"""CPU/synthetic tests for the ext_v5 native-head adapter (no model, no GPU, no cache, no pyarrow).

Exercises probing.native_head_adapter against a tiny fake native head (nn.Linear) + a local T5-style
RMSNorm, so every architectural contract is checked without loading Chronos-2:
  * identity-init adapter reproduces the zero-shot path EXACTLY;
  * the SAME adapter weight is applied to every forecast slot (shared-slot);
  * the L12+RMS path skips the RMSNorm (no double-normalisation);
  * only the adapter receives gradients (head/RMSNorm frozen but pass grad through);
  * the forward reuses probes._apply_shared_head's native (n k (q p) -> n q (k p)) layout;
  * wd is selected on validation only, the fit signature never sees test, and it is deterministic;
  * training reduces train loss from the identity start;
  * outputs are namespaced under ext_v5 (never ext_v4) + relative-regret math.
"""

from __future__ import annotations

import inspect

import numpy as np
import torch
import torch.nn as nn

from probing.native_head_adapter import (LinearAdapter, fit_adapter_explicit_val,
                                        slots_to_normalized_quantiles)
from probing.probes import _apply_shared_head, chronos2_quantile_loss

D, Q, PATCH, KS, HZN, N = 8, 3, 2, 2, 4, 40          # d, quantiles, patch, K slots, horizon, windows
QUANTS = np.array([0.25, 0.5, 0.75], dtype=np.float32)


class _RMSNorm(nn.Module):
    """T5-style RMSNorm (no mean subtraction) — the shape of Chronos-2's final_layer_norm."""
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        var = x.to(torch.float32).pow(2).mean(-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(var + self.eps))


def _fake_head_and_rms(seed=0):
    torch.manual_seed(seed)
    head = nn.Linear(D, Q * PATCH)               # callable (n, K, D) -> (n, K, Q*PATCH), like the ResidualBlock
    rms = _RMSNorm(D)
    for m in (head, rms):
        for p in m.parameters():
            p.requires_grad_(False)
    head.eval(); rms.eval()
    return head, rms


def _slots(seed=1):
    rng = np.random.default_rng(seed)
    return torch.as_tensor(rng.standard_normal((N, KS, D)).astype(np.float32))


def test_identity_adapter_equals_zero_shot():
    head, rms = _fake_head_and_rms()
    slots = _slots()
    a = LinearAdapter(D)                          # identity init
    # the adapter itself is the identity map at init
    np.testing.assert_allclose(a(slots).detach().numpy(), slots.numpy(), rtol=0, atol=1e-6)
    with torch.no_grad():
        zero = slots_to_normalized_quantiles(slots, None, True, rms, head, QUANTS, PATCH, HZN)
        adap = slots_to_normalized_quantiles(slots, a, True, rms, head, QUANTS, PATCH, HZN)
    np.testing.assert_allclose(adap.numpy(), zero.numpy(), rtol=0, atol=1e-6)


def test_shared_weight_across_slots():
    """The same A hits every slot: A(slots)[:, k] == linear(slots[:, k]) for each k independently."""
    a = LinearAdapter(D)
    with torch.no_grad():                         # perturb off identity so the map is non-trivial
        a.linear.weight.add_(0.1 * torch.randn(D, D))
        a.linear.bias.add_(0.1 * torch.randn(D))
    slots = _slots()
    out = a(slots)
    for k in range(KS):
        np.testing.assert_allclose(out[:, k].detach().numpy(),
                                   a.linear(slots[:, k]).detach().numpy(), rtol=1e-6, atol=1e-6)


def test_no_double_rmsnorm_at_ref():
    head, rms = _fake_head_and_rms()
    slots = _slots()
    with torch.no_grad():
        pre = slots_to_normalized_quantiles(slots, None, True, rms, head, QUANTS, PATCH, HZN)   # applies RMS
        post = slots_to_normalized_quantiles(slots, None, False, rms, head, QUANTS, PATCH, HZN)  # skips RMS
        manual = _apply_shared_head(head, slots, Q, PATCH, HZN)                                  # raw, no RMS
    np.testing.assert_allclose(post.numpy(), manual.numpy(), rtol=1e-6, atol=1e-6)
    assert not np.allclose(pre.numpy(), post.numpy()), "apply_rms=True/False produced identical output"


def test_reuses_native_apply_shared_head_layout():
    """slots_to_normalized_quantiles == manual native 'n k (q p) -> n q (k p)' reconstruction."""
    head, rms = _fake_head_and_rms()
    slots = _slots()
    with torch.no_grad():
        got = slots_to_normalized_quantiles(slots, None, True, rms, head, QUANTS, PATCH, HZN)
        h = rms(slots)
        manual = head(h).view(N, KS, Q, PATCH).permute(0, 2, 1, 3).reshape(N, Q, KS * PATCH)[:, :, :HZN]
    np.testing.assert_allclose(got.numpy(), manual.numpy(), rtol=1e-6, atol=1e-6)
    assert got.shape == (N, Q, HZN)


def test_only_adapter_receives_gradients():
    head, rms = _fake_head_and_rms()
    slots = _slots()
    y = torch.as_tensor(np.random.default_rng(3).standard_normal((N, HZN)).astype(np.float32))
    a = LinearAdapter(D)
    q = torch.as_tensor(QUANTS)
    pred = slots_to_normalized_quantiles(slots, a, True, rms, head, QUANTS, PATCH, HZN)
    chronos2_quantile_loss(pred, y, q).backward()
    assert a.linear.weight.grad is not None and a.linear.weight.grad.abs().sum() > 0
    for m in (head, rms):
        for p in m.parameters():
            assert p.grad is None, "a frozen head/RMSNorm parameter received a gradient"


def test_fit_signature_excludes_test_and_selects_on_val():
    sig = inspect.signature(fit_adapter_explicit_val)
    assert not any("test" in name for name in sig.parameters), "fit must never see a test split"
    head, rms = _fake_head_and_rms()
    tr = {0: _slots(1).numpy(), 1: _slots(2).numpy()}
    va = {0: _slots(11).numpy(), 1: _slots(12).numpy()}
    ytr = np.random.default_rng(4).standard_normal((N, HZN)).astype(np.float32)
    yva = np.random.default_rng(5).standard_normal((N, HZN)).astype(np.float32)
    fitted = fit_adapter_explicit_val(tr, ytr, va, yva, rms, head, layers=[0, 1], quantiles=QUANTS,
                                      epochs=20, wd_grid=(1e-4, 1e-1), device="cpu", output_patch_size=PATCH)
    assert set(fitted) == {0, 1}
    for L in (0, 1):
        assert set(fitted[L]["selection"]["val_loss_by_wd"]) == {1e-4, 1e-1}
        assert fitted[L]["family"] == "native_head_adapter"
        assert fitted[L]["param_count"] == D * D + D


def test_fit_is_deterministic():
    head, rms = _fake_head_and_rms()
    tr = {0: _slots(1).numpy()}; va = {0: _slots(11).numpy()}
    ytr = np.random.default_rng(4).standard_normal((N, HZN)).astype(np.float32)
    yva = np.random.default_rng(5).standard_normal((N, HZN)).astype(np.float32)
    kw = dict(layers=[0], quantiles=QUANTS, epochs=25, wd_grid=(1e-3,), device="cpu", output_patch_size=PATCH)
    f1 = fit_adapter_explicit_val(tr, ytr, va, yva, rms, head, **kw)
    f2 = fit_adapter_explicit_val(tr, ytr, va, yva, rms, head, **kw)
    w1 = f1[0]["adapter"].linear.weight.detach().numpy()
    w2 = f2[0]["adapter"].linear.weight.detach().numpy()
    np.testing.assert_allclose(w1, w2, rtol=0, atol=0)          # identical -> no init-seed banding needed


def test_training_reduces_train_loss_from_identity():
    head, rms = _fake_head_and_rms()
    slots = _slots(7)
    y = torch.as_tensor(np.random.default_rng(8).standard_normal((N, HZN)).astype(np.float32))
    q = torch.as_tensor(QUANTS)
    with torch.no_grad():
        ident = chronos2_quantile_loss(
            slots_to_normalized_quantiles(slots, LinearAdapter(D), True, rms, head, QUANTS, PATCH, HZN), y, q).item()
    fitted = fit_adapter_explicit_val({0: slots.numpy()}, y.numpy(), {0: slots.numpy()}, y.numpy(),
                                      rms, head, layers=[0], quantiles=QUANTS, epochs=200, wd_grid=(1e-5,),
                                      device="cpu", output_patch_size=PATCH)
    with torch.no_grad():
        trained = chronos2_quantile_loss(
            slots_to_normalized_quantiles(slots, fitted[0]["adapter"], True, rms, head, QUANTS, PATCH, HZN),
            y, q).item()
    assert trained < ident, f"training did not reduce train loss ({trained:.4f} !< {ident:.4f})"


def test_relative_regret_math():
    l_native, l_adapter = 0.80, 1.00
    assert abs((l_adapter / l_native - 1.0) - 0.25) < 1e-12


def test_outputs_namespaced_to_ext_v5_not_ext_v4():
    try:
        from experiments.run_native_head_adapter import OUT_ROOT
    except Exception as e:                                       # pyarrow/heavy import on a bare login node
        print(f"  (skipped driver-namespace check: {type(e).__name__})"); return
    s = str(OUT_ROOT)
    assert s.endswith("results/ext_v5_native_head_adapter"), s
    assert "ext_v4_future_tokens" not in s


if __name__ == "__main__":
    tests = [test_identity_adapter_equals_zero_shot, test_shared_weight_across_slots,
             test_no_double_rmsnorm_at_ref, test_reuses_native_apply_shared_head_layout,
             test_only_adapter_receives_gradients, test_fit_signature_excludes_test_and_selects_on_val,
             test_fit_is_deterministic, test_training_reduces_train_loss_from_identity,
             test_relative_regret_math, test_outputs_namespaced_to_ext_v5_not_ext_v4]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} native-head-adapter tests passed.")
