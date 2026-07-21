"""Focused tests for the configurable quantile sets (q1 / q9 / q21).

No model, no cache, no GPU — synthetic features only, so this runs on a login node:

    OMP_NUM_THREADS=2 python -m tests.test_quantile_sets

(also collectable by pytest if it is ever installed). Covers: registry + validation,
output dims and exact parameter counts, the (B, Q, H) prediction layout, loss identities
(mean_pinball = chronos/2Q; pinball at q=0.5 = 0.5*MAE; formula vs an explicit-loop
reference), q21 default-path preservation, median lookup without substitution, and both
probes end-to-end on pooled (content/REG-style) and forecast-slot features for all sets.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")   # shared login node: don't grab all threads

import numpy as np
import torch

from probing.config import NUM_LAYERS, SEED
from probing.probes import (CHRONOS2_QUANTILES, QUANTILE_SETS, validate_quantiles,
                            median_index, chronos2_quantile_loss,
                            chronos2_quantile_loss_per_window, mean_pinball_loss,
                            quantile_probe, shared_forecast_token_probe,
                            _fit_quantile_linear)

D_MODEL, H_FULL = 768, 64
# quantile set -> (Q, output_dim = Q*H, params = D_MODEL*Q*H + Q*H) at d=768, H=64
EXPECTED = {"q1": (1, 64, 49_216), "q9": (9, 576, 442_944), "q21": (21, 1344, 1_033_536)}


def _rand_pred_target(B=7, Q=9, H=12, seed=3):
    g = torch.Generator().manual_seed(seed)
    pred = torch.randn(B, Q, H, generator=g)
    target = torch.randn(B, H, generator=g)
    return pred, target


def _synth_pooled_feats(n, d, seed):
    """{layer: (n, d)} synthetic pooled features (content/REG-style)."""
    rng = np.random.default_rng(seed)
    return {i: rng.normal(size=(n, d)).astype(np.float32) for i in range(NUM_LAYERS)}


def _synth_slot_feats(n, K, d, seed):
    """{layer: (n, K, d)} synthetic forecast-slot features."""
    rng = np.random.default_rng(seed)
    return {i: rng.normal(size=(n, K, d)).astype(np.float32) for i in range(NUM_LAYERS)}


def test_registry_and_median_index():
    # q21 IS the verified Chronos-2 vector — same object contents, exactly
    np.testing.assert_array_equal(QUANTILE_SETS["q21"], CHRONOS2_QUANTILES)
    for name, (Q, _, _) in EXPECTED.items():
        q = validate_quantiles(QUANTILE_SETS[name])
        assert len(q) == Q, f"{name}: expected {Q} levels, got {len(q)}"
    # exact 0.5 lookup (never a nearest-neighbor substitute)
    assert median_index(QUANTILE_SETS["q1"]) == 0
    assert median_index(QUANTILE_SETS["q9"]) == 4
    assert median_index(QUANTILE_SETS["q21"]) == 10
    assert median_index([0.25, 0.75]) is None


def test_validate_quantiles_rejects_bad_inputs():
    for bad in ([], [0.0, 0.5], [0.5, 1.0], [-0.1, 0.5], [0.9, 0.1], [0.3, 0.3, 0.7]):
        try:
            validate_quantiles(bad)
        except AssertionError:
            continue
        raise AssertionError(f"validate_quantiles accepted invalid input {bad}")


def test_output_dim_and_param_counts():
    Xtr = torch.randn(8, D_MODEL)
    for name, (Q, out_dim, n_params) in EXPECTED.items():
        q = torch.as_tensor(QUANTILE_SETS[name])
        ytr = torch.randn(8, H_FULL)
        lin = _fit_quantile_linear(Xtr, ytr, q, weight_decay=0.0, epochs=1, lr=1e-2,
                                   device="cpu")
        assert lin.out_features == out_dim == Q * H_FULL, (
            f"{name}: out_features {lin.out_features} != {Q}*{H_FULL}")
        total = sum(p.numel() for p in lin.parameters())
        assert total == n_params, f"{name}: {total:,} params != expected {n_params:,}"


def test_reshape_convention_and_shape_asserts():
    # convention everywhere: predictions are (B, Q, H) — quantiles on -2, horizon on -1
    # (Chronos-2's native 'b q (n p)' layout; deliberately NOT (B, H, Q), which would
    # re-pair the seeded init with different (q, h) targets and move the q21 numbers)
    for name, (Q, _, _) in EXPECTED.items():
        q = torch.as_tensor(QUANTILE_SETS[name])
        B, H = 5, H_FULL
        lin = torch.nn.Linear(D_MODEL, Q * H)
        pred = lin(torch.randn(B, D_MODEL)).view(B, Q, H)
        assert pred.shape == (B, Q, H)
        assert pred.shape[-2] == len(QUANTILE_SETS[name]) and pred.shape[-1] == H
        target = torch.randn(B, H)
        assert torch.isfinite(chronos2_quantile_loss(pred, target, q))
        # a transposed (B, H, Q) prediction must be rejected, not silently mis-scored
        if Q != H:
            try:
                chronos2_quantile_loss(pred.transpose(1, 2), target, q)
            except AssertionError:
                pass
            else:
                raise AssertionError(f"{name}: loss accepted a (B, H, Q) prediction")


def test_loss_formula_against_explicit_reference():
    """Vectorized Chronos-2 loss == an explicit per-element loop (formula + reduction)."""
    pred, target = _rand_pred_target()
    q = torch.as_tensor(QUANTILE_SETS["q9"])
    B, Q, H = pred.shape
    per_batch = []
    for b in range(B):
        acc = 0.0
        for j in range(Q):
            row = 0.0
            for t in range(H):
                y, p, tau = target[b, t].item(), pred[b, j, t].item(), q[j].item()
                row += 2.0 * abs((y - p) * ((1.0 if y <= p else 0.0) - tau))
            acc += row / H                                # mean over horizon
        per_batch.append(acc)                             # sum over quantiles
    ref = float(np.mean(per_batch))                       # mean over batch
    got = float(chronos2_quantile_loss(pred, target, q))
    np.testing.assert_allclose(got, ref, rtol=1e-5)
    # per-window variant: same numbers before the batch mean
    pw = chronos2_quantile_loss_per_window(pred, target, q)
    np.testing.assert_allclose(pw.numpy(), per_batch, rtol=1e-5)
    np.testing.assert_allclose(float(pw.mean()), got, rtol=1e-6)


def test_pinball_identities():
    for name, (Q, _, _) in EXPECTED.items():
        q = torch.as_tensor(QUANTILE_SETS[name])
        pred, target = _rand_pred_target(Q=Q, seed=11)
        chron = float(chronos2_quantile_loss(pred, target, q))
        pin = float(mean_pinball_loss(pred, target, q))
        assert np.isfinite(pin) and pin >= 0
        # elementwise chronos term = 2*pinball; reductions differ only by the 1/Q factor
        np.testing.assert_allclose(pin, chron / (2 * Q), rtol=1e-5,
                                   err_msg=f"{name}: mean_pinball != chronos/(2Q)")
    # median-only: pinball(q=0.5) == 0.5 * MAE  (and chronos(q=[0.5]) == MAE)
    q1 = torch.as_tensor(QUANTILE_SETS["q1"])
    pred, target = _rand_pred_target(Q=1, seed=13)
    mae = float((target.unsqueeze(1) - pred).abs().mean())
    np.testing.assert_allclose(float(mean_pinball_loss(pred, target, q1)), 0.5 * mae, rtol=1e-6)
    np.testing.assert_allclose(float(chronos2_quantile_loss(pred, target, q1)), mae, rtol=1e-6)


def _run_pooled(feats_tr, feats_te, name, H, collect_median=True):
    rng = np.random.default_rng(SEED + 1)
    n_tr, n_te = feats_tr[0].shape[0], feats_te[0].shape[0]
    ytr = rng.normal(size=(n_tr, H)).astype(np.float32)
    yte = rng.normal(size=(n_te, H)).astype(np.float32)
    return quantile_probe(feats_tr, ytr, feats_te, yte, quantiles=QUANTILE_SETS[name],
                          epochs=5, wd_grid=None, device="cpu",
                          collect_test_median=collect_median)


def test_quantile_probe_all_sets_both_representations():
    """End-to-end pooled probe for q1/q9/q21 on two feature dicts standing in for the
    content-pooled and REG-token representations (the probe is representation-agnostic —
    identical code path for both; only the feature arrays differ)."""
    n_tr, n_te, d, H = 30, 12, 16, 8
    reps = {"content": (_synth_pooled_feats(n_tr, d, 1), _synth_pooled_feats(n_te, d, 2)),
            "reg":     (_synth_pooled_feats(n_tr, d, 3), _synth_pooled_feats(n_te, d, 4))}
    for rep, (ftr, fte) in reps.items():
        for name, (Q, _, _) in EXPECTED.items():
            out, diag = _run_pooled(ftr, fte, name, H)
            assert set(out) == set(range(NUM_LAYERS))
            assert all(np.isfinite(v) for v in out.values()), f"{rep}/{name}: non-finite loss"
            for i in range(NUM_LAYERS):
                assert diag["test_median"][i].shape == (n_te, H)
                np.testing.assert_allclose(diag["test_mean_pinball"][i],
                                           out[i] / (2 * Q), rtol=1e-4,
                                           err_msg=f"{rep}/{name}/L{i}")


def test_q21_default_path_preserved():
    """The default call (no quantiles argument) IS the q21 configuration, and the fit is
    deterministic — the code-level guarantee that selecting q21 reproduces the committed
    numbers (the byte-identical GPU check is the PLAN's git-diff equivalence protocol)."""
    n_tr, n_te, d, H = 30, 12, 16, 8
    ftr, fte = _synth_pooled_feats(n_tr, d, 1), _synth_pooled_feats(n_te, d, 2)
    rng = np.random.default_rng(SEED + 1)
    ytr = rng.normal(size=(n_tr, H)).astype(np.float32)
    yte = rng.normal(size=(n_te, H)).astype(np.float32)
    default = quantile_probe(ftr, ytr, fte, yte, epochs=5, wd_grid=None, device="cpu")
    explicit, _ = _run_pooled(ftr, fte, "q21", H)
    rerun, _ = _run_pooled(ftr, fte, "q21", H)
    assert default == explicit == rerun, "q21 path not deterministic / default drifted"


def test_median_never_substituted():
    """collect_test_median must fail loudly when 0.5 is absent — not fall back to 0.4/0.6."""
    n_tr, n_te, d, H = 20, 8, 8, 8
    ftr, fte = _synth_pooled_feats(n_tr, d, 5), _synth_pooled_feats(n_te, d, 6)
    rng = np.random.default_rng(SEED)
    ytr = rng.normal(size=(n_tr, H)).astype(np.float32)
    yte = rng.normal(size=(n_te, H)).astype(np.float32)
    try:
        quantile_probe(ftr, ytr, fte, yte, quantiles=[0.4, 0.6], epochs=2, wd_grid=None,
                       device="cpu", collect_test_median=True)
    except ValueError as e:
        assert "0.5" in str(e)
    else:
        raise AssertionError("collect_test_median without a 0.5 level did not raise")


def test_shared_forecast_probe_all_sets():
    n_tr, n_te, d, P = 24, 10, 16, 16
    H = 24                                                # K = ceil(24/16) = 2, with trim
    K = 2
    ftr, fte = _synth_slot_feats(n_tr, K, d, 7), _synth_slot_feats(n_te, K, d, 8)
    rng = np.random.default_rng(SEED + 2)
    ytr = rng.normal(size=(n_tr, H)).astype(np.float32)
    yte = rng.normal(size=(n_te, H)).astype(np.float32)
    for name, (Q, _, _) in EXPECTED.items():
        out, diag = shared_forecast_token_probe(
            ftr, ytr, fte, yte, quantiles=QUANTILE_SETS[name], epochs=5, wd_grid=None,
            device="cpu", collect_test_median=True, output_patch_size=P)
        assert all(np.isfinite(v) for v in out.values()), f"fslot/{name}: non-finite loss"
        for i in range(NUM_LAYERS):
            assert diag["test_median"][i].shape == (n_te, H)
            np.testing.assert_allclose(diag["test_mean_pinball"][i], out[i] / (2 * Q),
                                       rtol=1e-4, err_msg=f"fslot/{name}/L{i}")


TESTS = [test_registry_and_median_index,
         test_validate_quantiles_rejects_bad_inputs,
         test_output_dim_and_param_counts,
         test_reshape_convention_and_shape_asserts,
         test_loss_formula_against_explicit_reference,
         test_pinball_identities,
         test_quantile_probe_all_sets_both_representations,
         test_q21_default_path_preserved,
         test_median_never_substituted,
         test_shared_forecast_probe_all_sets]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(TESTS)} TESTS PASS")
