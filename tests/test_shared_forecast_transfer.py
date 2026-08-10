"""Focused tests for the frozen fit/predict split of the shared forecast-token probe.

No model, no GPU — synthetic (n, K, 768)-style forecast-slot features only, so this runs on a
login node:

    OMP_NUM_THREADS=2 python -m tests.test_shared_forecast_transfer

The v4 "future tokens" line makes shared_forecast_token_probe the headline readout, driven through
fit_shared_forecast_probe_explicit_val (rolling-tunnel wd on an explicit temporal val) +
predict_shared_forecast_probe (frozen PT-OOD scoring). These are the shared-head twins of
fit_quantile_probe_explicit_val / predict_quantile_probe, so the same invariants apply: the frozen
fit->predict reproduces the combined probe on the same split (no silent math change), one probe is
trained once and reused across targets without mutation, the target arrays never reach the fit, the
3-D slot contract is enforced, the selection dict carries {val_loss_by_wd, chosen_wd} (so the
source-selected-layer picker keeps working), and init_seed is the one knob the 3-run protocol turns.
"""

from __future__ import annotations

import inspect
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from probing.config import NUM_LAYERS, SEED, OUTPUT_PATCH_SIZE
from probing.probes import (QUANTILE_SETS, shared_forecast_token_probe,
                            fit_shared_forecast_probe_explicit_val,
                            predict_shared_forecast_probe)

Q9 = QUANTILE_SETS["q9"]
D = 6                                    # tiny feature dim (speed); real d = 768
P = int(OUTPUT_PATCH_SIZE)               # model output patch size
H = P + 8                                # horizon that forces multi-slot + trim
K = math.ceil(H / P)                     # native slot count for this (H, P) -> 2
assert K >= 2, "test wants K>=2 to exercise the concat+trim path"


def _synth_slot(n, d=D, seed=0, n_points=NUM_LAYERS):
    """Synthetic {layer: (n, K, d)} forecast-slot features + (n, H) trajectory labels — stands in
    for one dataset's K native forecast-slot states. Different `seed` = a different 'dataset'.
    n_points = NUM_LAYERS+1 mimics the fslot v4 line, where key NUM_LAYERS is the extra post-final-LN
    readout point beyond L12 (the shared-head fns iterate feature-dict keys, not range(NUM_LAYERS))."""
    rng = np.random.default_rng(seed)
    feats = {i: rng.normal(size=(n, K, d)).astype(np.float32) for i in range(n_points)}
    y = rng.normal(size=(n, H)).astype(np.float32)
    return feats, y


# 1. Diagonal reproduction: fit(explicit-val) -> predict on the SAME test split must reproduce
#    shared_forecast_token_probe. A single-candidate wd grid makes selection trivial, so the kept
#    full-train model equals the combined probe's full-train refit (val split can't move it).
def test_fit_predict_reproduces_shared_forecast_probe():
    f_tr, y_tr = _synth_slot(40, seed=1)
    f_va, y_va = _synth_slot(15, seed=9)          # val only drives wd selection (trivial here)
    f_te, y_te = _synth_slot(25, seed=2)
    ref = shared_forecast_token_probe(f_tr, y_tr, f_te, y_te, quantiles=Q9, epochs=5,
                                      wd_grid=(1e-2,), device="cpu", output_patch_size=P)
    fitted = fit_shared_forecast_probe_explicit_val(f_tr, y_tr, f_va, y_va, quantiles=Q9,
                                                    epochs=5, wd_grid=(1e-2,), device="cpu",
                                                    output_patch_size=P)
    out = predict_shared_forecast_probe(fitted, f_te, y_te, quantiles=Q9, device="cpu",
                                        output_patch_size=P)
    for i in range(NUM_LAYERS):
        np.testing.assert_allclose(out[i], ref[i], rtol=1e-5, atol=1e-6,
                                   err_msg=f"L{i}: fit->predict != shared_forecast_token_probe")


# 2. One frozen probe scores multiple targets; weights are never mutated; predict is deterministic.
def test_frozen_probe_reused_across_targets_not_mutated():
    f_tr, y_tr = _synth_slot(40, seed=1)
    f_va, y_va = _synth_slot(15, seed=9)
    fitted = fit_shared_forecast_probe_explicit_val(f_tr, y_tr, f_va, y_va, quantiles=Q9,
                                                    epochs=5, wd_grid=(1e-2,), device="cpu",
                                                    output_patch_size=P)
    w0 = fitted[0]["linear"].weight.detach().clone()
    tA, tB = _synth_slot(15, seed=10), _synth_slot(22, seed=11)
    outA = predict_shared_forecast_probe(fitted, tA[0], tA[1], quantiles=Q9, device="cpu",
                                         output_patch_size=P)
    outB = predict_shared_forecast_probe(fitted, tB[0], tB[1], quantiles=Q9, device="cpu",
                                         output_patch_size=P)
    assert torch.equal(fitted[0]["linear"].weight, w0), "predict mutated the frozen probe"
    outA2 = predict_shared_forecast_probe(fitted, tA[0], tA[1], quantiles=Q9, device="cpu",
                                          output_patch_size=P)
    for i in range(NUM_LAYERS):
        assert outA[i] == outA2[i], f"L{i}: predict is not deterministic / has state"
        assert outA[i] != outB[i], f"L{i}: two different targets gave identical loss (suspicious)"


# 3. The fit sees ONLY train + val — no target/test array can leak into training.
def test_fit_signature_excludes_target():
    params = list(inspect.signature(fit_shared_forecast_probe_explicit_val).parameters)
    lowered = " ".join(params).lower()
    assert "test" not in lowered and "target" not in lowered, \
        f"fit signature must not take a target/test array, got {params}"
    assert params[:4] == ["train_feats", "train_labels", "val_feats", "val_labels"]


# 4. The selection dict has the shape the source-selected-layer picker reads (val_loss_by_wd over a
#    real grid + chosen_wd) — the contract that keeps save_checkpoints / source_selected_layer working.
def test_selection_dict_shape_for_source_selected_layer():
    f_tr, y_tr = _synth_slot(40, seed=1)
    f_va, y_va = _synth_slot(15, seed=9)
    grid = (1e-3, 1e-1)
    fitted = fit_shared_forecast_probe_explicit_val(f_tr, y_tr, f_va, y_va, quantiles=Q9,
                                                    epochs=5, wd_grid=grid, device="cpu",
                                                    output_patch_size=P)
    for i in range(NUM_LAYERS):
        sel = fitted[i]["selection"]
        assert set(sel) == {"val_loss_by_wd", "chosen_wd"}, f"L{i}: bad selection keys {set(sel)}"
        assert set(sel["val_loss_by_wd"]) == {float(g) for g in grid}, f"L{i}: grid not recorded"
        assert sel["chosen_wd"] in {float(g) for g in grid}, f"L{i}: chosen_wd off-grid"
    # source-selected layer = argmin over layers of the min-over-wd val loss (never touches a target)
    per_layer_val = [min(fitted[i]["selection"]["val_loss_by_wd"].values()) for i in range(NUM_LAYERS)]
    assert 0 <= int(np.argmin(per_layer_val)) < NUM_LAYERS


# 5. The 3-D (n, K, 768) slot contract is enforced on both ends — pooled (n, 768) features are rejected.
def test_requires_3d_slot_features():
    pooled = {i: np.random.default_rng(i).normal(size=(10, D)).astype(np.float32)
              for i in range(NUM_LAYERS)}         # 2-D, wrong readout
    y = np.random.default_rng(0).normal(size=(10, H)).astype(np.float32)
    try:
        fit_shared_forecast_probe_explicit_val(pooled, y, pooled, y, quantiles=Q9, epochs=2,
                                               device="cpu", output_patch_size=P)
        raise AssertionError("fit accepted 2-D pooled features (should require (n, K, 768))")
    except ValueError:
        pass
    f_tr, y_tr = _synth_slot(12, seed=1)
    fitted = fit_shared_forecast_probe_explicit_val(f_tr, y_tr, f_tr, y_tr, quantiles=Q9, epochs=2,
                                                    device="cpu", output_patch_size=P)
    try:
        predict_shared_forecast_probe(fitted, pooled, y, quantiles=Q9, device="cpu",
                                      output_patch_size=P)
        raise AssertionError("predict accepted 2-D pooled features (should require (n, K, 768))")
    except ValueError:
        pass


# 6. init_seed is the ONLY randomness: same seed -> byte-identical weights, different seed -> different.
def test_init_seed_controls_and_is_deterministic():
    f_tr, y_tr = _synth_slot(30, seed=1)
    f_va, y_va = _synth_slot(12, seed=9)
    a = fit_shared_forecast_probe_explicit_val(f_tr, y_tr, f_va, y_va, quantiles=Q9, epochs=5,
                                               wd_grid=(1e-2,), device="cpu", init_seed=0,
                                               output_patch_size=P)
    b = fit_shared_forecast_probe_explicit_val(f_tr, y_tr, f_va, y_va, quantiles=Q9, epochs=5,
                                               wd_grid=(1e-2,), device="cpu", init_seed=0,
                                               output_patch_size=P)
    c = fit_shared_forecast_probe_explicit_val(f_tr, y_tr, f_va, y_va, quantiles=Q9, epochs=5,
                                               wd_grid=(1e-2,), device="cpu", init_seed=1,
                                               output_patch_size=P)
    assert torch.equal(a[0]["linear"].weight, b[0]["linear"].weight), "same init_seed not identical"
    assert not torch.equal(a[0]["linear"].weight, c[0]["linear"].weight), "init_seed did not vary the fit"


# 7. Per-window losses (for the cluster bootstrap) average back to the reported scalar loss.
def test_per_window_loss_mean_equals_scalar():
    f_tr, y_tr = _synth_slot(30, seed=1)
    f_va, y_va = _synth_slot(12, seed=9)
    f_te, y_te = _synth_slot(20, seed=2)
    fitted = fit_shared_forecast_probe_explicit_val(f_tr, y_tr, f_va, y_va, quantiles=Q9, epochs=5,
                                                    wd_grid=(1e-2,), device="cpu", output_patch_size=P)
    out, diag = predict_shared_forecast_probe(fitted, f_te, y_te, quantiles=Q9, device="cpu",
                                              collect_test_window_loss=True, output_patch_size=P)
    for i in range(NUM_LAYERS):
        pw = diag["test_window_loss"][i]
        assert pw.shape == (len(y_te),), f"L{i}: per-window loss shape {pw.shape}"
        np.testing.assert_allclose(pw.mean(), out[i], rtol=1e-5, atol=1e-6,
                                   err_msg=f"L{i}: per-window mean != scalar loss")


# 8. The v4 fslot line adds a 14th readout point (post-final-LN slots) as key NUM_LAYERS. The
#    shared-head fns must iterate feature-dict keys, so a 14-key dict produces 14-key outputs — and
#    the 13 shared keys must be BYTE-IDENTICAL to a 13-key run (the extra key can't perturb the rest,
#    since each layer is fit/scored independently). This is the invariant the post-LN change rests on.
def test_fourteen_key_dict_extra_post_ln_point():
    n_pts = NUM_LAYERS + 1
    f_tr13, y_tr = _synth_slot(40, seed=1)
    f_va13, y_va = _synth_slot(15, seed=9)
    f_te13, y_te = _synth_slot(25, seed=2)
    f_tr14, _ = _synth_slot(40, seed=1, n_points=n_pts)   # same seed → keys 0..12 identical to 13-key
    f_va14, _ = _synth_slot(15, seed=9, n_points=n_pts)
    f_te14, _ = _synth_slot(25, seed=2, n_points=n_pts)
    # sanity: the shared keys really are identical arrays across the 13- and 14-key dicts
    for i in range(NUM_LAYERS):
        assert np.array_equal(f_tr14[i], f_tr13[i]) and np.array_equal(f_te14[i], f_te13[i])

    kw = dict(quantiles=Q9, epochs=5, wd_grid=(1e-2,), device="cpu", output_patch_size=P)
    fit13 = fit_shared_forecast_probe_explicit_val(f_tr13, y_tr, f_va13, y_va, **kw)
    fit14 = fit_shared_forecast_probe_explicit_val(f_tr14, y_tr, f_va14, y_va, **kw)
    assert set(fit14) == set(range(n_pts)), f"14-key fit produced keys {sorted(fit14)}"
    assert NUM_LAYERS in fit14, "post-LN key (NUM_LAYERS) missing from the fit"

    out13 = predict_shared_forecast_probe(fit13, f_te13, y_te, quantiles=Q9, device="cpu",
                                          output_patch_size=P)
    out14 = predict_shared_forecast_probe(fit14, f_te14, y_te, quantiles=Q9, device="cpu",
                                          output_patch_size=P)
    assert set(out14) == set(range(n_pts)), f"14-key predict produced keys {sorted(out14)}"
    for i in range(NUM_LAYERS):        # the 13 shared points are unperturbed by the extra key
        np.testing.assert_allclose(out14[i], out13[i], rtol=1e-6, atol=1e-7,
                                   err_msg=f"L{i}: 14-key run changed a shared point")
    assert np.isfinite(out14[NUM_LAYERS]), "post-LN point produced a non-finite loss"


if __name__ == "__main__":
    tests = [test_fit_predict_reproduces_shared_forecast_probe,
             test_frozen_probe_reused_across_targets_not_mutated,
             test_fit_signature_excludes_target,
             test_selection_dict_shape_for_source_selected_layer,
             test_requires_3d_slot_features,
             test_init_seed_controls_and_is_deterministic,
             test_per_window_loss_mean_equals_scalar,
             test_fourteen_key_dict_extra_post_ln_point]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} shared-forecast transfer tests passed.")
