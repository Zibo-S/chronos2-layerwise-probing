"""No-GPU probe-level checks for the native-structure MLP forecast-slot head (v4 capacity control).

Synthetic (n, K, 768)-style forecast-slot features only — no model, no GPU, no HF data — so this
runs on a login node:

    OMP_NUM_THREADS=2 python -m tests.test_slot_mlp_probe

The MLP head (probing.heads.ResidualBlock, one shared head over the K native forecast slots) is the
NONLINEAR twin of the shared linear forecast-token probe. Its explicit-validation fit
(fit_forecast_slot_native_head_explicit_val) + frozen predict (predict_forecast_slot_native_head)
mirror the linear pair, so the same invariants apply: fit->predict reproduces the existing combined
slot-head path on the same split (no silent math change); one probe is trained once and reused across
targets without mutation; the target arrays never reach the fit; the selection dict carries the
{val_loss_by_wd, train_loss_by_wd, chosen_wd} contract; the 3-D slot contract is enforced; init_seed
is the one knob the 3-run protocol turns; ONE shared head (out = Q*P) is applied across all K slots;
the horizon is trimmed when H is not a multiple of P; the 14-key fslot dict adds the post-final-LN
point without perturbing the shared 13; the heads are freshly initialised (never native weights); and
the training diagnostics (per-epoch history + convergence) required for the overfitting audit exist.

Everything runs with the NATIVE-FAITHFUL dropout=0.1 the driver uses, to exercise that path (dropout
is on in train, off in eval, so predict stays deterministic).
"""

from __future__ import annotations

import inspect
import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from probing.config import NUM_LAYERS, SEED, OUTPUT_PATCH_SIZE
from probing.heads import ResidualBlock, build_head, head_param_count
from probing.probes import (QUANTILE_SETS, _apply_shared_head, _fit_slot_scaler, _slot_transform,
                            fit_forecast_slot_native_head,
                            fit_forecast_slot_native_head_explicit_val,
                            predict_forecast_slot_native_head)

Q9 = QUANTILE_SETS["q9"]
Q = len(Q9)
D = 8                                    # tiny feature dim (speed); real d = 768
P = int(OUTPUT_PATCH_SIZE)               # model output patch size
H = P + 8                                # horizon that forces multi-slot + trim (K*P > H)
K = math.ceil(H / P)                     # native slot count for this (H, P) -> 2
assert K >= 2 and (K * P) > H, "test wants K>=2 and a genuine trim (K*P > H)"
DROPOUT = 0.1                            # the native-faithful value the driver uses
HID = 16                                 # tiny hidden width (speed); real d_ff = 3072


def _synth_slot(n, d=D, seed=0, n_points=NUM_LAYERS):
    """Synthetic {layer: (n, K, d)} forecast-slot features + (n, H) trajectory labels — stands in for
    one dataset's K native forecast-slot states. Different `seed` = a different 'dataset'. n_points =
    NUM_LAYERS+1 mimics the fslot v4 line (key NUM_LAYERS = the post-final-LN readout point)."""
    rng = np.random.default_rng(seed)
    feats = {i: rng.normal(size=(n, K, d)).astype(np.float32) for i in range(n_points)}
    y = rng.normal(size=(n, H)).astype(np.float32)
    return feats, y


_KW = dict(quantiles=Q9, epochs=5, wd_grid=(1e-2,), device="cpu", hidden_dim=HID, dropout=DROPOUT,
           output_patch_size=P)


# 1. fit(explicit-val) -> predict reproduces the existing combined slot-head path (fit_forecast_slot_
#    native_head + predict) on the SAME test split. Single-wd grid + identical default init_seed=SEED
#    -> both keep a full-train head fit with the same seed, so the kept models are byte-identical.
def test_fit_predict_reproduces_combined_slot_head():
    f_tr, y_tr = _synth_slot(36, seed=1)
    f_va, y_va = _synth_slot(15, seed=9)                 # val only drives wd selection (trivial here)
    f_te, y_te = _synth_slot(22, seed=2)
    ref_fit = fit_forecast_slot_native_head(f_tr, y_tr, **_KW)          # 80/20 carve, final = full train
    ref = predict_forecast_slot_native_head(ref_fit, f_te, y_te, quantiles=Q9, device="cpu",
                                            output_patch_size=P)
    fitted = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_va, y_va, **_KW)
    out = predict_forecast_slot_native_head(fitted, f_te, y_te, quantiles=Q9, device="cpu",
                                            output_patch_size=P)
    for i in range(NUM_LAYERS):
        np.testing.assert_allclose(out[i], ref[i], rtol=1e-5, atol=1e-6,
                                   err_msg=f"L{i}: explicit-val fit->predict != combined slot-head path")


# 2. One frozen probe scores multiple targets; weights are never mutated; predict is deterministic
#    (dropout=0.1 is OFF in eval, so repeated predicts agree exactly).
def test_frozen_probe_reused_across_targets_not_mutated():
    f_tr, y_tr = _synth_slot(36, seed=1)
    f_va, y_va = _synth_slot(15, seed=9)
    fitted = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_va, y_va, **_KW)
    w0 = fitted[0]["head"].hidden_layer.weight.detach().clone()
    tA, tB = _synth_slot(15, seed=10), _synth_slot(20, seed=11)
    outA = predict_forecast_slot_native_head(fitted, tA[0], tA[1], quantiles=Q9, device="cpu",
                                             output_patch_size=P)
    outB = predict_forecast_slot_native_head(fitted, tB[0], tB[1], quantiles=Q9, device="cpu",
                                             output_patch_size=P)
    assert torch.equal(fitted[0]["head"].hidden_layer.weight, w0), "predict mutated the frozen probe"
    outA2 = predict_forecast_slot_native_head(fitted, tA[0], tA[1], quantiles=Q9, device="cpu",
                                              output_patch_size=P)
    for i in range(NUM_LAYERS):
        assert outA[i] == outA2[i], f"L{i}: predict is not deterministic / has state"
        assert outA[i] != outB[i], f"L{i}: two different targets gave identical loss (suspicious)"


# 3. The fit sees ONLY train + val — no target/test array can leak into training.
def test_fit_signature_excludes_target():
    params = list(inspect.signature(fit_forecast_slot_native_head_explicit_val).parameters)
    lowered = " ".join(params).lower()
    assert "test" not in lowered and "target" not in lowered, \
        f"fit signature must not take a target/test array, got {params}"
    assert params[:4] == ["train_feats", "train_labels", "val_feats", "val_labels"]


# 4. The selection dict carries the source-selected-layer contract (val + train grids + chosen_wd).
def test_selection_dict_shape_for_source_selected_layer():
    f_tr, y_tr = _synth_slot(36, seed=1)
    f_va, y_va = _synth_slot(15, seed=9)
    grid = (1e-3, 1e-1)
    fitted = fit_forecast_slot_native_head_explicit_val(
        f_tr, y_tr, f_va, y_va, quantiles=Q9, epochs=5, wd_grid=grid, device="cpu", hidden_dim=HID,
        dropout=DROPOUT, output_patch_size=P)
    for i in range(NUM_LAYERS):
        sel = fitted[i]["selection"]
        assert set(sel) == {"val_loss_by_wd", "train_loss_by_wd", "chosen_wd"}, f"L{i}: {set(sel)}"
        assert set(sel["val_loss_by_wd"]) == {float(g) for g in grid}, f"L{i}: val grid not recorded"
        assert set(sel["train_loss_by_wd"]) == {float(g) for g in grid}, f"L{i}: train grid not recorded"
        assert sel["chosen_wd"] in {float(g) for g in grid}, f"L{i}: chosen_wd off-grid"
    per_layer_val = [min(fitted[i]["selection"]["val_loss_by_wd"].values()) for i in range(NUM_LAYERS)]
    assert 0 <= int(np.argmin(per_layer_val)) < NUM_LAYERS


# 5. The 3-D (n, K, 768) slot contract is enforced on both ends — 2-D pooled features are rejected.
def test_requires_3d_slot_features():
    pooled = {i: np.random.default_rng(i).normal(size=(10, D)).astype(np.float32)
              for i in range(NUM_LAYERS)}
    y = np.random.default_rng(0).normal(size=(10, H)).astype(np.float32)
    try:
        fit_forecast_slot_native_head_explicit_val(pooled, y, pooled, y, quantiles=Q9, epochs=2,
                                                   device="cpu", hidden_dim=HID, output_patch_size=P)
        raise AssertionError("fit accepted 2-D pooled features (should require (n, K, 768))")
    except ValueError:
        pass
    f_tr, y_tr = _synth_slot(12, seed=1)
    fitted = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_tr, y_tr, quantiles=Q9, epochs=2,
                                                        device="cpu", hidden_dim=HID, output_patch_size=P)
    try:
        predict_forecast_slot_native_head(fitted, pooled, y, quantiles=Q9, device="cpu",
                                          output_patch_size=P)
        raise AssertionError("predict accepted 2-D pooled features (should require (n, K, 768))")
    except ValueError:
        pass


# 6. init_seed is the ONLY randomness: same seed -> identical head + dropout stream, different -> not.
def test_init_seed_controls_and_is_deterministic():
    f_tr, y_tr = _synth_slot(30, seed=1)
    f_va, y_va = _synth_slot(12, seed=9)
    kw = dict(quantiles=Q9, epochs=5, wd_grid=(1e-2,), device="cpu", hidden_dim=HID, dropout=DROPOUT,
              output_patch_size=P)
    a = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_va, y_va, init_seed=0, **kw)
    b = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_va, y_va, init_seed=0, **kw)
    c = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_va, y_va, init_seed=1, **kw)
    wa, wb, wc = (x[0]["head"].output_layer.weight for x in (a, b, c))
    assert torch.equal(wa, wb), "same init_seed did not reproduce the head"
    assert not torch.equal(wa, wc), "init_seed did not vary the fit (dropout stream + init)"


# 7. Per-window losses (for the cluster bootstrap) average back to the reported scalar loss.
def test_per_window_loss_mean_equals_scalar():
    f_tr, y_tr = _synth_slot(30, seed=1)
    f_va, y_va = _synth_slot(12, seed=9)
    f_te, y_te = _synth_slot(20, seed=2)
    fitted = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_va, y_va, **_KW)
    out, diag = predict_forecast_slot_native_head(fitted, f_te, y_te, quantiles=Q9, device="cpu",
                                                  collect_test_window_loss=True, output_patch_size=P)
    for i in range(NUM_LAYERS):
        pw = diag["test_window_loss"][i]
        assert pw.shape == (len(y_te),), f"L{i}: per-window loss shape {pw.shape}"
        np.testing.assert_allclose(pw.mean(), out[i], rtol=1e-5, atol=1e-6,
                                   err_msg=f"L{i}: per-window mean != scalar loss")


# 8. ONE shared head (out = Q*P, a single patch) is applied across ALL K slots, and the prediction is
#    (n, Q, H) with H trimmed from K*P. Not K separate heads: param_count == a single Q*P head's.
def test_one_shared_head_over_slots_and_shape():
    f_tr, y_tr = _synth_slot(24, seed=1)
    f_va, y_va = _synth_slot(10, seed=9)
    fitted = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_va, y_va, **_KW)
    head = fitted[0]["head"]
    assert fitted[0]["out_features"] == Q * P, "shared head must emit ONE Q*P patch, not Q*H"
    assert head.output_layer.out_features == Q * P
    ref_params = head_param_count(build_head(D, Q * P, hidden_dim=HID, dropout=DROPOUT))
    assert fitted[0]["param_count"] == ref_params, "param count != a single shared Q*P head"
    # applying the shared head to (n, K, D) yields (n, Q, H) after concat over K patches + trim to H
    f_te, _ = _synth_slot(7, seed=3)
    sc = fitted[0]["scaler"]
    Xte = torch.as_tensor(_slot_transform(sc, f_te[0]), dtype=torch.float32)
    pred = _apply_shared_head(head, Xte, Q, P, H)
    assert pred.shape == (7, Q, H), f"expected (n, Q, H) after trim, got {tuple(pred.shape)}"
    assert K * P > H, "sanity: this fixture must genuinely trim (K*P > H)"


# 9. The heads are FRESHLY initialised (build_head + init_seed) — they never load native Chronos-2
#    head weights. Structural: the head is a probing.heads.ResidualBlock, and training MOVED it away
#    from its seeded init (so weights come from the fit, not from a pretrained checkpoint).
def test_heads_are_fresh_not_native():
    f_tr, y_tr = _synth_slot(24, seed=1)
    f_va, y_va = _synth_slot(10, seed=9)
    fitted = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_va, y_va, **_KW)
    head = fitted[0]["head"]
    assert isinstance(head, ResidualBlock), "probe head must be the from-scratch probing.heads block"
    assert type(head).__module__ == "probing.heads", "head must not come from the chronos2 package"
    torch.manual_seed(SEED)                                  # the fresh init the fit started from
    fresh = build_head(D, Q * P, hidden_dim=HID, dropout=DROPOUT)
    assert not torch.equal(head.output_layer.weight, fresh.output_layer.weight), \
        "trained head equals its fresh init — training did nothing / weights were injected"


# 10. The v4 fslot line adds a 14th readout point (post-final-LN) as key NUM_LAYERS. fit/predict must
#     iterate feature-dict keys, so a 14-key dict yields 14-key outputs and the 13 shared points stay
#     byte-identical to a 13-key run (each layer is fit/scored independently).
def test_fourteen_key_dict_extra_post_ln_point():
    n_pts = NUM_LAYERS + 1
    f_tr13, y_tr = _synth_slot(30, seed=1)
    f_va13, y_va = _synth_slot(12, seed=9)
    f_te13, y_te = _synth_slot(18, seed=2)
    f_tr14, _ = _synth_slot(30, seed=1, n_points=n_pts)      # same seed -> keys 0..12 identical
    f_va14, _ = _synth_slot(12, seed=9, n_points=n_pts)
    f_te14, _ = _synth_slot(18, seed=2, n_points=n_pts)
    for i in range(NUM_LAYERS):
        assert np.array_equal(f_tr14[i], f_tr13[i]) and np.array_equal(f_te14[i], f_te13[i])
    fit13 = fit_forecast_slot_native_head_explicit_val(f_tr13, y_tr, f_va13, y_va, **_KW)
    fit14 = fit_forecast_slot_native_head_explicit_val(f_tr14, y_tr, f_va14, y_va, **_KW)
    assert set(fit14) == set(range(n_pts)) and NUM_LAYERS in fit14, f"14-key keys {sorted(fit14)}"
    out13 = predict_forecast_slot_native_head(fit13, f_te13, y_te, quantiles=Q9, device="cpu",
                                              output_patch_size=P)
    out14 = predict_forecast_slot_native_head(fit14, f_te14, y_te, quantiles=Q9, device="cpu",
                                              output_patch_size=P)
    assert set(out14) == set(range(n_pts)), f"14-key predict keys {sorted(out14)}"
    for i in range(NUM_LAYERS):
        np.testing.assert_allclose(out14[i], out13[i], rtol=1e-6, atol=1e-7,
                                   err_msg=f"L{i}: 14-key run changed a shared point")
    assert np.isfinite(out14[NUM_LAYERS]), "post-LN point produced a non-finite loss"


# 11. The training diagnostics the overfitting audit (§3) needs are present per layer: per-epoch
#     train + val history, final losses, chosen wd, lr, dropout, param count, and a convergence flag.
def test_training_diagnostics_present():
    f_tr, y_tr = _synth_slot(24, seed=1)
    f_va, y_va = _synth_slot(10, seed=9)
    fitted = fit_forecast_slot_native_head_explicit_val(f_tr, y_tr, f_va, y_va, **_KW)
    for i in range(NUM_LAYERS):
        f = fitted[i]
        assert len(f["history"]["train"]) >= 5 and len(f["history"]["val"]) >= 5, "no per-epoch history"
        assert np.isfinite(f["final_train_loss"]) and np.isfinite(f["final_val_loss"])
        for kk in ("wd", "lr", "dropout", "epochs", "selected_epoch", "param_count", "init_seed"):
            assert kk in f, f"L{i}: diagnostic field '{kk}' missing"
        assert f["dropout"] == DROPOUT, "recorded dropout must equal the fitted value"
        assert f["converged"] in (True, False, None), f"L{i}: bad convergence flag {f['converged']}"


if __name__ == "__main__":
    tests = [test_fit_predict_reproduces_combined_slot_head,
             test_frozen_probe_reused_across_targets_not_mutated,
             test_fit_signature_excludes_target,
             test_selection_dict_shape_for_source_selected_layer,
             test_requires_3d_slot_features,
             test_init_seed_controls_and_is_deterministic,
             test_per_window_loss_mean_equals_scalar,
             test_one_shared_head_over_slots_and_shape,
             test_heads_are_fresh_not_native,
             test_fourteen_key_dict_extra_post_ln_point,
             test_training_diagnostics_present]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(tests)} slot-MLP probe tests passed.")
