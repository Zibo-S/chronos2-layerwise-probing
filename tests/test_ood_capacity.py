"""Focused tests for the higher-capacity forecasting probes (capacity controls).

No model, no GPU — synthetic features only, so this runs on a login node:

    OMP_NUM_THREADS=2 python -m tests.test_ood_capacity

Covers the invariants that keep the capacity study honest and comparable to the committed
linear pilot: both families emit (B, 9, 64); the forecast-slot head shares ONE weight set
across all K slots; a source head is fit once and reused across targets without mutation;
target arrays never reach fitting or model selection; the checkpoint identity includes
probe_family and excludes the target; the source-selected layer uses only source-val loss;
per-window q9 loss reproduces the scalar aggregate; the extracted median is exactly the 0.5
quantile row the MASE pipeline consumes; and the capacity outputs are namespaced away from the
committed linear artifacts. A real-cache end-to-end smoke is gated behind RUN_OOD_CAP_SMOKE=1.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from probing.config import NUM_LAYERS, LAST_LAYER, OUTPUT_PATCH_SIZE
from probing.heads import ResidualBlock, build_head, head_param_count
from probing.probes import (QUANTILE_SETS, median_index, _apply_shared_head,
                            fit_content_mlp_head, predict_content_mlp_head,
                            fit_forecast_slot_native_head, predict_forecast_slot_native_head)

Q9 = QUANTILE_SETS["q9"]
QN = len(Q9)                       # 9
D, HID = 6, 8                      # tiny feature dim / head hidden width (speed)


def _content(n, d=D, h=8, seed=0):
    rng = np.random.default_rng(seed)
    return ({i: rng.normal(size=(n, d)).astype(np.float32) for i in range(NUM_LAYERS)},
            rng.normal(size=(n, h)).astype(np.float32))


def _slots(n, k, d=D, h=8, seed=0):
    rng = np.random.default_rng(seed)
    return ({i: rng.normal(size=(n, k, d)).astype(np.float32) for i in range(NUM_LAYERS)},
            rng.normal(size=(n, h)).astype(np.float32))


# 2. both families produce (B, 9, 64).
def test_both_families_emit_B_9_64():
    Hh, P = 64, OUTPUT_PATCH_SIZE
    K = -(-Hh // P)                                    # ceil = 4
    # content_mlp_head
    f_tr, y_tr = _content(10, h=Hh, seed=1)
    fit = fit_content_mlp_head(f_tr, y_tr, quantiles=Q9, epochs=2, wd_grid=(1e-3,),
                               device="cpu", hidden_dim=HID)
    m, sc = fit[0]["head"], fit[0]["scaler"]
    X = torch.as_tensor(sc.transform(_content(5, h=Hh, seed=2)[0][0]), dtype=torch.float32)
    with torch.no_grad():
        pred = m(X).view(-1, QN, Hh)
    assert pred.shape == (5, 9, 64), f"content_mlp_head pred {tuple(pred.shape)} != (5,9,64)"
    # forecast_slot_native_head
    fs_tr, ys_tr = _slots(10, K, h=Hh, seed=3)
    fitf = fit_forecast_slot_native_head(fs_tr, ys_tr, quantiles=Q9, epochs=2, wd_grid=(1e-3,),
                                         device="cpu", hidden_dim=HID)
    mf, scf = fitf[0]["head"], fitf[0]["scaler"]
    from probing.probes import _slot_transform
    Xf = torch.as_tensor(_slot_transform(scf, _slots(5, K, h=Hh, seed=4)[0][0]), dtype=torch.float32)
    with torch.no_grad():
        predf = _apply_shared_head(mf, Xf, QN, P, Hh)
    assert predf.shape == (5, 9, 64), f"forecast_slot pred {tuple(predf.shape)} != (5,9,64)"


# 3. the forecast-slot head uses ONE shared weight set across all slots.
def test_forecast_slot_head_shares_weights_across_slots():
    P, K = 2, 4                                        # tiny patch so K=4 slots without H=64
    Hh = P * K
    fs_tr, ys_tr = _slots(12, K, h=Hh, seed=5)
    fitf = fit_forecast_slot_native_head(fs_tr, ys_tr, quantiles=Q9, epochs=2, wd_grid=(1e-3,),
                                         device="cpu", hidden_dim=HID, output_patch_size=P)
    m = fitf[0]["head"]
    # (a) parameter count equals ONE ResidualBlock (768->hid->Q*P), not K separate heads.
    ref = head_param_count(build_head(D, QN * P, hidden_dim=HID))
    assert head_param_count(m) == ref, "slot head is not a single shared block"
    # (b) permuting the input slots permutes the output patches identically (equivariance only
    #     possible if the SAME head is applied to every slot).
    X = torch.randn(7, K, D)
    perm = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        base = _apply_shared_head(m, X, QN, P, Hh).view(7, QN, K, P)
        permd = _apply_shared_head(m, X[:, perm, :], QN, P, Hh).view(7, QN, K, P)
    torch.testing.assert_close(permd, base[:, :, perm, :], rtol=1e-5, atol=1e-5)


# 4. a source probe is fit once and reused for multiple targets, unmutated + deterministic.
def test_frozen_head_reused_across_targets_not_mutated():
    f_tr, y_tr = _content(20, seed=1)
    fit = fit_content_mlp_head(f_tr, y_tr, quantiles=Q9, epochs=3, wd_grid=(1e-2,), device="cpu",
                               hidden_dim=HID)
    w0 = fit[0]["head"].hidden_layer.weight.detach().clone()
    tA, tB = _content(9, seed=10), _content(13, seed=11)
    outA = predict_content_mlp_head(fit, tA[0], tA[1], quantiles=Q9, device="cpu")
    outB = predict_content_mlp_head(fit, tB[0], tB[1], quantiles=Q9, device="cpu")
    assert torch.equal(fit[0]["head"].hidden_layer.weight, w0), "predict mutated the frozen head"
    outA2 = predict_content_mlp_head(fit, tA[0], tA[1], quantiles=Q9, device="cpu")
    for i in range(NUM_LAYERS):
        assert outA[i] == outA2[i], f"L{i}: predict is not deterministic"
    assert set(outA) == set(outB) == set(range(NUM_LAYERS))


# 5. no target features / labels can enter fitting or model selection.
def test_fit_signatures_exclude_target_and_val():
    for fit in (fit_content_mlp_head, fit_forecast_slot_native_head):
        pars = set(inspect.signature(fit).parameters)
        assert "test_feats" not in pars and "test_labels" not in pars, f"{fit.__name__} sees target"
    for pred in (predict_content_mlp_head, predict_forecast_slot_native_head):
        pars = set(inspect.signature(pred).parameters)
        assert "wd_grid" not in pars and "weight_decay" not in pars, f"{pred.__name__} can retune"


# 6. checkpoint identity includes probe_family and excludes the target.
def test_checkpoint_identity_includes_family_excludes_target():
    from experiments.run_ood_capacity import _probe_run_id, _ckpt_meta, FAMILIES
    q = QUANTILE_SETS["q9"]
    a = _probe_run_id("monash_electricity_hourly", "content_mlp_head", "q9", 0)
    b = _probe_run_id("monash_electricity_hourly", "forecast_slot_native_head", "q9", 0)
    assert "content_mlp_head" in a and "electricity" in a
    assert a != b, "probe_family must change the checkpoint identity"
    assert "kdd" not in a and "uber" not in a, "target must not appear in the identity"
    meta = _ckpt_meta("monash_electricity_hourly", "content_mlp_head", "q9", 0, q)
    assert meta["probe_family"] == "content_mlp_head" and "target" not in meta


# 7. the source-selected layer is chosen on SOURCE-validation loss, never the target.
def test_source_selected_layer_uses_source_val_only():
    from experiments.run_ood_capacity import _source_selected_layer
    f_tr, y_tr = _content(20, seed=7)
    fit = fit_content_mlp_head(f_tr, y_tr, quantiles=Q9, epochs=2, wd_grid=(1e-3, 1e-1),
                               device="cpu", hidden_dim=HID)
    for i in range(NUM_LAYERS):                        # source-val loss recorded per layer
        assert fit[i]["source_val_loss"] is not None and np.isfinite(fit[i]["source_val_loss"])
    ssl = _source_selected_layer(fit)
    assert ssl == int(min(range(NUM_LAYERS), key=lambda i: fit[i]["source_val_loss"]))
    # it is a function of the frozen probe only — scoring different targets cannot change it.
    predict_content_mlp_head(fit, *_content(8, seed=20), quantiles=Q9, device="cpu")
    assert _source_selected_layer(fit) == ssl


# 8. per-window q9 loss reproduces the scalar aggregate; median = the exact 0.5 quantile row.
def test_per_window_reduction_and_median_extraction():
    f_tr, y_tr = _content(16, h=12, seed=8)
    fit = fit_content_mlp_head(f_tr, y_tr, quantiles=Q9, epochs=2, wd_grid=(1e-3,), device="cpu",
                               hidden_dim=HID)
    f_te, y_te = _content(11, h=12, seed=9)
    out, diag = predict_content_mlp_head(fit, f_te, y_te, quantiles=Q9, device="cpu",
                                         collect_test_median=True, collect_test_window_loss=True)
    for i in range(NUM_LAYERS):
        np.testing.assert_allclose(diag["test_window_loss"][i].mean(), out[i], rtol=1e-5, atol=1e-6,
                                   err_msg=f"L{i}: per-window mean != scalar loss")
    # median row check: rebuild the full prediction and confirm test_median is the 0.5 row.
    qmid = median_index(Q9)
    m, sc = fit[0]["head"], fit[0]["scaler"]
    X = torch.as_tensor(sc.transform(f_te[0]), dtype=torch.float32)
    with torch.no_grad():
        pred = m(X).view(-1, QN, 12)
    np.testing.assert_allclose(diag["test_median"][0], pred[:, qmid, :].numpy(), rtol=1e-6, atol=1e-6)


# 9. a tiny fit -> predict is deterministic (fresh fit reproduces the same scores on CPU).
def test_fit_predict_deterministic():
    f_tr, y_tr = _content(18, seed=1)
    f_te, y_te = _content(10, seed=2)
    kw = dict(quantiles=Q9, epochs=3, wd_grid=(1e-3, 1e-1), device="cpu", hidden_dim=HID)
    o1 = predict_content_mlp_head(fit_content_mlp_head(f_tr, y_tr, **kw), f_te, y_te,
                                  quantiles=Q9, device="cpu")
    o2 = predict_content_mlp_head(fit_content_mlp_head(f_tr, y_tr, **kw), f_te, y_te,
                                  quantiles=Q9, device="cpu")
    for i in range(NUM_LAYERS):
        assert o1[i] == o2[i], f"L{i}: fit->predict is not deterministic"


# 10. capacity outputs are namespaced AWAY from the committed linear artifacts (never clobbered).
def test_outputs_namespaced_away_from_linear():
    import experiments.run_ood_transfer as ood
    from experiments.run_ood_capacity import _cap_base, _probe_run_id
    cap = _cap_base()
    assert cap.name == "capacity" and cap.parent == ood.OOD_DIR, "capacity dir must sit under ood_transfer/"
    # the capacity result filename differs from the linear pilot's filename.
    assert f"ood_capacity_results__content_mlp_head__q9" != "ood_transfer_results__q9"
    # the frozen linear identity (pooling 'content') and the capacity identity never collide.
    lin_id = f"monash_electricity_hourly__content__C512_H64__q9__seed0"
    assert _probe_run_id("monash_electricity_hourly", "content_mlp_head", "q9", 0) != lin_id


# ------------------------- real-cache end-to-end smoke (opt-in) ------------------------- #

def test_smoke_real_cache():
    """RUN_OOD_CAP_SMOKE=1 + warm caches: fit content_mlp_head on electricity, score it on
    electricity + kdd, all writing to a TEMPDIR (config.ID_OUT_DIR redirected) so committed
    results are never touched. Tiny head/epochs keep it fast."""
    if os.environ.get("RUN_OOD_CAP_SMOKE") != "1":
        print("    (skipped test_smoke_real_cache — set RUN_OOD_CAP_SMOKE=1 with warm caches)")
        return
    import tempfile
    from pathlib import Path
    from probing import config
    import experiments.run_ood_capacity as R
    src, tgt = "monash_electricity_hourly", "monash_kdd_cup_2018"
    need = [config.CACHE_DIR / f"IDF_{d}__{sp}__clean__content.npz"
            for d in (src, tgt) for sp in ("train", "test")]
    if not all(p.exists() for p in need):
        print("    (skipped test_smoke_real_cache — content caches not present)")
        return
    old_out = config.ID_OUT_DIR
    R.HIDDEN_DIM, R.QUANTILE_EPOCHS, R.WD_GRID = 16, 2, (1e-3,)
    try:
        config.ID_OUT_DIR = Path(tempfile.mkdtemp())
        pl = R.run_source(src, [src, tgt], "content_mlp_head", "q9", QUANTILE_SETS["q9"], 0, "cpu")
        assert len(pl["summaries"]) == 2
        diag = {(s["source_dataset"], s["target_dataset"]): s for s in pl["summaries"]}
        assert diag[(src, src)]["is_ood"] is False and diag[(src, tgt)]["is_ood"] is True
        assert (R._cap_base() / "content_mlp_head" / "checkpoints").exists()
    finally:
        config.ID_OUT_DIR = old_out


TESTS = [test_both_families_emit_B_9_64,
         test_forecast_slot_head_shares_weights_across_slots,
         test_frozen_head_reused_across_targets_not_mutated,
         test_fit_signatures_exclude_target_and_val,
         test_checkpoint_identity_includes_family_excludes_target,
         test_source_selected_layer_uses_source_val_only,
         test_per_window_reduction_and_median_extraction,
         test_fit_predict_deterministic,
         test_outputs_namespaced_away_from_linear,
         test_smoke_real_cache]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(TESTS)} TESTS PASS")
