"""No-GPU checks for the two-axis shared forecast-token transfer driver (run_fslot_transfer).

Synthetic (n, K, d) forecast-slot features only — no model, no GPU, no HF data, no committed
checkpoints (the checkpoint round-trip builds + saves its own):

    OMP_NUM_THREADS=2 python -m tests.test_fslot_transfer

Covers the invariants the transfer experiments rest on: the 4×4 is 4 diagonal (PT-ID/Probe-ID) +
12 off-diagonal (PT-ID/Probe-OOD); off-diagonal evaluation is PREDICT-ONLY (no fit / scaler-fit
touches target data); a frozen checkpoint predicts identically to the in-memory source probe; the
PT-OOD experiment is 4 PT-ID sources × 3 PT-OOD targets; every curve has 14 depths with L12+LN as the
reference; the source tunnel / selected layer come only from source validation; records + figure
labels never use a bare "ID"/"OOD" token; and the content-probe pipeline is unchanged.
"""

from __future__ import annotations

import math
import os
import pathlib
import tempfile

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from probing.config import NUM_LAYERS, OUTPUT_PATCH_SIZE
from probing.probes import (QUANTILE_SETS, fit_quantile_probe, fit_shared_forecast_probe_explicit_val,
                            predict_forecast_slot_native_head, predict_quantile_probe,
                            predict_shared_forecast_probe, quantile_probe)
from probing.tunnel import PT_ID_TAGS, PT_OOD_TAGS
import probing.probes as pp
import experiments.run_fslot_transfer as t
import experiments.run_ptood_probing_ftok as ftok

MLP = ftok.PROBE_FAMILIES["native_mlp"]      # native-structure nonlinear head family (transfer)

Q9 = QUANTILE_SETS["q9"]
D = 6                                        # tiny feature dim (real d = 768)
P = int(OUTPUT_PATCH_SIZE)                   # model output patch size = predict's default
H = P + 8                                    # forces multi-slot + trim
K = math.ceil(H / P)                         # slot count -> 2
C = 64                                       # context length (> m=24 for the MASE denominator)
N_POINTS = NUM_LAYERS + 1                    # 14 fslot readout points


def _slot_feats(n, seed, n_points=N_POINTS):
    rng = np.random.default_rng(seed)
    return {i: rng.normal(size=(n, K, D)).astype(np.float32) for i in range(n_points)}


def _target_window(n, seed):
    """A synthetic build_windows-shaped test dict (only the fields the transfer path reads)."""
    rng = np.random.default_rng(seed)
    return {"X_test": rng.normal(size=(n, C)).astype(np.float32),
            "y_test": rng.normal(size=(n,)).astype(np.float32),
            "Y_test_traj": rng.normal(size=(n, H)).astype(np.float32),
            "series_test": np.repeat(np.arange(max(1, n // 2)), 2)[:n].astype(np.int64),
            "meta": {"sigma_eps": 1e-6, "n_test": n}}


def _source_probe(seed=1):
    f_tr, f_va = _slot_feats(40, seed), _slot_feats(15, seed + 100)
    y_tr = np.random.default_rng(seed).normal(size=(40, H)).astype(np.float32)
    y_va = np.random.default_rng(seed + 1).normal(size=(15, H)).astype(np.float32)
    return fit_shared_forecast_probe_explicit_val(f_tr, y_tr, f_va, y_va, quantiles=Q9, epochs=4,
                                                  wd_grid=(1e-2,), device="cpu", output_patch_size=P)


def _use_tmp_dirs(root):
    t.OUT_DIR, t.BOOT_IN_DIR = root, root / "boot"
    t.FIG_DIR, t.TAB_DIR = root / "fig", root / "tab"
    for d in (t.OUT_DIR, t.BOOT_IN_DIR, t.FIG_DIR, t.TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)


# 1. The 4×4 is exactly 4 diagonal (PT-ID/Probe-ID) + 12 off-diagonal (PT-ID/Probe-OOD).
def test_4x4_four_diagonal_twelve_offdiagonal():
    sources = targets = list(PT_ID_TAGS)
    assert len(sources) == 4, f"expected 4 PT-ID datasets, got {sources}"
    pairs = [(s, tg) for s in sources for tg in targets]
    diag = [p for p in pairs if p[0] == p[1]]
    off = [p for p in pairs if p[0] != p[1]]
    assert len(diag) == 4 and len(off) == 12, (len(diag), len(off))
    assert all(t._quadrant(s, tg) == "PT-ID / Probe-ID" for s, tg in diag)
    assert all(t._quadrant(s, tg) == "PT-ID / Probe-OOD" for s, tg in off)
    # pt_status keys off the TARGET, which is PT-ID for every 4×4 cell
    assert all(t._pt_label(tg) == "PT-ID" for _, tg in pairs)


# 2. Off-diagonal evaluation is predict-only: no fitting / scaler-fitting runs on target data.
def test_offdiagonal_eval_is_predict_only():
    with tempfile.TemporaryDirectory() as tmp:
        _use_tmp_dirs(pathlib.Path(tmp))
        fitted = _source_probe()
        w, feats = _target_window(12, seed=7), _slot_feats(12, seed=7)
        saved = {name: getattr(pp, name) for name in
                 ("_fit_slot_scaler", "_fit_shared_forecast_linear",
                  "fit_shared_forecast_probe_explicit_val")}

        def _boom(*a, **k):
            raise AssertionError("a fitting/scaler-fitting function ran during transfer eval")
        try:
            for name in saved:
                setattr(pp, name, _boom)
            res = t.eval_cell("m4_hourly", "uber_tlc_hourly", w, feats, fitted, "q9", 0, Q9, "cpu")
        finally:
            for name, fn in saved.items():
                setattr(pp, name, fn)
    assert len(res["quantile_loss"]) == N_POINTS and len(res["mase"]) == N_POINTS


# 3. A frozen checkpoint predicts identically to the in-memory source probe.
def test_frozen_checkpoint_matches_direct_prediction():
    with tempfile.TemporaryDirectory() as tmp:
        orig_root = ftok.OUT_ROOT
        ftok.OUT_ROOT = pathlib.Path(tmp)                       # redirect _ptid_ckpt_dir
        try:
            fitted = _source_probe()
            f_te = _slot_feats(20, seed=2)
            y_te = np.random.default_rng(2).normal(size=(20, H)).astype(np.float32)
            ref = predict_shared_forecast_probe(fitted, f_te, y_te, quantiles=Q9, device="cpu",
                                                output_patch_size=P)
            ckdir = ftok._ptid_ckpt_dir("uber_tlc_hourly", "q9", 0)
            ftok._save_ckpt(ckdir, fitted)
            reloaded = ftok.load_ptid_ckpt("uber_tlc_hourly", "q9", 0, device="cpu")
            got = predict_shared_forecast_probe(reloaded, f_te, y_te, quantiles=Q9, device="cpu",
                                                output_patch_size=P)
        finally:
            ftok.OUT_ROOT = orig_root
    assert sorted(reloaded) == list(range(N_POINTS)), sorted(reloaded)
    for i in range(N_POINTS):
        np.testing.assert_allclose(got[i], ref[i], rtol=1e-6, atol=1e-7,
                                   err_msg=f"L{i}: checkpoint predict != in-memory predict")


# 4. The PT-OOD experiment is 4 PT-ID sources × 3 PT-OOD targets, every cell PT-OOD/Probe-OOD.
def test_pt_ood_is_four_by_three_all_probe_ood():
    assert len(PT_ID_TAGS) == 4 and len(PT_OOD_TAGS) == 3, (PT_ID_TAGS, PT_OOD_TAGS)
    quads = {t._quadrant(s, tg) for s in PT_ID_TAGS for tg in PT_OOD_TAGS}
    assert quads == {"PT-OOD / Probe-OOD"}, quads          # no source is ever a PT-OOD target
    assert not hasattr(t, "PT_OOD_SOURCE")                 # single-source constant retired


# 5. Every curve has 14 depths and uses L12+LN as the reference.
def test_curves_have_fourteen_depths_ref_post_ln():
    assert N_POINTS == 14 and len(t.LAYER_LABELS) == 14
    assert t.LAYER_LABELS[-1] == "L12+LN" and t.REF_LABEL == t.LAYER_LABELS[-1]
    with tempfile.TemporaryDirectory() as tmp:
        _use_tmp_dirs(pathlib.Path(tmp))
        fitted = _source_probe()
        w, feats = _target_window(10, seed=3), _slot_feats(10, seed=3)
        res = t.eval_cell("uber_tlc_hourly", "m4_hourly", w, feats, fitted, "q9", 0, Q9, "cpu")
        assert len(res["quantile_loss"]) == N_POINTS and len(res["mase"]) == N_POINTS
    # the reference is the LAST point (index NUM_LAYERS), i.e. the post-final-LN slots
    assert t.LAYER_LABELS.index(t.REF_LABEL) == NUM_LAYERS


# 6. The source tunnel entrance + the selected layer come only from source VALIDATION.
def test_selected_layer_and_tunnel_from_source_validation_only():
    # a record whose validation argmin (L3) differs from a decoy test argmin (L9)
    val = [9, 8, 7, 1.0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]         # argmin at layer 3
    rec = {"l_start": 5, "mean_val_loss_by_layer": val,
           "mean_test_loss_by_layer": [1.0 if i == 9 else 5 for i in range(N_POINTS)]}
    assert t._val_selected_layer(rec) == 3, "selected layer must be the source-VAL argmin"
    # _write_records must stamp l_start_sustained from the source tunnel record, not recompute it
    with tempfile.TemporaryDirectory() as tmp:
        _use_tmp_dirs(pathlib.Path(tmp))
        s = "uber_tlc_hourly"
        cells = {(s, s): {"ql": [[1.0] * N_POINTS] * 2, "mase": [[0.5] * N_POINTS] * 2}}
        mean = {(s, s): np.ones(N_POINTS)}
        t._write_records([s], [s], cells, mean, mean, {s: 3}, {s: rec},
                         gap={(s, s): 0.0}, qset="q9", meta_cell={(s, s): {"n_test": 4, "n_clusters": 2}})
        import json
        rows = json.load(open(t.TAB_DIR / "transfer_by_layer__4x4__q9.json"))
    assert all(r["l_start_sustained"] == 5 for r in rows)
    assert all(r["tunnel_defined_on"] == f"{s}:validation" for r in rows)
    assert all(r["wd_selected_on"] == f"{s}:validation" for r in rows)


# 7. Records + figure labels never use a bare "ID"/"OOD" token.
def test_no_bare_id_ood_labels():
    # label helpers used in every figure title + record cell
    for s in PT_ID_TAGS:
        for tg in list(PT_ID_TAGS) + list(PT_OOD_TAGS):
            assert t._pt_label(tg) in ("PT-ID", "PT-OOD")
            assert t._probe_label(s, tg) in ("Probe-ID", "Probe-OOD")
            assert " / " in t._quadrant(s, tg)
    with tempfile.TemporaryDirectory() as tmp:
        _use_tmp_dirs(pathlib.Path(tmp))
        s = "m4_hourly"
        cells = {(s, s): {"ql": [[1.0] * N_POINTS], "mase": [[0.5] * N_POINTS]}}
        mean = {(s, s): np.ones(N_POINTS)}
        rec = {"l_start": 2, "mean_val_loss_by_layer": list(range(1, N_POINTS + 1))}
        t._write_records([s], [s], cells, mean, mean, {s: 0}, {s: rec},
                         gap={(s, s): 0.0}, qset="q9", meta_cell={(s, s): {"n_test": 2, "n_clusters": 1}})
        import json
        rows = json.load(open(t.TAB_DIR / "transfer_by_layer__4x4__q9.json"))
    banned = {"ID", "OOD"}
    for r in rows:
        for v in r.values():
            assert v not in banned, f"bare label {v!r} in record {r}"
    assert all(r["pt_status"] == "PT-ID" and r["probe_status"] == "Probe-ID" for r in rows)


# 8. The content-probe pipeline is unchanged: fit->predict still reproduces the combined quantile
#    probe, and run_ood_transfer still reads the content pooling.
def test_content_probe_pipeline_unchanged():
    import experiments.run_ood_transfer as ood
    assert ood.POOLING == "content"
    f_tr = {i: np.random.default_rng(i).normal(size=(30, D)).astype(np.float32) for i in range(NUM_LAYERS)}
    f_te = {i: np.random.default_rng(i + 50).normal(size=(16, D)).astype(np.float32) for i in range(NUM_LAYERS)}
    y_tr = np.random.default_rng(0).normal(size=(30, H)).astype(np.float32)
    y_te = np.random.default_rng(1).normal(size=(16, H)).astype(np.float32)
    ref = quantile_probe(f_tr, y_tr, f_te, y_te, quantiles=Q9, epochs=4, wd_grid=(1e-2,), device="cpu")
    fitted = fit_quantile_probe(f_tr, y_tr, quantiles=Q9, epochs=4, wd_grid=(1e-2,), device="cpu")
    out = predict_quantile_probe(fitted, f_te, y_te, quantiles=Q9, device="cpu")
    for i in range(NUM_LAYERS):
        np.testing.assert_allclose(out[i], ref[i], rtol=1e-5, atol=1e-6,
                                   err_msg=f"L{i}: content fit->predict != quantile_probe (pipeline changed)")


# ================= native-MLP family transfer-level checks (§10) ================= #
def _mlp_source_probe(seed=1):
    f_tr, f_va = _slot_feats(40, seed), _slot_feats(15, seed + 100)
    y_tr = np.random.default_rng(seed).normal(size=(40, H)).astype(np.float32)
    y_va = np.random.default_rng(seed + 1).normal(size=(15, H)).astype(np.float32)
    return MLP.fit(f_tr, y_tr, f_va, y_va, quantiles=Q9, epochs=4, wd_grid=(1e-2,), device="cpu",
                   init_seed=seed)


# 9. MLP checkpoint round-trip: save -> reload reproduces the in-memory frozen predictions exactly.
def test_mlp_checkpoint_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        orig = ftok.MLP_ROOT
        ftok.MLP_ROOT = pathlib.Path(tmp)                       # redirect _mlp_ckpt_dir
        try:
            fitted = _mlp_source_probe()
            f_te = _slot_feats(18, seed=2)
            y_te = np.random.default_rng(2).normal(size=(18, H)).astype(np.float32)
            ref = predict_forecast_slot_native_head(fitted, f_te, y_te, quantiles=Q9, device="cpu",
                                                    output_patch_size=P)
            ck = ftok._mlp_ckpt_dir("uber_tlc_hourly", "q9", 0)
            MLP.save_ckpt(ck, fitted)
            reloaded = MLP.load_ckpt("uber_tlc_hourly", "q9", 0, device="cpu")
            got = predict_forecast_slot_native_head(reloaded, f_te, y_te, quantiles=Q9, device="cpu",
                                                    output_patch_size=P)
        finally:
            ftok.MLP_ROOT = orig
    assert sorted(reloaded) == list(range(N_POINTS)), sorted(reloaded)
    for i in range(N_POINTS):
        np.testing.assert_allclose(got[i], ref[i], rtol=1e-6, atol=1e-7,
                                   err_msg=f"L{i}: MLP checkpoint predict != in-memory predict")


# 10. MLP off-diagonal evaluation is predict-only: no fit / scaler-fit runs on target data.
def test_mlp_offdiagonal_eval_is_predict_only():
    saved_family = t.FAMILY
    with tempfile.TemporaryDirectory() as tmp:
        _use_tmp_dirs(pathlib.Path(tmp))
        t.FAMILY = MLP
        fitted = _mlp_source_probe()
        w, feats = _target_window(12, seed=7), _slot_feats(12, seed=7)
        patched = {name: getattr(pp, name) for name in
                   ("_fit_slot_scaler", "_fit_forecast_slot_head",
                    "fit_forecast_slot_native_head_explicit_val")}

        def _boom(*a, **k):
            raise AssertionError("a fitting/scaler-fitting function ran during MLP transfer eval")
        try:
            for name in patched:
                setattr(pp, name, _boom)
            res = t.eval_cell("m4_hourly", "uber_tlc_hourly", w, feats, fitted, "q9", 0, Q9, "cpu")
        finally:
            for name, fn in patched.items():
                setattr(pp, name, fn)
            t.FAMILY = saved_family
    assert len(res["quantile_loss"]) == N_POINTS and len(res["mase"]) == N_POINTS


# 11. MLP artifact paths are separate from linear; feature caches are family-shared (no artifact tag).
def test_mlp_paths_separate_features_shared():
    lin = ftok.PROBE_FAMILIES["shared_linear"]
    lck, mck = lin.ckpt_dir("uber_tlc_hourly", "q9", 0), MLP.ckpt_dir("uber_tlc_hourly", "q9", 0)
    ltun, mtun = lin.tunnel_path("uber_tlc_hourly", "q9"), MLP.tunnel_path("uber_tlc_hourly", "q9")
    assert "fslot_mlp" in str(mck) and "fslot_mlp" in str(mtun), "MLP artifacts must be under fslot_mlp/"
    assert "fslot_mlp" not in str(lck) and "fslot_mlp" not in str(ltun), "linear paths must not move"
    assert mck != lck and mtun != ltun, "MLP and linear artifact paths must be disjoint"
    from probing import config as _cfg
    _cfg.set_dataset_set("extended_v3_rolling")
    saved = t.FAMILY
    try:
        t.FAMILY = MLP
        t.preflight_feature_cache(["uber_tlc_hourly"], rolling_ood=False)   # asserts cache is shared
    finally:
        t.FAMILY = saved


# 12. The diagonal transfer gap is 0 by construction (gap = L_{s->t}(ℓ_s)/L_{t->t}(ℓ_t) − 1, s==t).
def test_diagonal_transfer_gap_is_zero():
    sources = list(PT_ID_TAGS)
    rng = np.random.default_rng(0)
    mean_ql = {(s, tg): rng.uniform(1, 3, size=N_POINTS) for s in sources for tg in sources}
    ell = {s: int(rng.integers(0, N_POINTS)) for s in sources}
    for s in sources:                                          # the run_4x4 gap formula on the diagonal
        Ls = float(mean_ql[(s, s)][ell[s]])
        Lt = float(mean_ql[(s, s)][ell[s]])
        assert abs(Ls / Lt - 1.0) < 1e-12, "diagonal transfer gap must be exactly 0"


# 13. Each source row uses ONE source-defined tunnel across all its targets (stamped identically).
def test_one_source_tunnel_across_row():
    with tempfile.TemporaryDirectory() as tmp:
        _use_tmp_dirs(pathlib.Path(tmp))
        s, tgts = "monash_electricity_hourly", list(PT_ID_TAGS)
        rec = {"l_start": 4, "mean_val_loss_by_layer": list(range(1, N_POINTS + 1))}
        cells = {(s, tg): {"ql": [[1.0] * N_POINTS], "mase": [[0.5] * N_POINTS]} for tg in tgts}
        mean = {(s, tg): np.ones(N_POINTS) for tg in tgts}
        meta = {(s, tg): {"n_test": 2, "n_clusters": 1} for tg in tgts}
        t._write_records([s], tgts, cells, mean, mean, {s: 0}, {s: rec},
                         gap={(s, tg): 0.0 for tg in tgts}, qset="q9", meta_cell=meta)
        import json
        rows = json.load(open(t.TAB_DIR / "transfer_by_layer__4x4__q9.json"))
    by_src = [r for r in rows if r["source_dataset"] == s]
    assert by_src and all(r["l_start_sustained"] == 4 for r in by_src), "row must share one tunnel entrance"
    assert len({r["tunnel_defined_on"] for r in by_src}) == 1, "row must share one tunnel provenance"


if __name__ == "__main__":
    tests = [test_4x4_four_diagonal_twelve_offdiagonal,
             test_offdiagonal_eval_is_predict_only,
             test_frozen_checkpoint_matches_direct_prediction,
             test_pt_ood_is_four_by_three_all_probe_ood,
             test_curves_have_fourteen_depths_ref_post_ln,
             test_selected_layer_and_tunnel_from_source_validation_only,
             test_no_bare_id_ood_labels,
             test_content_probe_pipeline_unchanged,
             test_mlp_checkpoint_round_trip,
             test_mlp_offdiagonal_eval_is_predict_only,
             test_mlp_paths_separate_features_shared,
             test_diagonal_transfer_gap_is_zero,
             test_one_source_tunnel_across_row]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(tests)} fslot-transfer tests passed.")
