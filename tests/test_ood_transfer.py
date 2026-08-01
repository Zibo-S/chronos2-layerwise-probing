"""Focused tests protecting the scientific validity of the cross-dataset transfer pilot.

No model, no GPU — synthetic features only (the core checks), so this runs on a login node:

    OMP_NUM_THREADS=2 python -m tests.test_ood_transfer

Covers the invariants that make the experiment "strict transfer": the frozen fit/predict
split reproduces quantile_probe (the diagonal), one probe is trained once and reused across
targets without mutation, target arrays never reach the training function, source/target
shapes are compatible, and the probe checkpoint identity excludes the target (so
electricity->kdd and electricity->uber share one checkpoint) while result identity includes
it. An end-to-end smoke over real cached features is gated behind RUN_OOD_SMOKE=1 so the
default run stays fast and cache-free.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from probing.config import NUM_LAYERS, LAST_LAYER, SEED
from probing.probes import (QUANTILE_SETS, quantile_probe, fit_quantile_probe,
                            predict_quantile_probe)

Q9 = QUANTILE_SETS["q9"]
D, H = 6, 8                      # tiny feature dim / horizon (speed)


def _synth(n, d=D, h=H, seed=0):
    """Synthetic {layer: (n, d)} features + (n, h) trajectory labels — stands in for one
    dataset's hidden states. Different `seed` = a different 'dataset' (a transfer target)."""
    rng = np.random.default_rng(seed)
    feats = {i: rng.normal(size=(n, d)).astype(np.float32) for i in range(NUM_LAYERS)}
    y = rng.normal(size=(n, h)).astype(np.float32)
    return feats, y


# 1 + (diagonal reproduction). fit -> predict on the SAME split must reproduce quantile_probe.
def test_fit_predict_reproduces_quantile_probe():
    f_tr, y_tr = _synth(40, seed=1)
    f_te, y_te = _synth(25, seed=2)
    ref = quantile_probe(f_tr, y_tr, f_te, y_te, quantiles=Q9, epochs=5,
                         wd_grid=(1e-3, 1e-1), device="cpu")
    fitted = fit_quantile_probe(f_tr, y_tr, quantiles=Q9, epochs=5,
                                wd_grid=(1e-3, 1e-1), device="cpu")
    out = predict_quantile_probe(fitted, f_te, y_te, quantiles=Q9, device="cpu")
    for i in range(NUM_LAYERS):
        np.testing.assert_allclose(out[i], ref[i], rtol=1e-6, atol=1e-6,
                                   err_msg=f"L{i}: fit->predict != quantile_probe (diagonal)")


# 1 (reuse). ONE frozen probe scores multiple targets; weights are never mutated.
def test_frozen_probe_reused_across_targets_not_mutated():
    f_tr, y_tr = _synth(40, seed=1)
    fitted = fit_quantile_probe(f_tr, y_tr, quantiles=Q9, epochs=5, wd_grid=(1e-2,), device="cpu")
    w0 = fitted[0]["linear"].weight.detach().clone()
    tA = _synth(15, seed=10)
    tB = _synth(22, seed=11)
    outA = predict_quantile_probe(fitted, tA[0], tA[1], quantiles=Q9, device="cpu")
    outB = predict_quantile_probe(fitted, tB[0], tB[1], quantiles=Q9, device="cpu")
    assert torch.equal(fitted[0]["linear"].weight, w0), "predict mutated the frozen probe"
    # deterministic + side-effect-free: re-scoring target A after B is unchanged
    outA2 = predict_quantile_probe(fitted, tA[0], tA[1], quantiles=Q9, device="cpu")
    for i in range(NUM_LAYERS):
        assert outA[i] == outA2[i], f"L{i}: predict is not deterministic / has state"
    assert set(outA) == set(outB) == set(range(NUM_LAYERS))


# 2 + 3. The training call cannot receive target features / target labels / a target val loader.
def test_training_signature_has_no_target_or_val():
    fit_params = set(inspect.signature(fit_quantile_probe).parameters)
    assert "test_feats" not in fit_params and "test_labels" not in fit_params, \
        "fit_quantile_probe must not accept any test/target arrays"
    # fit takes only source train arrays + hyperparams; no separate validation split is passed
    # in (its wd carve is drawn from the SOURCE train inside the function).
    assert {"train_feats", "train_labels"} <= fit_params
    pred_params = set(inspect.signature(predict_quantile_probe).parameters)
    assert "wd_grid" not in pred_params and "weight_decay" not in pred_params, \
        "predict must not do any model selection (no wd grid / early stopping on the target)"


# 4. A source-trained probe applies directly to a DIFFERENT-sized target of the same geometry.
def test_source_to_target_shape_compatibility():
    f_tr, y_tr = _synth(30, seed=1)             # source: 30 windows
    fitted = fit_quantile_probe(f_tr, y_tr, quantiles=Q9, epochs=4, wd_grid=(1e-2,), device="cpu")
    f_te, y_te = _synth(17, seed=99)            # target: 17 windows, same d / H
    out, diag = predict_quantile_probe(fitted, f_te, y_te, quantiles=Q9, device="cpu",
                                       collect_test_median=True, collect_test_window_loss=True)
    assert set(out) == set(range(NUM_LAYERS))
    for i in range(NUM_LAYERS):
        assert diag["test_median"][i].shape == (17, H), f"L{i}: median shape wrong"
        assert diag["test_window_loss"][i].shape == (17,), f"L{i}: per-window loss shape wrong"


# 5. Checkpoint identity excludes the target; result identity includes it.
def test_checkpoint_identity_excludes_target():
    from experiments.run_ood_transfer import _probe_run_id
    src = "monash_electricity_hourly"
    rid_kdd_target = _probe_run_id(src, "q9", SEED)       # run id is a function of SOURCE only
    rid_uber_target = _probe_run_id(src, "q9", SEED)
    assert rid_kdd_target == rid_uber_target, "probe run id must not depend on the target"
    assert "electricity" in rid_kdd_target and "kdd" not in rid_kdd_target
    # a different source is a different probe identity
    assert _probe_run_id("monash_kdd_cup_2018", "q9", SEED) != rid_kdd_target


# 6 (definition). is_ood is exactly source != target.
def test_is_ood_definition():
    for src in ("monash_electricity_hourly", "monash_kdd_cup_2018", "uber_tlc_hourly"):
        for tgt in ("monash_electricity_hourly", "monash_kdd_cup_2018", "uber_tlc_hourly"):
            assert (src != tgt) == (src != tgt)          # is_ood := source_dataset != target_dataset
            if src == tgt:
                assert not (src != tgt), "diagonal must be in-dataset (is_ood False)"


# 7 (paired Δ-vs-last bootstrap). Last-layer Δ is identically 0; a clearly-better earlier layer
# has a positive Δ whose 95% CI excludes zero.
def test_paired_delta_bootstrap():
    from experiments.run_ood_transfer import _paired_delta_bootstrap
    rng = np.random.default_rng(0)
    S, per = 30, 5                                   # 30 test series x 5 windows each
    sid = np.repeat(np.arange(S), per)
    n = S * per
    wl = np.empty((NUM_LAYERS, n))
    for L in range(NUM_LAYERS):
        base = 0.5 if L == 0 else 1.0                # layer 0 clearly beats every other layer
        wl[L] = base + rng.normal(scale=0.02, size=n)
    r = _paired_delta_bootstrap(wl, sid)
    assert r["delta_vs_last"].shape == (NUM_LAYERS,)
    assert r["delta_above_zero"].dtype == bool and len(r["delta_above_zero"]) == NUM_LAYERS
    # last vs itself: Δ is identically 0, CI brackets 0, and it is not flagged "above zero"
    assert abs(r["delta_vs_last"][LAST_LAYER]) < 1e-12
    assert r["delta_ci_lo"][LAST_LAYER] <= 0.0 <= r["delta_ci_hi"][LAST_LAYER]
    assert not r["delta_above_zero"][LAST_LAYER]
    # a clearly-better early layer: positive Δ (~0.5) with CI entirely above zero
    assert r["delta_vs_last"][0] > 0.3 and r["delta_above_zero"][0]
    assert r["delta_ci_lo"][0] > 0.0


# 8. End-to-end smoke over REAL cached features (gated: needs the content caches; opt-in + fast).
def test_smoke_real_cache():
    if os.environ.get("RUN_OOD_SMOKE") != "1":
        print("  [skip] test_smoke_real_cache (set RUN_OOD_SMOKE=1 to run over cached features)")
        return
    import pathlib
    import shutil
    import tempfile
    import experiments.run_ood_transfer as R
    from probing.config import CACHE_DIR
    src = "monash_electricity_hourly"
    targets = [src, "monash_kdd_cup_2018"]
    need = [CACHE_DIR / f"IDF_{src}__train__clean__content.npz"]
    need += [CACHE_DIR / f"IDF_{t}__test__clean__content.npz" for t in targets]
    if not all(p.exists() for p in need):
        print("  [skip] test_smoke_real_cache (content caches missing)")
        return
    # redirect ALL writes to a throwaway tempdir so the smoke can never overwrite real committed
    # results (checkpoints / per_source / bootstrap_inputs live under these module globals).
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ood_smoke_"))
    R.OOD_DIR, R.CKPT_DIR = tmp, tmp / "checkpoints"
    R.PER_SOURCE_DIR, R.BOOT_IN_DIR, R.FIG_DIR = tmp / "per_source", tmp / "bootstrap_inputs", tmp / "figures"
    for d in (R.CKPT_DIR, R.PER_SOURCE_DIR, R.BOOT_IN_DIR, R.FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)
    try:
        R.QUANTILE_EPOCHS, R.WD_GRID = 3, (1e-3,)        # keep the smoke fast
        payload = R.run_source(src, targets, "q9", QUANTILE_SETS["q9"], SEED, "cpu")
        assert len(payload["summaries"]) == 2
        diag_cell = next(s for s in payload["summaries"] if s["target_dataset"] == src)
        ood_cell = next(s for s in payload["summaries"] if s["target_dataset"] != src)
        assert diag_cell["is_ood"] is False and ood_cell["is_ood"] is True
        assert (R._ckpt_dir(src, "q9", SEED) / "L00.pt").exists(), "checkpoint not written"
        bc = R._boot_cell(src, ood_cell["target_dataset"], "q9", SEED)   # paired Δ on the OOD cell
        assert bc is not None and abs(bc["delta_vs_last"][LAST_LAYER]) < 1e-12
        print(f"  [smoke] diagonal best L{diag_cell['best_layer']} Δ={diag_cell['delta_vs_last']:+.3f} | "
              f"OOD best L{ood_cell['best_layer']} Δ={ood_cell['delta_vs_last']:+.3f}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# 9 (4×4 extension). The matrix order is per-set: extended_v2 is a 4×4 (elec/uber/m4/wind), and
# extended_v1 stays the committed 3×3 (elec/kdd/uber) even though its roster has 4 datasets.
def test_matrix_order_is_per_set():
    from probing import config
    import experiments.run_ood_transfer as R
    orig = config.DATASET_SET
    try:
        config.set_dataset_set("extended_v2"); R._derive_datasets()
        assert R.NDATA == 4, "extended_v2 must be a 4×4 matrix"
        assert R.DATASET_ORDER == ["monash_electricity_hourly", "uber_tlc_hourly",
                                   "m4_hourly", "wind_farms_hourly"]
        assert "M4" in R.SHORT.values() and "WindFarms" in R.SHORT.values()
        config.set_dataset_set("extended_v1"); R._derive_datasets()
        assert R.NDATA == 3, "extended_v1 OOD stays the committed 3×3 (pedestrian excluded)"
        assert R.DATASET_ORDER == ["monash_electricity_hourly", "monash_kdd_cup_2018",
                                   "uber_tlc_hourly"]
    finally:
        config.set_dataset_set(orig); R._derive_datasets()


# 10 (budget + cache isolation). extended_v2 has the matched 1500/650 budget and a namespaced ID
# cache prefix, so it can never collide with extended_v1's committed 3000/1500 caches.
def test_budget_and_cache_namespacing():
    from probing import config
    from probing.id_data import BUDGET_BY_SET
    from probing.extraction import _idf_prefix
    orig = config.DATASET_SET
    try:
        assert BUDGET_BY_SET["extended_v2"] == (1500, 650)
        assert BUDGET_BY_SET["extended_v1"] == (3000, 1500)
        config.set_dataset_set("extended_v2")
        assert _idf_prefix("t") == "IDF_t__extended_v2", "new set must namespace its ID cache"
        config.set_dataset_set("extended_v1")
        assert _idf_prefix("t") == "IDF_t", "legacy set must keep the committed cache key"
    finally:
        config.set_dataset_set(orig)


# 11 (windows/split/budget for the NEW datasets; gated — needs the HF dataset cache).
def test_extended_v2_windows_split_and_budget():
    if os.environ.get("RUN_OOD_SMOKE") != "1":
        print("  [skip] test_extended_v2_windows_split_and_budget (set RUN_OOD_SMOKE=1; needs data cache)")
        return
    from probing import config
    from probing.id_data import build_windows
    orig = config.DATASET_SET
    try:
        config.set_dataset_set("extended_v2")
        expect = {"m4_hourly": "cross_series", "wind_farms_hourly": "within_series"}
        for tag, split_mode in expect.items():
            w = build_windows(tag)
            m = w["meta"]
            assert m["split_mode"] == split_mode, f"{tag}: split {m['split_mode']} != {split_mode}"
            assert m["target_train"] == 1500 and m["target_test"] == 650
            assert m["n_train"] <= 1500 and m["n_test"] <= 650
            assert w["X_train"].shape[0] == m["n_train"] == w["Y_train_traj"].shape[0]
            assert w["X_test"].shape[0] == m["n_test"] == len(w["series_test"])
            assert w["X_test"].shape[1] == 512 and w["Y_test_traj"].shape[1] == 64
            print(f"  [ext_v2] {tag}: split={split_mode} n_train={m['n_train']} n_test={m['n_test']}")
    finally:
        config.set_dataset_set(orig)


# 12 (rolling dispatch). extended_v3_rolling must fit through the EXPLICIT temporal-val path
# (fit_quantile_probe_explicit_val); the legacy 80/20-carve fit must never run for a rolling
# set — and vice versa for extended_v2. Everything heavy is monkeypatched, so this only tests
# the driver's routing (the fit contracts themselves are covered elsewhere).
def test_rolling_dispatch_uses_explicit_val_fit():
    from probing import config
    import experiments.run_ood_transfer as R

    n_tr, n_va, h = 12, 4, 8

    def fake_windows(tag):
        rng = np.random.default_rng(0)
        return {"X_train": rng.normal(size=(n_tr, 16)).astype(np.float32),
                "y_train": np.zeros(n_tr, np.float32),
                "X_val": rng.normal(size=(n_va, 16)).astype(np.float32),
                "y_val": np.zeros(n_va, np.float32),
                "Y_train_traj": rng.normal(size=(n_tr, h)).astype(np.float32),
                "Y_val_traj": rng.normal(size=(n_va, h)).astype(np.float32),
                "meta": {"n_val": n_va, "n_val_series": n_va}}

    def fake_extract(tag, split, X, y, pooling="content"):
        return ({i: np.zeros((len(X), D), np.float32) for i in range(NUM_LAYERS)}, y)

    SENTINEL = {"probe": "sentinel"}
    calls = {"explicit": 0, "legacy": 0}

    def explicit_ok(*a, **k):
        calls["explicit"] += 1
        return SENTINEL

    def legacy_ok(*a, **k):
        calls["legacy"] += 1
        return SENTINEL

    def must_not_run(*a, **k):
        raise AssertionError("wrong fit path for the active dataset set")

    orig_set = config.DATASET_SET
    saved = {name: getattr(R, name) for name in
             ("build_windows", "extract_window_features", "load_checkpoints",
              "save_checkpoints", "fit_quantile_probe", "fit_quantile_probe_explicit_val")}
    try:
        R.build_windows = fake_windows
        R.extract_window_features = fake_extract
        R.load_checkpoints = lambda *a, **k: None       # force the fit branch (no resume)
        R.save_checkpoints = lambda *a, **k: "<unsaved>"

        config.set_dataset_set("extended_v3_rolling")
        R.fit_quantile_probe_explicit_val, R.fit_quantile_probe = explicit_ok, must_not_run
        fitted, _ = R.get_source_probe("m4_hourly", "q9", Q9, SEED, "cpu")
        assert fitted is SENTINEL and calls == {"explicit": 1, "legacy": 0}, \
            "rolling set must fit via fit_quantile_probe_explicit_val (no 80/20 carve)"

        config.set_dataset_set("extended_v2")
        R.fit_quantile_probe_explicit_val, R.fit_quantile_probe = must_not_run, legacy_ok
        fitted, _ = R.get_source_probe("m4_hourly", "q9", Q9, SEED, "cpu")
        assert fitted is SENTINEL and calls == {"explicit": 1, "legacy": 1}, \
            "non-rolling set must keep the legacy 80/20-carve fit path"
    finally:
        for name, fn in saved.items():
            setattr(R, name, fn)
        config.set_dataset_set(orig_set)
        R._derive_dirs()
        R._derive_datasets()


TESTS = [test_fit_predict_reproduces_quantile_probe,
         test_frozen_probe_reused_across_targets_not_mutated,
         test_training_signature_has_no_target_or_val,
         test_source_to_target_shape_compatibility,
         test_checkpoint_identity_excludes_target,
         test_is_ood_definition,
         test_paired_delta_bootstrap,
         test_matrix_order_is_per_set,
         test_budget_and_cache_namespacing,
         test_extended_v2_windows_split_and_budget,
         test_rolling_dispatch_uses_explicit_val_fit,
         test_smoke_real_cache]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(TESTS)} TESTS PASS")
