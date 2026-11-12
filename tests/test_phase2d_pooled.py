"""PRE-RUN CONTRACT SUITE for Phase 2d (content-mean-pooled vs architecture-aware readouts).

Model-free: no checkpoint, no GPU, no feature cache. Groups 01, 10 and 17 read the COMMITTED
Phase-1 artifacts in results/ (all 42 cells), so the reproduction gate and the LOOP protocol are
checked on the real numbers, not on fixtures. Runs on a login node with 2 threads:

    OMP_NUM_THREADS=2 python -m tests.test_phase2d_pooled

The model-backed one-batch checks (all three models) run in the smoke job:
    python -m experiments.run_phase2d_pooled --smoke --device cuda

Numbered so a failure names the requirement it broke.
"""

from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2, registry                                      # noqa: E402
from probing import phase2d_pooled as p2d                                         # noqa: E402

torch.set_num_threads(2)
Q9, MED = phase1.quantile_set("q9")
FAST = {"epochs": 4, "lr": 1e-2, "wd_grid": (1e-3, 1e-1), "seed": 0}

#: The frozen Phase-1 sustained-5% entrances (notes/PLAN.md, Phase 2 section; LOOP from the
#: lr=1e-3 rerun). Group 10 re-derives every one of them from the committed artifacts.
FROZEN = {
    "m4_hourly": ("L8", "L16", "L12"), "monash_electricity_hourly": ("L9", "L15", "L11"),
    "uber_tlc_hourly": ("L4", "L1", "L10"), "wind_farms_hourly": ("L3", "Emb", "L11"),
    "sg_carpark": ("L10", "L12", "L11"), "coastal_ts": ("L1", "L5", "L1"),
    "boom_hourly": ("L3", "Emb", "L8"), "LOOP_SEATTLE_5T": ("L9", "L14", "L9"),
    "electricity_15min": ("L10", "L1", "L10"), "SZ_TAXI_15T": ("L2", "L1", "Emb"),
    "monash_london_smart_meters": ("L3", "L1", "L8"), "m5": ("Emb", "Emb", "Emb"),
    "wiki_daily_100k": ("L2", "Emb", "L1"), "monash_traffic": ("L11", "L12", "L12")}


def _raises(fn, exc=Exception) -> bool:
    try:
        fn()
    except exc:
        return True
    return False


def _synthetic(model, n=(40, 20, 20), d=16, K=None, seed=0):
    """Spec-shaped synthetic features {i: (n, K, d)} with a learnable target."""
    spec = phase1.model_spec(model)
    K = K or p2d.NATIVE_K[model]
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(K * d, phase1.PHASE1_H)) / np.sqrt(K * d)
    out = {}
    for split, m in zip(p2d.SPLITS, n):
        base = rng.normal(size=(m, K, d)).astype(np.float32)
        feats = {i: (base + 0.1 * (i + 1) * rng.normal(size=base.shape)).astype(np.float32)
                 for i in range(spec.n_points)}
        Y = (base.reshape(m, -1) @ W + 0.1 * rng.normal(size=(m, phase1.PHASE1_H)))
        out[split] = {"feats": feats, "Y": Y.astype(np.float32)}
    return out


def _real_cell(model, tag):
    root = phase2.resolve_phase1_root(tag, phase2.DEFAULT_PHASE1_ROOT)
    return phase2.Phase1Cell(root, model, tag)


# --------------------------------------------------------------------------- #
def test_01_parameter_counts():
    want = {("chronos2", "native"): 110_736, ("chronos2", "pooled"): 442_944,
            ("chronos2", "native_mean"): 442_944, ("chronos2", "concat"): 1_770_048,
            ("timesfm3", "native"): 737_856, ("timesfm3", "pooled"): 737_856,
            ("tirex", "native"): 147_744, ("tirex", "pooled"): 295_488,
            ("tirex", "native_mean"): 295_488, ("tirex", "concat"): 590_400}
    got = {(r["model"], r["arm"]): r["n_params"] for r in p2d.param_table()}
    assert got == want, got
    ratio = {(r["model"], r["arm"]): round(r["ratio_to_native"], 2) for r in p2d.param_table()}
    assert ratio[("chronos2", "concat")] == 15.98 and ratio[("timesfm3", "pooled")] == 1.0
    assert _raises(lambda: p2d.readout_params("timesfm3", "concat"), ValueError)
    # the documented native count IS the committed Phase-1 probe
    for m in phase1.MODELS:
        nat = p2d.native_from_phase1(_real_cell(m, "monash_electricity_hourly"),
                                     phase1.model_spec(m))
        assert set(nat["n_params"]) == {want[(m, "native")]}, (m, set(nat["n_params"]))
    # and the MEASURED count of every fitted arm matches (real d, real K)
    for m in ("chronos2", "tirex"):
        spec = phase1.model_spec(m)
        data = _synthetic(m, n=(12, 6, 6), d=spec.d)
        for arm in p2d.ARMS_BY_MODEL[m]:
            arm_data = {s: {"feats": p2d.arm_features(
                arm, pooled={i: v[:, 0] for i, v in data[s]["feats"].items()},
                native_states=data[s]["feats"]), "Y": data[s]["Y"]} for s in p2d.SPLITS}
            res = p2d.fit_shared_layerwise(m, arm_data, quantiles=Q9, median_idx=MED,
                                           device="cpu", epochs=1, lr=1e-2, wd_grid=(1e-3,),
                                           verbose=False)
            assert set(res["n_params"]) == {want[(m, arm)]}, (m, arm, set(res["n_params"]))
    from probing.timesfm3_last_token_probes import make_probe
    lin = make_probe(64, 9)
    assert sum(p.numel() for p in lin.parameters()) == want[("timesfm3", "pooled")]
    print(" 01  readout params: C2 110,736 / 442,944 / 1,770,048; TFM3 737,856 x2; TiRex 147,744 /"
          " 295,488 / 590,400 -- documented == measured == committed Phase-1        OK")


def test_02_token_masks():
    X = np.random.default_rng(0).normal(size=(5, 512)).astype(np.float32)
    m = p2d.chronos2_token_mask(X)
    assert m.shape == (5, 37) and m[:, :32].all() and not m[:, 32:].any()
    p2d.assert_token_positions(m, range(32), "c2")
    m = p2d.timesfm3_token_mask(X)
    assert m.shape == (5, 18) and m[:, :16].all() and not m[:, 16:].any()
    # TiRex: the context exactly as the model pads it (left NaN to 2048; pass 1 appends 32 NaN)
    pad0 = np.concatenate([np.full((5, 1536), np.nan, np.float32), X], axis=1)
    pad1 = np.concatenate([np.full((5, 1504), np.nan, np.float32), X,
                           np.full((5, 32), np.nan, np.float32)], axis=1)
    m0, m1 = p2d.tirex_token_mask(pad0), p2d.tirex_token_mask(torch.as_tensor(pad1))
    assert p2d.assert_token_positions(m0, range(48, 64), "p0")["first_valid_token"] == 48
    assert p2d.assert_token_positions(m1, range(47, 63), "p1")["last_valid_token"] == 62
    # a missing value inside the context invalidates that patch -- and the gate refuses it
    Xn = X.copy()
    Xn[2, 100] = np.nan
    assert not p2d.chronos2_token_mask(Xn)[2, 100 // 16]
    assert not p2d.timesfm3_token_mask(Xn)[2, 100 // 32]
    assert _raises(lambda: p2d.assert_token_positions(p2d.chronos2_token_mask(Xn), range(32),
                                                      "c2"), RuntimeError)
    print(" 02  valid tokens: C2 0..31 (REG + 4 slots out), TFM3 0..15 (2 placeholders out), "
          "TiRex pass0 48..63 / pass1 47..62; a NaN patch is refused             OK")


def test_03_masked_mean():
    rng = np.random.default_rng(1)
    h = rng.normal(size=(4, 10, 6))
    mask = np.zeros((4, 10), bool)
    mask[:, 3:9] = True
    h[~mask] = 1e9                                       # padding must not leak at all
    want = h[:, 3:9].mean(axis=1)
    got_np = p2d.masked_mean(h, mask)
    got_t = p2d.masked_mean(torch.as_tensor(h, dtype=torch.float32), torch.as_tensor(mask))
    assert got_np.dtype == np.float32 and np.allclose(got_np, want, rtol=1e-6)
    assert np.allclose(got_t.numpy(), want, rtol=1e-5)
    mask[1] = False
    assert _raises(lambda: p2d.masked_mean(h, mask), RuntimeError)
    print(" 03  masked mean: excludes padding exactly (1e9 pad does not leak), float64 "
          "accumulation, torch == numpy, empty window refused                    OK")


def test_04_no_future_input():
    parent = np.zeros((2, 512 + 64), np.float32)
    assert _raises(lambda: p2d.assert_contexts(parent), ValueError)
    geom_t = types.SimpleNamespace(C=512)
    assert _raises(lambda: p2d.extract_timesfm3_pooled(parent, model=None, device="cpu",
                                                       geom=geom_t, batch_size=2), ValueError)
    geom_x = types.SimpleNamespace(C=512)
    assert _raises(lambda: p2d.extract_tirex_pooled(parent, model=None, geom=geom_x,
                                                    batch_size=2), ValueError)
    assert _raises(lambda: p2d.chronos2_hidden_states(None, parent), ValueError)
    print(" 04  every pooled extractor refuses a (n, 576) parent window before touching a model"
          "                                                                      OK")


def test_05_arm_features():
    rng = np.random.default_rng(2)
    nat = {i: rng.normal(size=(3, 4, 5)).astype(np.float32) for i in range(2)}
    pooled = {i: rng.normal(size=(3, 5)).astype(np.float32) for i in range(2)}
    c = p2d.arm_features("concat", native_states=nat)
    assert c[0].shape == (3, 1, 20)
    assert np.array_equal(c[0][:, 0, 5:10], nat[0][:, 1])           # slot-major
    mu = p2d.arm_features("native_mean", native_states=nat)
    assert np.allclose(mu[1][:, 0], nat[1].mean(axis=1), atol=1e-7)
    po = p2d.arm_features("pooled", pooled=pooled)
    assert po[0].shape == (3, 1, 5) and np.array_equal(po[0][:, 0], pooled[0])
    assert _raises(lambda: p2d.arm_features("concat", native_states={0: np.zeros((3, 5))}),
                   ValueError)
    print(" 05  arm features: concat slot-major (n,1,K*d), native mean (n,1,d), pooled (n,1,d)"
          "                                                                      OK")


def test_06_fit_parity_chronos2():
    """The Phase-2d wrapper at the NATIVE K=4, P=16 == phase1_chronos2.fit_layerwise, bitwise."""
    import probing.phase1_chronos2 as c2
    data = _synthetic("chronos2")
    ours = p2d.fit_shared_layerwise("chronos2", data, quantiles=Q9, median_idx=MED,
                                    device="cpu", verbose=False, **FAST)
    ref = c2.fit_layerwise("t", data, quantiles=Q9, median_idx=MED, device="cpu",
                           horizon=64, verbose=False, **FAST)
    for k in ("train_loss", "val_loss", "test_loss", "wd", "n_params"):
        assert ours[k] == ref[k], (k, ours[k][:3], ref[k][:3])
    assert np.array_equal(ours["test_window_loss"], ref["test_window_loss"])
    print(" 06  Chronos-2 fit wrapper == phase1_chronos2.fit_layerwise BITWISE at K=4, P=16 "
          "(train/val/test, wd, params)                                          OK")


def test_07_fit_parity_tirex():
    import probing.phase1_tirex as tx
    spec = phase1.model_spec("tirex")
    data = _synthetic("tirex")
    ours = p2d.fit_shared_layerwise("tirex", data, quantiles=Q9, median_idx=MED, device="cpu",
                                    verbose=False, **FAST)
    ref_data = {s: {"feats": {spec.labels[i]: v for i, v in d["feats"].items()},
                    "targets": d["Y"]} for s, d in data.items()}
    geom = types.SimpleNamespace(output_patch=32, n_forecast_patches=2)
    ref = tx.fit_layerwise("t", ref_data, quantiles=Q9, median_idx=MED, geom=geom,
                           device="cpu", verbose=False, **FAST)
    for k in ("train_loss", "val_loss", "test_loss", "wd", "n_params"):
        assert ours[k] == ref[k], (k, ours[k][:3], ref[k][:3])
    print(" 07  TiRex fit wrapper == phase1_tirex.fit_layerwise BITWISE at K=2, P=32       OK")


def test_08_k1_head_is_the_pooled_quantile_probe():
    """At K=1, P=64 the shared-slot fitter IS a Linear(d, Q*H) quantile probe: same init, same
    layout, same loss as probes.fit_quantile_probe_explicit_val (the committed pooled line)."""
    from probing.probes import fit_quantile_probe_explicit_val, validate_quantiles
    data = _synthetic("chronos2", K=1)
    ours = p2d.fit_shared_layerwise("chronos2", data, quantiles=Q9, median_idx=MED,
                                    device="cpu", verbose=False, **FAST)
    q = validate_quantiles(Q9)
    flat = {s: {i: v[:, 0] for i, v in d["feats"].items()} for s, d in data.items()}
    ref = fit_quantile_probe_explicit_val(flat["train"], data["train"]["Y"], flat["val"],
                                          data["val"]["Y"], quantiles=q, epochs=FAST["epochs"],
                                          lr=FAST["lr"], wd_grid=list(FAST["wd_grid"]),
                                          device="cpu", init_seed=0)
    worst = 0.0
    for i in range(13):
        f = ref[i]
        X = torch.as_tensor(f["scaler"].transform(flat["test"][i]), dtype=torch.float32)
        with torch.no_grad():
            pr = f["linear"](X).view(-1, len(q), 64).numpy()
        worst = max(worst, float(np.abs(pr - ours["pred_test"][ours["labels"][i]]).max()))
        assert f["wd"] == ours["wd"][i]
    assert worst <= 1e-5, worst
    print(f" 08  K=1/P=64 shared head == the pooled quantile probe (max |d| {worst:.1e}, same wd)"
          "                                                                      OK")


def test_09_selection_never_sees_test():
    data = _synthetic("tirex")
    a = p2d.fit_shared_layerwise("tirex", data, quantiles=Q9, median_idx=MED, device="cpu",
                                 verbose=False, **FAST)
    poisoned = copy.deepcopy(data)
    for i in poisoned["test"]["feats"]:
        poisoned["test"]["feats"][i] = poisoned["test"]["feats"][i] * 0 + 1e3
    poisoned["test"]["Y"] = poisoned["test"]["Y"] * 0 - 5.0
    b = p2d.fit_shared_layerwise("tirex", poisoned, quantiles=Q9, median_idx=MED, device="cpu",
                                 verbose=False, **FAST)
    assert a["wd"] == b["wd"] and a["val_loss"] == b["val_loss"]
    assert a["test_loss"] != b["test_loss"]
    print(" 09  test features + targets replaced: wd selection and val curve unchanged      OK")


def test_10_native_reproduced_from_committed_artifacts():
    """All 42 committed Phase-1 cells: curves from per-window arrays, entrances re-derived at
    every tolerance, and the frozen entrance table -- before any GPU time is spent."""
    n = 0
    for t in registry.roster("paper14"):
        for k, m in enumerate(phase1.MODELS):
            spec = phase1.model_spec(m)
            nat = p2d.native_from_phase1(_real_cell(m, t), spec)
            rep = p2d.reproduce_native(nat, spec)
            assert rep["entrances"]["tol_0.05"] == FROZEN[t][k], (m, t, rep["entrances"])
            n += 1
    assert n == 42
    nat = p2d.native_from_phase1(_real_cell("chronos2", "monash_electricity_hourly"),
                                 phase1.model_spec("chronos2"))
    bad = copy.deepcopy(nat)
    bad["val_loss"][3] *= 1.01
    assert _raises(lambda: p2d.reproduce_native(bad, phase1.model_spec("chronos2")),
                   phase2.Phase1DependencyError)
    bad = copy.deepcopy(nat)
    bad["entrances"][0.05] = {"depth_axis_index": 2, "label": "L2"}
    assert _raises(lambda: p2d.reproduce_native(bad, phase1.model_spec("chronos2")),
                   phase2.Phase1DependencyError)
    print(" 10  42/42 committed native cells reproduced (curves + entrances at 1/2/5/10%) and "
          "equal to the frozen table; a perturbed curve or entrance aborts           OK")


def test_11_paired_rows():
    spec = phase1.model_spec("timesfm3")
    rng = np.random.default_rng(3)
    cid = np.repeat(np.arange(30), 4)
    A = rng.gamma(2.0, 0.1, size=(spec.n_points, len(cid)))
    rows = p2d.paired_rows("loss", "test", "native", A, "pooled", A + 0.05, cid, spec, B=500)
    r = rows[spec.reference_index]
    assert abs(r["delta_abs"] - 0.05) < 1e-12 and r["direction"] == "to_worse"
    assert r["delta_abs_ci_lo"] > 0.049 and r["is_reference_point"]
    same = p2d.paired_rows("loss", "test", "native", A, "pooled", A.copy(), cid, spec, B=500)
    assert all(x["delta_abs"] == 0 and not x["ci_excludes_zero"] for x in same)
    better = p2d.paired_rows("loss", "test", "native", A, "pooled", A * 0.8, cid, spec, B=500)
    assert all(x["direction"] == "to_better" for x in better)
    assert _raises(lambda: p2d.paired_rows("loss", "test", "a", A, "b", A[:, :-1], cid, spec))
    print(" 11  paired cluster bootstrap: exact delta, shared resampling, sign/direction, "
          "alignment refused when shapes differ                                  OK")


# --------------------------------------------------------------------------- #
# synthetic cells for the store / tables / figures path
# --------------------------------------------------------------------------- #
def _fake_native(model, n_val=24, n_test=24, seed=0):
    spec = phase1.model_spec(model)
    rng = np.random.default_rng(seed)
    cid_t, cid_v = np.repeat(np.arange(n_test // 2), 2), np.repeat(np.arange(n_val // 2), 2)
    curve = 0.2 + 0.3 * np.exp(-np.arange(spec.n_points) / 3.0)
    vw = curve[:, None] * rng.gamma(20, 0.05, size=(spec.n_points, n_val))
    tw = curve[:, None] * rng.gamma(20, 0.05, size=(spec.n_points, n_test))
    nat = {"labels": spec.labels, "train_loss": list(vw.mean(1) * 0.9),
           "val_loss": list(vw.mean(1)), "test_loss": list(tw.mean(1)),
           "wd": [1.0] * spec.n_points,
           "n_params": [p2d.readout_params(model, "native")["n_params"]] * spec.n_points,
           "val_window_loss": vw, "test_window_loss": tw, "mase_pw": tw * 5,
           "mae_pw": tw, "wql_num_pw": tw, "wql_den_pw": np.ones(n_test),
           "cluster_ids_test": cid_t, "cluster_ids_val": cid_v,
           "forecast": {"mase": np.full(n_test, 0.8)}, "probe": "native probe"}
    tun = phase1.tunnel_record(spec, nat["val_loss"], nat["test_loss"], tol=0.05,
                               tols=p2d.TUNNEL_TOLS)
    nat["entrances"] = {float(e["tolerance"]): {"depth_axis_index": e["depth_axis_index"],
                                                "label": e["label"]}
                        for e in tun["by_tolerance"].values()}
    return nat


def _fake_arm(model, arm, nat, scale=1.1, seed=1):
    spec = phase1.model_spec(model)
    rng = np.random.default_rng(seed)
    vw = nat["val_window_loss"] * scale * rng.uniform(0.98, 1.02, size=nat["val_window_loss"].shape)
    tw = nat["test_window_loss"] * scale
    npar = p2d.readout_params(model, arm)["n_params"]
    grid = list(phase1.PHASE1_WD_GRID)
    res = {"labels": spec.labels, "quantiles": list(Q9), "train_loss": list(vw.mean(1) * 0.8),
           "val_loss": list(vw.mean(1)), "test_loss": list(tw.mean(1)), "wd": [1.0] * spec.n_points,
           "wd_at_grid_max": [False] * spec.n_points, "n_params": [npar] * spec.n_points,
           "selection": [{str(w): 0.3 for w in grid}] * spec.n_points, "wd_grid": grid,
           "val_window_loss": vw, "test_window_loss": tw,
           "probe_description": p2d.readout_params(model, arm)["description"],
           "constant_forecast_floor": {"val_loss": 0.9, "test_loss": 0.9}}
    raw = {"mase_pw": tw * 5, "wql_num_pw": tw, "wql_den_pw": np.ones(tw.shape[1])}
    return {"res": res, "raw": raw, "cid_test": nat["cluster_ids_test"],
            "cid_val": nat["cluster_ids_val"]}


def _fake_cfg(model, tag):
    cfg = {"protocol": p2d.PROTOCOL, "model": model, "dataset": tag, "quantiles": list(Q9),
           "probe_lr": 1e-2, "arms": list(p2d.ARMS_BY_MODEL[model])}
    cfg["config_hash"] = p2d.config_hash(cfg)
    return cfg


def _build(root, model, tag, scales=None):
    nat = _fake_native(model)
    rep = p2d.reproduce_native(nat, phase1.model_spec(model))
    scales = scales or {}
    arms = {a: _fake_arm(model, a, nat, scales.get(a, 1.1)) for a in p2d.ARMS_BY_MODEL[model]}
    cfg = _fake_cfg(model, tag)
    bundle = p2d.assemble_cell(model, tag, native=nat, repro=rep, arms=arms, cfg=cfg, B=300)
    store = phase2.Phase2CellStore(root, p2d.KIND, model, tag, required=p2d.REQUIRED)
    stage = store.begin()
    p2d.write_cell(stage, bundle)
    store.commit(stage, cfg["config_hash"])
    return store, cfg, bundle


def test_12_cell_store_tables_figures():
    from experiments import make_phase2d_tables as mt
    tmp = Path(tempfile.mkdtemp())
    try:
        root, paper = tmp / "out", tmp / "paper"
        for m in phase1.MODELS:
            for t in ("monash_electricity_hourly", "uber_tlc_hourly"):
                store, cfg, bundle = _build(root, m, t, scales={"pooled": 1.2, "concat": 0.97})
        assert store.status(cfg["config_hash"])[0] == "complete"
        assert store.status("sha256:other")[0] == "incompatible"
        for a in p2d.REQUIRED:
            assert (store.final / a).exists(), a
        s = json.loads((store.final / "summary.json").read_text())
        assert set(s["per_arm"]) == {"native", "pooled", "concat", "native_mean"}
        fb = s["final_block_comparisons"]["native->pooled"]["loss"]
        assert abs(fb["delta_rel"] - 0.2) < 1e-9 and fb["direction"] == "to_worse"
        steps = {r["step"] for r in bundle["comparisons"]}
        assert steps == {"readout_structure", "vs_native", "pooling", "token_selection"}
        rc = mt.main(["--output-root", str(root), "--paper-dir", str(paper)])
        assert rc == 0
        agg = list(__import__("csv").DictReader(open(root / "combined" /
                                                     "aggregate_by_model.csv")))
        pooled_c2 = next(r for r in agg if r["model"] == "chronos2" and r["arm"] == "pooled")
        assert pooled_c2["n_cells"] == "2" and pooled_c2["n_final_worse_than_native_ci"] == "2"
        stats = json.loads((root / "combined" / "stats.json").read_text())
        assert stats["n_complete"] == 6 and len(stats["pending"]) == 36
        for f in ("tab_pooled_entrances.tex", "tab_pooled_final.tex", "tab_pooled_aggregate.tex",
                  "tab_pooled_params.tex", "pooled_macros.tex"):
            assert (paper / "tables" / f).exists(), f
        for f in ("pooled_curves_main.pdf", "pooled_curves_tirex.pdf",
                  "pooled_final_delta.pdf", "pooled_entrances.pdf"):
            assert (paper / f).exists(), f
        assert "\\pdChronosTwoPooledFinalWorse}{2}" in (paper / "tables" /
                                                        "pooled_macros.tex").read_text()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 12  cell store (atomic, resume, refuse other hash) -> tables (pending kept) -> LaTeX"
          " + figures, all from artifacts                                         OK")


def test_13_alignment_and_capacity_refusals():
    nat = _fake_native("tirex")
    rep = p2d.reproduce_native(nat, phase1.model_spec("tirex"))
    arms = {a: _fake_arm("tirex", a, nat) for a in p2d.ARMS_BY_MODEL["tirex"]}
    shuffled = copy.deepcopy(arms)
    shuffled["pooled"]["cid_test"] = shuffled["pooled"]["cid_test"][::-1].copy()
    assert _raises(lambda: p2d.assemble_cell("tirex", "t", native=nat, repro=rep, arms=shuffled,
                                             cfg=_fake_cfg("tirex", "t"), B=50), RuntimeError)
    wrong = copy.deepcopy(arms)
    wrong["concat"]["res"]["n_params"] = [123] * len(wrong["concat"]["res"]["n_params"])
    assert _raises(lambda: p2d.assemble_cell("tirex", "t", native=nat, repro=rep, arms=wrong,
                                             cfg=_fake_cfg("tirex", "t"), B=50), RuntimeError)
    print(" 13  permuted rows refused; a measured parameter count != documented refused     OK")


def test_14_tunnel_rule_per_arm():
    nat = _fake_native("chronos2")
    spec = phase1.model_spec("chronos2")
    rep = p2d.reproduce_native(nat, spec)
    arms = {a: _fake_arm("chronos2", a, nat) for a in p2d.ARMS_BY_MODEL["chronos2"]}
    b1 = p2d.assemble_cell("chronos2", "t", native=nat, repro=rep, arms=arms,
                           cfg=_fake_cfg("chronos2", "t"), B=50)
    moved = copy.deepcopy(arms)
    r = moved["pooled"]["res"]
    r["test_loss"] = [x * 50 for x in r["test_loss"]]               # test cannot move it
    r["val_loss"][spec.diagnostic_indices[0]] *= 1e-6                # nor the head-input point
    b2 = p2d.assemble_cell("chronos2", "t", native=nat, repro=rep, arms=moved,
                           cfg=_fake_cfg("chronos2", "t"), B=50)
    e1, e2 = b1["tunnels"]["pooled"]["headline"], b2["tunnels"]["pooled"]["headline"]
    assert e1["label"] == e2["label"]
    assert b1["tunnels"]["pooled"]["reference_point"] == "L12"
    assert b2["summary"]["per_arm"]["pooled"]["final_block"]["label"] == "L12"
    print(" 14  each arm's tunnel: validation only, reference = its OWN final block (L12), the "
          "head-input point is off the depth axis                                OK")


def test_15_separation():
    from experiments import run_phase2d_pooled as drv
    for name in ("three_model_phase2_lodo", "three_model_phase1_q9_expanded",
                 "three_model_phase1_q9_loop_lowlr", "three_model_phase2"):
        assert _raises(lambda: drv.assert_output_root(REPO_ROOT / "results" / name / "x"),
                       SystemExit), name
    drv.assert_output_root(drv.DEFAULT_OUT)
    drv.assert_output_root(drv.SMOKE_OUT)
    assert p2d.KIND not in ("h3", "lodo") and p2d.CACHE_VERSION.startswith("p2d-")
    d = p2d.pooled_cache_dir("/scratch/x/phase2d_pooled_cache", "tirex", "m5", "test")
    assert d.name.startswith("p2d-pooled-v1__tirex__") and "phase1_shared_cache" not in str(d)
    print(" 15  separation: Phase-1 / Phase-2 / LODO trees refused as output; own cache + kind"
          " namespace                                                            OK")


def test_16_pooled_cache_roundtrip():
    tmp = Path(tempfile.mkdtemp())
    try:
        d = p2d.pooled_cache_dir(tmp, "tirex", "m5", "test")
        assert p2d.load_pooled_cache(d, {"a": 1}) is None
        feats = {"Emb": np.ones((3, 4), np.float32), "L12+RMS": np.zeros((3, 4), np.float32)}
        meta = {"cache_version": p2d.CACHE_VERSION, "n_windows": 3, "dim": 4,
                "labels": list(feats), "batch_size": 256}
        p2d.save_pooled_cache(d, feats, meta, {"identity": {"ok": True}})
        assert not list(tmp.glob("*.tmp*"))
        got, aux = p2d.load_pooled_cache(d, meta)
        assert np.array_equal(got["L12+RMS"], feats["L12+RMS"]) and aux["identity"]["ok"]
        assert _raises(lambda: p2d.load_pooled_cache(d, {**meta, "batch_size": 128}),
                       RuntimeError)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 16  pooled cache: atomic write, exact round trip, any metadata change REFUSED      OK")


def test_17_protocol_read_from_phase1_cell():
    from experiments import run_phase2d_pooled as drv
    args = types.SimpleNamespace(phase1_root=str(phase2.DEFAULT_PHASE1_ROOT), phase1_override=[],
                                 suite="paper14", boot_b=5000)
    out = {}
    for t in ("monash_electricity_hourly", "LOOP_SEATTLE_5T"):
        cell, p1cfg, root = drv.phase1_source("tirex", t, args)
        out[t] = drv.cell_config("tirex", t, p1cfg, root, args, p2d.ARMS_BY_MODEL["tirex"], Q9)
    e, lp = out["monash_electricity_hourly"], out["LOOP_SEATTLE_5T"]
    assert e["probe_lr"] == 1e-2 and len(e["wd_grid"]) == 13 and max(e["wd_grid"]) == 90
    assert lp["probe_lr"] == 1e-3 and len(lp["wd_grid"]) == 21 and max(lp["wd_grid"]) == 990
    assert "loop_lowlr" in lp["phase1_root"]
    assert e["phase1_extraction"]["extract_batch_size"] == 256
    assert e["phase1_extraction"]["rollout_mode"] == "two_pass"
    assert p2d.config_hash(e) != p2d.config_hash(lp)
    print(" 17  training protocol read from each Phase-1 cell: lr 1e-2 / 13-value grid, LOOP "
          "lr 1e-3 / 21-value grid (rerouted); TiRex batch 256 two_pass            OK")


def test_18_decomposition_pairs():
    assert p2d.comparison_pairs("chronos2") == [
        ("vs_native", "native", "pooled"), ("readout_structure", "native", "concat"),
        ("vs_native", "native", "native_mean"), ("pooling", "concat", "native_mean"),
        ("token_selection", "native_mean", "pooled")]
    assert p2d.comparison_pairs("timesfm3") == [("representation", "native", "pooled")]
    assert p2d.comparison_pairs("tirex", ("pooled",)) == [("vs_native", "native", "pooled")]
    print(" 18  decomposition: native->concat (structure) -> native_mean (pooling) -> pooled "
          "(token selection); TimesFM-3: native->pooled (representation)         OK")


def test_19_chronos2_loader_refuses_missing_values():
    X = np.random.default_rng(4).normal(size=(4, 512)).astype(np.float32)
    X[1, 7] = np.nan
    w = {f"X_{s}": X for s in p2d.SPLITS}
    w.update({f"Y_{s}_traj": np.zeros((4, 64), np.float32) for s in p2d.SPLITS})
    w.update({f"series_{s}": np.arange(4) for s in p2d.SPLITS})
    w["meta"] = {"sigma_eps": 1e-6}
    assert _raises(lambda: p2d.load_chronos2("m5", w, {}), RuntimeError)
    print(" 19  Chronos-2 loader refuses a window whose context is not fully observed (the cached"
          " unmasked mean would then differ from the masked mean)               OK")


def _loader_output(model, n=24, seed=5):
    """What load_<model> returns, with real dimensions and real geometries, synthetic values."""
    spec = phase1.model_spec(model)
    rng = np.random.default_rng(seed)
    series = np.repeat(np.arange(n // 2), 2)
    splits = {}
    for s in p2d.SPLITS:
        X = (10 + rng.normal(size=(n, 512))).astype(np.float32)
        y_raw = (10 + rng.normal(size=(n, 64))).astype(np.float64)
        Y = rng.normal(size=(n, 64)).astype(np.float32)
        d = {"X": X, "y_raw": y_raw, "series": series, "Y": Y}
        if model == "timesfm3":
            d.update(feats={L: rng.normal(size=(n, spec.d)).astype(np.float32)
                            for L in range(spec.n_points)},
                     last_token_feats={L: rng.normal(size=(n, spec.d)).astype(np.float32)
                                       for L in range(spec.n_points)},
                     targets=Y, valid=np.ones(n, bool), mu=np.full(n, 10.0), sd=np.ones(n),
                     trend=np.zeros((n, 64)))
        else:
            K = p2d.NATIVE_K[model]
            d.update(pooled={i: rng.normal(size=(n, spec.d)).astype(np.float32)
                             for i in range(spec.n_points)},
                     native_states={i: rng.normal(size=(n, K, spec.d)).astype(np.float32)
                                    for i in range(spec.n_points)})
            if model == "tirex":
                d.update(targets=Y, loc=np.full((n, 2), 10.0, np.float32),
                         scale=np.ones((n, 2), np.float32),
                         native_by_label={lb: d["native_states"][i]
                                          for i, lb in enumerate(spec.labels)})
        splits[s] = d
    if model == "timesfm3":
        from probing.timesfm3_last_token import LastTokenGeometry
        geom = LastTokenGeometry(C=512, H=64, strict=True)
    elif model == "tirex":
        from probing.tirex_model import TiRexGeometry
        geom = TiRexGeometry(C=512, H=64, input_patch=32, output_patch=32, train_ctx_len=2048,
                             num_quantiles=9, median_index=4)
    else:
        geom = None
    return {"splits": splits, "geom": geom, "gates": {}, "provenance": {}}


def test_20_driver_glue_all_models():
    """fit_arms -> per-model raw_metrics -> assemble_cell -> write_cell, and the native refit
    check, for all three models on loader-shaped synthetic data: catches every signature /
    key mismatch in the driver before a GPU job does."""
    from experiments import run_phase2d_pooled as drv
    args = types.SimpleNamespace(device="cpu", quiet_probes=True)
    p1cfg = {"probe_epochs": 2, "probe_lr": 1e-2, "seed": 0, "wd_grid": [1e-3, 1e-1],
             "null_wd": 100.0}
    import contextlib
    import io
    tmp = Path(tempfile.mkdtemp())
    # the fits and the refit check print per-depth lines; the refit check REPORTS "FAIL" here by
    # design (a random synthetic refit vs a fake native curve) -- only "it runs" is under test
    quiet = contextlib.redirect_stdout(io.StringIO())
    try:
        quiet.__enter__()
        for m in phase1.MODELS:
            D = _loader_output(m)
            arm_res = drv.fit_arms(m, "monash_electricity_hourly", D, p2d.ARMS_BY_MODEL[m],
                                   p1cfg, Q9, MED, args)
            nat = _fake_native(m)
            rep = p2d.reproduce_native(nat, phase1.model_spec(m))
            cfg = _fake_cfg(m, "monash_electricity_hourly")
            bundle = p2d.assemble_cell(m, "monash_electricity_hourly", native=nat, repro=rep,
                                       arms=arm_res, cfg=cfg, B=100)
            st = tmp / m
            st.mkdir()
            p2d.write_cell(st, bundle)
            assert all((st / a).exists() for a in p2d.REQUIRED)
            arts = p2d.save_probe_artifacts(tmp / "arts", m, "x", "pooled", arm_res["pooled"]["res"])
            assert Path(arts["probe_weights"]).exists()
            rec = drv.refit_native_check(m, "monash_electricity_hourly",
                                         {"data": D, "native": nat, "p1cfg": p1cfg}, Q9, MED, args)
            assert rec["status"] in ("pass", "FAIL") and rec["n_points"] == phase1.model_spec(m).n_points
    finally:
        quiet.__exit__(None, None, None)
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 20  driver glue: fit_arms + raw_metrics + assemble + write + probe artifacts + native"
          " refit check run end to end for all three models                      OK")


def main(argv=None):
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"\nPHASE-2d POOLED-READOUT CONTRACTS  ({len(tests)} model-free groups)\n" + "=" * 78)
    failed = []
    for t in tests:
        try:
            t()
        except Exception:
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print("=" * 78)
    if failed:
        print(f"{len(failed)} of {len(tests)} Phase-2d contracts FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(tests)} Phase-2d contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
