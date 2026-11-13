"""PRE-RUN CONTRACT SUITE for Phase 2f (source-dataset scaling of zero-shot adapter transfer).

Model-free: no checkpoint, no dataset, no feature cache. The design contracts run on the REAL
paper-14 roster; the fit / cell / driver contracts reuse the Phase 2c Chronos-2-shaped mock pathway
(``tests.test_phase2_lodo``). Runs on a login node in about a minute with 2 threads:

    python -m tests.test_phase2f_source_scaling

Numbered so a failure names the requirement it broke.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2, registry                                      # noqa: E402
from probing import phase2_lodo as lodo                                           # noqa: E402
from probing import phase2f_source_scaling as ss                                  # noqa: E402
from probing.phase2 import MEDIAN_INDEX, QUANTILES, Phase2CellStore               # noqa: E402
from probing.phase2_align import (adapter_arrays_sha256, fit_residual_adapter,    # noqa: E402
                                  load_adapter, mse_loss)
from tests import test_phase2_lodo as t2c                                         # noqa: E402

torch.set_num_threads(2)
ROSTER = list(registry.roster("paper14"))
MOCK_ROSTER = list(t2c.MOCK_TAGS)          # m4, Electricity hourly, SG Carpark, Electricity 15min, M5
MOCK_TARGETS = ("m5", "sg_carpark")
MOCK_GRID = (1, 2, 4)                      # a 4-dataset pool: the same nested structure, small
MOCK_SEEDS = (0, 1)
FAST = {**lodo.DEFAULT_FIT_CFG, **t2c.FAST}
D6 = 6

#: The source orders, FROZEN 2026-11-13 before any Phase 2f fit (a pure function of the rule, the
#: target, the seed and the roster; sha256-keyed, so identical on every machine).
FROZEN_ORDERS = {
    "LOOP_SEATTLE_5T": [
        ["m5", "boom_hourly", "electricity_15min", "sg_carpark", "wiki_daily_100k",
         "wind_farms_hourly", "monash_traffic", "uber_tlc_hourly", "monash_electricity_hourly",
         "coastal_ts", "m4_hourly", "SZ_TAXI_15T", "monash_london_smart_meters"],
        ["wind_farms_hourly", "m5", "monash_traffic", "uber_tlc_hourly", "electricity_15min",
         "coastal_ts", "m4_hourly", "boom_hourly", "SZ_TAXI_15T", "monash_london_smart_meters",
         "sg_carpark", "monash_electricity_hourly", "wiki_daily_100k"],
        ["boom_hourly", "coastal_ts", "m4_hourly", "wiki_daily_100k", "SZ_TAXI_15T", "m5",
         "wind_farms_hourly", "sg_carpark", "uber_tlc_hourly", "electricity_15min",
         "monash_traffic", "monash_electricity_hourly", "monash_london_smart_meters"]],
    "monash_london_smart_meters": [
        ["wind_farms_hourly", "boom_hourly", "coastal_ts", "sg_carpark", "SZ_TAXI_15T",
         "wiki_daily_100k", "monash_traffic", "uber_tlc_hourly", "LOOP_SEATTLE_5T",
         "monash_electricity_hourly", "m5", "electricity_15min", "m4_hourly"],
        ["m4_hourly", "sg_carpark", "wiki_daily_100k", "wind_farms_hourly", "SZ_TAXI_15T", "m5",
         "LOOP_SEATTLE_5T", "coastal_ts", "electricity_15min", "monash_electricity_hourly",
         "uber_tlc_hourly", "boom_hourly", "monash_traffic"],
        ["monash_traffic", "wind_farms_hourly", "coastal_ts", "m4_hourly", "uber_tlc_hourly",
         "LOOP_SEATTLE_5T", "electricity_15min", "SZ_TAXI_15T", "wiki_daily_100k",
         "monash_electricity_hourly", "boom_hourly", "sg_carpark", "m5"]],
    "boom_hourly": [
        ["m4_hourly", "wiki_daily_100k", "uber_tlc_hourly", "monash_electricity_hourly", "m5",
         "SZ_TAXI_15T", "sg_carpark", "wind_farms_hourly", "monash_london_smart_meters",
         "coastal_ts", "monash_traffic", "LOOP_SEATTLE_5T", "electricity_15min"],
        ["electricity_15min", "wiki_daily_100k", "m4_hourly", "monash_london_smart_meters",
         "uber_tlc_hourly", "coastal_ts", "sg_carpark", "wind_farms_hourly", "SZ_TAXI_15T",
         "monash_traffic", "monash_electricity_hourly", "LOOP_SEATTLE_5T", "m5"],
        ["wind_farms_hourly", "m5", "LOOP_SEATTLE_5T", "monash_electricity_hourly", "sg_carpark",
         "m4_hourly", "SZ_TAXI_15T", "monash_london_smart_meters", "uber_tlc_hourly",
         "electricity_15min", "monash_traffic", "wiki_daily_100k", "coastal_ts"]],
    "m5": [
        ["boom_hourly", "electricity_15min", "sg_carpark", "monash_traffic", "SZ_TAXI_15T",
         "wiki_daily_100k", "m4_hourly", "monash_london_smart_meters", "wind_farms_hourly",
         "LOOP_SEATTLE_5T", "coastal_ts", "monash_electricity_hourly", "uber_tlc_hourly"],
        ["monash_london_smart_meters", "uber_tlc_hourly", "coastal_ts",
         "monash_electricity_hourly", "LOOP_SEATTLE_5T", "SZ_TAXI_15T", "wiki_daily_100k",
         "monash_traffic", "m4_hourly", "electricity_15min", "wind_farms_hourly", "boom_hourly",
         "sg_carpark"],
        ["wiki_daily_100k", "boom_hourly", "sg_carpark", "LOOP_SEATTLE_5T", "monash_traffic",
         "m4_hourly", "coastal_ts", "monash_london_smart_meters", "wind_farms_hourly",
         "monash_electricity_hourly", "uber_tlc_hourly", "electricity_15min", "SZ_TAXI_15T"]]}


def _quiet(*a, **k):
    return None


def _T(x):
    return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)


def _h3_world(pw, prep, depth, tags):
    """An H3-shaped committed world on the mock data: for every tag, the H3 NOA fit (the H3 code
    path: ``mse_loss``, one dataset) as the dataset-specific adapter, its fits.json-like record,
    the per-window arrays the gates compare against, and the Phase-1 native arrays."""
    lab = lodo.depth_label(depth)
    h3, p1, spec = {}, {}, {}
    for t in tags:
        P = prep[t]
        r = fit_residual_adapter(d=t2c.D, pathway_fn=pw.outputs, train_x=_T(P.train_states[depth]),
                                 train_y=_T(P.train_teacher), val_x=_T(P.val_states[depth]),
                                 val_y=_T(P.val_teacher), train_loss=mse_loss,
                                 val_criterion=mse_loss, wd_grid=FAST["wd_grid"],
                                 epochs=FAST["epochs"], lr=phase2.ADAPTER_LR,
                                 eval_every=FAST["eval_every"], seed=0)
        ad = r.pop("adapter")
        rec = json.loads(json.dumps({**r, "fitted": True,
                                     "adapter_sha256": adapter_arrays_sha256(ad)}, default=float))
        te, L = P.test, P.reference_index
        nat = lodo.score_arm(pw, te, t, te.feats[L])
        hard = lodo.score_arm(pw, te, t, te.feats[depth])
        noa = lodo.score_arm(pw, te, t, te.feats[depth], ad)
        n = len(te.target)
        H, S = np.full((L + 1, n), np.nan), np.full((L + 1, n), np.nan)
        H[depth], S[depth] = hard["mase"], noa["mase"]
        h3[t] = {"config_hash": f"h3-{t}",
                 "arrays": {"native_test_mase": nat["mase"], "hard__test_mase": H,
                            "noa__test_mase": S, "cluster_ids_test": np.asarray(te.cluster_ids)},
                 "fits": {"noa": {lab: rec}}}
        p1[t] = {"native_loss_window": nat["loss"], "native_mase_window": nat["mase"]}
        spec[t] = (ad, {"path": f"mock/{t}", "sha256": rec["adapter_sha256"]})
    return h3, p1, spec


def _mock_world(seed=0):
    pw = t2c._pathway(seed)
    prep = t2c._prepared(pw, t2c._mock_data(pw, seed))
    h3, p1, spec = _h3_world(pw, prep, D6, MOCK_ROSTER)
    cells = ss.plan_cells(MOCK_ROSTER, targets=MOCK_TARGETS, seeds=MOCK_SEEDS, n_grid=MOCK_GRID,
                          depth=D6)
    deps = {t: {"fit_hash": f"fit-{t}", "window_digest": f"wd-{t}"} for t in MOCK_ROSTER}
    for c in cells:
        c["config"] = ss.cell_config(c, deps, h3[c["target"]], FAST)
        c["hash"] = phase1._digest(c["config"])
    return pw, prep, h3, p1, spec, cells


def _run(pw, prep, h3, p1, spec, cells, root, **kw):
    from experiments import run_phase2f_source_scaling as drv
    return drv.run_cells(pw, prep, cells, out=root, adapter_root=Path(root) / "adapters",
                         h3refs=h3, p1arrays=p1, specific=spec, fit_cfg=FAST,
                         roster=MOCK_ROSTER, job="test", log=_quiet, **kw)


# =========================================================================== #
def test_01_targets_follow_the_frozen_metadata_rule():
    picked, audit = ss.select_targets(ROSTER)
    assert sorted(picked) == sorted(ss.TARGETS), picked
    assert audit["excluded_R1_series_siblings"] == ["monash_electricity_hourly",
                                                     "electricity_15min"]
    assert audit["excluded_R2_test_series_below_budget"] == {"coastal_ts": 24, "SZ_TAXI_15T": 156}
    assert len(audit["admissible_sets"]) == 2          # only the daily slot is open; R4 decides
    prov = {t: lodo.CHRONOS2_PROVENANCE[t][0] for t in ss.TARGETS}
    assert set(prov.values()) == set(lodo.PROVENANCE_GROUPS)
    assert len({ss.TARGET_RATIONALE[t]["frequency"] for t in ss.TARGETS}) == 4
    assert len({ss.DOMAIN_FAMILY[t] for t in ss.TARGETS}) == 4
    assert not set(ss.TARGETS) & set(lodo.SERIES_SIBLINGS)
    for t in ss.TARGETS:                               # the rationale table agrees with the data
        assert ss.TARGET_RATIONALE[t]["provenance"] == prov[t]
        assert ss.TARGET_RATIONALE[t]["seasonal_period"] == registry.seasonal_m(t)
    checked = 0                                        # TEST_SERIES vs the committed H3 windows
    for t in ROSTER:
        p = REPO_ROOT / "results" / "three_model_phase2" / "h3" / "chronos2" / t / \
            "bootstrap_inputs.npz"
        if p.exists():
            with np.load(p, allow_pickle=True) as z:
                assert len(np.unique(z["cluster_ids_test"])) == ss.TEST_SERIES[t], t
            checked += 1
    print(" 1   targets = the frozen metadata rule (R1 siblings, R2 test budget, R3 frequency /"
          " provenance / domain, R4 held out)"
          + (f"; test series of {checked} datasets match the H3 windows" if checked else "")
          + "   OK")


def test_02_subset_sizes_nesting_and_the_40_cells():
    cells = ss.plan_cells(ROSTER)
    assert len(cells) == 40 and {(c["target"], c["cell_id"]) for c in cells} == \
        ss.expected_cell_ids()
    for t in ss.TARGETS:
        ct = [c for c in cells if c["target"] == t]
        pool = ss.eligible_sources(t, ROSTER)
        assert len(pool) == 13 and t not in pool and pool == [x for x in ROSTER if x != t]
        assert sorted(c["n_sources"] for c in ct) == [1, 1, 1, 4, 4, 4, 8, 8, 8, 13]
        full = [c for c in ct if c["n_sources"] == 13]
        assert len(full) == 1 and full[0]["sources"] == pool
        assert full[0]["order_seed"] is None and full[0]["shared_across_seeds"] == [0, 1, 2]
        for c in ct:
            assert len(c["sources"]) == c["n_sources"] == len(set(c["sources"]))
            assert t not in c["sources"] and t in c["excluded"]
            assert c["sources"] == sorted(c["sources"], key=ROSTER.index)   # the 2c data order
            assert set(c["sources"]) | set(c["excluded"]) == set(ROSTER)
        for s in ss.ORDER_SEEDS:
            seq = [next(c for c in ct if c["cell_id"] == ss.cell_id(n, s)) for n in (1, 4, 8)]
            for a, b in zip(seq, seq[1:]):           # nested: a larger N only ADDS datasets
                assert b["selection_order"][:a["n_sources"]] == a["selection_order"]
                assert set(a["sources"]) < set(b["sources"])
            assert set(seq[-1]["sources"]) < set(full[0]["sources"])
    print(" 2   40 cells = 4 targets x (3 orders x N 1/4/8 + one shared N=13); sizes exact, "
          "subsets nested within an order, target never a source   OK")


def test_03_orders_are_deterministic_and_frozen():
    got = {t: [o["order"] for o in ss.source_orders(t, ROSTER)] for t in ss.TARGETS}
    assert got == FROZEN_ORDERS, "the source orders changed: the design is frozen"
    assert got == {t: [o["order"] for o in ss.source_orders(t, ROSTER)] for t in ss.TARGETS}
    code = ("import json,sys; sys.path.insert(0, %r); from probing import registry, "
            "phase2f_source_scaling as ss; r=list(registry.roster('paper14')); "
            "print(json.dumps({t: [o['order'] for o in ss.source_orders(t, r)] "
            "for t in ss.TARGETS}))" % str(REPO_ROOT))
    for hs in ("1", "4242"):                          # independent of Python's hash seed
        env = {**os.environ, "PYTHONHASHSEED": hs, "OMP_NUM_THREADS": "1"}
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, check=True).stdout.strip().splitlines()[-1]
        assert json.loads(out) == FROZEN_ORDERS, hs
    # the permutation is a pure function of (rule, target, seed, draw)
    assert ss._permutation(ROSTER, "m5", 0, 1) == ss._permutation(list(reversed(ROSTER)), "m5",
                                                                  0, 1)
    assert ss._permutation(ROSTER, "m5", 0, 1) != ss._permutation(ROSTER, "m5", 1, 1)
    print(" 3   source orders reproduce the frozen table, in fresh processes under two hash seeds,"
          " and do not depend on the input order                         OK")


def test_04_electricity_family_and_order_restrictions():
    pair = set(lodo.SERIES_SIBLINGS)
    assert ss.family_of("electricity_15min") == ss.family_of("monash_electricity_hourly")
    assert ss.family_of("m5") == "m5"
    for t in ss.TARGETS:
        orders = ss.source_orders(t, ROSTER)
        assert len({o["order"][0] for o in orders}) == 3, t      # three different N=1 sources
        for o in orders:
            assert len(pair & set(o["order"][:8])) <= 1, (t, o["seed"])
            assert o["restrictions"]["family_distinct_within_first"] == 8
    for c in ss.plan_cells(ROSTER):
        both = pair <= set(c["sources"])
        assert both == (c["n_sources"] == 13), c["cell_id"]
        assert c["n_families"] == (12 if c["n_sources"] == 13 else c["n_sources"])
    for bad in pair:
        try:
            ss.eligible_sources(bad, ROSTER)
            raise AssertionError("an Electricity target was accepted")
        except ValueError:
            pass
    print(" 4   the Electricity pair never shares an N<=8 prefix; N=13 holds both = 12 families;"
          " three distinct N=1 sources per target; Electricity targets refused   OK")


def test_05_the_cell_is_the_phase2c_fit_with_equal_source_weights():
    pw = t2c._pathway()
    prep = t2c._prepared(pw, t2c._mock_data(pw))
    h3, p1, spec = _h3_world(pw, prep, D6, ["m5"])
    cells = ss.plan_cells(MOCK_ROSTER, targets=["m5"], seeds=MOCK_SEEDS, n_grid=MOCK_GRID,
                          depth=D6)
    c = next(x for x in cells if x["n_sources"] == 2)
    res = lodo.run_lodo_cell(pw, prep, c, phase1_arrays=p1["m5"], h3_ref=h3["m5"],
                             specific_adapter=spec["m5"][0], fit_cfg=FAST, log=None)
    direct = lodo.fit_pooled_adapter(pw, prep, c["sources"], D6, fit_cfg=FAST, log=None)
    for x, y in zip(t2c._arrays(res["adapter"]), t2c._arrays(direct["adapter"])):
        assert np.array_equal(x, y), "the Phase 2f cell is not the Phase 2c pooled fit"
    assert res["fit"]["fit"]["optimizer"] == direct["fit"]["optimizer"]
    assert res["fit"]["fit"]["grid"] == [float(w) for w in FAST["wd_grid"]]
    # unequal validation counts: the criterion is the MEAN OF PER-SOURCE MSEs, not the pooled mean
    s0, s1 = c["sources"]
    p = prep[s0]
    prep2 = {**prep, s0: dataclasses.replace(
        p, val_states={l: v[:7] for l, v in p.val_states.items()}, val_teacher=p.val_teacher[:7])}
    f = lodo.fit_pooled_adapter(pw, prep2, c["sources"], D6, fit_cfg=FAST, log=None)
    per = [float(((pw.run(prep2[s].val_states[D6])[0].astype(np.float64)
                   - prep2[s].val_teacher.astype(np.float64)) ** 2).mean()) for s in (s0, s1)]
    assert abs(f["fit"]["hard_cut_val"] - np.mean(per)) <= 1e-5 * np.mean(per), \
        (f["fit"]["hard_cut_val"], per)
    pooled = float(np.mean(np.concatenate([
        ((pw.run(prep2[s].val_states[D6])[0].astype(np.float64)
          - prep2[s].val_teacher.astype(np.float64)) ** 2).reshape(len(prep2[s].val_teacher), -1)
        for s in (s0, s1)])))
    assert abs(pooled - np.mean(per)) > 1e-3 * np.mean(per)       # the two would differ here
    assert [r["n_val"] for r in f["per_source_val"]] == [7, len(prep[s1].val_teacher)]
    print(" 5   a Phase 2f cell IS the Phase 2c pooled fit (bitwise); with unequal val counts the"
          " criterion is the mean of per-source MSEs, not the pooled mean   OK")


def test_06_target_and_labels_never_reach_the_fit():
    pw = t2c._pathway()
    data = t2c._mock_data(pw)
    prep = t2c._prepared(pw, data)
    c = next(x for x in ss.plan_cells(MOCK_ROSTER, targets=["sg_carpark"], seeds=MOCK_SEEDS,
                                      n_grid=MOCK_GRID, depth=D6) if x["n_sources"] == 2)
    a = lodo.fit_pooled_adapter(pw, prep, c["sources"], D6, fit_cfg=FAST, log=None)
    t = c["target"]
    P = prep[t]
    rng = np.random.default_rng(3)
    poisoned = dataclasses.replace(
        P, train_states={l: np.full_like(v, np.nan) for l, v in P.train_states.items()},
        val_states={l: rng.normal(size=v.shape).astype(np.float32) * 1e3
                    for l, v in P.val_states.items()},
        train_teacher=np.full_like(P.train_teacher, np.nan),
        val_teacher=rng.normal(size=P.val_teacher.shape).astype(np.float32),
        test=dataclasses.replace(P.test, feats={l: np.full_like(v, np.nan)
                                                for l, v in P.test.feats.items()},
                                 target=np.full_like(P.test.target, np.nan),
                                 y_raw=np.full_like(P.test.y_raw, np.nan)))
    b = lodo.fit_pooled_adapter(pw, {**prep, t: poisoned}, c["sources"], D6, fit_cfg=FAST,
                                log=None)
    for x, y in zip(t2c._arrays(a["adapter"]), t2c._arrays(b["adapter"])):
        assert np.array_equal(x, y), "the target influenced the adapter"
    # no ground-truth future of ANY dataset reaches the fit (label-free objective)
    nolab = {}
    for tag, d in data.items():
        sp = dict(d.splits)
        for s in ("train", "val"):
            sp[s] = dataclasses.replace(sp[s], target=np.full_like(sp[s].target, np.nan),
                                        y_raw=np.full_like(sp[s].y_raw, np.nan))
        nolab[tag] = dataclasses.replace(d, splits=sp)
    c2 = lodo.fit_pooled_adapter(pw, t2c._prepared(pw, nolab), c["sources"], D6, fit_cfg=FAST,
                                 log=None)
    for x, y in zip(t2c._arrays(a["adapter"]), t2c._arrays(c2["adapter"])):
        assert np.array_equal(x, y), "a ground-truth future reached the adapter fit"
    try:
        lodo.pooled_split(prep, c["sources"], "test", D6)
        raise AssertionError("a pooled fit could read a test split")
    except ValueError:
        pass
    print(" 6   the target's train/val/test (states, teacher, labels) and every future label "
          "replaced by NaN/noise -> adapter BITWISE unchanged             OK")


def test_07_fit_reproduction_gate():
    pw = t2c._pathway()
    prep = t2c._prepared(pw, t2c._mock_data(pw))
    c = next(x for x in ss.plan_cells(MOCK_ROSTER, targets=["m5"], seeds=MOCK_SEEDS,
                                      n_grid=MOCK_GRID, depth=D6) if x["n_sources"] == 1)
    s = c["sources"][0]
    h3, p1, spec = _h3_world(pw, prep, D6, ["m5", s])
    res = lodo.run_lodo_cell(pw, prep, c, phase1_arrays=p1["m5"], h3_ref=h3["m5"],
                             specific_adapter=spec["m5"][0], fit_cfg=FAST, log=None)
    lab = lodo.depth_label(D6)
    sha = adapter_arrays_sha256(res["adapter"])
    te = prep[s].test
    st = lodo._repro(lodo.score_arm(pw, te, s, te.feats[D6], res["adapter"])["mase"],
                     h3[s]["arrays"]["noa__test_mase"][D6], ss.FIT_REPRO_MASE_RTOL)
    ok = ss.fit_reproduction(res["fit"]["fit"], sha, h3[s]["fits"]["noa"][lab], source=s,
                             label=lab, source_test=st)
    assert ok["passed"] and ok["bitwise_adapter"] and ok["val_curve_max_rel_diff"] <= 1e-6, ok
    ref = h3[s]["fits"]["noa"][lab]
    for field, bad in (("selected_epoch", ref["selected_epoch"] + 10),
                       ("selected_val", ref["selected_val"] * 1.01),
                       ("hard_cut_val", ref["hard_cut_val"] * (1 + 1e-5)),
                       ("selected_wd", (ref["selected_wd"] or 0.0) + 7.0)):
        r = ss.fit_reproduction(res["fit"]["fit"], sha, {**ref, field: bad}, source=s, label=lab,
                                source_test=st)
        assert not r["passed"] and not r["checks"][field]["passed"], field
    r = ss.fit_reproduction(res["fit"]["fit"], sha, ref, source=s, label=lab,
                            source_test={**st, "passed": False})
    assert not r["passed"]
    print(" 7   N=1 pooled fit reproduces the H3 fit of its source (bitwise adapter, same val "
          "curves); a moved epoch / val / hard-cut val / wd or source MASE fails   OK")


def test_08_driver_loop_writes_atomically_and_resumes():
    from experiments import run_phase2f_source_scaling as drv
    pw, prep, h3, p1, spec, cells = _mock_world()
    assert len(cells) == 2 * (2 * 2 + 1)
    tmp = Path(tempfile.mkdtemp())
    try:
        st = _run(pw, prep, h3, p1, spec, cells, tmp)
        assert st["rc"] == 0 and not st["failures"], st["failures"]
        assert len(st["fit_reproduction"]) == 4 and all(r["passed"] for r in st["fit_reproduction"])
        run = [k for k, v in st["cells"].items() if v["status"] == "complete"]
        assert len(run) == len(cells)
        for c in cells:
            store = drv._store(tmp, c)
            assert store.status(c["hash"]) == ("complete", "validated"), c["cell_id"]
            for name in ss.REQUIRED:
                assert (store.final / name).exists(), name
            sel = json.loads((store.final / "selection.json").read_text())
            assert sel["sources_fit_order"] == c["sources"] and sel["n_sources"] == c["n_sources"]
            assert sel["training_work"]["n_train_windows"] == sum(
                len(prep[s].train_teacher) for s in c["sources"])
            assert (sel["fit_reproduction"] is not None) == (c["n_sources"] == 1)
            te = prep[c["target"]].test
            with np.load(store.final / "forecasts.npz") as z, \
                    np.load(store.final / "per_window.npz") as w:
                from probing.phase1_metrics import raw_window_metrics
                m = raw_window_metrics(c["target"], te.X, te.y_raw,
                                       z["pooled_q_raw"].astype(np.float64), QUANTILES,
                                       MEDIAN_INDEX)["mase_pw"]
                assert np.allclose(m, w["pooled__mase"], rtol=1e-5, atol=0)
                assert np.array_equal(z["rows_test"], w["rows_test"])
            ad, meta = load_adapter(Path(tmp) / "adapters" / c["target"] / c["cell_id"] /
                                    f"noa__{lodo.depth_label(D6)}.npz")
            assert meta["sources"] == c["sources"]
        # resume: nothing is refit
        real = lodo.run_lodo_cell

        def boom(*a, **k):
            raise AssertionError("a COMPLETE cell was refit on resume")
        lodo.run_lodo_cell = boom
        try:
            again = _run(pw, prep, h3, p1, spec, cells, tmp)
        finally:
            lodo.run_lodo_cell = real
        assert again["rc"] == 0 and all(v.get("skipped") for v in again["cells"].values())
        # a killed cell leaves only its staging dir; the cleanup is scoped to the owned cells
        c0, c1 = cells[0], cells[1]
        dead = drv._store(tmp, c0).final.with_name(c0["cell_id"] + ".building-99999")
        other = drv._store(tmp, c1).final.with_name(c1["cell_id"] + ".building-99998")
        for d in (dead, other):
            d.mkdir(parents=True)
            (d / "metrics.json").write_text("{partial")
        removed = phase2.clean_staging(tmp, ss.KIND, cells=[(c0["target"], c0["cell_id"])])
        assert len(removed) == 1 and not dead.exists() and other.exists()
        # a changed configuration is refused, never silently reused
        assert drv._store(tmp, c0).status("another-hash")[0] == "incompatible"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 8   driver loop: every cell COMPLETE with all artifacts; forecasts reproduce the "
          "per-window MASE; resume refits nothing; staging cleanup scoped; new config refused  OK")


def test_09_a_failed_fit_reproduction_stops_the_job():
    from experiments import run_phase2f_source_scaling as drv
    pw, prep, h3, p1, spec, cells = _mock_world()
    lab = lodo.depth_label(D6)
    bad = {t: dict(v) for t, v in h3.items()}
    first = drv._run_order(cells, MOCK_ROSTER)[0]
    assert first["n_sources"] == 1                    # the gates run before any larger fit
    s = first["sources"][0]
    rec = dict(bad[s]["fits"]["noa"][lab])
    rec["selected_epoch"] = int(rec["selected_epoch"]) + 10
    bad[s] = {**bad[s], "fits": {"noa": {lab: rec}}}
    calls = []
    real = lodo.run_lodo_cell

    def counting(*a, **k):
        calls.append(a[2]["n_sources"])
        return real(*a, **k)
    tmp = Path(tempfile.mkdtemp())
    lodo.run_lodo_cell = counting
    try:
        st = _run(pw, prep, bad, p1, spec, cells, tmp)
    finally:
        lodo.run_lodo_cell = real
    try:
        assert st["rc"] == 4 and calls == [1], (st["rc"], calls)
        f = tmp / ss.KIND / "runs" / "test" / "fit_reproduction_failure.json"
        assert f.exists() and not json.loads(f.read_text())["reproduction"]["passed"]
        assert not list((tmp / ss.KIND).glob("*/N*/COMPLETE"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 9   a failed N=1 reproduction stops the job after that one fit (exit 4), writes the "
          "diagnostics, commits no cell                                     OK")


def test_10_outputs_are_isolated_from_every_other_experiment():
    from experiments import run_phase2f_source_scaling as drv
    res = REPO_ROOT / "results"
    scratch = Path(tempfile.mkdtemp())
    try:
        ok_ad = scratch / "chronos2" / "phase2f_source_scaling" / "adapters"
        ss.assert_isolated(drv.DEFAULT_ROOT, ok_ad, REPO_ROOT, scratch=scratch)
        for bad in ("three_model_phase2_lodo", "three_model_phase2_lodo/lodo",
                    "three_model_phase2d_pooled", "three_model_phase2e_noncontig/x",
                    "three_model_phase2", "three_model_phase2/h3", "three_model_phase1_q9_expanded"):
            try:
                ss.assert_isolated(res / bad, ok_ad, REPO_ROOT, scratch=scratch)
                raise AssertionError(f"output root {bad} accepted")
            except ValueError:
                pass
        try:
            ss.assert_isolated(res, ok_ad, REPO_ROOT, scratch=scratch)
            raise AssertionError("an output root CONTAINING other trees was accepted")
        except ValueError:
            pass
        for bad in (scratch / "chronos2_phase2" / "adapters_lodo", scratch / "chronos2_phase2",
                    scratch / "chronos2_phase2" / "adapters" / "chronos2",
                    scratch / "chronos2" / "phase2e_noncontig" / "adapters", scratch):
            try:
                ss.assert_isolated(drv.DEFAULT_ROOT, bad, REPO_ROOT, scratch=scratch)
                raise AssertionError(f"adapter root {bad} accepted")
            except ValueError:
                pass
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    a = drv.parse_args([])
    assert Path(a.output_root).name == "three_model_phase2f_source_scaling"
    assert "phase2f_source_scaling" in a.adapter_root and "adapters_lodo" not in a.adapter_root
    try:
        drv.main(["--plan", "--output-root", str(res / "three_model_phase2_lodo")])
        raise AssertionError("the driver accepted the Phase 2c tree as its output root")
    except ValueError:
        pass
    print(" 10  output / adapter roots overlapping Phase 1, H3, 2c, 2d or 2e (or containing them)"
          " are refused before anything is read; defaults are dedicated          OK")


def test_11_n13_is_the_phase2c_main_cell():
    pw = t2c._pathway()
    prep = t2c._prepared(pw, t2c._mock_data(pw))
    h3, p1, spec = _h3_world(pw, prep, D6, ["m5"])
    full = next(c for c in ss.plan_cells(MOCK_ROSTER, targets=["m5"], seeds=MOCK_SEEDS,
                                         n_grid=MOCK_GRID, depth=D6) if c["n_sources"] == 4)
    c2c = next(c for c in lodo.plan_cells(MOCK_ROSTER, depths=[D6], targets=["m5"],
                                          variants=[lodo.MAIN]))
    assert full["sources"] == c2c["sources"] and full["excluded"] == c2c["excluded"]
    a = lodo.run_lodo_cell(pw, prep, full, phase1_arrays=p1["m5"], h3_ref=h3["m5"],
                           specific_adapter=spec["m5"][0], fit_cfg=FAST, log=None)
    b = lodo.run_lodo_cell(pw, prep, c2c, phase1_arrays=p1["m5"], h3_ref=h3["m5"],
                           specific_adapter=spec["m5"][0], fit_cfg=FAST, log=None)
    for x, y in zip(t2c._arrays(a["adapter"]), t2c._arrays(b["adapter"])):
        assert np.array_equal(x, y)
    assert np.array_equal(a["per_window"]["pooled"]["mase"], b["per_window"]["pooled"]["mase"])
    deps = {t: {"fit_hash": f"fit-{t}", "window_digest": f"wd-{t}"} for t in MOCK_ROSTER}
    cf = ss.cell_config(full, deps, h3["m5"], FAST)
    cc = lodo.cell_config(c2c, deps, h3["m5"], FAST)
    assert {k: cf[k] for k in cc} == cc                # every 2c identity field, unchanged
    assert phase1._digest(cf) != phase1._digest(cc)    # but a separate Phase 2f identity
    print(" 11  the full-pool Phase 2f cell equals the Phase 2c main cell bitwise (same fold, "
          "same code); its config extends the 2c one, with its own hash        OK")


# --------------------------------------------------------------------------- #
def _fake_cell(root, cell, rng, deg, spec_deg=0.05, hard_deg=0.6, n=40, tamper=False):
    """Artifacts of one cell in the exact on-disk format, from synthetic per-window arrays (no fit):
    the tables and the report must work from artifacts alone."""
    cid = np.arange(n) // 2
    tr = np.random.default_rng(ROSTER.index(cell["target"]))    # one test split per target
    nat = tr.uniform(0.8, 1.2, n)
    arms = {"native": nat, "hard": nat * (1 + hard_deg) + tr.normal(0, 0.003, n),
            "specific": nat * (1 + spec_deg) + tr.normal(0, 0.003, n),
            "pooled": nat * (1 + deg) + rng.normal(0, 0.003, n)}
    pw = {a: {"loss": v, "mase": v, "mae": v, "wql_num": v, "wql_den": np.ones(n)}
          for a, v in arms.items()}
    stats = lodo.cell_statistics(pw, np.ones(n), cid, B=200, seed=0)
    k = cell["n_sources"]
    fit = {"fit": {"selected_wd": 1.0, "selected_epoch": 500, "selected_val": 0.1,
                   "hard_cut_val": 0.2, "selected_hard_cut": False,
                   "candidates": [{"wd": 1.0, "status": "completed", "val_curve": [[0, 0.2],
                                                                                 [500, 0.1]],
                                   "train_curve": [[1500, 0.1]], "final_train_loss": 0.1,
                                   "best_val": 0.1}],
                   "grid": [1.0], "epochs": 1500, "lr": 1e-3, "eval_every": 25,
                   "optimizer": "mock", "seed": 0},
           "flags": {"selected_hard_cut": False, "late_checkpoint": False, "grid_max_wd": False},
           "criterion_check": {}, "fit_seconds": 55.0 * k,
           "per_source_val": [{"source": s, "n_train": 1394, "n_val": 262}
                              for s in cell["sources"]],
           "n_train_rows": 1394 * k, "n_val_rows": 262 * k, "equal_train_counts": True}
    res = {"cell": cell, "label": "L6", "fit": fit, "per_window": pw,
           "verification": {"all_passed": not tamper}, "stats": stats,
           "timings": {"fit_s": 55.0 * k}, "fit_cfg": FAST, "test_cluster_ids": cid,
           "test_rows": np.arange(n)}
    deps = {t: {"fit_hash": f"fit-{t}", "window_digest": f"wd-{t}"} for t in ROSTER}
    ccfg = ss.cell_config(cell, deps, {"config_hash": "h3mock"}, FAST)
    chash = phase1._digest(ccfg)
    store = Phase2CellStore(root, ss.KIND, cell["target"], cell["cell_id"], required=ss.REQUIRED)
    stage = store.begin()
    lodo.save_lodo_cell(stage, res, ccfg, chash, {"path": "mock", "sha256": "x"},
                        {"path": "mock", "sha256": "y"})
    rep = ({"passed": True, "bitwise_adapter": True} if k == 1 else None)
    ss.save_extras(stage, cell, res, {"pooled_q_raw": np.zeros((n, 9, 64), np.float32)},
                   work=ss.training_work(fit, 4), fit_repro=rep, job="mock")
    store.commit(stage, chash)


def _fake_design(root, targets, rng):
    """LOOP Seattle improves strongly with N; the others are flat (truth known to the test)."""
    for c in ss.plan_cells(ROSTER, targets=targets):
        n = c["n_sources"]
        deg = {1: 0.40, 4: 0.25, 8: 0.15, 13: 0.10}[n] if c["target"] == "LOOP_SEATTLE_5T" \
            else 0.12
        _fake_cell(root, c, rng, deg)


def test_12_tables_figures_report_and_draft_from_artifacts():
    from experiments import make_phase2f_report as rep
    from experiments import make_phase2f_tables as mt
    tmp = Path(tempfile.mkdtemp())
    rng = np.random.default_rng(0)
    try:
        _fake_design(tmp, ["LOOP_SEATTLE_5T", "m5"], rng)
        s = mt.build(tmp, tmp / "paper", latency=tmp / "none.csv", lodo_root=tmp / "none",
                     boot_b=300)
        assert not s["complete"] and s["n_cells"] == 20 and len(s["missing"]) == 20
        steps = list(__import__("csv").DictReader(open(tmp / "combined" /
                                                       "source_scaling_steps.csv")))
        assert len(steps) == 2 * 3 * 4
        loop = [x for x in steps if x["target"] == "LOOP_SEATTLE_5T" and x["step"] == "1->13"]
        assert all(x["direction"] == "improves" for x in loop)
        q1 = s["analysis"]["q1_larger_pool_improves"]
        assert q1["overall_1_to_13"]["improves"] >= 3 and q1["n_pairs"] == 6
        assert s["analysis"]["q2_monotone"]["n_monotone_nonincreasing_point"] >= 3
        assert "LOOP_SEATTLE_5T" in s["analysis"]["q6_diminishing_returns"][
            "targets_strictly_diminishing"]           # gains 5 > 2.5 > 1 pp per added source
        for f in ("source_scaling_main.pdf", "source_scaling_gap_to_specific.pdf",
                  "source_scaling_per_target.pdf", "source_scaling_aggregate.png",
                  "source_scaling_cells_table.tex", "source_scaling_orders_table.tex",
                  "source_scaling_macros.tex"):
            assert (tmp / "paper" / f).exists(), f
        assert "INCOMPLETE" in (tmp / "paper" / "source_scaling_cells_table.tex").read_text()
        r = rep.build(tmp)
        md, rb = Path(r["report"]).read_text(), Path(r["rebuttal_draft"]).read_text()
        assert "INCOMPLETE RESULTS" in md and "INCOMPLETE RESULTS" in rb
        assert "2 of the four planned target datasets" in rb, rb
        for sec in ("## Design", "## Coverage and gates", "## Results per target",
                    "## Scaling analysis", "## Aggregate", "## Fixed limitations"):
            assert sec in md, sec
        for bad in ("--", "representative"):
            assert bad not in rb, bad
        # the full design: complete, no banner
        _fake_design(tmp, ["monash_london_smart_meters", "boom_hourly"], rng)
        s = mt.build(tmp, tmp / "paper", latency=tmp / "none.csv", lodo_root=tmp / "none",
                     figures=False, boot_b=300)
        assert s["complete"] and s["n_cells"] == 40
        r = rep.build(tmp)
        md, rb = Path(r["report"]).read_text(), Path(r["rebuttal_draft"]).read_text()
        assert "INCOMPLETE" not in md and "INCOMPLETE" not in rb
        assert all(n in rb for n in ("Loop Seattle", "London Smart Meters", "BOOM", "M5"))
        assert "do not show that a larger" in rb
        for bad in ("--", "representative"):
            assert bad not in rb, bad
        # a cell whose gates failed blocks the tables
        bad_cell = next(c for c in ss.plan_cells(ROSTER, targets=["m5"]) if c["n_sources"] == 4)
        _fake_cell(tmp, bad_cell, rng, 0.1, tamper=True)
        try:
            mt.build(tmp, tmp / "paper", latency=tmp / "none.csv", lodo_root=tmp / "none",
                     figures=False, boot_b=100)
            raise AssertionError("a gate-failed cell was tabulated")
        except RuntimeError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 12  tables, figures, report and rebuttal draft rebuild from artifacts only; paired "
          "steps recover a known trend; INCOMPLETE stamped; failed gates block   OK")


def test_13_job_script_contract():
    txt = (REPO_ROOT / "job_phase2f_source_scaling.sh").read_text()
    for need in ("#SBATCH --account=def-irina", "#SBATCH --gres=gpu:a100:1",
                 "#SBATCH --signal=B:USR1@900",
                 "#SBATCH --output=results/three_model_phase2f_source_scaling/logs/%x-%j.out",
                 "source ./job_rebuttal_common.sh",
                 "rb_stage tests python -m tests.test_phase2f_source_scaling",
                 "python -m experiments.run_phase2f_source_scaling --plan --check-inputs",
                 "python -m experiments.make_phase2f_tables",
                 "python -m experiments.make_phase2f_report", "CUBLAS_WORKSPACE_CONFIG=:4096:8"):
        assert need in txt, need
    hours = int(re.search(r"#SBATCH --time=(\d+):", txt).group(1))
    assert 1 <= hours <= 24
    i_beg, i_main, i_end = (txt.index("rb_main_begin"), txt.index("rb_stage main"),
                            txt.index("rb_main_end"))
    assert i_beg < i_main < i_end < txt.index("rb_stage tables") < txt.index("rb_stage archive")
    arch = txt[txt.index("rb_stage archive"):]
    assert 'if [ "$RB_MAIN_RC" = 0 ]' in txt[:txt.index("rb_stage archive")]
    for other in ("phase2_lodo", "phase2d", "phase2e", "adapters_lodo", "chronos2_phase2/"):
        assert other not in arch.split("else")[0], other
    print(" 13  job script: full A100, timeout continuation, tests -> plan -> main (GPU) -> "
          "tables -> report -> archive only after a clean main stage, own trees only   OK")


def main(argv=None):
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"\nPHASE-2f SOURCE-SCALING CONTRACTS  ({len(tests)} model-free groups)\n" + "=" * 78)
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
        print(f"{len(failed)} of {len(tests)} Phase-2f contracts FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(tests)} Phase-2f contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
