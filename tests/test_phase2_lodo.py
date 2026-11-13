"""PRE-RUN CONTRACT SUITE for Phase 2c (pooled LODO adapter transfer, Chronos-2).

Model-free: no checkpoint, no dataset, no feature cache. A Chronos-2-SHAPED mock pathway (4 forecast
slots, 21 native quantiles, frozen norm + residual head) on real roster tags, so MASE uses the real
seasonal periods. Runs on a login node in well under a minute with 2 threads:

    python -m tests.test_phase2_lodo

Numbered so a failure names the requirement it broke.
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2, registry                                      # noqa: E402
from probing import phase2_lodo as lodo                                           # noqa: E402
from probing.phase2 import QUANTILES, Phase2CellStore                             # noqa: E402
from probing.phase2_align import (ResidualAdapter, fit_residual_adapter,          # noqa: E402
                                  load_adapter, mse_loss, save_adapter)

torch.set_num_threads(2)

MOCK_TAGS = ("m4_hourly", "monash_electricity_hourly", "sg_carpark", "electricity_15min", "m5")
NATIVE_Q = (0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7,
            0.75, 0.8, 0.85, 0.9, 0.95, 0.99)
FAST = {"epochs": 30, "eval_every": 10, "wd_grid": [0.0, 1.0], "boot_b": 200, "boot_seed": 0,
        "seed": 0}
D = 16


class _Norm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return self.weight * x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6)


class _Head(nn.Module):
    """ResidualBlock-shaped: hidden + output_layer + residual."""

    def __init__(self, d, n_out):
        super().__init__()
        self.hidden = nn.Linear(d, 32)
        self.output_layer = nn.Linear(32, n_out)
        self.residual = nn.Linear(d, n_out)
        with torch.no_grad():
            for p in self.parameters():
                p.mul_(0.3)

    def forward(self, x):
        return self.output_layer(torch.relu(self.hidden(x))) + self.residual(x)


def _pathway(seed=0):
    from probing.phase2_pathways import Chronos2Pathway
    torch.manual_seed(seed)

    class MockPW(Chronos2Pathway):
        pass
    MockPW.d = D
    return MockPW(_Head(D, len(NATIVE_Q) * 16), _Norm(D), list(NATIVE_Q))


def _mock_data(pw, seed=0, sizes=None):
    """One CellData per mock tag: h_l = a dataset-specific rotation of h_L plus noise, so a shared
    adapter can recover part of the map and a per-dataset one more."""
    from probing.phase2_pathways import CellData, SplitData
    spec = phase1.model_spec("chronos2")
    L = spec.reference_index
    rng = np.random.default_rng(seed)
    shared = {l: np.linalg.qr(rng.normal(size=(D, D)))[0] for l in lodo.LODO_DEPTHS}
    sizes = sizes or {"train": 48, "val": 20, "test": 24}
    out = {}
    for k, tag in enumerate(MOCK_TAGS):
        own = {l: np.linalg.qr(np.eye(D) + 0.3 * rng.normal(size=(D, D)))[0]
               for l in lodo.LODO_DEPTHS}
        splits = {}
        for s, n in sizes.items():
            hL = (rng.normal(size=(n, 4, D)) + 0.2 * k).astype(np.float32)
            feats = {L: hL}
            for l in lodo.LODO_DEPTHS:
                M = shared[l] @ own[l]
                feats[l] = (hL @ M.T + 0.3 * rng.normal(size=hL.shape)).astype(np.float32)
            X = (50 + np.cumsum(rng.normal(size=(n, 512)), axis=1)).astype(np.float32)
            mu, sd = X.astype(np.float64).mean(1), np.maximum(X.astype(np.float64).std(1), 1e-6)
            with torch.no_grad():
                z = pw.q9(pw.outputs(torch.as_tensor(hL))).numpy()
            target = (z[:, 4, :] + 0.1 * rng.normal(size=(n, 64))).astype(np.float32)
            y_raw = mu[:, None] + sd[:, None] * np.sinh(target.astype(np.float64))
            splits[s] = SplitData(feats, target, y_raw, X, {"mu": mu, "sd": sd},
                                  np.arange(n) // 2, np.arange(n))
        out[tag] = CellData("chronos2", tag, splits, spec.depth_labels, L, {})
    return out


def _prepared(pw, data):
    return {t: lodo.prepare_dataset(pw, d, lodo.LODO_DEPTHS) for t, d in data.items()}


def _mock_cell(target, depth, variant=lodo.MAIN):
    roster = list(MOCK_TAGS)
    excluded = {target} | (set(lodo.SERIES_SIBLINGS.get(target, ())) if variant == lodo.MAIN
                           else set())
    src = [t for t in roster if t not in excluded]
    return {"variant": variant, "target": target, "depth": depth, "sources": src,
            "excluded": [t for t in roster if t not in src]}


def _h3_ref_and_specific(pw, prepared, target, depth, seed=1):
    """A dataset-specific adapter and an H3-shaped reference that the SAME code scores."""
    torch.manual_seed(seed)
    ad = ResidualAdapter(D)
    with torch.no_grad():
        ad.delta.copy_(0.05 * torch.randn(D, D))
    ad.eval()
    te = prepared[target].test
    L = prepared[target].reference_index
    nat = lodo.score_arm(pw, te, target, te.feats[L])
    hard = lodo.score_arm(pw, te, target, te.feats[depth])
    spec = lodo.score_arm(pw, te, target, te.feats[depth], ad)
    n = len(te.target)
    H = np.zeros((L + 1, n))
    S = np.zeros((L + 1, n))
    H[depth], S[depth] = hard["mase"], spec["mase"]
    ref = {"config_hash": "h3mock", "arrays": {"native_test_mase": nat["mase"],
                                               "hard__test_mase": H, "noa__test_mase": S,
                                               "cluster_ids_test": np.asarray(te.cluster_ids)},
           "fits": {}}
    p1 = {"native_loss_window": nat["loss"], "native_mase_window": nat["mase"]}
    return ref, p1, ad


def _arrays(ad):
    return ad.delta.detach().numpy().copy(), ad.bias.detach().numpy().copy()


# =========================================================================== #
def test_01_folds_sibling_exclusion_and_plan():
    roster = list(registry.roster("paper14"))
    assert len(roster) == 14
    for t in roster:
        src = lodo.fold_sources(t, roster)
        assert t not in src and src == [x for x in roster if x in src]
        if t in lodo.SERIES_SIBLINGS:
            assert len(src) == 12 and not set(lodo.SERIES_SIBLINGS[t]) & set(src), t
        else:
            assert len(src) == 13, t
    for t in lodo.SERIES_SIBLINGS:
        src = lodo.fold_sources(t, roster, lodo.SENSITIVITY)
        assert len(src) == 13 and set(lodo.SERIES_SIBLINGS[t]) <= set(src)
    try:
        lodo.fold_sources("m5", roster, lodo.SENSITIVITY)
        raise AssertionError("a sibling_included fold was built for a target with no sibling")
    except ValueError:
        pass
    cells = lodo.plan_cells(roster)
    main = [c for c in cells if c["variant"] == lodo.MAIN]
    sens = [c for c in cells if c["variant"] == lodo.SENSITIVITY]
    assert len(main) == 42 and len(sens) == 6
    assert {c["target"] for c in sens} == set(lodo.SERIES_SIBLINGS)
    for spec in ("1/2", "1/3"):
        n = int(spec.split("/")[1])
        parts = [lodo.chunk(roster, f"{i}/{n}") for i in range(1, n + 1)]
        assert sum(parts, []) == roster and all(parts)
    print(" 1   14 folds; target never a source; Electricity folds drop the sibling (12 src.);"
          " sensitivity = the 2 sibling targets x 3 depths; chunks partition   OK")


def test_02_equal_weight_criterion():
    torch.manual_seed(0)
    p, y = torch.randn(30, 4, 7), torch.randn(30, 4, 7)
    seg = [("a", 0, 10), ("b", 10, 20), ("c", 20, 30)]
    eq = lodo.equal_weight_mse(seg)(p, y)
    assert torch.allclose(eq, mse_loss(p, y), rtol=1e-6, atol=0), (eq, mse_loss(p, y))
    seg2 = [("a", 0, 5), ("b", 5, 30)]
    manual = 0.5 * (((p[:5] - y[:5]) ** 2).mean() + ((p[5:] - y[5:]) ** 2).mean())
    got = lodo.equal_weight_mse(seg2)(p, y)
    assert torch.allclose(got, manual) and not torch.allclose(got, mse_loss(p, y))
    # replicating one source's rows 10x must not change its weight
    pa, ya = p[:5].repeat(10, 1, 1), y[:5].repeat(10, 1, 1)
    p3, y3 = torch.cat([pa, p[5:]]), torch.cat([ya, y[5:]])
    assert torch.allclose(lodo.equal_weight_mse([("a", 0, 50), ("b", 50, 75)])(p3, y3), manual)
    try:
        lodo.equal_weight_mse([("a", 0, 0)])
        raise AssertionError("an empty segment was accepted")
    except ValueError:
        pass
    print(" 2   criterion = mean of per-source MSEs: equals the H3 pooled mean at equal counts,"
          " invariant to a source's window count                       OK")


def test_03_target_train_val_never_reach_the_fit():
    pw = _pathway()
    data = _mock_data(pw)
    prep = _prepared(pw, data)
    target, depth = "m4_hourly", 6
    src = _mock_cell(target, depth)["sources"]
    a = lodo.fit_pooled_adapter(pw, prep, src, depth, fit_cfg=FAST, log=None)
    rng = np.random.default_rng(9)
    p = prep[target]
    poisoned = dataclasses.replace(
        p, train_states={l: np.full_like(v, np.nan) for l, v in p.train_states.items()},
        val_states={l: rng.normal(size=v.shape).astype(np.float32) * 1e3
                    for l, v in p.val_states.items()},
        train_teacher=np.full_like(p.train_teacher, np.nan),
        val_teacher=rng.normal(size=p.val_teacher.shape).astype(np.float32),
        test=dataclasses.replace(p.test, feats={l: np.full_like(v, np.nan)
                                                for l, v in p.test.feats.items()}))
    prep2 = {**prep, target: poisoned}
    b = lodo.fit_pooled_adapter(pw, prep2, src, depth, fit_cfg=FAST, log=None)
    for x, y in zip(_arrays(a["adapter"]), _arrays(b["adapter"])):
        assert np.array_equal(x, y), "the target's train/val/test influenced the pooled adapter"
    assert a["fit"]["selected_wd"] == b["fit"]["selected_wd"]
    assert a["fit"]["selected_epoch"] == b["fit"]["selected_epoch"]
    try:
        lodo.pooled_split(prep, src, "test", depth)
        raise AssertionError("a pooled fit could read a test split")
    except ValueError:
        pass
    print(" 3   target train/val/test replaced by NaN/noise -> pooled adapter BITWISE unchanged;"
          " pooled data can never read a test split                       OK")


def test_04_no_future_labels_reach_the_fit():
    pw = _pathway()
    data = _mock_data(pw)
    prep = _prepared(pw, data)
    poisoned = {}
    for t, d in data.items():
        sp = dict(d.splits)
        for s in ("train", "val"):
            sp[s] = dataclasses.replace(sp[s], target=np.full_like(sp[s].target, np.nan),
                                        y_raw=np.full_like(sp[s].y_raw, np.nan))
        poisoned[t] = dataclasses.replace(d, splits=sp)
    prep2 = _prepared(pw, poisoned)
    for t in MOCK_TAGS:
        assert prep[t].fingerprints == prep2[t].fingerprints, t
    src = _mock_cell("m5", 9)["sources"]
    a = lodo.fit_pooled_adapter(pw, prep, src, 9, fit_cfg=FAST, log=None)
    b = lodo.fit_pooled_adapter(pw, prep2, src, 9, fit_cfg=FAST, log=None)
    for x, y in zip(_arrays(a["adapter"]), _arrays(b["adapter"])):
        assert np.array_equal(x, y), "a ground-truth future reached the adapter fit"
    print(" 4   every train/val future (normalized AND raw) set to NaN -> identical fit inputs"
          " and a bitwise-identical adapter: the objective is label-free   OK")


def test_05_one_source_pooled_fit_is_the_h3_fit():
    pw = _pathway()
    prep = _prepared(pw, _mock_data(pw))
    t, l = "sg_carpark", 4
    a = lodo.fit_pooled_adapter(pw, prep, [t], l, fit_cfg=FAST, log=None)
    P = prep[t]

    def T(x):
        return torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float32)
    h3 = fit_residual_adapter(d=D, pathway_fn=pw.outputs, train_x=T(P.train_states[l]),
                              train_y=T(P.train_teacher), val_x=T(P.val_states[l]),
                              val_y=T(P.val_teacher), train_loss=mse_loss, val_criterion=mse_loss,
                              wd_grid=FAST["wd_grid"], epochs=FAST["epochs"], lr=phase2.ADAPTER_LR,
                              eval_every=FAST["eval_every"], seed=0)
    for x, y in zip(_arrays(a["adapter"]), _arrays(h3["adapter"])):
        assert np.allclose(x, y, rtol=0, atol=1e-7), np.abs(x - y).max()
    assert a["fit"]["selected_wd"] == h3["selected_wd"]
    assert a["fit"]["selected_epoch"] == h3["selected_epoch"]
    assert a["fit"]["grid"] == h3["grid"] and a["fit"]["optimizer"] == h3["optimizer"]
    cc = a["criterion_check"]
    assert abs(cc["equal_weight_mean_of_per_source_hard"] - cc["fit_hard_cut_val"]) <= \
        1e-5 * cc["fit_hard_cut_val"]
    print(" 5   with one source the pooled fit IS the H3 NOA fit (same optimizer, grid, selection,"
          " adapter to 1e-7)                                                OK")


def test_06_cell_end_to_end_and_reproduction_gates():
    pw = _pathway()
    prep = _prepared(pw, _mock_data(pw))
    target, depth = "monash_electricity_hourly", 9
    cell = _mock_cell(target, depth)
    assert "electricity_15min" not in cell["sources"]
    ref, p1, spec_ad = _h3_ref_and_specific(pw, prep, target, depth)
    res = lodo.run_lodo_cell(pw, prep, cell, phase1_arrays=p1, h3_ref=ref,
                             specific_adapter=spec_ad, fit_cfg=FAST, log=None)
    v = res["verification"]
    assert v["all_passed"] and v["hard_vs_h3"]["bitwise"] and v["specific_vs_h3"]["bitwise"]
    assert v["identity_adapter_equals_hard_cut"]["passed"]
    s = res["stats"]
    for a in ("hard", "pooled", "specific"):
        assert np.isfinite(s["vs_native"][a]["mase"]["ratio"])
    assert s["absolute"]["native"]["mase"] > 0
    # a reference that the same code does NOT reproduce must abort the cell
    for key, arm in (("hard__test_mase", "hard_vs_h3"), ("noa__test_mase", "specific_vs_h3")):
        bad = {**ref, "arrays": {**ref["arrays"], key: ref["arrays"][key] * 1.001}}
        try:
            lodo.run_lodo_cell(pw, prep, cell, phase1_arrays=p1, h3_ref=bad,
                               specific_adapter=spec_ad, fit_cfg=FAST, log=None)
            raise AssertionError(f"{arm} did not fail on a 0.1% mismatch")
        except RuntimeError as e:
            assert arm in str(e), e
    bad_p1 = {**p1, "native_mase_window": p1["native_mase_window"] * 1.01}
    try:
        lodo.run_lodo_cell(pw, prep, cell, phase1_arrays=bad_p1, h3_ref=ref,
                           specific_adapter=spec_ad, fit_cfg=FAST, log=None)
        raise AssertionError("the native gate did not fail")
    except RuntimeError as e:
        assert "native_gate_vs_phase1" in str(e)
    # the sibling is refused as a source under the main protocol
    leak = {**cell, "sources": cell["sources"] + ["electricity_15min"]}
    try:
        lodo.run_lodo_cell(pw, prep, leak, phase1_arrays=p1, h3_ref=ref,
                           specific_adapter=spec_ad, fit_cfg=FAST, log=None)
        raise AssertionError("a main-protocol cell accepted the series sibling")
    except RuntimeError as e:
        assert "sibling" in str(e)
    print(" 6   cell end to end: native/hard/specific reproduce the references bitwise; a 0.1%"
          " hard/specific or 1% native mismatch aborts; sibling refused   OK")


def test_07_gap_closure_is_reported_only_when_there_is_a_gap():
    rng = np.random.default_rng(0)
    n = 40
    cid = np.arange(n) // 2
    nat = rng.uniform(0.9, 1.1, n)

    def stats(hard_shift, pooled_frac):
        hard = nat * (1 + hard_shift) + rng.normal(0, 0.002, n)
        pooled = nat + pooled_frac * (hard - nat)
        pw = {a: {"loss": v, "mase": v, "mae": v, "wql_num": v}
              for a, v in (("native", nat), ("hard", hard), ("pooled", pooled),
                           ("specific", pooled))}
        return lodo.cell_statistics(pw, np.ones(n), cid, B=400, seed=0)
    s = stats(0.5, 0.25)["gap"]["mase"]["pooled"]
    assert s["gap_sign"] == "positive" and abs(s["reported"] - 0.75) < 0.02, s
    s = stats(-0.2, 0.5)["gap"]["mase"]["pooled"]
    assert s["gap_sign"] == "negative" and s["reported"] is None, s
    s = stats(0.0, 0.5)["gap"]["mase"]["pooled"]
    assert s["gap_sign"] == "indistinguishable" and s["reported"] is None, s
    t = stats(0.5, 0.25)
    assert t["pooled_vs_hard"]["mase"]["improves"] and t["pooled_vs_hard"]["mase"]["improves_ci"]
    assert abs(t["pooled_vs_specific"]["mase"]["degradation_difference_pp"]) < 1e-9
    print(" 7   gap recovery reported only for a significantly POSITIVE hard-cut gap; negative /"
          " indistinguishable gaps -> None with the reason                 OK")


def _write_cell(root, pw, prep, cell, tamper_gate=False):
    ref, p1, spec_ad = _h3_ref_and_specific(pw, prep, cell["target"], cell["depth"])
    res = lodo.run_lodo_cell(pw, prep, cell, phase1_arrays=p1, h3_ref=ref,
                             specific_adapter=spec_ad, fit_cfg=FAST, log=None)
    deps = {t: {"fit_hash": f"fit-{t}", "window_digest": f"wd-{t}"} for t in MOCK_TAGS}
    ccfg = lodo.cell_config(cell, deps, ref, {**lodo.DEFAULT_FIT_CFG, **FAST})
    chash = phase1._digest(ccfg)
    store = Phase2CellStore(root, f"{lodo.KIND}/{cell['variant']}", cell["target"],
                            lodo.depth_label(cell["depth"]), required=lodo.LODO_REQUIRED)
    stage = store.begin()
    arec = save_adapter(Path(root) / "adapters" / f"{cell['target']}_{cell['depth']}.npz",
                        res["adapter"], {"target": cell["target"]})
    if tamper_gate:
        res["verification"]["all_passed"] = False
    lodo.save_lodo_cell(stage, res, ccfg, chash, arec, {"path": "mock", "sha256": "x"})
    store.commit(stage, chash, extra={"protocol": lodo.LODO_PROTOCOL_VERSION})
    return store, chash, res, arec


def test_08_cell_artifacts_resume_and_adapter_round_trip():
    pw = _pathway()
    prep = _prepared(pw, _mock_data(pw))
    tmp = Path(tempfile.mkdtemp())
    try:
        cell = _mock_cell("m5", 4)
        store, chash, res, arec = _write_cell(tmp, pw, prep, cell)
        assert store.status(chash) == ("complete", "validated")
        assert store.status("other")[0] == "incompatible"
        for name in lodo.LODO_REQUIRED:
            assert (store.final / name).exists(), name
        mk = json.loads((store.final / "COMPLETE").read_text())
        assert mk["protocol"] == lodo.LODO_PROTOCOL_VERSION
        fold = json.loads((store.final / "fold.json").read_text())
        assert fold["target"] not in fold["sources"]
        assert fold["chronos2_pretraining_provenance"]["status"] == "held_out"
        ad, _ = load_adapter(arec["path"])
        te = prep["m5"].test
        again = lodo.score_arm(pw, te, "m5", te.feats[4], ad)
        assert np.array_equal(again["mase"], res["per_window"]["pooled"]["mase"])
        with np.load(store.final / "per_window.npz") as z:
            assert np.array_equal(z["pooled__mase"], res["per_window"]["pooled"]["mase"])
            assert np.array_equal(z["cluster_ids_test"], np.asarray(te.cluster_ids))
        fit = json.loads((store.final / "fit.json").read_text())
        assert all("val_curve" in c and "train_curve" in c for c in fit["candidates"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 8   every artifact written atomically; resume = complete, a new config = incompatible;"
          " saved adapter + per-window arrays reproduce the cell bitwise    OK")


def test_09_tables_and_figures_from_artifacts_only():
    from experiments import make_phase2_lodo_tables as mt
    pw = _pathway()
    prep = _prepared(pw, _mock_data(pw))
    tmp = Path(tempfile.mkdtemp())
    try:
        for target in ("monash_electricity_hourly", "m5", "sg_carpark"):
            for depth in lodo.LODO_DEPTHS:
                _write_cell(tmp, pw, prep, _mock_cell(target, depth))
        for depth in lodo.LODO_DEPTHS:
            _write_cell(tmp, pw, prep, _mock_cell("monash_electricity_hourly", depth,
                                                  lodo.SENSITIVITY))
        s = mt.build(tmp, tmp / "paper", tmp / "no_latency.csv")
        assert s["n_cells"] == 12 and not s["complete"] and s["n_expected"] == 48
        rows = list(__import__("csv").DictReader(open(tmp / "combined" / "lodo_per_dataset.csv")))
        assert len(rows) == 12
        sens = [r for r in rows if r["variant"] == lodo.SENSITIVITY]
        assert all(r["unseen_series_by_adapter"] == "False" for r in sens)
        assert all(r["unseen_series_by_adapter"] == "True" for r in rows
                   if r["variant"] == lodo.MAIN)
        agg = {(a["label"], a["group"]): a for a in s["aggregate"]}
        assert agg[("L4", "all")]["n"] == 3 and agg[("L4", "held_out")]["n"] == 1
        assert len(s["sibling_sensitivity"]) == 3
        for f in ("lodo_per_dataset.pdf", "lodo_aggregate.png", "lodo_per_dataset_table.tex",
                  "lodo_aggregate_table.tex", "lodo_sibling_sensitivity_table.tex",
                  "lodo_macros.tex"):
            assert (tmp / "paper" / f).exists(), f
        assert "PARTIAL" in (tmp / "paper" / "lodo_aggregate_table.tex").read_text()
        # a cell whose gates did not pass must block the tables
        _write_cell(tmp, pw, prep, _mock_cell("m4_hourly", 4), tamper_gate=True)
        try:
            mt.build(tmp, tmp / "paper", tmp / "no_latency.csv", figures=False)
            raise AssertionError("a gate-failed cell was tabulated")
        except RuntimeError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 9   tables/figures/macros rebuild from cell artifacts only; sensitivity cells are"
          " never 'unseen series'; partial runs stamped; failed gates block   OK")


def test_10_provenance_is_the_papers_table():
    roster = set(registry.roster("paper14"))
    assert set(lodo.CHRONOS2_PROVENANCE) == roster
    counts = {g: sum(v[0] == g for v in lodo.CHRONOS2_PROVENANCE.values())
              for g in lodo.PROVENANCE_GROUPS}
    assert counts == {"held_out": 4, "not_listed": 3, "listed": 7}, counts
    assert {t for t, v in lodo.CHRONOS2_PROVENANCE.items() if v[0] == "held_out"} == \
        {"LOOP_SEATTLE_5T", "SZ_TAXI_15T", "m5", "monash_traffic"}
    tex = REPO_ROOT / "notes" / "help.tex"
    checked = False
    if tex.exists():
        txt = tex.read_text()
        i = txt.find(r"\label{tab:provenance}")
        body = txt[i:txt.find(r"\bottomrule", i)] if i >= 0 else ""
        names = {"M4": "m4_hourly", "Electricity": "monash_electricity_hourly",
                 "Uber TLC": "uber_tlc_hourly", "Wind Farms": "wind_farms_hourly",
                 "BOOM": "boom_hourly", "SG Carpark": "sg_carpark", "Coastal T-S": "coastal_ts",
                 "LOOP Seattle": "LOOP_SEATTLE_5T", "Electricity (15-minute)": "electricity_15min",
                 "SZ Taxi": "SZ_TAXI_15T", "London Smart Meters": "monash_london_smart_meters",
                 "M5": "m5", "Wiki Daily": "wiki_daily_100k", "Monash Traffic": "monash_traffic"}
        seen = 0
        for line in body.splitlines():
            parts = [p.strip() for p in line.split("&")]
            if len(parts) < 4 or parts[0] not in names:
                continue
            c2 = parts[1]
            status = ("held_out" if c2.startswith("held out") else
                      "not_listed" if c2.startswith("not listed") else
                      "listed" if c2.startswith("listed") else None)
            scope = "source" if "source" in c2 else "exact"
            assert lodo.CHRONOS2_PROVENANCE[names[parts[0]]] == (status, scope), (parts[0], c2)
            seen += 1
        assert seen == 14, seen
        checked = True
    print(" 10  provenance = the paper's Table (Chronos-2 column): 7 listed / 4 held out / 3 not"
          f" listed{' -- cross-checked against notes/help.tex' if checked else ''}       OK")


def test_11_preflight_gates_equal_the_cells_and_need_no_fit():
    """The driver's pre-fit stage must run the SAME gates as the cell (identical values) and fail
    on exactly the mismatches that abort a cell -- without fitting anything."""
    pw = _pathway()
    prep = _prepared(pw, _mock_data(pw))
    target, depth = "monash_electricity_hourly", 9
    cell = _mock_cell(target, depth)
    ref, p1, spec_ad = _h3_ref_and_specific(pw, prep, target, depth)
    before = {k: v.detach().clone() for k, v in spec_ad.state_dict().items()}
    pf = lodo.preflight_gates(pw, prep, target, depth, phase1_arrays=p1, h3_ref=ref,
                              specific_adapter=spec_ad, fit_cfg=FAST)
    assert pf["all_passed"] and not pf["failed"] and pf["label"] == "L9"
    assert set(lodo.PREFLIGHT_GATES) <= set(pf["gates"])
    assert all(torch.equal(before[k], v) for k, v in spec_ad.state_dict().items())
    res = lodo.run_lodo_cell(pw, prep, cell, phase1_arrays=p1, h3_ref=ref,
                             specific_adapter=spec_ad, fit_cfg=FAST, log=None)
    for k in lodo.PREFLIGHT_GATES:
        assert json.dumps(pf["gates"][k], sort_keys=True, default=str) == json.dumps(
            res["verification"][k], sort_keys=True, default=str), k
    for key, gate in (("hard__test_mase", "hard_vs_h3"), ("noa__test_mase", "specific_vs_h3"),
                      ("native_test_mase", "native_vs_h3")):
        bad = {**ref, "arrays": {**ref["arrays"], key: ref["arrays"][key] * 1.001}}
        r = lodo.preflight_gates(pw, prep, target, depth, phase1_arrays=p1, h3_ref=bad,
                                 specific_adapter=spec_ad, fit_cfg=FAST)
        assert not r["all_passed"] and gate in r["failed"], (gate, r["failed"])
    bad_p1 = {**p1, "native_mase_window": p1["native_mase_window"] * 1.01}
    r = lodo.preflight_gates(pw, prep, target, depth, phase1_arrays=bad_p1, h3_ref=ref,
                             specific_adapter=spec_ad, fit_cfg=FAST)
    assert not r["all_passed"] and "native_gate_vs_phase1" in r["failed"]
    rows = {**ref, "arrays": {**ref["arrays"],
                              "cluster_ids_test": ref["arrays"]["cluster_ids_test"][::-1].copy()}}
    r = lodo.preflight_gates(pw, prep, target, depth, phase1_arrays=p1, h3_ref=rows,
                             specific_adapter=spec_ad, fit_cfg=FAST)
    assert not r["all_passed"] and "test_rows_vs_h3" in r["failed"]
    print(" 11  preflight (driver, before any fit) = the cell's gates value for value; fails on "
          "a 0.1% hard/specific/native or 1% Phase-1 mismatch and on shuffled rows      OK")


def test_12_report_and_rebuttal_draft_from_tables_only():
    """make_phase2_lodo_report: built from the tables only, stamped INCOMPLETE on a partial run,
    states its coverage in the draft, and follows the paper's prose rules."""
    from experiments import make_phase2_lodo_report as rep
    from experiments import make_phase2_lodo_tables as mt
    pw = _pathway()
    prep = _prepared(pw, _mock_data(pw))
    tmp = Path(tempfile.mkdtemp())
    try:
        for target in ("monash_electricity_hourly", "m5"):
            for depth in lodo.LODO_DEPTHS:
                _write_cell(tmp, pw, prep, _mock_cell(target, depth))
        _write_cell(tmp, pw, prep, _mock_cell("monash_electricity_hourly", 9, lodo.SENSITIVITY))
        mt.build(tmp, tmp / "paper", tmp / "no_latency.csv", figures=False)
        pf = tmp / lodo.KIND / "runs" / "job1"
        pf.mkdir(parents=True)
        (pf / "preflight.json").write_text(json.dumps({"n_pairs": 6, "n_failed": 0,
                                                       "failed": [], "pairs": []}))
        s = rep.build(tmp)
        md = Path(s["report"]).read_text()
        rb = Path(s["rebuttal_draft"]).read_text()
        assert not s["complete"] and "INCOMPLETE RESULTS" in md and "INCOMPLETE RESULTS" in rb
        assert "2 of the 14 datasets" in rb, rb
        for sec in ("Coverage and gates", "Main protocol", "Per dataset", "Depth consistency",
                    "sibling sensitivity", "Fixed limitations"):
            assert sec in md, sec
        assert "6 of 6 (target, depth) pairs passed" in md
        assert "M5" in rb and "held out" in rb      # the held-out group is named in the draft
        for bad in ("--", "representative"):
            assert bad not in rb, bad
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 12  report + rebuttal draft from the tables only: INCOMPLETE stamped, coverage "
          "stated, held-out group named, no '--' / 'representative'                OK")


def main(argv=None):
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"\nPHASE-2c LODO CONTRACTS  ({len(tests)} model-free groups)\n" + "=" * 78)
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
        print(f"{len(failed)} of {len(tests)} LODO contracts FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(tests)} LODO contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
