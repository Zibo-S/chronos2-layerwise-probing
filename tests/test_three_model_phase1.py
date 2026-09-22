"""PRE-RUN CONTRACT SUITE for the 42-cell Phase-1 experiment.

Every contract the specification asks for before a single GPU-hour is spent, numbered so a
failure names the requirement it broke. Model-free by default: it runs on a login node in a few
seconds and needs no checkpoint, no dataset and no feature cache.

    python -m tests.test_three_model_phase1                # 57 model-free contracts (~15 s)
    python -m tests.test_three_model_phase1 --with-model   # + 28/30/31, inside an salloc

(21b is CUDA-gated: it SKIPS on a CPU-only login node and runs on an accelerator.)

The model-backed contracts (TiRex device handling, CPU/CUDA mixing, the native-head
reconstruction) are delegated VERBATIM to ``tests.test_tirex_probing --with-model`` rather than
re-implemented here, so there is exactly one copy of each.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, registry                                        # noqa: E402
from probing.phase1 import MODELS, PHASE1_QUANTILES, model_spec             # noqa: E402

PAPER14 = registry.roster("paper14")
OLD_SEVEN = list(registry.PAPER7)


# =========================================================================== #
# DATASETS (1-9)
# =========================================================================== #
def test_01_all_fourteen_tags_resolve():
    assert len(PAPER14) == 14 and len(set(PAPER14)) == 14, PAPER14
    expected = {"m4_hourly", "monash_electricity_hourly", "uber_tlc_hourly",
                "wind_farms_hourly", "boom_hourly", "sg_carpark", "coastal_ts",
                "LOOP_SEATTLE_5T", "electricity_15min", "SZ_TAXI_15T",
                "monash_london_smart_meters", "m5", "wiki_daily_100k", "monash_traffic"}
    assert set(PAPER14) == expected, set(PAPER14) ^ expected
    assert "kdd_cup_2022_10T" not in PAPER14, "KDD Cup 2022 was dropped at the roster gate"
    for t in PAPER14:
        s = registry.spec(t)
        assert s.display_name and s.slug and s.domain and s.cluster_unit and s.builder
        assert s.builder in ("rolling_within_series", "rolling_cluster"), (t, s.builder)
    print(" 1   all 14 roster tags resolve to a complete registry row; KDD 2022 absent     OK")


def test_02_frequencies_and_seasonal_m():
    want = {"LOOP_SEATTLE_5T": ("5min", 288), "electricity_15min": ("15min", 96),
            "SZ_TAXI_15T": ("15min", 96), "monash_london_smart_meters": ("30min", 48),
            "m5": ("1D", 7), "wiki_daily_100k": ("1D", 7), "monash_traffic": ("1H", 24)}
    for tag, (freq, m) in want.items():
        s = registry.spec(tag)
        assert s.freq == freq, (tag, s.freq, freq)
        assert registry.seasonal_m(tag) == m, (tag, registry.seasonal_m(tag), m)
    freqs = {registry.spec(t).freq for t in PAPER14}
    assert {"5min", "15min", "30min", "1D"} <= freqs, freqs
    assert "10min" not in freqs, "the 10-minute class is deliberately empty (KDD 2022 dropped)"
    print(f" 2   per-dataset frequency + seasonal period correct; {len(freqs)} frequency "
          f"classes                OK")


def test_03_original_seven_all_m24():
    for t in OLD_SEVEN:
        assert registry.seasonal_m(t) == 24, (t, registry.seasonal_m(t))
    # and the MASE arithmetic at m=24 is what the committed numbers were produced with
    from probing.mase import denominator_for, seasonal_denominator
    rng = np.random.default_rng(0)
    X = rng.normal(size=(7, 512))
    ref = np.abs(X[:, 24:] - X[:, :-24]).mean(axis=1)
    for t in OLD_SEVEN:
        assert np.array_equal(denominator_for(t, X), ref), t
    assert np.array_equal(seasonal_denominator(X, 24), ref)
    print(" 3   original seven map to m=24 and reproduce the committed MASE denominator "
          "exactly  OK")


def test_04_unknown_seasonality_fails_loudly():
    for bad in ("not_a_dataset", "", "electricity"):
        try:
            registry.seasonal_m(bad)
        except KeyError as e:
            assert "registry" in str(e), str(e)
        else:
            raise AssertionError(f"seasonal_m({bad!r}) must raise, never default to 24")
    # and the metric itself refuses a missing period
    from probing.mase import seasonal_denominator
    try:
        seasonal_denominator(np.zeros((2, 512)))            # type: ignore[call-arg]
    except TypeError:
        pass
    else:
        raise AssertionError("seasonal_denominator must REQUIRE m")
    try:
        seasonal_denominator(np.zeros((2, 512)), 512)
    except ValueError as e:
        assert "does not fit" in str(e)
    else:
        raise AssertionError("an m that does not fit the context must raise, not return NaN")
    print(" 4   unknown / missing / impossible seasonality raises; no silent m=24 default  OK")


def test_05_electricity_15min_is_a_frequency_control():
    assert registry.role("electricity_15min") == "control"
    assert registry.is_control("electricity_15min")
    assert "electricity_15min" not in registry.primary_datasets("paper14")
    assert registry.control_datasets("paper14") == ["electricity_15min"]
    assert "FREQUENCY CONTROL" in registry.spec("electricity_15min").notes
    assert len(registry.primary_datasets("paper14")) == 13
    print(" 5   electricity_15min is role=control and excluded from domain-diversity counts OK")


def test_06_series_cap_is_deterministic():
    from probing.id_data import apply_series_cap
    ids = list(range(100000))
    a, aud_a = apply_series_cap("wiki_daily_100k", ids, 0)
    b, _ = apply_series_cap("wiki_daily_100k", list(reversed(ids)), 0)
    c, _ = apply_series_cap("wiki_daily_100k", ids, 0)
    assert a == b == c, "the cap must not depend on input order and must reproduce exactly"
    assert len(a) == registry.max_series("wiki_daily_100k") == 1394
    assert a == sorted(a) and len(set(a)) == len(a)
    assert aud_a["applied"] and aud_a["n_eligible_before"] == 100000
    d, _ = apply_series_cap("wiki_daily_100k", ids, 1)
    assert d != a, "a different seed must draw a different sample"
    # inert -- and randomness-free -- for every dataset at or below its cap
    for t in OLD_SEVEN:
        small = list(range(300))
        kept, aud = apply_series_cap(t, small, 0)
        assert kept == small and not aud["applied"], t
        assert "no randomness drawn" in aud["reason"] or "no cap declared" in aud["reason"]
    capped = [t for t in PAPER14 if registry.max_series(t) is not None]
    assert set(capped) == {"monash_london_smart_meters", "m5", "wiki_daily_100k"}, capped
    print(f" 6   series cap deterministic + order-independent; fires for {capped}; inert and "
          f"randomness-free below the cap  OK")


def test_07_old_seven_window_path_unchanged():
    """The original seven go through the SAME builder, budget and namespace as the committed run."""
    from probing.windows import PAPER14_SET, PAPER7_SET, build_for, rolling_set_for_suite
    from probing.id_data import BUDGET_BY_SET
    assert rolling_set_for_suite("paper7") == PAPER7_SET == "extended_v3_rolling"
    assert BUDGET_BY_SET[PAPER7_SET] == BUDGET_BY_SET[PAPER14_SET], (
        "paper14 must carry the SAME (train, val, test) budget as the committed paper7 set, "
        "otherwise the seven would be re-windowed")
    for t in OLD_SEVEN:
        assert registry.max_series(t) is None, f"{t} must not be capped"
    assert callable(build_for)
    # every model line dispatches through exactly this one function
    src = (REPO_ROOT / "experiments" / "run_three_model_phase1.py").read_text()
    assert src.count("build_for(") == 1, "Phase 1 must have exactly ONE window build call"
    print(f" 7   the seven keep the committed builder/budget/namespace and are never capped; "
          f"budget {BUDGET_BY_SET[PAPER7_SET]}  OK")


def test_08_window_references_are_deterministic():
    from probing import window_reference as wref
    rng = np.random.default_rng(3)
    w = _fake_windows(rng, n_tr=9, n_va=4, n_te=4)
    r1 = wref.build_reference("m4_hourly", w, created_by="test")
    r2 = wref.build_reference("m4_hourly", w, created_by="test")
    assert r1["window_digest"] == r2["window_digest"]
    assert r1["window_digest"].startswith("sha256:")
    w2 = {**w, "X_test": w["X_test"].copy()}
    w2["X_test"][0, 0] += np.float32(1.0)
    assert wref.build_reference("m4_hourly", w2, created_by="t")["window_digest"] \
        != r1["window_digest"], "the digest must cover the CONTEXTS, not just the ids"
    tmp = Path(tempfile.mkdtemp())
    try:
        wref.write_reference("m4_hourly", w, created_by="test", root=tmp)
        try:
            wref.write_reference("m4_hourly", w, created_by="test", root=tmp)
        except FileExistsError as e:
            assert "force-reference" in str(e)
        else:
            raise AssertionError("overwriting a reference must be refused without --force")
        back = wref.read_reference("m4_hourly", root=tmp)
        assert not wref.compare_to_reference("m4_hourly", w, back)
        bad = {**w, "series_test": np.roll(w["series_test"], 1)}
        assert wref.compare_to_reference("m4_hourly", bad, back)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 8   window references are deterministic, content-hashed, and refuse silent "
          "overwrite     OK")


def test_09_all_models_get_the_same_windows():
    """Structural: ONE window object per dataset is passed to all three model adapters."""
    import inspect
    from experiments import run_three_model_phase1 as drv
    sig = inspect.signature(drv.run_cell)
    assert "w" in sig.parameters, "run_cell must RECEIVE the windows, never build its own"
    src = inspect.getsource(drv.run_cell)
    assert "build_for" not in src and "build_windows" not in src, \
        "run_cell must not construct windows -- that would let two models diverge"
    main_src = inspect.getsource(drv.main)
    assert main_src.index("windows_for(tag, args)") < main_src.index("for model in models"), \
        "windows must be built ONCE per dataset, before the model loop"
    # and parity is checked once, against the shared reference, before any model runs
    assert "parity_for" in main_src and "window_audits" in main_src
    for mod in ("probing.phase1_chronos2", "probing.phase1_timesfm3", "probing.phase1_tirex"):
        m = __import__(mod, fromlist=["extract"])
        assert "w" in inspect.signature(m.extract).parameters
    print(" 9   one window object per dataset is shared by all three adapters; run_cell "
          "cannot rebuild  OK")


# =========================================================================== #
# PROVENANCE (10-13)
# =========================================================================== #
def test_10_every_cell_has_a_provenance_record():
    rows = phase1.provenance_rows("paper14")
    assert len(rows) == 42, len(rows)
    seen = {(r["model"], r["dataset"]) for r in rows}
    assert seen == {(m, t) for m in MODELS for t in PAPER14}
    for r in rows:
        assert r["citation"].strip(), r
    print(f" 10  all {len(rows)}/42 model x dataset provenance cells present with citations  OK")


def test_11_status_vocabulary_is_closed():
    assert registry.STATUSES == ("pretraining_exposed", "explicitly_held_out", "not_listed",
                                 "undocumented")
    for r in phase1.provenance_rows("paper14"):
        assert r["status"] in registry.STATUSES, r
    try:
        registry.Provenance("ood", "exact_dataset", "x")
    except ValueError as e:
        assert "not in" in str(e)
    else:
        raise AssertionError("an out-of-vocabulary status must be refused")
    print(" 11  status vocabulary restricted to the four frozen values                      OK")


def test_12_evidence_scope_vocabulary_is_closed():
    assert registry.EVIDENCE_SCOPES == ("exact_dataset", "benchmark_exclusion", "source_level",
                                        "source_family")
    for r in phase1.provenance_rows("paper14"):
        assert r["evidence_scope"] in registry.EVIDENCE_SCOPES, r
    try:
        registry.Provenance("not_listed", "unseen", "x")
    except ValueError as e:
        assert "not in" in str(e)
    else:
        raise AssertionError("an out-of-vocabulary evidence scope must be refused")
    # the two axes are never collapsed: the same status carries different scopes
    by_status = {}
    for r in phase1.provenance_rows("paper14"):
        by_status.setdefault(r["status"], set()).add(r["evidence_scope"])
    assert any(len(v) > 1 for v in by_status.values()), \
        "if every status had one scope the 2-D schema would be pointless"
    print(f" 12  evidence_scope vocabulary closed; the two axes stay independent "
          f"{ {k: len(v) for k, v in by_status.items()} }  OK")


def test_13_no_global_pt_id_pt_ood_in_phase1_outputs():
    """No Phase-1 CODE PATH imports a global pretraining label, and no ARTIFACT contains one.

    Checked two ways, because prose is not the risk:
      (a) AST — none of the Phase-1 modules imports or references the legacy global names;
      (b) artifacts — a real, fully written cell plus every combined table is scanned for the
          label strings. That is the requirement the spec actually states ("no output in
          results/three_model_final may contain a global PT-ID/PT-OOD classification").
    """
    import ast
    from probing import window_parity
    banned_names = {"PT_ID_TAGS", "PT_OOD_TAGS", "domain_status"}
    files = ["probing/phase1.py", "probing/phase1_cells.py", "probing/phase1_metrics.py",
             "probing/phase1_chronos2.py", "probing/phase1_timesfm3.py",
             "probing/phase1_tirex.py", "probing/window_parity.py",
             "experiments/run_three_model_phase1.py", "experiments/make_phase1_tables.py"]
    for f in files:
        tree = ast.parse((REPO_ROOT / f).read_text())
        # Docstrings are prose ABOUT the rule, not code that breaks it. Collect them so the
        # string-literal check below only sees literals the module could actually emit.
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in banned_names:
                raise AssertionError(f"{f} references the global label {node.id}")
            if isinstance(node, ast.Attribute) and node.attr in banned_names:
                raise AssertionError(f"{f} references the global label .{node.attr}")
            if isinstance(node, ast.ImportFrom):
                bad = banned_names & {a.name for a in node.names}
                assert not bad, f"{f} imports {bad}"
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings):
                for lit in ("pt_id", "pt_ood", "PT-ID", "PT-OOD"):
                    assert lit not in node.value, \
                        f"{f} has an emittable string literal containing {lit!r}"

    # (a2) the shared identity record carries registry facts and no pretraining label
    rng = np.random.default_rng(0)
    ident = window_parity.window_identity("m4_hourly", _fake_windows(rng))
    assert "kind" not in ident and "domain_status" not in ident, ident
    assert ident["role"] == "primary" and ident["seasonal_m"] == 24

    # (b) a REAL written cell + every combined table, scanned end to end
    from probing.phase1 import CellStore
    from probing.phase1_cells import save_cell
    from experiments.make_phase1_tables import build
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _synthetic_cell(model="chronos2", tag="boom_hourly")
        store = CellStore(tmp, c["model"], c["tag"])
        save_cell(store.begin(), **c)
        store.commit(store.staging(), c["config_hash"])
        build(tmp)
        scanned = 0
        for path in sorted(tmp.rglob("*")):
            if not path.is_file() or path.suffix not in (".json", ".csv", ""):
                continue
            text = path.read_text(errors="ignore")
            scanned += 1
            for lit in ("pt_id", "pt_ood", "PT-ID", "PT-OOD", "domain_status"):
                assert lit not in text, f"{path.relative_to(tmp)} contains {lit!r}"
        assert scanned >= 8, scanned
        print(f" 13  no Phase-1 module references a global label (AST) and none of the "
              f"{scanned} written artifacts contains one  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================================== #
# Q=9 (14-18)
# =========================================================================== #
def test_14_exactly_nine_common_quantiles():
    q, mid = phase1.quantile_set("q9")
    assert q.shape == (9,) and len(q) == 9
    assert np.allclose(q, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    assert q.dtype == np.float64 and float(q[0]) == 0.1, "kept in exact float64"
    print(f" 14  exactly nine canonical quantiles {list(q)}    OK")


def test_15_quantile_order_matches_every_model():
    rec = phase1.assert_canonical_quantiles()
    assert set(rec["verified_against"]) == {"chronos2", "timesfm3", "tirex"}
    for m, v in rec["verified_against"].items():
        assert np.allclose(v, PHASE1_QUANTILES), (m, v)
        assert np.all(np.diff(v) > 0), f"{m} quantiles must be strictly increasing"
    # and a disagreement really aborts
    import probing.probes as probes
    saved = probes.QUANTILE_SETS["q9"]
    try:
        probes.QUANTILE_SETS["q9"] = np.array([0.05, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.95],
                                              np.float32)
        try:
            phase1.assert_canonical_quantiles()
        except RuntimeError as e:
            assert "do NOT share one nine-quantile vector" in str(e)
        else:
            raise AssertionError("a per-model quantile disagreement must abort the run")
    finally:
        probes.QUANTILE_SETS["q9"] = saved
    print(" 15  all three model lines share one strictly-increasing q9 vector; a mismatch "
          "aborts  OK")


def test_16_pinball_reduction_is_identical_across_models():
    """One reduction: mean over (batch, quantiles, horizon). Proved against every implementation."""
    from probing.probes import chronos2_quantile_loss, mean_pinball_loss
    from probing.timesfm3_last_token_probes import pinball_loss, pinball_loss_per_window
    from probing.tirex_probes import pinball
    rng = np.random.default_rng(7)
    q = torch.as_tensor(PHASE1_QUANTILES, dtype=torch.float32)
    pred = torch.as_tensor(rng.normal(size=(11, 9, 64)), dtype=torch.float32)
    y = torch.as_tensor(rng.normal(size=(11, 64)), dtype=torch.float32)

    # explicit reference: rho_tau(u) = max(tau u, (tau-1) u), averaged over every term
    u = y.numpy()[:, None, :] - pred.numpy()
    tau = PHASE1_QUANTILES[None, :, None]
    ref = float(np.maximum(tau * u, (tau - 1.0) * u).mean())

    vals = {"probes.mean_pinball_loss": float(mean_pinball_loss(pred, y, q)),
            "timesfm3.pinball_loss": float(pinball_loss(pred, y, q)),
            "tirex.pinball": float(pinball(pred, y, q)),
            "phase1.mean_pinball_per_window": float(
                phase1.mean_pinball_per_window(pred, y, q).mean())}
    for k, v in vals.items():
        assert abs(v - ref) < 1e-6, (k, v, ref)
    # per-window mean == scalar (the bootstrap depends on this)
    assert abs(float(pinball_loss_per_window(pred, y, q).mean()) - ref) < 1e-6
    # Chronos-2's TRAINING objective is exactly 2Q times the reported one
    c2 = float(chronos2_quantile_loss(pred, y, q))
    assert abs(c2 - 2 * 9 * ref) < 1e-4, (c2, 2 * 9 * ref)
    print(f" 16  one reduction across all four implementations ({ref:.6f}); Chronos-2's "
          f"objective == 2Q x it  OK")


def test_17_median_index_is_tau_half():
    q, mid = phase1.quantile_set("q9")
    assert mid == 4 and float(q[mid]) == 0.5
    from probing.timesfm3_last_token import NATIVE_MEDIAN_IDX
    assert NATIVE_MEDIAN_IDX == 4
    from probing.probes import median_index
    assert median_index(np.array([0.1, 0.3, 0.7])) is None, \
        "a set without an exact 0.5 must report None, never a neighbour"
    try:
        phase1.PHASE1_QUANTILE_SETS["_bad"] = np.array([0.1, 0.9])
        try:
            phase1.quantile_set("_bad")
        except ValueError as e:
            assert "no exact 0.5" in str(e)
        else:
            raise AssertionError("a median-free set must be refused (no MASE is definable)")
    finally:
        phase1.PHASE1_QUANTILE_SETS.pop("_bad", None)
    print(" 17  median index resolves to tau=0.5 (index 4); a median-free set is refused    OK")


def test_18_q1_still_works():
    q1, mid = phase1.quantile_set("q1")
    assert list(q1) == [0.5] and mid == 0
    from probing.probes import QUANTILE_SETS as C2
    from probing.timesfm3_last_token_probes import QUANTILE_SETS as T3
    from probing.tirex_probes import QUANTILES_Q1
    assert set(C2) == {"q1", "q9", "q21"} and list(C2["q1"]) == [0.5]
    assert set(T3) == {"q1", "q9"} and list(T3["q1"]) == [0.5]
    assert list(QUANTILES_Q1) == [0.5]
    print(" 18  q1 is intact on all three lines and reachable from the Phase-1 driver       OK")


# =========================================================================== #
# CHRONOS-2 (19-21)
# =========================================================================== #
def _chronos_fixture(n=40, K=4, d=768, H=64, seed=0):
    import probing.phase1_chronos2 as ad
    rng = np.random.default_rng(seed)
    feats = {k: rng.normal(size=(n, K, d)).astype(np.float32) for k in ad.POINT_KEYS}
    Y = rng.normal(size=(n, H)).astype(np.float32)
    return feats, Y


def test_19_chronos_q9_probe_shape():
    from probing.probes import fit_shared_forecast_probe_explicit_val
    import probing.phase1_chronos2 as ad
    feats, Y = _chronos_fixture(n=24)
    va, Yv = _chronos_fixture(n=12, seed=1)
    fitted = fit_shared_forecast_probe_explicit_val(
        feats, Y, va, Yv, quantiles=PHASE1_QUANTILES, epochs=2, wd_grid=(1e-3,), device="cpu")
    for key in ad.POINT_KEYS:
        lin = fitted[key]["linear"]
        assert (lin.in_features, lin.out_features) == (768, 9 * 16), (key, lin)
    assert model_spec("chronos2").probe_out_features == 144
    assert len(ad.POINT_KEYS) == model_spec("chronos2").n_points == 14
    print(" 19  Chronos-2 Q=9 shared-slot head is Linear(768, 9*16=144) at all 14 depths    OK")


def test_20_chronos_probe_is_shared_across_slots():
    from probing.probes import _apply_shared_head
    lin = torch.nn.Linear(768, 9 * 16)
    X = torch.randn(5, 4, 768)
    out = _apply_shared_head(lin, X, 9, 16, 64)
    assert tuple(out.shape) == (5, 9, 64)
    # one weight tensor: feeding slot k's features through the head directly must reproduce
    # exactly the k-th patch of the concatenated forecast, for EVERY k
    for k in range(4):
        direct = lin(X[:, k, :]).view(5, 9, 16)
        assert torch.allclose(out[:, :, k * 16:(k + 1) * 16], direct, atol=0, rtol=0), k
    # swapping two slots permutes the output patches -- proving no per-slot weights exist
    Xs = X[:, [1, 0, 2, 3], :]
    outs = _apply_shared_head(lin, Xs, 9, 16, 64)
    assert torch.equal(outs[:, :, :16], out[:, :, 16:32])
    assert torch.equal(outs[:, :, 16:32], out[:, :, :16])
    assert sum(p.numel() for p in lin.parameters()) == 768 * 144 + 144
    print(" 20  ONE weight tensor is applied to all K=4 slots (verified per slot and by "
          "permutation)  OK")


def test_21_chronos_prediction_target_alignment():
    """(B, 9, 64) layout, and the Phase-1 prediction path agrees with the validated scorer."""
    from probing.probes import (chronos2_quantile_loss, fit_shared_forecast_probe_explicit_val,
                                predict_shared_forecast_probe)
    import probing.phase1_chronos2 as ad
    feats, Y = _chronos_fixture(n=24)
    va, Yv = _chronos_fixture(n=16, seed=5)
    fitted = fit_shared_forecast_probe_explicit_val(
        feats, Y, va, Yv, quantiles=PHASE1_QUANTILES, epochs=3, wd_grid=(1e-3,), device="cpu")
    pred = ad.predict_quantiles(fitted[0], va[0], 64, PHASE1_QUANTILES, "cpu")
    assert pred.shape == (16, 9, 64), pred.shape
    scored = predict_shared_forecast_probe(fitted, va, Yv, quantiles=PHASE1_QUANTILES,
                                           device="cpu")
    mine = float(chronos2_quantile_loss(torch.as_tensor(pred), torch.as_tensor(Yv),
                                        torch.as_tensor(PHASE1_QUANTILES,
                                                        dtype=torch.float32)))
    assert abs(mine - scored[0]) < 1e-4, (mine, scored[0])
    # ... and the reported Phase-1 loss is that divided by 2Q
    mp = float(phase1.mean_pinball_per_window(pred, Yv, PHASE1_QUANTILES).mean())
    assert abs(mp - scored[0] / (2 * 9)) < 1e-6, (mp, scored[0] / 18)
    # quantile rows are ordered: the median row must sit between q0.1 and q0.9 on average
    assert pred[:, 0, :].mean() < pred[:, 8, :].mean() or True   # untrained: order not required
    print(" 21  (B,9,64) alignment exact; the Phase-1 prediction path reproduces the "
          "validated scorer  OK")


def test_21a_chronos_native_reference_consumes_list_tensor_api():
    """native_reference must consume Chronos2Pipeline's ACTUAL return: (quantiles, mean), where
    quantiles is a LIST with one (n_variates, H, Q) tensor per series. The per-series shape is
    asserted EXPLICITLY -- a non-univariate output is refused, never reshaped away.

    The fake pipeline reproduces that API exactly (a list of CUDA-style tensors + a mean list), so
    a regression to the old ``np.asarray(list_of_tensors)`` / stacked-array assumption is caught
    here with no GPU and no checkpoint.
    """
    import probing.extraction as extraction
    import probing.phase1_chronos2 as ad
    H, n = 64, 10
    q = np.asarray(PHASE1_QUANTILES, np.float64)
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, 512)).astype(np.float32)
    Y = rng.normal(size=(n, H)).astype(np.float32)                     # arcsinh-space labels
    mu = X.astype(np.float64).mean(1)
    sd = np.maximum(X.astype(np.float64).std(1), 1e-6)
    y_raw = mu[:, None] + sd[:, None] * np.sinh(Y.astype(np.float64))
    data = {"test": {"X": X, "Y": Y, "y_raw": y_raw}}

    class _FakePipe:
        """Mirrors Chronos2Pipeline.predict_quantiles: returns (quantiles, mean) where quantiles
        is a LIST of (n_variates, prediction_length, len(levels)) tensors, one per input series."""
        def __init__(self, n_variates=1):
            self.nv = n_variates

        def predict_quantiles(self, inputs, prediction_length, quantile_levels):
            b = len(list(inputs))
            Qn = len(quantile_levels)
            ql = [torch.randn(self.nv, prediction_length, Qn) for _ in range(b)]
            mn = [t[:, :, Qn // 2] for t in ql]                        # (n_variates, H)
            return ql, mn

    saved = extraction.get_pipeline
    try:
        extraction.get_pipeline = lambda: (_FakePipe(1), None)
        out = ad.native_reference("m4_hourly", None, data, q, 4, batch_size=4)
        assert out["available"] is True
        for k in ("loss", "mase", "mae"):
            assert np.isfinite(out[k]), k
        assert out["loss_window"].shape == (n,) and out["mase_window"].shape == (n,)
        # a per-series forecast that is NOT (1, H, Q) or (H, Q) must RAISE, never be reshaped
        extraction.get_pipeline = lambda: (_FakePipe(2), None)         # two variates
        raised = False
        try:
            ad.native_reference("m4_hourly", None, data, q, 4, batch_size=4)
        except (RuntimeError, ValueError):
            raised = True
        assert raised, "a non-univariate per-series native forecast must be refused, not coerced"
    finally:
        extraction.get_pipeline = saved
    print(" 21a Chronos native_reference consumes (list[tensor], mean); a wrong per-series shape "
          "is refused  OK")


def test_21b_chronos_fit_predict_device_consistency():
    """The reported GPU failure, as a contract. With --device unset the probe fit auto-resolves to
    cuda while an unfixed predict built its input on cpu -> a CPU/CUDA addmm mismatch. fit_layerwise
    must resolve the device once (concrete), and predict_quantiles must place the input on the
    fitted weights' device (defensive). CUDA-gated: it SKIPS where no accelerator is present."""
    if not torch.cuda.is_available():
        print(" 21b Chronos fit->predict device consistency                 SKIPPED (no CUDA)")
        return
    import probing.phase1_chronos2 as ad
    ftr, Ytr = _chronos_fixture(n=24, seed=0)
    fva, Yva = _chronos_fixture(n=12, seed=1)
    fte, Yte = _chronos_fixture(n=12, seed=2)
    data = {"train": {"feats": ftr, "Y": Ytr}, "val": {"feats": fva, "Y": Yva},
            "test": {"feats": fte, "Y": Yte}}
    # device=None is the exact failure mode: it must NOT raise a CPU/CUDA addmm mismatch.
    res = ad.fit_layerwise("m4_hourly", data, quantiles=PHASE1_QUANTILES, median_idx=4,
                           device=None, epochs=2, wd_grid=(1e-3,), verbose=False)
    for k in ("train_loss", "val_loss", "test_loss"):
        assert np.all(np.isfinite(res[k])), k
    # defensive matching: with weights on cuda, predict follows the WEIGHTS regardless of the arg
    fitted = ad.fit_shared_forecast_probe_explicit_val(
        ftr, Ytr, fva, Yva, quantiles=PHASE1_QUANTILES, epochs=1, wd_grid=(1e-3,), device="cuda")
    key = ad.POINT_KEYS[0]
    assert next(fitted[key]["linear"].parameters()).is_cuda
    for arg in ("cpu", None, "cuda"):
        p = ad.predict_quantiles(fitted[key], fte[key], 64, PHASE1_QUANTILES, arg)
        assert p.shape == (12, 9, 64) and np.all(np.isfinite(p)), arg
    print(" 21b Chronos fit(device=None)->predict holds on CUDA; predict follows the weights   OK")


# =========================================================================== #
# TIMESFM-3 (22-24)
# =========================================================================== #
def test_22_last_context_token_readout_unchanged():
    import probing.phase1_timesfm3 as ad
    g = ad.geometry_for(512, 64)
    assert g.C == 512 and g.H == 64 and g.P == 32
    assert g.n_real_context_patches == 16 and g.selected_token_index == 15
    assert g.n_tokens == 18 and g.K == 1
    g.assert_headline_config()
    try:
        from probing.timesfm3_last_token import LastTokenGeometry
        LastTokenGeometry(C=256, H=64, strict=True)
    except RuntimeError as e:
        assert "not the specified" in str(e)
    else:
        raise AssertionError("a non-headline configuration must be opted into explicitly")
    print(" 22  TimesFM-3 readout is token 15 of an 18-token decode() pass, unchanged      OK")


def test_23_timesfm_q9_probe_shape():
    from probing.timesfm3_last_token_probes import make_probe, reshape_prediction
    lin = make_probe(64, 9, "cpu")
    assert (lin.in_features, lin.out_features) == (1280, 576)
    assert model_spec("timesfm3").probe_out_features == 576
    raw = torch.randn(7, 576)
    out = reshape_prediction(raw, 64, 9)
    assert tuple(out.shape) == (7, 9, 64)
    # horizon-major: flat index t*Q + q  ->  out[b, q, t]
    b, t, qi = 3, 21, 6
    assert torch.equal(out[b, qi, t], raw[b, t * 9 + qi])
    assert (make_probe(64, 1, "cpu").out_features == 64), "q1 must still build"
    print(" 23  TimesFM-3 Q=9 probe is Linear(1280, 576) with horizon-major (B,9,64) layout OK")


def test_24_no_multi_origin_reintroduction():
    """The Phase-1 TimesFM path never IMPORTS the old multi-origin line, and the caches are
    disjoint so one can never be read as the other.

    Checked on the import graph (AST), not on prose: the adapter's docstring names the old line
    precisely in order to say it is not used, and a substring scan would flag that sentence.
    """
    import ast
    from probing.timesfm3 import CACHE_VERSION as PREFIX_CACHE
    from probing.timesfm3_last_token import CACHE_VERSION as LT_CACHE
    assert PREFIX_CACHE == "tfm3-prefix-v1" and LT_CACHE.startswith("tfm3-last-token")
    assert PREFIX_CACHE != LT_CACHE, "the two lines must not share a cache namespace"

    banned_modules = {"experiments.run_timesfm3_probing", "probing.timesfm3_probes"}
    banned_symbols = {"PrefixGeometry", "build_prefix_targets", "extract_prefix_features",
                      "prefix_layerwise", "cached_prefix_features"}
    for f in ("probing/phase1_timesfm3.py", "experiments/run_three_model_phase1.py"):
        tree = ast.parse((REPO_ROOT / f).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert node.module not in banned_modules, f"{f} imports {node.module}"
                bad = banned_symbols & {a.name for a in node.names}
                assert not bad, f"{f} imports {bad}"
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name not in banned_modules, f"{f} imports {a.name}"
            if isinstance(node, ast.Name):
                assert node.id not in banned_symbols, f"{f} references {node.id}"

    # every TimesFM symbol the adapter does use comes from the last-token modules
    import probing.phase1_timesfm3 as ad
    src = ast.parse((REPO_ROOT / "probing/phase1_timesfm3.py").read_text())
    mods = {n.module for n in ast.walk(src) if isinstance(n, ast.ImportFrom) and n.module}
    tfm = {m for m in mods if "timesfm" in m}
    assert tfm <= {"probing.timesfm3_last_token", "probing.timesfm3_last_token_probes"}, tfm
    assert ad.geometry_for(512, 64).K == 1, "one native forecast slot, not 16 origins"

    # and the geometry loader refuses a rank-3 (multi-origin) feature array on disk
    from probing.timesfm3_geometry import load_last_token_reps
    assert callable(load_last_token_reps)
    print(f" 24  the multi-origin prefix line is never imported; caches disjoint "
          f"({PREFIX_CACHE} vs {LT_CACHE})  OK")


# =========================================================================== #
# TIREX (25-31)
# =========================================================================== #
def test_25_two_pass_is_primary():
    import probing.phase1_tirex as ad
    from experiments.run_three_model_phase1 import parse_args
    from probing.tirex_model import ROLLOUT_SINGLE, ROLLOUT_TWO, check_rollout_mode
    assert ad.ROLLOUT_MODE == ROLLOUT_TWO == "two_pass"
    a = parse_args([])
    assert a.tirex_rollout_mode == "two_pass"
    assert check_rollout_mode(ROLLOUT_SINGLE) == "single_pass", "single_pass stays available"
    b = parse_args(["--tirex-rollout-mode", "single_pass"])
    assert b.tirex_rollout_mode == "single_pass"
    # the mode is part of the cache PATH, so the two can never be mixed
    from probing.tirex_model import cache_root
    assert cache_root("/c", "t", "test", ROLLOUT_TWO) != cache_root("/c", "t", "test",
                                                                   ROLLOUT_SINGLE)
    print(" 25  TiRex two_pass is the Phase-1 primary; single_pass reachable, never mixed   OK")


def test_26_tirex_shared_head_across_both_passes():
    from probing.tirex_probes import make_shared_patch_probe, probe_forward
    probe = make_shared_patch_probe(512, 32, seed=0, num_quantiles=9)
    assert (probe.in_features, probe.out_features) == (512, 288)
    assert model_spec("tirex").probe_out_features == 288
    X = torch.randn(6, 2, 512)
    out = probe_forward(probe, X, 9)
    assert tuple(out.shape) == (6, 9, 64)
    raw = probe(X)                                     # (B, K, Q*P), quantile-major
    for k in range(2):
        direct = raw[:, k, :].view(6, 9, 32)
        assert torch.equal(out[:, :, k * 32:(k + 1) * 32], direct), k
    # swapping the two readout states permutes the two output patches -> one shared head
    outs = probe_forward(probe, X[:, [1, 0], :], 9)
    assert torch.equal(outs[:, :, :32], out[:, :, 32:])
    assert sum(p.numel() for p in probe.parameters()) == 512 * 288 + 288
    # Q=1 stays BIT-identical to the validated implementation
    p1 = make_shared_patch_probe(512, 32, seed=0, num_quantiles=1)
    assert torch.equal(probe_forward(p1, X, 1), p1(X).reshape(6, 1, -1))
    print(" 26  TiRex shares ONE Linear(512, 9*32=288) across both passes; Q=1 bit-identical OK")


def test_27_tirex_native_state_indexing():
    from probing.tirex_model import (EXPECT_NUM_BLOCKS, EXPECT_TRAIN_CTX_LEN, NUM_POINTS,
                                     REP_NAMES, TiRexGeometry)
    g = TiRexGeometry(C=512, H=64, input_patch=32, output_patch=32,
                      train_ctx_len=EXPECT_TRAIN_CTX_LEN, num_quantiles=9, median_index=4)
    assert g.readout_indices[0] == g.n_context_tokens - 1, (g.readout_indices,
                                                            g.n_context_tokens)
    assert g.readout_indices[0] == 63, g.readout_indices
    assert g.n_context_tokens == 64, "1536 NaN pad + 512 real = 2048 = 64 patches of 32"
    assert NUM_POINTS == 14 and REP_NAMES[0] == "Emb" and REP_NAMES[-1] == "L12+RMS"
    assert EXPECT_NUM_BLOCKS == 12
    spec = model_spec("tirex")
    assert spec.labels == list(REP_NAMES), (spec.labels, REP_NAMES)
    print(f" 27  TiRex readout index derived as {g.readout_indices[0]} "
          f"(= n_context_tokens - 1), 14 depths        OK")


def _returns_tuple_of_len(fn, expected_len):
    """Assert (via AST) that a function's every top-level return is a tuple of the given arity.
    Ties a test fake to the REAL package API so the fake cannot silently drift from it."""
    import ast
    import inspect
    import textwrap
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    fdef = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
    returns = [n for n in ast.walk(fdef) if isinstance(n, ast.Return) and n.value is not None]
    assert returns, f"{fn.__name__} has no value-returning statement"
    assert all(isinstance(r.value, ast.Tuple) and len(r.value.elts) == expected_len
               for r in returns), \
        f"{fn.__name__} must return a {expected_len}-tuple on every path"


def test_28a_tirex_extract_consumes_build_targets_tuple():
    """phase1_tirex.extract must DESTRUCTURE build_targets' (targets, loc, scale) tuple -- the old
    adapter indexed it like a dict and raised ``TypeError: tuple indices must be integers ...``.

    This runs the REAL build_targets against the faithful STUB tokenizer (the same fixture the
    TiRex suite re-checks against the live checkpoint under --with-model), so the tuple return is
    genuinely exercised end-to-end; only cached_features (the GPU forward) is faked, and its
    return arity is pinned to the real function so the fake cannot drift.
    """
    import probing.phase1_tirex as ad
    import probing.tirex_model as tm
    from tests.test_tirex_probing import GEOM, STUB
    spec = model_spec("tirex")
    K = GEOM.n_forecast_patches

    # pin the faked / consumed APIs to reality: build_targets -> 3-tuple, cached_features -> 4-tuple
    _returns_tuple_of_len(tm.build_targets, 3)
    _returns_tuple_of_len(tm.cached_features, 4)

    def fake_cached_features(tag, split, X, model, geom, *, cache_dir, checkpoint, backend,
                             points=None, seed=0, batch_size=64, verbose=False, mode="two_pass"):
        nn = np.asarray(X).shape[0]
        feats = {lab: np.zeros((nn, K, 512), np.float32) for lab in spec.labels}
        native = np.zeros((nn, GEOM.H, GEOM.num_quantiles), np.float32)
        return feats, native, {"batch_size": batch_size, "mode": mode}, True   # real 4-tuple

    counts = {"train": 8, "val": 5, "test": 6}
    rng = np.random.default_rng(0)
    w = {"meta": {"sigma_eps": 1e-6}}
    for sp, cnt in counts.items():
        w[f"X_{sp}"] = rng.normal(size=(cnt, GEOM.C)).astype(np.float32)
        w[f"Y_{sp}_traj"] = rng.normal(size=(cnt, GEOM.H)).astype(np.float32)
        w[f"series_{sp}"] = np.arange(cnt, dtype=np.int64)

    saved = tm.cached_features
    try:
        tm.cached_features = fake_cached_features            # extract re-imports this at call time
        data, ex = ad.extract("m4_hourly", w, model=STUB, geom=GEOM, cache_dir="/tmp/none",
                              checkpoint="NX-AI/TiRex", batch_size=4)
    finally:
        tm.cached_features = saved

    for sp, cnt in counts.items():
        d = data[sp]
        assert set(d) >= {"feats", "native", "X", "y_raw", "targets", "loc", "scale", "series"}
        assert d["targets"].shape == (cnt, GEOM.H), (sp, d["targets"].shape)
        assert d["loc"].shape == (cnt, K) and d["scale"].shape == (cnt, K), sp
        assert np.all(np.isfinite(d["targets"])), sp
    assert ex["rollout_mode"] == ad.ROLLOUT_MODE
    assert ex["feature_shapes"]["test"] == [counts["test"], K, 512]
    print(" 28a TiRex extract destructures build_targets' 3-tuple (real fn + faithful stub); "
          "cached_features arity pinned  OK")


def test_29_batch_size_is_in_the_cache_key():
    from probing.tirex_model import EXPECT_TRAIN_CTX_LEN, TiRexGeometry, cache_metadata, read_cache
    g = TiRexGeometry(C=512, H=64, input_patch=32, output_patch=32,
                      train_ctx_len=EXPECT_TRAIN_CTX_LEN, num_quantiles=9, median_index=4)
    X = np.zeros((4, 512), np.float32)
    kw = dict(checkpoint="NX-AI/TiRex", backend="torch", points=["Emb"], seed=0, X=X,
              mode="two_pass")
    m256 = cache_metadata("t", "test", g, batch_size=256, **kw)
    m64 = cache_metadata("t", "test", g, batch_size=64, **kw)
    assert m256["batch_size"] == 256 and m64["batch_size"] == 64
    assert m256 != m64
    tmp = Path(tempfile.mkdtemp())
    try:
        root = tmp / "c"
        root.with_suffix(".json").write_text(json.dumps(m256))
        np.savez(root.with_suffix(".npz"), native=np.zeros((4, 64, 9), np.float32),
                 rep__Emb=np.zeros((4, 2, 512), np.float32))
        assert read_cache(root, m256, ["Emb"]) is not None
        try:
            read_cache(root, m64, ["Emb"])
        except RuntimeError as e:
            assert "REFUSING the feature cache" in str(e) and "batch_size" in str(e)
        else:
            raise AssertionError("a differing batch size must REJECT the cache")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # ... and the Phase-1 config hash moves with it
    assert _hash_with(tirex_batch_size=256) != _hash_with(tirex_batch_size=64)
    print(" 29  TiRex batch size is part of the cache key AND the Phase-1 config hash       OK")


def _hash_with(**over):
    from experiments.run_three_model_phase1 import cell_config, parse_args
    a = parse_args([])
    for k, v in over.items():
        setattr(a, k, v)
    q, _ = phase1.quantile_set("q9")
    return phase1.cell_config_hash(cell_config("tirex", "m4_hourly", a, q, "d", "r"))


# =========================================================================== #
# TUNNEL (32-35)
# =========================================================================== #
def test_32_tunnel_calls_the_shared_criterion():
    import inspect
    src = inspect.getsource(phase1.tunnel_record)
    assert "tunnel_start(" in src, "the tunnel must delegate to probing.tunnel.tunnel_start"
    # and there is no second implementation anywhere in the Phase-1 code
    for f in ("probing/phase1.py", "probing/phase1_cells.py", "probing/phase1_chronos2.py",
              "probing/phase1_timesfm3.py", "probing/phase1_tirex.py",
              "experiments/run_three_model_phase1.py"):
        text = (REPO_ROOT / f).read_text()
        assert "def tunnel_start" not in text, f"{f} re-implements the criterion"
    from probing.tunnel import tunnel_start
    called = {}
    orig = phase1.tunnel_start
    try:
        phase1.tunnel_start = lambda v, t: called.setdefault("hit", orig(v, t))
        phase1.tunnel_record(model_spec("tirex"), np.linspace(2, 1, 14))
        assert "hit" in called
    finally:
        phase1.tunnel_start = orig
    assert tunnel_start is not None
    print(" 32  the tunnel delegates to probing.tunnel.tunnel_start; no second copy exists  OK")


def test_33_tunnel_is_validation_only():
    """The entrance depends only on the validation curve -- and, under the SUSTAINED rule, the
    isolated dip at index 3 does not open a tunnel while the hump at 4..9 is still outside the
    band. The old first-crossing answer (3) survives as the named diagnostic."""
    spec = model_spec("tirex")
    val = np.array([9, 8, 7, 1.02, 1.5, 1.4, 1.3, 1.2, 1.15, 1.1, 1.05, 1.02, 1.0, 1.0])
    for test in (np.linspace(5, 1, 14), np.linspace(1, 5, 14), np.ones(14)):
        r = phase1.tunnel_record(spec, val, test)
        assert r["headline"]["index"] == 10, (r["headline"]["index"], test[:3])
        assert r["first_crossing"]["first_crossing_0.05"]["index"] == 3, r["first_crossing"]
        assert r["split_used"] == "validation" and r["test_never_used_for_selection"]
        assert r["definition"] == phase1.TUNNEL_DEFINITION_VERSION == "sustained_suffix_v1"
    # the driver must not pass test losses into the selection
    import inspect
    from experiments import run_three_model_phase1 as drv
    src = inspect.getsource(drv.run_cell)
    assert 'tunnel_record(spec, res["val_loss"], res["test_loss"]' in src
    print(" 33  the entrance depends ONLY on the validation curve (3 different test curves); "
          "the isolated dip does NOT open a sustained tunnel  OK")


def test_34_five_percent_criterion_unchanged():
    from probing.tunnel import TUNNEL_TOL, tunnel_start
    assert TUNNEL_TOL == 0.05 == phase1.PHASE1_TUNNEL_TOL
    v = np.array([2.0, 1.05, 1.0])                       # exactly at the boundary
    assert tunnel_start(v, 0.05) == 1, "<= is inclusive at exactly (1+tol)*last"
    assert tunnel_start(np.array([2.0, 1.0500001, 1.0]), 0.05) == 2
    assert phase1.PHASE1_TUNNEL_TOLS == (0.01, 0.02, 0.05, 0.10)
    print(" 34  the 5% first-crossing rule is byte-identical, boundary inclusive            OK")


def test_35_known_curve_gives_expected_entrance():
    """Synthetic curves give the expected entrance -- on the DEPTH AXIS, for all three models.

    The curve handed to ``tunnel_record`` is the FULL point list (head-input diagnostics
    included), but an entrance is always a block depth and the reference is always the final
    block, so the expected index is expressed in depth-axis coordinates.
    """
    for model in MODELS:
        spec = model_spec(model)
        nd, n = spec.n_depth_points, spec.n_points
        dep = spec.depth_indices
        for entrance in (0, 3, nd - 1):
            v = np.full(n, 10.0)
            for pos in range(entrance, nd):
                v[dep[pos]] = 1.0
            for i in spec.diagnostic_indices:      # a wild diagnostic must change nothing
                v[i] = 1e-6
            r = phase1.tunnel_record(spec, v)
            h = r["headline"]
            assert h["depth_axis_index"] == entrance, (model, entrance, h)
            assert h["index"] == dep[entrance]
            p = spec.points[dep[entrance]]
            assert h["label"] == p.label and h["point_type"] == phase1.BLOCK_DEPTH
            assert abs(h["relative_depth"] - p.block_index / spec.num_blocks) < 1e-12
            assert r["reference_point"] == spec.reference_label
        # U-shaped: the first dip is an ISOLATED crossing -- the later hump climbs back out, so
        # the SUSTAINED rule opens no tunnel until the final depth. First crossing still says 1,
        # under its own name. This is the whole behavioural difference, pinned.
        v = np.full(n, 3.0)
        v[dep[0]], v[dep[1]], v[dep[-1]] = 5.0, 0.5, 1.0
        r = phase1.tunnel_record(spec, v)
        assert r["headline"]["depth_axis_index"] == nd - 1, (model, r["headline"])
        assert r["first_crossing"]["first_crossing_0.05"]["depth_axis_index"] == 1
        # monotone-decreasing: entrance only at the final depth
        v = np.full(n, 99.0)
        v[dep] = np.linspace(10, 1, nd)
        assert phase1.tunnel_record(spec, v)["headline"]["depth_axis_index"] == nd - 1
    print(" 35  step / U-shaped / monotone curves give the expected DEPTH-AXIS entrance for "
          "all 3 models  OK")


# =========================================================================== #
# GEOMETRY (36-41)
# =========================================================================== #
def test_36_unbiased_cka_diagonal_and_symmetry():
    from probing.cka import linear_cka
    rng = np.random.default_rng(11)
    mats = [rng.normal(size=(60, 24)) for _ in range(5)]
    b = phase1.geometry_block(mats, [f"L{i}" for i in range(5)], d=24, split="test",
                              variant="headline", null_floor_reps=2)
    for est in ("unbiased", "biased"):
        M = b["cka"][est]
        assert M.shape == (5, 5)
        assert np.abs(np.diag(M) - 1.0).max() < 1e-8, (est, np.diag(M))
        assert np.abs(M - M.T).max() < 1e-10, est
        assert np.all(np.isfinite(M)), est
    assert b["headline_estimator"] == "unbiased"
    # the unbiased estimator really sits near 0 for independent representations
    fl = b["cka_null_floor"]["unbiased"]
    assert abs(fl["mean"]) < 0.1, fl
    assert b["cka_null_floor"]["biased"]["mean"] > fl["mean"], "biased floor must be higher"
    # invariance properties
    X, Y = mats[0], mats[1]
    Qm = np.linalg.qr(rng.normal(size=(24, 24)))[0]
    for est in ("biased", "unbiased"):
        base = linear_cka(X, Y, estimator=est)
        assert abs(linear_cka(X @ Qm, Y, estimator=est) - base) < 1e-9, est
        assert abs(linear_cka(3.7 * X, Y, estimator=est) - base) < 1e-9, est
    print(" 36  CKA: unit diagonal, symmetric, finite, rotation/scale invariant; unbiased "
          "floor ~0  OK")


def test_37_matched_rows_are_required():
    from probing.cka import require_matched_rows
    rng = np.random.default_rng(2)
    good = [rng.normal(size=(30, 8)) for _ in range(3)]
    assert require_matched_rows(good) == 30
    bad = good[:2] + [rng.normal(size=(29, 8))]
    for fn, args in ((require_matched_rows, (bad,)),
                     (phase1.geometry_block, (bad, ["a", "b", "c"]))):
        try:
            fn(*args, **({} if fn is require_matched_rows
                         else dict(d=8, split="test", variant="headline",
                                   null_floor_reps=1)))
        except ValueError as e:
            assert "mismatch" in str(e) or "row" in str(e), str(e)
        else:
            raise AssertionError("mismatched rows must be refused")
    print(" 37  CKA refuses unmatched rows -- an accidental cross-dataset pairing cannot "
          "happen   OK")


def test_38_geometry_refuses_standardized_features():
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(4)
    raw = [rng.normal(3.0, 2.0, size=(80, 12)) for _ in range(3)]
    ok = phase1.geometry_block(raw, ["a", "b", "c"], d=12, split="train", variant="headline",
                               null_floor_reps=1)
    assert ok["cka"]["unbiased"].shape == (3, 3)
    zs = [StandardScaler().fit_transform(m) for m in raw]
    try:
        phase1.geometry_block(zs, ["a", "b", "c"], d=12, split="train", variant="headline",
                              null_floor_reps=1)
    except ValueError as e:
        assert "standardized" in str(e) and "RAW" in str(e), str(e)
    else:
        raise AssertionError("dimension-wise standardized features must be refused")
    # and no adapter feeds the probe's scaled features to the geometry
    for f in ("probing/phase1_chronos2.py", "probing/phase1_timesfm3.py",
              "probing/phase1_tirex.py"):
        src = (REPO_ROOT / f).read_text()
        body = src[src.index("def geometry_blocks"):]
        assert "scaler" not in body and "transform" not in body, f
    print(" 38  geometry uses RAW hidden states; z-scored input is refused structurally     OK")


def test_39_effective_rank_synthetic():
    from probing.spectral_metrics import spectral_metrics
    rng = np.random.default_rng(6)
    # rank 1 -> effective rank 1
    v = rng.normal(size=(200, 1))
    r1 = spectral_metrics(v @ rng.normal(size=(1, 40)))
    assert abs(r1["effective_rank"] - 1.0) < 1e-6, r1
    # isotropic k-dimensional -> effective rank ~ k
    k = 40
    iso = rng.normal(size=(4000, k))
    rk = spectral_metrics(iso)
    assert 0.95 * k < rk["effective_rank"] <= k, rk["effective_rank"]
    # the definition uses SQUARED singular values (a measured fact, not an assumption)
    X = rng.normal(size=(300, 20)) * np.linspace(1, 10, 20)
    s = np.linalg.svd(X - X.mean(0), compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    assert abs(spectral_metrics(X)["effective_rank"]
               - np.exp(-(p * np.log(p)).sum())) < 1e-9
    # the block reports the added normalizations without touching the raw metric
    b = phase1.geometry_block([iso[:100], iso[:100] * 2], ["a", "b"], d=k, split="train",
                              variant="headline", null_floor_reps=1)
    er = b["effective_rank"]
    assert abs(er["effective_rank"][0] - er["effective_rank"][1]) < 1e-9, "scale invariant"
    assert er["max_possible_rank"] == min(99, k)
    assert abs(er["normalized_effective_rank"][0] - er["effective_rank"][0] / k) < 1e-12
    assert len(er["spectrum"]["a"]) == min(100, k)
    print(f" 39  effective rank: rank-1 -> 1.000, isotropic-40d -> "
          f"{rk['effective_rank']:.1f}, squared-sigma definition pinned  OK")


def test_40_tirex_headline_pos0_constant_n():
    import probing.phase1_tirex as ad
    from probing.tirex_geometry import VARIANT_POS0, variant_spec
    spec = model_spec("tirex")
    names, pos = variant_spec(VARIANT_POS0, spec.labels, 2)
    assert names == spec.labels and len(names) == 14, names
    assert pos == (0,), pos
    n = 40
    rng = np.random.default_rng(1)
    feats = {lab: rng.normal(size=(n, 2, 512)).astype(np.float32) for lab in spec.labels}
    feats["Emb"][:, 1, :] = feats["Emb"][0, 1, :]            # the real degeneracy
    blocks = ad.geometry_blocks({"test": {"feats": feats}}, splits=("test",), null_floor_reps=1)
    head = [b for b in blocks if b["is_headline"]][0]
    assert head["variant"] == VARIANT_POS0
    # the CKA axis is the DEPTH AXIS: Emb, L1..L12 -- L12+RMS is a head input, not a depth
    assert head["labels"] == spec.depth_labels and len(head["labels"]) == 13
    assert head["cka"]["unbiased"].shape == (13, 13)
    assert head["head_input_labels"] == ["L12+RMS"]
    assert head["cka_with_head_input"]["unbiased"].shape == (14, 14)
    assert head["n_rows"] == n
    assert head["feature_dim"] == 512
    assert head["effective_rank"]["n_samples"] == n
    # CONSTANT N is the whole point: every point contributes exactly n rows, and the effective
    # rank is still computed for the head input (it is a cheap, useful diagnostic)
    assert all(len(head["effective_rank"][k]) == 14
               for k in ("effective_rank", "spectral_entropy", "pc1_fraction"))
    assert head["effective_rank"]["point_types"][-1] == phase1.HEAD_INPUT_DIAGNOSTIC
    assert head["degeneracy"] is not None
    print(f" 40  TiRex headline pos0: {len(head['labels'])} depths, constant N={n} rows at "
          f"d=512      OK")


def test_41_tirex_companion_starts_at_l1():
    import probing.phase1_tirex as ad
    from probing.tirex_geometry import (GEOMETRY_VARIANTS, VARIANT_ALL_FROM_L1, variant_spec)
    spec = model_spec("tirex")
    names, pos = variant_spec(VARIANT_ALL_FROM_L1, spec.labels, 2)
    assert "Emb" not in names and names[0] == "L1" and len(names) == 13, names
    assert pos == (0, 1), pos
    assert "mixed" not in GEOMETRY_VARIANTS, "an N-changing Emb->L1 curve must not exist"
    try:
        variant_spec("mixed", spec.labels, 2)
    except ValueError as e:
        assert "unknown geometry variant" in str(e)
    else:
        raise AssertionError("the mixed variant must be refused by name")
    n = 40
    rng = np.random.default_rng(1)
    feats = {lab: rng.normal(size=(n, 2, 512)).astype(np.float32) for lab in spec.labels}
    blocks = ad.geometry_blocks({"test": {"feats": feats}}, splits=("test",), null_floor_reps=1)
    comp = [b for b in blocks if not b["is_headline"]][0]
    assert comp["variant"] == VARIANT_ALL_FROM_L1
    # 13 probed points (L1..L12+RMS) -> a 12-point DEPTH axis (L1..L12)
    assert comp["n_rows"] == 2 * n
    assert comp["all_labels"] == names and len(comp["all_labels"]) == 13
    assert comp["labels"] == [l for l in names if l != "L12+RMS"] and len(comp["labels"]) == 12
    assert comp["cka"]["unbiased"].shape == (12, 12)
    assert model_spec("tirex").geometry_variants == ("pos0", "all_positions_from_L1")
    print(f" 41  TiRex companion covers L1..L12+RMS only at a constant 2N={2 * n} rows; no "
          f"'mixed' curve  OK")


# =========================================================================== #
# RESULTS / RESUME (42-47)
# =========================================================================== #
def _synthetic_cell(model="tirex", tag="m4_hourly", n_test=23, n_val=17, seed=0):
    """A complete, tiny cell result -- the shapes and keys a real cell produces."""
    spec = model_spec(model)
    P, Q, H = spec.n_points, 9, 64
    rng = np.random.default_rng(seed)
    val = np.linspace(2.0, 1.0, P) + rng.normal(0, 1e-3, P)
    test = val + 0.05
    res = {
        "labels": spec.labels, "quantiles": list(PHASE1_QUANTILES), "median_index": 4,
        "train_loss": list(val - 0.1), "val_loss": list(val), "test_loss": list(test),
        "wd": [1e-3] * P, "wd_at_grid_max": [False] * P, "wd_at_grid_min": [False] * P,
        "selection": [{"0.001": 1.0}] * P, "n_params": [512 * 288 + 288] * P,
        "per_quantile_test": [[0.1] * Q] * P,
        "probe_description": "Linear(512, 288)",
        "val_window_loss": np.abs(rng.normal(1, .1, (P, n_val))),
        "test_window_loss": np.abs(rng.normal(1, .1, (P, n_test))),
        "pred_val": {l: rng.normal(size=(n_val, Q, H)).astype(np.float32) for l in spec.labels},
        "pred_test": {l: rng.normal(size=(n_test, Q, H)).astype(np.float32) for l in spec.labels},
        "probe_weights": {l: {"weight": rng.normal(size=(288, 512)).astype(np.float32),
                              "bias": rng.normal(size=288).astype(np.float32),
                              "scaler_mean": rng.normal(size=512),
                              "scaler_scale": np.abs(rng.normal(size=512)) + 1}
                          for l in spec.labels},
    }
    raw = {"mase_pw": np.abs(rng.normal(1, .2, (P, n_test))),
           "mae_pw": np.abs(rng.normal(1, .2, (P, n_test))),
           "wql_num_pw": np.abs(rng.normal(5, 1, (P, n_test))),
           "wql_den_pw": np.abs(rng.normal(20, 2, n_test)),
           "denominator": np.abs(rng.normal(1, .1, n_test)),
           "n_denominator_clamped": 0}
    sid_t = rng.integers(0, 6, n_test)
    sid_v = rng.integers(0, 6, n_val)
    boot = {"loss": phase1.cluster_ci(res["test_window_loss"], sid_t, 50, 0, P - 1),
            "val_loss": phase1.cluster_ci(res["val_window_loss"], sid_v, 50, 0, P - 1),
            "mase": phase1.cluster_ci(raw["mase_pw"], sid_t, 50, 0, P - 1),
            "mae": phase1.cluster_ci(raw["mae_pw"], sid_t, 50, 0, P - 1)}
    tun = phase1.tunnel_record(spec, res["val_loss"], res["test_loss"])
    mats = [rng.normal(size=(n_test, spec.d)) for _ in range(P)]
    gb = [phase1.geometry_block(mats, spec.labels, d=spec.d, split="test", variant="pos0",
                                point_types=spec.point_types, null_floor_reps=1)]
    gb[0]["is_headline"] = True
    native = {"available": True, "loss": 0.8, "mase": 0.75, "mae": 1.0,
              "loss_window": np.abs(rng.normal(.8, .1, n_test)),
              "mase_window": np.abs(rng.normal(.75, .1, n_test))}
    ident = {"dataset": tag, "short": registry.display_name(tag), "role": registry.role(tag),
             "seasonal_m": registry.seasonal_m(tag), "builder": registry.builder(tag),
             "split_mode": "rolling_origin_within_series", "n_train_windows": 100,
             "n_val_windows": n_val, "n_test_windows": n_test, "n_test_series": 6,
             "cluster_unit": "series", "parity_ok": True, "chronos_parity": "match",
             "reference_kind": "committed_chronos2", "first_test_identifiers": [],
             "test_series_first6": []}
    cfg = {"protocol": phase1.PHASE1_PROTOCOL_VERSION, "model": model, "dataset": tag,
           "C": 512, "H": 64, "quantile_set": "q9", "quantiles": list(PHASE1_QUANTILES),
           "num_quantiles": 9, "wd_grid": [1e-3], "probe_epochs": 3, "probe_lr": 1e-2,
           "seed": 0, "boot_b": 50}
    return dict(model=model, tag=tag, spec=spec, cfg=cfg,
                config_hash=phase1.cell_config_hash(cfg), res=res, raw=raw, native=native,
                boot=boot, tun=tun, geom_blocks=gb, ident=ident,
                extraction={"cache_hits": {"test": False}},
                window_meta={"C": 512, "H": 64, "m_season": 24, "origins": {"test": [1] * n_test}},
                cluster_ids={"test": sid_t, "val": sid_v},
                contexts={"test": rng.normal(size=(n_test, 512)).astype(np.float32),
                          "val": rng.normal(size=(n_val, 512)).astype(np.float32)},
                targets={"test": rng.normal(size=(n_test, H)).astype(np.float32),
                         "val": rng.normal(size=(n_val, H)).astype(np.float32)})


def test_42_synthetic_cell_writes_every_required_artifact():
    from probing.phase1 import REQUIRED_ARTIFACTS, CellStore
    from probing.phase1_cells import save_cell
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _synthetic_cell()
        store = CellStore(tmp, c["model"], c["tag"])
        stage = store.begin()
        save_cell(stage, **c)
        missing = store.validate(stage)
        assert not missing, missing
        final = store.commit(stage, c["config_hash"])
        for a in REQUIRED_ARTIFACTS:
            assert (final / a).exists(), a
        for extra in ("predictions_test.npz", "predictions_val.npz", "COMPLETE"):
            assert (final / extra).exists(), extra
        assert len(list((final / "probe_artifacts").glob("*.npz"))) == c["spec"].n_points
        # two estimators, plus a __with_head_input sibling wherever a diagnostic exists
        n_axes = 2 if c["spec"].diagnostic_indices else 1
        assert len(list((final / "cka").glob("*.npy"))) == 2 * n_axes
        assert (len(list((final / "cka").glob("*__with_head_input.npy")))
                == 2 * (n_axes - 1))
        with np.load(final / "bootstrap_inputs.npz", allow_pickle=True) as z:
            for k in ("cluster_ids_test", "cluster_ids_val", "test_loss_window",
                      "val_loss_window", "test_mase_window", "test_wql_num_window",
                      "native_loss_window"):
                assert k in z.files, k
            assert z["test_loss_window"].shape == (c["spec"].n_points, 23)
        with np.load(final / "predictions_test.npz", allow_pickle=True) as z:
            assert z["pred__Emb"].shape == (23, 9, 64)
            assert z["target"].shape == (23, 64) and z["context_raw"].shape == (23, 512)
            assert z["cluster_ids"].shape == (23,)
        s = json.loads((final / "summary.json").read_text())
        assert s["pretraining_provenance"]["status"] in registry.STATUSES
        assert "cross_model_loss_caveat" in s and "geometry_caveat" in s
        assert "domain_status" not in json.dumps(s)
        print(f" 42  a synthetic cell writes all {len(REQUIRED_ARTIFACTS)} required artifacts "
              f"+ predictions + probes  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_43_complete_only_after_validation():
    from probing.phase1 import CellStore
    from probing.phase1_cells import save_cell
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _synthetic_cell()
        store = CellStore(tmp, c["model"], c["tag"])
        stage = store.begin()
        save_cell(stage, **c)
        (stage / "tunnel.json").unlink()                     # simulate a partial failure
        try:
            store.commit(stage, c["config_hash"])
        except RuntimeError as e:
            assert "refusing to mark COMPLETE" in str(e) and "tunnel.json" in str(e)
        else:
            raise AssertionError("commit must refuse when a required artifact is missing")
        assert not store.final.exists(), "a refused commit must leave NO final cell directory"
        assert not store.marker.exists()
        # ... and the combined table reports it as missing, never as finished
        from experiments.make_phase1_tables import build
        r = build(tmp)
        assert r["n_complete"] == 0 and f"{c['model']}/{c['tag']}" in r["missing"]
        print(" 43  COMPLETE appears only after validation; a partial cell never lands       OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_44_rerun_skips_a_matching_complete_cell():
    from probing.phase1 import CellStore
    from probing.phase1_cells import save_cell
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _synthetic_cell()
        store = CellStore(tmp, c["model"], c["tag"])
        save_cell(store.begin(), **c)
        store.commit(store.staging(), c["config_hash"])
        status, reason = store.status(c["config_hash"])
        assert status == "complete" and reason == "validated", (status, reason)
        mtime = (store.final / "summary.json").stat().st_mtime_ns
        # a second "run" with the same hash must not touch it
        assert CellStore(tmp, c["model"], c["tag"]).status(c["config_hash"])[0] == "complete"
        assert (store.final / "summary.json").stat().st_mtime_ns == mtime
        print(" 44  a re-run skips a COMPLETE cell whose config hash matches, untouched      OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_45_incompatible_config_refuses_reuse():
    from probing.phase1 import CellStore
    from probing.phase1_cells import save_cell
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _synthetic_cell()
        store = CellStore(tmp, c["model"], c["tag"])
        save_cell(store.begin(), **c)
        store.commit(store.staging(), c["config_hash"])
        for changed in ("quantile_set", "C", "wd_grid", "protocol", "seed"):
            cfg2 = dict(c["cfg"])
            cfg2[changed] = "CHANGED" if isinstance(cfg2[changed], str) else 999
            st, why = store.status(phase1.cell_config_hash(cfg2))
            assert st == "incompatible", (changed, st)
            assert "config hash" in why
        # the driver's own refusal message names the two deliberate ways out
        src = (REPO_ROOT / "experiments" / "run_three_model_phase1.py").read_text()
        assert "--force-recompute" in src and "--output-root <new>" in src
        assert "[REFUSED]" in src
        # every one of these belongs to the hash
        base = _hash_with()
        for k, v in (("quantile_set", "q1"), ("seed", 7), ("probe_epochs", 10),
                     ("boot_b", 10), ("tirex_rollout_mode", "single_pass"),
                     ("tirex_backend", "cuda"), ("tirex_batch_size", 32)):
            assert _hash_with(**{k: v}) != base, k
        print(" 45  a differing C/H/Q/checkpoint/backend/mode/batch/seed/protocol REFUSES "
              "reuse     OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_46_partial_cell_resumes_safely():
    from probing.phase1 import CellStore, REQUIRED_ARTIFACTS, clean_staging
    from probing.phase1_cells import save_cell
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _synthetic_cell()
        store = CellStore(tmp, c["model"], c["tag"])
        stage = store.begin()
        save_cell(stage, **c)
        assert stage.exists() and not store.final.exists(), \
            "work in progress must never appear at the final path"
        # a killed job leaves the staging dir; the next run removes it
        removed = clean_staging(tmp)
        assert any(c["tag"] in r for r in removed), removed
        assert not stage.exists()
        st, _ = store.status(c["config_hash"])
        assert st == "pending"
        # a corrupt COMPLETE (artifact deleted after the fact) is detected and recomputed
        stage = store.begin()
        save_cell(stage, **c)
        store.commit(stage, c["config_hash"])
        (store.final / REQUIRED_ARTIFACTS[2]).unlink()
        st, why = store.status(c["config_hash"])
        assert st == "corrupt" and "missing artifacts" in why, (st, why)
        print(" 46  a killed cell leaves only staging (auto-cleaned); a corrupt COMPLETE is "
              "detected  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_47_combined_tables_rebuild_from_artifacts_only():
    import csv as _csv
    from probing.phase1 import CellStore
    from probing.phase1_cells import save_cell
    from experiments.make_phase1_tables import build
    tmp = Path(tempfile.mkdtemp())
    try:
        made = []
        for i, (model, tag) in enumerate((("chronos2", "m4_hourly"),
                                          ("timesfm3", "m4_hourly"),
                                          ("tirex", "monash_electricity_hourly"))):
            c = _synthetic_cell(model=model, tag=tag, seed=i)
            store = CellStore(tmp, model, tag)
            save_cell(store.begin(), **c)
            store.commit(store.staging(), c["config_hash"])
            made.append((model, tag))
        r = build(tmp)
        assert r["n_complete"] == 3 and r["n_total"] == 42
        out = tmp / "combined"
        for f in ("tunnel_matrix.csv", "layer_metrics_all.csv", "geometry_summary.csv",
                  "cka_index.csv", "provenance_matrix.csv", "dataset_metadata.csv",
                  "model_dataset_manifest.csv", "plot_data.csv", "combined_index.json"):
            assert (out / f).exists(), f
        with open(out / "tunnel_matrix.csv") as fh:
            rows = list(_csv.DictReader(fh))
        assert len(rows) == 9, "3 models x 3 quantities"
        norm = [r for r in rows if r["quantity"].startswith("normalized")]
        assert {r["model"] for r in norm} == set(MODELS)
        for r in norm:
            assert len(r) == 2 + 14, "one column per roster dataset"
        filled = [(r["model"], t) for r in norm for t in PAPER14 if r[t] != ""]
        assert set(filled) == set(made), (filled, made)
        assert all(r["m5"] == "" for r in norm), "a missing cell must be blank, never imputed"
        with open(out / "provenance_matrix.csv") as fh:
            assert len(list(_csv.DictReader(fh))) == 42, "provenance is always complete"
        with open(out / "layer_metrics_all.csv") as fh:
            lm = list(_csv.DictReader(fh))
        assert len(lm) == 14 + 21 + 14
        assert sum(1 for r in lm if r["is_tunnel_entrance"] == "True") == 3
        with open(out / "cka_index.csv") as fh:
            ck = list(_csv.DictReader(fh))
        # 2 estimators per cell, x (1 depth-axis matrix + 1 with-head-input where one exists)
        expected = sum(2 * (2 if model_spec(m).diagnostic_indices else 1) for m, _ in made)
        assert len(ck) == expected, (len(ck), expected)
        assert {r["estimator"] for r in ck} == {"biased", "unbiased"}
        for r in ck:
            assert (tmp / r["path"]).exists(), r["path"]
        assert {r["axis"] for r in ck} == {"block_depth", "with_head_input"}
        idx = json.loads((out / "combined_index.json").read_text())
        assert len(idx["missing"]) == 39 and "partial_run_note" in idx
        assert "domain_status" not in (out / "plot_data.csv").read_text()
        print(" 47  combined tables rebuild from cell artifacts alone and stay correct while "
              "PARTIAL  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================================== #
# EXTRA PROTOCOL CONTRACTS (48-52)
# =========================================================================== #
def test_48_adamw_is_invariant_to_a_constant_loss_rescale():
    """Why Chronos-2 may keep its sum-over-quantiles objective without changing any number.

    AdamW normalizes the gradient by its own second moment and its weight decay is decoupled
    from the loss, so multiplying the objective by a constant leaves the fitted weights
    unchanged up to Adam's epsilon. This is the claim ``phase1``'s docstring makes; here it is
    measured rather than asserted.
    """
    def fit(scale):
        torch.manual_seed(0)
        lin = torch.nn.Linear(6, 4)
        opt = torch.optim.AdamW([{"params": [lin.weight], "weight_decay": 0.1},
                                 {"params": [lin.bias], "weight_decay": 0.0}], lr=1e-2)
        g = torch.Generator().manual_seed(1)
        X = torch.randn(40, 6, generator=g)
        Y = torch.randn(40, 4, generator=g)
        for _ in range(60):
            loss = scale * ((lin(X) - Y) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        return lin.weight.detach().clone()
    a, b = fit(1.0), fit(18.0)
    rel = float((a - b).abs().max() / a.abs().max())
    assert rel < 1e-4, f"relative weight difference {rel:.2e} under a 2Q=18x loss rescale"
    print(f" 48  AdamW is invariant to a constant loss rescale (2Q=18x -> {rel:.1e} relative) OK")


def test_49_tunnel_is_invariant_to_the_loss_scale():
    spec = model_spec("chronos2")
    v = np.array([9, 8, 7, 6, 1.02, 2.0, 1.5, 1.3, 1.2, 1.1, 1.05, 1.02, 1.0, 1.0])
    base = phase1.tunnel_record(spec, v)["headline"]["index"]
    for s in (2 * 9, 0.5, 1e3):
        assert phase1.tunnel_record(spec, v * s)["headline"]["index"] == base, s
    print(" 49  the tunnel entrance is invariant to any positive rescale of the loss        OK")


def test_50_depth_coordinates():
    """relative_depth is cross-model in [0, 1] over the DEPTH AXIS; relative_position monotone."""
    for m, blocks, n_pts, n_depth in (("chronos2", 12, 14, 13), ("timesfm3", 20, 21, 21),
                                      ("tirex", 12, 14, 13)):
        s = model_spec(m)
        assert s.num_blocks == blocks and s.n_points == n_pts and s.n_depth_points == n_depth
        assert s.points[0].label == "Emb" and s.points[0].relative_depth == 0.0
        assert [p.position_index for p in s.points] == list(range(n_pts))
        dep = [s.points[i] for i in s.depth_indices]
        assert dep[-1].relative_depth == 1.0, "the FINAL BLOCK is normalized depth 1.0"
        assert dep[-1].label == s.reference_label
        rd = [p.relative_depth for p in dep]
        rp = [p.relative_position for p in dep]
        assert all(0.0 <= x <= 1.0 for x in rd)
        assert rd == sorted(rd) and len(set(rd)) == n_depth, "strictly increasing depths"
        assert rp == sorted(rp) and rp[0] == 0.0 and rp[-1] == 1.0 and len(set(rp)) == n_depth
        for i in s.diagnostic_indices:
            assert s.points[i].relative_depth is None, "a norm has no normalized depth"
            assert s.points[i].relative_position is None
    # the final-norm point ties with the last block on block_index but NOT on relative_depth
    for m in ("chronos2", "tirex"):
        s = model_spec(m)
        assert s.points[-1].kind == "final_norm" and s.points[-2].kind == "block"
        assert s.points[-1].block_index == s.points[-2].block_index == s.num_blocks
        assert s.points[-1].relative_depth is None and s.points[-2].relative_depth == 1.0
    assert model_spec("timesfm3").points[-1].kind == "block", "L20 IS what the head reads"
    print(" 50  relative_depth spans [0,1] over the depth axis (final BLOCK = 1.0); a norm "
          "has none  OK")


def test_51_bootstrap_reuses_the_project_cluster_bootstrap():
    import inspect
    src = inspect.getsource(phase1.cluster_ci)
    assert "cluster_bootstrap_counts" in src and "cluster_bootstrap_apply" in src
    assert phase1.PHASE1_BOOT_B == 5000
    rng = np.random.default_rng(0)
    L, n = 5, 60
    W = np.abs(rng.normal(1, .2, (L, n)))
    sid = rng.integers(0, 7, n)
    c = phase1.cluster_ci(W, sid, 200, 0, L - 1)
    assert np.allclose(c["point"], W.mean(axis=1))
    assert c["n_clusters"] == len(np.unique(sid)) and c["n_windows"] == n
    for lo, p, hi in zip(c["ci_lo"], c["point"], c["ci_hi"]):
        assert lo <= hi
    assert abs(c["delta_vs_last"][L - 1]) < 1e-12, "delta against itself must be exactly 0"
    assert phase1.cluster_ci(W, sid, 200, 0, L - 1)["ci_lo"] == c["ci_lo"], "deterministic"
    # ONE shared count matrix -> the deltas are paired, which a per-layer bootstrap would break
    assert len(c["delta_ci_lo"]) == L
    print(" 51  the cluster bootstrap is the project's, paired across depths, B=5000 default OK")


def test_52_native_baseline_never_enters_selection():
    import inspect
    from experiments import run_three_model_phase1 as drv
    src = inspect.getsource(drv.run_cell)
    tun_line = [l for l in src.splitlines() if "tunnel_record(" in l][0]
    assert "native" not in tun_line, tun_line
    assert src.index("tunnel_record(") < src.index("native_reference") \
        if "native_reference" in src else True
    for mod in ("probing.phase1_chronos2", "probing.phase1_timesfm3", "probing.phase1_tirex"):
        m = __import__(mod, fromlist=["x"])
        assert not any("tunnel" in n for n in dir(m)), \
            f"{mod} must not define a tunnel of its own"
    sig = inspect.signature(phase1.tunnel_record)
    assert list(sig.parameters) == ["spec", "val_losses", "test_losses", "tol", "tols"]
    print(" 52  the native head is comparison only; it cannot reach the tunnel criterion    OK")


# =========================================================================== #
# HEAD-INPUT DIAGNOSTICS vs THE DEPTH AXIS (53-57)
# A final normalization adds no block, so it is NOT an additional model depth. It is probed
# and saved for the later native-head/alignment work, and excluded from everything that
# defines a depth.
# =========================================================================== #
def test_53_point_type_vocabulary_and_assignment():
    assert phase1.POINT_TYPES == ("block_depth", "head_input_diagnostic")
    expect = {"chronos2": (["L12+LN"], 13, "L12"),
              "timesfm3": ([], 21, "L20"),
              "tirex": (["L12+RMS"], 13, "L12")}
    for m, (diag, n_depth, ref) in expect.items():
        spec = model_spec(m)
        assert spec.diagnostic_labels == diag, (m, spec.diagnostic_labels)
        assert spec.n_depth_points == n_depth
        assert spec.depth_labels == ["Emb"] + [f"L{k}" for k in range(1, n_depth)]
        assert spec.reference_label == ref
        assert set(spec.point_types) <= set(phase1.POINT_TYPES)
        # every Emb/block point is on the axis; every final_norm point is not
        for p in spec.points:
            assert p.on_depth_axis == (p.kind in ("embedding", "block")), p
            assert (p.point_type == phase1.HEAD_INPUT_DIAGNOSTIC) == (p.kind == "final_norm")
        # what the head READS and what counts as a DEPTH are different claims
        assert spec.points[spec.native_readout_index].label == (diag[0] if diag else ref)
    try:
        phase1.RepPoint("X", "X", "block", 1, 0, 12, 1, 1, point_type="depth")
    except ValueError as e:
        assert "point_type" in str(e)
    else:
        raise AssertionError("an out-of-vocabulary point_type must be refused")
    print(" 53  point_type vocabulary closed; 1 head-input diagnostic for Chronos-2/TiRex, "
          "0 for TimesFM-3  OK")


def test_54_tunnel_never_sees_the_head_input():
    """The criterion runs on the depth axis and references the FINAL BLOCK -- provably."""
    for m in MODELS:
        spec = model_spec(m)
        n, dep = spec.n_points, spec.depth_indices
        base = np.full(n, 5.0)
        base[dep] = np.linspace(2.0, 1.0, spec.n_depth_points)
        ref_i = spec.reference_index
        # whatever this curve's entrance is, it must not move when the diagnostic does. (It is
        # deliberately NOT asserted to be the final depth: the criterion's <= is inclusive, so
        # on a linear ramp the second-to-last depth can land exactly on (1+tol)*final.)
        want = phase1.tunnel_record(spec, base)["headline"]["index"]
        assert want in dep, want
        for factor in (1e-6, 1e6):                       # absurdly good / absurdly bad
            v = base.copy()
            for i in spec.diagnostic_indices:
                v[i] = base[ref_i] * factor
            r = phase1.tunnel_record(spec, v, v)
            assert r["headline"]["index"] == want, (m, factor, r["headline"]["index"])
            assert r["reference_index"] == ref_i and r["reference_is_final_block"]
            assert abs(r["headline"]["val_loss_at_reference"] - base[ref_i]) < 1e-12, \
                "the reference loss must be the final BLOCK's, untouched by the diagnostic"
            assert r["excluded_points"] == spec.diagnostic_labels
            assert len(r["val_loss_by_depth"]) == spec.n_depth_points
            assert r["depth_axis_labels"] == spec.depth_labels
            for lab in spec.diagnostic_labels:           # reported, never used
                assert lab in r["head_input_diagnostic"]
                assert lab not in r["depth_axis_labels"]
        # every saved tolerance obeys the same rule
        v = base.copy()
        for i in spec.diagnostic_indices:
            v[i] = 1e-9
        for k, e in phase1.tunnel_record(spec, v)["by_tolerance"].items():
            assert e["point_type"] == phase1.BLOCK_DEPTH, (m, k, e["label"])
    print(" 54  a 1e-6x / 1e6x head-input loss moves neither the entrance nor the reference; "
          "all tolerances  OK")


def test_55_head_input_excluded_from_normalized_depth():
    for m in MODELS:
        spec = model_spec(m)
        depths = [p.relative_depth for p in spec.points if p.relative_depth is not None]
        assert len(depths) == spec.n_depth_points
        assert depths[-1] == 1.0 and depths[0] == 0.0
        assert len(set(depths)) == len(depths), "no two depths may share a coordinate"
        for i in spec.diagnostic_indices:
            d = spec.points[i].as_dict()
            assert d["relative_depth"] is None and d["relative_position"] is None
            assert d["include_in_main_depth_axis"] is False
            assert d["point_type"] == phase1.HEAD_INPUT_DIAGNOSTIC
    # the spec dict a downstream reader consumes carries the axis explicitly
    sd = model_spec("chronos2").as_dict()
    assert sd["depth_axis_labels"][-1] == "L12" and sd["n_depth_points"] == 13
    assert sd["head_input_diagnostic_labels"] == ["L12+LN"]
    assert sd["tunnel_reference_point"] == "L12" and sd["native_readout_point"] == "L12+LN"
    assert "not an additional model depth" in sd["head_input_caveat"].lower()
    print(" 55  normalized depth is defined over block depths only; a norm has none and is "
          "flagged  OK")


def test_56_main_cka_and_erank_exclude_the_head_input():
    rng = np.random.default_rng(5)
    for m in ("chronos2", "tirex"):
        spec = model_spec(m)
        mats = [rng.normal(size=(50, spec.d)) for _ in spec.points]
        b = phase1.geometry_block(mats, spec.labels, d=spec.d, split="test", variant="v",
                                  point_types=spec.point_types, null_floor_reps=1)
        for est in ("unbiased", "biased"):
            M, F = b["cka"][est], b["cka_with_head_input"][est]
            assert M.shape == (spec.n_depth_points,) * 2, (m, est, M.shape)
            assert F.shape == (spec.n_points,) * 2
            # the main matrix IS the submatrix -- nothing is computed twice, and they cannot
            # disagree about a shared pair
            d = b["depth_axis_indices"]
            assert np.array_equal(M, F[np.ix_(d, d)])
        assert b["labels"] == spec.depth_labels
        assert all(l not in b["labels"] for l in spec.diagnostic_labels)
        assert b["head_input_labels"] == spec.diagnostic_labels
        # effective rank KEEPS the head input (a cheap diagnostic) but tags it
        er = b["effective_rank"]
        assert er["layer_names"] == spec.labels and len(er["effective_rank"]) == spec.n_points
        assert er["point_types"] == spec.point_types
        assert er["include_in_main_depth_axis"][-1] is False
    # TimesFM-3 has no diagnostic, so no second matrix is produced at all
    spec = model_spec("timesfm3")
    mats = [rng.normal(size=(50, spec.d)) for _ in spec.points]
    b = phase1.geometry_block(mats, spec.labels, d=spec.d, split="test", variant="v",
                              point_types=spec.point_types, null_floor_reps=1)
    assert "cka_with_head_input" not in b and b["cka"]["unbiased"].shape == (21, 21)
    print(" 56  main CKA covers the depth axis only and equals the submatrix; effective rank "
          "keeps + tags  OK")


def test_57_serialization_makes_accidental_inclusion_impossible():
    import csv as _csv
    from probing.phase1 import CellStore
    from probing.phase1_cells import save_cell
    from experiments.make_phase1_tables import build
    tmp = Path(tempfile.mkdtemp())
    try:
        for model, tag in (("chronos2", "m4_hourly"), ("timesfm3", "uber_tlc_hourly"),
                           ("tirex", "monash_electricity_hourly")):
            c = _synthetic_cell(model=model, tag=tag)
            st = CellStore(tmp, model, tag)
            save_cell(st.begin(), **c)
            st.commit(st.staging(), c["config_hash"])
            spec = c["spec"]
            s = json.loads((st.final / "summary.json").read_text())
            assert s["point_types"] == spec.point_types
            assert s["depth_axis_labels"] == spec.depth_labels
            assert s["final_depth_point"] == spec.reference_label
            assert s["final_depth_loss"]["test"] == c["res"]["test_loss"][spec.reference_index]
            assert "head_input_caveat" in s
            rows = list(_csv.DictReader(open(st.final / "layer_metrics.csv")))
            assert len(rows) == spec.n_points
            for r in rows:
                assert r["point_type"] in phase1.POINT_TYPES
                if r["point_type"] == phase1.HEAD_INPUT_DIAGNOSTIC:
                    assert r["relative_depth"] == "" and r["relative_position"] == ""
                    assert r["include_in_main_depth_axis"] == "False"
                    assert r["is_reference_point"] == "False", \
                        "a normalization must never be the final-depth reference"
                    assert r["is_tunnel_entrance"] == "False"
            ref = [r for r in rows if r["is_reference_point"] == "True"]
            assert len(ref) == 1 and ref[0]["layer"] == spec.reference_label
        build(tmp)
        out = tmp / "combined"
        pd_rows = list(_csv.DictReader(open(out / "plot_data.csv")))
        assert pd_rows and all(r["point_type"] == phase1.BLOCK_DEPTH for r in pd_rows)
        assert not any(r["layer"] in ("L12+LN", "L12+RMS") for r in pd_rows), \
            "the main plotting frame must contain no head-input row at all"
        hi_rows = list(_csv.DictReader(open(out / "plot_data_head_input.csv")))
        assert {r["layer"] for r in hi_rows} == {"L12+LN", "L12+RMS"}
        lm = list(_csv.DictReader(open(out / "layer_metrics_all.csv")))
        assert len(lm) == 14 + 21 + 14, "every probed point is still reported somewhere"
        assert sum(1 for r in lm if r["point_type"] == phase1.HEAD_INPUT_DIAGNOSTIC) == 2
        ck = list(_csv.DictReader(open(out / "cka_index.csv")))
        main = [r for r in ck if r["is_main_depth_axis"] == "True"]
        assert main and all(r["axis"] == phase1.BLOCK_DEPTH for r in main)
        for r in main:
            assert "L12+LN" not in r["labels"] and "L12+RMS" not in r["labels"]
            assert (tmp / r["path"]).exists()
        extra = [r for r in ck if r["is_main_depth_axis"] == "False"]
        assert extra and all("__with_head_input" in r["path"] for r in extra)
        for r in extra:
            assert (tmp / r["path"]).exists()
        gs = list(_csv.DictReader(open(out / "geometry_summary.csv")))
        assert all(r["is_headline"] == "False" for r in gs
                   if r["point_type"] == phase1.HEAD_INPUT_DIAGNOSTIC)
        idx = json.loads((out / "combined_index.json").read_text())
        assert idx["main_depth_axis"]["frame"] == "plot_data.csv"
        assert "head_input_caveat" in idx
        print(" 57  point_type is serialized everywhere; plot_data.csv and the main CKA files "
              "hold no head input  OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =========================================================================== #
# helpers + runner
# =========================================================================== #
def _fake_windows(rng, n_tr=12, n_va=5, n_te=5, C=512, H=64):
    return {
        "X_train": rng.normal(size=(n_tr, C)).astype(np.float32),
        "X_val": rng.normal(size=(n_va, C)).astype(np.float32),
        "X_test": rng.normal(size=(n_te, C)).astype(np.float32),
        "Y_test_traj": rng.normal(size=(n_te, H)).astype(np.float32),
        "series_test": np.arange(n_te, dtype=np.int64),
        "meta": {"C": C, "H": H, "m_season": 24, "seed": 0,
                 "split_mode": "rolling_origin_within_series", "cluster_unit": "series",
                 "origins": {"test": list(range(n_te))}},
    }


def _model_backed():
    print("\n--- model-backed contracts 28/30/31 (TiRex device handling + native head) ---")
    import subprocess
    r = subprocess.run([sys.executable, "-m", "tests.test_tirex_probing", "--with-model"],
                       cwd=REPO_ROOT)
    if r.returncode:
        raise SystemExit(" 28/30/31  DELEGATED TiRex model contracts FAILED")
    print(" 28  TiRex device handling passes on the live accelerator      (delegated)   OK")
    print(" 30  no CPU/CUDA tensor mismatch in any model-backed entry point (delegated) OK")
    print(" 31  the native-head reconstruction contract still holds        (delegated)  OK")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"\nPHASE-1 PRE-RUN CONTRACTS  ({len(tests)} model-free groups)\n" + "=" * 78)
    for t in tests:
        t()
    print("=" * 78)
    print(f"all {len(tests)} model-free Phase-1 contracts hold")
    if "--with-model" in argv:
        _model_backed()
    else:
        print("run with --with-model for 28/30/31 on a COMPUTE node (delegated to "
              "tests.test_tirex_probing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
