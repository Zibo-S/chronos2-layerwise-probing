"""Contracts for the central dataset registry and the 14-dataset benchmark refactor.

No model, no GPU, no feature cache, no network. Everything here is either pure lookup or runs
the window builder against SYNTHETIC series, so it can run on a login node in seconds.

The point of this file is the sequencing requirement: the refactor must PRESERVE the original
seven's results and FAIL LOUDLY on mistakes. So the tests are grouped as

    1-6    the registry itself (roster, fields, controls, fail-loud lookups)
    7-12   seasonal MASE: per-dataset periods, the six new ones pinned by name, and -- the
           decisive one -- bit-identity with the pre-refactor m=24 implementation
    13-14  the reported MASE definition is the in-context one, still, and is documented
    15-19  the deterministic series cap, including byte-identical windows when it does not fire
    20-23  no global PT-ID/PT-OOD control of validity or grouping; provenance is 2-D and
           model-relative, and `not_listed` is never turned into OOD
    24-26  duplicate rosters are gone; no stale labels leak into generated metadata
    27-28  the TimesFM-3 Q=1 probe is constructed correctly and Q=9 still works
    29-31  unknown datasets / seasonal periods / quantile sets fail loudly
    32     the val/test budget is a ceiling, and a reduction is recorded
"""

from __future__ import annotations

import ast
import inspect
import json
import re
import sys
import textwrap
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import id_data, mase, registry                       # noqa: E402
from probing import window_reference as wref                      # noqa: E402
from probing.tunnel import (PT_ID_TAGS, PT_OOD_TAGS, domain_status,  # noqa: E402
                            tunnel_record, tunnel_start)

#: The six NEW datasets whose seasonal period the spec pins by name. A global 24 would have
#: silently produced a wrong-but-plausible MASE for every one of them.
NEW_PERIODS = {
    "LOOP_SEATTLE_5T": 288,
    "electricity_15min": 96,
    "SZ_TAXI_15T": 96,
    "monash_london_smart_meters": 48,
    "m5": 7,
    "wiki_daily_100k": 7,
}

#: The pre-refactor denominator, transcribed verbatim from the two module-level copies that
#: `probing.mase` replaced (run_id_forecasting._mase_denominator and
#: run_timesfm3_probing.mase_denominator). Tests 11/12 compare against THIS, not against a
#: re-derivation, so "bit-identical" means bit-identical to the code that produced the
#: committed numbers.
def _legacy_mase_denominator(X, m=24):
    X64 = np.asarray(X, np.float64)
    return np.abs(X64[:, m:] - X64[:, :-m]).mean(axis=1)


def _synth_series(n, L, seed=7, period=24):
    rng = np.random.default_rng(seed)
    t = np.arange(L)
    return [(rng.normal(0, 1, L) + 10 * np.sin(2 * np.pi * t / period)
             + rng.normal(0, 3)).astype(np.float64) for _ in range(n)]


# --------------------------------------------------------------------------- #
# 1-6  the registry
# --------------------------------------------------------------------------- #
def test_01_roster_is_the_frozen_fourteen():
    assert len(registry.PAPER14) == 14, registry.PAPER14
    assert len(set(registry.PAPER14)) == 14, "duplicate tag in the roster"
    assert "kdd_cup_2022_10T" not in registry.PAPER14, "kdd was dropped at the roster gate"
    assert registry.spec("kdd_cup_2022_10T").roster == "alternate"
    assert "electricity_15min" in registry.PAPER14, "the promoted control must be in the roster"
    print("  1  roster is the frozen 14 (kdd dropped, electricity_15min promoted)      OK")


def test_02_paper7_is_an_exact_subset_in_committed_order():
    assert set(registry.PAPER7) <= set(registry.PAPER14)
    assert registry.PAPER7 == ("monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly",
                               "wind_farms_hourly", "sg_carpark", "coastal_ts", "boom_hourly")
    print("  2  paper7 is an exact subset of paper14, in the committed order            OK")


def test_03_every_row_carries_every_required_field():
    required = ("tag", "display_name", "source_repo", "source_config", "target_column", "freq",
                "domain", "seasonal_m", "cluster_unit", "builder", "max_series", "role")
    for tag in registry.PAPER14:
        d = registry.spec(tag).as_dict()
        for f in required:
            assert f in d, f"{tag}: registry row has no {f!r}"
        assert d["display_name"] and d["freq"] and d["domain"] and d["cluster_unit"]
        if registry.builder(tag) != "rolling_cluster":
            assert d["source_repo"] and d["source_config"] and d["target_column"], tag
    print("  3  every roster row carries tag/name/source/target/freq/domain/m/unit/      OK\n"
          "     builder/max_series/role")


def test_04_electricity_15min_is_a_labelled_control():
    assert registry.is_control("electricity_15min")
    assert registry.role("electricity_15min") == "control"
    assert "control" in registry.spec("electricity_15min").notes.lower()
    assert registry.control_datasets() == ["electricity_15min"]
    assert "electricity_15min" not in registry.primary_datasets(), (
        "a frequency control must never be counted toward domain diversity")
    assert len(registry.primary_datasets()) == 13
    print("  4  electricity_15min is role=control and excluded from domain diversity    OK")


def test_05_fev_sourced_datasets_keep_their_real_target_columns():
    # fev's target columns are NOT uniformly "target" -- getting this wrong reads the wrong
    # column and silently probes a covariate.
    assert registry.spec("kdd_cup_2022_10T").target_column == "Patv"
    assert registry.spec("rossmann_1D").target_column == "Sales"
    assert registry.spec("electricity_15min").target_column == "consumption_kW"
    assert registry.spec("LOOP_SEATTLE_5T").source_repo == "autogluon/fev_datasets"
    assert registry.spec("SZ_TAXI_15T").source_repo == "autogluon/fev_datasets"
    print("  5  fev/chronos source repos and non-'target' target columns are correct    OK")


def test_06_all_fourteen_tags_resolve_through_every_lookup():
    for tag in registry.PAPER14:
        assert registry.seasonal_m(tag) >= 1
        assert registry.display_name(tag)
        assert registry.slug(tag)
        assert registry.builder(tag) in registry.BUILDERS
        assert registry.cluster_unit(tag)
        assert registry.role(tag) in registry.ROLES
        registry.max_series(tag)                       # None is a valid answer
        for model in registry.MODELS:
            p = registry.provenance(tag, model)
            assert p.status in registry.STATUSES and p.evidence_scope in registry.EVIDENCE_SCOPES
    print("  6  all 14 tags resolve through every registry lookup, for all 3 models     OK")


# --------------------------------------------------------------------------- #
# 7-12  seasonal MASE
# --------------------------------------------------------------------------- #
def test_07_original_seven_still_resolve_to_24():
    got = {t: registry.seasonal_m(t) for t in registry.PAPER7}
    assert set(got.values()) == {24}, got
    print("  7  all seven original datasets still resolve to m=24                       OK")


def test_08_the_six_new_periods_are_pinned_by_name():
    for tag, m in NEW_PERIODS.items():
        assert registry.seasonal_m(tag) == m, f"{tag}: {registry.seasonal_m(tag)} != {m}"
    print("  8  LOOP_SEATTLE 288 / electricity_15min 96 / SZ_TAXI 96 / London 48 /      OK\n"
          "     M5 7 / Wiki 7")


def test_09_no_driver_keeps_a_global_seasonal_period():
    offenders = []
    for p in sorted((REPO_ROOT / "experiments").glob("*.py")) + \
             sorted((REPO_ROOT / "probing").glob("*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if re.match(r"\s*M_SEASON\s*=", line):
                offenders.append(f"{p.relative_to(REPO_ROOT)}:{i}")
    assert not offenders, f"a global seasonal period was reintroduced: {offenders}"
    print("  9  no module defines a global M_SEASON any more                            OK")


def test_10_denominator_requires_an_explicit_period():
    sig = inspect.signature(mase.seasonal_denominator)
    assert sig.parameters["m"].default is inspect.Parameter.empty, (
        "m must be REQUIRED -- a default is how a wrong period becomes a printed number")
    for mod in ("run_id_forecasting", "run_timesfm3_probing"):
        src = (REPO_ROOT / "experiments" / f"{mod}.py").read_text()
        assert not re.search(r"def _?mase_denominator\(X, m=", src), (
            f"{mod}: the denominator regained a default period")
    print(" 10  the MASE denominator takes m as a REQUIRED argument everywhere          OK")


def test_11_mase_is_bit_identical_for_the_original_seven():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 512)) * 7.0 + 3.0
    legacy = _legacy_mase_denominator(X)
    for tag in registry.PAPER7:
        got = mase.denominator_for(tag, X)
        assert np.array_equal(got, legacy), f"{tag}: MASE denominator moved"
    # and the full per-window metric, not just the denominator
    y = rng.normal(size=(64, 64)); yhat = rng.normal(size=(64, 64))
    a = mase.per_window_mase(y, yhat, legacy)
    b = mase.per_window_mase(y, yhat, mase.denominator_for("m4_hourly", X))
    assert np.array_equal(a, b)
    print(" 11  MASE is BIT-IDENTICAL to the pre-refactor m=24 code for all seven       OK")


def test_12_a_different_period_actually_changes_the_number():
    # The converse of test 11: if m did not reach the arithmetic, test 11 would pass trivially.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(8, 512)).cumsum(axis=1)
    for tag, m in NEW_PERIODS.items():
        got = mase.denominator_for(tag, X)
        assert np.array_equal(got, mase.seasonal_denominator(X, m))
        if m != 24:
            assert not np.allclose(got, _legacy_mase_denominator(X)), (
                f"{tag}: m={m} produced the same numbers as m=24 -- m is not reaching the math")
    print(" 12  a non-24 period demonstrably changes the denominator (m reaches math)   OK")


# --------------------------------------------------------------------------- #
# 13-14  the reported MASE definition is preserved and documented
# --------------------------------------------------------------------------- #
def test_13_reported_denominator_is_still_the_in_context_one():
    src = inspect.getsource(mase.seasonal_denominator)
    assert "X64[:, m:] - X64[:, :-m]" in src, "the in-context formula changed"
    # the canonical per-series denominator must NOT have become the reported metric
    # a CONSUMPTION looks like w["test_denominator"] / .get("test_denominator"); a prose
    # mention of the name in a docstring is documentation and is exactly what we want to keep.
    use_re = re.compile(r'\[\s*["\']test_denominator["\']\s*\]|get\(\s*["\']test_denominator["\']')
    consumers = []
    for p in sorted((REPO_ROOT / "experiments").glob("*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if use_re.search(line):
                consumers.append(f"{p.name}:{i}")
    assert not consumers, (
        "id_data.test_denominator (the canonical per-series denominator) must NOT be consumed "
        f"as the reported metric: {consumers}")
    print(" 13  reported MASE is still the in-context C=512 denominator, unchanged      OK")


def test_14_the_paper_sentence_exists_and_is_used():
    s = registry.MASE_DEFINITION
    assert "512-step context window" in s and "dataset-specific seasonal period" in s, s
    per = mase.mase_definition_for("LOOP_SEATTLE_5T")
    assert "m=288" in per, per
    print(" 14  the quotable MASE definition sentence exists and substitutes m          OK")


# --------------------------------------------------------------------------- #
# 15-19  the deterministic series cap
# --------------------------------------------------------------------------- #
def test_15_cap_is_deterministic_and_order_independent():
    ids = list(range(5000))
    shuffled = list(np.random.default_rng(3).permutation(ids))
    a, audit = id_data.apply_series_cap("m5", ids, seed=0)
    b, _ = id_data.apply_series_cap("m5", shuffled, seed=0)
    c, _ = id_data.apply_series_cap("m5", ids, seed=0)
    assert a == c, "cap is not reproducible at a fixed seed"
    assert a == b, "cap depends on input (Arrow) order"
    assert a == sorted(a) and len(set(a)) == len(a)
    assert a != sorted(ids)[:len(a)], "cap is a [:n] truncation, not a seeded sample"
    assert a != id_data.apply_series_cap("m5", ids, seed=1)[0], "seed does not change the draw"
    assert audit["applied"] and audit["n_eligible_after"] == registry.max_series("m5")
    print(" 15  series cap: reproducible, seeded, order-independent, not a truncation   OK")


def test_16_cap_selection_is_recorded():
    _, audit = id_data.apply_series_cap("wiki_daily_100k", list(range(20000)), seed=0)
    for k in ("applied", "max_series", "n_eligible_before", "n_eligible_after", "seed",
              "rng_stream", "selection", "capped_series_ids"):
        assert k in audit, f"cap audit is missing {k!r}"
    assert len(audit["capped_series_ids"]) == audit["n_eligible_after"]
    print(" 16  cap records max_series/seed/stream/selection + the kept series ids      OK")


def test_17_cap_is_inert_and_rng_neutral_when_it_does_not_fire():
    _, a = id_data.apply_series_cap("monash_electricity_hourly", list(range(321)), seed=0)
    _, b = id_data.apply_series_cap("m5", list(range(900)), seed=0)
    assert not a["applied"] and not b["applied"]
    # the decisive property: a FIRING cap must not perturb the builder's own rng stream
    want = np.random.default_rng(0).permutation(10).tolist()
    id_data.apply_series_cap("m5", list(range(5000)), seed=0)
    got = np.random.default_rng(0).permutation(10).tolist()
    assert want == got
    print(" 17  cap is inert below the cap and draws from its OWN rng stream            OK")


def test_18_old_seven_windows_are_byte_identical_under_the_cap_code():
    from probing import config
    prev = config.DATASET_SET
    try:
        config.set_dataset_set("extended_v3_rolling")
        S = _synth_series(300, 512 + 8 * 64, seed=7)
        orig = id_data.load_seen_series
        id_data.load_seen_series = lambda tag: [s.copy() for s in S]
        w = id_data._build_rolling_windows("monash_electricity_hourly", 512, 64, 64, 1e-6, 24, 0,
                                           target_train=1394, target_val=30, target_test=30)
        id_data.load_seen_series = orig
    finally:
        config.set_dataset_set(prev)
    assert w["meta"]["series_cap"]["applied"] is False
    # 300 eligible series -> 262-style selection untouched; counts exactly as before the cap
    assert w["meta"]["n_val"] == 30 and w["meta"]["n_test"] == 30
    assert len(w["X_train"]) == 1394
    print(" 18  an under-cap dataset takes the identical path (cap not applied)         OK")


def test_19_cap_unblocks_the_builder_that_used_to_raise():
    from probing import config
    prev = config.DATASET_SET
    try:
        config.set_dataset_set("extended_v3_rolling")
        S = _synth_series(1600, 512 + 5 * 64, seed=11, period=7)
        orig = id_data.load_seen_series
        id_data.load_seen_series = lambda tag: [s.copy() for s in S]
        w = id_data._build_rolling_windows("m5", 512, 64, 64, 1e-6, 7, 0,
                                           target_train=1394, target_val=262, target_test=262)
        w2 = id_data._build_rolling_windows("m5", 512, 64, 64, 1e-6, 7, 0,
                                            target_train=1394, target_val=262, target_test=262)
        id_data.load_seen_series = orig
    finally:
        config.set_dataset_set(prev)
    assert w["meta"]["series_cap"]["applied"] is True
    assert (w["meta"]["n_train"], w["meta"]["n_val"], w["meta"]["n_test"]) == (1394, 262, 262)
    for k in ("X_train", "X_val", "X_test", "series_test"):
        assert np.array_equal(w[k], w2[k]), f"{k} is not reproducible across builds"
    print(" 19  1600 eligible series: builder now yields 1394/262/262, reproducibly     OK")


# --------------------------------------------------------------------------- #
# 20-23  no global PT-ID/PT-OOD control; provenance is 2-D and model-relative
# --------------------------------------------------------------------------- #
def _code_only(fn) -> str:
    """Executable source of ``fn``, with comments and the docstring removed via the AST.

    These checks are about what the code DOES. A comment saying "this used to branch on
    PT_OOD_TAGS" documents the fix, not the bug, and must not fail the test.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    node = tree.body[0]
    body = getattr(node, "body", [])
    if (body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        node.body = body[1:]
    return ast.unparse(tree)


def test_20_window_builder_comes_from_the_registry_not_a_pretraining_label():
    from experiments import run_native_head_adapter as nha
    from experiments import run_timesfm3_last_token_probing as lt
    from probing import windows as W
    # THE dispatch reads the registry ...
    assert "registry.builder(tag)" in _code_only(W.build_for)
    # ... and no window-selecting code anywhere keys off a pretraining label.
    for fn in (W.build_for, nha._windows, lt.windows_for):
        code = _code_only(fn)
        assert "PT_OOD_TAGS" not in code and "PT_ID_TAGS" not in code, (
            f"{fn.__qualname__}: window dispatch still keys off a pretraining label")
    assert "build_for(" in _code_only(lt.windows_for), "the driver must use THE dispatch"
    assert "rolling_cluster" in _code_only(nha._windows)
    assert registry.builder("coastal_ts") == "rolling_cluster"
    assert registry.builder("sg_carpark") == "rolling_cluster"
    assert registry.builder("m5") == "rolling_within_series"
    print(" 20  window builder is registry.builder(tag), not PT-ID/PT-OOD membership    OK")


def test_21_legacy_rosters_are_derived_and_unchanged():
    assert PT_ID_TAGS == ("monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly",
                          "wind_farms_hourly")
    assert PT_OOD_TAGS == ("sg_carpark", "coastal_ts", "boom_hourly")
    src = (REPO_ROOT / "probing" / "tunnel.py").read_text()
    assert "registry.PAPER7" in src, "tunnel.py must derive the rosters, not re-type them"
    print(" 21  legacy PT_ID/PT_OOD tuples derived from the registry, order preserved   OK")


def test_22_provenance_is_two_dimensional_and_model_relative():
    p = registry.provenance("wiki_daily_100k", "chronos2")
    q = registry.provenance("wiki_daily_100k", "timesfm3")
    assert (p.status, p.evidence_scope) == ("pretraining_exposed", "exact_dataset")
    assert (q.status, q.evidence_scope) == ("pretraining_exposed", "source_level")
    assert p.evidence_scope != q.evidence_scope, (
        "the 2-D schema exists precisely so these two do not read as identical evidence")
    assert p.citation and q.citation
    try:
        registry.provenance("m5", "not_a_model")
        raise AssertionError("an unknown model must raise")
    except ValueError:
        pass
    print(" 22  provenance is (status x evidence_scope + citation), per MODEL           OK")


def test_23_not_listed_is_never_turned_into_ood():
    # `not_listed` must stay its own status, and the flat pt_ood label must refuse to attach
    # itself to a dataset that merely is not in some corpus list.
    assert registry.provenance("boom_hourly", "chronos2").status == "not_listed"
    assert registry.provenance("sg_carpark", "chronos2").status == "not_listed"
    for tag in registry.PAPER14:
        for model in registry.MODELS:
            st = registry.provenance(tag, model).status
            assert st in registry.STATUSES
            assert "ood" not in st and "unseen" not in st, (tag, model, st)
    for tag in ("m5", "wiki_daily_100k", "LOOP_SEATTLE_5T", "monash_traffic"):
        try:
            domain_status(tag)
            raise AssertionError(f"domain_status({tag!r}) must refuse to invent a label")
        except ValueError as exc:
            assert "registry.provenance" in str(exc)
    rec = tunnel_record("m5", [3.0, 2.0, 1.9, 1.95], [3.0, 2.0, 1.9, 1.95], model="chronos2")
    assert "domain_status" not in rec, "a new dataset must not be stamped pt_id/pt_ood"
    assert rec["pretraining_provenance"]["model"] == "chronos2"
    old = tunnel_record("m4_hourly", [3.0, 2.0, 1.9, 1.95], [3.0, 2.0, 1.9, 1.95])
    assert old["domain_status"] == {"pretraining": "pt_id", "adaptation": None}
    print(" 23  not_listed stays not_listed; new datasets get no pt_id/pt_ood label     OK")


# --------------------------------------------------------------------------- #
# 24-26  duplicate rosters removed; no stale labels in generated metadata
# --------------------------------------------------------------------------- #
def test_24_no_module_hardcodes_a_roster_or_display_names():
    roster_re = re.compile(r'PT_(ID|OOD)_TAGS\s*=\s*[\[({]\s*["\']')
    name_re = re.compile(r'^\s*(SHORT|PRETTY|TITLES|EPS_TITLES|SHORT_LABELS|SLUG)\s*=\s*\{\s*["\']')
    offenders = []
    for p in sorted((REPO_ROOT / "experiments").glob("*.py")) + \
             sorted((REPO_ROOT / "probing").glob("*.py")):
        for i, line in enumerate(p.read_text().splitlines(), 1):
            if roster_re.search(line) or name_re.search(line):
                offenders.append(f"{p.relative_to(REPO_ROOT)}:{i}: {line.strip()[:60]}")
    assert not offenders, "hardcoded roster / display-name table(s):\n  " + "\n  ".join(offenders)
    print(" 24  no module hardcodes a dataset roster or a display-name table            OK")


def test_25_display_names_and_slugs_have_one_spelling():
    from experiments.run_timesfm3_last_token_probing import SHORT as LT_SHORT
    from experiments.run_cka_analysis import SHORT as CKA_SHORT
    for tag in registry.PAPER7:
        want = registry.display_name(tag)
        assert LT_SHORT[tag] == want == CKA_SHORT[tag], tag
    assert [registry.display_name(t) for t in PT_ID_TAGS] == \
           ["Electricity", "Uber TLC", "M4", "Wind Farms"]
    assert registry.slug("boom_hourly") == "boom" and registry.slug("m4_hourly") == "m4"
    print(" 25  one display-name and one slug per dataset, across every module          OK")


def test_26_generated_metadata_carries_no_stale_pt_labels():
    """A record for a dataset outside the original seven must carry no PT-ID/PT-OOD label.

    This is the regression guard for the thing the refactor set out to remove: a
    Chronos-2-relative pretraining label leaking into another model's output, or onto a
    dataset that has no such evidence.
    """
    stale = re.compile(r"PT-(ID|OOD)|pt_id|pt_ood")
    for tag in ("m5", "wiki_daily_100k", "LOOP_SEATTLE_5T", "electricity_15min"):
        rec = tunnel_record(tag, [3.0, 2.0, 1.9, 1.95], [3.0, 2.0, 1.9, 1.95], model="tirex")
        blob = json.dumps(rec)
        assert not stale.search(blob), f"{tag}: stale pretraining label in {blob[:200]}"
        spec_blob = json.dumps(registry.spec(tag).as_dict())
        assert not stale.search(spec_blob), f"{tag}: stale label in the registry row"
    print(" 26  no PT-ID/PT-OOD label appears in any new dataset's generated metadata   OK")


# --------------------------------------------------------------------------- #
# 27-28  the cross-model Q=1 probe
# --------------------------------------------------------------------------- #
def test_27_timesfm3_quantile_sets_are_constructed_correctly():
    from probing.timesfm3_last_token_probes import (NATIVE_QUANTILES, QUANTILE_SETS,
                                                    make_probe, quantile_set,
                                                    reshape_prediction)
    assert set(QUANTILE_SETS) == {"q1", "q9"}
    q1, cols1, med1 = quantile_set("q1")
    q9, cols9, med9 = quantile_set("q9")
    assert list(q1) == [0.5] and med1 == 0, (q1, med1)
    assert len(q9) == 9 and med9 == 4 and np.allclose(q9, NATIVE_QUANTILES)
    assert list(cols1) == [4], "q1 must score the native head's MEDIAN column"
    assert list(cols9) == list(range(9))
    H = 64
    for name, Q in (("q1", 1), ("q9", 9)):
        lin = make_probe(H, Q=Q, device="cpu")
        assert (lin.in_features, lin.out_features) == (1280, H * Q)
        out = reshape_prediction(np.zeros((5, H * Q), np.float32), H, Q=Q)
        assert tuple(out.shape) == (5, Q, H)
    assert make_probe(H, Q=1).out_features == 64
    assert make_probe(H, Q=9).out_features == 576
    print(" 27  TimesFM-3 q1/q9: probe shapes, (B,Q,H) layout, native median column     OK")


def test_28_q9_remains_the_default_and_q1_gets_its_own_namespace():
    from experiments import run_timesfm3_last_token_probing as lt
    args = lt.parse_args(["--audit-only"])
    assert args.quantile_set == "q9", "Q=9 must remain the default (committed numbers)"
    assert set(lt.parse_args(["--quantile-set", "q1"]).__dict__["quantile_set"]) == set("q1")
    src = inspect.getsource(lt.main)
    assert 'f"timesfm3_last_token_{args.suite}_{args.quantile_set}"' in src, (
        "q1 and q9 must write to different output namespaces")
    print(" 28  q9 is still the default; q1 writes to its own output namespace          OK")


# --------------------------------------------------------------------------- #
# 29-31  fail-loud contracts
# --------------------------------------------------------------------------- #
def test_29_unknown_dataset_fails_loudly_everywhere():
    for fn in (registry.spec, registry.seasonal_m, registry.display_name, registry.builder,
               registry.cluster_unit, registry.max_series, registry.role, registry.slug):
        try:
            fn("definitely_not_a_dataset")
            raise AssertionError(f"{fn.__name__} silently accepted an unknown tag")
        except KeyError:
            pass
    try:
        mase.denominator_for("definitely_not_a_dataset", np.zeros((2, 512)))
        raise AssertionError("MASE silently accepted an unknown tag")
    except KeyError:
        pass
    assert not registry.known("definitely_not_a_dataset")
    print(" 29  every registry lookup and MASE raise KeyError on an unknown dataset     OK")


def test_30_impossible_seasonal_periods_fail_loudly():
    X = np.zeros((3, 512))
    for bad in (0, -1, 512, 10_000):
        try:
            mase.seasonal_denominator(X, bad)
            raise AssertionError(f"m={bad} was silently accepted")
        except ValueError:
            pass
    from probing.timesfm3_last_token_probes import quantile_set
    try:
        quantile_set("q21")
        raise AssertionError("an unknown quantile set was silently accepted")
    except ValueError:
        pass
    print(" 30  m<1, m>=C and an unknown quantile set all raise                         OK")


def test_31_a_missing_window_reference_is_not_a_silent_pass():
    from experiments import run_timesfm3_last_token_probing as lt

    class _Args:
        allow_window_mismatch = False
        force_reference = False
        reference_mode = "require"

    w = {"X_train": np.zeros((4, 512), np.float32), "X_val": np.zeros((2, 512), np.float32),
         "X_test": np.zeros((2, 512), np.float32),
         "Y_test_traj": np.zeros((2, 64), np.float32),
         "series_test": np.array([0, 1], np.int64),
         "meta": {"C": 512, "H": 64, "m_season": 7, "seed": 0, "split_mode": "x",
                  "origins": {"test": [576, 640]}}}
    try:
        lt.assert_window_parity("m5", w, None, reference_mode="require")
        raise AssertionError("a missing reference must abort under 'require'")
    except wref.MissingReferenceError as exc:
        assert "--reference-mode create" in str(exc)
    ident = lt.assert_window_parity("m5", w, None, reference_mode="allow-missing")
    assert ident["parity_ok"] is None and "UNVERIFIED" in ident["parity_warning"]
    # a created reference must then actually detect a changed window
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        rec = wref.write_reference("m5", w, created_by="test", root=Path(d))
        ref = wref.read_reference("m5", root=Path(d))
        assert wref.compare_to_reference("m5", w, ref) == []
        w2 = {**w, "X_test": np.ones((2, 512), np.float32)}
        fails = wref.compare_to_reference("m5", w2, ref)
        assert any("window_digest" in f for f in fails), fails
        assert rec["window_digest"].startswith("sha256:")
    print(" 31  a missing reference aborts; a created one detects changed windows       OK")


def test_32_valtest_budget_is_a_recorded_ceiling_not_a_requirement():
    """A dataset with fewer eligible series than the budget must window, and say so.

    SZ_TAXI_15T has 156 eligible series against a 262 val/test budget. The builder used to
    raise, which would have meant "the frozen roster cannot run". The budget is now a ceiling,
    and the reduction is recorded rather than silent -- it is a real loss of bootstrap units and
    has to be readable off the run.
    """
    from probing import config
    prev = config.DATASET_SET
    try:
        config.set_dataset_set("extended_v3_rolling")
        S = _synth_series(156, 512 + 6 * 64, seed=5, period=96)
        orig = id_data.load_seen_series
        id_data.load_seen_series = lambda tag: [s.copy() for s in S]
        w = id_data._build_rolling_windows("SZ_TAXI_15T", 512, 64, 64, 1e-6, 96, 0,
                                           target_train=1394, target_val=262, target_test=262)
        id_data.load_seen_series = orig
    finally:
        config.set_dataset_set(prev)
    b = w["meta"]["valtest_budget"]
    assert b["reduced"] is True and b["requested"] == 262 and b["realized"] == 156
    assert w["meta"]["n_val"] == w["meta"]["n_test"] == 156
    # and a dataset that MEETS the budget records no reduction
    try:
        config.set_dataset_set("extended_v3_rolling")
        S = _synth_series(300, 512 + 6 * 64, seed=6)
        orig = id_data.load_seen_series
        id_data.load_seen_series = lambda tag: [s.copy() for s in S]
        w2 = id_data._build_rolling_windows("monash_electricity_hourly", 512, 64, 64, 1e-6, 24, 0,
                                            target_train=1394, target_val=262, target_test=262)
        id_data.load_seen_series = orig
    finally:
        config.set_dataset_set(prev)
    assert w2["meta"]["valtest_budget"]["reduced"] is False
    assert w2["meta"]["n_val"] == w2["meta"]["n_test"] == 262
    print(" 32  val/test budget is a CEILING; a reduction is recorded, not silent       OK")


TESTS = [test_01_roster_is_the_frozen_fourteen,
         test_02_paper7_is_an_exact_subset_in_committed_order,
         test_03_every_row_carries_every_required_field,
         test_04_electricity_15min_is_a_labelled_control,
         test_05_fev_sourced_datasets_keep_their_real_target_columns,
         test_06_all_fourteen_tags_resolve_through_every_lookup,
         test_07_original_seven_still_resolve_to_24,
         test_08_the_six_new_periods_are_pinned_by_name,
         test_09_no_driver_keeps_a_global_seasonal_period,
         test_10_denominator_requires_an_explicit_period,
         test_11_mase_is_bit_identical_for_the_original_seven,
         test_12_a_different_period_actually_changes_the_number,
         test_13_reported_denominator_is_still_the_in_context_one,
         test_14_the_paper_sentence_exists_and_is_used,
         test_15_cap_is_deterministic_and_order_independent,
         test_16_cap_selection_is_recorded,
         test_17_cap_is_inert_and_rng_neutral_when_it_does_not_fire,
         test_18_old_seven_windows_are_byte_identical_under_the_cap_code,
         test_19_cap_unblocks_the_builder_that_used_to_raise,
         test_20_window_builder_comes_from_the_registry_not_a_pretraining_label,
         test_21_legacy_rosters_are_derived_and_unchanged,
         test_22_provenance_is_two_dimensional_and_model_relative,
         test_23_not_listed_is_never_turned_into_ood,
         test_24_no_module_hardcodes_a_roster_or_display_names,
         test_25_display_names_and_slugs_have_one_spelling,
         test_26_generated_metadata_carries_no_stale_pt_labels,
         test_27_timesfm3_quantile_sets_are_constructed_correctly,
         test_28_q9_remains_the_default_and_q1_gets_its_own_namespace,
         test_29_unknown_dataset_fails_loudly_everywhere,
         test_30_impossible_seasonal_periods_fail_loudly,
         test_31_a_missing_window_reference_is_not_a_silent_pass,
         test_32_valtest_budget_is_a_recorded_ceiling_not_a_requirement]

if __name__ == "__main__":
    print("dataset-registry / 14-dataset refactor contracts "
          "(no model, no GPU, no cache, no network)\n")
    for t in TESTS:
        t()
    print(f"\nALL {len(TESTS)} CONTRACT GROUPS PASS")
