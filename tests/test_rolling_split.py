"""Focused, no-GPU tests for the rolling-origin within-series split (extended_v3_rolling).

Synthetic series only (monkeypatched load) — no HF download, no model, no torch. Verifies the
split CONTRACT the design locked:
  * m4_hourly (and all four datasets) use rolling_origin_within_series, NOT cross_series;
  * contexts may overlap but train/val/test TARGETS never overlap within a series;
  * no context includes an observation at/after its forecast origin;
  * the SAME deterministic series carry validation and test;
  * every selected val/test series keeps >= 1 retained training window;
  * origins are deterministic under seed 0;
  * the canonical MASE denominator uses history STRICTLY BEFORE the test target;
  * windows whose context OR target contains a NaN are rejected (and a series left with < 3
    valid origins becomes ineligible);
  * constant contexts are rejected under the same sigma_eps rule as _make_examples;
  * the real (1394, 262, 262) budget resolves;
  * extended_v2 is unchanged (still cross_series for short M4, 1500/650, no val split).

Run: python -m tests.test_rolling_split   (or pytest tests/test_rolling_split.py)
"""

from __future__ import annotations

import contextlib

import numpy as np

from probing import config, id_data

C, H = 512, 64


def _series(n, L, base_step=10.0):
    """n synthetic finite, non-constant series of length L (arange + a per-series offset)."""
    return [np.arange(L, dtype=np.float64) + base_step * k for k in range(n)]


@contextlib.contextmanager
def _active(series, set_name, budget=None):
    """Temporarily point load_seen_series at `series`, activate `set_name` (optionally overriding
    its budget), and restore everything afterwards — so tests never leak global state."""
    prev_set = config.DATASET_SET
    prev_load = id_data.load_seen_series
    prev_budget = dict(id_data.BUDGET_BY_SET)
    try:
        id_data.load_seen_series = lambda tag: series
        if budget is not None:
            id_data.BUDGET_BY_SET[set_name] = budget
        config.set_dataset_set(set_name)
        yield
    finally:
        id_data.load_seen_series = prev_load
        id_data.BUDGET_BY_SET.clear()
        id_data.BUDGET_BY_SET.update(prev_budget)
        config.set_dataset_set(prev_set)


def test_rolling_split_mode_and_val_test_same_series():
    with _active(_series(8, 800), "extended_v3_rolling", (10, 4, 4)):
        w = id_data.build_windows("m4_hourly")
    m = w["meta"]
    assert m["split_mode"] == "rolling_origin_within_series"
    assert m["n_val"] == 4 and m["n_test"] == 4
    assert set(w["series_val"].tolist()) == set(w["series_test"].tolist()) == set(m["selected_series"])
    assert len(m["selected_series"]) == 4


def test_all_four_datasets_same_semantics():
    with _active(_series(8, 800), "extended_v3_rolling", (10, 4, 4)):
        for tag in id_data.ID_DATASET_SPECS["extended_v3_rolling"]:
            assert id_data.build_windows(tag)["meta"]["split_mode"] == "rolling_origin_within_series"


def test_targets_non_overlap_across_splits():
    with _active(_series(8, 800), "extended_v3_rolling", (10, 4, 4)):
        w = id_data.build_windows("m4_hourly")
    m = w["meta"]
    st, ot = w["series_train"], m["origins"]["train"]
    sv, ov = w["series_val"], m["origins"]["val"]
    se, oe = w["series_test"], m["origins"]["test"]
    for sid in m["selected_series"]:
        tr = [ot[j] for j in range(len(st)) if st[j] == sid]
        vo = next(ov[j] for j in range(len(sv)) if sv[j] == sid)
        to = next(oe[j] for j in range(len(se)) if se[j] == sid)
        assert tr, f"selected series {sid} contributed no train window"
        assert max(tr) < vo < to                         # train targets before val before test
        ivs = sorted([(o, o + H) for o in tr] + [(vo, vo + H), (to, to + H)])
        for (a0, a1), (b0, b1) in zip(ivs, ivs[1:]):     # explicit interval disjointness
            assert a1 <= b0, f"overlapping targets {(a0, a1)} / {(b0, b1)} for series {sid}"


def test_context_precedes_its_target():
    series = _series(8, 800)
    with _active(series, "extended_v3_rolling", (10, 4, 4)):
        w = id_data.build_windows("m4_hourly")
    st, ot = w["series_train"], w["meta"]["origins"]["train"]
    for j in range(len(st)):
        sid, o = int(st[j]), ot[j]
        assert np.allclose(w["X_train"][j], series[sid][o - C:o])     # exactly the C obs before o
        assert float(w["X_train"][j].max()) < series[sid][o]          # nothing at/after the origin


def test_deterministic_under_seed0():
    with _active(_series(8, 800), "extended_v3_rolling", (10, 4, 4)):
        w1 = id_data.build_windows("m4_hourly")
        w2 = id_data.build_windows("m4_hourly")
    assert np.array_equal(w1["series_test"], w2["series_test"])
    assert np.array_equal(w1["X_test"], w2["X_test"])
    assert w1["meta"]["origins"]["train"] == w2["meta"]["origins"]["train"]


def test_every_selected_series_has_a_train_window():
    with _active(_series(8, 800), "extended_v3_rolling", (10, 4, 4)):
        w = id_data.build_windows("m4_hourly")
    assert set(w["meta"]["selected_series"]).issubset(set(w["series_train"].tolist()))


def test_mase_history_excludes_test_target():
    series = _series(8, 800)
    for s in series:
        s[704:] = 1e9                                    # spike AT/AFTER the test target start (704)
    with _active(series, "extended_v3_rolling", (10, 4, 4)):
        w = id_data.build_windows("m4_hourly")
    # seasonal-naive scale of arange history before the test target is exactly 24; the 1e9 tail
    # would blow it up if the denominator wrongly included the test target region.
    assert np.allclose(w["test_denominator"], 24.0)
    assert w["meta"]["mase_canonical"] is True


def _origins_for(w, sid):
    """(train_origins, val_origin, test_origin) recorded for series `sid` (val/test None if absent)."""
    m = w["meta"]
    tr = [m["origins"]["train"][j] for j in range(len(w["series_train"])) if w["series_train"][j] == sid]
    va = [m["origins"]["val"][j] for j in range(len(w["series_val"])) if w["series_val"][j] == sid]
    te = [m["origins"]["test"][j] for j in range(len(w["series_test"])) if w["series_test"][j] == sid]
    return tr, (va[0] if va else None), (te[0] if te else None)


def test_nan_windows_rejected():
    # len-800 series -> valid ctx-starts {0,64,128,192}, targets at {512,576,640,704}.
    series = _series(8, 800)
    series[0][600] = np.nan     # kills origin 576 (target) AND 640/704 (contexts) -> 1 valid -> OUT
    series[1][0] = np.nan       # kills only origin 512 (context) -> 3 valid -> still eligible
    with _active(series, "extended_v3_rolling", (20, 7, 7)):     # all 7 eligible selected
        w = id_data.build_windows("m4_hourly")
    m = w["meta"]
    assert m["excluded_series"]["insufficient_valid"] == 1
    assert 0 not in set(w["series_train"].tolist()) and 0 not in set(m["selected_series"])
    tr, va, te = _origins_for(w, 1)                              # NaN dropped 512 leakage-free:
    assert (tr, va, te) == ([576], 640, 704)
    for X in (w["X_train"], w["X_val"], w["X_test"], w["Y_train_traj"], w["Y_val_traj"], w["Y_test_traj"]):
        assert np.isfinite(X).all(), "a NaN-bearing window leaked into a split"


def test_constant_context_rejected():
    series = _series(8, 800)
    series[0][:] = 5.0          # fully constant -> zero valid origins -> ineligible
    series[1][:512] = 7.0       # constant FIRST context only -> origin 512 invalid, 576/640/704 keep
    with _active(series, "extended_v3_rolling", (20, 7, 7)):
        w = id_data.build_windows("m4_hourly")
    m = w["meta"]
    assert m["excluded_series"]["insufficient_valid"] == 1
    assert 0 not in set(w["series_train"].tolist()) and 0 not in set(m["selected_series"])
    tr, va, te = _origins_for(w, 1)
    assert (tr, va, te) == ([576], 640, 704)                     # same rule as _make_examples


def test_real_budget_resolves():
    # 300 series x 7 origins each -> exercises the true (1394, 262, 262) budget (no budget override)
    with _active(_series(300, 1000), "extended_v3_rolling"):
        w = id_data.build_windows("m4_hourly")
    m = w["meta"]
    assert (m["target_train"], m["target_val"], m["target_test"]) == (1394, 262, 262)
    assert m["n_val"] == 262 and m["n_test"] == 262
    assert m["n_train"] == 1394                          # 300*5=1500 candidates capped to 1394
    assert set(m["selected_series"]).issubset(set(w["series_train"].tolist()))


def test_extended_v2_unchanged():
    # short series -> extended_v2 (NOT a rolling set) still auto-picks cross_series + 1500/650,
    # and emits no validation split.
    with _active(_series(20, 800), "extended_v2"):
        w = id_data.build_windows("m4_hourly")
    assert w["meta"]["split_mode"] == "cross_series"
    assert w["meta"]["target_train"] == 1500 and w["meta"]["target_test"] == 650
    assert "X_val" not in w


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in tests:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS ({len(tests)} tests)")
