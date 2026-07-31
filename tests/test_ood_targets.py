"""Minimal loader/window contracts for the documented pretraining-OOD targets.

Scope (deliberately small): one loader + shape + parent-cluster-id check per dataset, a
determinism + frozen-650-window check on build_ood_windows, and a data-free unit test of the
SG-Carpark 15min->hourly aggregation rule. Needs `module load arrow` for the arrow reads; the
data-dependent checks SKIP cleanly when the pre-downloaded shards / BOOM manifest are absent
(so this stays runnable on a bare checkout). Not a GPU / model test.

Run:  python -m tests.test_ood_targets
"""

from __future__ import annotations

import collections

import numpy as np

from probing import id_data
from probing.id_data import (load_ood_target_series, build_ood_windows,
                             _aggregate_15min_to_hourly, OOD_TARGET_ROOT)

C, H = 512, 64


def _has_shard(tag):
    if tag == "boom_hourly":
        return id_data.BOOM_MANIFEST.exists()
    return (OOD_TARGET_ROOT / tag / "data-00000-of-00001.arrow").exists()


def _skip(tag, reason=""):
    print(f"  [skip] {tag}: data absent ({reason or OOD_TARGET_ROOT / tag}) — "
          "pre-download it on the login node first")


# --------------------------------------------------------------------------- #
# data-free: SG-Carpark aggregation rule (mean of AVAILABLE samples, >=3-of-4 required, no fill)
# --------------------------------------------------------------------------- #

def test_aggregate_15min_to_hourly():
    # start at :15 -> offset 3 to reach the first :00; 3 + 4*2 = 11 samples -> 2 full hours
    x = np.array([9, 9, 9,  1, 2, 3, 4,  10, 20, 30, 40], dtype=float)
    out = _aggregate_15min_to_hourly(x, start_minute=15)
    assert out.shape == (2,), out.shape
    assert np.allclose(out, [2.5, 25.0]), out               # means of [1,2,3,4] and [10,20,30,40]
    # >=3-of-4 rule: ONE missing sample -> mean of the 3 present (NOT NaN, no fill)
    x2 = x.copy(); x2[5] = np.nan                           # hour 0 = [1, 2, NaN, 4]
    out2 = _aggregate_15min_to_hourly(x2, start_minute=15)
    assert np.isclose(out2[0], (1 + 2 + 4) / 3) and np.isclose(out2[1], 25.0), out2
    # TWO missing (only 2 present, below the >=3 threshold) -> that hour is NaN
    x3 = x.copy(); x3[5] = np.nan; x3[6] = np.nan           # hour 0 = [1, 2, NaN, NaN]
    out3 = _aggregate_15min_to_hourly(x3, start_minute=15)
    assert np.isnan(out3[0]) and np.isclose(out3[1], 25.0), out3
    # start exactly on the hour -> offset 0
    assert _aggregate_15min_to_hourly(np.arange(8.0), start_minute=0).shape == (2,)
    print("  test_aggregate_15min_to_hourly OK")


# --------------------------------------------------------------------------- #
# loaders: shape + parent-cluster-id contract
# --------------------------------------------------------------------------- #

def test_sg_carpark_loader():
    if not _has_shard("sg_carpark"):
        return _skip("sg_carpark")
    d = load_ood_target_series("sg_carpark")
    n = len(d["series"])
    assert d["cluster_unit"] == "carpark"
    assert n == 354, n
    assert len(d["cluster_ids"]) == n == len(d["cluster_names"])
    assert sorted(set(d["cluster_ids"])) == list(range(n))          # one carpark per series
    for s in d["series"][:5]:
        assert np.ndim(s) == 1 and len(s) > 3000                    # hourly (~3623 from ~14495 @15min)
    print(f"  test_sg_carpark_loader OK ({n} carparks, hourly len {len(d['series'][0])})")


def test_coastal_loader():
    if not _has_shard("coastal_ts"):
        return _skip("coastal_ts")
    d = load_ood_target_series("coastal_ts")
    assert d["cluster_unit"] == "station"
    cnt = collections.Counter(d["cluster_ids"])
    assert len(cnt) == 24, len(cnt)                                 # 24 stations
    assert all(v == 2 for v in cnt.values())                        # TEMP + PSAL per station
    assert len(d["series"]) == 48
    assert any(nm.endswith(":TEMP") for nm in d["cluster_names"])
    assert any(nm.endswith(":PSAL") for nm in d["cluster_names"])
    assert not any(nm.endswith(":PRES_REL") for nm in d["cluster_names"])   # locked: dropped
    for s in d["series"][:5]:
        assert np.ndim(s) == 1 and len(s) >= C + H
    print(f"  test_coastal_loader OK (24 stations x 2 variates = {len(d['series'])} series)")


def test_boom_loader():
    if not _has_shard("boom_hourly"):
        return _skip("boom_hourly", "manifest not built yet")
    d = load_ood_target_series("boom_hourly")
    assert d["cluster_unit"] == "metric_query"
    n = len(d["series"])
    assert len(d["cluster_ids"]) == n
    assert sorted(set(d["cluster_ids"])) == list(range(n))          # one variate per query
    print(f"  test_boom_loader OK ({n} hourly metric-query variates)")


# --------------------------------------------------------------------------- #
# build_ood_windows: determinism + frozen 650 + eval-only contract
# --------------------------------------------------------------------------- #

def test_build_ood_windows():
    tag = "coastal_ts" if _has_shard("coastal_ts") else ("sg_carpark" if _has_shard("sg_carpark") else None)
    if tag is None:
        return _skip("build_ood_windows", "no target shard present")
    w1 = build_ood_windows(tag, target_test=650)
    w2 = build_ood_windows(tag, target_test=650)
    assert np.array_equal(w1["X_test"], w2["X_test"]), "windows must be deterministic (seed 0)"
    assert np.array_equal(w1["series_test"], w2["series_test"])
    m = w1["meta"]
    assert m["ood_target"] is True and m["split_mode"] == "ood_eval_only"
    n = m["n_test"]
    assert n == min(650, m["n_test_windows_before_subsample"]), (n, m["n_test_windows_before_subsample"])
    assert w1["X_test"].shape == (n, C)
    assert w1["Y_test_traj"].shape == (n, H)
    assert w1["series_test"].shape == (n,)
    assert w1["series_test"].max() < m["n_clusters_total"]
    # query-balanced round-robin: broad cluster coverage + no single cluster dominates
    assert m["sampling"] == "cluster_balanced_round_robin"
    _u, _c = np.unique(w1["series_test"], return_counts=True)
    assert len(_u) == m["n_test_clusters"] == m["n_clusters_total"], (len(_u), m)
    assert int(_c.max()) - int(_c.min()) <= 1, ("clusters unbalanced", _c.min(), _c.max())
    # eval-only: no train, canonical MASE denominator undefined (in-context used downstream)
    assert w1["X_train"].shape == (0, C) and w1["y_train"].size == 0
    assert not np.isfinite(w1["test_denominator"]).any()
    print(f"  test_build_ood_windows OK ({tag}: {n} windows over "
          f"{m['n_test_clusters']}/{m['n_clusters_total']} {m['cluster_unit']}s, "
          f"windows/cluster min/med/max={m['windows_per_cluster_min_med_max']})")


def main():
    print("[test_ood_targets]")
    test_aggregate_15min_to_hourly()
    test_sg_carpark_loader()
    test_coastal_loader()
    test_boom_loader()
    test_build_ood_windows()
    print("done.")


if __name__ == "__main__":
    main()
