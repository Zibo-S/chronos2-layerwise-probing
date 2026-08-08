"""Validation gates for the distance-vs-gain ladder join.

Reads the committed artifacts under results/distance/ladder/ (skips if absent so the
suite stays runnable before the ladder has been computed).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from probing.config import OUT_DIR
from probing.distance_ladder import (
    ANCHOR_TOL_PP,
    ANCHOR_VERSION,
    ANCHORS,
    GAIN_VERSIONS,
    SOURCES,
    TARGETS_FAR,
    load_gains,
)

LADDER = OUT_DIR / "distance" / "ladder"
MAT = LADDER / "distance_matrix_7x7.json"

needs_artifacts = pytest.mark.skipif(not MAT.exists(),
                                     reason="ladder artifacts not computed yet")


# ---------------------------------------------------------------- gain-side gates
def test_anchor_values_match_v3_rolling_source_selected():
    """GATE: the three published far-cell values match ANCHOR_VERSION + source_selected
    to within 1pp. (They do NOT match extended_v2 — that mislabel is why this gate is
    pointed at extended_v3_rolling.)"""
    g = load_gains(ANCHOR_VERSION)
    for (src, tgt), expected in ANCHORS.items():
        row = g[(g.source == src) & (g.target == tgt)]
        assert len(row) == 1, f"{src}->{tgt}: expected exactly 1 row, got {len(row)}"
        got = float(row.gain.iloc[0])
        assert abs(got - expected) <= ANCHOR_TOL_PP, (
            f"{src}->{tgt}: read {got:+.4f}, anchor {expected:+.1f}, "
            f"|d|={abs(got - expected):.3f}pp > {ANCHOR_TOL_PP}pp")


@pytest.mark.parametrize("version", GAIN_VERSIONS)
def test_join_completeness_28_rows_no_nan(version):
    """GATE: exactly 28 cells (16 near + 12 far), no NaN gains, no duplicate cells."""
    g = load_gains(version)
    assert len(g) == 28, f"{version}: {len(g)} rows, expected 28"
    assert (g.tier == "near").sum() == 16 and (g.tier == "far").sum() == 12
    assert g.gain.notna().all(), f"{version}: NaN gains present"
    assert not g.duplicated(subset=["source", "target"]).any()
    assert set(g[g.tier == "far"].target) == set(TARGETS_FAR)
    assert set(g.source) == set(SOURCES)


# ------------------------------------------------------------ distance-side gates
@needs_artifacts
def test_distance_matrix_self_zero_and_symmetric():
    """Sanity anchor: distance(x, x) == 0, and the matrix is symmetric."""
    m = json.loads(MAT.read_text())["seed0"]
    M = np.asarray(m["matrix"])
    assert np.allclose(np.diag(M), 0.0, atol=1e-12), "self-distance must be exactly 0"
    assert np.allclose(M, M.T, atol=1e-12), "matrix must be symmetric"
    i = m["datasets"].index("monash_electricity_hourly")
    assert M[i, i] == 0.0


@needs_artifacts
def test_seed_stability_spearman_above_095():
    """GATE: seed=1 rerun preserves the pairwise ordering (Spearman > 0.95 vs seed=0)."""
    blob = json.loads(MAT.read_text())
    assert blob["seed_stability_spearman"] > 0.95, (
        f"seed stability rho={blob['seed_stability_spearman']:.4f} <= 0.95")


@needs_artifacts
def test_electricity_closer_to_uber_than_to_coastal():
    """Sanity anchor: d(electricity, uber) < d(electricity, coastal_ts).

    A failure here is reported as a FINDING (two hourly demand-style series ought to be
    nearer each other than either is to an oceanographic T/S record), not silently fixed."""
    m = json.loads(MAT.read_text())["seed0"]
    names, M = m["datasets"], np.asarray(m["matrix"])
    e = names.index("monash_electricity_hourly")
    d_uber = M[e, names.index("uber_tlc_hourly")]
    d_coast = M[e, names.index("coastal_ts")]
    assert d_uber < d_coast, (
        f"FINDING (not adjusted): d(electricity,uber)={d_uber:.4f} >= "
        f"d(electricity,coastal_ts)={d_coast:.4f}")


@needs_artifacts
@pytest.mark.parametrize("version", GAIN_VERSIONS)
def test_join_csv_matches_matrix_distances(version):
    """The written join CSV's distances agree with the 7x7 matrix."""
    blob = json.loads(MAT.read_text())
    m = blob["seed0"]
    names, M = m["datasets"], np.asarray(m["matrix"])
    j = pd.read_csv(LADDER / f"join_{version}.csv")
    assert len(j) == 28
    for _, r in j.iterrows():
        expect = M[names.index(r.source), names.index(r.target)]
        assert abs(r.distance - expect) < 1e-9, f"{r.source}->{r.target} distance mismatch"


@needs_artifacts
def test_boom_restricted_to_first_200():
    """BOOM must use exactly the first 200 manifest entries (only those shards are staged)."""
    m = json.loads(MAT.read_text())["seed0"]
    assert m["boom_first_n"] == 200
    assert m["n_raw_series"]["boom_hourly"] == 200


@needs_artifacts
def test_coastal_is_48_univariate_series():
    """Manifest preprocessing: 24 stations x (TEMP, PSAL) = 48 univariate series."""
    m = json.loads(MAT.read_text())["seed0"]
    assert m["n_raw_series"]["coastal_ts"] == 48, (
        f"coastal_ts raw series = {m['n_raw_series']['coastal_ts']}, expected 48 "
        "(24 stations x TEMP+PSAL)")
