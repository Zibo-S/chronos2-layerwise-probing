"""Known-answer tests for probing.repr_metrics (pure-numpy; no model, no data)."""

import math

import numpy as np
import pytest

from probing.repr_metrics import (
    effective_rank,
    matrix_entropy,
    normalized_matrix_entropy,
)


def test_orthonormal_rows_max_entropy_and_full_rank():
    """(a) Z with orthonormal rows: all singular values equal ->
    normalized entropy ~= 1.0 and effective_rank ~= min(N, D)."""
    N, D = 16, 64
    Z = np.eye(N, D)                          # rows are orthonormal e_1..e_N
    assert normalized_matrix_entropy(Z) == pytest.approx(1.0, abs=1e-9)
    assert matrix_entropy(Z) == pytest.approx(math.log(N), abs=1e-9)
    assert effective_rank(Z) == pytest.approx(min(N, D), abs=1e-6)


def test_orthonormal_rows_random_basis():
    """Same known answer under a random orthonormal basis (not axis-aligned)."""
    rng = np.random.default_rng(0)
    N, D = 12, 48
    Q, _ = np.linalg.qr(rng.standard_normal((D, N)))   # (D, N), orthonormal columns
    Z = Q.T                                            # rows orthonormal
    assert normalized_matrix_entropy(Z) == pytest.approx(1.0, abs=1e-9)
    assert effective_rank(Z) == pytest.approx(N, abs=1e-6)


def test_rank1_zero_entropy_unit_rank():
    """(b) rank-1 Z (outer product) -> normalized entropy ~= 0, effective_rank ~= 1."""
    rng = np.random.default_rng(1)
    u, v = rng.standard_normal(20), rng.standard_normal(50)
    Z = np.outer(u, v)
    assert matrix_entropy(Z) == pytest.approx(0.0, abs=1e-9)
    assert normalized_matrix_entropy(Z) == pytest.approx(0.0, abs=1e-9)
    assert effective_rank(Z) == pytest.approx(1.0, abs=1e-6)


def test_effrank_at_least_exp_entropy_random():
    """Provable direction of the theorem gate: exp(S1(Z)) <= EffRank(Z).

    (The spec's literal 'EffRank <= exp(S1)' is reversed: with p ∝ s^2 and q ∝ s,
    p is more concentrated, so H(p) <= H(q). Counterexample: s=(2,1) gives
    EffRank=1.890 > exp(S1)=1.649 — checked explicitly below.)"""
    rng = np.random.default_rng(2)
    for _ in range(5):
        Z = rng.standard_normal((30, 80))
        assert math.exp(matrix_entropy(Z)) <= effective_rank(Z) * (1 + 1e-9)
    # the explicit s=(2,1) counterexample via a diagonal matrix
    Z = np.diag([2.0, 1.0])
    er, e1 = effective_rank(Z), math.exp(matrix_entropy(Z))
    assert er == pytest.approx(1.8899, abs=1e-3)
    assert e1 == pytest.approx(1.6493, abs=1e-3)
    assert e1 < er


def test_eig_guard_drops_noise_floor():
    """Eigenvalues below 1e-12 * max must not perturb the entropy."""
    Z = np.diag([1.0, 1.0, 1e-9])             # lam = (1, 1, 1e-18) -> third dropped
    assert matrix_entropy(Z) == pytest.approx(math.log(2), abs=1e-9)


def test_ragged_series_handled_as_lists():
    """Per-series matrices of different N_patches must work as plain lists —
    metrics are computed per matrix, never on a padded tensor."""
    rng = np.random.default_rng(3)
    mats = [rng.standard_normal((n, 24)) for n in (8, 12, 5)]
    vals_ent = [matrix_entropy(Z) for Z in mats]
    vals_er = [effective_rank(Z) for Z in mats]
    assert all(np.isfinite(vals_ent)) and all(np.isfinite(vals_er))
    # padding WOULD have changed the spectrum: check the 5-row matrix's effrank <= 5
    assert vals_er[2] <= 5 + 1e-9
