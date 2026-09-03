"""Property tests for the debiased linear CKA (pure numpy; no caches, no model)."""

import numpy as np
import pytest

from probing.repr_metrics_cka import cka_debiased_matrix, hsic1


def _cka(X, Y):
    C = cka_debiased_matrix({"a": X, "b": Y}, ["a", "b"])
    return C[0, 1]


def test_self_cka_is_one():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((50, 20))
    C = cka_debiased_matrix({"a": X}, ["a"])
    assert C[0, 0] == pytest.approx(1.0, abs=1e-12)


def test_symmetry():
    rng = np.random.default_rng(1)
    X, Y = rng.standard_normal((40, 16)), rng.standard_normal((40, 24))
    C = cka_debiased_matrix({"a": X, "b": Y}, ["a", "b"])
    assert abs(C[0, 1] - C[1, 0]) < 1e-12


def test_orthogonal_invariance():
    """CKA(X, XR) = 1 for orthogonal R — the linear kernel XX^T is unchanged."""
    rng = np.random.default_rng(2)
    X = rng.standard_normal((60, 30))
    Q, _ = np.linalg.qr(rng.standard_normal((30, 30)))
    assert _cka(X, X @ Q) == pytest.approx(1.0, abs=1e-9)


def test_isotropic_scale_invariance():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((60, 30))
    assert _cka(X, 7.3 * X) == pytest.approx(1.0, abs=1e-9)


def test_independent_gaussians_near_zero():
    """Debiased CKA of independent representations concentrates near 0 (can be <0)."""
    rng = np.random.default_rng(4)
    vals = [_cka(rng.standard_normal((80, 32)), rng.standard_normal((80, 32)))
            for _ in range(5)]
    assert max(abs(v) for v in vals) < 0.15


def test_hsic1_matches_bruteforce_small():
    """Unbiased HSIC formula vs the O(n^2) elementwise definition on a tiny case."""
    rng = np.random.default_rng(5)
    n = 8
    X, Y = rng.standard_normal((n, 3)), rng.standard_normal((n, 4))
    K, L = X @ X.T, Y @ Y.T
    Kt = K.copy(); np.fill_diagonal(Kt, 0.0)
    Lt = L.copy(); np.fill_diagonal(Lt, 0.0)
    t1 = sum(Kt[i, j] * Lt[i, j] for i in range(n) for j in range(n))
    t2 = Kt.sum() * Lt.sum() / ((n - 1) * (n - 2))
    t3 = 2.0 / (n - 2) * sum(Kt[i, j] * Lt[j, k] for i in range(n)
                             for j in range(n) for k in range(n))
    expect = (t1 + t2 - t3) / (n * (n - 3))
    assert hsic1(K, L) == pytest.approx(expect, rel=1e-10)
