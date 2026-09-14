"""Contracts for the UNBIASED (debiased-HSIC) linear-CKA estimator in probing/cka.py.

Synthetic + pure numpy: no GPU, no model, no feature cache, no download — login-node safe.

The load-bearing test is `test_matches_independent_gram_implementation`: the shipped estimator is
the O(d^2) FEATURE-space form, and it is checked against a from-scratch O(n^2) GRAM-space HSIC_1
written directly from Song et al. (2012). Two independent derivations agreeing to 1e-9 is what
makes the exported numbers trustworthy.
"""

from __future__ import annotations

import numpy as np
import pytest

from probing import cka


# --------------------------------------------------------------------------- #
# independent reference: Gram-space HSIC_1 (Song et al. 2012), O(n^2)
# --------------------------------------------------------------------------- #
def _hsic1(K, L):
    n = K.shape[0]
    K, L = K.copy(), L.copy()
    np.fill_diagonal(K, 0.0)
    np.fill_diagonal(L, 0.0)
    return (np.sum(K * L)
            + K.sum() * L.sum() / ((n - 1) * (n - 2))
            - 2.0 / (n - 2) * float(np.sum(K.sum(1) * L.sum(1)))) / (n * (n - 3))


def _cka_unbiased_gram(X, Y):
    K = np.asarray(X, np.float64) @ np.asarray(X, np.float64).T
    L = np.asarray(Y, np.float64) @ np.asarray(Y, np.float64).T
    return _hsic1(K, L) / np.sqrt(_hsic1(K, K) * _hsic1(L, L))


def _stack(rng, n=240, d=64, rank=12, layers=6):
    """A layer stack with real structure: a shared low-rank source drifting layer to layer."""
    Z = rng.standard_normal((n, rank))
    W = rng.standard_normal((rank, d))
    out = []
    for _ in range(layers):
        W = W + 0.4 * rng.standard_normal((rank, d))
        out.append(Z @ W + 0.05 * rng.standard_normal((n, d)))
    return out


# --------------------------------------------------------------------------- #
def test_matches_independent_gram_implementation():
    rng = np.random.default_rng(0)
    for n, d in [(60, 8), (200, 30), (150, 300)]:
        X = rng.standard_normal((n, d))
        Y = rng.standard_normal((n, d)) @ rng.standard_normal((d, d))
        assert cka.linear_cka(X, Y, estimator="unbiased") == pytest.approx(
            _cka_unbiased_gram(X, Y), abs=1e-9)


def test_self_similarity_is_exactly_one():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((100, 20))
    assert cka.linear_cka(X, X, estimator="unbiased") == pytest.approx(1.0, abs=1e-12)


def test_invariant_to_orthogonal_transform_and_isotropic_scale():
    rng = np.random.default_rng(2)
    X, Y = rng.standard_normal((120, 30)), rng.standard_normal((120, 30))
    Q, _ = np.linalg.qr(rng.standard_normal((30, 30)))
    base = cka.linear_cka(X, Y, estimator="unbiased")
    assert cka.linear_cka(X @ Q, Y, estimator="unbiased") == pytest.approx(base, abs=1e-10)
    assert cka.linear_cka(4.2 * X, Y, estimator="unbiased") == pytest.approx(base, abs=1e-10)


def test_unbiased_removes_the_small_sample_inflation():
    """Independent representations: biased is visibly positive and shrinks as O(1/n); unbiased
    sits at ~0 for every n. This is the whole reason the estimator exists."""
    rng = np.random.default_rng(3)
    for n in (50, 400):
        b = [cka.linear_cka(rng.standard_normal((n, 25)), rng.standard_normal((n, 25)))
             for _ in range(40)]
        u = [cka.linear_cka(rng.standard_normal((n, 25)), rng.standard_normal((n, 25)),
                            estimator="unbiased") for _ in range(40)]
        assert np.mean(b) > 5 * abs(np.mean(u))
        assert abs(np.mean(u)) < 0.02
    assert np.mean(b) < 0.1                      # the n=400 biased mean has shrunk


def test_unbiased_can_go_negative():
    """Unbiasedness costs boundedness — an implementation clipping to [0,1] would be wrong."""
    rng = np.random.default_rng(4)
    vals = [cka.linear_cka(rng.standard_normal((40, 10)), rng.standard_normal((40, 10)),
                           estimator="unbiased") for _ in range(60)]
    assert min(vals) < 0.0


def test_matrix_agrees_with_pairwise_and_is_symmetric_with_unit_diagonal():
    rng = np.random.default_rng(5)
    reps = _stack(rng)
    M = cka.cka_matrix(reps, estimator="unbiased")
    assert M.shape == (len(reps), len(reps))
    assert np.allclose(M, M.T, atol=1e-14)
    assert np.allclose(np.diag(M), 1.0, atol=1e-12)
    for i in range(len(reps)):
        for j in range(len(reps)):
            assert M[i, j] == pytest.approx(
                cka.linear_cka(reps[i], reps[j], estimator="unbiased"), abs=1e-12)


def test_rectangular_cross_matrix_supports_the_estimator():
    rng = np.random.default_rng(6)
    a, b = _stack(rng, layers=4), _stack(rng, layers=3)
    M = cka.cka_matrix(a, b, estimator="unbiased")
    assert M.shape == (4, 3)
    assert M[2, 1] == pytest.approx(cka.linear_cka(a[2], b[1], estimator="unbiased"), abs=1e-12)


def test_curve_helpers_thread_the_estimator():
    rng = np.random.default_rng(7)
    a, b = _stack(rng, layers=5), _stack(rng, layers=5)
    d = cka.same_layer_diagonal(a, b, estimator="unbiased")
    r = cka.cka_to_reference(a, estimator="unbiased")
    assert d[3] == pytest.approx(cka.linear_cka(a[3], b[3], estimator="unbiased"), abs=1e-12)
    assert r[-1] == pytest.approx(1.0, abs=1e-12)
    assert not np.allclose(d, cka.same_layer_diagonal(a, b))       # differs from the biased curve


def test_biased_remains_the_default_and_is_byte_identical():
    """The committed pipelines must be untouched by this addition."""
    rng = np.random.default_rng(8)
    reps = _stack(rng)
    assert cka.linear_cka(reps[0], reps[1]) == cka.linear_cka(reps[0], reps[1], estimator="biased")
    assert np.array_equal(cka.cka_matrix(reps), cka.cka_matrix(reps, estimator="biased"))
    assert np.array_equal(cka.same_layer_diagonal(reps, reps),
                          cka.same_layer_diagonal(reps, reps, estimator="biased"))
    # and it is genuinely a different number from the unbiased one
    assert cka.linear_cka(reps[0], reps[1]) != cka.linear_cka(reps[0], reps[1],
                                                              estimator="unbiased")


def test_fails_loud_on_bad_estimator_and_too_few_rows():
    rng = np.random.default_rng(9)
    X, Y = rng.standard_normal((10, 4)), rng.standard_normal((10, 4))
    with pytest.raises(ValueError, match="unknown CKA estimator"):
        cka.linear_cka(X, Y, estimator="debiased")
    with pytest.raises(ValueError, match="unknown CKA estimator"):
        cka.cka_matrix([X, Y], estimator="Unbiased")
    with pytest.raises(ValueError, match="at least 4 rows"):
        cka.linear_cka(X[:3], Y[:3], estimator="unbiased")
    with pytest.raises(ValueError, match="at least 4 rows"):
        cka.cka_matrix([X[:3], Y[:3]], estimator="unbiased")


def test_degenerate_constant_representation_is_nan_not_a_crash():
    rng = np.random.default_rng(10)
    X = rng.standard_normal((50, 6))
    const = np.ones((50, 6))
    assert np.isnan(cka.linear_cka(const, X, estimator="unbiased"))
    M = cka.cka_matrix([X, const], estimator="unbiased")
    assert np.isnan(M[0, 1]) and np.isnan(M[1, 0])


def test_matched_row_guard_still_applies():
    rng = np.random.default_rng(11)
    with pytest.raises(ValueError, match="row-count mismatch"):
        cka.linear_cka(rng.standard_normal((40, 5)), rng.standard_normal((39, 5)),
                       estimator="unbiased")
