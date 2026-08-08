"""Gates for the series-level bootstrap (pure numpy; no caches, no model)."""

import numpy as np
import pytest

from probing.repr_metrics import effective_rank, matrix_entropy
from probing.repr_metrics_bootstrap import _metrics_from_gram_eigs


def _boot_effrank_ci(Z, B=500, seed=0):
    """Series-level bootstrap of effective_rank via the Gram path (rows = series)."""
    G = Z @ Z.T
    n = Z.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, size=(B, n))
    vals = np.empty(B)
    for b in range(B):
        i = idx[b]
        vals[b] = _metrics_from_gram_eigs(np.linalg.eigvalsh(G[np.ix_(i, i)])[::-1])[1]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), vals


def test_bootstrap_ci_contains_true_value_fixed_rank():
    """GATE: bootstrap of synthetic FIXED-RANK matrices gives a CI containing the true value.

    Uses the PROMPT-LEVEL estimator (mean of per-series effective ranks), which is a plain
    sample mean and therefore unbiased under series resampling. Every synthetic series has
    a flat rank-r spectrum, so its true effective rank is exactly r and the true mean is r.
    """
    rng = np.random.default_rng(0)
    n_series, n_patches, d, r = 200, 32, 64, 8
    true_value = float(r)

    per_series = []
    for _ in range(n_series):
        U, _ = np.linalg.qr(rng.standard_normal((n_patches, r)))   # orthonormal columns
        V, _ = np.linalg.qr(rng.standard_normal((d, r)))
        per_series.append(effective_rank(U @ V.T))                  # flat spectrum -> exactly r
    per_series = np.array(per_series)
    assert per_series == pytest.approx(true_value, rel=1e-6)

    B = 2000
    idx = np.random.default_rng(0).integers(0, n_series, size=(B, n_series))
    boot = per_series[idx].mean(axis=1)
    lo, hi = np.percentile(boot, 2.5), np.percentile(boot, 97.5)
    assert lo <= true_value <= hi, f"CI [{lo:.6f}, {hi:.6f}] must contain true {true_value}"
    assert boot.mean() == pytest.approx(true_value, rel=1e-6)       # unbiased


def test_dataset_level_effrank_is_downward_biased_under_resampling():
    """Documented property (NOT a bug): the dataset-level effective rank is *downward
    biased* under bootstrap row-resampling, because duplicated series rows reduce the
    number of independent directions. Magnitude scales with rank/n_series — negligible
    when the spectrum is concentrated, up to a few percent when it is flat.

    This is why the run-level gate compares the bootstrap mean to the point estimate with
    a 5% tolerance rather than expecting exact agreement."""
    rng = np.random.default_rng(0)
    n, d = 200, 64
    # concentrated (low effective rank): bias should be small
    V, _ = np.linalg.qr(rng.standard_normal((d, 8)))
    Z_low = rng.standard_normal((n, 8)) @ V.T
    _, _, vals_low = _boot_effrank_ci(Z_low, B=300)
    bias_low = (vals_low.mean() - effective_rank(Z_low)) / effective_rank(Z_low)

    # flat spectrum (maximal effective rank): bias is larger, still bounded and NEGATIVE
    Q, _ = np.linalg.qr(rng.standard_normal((120, 120)))
    W, _ = np.linalg.qr(rng.standard_normal((40, 40)))
    Z_flat = Q[:, :40] @ W.T
    _, _, vals_flat = _boot_effrank_ci(Z_flat, B=300)
    bias_flat = (vals_flat.mean() - effective_rank(Z_flat)) / effective_rank(Z_flat)

    assert bias_low < 0 and bias_flat < 0, "bias direction must be downward"
    assert abs(bias_low) < 0.02, f"concentrated-spectrum bias should be small, got {bias_low:.4f}"
    assert abs(bias_low) < abs(bias_flat), "bias must grow with spectral flatness"
    assert abs(bias_flat) < 0.10, f"even the flat case stays bounded, got {bias_flat:.4f}"


def test_gram_path_matches_committed_metrics():
    """The Gram shortcut must reproduce matrix_entropy / effective_rank exactly."""
    rng = np.random.default_rng(2)
    for shape in ((50, 80), (200, 768), (32, 768)):
        Z = rng.standard_normal(shape) @ np.diag(
            np.linspace(4.0, 0.05, shape[1]))
        lam = np.linalg.eigvalsh(Z @ Z.T)[::-1]
        s1g, erg = _metrics_from_gram_eigs(lam)
        assert s1g == pytest.approx(matrix_entropy(Z), abs=1e-8)
        assert erg == pytest.approx(effective_rank(Z), rel=1e-8)


def test_paired_draws_are_identical_across_layers():
    """One index array reused across layers -> differences are paired."""
    n, B = 200, 100
    idx = np.random.default_rng(0).integers(0, n, size=(B, n))
    idx2 = np.random.default_rng(0).integers(0, n, size=(B, n))
    assert np.array_equal(idx, idx2)            # same seed -> reproducible
    # a paired difference of a constant offset has zero variance
    a = np.arange(n, dtype=float)
    b = a + 5.0
    diff = b[idx].mean(axis=1) - a[idx].mean(axis=1)
    assert np.allclose(diff, 5.0)


def test_duplicates_are_kept():
    """Resampling with replacement must duplicate rows (not deduplicate)."""
    n, B = 200, 50
    idx = np.random.default_rng(0).integers(0, n, size=(B, n))
    uniq = np.array([len(np.unique(idx[b])) for b in range(B)])
    assert uniq.max() < n, "with replacement, some rows must repeat"
    assert uniq.mean() / n == pytest.approx(1 - 1 / np.e, abs=0.05)   # ~63.2% distinct
