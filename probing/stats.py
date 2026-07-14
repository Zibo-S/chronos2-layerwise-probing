"""Bootstrap confidence-interval helpers (reusable core).

Verbatim extraction of the CI functions originally defined in ``probe_improve.py`` —
the resampling logic (paired indices, percentile bounds, B and seed) is unchanged. Both
operate only on precomputed per-sample correctness vectors, so they never refit a probe
or re-extract features.
"""

from __future__ import annotations

import numpy as np

from probing.config import SEED, BOOT_B


def bootstrap_ci(correct, B=BOOT_B, rng=None):
    """Test-set bootstrap CI for an accuracy from a correctness vector.

    Returns (point=mean(correct), lo=2.5pct, hi=97.5pct).
    Operates only on the precomputed correctness vector — no refit.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    correct = np.asarray(correct, dtype=np.float64)
    n = correct.size
    idx = rng.integers(0, n, size=(B, n))
    means = correct[idx].mean(axis=1)
    return float(correct.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(correct_a, correct_b, B=BOOT_B, rng=None):
    """Paired test-set bootstrap CI for (acc_a - acc_b).

    The same resampled indices are applied to BOTH correctness vectors, preserving
    their per-sample correlation. Returns (point, lo, hi).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    assert a.size == b.size, "paired_diff_ci needs matched-length vectors"
    n = a.size
    idx = rng.integers(0, n, size=(B, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --------------------------------------------------------------------------- #
# Series-level CLUSTER bootstrap (for overlapping forecasting windows).
# --------------------------------------------------------------------------- #
# The test windows are strongly correlated within a series (stride 64 << span 576), so
# resampling individual windows would understate the CIs. Instead we resample whole
# SERIES with replacement and include ALL of a sampled series' windows. For a window-MEAN
# metric this collapses to a matmul: resampling S series with replacement (uniform) is
# exactly a Multinomial(S trials, uniform) count vector m, and the window-mean under
# duplication is (m . per_series_sum) / (m . per_series_count). Exact, not approximate.

def cluster_bootstrap_counts(n_series, B, seed):
    """(B, n_series) multinomial count matrix = B draws of "sample n_series series with
    replacement". Generate ONCE and reuse across every layer/metric/model so all share
    the same resampled series (required for valid paired differences)."""
    rng = np.random.default_rng(seed)
    return rng.multinomial(n_series, np.full(n_series, 1.0 / n_series), size=B).astype(np.float64)


def cluster_bootstrap_apply(M, per_series_sum, per_series_count):
    """Window-mean of a metric under cluster resampling.

    M                (B, S) : multinomial counts from cluster_bootstrap_counts
    per_series_sum   (S, L) : sum of the per-window metric over each series' windows
    per_series_count (S,)   : number of windows in each series
    Returns          (B, L) : (M @ sum) / (M @ count) -- equal weight per window (a series
                              with more windows contributes proportionally more, matching
                              the reported aggregate).
    """
    return (M @ per_series_sum) / (M @ per_series_count)[:, None]


def ci_bounds(boot, lo=2.5, hi=97.5):
    """95% percentile CI along the replicate axis (axis 0). Returns (lo_arr, hi_arr)."""
    return np.percentile(boot, lo, axis=0), np.percentile(boot, hi, axis=0)
