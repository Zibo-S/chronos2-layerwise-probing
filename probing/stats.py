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
