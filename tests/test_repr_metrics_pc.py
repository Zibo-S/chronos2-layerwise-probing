"""Sanity gates for the PC-robustness check (pure numpy; no model, no caches)."""

import numpy as np
import pytest

from probing.repr_metrics import effective_rank, normalized_matrix_entropy
from probing.repr_metrics_pc import pc_variance_shares, remove_top_pcs


def _one_dominant_direction(n=60, d=100, factor=100.0, seed=0):
    """Matrix whose centered spectrum has ONE singular value `factor`x the rest."""
    rng = np.random.default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((n, n)))
    V, _ = np.linalg.qr(rng.standard_normal((d, n)))
    s = np.ones(n); s[0] = factor
    Z = (U * s) @ V.T
    return Z - Z.mean(axis=0, keepdims=True)   # centered so PCA sees exactly this spectrum


def test_dominant_direction_ablation_raises_entropy_substantially():
    """GATE: one singular value 100x the rest -> -1PC ablation must raise
    normalized entropy substantially."""
    Z = _one_dominant_direction()
    before = normalized_matrix_entropy(Z)
    after = normalized_matrix_entropy(remove_top_pcs(Z, 1))
    assert before < 0.35, f"dominated matrix should have LOW normalized entropy, got {before:.3f}"
    assert after > 0.90, f"ablated matrix should be near-flat spectrum, got {after:.3f}"
    assert after - before > 0.5, f"substantial rise required, got +{after - before:.3f}"


def test_dominant_direction_pc1_share_and_effrank():
    import math
    from probing.repr_metrics import matrix_entropy
    Z = _one_dominant_direction()
    shares = pc_variance_shares(Z)
    assert shares[0] > 0.99                      # 100^2 / (100^2 + 59) ~ 0.994
    # variance-based exp(S1) collapses to ~1 under one dominant direction...
    assert math.exp(matrix_entropy(Z)) < 2.0
    # ...while effective rank (linear s, q ∝ s) is damped but NOT collapsed (~8.7 << n=60)
    er = effective_rank(Z)
    assert 2.0 < er < 15.0
    assert effective_rank(remove_top_pcs(Z, 1)) > 50.0   # ~n-1 equal directions remain


def test_remove_zero_pcs_is_centering_only():
    rng = np.random.default_rng(1)
    Z = rng.standard_normal((20, 30)) + 5.0      # nonzero mean
    Zc = remove_top_pcs(Z, 0)
    assert np.allclose(Zc.mean(axis=0), 0.0, atol=1e-12)
    assert np.allclose(Zc, Z - Z.mean(axis=0, keepdims=True))


def test_removed_pcs_are_orthogonal_to_result():
    """After projecting out top-k PCs, the result has no component along them."""
    rng = np.random.default_rng(2)
    Z = rng.standard_normal((40, 60)) @ np.diag(np.linspace(5, 0.1, 60))
    Zc = Z - Z.mean(axis=0, keepdims=True)
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    Z2 = remove_top_pcs(Z, 2)
    assert np.abs(Z2 @ Vt[:2].T).max() < 1e-9
    assert pc_variance_shares(Z2 + 0.0)[0] < 1.0  # still a valid spectrum


def test_float64_pipeline():
    Z32 = np.random.default_rng(3).standard_normal((10, 12)).astype(np.float32)
    out = remove_top_pcs(Z32, 1)
    assert out.dtype == np.float64
