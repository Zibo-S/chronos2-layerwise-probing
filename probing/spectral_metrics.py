"""Spectral geometry of representation matrices: effective rank & friends (label-free).

Definition used PROJECT-WIDE (keep consistent):
    Xc  = X - mean(X, axis=0)            # center across examples
    s   = svdvals(Xc)                    # singular values
    lam = s**2                           # covariance-energy spectrum
    p   = lam / lam.sum()                # normalized variance spectrum
    H   = -sum p_i log p_i               # spectral entropy (natural log)
    effective_rank = exp(H)              # Roy & Vetterli (2007)
    pc1_fraction   = p[0]

Independent of Chronos-specific code: takes any (N, d) numpy array or torch tensor.
rank(Xc) <= min(N - 1, d) (centering removes one dimension), so records carry N and d.

Uncertainty: `subsample_metrics` does repeated subsampling WITHOUT replacement (a naive
with-replacement bootstrap duplicates rows, which deterministically deflates rank estimates,
and is deliberately NOT implemented here). The subsample distribution is returned whole so
the CI procedure can change later without recomputation.
"""

from __future__ import annotations

import numpy as np

from probing.config import SEED


def _as_2d_float64(X):
    if hasattr(X, "detach"):                    # torch tensor (avoids importing torch here)
        X = X.detach().cpu().numpy()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] < 2:
        raise ValueError(f"need an (N>=2, d) matrix, got shape {X.shape}")
    return X


def spectral_metrics(X, eps: float = 1e-12, return_spectrum: bool = False) -> dict:
    """Effective rank / spectral entropy / PC1 fraction of one representation matrix.

    numerical_rank counts singular values above max(s) * max(N, d) * float64-machine-eps —
    the numpy.linalg.matrix_rank default tolerance — and is diagnostic only; effective_rank
    is the primary metric. `eps` guards the p*log(p) terms (p < eps contributes 0)."""
    X = _as_2d_float64(X)
    n, d = X.shape
    s = np.linalg.svd(X - X.mean(axis=0), compute_uv=False)
    lam = s ** 2
    total = lam.sum()
    if total <= 0:                              # constant matrix: rank 0, entropy 0 by convention
        p = np.zeros_like(lam)
        H = 0.0
    else:
        p = lam / total
        q = p[p > eps]
        H = float(-(q * np.log(q)).sum())
    out = {
        "effective_rank": float(np.exp(H)) if total > 0 else 0.0,
        "spectral_entropy": H,
        "pc1_fraction": float(p[0]) if p.size else 0.0,
        "n_samples": int(n),
        "feature_dim": int(d),
        "numerical_rank": int((s > s.max() * max(n, d) * np.finfo(np.float64).eps).sum())
        if s.size and s.max() > 0 else 0,
    }
    if return_spectrum:
        out["spectrum"] = p.tolist()
    return out


def subsample_metrics(X, n_subsamples: int = 200, frac: float = 0.8, seed: int = SEED,
                      keys=("effective_rank", "spectral_entropy", "pc1_fraction")) -> dict:
    """Subsampling (WITHOUT replacement) distribution of the spectral metrics.

    Each of `n_subsamples` draws keeps round(frac * N) distinct rows and recomputes the
    metrics. Returns {key: {"values": [...], "mean", "std", "ci": [2.5%, 97.5%]}}. The
    percentile interval describes subsample variability at the reduced size, not a formal
    CI for the full-sample estimate (rank estimates grow with N) — interpret accordingly."""
    X = _as_2d_float64(X)
    n = X.shape[0]
    m = max(2, int(round(frac * n)))
    if m >= n:
        raise ValueError(f"subsample size {m} must be < N={n}; lower frac ({frac})")
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in keys}
    for _ in range(n_subsamples):
        idx = rng.choice(n, size=m, replace=False)
        sm = spectral_metrics(X[idx])
        for k in keys:
            vals[k].append(sm[k])
    out = {}
    for k, v in vals.items():
        a = np.asarray(v, dtype=np.float64)
        out[k] = {"values": a.tolist(), "mean": float(a.mean()), "std": float(a.std()),
                  "ci": [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))],
                  "subsample_size": m, "n_subsamples": int(n_subsamples)}
    return out
