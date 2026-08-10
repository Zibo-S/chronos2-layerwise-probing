"""No-GPU contract tests for probing.spectral_metrics.

Run: python -m tests.test_spectral_metrics   (pytest-compatible)
"""

from __future__ import annotations

import numpy as np

from probing.spectral_metrics import spectral_metrics, subsample_metrics

RNG = np.random.default_rng(0)


def test_isotropic_high_rank():
    X = RNG.standard_normal((2000, 64))
    m = spectral_metrics(X)
    assert m["effective_rank"] > 0.9 * 64          # i.i.d. Gaussian: near-full effective rank
    assert m["numerical_rank"] == 64
    assert m["n_samples"] == 2000 and m["feature_dim"] == 64


def test_rank_one():
    u = RNG.standard_normal((500, 1))
    v = RNG.standard_normal((1, 32))
    m = spectral_metrics(u @ v)
    assert m["effective_rank"] < 1.01 and m["numerical_rank"] == 1
    assert m["pc1_fraction"] > 0.999


def test_concentration_monotone():
    # progressively concentrating variance into PC1 must decrease effective rank
    base = RNG.standard_normal((1000, 16))
    prev = np.inf
    for scale in (1.0, 4.0, 16.0):
        X = base.copy()
        X[:, 0] *= scale
        m = spectral_metrics(X)
        assert m["effective_rank"] < prev
        prev = m["effective_rank"]
    assert m["pc1_fraction"] > 0.9


def test_scale_invariance():
    X = RNG.standard_normal((300, 20))
    a, b = spectral_metrics(X), spectral_metrics(7.3 * X)
    for k in ("effective_rank", "spectral_entropy", "pc1_fraction", "numerical_rank"):
        assert np.isclose(a[k], b[k]), k


def test_centering_offset_invariance():
    X = RNG.standard_normal((300, 20))
    off = X + 1000.0 * np.ones((1, 20))            # constant row offset removed by centering
    a, b = spectral_metrics(X), spectral_metrics(off)
    assert np.isclose(a["effective_rank"], b["effective_rank"])
    assert np.isclose(a["spectral_entropy"], b["spectral_entropy"])


def test_rank_ceiling_small_n():
    X = RNG.standard_normal((10, 768))             # N-1 = 9 < d
    m = spectral_metrics(X)
    assert m["numerical_rank"] <= 9
    assert m["effective_rank"] <= 9 + 1e-9


def test_torch_and_numpy_agree():
    import torch
    X = RNG.standard_normal((200, 12))
    a = spectral_metrics(X)
    b = spectral_metrics(torch.as_tensor(X, dtype=torch.float32))
    assert abs(a["effective_rank"] - b["effective_rank"]) < 1e-3   # float32 round-trip


def test_normalized_spectrum():
    X = RNG.standard_normal((100, 8))
    m = spectral_metrics(X, return_spectrum=True)
    p = np.asarray(m["spectrum"])
    assert np.isclose(p.sum(), 1.0) and np.all(p[:-1] >= p[1:])    # normalized, descending
    assert np.isclose(m["pc1_fraction"], p[0])


def test_subsample_metrics_no_replacement_and_determinism():
    X = RNG.standard_normal((400, 16))
    d1 = subsample_metrics(X, n_subsamples=20, frac=0.8, seed=0)
    d2 = subsample_metrics(X, n_subsamples=20, frac=0.8, seed=0)
    er = d1["effective_rank"]
    assert len(er["values"]) == 20 and er["subsample_size"] == 320
    assert er["ci"][0] <= er["mean"] <= er["ci"][1]
    assert d1["effective_rank"]["values"] == d2["effective_rank"]["values"]   # deterministic
    # subsampling without replacement at frac<1 can only see distinct rows: every value must
    # stay below the mathematically maximal rank of the subsample
    assert max(er["values"]) <= 16.0 + 1e-9
    try:
        subsample_metrics(X, frac=1.0)
        assert False, "frac=1.0 must raise (subsample must be a strict subset)"
    except ValueError:
        pass


def test_degenerate_inputs():
    m = spectral_metrics(np.ones((50, 4)))         # constant matrix -> zero after centering
    assert m["effective_rank"] == 0.0 and m["numerical_rank"] == 0
    try:
        spectral_metrics(np.zeros((1, 4)))
        assert False, "N=1 must raise"
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} spectral-metric tests passed.")
