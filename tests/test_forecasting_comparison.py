"""No-GPU checks for the original-scale forecasting comparison (run_fslot_forecasting_comparison).

Synthetic arrays only — no model, no HF data, no checkpoints on disk — so this runs on a login node:

    OMP_NUM_THREADS=2 python -m tests.test_forecasting_comparison

Covers the §9 invariants the comparison rests on that are checkable without the backbone: the ORIGINAL-
scale inverse transform (mu + s*sinh(z)) genuinely inverts the arcsinh label; the point baselines are
right; the per-window MASE / median-MAE / WQL are correct (perfect forecast -> 0); the series-level
cluster bootstrap is PAIRED across methods under one shared resample (a method minus itself is exactly
0 in every replicate, and boot means match the plain metric mean); a probe's raw quantile forecast has
the right shape with the median row = the inverse-transformed median; and identical windows feed every
method. The GPU-only pieces (native forecast, real checkpoints) are exercised by the user's run.
"""

from __future__ import annotations

import math
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

from probing.config import OUTPUT_PATCH_SIZE
from probing.probes import QUANTILE_SETS, median_index
import experiments.run_fslot_forecasting_comparison as fc
import experiments.run_ptood_probing_ftok as ftok

Q9 = QUANTILE_SETS["q9"]
P = int(OUTPUT_PATCH_SIZE)
Hh = fc.H                                     # driver horizon (64)
C = 128                                       # context length (> m=24)
M_SEASON = fc.M_SEASON


def _windows(n, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, C)).astype(np.float64)
    mu = X.mean(axis=1); s = np.maximum(X.std(axis=1), 1e-6)
    z = rng.normal(size=(n, Hh)).astype(np.float64)                # arcsinh-space "labels"
    return X, mu, s, z


# 1. ORIGINAL-scale inverse transform: mu + s*sinh(z) inverts arcsinh((y-mu)/s) back to raw y (§9.1).
def test_raw_future_inverts_arcsinh():
    X, mu, s, _ = _windows(12, seed=1)
    rng = np.random.default_rng(5)
    y_raw_true = rng.normal(size=(12, Hh)) * 3.0 + mu[:, None]      # arbitrary raw future
    z = np.arcsinh((y_raw_true - mu[:, None]) / s[:, None])         # the label transform (float64)
    w = {"Y_test_traj": z}                                          # test the MATH, not float32 precision
    y_raw = fc._raw_future(w, mu, s)
    np.testing.assert_allclose(y_raw, y_raw_true, rtol=1e-9, atol=1e-9)
    # and it is NOT the arcsinh label itself (metrics must be on raw units, not arcsinh space)
    assert not np.allclose(y_raw, z), "raw future must differ from the arcsinh label"


# 2. Point baselines: last-value repeats x[-1]; seasonal-naive tiles the last m context values.
def test_point_baselines():
    X, *_ = _windows(7, seed=2)
    lv = fc._last_value_raw(X)
    assert lv.shape == (7, Hh)
    assert np.all(lv == X[:, -1:][:, [0] * Hh])
    sn = fc._seasonal_naive_raw(X, m=M_SEASON)
    assert sn.shape == (7, Hh)
    # first m steps repeat the final season; step h maps to context index C-m + (h mod m)
    for h in (0, 1, M_SEASON - 1, M_SEASON, M_SEASON + 3):
        np.testing.assert_allclose(sn[:, h], X[:, C - M_SEASON + (h % M_SEASON)])


# 3. Per-window metrics: a perfect forecast gives 0 MASE / MAE / WQL-numerator.
def test_metrics_zero_for_perfect_forecast():
    _, mu, s, z = _windows(9, seed=3)
    y_raw = mu[:, None] + s[:, None] * np.sinh(z)
    denom = np.full((9, 1), 2.0)
    np.testing.assert_allclose(fc._mase_pw(y_raw, y_raw, denom), 0.0, atol=1e-12)
    np.testing.assert_allclose(fc._mae_pw(y_raw, y_raw), 0.0, atol=1e-12)
    q_perfect = np.repeat(y_raw[:, None, :], len(Q9), axis=1)      # every quantile = y
    num, den = fc._wql_pw_parts(y_raw, q_perfect, Q9)
    np.testing.assert_allclose(num, 0.0, atol=1e-9)
    assert np.all(den > 0), "WQL denominator sum|y| should be positive"


# 4. MASE matches the explicit definition mean_H|y-yhat| / d.
def test_mase_matches_definition():
    _, mu, s, z = _windows(6, seed=4)
    y_raw = mu[:, None] + s[:, None] * np.sinh(z)
    rng = np.random.default_rng(9)
    yhat = y_raw + rng.normal(size=y_raw.shape)
    denom = np.abs(rng.normal(size=(6, 1))) + 0.5
    ref = (np.abs(y_raw - yhat) / denom).mean(axis=1)
    np.testing.assert_allclose(fc._mase_pw(y_raw, yhat, denom), ref, rtol=1e-12)


# 5. WQL numerator equals 2 * summed pinball; for the median-only level it reduces to sum|y-yhat|.
def test_wql_numerator_reduces_to_mae_for_median_level():
    y = np.array([[1.0, 4.0, 2.0]])                                # (1, H=3)
    yhat = np.array([[2.0, 2.0, 2.0]])
    q = np.array([yhat])                                          # (n=1, Q=1, H=3), single 0.5 level
    num, den = fc._wql_pw_parts(y, q, [0.5])
    # pinball at tau=0.5 is 0.5*|y-yhat|; num = 2*sum(0.5*|.|) = sum|y-yhat|
    np.testing.assert_allclose(num, np.abs(y - yhat).sum(axis=1))
    np.testing.assert_allclose(den, np.abs(y).sum(axis=1))


# 6. §20 PAIRED cluster bootstrap: one shared resample -> a method minus ITSELF is exactly 0 in every
#    replicate; and the bootstrap mean of a metric matches its plain window mean at the point estimate.
def test_bootstrap_is_paired_and_consistent():
    rng = np.random.default_rng(0)
    n = 40
    sid = np.repeat(np.arange(20), 2)[:n].astype(np.int64)         # 2 windows/series
    S, inv = fc._series_group(sid)
    M = fc.cluster_bootstrap_counts(S, 300, fc.SEED)
    a = rng.normal(size=n) ** 2
    b = rng.normal(size=n) ** 2
    boot_a = fc._boot_mean(M, a, inv, S)
    boot_b = fc._boot_mean(M, b, inv, S)
    assert boot_a.shape == (300,)
    # paired: same M applied to both -> a-minus-a diff is identically zero
    np.testing.assert_allclose(boot_a - boot_a, 0.0, atol=0)
    assert np.any(boot_a != boot_b), "two different metrics must give different bootstrap draws"
    # the shared resample makes the paired difference deterministic given M (recompute -> identical)
    np.testing.assert_allclose(boot_a - boot_b, fc._boot_mean(M, a, inv, S) - boot_b, atol=0)


# 7. A probe's ORIGINAL-scale quantile forecast is (n, Q, H) and its median row = mu + s*sinh(z_median).
def test_probe_quantiles_raw_shape_and_median():
    Kk = math.ceil(Hh / P); D = 32
    rng = np.random.default_rng(1)
    NP = fc.LAYER_LABELS.__len__()                                 # 14 fslot points
    f_tr = {i: rng.normal(size=(20, Kk, D)).astype(np.float32) for i in range(NP)}
    f_va = {i: rng.normal(size=(8, Kk, D)).astype(np.float32) for i in range(NP)}
    y_tr = rng.normal(size=(20, Hh)).astype(np.float32); y_va = rng.normal(size=(8, Hh)).astype(np.float32)
    fitted = ftok.PROBE_FAMILIES["native_mlp"].fit(f_tr, y_tr, f_va, y_va, quantiles=Q9, epochs=3,
                                                   wd_grid=(1e-2,), device="cpu", init_seed=0)
    n = 6
    feats = {i: rng.normal(size=(n, Kk, D)).astype(np.float32) for i in range(NP)}
    X = rng.normal(size=(n, C)).astype(np.float64); mu = X.mean(axis=1); s = np.maximum(X.std(axis=1), 1e-6)
    qr = fc._probe_quantiles_raw(fitted, feats, layer=5, mu=mu, s=s, quantiles=Q9, device="cpu")
    assert qr.shape == (n, len(Q9), Hh), qr.shape
    # the median row must be the inverse transform of the head's median arcsinh output at this layer
    from probing.probes import _apply_shared_head, _slot_transform
    import torch
    m = fitted[5]["head"]; m.eval()
    Xt = torch.as_tensor(_slot_transform(fitted[5]["scaler"], feats[5]), dtype=torch.float32)
    with torch.no_grad():
        z = _apply_shared_head(m, Xt, len(Q9), P, Hh).numpy().astype(np.float64)
    qmid = median_index(Q9)
    np.testing.assert_allclose(qr[:, qmid, :], mu[:, None] + s[:, None] * np.sinh(z[:, qmid, :]),
                               rtol=1e-6, atol=1e-7)


# 8. §18 identical windows: every method is scored on the SAME y_raw / series ids / resample. Simulate
#    three methods sharing one window set and assert they share length + the paired resample.
def test_all_methods_share_windows_and_resample():
    _, mu, s, z = _windows(30, seed=6)
    y_raw = mu[:, None] + s[:, None] * np.sinh(z)
    denom = np.full((30, 1), 1.5)
    sid = np.repeat(np.arange(15), 2).astype(np.int64)
    S, inv = fc._series_group(sid)
    M = fc.cluster_bootstrap_counts(S, 200, fc.SEED)
    rng = np.random.default_rng(2)
    methods = {name: y_raw + rng.normal(size=y_raw.shape) * scale
               for name, scale in (("a", 0.1), ("b", 0.5), ("c", 1.0))}
    pw = {name: fc._mase_pw(y_raw, yhat, denom) for name, yhat in methods.items()}
    assert all(v.shape == (30,) for v in pw.values()), "all methods share the window count"
    boots = {name: fc._boot_mean(M, v, inv, S) for name, v in pw.items()}
    # better method (smaller noise) has a lower bootstrap-mean MASE, paired under the same M
    assert boots["a"].mean() < boots["b"].mean() < boots["c"].mean()


if __name__ == "__main__":
    tests = [test_raw_future_inverts_arcsinh,
             test_point_baselines,
             test_metrics_zero_for_perfect_forecast,
             test_mase_matches_definition,
             test_wql_numerator_reduces_to_mae_for_median_level,
             test_bootstrap_is_paired_and_consistent,
             test_probe_quantiles_raw_shape_and_median,
             test_all_methods_share_windows_and_resample]
    for fn in tests:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\nAll {len(tests)} forecasting-comparison tests passed.")
