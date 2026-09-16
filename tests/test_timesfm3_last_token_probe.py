"""Contract tests for the TimesFM-3 LAST-CONTEXT-TOKEN probe (C=512 -> H=64, native Q=9).

    python -m tests.test_timesfm3_last_token_probe                # no model, no GPU
    python -m tests.test_timesfm3_last_token_probe --with-model    # + TimesFM-3 integration

The model-free block is login-node safe: synthetic arrays only, torch pinned to 2 threads, a
few seconds. The --with-model block loads the 330M checkpoint and runs forward passes, so it
belongs in salloc/sbatch, never on a login node.

Numbered to the spec's required-tests list:
     1 raw context length == 512                 12 no backbone grads after probe backward
     2 num real context patches == 16            13 probe DOES receive gradients
     3 selected token index == 15                14 preprocessing uses the context only
     4 selected token is real, not a placeholder  15 target inverse round-trip < 1e-4
     5 Emb,L1..L20 -> (B, 1280)                   16 Emb..L20 extraction succeeds
     6 probe is exactly Linear(1280, 64*9)        17 token-15 extraction is correct
     7 raw probe output is (B, 64*9)              18 L20 all-quantile native recon < 1e-4
     8 reshape has the native (H, Q) geometry     19 end-to-end probe smoke fit
     9 native quantile ORDER verified             20 5% tunnel entrance is correct
    10 Q=9 pinball vs a hand-computed example     21 the old shared-origin cache cannot load

NOTHING here is weakened to make a run pass: a failure means a geometry, layout, quantile,
preprocessing or cache assumption is wrong and must be investigated.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)          # never grab every core of a shared login node

from probing.probes import (chronos2_quantile_loss_per_window, mean_pinball_loss,  # noqa: E402
                            median_index, validate_quantiles)
from probing.timesfm3 import CACHE_VERSION as PREFIX_CACHE_VERSION  # noqa: E402
from probing.timesfm3_last_token import (CACHE_VERSION, LAYER_NAMES, LAST_LAYER,  # noqa: E402
                                         MODEL_DIMS, NATIVE_MEDIAN_IDX, NATIVE_QUANTILES,
                                         NUM_LAYERS, NUM_QUANTILES, SIGMA_EPS,
                                         LastTokenGeometry, assert_target_roundtrip,
                                         build_last_token_targets, cache_metadata, cache_root,
                                         context_detrend_params, context_trend, denormalize,
                                         raw_future_from_arcsinh, read_cache, safe_sigma,
                                         select_last_token)
from probing.timesfm3_last_token_probes import (fit_last_token_probe,  # noqa: E402
                                                last_token_layerwise, make_probe,
                                                median_pinball_per_window, pinball_loss,
                                                pinball_loss_per_window, reshape_prediction,
                                                tunnel_entrance, tunnel_entrances)

C, H, P, D, Q, NPATCH, TOK, NTOK = 512, 64, 32, MODEL_DIMS, NUM_QUANTILES, 16, 15, 18


# ------------------------------------------------------------------ 1, 2, 3, 4
def test_geometry():
    g = LastTokenGeometry()
    assert (g.C, g.H, g.P) == (C, H, P), (g.C, g.H, g.P)
    assert g.C == 512, f"raw context length must be 512, got {g.C}"
    assert g.n_real_context_patches == NPATCH == 16, g.n_real_context_patches
    assert g.selected_token_index == TOK == 15, g.selected_token_index
    assert g.selected_token_index < g.n_real_context_patches, "readout must be a REAL patch"
    assert g.n_tokens == NTOK == 18 and g.K == 1 and g.n_horizon_patches == 2
    assert g.selected_token_index == g.C // g.P - 1, "readout is the LAST real context patch"
    # the target is x[513:576] (1-based), i.e. the 64 steps right after the context
    x = np.arange(1, C + H + 1)                       # x[k] (0-based) holds the 1-based value
    tgt = x[g.target_start:g.target_end]
    assert (tgt[0], tgt[-1], len(tgt)) == (513, 576, H), (tgt[0], tgt[-1], len(tgt))
    assert x[:g.C][-1] == 512 and tgt[0] == g.C + 1, "target must start right after the context"
    d = g.as_dict()
    assert d["last_token_only"] is True and d["prefix_extraction"] is False
    assert d["selected_token_index"] == 15 and d["num_real_context_patches"] == 16
    assert np.allclose(d["native_quantiles"], NATIVE_QUANTILES)
    for bad in (500, 513):
        try:
            LastTokenGeometry(bad, H, P)
        except ValueError:
            pass
        else:
            raise AssertionError(f"C={bad} must be rejected (not a multiple of {P})")
    try:
        LastTokenGeometry(C, 96, P)
    except ValueError:
        pass
    else:
        raise AssertionError("H > output_patch_len must be rejected")
    try:
        LastTokenGeometry(256, H, P)                  # a valid layout, but NOT this experiment
    except RuntimeError:
        pass
    else:
        raise AssertionError("strict mode must reject C != 512")
    g2 = LastTokenGeometry(256, H, P, strict=False)
    assert (g2.n_real_context_patches, g2.selected_token_index) == (8, 7)
    print("  1  raw context length == 512                                              OK")
    print("  2  num_real_context_patches == 16                                         OK")
    print("  3  selected_token_index == 15 (last real context patch)                   OK")
    print("  4  token 15 < 16 real patches -> never a horizon placeholder              OK")


# ------------------------------------------------------------------ 5, 17 (model-free half)
def test_token_selection():
    g = LastTokenGeometry()
    B = 3
    h = torch.zeros(B, 1, NTOK, D)
    for t in range(NTOK):
        h[:, 0, t, :] = t + 1                         # token t is identifiable
    got = select_last_token(h, g.selected_token_index,
                            n_real_context_patches=g.n_real_context_patches,
                            n_tokens=g.n_tokens)
    assert got.shape == (B, D), got.shape
    assert torch.equal(got, h[:, 0, TOK, :]), "must return token 15 and nothing else"
    assert float(got[0, 0]) == TOK + 1, float(got[0, 0])
    kw = dict(n_real_context_patches=NPATCH, n_tokens=NTOK)
    for bad_h, why in [(torch.zeros(B, NTOK, D), "rank 3"),
                       (torch.zeros(B, 2, NTOK, D), "two variates"),
                       (torch.zeros(B, 1, 17, D), "17 tokens"),
                       (torch.zeros(B, 1, NTOK, 768), "wrong hidden width")]:
        try:
            select_last_token(bad_h, TOK, **kw)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"{why} must be rejected")
    for bad_tok in (16, 17, -1):
        try:
            select_last_token(h, bad_tok, **kw)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"token {bad_tok} must be rejected (not a real context patch)")
    assert len(LAYER_NAMES) == NUM_LAYERS == 21
    assert LAYER_NAMES[0] == "Emb" and LAYER_NAMES[LAST_LAYER] == "L20" and LAST_LAYER == 20
    print("  5  representation points Emb,L1..L20 select to (B, 1280)                  OK")
    print(" 17  token-15 selection is exact; every other index/shape is REFUSED        OK")


# ------------------------------------------------------------------ 6, 7, 8
def test_probe_shape_and_layout():
    lin = make_probe(H, Q, "cpu")
    assert isinstance(lin, torch.nn.Linear)
    assert (lin.in_features, lin.out_features) == (D, H * Q) == (1280, 576), \
        (lin.in_features, lin.out_features)
    assert sum(1 for _ in lin.parameters()) == 2, "exactly one weight and one bias"
    raw = lin(torch.randn(5, D))
    assert raw.shape == (5, H * Q) == (5, 576), raw.shape
    pred = reshape_prediction(raw, H, Q)
    assert pred.shape == (5, Q, H) == (5, 9, 64), pred.shape
    # THE layout: the native head's flat vector is HORIZON-major, flat index = t*Q + q
    probe_raw = torch.zeros(1, H * Q)
    for t in range(H):
        for qi in range(Q):
            probe_raw[0, t * Q + qi] = 100 * t + qi
    p = reshape_prediction(probe_raw, H, Q)
    for t in (0, 1, H - 1):
        for qi in (0, NATIVE_MEDIAN_IDX, Q - 1):
            assert float(p[0, qi, t]) == 100 * t + qi, (t, qi, float(p[0, qi, t]))
    # a quantile-major reading would give something else, so the two are distinguishable
    assert float(p[0, 1, 0]) != float(probe_raw[0, 1 * H + 0])
    for bad in (torch.randn(5, H * Q + 1), torch.randn(5, Q, H)):
        try:
            reshape_prediction(bad, H, Q)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"raw output {tuple(bad.shape)} must be rejected")
    print("  6  probe is exactly Linear(1280, 64*9) = Linear(1280, 576)                OK")
    print("  7  raw probe output is (B, 576)                                           OK")
    print("  8  reshape -> (B, 64, 9) -> (B, 9, 64), horizon-major like the native head OK")


# ------------------------------------------------------------------ 9
def test_native_quantiles():
    q = np.asarray(NATIVE_QUANTILES, np.float64)
    assert np.allclose(q, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], atol=1e-6), q.tolist()
    assert len(q) == NUM_QUANTILES == Q == 9
    assert np.all(np.diff(q) > 0), "must be strictly increasing"
    assert median_index(q) == NATIVE_MEDIAN_IDX == 4, median_index(q)
    assert q[NATIVE_MEDIAN_IDX] == 0.5, "the median must be EXACTLY 0.5"
    validate_quantiles(q)
    assert H * Q == 576 == 64 * 9, "the native head's width"
    try:
        validate_quantiles(q[::-1])
    except AssertionError:
        pass
    else:
        raise AssertionError("a decreasing quantile vector must be rejected")
    print("  9  native quantiles [0.1..0.9], strictly increasing, median at index 4    OK")


# ------------------------------------------------------------------ 10
def test_pinball_hand_computed():
    q = torch.as_tensor(NATIVE_QUANTILES)
    # (a) fully explicit single-step case, computed by hand from rho_tau(u)=max(tau u,(tau-1)u)
    #     y = 2, yhat_q = 1 for every q  -> u = +1 -> rho = tau
    #     mean over the 9 taus = mean(0.1..0.9) = 0.5
    pred = torch.ones(1, Q, 1)
    y = torch.full((1, 1), 2.0)
    got = float(pinball_loss(pred, y, q))
    assert abs(got - 0.5) < 1e-6, got
    #     y = 0, yhat_q = 1 -> u = -1 -> rho = 1 - tau -> mean = 0.5 as well
    assert abs(float(pinball_loss(pred, torch.zeros(1, 1), q)) - 0.5) < 1e-6
    #     asymmetry at one quantile: tau=0.9, u=+1 -> 0.9 ; u=-1 -> 0.1
    q9 = torch.as_tensor([0.9])
    assert abs(float(pinball_loss(torch.ones(1, 1, 1), torch.full((1, 1), 2.0), q9)) - 0.9) < 1e-6
    assert abs(float(pinball_loss(torch.ones(1, 1, 1), torch.zeros(1, 1), q9)) - 0.1) < 1e-6
    # (b) independent pure-python reference over a random (B, Q, H) case
    rng = np.random.default_rng(0)
    B, Hs = 3, 4
    Pd = rng.normal(0, 1, (B, Q, Hs))
    Yd = rng.normal(0, 1, (B, Hs))
    ref = 0.0
    for b in range(B):
        acc = 0.0
        for qi, tau in enumerate(NATIVE_QUANTILES.tolist()):
            for t in range(Hs):
                u = Yd[b, t] - Pd[b, qi, t]
                acc += max(tau * u, (tau - 1.0) * u)
        ref += acc / (Q * Hs)
    ref /= B
    tp = torch.as_tensor(Pd, dtype=torch.float32)
    ty = torch.as_tensor(Yd, dtype=torch.float32)
    got = float(pinball_loss(tp, ty, q))
    assert abs(got - ref) < 1e-5, (got, ref)
    # per-window reduction, and its mean IS the scalar
    pw = pinball_loss_per_window(tp, ty, q)
    assert pw.shape == (B,) and abs(float(pw.mean()) - got) < 1e-6
    # identical to the project's cross-set metric, and to Chronos-2's loss / (2Q)
    assert abs(float(mean_pinball_loss(tp, ty, q)) - got) < 1e-6
    c2 = chronos2_quantile_loss_per_window(tp, ty, q) / (2 * Q)
    assert torch.allclose(c2, pw, atol=1e-6), (c2[:3], pw[:3])
    # median-only diagnostic == 0.5 * MAE
    mpw = median_pinball_per_window(tp, ty, NATIVE_MEDIAN_IDX)
    mae = (ty - tp[:, NATIVE_MEDIAN_IDX, :]).abs().mean(dim=1)
    assert torch.allclose(mpw, 0.5 * mae, atol=1e-7)
    for bad in [(torch.randn(B, Q + 1, Hs), ty), (torch.randn(B, Q, Hs + 1), ty),
                (torch.randn(B, Hs), ty)]:
        try:
            pinball_loss(bad[0], bad[1], q)
        except RuntimeError:
            pass
        else:
            raise AssertionError("a shape-violating prediction must be rejected")
    print(" 10  Q=9 pinball matches hand-computed values, a python loop, mean_pinball_ OK")
    print("     loss, chronos2_quantile_loss/(2Q) and 0.5*MAE at tau=0.5")


# ------------------------------------------------------------------ 14, 15
def test_preprocessing():
    g = LastTokenGeometry()
    rng = np.random.default_rng(0)
    t = np.arange(C + H)
    Z = np.stack([5 + 0.09 * t + 2 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, C + H),
                  10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4, C + H)]).astype(
                      np.float64)
    # mu/sd exactly as the model computes them at token 15: stats of the DETRENDED context
    a, b, act = context_detrend_params(Z[:, :C])
    det = Z[:, :C] - context_trend(a, b, act, C, np.arange(C))
    mu, sd = det.mean(axis=1), det.std(axis=1)
    out = build_last_token_targets(Z, mu, sd, g)
    assert out["targets"].shape == (2, H) and out["trend"].shape == (2, H)
    assert out["valid"].all() and out["a"].shape == (2,)
    # 15: y_raw -> transformed -> y_raw
    rel = assert_target_roundtrip(Z, out["targets"], out["trend"], mu, sd, g, out["valid"])
    assert rel < 1e-4, rel
    back = denormalize(out["targets"], mu=mu, sd=sd, trend=out["trend"])
    assert np.abs(back - Z[:, C:]).max() / (np.abs(Z[:, C:]).mean()) < 1e-4
    # 14: the fit NEVER sees the future -- replace the target region with garbage
    Z2 = Z.copy()
    Z2[:, C:] = 9999.0
    o2 = build_last_token_targets(Z2, mu, sd, g)
    assert np.array_equal(out["a"], o2["a"]) and np.array_equal(out["b"], o2["b"])
    assert np.array_equal(out["active"], o2["active"])
    assert np.array_equal(out["trend"], o2["trend"]), "the extrapolated trend is future-blind"
    assert not np.allclose(out["targets"], o2["targets"]), "only the TARGET may change"
    # the detrend time axis is the FULL context length, not a shorter window
    a256, _, _ = context_detrend_params(Z[:, :256])
    assert not np.allclose(a, a256), "the fit must use all 512 observed context points"
    assert np.isclose(a[0], 0.09 * C, rtol=2e-2), (a[0], 0.09 * C)   # slope per context length
    assert act[0] and not act[1], "detrending fires on the trending series only"
    # a degenerate sigma is MASKED, never silently rescaled
    sd_bad = sd.copy()
    sd_bad[0] = SIGMA_EPS / 10
    v = build_last_token_targets(Z, mu, sd_bad, g)["valid"]
    assert not v[0] and v[1]
    assert safe_sigma(np.array([1e-9, 2.0])).tolist() == [1.0, 2.0]
    # a failing round-trip must ABORT, not warn
    try:
        assert_target_roundtrip(Z, out["targets"] + 1.0, out["trend"], mu, sd, g, out["valid"])
    except RuntimeError:
        pass
    else:
        raise AssertionError("a broken round-trip must raise")
    # the raw-future inversion the driver uses to rebuild Z from id_data's arcsinh labels
    X = Z[:, :C]
    mu0, s0 = X.mean(1), np.maximum(X.std(1), 1e-6)
    Yt = np.arcsinh((Z[:, C:] - mu0[:, None]) / s0[:, None]).astype(np.float32)
    assert np.abs(raw_future_from_arcsinh(X, Yt) - Z[:, C:]).max() < 1e-3
    print(" 14  preprocessing is fit on x[1:512] ONLY (garbage future -> same a,b,trend) OK")
    print(f" 15  target round-trip raw -> transformed -> raw: relative {rel:.2e} < 1e-4    OK")


# ------------------------------------------------------------------ 20
def test_tunnel_entrance():
    L = [0, 1, 2, 3, 20]
    v = [2.0, 1.5, 1.05, 1.01, 1.0]                  # threshold at 5% = 1.05 -> position 2
    e = tunnel_entrance(v, L, 0.05)
    assert e["position_in_curve"] == 2 and e["layer"] == 2, e
    assert abs(e["threshold"] - 1.05) < 1e-12 and e["reference_layer"] == 20
    assert abs(e["ratio_at_entrance"] - 1.05) < 1e-12
    e2 = tunnel_entrance(v, L, 0.02)                 # threshold 1.02: 1.05 fails, 1.01 passes
    assert e2["position_in_curve"] == 3 and e2["layer"] == 3, e2
    # a layer just ABOVE the tolerance does not open the tunnel
    assert tunnel_entrance([2.0, 1.5, 1.05, 1.04, 1.0], L, 0.02)["layer"] == 20
    # only the last layer qualifies
    e3 = tunnel_entrance([9.0, 8.0, 7.0, 6.0, 1.0], L, 0.05)
    assert e3["position_in_curve"] == 4 and e3["layer"] == 20
    # one-sided: a layer that BEATS the last layer satisfies the criterion
    e4 = tunnel_entrance([0.5, 2.0, 2.0, 2.0, 1.0], L, 0.05)
    assert e4["position_in_curve"] == 0 and e4["layer"] == 0
    assert e4["max_excursion_after_entrance"] > 0.9, e4["max_excursion_after_entrance"]
    # a hump after the entrance is allowed and reported, not silently hidden
    e5 = tunnel_entrance([1.0, 2.0, 1.0, 1.0, 1.0], L, 0.05)
    assert e5["position_in_curve"] == 0 and abs(e5["max_excursion_after_entrance"] - 1.0) < 1e-9
    allt = tunnel_entrances(v, L)
    assert set(allt) == {"0.01", "0.02", "0.05", "0.1"} and allt["0.05"]["layer"] == 2
    for bad_L, bad_v, why in [([0, 1, 2], v, "length mismatch"),
                              ([20, 1, 2, 3, 0], v, "unsorted layers"),
                              ([0, 1, 2, 3, 19], v, "curve not ending at L20")]:
        try:
            tunnel_entrance(bad_v, bad_L, 0.05)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{why} must be rejected")
    print(" 20  5% tunnel entrance = first l with L_val(l) <= 1.05*L_val(L20); 2% too   OK")


# ------------------------------------------------------------------ 19, 13 (model-free half)
def test_smoke_fit():
    """A genuine end-to-end layerwise fit on synthetic features (H=8 keeps it seconds-fast)."""
    Hs, n_tr, n_te, k = 4, 160, 64, 12
    rng = np.random.default_rng(0)
    W = rng.normal(0, 0.5, (k, Hs))

    def mk(m, seed):
        r = np.random.default_rng(seed)
        F = np.zeros((m, D), np.float32)
        F[:, :k] = r.normal(0, 1, (m, k))
        return F, (F[:, :k] @ W).astype(np.float32), np.ones(m, bool)

    Ftr, Ttr, Vtr = mk(n_tr, 1)
    Fte, Tte, Vte = mk(n_te, 2)
    scores, diag = last_token_layerwise({0: Ftr, 20: Ftr}, Ttr, Vtr, {0: Fte, 20: Fte}, Tte, Vte,
                                        H=Hs, epochs=60, lr=5e-2, wd_grid=(1e-3, 1e-1),
                                        layers=[0, 20], device="cpu", verbose=False)
    q = torch.as_tensor(NATIVE_QUANTILES)
    trivial = float(pinball_loss(torch.zeros(n_te, Q, Hs), torch.as_tensor(Tte), q))
    assert scores[0] < 0.6 * trivial, (scores[0], trivial)
    assert set(scores) == {0, 20} and diag["layers"] == [0, 20]
    assert diag["n_val_rows"] == max(1, int(0.2 * n_tr)) == 32, diag["n_val_rows"]
    assert diag["n_train_rows"][0] == n_tr, (diag["n_train_rows"][0], n_tr)
    assert diag["test_q9_window"][0].shape == (n_te,)
    assert abs(diag["test_q9_window"][0].mean() - scores[0]) < 1e-6
    assert diag["test_median_pred"][0].shape == (n_te, Hs)
    assert len(diag["test_per_quantile"][0]) == Q
    assert set(diag["selection"][0]["val_loss_by_wd"]) == {1e-3, 1e-1}
    assert diag["val_loss"][0] == min(diag["selection"][0]["val_loss_by_wd"].values())
    assert diag["quantiles"] == NATIVE_QUANTILES.tolist()
    # the carve splits WINDOWS and is disjoint
    assert not set(diag["carve_train_windows"]) & set(diag["carve_val_windows"])
    assert len(diag["carve_train_windows"]) + len(diag["carve_val_windows"]) == n_tr
    # 13 (model-free half): the probe receives gradients from the Q=9 objective
    lin = make_probe(Hs, Q, "cpu")
    X = torch.as_tensor(Ftr[:16])
    loss = pinball_loss(reshape_prediction(lin(X), Hs, Q), torch.as_tensor(Ttr[:16]), q)
    loss.backward()
    assert lin.weight.grad is not None and float(lin.weight.grad.abs().sum()) > 0
    assert lin.bias.grad is not None and float(lin.bias.grad.abs().sum()) > 0
    # wd selection is a real choice, and features of the wrong rank are refused
    a = fit_last_token_probe(X, torch.as_tensor(Ttr[:16]), q, Hs, 1e-5, 8, 1e-2, "cpu")
    b = fit_last_token_probe(X, torch.as_tensor(Ttr[:16]), q, Hs, 3.0, 8, 1e-2, "cpu")
    assert not torch.allclose(a.weight, b.weight), "weight decay must change the fit"
    try:
        last_token_layerwise({0: Ftr[:, None, :]}, Ttr, Vtr, {0: Fte[:, None, :]}, Tte, Vte,
                             H=Hs, epochs=2, layers=[0], device="cpu", verbose=False,
                             wd_grid=(1e-3,))
    except RuntimeError:
        pass
    else:
        raise AssertionError("rank-3 (shared-origin) features must be refused")
    print(f" 19  end-to-end layerwise fit: test Q9 {scores[0]:.4f} << trivial {trivial:.4f}  OK")
    print(" 13  the probe receives gradients from the Q=9 objective                    OK")


# ------------------------------------------------------------------ 21
def test_cache_isolation():
    g = LastTokenGeometry()
    layers = [0, LAST_LAYER]
    X = np.zeros((4, C), np.float32)
    meta_new = cache_metadata("ds", "test", g, checkpoint="ckpt", detrend=True, layers=layers,
                              seed=0)
    assert meta_new["cache_version"] == CACHE_VERSION == "tfm3-last-token-q9-v1"
    assert CACHE_VERSION != PREFIX_CACHE_VERSION == "tfm3-prefix-v1", \
        "the ablation's cache version must be untouched and different"
    assert meta_new["last_token_only"] is True and meta_new["prefix_extraction"] is False
    assert meta_new["selected_token_index"] == 15 and meta_new["feature_shape_rank"] == 2
    assert meta_new["num_quantiles"] == 9 and meta_new["hidden_dim"] == D

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # the two namespaces cannot even collide on disk
        new_root = cache_root(td, "ds", "test", g, True)
        assert CACHE_VERSION in new_root.name and PREFIX_CACHE_VERSION not in new_root.name

        # (a) a genuine OLD shared-origin cache, metadata and all
        old = td / f"{PREFIX_CACHE_VERSION}__ds__test__C512_H64"
        old.mkdir()
        old_meta = {"cache_version": PREFIX_CACHE_VERSION, "checkpoint": "ckpt",
                    "prefix_extraction": True, "n_origins": 16, "context_len": C, "horizon": H,
                    "dataset": "ds", "split": "test", "seed": 0}
        np.savez(old / "meta.npz", meta=json.dumps(old_meta), ctx_tail=X[:, -8:],
                 mu=np.zeros((4, 16)), sd=np.ones((4, 16)), native=np.zeros((4, H, Q)),
                 native_median_err=json.dumps(None))
        for L in layers:
            np.save(old / f"L{L:02d}.npy", np.zeros((4, 16, D), np.float16))
        try:
            read_cache(old, meta_new, X, layers)
        except RuntimeError as e:
            assert "INCOMPATIBLE" in str(e) and "cache_version" in str(e), str(e)
        else:
            raise AssertionError("the shared-origin cache must NOT load under the new tag")

        # (b) the harder case: correct metadata, but (N, 16, 1280) arrays
        sneaky = td / "sneaky"
        sneaky.mkdir()
        np.savez(sneaky / "meta.npz", meta=json.dumps(meta_new), ctx_tail=X[:, -8:],
                 mu=np.zeros(4), sd=np.ones(4), native=np.zeros((4, H, Q)),
                 native_check=json.dumps(None), dtype_check=json.dumps(None),
                 sorting=json.dumps(None))
        for L in layers:
            np.save(sneaky / f"L{L:02d}.npy", np.zeros((4, 16, D), np.float16))
        try:
            read_cache(sneaky, meta_new, X, layers)
        except RuntimeError as e:
            assert "rank 3" in str(e), str(e)
        else:
            raise AssertionError("a rank-3 feature array must NOT load")

        # (c) the same directory with correct 2-D arrays loads, and every metadata field guards
        for L in layers:
            np.save(sneaky / f"L{L:02d}.npy", np.zeros((4, D), np.float16))
        hit = read_cache(sneaky, meta_new, X, layers)
        assert hit is not None and hit["feats"][0].shape == (4, D) and hit["cache_hit"]
        for field, val in [("detrending", False), ("seed", 1), ("layers", [0]),
                           ("selected_token_index", 14), ("feature_dtype", "float32"),
                           ("checkpoint", "other")]:
            bad = dict(meta_new)
            bad[field] = val
            try:
                read_cache(sneaky, bad, X, layers if field != "layers" else [0])
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"a changed {field} must reject the cache")
        # different windows -> stale
        try:
            read_cache(sneaky, meta_new, X + 1.0, layers)
        except RuntimeError as e:
            assert "stale" in str(e)
        else:
            raise AssertionError("different context windows must reject the cache")
        # nothing there yet -> None (the caller extracts), never a silent empty hit
        assert read_cache(td / "nope", meta_new, X, layers) is None
    print(" 21  the (N,16,1280) shared-origin cache can NEVER load under the new tag    OK")


# ------------------------------------------------------------------ 11, 12 (helpers)
def test_freeze_helpers():
    from probing.timesfm3_last_token import assert_backbone_frozen, assert_no_backbone_grads
    m = torch.nn.Linear(4, 4)
    for p in m.parameters():
        p.requires_grad_(False)
    assert assert_backbone_frozen(m) == 2
    assert_no_backbone_grads(m)
    m.weight.requires_grad_(True)
    try:
        assert_backbone_frozen(m)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a trainable backbone parameter must raise")
    m.weight.requires_grad_(False)
    m.weight.grad = torch.zeros_like(m.weight)
    try:
        assert_no_backbone_grads(m)
    except RuntimeError:
        pass
    else:
        raise AssertionError("a backbone gradient must raise")
    print("     freeze/no-grad guards raise on a live parameter and on a stray grad     OK")


# ------------------------------------------------------------------ model-backed: 5,9,11,12,13,16,17,18
def model_tests():
    from probing.timesfm3_last_token import (assert_backbone_frozen, assert_native_geometry,
                                             assert_native_quantiles, assert_no_backbone_grads,
                                             extract_last_token_features, get_model,
                                             register_layer_hooks)
    g = LastTokenGeometry()
    model, device = get_model()
    rng = np.random.default_rng(0)
    t = np.arange(C)
    X = np.stack([10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, C),
                  5 + 0.09 * t + 2 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, C)]
                 ).astype(np.float32)

    n_par = assert_backbone_frozen(model)
    print(f"\n 11  backbone FROZEN: all {n_par} parameter tensors requires_grad=False      OK")
    qinfo = assert_native_quantiles(model)
    ginfo = assert_native_geometry(model, g)
    print(f"  9  model.quantiles = {qinfo['quantiles']}, median index "
          f"{qinfo['median_index']}, head out {qinfo['output_head_out_features']}       OK")
    print(f"      decode()'s own forecast index for (C=512, H=64): "
          f"{ginfo['native_forecast_indices']} == [15]                                  OK")

    # 12 / 13: a GRAD-ENABLED backbone forward (the autograd graph really passes through
    # the frozen blocks), then a probe backward -- no backbone parameter may collect a grad.
    for p in model.parameters():
        p.grad = None
    caps = {}
    hook = model.transformer_stack.layers[-1].register_forward_hook(
        lambda m, a, o: caps.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    dev = torch.device(device)
    try:
        model.forward({"values": torch.zeros(1, 1, NTOK, P, device=dev),
                       "masks": torch.zeros(1, 1, NTOK, P, dtype=torch.bool, device=dev),
                       "patch_is_target": torch.ones(1, 1, NTOK, dtype=torch.bool, device=dev)})
    finally:
        hook.remove()
    lin = make_probe(H, Q, device)
    h15 = select_last_token(caps["h"], g.selected_token_index,
                            n_real_context_patches=g.n_real_context_patches,
                            n_tokens=g.n_tokens)
    loss = pinball_loss(reshape_prediction(lin(h15.float()), H, Q),
                        torch.zeros(1, H, device=dev),
                        torch.as_tensor(NATIVE_QUANTILES, device=dev))
    loss.backward()
    assert_no_backbone_grads(model)
    assert lin.weight.grad is not None and float(lin.weight.grad.abs().sum()) > 0
    for p in model.parameters():
        p.grad = None
    print(" 12  no backbone gradients after a probe backward (graph ran THROUGH L1..L20) OK")
    print(" 13  the probe DOES get gradients through the frozen states                  OK")

    # an independent token-15 capture from a real decode() pass, for the identity check below
    caps2 = {}
    hs = register_layer_hooks(model, caps2)
    try:
        with torch.no_grad():
            model.decode(target=torch.from_numpy(X).to(device).unsqueeze(1), horizon=H)
        ref_last = caps2[f"L{LAST_LAYER}"].clone()
    finally:
        for h in hs:
            h.remove()

    # 16 / 17 / 18: extraction, token-15 identity, native reconstruction
    ex = extract_last_token_features(X, geom=g, model=model, device=device, batch_size=2,
                                     feature_dtype=np.float32, progress=False)
    for L in range(NUM_LAYERS):
        assert ex["feats"][L].shape == (2, D), (L, ex["feats"][L].shape)
    assert ex["native"].shape == (2, H, Q) and ex["mu"].shape == (2,) and ex["sd"].shape == (2,)
    print(f" 16  extraction: Emb,L1..L20 each (2, {D}); native (2, {H}, {Q})              OK")
    ref15 = ref_last[:, 0, TOK, :].float().cpu().numpy()
    scale = float(ref15.std()) + 1e-12
    d15 = float(np.abs(ex["feats"][LAST_LAYER] - ref15).max())
    d14 = float(np.abs(ex["feats"][LAST_LAYER] - ref_last[:, 0, TOK - 1, :].float().cpu().numpy()
                       ).max())
    d16 = float(np.abs(ex["feats"][LAST_LAYER] - ref_last[:, 0, TOK + 1, :].float().cpu().numpy()
                       ).max())
    assert d15 / scale < 1e-6, (d15, scale)
    assert min(d14, d16) / scale > 1e-2, (d14, d16, scale)     # the check discriminates
    print(f" 17  extracted L20 == independent capture at token 15 ({d15 / scale:.1e} of std; "
          f"tokens 14/16 differ by {min(d14, d16) / scale:.1e})  OK")
    nc = ex["native_check"]
    print(f" 18  L20 -> native head -> ALL {Q} quantiles vs decode(): max|d| = "
          f"{nc['max_abs']:.3e} (relative {nc['relative']:.3e})                 OK")
    print(f"      median-only {nc['median_max_abs']:.3e} | per-quantile max "
          f"{max(nc['per_quantile_max_abs']):.3e} | wrong-layout control "
          f"{nc['transposed_layout_relative']:.2e}")
    print(f"      quantile spread q0.9-q0.1 = {nc['quantile_spread_q90_minus_q10']:.4f} | "
          f"native clip fraction {nc['native_clip_fraction']:.2e} | decode monotone "
          f"{nc['official_monotone_in_quantile_fraction']:.1%}")
    print(f"      quantile sorting knobs: {ex['sorting']}")
    assert nc["relative"] < 1e-4 and nc["transposed_layout_relative"] > 1e-3

    # float16 storage, validated against the float32 extraction
    ex16 = extract_last_token_features(X, geom=g, model=model, device=device, batch_size=2,
                                      feature_dtype=np.float16, progress=False, verify=False)
    worst = max(float(np.abs(ex["feats"][L] - ex16["feats"][L].astype(np.float32)).max()
                      / (ex["feats"][L].std() + 1e-12)) for L in range(NUM_LAYERS))
    print(f"      float16 cache vs float32 extraction: worst {worst:.2e} of the layer std "
          f"({ex['dtype_check']})")
    assert worst < 1e-2, worst

    # the future can never reach the backbone
    try:
        extract_last_token_features(np.zeros((2, C + H), np.float32), geom=g, model=model,
                                   device=device, progress=False)
    except ValueError:
        print("      a (n, C+H) parent window is REFUSED by extraction                     OK")
    else:
        raise AssertionError("extraction must refuse anything but (n, 512) contexts")


if __name__ == "__main__":
    print("TimesFM-3 last-context-token (token 15) Q=9 probe contracts")
    for fn in (test_geometry, test_token_selection, test_probe_shape_and_layout,
               test_native_quantiles, test_pinball_hand_computed, test_preprocessing,
               test_tunnel_entrance, test_smoke_fit, test_cache_isolation,
               test_freeze_helpers):
        fn()
    if "--with-model" in sys.argv:
        model_tests()
        print("\nall contracts hold (model-backed checks included)")
    else:
        print("\nall model-free contracts hold "
              "(run with --with-model for 11,12,13,16,17,18 on a COMPUTE node)")
