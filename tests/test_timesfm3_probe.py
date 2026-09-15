"""Contract tests for the TimesFM-3 independent-causal-prefix probe.

    python -m tests.test_timesfm3_probe              # no model, no GPU -- login-node safe
    python -m tests.test_timesfm3_probe --with-model # + the TimesFM-3 integration checks

Tests 1-6, 11, 12 need no model. Tests 7-10 need the checkpoint and are skipped without
--with-model. NOTHING here is weakened to make a run pass: a failure means the geometry or
the preprocessing assumption is wrong and must be investigated.
"""

from __future__ import annotations

import sys

import numpy as np
import torch

from probing.timesfm3 import (INPUT_PATCH_LEN, LAYER_NAMES, LAST_LAYER, MODEL_DIMS,
                              NUM_LAYERS, OUTPUT_PATCH_LEN, SIGMA_EPS, PrefixGeometry,
                              build_prefix_targets, denormalize, normalize_target,
                              prefix_detrend_params, prefix_trend, raw_future_from_arcsinh,
                              safe_sigma)
from probing.timesfm3_probes import (MEDIAN_QUANTILE, Q, apply_probe_origins, flatten_valid,
                                     fit_shared_origin_probe, make_probe, median_pinball,
                                     shared_origin_layerwise)

C, H, P, J, D = 512, 64, 32, 16, 1280


# ---------------------------------------------------------------- 1
def test_target_indexing():
    """j=1: x[1:32]->x[33:96];  j=2: x[1:64]->x[65:128];  j=8: ->x[257:320]; j=16: ->x[513:576]"""
    g = PrefixGeometry(C, H, P)
    x = np.arange(1, C + H + 1)                       # x[k] (0-based) holds the 1-based value
    expect = {1: (32, 33, 96), 2: (64, 65, 128), 8: (256, 257, 320), 16: (512, 513, 576)}
    for j, (plen, t1, t2) in expect.items():
        i = j - 1
        assert int(g.prefix_len[i]) == plen, (j, g.prefix_len[i], plen)
        sl = slice(int(g.target_start[i]), int(g.target_end[i]))
        got = x[sl]
        assert got[0] == t1 and got[-1] == t2 and len(got) == H, (j, got[0], got[-1], len(got))
        pre = x[:int(g.prefix_len[i])]
        assert pre[0] == 1 and pre[-1] == plen, (j, pre[0], pre[-1])
        assert got[0] == plen + 1, f"origin {j}: target must start right after the prefix"
    assert g.span == C + H == 576 and g.J == 16
    print("  1  target indexing: j=1,2,8,16 vs np.arange 1-based ground truth        OK")


# ---------------------------------------------------------------- 2
def test_prefix_lengths_and_readout():
    g = PrefixGeometry(C, H, P)
    for j in range(1, J + 1):
        i = j - 1
        assert int(g.prefix_len[i]) == P * j
        assert int(g.prefix_len[i]) // P == j, "num real context patches must equal j"
        assert int(g.readout_token[i]) == j - 1, "readout must be context-token index j-1"
        assert int(g.readout_token[i]) < j, "readout must be a REAL context patch"
        assert g.n_tokens(j) == j + g.n_horizon_patches
    assert g.n_horizon_patches == 2 and g.K == 1
    assert g.headline_origin_1based == 16
    for bad in (500, 513):
        try:
            PrefixGeometry(bad, H, P)
        except ValueError:
            pass
        else:
            raise AssertionError(f"C={bad} must be rejected")
    try:
        PrefixGeometry(C, 96, P)
    except ValueError:
        pass
    else:
        raise AssertionError("H > output_patch_len must be rejected")
    print("  2  prefix geometry: len 32j, j real patches, readout j-1, 2 placeholders  OK")


# ---------------------------------------------------------------- 3
def test_shared_head_no_cross_origin():
    lin = make_probe(H, "cpu")
    assert isinstance(lin, torch.nn.Linear)
    assert (lin.in_features, lin.out_features) == (D, H), (lin.in_features, lin.out_features)
    X = torch.randn(5, J, D)
    out = apply_probe_origins(lin, X)
    assert out.shape == (5, J, H), out.shape
    X2 = X.clone()
    X2[2, 7] += 10.0                                   # perturb ONE origin of ONE window
    o2 = apply_probe_origins(lin, X2)
    d = (out - o2).abs().amax(dim=-1)                  # (B, J)
    moved = (d > 0).nonzero().tolist()
    assert moved == [[2, 7]], f"only origin 7 of window 2 may change, got {moved}"
    # the same parameter object serves every origin
    ids = {id(lin.weight), id(lin.bias)}
    for j in range(J):
        yj = lin(X[:, j, :])
        assert torch.allclose(yj, out[:, j, :], atol=0), f"origin {j} used different weights"
    assert len(ids) == 2, "there must be exactly one weight and one bias"
    assert sum(1 for _ in lin.parameters()) == 2, "exactly one Linear module per layer"
    assert out.shape[-1] == H == OUTPUT_PATCH_LEN, "each origin predicts its OWN H steps"
    print("  3  shared head: Linear(1280,64), one module, zero cross-origin mixing     OK")


# ---------------------------------------------------------------- 4
def test_q1_tau_half_loss():
    assert Q == 1 and MEDIAN_QUANTILE.tolist() == [0.5]
    rng = np.random.default_rng(0)
    pred = torch.as_tensor(rng.normal(0, 1, (7, H)), dtype=torch.float32)
    targ = torch.as_tensor(rng.normal(0, 1, (7, H)), dtype=torch.float32)
    got = median_pinball(pred, targ).item()
    mae = (targ - pred).abs().mean().item()
    assert abs(got - mae) < 1e-6, (got, mae)           # Chronos-2 Q=1 objective == MAE
    from probing.probes import mean_pinball_loss
    mp = mean_pinball_loss(pred.unsqueeze(1), targ,
                           torch.as_tensor(MEDIAN_QUANTILE)).item()
    assert abs(mp - 0.5 * mae) < 1e-6, (mp, mae)       # reported metric == 0.5*MAE == rho_0.5
    e = targ - pred
    rho = torch.where(e >= 0, 0.5 * e, -0.5 * e).mean().item()
    assert abs(mp - rho) < 1e-6, (mp, rho)             # matches the spec's rho_0.5 formula
    try:
        median_pinball(pred, targ, q=[0.1, 0.5, 0.9])
    except AssertionError:
        pass
    else:
        raise AssertionError("a multi-quantile vector must be rejected at Q=1")
    # prediction shape before flattening is (B, J, H)
    lin = make_probe(H, "cpu")
    o = apply_probe_origins(lin, torch.randn(3, J, D))
    assert o.shape == (3, J, H)
    print("  4  loss: Q=1, tau=0.5, == MAE; reported 0.5*MAE == rho_0.5; (B,16,64)     OK")


# ---------------------------------------------------------------- 5
def test_preprocessing_roundtrip():
    g = PrefixGeometry(C, H, P)
    rng = np.random.default_rng(0)
    t = np.arange(C + H)
    Z = np.stack([5 + 0.09 * t + 2 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, C + H),
                  10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.4, C + H)]).astype(
                      np.float32)
    # mu/sd exactly as the model computes them: stats of each prefix's OWN detrended values
    mu = np.zeros((2, J))
    sd = np.zeros((2, J))
    for i, j in enumerate(g.origins_1based):
        Lp = int(g.prefix_len[i])
        a, b, act = prefix_detrend_params(Z[:, :Lp])
        det = Z[:, :Lp] - prefix_trend(a, b, act, Lp, np.arange(Lp))
        mu[:, i], sd[:, i] = det.mean(axis=1), det.std(axis=1)
    out = build_prefix_targets(Z, mu, sd, g)
    assert out["targets"].shape == (2, J, H) and out["trend"].shape == (2, J, H)
    assert out["valid"].all()
    for i, j in enumerate(g.origins_1based):
        back = denormalize(out["targets"][:, i, :], mu=mu[:, i], sd=sd[:, i],
                           trend=out["trend"][:, i, :])
        want = Z[:, int(g.target_start[i]):int(g.target_end[i])]
        rel = np.abs(back - want).max() / (np.abs(want).mean() + 1e-12)
        assert rel < 1e-5, f"origin {j}: inverse preprocessing off by {rel:.3e}"
    # the fit never sees the future: replace the target region with garbage
    Z2 = Z.copy()
    Z2[:, C:] = 9999.0
    out2 = build_prefix_targets(Z2, mu, sd, g)
    assert np.array_equal(out["a"], out2["a"]) and np.array_equal(out["b"], out2["b"])
    assert np.array_equal(out["active"], out2["active"])
    assert np.array_equal(out["trend"][:, :J - 1], out2["trend"][:, :J - 1])
    # each origin's coefficients come from ITS prefix, not from the full 512 context
    a16, b16, _ = prefix_detrend_params(Z[:, :C])
    assert not np.allclose(out["a"][:, 0], a16), "origin 1 must not reuse the 512-point fit"
    assert np.allclose(out["a"][:, J - 1], a16), "origin 16's fit IS the 512-point fit"
    # degenerate sigma is masked, never silently rescaled
    sd2 = sd.copy()
    sd2[0, 3] = SIGMA_EPS / 10
    assert not build_prefix_targets(Z, mu, sd2, g)["valid"][0, 3]
    assert safe_sigma(np.array([1e-9, 2.0])).tolist() == [1.0, 2.0]
    # raw-future inversion used to rebuild Z from id_data labels
    X = Z[:, :C]
    mu0, s0 = X.astype(np.float64).mean(1), np.maximum(X.astype(np.float64).std(1), 1e-6)
    Yt = np.arcsinh((Z[:, C:] - mu0[:, None]) / s0[:, None]).astype(np.float32)
    assert np.abs(raw_future_from_arcsinh(X, Yt) - Z[:, C:]).max() < 1e-3
    print("  5  preprocessing: per-prefix fit, future-blind, exact inverse round-trip  OK")


# ---------------------------------------------------------------- 6
def test_prefix_time_axis_is_prefix_length():
    """The detrend line is normalized by the PREFIX length, not by C."""
    t = np.arange(C, dtype=np.float64)
    x = (3.0 + 0.05 * t)[None, :]
    for Lp in (32, 128, 512):
        a, b, act = prefix_detrend_params(x[:, :Lp])
        assert act[0]
        assert np.isclose(a[0], 0.05 * Lp, rtol=1e-6), (Lp, a[0], 0.05 * Lp)
        assert np.isclose(b[0], 3.0 + 0.05 * (Lp - 1), rtol=1e-6), (Lp, b[0])
        res = x[0, :Lp] - prefix_trend(a, b, act, Lp, np.arange(Lp))[0]
        assert np.abs(res).max() < 1e-6
        ext = prefix_trend(a, b, act, Lp, np.arange(Lp, Lp + H))[0]
        assert np.isclose(ext[0], 3.0 + 0.05 * Lp, rtol=1e-5), (Lp, ext[0])
    rng = np.random.default_rng(0)
    noisy = (10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 1, C))[None, :]
    assert not prefix_detrend_params(noisy)[2][0]
    assert not prefix_detrend_params(x, enabled=False)[2][0]
    print("  6  detrend axis: slope per PREFIX length, threshold, extrapolation        OK")


# ---------------------------------------------------------------- 11 / 12
def test_flatten_and_headline_only():
    n = 4
    F = np.arange(n * J * 3, dtype=np.float32).reshape(n, J, 3)
    T = np.zeros((n, J, H), np.float32)
    V = np.ones((n, J), bool)
    V[1, 5] = False
    X, Y, idx = flatten_valid(F, T, V)
    assert X.shape == (n * J - 1, 3) and idx.tolist() == [k for k in range(n * J)
                                                          if k != 1 * J + 5]
    assert np.array_equal(X[J], F[1, 0]), "row ordering must be i*J + j"

    rng = np.random.default_rng(0)
    d = 16
    W = rng.normal(0, 0.4, (d, H))
    def mk(m):
        f = rng.normal(0, 1, (m, J, d)).astype(np.float32)
        return f, np.einsum("mjd,dh->mjh", f, W).astype(np.float32), np.ones((m, J), bool)
    Ftr, Ttr, Vtr = mk(90)
    Fte, Tte, Vte = mk(40)
    # pad features to the real probe width so make_probe's Linear(1280, H) applies
    def pad(a):
        z = np.zeros((a.shape[0], J, D), np.float32)
        z[:, :, :a.shape[2]] = a
        return z
    scores, diag = shared_origin_layerwise({0: pad(Ftr)}, Ttr, Vtr, {0: pad(Fte)}, Tte, Vte,
                                           H=H, epochs=60, lr=5e-2, wd_grid=(1e-5, 1e-3),
                                           layers=[0], device="cpu", verbose=False)
    assert diag["headline_origin_idx"] == J - 1, "headline must be origin 16 (index 15)"
    zero = median_pinball(torch.zeros(len(Tte), H),
                          torch.as_tensor(Tte[:, J - 1, :])).item()
    assert scores[0] < 0.5 * zero, (scores[0], zero)
    # the headline number is origin 16 alone, not the all-origin average
    assert np.isclose(diag["test_window_loss"][0].mean(), scores[0], rtol=1e-5)
    assert len(diag["test_window_loss"][0]) == len(Tte)
    assert scores[0] != diag["test_loss_all_origins"][0]
    assert np.isclose(scores[0], diag["test_loss_by_origin"][0][J], rtol=1e-5), \
        "headline score must equal the per-origin entry for origin 16"
    assert set(diag["test_loss_by_origin"][0]) == set(range(1, J + 1))
    assert diag["test_median_pred"][0].shape == (len(Tte), H)
    print(" 11  end-to-end probe smoke: fits on all origins, loss << trivial          OK")
    print(" 12  headline evaluation uses ONLY origin 16 (index 15)                    OK")


def test_full_batch_semantics():
    torch.manual_seed(0)
    X, y = torch.randn(40, D), torch.randn(40, H)
    a = fit_shared_origin_probe(X, y, H, 1e-3, 5, 1e-2, "cpu", batch_size=0)
    b = fit_shared_origin_probe(X, y, H, 1e-3, 5, 1e-2, "cpu", batch_size=1000)
    assert torch.allclose(a.weight, b.weight), "batch_size >= n must equal full batch"
    c = fit_shared_origin_probe(X, y, H, 1e-3, 5, 1e-2, "cpu", batch_size=8)
    assert not torch.allclose(a.weight, c.weight)
    print("      fit: batch_size=0 is full batch (one loss, one backward, one step)   OK")


# ---------------------------------------------------------------- model-backed (7,8,9,10)
def model_tests():
    from probing.timesfm3 import (assert_backbone_frozen, assert_batching_invariant,
                                  assert_causal_prefixes, assert_no_backbone_grads,
                                  extract_prefix_features, get_model,
                                  prefix_vs_fullwindow_diagnostic)
    g = PrefixGeometry(C, H, P)
    model, device = get_model()
    rng = np.random.default_rng(0)
    t = np.arange(C)
    X = np.stack([10 + 3 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, C),
                  5 + 0.09 * t + 2 * np.sin(2 * np.pi * t / 24) + rng.normal(0, 0.3, C)]
                 ).astype(np.float32)

    n_par = assert_backbone_frozen(model)
    print(f"\n  7a backbone frozen: all {n_par} parameter tensors requires_grad=False    OK")

    # a real forward+backward through the backbone into a probe must leave no backbone grad
    for p in model.parameters():
        p.grad = None
    caps = {}
    h = model.transformer_stack.layers[-1].register_forward_hook(
        lambda m, a, o: caps.__setitem__("h", o[0]))
    dev = torch.device(device)
    model.forward({"values": torch.zeros(1, 1, 18, P, device=dev),
                   "masks": torch.zeros(1, 1, 18, P, dtype=torch.bool, device=dev),
                   "patch_is_target": torch.ones(1, 1, 18, dtype=torch.bool, device=dev)})
    h.remove()
    lin = make_probe(H, device)
    loss = median_pinball(lin(caps["h"][:, 0, 15, :]), torch.zeros(1, H, device=dev))
    loss.backward()
    assert_no_backbone_grads(model)
    assert lin.weight.grad is not None and lin.weight.grad.abs().sum() > 0
    for p in model.parameters():
        p.grad = None
    print("  7b no backbone gradients after a probe backward; probe DOES get grads    OK")

    ex = extract_prefix_features(X, geom=g, model=model, device=device, batch_size=2,
                                 feature_dtype=np.float32, progress=False)
    for L in range(NUM_LAYERS):
        assert ex["feats"][L].shape == (2, J, D), (L, ex["feats"][L].shape)
    assert len(LAYER_NAMES) == NUM_LAYERS == 21 and LAYER_NAMES[0] == "Emb"
    assert LAYER_NAMES[LAST_LAYER] == "L20"
    print(f"  8  extraction shapes: Emb,L1..L20 each (2, {J}, {D})                      OK")
    print(f"  9  last-real-context-token selection asserted per batch inside extract   OK")

    err = ex["native_median_err"]
    print(f" 10  L20 -> native head -> tau=0.5 reconstruction: max|d| = "
          f"{err['max_abs']:.3e} (relative {err['relative']:.3e})                OK")

    rep = assert_causal_prefixes(model, g, device, origins=(1, 4, 8, 12, 16))
    worst = max(v["max_abs_dh"] for v in rep.values())
    print(f"  7  STRICT CAUSALITY: perturbing x after each origin moves h by "
          f"max {worst:.3e}   OK")

    w = assert_batching_invariant(model, g, device, n=4)
    print(f"      batching: SEMANTIC (other rows' data changed) = {w['semantic_max_abs']:.3e} "
          f"[must be 0]")
    print(f"                NUMERIC  (batch size changed)       = {w['numeric_max_abs']:.3e} "
          f"({w['numeric_relative']:.2e} rel, float non-determinism)   OK")

    diag = prefix_vs_fullwindow_diagnostic(model, g, device, X, origins=(1, 8, 16))
    print("      prefix-vs-full-window DIAGNOSTIC (differences here are expected):")
    for j, v in diag.items():
        print(f"        origin {j:>2}: rel max diff {v['rel_max_diff']:>9.2%}  "
              f"prefix detrend rate {v['prefix_detrend_rate']:.0%}  "
              f"full-window rate {v['full_detrend_rate']:.0%}")


if __name__ == "__main__":
    print("TimesFM-3 independent-causal-prefix probe contracts")
    for fn in (test_target_indexing, test_prefix_lengths_and_readout,
               test_shared_head_no_cross_origin, test_q1_tau_half_loss,
               test_preprocessing_roundtrip, test_prefix_time_axis_is_prefix_length,
               test_flatten_and_headline_only, test_full_batch_semantics):
        fn()
    if "--with-model" in sys.argv:
        model_tests()
        print("\nall contracts hold (model-backed checks included)")
    else:
        print("\nall model-free contracts hold "
              "(run with --with-model for tests 7-10)")
