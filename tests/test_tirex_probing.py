"""Contract tests for the TiRex layer-wise probe (C=512 -> H=64, Q=1 shared patch head).

    python -m tests.test_tirex_probing                 # no model, no GPU, no network
    python -m tests.test_tirex_probing --with-model    # + the real NX-AI/TiRex checkpoint

The model-free block is login-node safe: synthetic arrays only, torch pinned to 2 threads, a
few seconds. The --with-model block loads the checkpoint and runs forward passes, so it belongs
in salloc/sbatch, never on a login node.

Numbered to the spec's required-tests list:
     1 checkpoint geometry discovery         11 ONE shared probe across both positions
     2 C=512 patch count                     12 concatenated probe output is exactly (B, 64)
     3 H=64 forecast-patch count             13 Q=1 only (a 9-quantile head is refused)
     4 exact forecast-producing indices      14 train-only feature standardization
     5 native-head manual reconstruction     15 validation/test separation
     6 input/target indexing                 16 tau=0.5 objective sanity (== 0.5 * MAE)
     7 normalization round trip              17 CKA diagonal / symmetry sanity
     8 no target leakage                     18 effective-rank synthetic sanity
     9 Emb/L1..L12/L12+RMS hook locations    19 extraction determinism (bitwise at fixed batch size; measured across batch sizes)
    10 hidden-state dimensions               20 native parameters stay frozen (+ no grads)

Additional contracts beyond the required list:
    21 geometry REFUSES incompatible configurations   24 feature cache refuses a foreign config
    22 the depth convention (block vs position)       25 CKA rows == probe rows, same order
    23 the tunnel reuses probing.tunnel verbatim      26 raw-vs-standardized guard for geometry
    27 the (Emb, masked-future) readout is constant across windows -- detected, not hidden
    28 the headline geometry is position-0-only at ALL depths (constant N), companion from L1
    29 the two_pass layout, derived per pass from the package's own context construction
    30 what the rollout appends is MISSING values, never predicted ones  [model-backed: 31]
    31 the two_pass native-head identity, per-pass states and per-pass normalization
    32 device hygiene: a CPU/numpy context against a model on any device (GPU regression guard)

NOTHING here is weakened to make a run pass: a failure means a geometry, indexing, layout,
normalization or cache assumption is wrong and must be investigated.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)          # never grab every core of a shared login node

from probing import cka as cka_mod                                          # noqa: E402
from probing.config import SEED                                             # noqa: E402
from probing.spectral_metrics import spectral_metrics                       # noqa: E402
from probing.tirex_geometry import (CKA_HEADLINE_VARIANT, DEGENERACY_CAVEAT,  # noqa: E402
                                    EMB_POINT, EMB_POLICY, ERANK_HEADLINE_VARIANT,
                                    GEOMETRY_VARIANTS, VARIANT_ALL_FROM_L1, VARIANT_POS0,
                                    assert_not_standardized, cka_layer_matrix, cka_null_floor,
                                    degenerate_readouts, effective_rank_curve,
                                    geometry_for_split, rep_matrices, rep_matrix, variant_spec)
from probing.tirex_model import (CACHE_VERSION, CONTEXT_LEN, EXPECT_QUANTILES,  # noqa: E402
                                 HORIZON, MODEL_DIMS, NUM_BLOCKS, NUM_POINTS,
                                 REP_NAMES, REP_POINTS, ROLLOUT_MODES, ROLLOUT_SINGLE,
                                 ROLLOUT_STEPS, ROLLOUT_TWO, SCALE_EPS,
                                 TiRexGeometry, assert_target_roundtrip, build_targets,
                                 cache_metadata, cache_root, check_rollout_mode, denormalize,
                                 normalize_raw, read_cache, rep_depth_table, rollout_contexts)
from probing.tirex_probes import (EPOCHS, QUANTILES_Q1, TAU, WD_GRID_TIREX,  # noqa: E402
                                  constant_forecast_floor, fit_shared_patch_probe,
                                  layerwise_probe, make_shared_patch_probe, per_patch_losses,
                                  per_window_loss, pinball_tau_half, predict, probe_forward,
                                  representation_norms, tau_half_is_half_mae,
                                  tunnel_entrance, tunnel_entrances)
from probing.tunnel import TUNNEL_TOL, tunnel_start                          # noqa: E402

C, H, P, D = CONTEXT_LEN, HORIZON, 32, MODEL_DIMS
TRAIN_CTX = 2048
GEOM = TiRexGeometry(C=C, H=H, input_patch=P, output_patch=P, train_ctx_len=TRAIN_CTX,
                     num_quantiles=9, median_index=4)


# --------------------------------------------------------------------------- #
# a faithful, minimal stand-in for TiRex's preprocessing -- TEST FIXTURE ONLY.
# It mirrors TiRexZero._adjust_context_length + PatchedTokenizer exactly (see
# tirex/models/tirex.py and tirex/models/patcher.py). The --with-model block re-runs the SAME
# assertions against the real checkpoint, so this stub can never be the only evidence.
# --------------------------------------------------------------------------- #
class _StubScalerState:
    def __init__(self, loc, scale):
        self.loc, self.scale = loc, scale


class _StubTokenizer:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def input_transform(self, data):
        loc = torch.nan_to_num(torch.nanmean(data, dim=-1, keepdim=True), nan=0.0)
        scale = torch.nan_to_num(torch.nanmean((data - loc).square(), dim=-1, keepdim=True).sqrt(),
                                 nan=1.0)
        scale = torch.where(scale == 0, torch.abs(loc) + SCALE_EPS, scale)
        patched = ((data - loc) / scale).unfold(-1, self.patch_size, self.patch_size)
        return patched, _StubScalerState(loc, scale)


class _StubModel:
    def __init__(self, patch=P, train_ctx_len=TRAIN_CTX):
        self.tokenizer = _StubTokenizer(patch)
        self._tcl = train_ctx_len

    def _adjust_context_length(self, context, min_context, max_context):
        pad_len = 0
        if context.shape[-1] > max_context:
            context = context[..., -max_context:]
        if context.shape[-1] < min_context:
            pad_len = min_context - context.shape[-1]
            pad = torch.full((context.shape[0], pad_len), float("nan"), dtype=context.dtype)
            context = torch.cat((pad, context), dim=1)
        return context, pad_len


STUB = _StubModel()


def _series(n, seed=0, span=C + H):
    rng = np.random.default_rng(seed)
    t = np.arange(span, dtype=np.float32)
    return np.stack([10 + 3 * np.sin(2 * np.pi * t / 24) + 0.02 * t
                     + rng.normal(0, 0.5, span).astype(np.float32) for _ in range(n)])


# ------------------------------------------------------------------ 2, 3, 4, 10, 21
def test_geometry():
    g = GEOM
    # 2 -- the C=512 patch count, and the pad that the paper's "16 patches" assumption misses
    assert g.n_real_context_patches == C // P == 16, g.n_real_context_patches
    assert g.pad_len == TRAIN_CTX - C == 1536 and g.n_pad_patches == 48, (g.pad_len, g.n_pad_patches)
    assert g.n_context_tokens == TRAIN_CTX // P == 64, g.n_context_tokens
    print(f"  2  C={C} -> {g.n_real_context_patches} real patches + {g.n_pad_patches} NaN-pad "
          f"patches = {g.n_context_tokens} context tokens                OK")

    # 3 -- the H=64 forecast-patch count and the single-pass token total
    assert g.n_forecast_patches == H // P == 2, g.n_forecast_patches
    assert g.rollout_steps == ROLLOUT_STEPS == 2, g.rollout_steps
    assert g.n_tokens == g.n_context_tokens + g.n_forecast_patches - 1 == 65, g.n_tokens
    print(f"  3  H={H} -> {g.n_forecast_patches} forecast patches, {g.n_tokens} tokens in the "
          f"single accelerated pass            OK")

    # 4 -- the exact readout indices, DERIVED, never written down
    assert g.readout_indices == (63, 64), g.readout_indices
    assert g.first_readout == g.n_context_tokens - 1 == 63
    assert g.masked_readouts == (64,)
    assert g.first_readout < g.n_context_tokens <= g.readout_indices[-1] + 1
    assert g.target_slice(0) == slice(0, 32) and g.target_slice(1) == slice(32, 64)
    print(f"  4  readout tokens {g.readout_indices}: {g.first_readout} = last REAL context patch, "
          f"{g.masked_readouts[0]} = masked future    OK")

    # 10 -- hidden-state dimensions
    assert MODEL_DIMS == 512 and NUM_BLOCKS == 12 and NUM_POINTS == 14
    assert g.as_dict()["model_dims"] == 512
    print(f" 10  d={MODEL_DIMS}, {NUM_BLOCKS} blocks, {NUM_POINTS} representation points   OK")

    # 21 -- incompatible configurations RAISE instead of being silently repaired
    bad = [
        ("H not a multiple of the output patch (would silently truncate)",
         dict(C=C, H=48, input_patch=P, output_patch=P, train_ctx_len=TRAIN_CTX)),
        ("C not a multiple of the patch size",
         dict(C=500, H=H, input_patch=P, output_patch=P, train_ctx_len=TRAIN_CTX)),
        ("C longer than train_ctx_len (would be left-truncated)",
         dict(C=4096, H=H, input_patch=P, output_patch=P, train_ctx_len=TRAIN_CTX)),
        ("input_patch != output_patch",
         dict(C=C, H=H, input_patch=P, output_patch=16, train_ctx_len=TRAIN_CTX)),
    ]
    for why, kw in bad:
        try:
            TiRexGeometry(num_quantiles=9, median_index=4, **kw)
        except ValueError:
            pass
        else:
            raise AssertionError(f"geometry accepted an invalid configuration: {why}")
    print(f" 21  {len(bad)} invalid configurations REFUSED (no silent truncation/padding)  OK")


# ------------------------------------------------------------------ 22
def test_depth_convention():
    tbl = rep_depth_table()
    assert [t["name"] for t in tbl] == list(REP_NAMES)
    assert REP_NAMES[0] == "Emb" and REP_NAMES[-1] == f"L{NUM_BLOCKS}+RMS"
    assert [t["block_index"] for t in tbl] == list(range(NUM_BLOCKS + 1)) + [NUM_BLOCKS]
    assert [t["position_index"] for t in tbl] == list(range(NUM_POINTS))
    # relative_depth: block-based, cross-model; L12 and L12+RMS deliberately tie at 1.0
    rd = [t["relative_depth"] for t in tbl]
    assert rd[0] == 0.0 and rd[-1] == 1.0 and rd[-2] == 1.0, rd
    assert all(b <= a for a, b in zip(rd[1:], rd[:-1])), "relative_depth must be non-decreasing"
    # relative_position: strictly monotone, safe as a plotting axis
    rp = [t["relative_position"] for t in tbl]
    assert rp[0] == 0.0 and rp[-1] == 1.0 and all(b < a for a, b in zip(rp[1:], rp[:-1]))
    kinds = [t["kind"] for t in tbl]
    assert kinds[0] == "embedding" and kinds[-1] == "final_norm" and set(kinds[1:-1]) == {"block"}
    assert sum(t["is_native_readout"] for t in tbl) == 1 and tbl[-1]["is_native_readout"]
    print(f" 22  depth convention: relative_depth in [0,1] (block-based, ties at L12/L12+RMS), "
          f"relative_position strictly monotone   OK")


# ------------------------------------------------------------------ 6, 7, 8 (model-free half)
def test_targets_and_normalization():
    full = _series(6, seed=1)
    ctx, fut = full[:, :C], full[:, C:]

    # 6 -- input/target indexing: the context is x[T-C:T], the target x[T:T+H]
    assert ctx.shape == (6, C) and fut.shape == (6, H)
    assert np.array_equal(ctx[:, -1], full[:, C - 1]) and np.array_equal(fut[:, 0], full[:, C])
    tgt, loc, scale = build_targets(ctx, fut, STUB, GEOM)
    K = GEOM.n_forecast_patches
    assert tgt.shape == (6, H) and loc.shape == (6, K) and scale.shape == (6, K)

    # the normalization convention, READ OFF patcher.py: loc = mean(context), scale = population
    # std(context) -- over the CONTEXT ONLY; the NaN pad is excluded by nanmean
    for k in range(K):
        assert np.allclose(loc[:, k], ctx.mean(axis=1), atol=1e-4), (k, loc[:2, k])
        assert np.allclose(scale[:, k], ctx.std(axis=1), atol=1e-4), (k, scale[:2, k])
    print(f"  6  context=x[T-C:T], target=x[T:T+H]; loc=mean(ctx), scale=popstd(ctx)      OK")

    # 7 -- round trip
    rt = assert_target_roundtrip(fut, tgt, loc, scale, GEOM)
    assert np.allclose(denormalize(tgt, loc, scale, GEOM), fut, atol=1e-3)
    assert np.allclose(normalize_raw(fut, loc, scale, GEOM), tgt, atol=1e-4)
    print(f"  7  denormalize(normalize(y)) == y : max|d| {rt['max_abs_error']:.2e}, "
          f"relative {rt['relative_error']:.2e}                OK")

    # 8 (model-free half) -- changing the FUTURE cannot change loc/scale, so it cannot change
    # the target space, the representations or any statistic the probe sees.
    alt = full.copy()
    alt[:, C:] = alt[:, C:] * 7.0 - 100.0
    _, loc2, scale2 = build_targets(alt[:, :C], alt[:, C:], STUB, GEOM)
    assert np.array_equal(loc, loc2) and np.array_equal(scale, scale2)
    print(f"  8  mutating the future leaves (loc, scale) bit-identical                     OK")

    # degenerate variance (spec F.4): TiRex's own guard scale <- |loc| + 1e-5
    flat = np.stack([np.full(C, 5.0, np.float32), np.zeros(C, np.float32)])
    _, l0, s0 = build_targets(flat, np.zeros((2, H), np.float32), STUB, GEOM)
    assert np.allclose(s0[:, 0], [5.0 + SCALE_EPS, SCALE_EPS], atol=1e-6), s0
    assert np.all(np.isfinite(s0)) and np.all(s0 > 0)
    print(f"  F4 constant context -> scale = |loc| + {SCALE_EPS:g} (TiRex's own guard), finite  OK")

    # a parent (n, C+H) window must be REFUSED, so the future can never be handed to the model
    for bad, why in [(full, "(n, C+H) parent window"), (ctx[:, :-P], "short context")]:
        try:
            build_targets(bad, fut, STUB, GEOM)
        except ValueError:
            pass
        else:
            raise AssertionError(f"build_targets accepted a {why}")


# ------------------------------------------------------------------ 11, 12, 13
def test_probe_structure():
    B, K = 7, GEOM.n_forecast_patches
    probe = make_shared_patch_probe(D, GEOM.output_patch, seed=SEED)

    # 13 -- Q=1 only. The head is (d -> output_patch), NOT (d -> Q*output_patch).
    assert QUANTILES_Q1 == (0.5,) and TAU == 0.5 and len(QUANTILES_Q1) == 1
    assert probe.in_features == D and probe.out_features == GEOM.output_patch == 32
    assert probe.weight.shape == (32, D) and probe.bias is not None
    assert probe.out_features != len(EXPECT_QUANTILES) * GEOM.output_patch
    print(f" 13  Q=1: Linear({D}, {probe.out_features}) = {sum(p.numel() for p in probe.parameters())} "
          f"params (a 9-quantile head would be {9 * 32 * D + 9 * 32})      OK")

    X = torch.randn(B, K, D)
    out = probe_forward(probe, X)

    # 12 -- concatenation is exactly (B, 1, 64), i.e. (B, Q=1, H)
    assert out.shape == (B, 1, H), out.shape
    assert out.squeeze(1).shape == (B, H)
    print(f" 12  probe output {tuple(out.shape)} == (B, Q=1, H={H})                           OK")

    # 11 -- THE SAME weight tensor is applied at BOTH positions. Proven three ways.
    assert len([p for p in probe.parameters()]) == 2, "more than one weight+bias => not shared"
    manual0 = probe(X[:, 0, :])
    manual1 = probe(X[:, 1, :])
    assert torch.equal(out[:, 0, :GEOM.output_patch], manual0)
    assert torch.equal(out[:, 0, GEOM.output_patch:], manual1)
    # swapping the two positions swaps the two output halves EXACTLY -- only possible if one map
    swapped = probe_forward(probe, X.flip(dims=[1]))
    assert torch.equal(swapped[:, 0, :32], out[:, 0, 32:]) and torch.equal(swapped[:, 0, 32:],
                                                                           out[:, 0, :32])
    # feeding the SAME state at both positions gives two IDENTICAL halves
    same = probe_forward(probe, X[:, :1, :].repeat(1, K, 1))
    assert torch.equal(same[:, 0, :32], same[:, 0, 32:])
    print(f" 11  one weight tensor at both readout positions (position swap / duplication "
          f"identities hold exactly)         OK")


# ------------------------------------------------------------------ 16
def test_tau_half_objective():
    B = 64
    g = torch.Generator().manual_seed(0)
    pred = torch.randn(B, 1, H, generator=g)
    tgt = torch.randn(B, H, generator=g)
    rec = tau_half_is_half_mae(pred, tgt)
    assert rec["identity_holds"], rec
    # hand-computed pinball on a tiny example
    p = torch.tensor([[[1.0, 2.0]]])
    t = torch.tensor([[3.0, 0.0]])
    # e = t - p = [2, -2]; rho_.5(e) = max(.5e, -.5e) = .5|e| -> mean = 1.0
    assert abs(float(pinball_tau_half(p, t)) - 1.0) < 1e-7, float(pinball_tau_half(p, t))
    # the constant factor 2 cannot move an argmin: scaling a curve by 2 preserves every ratio,
    # so the tunnel entrance under MAE == the tunnel entrance under tau=0.5 pinball
    curve = np.array([1.30, 1.11, 1.02, 1.05, 1.00])
    assert tunnel_start(curve, TUNNEL_TOL) == tunnel_start(2.0 * curve, TUNNEL_TOL)
    print(f" 16  tau=0.5 pinball == 0.5*MAE (|d| {rec['abs_diff']:.1e}); the factor 2 leaves "
          f"the tunnel entrance unchanged       OK")


# ------------------------------------------------------------------ 14, 15, 19 (fit half)
def test_fit_protocol():
    rng = np.random.default_rng(0)
    K, ntr, nva, nte = GEOM.n_forecast_patches, 600, 262, 262
    # A planted SHARED patch map: ONE R^512 -> R^32 explains BOTH positions, which is exactly the
    # hypothesis class the probe implements. Two properties of the fixture are deliberate and
    # match the real setting rather than being conveniences:
    #   * targets are O(1), because the real targets are (y - loc)/scale. Under an MAE-like
    #     objective AdamW takes ~lr-sized steps, so 300 epochs at lr=1e-2 move a parameter at
    #     most ~3.0 from init; an O(10) target offset would not be reachable in the budget.
    #   * n_rows = 2*600 = 1200 > d = 512, matching the real 2*1394 = 2788 > 512. Below d the
    #     system is underdetermined and the fixture would measure regularization, not learnability.
    W = rng.normal(size=(D, GEOM.output_patch)).astype(np.float32) / np.sqrt(D)

    def make(n, s):
        r = np.random.default_rng(s)
        F = (r.normal(size=(n, K, D)).astype(np.float32) * 3.0 + 7.0)   # NOT standardized
        y = np.concatenate([F[:, k] @ W for k in range(K)], axis=1).astype(np.float32) / 3.0
        return F, y + r.normal(0, 0.05, y.shape).astype(np.float32)

    Ftr, ytr = make(ntr, 1)
    Fva, yva = make(nva, 2)
    Fte, yte = make(nte, 3)

    fit = fit_shared_patch_probe(Ftr, ytr, Fva, yva, out_patch=GEOM.output_patch,
                                 wd_grid=(1e-4, 1e-2), epochs=EPOCHS, device="cpu")

    # 14 -- ONE scaler, fit on the STACKED TRAIN rows only
    sc = fit["scaler"]
    stacked = Ftr.reshape(ntr * K, D)
    assert np.allclose(sc.mean_, stacked.mean(axis=0), atol=1e-4)
    assert np.allclose(sc.scale_, stacked.std(axis=0), atol=1e-4)
    # it must NOT be the val/test statistics, and NOT a per-position statistic
    assert not np.allclose(sc.mean_, Fva.reshape(nva * K, D).mean(axis=0), atol=1e-3)
    assert not np.allclose(sc.mean_, Ftr[:, 0].mean(axis=0), atol=1e-3)
    print(f" 14  StandardScaler fit on the stacked ({ntr * K}, {D}) TRAIN rows only -- not per "
          f"position, not on val/test   OK")

    # 15 -- validation selects, test is never consulted during selection
    assert set(fit["selection"]["val_loss_by_wd"]) == {1e-4, 1e-2}
    assert fit["wd"] == min(fit["selection"]["val_loss_by_wd"],
                            key=fit["selection"]["val_loss_by_wd"].get)
    # scrambling the TEST set cannot change the fitted probe or the chosen wd
    fit2 = fit_shared_patch_probe(Ftr, ytr, Fva, yva, out_patch=GEOM.output_patch,
                                  wd_grid=(1e-4, 1e-2), epochs=EPOCHS, device="cpu")
    assert fit2["wd"] == fit["wd"] and abs(fit2["val_loss"] - fit["val_loss"]) < 1e-9
    assert torch.equal(fit2["probe"].weight, fit["probe"].weight)       # 19: deterministic fit
    print(f" 15  wd chosen on validation (chosen={fit['wd']:g}); refits are bit-identical     OK")

    # the probe actually recovers the planted SHARED map, measured against the NO-INFORMATION
    # floor: the best CONSTANT forecast under the same objective, i.e. the per-step median of
    # the TRAIN targets (the tau=0.5 optimum of a bias-only predictor).
    pred = predict(fit, Fte)
    assert pred.shape == (nte, H)
    floor = constant_forecast_floor(ytr, yte)
    got = 0.5 * float(np.abs(yte - pred).mean())
    assert abs(got - float(per_window_loss(pred, yte).mean())) < 1e-6
    assert got < 0.5 * floor["test_loss"], (got, floor["test_loss"])
    assert fit["train_loss"] < floor["train_loss"], (fit["train_loss"], floor["train_loss"])
    print(f" 19  deterministic fit; TEST loss {got:.4f} << constant-forecast floor "
          f"{floor['test_loss']:.4f} (ratio {got / floor['test_loss']:.3f})   OK")


# ------------------------------------------------------------------ 20 (probe-side half)
def test_no_backbone_grads_during_probe_training():
    """Probe training must not leak gradients into anything but the probe. The backbone is
    represented here by a parameter tensor standing in for a frozen TiRex weight; the extracted
    features are plain numpy, so there is structurally no path back into the model -- this test
    pins that the optimizer only ever sees the probe's two tensors."""
    K = GEOM.n_forecast_patches
    frozen = torch.nn.Linear(D, D)
    for p in frozen.parameters():
        p.requires_grad_(False)
    feats = torch.randn(8, K, D)                       # numpy->tensor: no graph to the backbone
    assert not feats.requires_grad and feats.grad_fn is None
    probe = make_shared_patch_probe(D, GEOM.output_patch, seed=SEED)
    opt = torch.optim.AdamW([{"params": [probe.weight], "weight_decay": 1e-3},
                             {"params": [probe.bias], "weight_decay": 0.0}], lr=1e-2)
    loss = pinball_tau_half(probe_forward(probe, feats), torch.randn(8, H))
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    assert probe.weight.grad is not None and probe.bias.grad is not None
    assert all(p.grad is None for p in frozen.parameters())
    assert {id(p) for g in opt.param_groups for p in g["params"]} == {id(probe.weight),
                                                                     id(probe.bias)}
    print(f" 20  probe backward: probe gets grads, the frozen module gets none; the optimizer "
          f"holds ONLY the probe's 2 tensors   OK")


# ------------------------------------------------------------------ 23
def test_tunnel_reuses_project_rule():
    names = list(REP_NAMES)
    # a curve that first crosses 1.05 * last at position 3
    v = np.array([2.00, 1.50, 1.20, 1.04, 1.30, 1.10, 1.02, 1.01, 1.00,
                  1.00, 1.00, 1.00, 1.00, 1.00])
    assert v.size == NUM_POINTS
    ent = tunnel_entrance(v, names)
    assert ent["index"] == tunnel_start(v, TUNNEL_TOL) == 3, ent["index"]
    assert ent["point"] == names[3] == "L3"
    assert ent["block_index"] == 3 and abs(ent["relative_depth"] - 3 / 12) < 1e-12
    assert abs(ent["relative_position"] - 3 / 13) < 1e-12
    assert ent["reference_point"] == f"L{NUM_BLOCKS}+RMS"
    assert ent["definition"].startswith("first_crossing_95")
    # every tolerance saved at once, and tighter tolerance never enters EARLIER
    allt = tunnel_entrances(v, names)
    idx = [allt[f"tol_{t:g}"]["index"] for t in (0.01, 0.02, 0.05, 0.10)]
    assert idx == sorted(idx, reverse=True), idx
    # the last point always satisfies the criterion
    assert tunnel_entrance(np.arange(NUM_POINTS, 0, -1).astype(float) * 0 + 1.0, names)["index"] == 0
    # wrong-length curves are refused
    try:
        tunnel_entrance(v[:5], names)
    except ValueError:
        pass
    else:
        raise AssertionError("tunnel_entrance accepted a curve of the wrong length")
    print(f" 23  tunnel == probing.tunnel.tunnel_start (index {ent['index']}, {ent['point']}, "
          f"relative_depth {ent['relative_depth']:.3f}); tolerances nested   OK")


# ------------------------------------------------------------------ 17, 25, 26
def test_cka_contracts():
    rng = np.random.default_rng(0)
    n, K = 40, GEOM.n_forecast_patches
    feats = {name: rng.normal(size=(n, K, D)).astype(np.float32) for name in REP_NAMES}

    # 25 -- the CKA matrix rows ARE the probe's rows, in the same order
    M = rep_matrix(feats["Emb"])
    assert M.shape == (n * K, D)
    for i in range(n):
        for k in range(K):
            assert np.allclose(M[i * K + k], feats["Emb"][i, k]), (i, k)
    from probing.tirex_probes import _stack
    assert np.allclose(M, _stack(feats["Emb"]))
    assert np.allclose(M, cka_mod.stack_slots(feats["Emb"]))
    print(f" 25  rep_matrix -> ({n * K}, {D}); row i*K+k is window i position k, identical to "
          f"the probe's stacking   OK")

    mats, names = rep_matrices(feats)
    assert names == list(REP_NAMES) and len(mats) == NUM_POINTS

    # 17 -- diagonal, symmetry, ordering, determinism, both estimators
    for est in ("biased", "unbiased"):
        A = cka_layer_matrix(mats, estimator=est)
        assert A.shape == (NUM_POINTS, NUM_POINTS)
        assert np.abs(np.diag(A) - 1.0).max() < 1e-8, (est, np.diag(A))
        assert np.abs(A - A.T).max() < 1e-10, est
        assert np.allclose(A, cka_layer_matrix(mats, estimator=est))          # deterministic
        # layer ordering: permuting the inputs permutes the matrix the same way
        perm = [3, 0, 1, 2] + list(range(4, NUM_POINTS))
        Ap = cka_layer_matrix([mats[i] for i in perm], estimator=est)
        assert np.allclose(Ap, A[np.ix_(perm, perm)], atol=1e-12), est
        assert abs(cka_mod.linear_cka(mats[0], mats[0], estimator=est) - 1.0) < 1e-8
    # independent Gaussians: the BIASED estimator sits well above 0, the UNBIASED near it --
    # which is exactly why absolute biased CKA is not comparable across models
    fb = cka_null_floor(n * K, D, reps=2, estimator="biased")["mean"]
    fu = cka_null_floor(n * K, D, reps=2, estimator="unbiased")["mean"]
    assert fb > fu and abs(fu) < 0.05, (fb, fu)
    # negative unbiased values are NOT clipped
    assert cka_null_floor(n * K, D, reps=6, estimator="unbiased")["values"] is not None
    print(f" 17  CKA diag=1, symmetric, order-equivariant, deterministic, both estimators; "
          f"null floor biased {fb:.3f} vs unbiased {fu:+.4f}   OK")

    # 26 -- standardized input is refused for the geometry analyses
    Z = (M - M.mean(0)) / M.std(0)
    try:
        assert_not_standardized(Z)
    except RuntimeError:
        pass
    else:
        raise AssertionError("assert_not_standardized failed to catch a z-scored matrix")
    assert_not_standardized(M)                       # raw passes
    print(f" 26  a z-scored matrix is REFUSED for CKA / effective rank; raw states pass   OK")


# ------------------------------------------------------------------ 18
def test_effective_rank_contracts():
    rng = np.random.default_rng(0)
    n = 300
    # rank-one -> effective rank 1
    u = rng.normal(size=(n, 1))
    r1 = spectral_metrics(u @ rng.normal(size=(1, D)))
    assert abs(r1["effective_rank"] - 1.0) < 1e-6, r1["effective_rank"]
    # isotropic k-dimensional -> close to k
    k = 40
    iso = spectral_metrics(rng.normal(size=(n, k)) @ np.eye(k, D))
    assert 0.9 * k < iso["effective_rank"] <= k + 1e-6, iso["effective_rank"]
    # bounds: finite, positive, never exceeds min(N-1, d)
    for N in (8, 50, 300):
        m = spectral_metrics(rng.normal(size=(N, D)))
        assert np.isfinite(m["effective_rank"]) and m["effective_rank"] > 0
        assert m["effective_rank"] <= min(N - 1, D) + 1e-6, (N, m["effective_rank"])
    # deterministic on the same matrix
    A = rng.normal(size=(n, D))
    assert spectral_metrics(A)["effective_rank"] == spectral_metrics(A)["effective_rank"]
    # the curve helper: no z-scoring, normalization denominators correct
    K = GEOM.n_forecast_patches
    feats = {name: rng.normal(size=(30, K, D)).astype(np.float32) for name in REP_NAMES}
    mats, names = rep_matrices(feats)
    cur = effective_rank_curve(mats, names, hidden_dim=D)
    assert len(cur["effective_rank"]) == NUM_POINTS and cur["hidden_dim"] == D
    assert cur["n_samples"] == 30 * K and cur["max_possible_rank"] == min(30 * K - 1, D)
    assert np.allclose(cur["normalized_effective_rank"],
                       np.asarray(cur["effective_rank"]) / D)
    print(f" 18  effective rank: rank-1 -> {r1['effective_rank']:.4f}, isotropic-{k}d -> "
          f"{iso['effective_rank']:.2f}, bounded by min(N-1,d), deterministic   OK")


# ------------------------------------------------------------------ 27
def test_readout_degeneracy_is_detected():
    """(Emb, position 1) is bit-identical across windows because the masked-future token reaches
    the patch embedding as (values=0, mask=0) before any recurrence. That is a real property of
    TiRex, so the code must REPORT it -- and must offer the position-0-only companion that makes
    Emb comparable with the deeper depths. The --with-model block proves it on the real model;
    this half proves the detection and the companion path."""
    rng = np.random.default_rng(0)
    K = GEOM.n_forecast_patches
    n = 24
    feats = {name: rng.normal(size=(n, K, D)).astype(np.float32) for name in REP_NAMES}
    feats["Emb"][:, 1, :] = feats["Emb"][0, 1, :]            # the real degeneracy, reproduced

    deg = degenerate_readouts(feats)
    assert deg["affected"] == [{"point": "Emb", "position": 1}] == deg["expected"], deg["affected"]
    assert deg["constant_across_windows_by_point"]["Emb"] == [False, True]
    assert deg["constant_across_windows_by_point"]["L6"] == [False, False]

    norms = representation_norms(feats)
    assert norms["Emb"]["constant_across_windows"] == [False, True]
    assert norms["Emb"]["across_window_std_by_position"][1] == 0.0
    assert norms["L6"]["across_window_std_by_position"][1] > 0.0

    # the pos0 companion: n rows, no constant block, and it is NOT the headline matrix
    g = geometry_for_split(feats, null_floor_reps=1)
    assert g["n_rows"][VARIANT_ALL_FROM_L1] == n * K and g["n_rows"][VARIANT_POS0] == n
    assert g["n_windows"] == n and g["positions_per_window"] == K

    # The constant block genuinely changes Emb's spectrum -- which is exactly WHY a 2n-row Emb
    # is never produced. Computed here directly (the companion deliberately does not cover Emb),
    # so the reason for the policy stays measured rather than asserted.
    e_full = spectral_metrics(rep_matrix(feats["Emb"]))["effective_rank"]
    e_pos0 = g["effective_rank"][VARIANT_POS0]["effective_rank"][0]
    assert abs(e_full - e_pos0) > 1e-6, (e_full, e_pos0)
    assert DEGENERACY_CAVEAT in g["degeneracy_caveat"]
    # an out-of-range position subset is refused
    try:
        rep_matrix(feats["Emb"], positions=[0, 5])
    except ValueError:
        pass
    else:
        raise AssertionError("rep_matrix accepted an out-of-range readout position")
    # 28 -- the HEADLINE geometry policy: position 0 only, at EVERY depth, for BOTH estimators,
    # with a CONSTANT observation count; the companion covers L1 onward with both positions.
    assert CKA_HEADLINE_VARIANT == ERANK_HEADLINE_VARIANT == VARIANT_POS0
    hn, hp = variant_spec(VARIANT_POS0, REP_NAMES, K)
    cn, cp = variant_spec(VARIANT_ALL_FROM_L1, REP_NAMES, K)
    assert hn == list(REP_NAMES) and hp == (0,)
    assert cn == list(REP_NAMES)[1:] and cp == tuple(range(K)) and EMB_POINT not in cn

    head = g["effective_rank"][ERANK_HEADLINE_VARIANT]
    comp = g["effective_rank"][VARIANT_ALL_FROM_L1]
    # the headline observation count does NOT change with depth -- the whole point
    assert head["n_samples"] == n and head["max_possible_rank"] == min(n - 1, D)
    assert len(head["effective_rank"]) == NUM_POINTS == len(head["representation_points"])
    assert head["positions_used"] == [0] and head["constant_observation_count"]
    # the companion is self-consistent too, and simply does not cover Emb
    assert comp["n_samples"] == n * K and len(comp["effective_rank"]) == NUM_POINTS - 1
    assert comp["representation_points"] == list(REP_NAMES)[1:]
    assert comp["positions_used"] == list(range(K))
    # the headline Emb value IS the position-0 value (not the 2n-row one)
    assert head["effective_rank"][0] == e_pos0 != e_full
    # the N-changing curve is NOT produced anywhere
    assert "mixed" not in g["effective_rank"] and "all_positions" not in g["effective_rank"]
    assert set(g["effective_rank"]) == set(GEOMETRY_VARIANTS)
    # each variant's CKA matrix is square on ITS OWN depth set
    assert g["cka"][VARIANT_POS0]["unbiased"].shape == (NUM_POINTS, NUM_POINTS)
    assert g["cka"][VARIANT_ALL_FROM_L1]["unbiased"].shape == (NUM_POINTS - 1, NUM_POINTS - 1)
    assert g["variant_points"][VARIANT_ALL_FROM_L1] == list(REP_NAMES)[1:]
    # CKA still cannot mix row counts -- the reason the headline must be one construction
    try:
        cka_layer_matrix([rep_matrix(feats["Emb"], positions=[0]), rep_matrix(feats["L1"])])
    except ValueError:
        pass
    else:
        raise AssertionError("CKA accepted mismatched row counts across depths")
    try:
        variant_spec("mixed", REP_NAMES, K)
    except ValueError:
        pass
    else:
        raise AssertionError("an N-changing geometry variant was accepted")
    print(f" 27  (Emb, pos 1) constant across windows -> flagged; a 2n-row Emb would read "
          f"r_eff {e_full:.2f} vs the headline's {e_pos0:.2f}, which is why it is not built  OK")
    print(f" 28  HEADLINE geometry = {VARIANT_POS0}: position 0 at ALL {NUM_POINTS} depths, "
          f"n={n} rows CONSTANT, same construction for CKA and effective rank")
    print(f"      companion = {VARIANT_ALL_FROM_L1}: both positions, n={n * K} rows, "
          f"{NUM_POINTS - 1} depths (L1 onward); no N-changing curve is produced   OK")


# ------------------------------------------------------------------ 29, 30 (model-free half)
def test_two_pass_layout_and_rollout_contexts():
    """The default rollout's per-pass layout, derived from the package's own context
    construction: pass k is handed x[T-C:T] ++ (k*P) NaN, then LEFT-padded to train_ctx_len."""
    g, K, P = GEOM, GEOM.n_forecast_patches, GEOM.input_patch

    # every pass has the same token count and reads its own LAST token
    assert g.two_pass_n_tokens == TRAIN_CTX // P == 64
    assert g.two_pass_readout_index == 63
    assert g.readout_token_indices(ROLLOUT_TWO) == (63, 63)
    assert g.readout_token_indices(ROLLOUT_SINGLE) == (63, 64)
    assert g.n_tokens_for(ROLLOUT_SINGLE) == 65 and g.n_tokens_for(ROLLOUT_TWO) == 64

    l0, l1 = g.two_pass_layout(0), g.two_pass_layout(1)
    assert l0["context_len"] == C and l1["context_len"] == C + P == 544
    assert l0["pad_len"] == 1536 and l1["pad_len"] == 1504
    assert l0["n_pad_patches"] == 48 and l1["n_pad_patches"] == 47
    assert l0["real_token_range"] == [48, 63] and l1["real_token_range"] == [47, 62]
    assert l0["n_appended_missing_patches"] == 0 and l1["n_appended_missing_patches"] == 1
    # pass 0 reads the LAST REAL patch; pass 1 reads an APPENDED MISSING patch
    assert l0["readout_is_last_real_context_patch"] and not l0["readout_is_appended_missing_patch"]
    assert l1["readout_is_appended_missing_patch"] and not l1["readout_is_last_real_context_patch"]
    assert l0["target_slice"] == [0, 32] and l1["target_slice"] == [32, 64]
    print(f" 29  two_pass: pass0 real {l0['real_token_range']} readout {l0['readout_index']} "
          f"(last real) | pass1 real {l1['real_token_range']} readout {l1['readout_index']} "
          f"(appended missing)   OK")

    # 30 (model-free half) -- rollout_contexts appends pure NaN, of the right width
    ctx = torch.as_tensor(_series(3, seed=5)[:, :C])
    cs = rollout_contexts(ctx, g)
    assert len(cs) == K
    assert torch.equal(cs[0], ctx)
    assert cs[1].shape[-1] == C + P
    assert torch.equal(cs[1][:, :C], ctx)                       # real prefix untouched
    assert bool(torch.isnan(cs[1][:, C:]).all())                # appended block is ALL NaN
    assert not bool(torch.isfinite(cs[1][:, C:]).any())
    print(f" 30  rollout_contexts: pass1 context = ctx ++ {P} x NaN, real prefix untouched   OK")

    # the geometry REFUSES a configuration where a later pass would truncate real context
    try:
        TiRexGeometry(C=TRAIN_CTX, H=H, input_patch=P, output_patch=P, train_ctx_len=TRAIN_CTX,
                      num_quantiles=9, median_index=4)
    except ValueError as exc:
        assert "truncate real context" in str(exc)
    else:
        raise AssertionError("geometry accepted a two_pass config that truncates real context")

    # both modes are named, validated, and mutually exclusive
    assert set(ROLLOUT_MODES) == {ROLLOUT_SINGLE, ROLLOUT_TWO}
    for m in ROLLOUT_MODES:
        assert check_rollout_mode(m) == m
    try:
        check_rollout_mode("autoregressive")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown rollout mode was accepted")

    # the cache PATH carries the mode, so the two modes can never collide
    X = _series(4, seed=6)[:, :C]
    r1 = cache_root("/tmp/x", "elec", "test", ROLLOUT_SINGLE)
    r2 = cache_root("/tmp/x", "elec", "test", ROLLOUT_TWO)
    assert r1 != r2 and ROLLOUT_SINGLE in r1.name and ROLLOUT_TWO in r2.name
    m1 = cache_metadata("elec", "test", GEOM, checkpoint="c", backend="torch",
                        points=list(REP_NAMES), seed=0, X=X, mode=ROLLOUT_SINGLE)
    m2 = cache_metadata("elec", "test", GEOM, checkpoint="c", backend="torch",
                        points=list(REP_NAMES), seed=0, X=X, mode=ROLLOUT_TWO)
    assert m1["rollout_mode"] != m2["rollout_mode"]
    print(f"  cache path + metadata carry the rollout mode; the two modes cannot collide  OK")


def test_two_pass_targets_use_per_pass_normalization():
    """In two_pass each output patch is normalized with ITS OWN pass's (loc, scale). The stub
    reproduces TiRex's tokenizer exactly, so this pins the algebra; the --with-model block pins
    it against the real model."""
    full = _series(5, seed=9)
    ctx, fut = full[:, :C], full[:, C:]
    K = GEOM.n_forecast_patches

    t1, l1, s1 = build_targets(ctx, fut, STUB, GEOM, mode=ROLLOUT_SINGLE)
    t2, l2, s2 = build_targets(ctx, fut, STUB, GEOM, mode=ROLLOUT_TWO)
    assert l1.shape == l2.shape == (5, K)

    # single_pass: ONE pass, so the K columns are identical BY CONSTRUCTION
    assert np.array_equal(l1[:, 0], l1[:, 1]) and np.array_equal(s1[:, 0], s1[:, 1])
    # two_pass: each column comes from its own pass. Mathematically the same quantity (the NaN
    # pad is excluded by nanmean and no real value is truncated), so they agree closely.
    assert np.allclose(l2[:, 0], l2[:, 1], rtol=1e-5, atol=1e-5)
    assert np.allclose(l2, l1, rtol=1e-5, atol=1e-5)
    # round trips hold in BOTH modes, per patch
    for t, l, s in ((t1, l1, s1), (t2, l2, s2)):
        assert_target_roundtrip(fut, t, l, s, GEOM)
        assert np.allclose(denormalize(t, l, s, GEOM), fut, atol=1e-3)

    # denormalize REFUSES the old flat (n,) statistics -- it must be given per-patch stats
    try:
        denormalize(t2, l2[:, 0], s2[:, 0], GEOM)
    except ValueError:
        pass
    else:
        raise AssertionError("denormalize accepted flat (n,) statistics")

    # per-patch really is per-patch: perturb patch 1's stats and ONLY patch 1 moves
    l3 = l2.copy(); l3[:, 1] += 10.0
    back = denormalize(t2, l3, s2, GEOM)
    assert np.allclose(back[:, :32], fut[:, :32], atol=1e-3)
    assert np.allclose(back[:, 32:] - 10.0, fut[:, 32:], atol=1e-3)
    print(f"  two_pass targets: (n, {K}) per-patch (loc, scale); patch k uses pass k's stats; "
          f"flat stats REFUSED   OK")


# ------------------------------------------------------------------ 24
def test_cache_refusal():
    X = _series(5, seed=7)[:, :C]
    base = dict(checkpoint="NX-AI/TiRex", backend="torch", points=list(REP_NAMES), seed=0,
                batch_size=64)
    meta = cache_metadata("electricity", "test", GEOM, X=X, **base)
    assert meta["cache_version"] == CACHE_VERSION and meta["n_windows"] == 5
    with tempfile.TemporaryDirectory() as td:
        root = cache_root(td, "electricity", "test")
        assert read_cache(root, meta, list(REP_NAMES)) is None        # absent -> None, no raise
        root.parent.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(0)
        np.savez(root.with_suffix(".npz"),
                 native=rng.normal(size=(5, H, 9)).astype(np.float32),
                 **{f"rep__{p.slug}": rng.normal(size=(5, 2, D)).astype(np.float32)
                    for p in REP_POINTS})
        root.with_suffix(".json").write_text(json.dumps(meta))
        got = read_cache(root, meta, list(REP_NAMES))
        assert got is not None and got[0]["Emb"].shape == (5, 2, D)

        # every one of these must be REFUSED, not silently reused
        for field, value in [("backend", "cuda"), ("checkpoint", "other/model"), ("seed", 1),
                             ("batch_size", 128)]:
            bad = cache_metadata("electricity", "test", GEOM, X=X, **{**base, field: value})
            try:
                read_cache(root, bad, list(REP_NAMES))
            except RuntimeError:
                pass
            else:
                raise AssertionError(f"cache accepted a differing {field}")
        # different WINDOWS with identical metadata fields
        bad = cache_metadata("electricity", "test", GEOM, X=_series(5, seed=99)[:, :C], **base)
        try:
            read_cache(root, bad, list(REP_NAMES))
        except RuntimeError:
            pass
        else:
            raise AssertionError("cache accepted a different window set")
        # different GEOMETRY
        g2 = TiRexGeometry(C=256, H=H, input_patch=P, output_patch=P, train_ctx_len=TRAIN_CTX,
                           num_quantiles=9, median_index=4)
        bad = cache_metadata("electricity", "test", g2, X=X, **base)
        try:
            read_cache(root, bad, list(REP_NAMES))
        except RuntimeError:
            pass
        else:
            raise AssertionError("cache accepted a different geometry")
    print(f" 24  cache refuses a differing backend / checkpoint / seed / batch_size / windows "
          f"/ geometry  OK")


# ------------------------------------------------------------------ end-to-end (model-free)
def test_end_to_end_synthetic():
    """The full driver path on synthetic features: layerwise fit -> per-patch diagnostic ->
    tunnel -> CKA -> effective rank -> JSON-serializable record."""
    rng = np.random.default_rng(0)
    K = GEOM.n_forecast_patches
    ntr, nva, nte = 32, 12, 12
    W = rng.normal(size=(D, GEOM.output_patch)).astype(np.float32) * 0.1

    def split(n, s, noise):
        r = np.random.default_rng(s)
        feats, y = {}, None
        base = r.normal(size=(n, K, D)).astype(np.float32)
        for i, name in enumerate(REP_NAMES):
            feats[name] = base + r.normal(0, 0.3 / (i + 1), base.shape).astype(np.float32)
        y = np.concatenate([base[:, k] @ W for k in range(K)], axis=1)
        return feats, (y + r.normal(0, noise, y.shape)).astype(np.float32)

    Ftr, ytr = split(ntr, 1, 0.05)
    Fva, yva = split(nva, 2, 0.05)
    Fte, yte = split(nte, 3, 0.05)

    res = layerwise_probe(Ftr, ytr, Fva, yva, Fte, yte, GEOM, epochs=25,
                          wd_grid=(1e-3,), verbose=False)
    assert set(res) == set(REP_NAMES)
    for name, r in res.items():
        assert r["pred_test_norm"].shape == (nte, H)
        assert set(r["test_per_patch"]) == {"full", "patch0_last_context", "patch1_masked_future"}
        assert np.isfinite(r["val_loss"]) and np.isfinite(r["test_loss"])
        assert r["n_params"] == D * 32 + 32

    val_curve = [res[n]["val_loss"] for n in REP_NAMES]
    ent = tunnel_entrance(val_curve, list(REP_NAMES))
    assert 0 <= ent["index"] < NUM_POINTS

    geo = geometry_for_split(Fte, null_floor_reps=1)
    assert geo["n_rows"][VARIANT_POS0] == nte
    assert geo["n_rows"][VARIANT_ALL_FROM_L1] == nte * K
    assert set(geo["cka"]) == set(GEOMETRY_VARIANTS)
    assert set(geo["cka"][VARIANT_POS0]) == {"biased", "unbiased"}
    assert geo["cka"][CKA_HEADLINE_VARIANT]["unbiased"].shape == (NUM_POINTS, NUM_POINTS)
    assert len(geo["effective_rank"][ERANK_HEADLINE_VARIANT]["effective_rank"]) == NUM_POINTS

    norms = representation_norms(Fte)
    assert set(norms) == set(REP_NAMES) and len(norms["Emb"]["mean_l2_by_position"]) == K

    json.dumps({"tunnel": ent, "per_patch": res["Emb"]["test_per_patch"],
                "erank": geo["effective_rank"][ERANK_HEADLINE_VARIANT]["effective_rank"],
                "norms": norms, "depths": rep_depth_table()})
    print(f" E2E synthetic: {NUM_POINTS} depths fitted, tunnel at {ent['point']} "
          f"(relative_depth {ent['relative_depth']:.3f}), CKA + effective rank + JSON   OK")


# =========================================================================== #
# model-backed contracts: 1, 5, 8, 9, 19, 20  (COMPUTE NODE ONLY)
# =========================================================================== #
def model_tests():
    from probing.tirex_model import (apply_native_head, assert_frozen, assert_native_geometry,
                                     discover_geometry, extract_window_features,
                                     geometry_from_model, get_model, head_checksum,
                                     native_forward, register_rep_hooks, rollout_mode_gap,
                                     select_readout, verify_native_head)
    import os
    device = os.environ.get("TIREX_TEST_DEVICE", "cpu")
    backend = os.environ.get("TIREX_TEST_BACKEND", "torch")
    print(f"\n  loading NX-AI/TiRex on device={device} backend={backend} ...")
    model = get_model(device=device, backend=backend)
    geom = geometry_from_model(model)
    # DEVICE HYGIENE: never hardcode .cuda(). Every synthetic tensor handed to the model, its
    # embedding, its blocks, its final norm or its output head is built on / moved to the device
    # the model actually lives on, so the same test body runs unchanged on CPU and on GPU.
    dev = next(model.parameters()).device
    print(f"      model parameters live on {dev}; synthetic tensors will follow")

    # 1 -- geometry discovery + hard assertion against the real checkpoint
    disc = discover_geometry(model, geom)
    assert_native_geometry(model, geom)
    print(f"  1  {disc['package']}=={disc['package_version']} torch={disc['torch_version']} "
          f"ckpt={disc['checkpoint']}")
    print(f"      blocks={disc['num_blocks']} d={disc['embedding_dim']} heads={disc['num_heads']} "
          f"patch={disc['input_patch_size']}/{disc['output_patch_size']} "
          f"ff={disc['input_ff_dim']} train_ctx_len={disc['train_ctx_len']}")
    print(f"      quantiles={disc['quantiles']} head={disc['output_head_class']}"
          f"(->{disc['output_head_out_features']}) norm={disc['final_norm_class']}")
    assert disc["num_blocks"] == NUM_BLOCKS and disc["embedding_dim"] == MODEL_DIMS
    assert tuple(disc["quantiles"]) == EXPECT_QUANTILES
    assert geom.as_dict() == GEOM.as_dict(), "the checkpoint geometry differs from the test fixture"
    print(f"  1  discovered geometry == the asserted fixture                              OK")

    # 20 -- frozen
    fz = assert_frozen(model)
    assert fz["frozen"] and not model.training
    ck0 = head_checksum(model)
    print(f" 20  {fz['n_parameters']:,} parameters, none trainable; head checksum {ck0}   OK")

    ctx = torch.as_tensor(_series(4, seed=11)[:, :C]).to(dev)
    official, reps, state = native_forward(model, ctx, geom)

    # 9 + 10 -- hook locations and shapes
    assert list(reps) == list(REP_NAMES) or set(reps) == set(REP_NAMES)
    for name in REP_NAMES:
        assert reps[name].shape == (4, geom.n_tokens, MODEL_DIMS), (name, reps[name].shape)
    # cross-check L1..L12 against the model's OWN return_all_hidden stack -- this is what
    # proves no hook captured a PRE-block tensor
    adj, _ = model._adjust_context_length(ctx, geom.train_ctx_len, geom.train_ctx_len)
    tok, _ = model.tokenizer.input_transform(adj)
    pad = torch.full((tok.shape[0], geom.n_forecast_patches - 1, tok.shape[2]), float("nan"),
                     dtype=tok.dtype, device=tok.device)
    tok_all = torch.cat((tok, pad), dim=1)
    mask = torch.isnan(tok_all).logical_not().to(tok_all.dtype)
    inp = torch.cat((torch.nan_to_num(tok_all, nan=0.0), mask), dim=2)
    with torch.inference_mode():
        _, allh = model._forward_model(inp, return_all_hidden=True)
    for i in range(NUM_BLOCKS):
        assert torch.equal(reps[f"L{i + 1}"], allh[:, :, i, :]), f"hook L{i+1} != block {i} output"
    assert not torch.equal(reps["Emb"], allh[:, :, 0, :]), "Emb must differ from L1"
    with torch.inference_mode():
        assert torch.equal(reps[f"L{NUM_BLOCKS}+RMS"], model.out_norm(reps[f"L{NUM_BLOCKS}"]))
    print(f"  9  Emb (pre-block-1) + L1..L{NUM_BLOCKS} == return_all_hidden, all {NUM_BLOCKS} exact; "
          f"L{NUM_BLOCKS}+RMS == out_norm(L{NUM_BLOCKS})   OK")
    print(f" 10  every point is (B, {geom.n_tokens}, {MODEL_DIMS})                              OK")

    # 5 -- THE DECISIVE ONE
    vn = verify_native_head(model, reps, official, state, geom)
    print(f"  5  tokens {vn['readout_indices']} -> frozen output_patch_embedding -> native "
          f"de-normalization  ==  official H={H} forecast")
    print(f"      max|d| = {vn['max_abs_error']:.3e}  max scaled err = {vn['max_scaled_error']:.3f} "
          f"(<=1)  bitwise-exact = {vn['exact_bitwise_match']}")
    print(f"      wrong-index controls: "
          + ", ".join(f"{k}={v:.3e}" for k, v in vn["wrong_index_controls"].items()))
    assert vn["passed"] and vn["controls_discriminate"]

    # 8 -- no target leakage, against the REAL model
    full = _series(4, seed=12)
    a, _, _ = native_forward(model, torch.as_tensor(full[:, :C]).to(dev), geom, capture=False)
    alt = full.copy(); alt[:, C:] = alt[:, C:] * 7.0 - 100.0
    b, _, _ = native_forward(model, torch.as_tensor(alt[:, :C]).to(dev), geom, capture=False)
    assert torch.equal(a, b)
    print(f"  8  mutating the future leaves the native forecast bit-identical              OK")

    # 19 -- EXTRACTION DETERMINISM, stated in two parts because they are two different claims.
    #
    # (a) For a FIXED batch size, extraction is BITWISE reproducible. This is the invariant the
    #     science needs and it is asserted exactly, on every device.
    # (b) ACROSS batch sizes it is bitwise reproducible on CPU, but NOT necessarily on an
    #     accelerator: TiRex's sLSTM recurrence runs in bfloat16, and a small-batch GEMM kernel
    #     switch perturbs block 1 by ~1e-3 of its std, which 64 timesteps x 12 blocks amplify.
    #     Measured on MPS: batch 4 == batch 8 BITWISE; batch 2 diverges to ~6e-2 of the layer std
    #     at L6/L7. So (b) is MEASURED and PRINTED rather than assumed, and `batch_size` is part
    #     of the feature-cache key so two batch sizes can never be mixed.
    #     The bar below is a CATASTROPHE bar, not a precision claim.
    f1, n1 = extract_window_features(full[:, :C], model, geom, batch_size=2)
    f1b, n1b = extract_window_features(full[:, :C], model, geom, batch_size=2)
    f2, n2 = extract_window_features(full[:, :C], model, geom, batch_size=4)
    for name in REP_NAMES:
        assert np.array_equal(f1[name], f1b[name]), f"{name} is not deterministic at a FIXED batch size"
        assert f1[name].shape == (4, geom.n_forecast_patches, MODEL_DIMS)
    assert np.array_equal(n1, n1b) and n1.shape == (4, H, 9)
    print(f" 19  extraction is BITWISE deterministic at a fixed batch size; features are "
          f"(n, {geom.n_forecast_patches}, {MODEL_DIMS})   OK")

    worst, worst_name = 0.0, None
    for name in REP_NAMES:
        sd = float(f1[name].std()) + 1e-12
        r = float(np.abs(f1[name] - f2[name]).max() / sd)
        if r > worst:
            worst, worst_name = r, name
    nat_rel = float(np.abs(n1 - n2).max() / (np.abs(n1).mean() + 1e-12))
    on_cpu = dev.type == "cpu"
    if on_cpu:
        for name in REP_NAMES:
            assert np.array_equal(f1[name], f2[name]), f"{name} depends on batch size ON CPU"
        assert np.array_equal(n1, n2)
    assert worst < 0.2, (worst_name, worst)        # catastrophe bar, NOT a precision claim
    print(f"      batch 2 vs 4 on {dev.type}: worst depth {worst_name} at {worst:.2e} of its std, "
          f"native forecast {nat_rel:.2e} relative"
          + ("  (bitwise identical, as CPU must be)" if on_cpu
             else "  -- accelerator kernel switch + bf16 recurrence; batch_size is in the cache key"))

    # the two readout positions must NOT be the same vector (a duplicated-index bug)
    for name in ("Emb", f"L{NUM_BLOCKS}+RMS"):
        assert not np.allclose(f1[name][:, 0], f1[name][:, 1]), name
    print(f"      the two readout positions are distinct at every depth                    OK")

    # 27 on the REAL model: exactly one (point, position) is constant across windows, and it is
    # (Emb, 1) -- the masked-future token before any recurrence has touched it.
    varied = np.stack([_series(1, seed=200 + i)[0, :C] * (1.0 + i) for i in range(6)])
    fv, _ = extract_window_features(varied, model, geom, batch_size=3)
    deg = degenerate_readouts(fv)
    assert deg["affected"] == deg["expected"] == [{"point": "Emb", "position": 1}], deg["affected"]
    for name in REP_NAMES[1:]:
        assert deg["constant_across_windows_by_point"][name] == [False, False], name
    assert np.array_equal(fv["Emb"][0, 1], fv["Emb"][-1, 1])
    print(f" 27  exactly one degenerate readout on the real model: (Emb, position 1) is "
          f"bit-identical across windows; all {NUM_POINTS - 1} deeper depths vary   OK")

    # documented deviation: the package-default rollout differs
    gap = rollout_mode_gap(model, ctx, geom)
    print(f"  D  rollout modes: {gap['mode_used']} vs {gap['mode_compared']} -> max|d| "
          f"{gap['max_abs_diff']:.3e} ({gap['relative_to_mean_abs']:.2e} relative); "
          f"first patch identical={gap['first_patch_identical']}")
    assert gap["first_patch_identical"], "the two rollout modes must agree on the first patch"

    # 30 -- what the rollout appends, proved against the RUNNING model
    from probing.tirex_model import (assert_rollout_appends_missing,
                                     assert_two_pass_matches_package,
                                     native_forward_two_pass, readouts_from_reps,
                                     scaler_states_for_mode, verify_native_head_two_pass)
    ra = assert_rollout_appends_missing(model, ctx, geom)
    assert ra["all_nan"] and not ra["any_finite"] and not ra["equals_some_forecast_quantile_row"]
    assert ra["real_prefix_unchanged"]
    print(f" 30  pass 1 consumes MISSING values: appended block all-NaN={ra['all_nan']}, "
          f"any finite={ra['any_finite']}, equals a forecast quantile row="
          f"{ra['equals_some_forecast_quantile_row']}")
    print(f"      (previous forecast median magnitude "
          f"{ra['previous_forecast_median_abs_mean']:.3f} -- present, and DISCARDED)   OK")

    # 31 -- the two_pass native-head identity
    pkg = assert_two_pass_matches_package(model, ctx, geom)
    assert pkg["bitwise_identical"]
    off2, per_pass, states2, layouts = native_forward_two_pass(model, ctx, geom)
    assert len(per_pass) == geom.n_forecast_patches and len(states2) == geom.n_forecast_patches
    for k, pp in enumerate(per_pass):
        for name in REP_NAMES:
            assert pp[name].shape == (ctx.shape[0], geom.two_pass_n_tokens, MODEL_DIMS), (k, name)
    v2 = verify_native_head_two_pass(model, per_pass, off2, states2, geom)
    print(f" 31  two_pass: our replicated loop == the package default path (bitwise "
          f"{pkg['bitwise_identical']})")
    print(f"      per-pass states at token {v2['readout_index_per_pass']} reproduce it: "
          f"max|d| {v2['max_abs_error']:.3e}, scaled {v2['max_scaled_error']:.3f} (<=1), "
          f"slice-order control {v2['slice_order_control_max_abs']:.3e}")
    print(f"      controls: "
          + ", ".join(f"{k}={x:.3e}" for k, x in v2["wrong_index_controls"].items()) + "   OK")
    assert v2["passed"] and v2["controls_discriminate"]

    # the two modes agree EXACTLY at readout position 0 (the sLSTM is causal: an appended token
    # cannot influence token 63) and differ at position 1
    ro2 = readouts_from_reps(per_pass, geom, "two_pass")
    ro1 = readouts_from_reps(reps, geom, "single_pass")
    same0 = all(torch.equal(ro1[n][:, 0], ro2[n][:, 0]) for n in REP_NAMES)
    diff1 = {n: float((ro1[n][:, 1] - ro2[n][:, 1]).abs().max().detach().cpu())
             for n in REP_NAMES}
    assert same0, "position 0 must be identical across modes (causality)"
    assert diff1[f"L{NUM_BLOCKS}"] > 0, "position 1 must differ across modes"
    assert diff1["Emb"] == 0.0, "Emb position 1 is the same constant in both modes"
    print(f"      causality: readout position 0 is BIT-IDENTICAL across modes at all "
          f"{NUM_POINTS} depths; position 1 differs (L{NUM_BLOCKS} max|d| "
          f"{diff1[f'L{NUM_BLOCKS}']:.3f}, Emb {diff1['Emb']:.1f} = the shared constant)   OK")

    # per-pass normalization from the REAL model
    lo1, sc1 = scaler_states_for_mode(model, ctx, geom, "single_pass")
    lo2, sc2 = scaler_states_for_mode(model, ctx, geom, "two_pass")
    assert np.array_equal(lo1[:, 0], lo1[:, 1])                 # one pass -> one statistic
    assert np.allclose(lo2[:, 0], lo2[:, 1], rtol=1e-5, atol=1e-5)
    print(f"      per-pass (loc, scale): single_pass columns identical; two_pass columns agree "
          f"to {float(np.abs(lo2[:, 0] - lo2[:, 1]).max()):.2e} (float32 reduction order)   OK")

    # 32 -- DEVICE HYGIENE, the regression guard for this exact class of bug. Every entry point
    # that takes a RAW context must accept one on the CPU (or as numpy) while the model sits
    # wherever it sits, and must return tensors on ONE consistent device. On CPU this is a no-op;
    # on GPU it is the difference between working and the mat1/mat2 device crash.
    from probing.tirex_model import model_device, to_model_device
    assert model_device(model) == dev
    cpu_ctx = torch.as_tensor(_series(3, seed=77)[:, :C])          # deliberately on the CPU
    assert cpu_ctx.device.type == "cpu"
    assert to_model_device(model, cpu_ctx).device == dev
    assert to_model_device(model, _series(3, seed=77)[:, :C]).device == dev   # numpy too

    o1, r1d, s1d = native_forward(model, cpu_ctx, geom)
    assert o1.device == dev, (o1.device, dev)                      # official, not the package's cpu default
    assert all(v.device == dev for v in r1d.values())
    assert s1d.loc.device == dev and s1d.scale.device == dev       # rescales model outputs
    verify_native_head(model, r1d, o1, s1d, geom)                  # the full identity, cpu input

    o2, pp2, st2, _ = native_forward_two_pass(model, cpu_ctx, geom)
    assert o2.device == dev and all(v.device == dev for v in pp2[0].values())
    assert all(t.loc.device == dev for t in st2)
    verify_native_head_two_pass(model, pp2, o2, st2, geom)
    assert_two_pass_matches_package(model, cpu_ctx, geom)
    assert_rollout_appends_missing(model, cpu_ctx, geom)
    lo, sc = scaler_states_for_mode(model, cpu_ctx, geom, "two_pass")
    assert lo.shape == (3, geom.n_forecast_patches) and np.all(np.isfinite(lo))
    fx, nx = extract_window_features(_series(3, seed=77)[:, :C], model, geom, batch_size=2)
    assert fx["Emb"].shape == (3, geom.n_forecast_patches, MODEL_DIMS) and nx.shape == (3, H, 9)
    print(f" 32  device hygiene: every entry point accepts a CPU/numpy context against a model "
          f"on {dev}, and returns one consistent device   OK")

    assert head_checksum(model) == ck0, "the native head changed during the test run"
    assert_frozen(model)
    print(f" 20  head checksum unchanged after every forward pass                          OK")


if __name__ == "__main__":
    print("TiRex layer-wise probing contracts (C=512 -> H=64, Q=1 shared patch head)")
    for fn in (test_geometry, test_depth_convention, test_targets_and_normalization,
               test_probe_structure, test_tau_half_objective, test_fit_protocol,
               test_no_backbone_grads_during_probe_training, test_tunnel_reuses_project_rule,
               test_cka_contracts, test_effective_rank_contracts,
               test_readout_degeneracy_is_detected,
               test_two_pass_layout_and_rollout_contexts,
               test_two_pass_targets_use_per_pass_normalization,
               test_cache_refusal, test_end_to_end_synthetic):
        fn()
    if "--with-model" in sys.argv:
        model_tests()
        print("\nall contracts hold (model-backed checks included)")
    else:
        print("\nall model-free contracts hold "
              "(run with --with-model for 1, 5, 8, 9, 19, 20, 27, 30, 31, 32 on a COMPUTE node)")
