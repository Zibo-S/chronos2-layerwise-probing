"""Phase 2d: generic content-mean-pooled layerwise readouts vs the architecture-aware ones.

THE QUESTION (reviewer dHTD). Phase 1 reads every model where its own forecasting head reads
(Chronos-2: the 4 forecast slots; TimesFM-3: the last real context token; TiRex: the token-63
state of each of its two native passes). Does a GENERIC readout -- the mean over the valid
content tokens, one linear map to the whole horizon -- change the layerwise loss curves, the
sustained 5% entrance, or the early-recoverability conclusion?

DESIGN PRINCIPLE. Every pooled arm is read off THE SAME FORWARD PASS as the native arm, so the
backbone computation is identical and only WHICH token states are read changes.

    model      native readout (Phase 1)         pooled readout (Phase 2d)            excluded
    Chronos-2  4 forecast slots, K=4 pass        mean of context patches 0..31        REG, 4 slots
    TimesFM-3  token 15 of one decode() pass     mean of tokens 0..15                 2 horizon placeholders
    TiRex      token 63 of pass 0 and pass 1     mean of pass-0 real tokens 48..63    48 left-pad, all of pass 1

THE ARMS. Four readouts, so that token selection, pooling and readout structure can be told
apart instead of attributing everything to "pooling":

    native       architecture-aware head (Phase 1, READ FROM ARTIFACTS, never refit here)
    concat       native readout states concatenated -> ONE Linear(K*d, H*Q)
    native_mean  mean of the native readout states  -> ONE Linear(d, H*Q)
    pooled       mean of the valid content tokens   -> ONE Linear(d, H*Q)

    native -> concat       READOUT STRUCTURE  same token states; shared patch head -> one full-horizon head
    concat -> native_mean  POOLING            same tokens, same head form; K states concatenated -> averaged
                                             (input K*d -> d: capacity ALSO drops K-fold, so this step is
                                             pooling + capacity, not pooling alone)
    native_mean -> pooled  TOKEN SELECTION    same head, same parameter count; readout tokens -> content tokens

TimesFM-3 has K=1: its native head already IS one Linear(1280, 576) on one state, so concat and
native_mean coincide with native and only ``pooled`` is fitted. Its native -> pooled step changes
the representation alone (token 15 -> mean of tokens 0..15), with an identical head.

Every capacity change is reported (``readout_params``): the controls change the parameter count
too, and a difference must never be read as a pure representation effect when it is not.

TRAINING. Each arm uses its model's OWN Phase-1 fitter with the head reshaped to K=1 slot of
P=H steps: Chronos-2 ``probes.fit_shared_forecast_probe_explicit_val`` (sum-over-quantile loss),
TiRex ``tirex_probes.fit_shared_patch_probe``, TimesFM-3 ``phase1_timesfm3.fit_layerwise``
unchanged. Learning rate, epochs, seed and the weight-decay grid are READ FROM THE PHASE-1 CELL,
so LOOP Seattle automatically keeps the paper's lr=1e-3 / 21-value-grid exception. Same targets,
same windows, same valid rows, validation selects, test is read once.

Nothing in this module writes to a Phase-1 or Phase-2c location. Phase-1 caches are opened with
their own read-only readers and a missing one RAISES; pooled states live in their own namespace.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from probing import phase1
from probing.phase2 import Phase1DependencyError

__all__ = ["KIND", "PROTOCOL", "CACHE_VERSION", "ARMS_BY_MODEL", "ARM_NATIVE", "ARM_POOLED",
           "ARM_CONCAT", "ARM_NATIVE_MEAN", "DECOMPOSITION", "POOLING", "NATIVE_K",
           "readout_params", "param_table", "masked_mean", "chronos2_token_mask",
           "timesfm3_token_mask", "tirex_token_mask", "assert_token_positions",
           "assert_contexts", "arm_features", "fit_shared_layerwise", "native_from_phase1",
           "reproduce_native", "paired_rows", "arm_summary", "REQUIRED"]

KIND = "phase2d_pooled"
PROTOCOL = "phase2d-pooled/v1"
CACHE_VERSION = "p2d-pooled-v1"
C, H = phase1.PHASE1_C, phase1.PHASE1_H
SPLITS = ("train", "val", "test")
BOOT_B = 5000
SEED = 0
HEADLINE_TOL = 0.05
TUNNEL_TOLS = (0.01, 0.02, 0.05, 0.10)
#: max|ours - Phase-1 cache| / std(layer) for the "same pass, same windows" identity gate. A
#: window or token mix-up is O(1); a numerical drift between two runs of the same code is ~1e-6.
IDENTITY_TOL = 1e-3
#: native curves recomputed from the saved per-window arrays vs the committed layer_metrics.csv
#: (TimesFM-3's sweep reports float32 means: measured 1.1e-7 on Electricity).
REPRO_RTOL = 1e-6

ARM_NATIVE, ARM_POOLED, ARM_CONCAT, ARM_NATIVE_MEAN = "native", "pooled", "concat", "native_mean"
ARMS_BY_MODEL = {"chronos2": (ARM_POOLED, ARM_CONCAT, ARM_NATIVE_MEAN),
                 "timesfm3": (ARM_POOLED,),
                 "tirex": (ARM_POOLED, ARM_CONCAT, ARM_NATIVE_MEAN)}
ALL_ARMS = (ARM_NATIVE, ARM_CONCAT, ARM_NATIVE_MEAN, ARM_POOLED)
NATIVE_K = {"chronos2": 4, "timesfm3": 1, "tirex": 2}

#: (step name, from arm, to arm, what changes). The chain native -> concat -> native_mean ->
#: pooled changes exactly one factor per step; TimesFM-3 has only the last row.
DECOMPOSITION = (
    ("readout_structure", ARM_NATIVE, ARM_CONCAT,
     "same token states; native patch-shared head -> one unshared full-horizon head"),
    ("pooling", ARM_CONCAT, ARM_NATIVE_MEAN,
     "same tokens and head form; K readout states concatenated -> averaged. NOT pure pooling: the input shrinks K*d -> d, so the readout capacity also drops K-fold"),
    ("token_selection", ARM_NATIVE_MEAN, ARM_POOLED,
     "same head and parameter count; native readout tokens -> mean of content tokens. The cleanest comparison: only the choice of tokens changes"),
    ("representation", ARM_NATIVE, ARM_POOLED,
     "TimesFM-3 only: same head and parameter count; token 15 -> mean of tokens 0..15"),
)

POOLING = {
    "chronos2": {"pass": "the Phase-1 num_output_patches=4 encoder pass (bidirectional)",
                 "pooled_tokens": "context patches 0..31 (all fully observed)",
                 "excluded_tokens": "REG token (32) and the 4 forecast slots (33..36)",
                 "source": "Phase-1 K4 cache content_L* (mean over [:ncp]); no new extraction"},
    "timesfm3": {"pass": "one full-context decode() pass (causal attention)",
                 "pooled_tokens": "tokens 0..15 (the 16 real context patches)",
                 "excluded_tokens": "the 2 horizon placeholder tokens (16, 17)",
                 "source": "new extraction; token 15 of the same pass checked against Phase 1"},
    "tirex": {"pass": "pass 0 of the package-default two_pass rollout (causal recurrence)",
              "pooled_tokens": "pass-0 tokens 48..63 (the 16 real context patches; token 63 is "
                               "native readout state 0)",
              "excluded_tokens": "pass-0 left-pad tokens 0..47; ALL of pass 1 (same 512 values "
                                 "one pad step earlier, plus an appended missing patch)",
              "source": "new extraction; both token-63 readouts checked against Phase 1"},
}

CAUSALITY_NOTE = (
    "TimesFM-3 (causal attention) and TiRex (causal recurrence): content token j has seen only "
    "patches <= j, so the content mean mixes states conditioned on truncated history; only the "
    "last token has seen all 512 points. Chronos-2's encoder is bidirectional: every content "
    "token sees the whole context (and the forecast-slot placeholders, which carry no values).")

REQUIRED = ("cell_config.json", "summary.json", "layer_metrics.csv", "comparisons.csv",
            "tunnels.json", "wd_selection.csv", "bootstrap_inputs.npz", "gates.json")


# --------------------------------------------------------------------------- #
# readout capacity -- reported for EVERY configuration
# --------------------------------------------------------------------------- #
def readout_params(model: str, arm: str, Q: int = 9, horizon: int = H) -> dict:
    """In/out width, parameter count and structure of one readout configuration."""
    spec = phase1.model_spec(model)
    d, K = spec.d, NATIVE_K[model]
    if arm == ARM_NATIVE:
        if K == 1:
            fin, fout, shared, desc = d, Q * horizon, 1, f"Linear({d}, {horizon}*{Q})"
        else:
            P = horizon // K
            fin, fout, shared = d, Q * P, K
            desc = f"ONE Linear({d}, {Q}*{P}) shared across the K={K} native readout states"
    elif arm in (ARM_POOLED, ARM_NATIVE_MEAN):
        fin, fout, shared = d, Q * horizon, 1
        what = "mean of valid content tokens" if arm == ARM_POOLED else \
            f"mean of the K={K} native readout states"
        desc = f"Linear({d}, {horizon}*{Q}) on the {what}"
    elif arm == ARM_CONCAT:
        if K == 1:
            raise ValueError(f"{model}: K=1, the concatenated control IS the native readout")
        fin, fout, shared = K * d, Q * horizon, 1
        desc = f"Linear({K}*{d}, {horizon}*{Q}) on the K={K} native states concatenated"
    else:
        raise ValueError(f"unknown arm {arm!r}")
    return {"model": model, "arm": arm, "in_features": int(fin), "out_features": int(fout),
            "n_params": int(fin * fout + fout), "head_applications_per_window": int(shared),
            "train_rows_per_window": int(shared), "description": desc}


def param_table(Q: int = 9) -> list[dict]:
    rows = []
    for m in phase1.MODELS:
        nat = readout_params(m, ARM_NATIVE, Q)["n_params"]
        for arm in (ARM_NATIVE,) + ARMS_BY_MODEL[m]:
            r = readout_params(m, arm, Q)
            r["ratio_to_native"] = r["n_params"] / nat
            rows.append(r)
    return rows


# --------------------------------------------------------------------------- #
# token masks + masked mean
# --------------------------------------------------------------------------- #
def assert_contexts(X, Cc: int = C) -> np.ndarray:
    """(n, C) raw CONTEXTS only. A parent window (context ++ future) is refused, so the future
    can never reach a backbone through a pooled extractor."""
    X = np.asarray(X, np.float32)
    if X.ndim != 2 or X.shape[1] != Cc:
        raise ValueError(f"expected (n, {Cc}) raw CONTEXTS (no future values), got {X.shape}")
    return X


def _patch_observed(X, P: int):
    """(n, T) -> (n, T // P): True where every value of the patch is finite."""
    n, T = X.shape
    if T % P:
        raise ValueError(f"length {T} is not a multiple of the patch size {P}")
    return np.isfinite(np.asarray(X, np.float64)).reshape(n, T // P, P).all(axis=-1)


def chronos2_token_mask(X, patch: int = 16, n_reg: int = 1, K: int = 4) -> np.ndarray:
    """Chronos-2 sequence [content(ncp) | REG | K forecast slots] -> valid content tokens."""
    obs = _patch_observed(assert_contexts(X), patch)
    return np.concatenate([obs, np.zeros((len(obs), n_reg + K), bool)], axis=1)


def timesfm3_token_mask(X, P: int = 32, n_tokens: int = 18) -> np.ndarray:
    """TimesFM-3 decode() sequence [16 real context patches | horizon placeholders]."""
    obs = _patch_observed(assert_contexts(X), P)
    return np.concatenate([obs, np.zeros((len(obs), n_tokens - obs.shape[1]), bool)], axis=1)


def tirex_token_mask(padded, P: int = 32):
    """The context AS THE MODEL PADS IT (left NaN pad to train_ctx_len, plus any appended
    missing patch) -> (n, n_tokens) valid tokens. Works on torch or numpy."""
    try:
        import torch
        if torch.is_tensor(padded):
            b, L = padded.shape
            return torch.isfinite(padded).reshape(b, L // P, P).all(dim=-1)
    except ImportError:                                            # pragma: no cover
        pass
    return _patch_observed(padded, P)


def assert_token_positions(mask, expected, what: str) -> dict:
    """Every window's valid tokens must be EXACTLY ``expected``. Never assumed, never repaired."""
    m = np.asarray(mask.cpu() if hasattr(mask, "cpu") else mask, bool)
    exp = np.zeros(m.shape[1], bool)
    exp[list(expected)] = True
    bad = np.flatnonzero((m != exp).any(axis=1))
    if bad.size:
        raise RuntimeError(f"{what}: {bad.size}/{len(m)} windows have a valid-token set other than "
                           f"{list(expected)[0]}..{list(expected)[-1]} (first bad window "
                           f"{bad[0]}: {np.flatnonzero(m[bad[0]]).tolist()}). The window filter "
                           "promised fully observed contexts -- refusing to pool.")
    return {"n_windows": int(len(m)), "valid_tokens_per_window": int(exp.sum()),
            "first_valid_token": int(np.flatnonzero(exp)[0]),
            "last_valid_token": int(np.flatnonzero(exp)[-1]), "n_tokens": int(m.shape[1])}


def masked_mean(h, mask):
    """(b, T, d) states x (b, T) mask -> (b, d) float32 mean over the valid tokens, accumulated
    in float64. Refuses a window with no valid token."""
    try:
        import torch
        if torch.is_tensor(h):
            m = torch.as_tensor(mask, device=h.device).to(torch.float64)
            cnt = m.sum(dim=1)
            if bool((cnt == 0).any()):
                raise RuntimeError("masked_mean: a window has no valid token")
            s = (h.to(torch.float64) * m[..., None]).sum(dim=1)
            return (s / cnt[:, None]).to(torch.float32)
    except ImportError:                                            # pragma: no cover
        pass
    h = np.asarray(h, np.float64)
    m = np.asarray(mask, np.float64)
    cnt = m.sum(axis=1)
    if np.any(cnt == 0):
        raise RuntimeError("masked_mean: a window has no valid token")
    return ((h * m[..., None]).sum(axis=1) / cnt[:, None]).astype(np.float32)


def state_identity(ours: dict, ref: dict) -> dict:
    """Per point: max|ours - ref|, relative to the reference layer's std; bitwise flag."""
    out = {}
    for k in ours:
        a, b = np.asarray(ours[k], np.float64), np.asarray(ref[k], np.float64)
        if a.shape != b.shape:
            raise RuntimeError(f"identity {k}: shape {a.shape} vs Phase-1 {b.shape}")
        mx = float(np.abs(a - b).max()) if a.size else 0.0
        sd = float(b.std()) + 1e-12
        out[str(k)] = {"max_abs": mx, "rel_to_std": mx / sd, "bitwise": bool(np.array_equal(a, b))}
    worst = max(out.values(), key=lambda r: r["rel_to_std"])
    return {"per_point": out, "worst_rel_to_std": worst["rel_to_std"],
            "all_bitwise": all(r["bitwise"] for r in out.values()), "tol": IDENTITY_TOL}


def assert_identity(rec: dict, what: str) -> dict:
    if not rec["worst_rel_to_std"] <= rec.get("tol", IDENTITY_TOL):
        raise RuntimeError(f"{what}: our pass does not reproduce the Phase-1 cached states "
                           f"(worst max|d|/std {rec['worst_rel_to_std']:.3e} > "
                           f"{rec.get('tol', IDENTITY_TOL):g}). Not the same pass / windows / batch "
                           "-- refusing to compare a pooled readout against it.")
    return rec


# --------------------------------------------------------------------------- #
# the pooled cache (Phase-2d namespace only)
# --------------------------------------------------------------------------- #
def _slug(label) -> str:
    return str(label).replace("+", "_")


def window_hash(X) -> str:
    a = np.ascontiguousarray(np.asarray(X, np.float32))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]


def pooled_cache_dir(root, model: str, tag: str, split: str, suite: str = "paper14") -> Path:
    if not root:
        raise ValueError("no pooled cache root (pass --pooled-cache-root or set $SCRATCH)")
    return Path(root) / f"{CACHE_VERSION}__{model}__{suite}__{tag}__{split}"


def save_pooled_cache(d: Path, feats: dict, meta: dict, aux: dict) -> Path:
    """Atomic: written into a temp dir, then os.replace'd. ``meta.json`` last."""
    d = Path(d)
    tmp = d.with_name(d.name + f".tmp{os.getpid()}")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    for label, arr in feats.items():
        np.save(tmp / f"{_slug(label)}.npy", np.asarray(arr, np.float32))
    (tmp / "aux.json").write_text(json.dumps(aux, indent=2, default=_jd))
    (tmp / "meta.json").write_text(json.dumps(meta, indent=2, default=_jd))
    if d.exists():
        shutil.rmtree(d)
    os.replace(tmp, d)
    return d


def load_pooled_cache(d: Path, meta_expected: dict):
    """(feats {label: (n, dim)}, aux) or None when absent. RAISES on any metadata disagreement."""
    d = Path(d)
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    diff = {k: (meta.get(k), v) for k, v in meta_expected.items() if meta.get(k) != v}
    if diff:
        raise RuntimeError(f"REFUSING pooled cache {d}: written under a different configuration "
                           f"{diff}. Delete it (or pass --force-extract); nothing is reused "
                           "silently.")
    feats = {}
    for label in meta["labels"]:
        a = np.load(d / f"{_slug(label)}.npy", mmap_mode="r")
        if a.shape != (meta["n_windows"], meta["dim"]):
            raise RuntimeError(f"{d}/{_slug(label)}.npy is {a.shape}, expected "
                               f"({meta['n_windows']}, {meta['dim']})")
        feats[label] = np.asarray(a, np.float32)
    return feats, json.loads((d / "aux.json").read_text())


def _jd(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    return str(o)


# --------------------------------------------------------------------------- #
# extraction (GPU) -- TimesFM-3 and TiRex. Chronos-2 needs none.
# --------------------------------------------------------------------------- #
def extract_timesfm3_pooled(X, *, model, device, geom, batch_size: int, detrend: bool = True,
                            progress: bool = True) -> dict:
    """ONE decode() pass per batch, exactly the Phase-1 call. Per layer it returns the masked
    mean over the real context tokens AND token 15 (for the identity gate), plus the RevIN
    statistics at token 15 (the target space)."""
    import torch
    from probing.timesfm3 import _hooks, detrending
    from probing.timesfm3_last_token import (MODEL_DIMS, NUM_LAYERS, assert_backbone_frozen,
                                             assert_native_geometry, assert_native_quantiles,
                                             no_quantile_sorting, select_last_token)
    X = assert_contexts(X, geom.C)
    assert_backbone_frozen(model)
    assert_native_quantiles(model)
    assert_native_geometry(model, geom)
    n, tok = len(X), geom.selected_token_index
    mask_all = timesfm3_token_mask(X, geom.P, geom.n_tokens)
    tokens = assert_token_positions(mask_all, range(geom.n_real_context_patches), "timesfm3")
    pooled = {L: np.zeros((n, MODEL_DIMS), np.float32) for L in range(NUM_LAYERS)}
    last = {L: np.zeros((n, MODEL_DIMS), np.float32) for L in range(NUM_LAYERS)}
    mu, sd = np.full(n, np.nan), np.full(n, np.nan)
    caps: dict = {}
    hs = _hooks(model, caps)
    try:
        with detrending(model, detrend), no_quantile_sorting(model, True) as srt:
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                tgt = torch.from_numpy(X[s:e]).to(device).unsqueeze(1)          # (b, 1, C)
                caps.clear()
                with torch.no_grad():
                    _off, aux = model.decode(target=tgt, horizon=geom.H, return_aux_outputs=True,
                                             **srt.decode_kwargs)
                m = torch.as_tensor(mask_all[s:e])
                for L in range(NUM_LAYERS):
                    h = caps[f"L{L}"]                                            # (b,1,18,1280)
                    last[L][s:e] = select_last_token(
                        h, tok, n_real_context_patches=geom.n_real_context_patches,
                        n_tokens=geom.n_tokens).float().cpu().numpy()
                    pooled[L][s:e] = masked_mean(h[:, 0], m).cpu().numpy()
                run_mu, run_sd = aux["revin_stats"]
                mu[s:e] = run_mu[:, 0, tok].float().cpu().numpy()
                sd[s:e] = run_sd[:, 0, tok].float().cpu().numpy()
                if progress and (s // max(batch_size, 1)) % 10 == 0:
                    print(f"  [pooled extract timesfm3] {e}/{n}", flush=True)
    finally:
        for hk in hs:
            hk.remove()
    if np.isnan(mu).any() or np.isnan(sd).any():
        raise RuntimeError("timesfm3 pooled extraction left NaN RevIN statistics")
    return {"pooled": pooled, "last_token": last, "mu": mu, "sd": sd, "tokens": tokens}


def extract_tirex_pooled(X, *, model, geom, batch_size: int, progress: bool = False) -> dict:
    """The package-default two_pass rollout (``native_forward_two_pass``, the Phase-1 call).
    Pools pass 0's real context tokens; records both passes' token-63 readouts for the identity
    gate and how far pass 1's real-token mean sits from pass 0's (the near-duplicate claim)."""
    import torch
    from probing.tirex_model import (MODEL_DIMS, REP_NAMES, model_device, native_forward_two_pass,
                                     readouts_from_reps, rollout_contexts, to_model_device)
    X = assert_contexts(X, geom.C)
    n, K = len(X), geom.n_forecast_patches
    lay = [geom.two_pass_layout(k) for k in range(K)]
    exp = [range(L["real_token_range"][0], L["real_token_range"][1] + 1) for L in lay]
    dev = model_device(model)
    pooled = {nm: np.zeros((n, MODEL_DIMS), np.float32) for nm in REP_NAMES}
    readouts = {nm: np.zeros((n, K, MODEL_DIMS), np.float32) for nm in REP_NAMES}
    dup = {nm: {"max_abs": 0.0} for nm in REP_NAMES}
    p0_abs_max = {nm: 0.0 for nm in REP_NAMES}
    tokens = None
    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        ctx = torch.as_tensor(X[s:e]).to(dev)
        _off, per_pass, _st, _lay = native_forward_two_pass(model, ctx, geom, capture=True)
        masks = []
        with torch.inference_mode():
            for k, c in enumerate(rollout_contexts(ctx, geom)):
                adj, _pad = model._adjust_context_length(to_model_device(model, c),
                                                         geom.train_ctx_len, geom.train_ctx_len)
                mk = tirex_token_mask(adj, geom.input_patch)
                rec = assert_token_positions(mk, exp[k], f"tirex pass {k}")
                if k == 0:
                    tokens = rec
                masks.append(mk)
        ro = readouts_from_reps(per_pass, geom, "two_pass")
        for nm in REP_NAMES:
            p0 = masked_mean(per_pass[0][nm], masks[0])
            p1 = masked_mean(per_pass[1][nm], masks[1])
            pooled[nm][s:e] = p0.cpu().numpy()
            readouts[nm][s:e] = ro[nm].float().cpu().numpy()
            dup[nm]["max_abs"] = max(dup[nm]["max_abs"], float((p1 - p0).abs().max()))
            p0_abs_max[nm] = max(p0_abs_max[nm], float(p0.abs().max()))
        if progress and (s // max(batch_size, 1)) % 10 == 0:
            print(f"  [pooled extract tirex] {e}/{n}", flush=True)
    for nm in REP_NAMES:
        sd = float(pooled[nm].std()) + 1e-12
        dup[nm]["rel_to_std"] = dup[nm]["max_abs"] / sd
    return {"pooled": pooled, "readouts": readouts, "tokens": tokens,
            "pass1_real_token_mean_vs_pass0": dup,
            "pass_layouts": lay}


def chronos2_hidden_states(pipeline, X, K: int = 4):
    """The Phase-1 Chronos-2 forward (``extract_kout_features``' hooks + ``model.encode``) on
    one batch: 13 block-hook states (L0 = embedding, L1..L12) and the final-norm state, all
    (b, P, 768) on the CPU. Used only by the one-batch check."""
    import torch
    from probing.config import NUM_LAYERS
    X = assert_contexts(X)
    model = pipeline.model
    device = next(model.parameters()).device
    cap: dict = {}

    def mk(i):
        def hook(_m, _inp, out):
            hs = out.hidden_states if getattr(out, "hidden_states", None) is not None else out[0]
            cap[i] = hs.detach().cpu()
        return hook

    pre = model.encoder.block[0].register_forward_pre_hook(
        lambda _m, a: cap.__setitem__(0, a[0].detach().cpu()))
    hooks = [b.register_forward_hook(mk(i + 1)) for i, b in enumerate(model.encoder.block)]
    was = model.training
    model.eval()
    try:
        with torch.no_grad():
            enc, *_ = model.encode(context=torch.from_numpy(np.ascontiguousarray(X)).to(device),
                                   num_output_patches=K)
    finally:
        pre.remove()
        for h_ in hooks:
            h_.remove()
        model.train(was)
    return [cap[i] for i in range(NUM_LAYERS)], enc[0].detach().cpu()


# --------------------------------------------------------------------------- #
# arm features: {point index: (n, 1, D)} -- ONE slot, so the head predicts all H steps
# --------------------------------------------------------------------------- #
def arm_features(arm: str, *, pooled=None, native_states=None) -> dict:
    """``pooled`` {i: (n, d)} content means; ``native_states`` {i: (n, K, d)} readout states."""
    if arm == ARM_POOLED:
        return {i: np.asarray(a, np.float32)[:, None, :] for i, a in pooled.items()}
    out = {}
    for i, a in native_states.items():
        a = np.asarray(a, np.float32)
        if a.ndim != 3:
            raise ValueError(f"native readout states must be (n, K, d), got {a.shape}")
        n, K, d = a.shape
        if arm == ARM_CONCAT:
            out[i] = a.reshape(n, 1, K * d)          # slot-major: [state 0 | state 1 | ...]
        elif arm == ARM_NATIVE_MEAN:
            out[i] = a.astype(np.float64).mean(axis=1, keepdims=True).astype(np.float32)
        else:
            raise ValueError(f"arm {arm!r} has no native-state feature construction")
    return out


# --------------------------------------------------------------------------- #
# the layerwise fit for the Chronos-2 / TiRex arms -- the models' OWN fitters
# --------------------------------------------------------------------------- #
def fit_shared_layerwise(model: str, data: dict, *, quantiles, median_idx: int, device=None,
                         epochs: int = 300, lr: float = 1e-2, wd_grid=phase1.PHASE1_WD_GRID,
                         seed: int = SEED, verbose: bool = True, description: str = "") -> dict:
    """``data[split] = {"feats": {i: (n, K, D)}, "Y": (n, H)}`` -> the Phase-1 ``res`` schema.

    Calls the model's own Phase-1 fitter with K slots of P = H/K steps: K=1 for every Phase-2d
    arm (one vector -> the whole horizon), the native K for the parity contract. Mirrors
    ``phase1_chronos2.fit_layerwise`` / ``phase1_tirex.fit_layerwise`` line for line so a
    contract can pin it to them bitwise at the native K.
    """
    import torch
    spec = phase1.model_spec(model)
    tr, va, te = data["train"], data["val"], data["test"]
    keys = list(range(spec.n_points))
    K = int(np.shape(tr["feats"][0])[1])
    Hh = int(np.shape(tr["Y"])[1])
    if Hh % K:
        raise ValueError(f"H={Hh} is not a multiple of K={K}")
    P = Hh // K
    wd_grid = phase1.assert_wd_grid(wd_grid, lr, null_wd=phase1.PHASE1_WD_NULL)
    gmax, gmin = float(max(wd_grid)), float(min(wd_grid))
    sets = (("train", tr), ("val", va), ("test", te))

    if model == "chronos2":
        from probing.phase1_chronos2 import predict_quantiles as _pred
        from probing.probes import fit_shared_forecast_probe_explicit_val, validate_quantiles
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        q = validate_quantiles(quantiles)
        fitted = fit_shared_forecast_probe_explicit_val(
            {k: tr["feats"][k] for k in keys}, tr["Y"], {k: va["feats"][k] for k in keys},
            va["Y"], quantiles=q, epochs=epochs, lr=lr, wd_grid=tuple(wd_grid), device=device,
            init_seed=seed, output_patch_size=P)

        def predict(k, F):
            return _pred(fitted[k], F, Hh, q, device)

        def info(k, preds_train_loss):
            f = fitted[k]
            lin = f["linear"]
            return {"wd": float(f["wd"]),
                    "selection": {str(a): float(b) for a, b in
                                  (f["selection"] or {}).get("val_loss_by_wd", {}).items()},
                    "n_params": int(sum(p.numel() for p in lin.parameters())),
                    "train_loss": preds_train_loss,
                    "weights": {"weight": lin.weight.detach().cpu().numpy().astype(np.float32),
                                "bias": lin.bias.detach().cpu().numpy().astype(np.float32),
                                "scaler_mean": np.asarray(f["scaler"].mean_, np.float64),
                                "scaler_scale": np.asarray(f["scaler"].scale_, np.float64)}}
    elif model == "tirex":
        from probing.tirex_probes import fit_shared_patch_probe
        from probing.tirex_probes import predict_quantiles as _pred
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        q = np.asarray(quantiles, np.float64)
        fits = {}

        def predict(k, F):
            return _pred(fits[k], F, device=device)

        def info(k, _preds_train_loss):
            f = fits[k]
            return {"wd": float(f["wd"]),
                    "selection": {str(a): float(b)
                                  for a, b in f["selection"]["val_loss_by_wd"].items()},
                    "n_params": int(f["n_params"]), "train_loss": float(f["train_loss"]),
                    "weights": {"weight": f["probe"].weight.detach().cpu().numpy()
                                .astype(np.float32),
                                "bias": f["probe"].bias.detach().cpu().numpy().astype(np.float32),
                                "scaler_mean": np.asarray(f["scaler"].mean_, np.float64),
                                "scaler_scale": np.asarray(f["scaler"].scale_, np.float64)}}
    else:
        raise ValueError(f"{model}: use phase1_timesfm3.fit_layerwise (its head is already "
                         "one Linear(1280, H*Q) on one state)")

    res = {"labels": spec.labels, "quantiles": [float(x) for x in q], "median_index": median_idx,
           "train_loss": [], "val_loss": [], "test_loss": [], "wd": [], "selection": [],
           "n_params": [], "wd_at_grid_max": [], "wd_at_grid_min": [], "val_window_loss": [],
           "test_window_loss": [], "per_quantile_test": [], "pred_val": {}, "pred_test": {},
           "probe_weights": {}}
    for k, label in zip(keys, spec.labels):
        if model == "tirex":
            fits[k] = fit_shared_patch_probe(tr["feats"][k], tr["Y"], va["feats"][k], va["Y"],
                                             out_patch=P, wd_grid=wd_grid, epochs=epochs, lr=lr,
                                             device=device, init_seed=seed, quantiles=q)
            pred_sets = sets[1:]
        else:
            pred_sets = sets
        preds = {s: predict(k, d["feats"][k]) for s, d in pred_sets}
        pw = {s: phase1.mean_pinball_per_window(preds[s], d["Y"], q) for s, d in pred_sets}
        meta = info(k, float(pw["train"].mean()) if "train" in pw else None)
        res["train_loss"].append(float(meta["train_loss"]))
        res["val_loss"].append(float(pw["val"].mean()))
        res["test_loss"].append(float(pw["test"].mean()))
        res["val_window_loss"].append(pw["val"])
        res["test_window_loss"].append(pw["test"])
        res["per_quantile_test"].append(phase1.per_quantile_mean_loss(preds["test"], te["Y"], q))
        res["wd"].append(meta["wd"])
        res["wd_at_grid_max"].append(bool(meta["wd"] == gmax))
        res["wd_at_grid_min"].append(bool(meta["wd"] == gmin))
        res["selection"].append(meta["selection"])
        res["n_params"].append(meta["n_params"])
        res["pred_val"][label] = preds["val"]
        res["pred_test"][label] = preds["test"]
        res["probe_weights"][label] = meta["weights"]
        if verbose:
            print(f"    [probe K={K} P={P}] {label:<8s} wd={meta['wd']:<8g} "
                  f"train={res['train_loss'][-1]:.5f} val={res['val_loss'][-1]:.5f} "
                  f"test={res['test_loss'][-1]:.5f}", flush=True)
    res["val_window_loss"] = np.stack(res["val_window_loss"])
    res["test_window_loss"] = np.stack(res["test_window_loss"])
    res["probe_description"] = description or f"Linear(D, {len(q)}*{P}) over K={K} slot(s)"
    res["wd_grid"] = [float(w) for w in wd_grid]
    res["constant_forecast_floor"] = phase1.constant_forecast_floor(
        tr["Y"], {"train": tr["Y"], "val": va["Y"], "test": te["Y"]}, q)
    return res


# --------------------------------------------------------------------------- #
# the native arm -- READ from the committed Phase-1 cell, and reproduced
# --------------------------------------------------------------------------- #
def native_from_phase1(cell, spec) -> dict:
    """Everything the comparison needs about the architecture-aware readout, from artifacts."""
    a = cell.arrays()
    labels = [str(x) for x in a["labels"]]
    if labels != spec.labels:
        raise Phase1DependencyError(f"{cell.path}: labels {labels} != {spec.labels}")
    rows = sorted(csv.DictReader(open(cell.path / "layer_metrics.csv")),
                  key=lambda r: int(r["point_index"]))
    if [r["layer"] for r in rows] != spec.labels:
        raise Phase1DependencyError(f"{cell.path}/layer_metrics.csv rows do not follow the spec")
    f = lambda k: [float(r[k]) for r in rows]                            # noqa: E731
    return {"labels": labels, "train_loss": f("train_loss"), "val_loss": f("val_loss"),
            "test_loss": f("test_loss"), "wd": f("weight_decay"),
            "n_params": [int(float(r["n_probe_params"])) for r in rows],
            "val_window_loss": np.asarray(a["val_loss_window"], np.float64),
            "test_window_loss": np.asarray(a["test_loss_window"], np.float64),
            "mase_pw": np.asarray(a["test_mase_window"], np.float64),
            "mae_pw": np.asarray(a["test_mae_window"], np.float64),
            "wql_num_pw": np.asarray(a["test_wql_num_window"], np.float64),
            "wql_den_pw": np.asarray(a["test_wql_den_window"], np.float64),
            "cluster_ids_test": np.asarray(a["cluster_ids_test"], np.int64),
            "cluster_ids_val": np.asarray(a["cluster_ids_val"], np.int64),
            "forecast": {k: np.asarray(a[f"native_{k}_window"], np.float64)
                         for k in ("loss", "mase", "mae") if f"native_{k}_window" in a},
            "entrances": cell.entrances(TUNNEL_TOLS),
            "probe": cell.config().get("probe")}


def reproduce_native(native: dict, spec, tols=TUNNEL_TOLS, rtol: float = REPRO_RTOL) -> dict:
    """The model-free reproduction gate: (1) the per-window arrays average to the committed
    curves, (2) the frozen sustained-entrance rule on the committed validation curve gives the
    committed entrance at EVERY tolerance. Aborts on either mismatch."""
    v_pw = native["val_window_loss"].mean(axis=1)
    t_pw = native["test_window_loss"].mean(axis=1)
    dv = float(np.max(np.abs(v_pw / np.asarray(native["val_loss"]) - 1.0)))
    dt = float(np.max(np.abs(t_pw / np.asarray(native["test_loss"]) - 1.0)))
    if not (dv <= rtol and dt <= rtol):
        raise Phase1DependencyError(f"{spec.name}: committed per-window arrays do not average to "
                                    f"the committed curves (val {dv:.2e}, test {dt:.2e} > {rtol})")
    tun = phase1.tunnel_record(spec, native["val_loss"], native["test_loss"], tol=HEADLINE_TOL,
                               tols=tuple(tols))
    tun_pw = phase1.tunnel_record(spec, v_pw, t_pw, tol=HEADLINE_TOL, tols=tuple(tols))
    ent, ent_pw = {}, {}
    for t in tols:
        key = f"tol_{float(t):g}"
        got = tun["by_tolerance"][key]["depth_axis_index"]
        want = native["entrances"][float(t)]["depth_axis_index"]
        if got != want:
            raise Phase1DependencyError(f"{spec.name}: entrance at tol {t:g} re-derived as depth "
                                        f"{got}, committed {want}")
        ent[key] = tun["by_tolerance"][key]["label"]
        ent_pw[key] = tun_pw["by_tolerance"][key]["label"]
    return {"max_rel_val_curve": dv, "max_rel_test_curve": dt, "rtol": rtol,
            "entrances": ent, "entrances_from_per_window_means": ent_pw,
            "per_window_entrances_identical": ent == ent_pw, "tunnel": tun, "passed": True}


# --------------------------------------------------------------------------- #
# paired comparisons -- ONE cluster resampling for both arms
# --------------------------------------------------------------------------- #
def paired_rows(metric: str, split: str, a_name: str, a_pw, b_name: str, b_pw, cluster_ids,
                spec, *, B: int = BOOT_B, seed: int = SEED, step: str = "") -> list[dict]:
    """Per point: mean_a, mean_b, delta = b - a and rel = b/a - 1 with paired 95% cluster CIs.
    Positive delta = the ``b`` arm has the HIGHER (worse) loss / MASE."""
    from probing.phase2 import cluster_replicates
    A, Bm = np.asarray(a_pw, np.float64), np.asarray(b_pw, np.float64)
    if A.shape != Bm.shape or A.shape[1] != len(cluster_ids):
        raise ValueError(f"paired arrays must be aligned: {A.shape} vs {Bm.shape}, "
                         f"{len(cluster_ids)} cluster ids")
    Pn = A.shape[0]
    reps = cluster_replicates(np.vstack([A, Bm]), cluster_ids, B, seed)
    ra, rb = reps[:, :Pn], reps[:, Pn:]
    ma, mb = A.mean(axis=1), Bm.mean(axis=1)
    dlo, dhi = np.percentile(rb - ra, [2.5, 97.5], axis=0)
    rlo, rhi = np.percentile(rb / ra - 1.0, [2.5, 97.5], axis=0)
    rows = []
    for i, p in enumerate(spec.points):
        rows.append({"metric": metric, "split": split, "step": step, "from_arm": a_name,
                     "to_arm": b_name, "point_index": i, "layer": p.label,
                     "point_type": p.point_type, "include_in_main_depth_axis": p.on_depth_axis,
                     "is_reference_point": i == spec.reference_index,
                     "mean_from": float(ma[i]), "mean_to": float(mb[i]),
                     "delta_abs": float(mb[i] - ma[i]), "delta_abs_ci_lo": float(dlo[i]),
                     "delta_abs_ci_hi": float(dhi[i]),
                     "delta_rel": float(mb[i] / ma[i] - 1.0), "delta_rel_ci_lo": float(rlo[i]),
                     "delta_rel_ci_hi": float(rhi[i]),
                     "ci_excludes_zero": bool(dlo[i] > 0 or dhi[i] < 0),
                     "direction": ("to_worse" if dlo[i] > 0 else "to_better" if dhi[i] < 0
                                   else "inconclusive"),
                     "n_clusters": int(len(np.unique(cluster_ids))), "B": int(B)})
    return rows


def comparison_pairs(model: str, arms=None) -> list[tuple[str, str, str]]:
    """(step, from, to): every fitted arm vs native, plus the one-factor decomposition steps.

    native -> concat IS the readout-structure step and (TimesFM-3) native -> pooled IS the
    representation step, so those vs-native rows carry the step name instead of a duplicate."""
    arms = tuple(arms or ARMS_BY_MODEL[model])
    have = {ARM_NATIVE, *arms}
    pairs = []
    for b in arms:
        if b == ARM_CONCAT:
            step = "readout_structure"
        elif model == "timesfm3" and b == ARM_POOLED:
            step = "representation"
        else:
            step = "vs_native"
        pairs.append((step, ARM_NATIVE, b))
    for step, a, b, _ in DECOMPOSITION:
        if step in ("pooling", "token_selection") and a in have and b in have:
            pairs.append((step, a, b))
    return pairs


# --------------------------------------------------------------------------- #
# per-arm summary
# --------------------------------------------------------------------------- #
def arm_summary(arm: str, model: str, spec, curves: dict, tun: dict, boot_loss: dict,
                boot_mase: dict, params: dict, description: str, floor: dict | None = None,
                wd_at_grid_max=None) -> dict:
    ri = spec.reference_index
    last_pos = spec.n_depth_points - 1
    h = tun["headline"]
    ent = {k: {"label": e["label"], "depth_axis_index": int(e["depth_axis_index"]),
               "relative_depth": e["relative_depth"]} for k, e in tun["by_tolerance"].items()}
    ei = int(h["index"])
    out = {"arm": arm, "readout": description, "params": params,
           "entrance": ent, "headline_entrance": h["label"],
           "headline_depth_axis_index": int(h["depth_axis_index"]),
           "headline_relative_depth": h["relative_depth"],
           "pre_final_entrance": bool(h["depth_axis_index"] < last_pos),
           "final_block": {"label": spec.reference_label,
                           "val_loss": float(curves["val_loss"][ri]),
                           "test_loss": float(curves["test_loss"][ri]),
                           "test_loss_ci": [boot_loss["ci_lo"][ri], boot_loss["ci_hi"][ri]],
                           "test_mase": float(boot_mase["point"][ri]),
                           "test_mase_ci": [boot_mase["ci_lo"][ri], boot_mase["ci_hi"][ri]]},
           "at_entrance": {"label": h["label"], "val_loss": float(curves["val_loss"][ei]),
                           "test_loss": float(curves["test_loss"][ei]),
                           "test_mase": float(boot_mase["point"][ei])},
           "val_argmin_point": tun.get("val_argmin_point"),
           "test_argmin_point": tun.get("test_argmin_point")}
    if wd_at_grid_max is not None:
        out["wd_at_grid_max_count"] = int(sum(bool(x) for x in wd_at_grid_max))
    if floor:
        out["constant_forecast_floor"] = floor
        if floor.get("val_loss"):
            out["final_block"]["val_over_floor"] = float(curves["val_loss"][ri]) / floor["val_loss"]
    return out


def config_hash(cfg: dict) -> str:
    keep = {k: v for k, v in cfg.items() if k not in ("phase1_root", "config_hash")}
    return "sha256:" + hashlib.sha256(json.dumps(keep, sort_keys=True, default=str)
                                      .encode()).hexdigest()


# --------------------------------------------------------------------------- #
# per-model data: Phase-1 caches READ-ONLY, pooled states from the Phase-2d cache
# --------------------------------------------------------------------------- #
CHRONOS2_INPUT_PATCH = 16          # 512 / 16 = 32 context patches; the one-batch check verifies
CHRONOS2_N_REG = 1                 # the realized sequence is 32 + 1 + 4 = 37 tokens


def _raw_future(w, split, X):
    from probing.timesfm3 import raw_future_from_arcsinh
    return raw_future_from_arcsinh(X, w[f"Y_{split}_traj"], w["meta"]["sigma_eps"])


def load_chronos2(tag: str, w: dict, phase1_arrays: dict, *, cache_dir=None,
                  splits=SPLITS) -> dict:
    """Content means AND forecast-slot states from the SAME Phase-1 K=4 cache file.

    The cache's ``content_L*`` is ``hs[:, :ncp].mean`` -- an unmasked mean over the 32 context
    patches. It equals the masked mean exactly when every patch is fully observed, which is
    asserted here per window (the window filter guarantees it), so no re-extraction is needed."""
    from probing.config import NUM_LAYERS
    from probing.phase2_pathways import _resolve_chronos2_cache, check_cluster_ids
    spec = phase1.model_spec("chronos2")
    ncp, K = C // CHRONOS2_INPUT_PATCH, NATIVE_K["chronos2"]
    out, gates, prov = {}, {"tokens": {}}, {}
    for split in splits:
        X = assert_contexts(w[f"X_{split}"])
        Y = np.asarray(w[f"Y_{split}_traj"], np.float32)
        mask = chronos2_token_mask(X, CHRONOS2_INPUT_PATCH, CHRONOS2_N_REG, K)
        gates["tokens"][split] = assert_token_positions(mask, range(ncp), f"chronos2/{split}")
        path, _tried = _resolve_chronos2_cache(tag, split, Y, cache_dir)
        with np.load(path, allow_pickle=True) as z:
            if (str(z["slot_tokens"]) if "slot_tokens" in z.files else "forecast") != "forecast":
                raise Phase1DependencyError(f"{path.name} does not hold the native forecast slots")
            content = {i: np.asarray(z[f"content_L{i}"], np.float32) for i in range(NUM_LAYERS)}
            fslot = {i: np.asarray(z[f"fslot_L{i}"], np.float32) for i in range(NUM_LAYERS)}
            content[NUM_LAYERS] = np.asarray(z["content_final"], np.float32)
            fslot[NUM_LAYERS] = np.asarray(z["fslot_final"], np.float32)
        if len(content) != spec.n_points:
            raise RuntimeError(f"chronos2: {len(content)} points, spec has {spec.n_points}")
        for i in content:
            if content[i].shape != (len(X), spec.d) or fslot[i].shape != (len(X), K, spec.d):
                raise Phase1DependencyError(f"{path.name} point {i}: content {content[i].shape}, "
                                            f"fslot {fslot[i].shape}")
        cid = np.asarray(w[f"series_{split}"], np.int64)
        if split in ("val", "test"):
            check_cluster_ids("chronos2", tag, split, cid, phase1_arrays)
        out[split] = {"pooled": content, "native_states": fslot, "X": X, "Y": Y,
                      "y_raw": _raw_future(w, split, X), "series": cid}
        prov[split] = str(path)
    gates["content_mean_equals_masked_mean"] = ("every window has all 32 context patches fully "
                                                "observed, so the cached unmasked mean over "
                                                "[:32] IS the masked mean")
    return {"splits": out, "gates": gates, "provenance": {"phase1_k4_cache": prov},
            "geom": None}


def load_timesfm3(tag: str, w: dict, p1cfg: dict, phase1_arrays: dict, *, phase1_cache_root,
                  pooled_cache_root, model=None, device=None, force_extract: bool = False,
                  splits=SPLITS, progress: bool = True) -> dict:
    """Phase-1 token-15 cache (READ-ONLY: targets, RevIN stats, native forecast) + the pooled
    states of the same decode() pass (Phase-2d cache, extracted on a miss)."""
    from probing.timesfm3_last_token import (LastTokenGeometry, assert_target_roundtrip,
                                             build_last_token_targets, cache_metadata, cache_root,
                                             read_cache, timesfm_version)
    spec = phase1.model_spec("timesfm3")
    geom = LastTokenGeometry(C=C, H=H, strict=True)
    layers = list(range(spec.n_points))
    detrend = bool(p1cfg.get("detrend", True))
    suite = p1cfg.get("suite", "paper14")
    dtype = np.dtype(p1cfg.get("feature_dtype", "float32"))
    bs = int(p1cfg.get("extract_batch_size", 64))
    seed = int(p1cfg.get("seed", 0))
    out, gates, prov = {}, {"identity_token15": {}, "revin": {}, "tokens": {}, "roundtrip": {}}, {}
    for split in splits:
        X = assert_contexts(w[f"X_{split}"])
        meta1 = cache_metadata(tag, split, geom, checkpoint=p1cfg["checkpoint"], detrend=detrend,
                               layers=layers, seed=seed, feature_dtype=dtype, suite=suite,
                               n_windows=len(X))
        root1 = cache_root(phase1_cache_root, tag, split, geom, detrend, suite)
        hit1 = read_cache(root1, meta1, X, layers)
        if hit1 is None:
            raise Phase1DependencyError(f"timesfm3/{tag}/{split}: no Phase-1 last-token cache at "
                                        f"{root1} (Phase 2d never writes there)")
        pmeta = {"cache_version": CACHE_VERSION, "model": "timesfm3", "dataset": tag,
                 "split": split, "suite": suite, "checkpoint": p1cfg["checkpoint"],
                 "timesfm_version": timesfm_version(), "detrend": detrend, "batch_size": bs,
                 "n_windows": int(len(X)), "window_hash": window_hash(X),
                 "labels": spec.labels, "dim": spec.d,
                 "pooling": POOLING["timesfm3"]["pooled_tokens"]}
        pdir = pooled_cache_dir(pooled_cache_root, "timesfm3", tag, split, suite)
        got = None if force_extract else load_pooled_cache(pdir, pmeta)
        if got is None:
            if model is None:
                raise RuntimeError(f"timesfm3/{tag}/{split}: no pooled cache at {pdir} and no "
                                   "model loaded to build one")
            ex = extract_timesfm3_pooled(X, model=model, device=device, geom=geom,
                                         batch_size=bs, detrend=detrend, progress=progress)
            ident = state_identity(ex["last_token"], {L: hit1["feats"][L] for L in layers})
            mu_rel = float(np.max(np.abs(ex["mu"] - hit1["mu"]) / (np.abs(hit1["mu"]) + 1e-12)))
            sd_rel = float(np.max(np.abs(ex["sd"] - hit1["sd"]) / (np.abs(hit1["sd"]) + 1e-12)))
            aux = {"identity_token15": ident, "revin_mu_max_rel": mu_rel,
                   "revin_sd_max_rel": sd_rel, "tokens": ex["tokens"]}
            assert_identity(ident, f"timesfm3/{tag}/{split} token 15")
            if not (mu_rel <= 1e-5 and sd_rel <= 1e-5):
                raise RuntimeError(f"timesfm3/{tag}/{split}: RevIN statistics differ from Phase 1 "
                                   f"(mu {mu_rel:.2e}, sd {sd_rel:.2e}) -- not the same target "
                                   "space")
            save_pooled_cache(pdir, {spec.labels[L]: ex["pooled"][L] for L in layers}, pmeta, aux)
            got = load_pooled_cache(pdir, pmeta)
        feats_by_label, aux = got
        assert_identity(aux["identity_token15"], f"timesfm3/{tag}/{split} token 15 (stored)")
        gates["identity_token15"][split] = {k: aux["identity_token15"][k] for k in
                                            ("worst_rel_to_std", "all_bitwise", "tol")}
        gates["revin"][split] = {"mu_max_rel": aux["revin_mu_max_rel"],
                                 "sd_max_rel": aux["revin_sd_max_rel"]}
        gates["tokens"][split] = aux["tokens"]
        Z = np.concatenate([X.astype(np.float64), _raw_future(w, split, X)], axis=1)
        t = build_last_token_targets(Z, hit1["mu"], hit1["sd"], geom, detrend=detrend)
        gates["roundtrip"][split] = float(assert_target_roundtrip(
            Z, t["targets"], t["trend"], hit1["mu"], hit1["sd"], geom, t["valid"]))
        out[split] = {"feats": {L: feats_by_label[spec.labels[L]] for L in layers},
                      "last_token_feats": {L: np.asarray(hit1["feats"][L], np.float32)
                                           for L in layers},
                      "mu": hit1["mu"], "sd": hit1["sd"], "native": hit1["native"], "X": X,
                      "Z": Z, "targets": t["targets"], "trend": t["trend"], "valid": t["valid"],
                      "series": np.asarray(w[f"series_{split}"], np.int64),
                      "y_raw": np.asarray(Z, np.float64)[:, geom.target_start:geom.target_end]}
        prov[split] = {"phase1_cache": str(root1), "pooled_cache": str(pdir)}
    return {"splits": out, "gates": gates, "provenance": prov, "geom": geom}


def load_tirex(tag: str, w: dict, p1cfg: dict, phase1_arrays: dict, *, phase1_cache_root,
               pooled_cache_root, model, geom, force_extract: bool = False, splits=SPLITS,
               progress: bool = False) -> dict:
    """Phase-1 two_pass readout cache (READ-ONLY) + pass-0 content means of the same rollout
    (Phase-2d cache, extracted on a miss). Targets are the per-pass normalized ones."""
    import torch
    from probing.phase2_pathways import check_cluster_ids
    from probing.tirex_model import (REP_NAMES, build_targets, cache_metadata, cache_root,
                                     read_cache, tirex_version)
    spec = phase1.model_spec("tirex")
    if list(REP_NAMES) != spec.labels:
        raise RuntimeError(f"tirex REP_NAMES {list(REP_NAMES)} != Phase-1 labels {spec.labels}")
    mode = p1cfg.get("rollout_mode", "two_pass")
    backend = p1cfg.get("backend", "torch")
    bs = int(p1cfg.get("extract_batch_size", 256))
    seed = int(p1cfg.get("seed", 0))
    out, gates, prov = {}, {"identity_readouts": {}, "tokens": {}, "pass1_duplicate": {}}, {}
    for split in splits:
        X = assert_contexts(w[f"X_{split}"])
        y_raw = _raw_future(w, split, X)
        meta1 = cache_metadata(tag, split, geom, checkpoint=p1cfg["checkpoint"], backend=backend,
                               points=list(REP_NAMES), seed=seed, X=X, mode=mode, batch_size=bs)
        root1 = cache_root(phase1_cache_root, tag, split, mode)
        got1 = read_cache(root1, meta1, list(REP_NAMES))
        if got1 is None:
            raise Phase1DependencyError(f"tirex/{tag}/{split}: no Phase-1 feature cache at "
                                        f"{root1}.npz (Phase 2d never writes there)")
        named, native_fc, _m = got1
        pmeta = {"cache_version": CACHE_VERSION, "model": "tirex", "dataset": tag,
                 "split": split, "suite": p1cfg.get("suite", "paper14"),
                 "checkpoint": p1cfg["checkpoint"], "tirex_version": tirex_version(),
                 "torch_version": torch.__version__, "backend": backend, "rollout_mode": mode,
                 "batch_size": bs, "n_windows": int(len(X)), "window_hash": window_hash(X),
                 "labels": spec.labels, "dim": spec.d,
                 "pooling": POOLING["tirex"]["pooled_tokens"]}
        pdir = pooled_cache_dir(pooled_cache_root, "tirex", tag, split, pmeta["suite"])
        got = None if force_extract else load_pooled_cache(pdir, pmeta)
        if got is None:
            if model is None:
                raise RuntimeError(f"tirex/{tag}/{split}: no pooled cache and no model loaded")
            ex = extract_tirex_pooled(X, model=model, geom=geom, batch_size=bs, progress=progress)
            ident = state_identity(ex["readouts"], named)
            assert_identity(ident, f"tirex/{tag}/{split} token-63 readouts")
            aux = {"identity_readouts": ident, "tokens": ex["tokens"],
                   "pass1_real_token_mean_vs_pass0": ex["pass1_real_token_mean_vs_pass0"],
                   "pass_layouts": ex["pass_layouts"]}
            save_pooled_cache(pdir, ex["pooled"], pmeta, aux)
            got = load_pooled_cache(pdir, pmeta)
        feats_by_label, aux = got
        assert_identity(aux["identity_readouts"], f"tirex/{tag}/{split} readouts (stored)")
        gates["identity_readouts"][split] = {k: aux["identity_readouts"][k] for k in
                                             ("worst_rel_to_std", "all_bitwise", "tol")}
        gates["tokens"][split] = aux["tokens"]
        gates["pass1_duplicate"][split] = {
            k: v["rel_to_std"] for k, v in aux["pass1_real_token_mean_vs_pass0"].items()}
        tgt, loc, scale = build_targets(X, y_raw, model, geom, mode=mode)
        cid = np.asarray(w[f"series_{split}"], np.int64)
        if split in ("val", "test"):
            check_cluster_ids("tirex", tag, split, cid, phase1_arrays)
        out[split] = {"pooled": {i: feats_by_label[lb] for i, lb in enumerate(spec.labels)},
                      "native_states": {i: np.asarray(named[lb], np.float32)
                                        for i, lb in enumerate(spec.labels)},
                      "native_by_label": named, "Y": np.asarray(tgt, np.float32),
                      "targets": np.asarray(tgt, np.float32), "loc": loc, "scale": scale,
                      "X": X, "y_raw": np.asarray(y_raw, np.float64), "series": cid,
                      "native": native_fc}
        prov[split] = {"phase1_cache": str(root1) + ".npz", "pooled_cache": str(pdir)}
    return {"splits": out, "gates": gates, "provenance": prov, "geom": geom}


# --------------------------------------------------------------------------- #
# one-batch checks (model-backed, run by the smoke job for all three models)
# --------------------------------------------------------------------------- #
def _refuses_future(fn) -> bool:
    try:
        fn()
    except ValueError:
        return True
    return False


def one_batch_chronos2(tag: str, w: dict, *, pipeline, cache_dir=None, split: str = "test",
                       batch: int = 128) -> dict:
    """Recompute Phase-1 batch 0 with hooks: the masked content mean must equal the cached
    ``content_L*`` and the slots the cached ``fslot_L*`` (same pass, same tokens)."""
    from probing.config import NUM_LAYERS
    from probing.phase2_pathways import _resolve_chronos2_cache
    X = assert_contexts(w[f"X_{split}"])[:batch]
    Y = np.asarray(w[f"Y_{split}_traj"], np.float32)
    K, ncp = NATIVE_K["chronos2"], C // CHRONOS2_INPUT_PATCH
    hs, final = chronos2_hidden_states(pipeline, X, K)
    T = int(hs[0].shape[1])
    if T != ncp + CHRONOS2_N_REG + K:
        raise RuntimeError(f"chronos2 sequence has {T} tokens, expected {ncp}+1+{K}")
    mask = chronos2_token_mask(X, CHRONOS2_INPUT_PATCH, CHRONOS2_N_REG, K)
    tokens = assert_token_positions(mask, range(ncp), "chronos2 one-batch")
    path, _ = _resolve_chronos2_cache(tag, split, Y, cache_dir)
    with np.load(path, allow_pickle=True) as z:
        c_ref = {i: z[f"content_L{i}"][:batch] for i in range(NUM_LAYERS)}
        s_ref = {i: z[f"fslot_L{i}"][:batch] for i in range(NUM_LAYERS)}
        c_ref[NUM_LAYERS], s_ref[NUM_LAYERS] = z["content_final"][:batch], z["fslot_final"][:batch]
    states = list(hs) + [final]
    ours_c = {i: masked_mean(states[i].numpy(), mask) for i in range(len(states))}
    ours_s = {i: states[i][:, -K:, :].numpy() for i in range(len(states))}
    rc, rs = state_identity(ours_c, c_ref), state_identity(ours_s, s_ref)
    assert_identity(rc, "chronos2 content mean")
    assert_identity(rs, "chronos2 forecast slots")
    pooled_vs_slot = float(np.abs(ours_c[12] - ours_s[12].mean(axis=1)).max())
    return {"model": "chronos2", "dataset": tag, "split": split, "n_windows": int(len(X)),
            "sequence_tokens": T, "tokens": tokens,
            "excluded_positions": list(range(ncp, T)),
            "content_mean_vs_cache": {k: rc[k] for k in ("worst_rel_to_std", "all_bitwise")},
            "forecast_slots_vs_cache": {k: rs[k] for k in ("worst_rel_to_std", "all_bitwise")},
            "pooled_differs_from_slots_L12_max_abs": pooled_vs_slot,
            "refuses_parent_window": _refuses_future(lambda: chronos2_hidden_states(
                pipeline, np.zeros((2, C + H), np.float32), K)),
            "passed": True}


def one_batch_timesfm3(tag: str, w: dict, p1cfg: dict, *, phase1_cache_root, model, device,
                       split: str = "test") -> dict:
    from probing.timesfm3_last_token import (LastTokenGeometry, cache_metadata, cache_root,
                                             read_cache)
    spec = phase1.model_spec("timesfm3")
    geom = LastTokenGeometry(C=C, H=H, strict=True)
    bs = int(p1cfg.get("extract_batch_size", 64))
    Xall = assert_contexts(w[f"X_{split}"])
    layers = list(range(spec.n_points))
    meta1 = cache_metadata(tag, split, geom, checkpoint=p1cfg["checkpoint"],
                           detrend=bool(p1cfg.get("detrend", True)), layers=layers,
                           seed=int(p1cfg.get("seed", 0)),
                           feature_dtype=np.dtype(p1cfg.get("feature_dtype", "float32")),
                           suite=p1cfg.get("suite", "paper14"), n_windows=len(Xall))
    hit1 = read_cache(cache_root(phase1_cache_root, tag, split, geom,
                                 bool(p1cfg.get("detrend", True)), p1cfg.get("suite", "paper14")),
                      meta1, Xall, layers)
    if hit1 is None:
        raise Phase1DependencyError(f"timesfm3/{tag}/{split}: no Phase-1 cache")
    X = Xall[:bs]
    ex = extract_timesfm3_pooled(X, model=model, device=device, geom=geom, batch_size=bs,
                                 detrend=bool(p1cfg.get("detrend", True)), progress=False)
    ident = assert_identity(state_identity(
        ex["last_token"], {L: np.asarray(hit1["feats"][L])[:bs] for L in layers}),
        "timesfm3 token 15")
    return {"model": "timesfm3", "dataset": tag, "split": split, "n_windows": int(len(X)),
            "tokens": ex["tokens"], "excluded_positions": list(range(geom.n_real_context_patches,
                                                                      geom.n_tokens)),
            "token15_vs_cache": {k: ident[k] for k in ("worst_rel_to_std", "all_bitwise")},
            "revin_mu_max_abs": float(np.abs(ex["mu"] - hit1["mu"][:bs]).max()),
            "revin_sd_max_abs": float(np.abs(ex["sd"] - hit1["sd"][:bs]).max()),
            "pooled_differs_from_token15_L20_max_abs":
                float(np.abs(ex["pooled"][20] - ex["last_token"][20]).max()),
            "refuses_parent_window": _refuses_future(lambda: extract_timesfm3_pooled(
                np.zeros((2, C + H), np.float32), model=model, device=device, geom=geom,
                batch_size=2)),
            "passed": True}


def one_batch_tirex(tag: str, w: dict, p1cfg: dict, *, phase1_cache_root, model, geom,
                    split: str = "test") -> dict:
    from probing.tirex_model import REP_NAMES, cache_metadata, cache_root, read_cache
    bs = int(p1cfg.get("extract_batch_size", 256))
    mode = p1cfg.get("rollout_mode", "two_pass")
    Xall = assert_contexts(w[f"X_{split}"])
    meta1 = cache_metadata(tag, split, geom, checkpoint=p1cfg["checkpoint"],
                           backend=p1cfg.get("backend", "torch"), points=list(REP_NAMES),
                           seed=int(p1cfg.get("seed", 0)), X=Xall, mode=mode, batch_size=bs)
    got = read_cache(cache_root(phase1_cache_root, tag, split, mode), meta1, list(REP_NAMES))
    if got is None:
        raise Phase1DependencyError(f"tirex/{tag}/{split}: no Phase-1 cache")
    named = got[0]
    X = Xall[:bs]
    ex = extract_tirex_pooled(X, model=model, geom=geom, batch_size=bs)
    ident = assert_identity(state_identity(ex["readouts"], {k: v[:bs] for k, v in named.items()}),
                            "tirex readouts")
    return {"model": "tirex", "dataset": tag, "split": split, "n_windows": int(len(X)),
            "tokens_pass0": ex["tokens"], "pass_layouts": ex["pass_layouts"],
            "readouts_vs_cache": {k: ident[k] for k in ("worst_rel_to_std", "all_bitwise")},
            "pass1_real_token_mean_vs_pass0_rel_to_std": {
                k: v["rel_to_std"] for k, v in ex["pass1_real_token_mean_vs_pass0"].items()},
            "pooled_differs_from_readout0_L12_max_abs":
                float(np.abs(ex["pooled"]["L12"] - ex["readouts"]["L12"][:, 0]).max()),
            "refuses_parent_window": _refuses_future(lambda: extract_tirex_pooled(
                np.zeros((2, C + H), np.float32), model=model, geom=geom, batch_size=2)),
            "passed": True}


# --------------------------------------------------------------------------- #
# the cell: pure numpy in, artifacts out (testable with no model)
# --------------------------------------------------------------------------- #
def _per_window(arm_or_native: dict, is_native: bool) -> dict:
    if is_native:
        n = arm_or_native
        return {"test_loss": n["test_window_loss"], "val_loss": n["val_window_loss"],
                "mase": n["mase_pw"], "wql_num": n["wql_num_pw"], "wql_den": n["wql_den_pw"]}
    res, raw = arm_or_native["res"], arm_or_native["raw"]
    return {"test_loss": np.asarray(res["test_window_loss"], np.float64),
            "val_loss": np.asarray(res["val_window_loss"], np.float64),
            "mase": np.asarray(raw["mase_pw"], np.float64),
            "wql_num": np.asarray(raw["wql_num_pw"], np.float64),
            "wql_den": np.asarray(raw["wql_den_pw"], np.float64)}


def assemble_cell(model: str, tag: str, *, native: dict, repro: dict, arms: dict, cfg: dict,
                  B: int = BOOT_B, seed: int = SEED, tols=TUNNEL_TOLS, gates: dict | None = None,
                  extra: dict | None = None) -> dict:
    """``arms[arm] = {"res": Phase-1-schema fit result, "raw": raw_metrics output,
    "cid_test": ..., "cid_val": ...}`` + the native record -> every artifact of one cell.

    Enforces row alignment (identical cluster ids in identical order for every arm) and that each
    MEASURED parameter count equals the documented ``readout_params`` -- so the capacity table in
    the paper is a measurement, not a formula."""
    spec = phase1.model_spec(model)
    ri = spec.reference_index
    cid_t, cid_v = native["cluster_ids_test"], native["cluster_ids_val"]
    names = [ARM_NATIVE] + [a for a in ALL_ARMS if a in arms]
    per = {ARM_NATIVE: _per_window(native, True)}
    curves = {ARM_NATIVE: {k: list(native[k]) for k in
                           ("train_loss", "val_loss", "test_loss", "wd", "n_params")}}
    tunnels = {ARM_NATIVE: repro["tunnel"]}
    params = {ARM_NATIVE: readout_params(model, ARM_NATIVE, len(cfg["quantiles"]))}
    if int(native["n_params"][0]) != params[ARM_NATIVE]["n_params"]:
        raise RuntimeError(f"{model}: Phase-1 native probe has {native['n_params'][0]} params, "
                           f"documented {params[ARM_NATIVE]['n_params']}")
    for arm in names[1:]:
        a = arms[arm]
        for nm, mine, ref in (("test", a["cid_test"], cid_t), ("val", a["cid_val"], cid_v)):
            if not np.array_equal(np.asarray(mine), np.asarray(ref)):
                raise RuntimeError(f"{model}/{tag}/{arm}: {nm} rows differ from the Phase-1 cell "
                                   "-- refusing a comparison across two row sets")
        per[arm] = _per_window(a, False)
        for k in ("test_loss", "mase"):
            if per[arm][k].shape != per[ARM_NATIVE][k].shape:
                raise RuntimeError(f"{model}/{tag}/{arm}: {k} {per[arm][k].shape} vs native "
                                   f"{per[ARM_NATIVE][k].shape}")
        r = a["res"]
        curves[arm] = {k: list(r[k]) for k in ("train_loss", "val_loss", "test_loss", "wd",
                                                "n_params")}
        tunnels[arm] = phase1.tunnel_record(spec, r["val_loss"], r["test_loss"],
                                            tol=HEADLINE_TOL, tols=tuple(tols))
        params[arm] = readout_params(model, arm, len(cfg["quantiles"]))
        measured = {int(x) for x in r["n_params"]}
        if measured != {params[arm]["n_params"]}:
            raise RuntimeError(f"{model}/{arm}: measured params {measured} != documented "
                               f"{params[arm]['n_params']}")
    boot = {a: {"loss": phase1.cluster_ci(per[a]["test_loss"], cid_t, B, seed, ri),
                "mase": phase1.cluster_ci(per[a]["mase"], cid_t, B, seed, ri)} for a in names}

    comparisons = []
    for step, a, b in comparison_pairs(model, tuple(arms)):
        for metric, split, key, cid in (("loss", "test", "test_loss", cid_t),
                                        ("loss", "val", "val_loss", cid_v),
                                        ("mase", "test", "mase", cid_t)):
            comparisons += paired_rows(metric, split, a, per[a][key], b, per[b][key], cid, spec,
                                       B=B, seed=seed, step=step)

    floors = {a: arms[a]["res"].get("constant_forecast_floor") for a in names[1:]}
    summaries = {}
    for a in names:
        desc = native.get("probe") if a == ARM_NATIVE else arms[a]["res"]["probe_description"]
        wmax = None if a == ARM_NATIVE else arms[a]["res"]["wd_at_grid_max"]
        summaries[a] = arm_summary(a, model, spec, curves[a], tunnels[a], boot[a]["loss"],
                                   boot[a]["mase"], params[a], desc or params[a]["description"],
                                   floors.get(a), wmax)

    def _final(step_from, step_to, metric="loss", split="test"):
        for r in comparisons:
            if (r["from_arm"], r["to_arm"], r["metric"], r["split"]) == \
                    (step_from, step_to, metric, split) and r["is_reference_point"]:
                return {k: r[k] for k in ("delta_abs", "delta_abs_ci_lo", "delta_abs_ci_hi",
                                          "delta_rel", "delta_rel_ci_lo", "delta_rel_ci_hi",
                                          "direction")}
        return None

    final_block = {f"{a}->{b}": {"step": s, "loss": _final(a, b), "mase": _final(a, b, "mase")}
                   for s, a, b in comparison_pairs(model, tuple(arms))}
    nat_fc = native.get("forecast", {})
    summary = {
        "schema": "phase2d_cell_summary/v1", "protocol": PROTOCOL, "model": model,
        "dataset": tag, "display_name": _display(tag), "config_hash": cfg.get("config_hash"),
        "arms": names, "readout_params": params, "pooling": POOLING[model],
        "causality_note": CAUSALITY_NOTE, "decomposition": [
            {"step": s, "from": a, "to": b, "what_changes": w} for s, a, b, w in DECOMPOSITION
            if a in names and b in names],
        "tunnel_rule": {"definition": phase1.TUNNEL_DEFINITION_VERSION,
                        "headline_tolerance": HEADLINE_TOL, "tolerances": list(tols),
                        "reference": "each arm's OWN final block "
                                     f"({spec.reference_label}); validation only"},
        "per_arm": summaries, "final_block_comparisons": final_block,
        "native_forecast": {k: float(np.mean(v)) for k, v in nat_fc.items()},
        "native_reproduction": {k: v for k, v in repro.items() if k != "tunnel"},
        "depth_axis_labels": spec.depth_labels, "reference_label": spec.reference_label,
        "n_test_windows": int(len(cid_t)), "n_test_clusters": int(len(np.unique(cid_t))),
        "B": int(B), "seed": int(seed), "gates": gates or {}, **(extra or {})}

    layer_rows = []
    for a in names:
        tun, c, bl, bm = tunnels[a], curves[a], boot[a]["loss"], boot[a]["mase"]
        pos_of = {i: p for p, i in enumerate(spec.depth_indices)}
        ent_idx = {k: e["index"] for k, e in tun["by_tolerance"].items()}
        for i, p in enumerate(spec.points):
            pos = pos_of.get(i)
            row = {"model": model, "dataset": tag, "arm": a, "point_index": i, "layer": p.label,
                   "point_type": p.point_type, "include_in_main_depth_axis": p.on_depth_axis,
                   "relative_depth": "" if p.relative_depth is None else p.relative_depth,
                   "is_reference_point": i == spec.reference_index,
                   "train_loss": c["train_loss"][i], "val_loss": c["val_loss"][i],
                   "test_loss": c["test_loss"][i], "test_loss_ci_lo": bl["ci_lo"][i],
                   "test_loss_ci_hi": bl["ci_hi"][i], "test_mase": bm["point"][i],
                   "test_mase_ci_lo": bm["ci_lo"][i], "test_mase_ci_hi": bm["ci_hi"][i],
                   "val_ratio_to_final": "" if pos is None else tun["val_ratio_by_depth"][pos],
                   "val_suffix_excursion": ("" if pos is None else
                                            tun["val_suffix_excursion_by_depth"][pos]),
                   "weight_decay": c["wd"][i], "n_probe_params": c["n_params"][i],
                   "readout": summaries[a]["readout"]}
            for k, idx in ent_idx.items():
                row[f"is_tunnel_entrance_{k[4:]}"] = bool(i == idx)
            layer_rows.append(row)

    wd_rows = []
    for a in names[1:]:
        r = arms[a]["res"]
        for row in phase1.wd_selection_rows(model, tag, spec, r, r["wd_grid"], lr=cfg["probe_lr"],
                                            floor=r.get("constant_forecast_floor")):
            wd_rows.append({"arm": a, **row})

    npz = {"labels": np.asarray(spec.labels), "cluster_ids_test": np.asarray(cid_t),
           "cluster_ids_val": np.asarray(cid_v)}
    for a in names:
        for k, v in per[a].items():
            npz[f"{a}__{k}_window"] = np.asarray(v, np.float64)
    return {"cfg": cfg, "summary": summary, "layer_rows": layer_rows, "comparisons": comparisons,
            "tunnels": tunnels, "wd_rows": wd_rows, "npz": npz, "gates": gates or {}}


def write_cell(stage: Path, bundle: dict) -> Path:
    stage = Path(stage)
    phase1.atomic_write_json(stage / "cell_config.json", bundle["cfg"])
    phase1.atomic_write_json(stage / "summary.json", bundle["summary"])
    phase1.atomic_write_json(stage / "tunnels.json", bundle["tunnels"])
    phase1.atomic_write_json(stage / "gates.json", bundle["gates"])
    phase1.write_csv(stage / "layer_metrics.csv", bundle["layer_rows"])
    phase1.write_csv(stage / "comparisons.csv", bundle["comparisons"])
    phase1.write_csv(stage / "wd_selection.csv", bundle["wd_rows"])
    np.savez(stage / "bootstrap_inputs.npz", **bundle["npz"])
    return stage


def save_probe_artifacts(root, model: str, tag: str, arm: str, res: dict) -> dict:
    """Readout checkpoints + predictions, OUTSIDE the repo (scratch), one file pair per arm."""
    d = Path(root) / model / tag
    d.mkdir(parents=True, exist_ok=True)
    pw = d / f"{arm}__probe_weights.npz"
    np.savez(pw, **{f"{_slug(lb)}__{k}": v for lb, wts in res["probe_weights"].items()
                    for k, v in wts.items()})
    pr = d / f"{arm}__predictions.npz"
    np.savez(pr, **{f"val__{_slug(lb)}": v for lb, v in res["pred_val"].items()},
             **{f"test__{_slug(lb)}": v for lb, v in res["pred_test"].items()})
    return {"probe_weights": str(pw), "predictions": str(pr)}


def _display(tag: str) -> str:
    try:
        from probing import registry
        return registry.display_name(tag)
    except Exception:                                              # pragma: no cover
        return tag


def read_cell(root, model: str, tag: str) -> dict | None:
    """A COMPLETE Phase-2d cell's summary + comparisons + layer rows, or None."""
    d = Path(root) / KIND / model / tag
    if not (d / "COMPLETE").exists():
        return None
    return {"summary": json.loads((d / "summary.json").read_text()),
            "layer_rows": list(csv.DictReader(open(d / "layer_metrics.csv"))),
            "comparisons": list(csv.DictReader(open(d / "comparisons.csv"))),
            "path": d}
