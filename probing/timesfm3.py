"""Frozen TimesFM-3 backbone access + INDEPENDENT-CAUSAL-PREFIX feature extraction.

The TimesFM-3 twin of ``probing/extraction.py``. Chronos-2 is untouched.

Why independent prefixes
------------------------
TimesFM-3's transformer is causal, and its running RevIN is causal, but its optional linear
detrending fits one (a, b) on the WHOLE context handed to a forward pass. So token j of a
single 512-point pass is NOT a causal function of x[1:32j]: when the detrend threshold fires,
(a, b) already saw x[32j+1:512]. Measured on a steep-trend series, token j of the full pass
differs from the same token of an independent prefix pass by 27-775% of the feature std.

Therefore every forecast origin j is its OWN forward pass on x[1:32j] only, and the readout is
that pass's LAST REAL CONTEXT token, index j-1. Everything about origin j -- hidden state,
detrend coefficients, RevIN statistics, target normalization -- then depends on x[1:32j] alone.
``assert_causal_prefixes`` enforces this at runtime by perturbing the future and requiring a
bit-identical representation.

Geometry (C=512, H=64, P=32), re-derived from decode() and asserted against the loaded model:
    origin j in 1..16   prefix x[0:32j]  ->  j real context patches + 2 horizon placeholders
    readout token       index j-1 (0-based), a REAL context patch, never a placeholder
    target              x[32j : 32j+64]          (1-based: x_{32j+1 .. 32j+64})
    per layer           h_{l,j} in R^1280,  stacked to (N, 16, 1280)

Nothing here silently repairs an unexpected shape: every geometric assumption is an assert
with an explicit message.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch

# ---- model facts, all asserted against the checkpoint in get_model() ---- #
NUM_LAYERS = 21                  # Emb (pre_transformer_resblock output) + L1..L20
LAST_LAYER = NUM_LAYERS - 1      # L20 -- the native quantile head reads this directly
LAYER_NAMES = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)]
INPUT_PATCH_LEN = 32             # P
OUTPUT_PATCH_LEN = 64            # the native head's per-token horizon
MODEL_DIMS = 1280                # d
NATIVE_QUANTILES = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], dtype=np.float32)
NATIVE_MEDIAN_IDX = 4            # index of 0.5 in NATIVE_QUANTILES
SIGMA_EPS = 1e-6                 # timesfm3.torch.util._TOLERANCE
DETREND_THRESHOLD = 0.5          # TimesFM3Torch.linear_detrending_threshold
CACHE_VERSION = "tfm3-prefix-v1"

DEFAULT_CHECKPOINT = os.environ.get("TIMESFM3_CHECKPOINT", "google/timesfm-3.0-pytorch")

_MODEL_CACHE: dict[tuple[str, str], tuple] = {}


def timesfm_version() -> str:
    import importlib.metadata as md
    try:
        return md.version("timesfm")
    except Exception:
        return "unknown"


def resolve_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def get_model(checkpoint: str | None = None, device: str | None = None):
    """Load the FROZEN TimesFM-3 backbone (cached per checkpoint/device).

    ``checkpoint`` is an HF repo id or a LOCAL directory holding config.json +
    model.safetensors; use the local form on offline compute nodes. Set HF_HOME before the
    first call. Every architectural assumption this module makes is asserted here, so a
    different checkpoint fails loudly instead of producing quietly wrong probe numbers.
    """
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    device = resolve_device(device)
    key = (checkpoint, device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    from timesfm3.torch import TimesFM3Forecaster

    model = TimesFM3Forecaster.from_pretrained(checkpoint, device=device).model.eval()
    for p in model.parameters():
        p.requires_grad_(False)                       # frozen backbone: probes only

    tc = model.transformer_config.transformer
    n_blocks = len(model.transformer_stack.layers)
    checks = [
        (model.input_patch_len == INPUT_PATCH_LEN, f"input_patch_len {model.input_patch_len}"),
        (model.output_patch_len == OUTPUT_PATCH_LEN, f"output_patch_len {model.output_patch_len}"),
        (tc.model_dims == MODEL_DIMS, f"model_dims {tc.model_dims}"),
        (n_blocks + 1 == NUM_LAYERS, f"{n_blocks} blocks -> {n_blocks + 1} representation points"),
        (np.allclose(model.quantiles, NATIVE_QUANTILES), f"quantiles {list(model.quantiles)}"),
        (abs(model.linear_detrending_threshold - DETREND_THRESHOLD) < 1e-9,
         f"detrend threshold {model.linear_detrending_threshold}"),
        (tc.causal_attention, "sequence attention is NOT causal"),
        (model.output_head.in_features == MODEL_DIMS
         and model.output_head.out_features == OUTPUT_PATCH_LEN * len(NATIVE_QUANTILES),
         f"output_head {model.output_head.in_features}->{model.output_head.out_features}"),
    ]
    bad = [msg for ok, msg in checks if not ok]
    if bad:
        raise RuntimeError(
            f"TimesFM-3 checkpoint {checkpoint!r} does not match this module's verified "
            f"assumptions: {bad}. Re-derive the readout before trusting any probe numbers.")
    _MODEL_CACHE[key] = (model, device)
    return _MODEL_CACHE[key]


def assert_backbone_frozen(model) -> int:
    """Every backbone parameter must be frozen. Returns the parameter count checked."""
    live = [n for n, p in model.named_parameters() if p.requires_grad]
    if live:
        raise RuntimeError(f"TimesFM-3 backbone is NOT frozen: {len(live)} trainable "
                           f"parameters, e.g. {live[:3]}")
    return sum(1 for _ in model.parameters())


def assert_no_backbone_grads(model) -> None:
    """After a probe backward pass no backbone parameter may carry a gradient."""
    grads = [n for n, p in model.named_parameters() if p.grad is not None]
    if grads:
        raise RuntimeError(f"TimesFM-3 backbone received gradients: {len(grads)} parameters, "
                           f"e.g. {grads[:3]} -- the backbone must stay frozen")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

class PrefixGeometry:
    """Independent-causal-prefix layout. All indices 0-based unless the name says otherwise."""

    def __init__(self, C: int = 512, H: int = 64, P: int = INPUT_PATCH_LEN):
        if C % P:
            raise ValueError(f"context length {C} must be a multiple of the patch length {P}")
        if H > OUTPUT_PATCH_LEN:
            raise ValueError(
                f"horizon {H} > output_patch_len {OUTPUT_PATCH_LEN}: one origin cannot cover it "
                "and the native readout would span horizon placeholders whose RevIN stats come "
                "from the nonlinear cpm_iterative_revin_refine scan")
        self.C, self.H, self.P = C, H, P
        self.J = C // P                                   # 16 origins
        self.rolls = OUTPUT_PATCH_LEN // P
        self.origins_1based = list(range(1, self.J + 1))
        # origin j (1-based): prefix x[0 : P*j], readout token P-index j-1,
        # target x[P*j : P*j + H]
        self.prefix_len = np.array([P * j for j in self.origins_1based], dtype=np.int64)
        self.readout_token = np.array([j - 1 for j in self.origins_1based], dtype=np.int64)
        self.target_start = self.prefix_len.copy()
        self.target_end = self.target_start + H           # exclusive
        self.span = int(self.target_end[-1])              # = C + H
        if self.span != C + H:
            raise ValueError(f"origin targets span {self.span}, expected C+H={C + H}")
        # horizon placeholder patches appended by decode(), identical for every prefix
        extract_len = min(2 * P, OUTPUT_PATCH_LEN)
        self.K = max(math.ceil((H - (extract_len - P)) / P), 1)
        self.n_horizon_patches = self.K + self.rolls - 1
        self.headline_origin_1based = self.J              # origin 16 -- THE reported task

    def n_tokens(self, j_1based: int) -> int:
        return j_1based + self.n_horizon_patches

    def as_dict(self) -> dict:
        return {"C": self.C, "H": self.H, "P": self.P, "J": self.J,
                "prefix_extraction": True,
                "prefix_len": self.prefix_len.tolist(),
                "readout_token_0based": self.readout_token.tolist(),
                "target_start_0based": self.target_start.tolist(),
                "target_end_0based_exclusive": self.target_end.tolist(),
                "n_horizon_patches": int(self.n_horizon_patches), "native_K": int(self.K),
                "headline_origin_1based": self.headline_origin_1based,
                "layer_names": LAYER_NAMES, "d": MODEL_DIMS}

    def audit_rows(self, detrend_active=None):
        """1-BASED human-readable audit rows, one per origin (see the driver's audit table)."""
        rows = []
        for i, j in enumerate(self.origins_1based):
            rows.append({
                "origin": j,
                "raw_prefix_length": int(self.prefix_len[i]),
                "num_real_context_patches": j,
                "selected_token_index_0based": int(self.readout_token[i]),
                "target_start_1based": int(self.target_start[i]) + 1,
                "target_end_1based": int(self.target_end[i]),
                "detrend_active": (None if detrend_active is None
                                   else float(np.mean(detrend_active[:, i]))),
                "hidden_shape": (MODEL_DIMS,),
                "transformed_target_shape": (self.H,),
            })
        return rows


def native_readout_token(model, prefix_len: int, H: int) -> list[int]:
    """decode()'s own forecast_indices for a context of ``prefix_len``, from model attributes."""
    P = model.input_patch_len
    n_ctx = prefix_len // P
    if model.use_stitching:
        overlap = model._stitching_extract_len - P
        K = max(math.ceil((H - overlap) / P), 1)
        return list(range(n_ctx - 1, n_ctx - 1 + K))
    K = (H + model.output_patch_len - 1) // model.output_patch_len
    return [n_ctx - 1 + i * model.rolls for i in range(K)]


# --------------------------------------------------------------------------- #
# per-prefix preprocessing: detrending + causal RevIN  (numpy twin of decode())
# --------------------------------------------------------------------------- #

def prefix_detrend_params(prefix: np.ndarray, threshold: float = DETREND_THRESHOLD,
                          enabled: bool = True):
    """Masked-OLS detrend parameters for ONE prefix length, mirroring decode().

    prefix : (n, Lp) raw prefixes, ALL of the same length Lp.
    Returns (a, b, active), each (n,). The trend at absolute window index k (0-based, measured
    from the START of the parent window, which is also the start of the prefix) is

        trend(k) = a * (k - (Lp - 1)) / Lp + b

    NOTE the time axis is normalized by THIS PREFIX's length Lp, exactly as decode() does
    (t_ctx = arange(-(context-1), 1) / context). Using 512 here for a shorter prefix would be
    a different line. Nothing outside ``prefix`` enters the fit -- in particular no target
    value ever does, at any origin.
    """
    Xp = np.asarray(prefix, dtype=np.float64)
    if Xp.ndim != 2:
        raise ValueError(f"prefix must be (n, Lp), got {Xp.shape}")
    n, Lp = Xp.shape
    if not enabled:
        z = np.zeros(n, dtype=np.float64)
        return z, z, np.zeros(n, dtype=bool)
    t = (np.arange(Lp, dtype=np.float64) - (Lp - 1)) / Lp
    sum_t, sum_t2 = t.sum(), (t ** 2).sum()
    sum_y, sum_ty = Xp.sum(axis=1), (Xp * t).sum(axis=1)
    det = Lp * sum_t2 - sum_t ** 2
    a = (Lp * sum_ty - sum_t * sum_y) / (det if det != 0 else 1.0)
    b = (sum_y - a * sum_t) / Lp
    resid = Xp - (a[:, None] * t[None, :] + b[:, None])
    mean_y = sum_y / Lp
    std_raw = np.sqrt(np.maximum((Xp ** 2).sum(axis=1) / Lp - mean_y ** 2, 0.0))
    mean_r = resid.sum(axis=1) / Lp
    std_res = np.sqrt(np.maximum((resid ** 2).sum(axis=1) / Lp - mean_r ** 2, 0.0))
    return a, b, std_res < threshold * std_raw


def prefix_trend(a, b, active, prefix_len: int, idx: np.ndarray) -> np.ndarray:
    """Evaluate a prefix's fitted line at absolute indices ``idx``; 0 where inactive.

    ``idx`` may run past the prefix (that is the extrapolation onto the target window, exactly
    what decode() does with t_forecast = arange(1, H+1) / context). Returns (n, len(idx)).
    """
    t = (np.asarray(idx, dtype=np.float64) - (prefix_len - 1)) / prefix_len
    out = np.asarray(a)[:, None] * t[None, :] + np.asarray(b)[:, None]
    return np.where(np.asarray(active)[:, None], out, 0.0)


def safe_sigma(sd) -> np.ndarray:
    """util._make_safe_for_division: forward RevIN divides by 1.0 when sigma < 1e-6.

    Reverse RevIN multiplies by the RAW sigma, so the library is not self-inverse in that
    regime; origins with sd < SIGMA_EPS are masked out rather than silently rescaled.
    """
    sd = np.asarray(sd)
    return np.where(sd < SIGMA_EPS, 1.0, sd)


def normalize_target(raw_target, mu, sd):
    """Forward RevIN of an already-detrended target: (y - mu) / safe(sigma)."""
    return (np.asarray(raw_target, np.float64) - np.asarray(mu)[:, None]) \
        / safe_sigma(sd)[:, None]


def denormalize(pred, *, mu, sd, trend):
    """Inverse: util.revin(reverse) + trend add-back. pred (..., H); mu/sd (n,); trend (n,H).

    mu and sd are KEYWORD-ONLY: they have identical shapes, so a positional swap would
    silently produce a plausible forecast on the wrong scale.
    """
    pred = np.asarray(pred, np.float64)
    extra = pred.ndim - 2
    shape = (-1,) + (1,) * extra + (1,)
    return pred * np.asarray(sd).reshape(shape) + np.asarray(mu).reshape(shape) \
        + np.asarray(trend).reshape((-1,) + (1,) * extra + (np.shape(trend)[-1],))


def raw_future_from_arcsinh(X, Y_traj, sigma_eps: float = 1e-6):
    """Invert id_data._make_examples' label transform exactly (sinh(arcsinh(x)) == x)."""
    X64 = np.asarray(X, np.float64)
    mu = X64.mean(axis=1)
    s = np.maximum(X64.std(axis=1), sigma_eps)
    return mu[:, None] + s[:, None] * np.sinh(np.asarray(Y_traj, np.float64))


def build_prefix_targets(Z, mu, sd, geom: PrefixGeometry, *, detrend: bool = True,
                         threshold: float = DETREND_THRESHOLD):
    """Per-origin targets in each prefix's OWN normalized space.

    Z     : (n, C+H) raw parent windows (context ++ future)
    mu/sd : (n, J)   the model's running RevIN stats at each prefix's last real context token
    Returns dict with
        targets (n, J, H) float32   normalized supervision
        trend   (n, J, H) float64   the add-back needed to invert an origin's prediction
        active  (n, J)    bool      did detrending fire for that prefix
        a, b    (n, J)    float64   that prefix's fitted line
        valid   (n, J)    bool      usable origins (finite target, sigma >= SIGMA_EPS)
    """
    Z = np.asarray(Z, np.float64)
    n, span = Z.shape
    if span != geom.span:
        raise ValueError(f"parent windows must be (n, {geom.span}), got {Z.shape}")
    if np.shape(mu) != (n, geom.J) or np.shape(sd) != (n, geom.J):
        raise ValueError(f"mu/sd must be (n, {geom.J}), got {np.shape(mu)} / {np.shape(sd)}")

    J, H = geom.J, geom.H
    targets = np.empty((n, J, H), np.float64)
    trend = np.empty((n, J, H), np.float64)
    active = np.empty((n, J), bool)
    A = np.empty((n, J), np.float64)
    B = np.empty((n, J), np.float64)

    for i, j in enumerate(geom.origins_1based):
        Lp = int(geom.prefix_len[i])
        # FIT on x[0:Lp] only -- the target slice is never passed in.
        a, b, act = prefix_detrend_params(Z[:, :Lp], threshold, enabled=detrend)
        sl = np.arange(geom.target_start[i], geom.target_end[i])
        tr = prefix_trend(a, b, act, Lp, sl)              # extrapolated onto the target
        targets[:, i, :] = normalize_target(Z[:, sl] - tr, mu[:, i], sd[:, i])
        trend[:, i, :] = tr
        active[:, i], A[:, i], B[:, i] = act, a, b

    valid = (np.asarray(sd) >= SIGMA_EPS) & np.isfinite(targets).all(axis=2)
    return {"targets": targets.astype(np.float32), "trend": trend, "active": active,
            "a": A, "b": B, "valid": valid}


# --------------------------------------------------------------------------- #
# extraction
# --------------------------------------------------------------------------- #

def _hooks(model, caps: dict):
    layers = model.transformer_stack.layers
    def grab(key):
        def hook(_m, _a, out):      # MUST return None -- a return value REPLACES the output
            caps[key] = (out[0] if isinstance(out, tuple) else out).detach()
        return hook
    hs = [layers[0].register_forward_pre_hook(
        lambda _m, a: caps.__setitem__("L0", a[0].detach()))]
    hs += [ly.register_forward_hook(grab(f"L{i + 1}")) for i, ly in enumerate(layers)]
    return hs


def extract_prefix_features(X, *, geom: PrefixGeometry, model=None, checkpoint=None,
                            device=None, batch_size: int = 64, layers=None,
                            feature_dtype=np.float16, detrend: bool = True,
                            origins=None, verify: bool = True, progress: bool = True):
    """One INDEPENDENT decode() per (origin, batch) -- prefixes bucketed by length.

    X : (n, C) raw contexts. For each origin j the whole batch's x[0:32j] goes through decode()
    as one tensor; TimesFM-3 computes detrending and RevIN per (batch element, variate), so
    batching rows NEVER shares preprocessing statistics between windows -- ``assert_batching_
    invariant`` proves it. Each row is still an independent TimesFM-3 input.

    Returns feats {layer: (n, J, 1280)}, mu/sd (n, J), native (n, H, 9) from origin J only.
    """
    if model is None:
        model, device = get_model(checkpoint, device)
    else:
        device = resolve_device(device)
    assert_backbone_frozen(model)
    layers = list(range(NUM_LAYERS)) if layers is None else sorted(layers)
    origins = list(geom.origins_1based) if origins is None else sorted(origins)
    bad = [j for j in origins if j not in geom.origins_1based]
    if bad:
        raise ValueError(f"origins {bad} outside 1..{geom.J}")

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != geom.C:
        raise ValueError(f"X must be (n, {geom.C}), got {X.shape}")
    n, J, H = len(X), geom.J, geom.H

    feats = {i: np.zeros((n, J, MODEL_DIMS), dtype=feature_dtype) for i in layers}
    mu = np.full((n, J), np.nan, np.float64)
    sd = np.full((n, J), np.nan, np.float64)
    native = np.zeros((n, H, len(NATIVE_QUANTILES)), np.float32)
    checks = {}

    caps: dict = {}
    hs = _hooks(model, caps)
    try:
        with detrending(model, detrend):
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                for j in origins:
                    i = j - 1
                    Lp = int(geom.prefix_len[i])
                    tok = int(geom.readout_token[i])
                    tgt = torch.from_numpy(X[s:e, :Lp]).to(device).unsqueeze(1)  # (b,1,Lp)
                    assert tgt.shape[-1] == Lp == INPUT_PATCH_LEN * j, (
                        f"origin {j}: prefix length {tgt.shape[-1]} != 32*{j}")
                    caps.clear()
                    with torch.no_grad():
                        out = model.decode(target=tgt, horizon=H, return_aux_outputs=True)
                    official, aux = out
                    # --- geometry, asserted every batch, never repaired ---
                    n_tok = caps[f"L{LAST_LAYER}"].shape[2]
                    if n_tok != geom.n_tokens(j):
                        raise RuntimeError(
                            f"origin {j}: decode() produced {n_tok} tokens, expected "
                            f"{geom.n_tokens(j)} = {j} real context + "
                            f"{geom.n_horizon_patches} horizon placeholders")
                    if tok >= j:
                        raise RuntimeError(
                            f"origin {j}: readout token {tok} is not a REAL context patch "
                            f"(only indices 0..{j-1} are; {j}..{n_tok-1} are placeholders)")
                    nat_tok = native_readout_token(model, Lp, H)
                    if nat_tok != [tok]:
                        raise RuntimeError(
                            f"origin {j}: decode()'s own forecast_indices {nat_tok} != our "
                            f"readout [{tok}] -- the native readout moved")
                    for L in layers:
                        h = caps[f"L{L}"]
                        if h.shape[1] != 1 or h.shape[3] != MODEL_DIMS:
                            raise RuntimeError(
                                f"origin {j} layer {L}: state {tuple(h.shape)} is not "
                                f"(b, 1, n_tok, {MODEL_DIMS})")
                        feats[L][s:e, i, :] = h[:, 0, tok, :].cpu().numpy().astype(feature_dtype)
                    run_mu, run_sd = aux["revin_stats"]
                    mu[s:e, i] = run_mu[:, 0, tok].cpu().numpy()
                    sd[s:e, i] = run_sd[:, 0, tok].cpu().numpy()
                    if j == geom.headline_origin_1based:
                        native[s:e] = official[:, 0].cpu().numpy()
                        if verify and "native" not in checks:
                            checks["native"] = verify_native_median(
                                model, caps[f"L{LAST_LAYER}"], official, aux["revin_stats"],
                                X[s:e], geom, detrend=detrend)
                if progress and (s // batch_size) % 5 == 0:
                    print(f"  [extract] {e}/{n} windows x {len(origins)} prefixes", flush=True)
    finally:
        for h in hs:
            h.remove()

    if np.isnan(mu[:, [j - 1 for j in origins]]).any():
        raise RuntimeError("extraction left NaN RevIN statistics -- incomplete pass")
    return {"feats": feats, "mu": mu, "sd": sd, "native": native,
            "native_median_err": checks.get("native"), "origins": origins}


class detrending:
    """Temporarily force TimesFM3Torch.use_linear_detrending.

    Must be toggled on the MODEL, not just in the target builder: decode() detrends before
    patching, so turning it off only in ``build_prefix_targets`` would normalize the targets
    against a series the backbone never saw.
    """

    def __init__(self, model, enabled: bool):
        self.model, self.enabled = model, bool(enabled)

    def __enter__(self):
        self._prev = self.model.use_linear_detrending
        self.model.use_linear_detrending = self.enabled
        return self.model

    def __exit__(self, *exc):
        self.model.use_linear_detrending = self._prev
        return False


def verify_native_median(model, last_states, official, revin_stats, X_batch,
                         geom: PrefixGeometry, *, detrend: bool, rtol: float = 1e-4):
    """L20 -> native head -> native MEDIAN must reproduce decode()'s median. Abort otherwise.

    Proves we extract the representation the native head actually reads. Runs on the first
    headline-origin batch of every extraction, so a TimesFM release that changes the readout,
    the RevIN convention or the trend add-back fails here, not silently in the results.
    Consumes the SAME pass's states and revin stats -- no second forward.
    """
    from timesfm3.torch import util as tfm_util
    P, Q = model.input_patch_len, model.num_quantiles
    tok = int(geom.readout_token[-1])
    run_mu, run_sd = revin_stats
    with torch.no_grad():
        raw = model.output_head(last_states)[:, :, [tok], :]
        den = tfm_util.revin(raw, run_mu[:, :, [tok]], run_sd[:, :, [tok]], reverse=True)
        den = torch.clamp(den, -model.value_clip, model.value_clip)
        den = den.view(len(X_batch), 1, 1, model.output_patch_len, Q)
        rec = tfm_util.stitch_patches(den[:, :, :, :min(2 * P, model.output_patch_len), :],
                                      P)[:, :, :geom.H, :]
    rec = rec[:, 0].cpu().numpy().astype(np.float64)                 # (b, H, Q)
    a, b, act = prefix_detrend_params(X_batch[:, :geom.C], enabled=detrend)
    rec = rec + prefix_trend(a, b, act, geom.C,
                             np.arange(geom.C, geom.C + geom.H))[:, :, None]
    ref = official[:, 0].cpu().numpy().astype(np.float64)
    med_rec, med_ref = rec[:, :, NATIVE_MEDIAN_IDX], ref[:, :, NATIVE_MEDIAN_IDX]
    err = float(np.abs(med_rec - med_ref).max())
    scale = float(np.abs(med_ref).mean()) + 1e-12
    if err / scale > rtol:
        raise RuntimeError(
            f"L20 native-median reconstruction FAILED: max|d| = {err:.3e} "
            f"(relative {err / scale:.3e} > {rtol}). The installed timesfm3 postprocessing no "
            "longer matches probing/timesfm3.py -- do NOT trust any probe numbers.")
    return {"max_abs": err, "relative": err / scale,
            "all_quantiles_max_abs": float(np.abs(rec - ref).max())}


# --------------------------------------------------------------------------- #
# runtime correctness checks (called by the driver and the tests)
# --------------------------------------------------------------------------- #

def assert_causal_prefixes(model, geom, device, origins=(1, 4, 8, 12, 16), seed=0,
                           detrend=True, atol=0.0):
    """THE causality test: changing x AFTER the origin must not move its representation.

    Builds two parents identical on x[0:32j] and arbitrarily different after, extracts both,
    and requires bit-identical states at every representation point. Raises on any difference.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(geom.C)
    base = (10 + 0.08 * t + 3 * np.sin(2 * np.pi * t / 24)
            + rng.normal(0, 0.3, geom.C)).astype(np.float32)          # steep: detrending fires
    report = {}
    for j in origins:
        Lp = int(geom.prefix_len[j - 1])
        A = base[None, :].copy()
        B = base[None, :].copy()
        B[0, Lp:] = rng.normal(50, 20, geom.C - Lp) if Lp < geom.C else B[0, Lp:]
        fa = extract_prefix_features(A, geom=geom, model=model, device=device, batch_size=1,
                                     feature_dtype=np.float32, detrend=detrend, origins=[j],
                                     verify=False, progress=False)
        fb = extract_prefix_features(B, geom=geom, model=model, device=device, batch_size=1,
                                     feature_dtype=np.float32, detrend=detrend, origins=[j],
                                     verify=False, progress=False)
        d = max(float(np.abs(fa["feats"][L][0, j - 1] - fb["feats"][L][0, j - 1]).max())
                for L in range(NUM_LAYERS))
        dstat = max(float(abs(fa[k][0, j - 1] - fb[k][0, j - 1])) for k in ("mu", "sd"))
        report[j] = {"max_abs_dh": d, "max_abs_dstat": dstat}
        if d > atol or dstat > atol:
            raise RuntimeError(
                f"CAUSALITY VIOLATION at origin {j}: changing x[{Lp}:] moved the prefix "
                f"representation by max|dh| = {d:.3e} (RevIN stats {dstat:.3e}). The "
                "independent-prefix extraction is not causal -- investigate before proceeding.")
    return report


def assert_batching_invariant(model, geom, device, n=4, seed=1, detrend=True,
                              numeric_rtol=1e-2):
    """Batching prefixes must not change SEMANTICS. Two separable properties:

    1. SEMANTIC (hard, must be exactly 0): at a FIXED batch size, changing the OTHER rows'
       data must leave a row's representation bit-identical. This is the property that
       matters -- it is what "no preprocessing statistic is shared across examples" means,
       and it holds because TimesFM-3 reduces detrending and RevIN per (batch, variate).

    2. NUMERIC (reported, tolerance): changing the batch SIZE re-tiles the GEMMs, so
       backends that are not batch-shape-invariant (MPS, and cuBLAS in general) return
       slightly different floats for identical inputs. Measured: 0.0 on CPU, ~3e-4 absolute
       (~8e-4 relative) on MPS across 20 layers. That is float non-determinism, not leakage,
       so it gets a tolerance rather than an equality -- but a gross violation still fails.

    Asserting bit-identity across batch SIZES would conflate the two and fail on any
    non-deterministic accelerator while proving nothing extra about leakage.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(geom.C)

    def make(k, s):
        return (10 + (1 + k) * np.sin(2 * np.pi * t / 24) + 0.03 * k * t
                + np.random.default_rng(s).normal(0, 0.3, geom.C)).astype(np.float32)

    row0 = make(0, seed)
    A = np.stack([row0] + [make(k, seed + 100 + k) for k in range(1, n)])
    B = np.stack([row0] + [make(k, seed + 900 + k) for k in range(1, n)])
    kw = dict(geom=geom, model=model, device=device, feature_dtype=np.float32,
              detrend=detrend, verify=False, progress=False)
    fa = extract_prefix_features(A, batch_size=n, **kw)
    fb = extract_prefix_features(B, batch_size=n, **kw)
    d_sem = max([float(np.abs(fa["feats"][L][0] - fb["feats"][L][0]).max())
                 for L in range(NUM_LAYERS)]
                + [float(np.abs(fa["mu"][0] - fb["mu"][0]).max()),
                   float(np.abs(fa["sd"][0] - fb["sd"][0]).max())])
    if d_sem != 0.0:
        raise RuntimeError(
            f"BATCHING LEAKS ACROSS ROWS: at a fixed batch size, changing the OTHER rows' "
            f"data moved row 0 by max|d| = {d_sem:.3e}. Preprocessing statistics are shared "
            "between examples -- the independent-prefix construction is invalid.")

    f1 = extract_prefix_features(A[:1], batch_size=1, **kw)
    d_num = max(float(np.abs(fa["feats"][L][0] - f1["feats"][L][0]).max())
                for L in range(NUM_LAYERS))
    scale = float(np.asarray(fa["feats"][LAST_LAYER][0]).std()) + 1e-12
    if d_num / scale > numeric_rtol:
        raise RuntimeError(
            f"batch-size sensitivity {d_num:.3e} ({d_num/scale:.2e} relative) exceeds "
            f"{numeric_rtol}: too large for float non-determinism -- investigate.")
    return {"semantic_max_abs": d_sem, "numeric_max_abs": d_num,
            "numeric_relative": d_num / scale}


def prefix_vs_fullwindow_diagnostic(model, geom, device, X, origins=(1, 4, 8, 12, 16),
                                    detrend=True):
    """DIAGNOSTIC ONLY: how far token j of one 512-pass is from the independent prefix state.

    Allowed to differ (that is the whole point of the prefix construction). Zero when the
    detrend threshold does not fire, because then nothing global touches the token.
    """
    caps: dict = {}
    hs = _hooks(model, caps)
    try:
        with detrending(model, detrend), torch.no_grad():
            model.decode(target=torch.from_numpy(np.asarray(X, np.float32)).to(device
                         ).unsqueeze(1), horizon=geom.H)
            full = {k: v.clone() for k, v in caps.items()}
    finally:
        for h in hs:
            h.remove()
    pre = extract_prefix_features(X, geom=geom, model=model, device=device,
                                  batch_size=len(X), feature_dtype=np.float32,
                                  detrend=detrend, origins=list(origins), verify=False,
                                  progress=False)
    _, _, act_full = prefix_detrend_params(np.asarray(X, np.float64), enabled=detrend)
    out = {}
    for j in origins:
        tok = j - 1
        rel = max(float(np.abs(pre["feats"][L][:, j - 1]
                               - full[f"L{L}"][:, 0, tok, :].cpu().numpy()).max()
                        / (full[f"L{L}"][:, 0, tok, :].cpu().numpy().std() + 1e-12))
                  for L in range(NUM_LAYERS))
        _, _, act_pre = prefix_detrend_params(np.asarray(X, np.float64)[:, :32 * j],
                                              enabled=detrend)
        out[j] = {"rel_max_diff": rel, "prefix_detrend_rate": float(act_pre.mean()),
                  "full_detrend_rate": float(act_full.mean())}
    return out


# --------------------------------------------------------------------------- #
# cache
# --------------------------------------------------------------------------- #

def cache_metadata(tag, split, geom, *, checkpoint, detrend, layers, seed) -> dict:
    return {"cache_version": CACHE_VERSION, "checkpoint": checkpoint,
            "timesfm_version": timesfm_version(), "context_len": geom.C, "horizon": geom.H,
            "patch_size": geom.P, "prefix_extraction": True, "n_origins": geom.J,
            "layer_names": [LAYER_NAMES[i] for i in layers], "layers": list(layers),
            "hidden_dim": MODEL_DIMS, "detrending": bool(detrend),
            "dataset": tag, "split": split, "seed": int(seed)}


def cached_prefix_features(tag, split, X, *, geom, cache_dir, checkpoint=None, seed=0,
                           layers=None, detrend=True, force=False, **kw):
    """Disk cache, one .npy per representation point so a probe fit holds ONE layer at a time.

    Refuses any cache whose metadata disagrees with the current run -- checkpoint, timesfm
    version, geometry, layer set, detrending, dataset/split or seed. Never silently reused.
    """
    layers = list(range(NUM_LAYERS)) if layers is None else sorted(layers)
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    meta = cache_metadata(tag, split, geom, checkpoint=checkpoint, detrend=detrend,
                          layers=layers, seed=seed)
    root = Path(cache_dir) / f"{CACHE_VERSION}__{tag}__{split}__C{geom.C}_H{geom.H}" \
                             f"{'' if detrend else '__nodetrend'}"
    mp = root / "meta.npz"
    X = np.asarray(X, dtype=np.float32)

    if mp.exists() and not force:
        with np.load(mp, allow_pickle=False) as d:
            cached = json.loads(str(d["meta"]))
            diffs = {k: (cached.get(k), meta[k]) for k in meta if cached.get(k) != meta[k]}
            if diffs:
                raise RuntimeError(
                    f"INCOMPATIBLE feature cache {root}:\n" +
                    "\n".join(f"    {k}: cached={c!r} current={n!r}" for k, (c, n) in diffs.items())
                    + "\n  Delete the directory or point --cache-dir elsewhere. Never reused "
                      "silently.")
            if d["ctx_tail"].shape[0] != len(X) or not np.allclose(d["ctx_tail"], X[:, -8:]):
                raise RuntimeError(
                    f"stale feature cache {root}: cached contexts do not match the current "
                    f"{split} windows -- delete the directory and re-run")
            missing = [i for i in layers if not (root / f"L{i:02d}.npy").exists()]
            if not missing:
                print(f"  [cache HIT]  {root.name}  ({len(X)} windows, {len(layers)} layers)")
                return {"feats": {i: np.load(root / f"L{i:02d}.npy", mmap_mode="r")
                                  for i in layers},
                        "mu": d["mu"], "sd": d["sd"], "native": d["native"],
                        "native_median_err": json.loads(str(d["native_median_err"])),
                        "origins": list(geom.origins_1based)}

    root.mkdir(parents=True, exist_ok=True)
    out = extract_prefix_features(X, geom=geom, checkpoint=checkpoint, layers=layers,
                                  detrend=detrend, **kw)
    for i, arr in out["feats"].items():
        np.save(root / f"L{i:02d}.npy", arr)
    np.savez(mp, meta=json.dumps(meta), ctx_tail=X[:, -8:], mu=out["mu"], sd=out["sd"],
             native=out["native"],
             native_median_err=json.dumps(out["native_median_err"]))
    print(f"  [saved]      {root}  ({len(X)} windows, {len(layers)} layers)")
    out["feats"] = {i: np.load(root / f"L{i:02d}.npy", mmap_mode="r") for i in layers}
    return out
