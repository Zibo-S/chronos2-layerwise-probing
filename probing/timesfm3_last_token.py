"""Frozen TimesFM-3: LAST-CONTEXT-TOKEN extraction for the NATIVE C=512 -> H=64 task.

The headline TimesFM-3 line. It probes the one representation the native forecasting head
actually reads, and nothing else. ``probing/timesfm3.py`` (the 16-prefix / shared-origin
ABLATION) and every Chronos-2 file are imported read-only and never modified.

    Chronos-2  : the native head reads K=4 forecast-slot states for H=64  -> probe those slots
    TimesFM-3  : the native head reads ONE last-context state for H=64    -> probe that token

Geometry, re-derived from decode() and asserted against the loaded checkpoint:

    input            x[1:512]                  ONE full-context decode() pass per batch
    patches          512 / 32 = 16 REAL context patches + 2 horizon placeholders = 18 tokens
    readout          token index 15 (0-based) = the LAST REAL context patch, never a placeholder
                     == decode()'s own forecast_indices for (C=512, H=64)
    per layer        h_{l,15} in R^1280 for l in {Emb, L1..L20}   ->  cache (N, 1280)
    target           x[513:576] in the native normalized forecasting space
    native head      Linear(1280, 64*9) consumes h_{L20,15} directly

Why the full context is legitimate here (it is NOT in the prefix ablation)
-------------------------------------------------------------------------
TimesFM-3's optional linear detrending fits one (a, b) on the WHOLE context handed to a
forward pass. In the prefix ablation that made token j of a 512-point pass non-causal w.r.t.
x[1:32j]. Here the forecast origin IS the end of the context: all 512 points are legitimately
observed for the C=512 -> H=64 task, so fitting the trend on x[1:512] leaks nothing. The
future x[513:576] is never passed to the model (asserted) and never enters any preprocessing
statistic (tested).

Nothing here silently repairs an unexpected shape, token position, quantile order or cache:
every assumption is an explicit raise.
"""

from __future__ import annotations

import inspect
import json
import math
from pathlib import Path

import numpy as np
import torch

from probing.probes import median_index, validate_quantiles
# Reused UNCHANGED from the validated shared-origin module (read-only import; that file is the
# 16-prefix ablation and must stay byte-identical). The detrend/RevIN algebra is prefix-length
# agnostic: calling it at prefix length C=512 IS decode()'s full-context preprocessing.
from probing.timesfm3 import (DEFAULT_CHECKPOINT, DETREND_THRESHOLD, INPUT_PATCH_LEN,
                              LAST_LAYER, LAYER_NAMES, MODEL_DIMS, NATIVE_MEDIAN_IDX,
                              NATIVE_QUANTILES, NUM_LAYERS, OUTPUT_PATCH_LEN, SIGMA_EPS,
                              assert_backbone_frozen, assert_no_backbone_grads, denormalize,
                              detrending, get_model, native_readout_token, normalize_target,
                              prefix_detrend_params, prefix_trend, raw_future_from_arcsinh,
                              resolve_device, safe_sigma, timesfm_version)
from probing.timesfm3 import _hooks as register_layer_hooks

__all__ = ["CACHE_VERSION", "NUM_QUANTILES", "LastTokenGeometry", "assert_native_geometry",
           "assert_native_quantiles", "context_detrend_params", "context_trend",
           "build_last_token_targets", "assert_target_roundtrip", "select_last_token",
           "extract_last_token_features", "cached_last_token_features", "cache_metadata",
           "read_cache", "cache_root", "verify_native_head", "decode_sorting_knobs",
           "no_quantile_sorting", "feature_dtype_report",
           # intentional re-exports so a driver imports ONE module for this experiment
           "assert_backbone_frozen", "assert_no_backbone_grads", "denormalize", "safe_sigma",
           "raw_future_from_arcsinh", "LAYER_NAMES", "NUM_LAYERS", "LAST_LAYER", "MODEL_DIMS",
           "NATIVE_QUANTILES", "NATIVE_MEDIAN_IDX", "SIGMA_EPS"]

CACHE_VERSION = "tfm3-last-token-q9-v1"    # MUST differ from the prefix ablation's tfm3-prefix-v1
NUM_QUANTILES = len(NATIVE_QUANTILES)      # 9 -- TimesFM-3's full native quantile vector
LAST_TOKEN_ONLY = True


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

class LastTokenGeometry:
    """The single-readout layout. All token indices are 0-based; time positions 1-based only
    where the name says so.

    ``strict=True`` (the default) hard-asserts the ONE configuration this experiment is
    specified for -- C=512, H=64, P=32, 16 real context patches, readout token 15. Any other
    configuration must be opted into explicitly, because the readout token and the horizon
    placeholder count are both configuration-dependent.
    """

    def __init__(self, C: int = 512, H: int = 64, P: int = INPUT_PATCH_LEN, strict: bool = True):
        if C % P:
            raise ValueError(f"context length {C} must be a multiple of the patch length {P}")
        if H > OUTPUT_PATCH_LEN:
            raise ValueError(
                f"horizon {H} > output_patch_len {OUTPUT_PATCH_LEN}: one token cannot cover it, "
                "so the readout would span more than the last real context patch and the "
                "RevIN statistics of the extra slots come from decode()'s nonlinear refine scan")
        self.C, self.H, self.P = int(C), int(H), int(P)
        self.n_real_context_patches = self.C // self.P            # 16
        self.selected_token_index = self.n_real_context_patches - 1   # 15
        self.rolls = OUTPUT_PATCH_LEN // self.P                   # 2
        self.extract_len = min(2 * self.P, OUTPUT_PATCH_LEN)      # decode()'s stitching window
        self.K = max(math.ceil((self.H - (self.extract_len - self.P)) / self.P), 1)   # 1
        self.n_horizon_patches = self.K + self.rolls - 1          # 2
        self.n_tokens = self.n_real_context_patches + self.n_horizon_patches          # 18
        self.target_start = self.C                                # 0-based, inclusive
        self.target_end = self.C + self.H                         # 0-based, exclusive
        self.span = self.C + self.H
        if strict:
            self.assert_headline_config()

    def assert_headline_config(self) -> None:
        """The spec's literal invariants. Raises with the offending value, never repairs."""
        checks = [
            (self.C == 512, f"raw context length is {self.C}, must be exactly 512"),
            (self.H == 64, f"horizon is {self.H}, must be exactly 64"),
            (self.P == INPUT_PATCH_LEN == 32, f"input patch size is {self.P}, must be 32"),
            (self.n_real_context_patches == 16,
             f"num_real_context_patches == {self.n_real_context_patches}, must be 16"),
            (self.selected_token_index == 15,
             f"selected_token_index == {self.selected_token_index}, must be 15"),
            (self.selected_token_index < self.n_real_context_patches,
             f"selected_token_index {self.selected_token_index} is not a REAL context patch "
             f"(only 0..{self.n_real_context_patches - 1} are)"),
            (self.n_tokens == 18, f"decode() layout is {self.n_tokens} tokens, must be 18 "
                                  f"(16 real context + 2 horizon placeholders)"),
            (self.K == 1, f"native forecast slot count K={self.K}, must be 1 for H=64"),
        ]
        bad = [m for ok, m in checks if not ok]
        if bad:
            raise RuntimeError("LastTokenGeometry is not the specified C=512 -> H=64 "
                               f"configuration: {bad}. Pass strict=False only if you have "
                               "re-derived the readout token for the new configuration.")

    def as_dict(self) -> dict:
        return {"C": self.C, "H": self.H, "P": self.P,
                "num_real_context_patches": self.n_real_context_patches,
                "selected_token_index": self.selected_token_index,
                "last_token_only": LAST_TOKEN_ONLY, "prefix_extraction": False,
                "n_horizon_patches": self.n_horizon_patches, "n_tokens": self.n_tokens,
                "native_K": self.K, "rolls": self.rolls, "extract_len": self.extract_len,
                "target_start_0based": self.target_start,
                "target_end_0based_exclusive": self.target_end,
                "target_start_1based": self.target_start + 1,
                "target_end_1based": self.target_end,
                "layer_names": LAYER_NAMES, "d": MODEL_DIMS,
                "num_quantiles": NUM_QUANTILES,
                "native_quantiles": [float(x) for x in NATIVE_QUANTILES]}

    def audit_row(self, detrend_active=None) -> dict:
        """The one-line audit the driver prints (the spec's table, collapsed to one origin)."""
        return {"raw_context_length": self.C,
                "num_real_context_patches": self.n_real_context_patches,
                "selected_token_index_0based": self.selected_token_index,
                "n_tokens_in_decode_pass": self.n_tokens,
                "target_start_1based": self.target_start + 1,
                "target_end_1based": self.target_end,
                "detrend_active": (None if detrend_active is None
                                   else float(np.mean(detrend_active))),
                "hidden_shape": (MODEL_DIMS,),
                "transformed_target_shape": (self.H,),
                "probe": f"Linear({MODEL_DIMS}, {self.H}*{NUM_QUANTILES})"}


def assert_native_geometry(model, geom: LastTokenGeometry) -> dict:
    """decode()'s OWN readout for this task must be exactly [geom.selected_token_index].

    This is the invariant that matters: not "token 15 looks right" but "the native forecast
    index computed from the model's own attributes is 15, and nothing else".
    """
    nat = native_readout_token(model, geom.C, geom.H)
    checks = [
        (model.input_patch_len == geom.P, f"model.input_patch_len {model.input_patch_len} "
                                          f"!= {geom.P}"),
        (model.output_patch_len == OUTPUT_PATCH_LEN,
         f"model.output_patch_len {model.output_patch_len} != {OUTPUT_PATCH_LEN}"),
        (nat == [geom.selected_token_index],
         f"decode()'s forecast_indices for (C={geom.C}, H={geom.H}) are {nat}, not "
         f"[{geom.selected_token_index}] -- the native readout moved"),
        (model.output_head.in_features == MODEL_DIMS,
         f"output_head.in_features {model.output_head.in_features} != {MODEL_DIMS}"),
        (model.output_head.out_features == OUTPUT_PATCH_LEN * NUM_QUANTILES,
         f"output_head.out_features {model.output_head.out_features} != "
         f"{OUTPUT_PATCH_LEN}*{NUM_QUANTILES}"),
    ]
    if getattr(model, "use_stitching", False):
        el = int(getattr(model, "_stitching_extract_len", geom.extract_len))
        checks.append((el == geom.extract_len,
                       f"model._stitching_extract_len {el} != geometry's {geom.extract_len}"))
    bad = [m for ok, m in checks if not ok]
    if bad:
        raise RuntimeError(f"native geometry mismatch: {bad}")
    return {"native_forecast_indices": nat, "selected_token_index": geom.selected_token_index,
            "num_real_context_patches": geom.n_real_context_patches,
            "use_stitching": bool(getattr(model, "use_stitching", False)),
            "value_clip": float(getattr(model, "value_clip", float("nan")))}


def assert_native_quantiles(model) -> dict:
    """Verify the quantile VECTOR and its ORDER from the loaded model, not from a constant.

    get_model() already refuses a checkpoint whose ``model.quantiles`` differ from
    NATIVE_QUANTILES; this adds the ordering/uniqueness contract and the exact 0.5 index that
    every median metric depends on.
    """
    qraw = model.quantiles
    q = np.asarray(qraw.detach().cpu().numpy() if torch.is_tensor(qraw) else qraw,
                   dtype=np.float64)
    validate_quantiles(q)                      # non-empty, strictly inside (0,1), increasing
    if q.size != NUM_QUANTILES:
        raise RuntimeError(f"model exposes {q.size} quantiles, expected {NUM_QUANTILES}")
    if not np.allclose(q, NATIVE_QUANTILES):
        raise RuntimeError(f"model.quantiles {q.tolist()} != {NATIVE_QUANTILES.tolist()}")
    mid = median_index(q)
    if mid != NATIVE_MEDIAN_IDX:
        raise RuntimeError(f"exact 0.5 sits at index {mid}, expected {NATIVE_MEDIAN_IDX}")
    nq = int(getattr(model, "num_quantiles", q.size))
    if nq != NUM_QUANTILES:
        raise RuntimeError(f"model.num_quantiles {nq} != {NUM_QUANTILES}")
    return {"quantiles": q.tolist(), "num_quantiles": nq, "median_index": mid,
            "strictly_increasing": True,
            "output_head_out_features": int(model.output_head.out_features)}


# --------------------------------------------------------------------------- #
# full-context preprocessing (numpy twin of decode(), fit on x[1:C] ONLY)
# --------------------------------------------------------------------------- #

def context_detrend_params(X, threshold: float = DETREND_THRESHOLD, enabled: bool = True):
    """Masked-OLS detrend parameters fit on the FULL observed context x[1:C].

    Delegates to the validated ``prefix_detrend_params`` at prefix length C, which normalizes
    the time axis by the supplied length -- exactly decode()'s t_ctx = arange(-(C-1), 1)/C.
    ``X`` must be the CONTEXT ONLY: passing the parent window would fit the trend on the future.
    """
    return prefix_detrend_params(X, threshold, enabled=enabled)


def context_trend(a, b, active, C: int, idx) -> np.ndarray:
    """Evaluate the context's fitted line at absolute indices ``idx`` (0 where inactive).

    ``idx >= C`` is the extrapolation onto the target window, i.e. decode()'s
    t_forecast = arange(1, H+1)/context.
    """
    return prefix_trend(a, b, active, C, idx)


def build_last_token_targets(Z, mu, sd, geom: LastTokenGeometry, *, detrend: bool = True,
                             threshold: float = DETREND_THRESHOLD) -> dict:
    """Targets in the native normalized forecasting space of the LAST REAL CONTEXT TOKEN.

    Z     : (n, C+H) raw parent windows (context ++ raw future)
    mu/sd : (n,)     the model's running RevIN statistics AT token ``selected_token_index``
    Returns
        targets (n, H) float32   probe supervision
        trend   (n, H) float64   add-back needed to invert a prediction to raw units
        a, b    (n,)   float64   the context's fitted line
        active  (n,)   bool      did detrending fire
        valid   (n,)   bool      usable window (finite target, sigma >= SIGMA_EPS)

    The detrend fit sees Z[:, :C] and nothing else, so no target value can enter any
    preprocessing statistic. sigma < SIGMA_EPS is MASKED, never silently rescaled: forward
    RevIN divides by 1.0 there while reverse RevIN multiplies by the raw sigma, so the library
    is not self-inverse in that regime.
    """
    Z = np.asarray(Z, np.float64)
    if Z.ndim != 2 or Z.shape[1] != geom.span:
        raise ValueError(f"parent windows must be (n, {geom.span}), got {Z.shape}")
    n = Z.shape[0]
    mu = np.asarray(mu, np.float64).reshape(-1)
    sd = np.asarray(sd, np.float64).reshape(-1)
    if mu.shape != (n,) or sd.shape != (n,):
        raise ValueError(f"mu/sd must be (n,) = ({n},), got {mu.shape} / {sd.shape}")

    a, b, active = context_detrend_params(Z[:, :geom.C], threshold, enabled=detrend)
    idx = np.arange(geom.target_start, geom.target_end)
    trend = context_trend(a, b, active, geom.C, idx)                        # (n, H)
    targets = normalize_target(Z[:, idx] - trend, mu, sd)                   # (n, H)
    valid = (sd >= SIGMA_EPS) & np.isfinite(targets).all(axis=1)
    return {"targets": targets.astype(np.float32), "trend": trend, "a": a, "b": b,
            "active": active, "valid": valid}


def assert_target_roundtrip(Z, targets, trend, mu, sd, geom: LastTokenGeometry, valid=None,
                            rtol: float = 1e-4) -> float:
    """y_raw -> y_transformed -> y_raw must close to relative max error < rtol. Abort otherwise.

    Guards the whole preprocessing chain (detrend fit, extrapolated trend, RevIN stats at
    token 15, safe-sigma masking) in ONE number, per dataset and per split.
    """
    rows = np.arange(len(Z)) if valid is None else np.flatnonzero(valid)
    if rows.size == 0:
        raise RuntimeError("no valid window to check the target round-trip on")
    back = denormalize(np.asarray(targets)[rows], mu=np.asarray(mu).reshape(-1)[rows],
                       sd=np.asarray(sd).reshape(-1)[rows], trend=np.asarray(trend)[rows])
    want = np.asarray(Z, np.float64)[rows, geom.target_start:geom.target_end]
    rel = float(np.abs(back - want).max() / (np.abs(want).mean() + 1e-12))
    if not np.isfinite(rel) or rel > rtol:
        raise RuntimeError(
            f"TARGET ROUND-TRIP FAILED: y_raw -> transformed -> y_raw misses the raw future by "
            f"relative max error {rel:.3e} > {rtol}. The RevIN/detrend algebra for the last "
            "real context token is wrong -- do NOT trust any probe numbers.")
    return rel


# --------------------------------------------------------------------------- #
# quantile-sorting discovery (sorting is POSTPROCESSING, never the reference)
# --------------------------------------------------------------------------- #

def decode_sorting_knobs(model) -> dict:
    """Find every way the installed timesfm3 could sort quantiles inside decode().

    Quantile sorting is a NONLINEAR postprocessing step. The L20 reconstruction check must
    compare against the RAW output-head path, so any such knob is disabled for the reference
    (and recorded, so the report states what the installed library actually does).
    """
    info: dict = {"decode_kwargs": {}, "decode_signature_defaults": {}, "attrs": {},
                  "source_mentions": []}
    try:
        sig = inspect.signature(model.decode)
    except (TypeError, ValueError):
        sig = None
    if sig is not None:
        info["decode_signature"] = str(sig)
        for name, p in sig.parameters.items():
            if "sort" in name.lower():
                info["decode_kwargs"][name] = False
                info["decode_signature_defaults"][name] = (
                    None if p.default is inspect.Parameter.empty else bool(p.default))
    for name in dir(type(model)) + list(vars(model).keys()):
        if "sort" not in name.lower() or name.startswith("__"):
            continue
        try:
            v = getattr(model, name)
        except Exception:
            continue
        if isinstance(v, bool):
            info["attrs"][name] = v
    try:
        src = inspect.getsource(type(model).decode)
        info["source_mentions"] = [ln.strip() for ln in src.splitlines()
                                   if "sort" in ln.lower()][:10]
    except Exception:
        pass
    return info


class no_quantile_sorting:
    """Context manager: temporarily disable any discovered quantile-sorting knob.

    ``.decode_kwargs`` are forwarded to decode(); boolean attributes are set False and
    restored on exit. Nothing is invented -- only names the installed library actually has.
    """

    def __init__(self, model, enabled: bool = True):
        self.model, self.enabled = model, bool(enabled)
        self.info = decode_sorting_knobs(model)
        self.decode_kwargs = dict(self.info["decode_kwargs"]) if enabled else {}
        self._prev: dict = {}

    def __enter__(self):
        if self.enabled:
            for name, val in self.info["attrs"].items():
                if val:                                    # only flip what is ON
                    self._prev[name] = val
                    setattr(self.model, name, False)
        return self

    def __exit__(self, *exc):
        for name, val in self._prev.items():
            setattr(self.model, name, val)
        self._prev.clear()
        return False

    def as_dict(self) -> dict:
        return {"bypass_enabled": self.enabled, "decode_kwargs": self.decode_kwargs,
                "attrs_found": self.info["attrs"], "attrs_overridden": sorted(self._prev),
                "decode_signature_defaults": self.info["decode_signature_defaults"],
                "source_mentions": self.info["source_mentions"]}


# --------------------------------------------------------------------------- #
# extraction: ONE full-context decode() pass per batch
# --------------------------------------------------------------------------- #

def select_last_token(h, token_index: int, *, n_real_context_patches: int, n_tokens: int,
                      d: int = MODEL_DIMS) -> torch.Tensor:
    """(b, n_var, n_tok, d) hidden states -> (b, d) at ``token_index``. Nothing is repaired.

    Refuses a multivariate pass, a wrong token count, a wrong hidden width, and any token
    index that is not a REAL context patch.
    """
    if h.ndim != 4:
        raise RuntimeError(f"hidden state must be (b, n_var, n_tok, d), got {tuple(h.shape)}")
    if h.shape[1] != 1:
        raise RuntimeError(f"expected ONE variate per pass, got {h.shape[1]} "
                           "-- TimesFM-3 is probed univariately here")
    if h.shape[2] != n_tokens:
        raise RuntimeError(f"decode() produced {h.shape[2]} tokens, expected {n_tokens} "
                           f"({n_real_context_patches} real context + "
                           f"{n_tokens - n_real_context_patches} horizon placeholders)")
    if h.shape[3] != d:
        raise RuntimeError(f"hidden width {h.shape[3]} != {d}")
    if not 0 <= token_index < n_real_context_patches:
        raise RuntimeError(f"token {token_index} is not a REAL context patch (real indices are "
                           f"0..{n_real_context_patches - 1}; "
                           f"{n_real_context_patches}..{n_tokens - 1} are placeholders)")
    return h[:, 0, token_index, :]


def feature_dtype_report(caps, layers, geom: LastTokenGeometry, feature_dtype) -> dict:
    """Quantify the storage cast, measured against the float32 states of the SAME pass.

    float16 halves the cache; this records exactly what that costs (max |h32 - cast(h32)|,
    absolute and relative to the layer's own std), so "float16 is fine" is a measurement.
    """
    dt = np.dtype(feature_dtype)
    worst_abs, worst_rel, worst_layer = 0.0, 0.0, None
    for L in layers:
        h32 = caps[f"L{L}"][:, 0, geom.selected_token_index, :].float().cpu().numpy()
        back = h32.astype(dt).astype(np.float32)
        a = float(np.abs(h32 - back).max())
        r = a / (float(h32.std()) + 1e-12)
        if r > worst_rel:
            worst_abs, worst_rel, worst_layer = a, r, LAYER_NAMES[L]
    return {"dtype": dt.name, "max_abs": worst_abs, "max_relative_to_layer_std": worst_rel,
            "worst_layer": worst_layer}


def extract_last_token_features(X, *, geom: LastTokenGeometry, model=None, checkpoint=None,
                                device=None, batch_size: int = 64, layers=None,
                                feature_dtype=np.float16, detrend: bool = True,
                                verify: bool = True, progress: bool = True,
                                allow_sorted_reference: bool = False,
                                bypass_sorting: bool = True) -> dict:
    """ONE decode() pass per batch; capture Emb,L1..L20; keep only token 15.

    X : (n, C) raw CONTEXTS. The future is never passed to the model -- asserted on the shape,
    so no call site can leak it.

    Returns feats {layer: (n, 1280)}, mu/sd (n,), native (n, H, 9) from decode(),
    plus the native-head reconstruction / dtype / sorting reports.

    NOTE on the native baseline: decode() runs INSIDE ``no_quantile_sorting``, so when the
    installed library exposes a sorting knob the captured native forecast is the RAW head path
    -- the same path the L20 check validates and the probes live in. Sorting is a monotone
    rearrangement that can only lower a pinball loss, so this makes the native baseline mildly
    CONSERVATIVE rather than flattering. Whatever was found and overridden is recorded in the
    returned ``sorting`` report and saved with the results.
    """
    # validated BEFORE the model is touched, so a caller that accidentally passes the parent
    # window (context ++ future) fails immediately and cheaply -- the future must never reach
    # the backbone.
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != geom.C:
        raise ValueError(f"X must be (n, {geom.C}) raw CONTEXTS (no future values), "
                         f"got {X.shape}")
    layers = list(range(NUM_LAYERS)) if layers is None else sorted(set(layers))
    bad = [L for L in layers if not 0 <= L < NUM_LAYERS]
    if bad:
        raise ValueError(f"layers {bad} outside 0..{NUM_LAYERS - 1}")

    if model is None:
        model, device = get_model(checkpoint, device)
    else:
        device = resolve_device(device)
    assert_backbone_frozen(model)
    assert_native_quantiles(model)
    assert_native_geometry(model, geom)
    n, H = len(X), geom.H

    feats = {L: np.zeros((n, MODEL_DIMS), dtype=feature_dtype) for L in layers}
    mu = np.full(n, np.nan, np.float64)
    sd = np.full(n, np.nan, np.float64)
    native = np.zeros((n, H, NUM_QUANTILES), np.float32)
    checks: dict = {}

    caps: dict = {}
    hs = register_layer_hooks(model, caps)
    try:
        with detrending(model, detrend), no_quantile_sorting(model, bypass_sorting) as srt:
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                tgt = torch.from_numpy(X[s:e]).to(device).unsqueeze(1)      # (b, 1, C)
                if tgt.shape[-1] != geom.C:
                    raise RuntimeError(f"context length {tgt.shape[-1]} != {geom.C}")
                caps.clear()
                with torch.no_grad():
                    official, aux = model.decode(target=tgt, horizon=H,
                                                 return_aux_outputs=True, **srt.decode_kwargs)
                # ---- geometry, asserted on EVERY batch, never repaired ----
                for L in layers:
                    h = select_last_token(
                        caps[f"L{L}"], geom.selected_token_index,
                        n_real_context_patches=geom.n_real_context_patches,
                        n_tokens=geom.n_tokens)
                    feats[L][s:e] = h.float().cpu().numpy().astype(feature_dtype)
                run_mu, run_sd = aux["revin_stats"]
                tok = geom.selected_token_index
                mu[s:e] = run_mu[:, 0, tok].float().cpu().numpy()
                sd[s:e] = run_sd[:, 0, tok].float().cpu().numpy()
                if official.shape[1:] != (1, H, NUM_QUANTILES):
                    raise RuntimeError(
                        f"decode() returned {tuple(official.shape)}, expected "
                        f"(b, 1, {H}, {NUM_QUANTILES})")
                native[s:e] = official[:, 0].float().cpu().numpy()
                if verify and "native" not in checks:
                    checks["sorting"] = srt.as_dict()
                    checks["dtype"] = feature_dtype_report(caps, layers, geom, feature_dtype)
                    checks["native"] = verify_native_head(
                        model, caps[f"L{LAST_LAYER}"], official, aux["revin_stats"],
                        X[s:e], geom, detrend=detrend,
                        allow_sorted_reference=allow_sorted_reference,
                        sorting=checks["sorting"])
                if progress and (s // max(batch_size, 1)) % 10 == 0:
                    print(f"  [extract] {e}/{n} windows (1 full-context pass per batch)",
                          flush=True)
    finally:
        for h in hs:
            h.remove()

    if np.isnan(mu).any() or np.isnan(sd).any():
        raise RuntimeError("extraction left NaN RevIN statistics -- incomplete pass")
    return {"feats": feats, "mu": mu, "sd": sd, "native": native,
            "native_check": checks.get("native"), "dtype_check": checks.get("dtype"),
            "sorting": checks.get("sorting"), "layers": layers}


# --------------------------------------------------------------------------- #
# THE mandatory L20 check: h_{L20,15} -> native Linear(1280, 64*9) -> decode()'s own output
# --------------------------------------------------------------------------- #

def verify_native_head(model, last_states, official, revin_stats, X_batch,
                       geom: LastTokenGeometry, *, detrend: bool, rtol: float = 1e-4,
                       allow_sorted_reference: bool = False, sorting=None) -> dict:
    """Push h_{L20,15} through the REAL native output head and reproduce decode(), all 9 q.

    Proves three things at once, on the SAME forward pass (no second decode):
      1. we extract the state the native head actually consumes (token 15, layer L20);
      2. the flat head layout is HORIZON-major -- (H, Q), index = t*Q + q. The transposed
         layout is computed too and must be clearly REJECTED, so the check has power;
      3. our inverse RevIN + clamp + stitch + trend add-back is the native one.

    Compares ALL NINE quantiles (not just the median) and aborts above ``rtol``. Quantile
    sorting, if the installed library applies any, is POSTPROCESSING: it is bypassed for this
    reference when a knob exists, and ``allow_sorted_reference`` must be set EXPLICITLY before
    a sorted reference is ever accepted.
    """
    from timesfm3.torch import util as tfm_util
    P, Q = model.input_patch_len, model.num_quantiles
    if Q != NUM_QUANTILES:
        raise RuntimeError(f"model.num_quantiles {Q} != {NUM_QUANTILES}")
    tok = geom.selected_token_index
    b = len(X_batch)
    run_mu, run_sd = revin_stats
    with torch.no_grad():
        raw = model.output_head(last_states)[:, :, [tok], :]         # (b, 1, 1, H*Q)
        if raw.shape[-1] != model.output_patch_len * Q:
            raise RuntimeError(f"native head emitted {raw.shape[-1]} values, expected "
                               f"{model.output_patch_len}*{Q}")
        den = tfm_util.revin(raw, run_mu[:, :, [tok]], run_sd[:, :, [tok]], reverse=True)
        clip_frac = float((den.abs() > model.value_clip).to(torch.float64).mean().item())
        den = torch.clamp(den, -model.value_clip, model.value_clip)

        def stitch(view5):
            out = tfm_util.stitch_patches(view5[:, :, :, :geom.extract_len, :], P)
            return out[:, :, :geom.H, :][:, 0].float().cpu().numpy().astype(np.float64)

        # assumed layout: (..., output_patch_len, Q) -- horizon-major (flat index t*Q + q)
        rec = stitch(den.reshape(b, 1, 1, model.output_patch_len, Q))
        # the WRONG layout (flat index q*H + t), kept as a discrimination control
        rec_T = stitch(den.reshape(b, 1, 1, Q, model.output_patch_len).transpose(-1, -2)
                       .contiguous())

    a, bb, act = context_detrend_params(np.asarray(X_batch, np.float64)[:, :geom.C],
                                        enabled=detrend)
    trend = context_trend(a, bb, act, geom.C,
                          np.arange(geom.target_start, geom.target_end))[:, :, None]
    rec, rec_T = rec + trend, rec_T + trend
    ref = official[:, 0].float().cpu().numpy().astype(np.float64)   # (b, H, Q)
    if ref.shape != rec.shape:
        raise RuntimeError(f"decode() output {ref.shape} != reconstruction {rec.shape}")

    scale = float(np.abs(ref).mean()) + 1e-12
    d_all = np.abs(rec - ref)
    err = float(d_all.max())
    rel = err / scale
    rel_T = float(np.abs(rec_T - ref).max()) / scale
    spread = float(np.mean(rec[:, :, -1] - rec[:, :, 0]))            # q0.9 - q0.1, raw units
    mono = float(np.mean(np.all(np.diff(ref, axis=-1) >= -1e-9, axis=-1)))
    rec_sorted = np.sort(rec, axis=-1)
    rel_sorted = float(np.abs(rec_sorted - ref).max()) / scale
    report = {
        "max_abs": err, "relative": rel, "all_quantiles_max_abs": err,
        "median_max_abs": float(d_all[:, :, NATIVE_MEDIAN_IDX].max()),
        "per_quantile_max_abs": d_all.max(axis=(0, 1)).tolist(),
        "quantiles": [float(x) for x in NATIVE_QUANTILES],
        "reference_scale_mean_abs": scale, "rtol": rtol,
        "transposed_layout_relative": rel_T, "quantile_spread_q90_minus_q10": spread,
        "native_clip_fraction": clip_frac,
        "official_monotone_in_quantile_fraction": mono,
        "sorted_reconstruction_relative": rel_sorted, "sorted_reference_used": False,
        "n_windows_checked": int(b), "sorting": sorting,
    }
    if rel > rtol:
        if allow_sorted_reference and rel_sorted <= rtol:
            report.update(sorted_reference_used=True, relative=rel_sorted,
                          max_abs=float(np.abs(rec_sorted - ref).max()))
            print("  [WARN] decode() SORTS its quantiles: the raw head path differs "
                  f"(relative {rel:.3e}) but sort(reconstruction) matches "
                  f"({rel_sorted:.3e}). Accepted only because --allow-sorted-reference was "
                  "passed; sorting is treated as postprocessing and is NOT part of the "
                  "linear-readout analysis.")
            return report
        raise RuntimeError(
            f"L20 NATIVE ALL-QUANTILE RECONSTRUCTION FAILED: max|d| = {err:.3e} "
            f"(relative {rel:.3e} > {rtol}) over all {Q} quantiles; median-only "
            f"{report['median_max_abs']:.3e}; per-quantile "
            f"{report['per_quantile_max_abs']}.\n"
            f"  transposed-layout relative error : {rel_T:.3e}\n"
            f"  sort(reconstruction) vs decode() : {rel_sorted:.3e}\n"
            f"  decode() monotone in quantile    : {mono:.1%} of (window, step)\n"
            f"  sorting knobs found              : {sorting}\n"
            "  If sort(reconstruction) matches, decode() applies quantile SORTING and no "
            "bypass knob was found: re-run with --allow-sorted-reference to record that "
            "explicitly. Otherwise the installed timesfm3 postprocessing no longer matches "
            "this module -- do NOT trust any probe numbers.")
    if rel_T <= 10 * rtol:
        raise RuntimeError(
            f"the output-head LAYOUT CHECK HAS NO POWER: the transposed (Q, H) layout also "
            f"matches decode() (relative {rel_T:.3e}). Cannot claim the horizon-major layout "
            "is verified -- investigate before trusting the probe's reshape.")
    if spread <= 100 * rtol * scale:
        raise RuntimeError(
            f"the reconstructed quantiles are nearly identical (mean q0.9-q0.1 = {spread:.3e}), "
            f"so an all-quantile agreement of {err:.3e} proves nothing about the layout.")
    return report


# --------------------------------------------------------------------------- #
# cache -- a namespace the prefix ablation can never be read from
# --------------------------------------------------------------------------- #

def cache_metadata(tag, split, geom: LastTokenGeometry, *, checkpoint, detrend, layers, seed,
                   feature_dtype=np.float16) -> dict:
    """Everything that could change a feature value. Any disagreement REJECTS the cache."""
    return {"cache_version": CACHE_VERSION, "model": "timesfm-3.0", "checkpoint": checkpoint,
            "timesfm_version": timesfm_version(),
            "context_len": geom.C, "horizon": geom.H, "patch_size": geom.P,
            "num_real_context_patches": geom.n_real_context_patches,
            "selected_token_index": geom.selected_token_index,
            "last_token_only": True, "prefix_extraction": False, "n_origins": 1,
            "num_quantiles": NUM_QUANTILES,
            "native_quantiles": [float(x) for x in NATIVE_QUANTILES],
            "layers": list(layers), "layer_names": [LAYER_NAMES[i] for i in layers],
            "hidden_dim": MODEL_DIMS, "feature_shape_rank": 2,
            "feature_dtype": np.dtype(feature_dtype).name,
            "detrending": bool(detrend), "dataset": tag, "split": split, "seed": int(seed)}


def cache_root(cache_dir, tag, split, geom: LastTokenGeometry, detrend: bool) -> Path:
    return Path(cache_dir) / (f"{CACHE_VERSION}__{tag}__{split}__C{geom.C}_H{geom.H}"
                              f"{'' if detrend else '__nodetrend'}")


def read_cache(root, meta_expected: dict, X, layers) -> dict | None:
    """Load a cache ONLY if every metadata field and every array shape agrees.

    Returns None when there is nothing to load (no meta, or a missing layer file). Raises on
    any DISAGREEMENT -- in particular a (N, 16, 1280) shared-origin array is rejected by the
    rank/shape check even if its directory were renamed by hand. No model is needed to reach
    either outcome, so this is testable on a login node.
    """
    root = Path(root)
    mp = root / "meta.npz"
    if not mp.exists():
        return None
    X = np.asarray(X, dtype=np.float32)
    with np.load(mp, allow_pickle=False) as d:
        cached = json.loads(str(d["meta"]))
        diffs = {k: (cached.get(k, "<missing>"), meta_expected[k])
                 for k in meta_expected if cached.get(k, "<missing>") != meta_expected[k]}
        if diffs:
            raise RuntimeError(
                f"INCOMPATIBLE feature cache {root}:\n" +
                "\n".join(f"    {k}: cached={c!r} current={n!r}" for k, (c, n) in diffs.items())
                + f"\n  This namespace is {CACHE_VERSION} (last-token, 2-D features). The "
                  "16-prefix ablation's cache is NEVER compatible. Delete the directory or "
                  "point --cache-dir elsewhere; nothing is reused silently.")
        if d["ctx_tail"].shape[0] != len(X) or not np.allclose(d["ctx_tail"], X[:, -8:]):
            raise RuntimeError(f"stale feature cache {root}: cached contexts do not match the "
                               f"current {meta_expected['split']} windows -- delete it and re-run")
        missing = [L for L in layers if not (root / f"L{L:02d}.npy").exists()]
        if missing:
            return None
        feats = {}
        for L in layers:
            arr = np.load(root / f"L{L:02d}.npy", mmap_mode="r")
            if arr.ndim != 2 or arr.shape != (len(X), MODEL_DIMS):
                raise RuntimeError(
                    f"cache {root / f'L{L:02d}.npy'} holds {arr.shape} (rank {arr.ndim}); this "
                    f"experiment needs ({len(X)}, {MODEL_DIMS}). A rank-3 array is the "
                    "shared-origin (N, 16, 1280) cache -- it can NOT be loaded here.")
            feats[L] = arr
        return {"feats": feats, "mu": d["mu"], "sd": d["sd"], "native": d["native"],
                "native_check": json.loads(str(d["native_check"])),
                "dtype_check": json.loads(str(d["dtype_check"])),
                "sorting": json.loads(str(d["sorting"])), "layers": list(layers),
                "cache_hit": True}


def cached_last_token_features(tag, split, X, *, geom: LastTokenGeometry, cache_dir,
                              checkpoint=None, seed=0, layers=None, detrend=True,
                              feature_dtype=np.float16, force=False, **kw) -> dict:
    """Disk cache, one .npy per representation point (so a probe fit holds ONE layer at a time)."""
    layers = list(range(NUM_LAYERS)) if layers is None else sorted(set(layers))
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    meta = cache_metadata(tag, split, geom, checkpoint=checkpoint, detrend=detrend,
                          layers=layers, seed=seed, feature_dtype=feature_dtype)
    root = cache_root(cache_dir, tag, split, geom, detrend)
    X = np.asarray(X, dtype=np.float32)

    if not force:
        hit = read_cache(root, meta, X, layers)
        if hit is not None:
            print(f"  [cache HIT]  {root.name}  ({len(X)} windows, {len(layers)} layers, "
                  f"(N, {MODEL_DIMS}) per layer)")
            return hit

    root.mkdir(parents=True, exist_ok=True)
    out = extract_last_token_features(X, geom=geom, checkpoint=checkpoint, layers=layers,
                                      detrend=detrend, feature_dtype=feature_dtype, **kw)
    for L, arr in out["feats"].items():
        np.save(root / f"L{L:02d}.npy", arr)
    np.savez(root / "meta.npz", meta=json.dumps(meta), ctx_tail=X[:, -8:], mu=out["mu"],
             sd=out["sd"], native=out["native"],
             native_check=json.dumps(out["native_check"]),
             dtype_check=json.dumps(out["dtype_check"]),
             sorting=json.dumps(out["sorting"]))
    nbytes = sum((root / f"L{L:02d}.npy").stat().st_size for L in layers)
    print(f"  [saved]      {root}  ({len(X)} windows, {len(layers)} layers, "
          f"{nbytes / 1e6:.0f} MB)")
    out["feats"] = {L: np.load(root / f"L{L:02d}.npy", mmap_mode="r") for L in layers}
    out["cache_hit"] = False
    return out
