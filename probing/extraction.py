"""Chronos-2 hidden-state extraction and per-layer probe fitting (reusable core).

This module is the workhorse the whole pipeline is built on. It is a verbatim extraction
of the reusable functions originally defined in ``probe_pipeline.py`` — the logic and the
numbers are unchanged; only the module location and the imports moved.

Public API (imported across the codebase):
    get_pipeline()          -> load the frozen Chronos-2 pipeline (float32, MPS/CPU, no grad)
    extract_features(...)   -> per-layer, per-pooling features with on-disk caching
    fit_layerwise_probes()  -> one StandardScaler + LogisticRegression per layer

Chronos-2 weights are frozen throughout (eval, no_grad, requires_grad=False).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probing.config import SEED, NUM_LAYERS, CACHE_DIR

# Module-level singletons for the lazily-loaded Chronos-2 pipeline (populated by
# get_pipeline on first call). Declared here so get_pipeline's `global` refers to a
# defined name; every post-refactor run so far was a cache hit, which is why this was
# never exercised until now.
_PIPELINE = None
_CFG = None


def get_pipeline():
    global _PIPELINE, _CFG
    if _PIPELINE is not None:
        return _PIPELINE, _CFG

    from chronos import Chronos2Pipeline

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[model] loading amazon/chronos-2 (float32) -> target device {device}")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", torch_dtype=torch.float32)
    inner = pipeline.model
    try:
        inner.to(device)
    except Exception as e:  # pragma: no cover
        warnings.warn(f"to({device}) failed: {e}; falling back to CPU")
        inner.to("cpu")
    inner.eval()
    for p in inner.parameters():
        p.requires_grad_(False)
    print(f"[model] params on {next(inner.parameters()).device}, "
          f"d_model={inner.config.d_model}, num_layers={inner.config.num_layers}, "
          f"patch_size={inner.chronos_config.input_patch_size}")
    _PIPELINE = pipeline
    _CFG = inner.config
    return pipeline, _CFG


# ----------------------------------------------------------------------- #
# Phase A — extract_features
# ----------------------------------------------------------------------- #

def _corr_repr(corruption):
    if corruption is None:
        return "clean"
    parts = [f"{k}={corruption[k]}" for k in sorted(corruption.keys())]
    return ",".join(parts)


def _cache_path(dataset, split, corruption, pooling):
    return CACHE_DIR / f"{dataset}__{split}__{_corr_repr(corruption)}__{pooling}.npz"


def _all_pool_caches_exist(dataset, split, corruption):
    return all(_cache_path(dataset, split, corruption, p).exists()
               for p in ("content", "all", "reg"))


def _load_cache(path):
    d = np.load(path, allow_pickle=True)
    features = {int(k.split("_")[1]): d[k] for k in d.files if k.startswith("layer_")}
    y = d["y"]
    return features, y


def _apply_gaussian_noise(X, alpha, seed):
    """Per-series, per-channel Gaussian noise with std = alpha * std(series, channel)."""
    rng = np.random.default_rng(seed)
    std = X.std(axis=-1, keepdims=True)  # (n, c, 1)
    noise = rng.standard_normal(X.shape).astype(np.float32) * (alpha * std).astype(np.float32)
    return (X + noise).astype(np.float32)


def _apply_timewarp(X, factor):
    """Label-preserving time warp via linear interpolation.

    Each series of length t is resampled onto a new grid of length round(t*factor),
    then center-cropped or edge-padded back to t. factor=1 is identity.
    """
    n, c, t = X.shape
    new_length = int(round(t * factor))
    old_idx = np.arange(t, dtype=np.float64)
    new_idx = np.linspace(0.0, t - 1.0, new_length)
    out = np.empty_like(X)
    for i in range(n):
        for ch in range(c):
            warped = np.interp(new_idx, old_idx, X[i, ch])
            if new_length == t:
                out[i, ch] = warped
            elif new_length > t:
                start = (new_length - t) // 2
                out[i, ch] = warped[start:start + t]
            else:
                pad_left = (t - new_length) // 2
                pad_right = t - new_length - pad_left
                out[i, ch] = np.pad(warped, (pad_left, pad_right), mode="edge")
    return out.astype(np.float32)


def _apply_drift(X, amplitude):
    """Half-cycle sinusoid added to each (series, channel), scaled by per-series-channel std."""
    n, c, t = X.shape
    std = X.std(axis=-1, keepdims=True)  # (n, c, 1)
    phase = np.sin(np.linspace(0.0, np.pi, t)).astype(np.float32)
    drift = (amplitude * std).astype(np.float32) * phase[None, None, :]
    return (X + drift).astype(np.float32)


def _apply_corruption(X, corruption):
    if corruption is None:
        return X
    kind = corruption["kind"]
    if kind == "gauss":
        return _apply_gaussian_noise(X, alpha=corruption["alpha"], seed=corruption["seed"])
    if kind == "timewarp":
        return _apply_timewarp(X, factor=corruption["factor"])
    if kind == "drift":
        return _apply_drift(X, amplitude=corruption["amplitude"])
    raise ValueError(f"unknown corruption kind: {kind!r}")


def extract_features(dataset_name, split, corruption=None, batch_size=64, pooling="content"):
    """Hook-based per-layer feature extraction with on-disk caching.

    Layout note (verified from chronos.chronos2.model.Chronos2Model.encode):
        encoder input is [content_patches..., REG, masked_future_patch].
        For embed() prediction_length=0 -> num_output_patches=1, so
        P = num_context_patches + 2  where num_context_patches = ceil(t/patch_size).

    Pooling variants computed in one pass and cached separately:
        - pool_content : mean over [0 .. num_context_patches-1]
        - pool_all     : mean over all P positions
        - pool_reg     : the REG token at index num_context_patches
    """
    cache_path = _cache_path(dataset_name, split, corruption, pooling)
    if cache_path.exists():
        print(f"  [cache HIT]  {cache_path.name}")
        return _load_cache(cache_path)

    if _all_pool_caches_exist(dataset_name, split, corruption):
        print(f"  [cache HIT]  {cache_path.name} (sibling pools present, requested missing)")
        # extremely unlikely branch but defensive

    print(f"  [cache MISS] extracting {dataset_name}/{split}  corruption={_corr_repr(corruption)}")

    from aeon.datasets import load_classification
    X, y = load_classification(dataset_name, split=split)
    X = np.asarray(X, dtype=np.float32)
    X = _apply_corruption(X, corruption)
    n, c, t = X.shape
    print(f"     X.shape={X.shape}  classes={sorted(np.unique(y).tolist())}")

    pipeline, cfg = get_pipeline()
    patch_size = pipeline.model.chronos_config.input_patch_size
    num_context_patches = math.ceil(t / patch_size)
    P_expected = num_context_patches + 2
    print(f"     patch_size={patch_size}  num_context_patches={num_context_patches}  "
          f"P_expected={P_expected}")

    # per_channel[pool][ch][layer] = (n, 768) float32 numpy array
    per_channel = {p: {ch: {} for ch in range(c)} for p in ("content", "all", "reg")}

    for ch in range(c):
        # Build univariate input list (one 1D tensor per case)
        inputs = [torch.from_numpy(np.ascontiguousarray(X[i, ch])).to(torch.float32) for i in range(n)]

        captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}

        def make_hook(layer_idx):
            def hook(_m, _inp, output):
                # Chronos2EncoderBlockOutput is a HF ModelOutput; .hidden_states is set,
                # and [0] also returns it (matches encoder.forward's layer_outputs[0]).
                if hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hs = output.hidden_states
                elif isinstance(output, (tuple, list)):
                    hs = output[0]
                else:
                    hs = output[0]
                # Move off-device immediately to keep MPS memory low.
                captured[layer_idx].append(hs.detach().to("cpu"))
            return hook

        hooks = [blk.register_forward_hook(make_hook(i))
                 for i, blk in enumerate(pipeline.model.encoder.block)]
        try:
            with torch.no_grad():
                _ = pipeline.embed(inputs, batch_size=batch_size)
        finally:
            for h in hooks:
                h.remove()

        for i in range(NUM_LAYERS):
            full = torch.cat(captured[i], dim=0)  # (n, P, 768)
            assert full.shape[0] == n, f"hook batch dim {full.shape[0]} != n={n} (ch={ch}, layer={i})"
            P_actual = full.shape[1]
            if ch == 0 and i == 0:
                assert P_actual == P_expected, (
                    f"P mismatch: expected {P_expected} (= ceil({t}/{patch_size})+2), got {P_actual}"
                )
                print(f"     P verified: actual={P_actual} (matches expected)")
            per_channel["content"][ch][i] = full[:, :num_context_patches, :].mean(dim=1).numpy()
            per_channel["all"][ch][i]     = full.mean(dim=1).numpy()
            per_channel["reg"][ch][i]     = full[:, num_context_patches, :].numpy()

        del captured

    # Concat across channels and write all three pooling caches.
    for pool_name in ("content", "all", "reg"):
        features = {
            i: np.concatenate([per_channel[pool_name][ch][i] for ch in range(c)], axis=1)
            for i in range(NUM_LAYERS)
        }
        cp = _cache_path(dataset_name, split, corruption, pool_name)
        save_kwargs = {f"layer_{i}": features[i] for i in range(NUM_LAYERS)}
        save_kwargs["y"] = y
        np.savez(cp, **save_kwargs)
        print(f"  [saved]      {cp.name}  per-layer shape={features[0].shape}")

    # Return the requested pooling.
    features = {
        i: np.concatenate([per_channel[pooling][ch][i] for ch in range(c)], axis=1)
        for i in range(NUM_LAYERS)
    }
    return features, y


# ----------------------------------------------------------------------- #
# Probe utilities
# ----------------------------------------------------------------------- #

def fit_layerwise_probes(train_features, y_train):
    """Fit one StandardScaler + LogisticRegression per layer. Returns dict per layer."""
    classes = sorted(np.unique(y_train).tolist())
    print(f"     n_train={len(y_train)}  classes={classes}")
    probes = {}
    for i in range(NUM_LAYERS):
        X_tr = train_features[i]
        scaler = StandardScaler().fit(X_tr)
        Xs = scaler.transform(X_tr)
        clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xs, y_train)
        probes[i] = {"scaler": scaler, "clf": clf}
    return probes
