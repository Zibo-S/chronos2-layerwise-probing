"""Chronos-2 hidden-state extraction (reusable core).

Loads the frozen Chronos-2 encoder and pulls per-layer hidden states off it via forward
hooks, pooling each layer to a fixed-width feature per window and caching to disk.

Public API (imported across the codebase):
    get_pipeline()             -> load the frozen Chronos-2 pipeline (float32, CUDA/MPS/CPU, no grad)
    extract_window_features()  -> per-layer, per-pooling windowed features with on-disk caching
    extract_features()         -> UEA classification features (per-layer, per-pooling, cached)
    fit_layerwise_probes()     -> one StandardScaler + LogisticRegression per layer

Chronos-2 weights are frozen throughout (eval, no_grad, requires_grad=False).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from probing.config import SEED, NUM_LAYERS, CACHE_DIR, OUTPUT_PATCH_SIZE

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

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
# Cache-path helpers (shared by the windowed extractor below)
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
# Windowed ID-forecasting extraction. Memory-safe: pools each batch immediately
# instead of accumulating all (n, P, 768) hidden states, so a large window count
# stays well within GPU memory. Cached under an ``IDF_`` prefix.
# ----------------------------------------------------------------------- #

def extract_window_features(tag, split, contexts, y, pooling="content", batch_size=128):
    """Per-layer features for a set of univariate context windows.

    Parameters
    ----------
    tag       : dataset tag (e.g. "solar_1h"); cached as ``IDF_<tag>``.
    split     : "train" | "test".
    contexts  : float32 array (n_windows, C) — one context window per row.
    y         : label array (n_windows,), stored alongside the features.
    pooling   : "content" | "all" | "reg" (per-patch pooling variant).

    Returns ({layer: (n_windows, 768)}, y). All three poolings are written to the
    cache in one pass, so a later request for a different pooling is free.
    """
    cache_dataset = f"IDF_{tag}"
    cache_path = _cache_path(cache_dataset, split, None, pooling)
    if cache_path.exists():
        feats, y_cached = _load_cache(cache_path)
        # the cache key ignores windowing params, so verify row-alignment against the
        # labels the caller just built — a stale cache must fail loudly, not misalign.
        if len(y_cached) != len(y) or not np.allclose(y_cached, np.asarray(y)):
            raise RuntimeError(
                f"stale feature cache {cache_path.name}: cached labels do not match the "
                f"current windows (n_cached={len(y_cached)}, n_now={len(y)}) — the "
                f"windowing changed; delete features_cache/IDF_{tag}__* and re-extract")
        print(f"  [cache HIT]  {cache_path.name}")
        return feats, y_cached

    print(f"  [cache MISS] extracting {cache_dataset}/{split}  n_windows={len(contexts)}")
    contexts = np.asarray(contexts, dtype=np.float32)
    n, C = contexts.shape

    pipeline, cfg = get_pipeline()
    patch_size = pipeline.model.chronos_config.input_patch_size
    num_context_patches = math.ceil(C / patch_size)
    P_expected = num_context_patches + 2
    print(f"     C={C}  patch_size={patch_size}  num_context_patches={num_context_patches}  "
          f"P_expected={P_expected}  batches of {batch_size}")

    # accumulate only the POOLED (n, 768) arrays per pooling per layer
    pooled = {p: {i: [] for i in range(NUM_LAYERS)} for p in ("content", "all", "reg")}

    for b0 in range(0, n, batch_size):
        batch = contexts[b0:b0 + batch_size]
        inputs = [torch.from_numpy(np.ascontiguousarray(batch[j])).to(torch.float32)
                  for j in range(len(batch))]
        captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}

        def make_hook(layer_idx):
            def hook(_m, _inp, output):
                if hasattr(output, "hidden_states") and output.hidden_states is not None:
                    hs = output.hidden_states
                else:
                    hs = output[0]
                captured[layer_idx].append(hs.detach().to("cpu"))
            return hook

        hooks = [blk.register_forward_hook(make_hook(i))
                 for i, blk in enumerate(pipeline.model.encoder.block)]
        try:
            with torch.no_grad():
                _ = pipeline.embed(inputs, batch_size=len(inputs))
        finally:
            for h in hooks:
                h.remove()

        for i in range(NUM_LAYERS):
            full = torch.cat(captured[i], dim=0)          # (b, P, 768)
            if b0 == 0 and i == 0:
                assert full.shape[1] == P_expected, f"P {full.shape[1]} != expected {P_expected}"
            pooled["content"][i].append(full[:, :num_context_patches, :].mean(dim=1).numpy())
            pooled["all"][i].append(full.mean(dim=1).numpy())
            pooled["reg"][i].append(full[:, num_context_patches, :].numpy())
        del captured

    for pool_name in ("content", "all", "reg"):
        features = {i: np.concatenate(pooled[pool_name][i], axis=0) for i in range(NUM_LAYERS)}
        cp = _cache_path(cache_dataset, split, None, pool_name)
        save_kwargs = {f"layer_{i}": features[i] for i in range(NUM_LAYERS)}
        save_kwargs["y"] = np.asarray(y)
        np.savez(cp, **save_kwargs)
        print(f"  [saved]      {cp.name}  per-layer shape={features[0].shape}")

    features = {i: np.concatenate(pooled[pooling][i], axis=0) for i in range(NUM_LAYERS)}
    return features, np.asarray(y)


# ----------------------------------------------------------------------- #
# K-output-patch extraction. extract_window_features goes through pipeline.embed(),
# which hardcodes num_output_patches=1 (a single forecast slot). This runs the encoder
# with K = ceil(horizon / output_patch_size) forecast tokens — Chronos-2's own rule for a
# horizon-H forecast (pipeline.get_num_output_patches) — via a direct
# model.encode(..., num_output_patches=K) call, and returns content/REG/forecast-slot
# states (plus the post-final-norm state) from that ONE pass. Cached under a
# ``K<K>_H<horizon>`` tag so different horizons never collide.
# ----------------------------------------------------------------------- #

def extract_kout_features(tag, split, contexts, y, horizon, batch_size=128):
    """content / REG / forecast-slot states from ONE num_output_patches=K forward pass,
    with K = ceil(horizon / OUTPUT_PATCH_SIZE) derived here (NOT passed in) so extraction
    and the shared probe can never disagree on the slot count.

    Because the encoder's self-attention is NON-causal, the context/REG states under K forecast
    tokens differ from the K=1 cache; we therefore re-derive content_K and reg_K from THIS pass
    so the pooled-vs-shared comparison is fully controlled (one encoder config for all three).

    Also returns the post-final-layer-norm representation ('final') = the actual tensor the native
    output head consumes (encoder.final_layer_norm output, model.py:190). NOTE: this is ONLY the
    real final output (= L11 run through the final norm). The L0..L11 block-hook states are
    PRE-final-norm and are NOT native-head-compatible for L0..L10.

    Returns (feats, final, y):
      feats = {"content": {L:(n,768)}, "reg": {L:(n,768)}, "fslot": {L:(n,K,768)}}  L in 0..11
      final = {"content": (n,768),     "reg": (n,768),     "fslot": (n,K,768)}       post-final-norm
    """
    K = math.ceil(horizon / OUTPUT_PATCH_SIZE)   # native rule: pipeline.get_num_output_patches
    cache_path = _cache_path(f"IDF_{tag}", split, None, f"K{K}_H{horizon}")
    types = ("content", "reg", "fslot")
    if cache_path.exists():
        d = np.load(cache_path, allow_pickle=True)
        y_cached = d["y"]
        if len(y_cached) != len(y) or not np.allclose(y_cached, np.asarray(y)):
            raise RuntimeError(
                f"stale K{K} cache {cache_path.name}: cached labels do not match the current "
                f"windows — delete features_cache/{cache_path.name} and re-extract")
        feats = {t: {i: d[f"{t}_L{i}"] for i in range(NUM_LAYERS)} for t in types}
        final = {t: d[f"{t}_final"] for t in types}
        assert feats["fslot"][0].shape[1] == K, (
            f"cache {cache_path.name} carries {feats['fslot'][0].shape[1]} forecast slots but the "
            f"filename/horizon implies K={K} — file renamed or corrupted; delete and re-extract")
        print(f"  [cache HIT]  {cache_path.name}")
        return feats, final, y_cached

    print(f"  [cache MISS] extracting {tag}/{split}  K={K}  n_windows={len(contexts)}")
    contexts = np.asarray(contexts, dtype=np.float32)
    n, C = contexts.shape

    pipeline, cfg = get_pipeline()
    model = pipeline.model
    model.eval()                                           # no_grad does NOT disable dropout; eval does
    device = next(model.parameters()).device
    patch_size = model.chronos_config.input_patch_size
    # K was derived above as ceil(horizon / OUTPUT_PATCH_SIZE) from the config constant; fail
    # loudly if a swapped-in model's actual output patch size differs from that constant.
    assert model.chronos_config.output_patch_size == OUTPUT_PATCH_SIZE, (
        f"model output_patch_size {model.chronos_config.output_patch_size} != config "
        f"OUTPUT_PATCH_SIZE {OUTPUT_PATCH_SIZE}; K was derived from the constant — update config")
    assert K == math.ceil(horizon / model.chronos_config.output_patch_size), (
        f"K={K} != ceil({horizon}/{model.chronos_config.output_patch_size}) — K derivation drifted")
    num_special = int(model.chronos_config.use_reg_token)  # 1 REG token for chronos-2
    ncp = math.ceil(C / patch_size)
    reg_idx = ncp                                          # [context(ncp) | REG | K forecast slots]
    P_expected = ncp + num_special + K
    print(f"     C={C}  patch_size={patch_size}  ncp={ncp}  use_reg={num_special}  K={K}  "
          f"P_expected={P_expected}  batches of {batch_size}")

    def pool_content(hs): return hs[:, :ncp, :].mean(dim=1).numpy()
    def pool_reg(hs):     return hs[:, reg_idx, :].numpy() if num_special else pool_content(hs)
    def pool_fslot(hs):   return hs[:, -K:, :].numpy()
    poolers = {"content": pool_content, "reg": pool_reg, "fslot": pool_fslot}

    acc = {t: {i: [] for i in range(NUM_LAYERS)} for t in types}
    acc_final = {t: [] for t in types}

    for b0 in range(0, n, batch_size):
        batch = contexts[b0:b0 + batch_size]
        ctx = torch.from_numpy(np.ascontiguousarray(batch)).to(device=device, dtype=torch.float32)
        captured: dict[int, list[torch.Tensor]] = {i: [] for i in range(NUM_LAYERS)}

        def make_hook(layer_idx):
            def hook(_m, _inp, output):
                hs = output.hidden_states if getattr(output, "hidden_states", None) is not None else output[0]
                captured[layer_idx].append(hs.detach().to("cpu"))
            return hook

        hooks = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(model.encoder.block)]
        try:
            with torch.no_grad():
                # model.encode returns a 4-tuple; enc_out = Chronos2EncoderOutput, enc_out[0] = hidden states
                enc_out, *_ = model.encode(context=ctx, num_output_patches=K)  # group_ids=None -> independent
        finally:
            for h in hooks:
                h.remove()

        final_hs = enc_out[0].detach().cpu()               # (b, P, 768) POST final_layer_norm = native-head input
        if b0 == 0:
            assert final_hs.ndim == 3, f"expected (b, P, 768), got {tuple(final_hs.shape)}"
            assert final_hs.shape[1] == P_expected, f"P={final_hs.shape[1]} != expected {P_expected}"
        for t in types:
            acc_final[t].append(poolers[t](final_hs))

        for i in range(NUM_LAYERS):
            hs = torch.cat(captured[i], dim=0)             # (b, P, 768) block output = PRE final-norm
            for t in types:
                acc[t][i].append(poolers[t](hs))
        del captured

    feats = {t: {i: np.concatenate(acc[t][i], axis=0) for i in range(NUM_LAYERS)} for t in types}
    final = {t: np.concatenate(acc_final[t], axis=0) for t in types}

    save = {"y": np.asarray(y)}
    for t in types:
        for i in range(NUM_LAYERS):
            save[f"{t}_L{i}"] = feats[t][i]
        save[f"{t}_final"] = final[t]
    np.savez(cache_path, **save)
    print(f"  [saved]      {cache_path.name}  fslot per-layer {feats['fslot'][0].shape}")
    return feats, final, np.asarray(y)


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
