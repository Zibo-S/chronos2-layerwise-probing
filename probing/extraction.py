"""Chronos-2 hidden-state extraction (reusable core).

Loads the frozen Chronos-2 encoder and pulls per-layer hidden states off it via forward
hooks, pooling each layer to a fixed-width feature per window and caching to disk.

Public API (imported across the codebase):
    get_pipeline()             -> load the frozen Chronos-2 pipeline (float32, CUDA/MPS/CPU, no grad)
    extract_window_features()  -> per-layer, per-pooling windowed features with on-disk caching

Chronos-2 weights are frozen throughout (eval, no_grad, requires_grad=False).
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import torch

from probing.config import NUM_LAYERS, CACHE_DIR

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


def _load_cache(path):
    d = np.load(path, allow_pickle=True)
    features = {int(k.split("_")[1]): d[k] for k in d.files if k.startswith("layer_")}
    y = d["y"]
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
        print(f"  [cache HIT]  {cache_path.name}")
        return _load_cache(cache_path)

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
