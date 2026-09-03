"""Overnight helpers: one-pass 14-layer extractor + metrics for sensitivity runs.

Captures Embed + L1..L12 + L12_postln in a SINGLE forward pass per batch (the committed
pipeline used two passes because postln was a later addition; numerics are identical —
same hooks, same content-patch slice). Used by the overnight tasks only; writes exclusively
to NEW cache files/dirs. Existing results/ files are never touched.

Protocol identical to probing.repr_metrics: window = LAST C=512 points (for NaN-bearing
series, the last 512 points of the LONGEST CONTIGUOUS FINITE SEGMENT, segment >= 512
required — the distance-ladder rule), content patches only, float32 states, metrics in
float64.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from probing.repr_metrics import (
    C,
    effective_rank,
    matrix_entropy,
    normalized_matrix_entropy,
)

AXIS14 = ["Embed"] + [f"L{i}" for i in range(1, 13)] + ["L12_postln"]


def longest_finite_segment(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    f = np.isfinite(x)
    if not f.any():
        return x[:0]
    d = np.diff(np.concatenate(([0], f.view(np.int8), [0])))
    st, en = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    k = int(np.argmax(en - st))
    return x[st[k]:en[k]]


def prepare_windows(series: list[np.ndarray], max_series: int | None, seed: int):
    """Segment rule (>=C) -> last-C window of the segment -> optional seed-sampled cap."""
    wins, kept_idx = [], []
    for i, s in enumerate(series):
        seg = longest_finite_segment(s)
        if len(seg) < C:
            continue
        w = seg[-C:]
        if w.std() < 1e-12:
            continue
        wins.append(w.astype(np.float32))
        kept_idx.append(i)
    if max_series is not None and len(wins) > max_series:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(wins), size=max_series, replace=False))
        wins = [wins[j] for j in sel]
        kept_idx = [kept_idx[j] for j in sel]
    return np.stack(wins), kept_idx


def extract14(windows: np.ndarray, cache_file: Path, batch_size: int = 64) -> dict:
    """{layer: [ (32,768) float32 ]} for AXIS14, cached to cache_file (never recomputed)."""
    if cache_file.exists():
        d = np.load(cache_file, allow_pickle=True)
        print(f"  [cache HIT]  {cache_file}")
        return {ln: list(d[f"hs_{ln}"]) for ln in AXIS14}

    import torch
    from probing.extraction import get_pipeline

    pipe, _ = get_pipeline()
    patch = pipe.model.chronos_config.input_patch_size
    n_content = math.ceil(C / patch)
    n = len(windows)
    print(f"  [extract] n={n} windows, one pass, 14 layers, device="
          f"{next(pipe.model.parameters()).device}")

    per: dict[str, list[np.ndarray]] = {ln: [] for ln in AXIS14}
    for b0 in range(0, n, batch_size):
        batch = windows[b0:b0 + batch_size]
        inputs = [torch.from_numpy(np.ascontiguousarray(batch[j])) for j in range(len(batch))]
        emb_hits, pln_hits = [], []
        blk_hits = {i: [] for i in range(12)}

        def emb_hook(_m, _i, out):
            if out.shape[1] == n_content:          # context call only (future call is (b,1,768))
                emb_hits.append(out.detach().to("cpu"))

        def pln_hook(_m, _i, out):
            pln_hits.append(out.detach().to("cpu"))

        def mk(i):
            def h(_m, _i, out):
                hs = out.hidden_states if hasattr(out, "hidden_states") and out.hidden_states is not None else out[0]
                blk_hits[i].append(hs.detach().to("cpu"))
            return h

        hs_handles = [pipe.model.input_patch_embedding.register_forward_hook(emb_hook),
                      pipe.model.encoder.final_layer_norm.register_forward_hook(pln_hook)]
        hs_handles += [b.register_forward_hook(mk(i)) for i, b in enumerate(pipe.model.encoder.block)]
        try:
            with torch.no_grad():
                _ = pipe.embed(inputs, batch_size=len(inputs))
        finally:
            for h in hs_handles:
                h.remove()

        emb = torch.cat(emb_hits, 0)
        assert emb.shape[0] == len(batch) and emb.shape[1] == n_content
        pln = torch.cat(pln_hits, 0)
        assert pln.shape[0] == len(batch) and pln.shape[1] == n_content + 2
        for j in range(len(batch)):
            per["Embed"].append(emb[j].numpy().astype(np.float32))
            per["L12_postln"].append(pln[j, :n_content].numpy().astype(np.float32))
        for i in range(12):
            full = torch.cat(blk_hits[i], 0)
            assert full.shape[1] == n_content + 2
            for j in range(len(batch)):
                per[f"L{i + 1}"].append(full[j, :n_content].numpy().astype(np.float32))

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    save = {}
    for ln in AXIS14:
        arr = np.empty(len(per[ln]), dtype=object)
        arr[:] = per[ln]
        save[f"hs_{ln}"] = arr
    np.savez_compressed(cache_file, **save)
    print(f"  [saved] {cache_file}  ({cache_file.stat().st_size / 1e6:.0f} MB)")
    return per


def metrics14(states: dict, provenance: dict) -> dict:
    per_layer = []
    for ln in AXIS14:
        mats = [np.asarray(m, np.float64) for m in states[ln]]
        s1n = np.array([normalized_matrix_entropy(m) for m in mats])
        er = np.array([effective_rank(m) for m in mats])
        Zd = np.stack([m.mean(0) for m in mats])
        per_layer.append({
            "layer": ln,
            "prompt_entropy_norm_mean": float(s1n.mean()),
            "prompt_entropy_norm_std": float(s1n.std()),
            "prompt_effrank_mean": float(er.mean()),
            "prompt_effrank_std": float(er.std()),
            "dataset_entropy": matrix_entropy(Zd),
            "dataset_effrank": effective_rank(Zd),
        })
    return {"provenance": provenance, "layer_axis": AXIS14,
            "n_series": len(states["Embed"]), "D": 768,
            "per_layer": per_layer}
