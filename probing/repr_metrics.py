"""Label-free, probe-free representation metrics on Chronos-2 hidden states.

Per-layer (Embed + L1..L12 = 13 entries) matrix-based entropy and effective rank,
for BOTH the pretrained model and a randomly initialized Chronos-2. No probes, no
fitting, no labels.

Metrics (kept strictly separate):
  - matrix_entropy(Z):  SVD of Z (N x D, rows = samples); lambda_i = s_i^2;
        p_i = lambda_i / sum(lambda); S1 = -sum p_i log p_i.
        Guard: eigenvalues < 1e-12 * max are dropped before normalizing.
  - effective_rank(Z):  Roy & Vetterli 2007 — Shannon entropy of the NORMALIZED
        SINGULAR VALUES q_i = s_i / sum(s), then exp.

  NOTE (theorem-direction correction, flagged to the author): the requested gate was
  "EffRank(Z) <= exp(S1(Z))". With p ∝ s^2 and q ∝ s, p is strictly more concentrated
  than q, hence H(p) <= H(q) and therefore  exp(S1) <= EffRank  — the reverse
  direction. Counterexample to the literal spec: s = (2, 1) gives EffRank = 1.890,
  exp(S1) = 1.649. This module asserts the provable direction and prints per-layer
  pairs so the relationship is visible.

Matrix types:
  - prompt-level: one Z per series = its content-patch embeddings (N_patches x D) at
    one layer; per-layer mean/std across series, plus normalized S1 / log(min(N, D)).
  - dataset-level: one Z per layer = per-series mean embeddings (mean over content
    patches), one row per series (n_series x D).

Data convention (same as probing.dataset_distance): phase0_trio loaders, <= 200 series
per dataset, rng seed 0, HF_DATASETS_OFFLINE=1. One context window per series = the
LAST C=512 points (same C / patch_size=16 patching as the ID pipeline -> 32 content
patches). Content patches ONLY: the REG token and masked_future positions are excluded,
and the Embed layer is the input_patch_embedding output for the context patches (the
REG/future embeddings come from other module calls and are filtered out).

Per-series matrices are kept as lists / object arrays end-to-end — NEVER zero-padded
into a rectangular tensor (padding would corrupt the entropy spectrum).

Hidden-state caches (never recomputed when present):
  results/repr_metrics/pretrained_cache/<tag>__seed<S>__C512.npz   (authorized new pass)
  results/repr_metrics/randinit_cache/<tag>__seed<S>__C512.npz     (the ONLY other pass)

Outputs:
  results/repr_metrics/{electricity,m4,electricity_randinit}/metrics.json
    + fig_entropy_by_layer.png / fig_effrank_by_layer.png   (rendered from JSON only)
  results/repr_metrics/fig_pretrained_vs_randinit_electricity.png

Run:
  python -m probing.repr_metrics                # everything: extract (cache), metrics,
                                                # figures, validation gates
  python -m probing.repr_metrics --figures-only # re-render all figures from JSONs
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")   # same convention as dataset_distance

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.config import OUT_DIR, SEED

# ----------------------------------------------------------------------------- #
# configuration
# ----------------------------------------------------------------------------- #
CHECKPOINT = "amazon/chronos-2"
C = 512                      # context window length (same as the ID pipeline)
MAX_SERIES = 200             # same convention as dataset_distance
SIGMA_EPS = 1e-12
EIG_GUARD = 1e-12            # relative eigenvalue / singular-value floor

RM_DIR = OUT_DIR / "repr_metrics"
PRETRAINED_CACHE = RM_DIR / "pretrained_cache"
RANDINIT_CACHE = RM_DIR / "randinit_cache"

# short output names -> phase0_trio tags
DATASETS = {"electricity": "monash_electricity_hourly", "m4": "m4_hourly"}

N_LAYERS_AXIS = 13
LAYER_AXIS = ["Embed"] + [f"L{i}" for i in range(1, 13)]


# ----------------------------------------------------------------------------- #
# metrics (pure numpy; unit-tested with known answers)
# ----------------------------------------------------------------------------- #

def matrix_entropy(Z: np.ndarray, alpha: float = 1.0) -> float:
    """Matrix-based (spectral) entropy of Z (N x D, rows = samples).

    SVD -> lambda_i = s_i^2 -> p_i = lambda_i / sum(lambda) -> S1 = -sum p log p.
    Eigenvalues < EIG_GUARD * max(lambda) are dropped BEFORE normalizing.
    alpha != 1 gives the Renyi generalization (1/(1-a)) log sum p^a; the default
    alpha=1.0 (Shannon) is the only order used in this study.
    """
    Z = np.asarray(Z, dtype=np.float64)
    s = np.linalg.svd(Z, compute_uv=False)
    lam = s ** 2
    if lam.size == 0 or lam.max() <= 0.0:
        return 0.0
    lam = lam[lam >= EIG_GUARD * lam.max()]
    p = lam / lam.sum()
    if abs(alpha - 1.0) < 1e-12:
        return float(-(p * np.log(p)).sum())
    return float(np.log((p ** alpha).sum()) / (1.0 - alpha))


def effective_rank(Z: np.ndarray) -> float:
    """Roy & Vetterli (2007) effective rank: exp(Shannon entropy of s_i / sum(s)).

    Operates on the NORMALIZED SINGULAR VALUES (NOT the eigenvalues) — strictly
    separate from matrix_entropy. Same relative floor guard for numerical stability.
    """
    Z = np.asarray(Z, dtype=np.float64)
    s = np.linalg.svd(Z, compute_uv=False)
    if s.size == 0 or s.max() <= 0.0:
        return 1.0
    s = s[s >= EIG_GUARD * s.max()]
    q = s / s.sum()
    return float(np.exp(-(q * np.log(q)).sum()))


def normalized_matrix_entropy(Z: np.ndarray) -> float:
    """S1(Z) / log(min(N, D)) — in [0, 1]; NaN for a 1-row/1-col matrix (log 1 = 0)."""
    Z = np.asarray(Z)
    denom = math.log(min(Z.shape[0], Z.shape[1]))
    if denom <= 0.0:
        return float("nan")
    return matrix_entropy(Z) / denom


# ----------------------------------------------------------------------------- #
# data loading (reuses the phase0_trio loaders; ids come from the same HF cache)
# ----------------------------------------------------------------------------- #

def _load_series_with_ids(tag: str) -> tuple[list[np.ndarray], list[str]]:
    from probing import config
    config.set_dataset_set("phase0_trio")
    from probing.id_data import load_seen_series, _HF_REPO, _active_specs
    from datasets import load_dataset

    series = load_seen_series(tag)
    ds = load_dataset(_HF_REPO, _active_specs()[tag]["hf_config"], split="train")
    ids = [str(r) for r in ds["id"]]
    assert len(ids) == len(series), f"id/series length mismatch for {tag}"
    return series, ids


def _sample_windows(tag: str, seed: int) -> tuple[np.ndarray, list[str], dict]:
    """<= MAX_SERIES series (fixed rng), one window per series = LAST C points.

    Drops series with a non-finite or near-constant window (guarded, logged).
    Returns (windows (n, C) float32, kept_ids, provenance dict).
    """
    series, ids = _load_series_with_ids(tag)
    rng = np.random.default_rng(seed)
    n_total = len(series)
    take = min(MAX_SERIES, n_total)
    sel = np.sort(rng.choice(n_total, size=take, replace=False))

    wins, kept_ids, dropped = [], [], []
    for i in sel:
        x = np.asarray(series[i], dtype=np.float64)
        if x.size < C:
            dropped.append((ids[i], "too_short"))
            continue
        w = x[-C:]
        if not np.all(np.isfinite(w)) or w.std() < SIGMA_EPS:
            dropped.append((ids[i], "nonfinite_or_constant"))
            continue
        wins.append(w.astype(np.float32))
        kept_ids.append(ids[i])

    prov = {
        "tag": tag, "seed": seed, "C": C, "window_rule": "last_C_points",
        "n_series_total": n_total, "n_sampled": int(take),
        "n_kept": len(wins), "n_dropped": len(dropped),
        "dropped": dropped[:20],
    }
    return np.stack(wins), kept_ids, prov


# ----------------------------------------------------------------------------- #
# per-patch extraction: Embed + L1..L12, content patches only
# ----------------------------------------------------------------------------- #

def _build_randinit_model():
    """Same architecture config as CHECKPOINT, RANDOM weights (torch.manual_seed(0)).

    Loads ONLY config.json via Chronos2Model.config_class.from_pretrained — no weight
    checkpoint is ever loaded (that is the state_dict source printed by the gate).
    """
    import torch
    from chronos import Chronos2Pipeline
    from chronos.chronos2.model import Chronos2Model

    cfg = Chronos2Model.config_class.from_pretrained(CHECKPOINT)
    torch.manual_seed(0)
    model = Chronos2Model(cfg)          # random init via HF post_init; CPU
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return Chronos2Pipeline(model), (
        f"randomly initialized: Chronos2Model(config) under torch.manual_seed(0); "
        f"config-only from_pretrained('{CHECKPOINT}'); NO weight checkpoint loaded"
    )


def _get_model(kind: str):
    """kind in {'pretrained','randinit'} -> (pipeline, state_dict_source string)."""
    if kind == "pretrained":
        from probing.extraction import get_pipeline   # existing loader, unchanged
        pipe, _ = get_pipeline()
        return pipe, f"checkpoint: from_pretrained('{CHECKPOINT}') via probing.extraction.get_pipeline"
    if kind == "randinit":
        return _build_randinit_model()
    raise ValueError(kind)


def _cache_path(kind: str, tag: str, seed: int) -> Path:
    d = PRETRAINED_CACHE if kind == "pretrained" else RANDINIT_CACHE
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{tag}__seed{seed}__C{C}.npz"


def _obj_array(mats: list[np.ndarray]) -> np.ndarray:
    """Object array of per-series matrices (NEVER a padded rectangular tensor)."""
    arr = np.empty(len(mats), dtype=object)
    arr[:] = mats
    return arr


def extract_perpatch_states(kind: str, tag: str, seed: int, batch_size: int = 64):
    """Per-series content-patch states for 13 layers (Embed + L1..L12), cached.

    Returns (states, ids, meta) where states = {layer_name: [ (N_patches_i, 768) ]},
    one matrix per kept series, patch axis = content patches only (REG and
    masked_future excluded).
    """
    cp = _cache_path(kind, tag, seed)
    if cp.exists():
        d = np.load(cp, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
        states = {ln: list(d[f"hs_{ln}"]) for ln in LAYER_AXIS}
        print(f"  [cache HIT]  {cp.relative_to(OUT_DIR)}  "
              f"(n_series={meta['n_kept']}, source: {meta['state_dict_source']})")
        return states, list(d["series_ids"]), meta

    import torch

    windows, ids, prov = _sample_windows(tag, seed)
    n, _ = windows.shape
    pipe, sd_source = _get_model(kind)
    patch_size = pipe.model.chronos_config.input_patch_size
    n_content = math.ceil(C / patch_size)
    print(f"  [cache MISS] {kind}/{tag} seed={seed}: n_series={n}, C={C}, "
          f"patch_size={patch_size} -> {n_content} content patches; extracting on "
          f"{next(pipe.model.parameters()).device}")
    print(f"     checkpoint/source: {sd_source}")

    per_layer: dict[str, list[np.ndarray]] = {ln: [] for ln in LAYER_AXIS}

    for b0 in range(0, n, batch_size):
        batch = windows[b0:b0 + batch_size]
        inputs = [torch.from_numpy(np.ascontiguousarray(batch[j])) for j in range(len(batch))]

        embed_hits: list[torch.Tensor] = []
        block_hits: dict[int, list[torch.Tensor]] = {i: [] for i in range(12)}

        def embed_hook(_m, _inp, out):
            # input_patch_embedding fires twice per forward: context patches
            # (b, n_content, 768) and the masked_future patch (b, 1, 768).
            # Keep ONLY the context call -> the "Embed" layer, content patches only.
            if out.shape[1] == n_content:
                embed_hits.append(out.detach().to("cpu"))

        def make_block_hook(i):
            def hook(_m, _inp, out):
                hs = out.hidden_states if hasattr(out, "hidden_states") and out.hidden_states is not None else out[0]
                block_hits[i].append(hs.detach().to("cpu"))
            return hook

        handles = [pipe.model.input_patch_embedding.register_forward_hook(embed_hook)]
        handles += [blk.register_forward_hook(make_block_hook(i))
                    for i, blk in enumerate(pipe.model.encoder.block)]
        try:
            with torch.no_grad():
                _ = pipe.embed(inputs, batch_size=len(inputs))
        finally:
            for h in handles:
                h.remove()

        emb = torch.cat(embed_hits, dim=0)                    # (b, n_content, 768)
        assert emb.shape[0] == len(batch) and emb.shape[1] == n_content
        for j in range(len(batch)):
            per_layer["Embed"].append(emb[j].numpy().astype(np.float32))
        for i in range(12):
            full = torch.cat(block_hits[i], dim=0)            # (b, P, 768)
            assert full.shape[0] == len(batch) and full.shape[1] == n_content + 2
            content = full[:, :n_content, :]                  # REG + masked_future excluded
            for j in range(len(batch)):
                per_layer[f"L{i + 1}"].append(content[j].numpy().astype(np.float32))

    meta = dict(prov)
    meta.update({"kind": kind, "checkpoint": CHECKPOINT if kind == "pretrained" else None,
                 "state_dict_source": sd_source, "patch_size": int(patch_size),
                 "n_content_patches": int(n_content), "D": 768,
                 "layer_axis": LAYER_AXIS})
    save = {f"hs_{ln}": _obj_array(per_layer[ln]) for ln in LAYER_AXIS}
    save["series_ids"] = np.array(ids)
    save["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(cp, **save)
    print(f"  [saved]      {cp.relative_to(OUT_DIR)}  ({cp.stat().st_size/1e6:.0f} MB)")
    return per_layer, ids, meta


# ----------------------------------------------------------------------------- #
# metric computation per run
# ----------------------------------------------------------------------------- #

def compute_metrics(states: dict[str, list[np.ndarray]], ids: list[str], meta: dict) -> dict:
    out_layers = []
    for ln in LAYER_AXIS:
        mats = states[ln]
        s1 = np.array([matrix_entropy(Z) for Z in mats])
        s1n = np.array([normalized_matrix_entropy(Z) for Z in mats])
        er = np.array([effective_rank(Z) for Z in mats])
        # dataset-level: rows = per-series mean over content patches
        Zd = np.stack([Z.mean(axis=0) for Z in mats])         # (n_series, 768)
        out_layers.append({
            "layer": ln,
            "prompt_entropy_mean": float(s1.mean()), "prompt_entropy_std": float(s1.std()),
            "prompt_entropy_norm_mean": float(s1n.mean()), "prompt_entropy_norm_std": float(s1n.std()),
            "prompt_effrank_mean": float(er.mean()), "prompt_effrank_std": float(er.std()),
            "dataset_entropy": matrix_entropy(Zd),
            "dataset_entropy_norm": normalized_matrix_entropy(Zd),
            "dataset_effrank": effective_rank(Zd),
            # theorem pairs (provable direction: exp(S1) <= EffRank; see module docstring)
            "dataset_exp_S1": float(np.exp(matrix_entropy(Zd))),
            "prompt_exp_S1_mean": float(np.exp(s1).mean()),
        })
    return {
        "provenance": meta,
        "n_series": len(ids), "series_ids": ids,
        "n_patches": [int(states["Embed"][k].shape[0]) for k in range(len(ids))],
        "D": 768, "layer_axis": LAYER_AXIS,
        "per_layer": out_layers,
    }


def theorem_gate(states: dict[str, list[np.ndarray]], label: str) -> list[tuple[str, float, float]]:
    """Assert exp(S1(Z)) <= EffRank(Z) (+eps) for every layer and every matrix.

    (Direction corrected from the spec — see module docstring. Counterexample to the
    literal 'EffRank <= exp(S1)': s=(2,1) -> EffRank 1.890 > exp(S1) 1.649.)
    Returns dataset-level (layer, EffRank, exp(S1)) pairs for printing.
    """
    pairs = []
    for ln in LAYER_AXIS:
        mats = states[ln]
        for Z in mats:                                   # every per-series matrix
            er, e1 = effective_rank(Z), math.exp(matrix_entropy(Z))
            assert e1 <= er * (1 + 1e-9) + 1e-9, f"{label}/{ln}: exp(S1)={e1} > EffRank={er}"
        Zd = np.stack([Z.mean(axis=0) for Z in mats])
        er_d, e1_d = effective_rank(Zd), math.exp(matrix_entropy(Zd))
        assert e1_d <= er_d * (1 + 1e-9) + 1e-9, f"{label}/{ln} dataset-level"
        pairs.append((ln, er_d, e1_d))
    return pairs


# ----------------------------------------------------------------------------- #
# figures (read ONLY metrics.json)
# ----------------------------------------------------------------------------- #

def _loadj(p: Path) -> dict:
    return json.loads(p.read_text())


def plot_dataset_figs(run_dir: Path) -> None:
    m = _loadj(run_dir / "metrics.json")
    xs = np.arange(len(m["layer_axis"]))
    pl = m["per_layer"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    mu = np.array([r["prompt_entropy_norm_mean"] for r in pl])
    sd = np.array([r["prompt_entropy_norm_std"] for r in pl])
    ax.fill_between(xs, mu - sd, mu + sd, alpha=0.2, color="C0", lw=0)
    ax.plot(xs, mu, "o-", color="C0", label="prompt entropy (normalized, mean±std)")
    ax.plot(xs, [r["dataset_entropy_norm"] for r in pl], "s--", color="C3",
            label="dataset entropy (normalized)")
    ax.set_xticks(xs); ax.set_xticklabels(m["layer_axis"], rotation=45)
    ax.set_xlabel("layer"); ax.set_ylabel("normalized matrix entropy")
    ax.set_title(f"{run_dir.name}: matrix-based entropy by layer")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run_dir / "fig_entropy_by_layer.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    mu = np.array([r["prompt_effrank_mean"] for r in pl])
    sd = np.array([r["prompt_effrank_std"] for r in pl])
    ax.fill_between(xs, mu - sd, mu + sd, alpha=0.2, color="C0", lw=0)
    ax.plot(xs, mu, "o-", color="C0", label="prompt-level effective rank (mean±std)")
    ax.plot(xs, [r["dataset_effrank"] for r in pl], "s--", color="C3",
            label="dataset-level effective rank")
    ax.set_xticks(xs); ax.set_xticklabels(m["layer_axis"], rotation=45)
    ax.set_xlabel("layer"); ax.set_ylabel("effective rank")
    ax.set_title(f"{run_dir.name}: effective rank by layer")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(run_dir / "fig_effrank_by_layer.png", dpi=140)
    plt.close(fig)
    print(f"  [saved] {run_dir.name}/fig_entropy_by_layer.png + fig_effrank_by_layer.png")


def plot_comparison() -> None:
    a = _loadj(RM_DIR / "electricity" / "metrics.json")
    b = _loadj(RM_DIR / "electricity_randinit" / "metrics.json")
    xs = np.arange(len(a["layer_axis"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for m, lab, c in ((a, "pretrained", "C0"), (b, "randinit", "C1")):
        pl = m["per_layer"]
        mu = np.array([r["prompt_entropy_norm_mean"] for r in pl])
        sd = np.array([r["prompt_entropy_norm_std"] for r in pl])
        ax1.fill_between(xs, mu - sd, mu + sd, alpha=0.18, color=c, lw=0)
        ax1.plot(xs, mu, "o-", color=c, label=lab)
        mu = np.array([r["prompt_effrank_mean"] for r in pl])
        sd = np.array([r["prompt_effrank_std"] for r in pl])
        ax2.fill_between(xs, mu - sd, mu + sd, alpha=0.18, color=c, lw=0)
        ax2.plot(xs, mu, "o-", color=c, label=lab)
    for ax, t, yl in ((ax1, "normalized prompt entropy", "S1 / log(min(N,D))"),
                      (ax2, "prompt-level effective rank", "effective rank")):
        ax.set_xticks(xs); ax.set_xticklabels(a["layer_axis"], rotation=45)
        ax.set_xlabel("layer"); ax.set_ylabel(yl); ax.set_title(t)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("electricity: pretrained vs randomly-initialized Chronos-2", y=1.02)
    fig.tight_layout()
    out = RM_DIR / "fig_pretrained_vs_randinit_electricity.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.relative_to(OUT_DIR)}")


# ----------------------------------------------------------------------------- #
# runner + validation gates
# ----------------------------------------------------------------------------- #

def run_one(kind: str, short: str, tag: str, seed: int, out_name: str):
    print(f"\n=== {out_name} ({kind}, tag={tag}, seed={seed}) ===")
    states, ids, meta = extract_perpatch_states(kind, tag, seed)
    metrics = compute_metrics(states, ids, meta)
    run_dir = RM_DIR / out_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"  [saved] {out_name}/metrics.json  (n_series={metrics['n_series']}, "
          f"N_patches={metrics['n_patches'][0]}, D={metrics['D']})")
    return states, metrics


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)

    RM_DIR.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        for name in ("electricity", "m4", "electricity_randinit"):
            plot_dataset_figs(RM_DIR / name)
        plot_comparison()
        return

    # ---- runs ----
    st_e, _ = run_one("pretrained", "electricity", DATASETS["electricity"], SEED, "electricity")
    st_m, _ = run_one("pretrained", "m4", DATASETS["m4"], SEED, "m4")
    st_r, _ = run_one("randinit", "electricity", DATASETS["electricity"], SEED, "electricity_randinit")
    st_e1, _ = run_one("pretrained", "electricity", DATASETS["electricity"], 1, "electricity_seed1")

    # ---- figures (from JSON only) ----
    print()
    for name in ("electricity", "m4", "electricity_randinit"):
        plot_dataset_figs(RM_DIR / name)
    plot_comparison()

    # ---- gate: theorem pairs, every layer, both models (+ m4) ----
    print("\n=== GATE: exp(S1) <= EffRank per layer (direction corrected from spec; "
          "counterexample to literal spec: s=(2,1) -> EffRank 1.890 > exp(S1) 1.649) ===")
    for label, st in (("electricity/pretrained", st_e), ("m4/pretrained", st_m),
                      ("electricity/randinit", st_r)):
        pairs = theorem_gate(st, label)
        print(f"  [{label}] dataset-level (layer: EffRank >= exp(S1)):")
        for ln, er, e1 in pairs:
            print(f"     {ln:>6}: EffRank={er:8.3f}  exp(S1)={e1:8.3f}  ok={e1 <= er*(1+1e-9)+1e-9}")
        print(f"  [{label}] all per-series matrices asserted (13 layers x n_series) ✓")

    # ---- gate: seed=1 rerun Spearman (pretrained electricity) ----
    from scipy.stats import spearmanr
    er0 = np.array([np.mean([effective_rank(Z) for Z in st_e[ln]]) for ln in LAYER_AXIS])
    er1 = np.array([np.mean([effective_rank(Z) for Z in st_e1[ln]]) for ln in LAYER_AXIS])
    er0_d = np.array([effective_rank(np.stack([Z.mean(0) for Z in st_e[ln]])) for ln in LAYER_AXIS])
    er1_d = np.array([effective_rank(np.stack([Z.mean(0) for Z in st_e1[ln]])) for ln in LAYER_AXIS])
    rho_p = spearmanr(er0, er1).statistic
    rho_d = spearmanr(er0_d, er1_d).statistic
    print(f"\n=== GATE: seed0 vs seed1 per-layer effective-rank Spearman (electricity/pretrained) ===")
    print(f"  prompt-level mean effrank:  rho = {rho_p:.4f}  (> 0.95: {rho_p > 0.95})")
    print(f"  dataset-level effrank:      rho = {rho_d:.4f}  (> 0.95: {rho_d > 0.95})")
    assert rho_p > 0.95, "seed-stability gate failed (prompt-level)"

    # ---- gate: randinit differs from pretrained; state_dict sources ----
    common = min(len(st_e["L6"]), len(st_r["L6"]))
    diffs = [float(np.abs(st_e["L6"][k] - st_r["L6"][k]).mean()) for k in range(common)]
    mad = float(np.mean(diffs))
    print(f"\n=== GATE: randinit sanity ===")
    print(f"  mean |pretrained - randinit| at L6 over {common} series: {mad:.4f}  (> 0: {mad > 0})")
    assert mad > 0
    for name in ("electricity", "electricity_randinit"):
        src = _loadj(RM_DIR / name / "metrics.json")["provenance"]["state_dict_source"]
        print(f"  state_dict source [{name}]: {src}")


if __name__ == "__main__":
    main()
