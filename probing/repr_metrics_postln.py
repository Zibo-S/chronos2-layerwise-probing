"""L12_postln follow-up: metrics for the encoder output AFTER the final layer norm.

Decides whether the observed L12 rank collapse (electricity dataset effrank 86->23,
m4 66->16) is representational or a PRE-final-norm scale artifact. The existing 13-entry
axis (Embed + L1..L12, block outputs pre final norm) is unchanged; this adds ONE more
representation, ``L12_postln`` = ``encoder.final_layer_norm`` output (the tensor the
decoder/head actually consumes; dropout after it is identity in eval), content patches
only, pretrained electricity + m4 only.

Additive by construction (probing/repr_metrics.py is NOT modified):
  - windows/series are re-derived with the same seed-0 sampler and ASSERTED to match the
    ids stored in the existing pretrained cache; everything except the new tensor is
    reused from that cache.
  - the new states are cached to  pretrained_cache/<tag>__seed0__C512__postln.npz
    (a sibling file — existing caches are not touched or invalidated).
  - metrics.json gains a clearly-labeled 14th per_layer entry + "L12_postln" on
    layer_axis (idempotent: re-running replaces the entry, never duplicates it).
  - figures are re-rendered FROM THE JSON ONLY, with L12_postln as a distinct open
    square; the randinit curve keeps its 13 points (no postln pass for randinit).

Run:  python -m probing.repr_metrics_postln
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.config import OUT_DIR
from probing.repr_metrics import (
    C,
    DATASETS,
    PRETRAINED_CACHE,
    RM_DIR,
    SEED,
    _cache_path,
    _obj_array,
    _sample_windows,
    effective_rank,
    matrix_entropy,
    normalized_matrix_entropy,
)

POSTLN = "L12_postln"


# --------------------------------------------------------------------------- #
# extraction: ONE new forward computation, hooked on encoder.final_layer_norm
# --------------------------------------------------------------------------- #

def extract_postln(tag: str, seed: int = SEED, batch_size: int = 64):
    """Content-patch states of the post-final-layer-norm encoder output (pretrained).

    Returns (states list[(N_patches, 768)], ids, hook_info). Cached as a sibling of the
    existing pretrained cache; the existing npz is read (ids + L12) but never rewritten.
    """
    base_cp = _cache_path("pretrained", tag, seed)
    assert base_cp.exists(), f"missing base cache {base_cp} — run probing.repr_metrics first"
    base = np.load(base_cp, allow_pickle=True)
    base_ids = list(base["series_ids"])

    cp = base_cp.with_name(base_cp.stem + "__postln.npz")
    if cp.exists():
        d = np.load(cp, allow_pickle=True)
        info = json.loads(str(d["meta"]))
        print(f"  [cache HIT]  {cp.relative_to(OUT_DIR)}  (hook: {info['hook_module']}, "
              f"shape per batch: {info['hook_shape']})")
        return list(d["hs"]), list(d["series_ids"]), info

    import torch
    from probing.extraction import get_pipeline

    windows, ids, prov = _sample_windows(tag, seed)
    assert ids == base_ids, f"window sampler no longer matches cached ids for {tag}"
    n = len(ids)

    pipe, _ = get_pipeline()
    patch_size = pipe.model.chronos_config.input_patch_size
    n_content = math.ceil(C / patch_size)

    # the hooked module, by qualified name (printed as a validation gate)
    hook_module_name = next(name for name, m in pipe.model.named_modules()
                            if m is pipe.model.encoder.final_layer_norm)
    hook_shape_seen = {}

    mats: list[np.ndarray] = []
    for b0 in range(0, n, batch_size):
        batch = windows[b0:b0 + batch_size]
        inputs = [torch.from_numpy(np.ascontiguousarray(batch[j])) for j in range(len(batch))]
        hits: list[torch.Tensor] = []

        def hook(_m, _inp, out):
            hits.append(out.detach().to("cpu"))

        h = pipe.model.encoder.final_layer_norm.register_forward_hook(hook)
        try:
            with torch.no_grad():
                _ = pipe.embed(inputs, batch_size=len(inputs))
        finally:
            h.remove()

        full = torch.cat(hits, dim=0)                       # (b, P, 768) post-LN
        assert full.shape[0] == len(batch) and full.shape[1] == n_content + 2
        hook_shape_seen = tuple(full.shape)
        content = full[:, :n_content, :]                    # content patches only
        for j in range(len(batch)):
            mats.append(content[j].numpy().astype(np.float32))

    info = dict(prov)
    info.update({"layer": POSTLN, "hook_module": hook_module_name,
                 "hook_shape": str(hook_shape_seen), "n_content_patches": int(n_content)})
    np.savez_compressed(cp, hs=_obj_array(mats), series_ids=np.array(ids),
                        meta=np.array(json.dumps(info)))
    print(f"  [saved]      {cp.relative_to(OUT_DIR)}  ({cp.stat().st_size/1e6:.0f} MB)")
    print(f"  [hook] module = pipe.model.{hook_module_name}  "
          f"({type(pipe.model.encoder.final_layer_norm).__name__}); "
          f"output shape per final batch = {hook_shape_seen}")
    return mats, ids, info


# --------------------------------------------------------------------------- #
# metrics.json append (idempotent) + numeric table
# --------------------------------------------------------------------------- #

def append_postln_metrics(short: str, mats: list[np.ndarray], info: dict) -> dict:
    p = RM_DIR / short / "metrics.json"
    m = json.loads(p.read_text())
    s1 = np.array([matrix_entropy(Z) for Z in mats])
    s1n = np.array([normalized_matrix_entropy(Z) for Z in mats])
    er = np.array([effective_rank(Z) for Z in mats])
    Zd = np.stack([Z.mean(axis=0) for Z in mats])
    entry = {
        "layer": POSTLN,
        "prompt_entropy_mean": float(s1.mean()), "prompt_entropy_std": float(s1.std()),
        "prompt_entropy_norm_mean": float(s1n.mean()), "prompt_entropy_norm_std": float(s1n.std()),
        "prompt_effrank_mean": float(er.mean()), "prompt_effrank_std": float(er.std()),
        "dataset_entropy": matrix_entropy(Zd),
        "dataset_entropy_norm": normalized_matrix_entropy(Zd),
        "dataset_effrank": effective_rank(Zd),
        "dataset_exp_S1": float(np.exp(matrix_entropy(Zd))),
        "prompt_exp_S1_mean": float(np.exp(s1).mean()),
        "note": "post final-layer-norm encoder output (the tensor the head consumes); "
                "L12 above is the same block output PRE final norm",
        "hook_module": info["hook_module"],
    }
    if POSTLN in m["layer_axis"]:                       # idempotent re-run: replace
        m["per_layer"][m["layer_axis"].index(POSTLN)] = entry
    else:
        m["layer_axis"].append(POSTLN)
        m["per_layer"].append(entry)
    p.write_text(json.dumps(m, indent=1))
    print(f"  [saved] {short}/metrics.json  (+{POSTLN} as point {len(m['layer_axis'])})")
    return m


def write_table(short: str) -> None:
    m = json.loads((RM_DIR / short / "metrics.json").read_text())
    lines = [f"{'layer':>12} | {'prompt_entropy_norm (mean±std)':>30} | {'dataset_effrank':>15}",
             "-" * 65]
    for r in m["per_layer"]:
        lines.append(f"{r['layer']:>12} | {r['prompt_entropy_norm_mean']:14.4f} ± "
                     f"{r['prompt_entropy_norm_std']:6.4f}      | {r['dataset_effrank']:15.3f}")
    txt = "\n".join(lines)
    out = RM_DIR / short / "table_by_layer.txt"
    out.write_text(txt + "\n")
    print(f"\n===== {short}: per-layer table (saved to {out.relative_to(OUT_DIR)}) =====")
    print(txt)


# --------------------------------------------------------------------------- #
# figures (JSON-only; L12_postln = distinct open square)
# --------------------------------------------------------------------------- #

def _series(m: dict, key: str) -> np.ndarray:
    return np.array([r[key] for r in m["per_layer"]])


def _plot_curve_with_postln(ax, xs, mu, sd, axis, color, label, ls="-"):
    """Solid curve over the 13 base layers; L12_postln (if present) as an open square."""
    has_pln = axis and axis[-1] == POSTLN
    k = len(axis) - 1 if has_pln else len(axis)
    if sd is not None:
        ax.fill_between(xs[:k], (mu - sd)[:k], (mu + sd)[:k], alpha=0.18, color=color, lw=0)
    ax.plot(xs[:k], mu[:k], "o", ls=ls, color=color, label=label)
    if has_pln:
        if sd is not None:
            ax.errorbar([xs[k]], [mu[k]], yerr=[sd[k]], fmt="none", ecolor=color, alpha=0.5)
        ax.plot([xs[k]], [mu[k]], marker="s", mfc="none", mec=color, ms=9, ls="none",
                label=f"{label} ({POSTLN})" if label else None)


def plot_dataset_figs(short: str) -> None:
    m = json.loads((RM_DIR / short / "metrics.json").read_text())
    axis = m["layer_axis"]
    xs = np.arange(len(axis))

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    _plot_curve_with_postln(ax, xs, _series(m, "prompt_entropy_norm_mean"),
                            _series(m, "prompt_entropy_norm_std"), axis, "C0",
                            "prompt entropy (normalized, mean±std)")
    _plot_curve_with_postln(ax, xs, _series(m, "dataset_entropy_norm"), None, axis, "C3",
                            "dataset entropy (normalized)", ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(axis, rotation=45)
    ax.set_xlabel("layer"); ax.set_ylabel("normalized matrix entropy")
    ax.set_title(f"{short}: matrix-based entropy by layer (postln = open square)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RM_DIR / short / "fig_entropy_by_layer.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    _plot_curve_with_postln(ax, xs, _series(m, "prompt_effrank_mean"),
                            _series(m, "prompt_effrank_std"), axis, "C0",
                            "prompt-level effective rank (mean±std)")
    _plot_curve_with_postln(ax, xs, _series(m, "dataset_effrank"), None, axis, "C3",
                            "dataset-level effective rank", ls="--")
    ax.set_xticks(xs); ax.set_xticklabels(axis, rotation=45)
    ax.set_xlabel("layer"); ax.set_ylabel("effective rank")
    ax.set_title(f"{short}: effective rank by layer (postln = open square)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(RM_DIR / short / "fig_effrank_by_layer.png", dpi=140)
    plt.close(fig)
    print(f"  [saved] {short}/fig_entropy_by_layer.png + fig_effrank_by_layer.png (14-pt axis)")


def plot_comparison() -> None:
    a = json.loads((RM_DIR / "electricity" / "metrics.json").read_text())
    b = json.loads((RM_DIR / "electricity_randinit" / "metrics.json").read_text())
    axis = a["layer_axis"]                              # 14 entries (randinit keeps 13)
    xs = np.arange(len(axis))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.6))
    for ax, key_mu, key_sd, title, yl in (
        (ax1, "prompt_entropy_norm_mean", "prompt_entropy_norm_std",
         "normalized prompt entropy", "S1 / log(min(N,D))"),
        (ax2, "prompt_effrank_mean", "prompt_effrank_std",
         "prompt-level effective rank", "effective rank"),
    ):
        _plot_curve_with_postln(ax, xs, _series(a, key_mu), _series(a, key_sd),
                                axis, "C0", "pretrained")
        mu_b, sd_b = _series(b, key_mu), _series(b, key_sd)
        ax.fill_between(xs[:len(mu_b)], mu_b - sd_b, mu_b + sd_b, alpha=0.18, color="C1", lw=0)
        ax.plot(xs[:len(mu_b)], mu_b, "o-", color="C1", label="randinit")
        ax.set_xticks(xs); ax.set_xticklabels(axis, rotation=45)
        ax.set_xlabel("layer"); ax.set_ylabel(yl); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("electricity: pretrained vs randomly-initialized Chronos-2 "
                 "(L12_postln = open square; pretrained only)", y=1.02)
    fig.tight_layout()
    out = RM_DIR / "fig_pretrained_vs_randinit_electricity.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.relative_to(OUT_DIR)}")


# --------------------------------------------------------------------------- #
# main + gates
# --------------------------------------------------------------------------- #

def main() -> None:
    for short, tag in DATASETS.items():                 # electricity, m4 (pretrained only)
        print(f"\n=== {short} ({tag}): {POSTLN} ===")
        mats, ids, info = extract_postln(tag)
        print(f"  [gate] hooked module: pipe.model.{info['hook_module']}  "
              f"| captured shape (last batch): {info['hook_shape']}")

        # gate: postln must differ from the cached PRE-norm L12 states
        base = np.load(_cache_path("pretrained", tag, SEED), allow_pickle=True)
        l12 = list(base["hs_L12"])
        diffs = [float(np.abs(mats[k] - l12[k]).mean()) for k in range(len(mats))]
        mad = float(np.mean(diffs))
        print(f"  [gate] mean |{POSTLN} - L12_prenorm| over {len(mats)} series: {mad:.4f}  "
              f"(> 0: {mad > 0})")
        assert mad > 0.0

        append_postln_metrics(short, mats, info)
        write_table(short)

    print()
    for short in DATASETS:
        plot_dataset_figs(short)
    plot_comparison()


if __name__ == "__main__":
    main()
