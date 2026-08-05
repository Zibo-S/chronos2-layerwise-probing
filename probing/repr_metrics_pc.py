"""PC-robustness check for the repr-metrics curves (cache-only; zero forward passes).

Verifies, on OUR pipeline, Amartya's finding on Qwen3-LangInit — that a metric valley
can be an artifact of one dominant channel/direction (his dim 35 held 96.6% of
across-window variance; removing 1-2 PCs erased the valley) — and his claim that
"Chronos-2 barely shifts".

Per layer (Embed..L12 + L12_postln where cached), for pretrained electricity + m4 and
randinit electricity (randinit has NO postln cache — it was never extracted and this
task allows no new forward passes, so its axis is 13 points; noted in the JSON):

  1) top-PC variance shares of the dataset-level matrix (per-series mean embeddings):
     PC1 and PC1+PC2 fractions of total variance (PCA = SVD of the CENTERED matrix).
  2) dataset-level effective_rank / matrix_entropy after projecting out the top-1 and
     top-2 PCs (center, remove PC directions, then the UNCHANGED metric functions).
  3) prompt-level: same ablation per-series on the (N_patches x 768) matrices,
     normalized prompt entropy averaged across series.

Protocol notes (recorded in the JSON): "full" = the committed convention (metric on the
UNCENTERED matrix — ties back to metrics.json exactly); ablated variants are computed on
the centered matrix with top-k PC directions projected out (k=0 centered baseline also
reported, so centering's own effect is separable from PC removal). All SVDs in float64.

Outputs: results/repr_metrics/pc_robustness/{run}/pc_check.json + fig_pc_robustness.png
Run:  python -m probing.repr_metrics_pc
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
    RM_DIR,
    SEED,
    _cache_path,
    effective_rank,
    matrix_entropy,
    normalized_matrix_entropy,
)

PC_DIR = RM_DIR / "pc_robustness"
POSTLN = "L12_postln"
BASE_AXIS = ["Embed"] + [f"L{i}" for i in range(1, 13)]

# run name -> (kind, tag, has_postln)
RUNS = {
    "electricity":          ("pretrained", "monash_electricity_hourly", True),
    "m4":                   ("pretrained", "m4_hourly", True),
    "electricity_randinit": ("randinit",  "monash_electricity_hourly", False),
}


# --------------------------------------------------------------------------- #
# PC helpers (float64 everywhere)
# --------------------------------------------------------------------------- #

def pc_variance_shares(Z: np.ndarray) -> np.ndarray:
    """Explained-variance ratios of the CENTERED matrix (standard PCA), float64."""
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    s = np.linalg.svd(Zc, compute_uv=False)
    lam = s ** 2
    tot = lam.sum()
    return lam / tot if tot > 0 else lam


def remove_top_pcs(Z: np.ndarray, k: int) -> np.ndarray:
    """Center Z and project out its top-k principal directions (k=0 -> centered Z)."""
    Z = np.asarray(Z, dtype=np.float64)
    Zc = Z - Z.mean(axis=0, keepdims=True)
    if k <= 0:
        return Zc
    _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
    top = Vt[:k]                                        # (k, D)
    return Zc - (Zc @ top.T) @ top


# --------------------------------------------------------------------------- #
# cache loading (reuse only; never recompute)
# --------------------------------------------------------------------------- #

def load_states(kind: str, tag: str, has_postln: bool):
    """{layer: list[(N_patches, 768) float64]} from the existing caches."""
    cp = _cache_path(kind, tag, SEED)
    assert cp.exists(), f"missing cache {cp}"
    d = np.load(cp, allow_pickle=True)
    states = {ln: [np.asarray(Z, dtype=np.float64) for Z in d[f"hs_{ln}"]] for ln in BASE_AXIS}
    files = [str(cp.relative_to(OUT_DIR))]
    if has_postln:
        pp = cp.with_name(cp.stem + "__postln.npz")
        assert pp.exists(), f"missing postln cache {pp}"
        dp = np.load(pp, allow_pickle=True)
        states[POSTLN] = [np.asarray(Z, dtype=np.float64) for Z in dp["hs"]]
        files.append(str(pp.relative_to(OUT_DIR)))
    return states, files


# --------------------------------------------------------------------------- #
# per-run computation
# --------------------------------------------------------------------------- #

def run_check(name: str, kind: str, tag: str, has_postln: bool) -> dict:
    states, cache_files = load_states(kind, tag, has_postln)
    axis = BASE_AXIS + ([POSTLN] if has_postln else [])

    rows = []
    for ln in axis:
        mats = states[ln]
        Zd = np.stack([Z.mean(axis=0) for Z in mats])   # dataset-level, (n_series, 768)

        shares = pc_variance_shares(Zd)
        pc1 = float(shares[0])
        pc12 = float(shares[:2].sum())

        er_full = effective_rank(Zd)                    # committed convention (uncentered)
        ent_full = matrix_entropy(Zd)
        er_c = effective_rank(remove_top_pcs(Zd, 0))    # centered baseline
        er_m1 = effective_rank(remove_top_pcs(Zd, 1))
        er_m2 = effective_rank(remove_top_pcs(Zd, 2))
        ent_m1 = matrix_entropy(remove_top_pcs(Zd, 1))
        ent_m2 = matrix_entropy(remove_top_pcs(Zd, 2))

        pe_full = float(np.mean([normalized_matrix_entropy(Z) for Z in mats]))
        pe_m1 = float(np.mean([normalized_matrix_entropy(remove_top_pcs(Z, 1)) for Z in mats]))
        pe_m2 = float(np.mean([normalized_matrix_entropy(remove_top_pcs(Z, 2)) for Z in mats]))

        rows.append({
            "layer": ln, "pc1_share": pc1, "pc12_share": pc12,
            "dataset_effrank_full": er_full, "dataset_effrank_centered": er_c,
            "dataset_effrank_minus1pc": er_m1, "dataset_effrank_minus2pc": er_m2,
            "dataset_entropy_full": ent_full,
            "dataset_entropy_minus1pc": ent_m1, "dataset_entropy_minus2pc": ent_m2,
            "prompt_entropy_norm_mean_full": pe_full,
            "prompt_entropy_norm_mean_minus1pc": pe_m1,
            "prompt_entropy_norm_mean_minus2pc": pe_m2,
        })

    out = {
        "provenance": {
            "run": name, "kind": kind, "tag": tag, "seed": SEED,
            "cache_files": cache_files, "float64_svd": True,
            "protocol": "full = metric on UNCENTERED matrix (committed convention); "
                        "ablated = centered matrix with top-k PCs projected out; "
                        "variance shares from PCA (centered SVD)",
            "postln_included": has_postln,
            "note": None if has_postln else
                    "randinit has no L12_postln cache (never extracted; this task "
                    "permits no new forward passes) -> 13-point axis",
        },
        "layer_axis": axis,
        "per_layer": rows,
    }
    d = PC_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "pc_check.json").write_text(json.dumps(out, indent=1))

    # ---- printed table ----
    print(f"\n===== {name} ({kind}) — PC-robustness =====")
    print(f"{'layer':>12} | {'PC1':>6} | {'PC1+2':>6} | {'er full':>8} | {'er -1PC':>8} | {'er -2PC':>8}")
    print("-" * 62)
    for r in rows:
        print(f"{r['layer']:>12} | {r['pc1_share']:6.3f} | {r['pc12_share']:6.3f} | "
              f"{r['dataset_effrank_full']:8.2f} | {r['dataset_effrank_minus1pc']:8.2f} | "
              f"{r['dataset_effrank_minus2pc']:8.2f}")
    print(f"  [saved] {(d / 'pc_check.json').relative_to(OUT_DIR)}")
    return out


# --------------------------------------------------------------------------- #
# figure (JSON-only)
# --------------------------------------------------------------------------- #

def plot_run(name: str) -> None:
    m = json.loads((PC_DIR / name / "pc_check.json").read_text())
    axis = m["layer_axis"]
    xs = np.arange(len(axis))
    has_pln = axis[-1] == POSTLN
    k = len(axis) - 1 if has_pln else len(axis)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    for key, lab, c in (("dataset_effrank_full", "full (uncentered, committed conv.)", "C0"),
                        ("dataset_effrank_minus1pc", "minus top-1 PC", "C1"),
                        ("dataset_effrank_minus2pc", "minus top-2 PCs", "C2")):
        ys = np.array([r[key] for r in m["per_layer"]])
        ax.plot(xs[:k], ys[:k], "o-", color=c, label=lab)
        if has_pln:
            ax.plot([xs[k]], [ys[k]], marker="s", mfc="none", mec=c, ms=9, ls="none")
    ax.set_xticks(xs); ax.set_xticklabels(axis, rotation=45)
    ax.set_xlabel("layer"); ax.set_ylabel("dataset-level effective rank")
    ax.set_title(f"{name}: effective rank vs top-PC ablation"
                 + (" (postln = open square)" if has_pln else " (no postln cache)"))
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = PC_DIR / name / "fig_pc_robustness.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  [saved] {out.relative_to(OUT_DIR)}")


def main() -> None:
    PC_DIR.mkdir(parents=True, exist_ok=True)
    for name, (kind, tag, has_pln) in RUNS.items():
        run_check(name, kind, tag, has_pln)
    print()
    for name in RUNS:
        plot_run(name)


if __name__ == "__main__":
    main()
