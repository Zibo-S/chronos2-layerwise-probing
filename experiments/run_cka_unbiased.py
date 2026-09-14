"""Biased vs UNBIASED (debiased-HSIC) linear CKA on the ext_v4 forecast-slot representation.

Robustness check + export for downstream use. The committed `--extv4-fslot` matrices (the ones
behind results/cka/ext_v4_future_tokens_fslot/figures/main_id_cka.png) use the BIASED estimator,
whose O(1/n) upward bias is only material when n is small relative to the representations'
effective rank. This driver recomputes the same 14x14 layer x layer matrices with BOTH estimators
on the SAME cached rows and exports the pair.

Reuses `run_cka_analysis.read_extv4_fslot_reps`, so the cache, the 14 keys, the (n,K,768)->(n*K,768)
stacking, the split and the subsample are identical to the committed run by construction. That
gives a free correctness gate, enforced below: the recomputed BIASED matrix must reproduce the
committed .npy, otherwise provenance has drifted and nothing is written.

Cache-only and CPU-only: no model load, no GPU, no probe (CKA is probe-independent — quantile set
and weight decay never enter). Still a few hundred 768x1048 matmuls per dataset, so run it under
salloc with OMP_NUM_THREADS set, NOT on the login node.

    python -m experiments.run_cka_unbiased
    python -m experiments.run_cka_unbiased --tags monash_electricity_hourly m4_hourly
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from probing import cka
from experiments.run_cka_analysis import (          # importing pins DATASET_SET = extended_v3_rolling
    EXTV4_TAGS, LABELS_14, OUT, PT_ID_TAGS, SHORT, _fslot_split, read_extv4_fslot_reps)

# The attached-figure label style: spelled-out endpoints rather than the compact LABELS_14.
LEGACY_LABELS_14 = ["Embed"] + [f"L{i}" for i in range(1, len(LABELS_14) - 1)] + ["L12 (post LN)"]

DEFAULT_TAGS = ["monash_electricity_hourly", "m4_hourly"]
GATE_TOL = 1e-10


def plain_heatmap(M, labels, path, *, vmin=0.0, vmax=1.0, dpi=300):
    """Bare square CKA heatmap in the original ad-hoc style: no title, no axis labels, no colorbar
    label, fixed [vmin, vmax] viridis, 45-degree x ticks. Writes `path` (.png) + its .pdf sibling."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    M = np.asarray(M, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal", origin="upper")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(length=2, pad=1.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=6, length=2)
    cb.outline.set_linewidth(0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)
    return path


def export(tag, split, max_rows, seed, root, gate=True):
    """Both estimators for one dataset: gate, matrices, tables, figures. Returns a summary dict."""
    reps = read_extv4_fslot_reps(tag, split)
    n_avail = reps[0].shape[0]
    idx = cka.subsample_indices(n_avail, max_rows, seed)
    reps = [r[idx] for r in reps]
    n = len(idx)

    B = cka.cka_matrix(reps, estimator="biased")
    U = cka.cka_matrix(reps, estimator="unbiased")

    # GATE: the biased half must reproduce the committed matrix behind main_id_cka.png.
    ref_path = OUT / "ext_v4_future_tokens_fslot" / "matrices" / f"{tag}__fslot__layerxlayer.npy"
    dev = float("nan")
    if gate:
        if not ref_path.exists():
            raise FileNotFoundError(
                f"missing committed reference {ref_path} — run `python -m experiments."
                f"run_cka_analysis --extv4-fslot` first, or pass --no-gate to skip the check")
        dev = float(np.nanmax(np.abs(B - np.load(ref_path))))
        if not dev < GATE_TOL:
            raise RuntimeError(
                f"{tag}: recomputed BIASED CKA deviates from {ref_path.name} by {dev:.3e} "
                f"(tol {GATE_TOL:g}) — provenance drift; refusing to write unbiased numbers")

    short = SHORT.get(tag, tag)
    np.save(root / "matrices" / f"{tag}__fslot__layerxlayer__biased.npy", B)
    np.save(root / "matrices" / f"{tag}__fslot__layerxlayer__unbiased.npy", U)
    for name, M in (("biased", B), ("unbiased", U), ("delta_biased_minus_unbiased", B - U)):
        cka.save_matrix_csv(M, LABELS_14, LABELS_14,
                            root / "tables" / f"{tag}__fslot__{name}.csv")
    with open(root / "tables" / f"{tag}__fslot__paired_long.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "row_layer", "col_layer", "cka_biased", "cka_unbiased", "delta"])
        for i, ri in enumerate(LEGACY_LABELS_14):
            for j, cj in enumerate(LEGACY_LABELS_14):
                w.writerow([tag, ri, cj, f"{B[i, j]:.8f}", f"{U[i, j]:.8f}", f"{B[i, j] - U[i, j]:.8f}"])

    for name, M in (("biased", B), ("unbiased", U)):
        plain_heatmap(M, LEGACY_LABELS_14, root / "figures" / f"{tag}__fslot__{name}.png")

    off = ~np.eye(len(LABELS_14), dtype=bool)
    summary = {"dataset": tag, "short": short,
               "kind": "PT-ID" if tag in PT_ID_TAGS else "PT-OOD",
               "cache_split": _fslot_split(tag, split),
               "rows_available": int(n_avail), "rows_used": int(n), "windows": int(n_avail // 4),
               "biased_gate_max_dev": dev,
               "offdiag_biased_min": float(B[off].min()), "offdiag_biased_max": float(B[off].max()),
               "offdiag_biased_mean": float(B[off].mean()),
               "offdiag_unbiased_min": float(U[off].min()), "offdiag_unbiased_max": float(U[off].max()),
               "offdiag_unbiased_mean": float(U[off].mean()),
               "max_abs_delta_offdiag": float(np.abs(B - U)[off].max())}
    print(f"[cka-unbiased] {short:<12} n={n} rows ({n // 4} windows x K=4)  gate dev={dev:.2e}")
    print(f"     off-diag biased   min/mean/max = {summary['offdiag_biased_min']:.4f} / "
          f"{summary['offdiag_biased_mean']:.4f} / {summary['offdiag_biased_max']:.4f}")
    print(f"     off-diag unbiased min/mean/max = {summary['offdiag_unbiased_min']:.4f} / "
          f"{summary['offdiag_unbiased_mean']:.4f} / {summary['offdiag_unbiased_max']:.4f}")
    print(f"     max |biased - unbiased| off-diagonal = {summary['max_abs_delta_offdiag']:.6f}")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tags", nargs="+", default=DEFAULT_TAGS,
                    help=f"datasets to export (default: {' '.join(DEFAULT_TAGS)})")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--max-rows", type=int, default=4096, help="match the committed provenance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-gate", action="store_true",
                    help="skip the biased-reproduces-committed check (only for a new split)")
    args = ap.parse_args()

    unknown = [t for t in args.tags if t not in EXTV4_TAGS]
    if unknown:
        raise SystemExit(f"unknown tag(s) {unknown}; choose from {EXTV4_TAGS}")

    root = OUT / "ext_v4_future_tokens_fslot" / "unbiased"
    for sub in ("matrices", "tables", "figures"):
        (root / sub).mkdir(parents=True, exist_ok=True)

    rows = [export(t, args.split, args.max_rows, args.seed, root, gate=not args.no_gate)
            for t in args.tags]

    with open(root / "tables" / "summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader(); w.writerows(rows)
    json.dump({"analysis": "ext_v4_future_tokens_fslot / biased vs unbiased linear CKA",
               "representation": "forecast slots (n,K,768) -> (n*K,768), 14 points Emb..L12+LN",
               "backbone": "pretrained amazon/chronos-2 (frozen)",
               "biased_estimator": "Kornblith et al. 2019 feature-space ratio (repo default)",
               "unbiased_estimator": "Song et al. 2012 unbiased HSIC_1, feature-space O(d^2) form",
               "requested_split": args.split, "max_rows": args.max_rows, "seed": args.seed,
               "gate": ("recomputed biased == committed matrices to "
                        f"{GATE_TOL:g}" if not args.no_gate else "SKIPPED (--no-gate)"),
               "per_dataset": rows},
              open(root / "provenance.json", "w"), indent=2)
    print(f"\nwrote -> {root}")


if __name__ == "__main__":
    main()
