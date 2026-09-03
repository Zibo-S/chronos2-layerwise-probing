"""Inter-layer linear CKA (debiased HSIC) on cached Chronos-2 representations.

Verifies, on OUR pipeline, whether the isolated B11 (= our L12) seam in Amartya's
inter-layer CKA matrices is the documented pre-final-norm artifact: if so, L12_postln
(the post-final-layer-norm variant of the same block) should rejoin the late-layer block
that pre-norm L12 separates from.

Estimator: linear-kernel CKA with the UNBIASED (debiased) HSIC estimator of Song et al.
(2012), as used by Kornblith et al.'s debiased CKA:

    K = X X^T,  L = Y Y^T   (linear kernels; X, Y are (n, D) float64)
    K~, L~ = kernels with the diagonal zeroed
    HSIC_1(K, L) = 1/(n(n-3)) * [ tr(K~ L~)
                                  + (1'K~1)(1'L~1) / ((n-1)(n-2))
                                  - (2/(n-2)) * 1' K~ L~ 1 ]
    CKA_deb(X, Y) = HSIC_1(K, L) / sqrt( HSIC_1(K, K) * HSIC_1(L, L) )

All math float64. Representations: dataset-level per-series mean content-patch embeddings
(n<=200 series, seed-0 sample — the same matrices as dataset_effrank), 14 taps for
pretrained (Embed, L1..L12, L12_postln), 13 taps for randinit (no postln cache exists;
recorded in provenance).

Axis-comparison note (provenance): Amartya's panels are electricity / m4_daily / solar_1h;
ours are electricity / m4_HOURLY — electricity is the only directly comparable panel, and
his axis has no postln tap, hence the 13-tap variant.

Outputs: results/repr_metrics/cka/{run}/cka.json + fig_cka_matrix.png
Run:  python -m probing.repr_metrics_cka [--figures-only]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.config import OUT_DIR
from probing.repr_metrics import RM_DIR, SEED, _cache_path

CKA_DIR = RM_DIR / "cka"
POSTLN = "L12_postln"
AXIS13 = ["Embed"] + [f"L{i}" for i in range(1, 13)]
LATE_BLOCK = [f"L{i}" for i in range(6, 12)]        # L6..L11 (seam reference block)

RUNS = {  # run -> (kind, tag, has_postln)
    "electricity":          ("pretrained", "monash_electricity_hourly", True),
    "m4":                   ("pretrained", "m4_hourly", True),
    "electricity_randinit": ("randinit",  "monash_electricity_hourly", False),
}


# --------------------------------------------------------------------------- #
# debiased linear CKA (float64)
# --------------------------------------------------------------------------- #

def hsic1(K: np.ndarray, L: np.ndarray) -> float:
    """Unbiased HSIC (Song et al. 2012) for symmetric kernel matrices K, L (n >= 4)."""
    n = K.shape[0]
    assert n >= 4, "unbiased HSIC needs n >= 4"
    Kt = K.copy(); np.fill_diagonal(Kt, 0.0)
    Lt = L.copy(); np.fill_diagonal(Lt, 0.0)
    one = np.ones(n, dtype=np.float64)
    tKL = float((Kt * Lt).sum())                     # tr(K~ L~) for symmetric K~, L~
    sK = float(Kt.sum()); sL = float(Lt.sum())
    cross = float(one @ Kt @ (Lt @ one))             # 1' K~ L~ 1
    return (tKL + sK * sL / ((n - 1) * (n - 2)) - 2.0 * cross / (n - 2)) / (n * (n - 3))


def cka_debiased_matrix(feats: dict[str, np.ndarray], axis: list[str]) -> np.ndarray:
    """Full pairwise debiased linear CKA over `axis` (every (i,j) computed independently,
    so the symmetry gate is a genuine check rather than true by construction)."""
    kern = {ln: np.asarray(feats[ln], np.float64) @ np.asarray(feats[ln], np.float64).T
            for ln in axis}
    diag_h = {ln: hsic1(kern[ln], kern[ln]) for ln in axis}
    C = np.empty((len(axis), len(axis)), dtype=np.float64)
    for i, a in enumerate(axis):
        for j, b in enumerate(axis):
            C[i, j] = hsic1(kern[a], kern[b]) / np.sqrt(diag_h[a] * diag_h[b])
    return C


# --------------------------------------------------------------------------- #
# data loading (dataset-level per-series mean embeddings, same as dataset_effrank)
# --------------------------------------------------------------------------- #

def load_feats(kind: str, tag: str, has_postln: bool) -> tuple[dict, list[str], list[str]]:
    cp = _cache_path(kind, tag, SEED)
    assert cp.exists(), f"MISSING CACHE: {cp}"
    d = np.load(cp, allow_pickle=True)
    feats = {ln: np.stack([np.asarray(m, np.float64).mean(axis=0) for m in d[f"hs_{ln}"]])
             for ln in AXIS13}
    files = [str(cp.relative_to(OUT_DIR.parent))]
    axis = list(AXIS13)
    if has_postln:
        pp = cp.with_name(cp.stem + "__postln.npz")
        assert pp.exists(), f"MISSING CACHE: {pp}"
        dp = np.load(pp, allow_pickle=True)
        feats[POSTLN] = np.stack([np.asarray(m, np.float64).mean(axis=0) for m in dp["hs"]])
        files.append(str(pp.relative_to(OUT_DIR.parent)))
        axis.append(POSTLN)
    return feats, axis, files


# --------------------------------------------------------------------------- #
# per-run computation + gates
# --------------------------------------------------------------------------- #

def run_cka(run: str, kind: str, tag: str, has_postln: bool) -> dict:
    feats, axis, files = load_feats(kind, tag, has_postln)
    n = feats["Embed"].shape[0]
    print(f"\n=== CKA {run} ({kind}, {tag}) — n_series={n}, taps={len(axis)} ===")

    C = cka_debiased_matrix(feats, axis)

    # ---- gates ----
    diag = np.diag(C)
    print(f"  [gate] CKA(X,X) per layer: "
          f"{'ALL == 1.0 exactly' if np.all(diag == 1.0) else 'max|diag-1|=%.3e' % np.abs(diag-1).max()}")
    for ln, v in zip(axis, diag):
        print(f"          {ln:>11}: {v!r}")
    asym = float(np.abs(C - C.T).max())
    print(f"  [gate] max asymmetry |C - C.T| = {asym:.3e}  (< 1e-10: {asym < 1e-10})")
    assert asym < 1e-10

    idx = {ln: i for i, ln in enumerate(axis)}
    late = [idx[ln] for ln in LATE_BLOCK]
    seam = {"late_block": LATE_BLOCK,
            "mean_cka_L12_vs_late": float(np.mean([C[idx["L12"], j] for j in late])),
            "mean_pairwise_cka_within_late": float(np.mean(
                [C[i, j] for i in late for j in late if i < j]))}
    if has_postln:
        seam["mean_cka_postln_vs_late"] = float(np.mean([C[idx[POSTLN], j] for j in late]))
        seam["cka_postln_vs_L12"] = float(C[idx[POSTLN], idx["L12"]])

    out = {
        "provenance": {
            "run": run, "kind": kind, "tag": tag, "seed": SEED, "n_series": int(n),
            "representation": "dataset-level per-series mean content-patch embeddings "
                              "(same matrices as dataset_effrank)",
            "estimator": "linear-kernel CKA, unbiased (debiased) HSIC of Song et al. 2012",
            "dtype": "float64",
            "cache_files": files,
            "taps": len(axis),
            "postln_note": None if has_postln else
                "randinit has no L12_postln cache (never extracted) -> 13 taps",
            "amartya_axis_note": "Amartya's panels: electricity / m4_daily / solar_1h; ours: "
                                 "electricity / m4_HOURLY. Electricity is the only directly "
                                 "comparable panel; m4_daily was NOT extracted (mismatch "
                                 "recorded, per brief). His axis lacks postln -> 13-tap variant.",
        },
        "axis": axis,
        "cka_matrix": C.tolist(),
        "cka_matrix_13tap": C[np.ix_(range(13), range(13))].tolist(),
        "axis_13tap": AXIS13,
        "gates": {"diag_values": diag.tolist(),
                  "diag_all_exactly_1": bool(np.all(diag == 1.0)),
                  "max_asymmetry": asym},
        "seam_metrics": seam,
    }
    d = CKA_DIR / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "cka.json").write_text(json.dumps(out, indent=1))
    print(f"  [saved] {(d / 'cka.json').relative_to(OUT_DIR)}")
    print(f"  seam: CKA(L12, L6..L11) mean = {seam['mean_cka_L12_vs_late']:.4f} | "
          f"within-late mean = {seam['mean_pairwise_cka_within_late']:.4f}"
          + (f" | CKA(postln, L6..L11) mean = {seam['mean_cka_postln_vs_late']:.4f}"
             if has_postln else ""))
    return out


# --------------------------------------------------------------------------- #
# figure (reads cka.json only)
# --------------------------------------------------------------------------- #

def plot_run(run: str) -> None:
    m = json.loads((CKA_DIR / run / "cka.json").read_text())
    axis = m["axis"]
    C = np.asarray(m["cka_matrix"])
    has_pln = axis[-1] == POSTLN

    ncols = 2 if has_pln else 1
    fig, axs = plt.subplots(1, ncols, figsize=(7.2 * ncols, 6.2))
    axs = np.atleast_1d(axs)

    # (a) 13-tap variant, Amartya-comparable axis
    C13 = np.asarray(m["cka_matrix_13tap"])
    ax = axs[0]
    im = ax.imshow(C13, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(13)); ax.set_xticklabels(m["axis_13tap"], rotation=45, fontsize=7)
    ax.set_yticks(range(13)); ax.set_yticklabels(m["axis_13tap"], fontsize=7)
    ax.set_title(f"{run}: debiased linear CKA — 13 taps (external-comparison axis)", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)

    # (b) 14-tap version with postln separated by a thin white line
    if has_pln:
        ax = axs[1]
        im = ax.imshow(C, vmin=0, vmax=1, cmap="viridis")
        ax.axhline(12.5, color="white", lw=2)
        ax.axvline(12.5, color="white", lw=2)
        ax.set_xticks(range(len(axis))); ax.set_xticklabels(axis, rotation=45, fontsize=7)
        ax.set_yticks(range(len(axis))); ax.set_yticklabels(axis, fontsize=7)
        ax.set_title(f"{run}: 14 taps (postln beyond the white line)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.tight_layout()
    out = CKA_DIR / run / "fig_cka_matrix.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.relative_to(OUT_DIR)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)
    CKA_DIR.mkdir(parents=True, exist_ok=True)
    if not args.figures_only:
        for run, spec in RUNS.items():
            run_cka(run, *spec)
    print()
    for run in RUNS:
        plot_run(run)


if __name__ == "__main__":
    main()
