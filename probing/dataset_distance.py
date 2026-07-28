"""Pairwise dataset distance from RAW series (catch22 features + energy distance).

Quantitative distance between time-series DATASETS for the Near/Mid/Far OOD ladder's
x-axis. Operates on raw series only — never on hidden states.

Loading REUSES the existing Phase 0 paths (no new downloader):
  - forecasting trio (m4_hourly, monash_electricity_hourly, solar_1h):
    probing.id_data.load_seen_series under the "phase0_trio" dataset set
    (HF cache: autogluon/chronos_datasets; run with HF_DATASETS_OFFLINE=1 so a
    missing cache errors instead of downloading)
  - UEA sets: aeon.datasets.load_classification with extract_path=UEA_ROOT
    (the pre-existing local archive); a set is only included if its directory
    already exists there.

Pipeline (per dataset): sample up to MAX_SERIES series (fixed rng seed), z-normalize
each series, compute catch22 features, drop non-finite/constant series (counts
logged + recorded in the JSON). Features are standardized by the POOLED mean/std
across all datasets, then pairwise dataset distance = energy distance between the
two standardized feature clouds.

Outputs (OUT_DIR):
  distance_matrix.json      full provenance + pairwise matrix
  fig_distance_heatmap.png  annotated heatmap, ordered by mean distance to the ID trio
  fig_distance_mds.png      classical MDS (PCoA) projection, labeled points
Both figures are rendered FROM THE JSON ONLY (see --figures-only).

Run:
  python -m probing.dataset_distance                 # compute + figures (seed 0)
  python -m probing.dataset_distance --figures-only  # re-render figures from the JSON
  python -m probing.dataset_distance --sanity        # electricity split-half check
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

# fail loudly instead of silently downloading if an HF cache entry is missing
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing import config
from probing.config import OUT_DIR, SEED

# ----------------------------------------------------------------------------- #
# configuration (provenance for all of this is written into the JSON)
# ----------------------------------------------------------------------------- #
ID_TRIO = ("m4_hourly", "monash_electricity_hourly", "solar_1h")
UEA_ROOT = Path.home() / "aeon_data"          # pre-existing raw UEA archive
UEA_SPLIT = "train"
UEA_CHANNEL_POLICY = "channel_mean"           # multivariate case -> mean over channels
MAX_SERIES = 200
SIGMA_EPS = 1e-12
DIST_DIR = OUT_DIR / "distance"


def _uea_available() -> list[str]:
    """UEA sets whose raw archive already exists locally (never triggers a download)."""
    if not UEA_ROOT.is_dir():
        return []
    return sorted(p.name for p in UEA_ROOT.iterdir()
                  if p.is_dir() and any(p.glob(f"{p.name}_TRAIN*")))


# ----------------------------------------------------------------------------- #
# loading (reused Phase 0 paths)
# ----------------------------------------------------------------------------- #

def load_raw_series(name: str) -> list[np.ndarray]:
    """Raw 1-D series for one dataset via the EXISTING loaders."""
    if name in ID_TRIO:
        from probing.id_data import load_seen_series
        config.set_dataset_set("phase0_trio")     # trio tags live in this named set
        return load_seen_series(name)
    # UEA: one series per case; multivariate cases reduced per UEA_CHANNEL_POLICY
    from aeon.datasets import load_classification
    X, _y = load_classification(name, split=UEA_SPLIT, extract_path=str(UEA_ROOT))
    X = np.asarray(X, dtype=np.float64)           # (n_cases, n_channels, t)
    return [X[i].mean(axis=0) for i in range(X.shape[0])]


def sample_series(name: str, seed: int) -> list[np.ndarray]:
    series = load_raw_series(name)
    if len(series) <= MAX_SERIES:
        return series
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(series), size=MAX_SERIES, replace=False))
    return [series[i] for i in idx]


# ----------------------------------------------------------------------------- #
# features + distance
# ----------------------------------------------------------------------------- #

def catch22_features(series_list: list[np.ndarray]) -> tuple[np.ndarray, int, list[str]]:
    """(n_kept, 22) catch22 features of z-normalized series; returns (F, n_dropped, names)."""
    import pycatch22
    rows, names, dropped = [], None, 0
    for s in series_list:
        s = np.asarray(s, dtype=np.float64)
        if not np.all(np.isfinite(s)) or s.std() < SIGMA_EPS or len(s) < 10:
            dropped += 1
            continue
        z = (s - s.mean()) / s.std()
        out = pycatch22.catch22_all(z.tolist())
        vals = np.asarray(out["values"], dtype=np.float64)
        if names is None:
            names = list(out["names"])
        if not np.all(np.isfinite(vals)):
            dropped += 1
            continue
        rows.append(vals)
    return np.asarray(rows), dropped, (names or [])


def pooled_standardize(feats: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Standardize every dataset's features by the pooled mean/std across ALL datasets."""
    allf = np.concatenate(list(feats.values()), axis=0)
    mu, sd = allf.mean(axis=0), np.maximum(allf.std(axis=0), SIGMA_EPS)
    return {k: (v - mu) / sd for k, v in feats.items()}


def energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Energy distance E(A,B) = 2*E||a-b|| - E||a-a'|| - E||b-b'|| (V-statistic:
    all-pairs means, within-terms including the zero diagonal)."""
    from scipy.spatial.distance import cdist
    return float(2.0 * cdist(A, B).mean() - cdist(A, A).mean() - cdist(B, B).mean())


def compute(seed: int, out_dir: Path) -> Path:
    import pycatch22
    names = list(ID_TRIO) + _uea_available()
    feats, n_used, n_drop = {}, {}, {}
    for name in names:
        sampled = sample_series(name, seed)
        F, dropped, feat_names = catch22_features(sampled)
        feats[name], n_used[name], n_drop[name] = F, int(F.shape[0]), int(dropped)
        print(f"[{name:>26}] sampled={len(sampled)}  used={F.shape[0]}  dropped={dropped}")
    z = pooled_standardize(feats)
    n = len(names)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = energy_distance(z[names[i]], z[names[j]])

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "datasets": names,
        "id_trio": list(ID_TRIO),
        "n_series_used": n_used,
        "n_series_dropped": n_drop,
        "max_series_per_dataset": MAX_SERIES,
        "feature_dim": int(next(iter(feats.values())).shape[1]),
        "feature_names": feat_names,
        "seed": int(seed),
        "catch22_package": "pycatch22",
        "catch22_version": getattr(pycatch22, "__version__", "unknown"),
        "per_series_preprocessing": "z-normalize (drop non-finite, constant, or len<10 series)",
        "uea_channel_policy": UEA_CHANNEL_POLICY,
        "uea_split": UEA_SPLIT,
        "uea_extract_path": str(UEA_ROOT),
        "trio_loader": "probing.id_data.load_seen_series (dataset set phase0_trio, HF offline)",
        "feature_standardization": "pooled mean/std across all datasets (std clamped at 1e-12)",
        "distance_estimator": "energy distance, V-statistic (all-pairs means incl. zero diagonal), "
                              "Euclidean metric on standardized catch22 features",
        "matrix": M.tolist(),
    }
    path = out_dir / "distance_matrix.json"
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[saved] {path}")
    return path


# ----------------------------------------------------------------------------- #
# figures — rendered from the JSON ONLY (no recomputation)
# ----------------------------------------------------------------------------- #

def make_figures(json_path: Path) -> None:
    d = json.load(open(json_path))
    names, M = d["datasets"], np.asarray(d["matrix"])
    trio_idx = [names.index(t) for t in d["id_trio"]]
    mean_to_trio = M[:, trio_idx].mean(axis=1)
    order = np.argsort(mean_to_trio)
    names_o = [names[i] for i in order]
    Mo = M[np.ix_(order, order)]
    out_dir = json_path.parent

    fig, ax = plt.subplots(figsize=(9.5, 8))
    im = ax.imshow(Mo, cmap="viridis")
    ax.set_xticks(range(len(names_o)), names_o, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(names_o)), names_o, fontsize=8)
    for i in range(len(names_o)):
        for j in range(len(names_o)):
            ax.text(j, i, f"{Mo[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                    color="white" if Mo[i, j] < Mo.max() * 0.6 else "black")
    ax.set_title("Pairwise dataset energy distance (catch22 features)\n"
                 "ordered by mean distance to the ID trio "
                 f"(seed={d['seed']}, n≤{d['max_series_per_dataset']}/dataset)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="energy distance")
    fig.tight_layout()
    out = out_dir / "fig_distance_heatmap.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")

    # classical MDS / PCoA: deterministic double-centering + eigendecomposition
    D2 = M ** 2
    J = np.eye(len(names)) - np.ones((len(names), len(names))) / len(names)
    Bmat = -0.5 * J @ D2 @ J
    w, V = np.linalg.eigh(Bmat)
    top = np.argsort(w)[::-1][:2]
    XY = V[:, top] * np.sqrt(np.maximum(w[top], 0.0))
    fig, ax = plt.subplots(figsize=(8.5, 7))
    trio = set(d["id_trio"])
    for i, nm in enumerate(names):
        c = "#d62728" if nm in trio else "#1f77b4"
        ax.scatter(XY[i, 0], XY[i, 1], s=60, color=c, zorder=3)
        ax.annotate(nm, (XY[i, 0], XY[i, 1]), fontsize=8,
                    xytext=(5, 4), textcoords="offset points")
    ax.scatter([], [], color="#d62728", label="ID trio")
    ax.scatter([], [], color="#1f77b4", label="UEA")
    var = np.maximum(w[top], 0.0)
    tot = np.maximum(w, 0.0).sum() or 1.0
    ax.set_xlabel(f"PCoA 1 ({100 * var[0] / tot:.0f}% of positive inertia)")
    ax.set_ylabel(f"PCoA 2 ({100 * var[1] / tot:.0f}%)")
    ax.set_title(f"Classical MDS (PCoA) of the dataset distance matrix (seed={d['seed']})")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.tight_layout()
    out = out_dir / "fig_distance_mds.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"[saved] {out}")


# ----------------------------------------------------------------------------- #
# sanity: distance(electricity, electricity-other-half) vs distance(electricity, UWave)
# ----------------------------------------------------------------------------- #

def sanity(seed: int) -> None:
    elec = sample_series("monash_electricity_hourly", seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(elec))
    half_a = [elec[i] for i in perm[: len(elec) // 2]]
    half_b = [elec[i] for i in perm[len(elec) // 2:]]
    uwave = sample_series("UWaveGestureLibrary", seed)
    Fa, da, _ = catch22_features(half_a)
    Fb, db, _ = catch22_features(half_b)
    Fu, du, _ = catch22_features(uwave)
    z = pooled_standardize({"a": Fa, "b": Fb, "u": Fu})
    d_split = energy_distance(z["a"], z["b"])
    d_uwave = energy_distance(z["a"], z["u"])
    print(f"n: elec_half_a={Fa.shape[0]} (drop {da})  elec_half_b={Fb.shape[0]} (drop {db})  "
          f"uwave={Fu.shape[0]} (drop {du})")
    print(f"distance(electricity_half_a, electricity_half_b) = {d_split:.4f}")
    print(f"distance(electricity_half_a, UWaveGestureLibrary) = {d_uwave:.4f}")
    print(f"ratio = {d_uwave / max(d_split, 1e-12):.1f}x")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out-dir", type=Path, default=DIST_DIR)
    ap.add_argument("--figures-only", action="store_true",
                    help="re-render both figures from the existing JSON (no recompute)")
    ap.add_argument("--sanity", action="store_true",
                    help="electricity split-half vs UWave check (prints numbers only)")
    args = ap.parse_args()
    if args.sanity:
        sanity(args.seed)
        return
    json_path = args.out_dir / "distance_matrix.json"
    if not args.figures_only:
        json_path = compute(args.seed, args.out_dir)
    make_figures(json_path)


if __name__ == "__main__":
    main()
