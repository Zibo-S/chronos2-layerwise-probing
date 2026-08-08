"""Distance-vs-gain ladder join: catch22 energy distance (source->target) vs layer-selection gain.

Post-processing + a raw-series distance computation. Zero model forward passes.

WHAT THIS JOINS
  x = catch22 energy distance from a SOURCE dataset to a TARGET dataset
  y = source-val-selected relative gain over L12 (%) for that (source, target) cell
  28 cells = 16 "near" (the 4x4 source-set grid, incl. the 4 ID diagonal cells)
            + 12 "far"  (the 4x3 pretraining-OOD grid)
rendered for BOTH gain versions (extended_v2, extended_v3_rolling) as side-by-side panels,
plus a variant excluding the m4_hourly source row.

REUSE (nothing reimplemented)
  * probing.dataset_distance : sample_series / catch22_features / pooled_standardize /
    energy_distance / MAX_SERIES / SIGMA_EPS  — identical conventions to the prototype.
  * probing.id_data.load_ood_target_series : the OOD target loaders carry the manifest
    preprocessing verbatim (SG 15min->hourly requiring >=3 of 4 samples, no fill; Coastal
    TEMP+PSAL only -> 24 stations x 2 = 48 univariate series; BOOM one manifest-pinned
    variate per query). They read from OOD_TARGET_ROOT, which this module points at the
    local raw cache.
  BOOM is additionally restricted to the FIRST 200 manifest entries by file order (only
  those shards are staged); the per-variate read is still id_data's verbatim reader.

The full 7x7 matrix (4 sources + 3 targets) is computed in ONE run so that the pooled
feature standardization — and hence every distance — comes from a single fit.

Gain columns (verified against Egor's published figure values, which correspond to
extended_v3_rolling + source_selected):
  near: source_val_relative_gain_summary_q9.csv -> relative_gain_pct (one row per cell)
  far : ood_pretrain_transfer_results__q9.csv   -> relative_gain_pct, filtered to
        selection_method == "source_selected" (the file also carries an "oracle" row/cell)

Outputs (results/distance/ladder/):
  distance_matrix_7x7.json, join_extended_v2.csv, join_extended_v3_rolling.csv,
  fig_distance_vs_gain.png, fig_distance_vs_gain_noM4.png

Run:  python -m probing.distance_ladder [--figures-only]
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.config import OUT_DIR, SEED

# point the OOD loaders at the local raw cache BEFORE importing id_data (module-level read)
RAW_CACHE = (OUT_DIR / "distance" / "raw_cache").resolve()
os.environ.setdefault("OOD_TARGET_ROOT", str(RAW_CACHE))

from probing.dataset_distance import (          # noqa: E402  (env must be set first)
    MAX_SERIES,
    catch22_features,
    energy_distance,
    pooled_standardize,
    sample_series,
)

LADDER_DIR = OUT_DIR / "distance" / "ladder"
BOOM_FIRST_N = 200                    # only these shards are staged (first N by file order)

# Missing-data rule, applied UNIFORMLY to all 7 datasets before feature extraction.
# The prototype's filter dropped any series containing a NaN, which discarded 95.5% of
# wind_farms_hourly (322/337 contain NaNs) and left 5-7 usable series — too few for an
# energy distance and the sole cause of a seed-stability failure. We instead take each
# series' LONGEST CONTIGUOUS FINITE SEGMENT and keep the series if that segment is at
# least MIN_SEGMENT points. Segments are NOT concatenated across gaps: splicing around a
# NaN run would corrupt the autocorrelation-family catch22 features (AC_nl_036, etc.).
MIN_SEGMENT = 512

SOURCES = ("m4_hourly", "monash_electricity_hourly", "uber_tlc_hourly", "wind_farms_hourly")
TARGETS_FAR = ("sg_carpark", "coastal_ts", "boom_hourly")
ALL_DATASETS = SOURCES + TARGETS_FAR

GAIN_VERSIONS = ("extended_v2", "extended_v3_rolling")
# anchors live in v3_rolling + source_selected (Egor's shared figure); 1pp threshold
ANCHOR_VERSION = "extended_v3_rolling"
ANCHORS = {("monash_electricity_hourly", "sg_carpark"): 5.8,
           ("uber_tlc_hourly", "sg_carpark"): 11.7,
           ("m4_hourly", "boom_hourly"): -40.3}
ANCHOR_TOL_PP = 1.0

SHORT = {"m4_hourly": "m4", "monash_electricity_hourly": "electricity",
         "uber_tlc_hourly": "uber", "wind_farms_hourly": "wind_farms",
         "sg_carpark": "sg_carpark", "coastal_ts": "coastal_ts", "boom_hourly": "boom_hourly"}


# --------------------------------------------------------------------------- #
# raw series for all 7 datasets
# --------------------------------------------------------------------------- #

def load_series(name: str) -> list[np.ndarray]:
    """Raw 1-D series. Sources via the existing HF loader; targets via id_data's
    manifest-verbatim OOD loaders (BOOM restricted to the first BOOM_FIRST_N entries)."""
    from probing import config, id_data

    if name in SOURCES:
        config.set_dataset_set("extended_v2")            # the set that names all 4 sources
        return [np.asarray(s, dtype=np.float64) for s in id_data.load_seen_series(name)]

    if name == "boom_hourly":
        sel = json.load(open(id_data.BOOM_MANIFEST))["selected"][:BOOM_FIRST_N]
        root = Path(os.environ["OOD_TARGET_ROOT"]) / "boom_hourly"
        out = []
        for e in sel:
            p = root / e["query_dir"] / "data-00000-of-00001.arrow"
            if not p.exists():
                raise FileNotFoundError(f"BOOM shard missing: {p}")
            out.append(id_data._boom_read_variate(p, e["variate_index"]))   # verbatim reader
        return out

    return [np.asarray(s, dtype=np.float64)
            for s in id_data.load_ood_target_series(name)["series"]]


# --------------------------------------------------------------------------- #
# 7x7 distance matrix (one pooled standardization)
# --------------------------------------------------------------------------- #

def longest_finite_segment(x: np.ndarray) -> np.ndarray:
    """The longest run of consecutive finite values in x (never spliced across gaps)."""
    x = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not finite.any():
        return x[:0]
    edges = np.diff(np.concatenate(([0], finite.view(np.int8), [0])))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    k = int(np.argmax(ends - starts))
    return x[starts[k]:ends[k]]


def apply_segment_rule(series: list[np.ndarray]) -> tuple[list[np.ndarray], dict]:
    """Longest contiguous finite segment per series; keep it iff >= MIN_SEGMENT points."""
    segs = [longest_finite_segment(s) for s in series]
    kept = [g for g in segs if len(g) >= MIN_SEGMENT]
    lens = np.array([len(g) for g in kept], dtype=float)
    stats = {"n_raw": len(series), "n_kept": len(kept),
             "n_dropped_short_segment": len(series) - len(kept),
             "median_segment_length": float(np.median(lens)) if len(kept) else 0.0,
             "min_segment_length": float(lens.min()) if len(kept) else 0.0,
             "median_raw_length": float(np.median([len(s) for s in series]))}
    return kept, stats


def compute_matrix(seed: int, verbose: bool = True) -> dict:
    import pycatch22

    feats, n_used, n_drop, n_raw, seg_stats = {}, {}, {}, {}, {}
    for name in ALL_DATASETS:
        raw = load_series(name)
        n_raw[name] = len(raw)
        series, stats = apply_segment_rule(raw)          # uniform missing-data rule
        seg_stats[name] = stats
        sampled = sample_series_list(series, seed)       # <=200 of the QUALIFYING series
        F, dropped, feat_names = catch22_features(sampled)
        feats[name] = F
        n_used[name] = int(F.shape[0])
        n_drop[name] = int(dropped)
        if verbose:
            print(f"    {name:>26}: raw={stats['n_raw']:>4} "
                  f"seg>={MIN_SEGMENT}:{stats['n_kept']:>4} "
                  f"(drop {stats['n_dropped_short_segment']:>3}, med_seg "
                  f"{stats['median_segment_length']:>7.0f}) sampled={len(sampled):>4} "
                  f"kept={F.shape[0]:>4} dropped={dropped:>3}")

    z = pooled_standardize(feats)                        # ONE fit across all 7
    names = list(ALL_DATASETS)
    M = np.zeros((len(names), len(names)))
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            M[i, j] = M[j, i] = energy_distance(z[names[i]], z[names[j]])

    return {
        "seed": seed, "datasets": names,
        "matrix": M.tolist(),
        "n_raw_series": n_raw, "n_series_used": n_used, "n_series_dropped": n_drop,
        "segment_rule": {
            "rule": "per series, take the LONGEST CONTIGUOUS FINITE segment; keep the series "
                    "iff that segment has >= min_segment_points. Segments are NEVER "
                    "concatenated across NaN gaps — splicing would corrupt the "
                    "autocorrelation-family catch22 features.",
            "min_segment_points": MIN_SEGMENT,
            "applied_to": "all 7 datasets uniformly, before sampling and feature extraction",
            "supersedes": "the prototype's drop-series-if-any-NaN filter, which discarded "
                          "322/337 wind_farms_hourly series (5-7 usable) and caused a "
                          "seed-stability failure (Spearman 0.9117 < 0.95)",
            "per_dataset": seg_stats,
        },
        "catch22_package": "pycatch22",
        "catch22_version": getattr(pycatch22, "__version__", "unknown"),
        "max_series": MAX_SERIES,
        "boom_first_n": BOOM_FIRST_N,
    }


def sample_series_list(series: list[np.ndarray], seed: int) -> list[np.ndarray]:
    """dataset_distance.sample_series takes a NAME; this is the same rule applied to a list."""
    if len(series) <= MAX_SERIES:
        return series
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(len(series), size=MAX_SERIES, replace=False))
    return [series[i] for i in idx]


def dist_lookup(mat: dict) -> dict:
    names = mat["datasets"]
    M = np.asarray(mat["matrix"])
    return {(a, b): float(M[i, j]) for i, a in enumerate(names) for j, b in enumerate(names)}


# --------------------------------------------------------------------------- #
# gains
# --------------------------------------------------------------------------- #

def load_gains(version: str) -> pd.DataFrame:
    """28 rows: 16 near (4x4) + 12 far (4x3, source_selected only)."""
    near = pd.read_csv(OUT_DIR / version / "ood_transfer" /
                       "source_val_relative_gain_summary_q9.csv")
    far = pd.read_csv(OUT_DIR / version / "ood_pretrain_transfer" /
                      "ood_pretrain_transfer_results__q9.csv")
    far = far[far.selection_method == "source_selected"].copy()

    n = near.rename(columns={"source_dataset": "source", "target_dataset": "target",
                             "relative_gain_pct": "gain"})[["source", "target", "gain"]]
    n["tier"] = "near"
    f = far.rename(columns={"relative_gain_pct": "gain"})[["source", "target", "gain"]]
    f["tier"] = "far"
    out = pd.concat([n, f], ignore_index=True)
    out["gain_version"] = version
    return out


# --------------------------------------------------------------------------- #
# join + figures
# --------------------------------------------------------------------------- #

def build_join(version: str, dl: dict) -> pd.DataFrame:
    g = load_gains(version)
    g["distance"] = [dl[(s, t)] for s, t in zip(g.source, g.target)]
    return g[["source", "target", "tier", "distance", "gain", "gain_version"]]


def _spearman(x, y) -> float:
    from scipy.stats import spearmanr
    if len(x) < 3:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def _panel(ax, df, title):
    colors = {s: f"C{i}" for i, s in enumerate(SOURCES)}
    for src in SOURCES:
        for tier, marker, size in (("near", "o", 55), ("far", "^", 70)):
            sub = df[(df.source == src) & (df.tier == tier)]
            if sub.empty:
                continue
            ax.scatter(sub.distance, sub.gain, marker=marker, s=size,
                       color=colors[src], alpha=0.85, edgecolor="black", linewidth=0.4,
                       label=f"{SHORT[src]} ({tier})")
    ax.axhline(0, color="gray", ls=":", lw=1)
    rho = _spearman(df.distance.values, df.gain.values)
    ax.set_title(f"{title}\nSpearman rho(distance, gain) = {rho:+.3f}  (n={len(df)})", fontsize=10)
    ax.set_xlabel("catch22 energy distance (source -> target)")
    ax.set_ylabel("source-val-selected relative gain over L12 (%)")
    ax.grid(alpha=0.3)
    return rho


def make_figure(joins: dict[str, pd.DataFrame], path: Path, suptitle: str) -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.0), sharey=False)
    rhos = {}
    for ax, v in zip(axes, GAIN_VERSIONS):
        rhos[v] = _panel(ax, joins[v], v)
    h, l = axes[0].get_legend_handles_labels()
    axes[0].legend(h, l, fontsize=7, ncol=2, loc="best")
    fig.suptitle(suptitle, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path.relative_to(OUT_DIR)}")
    return rhos


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)
    LADDER_DIR.mkdir(parents=True, exist_ok=True)
    mat_path = LADDER_DIR / "distance_matrix_7x7.json"

    if args.figures_only:
        blob = json.loads(mat_path.read_text())
        joins = {v: pd.read_csv(LADDER_DIR / f"join_{v}.csv") for v in GAIN_VERSIONS}
    else:
        print(f"=== distance matrix, seed={SEED} (OOD_TARGET_ROOT={os.environ['OOD_TARGET_ROOT']}) ===")
        m0 = compute_matrix(SEED)
        print(f"\n=== distance matrix, seed=1 (stability gate) ===")
        m1 = compute_matrix(1)

        names = m0["datasets"]
        M0, M1 = np.asarray(m0["matrix"]), np.asarray(m1["matrix"])
        iu = np.triu_indices(len(names), k=1)
        rho_seed = _spearman(M0[iu], M1[iu])

        blob = {
            "provenance": {
                "seed": SEED, "sources": list(SOURCES), "targets_far": list(TARGETS_FAR),
                "max_series": MAX_SERIES, "boom_first_n": BOOM_FIRST_N,
                "raw_cache": str(RAW_CACHE),
                "reused_modules": ["probing.dataset_distance (catch22 / energy distance / "
                                   "pooled standardization / <=200 series)",
                                   "probing.id_data.load_ood_target_series (manifest-verbatim "
                                   "target preprocessing)"],
                "preprocessing": {
                    "sg_carpark": "15min->hourly mean of AVAILABLE samples, >=3 of 4 required, "
                                  "else NaN; no fill / no cross-hour interpolation",
                    "coastal_ts": "TEMP + PSAL only (PRES_REL dropped); 24 stations x 2 = 48 "
                                  "univariate series, fixed variate order",
                    "boom_hourly": f"manifest-pinned variate per query; FIRST {BOOM_FIRST_N} "
                                   "entries by file order",
                    "missing_data_rule": f"longest contiguous finite segment per series, kept "
                                         f"iff >= {MIN_SEGMENT} points; no splicing across gaps; "
                                         f"applied uniformly to all 7 datasets",
                },
                "one_pooled_standardization": "all 7 datasets standardized by a single pooled "
                                              "mean/std, so within-4x4 and source->target "
                                              "distances are on the same scale",
                "gain_columns": {
                    "near": "results/<version>/ood_transfer/source_val_relative_gain_summary_q9"
                            ".csv :: relative_gain_pct",
                    "far": "results/<version>/ood_pretrain_transfer/ood_pretrain_transfer_"
                           "results__q9.csv :: relative_gain_pct where selection_method="
                           "'source_selected'"},
                "anchor_provenance": "the three published cross-check values correspond to "
                                     f"{ANCHOR_VERSION} + source_selected (Egor's shared figure), "
                                     "NOT extended_v2; the gate is pointed there",
                "note_m4_boom_oracle_degenerate": "m4_hourly->boom_hourly has oracle gain exactly "
                                                  "0.0000 in both versions (the oracle selected "
                                                  "L12 itself); the join uses source_selected, so "
                                                  "this does not affect any joined cell",
            },
            "seed0": m0, "seed1": m1,
            "seed_stability_spearman": rho_seed,
        }
        mat_path.write_text(json.dumps(blob, indent=1))
        print(f"\n  [saved] {mat_path.relative_to(OUT_DIR)}")

        dl = dist_lookup(m0)
        joins = {}
        for v in GAIN_VERSIONS:
            j = build_join(v, dl)
            j.to_csv(LADDER_DIR / f"join_{v}.csv", index=False)
            joins[v] = j
            print(f"  [saved] ladder/join_{v}.csv  ({len(j)} rows)")

    # ---------------- figures ----------------
    print()
    rhos_all = make_figure(joins, LADDER_DIR / "fig_distance_vs_gain.png",
                           "Distance vs source-val-selected gain — all 28 cells "
                           "(circles = near 4x4, triangles = far 4x3)")
    joins_no_m4 = {v: d[d.source != "m4_hourly"].reset_index(drop=True) for v, d in joins.items()}
    rhos_nom4 = make_figure(joins_no_m4, LADDER_DIR / "fig_distance_vs_gain_noM4.png",
                            "Distance vs gain — EXCLUDING the m4_hourly source row (21 cells)")

    blob["spearman"] = {"all_cells": rhos_all, "excluding_m4_source": rhos_nom4}
    mat_path.write_text(json.dumps(blob, indent=1))

    print("\n=== Spearman rho(distance, gain) ===")
    for v in GAIN_VERSIONS:
        print(f"  {v:>20}: all 28 cells rho={rhos_all[v]:+.4f} | "
              f"excl. m4 source (21) rho={rhos_nom4[v]:+.4f}")


if __name__ == "__main__":
    main()
