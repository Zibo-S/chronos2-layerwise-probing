"""Series-level resampling CIs / stability bands for the repr-metrics results.

Post-processing ONLY — reads the existing per-patch caches; zero forward passes.

RESAMPLE UNIT = SERIES. Patches (and windows) within a series are highly correlated, so
the resample unit must be the series, matching the cluster-bootstrap convention used for
the probe CIs. Patches are NEVER resampled.

----------------------------------------------------------------------------------------
TWO SCHEMES, DELIBERATELY SPLIT (see BIAS_NOTE below)
----------------------------------------------------------------------------------------
  * bootstrap, m = n = 200, WITH replacement
      -> used ONLY for the prompt-level metrics (normalized entropy mean, effective-rank
         mean). These are plain sample means, hence unbiased under resampling, so their
         percentile intervals are genuine 95% CIs.

  * subsampling, m = 126 = floor(0.632 * 200), WITHOUT replacement
      -> used for (i) the dataset-level spectral statistics' STABILITY BAND and (ii) ALL
         comparative/derived answers (dip test, argmax/peak separation, flatness range).
         Spectral statistics grow with the number of DISTINCT series, so comparisons are
         only meaningful at a FIXED distinct-n, where the shared offset cancels.

No naive-bootstrap CIs are produced for the spectral statistics anywhere. The dataset-level
headline number stays the full-n (n=200) point estimate.

PAIRED DRAWS: one index array per scheme per run, reused for every layer and metric, so
layer differences (e.g. L11 - L12_postln) are properly paired. Hashes are printed/stored.

EXACTNESS NOTE: dataset-level metrics need only the singular values of Z[idx]. Since
Z[idx] Z[idx]ᵀ = (Z Zᵀ)[idx][:, idx], the 200x200 Gram matrix is computed ONCE per layer
and each draw takes eigenvalues of its resampled submatrix. ``_verify_gram_equivalence``
asserts this reproduces the committed ``matrix_entropy`` / ``effective_rank`` on every
layer before any resampling runs. All eigendecompositions are float64.

Outputs:
  results/repr_metrics/bootstrap/{run}/bootstrap.json
  results/repr_metrics/bootstrap/{run}/fig_effrank_by_layer_ci.png       (pretrained runs)
  results/repr_metrics/bootstrap/fig_pretrained_vs_randinit_electricity_ci.png
Figures render from the JSONs only (--figures-only). Originals are never overwritten.

Run:  python -m probing.repr_metrics_bootstrap [--B 5000] [--figures-only]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

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

BOOT_DIR = RM_DIR / "bootstrap"
POSTLN = "L12_postln"
BASE_AXIS = ["Embed"] + [f"L{i}" for i in range(1, 13)]
DEFAULT_B = 5000
SUB_FRACTION = 0.632                 # -> m = 126 of 200, matching the diagnostic
GATE_M_OFFSET = 1                    # near-full subsample for the 2% gate: m = n - 1 = 199
GATE_B = 1000

PROMPT_METRICS = ("prompt_entropy_norm_mean", "prompt_effrank_mean")
DATASET_METRICS = ("dataset_effrank", "dataset_entropy")

RUNS = {                             # run -> (kind, tag, has_postln)
    "electricity":          ("pretrained", "monash_electricity_hourly", True),
    "m4":                   ("pretrained", "m4_hourly", True),
    "electricity_randinit": ("randinit",  "monash_electricity_hourly", False),
}

ENTROPY_BAND = ("L6", "L7")
EFFRANK_BAND = ("L10", "L11")

BIAS_NOTE = (
    "Spectral statistics of the dataset-level matrix (effective rank, matrix entropy) are "
    "NOT smooth functionals of the empirical distribution at fixed n: they grow with the "
    "number of DISTINCT sampling units. A classical bootstrap draw (n=200 with replacement) "
    "retains only ~63.2% distinct series (~126.5 of 200 observed), so the resampled statistic "
    "sits systematically BELOW the full-sample value. The compression is layer-dependent, "
    "scaling with the layer's rank: on pretrained electricity we measured -11.2% at Embed "
    "(point effrank 17.4), -26.1% at L12 (23.0), -29.9% at L6 (62.6) and -32.3% at L11 (86.3). "
    "Because high-rank layers are compressed more than low-rank ones, a naive bootstrap does "
    "not merely widen these curves, it distorts their SHAPE and drags the argmax toward lower "
    "layers. Controls confirm this is an estimator property, not a pipeline error: an "
    "identity index (all 200 series, each once) reproduces the point estimate exactly, and "
    "evaluating only the DISTINCT rows of a draw (no duplicates) recovers most of the gap "
    "(e.g. L11: 60.4 distinct-only vs 58.4 with duplicates vs 86.3 full-n) — i.e. the effect "
    "is driven by distinct-unit coverage, with duplication a smaller second-order term. We "
    "therefore report the full-n point estimate as primary for spectral statistics, use "
    "m-out-of-n subsampling WITHOUT replacement (m=126) for every cross-layer comparison so "
    "the shared offset cancels, and reserve true bootstrap CIs for the prompt-level means, "
    "which are unbiased sample means."
)


# --------------------------------------------------------------------------- #
# spectrum helpers (float64; guards mirror the committed metric functions)
# --------------------------------------------------------------------------- #

def _metrics_from_gram_eigs(lam: np.ndarray) -> tuple[float, float]:
    """(matrix_entropy, effective_rank) from Gram eigenvalues lam = s^2."""
    lam = np.clip(np.asarray(lam, dtype=np.float64), 0.0, None)
    mx = lam.max()
    if mx <= 0.0:
        return 0.0, 1.0
    le = lam[lam >= 1e-12 * mx]
    p = le / le.sum()
    s1 = float(-(p * np.log(p)).sum())
    se = np.sqrt(lam[lam >= 1e-24 * mx])
    q = se / se.sum()
    er = float(np.exp(-(q * np.log(q)).sum()))
    return s1, er


def _verify_gram_equivalence(Z: np.ndarray, layer: str, tol: float = 1e-8) -> None:
    lam = np.linalg.eigvalsh(Z @ Z.T)[::-1]
    s1g, erg = _metrics_from_gram_eigs(lam)
    s1d, erd = matrix_entropy(Z), effective_rank(Z)
    assert abs(s1g - s1d) < tol, f"{layer}: gram entropy {s1g} vs direct {s1d}"
    assert abs(erg - erd) / max(erd, 1e-12) < tol, f"{layer}: gram effrank {erg} vs direct {erd}"


def _subsample_indices(n: int, m: int, B: int, seed: int) -> np.ndarray:
    """(B, m) indices drawn WITHOUT replacement (distinct series within each draw)."""
    rng = np.random.default_rng(seed)
    return np.argsort(rng.random((B, n)), axis=1)[:, :m]


def _hash(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a, dtype=np.int64).tobytes()).hexdigest()[:16]


def _spectral_draws(G: np.ndarray, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    B = idx.shape[0]
    en = np.empty(B); er = np.empty(B)
    for b in range(B):
        i = idx[b]
        en[b], er[b] = _metrics_from_gram_eigs(np.linalg.eigvalsh(G[np.ix_(i, i)])[::-1])
    return en, er


def _pct(a: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #

def _load_layer_mats(kind: str, tag: str, layer: str) -> list[np.ndarray]:
    p = _cache_path(kind, tag, SEED)
    if layer == POSTLN:
        p = p.with_name(p.stem + "__postln.npz")
        assert p.exists(), f"MISSING CACHE: {p}"
        return [np.asarray(Z, dtype=np.float64) for Z in np.load(p, allow_pickle=True)["hs"]]
    assert p.exists(), f"MISSING CACHE: {p}"
    return [np.asarray(Z, dtype=np.float64) for Z in np.load(p, allow_pickle=True)[f"hs_{layer}"]]


def _cache_paths(kind: str, tag: str, has_postln: bool) -> list[str]:
    p = _cache_path(kind, tag, SEED)
    out = [str(p.relative_to(OUT_DIR.parent))]
    if has_postln:
        out.append(str(p.with_name(p.stem + "__postln.npz").relative_to(OUT_DIR.parent)))
    return out


# --------------------------------------------------------------------------- #
# per-run computation
# --------------------------------------------------------------------------- #

def run_resampling(run: str, kind: str, tag: str, has_postln: bool, B: int) -> dict:
    axis = BASE_AXIS + ([POSTLN] if has_postln else [])
    print(f"\n=== {run} ({kind}, {tag}) — B={B}, unit=series ===")

    grams, pe_norm, pe_rank, point = {}, {}, {}, {}
    n_series = None
    for ln in axis:
        mats = _load_layer_mats(kind, tag, ln)
        if n_series is None:
            n_series = len(mats)
        assert len(mats) == n_series, f"{ln}: {len(mats)} != {n_series}"
        Z = np.stack([m.mean(axis=0) for m in mats])
        _verify_gram_equivalence(Z, ln)
        grams[ln] = Z @ Z.T
        pe_norm[ln] = np.array([normalized_matrix_entropy(m) for m in mats])
        pe_rank[ln] = np.array([effective_rank(m) for m in mats])
        point[ln] = {"dataset_effrank": effective_rank(Z), "dataset_entropy": matrix_entropy(Z),
                     "prompt_entropy_norm_mean": float(pe_norm[ln].mean()),
                     "prompt_effrank_mean": float(pe_rank[ln].mean())}
        del mats
    m_sub = int(np.floor(SUB_FRACTION * n_series))
    m_gate = n_series - GATE_M_OFFSET
    print(f"  n_series={n_series}, layers={len(axis)}; gram-equivalence verified on all layers")
    print(f"  schemes: bootstrap m={n_series} WITH replacement (prompt means only) | "
          f"subsampling m={m_sub} WITHOUT replacement (spectral bands + all comparisons)")

    # paired index arrays: one per scheme, reused across every layer and metric
    boot_idx = np.random.default_rng(SEED).integers(0, n_series, size=(B, n_series))
    sub_idx = _subsample_indices(n_series, m_sub, B, SEED)
    gate_idx = _subsample_indices(n_series, m_gate, GATE_B, SEED + 1)
    print(f"  boot idx sha256[:16]={_hash(boot_idx)} shape={boot_idx.shape}")
    print(f"  sub  idx sha256[:16]={_hash(sub_idx)} shape={sub_idx.shape}  (all rows distinct)")

    prompt_ci, sub_band, sub_draws, prompt_sub_draws = {}, {}, {}, {}
    t0 = time.time()
    for li, ln in enumerate(axis):
        # (c)/(d) true bootstrap CIs — plain sample means, unbiased
        prompt_ci[ln] = {}
        for m, vals in ((PROMPT_METRICS[0], pe_norm[ln]), (PROMPT_METRICS[1], pe_rank[ln])):
            d = vals[boot_idx].mean(axis=1)
            lo, hi = _pct(d)
            prompt_ci[ln][m] = {"point": point[ln][m], "boot_mean": float(d.mean()),
                                "lo": lo, "hi": hi}
        # subsampling m=126: spectral stability band + paired draws for comparisons
        en, er = _spectral_draws(grams[ln], sub_idx)
        sub_draws[ln] = {"dataset_entropy": en, "dataset_effrank": er}
        sub_band[ln] = {}
        for m, d in (("dataset_effrank", er), ("dataset_entropy", en)):
            lo, hi = _pct(d)
            sub_band[ln][m] = {"point_full_n": point[ln][m], "sub_mean": float(d.mean()),
                               "lo": lo, "hi": hi,
                               "signed_offset_vs_full_n": float((d.mean() - point[ln][m])
                                                                / max(abs(point[ln][m]), 1e-12))}
        # prompt-level under the SAME subsample draws (for the argmax comparison)
        prompt_sub_draws[ln] = {m: v[sub_idx].mean(axis=1)
                                for m, v in ((PROMPT_METRICS[0], pe_norm[ln]),
                                             (PROMPT_METRICS[1], pe_rank[ln]))}
        print(f"    [{li + 1:2d}/{len(axis)}] {ln:>11}  ({time.time() - t0:5.1f}s)", end="\r")
    print(f"\n  resampling done in {time.time() - t0:.1f}s")

    # ---------------- gates ----------------
    # (1) near-full subsample (m = n-1) must land within 2% of the full-n point estimate
    g1 = {}
    for ln in axis:
        _, er_g = _spectral_draws(grams[ln], gate_idx)
        dev = (er_g.mean() - point[ln]["dataset_effrank"]) / max(point[ln]["dataset_effrank"], 1e-12)
        g1[ln] = {"m": m_gate, "sub_mean": float(er_g.mean()),
                  "point_full_n": point[ln]["dataset_effrank"], "signed_deviation": float(dev)}
    worst_ln = max(g1, key=lambda k: abs(g1[k]["signed_deviation"]))
    worst = g1[worst_ln]["signed_deviation"]
    print(f"  [gate m={m_gate}] worst signed deviation vs full-n effrank: "
          f"{worst * 100:+.3f}% at {worst_ln}  (|dev| < 2%: {abs(worst) < 0.02})")
    assert abs(worst) < 0.02, (f"{run}: near-full subsample deviates {worst*100:.2f}% at "
                               f"{worst_ln} — expected < 2%")

    # (2) the m=126 offset must be monotone in the layer's rank
    ranks = np.array([point[ln]["dataset_effrank"] for ln in axis])
    offs = np.array([sub_band[ln]["dataset_effrank"]["signed_offset_vs_full_n"] for ln in axis])
    order = np.argsort(ranks)
    from scipy.stats import spearmanr
    rho = float(spearmanr(ranks, offs).statistic)
    rank_spread = float(ranks.max() / max(ranks.min(), 1e-12))
    print(f"\n  [gate] m={m_sub} offset vs layer rank (sorted by rank):")
    print(f"      {'layer':>11} | {'point effrank':>13} | {'offset %':>9}")
    for i in order:
        print(f"      {axis[i]:>11} | {ranks[i]:13.3f} | {offs[i] * 100:+8.3f}%")
    print(f"      Spearman rho(rank, signed offset) = {rho:+.4f}  "
          f"(more negative offset at higher rank => rho < 0); rank spread x{rank_spread:.2f}")
    monotonic = {"spearman_rho": rho, "rank_spread_ratio": rank_spread,
                 "asserted": bool(rank_spread > 1.5)}
    if rank_spread > 1.5:
        assert rho < -0.5, f"{run}: offset not monotone in rank (rho={rho:+.3f})"
    else:
        print(f"      (rank spread < 1.5x — monotonicity not asserted for this run)")

    # ---------------- derived answers: ALL from m=126 subsampling ----------------
    depth = BASE_AXIS
    derived = {
        "resampling_scheme": f"subsampling (m={m_sub}/{n_series}, no replacement), B={B}, paired",
        "argmax_axis": depth,
        "argmax_axis_note": f"{POSTLN} excluded from argmax (representation variant of L12, "
                            f"not a depth position)",
    }
    if has_postln:
        dip = sub_draws["L11"]["dataset_effrank"] - sub_draws[POSTLN]["dataset_effrank"]
        lo, hi = _pct(dip)
        derived["dip_L11_minus_L12postln"] = {
            "metric": "dataset_effrank",
            "scheme": f"subsampling m={m_sub}, no replacement (shared offset cancels)",
            "point_full_n": point["L11"]["dataset_effrank"] - point[POSTLN]["dataset_effrank"],
            "sub_mean": float(dip.mean()), "lo": lo, "hi": hi,
            "excludes_0": bool(lo > 0 or hi < 0)}

    ent_mat = np.stack([prompt_sub_draws[ln][PROMPT_METRICS[0]] for ln in depth], axis=1)
    er_mat = np.stack([sub_draws[ln]["dataset_effrank"] for ln in depth], axis=1)
    for name, mat, band in ((PROMPT_METRICS[0], ent_mat, ENTROPY_BAND),
                            ("dataset_effrank", er_mat, EFFRANK_BAND)):
        am = mat.argmax(axis=1)
        counts = np.bincount(am, minlength=len(depth))
        derived[f"argmax_{name}"] = {
            "mode_layer": depth[int(counts.argmax())],
            "mode_fraction": float(counts.max() / B),
            "distribution": {depth[i]: float(counts[i] / B)
                             for i in range(len(depth)) if counts[i] > 0},
            "band": list(band),
            "fraction_in_band": float(sum(counts[depth.index(b)] for b in band) / B)}

    flat = er_mat.max(axis=1) - er_mat.min(axis=1)
    lo, hi = _pct(flat)
    derived["effrank_range_across_layers"] = {
        "metric": f"max-min dataset_effrank over {depth[0]}..{depth[-1]}",
        "scheme": f"subsampling m={m_sub}, no replacement",
        "point_full_n": float(max(point[ln]["dataset_effrank"] for ln in depth)
                              - min(point[ln]["dataset_effrank"] for ln in depth)),
        "sub_mean": float(flat.mean()), "lo": lo, "hi": hi}

    out = {
        "provenance": {
            "run": run, "kind": kind, "tag": tag, "B": B, "seed": SEED,
            "n_series": int(n_series), "resample_unit": "series",
            "schemes": {
                "bootstrap_with_replacement": {
                    "m": int(n_series), "used_for": list(PROMPT_METRICS),
                    "validity": "plain sample means -> unbiased -> genuine 95% CIs",
                    "index_sha256_16": _hash(boot_idx)},
                "subsampling_without_replacement": {
                    "m": int(m_sub), "fraction": SUB_FRACTION,
                    "used_for": list(DATASET_METRICS) + ["all derived comparisons"],
                    "validity": "fixed distinct-n -> shared offset cancels in cross-layer "
                                "comparisons; NOT a CI for the n=200 value",
                    "index_sha256_16": _hash(sub_idx)}},
            "dtype": "float64",
            "method": "dataset-level spectra via eigenvalues of the resampled Gram submatrix; "
                      "verified equal to the committed metric functions on every layer",
            "cache_files": _cache_paths(kind, tag, has_postln),
            "postprocessing_only": "no forward passes, no probe training",
            "B_reduced": False,
            "BIAS_NOTE": BIAS_NOTE,
        },
        "layer_axis": axis,
        "point_estimates_full_n": point,
        "prompt_level_bootstrap_ci": prompt_ci,
        "dataset_level_subsampling_band": sub_band,
        "dataset_level_band_label": f"subsampling spread at m={m_sub}, not a CI for the "
                                    f"n={n_series} value",
        "gates": {f"near_full_subsample_m{m_gate}": g1,
                  "offset_monotone_in_rank": monotonic},
        "derived": derived,
    }
    d = BOOT_DIR / run
    d.mkdir(parents=True, exist_ok=True)
    (d / "bootstrap.json").write_text(json.dumps(out, indent=1))
    print(f"  [saved] {(d / 'bootstrap.json').relative_to(OUT_DIR)}")
    return out


# --------------------------------------------------------------------------- #
# figures (read the JSONs only)
# --------------------------------------------------------------------------- #

def _load(run: str) -> dict:
    p = BOOT_DIR / run / "bootstrap.json"
    assert p.exists(), f"MISSING: {p}"
    return json.loads(p.read_text())


def plot_effrank_ci(run: str) -> None:
    m = _load(run)
    axis = m["layer_axis"]
    xs = np.arange(len(axis))
    has_pln = axis[-1] == POSTLN
    k = len(axis) - 1 if has_pln else len(axis)
    msub = m["provenance"]["schemes"]["subsampling_without_replacement"]["m"]

    pt = np.array([m["point_estimates_full_n"][ln]["dataset_effrank"] for ln in axis])
    lo = np.array([m["dataset_level_subsampling_band"][ln]["dataset_effrank"]["lo"] for ln in axis])
    hi = np.array([m["dataset_level_subsampling_band"][ln]["dataset_effrank"]["hi"] for ln in axis])
    ppt = np.array([m["prompt_level_bootstrap_ci"][ln]["prompt_effrank_mean"]["point"] for ln in axis])
    plo = np.array([m["prompt_level_bootstrap_ci"][ln]["prompt_effrank_mean"]["lo"] for ln in axis])
    phi = np.array([m["prompt_level_bootstrap_ci"][ln]["prompt_effrank_mean"]["hi"] for ln in axis])

    smean = np.array([m["dataset_level_subsampling_band"][ln]["dataset_effrank"]["sub_mean"]
                      for ln in axis])

    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    # dataset-level: full-n point estimate is the PRIMARY number (solid).
    ax.plot(xs[:k], pt[:k], "o-", color="C3", lw=2,
            label="dataset-level effective rank — full n=200 (primary)")
    # the m=126 band is a SEPARATE element around its own mean: it sits below the point
    # estimate by construction, so it must NOT be drawn as an interval around it.
    ax.fill_between(xs[:k], lo[:k], hi[:k], color="C3", alpha=0.15, lw=0)
    ax.plot(xs[:k], smean[:k], ":", color="C3", lw=1.4,
            label=f"subsampling spread at m={msub} (offset low by construction, NOT a CI)")
    # prompt-level: genuine bootstrap CI (contains the point estimate)
    ax.fill_between(xs[:k], plo[:k], phi[:k], color="C0", alpha=0.20, lw=0)
    ax.plot(xs[:k], ppt[:k], "o-", color="C0",
            label="prompt-level effective rank, mean (95% bootstrap CI)")
    if has_pln:
        # postln: point as open square; its m=126 interval as a separate vertical segment
        ax.vlines(xs[k], lo[k], hi[k], color="C3", alpha=0.45, lw=3)
        ax.plot([xs[k]], [pt[k]], marker="s", mfc="none", mec="C3", ms=10, ls="none")
        ax.errorbar([xs[k]], [ppt[k]], yerr=[[ppt[k] - plo[k]], [phi[k] - ppt[k]]],
                    fmt="none", ecolor="C0", alpha=0.7)
        ax.plot([xs[k]], [ppt[k]], marker="s", mfc="none", mec="C0", ms=9, ls="none")
    ax.set_xticks(xs); ax.set_xticklabels(axis, rotation=45)
    ax.set_ylabel("effective rank")
    ax.set_title(f"{run}: effective rank by layer (postln = open square)\n"
                 f"dataset-level = full-n point + m={msub} subsampling spread; "
                 f"prompt-level = true bootstrap CI", fontsize=9)
    ax.grid(alpha=0.3); ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    out = BOOT_DIR / run / "fig_effrank_by_layer_ci.png"
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"  [saved] {out.relative_to(OUT_DIR)}")


def plot_comparison_ci() -> None:
    a, b = _load("electricity"), _load("electricity_randinit")
    axis = a["layer_axis"]
    xs = np.arange(len(axis))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 4.8))
    for ax, metric, title, yl in (
        (ax1, "prompt_entropy_norm_mean", "normalized prompt entropy", "S1 / log(min(N,D))"),
        (ax2, "prompt_effrank_mean", "prompt-level effective rank", "effective rank"),
    ):
        for m, lab, c in ((a, "pretrained", "C0"), (b, "randinit", "C1")):
            ax_ = m["layer_axis"]
            kk = len(ax_) - 1 if ax_[-1] == POSTLN else len(ax_)
            pt = np.array([m["prompt_level_bootstrap_ci"][ln][metric]["point"] for ln in ax_])
            lo = np.array([m["prompt_level_bootstrap_ci"][ln][metric]["lo"] for ln in ax_])
            hi = np.array([m["prompt_level_bootstrap_ci"][ln][metric]["hi"] for ln in ax_])
            ax.fill_between(np.arange(kk), lo[:kk], hi[:kk], color=c, alpha=0.20, lw=0)
            ax.plot(np.arange(kk), pt[:kk], "o-", color=c, label=lab)
            if ax_[-1] == POSTLN:
                ax.errorbar([kk], [pt[kk]], yerr=[[pt[kk] - lo[kk]], [hi[kk] - pt[kk]]],
                            fmt="none", ecolor=c, alpha=0.7)
                ax.plot([kk], [pt[kk]], marker="s", mfc="none", mec=c, ms=9, ls="none",
                        label=f"{lab} ({POSTLN})")
        ax.set_xticks(xs); ax.set_xticklabels(axis, rotation=45)
        ax.set_ylabel(yl); ax.set_title(title)
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    fig.suptitle("electricity: pretrained vs randinit — prompt-level metrics, "
                 f"95% series-level bootstrap CI (B={a['provenance']['B']}, unbiased means)",
                 y=1.02)
    fig.tight_layout()
    out = BOOT_DIR / "fig_pretrained_vs_randinit_electricity_ci.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {out.relative_to(OUT_DIR)}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--B", type=int, default=DEFAULT_B)
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)

    BOOT_DIR.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        for run in ("electricity", "m4"):
            plot_effrank_ci(run)
        plot_comparison_ci()
        return

    res = {run: run_resampling(run, *spec, args.B) for run, spec in RUNS.items()}

    print("\n=== GATE: paired draws (one index array per scheme, reused across layers) ===")
    for run, m in res.items():
        s = m["provenance"]["schemes"]
        print(f"  {run:>22}: boot(m={s['bootstrap_with_replacement']['m']}) "
              f"sha={s['bootstrap_with_replacement']['index_sha256_16']} | "
              f"sub(m={s['subsampling_without_replacement']['m']}) "
              f"sha={s['subsampling_without_replacement']['index_sha256_16']}")

    print("\n=== ANSWER 1: dip test — effrank(L11) - effrank(L12_postln) ===")
    for run in ("electricity", "m4"):
        d = res[run]["derived"]["dip_L11_minus_L12postln"]
        print(f"  {run:>12}: full-n point={d['point_full_n']:+8.3f} | subsampling mean="
              f"{d['sub_mean']:+8.3f}  95% [{d['lo']:+.3f}, {d['hi']:+.3f}]  "
              f"excludes 0: {d['excludes_0']}   [{d['scheme']}]")

    print("\n=== ANSWER 2: peak separation (argmax over Embed..L12; postln excluded) ===")
    for run in ("electricity", "m4"):
        d = res[run]["derived"]
        e, r = d["argmax_prompt_entropy_norm_mean"], d["argmax_dataset_effrank"]
        print(f"  [{run}]  scheme: {d['resampling_scheme']}")
        print(f"     prompt-entropy  argmax: mode={e['mode_layer']} ({e['mode_fraction']:.3f})"
              f"  fraction in {{{','.join(e['band'])}}} = {e['fraction_in_band']:.4f}")
        print(f"        dist: { {k: round(v, 3) for k, v in e['distribution'].items()} }")
        print(f"     dataset-effrank argmax: mode={r['mode_layer']} ({r['mode_fraction']:.3f})"
              f"  fraction in {{{','.join(r['band'])}}} = {r['fraction_in_band']:.4f}")
        print(f"        dist: { {k: round(v, 3) for k, v in r['distribution'].items()} }")

    print("\n=== ANSWER 3: flatness — max-min effrank across layers ===")
    for run, m in res.items():
        f = m["derived"]["effrank_range_across_layers"]
        tag = "  <-- randinit" if run.endswith("randinit") else ""
        print(f"  {run:>22}: full-n point={f['point_full_n']:8.3f} | subsampling mean="
              f"{f['sub_mean']:8.3f}  95% [{f['lo']:.3f}, {f['hi']:.3f}]{tag}")

    print()
    for run in ("electricity", "m4"):
        plot_effrank_ci(run)
    plot_comparison_ci()


if __name__ == "__main__":
    main()
