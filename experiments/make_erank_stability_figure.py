"""Appendix figure: stability of the layerwise effective-rank estimates under subsampling.

POST-HOC PLOTTING ONLY. Reads the committed spectral records and re-plots them; it never loads
Chronos-2, never touches a feature cache, and never recomputes an effective rank. Safe on a login
node (a few seconds of CPU, numpy + matplotlib only — no torch import anywhere in its chain).

Input (written by experiments/run_spectral.py --readout fslot):
    results/ext_v4_future_tokens/spectral/spectral__<tag>__fslot__probe_input__train.json

Each record carries, per representation point, the full-sample effective rank at N stacked
forecast-slot rows AND the subsampling distribution used as the stability diagnostic:
`probing.spectral_metrics.subsample_metrics` draws B=200 subsamples of m=round(0.8*N) DISTINCT
rows (WITHOUT replacement, seed 0) and recomputes the metric on each. The shaded band below is
the 2.5th-97.5th percentile range over those 200 draws — a SUBSAMPLE / STABILITY INTERVAL, not
a confidence interval: it describes variability at the reduced size m within one fixed row
sample, and the rows are not independent (K slots per window, windows nested in series).

PEAK-STABILITY ANNOTATION. run_spectral calls subsample_metrics once per representation point
with a CONSTANT seed, and every point shares the same (N, m). np.random.default_rng(seed).choice
therefore emits the same sequence of index sets at every depth, so draw i is the SAME m rows at
every point. That pairing is what makes a per-draw argmax over points meaningful: the annotation
counts how many of the 200 subsamples recover the full-sample peak point. It is a resampling
diagnostic on ONE representation set — NOT 200 independent training runs.

Usage (repo root):
    python -m experiments.make_erank_stability_figure                # paper 2x2 (PDF + PNG)
    python -m experiments.make_erank_stability_figure --diagnostic   # + the internal 2x4 panel
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.config import REPO_ROOT

SPEC_DIR = REPO_ROOT / "results" / "ext_v4_future_tokens" / "spectral"
FIG_DIR = SPEC_DIR / "figures"
STEM = "erank_stability_{layout}__fslot__probe_input__train"
DIAG_STEM = "erank_stability_diagnostic_2x4__fslot__probe_input__train"

TAGS = {"monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber",
        "m4_hourly": "M4", "wind_farms_hourly": "WindFarms"}

# Paper labels. The final norm is T5-style RMSNorm (encoder.final_layer_norm), so the last point
# is L12+RMS here even though the ext_v4 filenames/records spell it "L12+LN".
LABELS = ["Emb"] + [f"L{i}" for i in range(1, 13)] + ["L12+RMS"]

CURVE, BAND = "#5E3C99", "#BFA8E8"

PAPER_RC = {"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 10,
            "xtick.labelsize": 8, "ytick.labelsize": 8.5, "axes.linewidth": 0.8,
            "xtick.major.width": 0.7, "ytick.major.width": 0.7, "lines.linewidth": 1.4,
            "pdf.fonttype": 42, "ps.fonttype": 42}          # TrueType: camera-ready safe


def load(tag: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """(full-sample curve, (n_points, B) subsample draws, metadata) for one dataset."""
    path = SPEC_DIR / f"spectral__{tag}__fslot__probe_input__train.json"
    if not path.exists():
        raise FileNotFoundError(f"missing spectral record {path} — run "
                                "`python -m experiments.run_spectral --readout fslot` first")
    rec = json.load(open(path))
    pts = rec["layers"]
    full = np.array([p["effective_rank"] for p in pts], dtype=np.float64)
    draws = np.array([p["subsample_dists"]["effective_rank"]["values"] for p in pts],
                     dtype=np.float64)
    if len(full) != len(LABELS):
        raise ValueError(f"{path.name}: {len(full)} representation points, expected {len(LABELS)}")
    sub = pts[0]["subsample_dists"]["effective_rank"]
    meta = {"N": rec["sample_size"], "n_available": rec["n_available"], "m": sub["subsample_size"],
            "B": draws.shape[1], "split": rec["split"], "d": pts[0]["feature_dim"]}
    return full, draws, meta


def summarize(full: np.ndarray, draws: np.ndarray) -> dict:
    """Stability numbers used for the annotation and the printed verification table."""
    lo, hi = np.percentile(draws, 2.5, axis=1), np.percentile(draws, 97.5, axis=1)
    sub_mean = draws.mean(axis=1)
    peak = int(full.argmax())
    return {"lo": lo, "hi": hi, "sub_mean": sub_mean, "peak": peak,
            # draws are PAIRED across points (same seed, same (N, m)) -> per-draw argmax is valid
            "peak_hits": int((draws.argmax(axis=0) == peak).sum()),
            "max_halfwidth_pct": float((100.0 * (hi - lo) / 2.0 / sub_mean).max()),
            "max_mean_gap_pct": float((100.0 * np.abs(full - sub_mean) / full).max())}


def make_paper_figure(data: dict, meta: dict, layout: str = "2x2", dpi: int = 400):
    """Manuscript figure: full-sample curve + subsample interval only. `layout` picks the panel
    grid; both variants are authored at (or near) NeurIPS \\textwidth so inclusion needs little or
    no rescaling."""
    grid = {"2x2": ((2, 2), (7.0, 5.7)), "1x4": ((1, 4), (10.0, 3.2))}
    if layout not in grid:
        raise ValueError(f"unknown layout {layout!r}; choose from {sorted(grid)}")
    (nr, nc), figsize = grid[layout]
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(nr, nc, figsize=figsize, layout="constrained", squeeze=False)
        fig.get_layout_engine().set(h_pad=0.05, w_pad=0.06, hspace=0.10, wspace=0.09)
        x = np.arange(len(LABELS))
        for ax, (tag, short) in zip(axes.ravel(), TAGS.items()):
            full, s = data[tag]["full"], data[tag]["stats"]
            ax.fill_between(x, s["lo"], s["hi"], color=BAND, alpha=0.9, lw=0,
                            label=f"95% subsample interval ($m={meta['m']}$)")
            ax.plot(x, full, "-o", ms=3.0, color=CURVE, mfc=CURVE, mec=CURVE,
                    label=f"Full-sample effective rank ($N={meta['N']}$)")
            peak = s["peak"]
            ax.plot([peak], [full[peak]], "*", ms=8.0, color=CURVE, mec="white", mew=0.5, zorder=5)
            # "(200/200)" = subsamples whose peak matches the full-sample peak; spelled out in the caption
            ax.text(0.03, 0.96, f"Peak: {LABELS[peak]} ({s['peak_hits']}/{meta['B']})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color="0.45")
            ax.set_title(short, fontweight="bold")
            ax.set_ylim(0, full.max() * 1.22)                 # headroom for the annotation
            ax.set_xticks(x)
            # narrow 1x4 panels cannot fit 14 rotated labels: keep every tick, label alternates
            last = len(LABELS) - 1
            shown = LABELS if nc < 4 else [    # 1x4 panels are narrow: label alternates
                t if (i == last or (i % 2 == 0 and i != last - 1)) else ""
                for i, t in enumerate(LABELS)]
            ax.set_xticklabels(shown, rotation=45, ha="right")
            ax.tick_params(length=2.5, pad=1.5)
            ax.grid(axis="y", alpha=0.18, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("Effective rank")
        for ax in axes[-1, :]:                            # supxlabel would collide with the
            ax.set_xlabel("Representation point")          # outside legend under constrained_layout
        h, l = axes[0, 0].get_legend_handles_labels()
        fig.legend(h[::-1], l[::-1], loc="outside lower center", ncol=2, frameon=False,
                   handlelength=1.8, handletextpad=0.5, columnspacing=1.8,
                   borderpad=0.2, borderaxespad=0.35)
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        stem = STEM.format(layout=layout)
        pdf, png = FIG_DIR / f"{stem}.pdf", FIG_DIR / f"{stem}.png"
        fig.savefig(pdf)                                      # vector, never rasterized
        fig.savefig(png, dpi=dpi)
        plt.close(fig)
    return pdf, png


def make_diagnostic_figure(data: dict, meta: dict, dpi: int = 200):
    """Internal-only 2x4: the two stability axes (sampling noise vs size sensitivity) kept out of
    the paper figure. Not for the manuscript."""
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.2), sharex="col")
    x = np.arange(len(LABELS))
    for col, (tag, short) in enumerate(TAGS.items()):
        full, s = data[tag]["full"], data[tag]["stats"]
        ax = axes[0, col]
        ax.fill_between(x, s["lo"], s["hi"], color=BAND, alpha=0.45, lw=0,
                        label=f"95% subsample interval (m={meta['m']})")
        ax.plot(x, full, "-o", ms=3.5, color=CURVE, label=f"effective rank (N={meta['N']})")
        ax.plot(x, s["sub_mean"], "--", lw=1.1, color="k", alpha=0.55,
                label=f"subsample mean (m={meta['m']})")
        ax.plot([s["peak"]], [full[s["peak"]]], "*", ms=15, color="k", zorder=5)
        ax.set_ylim(0, full.max() * 1.34)
        ax.set_title(short, fontsize=12, fontweight="bold")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.92)
        if col == 0:
            ax.set_ylabel("effective rank $e^{H}$\n(fslot, train)", fontsize=10)
        ax = axes[1, col]
        ax.plot(x, 100.0 * (s["hi"] - s["lo"]) / 2.0 / s["sub_mean"], "-o", ms=3.5, color="tab:red",
                label="sampling noise: 95% half-width (% of mean)")
        ax.plot(x, 100.0 * (full - s["sub_mean"]) / full, "-s", ms=3.5, color="tab:gray",
                label="size sensitivity: (N $-$ 0.8N) / N  (%)")
        ax.axhline(0, color="k", lw=0.6, alpha=0.4)
        ax.set_ylim(-0.7, 7.6)
        ax.set_xticks(x); ax.set_xticklabels(LABELS, rotation=90, fontsize=8)
        ax.grid(alpha=0.25); ax.legend(fontsize=7.5, loc="upper left")
        ax.set_xlabel("fslot readout point", fontsize=9)
        if col == 0:
            ax.set_ylabel("relative spread (% of estimate)", fontsize=10)
    fig.suptitle("Effective-rank stability diagnostic (internal) — ID datasets, pretrained "
                 f"backbone, {meta['split']} split", fontsize=12.5, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = FIG_DIR / f"{DIAG_STEM}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--layout", default="2x2", choices=("2x2", "1x4", "both"),
                    help="panel grid for the manuscript figure (default: 2x2)")
    ap.add_argument("--diagnostic", action="store_true",
                    help="also write the internal 2x4 noise / size-sensitivity panel")
    ap.add_argument("--dpi", type=int, default=400, help="PNG resolution (the PDF stays vector)")
    args = ap.parse_args()

    data, meta = {}, None
    for tag in TAGS:
        full, draws, m = load(tag)
        if meta is None:
            meta = m
        elif (m["N"], m["m"], m["B"]) != (meta["N"], meta["m"], meta["B"]):
            raise ValueError(f"{tag}: sampling protocol {m} differs from {meta} — one shared "
                             "legend would misdescribe at least one panel")
        data[tag] = {"full": full, "stats": summarize(full, draws)}

    print(f"[erank-stability] N={meta['N']} of {meta['n_available']} rows (d={meta['d']}), "
          f"split={meta['split']}, B={meta['B']} subsamples, m={meta['m']} "
          f"({meta['m'] / meta['N']:.0%}), without replacement")
    print(f"{'dataset':<13}{'peak':>8}{'peak held':>12}{'max 95% half-w':>17}"
          f"{'max |full-sub mean|':>22}")
    for tag, short in TAGS.items():
        s = data[tag]["stats"]
        held = "{}/{}".format(s["peak_hits"], meta["B"])
        print(f"{short:<13}{LABELS[s['peak']]:>8}{held:>12}"
              f"{s['max_halfwidth_pct']:>16.2f}%{s['max_mean_gap_pct']:>21.2f}%")

    for lay in (("2x2", "1x4") if args.layout == "both" else (args.layout,)):
        pdf, png = make_paper_figure(data, meta, layout=lay, dpi=args.dpi)
        print(f"[paper {lay}] {pdf}\n[paper {lay}] {png}")
    if args.diagnostic:
        print(f"[diagnostic] {make_diagnostic_figure(data, meta)}")


if __name__ == "__main__":
    main()
