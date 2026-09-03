"""Paper figures: cosmetic regeneration from CACHED result files only (no recompute).

Double-blind rules: no personal names, no run names (the two gain versions are labeled
"Primary split (rolling-origin)" / "Alternative split"), no code flags or variable names
in any rendered text. Axis = paper axis: Embed, L1..L12, "L12 (post-LN)".
All output 300 dpi PNG to paper_figs/. Run: python -m probing.paper_figs
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "paper_figs"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 17, "axes.labelsize": 18, "xtick.labelsize": 15, "ytick.labelsize": 15,
    "legend.fontsize": 13.5, "axes.titlesize": 18, "figure.dpi": 100, "savefig.dpi": 150,
})

# Overleaf free-plan compile-timeout budget: dpi<=150, width<=2000px, file<~400KB
EXPORT_DPI = 150
MAX_WIDTH_PX = 2000

PAPER_AXIS_13 = ["Embed"] + [f"L{i}" for i in range(1, 13)]
PAPER_AXIS_14 = PAPER_AXIS_13 + ["L12 (post-LN)"]
PROBE_AXIS_12 = [f"L{i}" for i in range(1, 13)]        # probe idx k -> L(k+1)

# display names (double-blind safe)
DISP = {"m4_hourly": "M4", "monash_electricity_hourly": "Electricity",
        "uber_tlc_hourly": "Uber TLC", "wind_farms_hourly": "Wind Farms",
        "sg_carpark": "SG Carpark", "coastal_ts": "Coastal", "boom_hourly": "BOOM",
        "solar_1h": "Solar 1H"}

SOURCES = ("m4_hourly", "monash_electricity_hourly", "uber_tlc_hourly", "wind_farms_hourly")
CACHE_LOG: dict[str, list[str]] = {}


def _save(fig, name, sources):
    p = OUT / name
    dpi = min(EXPORT_DPI, MAX_WIDTH_PX / fig.get_figwidth())
    fig.savefig(p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    # pngquant-equivalent: median-cut palette quantization (lossless for layout/text/data
    # placement — only the color depth is reduced)
    from PIL import Image
    img = Image.open(p)
    img.convert("RGB").quantize(colors=256, method=Image.MEDIANCUT).save(p, optimize=True)
    CACHE_LOG[name] = sources
    kb = p.stat().st_size / 1024
    print(f"[saved] paper_figs/{name}  {Image.open(p).size[0]}px wide, {kb:.0f} KB"
          f"   <- {', '.join(sources)}")


# ------------------------------------------------------------------ masarczyk x2
def masarczyk(short, out_name):
    crit_p = f"results/repr_metrics/masarczyk_criterion/{short}/criterion.json"
    met_p = f"results/repr_metrics/{short}/metrics.json"
    c = json.load(open(RES / f"repr_metrics/masarczyk_criterion/{short}/criterion.json"))
    m = json.load(open(RES / f"repr_metrics/{short}/metrics.json"))
    rank = {r["layer"]: r["dataset_effrank"] for r in m["per_layer"]}
    axis = c["probe_axis"]                              # L1..L12 (already paper axis)
    xs = np.arange(len(axis))
    acc = np.array([c["accuracy_by_layer"][ln] for ln in axis])
    rk = np.array([rank[ln] for ln in axis])
    t95 = c["thresholds"]["0.95"]

    fig, ax1 = plt.subplots(figsize=(10, 5.6))
    ax1.axvspan(t95["tunnel_start_probe_index"] - 0.35, len(axis) - 1 + 0.35,
                color="lightgreen", alpha=0.30, label="nominal tunnel (95% rule)")
    ax1.plot(xs, acc, "o-", color="tab:blue", lw=2.5, ms=8, label="linear-probe accuracy")
    ax1.axhline(c["accuracy_final_L12"] * 0.95, ls=":", color="tab:blue", alpha=0.7, lw=2,
                label="95% of final-layer accuracy")
    ax1.set_ylabel("probe accuracy", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(list(xs) + [len(axis)])
    ax1.set_xticklabels(axis + ["L12 (post-LN)"], rotation=45)
    ax2 = ax1.twinx()
    ax2.plot(xs, rk, "s--", color="tab:red", lw=2.5, ms=8, label="effective rank")
    ax2.plot([len(axis)], [rank["L12_postln"]], marker="s", mfc="none", mec="tab:red",
             ms=13, mew=2.5, ls="none", label="effective rank, post-LN")
    ax2.set_ylabel("effective rank", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    # legend below the axes (outside the data area; bbox_inches="tight" includes it)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.01),
               ncol=3, framealpha=0.95, fontsize=13, handletextpad=0.4, columnspacing=0.9)
    _save(fig, out_name, [crit_p, met_p])


# ------------------------------------------------------- distance-vs-gain combined
def distance_combined():
    srcs = ["results/distance/ladder/join_extended_v3_rolling.csv",
            "results/distance/ladder/join_extended_v2.csv"]
    headers = ["Primary split (rolling-origin), ρ = −0.41",
               "Alternative split, ρ = −0.10"]
    colors = {s: f"C{i}" for i, s in enumerate(SOURCES)}
    fig, axs = plt.subplots(1, 2, figsize=(16, 6.4), sharey=False)
    for ax, path, hdr in zip(axs, srcs, headers):
        df = pd.read_csv(ROOT / path)
        for src in SOURCES:
            for tier, marker, size in (("near", "o", 110), ("far", "^", 140)):
                sub = df[(df.source == src) & (df.tier == tier)]
                if sub.empty:
                    continue
                ax.scatter(sub.distance, sub.gain, marker=marker, s=size, color=colors[src],
                           alpha=0.85, edgecolor="black", linewidth=0.5)
        ax.axhline(0, color="gray", ls=":", lw=1.5)
        ax.set_title(hdr)
        ax.set_xlabel("catch22 energy distance (source → target)")
    axs[0].set_ylabel("relative gain over L12 (%)")
    # legend: sources by color + tier by marker shape (corpus-seen / corpus-unseen)
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", ls="none", color=colors[s], mec="black",
                      ms=11, label=DISP[s]) for s in SOURCES]
    handles += [Line2D([0], [0], marker="o", ls="none", color="0.45", mec="black", ms=11,
                       label="corpus-seen target (circles)"),
                Line2D([0], [0], marker="^", ls="none", color="0.45", mec="black", ms=12,
                       label="corpus-unseen target (triangles)")]
    axs[0].legend(handles=handles, loc="upper right", ncol=2, framealpha=0.92,
                  handletextpad=0.3, columnspacing=0.8)
    for ax in axs:
        ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_distance_vs_gain_combined.png", srcs)


# ---------------------------------------------------------------------- forest
def forest():
    p = "results/uea/perdataset_summary.json"
    d = json.load(open(RES / "uea/perdataset_summary.json"))["datasets"]
    rows = [(k, v["id_late_drop_band"]) for k, v in d.items() if v and not v["saturated"]]
    rows.sort(key=lambda t: t[1]["point"])
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ys = np.arange(len(rows))
    for y, (name, ld) in zip(ys, rows):
        signif = ld["excludes_0"]
        ax.plot([ld["lo"], ld["hi"]], [y, y], color="C2" if signif else "0.55", lw=5, alpha=0.5)
        ax.plot([ld["point"]], [y], "o", color="black", mfc="white", ms=10, mew=2)
    ax.axvline(0, color="gray", ls="--", lw=1.5)
    ax.set_yticks(ys)
    ax.set_yticklabels([name for name, _ in rows])
    ax.set_xlabel("mean(L4:L9) − L12 late-layer deficit (95% paired CI)")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    _save(fig, "forest.png", [p])


# --------------------------------------------------------------------- dropoff
def dropoff():
    p = "results/phase0_trio/id_probing_summary.json"
    d = json.load(open(RES / "phase0_trio/id_probing_summary.json"))
    xs = np.arange(12)

    def rel_drop(a):
        a = np.asarray(a, float)
        return (a - a.max()) / a.max()

    fig, ax = plt.subplots(figsize=(10.5, 6))
    for name, acc in d["uea_classification_reference"].items():
        ax.plot(xs, rel_drop(acc), color="steelblue", alpha=0.35, lw=1.6,
                label="UEA classification (n=6)" if name == list(
                    d["uea_classification_reference"])[0] else None)
    style = {"m4_hourly":                 ("M4 (cross-series)", "M4 (register-token pooling)", "#d62728", 1.0),
             "monash_electricity_hourly": ("Electricity", "Electricity (register-token pooling)", "#ff7f0e", 1.0),
             "solar_1h":                  ("Solar 1H (label pathology)", None, "#8c564b", 0.45)}
    for tag, (lab_c, lab_r, color, alpha) in style.items():
        res = d["id_datasets"][tag]["poolings"]
        lw = 2.8 if alpha == 1.0 else 1.4
        ax.plot(xs, rel_drop(res["content"]["binned_accuracy"]), "o-", color=color,
                alpha=alpha, lw=lw, ms=7, label=lab_c)
        if lab_r:
            ax.plot(xs, rel_drop(res["reg"]["binned_accuracy"]), "s--", color=color,
                    alpha=alpha * 0.9, lw=lw * 0.7, ms=6, label=lab_r)
    ax.axhline(0, color="gray", ls=":", lw=1.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(PROBE_AXIS_12, rotation=45)
    ax.set_ylabel("relative drop from own peak")
    ax.legend(loc="lower right", ncol=2, framealpha=0.95, fontsize=13, handletextpad=0.4)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "dropoff.png", [p])


# ------------------------------------------------------------------- cka matrix
def cka():
    p = "results/repr_metrics/cka/electricity/cka.json"
    m = json.load(open(RES / "repr_metrics/cka/electricity/cka.json"))
    C13 = np.asarray(m["cka_matrix_13tap"])
    C14 = np.asarray(m["cka_matrix"])
    fig, axs = plt.subplots(1, 2, figsize=(15.5, 7))
    for ax, Cm, labels in ((axs[0], C13, PAPER_AXIS_13), (axs[1], C14, PAPER_AXIS_14)):
        im = ax.imshow(Cm, vmin=0, vmax=1, cmap="viridis")
        if len(labels) == 14:
            ax.axhline(12.5, color="white", lw=2.5)
            ax.axvline(12.5, color="white", lw=2.5)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=60, fontsize=12)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=12)
        fig.colorbar(im, ax=ax, shrink=0.75)
    fig.tight_layout()
    _save(fig, "fig_cka_matrix.png", [p])


# ------------------------------------------------------------- common pooling
def common_pooling():
    p = "results/phase0_trio/quantile_loss/pooling_comparison/common_pooling_recompute.json"
    d = json.load(open(RES / "phase0_trio/quantile_loss/pooling_comparison/common_pooling_recompute.json"))
    xs = np.arange(12)
    c = d["poolings"]["content"]
    # (key, legend label, higher_is_better, color, linestyle, marker)
    METRICS = [("binned_accuracy", "Binned accuracy", True,  "#1f77b4", "-",  "o"),
               ("ridge_r2",        "Ridge R²",        True,  "#2ca02c", "--", "s"),
               ("quantile_loss",   "Quantile loss",   False, "#d62728", "-",  "^"),
               ("mase_context",    "MASE",            False, "#9467bd", "--", "D")]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    for key, lab, hib, color, ls, marker in METRICS:
        v = np.asarray(c[key], np.float64)
        norm = (v - v.min()) / (v.max() - v.min())      # within-objective [0,1]
        if not hib:
            norm = 1.0 - norm                           # 1 = best layer, 0 = worst
        ax.plot(xs, norm, ls, marker=marker, color=color, lw=2.5, ms=7, label=lab)
        k = int(norm.argmax())
        ax.plot([k], [norm[k]], "k*", ms=17, zorder=5,
                label="best layer" if key == "binned_accuracy" else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(PROBE_AXIS_12, rotation=45)
    ax.set_xlabel("layer")
    ax.set_ylabel("normalized score (1 = best layer)")
    ax.set_ylim(-0.05, 1.12)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=3,
               framealpha=0.95, fontsize=13, handletextpad=0.4, columnspacing=0.9)
    _save(fig, "fig_common_pooling.png", [p])


# ------------------------------------------- content-pooling flatness (appendix)
def content_pooling_flatness():
    p = "results/phase0_trio/quantile_loss/pooling_comparison/common_pooling_recompute.json"
    c = json.load(open(RES / "phase0_trio/quantile_loss/pooling_comparison/common_pooling_recompute.json"))["poolings"]["content"]
    xs = np.arange(12)
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.axhspan(95, 105, color="0.85", alpha=0.6, zorder=0, label="±5% of final layer")
    ax.axhline(100, color="gray", ls="--", lw=1.5, zorder=1)
    for key, lab, mirror, color, ls, marker in (
            ("binned_accuracy", "Binned accuracy", False, "#1f77b4", "-", "o"),
            ("ridge_r2", "Ridge R²", False, "#2ca02c", "--", "s"),
            ("quantile_loss", "Quantile loss", True, "#d62728", "-", "^")):
        v = np.asarray(c[key], np.float64)
        pct = 100 * v / v[-1]
        if mirror:                     # lower-is-better: mirror about 100 so above = better
            pct = 200 - pct
        ax.plot(xs, pct, ls, marker=marker, color=color, lw=2.5, ms=7, label=lab)
        if key == "ridge_r2":          # annotate the two points clipped below the y-range
            for k in (0, 1):
                ax.annotate(f"{pct[k]:.0f}", (k, 80.6), ha="center", va="bottom",
                            fontsize=12, color=color,
                            arrowprops=None)
                ax.plot([k], [80.4], marker="v", color=color, ms=8, clip_on=False)
    ax.set_ylim(80, 110)
    ax.set_xticks(xs)
    ax.set_xticklabels(PROBE_AXIS_12, rotation=45)
    ax.set_xlabel("layer")
    ax.set_ylabel("% of final layer\n(above 100 = better)")
    ax.legend(loc="lower right", framealpha=0.95, fontsize=12.5)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "appendix_content_pooling_flatness.png", [p])


# ------------------------------------------------------------------ native head
def native_head_elec():
    p = "results/repr_metrics/native_head/electricity/native_head.json"
    p2 = "results/phase0_trio/id_probing_summary.json"
    m = json.load(open(RES / "repr_metrics/native_head/electricity/native_head.json"))
    axis = m["layer_axis"]                              # Embed..L12 + L12_postln
    xs = np.arange(len(axis))
    q9 = [m["per_layer"][ln]["q9_loss"] for ln in axis]
    fig, ax = plt.subplots(figsize=(10.5, 6))
    ax.plot(xs[:-1], q9[:-1], "o-", color="C3", lw=2.5, ms=8,
            label="frozen native head on layer-k states")
    ax.plot([xs[-1]], [q9[-1]], marker="s", mfc="none", mec="C3", ms=13, mew=2.5, ls="none",
            label="post-LN (native input)")
    ax.axhline(m["native"]["q9_loss"], ls="--", color="black", lw=1.8,
               label="native forecast baseline")
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ax.set_xticklabels(PAPER_AXIS_14, rotation=45)
    ax.set_ylabel("q9 quantile loss (arcsinh scale)")
    summ = json.load(open(RES / "phase0_trio/id_probing_summary.json"))
    acc = summ["id_datasets"]["monash_electricity_hourly"]["poolings"]["content"]["binned_accuracy"]
    ax2 = ax.twinx()
    ax2.plot(xs[1:13], acc, "-", color="grey", alpha=0.35, lw=2.5,
             label="linear-probe accuracy (right axis)")
    ax2.set_ylabel("probe accuracy", color="grey")
    ax2.tick_params(axis="y", labelcolor="grey")
    lo, hi = ax.get_ylim()                     # headroom (log axis: multiplicative)
    ax.set_ylim(lo, hi * (hi / lo) ** 0.40)
    lo2, hi2 = ax2.get_ylim()
    ax2.set_ylim(lo2, hi2 + 0.40 * (hi2 - lo2))
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", ncol=2, framealpha=0.95,
              fontsize=13, handletextpad=0.4, columnspacing=0.9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    _save(fig, "fig_native_head_elec.png", [p, p2])


def main():
    masarczyk("electricity", "fig_masarczyk_overlay_elec.png")
    masarczyk("m4", "fig_masarczyk_overlay_m4.png")
    distance_combined()
    forest()
    dropoff()
    cka()
    native_head_elec()
    common_pooling()
    content_pooling_flatness()
    (OUT / "_cache_sources.json").write_text(json.dumps(CACHE_LOG, indent=1))


if __name__ == "__main__":
    main()
