"""Electricity common-pooling test: hold the representation fixed (mean-pooled content
tokens, 13 depths Embed+L1..L12), vary only the probe objective, and ask whether the
objectives still imply different best Chronos-2 layers. Post-hoc, no GPU, no UEA — reads
the committed results/<set>/id_probing_summary.json. Run: python -m experiments.electricity_common_pooling
"""
from __future__ import annotations
import csv, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.config import ID_OUT_DIR, NUM_LAYERS, CACHE_DIR

TAG = "monash_electricity_hourly"
# (label, dotted path into the dataset block, higher_is_better). Forecasting loss is plotted
# as mean_pinball_loss (per-element average) — NOT called WQL, since the repo does not divide
# by the target-magnitude denominator of normalized weighted quantile loss.
OBJECTIVES = [
    ("Binned accuracy",   "poolings.content.binned_accuracy",   True),
    ("Ridge R²",     "poolings.content.ridge_r2",          True),
    ("Mean pinball loss", "poolings.content.mean_pinball_loss", False),
    ("MASE",              "mase.poolings.content",              False),
]
# Retained in the summary only (NOT plotted): raw Chronos-2 quantile loss. Within q21 it is
# exactly 2*Q = 42x mean_pinball_loss, so its min-max-normalized curve and winning layer are
# identical to "Mean pinball loss"; only its raw best-vs-final gap is 42x larger.
RAW_QLOSS_PATH = "poolings.content.quantile_loss"
COLORS = {"Binned accuracy": "#1f77b4", "Ridge R²": "#2ca02c",
          "Mean pinball loss": "#d62728", "MASE": "#9467bd"}


def _dig(d, dotted):
    for k in dotted.split("."):
        d = d[k]
    return d


def _relative(curve, higher):
    """Unitless within-objective curve in [0,1]: 1=best layer, 0=worst. Zero-range safe."""
    a = np.asarray(curve, float)
    rng = a.max() - a.min()
    if rng == 0:
        return np.full_like(a, 0.5)
    return (a - a.min()) / rng if higher else (a.max() - a) / rng


def _best_stats(curve, higher, last):
    a = np.asarray(curve, float)
    bi = int(a.argmax() if higher else a.argmin())
    best, final = float(a[bi]), float(a[last])
    gap = (best - final) if higher else (final - best)  # >=0: how much best beats final
    return bi, best, final, gap


def main():
    summ_path = ID_OUT_DIR / "id_probing_summary.json"
    if not summ_path.exists():
        raise FileNotFoundError(f"{summ_path} missing — run experiments.run_id_forecasting first")
    e = json.load(open(summ_path))["id_datasets"][TAG]
    n = len(e["poolings"]["content"]["binned_accuracy"])
    if n != NUM_LAYERS:
        raise RuntimeError(f"summary has {n} depths, expected NUM_LAYERS={NUM_LAYERS} "
                           "— regenerate id_probing_summary.json under the current layer scheme")
    last, xs = n - 1, np.arange(n)
    xlabels = ["Embed"] + [str(i) for i in range(1, n)]

    print("=== VALIDATION ===")
    for split in ("train", "test"):
        d = np.load(CACHE_DIR / f"IDF_{TAG}__{split}__clean__content.npz", allow_pickle=True)
        ls = sorted(int(k.split("_")[1]) for k in d.files if k.startswith("layer_"))
        if ls != list(range(NUM_LAYERS)):
            raise RuntimeError(f"{split} content cache depths {ls} != 0..{NUM_LAYERS-1}")
        print(f"[ok] {split} content cache: depths {ls[0]}..{ls[-1]}, rows={len(d['y'])}, d={d['layer_0'].shape[1]}")
    for label, path, _ in OBJECTIVES:
        assert len(_dig(e, path)) == n, f"{label}: wrong length"
    print("[ok] 4 objectives x 13 layers, all on shared content pooling; no UEA/REG/fslot touched\n")

    out_dir = ID_OUT_DIR / "common_pooling" / "electricity"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, rel, summary = [], {}, {
        "dataset": TAG,
        "representation": "mean-pooled content tokens (13 depths: Embed + L1..L12)",
        "native_mase_reference": e["mase"]["native_mase"], "objectives": {}}
    print("=== RAW SUMMARY ===")
    print(f"{'objective':<18} | {'best L':>6} | {'best':>9} | {'final':>9} | {'gap':>8} | dir")
    for label, path, higher in OBJECTIVES:
        curve = [float(x) for x in _dig(e, path)]
        bi, best, final, gap = _best_stats(curve, higher, last)
        rel[label] = _relative(curve, higher)
        print(f"{label:<18} | {xlabels[bi]:>6} | {best:>9.4f} | {final:>9.4f} | {gap:>8.4f} | "
              f"{'higher' if higher else 'lower'}")
        summary["objectives"][label] = {
            "higher_is_better": higher, "raw_by_layer": curve,
            "best_layer": bi, "best_layer_label": xlabels[bi], "best_value": best,
            "final_layer": last, "final_value": final, "gap_best_minus_final": gap}
        for L in range(n):
            rows.append({"objective": label, "layer": L, "layer_label": xlabels[L],
                         "raw": curve[L], "relative": float(rel[label][L]),
                         "higher_is_better": higher, "is_best": L == bi, "is_final": L == last})

    # Retain BOTH forecasting-loss scales in the summary. Raw quantile_loss = Chronos-2's
    # training objective (sum over Q); mean_pinball_loss = its per-element average. Within q21:
    # raw = 2*Q * pinball = 42 * pinball -> identical normalized curve + winning layer, but a
    # 42x-larger raw best-vs-final gap. Not plotted (would overlap the pinball curve).
    raw_ql = [float(x) for x in _dig(e, RAW_QLOSS_PATH)]
    rbi, rbest, rfinal, rgap = _best_stats(raw_ql, False, last)
    pin_gap = summary["objectives"]["Mean pinball loss"]["gap_best_minus_final"]
    summary["objectives"]["Mean pinball loss"]["raw_quantile_loss"] = {
        "note": "Chronos-2 raw quantile loss (sum over Q). raw = 2*Q * mean_pinball_loss; "
                "here Q=21 -> factor 42. Same normalized curve and winning layer; not plotted.",
        "raw_by_layer": raw_ql, "best_layer": rbi, "best_layer_label": xlabels[rbi],
        "best_value": rbest, "final_value": rfinal, "gap_best_minus_final": rgap,
        "gap_ratio_raw_over_pinball": (rgap / pin_gap) if pin_gap else None}
    for L in range(n):                # complete the raw table too (marked not-plotted)
        rows.append({"objective": "Quantile loss (raw, not plotted)", "layer": L,
                     "layer_label": xlabels[L], "raw": raw_ql[L],
                     "relative": float(_relative(raw_ql, False)[L]),
                     "higher_is_better": False, "is_best": L == rbi, "is_final": L == last})
    print(f"\nnative Chronos-2 MASE (reference, not a probe): {e['mase']['native_mase']:.4f}")
    print(f"raw quantile_loss best-vs-final gap = {rgap:.4f} = {rgap/pin_gap:.1f}x the "
          f"mean-pinball gap ({pin_gap:.4f}); same winning layer L{rbi}")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for label, _, _ in OBJECTIVES:
        y = rel[label]
        ax.plot(xs, y, marker="o", ms=4, lw=2.0, color=COLORS[label], label=label)
        bi = int(np.argmax(y))
        ax.plot(bi, y[bi], marker="*", ms=14, color=COLORS[label],
                markeredgecolor="k", markeredgewidth=0.5, zorder=5)
    ax.set_xticks(xs); ax.set_xticklabels(xlabels)
    ax.set_xlabel("representation")
    ax.set_ylabel("within-objective relative performance\n(1 = best layer, 0 = worst)")
    ax.set_ylim(-0.05, 1.08); ax.grid(alpha=0.3); ax.legend(fontsize=9, loc="lower center", ncol=2)
    ax.set_title("Electricity common-pooling test\n"
                 "All probes use the same mean-pooled content-token representation", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "electricity_common_pooling.png", dpi=140, bbox_inches="tight"); plt.close(fig)

    with open(out_dir / "electricity_common_pooling_raw.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    json.dump(summary, open(out_dir / "electricity_common_pooling_summary.json", "w"), indent=2)
    for name in ("electricity_common_pooling.png", "electricity_common_pooling_raw.csv",
                 "electricity_common_pooling_summary.json"):
        print(f"[saved] {out_dir / name}")


if __name__ == "__main__":
    main()
