"""Masarczyk-criterion overlay: ID probe accuracy vs measured effective rank.

Post-processing ONLY — reads committed result JSONs; zero probe training, zero forward
passes.

Masarczyk et al. (2023) define the tunnel operationally: it STARTS at the first layer
whose in-distribution linear-probe accuracy reaches 95% (strict: 98%) of the FINAL
layer's accuracy, and INSIDE the tunnel representation rank should collapse while
accuracy stays flat. This module applies that criterion to Chronos-2 and overlays our
measured dataset-level effective rank on the nominal tunnel region.

Inputs (paths printed as a gate):
  results/phase0_trio/id_probing_summary.json
      -> id_datasets[<tag>].poolings.content.binned_accuracy  (12 probe points)
  results/repr_metrics/{electricity,m4}/metrics.json
      -> per_layer[*].dataset_effrank  (Embed + L1..L12 + L12_postln)

LAYER ALIGNMENT (fixed, not inferred):
  probe layer_k (k = 0..11) = block output k+1 = repr-metrics label L(k+1).
  Embed has NO probe point. The overlay x-axis is L1..L12; L12_postln is appended for
  RANK ONLY (it is a representation variant, not a separate probe point).
  Consequently "acc(L12)" — the criterion's final-layer accuracy — is probe index 11.

Outputs:
  results/repr_metrics/masarczyk_criterion/{dataset}/criterion.json
  results/repr_metrics/masarczyk_criterion/{dataset}/fig_masarczyk_overlay.png
The figure regenerates from criterion.json + metrics.json alone (--figures-only).

Run:  python -m probing.repr_metrics_masarczyk [--figures-only]
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
from probing.repr_metrics import RM_DIR

PROBE_JSON = OUT_DIR / "phase0_trio" / "id_probing_summary.json"
CRIT_DIR = RM_DIR / "masarczyk_criterion"
POSTLN = "L12_postln"
THRESHOLDS = (0.95, 0.98)

# output name -> probe tag in the summary JSON
DATASETS = {"electricity": "monash_electricity_hourly", "m4": "m4_hourly"}

# sanity anchors from the task spec (peak, final); tolerance 0.005
ANCHORS = {"m4": (0.7616, 0.7049), "electricity": (0.4133, 0.388)}

# probe index k -> repr label L(k+1)
PROBE_AXIS = [f"L{k + 1}" for k in range(12)]


def _probe_index(label: str) -> int:
    return PROBE_AXIS.index(label)


# --------------------------------------------------------------------------- #
# loading + gates
# --------------------------------------------------------------------------- #

def load_accuracy(short: str, verbose: bool = True) -> np.ndarray:
    assert PROBE_JSON.exists(), f"MISSING INPUT: {PROBE_JSON}"
    d = json.loads(PROBE_JSON.read_text())
    tag = DATASETS[short]
    acc = np.array(d["id_datasets"][tag]["poolings"]["content"]["binned_accuracy"], dtype=float)
    assert acc.size == 12, f"expected 12 probe points, got {acc.size}"
    if verbose:
        print(f"  [input] {PROBE_JSON}")
        print(f"  [{short} / {tag}] binned_accuracy (content), probe idx 0..11:")
        print("      ", [round(float(a), 4) for a in acc])
        print(f"       peak={acc.max():.4f} @ probe idx {int(acc.argmax())} "
              f"(= repr {PROBE_AXIS[int(acc.argmax())]}) | final acc(L12) = probe idx 11 "
              f"= {acc[11]:.4f}")
    # anchor gate
    pk_exp, fin_exp = ANCHORS[short]
    dpk, dfin = abs(acc.max() - pk_exp), abs(acc[11] - fin_exp)
    assert dpk <= 0.005, f"{short}: peak {acc.max():.4f} vs anchor {pk_exp} (|d|={dpk:.4f} > 0.005)"
    assert dfin <= 0.005, f"{short}: final {acc[11]:.4f} vs anchor {fin_exp} (|d|={dfin:.4f} > 0.005)"
    if verbose:
        print(f"       anchor gate OK (|dpeak|={dpk:.4f}, |dfinal|={dfin:.4f}, tol 0.005)")
    return acc


def load_rank(short: str) -> tuple[dict[str, float], Path]:
    p = RM_DIR / short / "metrics.json"
    assert p.exists(), f"MISSING INPUT: {p}"
    m = json.loads(p.read_text())
    rank = {r["layer"]: float(r["dataset_effrank"]) for r in m["per_layer"]}
    for ln in PROBE_AXIS:
        assert ln in rank, f"{p}: missing rank for {ln}"
    return rank, p


def print_mapping_table() -> None:
    print("\n=== GATE: layer mapping (probe index -> repr label) ===")
    print(f"  {'probe':>7} | {'repr':>11} | note")
    print("  " + "-" * 42)
    print(f"  {'—':>7} | {'Embed':>11} | no probe point (input embedding)")
    for k, ln in enumerate(PROBE_AXIS):
        note = "final layer -> acc(L12) for the criterion" if k == 11 else f"block output {k + 1}"
        print(f"  {('layer_%d' % k):>7} | {ln:>11} | {note}")
    print(f"  {'—':>7} | {POSTLN:>11} | rank only (post final-layer-norm variant)")


# --------------------------------------------------------------------------- #
# criterion
# --------------------------------------------------------------------------- #

def compute_criterion(short: str) -> dict:
    acc = load_accuracy(short)
    rank, rank_path = load_rank(short)
    acc_final = float(acc[11])                      # acc(L12) = probe index 11

    out_thr = {}
    for t in THRESHOLDS:
        target = t * acc_final
        hits = [i for i in range(12) if acc[i] >= target]
        assert hits, f"{short}: no layer reaches {t}x final accuracy"
        start_i = hits[0]
        start_label = PROBE_AXIS[start_i]
        region = PROBE_AXIS[start_i:]               # [tunnel_start .. L12]
        region_ranks = {ln: rank[ln] for ln in region}
        rmax_label = max(region_ranks, key=region_ranks.get)
        out_thr[f"{t:g}"] = {
            "threshold": t,
            "target_accuracy": target,
            "tunnel_start": start_label,
            "tunnel_start_probe_index": start_i,
            "nominal_tunnel_region": [region[0], region[-1]],
            "region_layers": region,
            "accuracy_at_start": float(acc[start_i]),
            "rank_at_tunnel_start": rank[start_label],
            "rank_max_within_region": region_ranks[rmax_label],
            "rank_max_layer": rmax_label,
            "rank_at_L11": rank["L11"],
            "ratio_rank_L11_over_start": rank["L11"] / rank[start_label],
        }

    return {
        "provenance": {
            "dataset": short, "probe_tag": DATASETS[short],
            "accuracy_source": str(PROBE_JSON.relative_to(OUT_DIR.parent)),
            "accuracy_field": "id_datasets[tag].poolings.content.binned_accuracy",
            "rank_source": str(rank_path.relative_to(OUT_DIR.parent)),
            "rank_field": "per_layer[*].dataset_effrank",
            "layer_mapping": "probe layer_k -> repr L(k+1); Embed has no probe point; "
                             f"{POSTLN} is rank-only",
            "thresholds": list(THRESHOLDS),
            "criterion": "Masarczyk et al. 2023: tunnel starts at the first layer whose ID "
                         "probe accuracy >= t * acc(final layer); inside the tunnel rank "
                         "should collapse while accuracy stays flat",
            "note_postprocessing_only": "no probe training, no forward passes",
        },
        "probe_axis": PROBE_AXIS,
        "accuracy_by_layer": {PROBE_AXIS[k]: float(acc[k]) for k in range(12)},
        "accuracy_final_L12": acc_final,
        "rank_by_layer": {ln: rank[ln] for ln in PROBE_AXIS},
        "rank_L12_postln": rank.get(POSTLN),
        "thresholds": out_thr,
    }


# --------------------------------------------------------------------------- #
# figure (regenerates from criterion.json + metrics.json alone)
# --------------------------------------------------------------------------- #

def plot_overlay(short: str) -> None:
    c = json.loads((CRIT_DIR / short / "criterion.json").read_text())
    rank, _ = load_rank(short)
    axis = c["probe_axis"]
    xs = np.arange(len(axis))
    acc = np.array([c["accuracy_by_layer"][ln] for ln in axis])
    rk = np.array([rank[ln] for ln in axis])
    t95 = c["thresholds"]["0.95"]
    start_i = t95["tunnel_start_probe_index"]

    fig, ax1 = plt.subplots(figsize=(9.5, 5.0))
    ax1.axvspan(start_i - 0.35, len(axis) - 1 + 0.35, color="lightgreen", alpha=0.30,
                label=f"nominal tunnel (95%): {t95['tunnel_start']}–L12")
    ax1.plot(xs, acc, "o-", color="tab:blue", lw=2, label="ID probe accuracy (binned, content)")
    ax1.axhline(c["accuracy_final_L12"] * 0.95, ls=":", color="tab:blue", alpha=0.6,
                label="95% of final accuracy")
    ax1.set_ylabel("ID probe accuracy", color="tab:blue")  # x ticks (L1..L12) self-label
    ax1.tick_params(axis="y", labelcolor="tab:blue")
    ax1.set_xticks(list(xs) + [len(axis)])
    ax1.set_xticklabels(axis + [POSTLN], rotation=45)

    ax2 = ax1.twinx()
    ax2.plot(xs, rk, "s--", color="tab:red", lw=2, label="dataset effective rank")
    if c.get("rank_L12_postln") is not None:
        ax2.plot([len(axis)], [c["rank_L12_postln"]], marker="s", mfc="none",
                 mec="tab:red", ms=11, ls="none", label=f"effective rank ({POSTLN})")
    ax2.set_ylabel("dataset-level effective rank", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    r_start, r_l11 = t95["rank_at_tunnel_start"], t95["rank_at_L11"]
    verdict = "RISES" if r_l11 > r_start else "collapses"
    fig.suptitle(f"{short}: Masarczyk tunnel criterion vs measured effective rank", y=0.99)
    ax1.set_title(f"95% tunnel starts at {t95['tunnel_start']} "
                  f"(98%: {c['thresholds']['0.98']['tunnel_start']}); inside the region rank "
                  f"{verdict} {r_start:.1f} → {r_l11:.1f} at L11 "
                  f"(x{t95['ratio_rank_L11_over_start']:.2f}), then {rank['L12']:.1f} at L12",
                  fontsize=9)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    # legend below the axes so it never occludes the curves
    ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, ncol=3,
               loc="upper center", bbox_to_anchor=(0.5, -0.32), frameon=False)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    out = CRIT_DIR / short / "fig_masarczyk_overlay.png"
    fig.savefig(out, dpi=140)
    plt.close(fig)
    print(f"  [saved] {out.relative_to(OUT_DIR)}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)

    CRIT_DIR.mkdir(parents=True, exist_ok=True)
    if args.figures_only:
        for short in DATASETS:
            plot_overlay(short)
        return

    print("=== GATE: inputs + anchor check ===")
    results = {}
    for short in DATASETS:
        c = compute_criterion(short)
        d = CRIT_DIR / short
        d.mkdir(parents=True, exist_ok=True)
        (d / "criterion.json").write_text(json.dumps(c, indent=1))
        results[short] = c
        print(f"  [saved] {(d / 'criterion.json').relative_to(OUT_DIR)}")

    print_mapping_table()

    # ---- summary table ----
    print("\n=== Masarczyk criterion summary ===")
    hdr = (f"{'dataset':>12} | {'start95':>7} | {'start98':>7} | "
           f"{'rank@start -> rank@L11':>24} | {'rank L12':>8} | {'rank postln':>11}")
    print(hdr); print("-" * len(hdr))
    for short, c in results.items():
        t95, t98 = c["thresholds"]["0.95"], c["thresholds"]["0.98"]
        traj = f"{t95['rank_at_tunnel_start']:8.2f} -> {t95['rank_at_L11']:8.2f}"
        print(f"{short:>12} | {t95['tunnel_start']:>7} | {t98['tunnel_start']:>7} | "
              f"{traj:>24} | {c['rank_by_layer']['L12']:8.2f} | "
              f"{(c['rank_L12_postln'] or float('nan')):11.2f}")

    # ---- assert gate (a failure here is a FINDING, reported as-is) ----
    print("\n=== GATE: rank(L11) > rank(tunnel_start(0.95)) ? ===")
    failures = []
    for short, c in results.items():
        t95 = c["thresholds"]["0.95"]
        rs, r11 = t95["rank_at_tunnel_start"], t95["rank_at_L11"]
        ok = r11 > rs
        print(f"  {short:>12}: rank(L11)={r11:.3f}  vs  rank({t95['tunnel_start']})={rs:.3f}"
              f"  -> {'PASS' if ok else 'FAIL'} (ratio x{r11 / rs:.3f})")
        if not ok:
            failures.append(short)
    print(f"  interpretation: rank RISES through the nominal tunnel on "
          f"{len(results) - len(failures)}/{len(results)} datasets "
          f"(Masarczyk predicts a COLLAPSE inside the tunnel)")
    assert not failures, f"rank(L11) <= rank(tunnel_start) for {failures} — reported as-is"

    print()
    for short in DATASETS:
        plot_overlay(short)


if __name__ == "__main__":
    main()
