"""
Improvement run on top of probe_pipeline.py.

Changes vs probe_pipeline.py:
  - SCP1 is the primary ID + synthetic-shift experiment (has headroom).
  - Bootstrap (B=2000) 95% CIs everywhere, including paired CIs for layer-vs-layer
    and ID-vs-OOD comparisons.
  - Transfer plot uses RAW accuracy (no min-max normalization).
  - Optional Handwriting headroom point (try/except so failure does not lose
    Phase 1/2 artifacts).

Chronos-2 weights stay frozen; cached features under ./features_cache/ are reused.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Reuse extraction + probe-fit logic from probe_pipeline (no rewrite).
from probe_pipeline import (
    NUM_LAYERS,
    SEED,
    extract_features,
    fit_layerwise_probes,
)


OUT_DIR = Path(".")
BOOT_B = 2000


# --------------------------------------------------------------------- #
# Phase 0 helpers
# --------------------------------------------------------------------- #

def score_layerwise_correctness(probes, features, y_true):
    """Returns dict {layer_idx: float32 correctness array of shape (n_test,)}."""
    y_true = np.asarray(y_true)
    out = {}
    for i in range(NUM_LAYERS):
        Xs = probes[i]["scaler"].transform(features[i])
        y_pred = probes[i]["clf"].predict(Xs)
        out[i] = (y_pred == y_true).astype(np.float32)
    return out


def bootstrap_ci(correct, B=BOOT_B, rng=None):
    """Test-set bootstrap CI for an accuracy from a correctness vector.

    Returns (point=mean(correct), lo=2.5pct, hi=97.5pct).
    Operates only on the precomputed correctness vector — no refit.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    correct = np.asarray(correct, dtype=np.float64)
    n = correct.size
    idx = rng.integers(0, n, size=(B, n))
    means = correct[idx].mean(axis=1)
    return float(correct.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(correct_a, correct_b, B=BOOT_B, rng=None):
    """Paired test-set bootstrap CI for (acc_a - acc_b).

    The same resampled indices are applied to BOTH correctness vectors, preserving
    their per-sample correlation. Returns (point, lo, hi).
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    a = np.asarray(correct_a, dtype=np.float64)
    b = np.asarray(correct_b, dtype=np.float64)
    assert a.size == b.size, "paired_diff_ci needs matched-length vectors"
    n = a.size
    idx = rng.integers(0, n, size=(B, n))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(a.mean() - b.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


# --------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------- #

def plot_layerwise_with_ci(path, curves, chance=None, title="", ylim=None, ylabel="test accuracy"):
    """curves: list of (label, point[L], lo[L], hi[L], color, kw)."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.arange(NUM_LAYERS)
    for label, point, lo, hi, color, kw in curves:
        kw = dict(kw or {})
        ax.fill_between(xs, lo, hi, alpha=0.18, color=color, linewidth=0)
        ax.plot(xs, point, marker="o", color=color, label=label, **kw)
    if chance is not None:
        ax.axhline(chance, ls="--", color="gray", linewidth=1, label=f"chance ({chance:.3f})")
    ax.set_xlabel("encoder layer index")
    ax.set_ylabel(ylabel)
    ax.set_xticks(xs)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [saved] {path}")


def plot_gap_with_ci(path, gap_point, gap_lo, gap_hi, title=""):
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.arange(NUM_LAYERS)
    ax.fill_between(xs, gap_lo, gap_hi, alpha=0.20, color="C2", linewidth=0,
                    label="95% paired bootstrap CI")
    ax.plot(xs, gap_point, marker="o", color="C2", label="ID - OOD(α=0.25)")
    ax.axhline(0.0, ls="--", color="gray", linewidth=1)
    ax.set_xlabel("encoder layer index")
    ax.set_ylabel("accuracy gap  (ID - OOD)")
    ax.set_xticks(xs)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  [saved] {path}")


# --------------------------------------------------------------------- #
# Per-dataset routine: ID + OOD sweep with CIs
# --------------------------------------------------------------------- #

def run_id_ood(dataset, alphas, rng, fig_id_ood, fig_sweep, fig_gap=None,
               saturation_msg=None):
    """Returns a results dict with everything needed for the slide-ready summary."""
    print(f"\nExtracting {dataset}/train ...")
    f_tr, y_tr = extract_features(dataset, split="train")
    print(f"Extracting {dataset}/test  ...")
    f_te, y_te = extract_features(dataset, split="test")

    classes = sorted(np.unique(y_tr).tolist())
    n_classes = len(classes)
    chance = 1.0 / n_classes

    print(f"\nFitting per-layer probes ({dataset}) ...")
    probes = fit_layerwise_probes(f_tr, y_tr)

    # ID correctness + CIs
    correct_id = score_layerwise_correctness(probes, f_te, y_te)
    id_pt = np.zeros(NUM_LAYERS); id_lo = np.zeros(NUM_LAYERS); id_hi = np.zeros(NUM_LAYERS)
    for i in range(NUM_LAYERS):
        id_pt[i], id_lo[i], id_hi[i] = bootstrap_ci(correct_id[i], rng=rng)

    saturated_layers = int((id_pt >= 0.95).sum())
    if saturated_layers >= 3:
        print(f"\n  *** SATURATION WARNING ({dataset}) ***  "
              f"{saturated_layers}/{NUM_LAYERS} layers >= 0.95"
              + (f"  -- {saturation_msg}" if saturation_msg else ""))
    else:
        print(f"\n  Saturation check ({dataset}): {saturated_layers} layers >= 0.95 (not saturating)")

    # OOD per alpha
    correct_ood = {}
    ood_pt = {a: np.zeros(NUM_LAYERS) for a in alphas}
    ood_lo = {a: np.zeros(NUM_LAYERS) for a in alphas}
    ood_hi = {a: np.zeros(NUM_LAYERS) for a in alphas}
    for alpha in alphas:
        print(f"\nExtracting {dataset}/test  corruption=gauss alpha={alpha} ...")
        f_te_n, y_te_n = extract_features(
            dataset, split="test",
            corruption={"kind": "gauss", "alpha": alpha, "seed": SEED},
        )
        assert np.array_equal(y_te_n, y_te), "labels changed under corruption"
        correct_ood[alpha] = score_layerwise_correctness(probes, f_te_n, y_te)
        for i in range(NUM_LAYERS):
            ood_pt[alpha][i], ood_lo[alpha][i], ood_hi[alpha][i] = bootstrap_ci(correct_ood[alpha][i], rng=rng)

    # ----- print tables -----
    print(f"\n  Per-layer {dataset} ID vs OOD@0.25 (95% bootstrap CI, B={BOOT_B}):")
    print(f"  {'layer':>5}  {'ID':>20}  {'OOD0.25':>20}")
    for i in range(NUM_LAYERS):
        print(f"  {i:>5d}  {id_pt[i]:>6.4f} [{id_lo[i]:.4f},{id_hi[i]:.4f}]  "
              f"{ood_pt[0.25][i]:>6.4f} [{ood_lo[0.25][i]:.4f},{ood_hi[0.25][i]:.4f}]")
    print(f"  chance = {chance:.4f}")

    # ----- early/mid argmax vs layer-11 paired diffs -----
    best_early = int(np.argmax(id_pt[:6]))  # over layers 0..5
    print(f"\n  best early/mid layer (0..5) by ID acc: layer {best_early} "
          f"(acc={id_pt[best_early]:.4f})")

    drop_id_pt, drop_id_lo, drop_id_hi = paired_diff_ci(
        correct_id[best_early], correct_id[NUM_LAYERS - 1], rng=rng
    )
    excl_id = (drop_id_lo > 0) or (drop_id_hi < 0)
    print(f"  late-layer drop (CLEAN ID)   = acc(L{best_early}) - acc(L{NUM_LAYERS-1}) = "
          f"{drop_id_pt:+.4f}  95% CI [{drop_id_lo:+.4f}, {drop_id_hi:+.4f}]  "
          f"{'EXCLUDES 0' if excl_id else 'includes 0'}")

    drop_ood_pt, drop_ood_lo, drop_ood_hi = paired_diff_ci(
        correct_ood[0.25][best_early], correct_ood[0.25][NUM_LAYERS - 1], rng=rng
    )
    excl_ood = (drop_ood_lo > 0) or (drop_ood_hi < 0)
    print(f"  late-layer drop (OOD@0.25)   = acc(L{best_early}) - acc(L{NUM_LAYERS-1}) = "
          f"{drop_ood_pt:+.4f}  95% CI [{drop_ood_lo:+.4f}, {drop_ood_hi:+.4f}]  "
          f"{'EXCLUDES 0' if excl_ood else 'includes 0'}")

    # ----- per-layer gap with paired CI -----
    gap_pt = np.zeros(NUM_LAYERS); gap_lo = np.zeros(NUM_LAYERS); gap_hi = np.zeros(NUM_LAYERS)
    print(f"\n  Per-layer (ID - OOD@0.25) gap with paired-bootstrap 95% CI:")
    print(f"  {'layer':>5}  {'gap':>8}  {'95% CI':>22}")
    for i in range(NUM_LAYERS):
        gap_pt[i], gap_lo[i], gap_hi[i] = paired_diff_ci(correct_id[i], correct_ood[0.25][i], rng=rng)
        print(f"  {i:>5d}  {gap_pt[i]:>+8.4f}  [{gap_lo[i]:+.4f}, {gap_hi[i]:+.4f}]")

    # Tunnel-direction check: trend of mid->late gap
    early_gap = gap_pt[:6].mean()
    late_gap = gap_pt[6:].mean()
    gap_widens = late_gap > early_gap
    print(f"  mean gap layers 0-5  = {early_gap:+.4f}")
    print(f"  mean gap layers 6-11 = {late_gap:+.4f}  -> "
          f"{'gap widens toward late layers (tunnel direction)' if gap_widens else 'gap does not widen toward late layers'}")

    # ----- figures -----
    plot_layerwise_with_ci(
        fig_id_ood,
        curves=[
            (f"{dataset} ID (clean test)", id_pt, id_lo, id_hi, "C0", None),
            (f"OOD (gauss α=0.25)",       ood_pt[0.25], ood_lo[0.25], ood_hi[0.25], "C3", None),
        ],
        chance=chance,
        title=f"{dataset}: ID vs synthetic noise OOD (95% bootstrap CI)",
        ylim=(max(0.0, chance - 0.15), 1.02),
    )

    sweep = [(f"{dataset} ID (clean test)", id_pt, id_lo, id_hi, "C0", {"linewidth": 2})]
    palette = ["C1", "C3", "C4"]
    for j, alpha in enumerate(alphas):
        sweep.append((f"OOD α={alpha}", ood_pt[alpha], ood_lo[alpha], ood_hi[alpha],
                      palette[j], {"linestyle": "--"}))
    plot_layerwise_with_ci(
        fig_sweep,
        curves=sweep,
        chance=chance,
        title=f"{dataset}: ID vs Gaussian-noise OOD sweep (95% CI)",
        ylim=(max(0.0, chance - 0.15), 1.02),
    )

    if fig_gap is not None:
        plot_gap_with_ci(
            fig_gap, gap_pt, gap_lo, gap_hi,
            title=f"{dataset}: per-layer ID-OOD(α=0.25) gap (paired 95% CI)"
        )

    return {
        "dataset": dataset,
        "chance": chance,
        "id_pt": id_pt, "id_lo": id_lo, "id_hi": id_hi,
        "ood_pt": ood_pt, "ood_lo": ood_lo, "ood_hi": ood_hi,
        "correct_id": correct_id, "correct_ood": correct_ood,
        "best_early": best_early,
        "drop_id": (drop_id_pt, drop_id_lo, drop_id_hi, excl_id),
        "drop_ood": (drop_ood_pt, drop_ood_lo, drop_ood_hi, excl_ood),
        "gap_pt": gap_pt, "gap_lo": gap_lo, "gap_hi": gap_hi,
        "gap_widens": gap_widens,
        "early_gap": float(early_gap), "late_gap": float(late_gap),
        "saturated_layers": saturated_layers,
        "probes": probes, "y_te": y_te,
    }


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    alphas = [0.1, 0.25, 0.5]
    pngs_written = []

    # --------- Phase 1: SCP1 primary ---------
    print("\n" + "=" * 72)
    print("PHASE 1  --  SelfRegulationSCP1: primary ID + synthetic-shift sweep")
    print("=" * 72)
    scp = run_id_ood(
        dataset="SelfRegulationSCP1",
        alphas=alphas,
        rng=rng,
        fig_id_ood=OUT_DIR / "fig_scp1_id_ood.png",
        fig_sweep=OUT_DIR / "fig_scp1_sweep.png",
        fig_gap=OUT_DIR / "fig_scp1_gap.png",
    )
    pngs_written += ["fig_scp1_id_ood.png", "fig_scp1_sweep.png", "fig_scp1_gap.png"]

    # --------- Phase 2: transfer plot with raw accuracy ---------
    print("\n" + "=" * 72)
    print("PHASE 2  --  cross-domain transfer (RAW accuracy)")
    print("=" * 72)
    print("Loading Epilepsy/train ...")
    f_tr_e, y_tr_e = extract_features("Epilepsy", split="train")
    print("Loading Epilepsy/test  ...")
    f_te_e, y_te_e = extract_features("Epilepsy", split="test")
    print("\nFitting per-layer probes (Epilepsy) ...")
    probes_e = fit_layerwise_probes(f_tr_e, y_tr_e)
    correct_e = score_layerwise_correctness(probes_e, f_te_e, y_te_e)
    chance_e = 1.0 / len(np.unique(y_tr_e))
    epi_pt = np.zeros(NUM_LAYERS); epi_lo = np.zeros(NUM_LAYERS); epi_hi = np.zeros(NUM_LAYERS)
    for i in range(NUM_LAYERS):
        epi_pt[i], epi_lo[i], epi_hi[i] = bootstrap_ci(correct_e[i], rng=rng)

    print(f"\n  Raw per-layer accuracy with 95% CIs:")
    print(f"  {'layer':>5}  {'Epilepsy':>22}  {'SCP1':>22}")
    for i in range(NUM_LAYERS):
        print(f"  {i:>5d}  {epi_pt[i]:.4f} [{epi_lo[i]:.4f},{epi_hi[i]:.4f}]  "
              f"{scp['id_pt'][i]:.4f} [{scp['id_lo'][i]:.4f},{scp['id_hi'][i]:.4f}]")
    j_epi = int(np.argmax(epi_pt))
    j_scp = int(np.argmax(scp["id_pt"]))
    print(f"  Epilepsy: chance={chance_e:.4f}  argmax=layer {j_epi}  acc={epi_pt[j_epi]:.4f}")
    print(f"  SCP1:     chance={scp['chance']:.4f}  argmax=layer {j_scp}  acc={scp['id_pt'][j_scp]:.4f}")

    # transfer figure
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.arange(NUM_LAYERS)
    ax.fill_between(xs, epi_lo, epi_hi, alpha=0.18, color="C0", linewidth=0)
    ax.plot(xs, epi_pt, marker="o", color="C0", label="Epilepsy ID")
    ax.fill_between(xs, scp["id_lo"], scp["id_hi"], alpha=0.18, color="C3", linewidth=0)
    ax.plot(xs, scp["id_pt"], marker="s", color="C3", label="SelfRegulationSCP1 ID")
    ax.axhline(chance_e, ls="--", color="C0", linewidth=1, alpha=0.6,
               label=f"Epilepsy chance ({chance_e:.3f})")
    ax.axhline(scp["chance"], ls="--", color="C3", linewidth=1, alpha=0.6,
               label=f"SCP1 chance ({scp['chance']:.3f})")
    ax.annotate(f"argmax L{j_epi}", xy=(j_epi, epi_pt[j_epi]),
                xytext=(j_epi, min(epi_pt[j_epi] + 0.04, 1.04)),
                ha="center", fontsize=8, color="C0")
    ax.annotate(f"argmax L{j_scp}", xy=(j_scp, scp["id_pt"][j_scp]),
                xytext=(j_scp, scp["id_pt"][j_scp] + 0.04),
                ha="center", fontsize=8, color="C3")
    ax.set_xlabel("encoder layer index")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(xs)
    ax.set_title("Layer-profile comparison across domains (raw accuracy, not classifier transfer)")
    ax.set_ylim(0.0, 1.08)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_transfer_raw.png", dpi=140)
    plt.close(fig)
    print(f"  [saved] {OUT_DIR / 'fig_transfer_raw.png'}")
    pngs_written.append("fig_transfer_raw.png")
    print("\n  NOTE: this is a layer-profile comparison across domains, "
          "NOT classifier transfer (different label spaces, separate probes per dataset).")

    # --------- Phase 3: optional Handwriting ---------
    print("\n" + "=" * 72)
    print("PHASE 3  --  optional Handwriting headroom point")
    print("=" * 72)
    hw = None
    try:
        hw = run_id_ood(
            dataset="Handwriting",
            alphas=alphas,
            rng=rng,
            fig_id_ood=OUT_DIR / "fig_handwriting_id_ood.png",
            fig_sweep=OUT_DIR / "fig_handwriting_sweep.png",
            fig_gap=None,
            saturation_msg="(unexpected — Handwriting was the swap-target)",
        )
        pngs_written += ["fig_handwriting_id_ood.png", "fig_handwriting_sweep.png"]
    except Exception as e:
        print(f"  Handwriting phase failed: {type(e).__name__}: {e}")
        print(f"  Phase 1 + Phase 2 artifacts are unaffected.")

    # --------- Slide-ready summary ---------
    print("\n" + "=" * 72)
    print("SLIDE-READY SUMMARY")
    print("=" * 72)

    def fmt_summary(r):
        ds = r["dataset"]
        be = r["best_early"]
        d_pt, d_lo, d_hi, excl = r["drop_id"]
        do_pt, do_lo, do_hi, exclo = r["drop_ood"]
        argmax = int(np.argmax(r["id_pt"]))
        argmax_acc = r["id_pt"][argmax]
        gap_widens = r["gap_widens"]
        return (
            f"{ds}: argmax layer L{argmax} (acc {argmax_acc:.3f}). "
            f"Late-layer drop = acc(L{be}) - acc(L{NUM_LAYERS-1}) = {d_pt:+.3f} "
            f"95%CI [{d_lo:+.3f}, {d_hi:+.3f}] ({'excludes 0' if excl else 'includes 0'}); "
            f"under OOD(α=0.25) drop = {do_pt:+.3f} CI [{do_lo:+.3f}, {do_hi:+.3f}] "
            f"({'excludes 0' if exclo else 'includes 0'}). "
            f"ID-OOD gap {'WIDENS' if gap_widens else 'does not widen'} from early "
            f"({r['early_gap']:+.3f}) to late ({r['late_gap']:+.3f}) layers."
        )

    print("\n" + fmt_summary(scp))
    if hw is not None:
        print("\n" + fmt_summary(hw))

    print(f"\nPNGs written ({len(pngs_written)}):")
    for p in pngs_written:
        path = OUT_DIR / p
        sz = path.stat().st_size if path.exists() else 0
        flag = "OK" if path.exists() else "MISSING"
        print(f"  {flag}  {path}  ({sz} bytes)")


if __name__ == "__main__":
    main()
