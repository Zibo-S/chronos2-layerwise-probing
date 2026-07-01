"""
Hardening run: five additions on top of probe_pipeline.py / probe_improve.py.

  Phase 1 — pooling ablation (content/all/reg) on SCP1 + Handwriting
  Phase 2 — replication across more datasets (forest plot of late-layer deficit)
  Phase 3 — Part-2 amplification test on SCP1 under Gaussian(α=0.25)
  Phase 4 — 20-split stability of late_drop_band on SCP1 + Handwriting
  Phase 5 — second OOD shift types (time warp + baseline drift)

Reuses extract_features, fit_layerwise_probes, score_layerwise_correctness,
bootstrap_ci, paired_diff_ci from probe_pipeline.py / probe_improve.py.

Cached features are reused; new corruption types (timewarp, drift) trigger fresh
forward passes and are written to the same ./features_cache/ directory.

Chronos-2 stays frozen (eval, no_grad, requires_grad_=False), float32, MPS or CPU.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probing.config import NUM_LAYERS, SEED, OUT_DIR
from probing.extraction import extract_features, fit_layerwise_probes
from probing.probes import score_layerwise_correctness
from probing.stats import bootstrap_ci, paired_diff_ci


# ----------------------------------------------------------------- #
# Shared
# ----------------------------------------------------------------- #

BOOT_B = 2000
MIDDLE_BAND = list(range(3, 9))  # a priori, NOT data-selected: layers 3..8 inclusive
LAST_LAYER = NUM_LAYERS - 1      # = 11

ARTIFACTS_PATH = OUT_DIR / "probe_harden_artifacts.json"


def _serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serializable(v) for v in obj]
    return obj


def save_artifact(key, value, all_artifacts):
    all_artifacts[key] = _serializable(value)
    ARTIFACTS_PATH.write_text(json.dumps(all_artifacts, indent=2))


def band_correctness(correct):
    """Per-sample correctness averaged across MIDDLE_BAND layers.

    For each test sample i, band_correctness[i] = mean_{L in band} correct[L][i] in [0,1].
    Then mean(band_correctness) = mean over the band of per-layer accuracies (the
    quantity we want for late_drop_band).
    """
    return np.stack([correct[L] for L in MIDDLE_BAND], axis=0).mean(axis=0)


def late_drop_band_ci(correct, B=BOOT_B, rng=None):
    """Paired bootstrap CI of mean(acc over band) - acc(L11)."""
    band = band_correctness(correct)
    return paired_diff_ci(band, correct[LAST_LAYER], B=B, rng=rng)


def late_drop_argmax_ci(correct, B=BOOT_B, rng=None):
    """Paired bootstrap CI of (max_{L<=10} acc(L)) - acc(L11).

    The argmax is data-selected, so this is selection-biased; report it but label
    it that way (band variant is the principled headline).
    """
    accs = np.array([correct[L].mean() for L in range(NUM_LAYERS - 1)])
    L_star = int(np.argmax(accs))
    pt, lo, hi = paired_diff_ci(correct[L_star], correct[LAST_LAYER], B=B, rng=rng)
    return pt, lo, hi, L_star


def amplification_ci(correct_id, correct_ood, layer_mid, B=BOOT_B, rng=None):
    """Paired bootstrap CI for (gap_OOD - gap_ID), gap = acc(layer_mid) - acc(L11).

    Same resampled indices applied to all four correctness vectors.
    """
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = correct_id[layer_mid].size
    idx = rng.integers(0, n, size=(B, n))
    a_id_m = correct_id[layer_mid][idx].mean(axis=1)
    a_id_l = correct_id[LAST_LAYER][idx].mean(axis=1)
    a_oo_m = correct_ood[layer_mid][idx].mean(axis=1)
    a_oo_l = correct_ood[LAST_LAYER][idx].mean(axis=1)
    amp_boot = (a_oo_m - a_oo_l) - (a_id_m - a_id_l)
    pt = (float(correct_ood[layer_mid].mean()) - float(correct_ood[LAST_LAYER].mean())) \
       - (float(correct_id[layer_mid].mean())  - float(correct_id[LAST_LAYER].mean()))
    return float(pt), float(np.percentile(amp_boot, 2.5)), float(np.percentile(amp_boot, 97.5))


def amplification_band_ci(correct_id, correct_ood, B=BOOT_B, rng=None):
    if rng is None:
        rng = np.random.default_rng(SEED)
    n = correct_id[LAST_LAYER].size
    idx = rng.integers(0, n, size=(B, n))
    band_id  = band_correctness(correct_id)
    band_ood = band_correctness(correct_ood)
    g_id  = band_id[idx].mean(axis=1)  - correct_id[LAST_LAYER][idx].mean(axis=1)
    g_ood = band_ood[idx].mean(axis=1) - correct_ood[LAST_LAYER][idx].mean(axis=1)
    amp_boot = g_ood - g_id
    pt = (float(band_ood.mean()) - float(correct_ood[LAST_LAYER].mean())) \
       - (float(band_id.mean())  - float(correct_id[LAST_LAYER].mean()))
    return float(pt), float(np.percentile(amp_boot, 2.5)), float(np.percentile(amp_boot, 97.5))


def saturation_check(accs):
    sat = int((np.asarray(accs) >= 0.95).sum())
    return sat >= 3, sat


def excl0(lo, hi):
    return (lo > 0) or (hi < 0)


# ----------------------------------------------------------------- #
# Phase 1 — pooling ablation
# ----------------------------------------------------------------- #

def phase1(rng, artifacts):
    print("\n" + "=" * 72)
    print("PHASE 1  --  pooling ablation: content vs all vs reg")
    print("=" * 72)

    rows = []
    for dataset in ("SelfRegulationSCP1", "Handwriting"):
        for pool in ("content", "all", "reg"):
            print(f"\n[{dataset} / pool={pool}]")
            f_tr, y_tr = extract_features(dataset, split="train", pooling=pool)
            f_te, y_te = extract_features(dataset, split="test", pooling=pool)
            probes = fit_layerwise_probes(f_tr, y_tr)
            correct = score_layerwise_correctness(probes, f_te, y_te)
            accs = np.array([correct[L].mean() for L in range(NUM_LAYERS)])
            pt, lo, hi = late_drop_band_ci(correct, rng=rng)
            print(f"  per-layer acc: {np.array2string(accs, precision=3, suppress_small=True)}")
            print(f"  late_drop_band = {pt:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                  f"{'EXCLUDES 0' if excl0(lo, hi) else 'includes 0'}")
            rows.append({
                "dataset": dataset, "pooling": pool,
                "accs": accs.tolist(),
                "late_drop_band": pt, "lo": lo, "hi": hi,
                "excludes_0": excl0(lo, hi),
            })

    print("\n  --- Phase 1 summary table ---")
    print(f"  {'dataset':>22s}  {'pooling':>8s}  {'late_drop_band':>16s}  {'95% CI':>22s}  {'excl0':>6s}")
    for r in rows:
        print(f"  {r['dataset']:>22s}  {r['pooling']:>8s}  "
              f"{r['late_drop_band']:>+16.4f}  [{r['lo']:+.4f},{r['hi']:+.4f}]  "
              f"{'YES' if r['excludes_0'] else 'no':>6s}")

    # grouped bar chart
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    datasets = ["SelfRegulationSCP1", "Handwriting"]
    pools = ["content", "all", "reg"]
    x = np.arange(len(datasets))
    width = 0.25
    colors = {"content": "C0", "all": "C1", "reg": "C2"}
    for j, pool in enumerate(pools):
        pts = [next(r for r in rows if r["dataset"] == ds and r["pooling"] == pool)["late_drop_band"]
               for ds in datasets]
        los = [next(r for r in rows if r["dataset"] == ds and r["pooling"] == pool)["lo"]
               for ds in datasets]
        his = [next(r for r in rows if r["dataset"] == ds and r["pooling"] == pool)["hi"]
               for ds in datasets]
        err_lo = [p - l for p, l in zip(pts, los)]
        err_hi = [h - p for p, h in zip(pts, his)]
        ax.bar(x + (j - 1) * width, pts, width, color=colors[pool], label=pool, alpha=0.85,
               yerr=[err_lo, err_hi], capsize=4)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("late_drop_band  =  mean(acc L3..L8) - acc(L11)")
    ax.set_title("Phase 1: pooling ablation of late-layer deficit (95% paired bootstrap CI)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(title="pooling", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_pooling_ablation.png", dpi=140)
    plt.close(fig)
    print(f"  [saved] fig_pooling_ablation.png")
    save_artifact("phase1", rows, artifacts)
    return rows


# ----------------------------------------------------------------- #
# Phase 2 — more datasets, replication breadth
# ----------------------------------------------------------------- #

PART2_DATASETS = [
    # already-cached, fast
    "Epilepsy", "SelfRegulationSCP1", "Handwriting",
    # new
    "UWaveGestureLibrary", "EthanolConcentration", "SelfRegulationSCP2",
    "LSST", "Cricket",
]


def phase2(rng, artifacts):
    print("\n" + "=" * 72)
    print("PHASE 2  --  replication breadth")
    print("=" * 72)

    rows = []
    failures = []
    for ds in PART2_DATASETS:
        print(f"\n[{ds}]")
        try:
            f_tr, y_tr = extract_features(ds, split="train", pooling="content")
            # check channel cap from a feature shape: f_tr[0] has shape (n, c*768)
            c_inferred = f_tr[0].shape[1] // 768
            if c_inferred > 8:
                print(f"  SKIP: {c_inferred} channels exceeds cap of 8")
                failures.append({"dataset": ds, "reason": f"too many channels ({c_inferred})"})
                continue
            f_te, y_te = extract_features(ds, split="test", pooling="content")
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failures.append({"dataset": ds, "reason": f"{type(e).__name__}: {e}"})
            continue

        probes = fit_layerwise_probes(f_tr, y_tr)
        correct = score_layerwise_correctness(probes, f_te, y_te)
        accs = np.array([correct[L].mean() for L in range(NUM_LAYERS)])
        sat, sat_count = saturation_check(accs)

        pt_b, lo_b, hi_b = late_drop_band_ci(correct, rng=rng)
        pt_a, lo_a, hi_a, L_star = late_drop_argmax_ci(correct, rng=rng)
        argmax_layer = int(np.argmax(accs))
        chance = 1.0 / len(np.unique(y_te))

        print(f"  n_test={len(y_te)}  n_classes={len(np.unique(y_te))}  chance={chance:.4f}  "
              f"saturating={sat}  ({sat_count}/{NUM_LAYERS} layers >= .95)")
        print(f"  argmax_layer = L{argmax_layer}  acc={accs.max():.4f}")
        print(f"  late_drop_band   = {pt_b:+.4f}  CI [{lo_b:+.4f},{hi_b:+.4f}]  "
              f"{'EXCLUDES 0' if excl0(lo_b, hi_b) else 'includes 0'}")
        print(f"  late_drop_argmax = {pt_a:+.4f}  CI [{lo_a:+.4f},{hi_a:+.4f}]  "
              f"(L_star=L{L_star}; selection-biased)")

        rows.append({
            "dataset": ds, "n_test": int(len(y_te)),
            "n_classes": int(len(np.unique(y_te))), "chance": chance,
            "saturating": bool(sat), "sat_count": int(sat_count),
            "argmax_layer": argmax_layer,
            "argmax_acc": float(accs.max()),
            "accs": accs.tolist(),
            "late_drop_band": (pt_b, lo_b, hi_b),
            "late_drop_argmax": (pt_a, lo_a, hi_a, L_star),
            "excludes_0_band": excl0(lo_b, hi_b),
        })

        # Save artifacts incrementally — one phase2 row at a time
        save_artifact("phase2_rows", rows, artifacts)
        save_artifact("phase2_failures", failures, artifacts)

    # Full table
    print("\n  --- Phase 2 summary table ---")
    print(f"  {'dataset':>22s}  {'n_test':>7}  {'n_cls':>6}  {'chance':>7}  "
          f"{'sat?':>5}  {'argmax':>7}  {'late_drop_band':>16}  {'95% CI':>22}  {'excl0':>6}")
    for r in rows:
        pt, lo, hi = r["late_drop_band"]
        print(f"  {r['dataset']:>22s}  {r['n_test']:>7d}  {r['n_classes']:>6d}  "
              f"{r['chance']:>7.4f}  {'YES' if r['saturating'] else 'no':>5s}  "
              f"L{r['argmax_layer']:<6d}  {pt:>+16.4f}  [{lo:+.4f},{hi:+.4f}]  "
              f"{'YES' if r['excludes_0_band'] else 'no':>6s}")

    nonsat = [r for r in rows if not r["saturating"]]
    K = len(nonsat)
    J = sum(1 for r in nonsat if r["excludes_0_band"] and r["late_drop_band"][0] > 0)
    print(f"\n  Headline (non-saturated only): final layer significantly worse than middle band "
          f"(CI excludes 0 and point > 0) in {J} of {K} datasets.")

    if failures:
        print(f"\n  Failures/skips ({len(failures)}):")
        for f in failures:
            print(f"    - {f['dataset']}: {f['reason']}")

    # Forest plot: non-saturated datasets only
    if nonsat:
        nonsat_sorted = sorted(nonsat, key=lambda r: r["late_drop_band"][0])
        fig, ax = plt.subplots(figsize=(7.5, max(3.0, 0.45 * len(nonsat_sorted) + 1.5)))
        names = [r["dataset"] for r in nonsat_sorted]
        pts   = np.array([r["late_drop_band"][0] for r in nonsat_sorted])
        los   = np.array([r["late_drop_band"][1] for r in nonsat_sorted])
        his   = np.array([r["late_drop_band"][2] for r in nonsat_sorted])
        ys = np.arange(len(names))
        colors = ["C2" if (lo > 0 or hi < 0) else "C7" for lo, hi in zip(los, his)]
        ax.errorbar(pts, ys, xerr=[pts - los, his - pts], fmt="o", color="black",
                    ecolor="black", markerfacecolor="white", capsize=3, elinewidth=1.2)
        for y, lo, hi, c in zip(ys, los, his, colors):
            ax.add_patch(plt.Rectangle((lo, y - 0.15), hi - lo, 0.30,
                                       facecolor=c, alpha=0.25, edgecolor="none"))
        ax.set_yticks(ys)
        ax.set_yticklabels(names)
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1)
        ax.set_xlabel("late_drop_band  =  mean(acc L3..L8) - acc(L11)  (95% paired CI)")
        ax.set_title(f"Phase 2: late-layer deficit, non-saturated datasets only "
                     f"({J}/{K} CIs exclude 0)")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "fig_dataset_forest.png", dpi=140)
        plt.close(fig)
        print(f"  [saved] fig_dataset_forest.png")

    save_artifact("phase2_rows", rows, artifacts)
    save_artifact("phase2_failures", failures, artifacts)
    save_artifact("phase2_headline", {"J": J, "K": K}, artifacts)
    return rows, failures


# ----------------------------------------------------------------- #
# Phase 3 — Part-2 amplification on SCP1
# ----------------------------------------------------------------- #

def phase3(rng, artifacts):
    print("\n" + "=" * 72)
    print("PHASE 3  --  Part-2 amplification on SCP1 under Gaussian α=0.25")
    print("=" * 72)

    ds = "SelfRegulationSCP1"
    f_tr, y_tr = extract_features(ds, split="train", pooling="content")
    f_te_clean, y_te = extract_features(ds, split="test", pooling="content")
    f_te_ood, y_te_o = extract_features(
        ds, split="test", pooling="content",
        corruption={"kind": "gauss", "alpha": 0.25, "seed": SEED},
    )
    assert np.array_equal(y_te, y_te_o), "labels changed under corruption"

    probes = fit_layerwise_probes(f_tr, y_tr)
    correct_id  = score_layerwise_correctness(probes, f_te_clean, y_te)
    correct_ood = score_layerwise_correctness(probes, f_te_ood,   y_te)

    # best mid-layer = argmax of CLEAN-ID acc within MIDDLE_BAND
    mid_accs = np.array([correct_id[L].mean() for L in MIDDLE_BAND])
    best_mid = MIDDLE_BAND[int(np.argmax(mid_accs))]
    print(f"  best_mid_layer (argmax of ID acc within L{MIDDLE_BAND[0]}..L{MIDDLE_BAND[-1]}) = L{best_mid}  "
          f"acc={mid_accs.max():.4f}")
    print(f"  acc(L11) ID  = {correct_id[LAST_LAYER].mean():.4f}")
    print(f"  acc(L11) OOD = {correct_ood[LAST_LAYER].mean():.4f}")

    # amplification with best_mid
    amp_pt, amp_lo, amp_hi = amplification_ci(correct_id, correct_ood, best_mid, rng=rng)
    print(f"\n  amplification @ best_mid (L{best_mid}) = (gap_OOD - gap_ID) = "
          f"{amp_pt:+.4f}  95% CI [{amp_lo:+.4f}, {amp_hi:+.4f}]  "
          f"{'EXCLUDES 0' if excl0(amp_lo, amp_hi) else 'includes 0'}")

    # band-mean version (principled)
    amp_b_pt, amp_b_lo, amp_b_hi = amplification_band_ci(correct_id, correct_ood, rng=rng)
    print(f"  amplification @ band (mean L3..L8) = {amp_b_pt:+.4f}  "
          f"95% CI [{amp_b_lo:+.4f}, {amp_b_hi:+.4f}]  "
          f"{'EXCLUDES 0' if excl0(amp_b_lo, amp_b_hi) else 'includes 0'}")

    result = {
        "best_mid": best_mid,
        "amp_best_mid": (amp_pt, amp_lo, amp_hi),
        "amp_band": (amp_b_pt, amp_b_lo, amp_b_hi),
    }
    save_artifact("phase3", result, artifacts)
    return result


# ----------------------------------------------------------------- #
# Phase 4 — multi-split stability
# ----------------------------------------------------------------- #

def phase4(rng, artifacts):
    print("\n" + "=" * 72)
    print("PHASE 4  --  20-split stability on SCP1 + Handwriting (refit only)")
    print("=" * 72)
    from sklearn.model_selection import StratifiedShuffleSplit
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    summary = {}
    for ds in ("SelfRegulationSCP1", "Handwriting"):
        print(f"\n[{ds}]")
        f_tr, y_tr = extract_features(ds, split="train", pooling="content")
        f_te, y_te = extract_features(ds, split="test", pooling="content")
        # Concatenate per-layer features and labels; same row order across layers.
        X_full = {L: np.concatenate([f_tr[L], f_te[L]], axis=0) for L in range(NUM_LAYERS)}
        y_full = np.concatenate([np.asarray(y_tr), np.asarray(y_te)], axis=0)
        n_total = len(y_full)
        canonical_test_frac = len(y_te) / n_total
        print(f"  n_total={n_total}  canonical test frac = {len(y_te)}/{n_total} = {canonical_test_frac:.4f}")

        drops_band = []
        drops_argmax = []
        argmax_layers = []
        for s in range(20):
            sss = StratifiedShuffleSplit(n_splits=1, test_size=canonical_test_frac, random_state=s)
            (tr_idx, te_idx), = sss.split(np.zeros(n_total), y_full)
            # per-layer fit+score
            correct_s = {}
            accs_s = np.zeros(NUM_LAYERS)
            for L in range(NUM_LAYERS):
                Xtr = X_full[L][tr_idx]
                Xte = X_full[L][te_idx]
                ytr = y_full[tr_idx]
                yte = y_full[te_idx]
                scaler = StandardScaler().fit(Xtr)
                clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(scaler.transform(Xtr), ytr)
                yp = clf.predict(scaler.transform(Xte))
                correct_s[L] = (yp == yte).astype(np.float32)
                accs_s[L] = correct_s[L].mean()
            band_mean = band_correctness(correct_s).mean()
            drop_b = band_mean - accs_s[LAST_LAYER]
            drops_band.append(drop_b)
            argmax_L = int(np.argmax(accs_s[:NUM_LAYERS - 1]))
            argmax_layers.append(argmax_L)
            drops_argmax.append(accs_s[argmax_L] - accs_s[LAST_LAYER])

        drops_band = np.array(drops_band)
        drops_argmax = np.array(drops_argmax)
        print(f"  late_drop_band across 20 splits: mean={drops_band.mean():+.4f}  "
              f"2.5/97.5 pct = [{np.percentile(drops_band, 2.5):+.4f}, "
              f"{np.percentile(drops_band, 97.5):+.4f}]")
        print(f"  fraction with late_drop_band > 0: {(drops_band > 0).mean():.2f}  "
              f"({int((drops_band > 0).sum())}/{len(drops_band)})")
        # histogram of argmax layer
        argmax_counts = {L: int((np.array(argmax_layers) == L).sum()) for L in range(NUM_LAYERS - 1)}
        nonzero = {L: c for L, c in argmax_counts.items() if c > 0}
        print(f"  argmax-layer histogram (over L0..L10): {nonzero}")

        summary[ds] = {
            "drops_band": drops_band.tolist(),
            "drops_argmax": drops_argmax.tolist(),
            "argmax_layers": argmax_layers,
            "argmax_counts": argmax_counts,
            "frac_gt0": float((drops_band > 0).mean()),
            "canonical_test_frac": canonical_test_frac,
        }
        save_artifact("phase4", summary, artifacts)

    # Box plot
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = list(summary.keys())
    data = [summary[ds]["drops_band"] for ds in labels]
    bp = ax.boxplot(data, labels=labels, showfliers=True, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], ["C0", "C3"]):
        patch.set_facecolor(color); patch.set_alpha(0.35)
    # Overlay individual split points
    for i, vals in enumerate(data):
        x = np.full(len(vals), i + 1) + np.random.uniform(-0.06, 0.06, size=len(vals))
        ax.scatter(x, vals, color="black", s=14, alpha=0.7, zorder=3)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_ylabel("late_drop_band over 20 stratified resplits")
    ax.set_title("Phase 4: re-split robustness of the late-layer deficit\n"
                 "(distinct from the canonical-split headline numbers)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_multisplit_stability.png", dpi=140)
    plt.close(fig)
    print(f"  [saved] fig_multisplit_stability.png")
    save_artifact("phase4", summary, artifacts)
    return summary


# ----------------------------------------------------------------- #
# Phase 5 — second OOD shift types
# ----------------------------------------------------------------- #

def phase5(rng, artifacts):
    print("\n" + "=" * 72)
    print("PHASE 5  --  second OOD shift types (time warp, baseline drift)")
    print("=" * 72)

    # Shift definitions — fixed and label-preserving by construction.
    shifts = [
        ("gauss(α=0.25)", {"kind": "gauss", "alpha": 0.25, "seed": SEED}),
        ("timewarp(f=1.2)", {"kind": "timewarp", "factor": 1.2}),
        ("drift(amp=0.3)", {"kind": "drift", "amplitude": 0.3}),
    ]

    results = {}
    for ds in ("SelfRegulationSCP1", "Handwriting"):
        print(f"\n[{ds}]")
        f_tr, y_tr = extract_features(ds, split="train", pooling="content")
        f_te_clean, y_te = extract_features(ds, split="test", pooling="content")
        probes = fit_layerwise_probes(f_tr, y_tr)
        correct_id = score_layerwise_correctness(probes, f_te_clean, y_te)

        # best_mid is dataset-specific; pick once per dataset
        mid_accs = np.array([correct_id[L].mean() for L in MIDDLE_BAND])
        best_mid = MIDDLE_BAND[int(np.argmax(mid_accs))]
        chance = 1.0 / len(np.unique(y_tr))
        print(f"  best_mid_layer = L{best_mid}  acc={mid_accs.max():.4f}  chance={chance:.4f}")

        per_shift = {}
        for label, corr in shifts:
            print(f"  -- shift: {label} --")
            f_te_o, y_te_o = extract_features(ds, split="test", pooling="content", corruption=corr)
            assert np.array_equal(y_te_o, y_te), "labels changed under corruption"
            correct_oo = score_layerwise_correctness(probes, f_te_o, y_te)
            acc_best_clean = float(correct_id[best_mid].mean())
            acc_best_ood   = float(correct_oo[best_mid].mean())
            print(f"     label-sanity: acc(L{best_mid}) clean={acc_best_clean:.4f}  "
                  f"shifted={acc_best_ood:.4f}  (chance {chance:.4f})")
            if acc_best_ood < chance + 0.05:
                print(f"     WARNING: shifted accuracy near chance; shift may have destroyed task")

            amp_pt, amp_lo, amp_hi = amplification_ci(correct_id, correct_oo, best_mid, rng=rng)
            ampb_pt, ampb_lo, ampb_hi = amplification_band_ci(correct_id, correct_oo, rng=rng)
            ood_accs = np.array([correct_oo[L].mean() for L in range(NUM_LAYERS)])
            print(f"     amplification @ best_mid (L{best_mid}) = {amp_pt:+.4f}  "
                  f"95% CI [{amp_lo:+.4f}, {amp_hi:+.4f}]  "
                  f"{'EXCLUDES 0' if excl0(amp_lo, amp_hi) else 'includes 0'}")
            print(f"     amplification @ band                  = {ampb_pt:+.4f}  "
                  f"95% CI [{ampb_lo:+.4f}, {ampb_hi:+.4f}]  "
                  f"{'EXCLUDES 0' if excl0(ampb_lo, ampb_hi) else 'includes 0'}")
            per_shift[label] = {
                "ood_accs": ood_accs.tolist(),
                "acc_best_clean": acc_best_clean,
                "acc_best_ood": acc_best_ood,
                "amp_best_mid": (amp_pt, amp_lo, amp_hi),
                "amp_band":     (ampb_pt, ampb_lo, ampb_hi),
            }

        results[ds] = {
            "best_mid": best_mid,
            "chance": chance,
            "id_accs": [float(correct_id[L].mean()) for L in range(NUM_LAYERS)],
            "shifts": per_shift,
        }
        save_artifact("phase5", results, artifacts)

    # ---------- Figures ----------
    # Bar chart: amplification per shift per dataset
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    shift_labels = [s[0] for s in shifts]
    datasets = list(results.keys())
    x = np.arange(len(shift_labels))
    width = 0.38
    colors = {"SelfRegulationSCP1": "C3", "Handwriting": "C1"}
    for i, ds in enumerate(datasets):
        pts, los, his = [], [], []
        for sl in shift_labels:
            p, lo, hi = results[ds]["shifts"][sl]["amp_band"]
            pts.append(p); los.append(lo); his.append(hi)
        pts = np.array(pts); los = np.array(los); his = np.array(his)
        err_lo = pts - los; err_hi = his - pts
        ax.bar(x + (i - 0.5) * width, pts, width, yerr=[err_lo, err_hi], capsize=4,
               color=colors[ds], alpha=0.85, label=ds)
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(shift_labels)
    ax.set_ylabel("amplification = (band - L11)_OOD - (band - L11)_ID  (95% paired CI)")
    ax.set_title("Phase 5: does the mid-vs-last gap widen under multiple shift types?")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_shift_amplification.png", dpi=140)
    plt.close(fig)
    print(f"  [saved] fig_shift_amplification.png")

    # SCP1 ID vs timewarp-OOD curve (CI bands)
    scp = results["SelfRegulationSCP1"]
    id_accs = np.array(scp["id_accs"])
    tw_accs = np.array(scp["shifts"]["timewarp(f=1.2)"]["ood_accs"])
    # Reload correctness for CI bands on the same call to ensure paired bootstraps
    f_tr, y_tr = extract_features("SelfRegulationSCP1", split="train", pooling="content")
    f_te, y_te = extract_features("SelfRegulationSCP1", split="test", pooling="content")
    f_te_tw, _ = extract_features("SelfRegulationSCP1", split="test", pooling="content",
                                  corruption={"kind": "timewarp", "factor": 1.2})
    probes = fit_layerwise_probes(f_tr, y_tr)
    correct_id  = score_layerwise_correctness(probes, f_te, y_te)
    correct_tw  = score_layerwise_correctness(probes, f_te_tw, y_te)
    id_pt = np.zeros(NUM_LAYERS); id_lo = np.zeros(NUM_LAYERS); id_hi = np.zeros(NUM_LAYERS)
    tw_pt = np.zeros(NUM_LAYERS); tw_lo = np.zeros(NUM_LAYERS); tw_hi = np.zeros(NUM_LAYERS)
    for L in range(NUM_LAYERS):
        id_pt[L], id_lo[L], id_hi[L] = bootstrap_ci(correct_id[L], rng=rng)
        tw_pt[L], tw_lo[L], tw_hi[L] = bootstrap_ci(correct_tw[L], rng=rng)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    xs = np.arange(NUM_LAYERS)
    ax.fill_between(xs, id_lo, id_hi, alpha=0.18, color="C0", linewidth=0)
    ax.plot(xs, id_pt, marker="o", color="C0", label="SCP1 ID (clean test)")
    ax.fill_between(xs, tw_lo, tw_hi, alpha=0.18, color="C4", linewidth=0)
    ax.plot(xs, tw_pt, marker="s", color="C4", label="SCP1 OOD (timewarp f=1.2)")
    ax.axhline(scp["chance"], ls="--", color="gray", linewidth=1, label=f"chance ({scp['chance']:.3f})")
    ax.set_xlabel("encoder layer index")
    ax.set_ylabel("test accuracy")
    ax.set_xticks(xs)
    ax.set_title("SCP1: layer-wise probe accuracy under time-warp OOD (95% CI)")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fig_scp1_timewarp_idood.png", dpi=140)
    plt.close(fig)
    print(f"  [saved] fig_scp1_timewarp_idood.png")
    save_artifact("phase5", results, artifacts)
    return results


# ----------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------- #

def main():
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    artifacts = {}
    if ARTIFACTS_PATH.exists():
        # start fresh — incremental saves overwrite as we go
        artifacts = {}

    phase1_rows = phase1(rng, artifacts)
    phase2_rows, phase2_failures = phase2(rng, artifacts)
    phase3_result = phase3(rng, artifacts)
    phase4_summary = phase4(rng, artifacts)
    phase5_results = phase5(rng, artifacts)

    # ----------- slide-ready summary -----------
    print("\n" + "=" * 72)
    print("SLIDE-READY SUMMARY")
    print("=" * 72)

    # 1) Pooling invariance — Phase 1
    pool_excl = {}
    for ds in ("SelfRegulationSCP1", "Handwriting"):
        pool_excl[ds] = {r["pooling"]: r["excludes_0"] for r in phase1_rows if r["dataset"] == ds}
    print("\n(1) Pooling invariance of late-layer deficit:")
    for ds in pool_excl:
        excl = pool_excl[ds]
        n_excl = sum(excl.values())
        print(f"    {ds}: CI excludes 0 in {n_excl}/3 pooling variants  ({excl})")

    # Phase-2 headline
    nonsat = [r for r in phase2_rows if not r["saturating"]]
    K = len(nonsat)
    J = sum(1 for r in nonsat if r["excludes_0_band"] and r["late_drop_band"][0] > 0)
    print(f"    Across {len(phase2_rows)} loaded datasets, {K} non-saturated; "
          f"final layer significantly worse than middle band in {J}/{K}.")

    # 2) Phase-5 amplification
    print("\n(2) Part-2 amplification under multiple shift types (band-mean, 95% CI):")
    for ds in phase5_results:
        for sl, info in phase5_results[ds]["shifts"].items():
            p, lo, hi = info["amp_band"]
            print(f"    {ds:>22s}  {sl:>22s}  {p:+.4f}  [{lo:+.4f}, {hi:+.4f}]  "
                  f"{'EXCLUDES 0' if (lo>0 or hi<0) else 'includes 0'}")

    # 3) Phase-4 multi-split stability
    print("\n(3) Multi-split stability of late_drop_band (band-mean variant):")
    for ds, s in phase4_summary.items():
        db = np.array(s["drops_band"])
        print(f"    {ds:>22s}  mean={db.mean():+.4f}  "
              f"pct [2.5,97.5]=[{np.percentile(db,2.5):+.4f}, {np.percentile(db,97.5):+.4f}]  "
              f"frac > 0 = {s['frac_gt0']:.2f}")

    # PNG inventory
    pngs = [
        "fig_pooling_ablation.png",
        "fig_dataset_forest.png",
        "fig_multisplit_stability.png",
        "fig_shift_amplification.png",
        "fig_scp1_timewarp_idood.png",
    ]
    print("\nPNGs written:")
    for p in pngs:
        path = OUT_DIR / p
        sz = path.stat().st_size if path.exists() else 0
        flag = "OK" if path.exists() else "MISSING"
        print(f"  {flag}  {path}  ({sz} bytes)")

    print(f"\nArtifacts JSON: {ARTIFACTS_PATH}")


if __name__ == "__main__":
    main()
