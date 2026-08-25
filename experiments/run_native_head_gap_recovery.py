"""ext_v5 GAP-RECOVERY diagnostic — lightweight post-hoc analysis of the completed native-head-adapter
results. NO retraining, NO re-extraction, NO model load: it only reads the committed per-window
``bootstrap_inputs/native_head_adapter__<tag>.npz`` and reuses the driver's own bootstrap machinery.

For each dataset and layer l it reports the fraction of the zero-shot -> native MASE gap that the shared
linear adapter A_l closes:

        R_l = ( L_zero(l) - L_adapter(l) ) / ( L_zero(l) - L_native )        (lower MASE = better)

R_l = 0  -> adapter did nothing (L_adapter == L_zero);  R_l = 1 -> adapter fully closed the gap to
native.  Values are NOT clipped: R_l > 1 means the adapter beat native, R_l < 0 means it hurt.

Degeneracy: index 13 (L12+RMS) IS RMSNorm(index 12), and zero-shot@L12 applies that same RMSNorm, so
zero-shot@L12 == native EXACTLY (verified per-window on all 7 datasets). The gap denominator therefore
collapses to 0 at BOTH L12 and L12+RMS, and R_l is undefined there. Such points are flagged NaN and
omitted from the plotted curve (never manufactured as 0 or 1). Two documented criteria decide validity:
  (1) point:  |gap_l| >= DENOM_REL_TOL * max_k |gap_k|      (2% of the dataset's largest gap)
  (2) bootstrap: < DENOM_SIGN_FRAC of resamples may have denom <= 0 (else the ratio is sign-unstable).

Caveat kept in the caption/table: A_l is a SUPERVISED ~590k-param linear map (768->768) fit to the
forecast target through the frozen native head. A large R_l says a supervised linear transform of layer l
is SUFFICIENT to make it native-head-usable; it does NOT recover the true h_l -> h_12 transform, and does
NOT imply later layers merely do coordinate alignment.

Outputs (every name carries ``gap_recovery``; nothing existing is overwritten; ext_v4 untouched):
    plots/native_head_adapter__gap_recovery__overview_2x4.png
    plots/native_head_adapter__gap_recovery__<tag>.png
    tables/native_head_adapter__gap_recovery__all.{csv,json}

Run (login/CPU, seconds; same cost as the existing ``--figures`` bootstrap):
    python -m experiments.run_native_head_gap_recovery
"""
from __future__ import annotations

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probing.config import SEED
from probing.stats import cluster_bootstrap_counts, ci_bounds
# reuse the committed driver verbatim (import only — no behaviour change, no outputs of its own)
from experiments.run_native_head_adapter import (
    BOOT_B, DATASET_KIND, LAYER_LABELS, OUT_ROOT, REF_IDX, ALL_TAGS,
    _boot_mean, _load_pw, _series_group)
from experiments.run_ptood_probing_ftok import SHORT

# --- documented numerical tolerances (MASE scale) --------------------------- #
GAP_METRIC = "mase"          # the primary, plotted native-head-adapter metric
DENOM_REL_TOL = 0.02         # |gap| below 2% of the dataset's max gap -> undefined (catches L12 collapse)
DENOM_SIGN_FRAC = 0.01       # >=99% of bootstrap resamples must keep denom > 0, else ratio is unstable
PLOT_LAYERS = list(range(REF_IDX))   # 0..12 = Emb..L12 ; REF_IDX (13, L12+RMS) is the native endpoint


def gap_recovery_curve(tag):
    """Per-layer R_l with a paired series-cluster bootstrap CI, reusing the driver's resampler.

    Returns a list of row-dicts (one per layer in PLOT_LAYERS)."""
    pw, series = _load_pw(tag)
    S, inv = _series_group(series)
    M = cluster_bootstrap_counts(S, BOOT_B, SEED)            # ONE shared resample -> paired conditions
    nat = pw[("native", REF_IDX)][GAP_METRIC]
    nat_mean = float(nat.mean())
    nb = _boot_mean(M, nat, inv, S)                          # (B,) native, identical across layers

    gaps = {L: float(pw[("zero_shot", L)][GAP_METRIC].mean()) - nat_mean for L in PLOT_LAYERS}
    gap_scale = max(abs(g) for g in gaps.values())           # per-dataset scale for the point tolerance
    tol = DENOM_REL_TOL * gap_scale

    rows = []
    for L in PLOT_LAYERS:
        z = pw[("zero_shot", L)][GAP_METRIC]
        a = pw[("linear_adapter", L)][GAP_METRIC]
        zmean, amean, gap = float(z.mean()), float(a.mean()), gaps[L]
        zb, ab = _boot_mean(M, z, inv, S), _boot_mean(M, a, inv, S)
        denom_b = zb - nb
        frac_neg = float(np.mean(denom_b <= 0.0))
        point_ok = abs(gap) >= tol
        sign_ok = frac_neg < DENOM_SIGN_FRAC
        if not point_ok:
            flag, R, lo, hi = "undefined:gap~0", float("nan"), float("nan"), float("nan")
        elif not sign_ok:
            flag, R, lo, hi = "unstable:denom_sign", float("nan"), float("nan"), float("nan")
        else:
            R = (zmean - amean) / gap                        # plug-in ratio of means (the central estimate)
            lo, hi = ci_bounds((zb - ab) / denom_b)          # percentile CI of the paired bootstrap ratio
            flag = "valid"
        rows.append({
            "dataset": tag, "kind": DATASET_KIND[tag], "layer": L, "layer_label": LAYER_LABELS[L],
            "zero_shot_mase": round(zmean, 6), "adapter_mase": round(amean, 6),
            "native_mase": round(nat_mean, 6), "gap_denominator": round(gap, 6),
            "R_l": (None if np.isnan(R) else round(R, 6)),
            "R_boot_lo": (None if np.isnan(lo) else round(lo, 6)),
            "R_boot_hi": (None if np.isnan(hi) else round(hi, 6)),
            "frac_denom_nonpositive": round(frac_neg, 6),
            "valid_flag": flag, "n_windows": int(series.size), "n_series": int(S),
        })
    return rows


def _panel(ax, rows):
    tag = rows[0]["dataset"]; kind = rows[0]["kind"]
    x = np.arange(len(PLOT_LAYERS))
    valid = [r for r in rows if r["valid_flag"] == "valid"]
    xs = [r["layer"] for r in valid]
    R = np.array([r["R_l"] for r in valid]); lo = np.array([r["R_boot_lo"] for r in valid])
    hi = np.array([r["R_boot_hi"] for r in valid])
    if kind == "PT-OOD":                                     # clearly mark PT-ID vs PT-OOD
        ax.set_facecolor("#fbf3ee")
    ax.axhline(1.0, color="tab:green", ls="--", lw=1.2, label="R = 1  (gap fully closed to native)")
    ax.axhline(0.0, color="0.4", ls=":", lw=1.2, label="R = 0  (no gain over zero-shot)")
    ax.fill_between(xs, lo, hi, color="tab:blue", alpha=0.18)
    ax.plot(xs, R, marker="s", ms=4, color="tab:blue", label="R (shared linear adapter)")
    ax.set_xticks(x); ax.set_xticklabels([LAYER_LABELS[i] for i in PLOT_LAYERS], rotation=60, fontsize=6)
    ax.set_title(f"{SHORT.get(tag, tag)}  [{kind}]", fontsize=9,
                 color=("tab:red" if kind == "PT-OOD" else "black"))
    ax.grid(alpha=0.25)


def make_gap_recovery(out_root=OUT_ROOT):
    (out_root / "plots").mkdir(parents=True, exist_ok=True)
    (out_root / "tables").mkdir(parents=True, exist_ok=True)
    have = [t for t in ALL_TAGS
            if (out_root / "bootstrap_inputs" / f"native_head_adapter__{t}.npz").exists()]
    if not have:
        print("[gap_recovery] no bootstrap_inputs found — run the adapter (`--adapt`) first"); return
    all_rows = {t: gap_recovery_curve(t) for t in have}

    # per-dataset panels
    for tag in have:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        _panel(ax, all_rows[tag])
        ax.set_ylabel("R  (fraction of zero-shot->native MASE gap recovered)")
        ax.set_xlabel("representation depth")
        ax.legend(fontsize=7)
        fig.suptitle(f"gap recovery — {SHORT.get(tag, tag)} [{DATASET_KIND[tag]}]\n"
                     "fraction of the zero-shot->native gap closed by the shared linear adapter", fontsize=9)
        fig.tight_layout()
        out = out_root / "plots" / f"native_head_adapter__gap_recovery__{tag}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"  [saved] {out.name}")

    # 2x4 overview (7 datasets + legend in the 8th panel), matching the existing overview layout
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5)); axes = axes.ravel()
    for k, tag in enumerate(ALL_TAGS):
        if tag not in have:
            axes[k].set_visible(False); continue
        _panel(axes[k], all_rows[tag])
        if k % 4 == 0:
            axes[k].set_ylabel("R (gap recovered)")
    axes[7].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[7].legend(h, l, loc="center", fontsize=10, title="gap-recovery R")
    fig.suptitle("Fraction of zero-shot->native gap recovered by the shared linear adapter (MASE)\n"
                 "R_l = (L_zero(l) - L_adapter(l)) / (L_zero(l) - L_native);  4 PT-ID + 3 PT-OOD;  "
                 "L12/L12+RMS undefined (zero-shot == native) and omitted", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = out_root / "plots" / "native_head_adapter__gap_recovery__overview_2x4.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  [saved] {out.name}")

    # machine-readable table
    flat = [r for tag in have for r in all_rows[tag]]
    stem = out_root / "tables" / "native_head_adapter__gap_recovery__all"
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(flat[0])); wr.writeheader(); wr.writerows(flat)
    json.dump({"tolerances": {"denom_rel_tol": DENOM_REL_TOL, "denom_sign_frac": DENOM_SIGN_FRAC,
                              "metric": GAP_METRIC, "bootstrap_B": BOOT_B},
               "rows": flat}, open(f"{stem}.json", "w"), indent=2)
    print(f"  [saved] {stem.name}.csv/.json ({len(flat)} rows)")
    return all_rows


if __name__ == "__main__":
    make_gap_recovery()
