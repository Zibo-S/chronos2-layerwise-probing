"""Pretraining-OOD probe transfer: 4 frozen ID-source probes -> 3 documented-OOD targets (4x3).

Question: does the intermediate-vs-final-layer advantage GROW when the target is outside
Chronos-2's documented pretraining distribution? We take the 4 FROZEN extended_v2 source probes
(electricity, uber_tlc, m4_hourly, wind_farms_hourly — all pretraining-ID) and score them, without
any adaptation, on 3 pretraining-OOD targets (SG Carpark, Coastal T-S, BOOM). Every cell is OOD
(no diagonal). Nothing is trained here — the source checkpoints and the target windows are frozen.

Protocol (inherited unchanged): frozen amazon/chronos-2, content-mean pooling, C=512/H=64, q9,
arcsinh-context normalization, Chronos-2 quantile loss, in-context seasonal-naive MASE (m=24),
seed 0. Targets: 650 deterministic windows from id_data.build_ood_windows (query-balanced), frozen
before any layer is inspected. Cluster unit = carpark / station / metric-query.

Two layer analyses (PRIMARY is source-validation-selected — the unseen targets have NO validation
split, so the ONLY leakage-free selection is the source's own ID validation carve):
  A. SOURCE-SELECTED (PRIMARY, fair OOD): one layer per source chosen on the SOURCE's own ID
     validation loss (reconstructed from the checkpoint's wd-selection record) — never touches any
     OOD-target loss. Frozen: elec L3, uber L3, m4 L1, wind L3. Gains may be negative.
  B. ORACLE (DIAGNOSTIC): best layer = target-test argmin — optimistically biased (post-selection),
     kept only for reference and clearly labeled.

Primary layer metric: relative_gain_pct = 100*(loss[L12]-loss[layer])/loss[L12], q9; its 95% CI is
the paired series-level cluster bootstrap RATIO computed inside each replicate (reused verbatim from
run_ood_transfer). Also retained: best/L12 q9 loss, raw delta, MASE, last-value, seasonal-naive,
native Chronos-2.

Run (extraction + native need a GPU / warm cache; aggregation is CPU):
    python -m experiments.run_ood_pretrain_transfer                 # eval all cells + aggregate
    python -m experiments.run_ood_pretrain_transfer --target sg_carpark
    python -m experiments.run_ood_pretrain_transfer --figure-only   # aggregate saved cells (CPU)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probing import config
from probing.config import NUM_LAYERS, LAST_LAYER, SEED
from probing.id_data import build_ood_windows, OOD_CLUSTER_UNIT
from probing.extraction import extract_window_features
from probing.probes import QUANTILE_SETS, median_index, predict_quantile_probe
from experiments.run_id_forecasting import (compute_mase, _ctx_stats, _mase_denominator,
                                            native_median_forecast, M_SEASON, ID_STYLE)
# reuse the committed-transfer frame: frozen-checkpoint loader, paired bootstrap, relative gain.
from experiments import run_ood_transfer as ood

# ---- fixed frame (must match the source probes) ----
SOURCE_SET = "extended_v2"          # the 4 frozen source probes live under results/extended_v2/
SOURCES = ["monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly"]
TARGETS = ["sg_carpark", "coastal_ts", "boom_hourly"]
POOLING, C, H = ood.POOLING, ood.C, ood.H
BOOT_B, CI_LO, CI_HI = ood.BOOT_B, ood.CI_LO, ood.CI_HI
SRC_SHORT = {t: ood.SHORT_LABELS.get(t, t) for t in SOURCES}
TGT_SHORT = {"sg_carpark": "SG Carpark", "coastal_ts": "Coastal T-S", "boom_hourly": "BOOM"}
TGT_FREQ = {"sg_carpark": "H (agg 15min)", "coastal_ts": "H", "boom_hourly": "H"}
XLABELS = ["Embed"] + [str(i) for i in range(1, NUM_LAYERS)]


def _derive_dirs():
    global OOD_PT_DIR, PER_CELL_DIR, BOOT_IN_DIR, TGT_DIR, FIG_DIR
    OOD_PT_DIR = config.ID_OUT_DIR / "ood_pretrain_transfer"
    PER_CELL_DIR = OOD_PT_DIR / "per_cell"
    BOOT_IN_DIR = OOD_PT_DIR / "bootstrap_inputs"
    TGT_DIR = OOD_PT_DIR / "targets"
    FIG_DIR = OOD_PT_DIR / "figures"
    for d in (OOD_PT_DIR, PER_CELL_DIR, BOOT_IN_DIR, TGT_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# source-selected layer (Part B) — reconstructed from the SOURCE checkpoint's wd-selection
# record (source 80/20 ID validation loss). Never touches any OOD target.
# --------------------------------------------------------------------------- #

def source_selected_layer(fitted):
    """(argmin layer, per-layer val loss) from the frozen source probe's saved wd selection —
    each layer's validation loss AT its chosen weight decay (source ID val carve only)."""
    vals = []
    for i in range(NUM_LAYERS):
        sel = fitted[i].get("selection")
        if not sel or "val_loss_by_wd" not in sel:
            return None, None                        # wd grid was off -> no source-selected layer
        vals.append(float(min(sel["val_loss_by_wd"].values())))
    return int(np.argmin(vals)), vals


# --------------------------------------------------------------------------- #
# per-target reference forecasts (once per target; native needs GPU/cache)
# --------------------------------------------------------------------------- #

def ood_target_baselines(target, w):
    """last-value / seasonal-naive(m=24) / native-Chronos-2 median MASE on the target's 650
    windows, in-context seasonal-naive denominator (leakage-free, context-only). Returns scalar
    metrics + per-window MASE (n,) for the paired improvement CIs."""
    X = np.asarray(w["X_test"], np.float64)
    n, Cx = X.shape
    Y = w["Y_test_traj"]
    mu, s = _ctx_stats(X, w["meta"]["sigma_eps"])
    y_raw = mu[:, None] + s[:, None] * np.sinh(Y.astype(np.float64))
    d = np.maximum(_mase_denominator(X), 1e-8)[:, None]
    seas = X[:, Cx - M_SEASON + (np.arange(H) % M_SEASON)]
    fc = {"last_value": np.repeat(X[:, -1:], H, axis=1), "seasonal_naive": seas,
          "native_chronos2": native_median_forecast(target, w["X_test"], H).astype(np.float64)}
    metrics, pw = {}, {}
    for name, f in fc.items():
        A = np.abs(y_raw - f) / d
        pw[name] = A.mean(axis=1)
        metrics[name] = float(A.mean())
    return metrics, pw, (mu, s, y_raw, d)


# --------------------------------------------------------------------------- #
# eval one target across all sources (frozen checkpoints; no training)
# --------------------------------------------------------------------------- #

def eval_target(target, sources, qset, quantiles, seed, device):
    print(f"\n{'='*70}\n[target={target}] build 650 windows + extract features\n{'='*70}")
    w = build_ood_windows(target)
    m = w["meta"]
    print(f"  windows={m['n_test']} clusters={m['n_test_clusters']}/{m['n_clusters_total']} "
          f"unit={m['cluster_unit']} wpc={m['windows_per_cluster_min_med_max']}")
    if int(m["n_test"]) == 0:                              # fail loud: no clean C+H span in the target
        raise SystemExit(
            f"target {target} produced 0 evaluation windows "
            f"({m['n_test_windows_before_subsample']} candidates before subsample, "
            f"{m['n_skipped_windows']} skipped as non-finite / near-constant). The preprocessing "
            f"yielded no fully-finite C+H={C + H} span — check the target's missingness (run "
            f"`run_ood_screen --ood-targets {target}` first) before the GPU eval.")
    f_te, _ = extract_window_features(target, "test", w["X_test"], w["y_test"], pooling=POOLING)

    base_metrics, base_pw, (mu, s, y_raw, d) = ood_target_baselines(target, w)
    sid = np.asarray(w["series_test"], np.int64)
    json.dump({"target": target, "frequency": TGT_FREQ[target], "cluster_unit": m["cluster_unit"],
               "n_windows": int(m["n_test"]), "n_clusters": int(m["n_test_clusters"]),
               "n_clusters_total": int(m["n_clusters_total"]),
               "windows_per_cluster_min_med_max": m["windows_per_cluster_min_med_max"],
               "n_windows_before_subsample": int(m["n_test_windows_before_subsample"]),
               "n_skipped_windows": int(m["n_skipped_windows"]),
               "baselines_mase": base_metrics, "notes": m["notes"]},
              open(TGT_DIR / f"{target}.json", "w"), indent=2)
    np.savez(TGT_DIR / f"{target}__baseline_pw.npz", series_test=sid,
             native=base_pw["native_chronos2"], seasonal=base_pw["seasonal_naive"],
             last=base_pw["last_value"])

    for src in sources:
        print(f"  [eval OOD] {SRC_SHORT[src]} -> {TGT_SHORT[target]}")
        fitted = ood.load_checkpoints(src, qset, seed, quantiles, device)
        if fitted is None:
            raise SystemExit(f"missing frozen source checkpoints for {src} ({qset}) under "
                             f"{ood.CKPT_DIR} — run run_ood_transfer --dataset-set {SOURCE_SET} first")
        out, diag = predict_quantile_probe(fitted, f_te, w["Y_test_traj"], quantiles=quantiles,
                                           device=device, collect_test_median=True,
                                           collect_test_window_loss=True)
        mase_entry, _mpw = compute_mase(target, w, {POOLING: diag})
        ss_layer, ss_val = source_selected_layer(fitted)

        wl = np.stack([diag["test_window_loss"][i] for i in range(NUM_LAYERS)]).astype(np.float64)
        # per-window MASE per layer (probe median, raw units) for baseline improvement CIs
        wmase = np.stack([(np.abs(y_raw - (mu[:, None] + s[:, None]
                          * np.sinh(diag["test_median"][i].astype(np.float64)))) / d).mean(axis=1)
                          for i in range(NUM_LAYERS)]).astype(np.float64)
        np.savez(BOOT_IN_DIR / f"{src}__to__{target}__{qset}__seed{seed}.npz",
                 window_loss=wl, window_mase=wmase, series_test=sid)

        rows = [{"source_dataset": src, "target_dataset": target, "layer": i,
                 "is_ood": True, "ood_category": "pretraining_ood", "pooling": POOLING,
                 "quantile_set": qset, "quantile_config": int(len(quantiles)),
                 "quantile_loss": float(out[i]),
                 "mean_pinball_loss": float(diag["test_mean_pinball"][i]),
                 "mase": float(mase_entry["poolings"][POOLING][i])}
                for i in range(NUM_LAYERS)]
        ql = np.array([out[i] for i in range(NUM_LAYERS)])
        summary = {"source_dataset": src, "target_dataset": target, "seed": int(seed),
                   "quantile_set": qset, "is_ood": True, "ood_category": "pretraining_ood",
                   "target_frequency": TGT_FREQ[target], "cluster_unit": m["cluster_unit"],
                   "n_test_windows": int(m["n_test"]), "n_test_clusters": int(m["n_test_clusters"]),
                   "oracle_best_layer": int(ql.argmin()),
                   "source_selected_layer": ss_layer, "source_val_loss_by_layer": ss_val,
                   "last_layer": LAST_LAYER, "last_layer_loss": float(ql[LAST_LAYER]),
                   "native_mase": float(mase_entry["native_mase"]),
                   "preprocessing_notes": m["notes"]}
        json.dump({"rows": rows, "summary": summary},
                  open(PER_CELL_DIR / f"{src}__{target}__{qset}__seed{seed}.json", "w"), indent=2)
    del w, f_te
    gc.collect()


# --------------------------------------------------------------------------- #
# aggregate: bootstrap every cell, both layer analyses, tables + CSV + figures
# --------------------------------------------------------------------------- #

def _load_cells(qset, seed):
    cells = {}
    for src in SOURCES:
        for tgt in TARGETS:
            p = PER_CELL_DIR / f"{src}__{tgt}__{qset}__seed{seed}.json"
            bp = BOOT_IN_DIR / f"{src}__to__{tgt}__{qset}__seed{seed}.npz"
            if p.exists() and bp.exists():
                d = json.load(open(p))
                with np.load(bp) as z:
                    bc = ood._paired_delta_bootstrap(z["window_loss"], z["series_test"])
                    wmase = z["window_mase"]; sid = z["series_test"]
                cells[(src, tgt)] = {"summary": d["summary"], "rows": d["rows"], "bc": bc,
                                     "wmase": wmase, "sid": sid,
                                     "curve": np.array([r["quantile_loss"] for r in d["rows"]]),
                                     "mase": np.array([r["mase"] for r in d["rows"]])}
    return cells


def _cell_record(src, tgt, cell, method):
    """Per-cell record for one layer-analysis method ('oracle' | 'source_selected')."""
    bc, summ = cell["bc"], cell["summary"]
    layer = summ["oracle_best_layer"] if method == "oracle" else summ["source_selected_layer"]
    if layer is None:
        return None
    rg = ood._relative_gain_cell(bc, layer)
    l12 = float(bc["point"][LAST_LAYER])
    return {"source": src, "target": tgt, "ood_category": "pretraining_ood",
            "target_frequency": summ["target_frequency"],
            "selection_method": method, "selected_layer": int(layer),
            "q9_loss": round(float(bc["point"][layer]), 6), "l12_loss": round(l12, 6),
            "raw_gain": round(l12 - float(bc["point"][layer]), 6),
            "relative_gain_pct": round(rg["relative_gain_pct"], 4),
            "relative_ci": f"[{rg['ci_lo']:+.3f}, {rg['ci_hi']:+.3f}]",
            "relative_ci_lo": round(rg["ci_lo"], 4), "relative_ci_hi": round(rg["ci_hi"], 4),
            "relative_ci_excludes_zero": rg["excludes_zero"],
            "mase": round(float(cell["mase"][layer]), 6),
            "cluster_count": int(summ["n_test_clusters"]), "window_count": int(summ["n_test_windows"]),
            "preprocessing_notes": summ["preprocessing_notes"]}


def aggregate(qset, seed):
    cells = _load_cells(qset, seed)
    if not cells:
        print("  [aggregate] no evaluated cells yet — run the eval (needs GPU) first")
        return
    # CSV (both methods) + JSON
    recs = [r for (src, tgt), cell in cells.items() for method in ("oracle", "source_selected")
            for r in [_cell_record(src, tgt, cell, method)] if r is not None]
    fields = ["source", "target", "ood_category", "target_frequency", "selection_method",
              "selected_layer", "q9_loss", "l12_loss", "raw_gain", "relative_gain_pct",
              "relative_ci", "relative_ci_lo", "relative_ci_hi", "relative_ci_excludes_zero",
              "mase", "cluster_count", "window_count", "preprocessing_notes"]
    with open(OOD_PT_DIR / f"ood_pretrain_transfer_results__{qset}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(recs)
    json.dump(recs, open(OOD_PT_DIR / f"ood_pretrain_transfer_results__{qset}.json", "w"), indent=2)
    print(f"  [saved] results CSV/JSON ({len(recs)} records) -> {OOD_PT_DIR}")

    make_loss_grid(cells, qset, seed)
    make_relative_gain_heatmap(cells, qset, seed)
    make_source_selected_figure(cells, qset, seed)
    make_baseline_comparison(cells, qset, seed)
    make_three_group_plot(cells, qset, seed)
    _print_summary(recs)


def _print_summary(recs):
    label = {"source_selected": "source_val PRIMARY ", "oracle": "oracle    DIAGNOSTIC"}
    for method in ("source_selected", "oracle"):
        sub = [r for r in recs if r["selection_method"] == method]
        if not sub:
            continue
        v = np.array([r["relative_gain_pct"] for r in sub])
        nsig = sum(r["relative_ci_excludes_zero"] for r in sub)
        nsig_pos = sum(r["relative_ci_excludes_zero"] and r["relative_gain_pct"] > 0 for r in sub)
        print(f"  [{label[method]}] rel-gain over L12: mean {v.mean():+.2f}%  median {np.median(v):+.2f}%  "
              f"positive {int((v>0).sum())}/{len(v)}  CI≠0 {nsig}/{len(v)} (of which CI>0 {nsig_pos}/{len(v)})")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def make_loss_grid(cells, qset, seed):
    """4x3 grid: rows=source probe, cols=OOD target. Per-layer q9 loss (Embed+L1..L12) with the
    paired cluster-bootstrap band; ★ = oracle best layer, ◆ = source-selected layer, filled black-
    edged dots where the paired Δ-vs-last CI excludes 0; dotted line at L12. y shared within column."""
    xs = np.arange(NUM_LAYERS)
    fig, axes = plt.subplots(len(SOURCES), len(TARGETS),
                             figsize=(5.2 * len(TARGETS), 4.1 * len(SOURCES)),
                             sharex=True, sharey="col", squeeze=False)
    for ri, src in enumerate(SOURCES):
        color = ID_STYLE.get(src, {}).get("color", "#333333")
        for ci, tgt in enumerate(TARGETS):
            ax = axes[ri][ci]
            cell = cells.get((src, tgt))
            if cell is None:
                ax.text(0.5, 0.5, "(not run)", ha="center", va="center", transform=ax.transAxes,
                        color="0.6"); ax.set_xticks(xs); continue
            bc, summ, curve = cell["bc"], cell["summary"], cell["curve"]
            ax.plot(xs, curve, color=color, lw=2.2, marker="o", ms=4, zorder=3)
            ax.fill_between(xs, bc["ci_lo"], bc["ci_hi"], color=color, alpha=0.16, lw=0)
            sig = bc["delta_above_zero"].copy(); sig[LAST_LAYER] = False
            if sig.any():
                ax.plot(xs[sig], curve[sig], "o", color=color, ms=7, mec="k", mew=0.9, zorder=4)
            bo = summ["oracle_best_layer"]
            ss = summ["source_selected_layer"]
            # PRIMARY: source-validation-selected layer (◆, filled); SECONDARY: oracle test-best (hollow ★)
            if ss is not None:
                ax.plot(ss, curve[ss], marker="D", ms=12, color=color, mec="k", mew=1.4, zorder=6,
                        label=f"source-val sel = {XLABELS[ss]}")
            ax.plot(bo, curve[bo], marker="*", ms=14, mfc="none", mec="0.25", mew=1.1, zorder=5,
                    label=f"oracle test-best = {XLABELS[bo]}")
            ax.axvline(LAST_LAYER, color="0.4", ls=":", lw=1.1)
            chosen = ss if ss is not None else bo
            rgc = ood._relative_gain_cell(bc, chosen)["relative_gain_pct"]
            sel_word = "source-val sel" if ss is not None else "oracle (no source val)"
            ax.set_title(f"{SRC_SHORT[src]} → {TGT_SHORT[tgt]}  [OOD]\n"
                         f"{sel_word} {XLABELS[chosen]}: Δ vs last = "
                         f"{summ['last_layer_loss'] - curve[chosen]:+.3f}  ({rgc:+.1f}%)", fontsize=9)
            ax.set_xticks(xs); ax.set_xticklabels(XLABELS, fontsize=7)
            ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("representation (Embed + L1..L12)")
    for ri, src in enumerate(SOURCES):
        axes[ri][0].set_ylabel(f"probe trained on {SRC_SHORT[src]}\nChronos-2 q9 loss (test)")
    fig.suptitle(f"Pretraining-OOD probe transfer (4×3) — Chronos-2 q9 loss by layer  [{qset}, seed {seed}]\n"
                 "rows = ID source probe, cols = documented pretraining-OOD target;  every cell is OOD;  "
                 "◆ source-validation-selected (PRIMARY, no target data), hollow ★ oracle target-test-best "
                 "(secondary/diagnostic);  Δ/gain in title = ◆ and MAY be negative;  filled dot = paired "
                 "Δ-vs-last CI excludes 0;  y shared within a column", fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIG_DIR / f"loss_grid_4x3__{qset}.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"  [saved] {out}")


def make_relative_gain_heatmap(cells, qset, seed):
    """4x3 relative-gain heatmap in BOTH layer-selection views. 100*(loss[L12]-loss[L])/loss[L12],
    diverging at 0; each cell annotated with the chosen layer + gain% + 95% paired CI; ★/bold where
    the CI excludes 0. The layer is fixed BEFORE the bootstrap -> CIs descriptive. Writes:
      PRIMARY  source_val_relative_gain_heatmap_4x3 — L = source-validation-selected (fair, no target),
      DIAGNOSTIC oracle_relative_gain_heatmap_4x3   — L = target-test argmin (optimistically biased)."""
    for mode in ("source_val", "oracle"):
        M = np.full((len(SOURCES), len(TARGETS)), np.nan)
        ann = {}
        for (src, tgt), cell in cells.items():
            summ = cell["summary"]
            L = summ["source_selected_layer"] if mode == "source_val" else summ["oracle_best_layer"]
            if L is None:
                continue
            rg = ood._relative_gain_cell(cell["bc"], L)
            M[SOURCES.index(src), TARGETS.index(tgt)] = rg["relative_gain_pct"]
            ann[(src, tgt)] = (L, summ["oracle_best_layer"], rg)
        if not ann:
            continue
        vmax = float(np.nanmax(np.abs(M))) or 1.0
        fig, ax = plt.subplots(figsize=(2.7 * len(TARGETS) + 3, 2.1 * len(SOURCES) + 2))
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        for (src, tgt), (L, oracle_L, rg) in ann.items():
            sig = rg["excludes_zero"]
            if mode == "source_val":
                head, tail = f"sel {XLABELS[L]}", f"\n(oracle {XLABELS[oracle_L]})"
            else:
                head, tail = f"oracle {XLABELS[L]}", ""
            ax.text(TARGETS.index(tgt), SOURCES.index(src),
                    ("★ " if sig else "") + f"{head}\n{rg['relative_gain_pct']:+.1f}%\n"
                    f"95% CI [{rg['ci_lo']:+.1f},{rg['ci_hi']:+.1f}]" + tail, ha="center", va="center",
                    fontsize=7.5, fontweight="bold" if sig else "normal")
        ax.set_xticks(range(len(TARGETS))); ax.set_xticklabels([TGT_SHORT[t] for t in TARGETS])
        ax.set_yticks(range(len(SOURCES))); ax.set_yticklabels([SRC_SHORT[s] for s in SOURCES])
        ax.set_xlabel("pretraining-OOD target"); ax.set_ylabel("ID source (probe)")
        if mode == "source_val":
            title = (f"PRIMARY — relative improvement over L12 at the SOURCE-VALIDATION-selected layer  "
                     f"[{qset}, seed {seed}]\nlayer chosen on each source's own ID validation ONLY (never "
                     "touches OOD-target loss); warm = beats L12, cool = worse; ★/bold = 95% paired CI "
                     "excludes 0.\ngains may be NEGATIVE; CIs descriptive (layer fixed before the bootstrap)")
            prefix = "source_val_relative_gain_heatmap_4x3"
        else:
            title = (f"DIAGNOSTIC (oracle) — relative improvement over L12 at the TARGET-TEST-best layer  "
                     f"[{qset}, seed {seed}]\noptimistically biased (L = target-test argmin, ≥0 by "
                     "construction); ★/bold = CI excludes 0.\nNOT the primary result — see the source_val heatmap")
            prefix = "oracle_relative_gain_heatmap_4x3"
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Relative improvement over L12 (%)")
        fig.tight_layout()
        out = FIG_DIR / f"{prefix}__{qset}.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig); print(f"  [saved] {out}")


def make_source_selected_figure(cells, qset, seed):
    """Part B: each source's SOURCE-SELECTED layer (fixed from source ID val) evaluated on all 3
    OOD targets — grouped bars of relative gain over L12 with the paired 95% CI as error bars."""
    fig, ax = plt.subplots(figsize=(9.5, 6.0))
    width = 0.8 / len(TARGETS)
    xbase = np.arange(len(SOURCES))
    for ti, tgt in enumerate(TARGETS):
        vals, los, his, labs = [], [], [], []
        for src in SOURCES:
            cell = cells.get((src, tgt))
            L = cell["summary"]["source_selected_layer"] if cell else None
            if cell is None or L is None:
                vals.append(np.nan); los.append(0); his.append(0); continue
            rg = ood._relative_gain_cell(cell["bc"], L)
            vals.append(rg["relative_gain_pct"])
            los.append(rg["relative_gain_pct"] - rg["ci_lo"]); his.append(rg["ci_hi"] - rg["relative_gain_pct"])
        ax.bar(xbase + ti * width, vals, width, yerr=[los, his], capsize=3,
               label=TGT_SHORT[tgt], edgecolor="k", lw=0.5)
    ax.axhline(0, color="0.4", lw=1)
    ax.set_xticks(xbase + width * (len(TARGETS) - 1) / 2)
    ax.set_xticklabels([f"{SRC_SHORT[s]}\n(L{cells.get((s, TARGETS[0]),{}).get('summary',{}).get('source_selected_layer','?')})"
                        for s in SOURCES], fontsize=8)
    ax.set_ylabel("Relative improvement over L12 (%)")
    ax.set_title(f"Part B — SOURCE-SELECTED layer (fair OOD) on pretraining-OOD targets  [{qset}, seed {seed}]\n"
                 "layer chosen on each source's own ID validation loss (never OOD); error bars = paired "
                 "95% cluster-bootstrap CI", fontsize=9)
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"source_selected_layer__{qset}.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"  [saved] {out}")

    # companion table
    rows = []
    for src in SOURCES:
        for tgt in TARGETS:
            cell = cells.get((src, tgt))
            if cell is None:
                continue
            r = _cell_record(src, tgt, cell, "source_selected")
            if r:
                rows.append(r)
    with open(OOD_PT_DIR / f"source_selected_layer__{qset}.csv", "w", newline="") as f:
        if rows:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys())); wr.writeheader(); wr.writerows(rows)


def make_baseline_comparison(cells, qset, seed):
    """Per target: MASE of the source-selected probe (per source) vs target-only
    baselines (native / seasonal / last). LOWER = better. Baselines identical within a target."""
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(6.2 * len(TARGETS), 5.2), squeeze=False)
    for ci, tgt in enumerate(TARGETS):
        ax = axes[0][ci]
        base = json.load(open(TGT_DIR / f"{tgt}.json"))["baselines_mase"]
        labels, vals, colors = [], [], []
        for src in SOURCES:
            cell = cells.get((src, tgt))
            if cell is None:
                continue
            ss = cell["summary"]["source_selected_layer"]
            labels.append(f"{SRC_SHORT[src]}\nsel L{ss}"); vals.append(float(cell["mase"][ss]))
            colors.append(ID_STYLE.get(src, {}).get("color", "#333"))
        labels += ["native", "seasonal", "last"]
        vals += [base["native_chronos2"], base["seasonal_naive"], base["last_value"]]
        colors += ["#444", "#999", "#c8c8c8"]
        ax.bar(range(len(vals)), vals, color=colors, edgecolor="k", lw=0.5)
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, fontsize=6, rotation=45, ha="right")
        ax.set_title(f"{TGT_SHORT[tgt]}  [OOD]", fontsize=10); ax.grid(axis="y", alpha=0.3)
        if ci == 0:
            ax.set_ylabel("MASE (test, lower = better)")
    fig.suptitle(f"Transferred-probe MASE vs target baselines  [{qset}, seed {seed}]  —  probe bars = "
                 "source-validation-selected (PRIMARY); native/seasonal/last are target-only references",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    out = FIG_DIR / f"baseline_comparison__{qset}.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig); print(f"  [saved] {out}")


def make_three_group_plot(cells, qset, seed):
    """Descriptive 3-group view of the relative gain over L12, in BOTH selection views:
       (1) ID cells (extended_v2 diagonal), (2) cross-dataset pretraining-ID cells (extended_v2
       off-diagonal), (3) pretraining-OOD cells (this experiment). PRIMARY uses the source-validation-
       selected layer (groups 1&2 read from the 4x4 source_val summary, group 3 from each cell's
       source_selected layer); DIAGNOSTIC uses the oracle target-test-best layer (biased). Individual
       cells + group mean (solid) + median (dashed). Cells are correlated -> descriptive only."""
    ref_stem = {"source_val": "source_val_relative_gain_summary",
                "oracle": "oracle_relative_gain_summary"}
    for mode in ("source_val", "oracle"):
        groups = {"ID\n(in-dataset)": [], "Cross-dataset\n(pretraining-ID)": [], "Pretraining-OOD": []}
        # groups 1&2 from the committed extended_v2 4x4 relative-gain summary (SAME selection view)
        ref = config.OUT_DIR / SOURCE_SET / "ood_transfer" / f"{ref_stem[mode]}_{qset}.json"
        if ref.exists():
            for r in json.load(open(ref)):
                key = "ID\n(in-dataset)" if r["id_or_ood"] == "ID" else "Cross-dataset\n(pretraining-ID)"
                groups[key].append(r["relative_gain_pct"])
        else:
            print(f"  [warn] {ref} absent — 3-group ({mode}) plot shows only the OOD group")
        for (src, tgt), cell in cells.items():
            summ = cell["summary"]
            L = summ["source_selected_layer"] if mode == "source_val" else summ["oracle_best_layer"]
            if L is None:
                continue
            groups["Pretraining-OOD"].append(ood._relative_gain_cell(cell["bc"], L)["relative_gain_pct"])

        fig, ax = plt.subplots(figsize=(9.0, 6.5))
        rng = np.random.default_rng(SEED)
        palette = ["#4c72b0", "#55a868", "#dd8452"]
        for gi, (label, vals) in enumerate(groups.items()):
            if not vals:
                continue
            v = np.array(vals, float)
            ax.scatter(gi + rng.uniform(-0.08, 0.08, len(v)), v, s=55, color=palette[gi],
                       edgecolor="k", lw=0.5, zorder=3, alpha=0.9)
            ax.hlines(v.mean(), gi - 0.25, gi + 0.25, color=palette[gi], lw=2.6, zorder=4,
                      label=f"{label.split(chr(10))[0]} mean {v.mean():+.1f}%")
            ax.hlines(np.median(v), gi - 0.25, gi + 0.25, color=palette[gi], lw=1.5, ls="--", zorder=4)
        ax.axhline(0, color="0.5", ls=":", lw=1)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([f"{k}\n(n={len(v)})" for k, v in groups.items()], fontsize=9)
        if mode == "source_val":
            ax.set_ylabel("Relative improvement over L12 (%) — source-validation-selected layer")
            title = (f"Early-layer advantage by pretraining-distribution group (PRIMARY: source-"
                     f"validation-selected)  [{qset}, seed {seed}]\neach point = one source→target cell "
                     "(fair layer, no target data); solid = mean, dashed = median; gains may be negative.\n"
                     "DESCRIPTIVE — cells correlated (shared source probes / targets); no cross-cell test.")
            prefix = "source_val_three_group_relative_gain"
        else:
            ax.set_ylabel("Relative improvement over L12 (%) — oracle target-test-best layer")
            title = (f"Early-layer advantage by pretraining-distribution group (DIAGNOSTIC: oracle test-"
                     f"best)  [{qset}, seed {seed}]\noptimistically biased; each point = one source→target "
                     "cell; solid = mean, dashed = median.\nDESCRIPTIVE — NOT the primary result.")
            prefix = "oracle_three_group_relative_gain"
        ax.set_title(title, fontsize=9)
        ax.legend(fontsize=8, loc="best"); ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        out = FIG_DIR / f"{prefix}__{qset}.png"; fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig); print(f"  [saved] {out}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Pretraining-OOD probe transfer (4 ID sources x 3 OOD "
                                             "targets); frozen source probes, no target adaptation.")
    ap.add_argument("--sources", nargs="+", default=None, help="subset of sources (default: all 4).")
    ap.add_argument("--target", default=None, help="evaluate a single target (default: all 3).")
    ap.add_argument("--source-set", default=SOURCE_SET,
                    help="dataset set whose frozen 4×4 source probes to transfer (default: "
                         "extended_v2; pass extended_v3_rolling for the rolling-trained sources). "
                         "Also namespaces outputs to results/<source-set>/ood_pretrain_transfer/.")
    ap.add_argument("--quantile-set", choices=sorted(QUANTILE_SETS), default="q9")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available).")
    ap.add_argument("--figure-only", action="store_true", help="aggregate saved cells only (CPU).")
    return ap.parse_args(argv)


def main():
    args = _parse_args()
    global SOURCE_SET
    SOURCE_SET = args.source_set
    # source probes live under SOURCE_SET; point config there so ood.load_checkpoints resolves them
    # AND outputs namespace to results/<SOURCE_SET>/ood_pretrain_transfer/.
    config.set_dataset_set(SOURCE_SET)
    ood._derive_dirs(); ood._derive_datasets()
    _derive_dirs()
    qset = args.quantile_set
    quantiles = QUANTILE_SETS[qset]
    if median_index(quantiles) is None:
        raise SystemExit(f"{qset} has no 0.5 level — MASE/median undefined")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[config] sources under {SOURCE_SET}; targets={TARGETS}; qset={qset} Q={len(quantiles)} "
          f"device={device}\n[config] outputs -> {OOD_PT_DIR}")

    if not args.figure_only:
        sources = args.sources or SOURCES
        targets = [args.target] if args.target else TARGETS
        for tgt in targets:
            eval_target(tgt, sources, qset, quantiles, SEED, device)
    aggregate(qset, SEED)


if __name__ == "__main__":
    main()
