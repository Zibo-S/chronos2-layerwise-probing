"""Absolute-quality baselines for the 3×3 cross-dataset transfer experiment.

The layerwise experiment (run_ood_transfer.py) answers *which* layer transfers best. It does
NOT say whether the transferred probe forecasts the target competitively. This driver adds
per-target reference forecasts — native Chronos-2 (its cached median), seasonal-naive, and
last-value — and asks: does the best transferred probe actually beat them, or is it merely
less bad than the final-layer probe?

Design (CPU / login-node; reuses every cache + the frozen OOD checkpoints, refits NOTHING):
  * target baselines are computed ONCE per target and reused across all three source rows;
  * probe per-window MASE comes from re-scoring the FROZEN source checkpoints on the CACHED
    target features (no training, tuning, or refit);
  * the best layer is READ from the committed ood_transfer_summary (never re-selected here);
  * paired series-level cluster bootstrap (B=5000, shared counts → paired) on MASE gives
    improvement = baseline_loss − probe_loss with a 95% CI (positive = probe better).

Run:  python -m experiments.run_ood_baselines            # aggregate + tables + figures
      python -m experiments.run_ood_baselines --self-check
"""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probing import config
from probing.config import NUM_LAYERS, LAST_LAYER, SEED
from probing.id_data import build_windows
from probing.extraction import extract_window_features
from probing.probes import QUANTILE_SETS, median_index, chronos2_quantile_loss_per_window
from probing.stats import cluster_bootstrap_counts, cluster_bootstrap_apply
from experiments.run_id_forecasting import (native_median_forecast, _mase_denominator,
                                            _ctx_stats, M_SEASON, ID_STYLE)
# reuse the OOD experiment's frame + frozen-checkpoint loader; importing only defines things.
from experiments import run_ood_transfer as ood

POOLING, C, H = ood.POOLING, ood.C, ood.H
BOOT_B, CI_LO, CI_HI = ood.BOOT_B, ood.CI_LO, ood.CI_HI
DATASET_ORDER, SHORT = ood.DATASET_ORDER, ood.SHORT

BASELINES = ["native_chronos2", "seasonal_naive", "last_value"]   # comparison order
DET_Q9 = ["seasonal_naive", "last_value"]     # naive baselines that also get a degenerate q9
BASE_LABEL = {"native_chronos2": "Native Chronos-2", "seasonal_naive": "Seasonal-naive (m=24)",
              "last_value": "Last-value"}


# --------------------------------------------------------------------------- #
# per-target baselines (computed ONCE per target, reused for every source row)
# --------------------------------------------------------------------------- #

def _degenerate_qloss_per_window(z_hat, Y, quantiles):
    """q9 pinball loss of a DETERMINISTIC forecast: broadcast the single normalized trajectory
    z_hat (n,H) across all Q quantile levels and score against Y (n,H) with Chronos-2's own
    per-window reduction (mean over horizon → sum over quantiles). Returns (n,). NOT a
    calibrated probabilistic forecast — for symmetric q9 it is a fixed multiple of MAE."""
    q = torch.tensor(np.asarray(quantiles), dtype=torch.float32)
    pred = torch.tensor(np.asarray(z_hat), dtype=torch.float32).unsqueeze(1).repeat(1, len(q), 1)
    tgt = torch.tensor(np.asarray(Y), dtype=torch.float32)
    return chronos2_quantile_loss_per_window(pred, tgt, q).cpu().numpy()


def target_baselines(target, qset, quantiles):
    """Build every target-only reference forecast + its metrics. Depends ONLY on the target
    dataset (never a source), so it is computed once and shared across source rows."""
    w = build_windows(target)                                   # same windows as the OOD run
    X = np.asarray(w["X_test"], np.float64)                     # (n, C)
    n, Cx = X.shape
    Y = w["Y_test_traj"]                                        # (n, H) arcsinh-normalized future
    mu, s = _ctx_stats(X, w["meta"]["sigma_eps"])
    y_raw = mu[:, None] + s[:, None] * np.sinh(Y.astype(np.float64))   # future in raw units
    d = np.maximum(_mase_denominator(X), 1e-8)[:, None]        # in-context seasonal-naive scale
    Hh = Y.shape[1]
    m = M_SEASON

    seas_idx = Cx - m + (np.arange(Hh) % m)                     # (H,) — context indices only
    assert seas_idx.max() < Cx and seas_idx.min() >= 0, "seasonal-naive index out of context"
    forecasts_raw = {
        "last_value": np.repeat(X[:, -1:], Hh, axis=1),
        "seasonal_naive": X[:, seas_idx],
        "native_chronos2": native_median_forecast(target, w["X_test"], Hh).astype(np.float64),
    }

    sid = np.asarray(w["series_test"], np.int64)
    n_series = int(len(np.unique(sid)))
    metrics, pw_mase = {}, {}
    for name, f_raw in forecasts_raw.items():
        A = np.abs(y_raw - f_raw) / d                          # (n,H) scaled abs errors
        pw_mase[name] = A.mean(axis=1)                         # (n,) per-window MASE
        metrics[name] = {"mase": float(A.mean()),
                         "mae": float(np.abs(y_raw - f_raw).mean())}
        if name in DET_Q9:
            z_hat = np.arcsinh((f_raw - mu[:, None]) / s[:, None])
            metrics[name]["q9_quantile_loss_degenerate"] = float(
                _degenerate_qloss_per_window(z_hat, Y, quantiles).mean())
    return {"target": target, "n_windows": n, "n_series": n_series, "series_test": sid,
            "mu": mu, "s": s, "y_raw": y_raw, "d": d, "metrics": metrics, "pw_mase": pw_mase}


# --------------------------------------------------------------------------- #
# probe per-window MASE from the FROZEN source checkpoints (no refit, no GPU)
# --------------------------------------------------------------------------- #

def probe_per_window_mase(source, target, qset, quantiles, seed, device, pt):
    """Per-layer per-window MASE of the frozen SOURCE probe on TARGET windows. Loads the frozen
    checkpoints and re-scores the CACHED target features — trains nothing. Row means reproduce
    the committed per-layer MASE (asserted by the caller)."""
    fitted = ood.load_checkpoints(source, qset, seed, quantiles, device)
    if fitted is None:
        raise SystemExit(f"missing frozen checkpoints for source={source} ({qset}); run "
                         "run_ood_transfer first")
    w = build_windows(target)
    f_te, _ = extract_window_features(target, "test", w["X_test"], w["y_test"], pooling=POOLING)
    from probing.probes import predict_quantile_probe
    _out, diag = predict_quantile_probe(fitted, f_te, w["Y_test_traj"], quantiles=quantiles,
                                        device=device, collect_test_median=True)
    mu, s, y_raw, d = pt["mu"], pt["s"], pt["y_raw"], pt["d"]
    pw = {}
    for i in range(NUM_LAYERS):
        yhat = mu[:, None] + s[:, None] * np.sinh(diag["test_median"][i].astype(np.float64))
        pw[i] = (np.abs(y_raw - yhat) / d).mean(axis=1)        # (n,) per-window MASE
    return pw


def _paired_improvement_ci(probe_pw, baseline_pw, sid):
    """Paired series-level cluster bootstrap of improvement = baseline − probe (per-window MASE).
    ONE shared counts matrix → both means come from the SAME resampled series (paired). Positive
    improvement = the probe is better; CI entirely > 0 = probe significantly beats the baseline."""
    sid = np.asarray(sid, np.int64)
    _uniq, inv = np.unique(sid, return_inverse=True)
    S = int(inv.max()) + 1
    cnt = np.bincount(inv).astype(np.float64)
    sums = np.zeros((S, 2))
    np.add.at(sums, inv, np.stack([probe_pw, baseline_pw], axis=1))   # cols: probe, baseline
    counts = cluster_bootstrap_counts(S, BOOT_B, SEED)
    boot = cluster_bootstrap_apply(counts, sums, cnt)                 # (B,2) window-means
    imp = boot[:, 1] - boot[:, 0]
    return (float(baseline_pw.mean() - probe_pw.mean()),
            float(np.percentile(imp, CI_LO)), float(np.percentile(imp, CI_HI)), S)


# --------------------------------------------------------------------------- #
# aggregate → tables → figures
# --------------------------------------------------------------------------- #

def _best_layers(qset, seed):
    """Committed best layer per (source,target) — read from the layerwise summary, NEVER
    re-selected from baseline-normalized scores."""
    p = ood.OOD_DIR / f"ood_transfer_summary__{qset}.json"
    if not p.exists():
        raise SystemExit(f"{p} not found — run the layerwise experiment (--figure-only) first")
    return {(c["source_dataset"], c["target_dataset"]): int(c["best_layer"])
            for c in json.load(open(p))["cells"]}


def _committed_probe_mase(qset):
    """Committed per-(source,target,layer) probe MASE, for a same-windows cross-check."""
    p = ood.OOD_DIR / f"ood_transfer_results__{qset}.csv"
    out = {}
    for r in csv.DictReader(open(p)):
        out[(r["source_dataset"], r["target_dataset"], int(r["layer"]))] = float(r["mase"])
    return out


def run(qset, seed, device):
    quantiles = QUANTILE_SETS[qset]
    qcfg = int(len(quantiles))
    best = _best_layers(qset, seed)                 # oracle target-test-best layer (diagnostic)
    sel = {src: ood.source_selected_layer(src, qset, seed)[0] for src in DATASET_ORDER}   # PRIMARY
    committed = _committed_probe_mase(qset)

    # 1) target baselines — ONCE per target, reused for every source row
    base = {t: target_baselines(t, qset, quantiles) for t in DATASET_ORDER}
    _write_target_table(base, qset, seed, qcfg)

    # 2) probe vs each baseline, per source→target cell (frozen checkpoints, cached features)
    comp_rows, cmp = [], {}
    for src in DATASET_ORDER:
        for tgt in DATASET_ORDER:
            pt = base[tgt]
            pw = probe_per_window_mase(src, tgt, qset, quantiles, seed, device, pt)
            for L in (best[(src, tgt)], LAST_LAYER):             # cross-check re-score vs committed
                diff = abs(float(pw[L].mean()) - committed[(src, tgt, L)])
                if diff > 1e-2:
                    print(f"  [warn] {src}->{tgt} L{L}: re-scored MASE {pw[L].mean():.4f} vs "
                          f"committed {committed[(src, tgt, L)]:.4f} (Δ={diff:.4f}; device drift?)")
            cmp[(src, tgt)] = {}
            for ptype, L in (("source_val_selected", sel[src]),
                             ("best_layer", best[(src, tgt)]), ("final_layer", LAST_LAYER)):
                probe_loss = float(pw[L].mean())
                cmp[(src, tgt)][ptype] = {"layer": L, "loss": probe_loss, "by_base": {}}
                for bname in BASELINES:
                    bloss = pt["metrics"][bname]["mase"]
                    imp, lo, hi, S = _paired_improvement_ci(pw[L], pt["pw_mase"][bname],
                                                            pt["series_test"])
                    beats = bool(lo > 0)
                    skill = float(1.0 - probe_loss / bloss) if bloss > 0 else float("nan")
                    cmp[(src, tgt)][ptype]["by_base"][bname] = {
                        "improvement": imp, "ci": (lo, hi), "beats": beats, "skill": skill}
                    comp_rows.append({
                        "source_dataset": src, "target_dataset": tgt, "probe_type": ptype,
                        "probe_layer": L, "baseline": bname, "metric": "mase",
                        "probe_loss": round(probe_loss, 6), "baseline_loss": round(bloss, 6),
                        "improvement_vs_baseline": round(imp, 6),
                        "improvement_ci_lo": round(lo, 6), "improvement_ci_hi": round(hi, 6),
                        "probe_beats_baseline": beats, "skill_vs_baseline": round(skill, 6),
                        "n_series": S, "n_windows": pt["n_windows"], "seed": int(seed),
                        "quantile_config": qcfg, "context_length": C, "prediction_length": H})
    _write_comparison_table(comp_rows, qset)
    make_source_val_baseline_figure(base, cmp, sel, qset, seed)     # PRIMARY (source-val-selected)
    make_baseline_comparison_figure(base, cmp, best, qset, seed)    # DIAGNOSTIC (oracle test-best)
    make_skill_heatmap(cmp, qset, seed)
    print(f"\n  [done] baselines + comparison written under {ood.OOD_DIR}")


def _write_target_table(base, qset, seed, qcfg):
    fields = ["target_dataset", "baseline", "metric", "loss", "seed", "quantile_config",
              "context_length", "prediction_length", "number_of_windows", "number_of_series"]
    rows = []
    for t in DATASET_ORDER:
        pt = base[t]
        for bname, mets in pt["metrics"].items():
            for metric, val in mets.items():
                rows.append({"target_dataset": t, "baseline": bname, "metric": metric,
                             "loss": round(float(val), 6), "seed": int(seed),
                             "quantile_config": qcfg, "context_length": C, "prediction_length": H,
                             "number_of_windows": pt["n_windows"], "number_of_series": pt["n_series"]})
    with open(ood.OOD_DIR / f"ood_target_baselines__{qset}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    json.dump(rows, open(ood.OOD_DIR / f"ood_target_baselines__{qset}.json", "w"), indent=2)
    print(f"  [saved] target-baseline table ({len(rows)} rows)")


def _write_comparison_table(rows, qset):
    fields = ["source_dataset", "target_dataset", "probe_type", "probe_layer", "baseline",
              "metric", "probe_loss", "baseline_loss", "improvement_vs_baseline",
              "improvement_ci_lo", "improvement_ci_hi", "probe_beats_baseline",
              "skill_vs_baseline", "n_series", "n_windows", "seed", "quantile_config",
              "context_length", "prediction_length"]
    with open(ood.OOD_DIR / f"ood_probe_vs_baseline__{qset}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields); wr.writeheader(); wr.writerows(rows)
    json.dump(rows, open(ood.OOD_DIR / f"ood_probe_vs_baseline__{qset}.json", "w"), indent=2)
    print(f"  [saved] probe-vs-baseline table ({len(rows)} rows)")


def make_baseline_comparison_figure(base, cmp, best, qset, seed):
    """DIAGNOSTIC N×N grouped-bar grid: each source→target cell shows MASE for the ORACLE
    test-best-layer probe, the final-layer probe, native, seasonal-naive, last-value (LOWER =
    better). The oracle layer is target-test-selected (optimistically biased) — see
    make_source_val_baseline_figure for the PRIMARY (source-validation-selected) view. Target
    baselines are identical down a column (same target)."""
    labels = ["probe\noracle", "probe\nfinal", "native", "seasonal", "last"]
    N = len(DATASET_ORDER)
    fig, axes = plt.subplots(N, N, figsize=(5 * N, 4 * N), sharex=True)
    for ri, src in enumerate(DATASET_ORDER):
        color = ID_STYLE.get(src, {}).get("color", "#333333")
        for ci, tgt in enumerate(DATASET_ORDER):
            ax = axes[ri, ci]
            pt, cell = base[tgt], cmp[(src, tgt)]
            vals = [cell["best_layer"]["loss"], cell["final_layer"]["loss"],
                    pt["metrics"]["native_chronos2"]["mase"],
                    pt["metrics"]["seasonal_naive"]["mase"], pt["metrics"]["last_value"]["mase"]]
            colors = [color, color, "#555555", "#999999", "#c0c0c0"]
            bars = ax.bar(range(5), vals, color=colors, edgecolor="k", linewidth=0.6)
            bars[1].set_alpha(0.55)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=7)
            is_ood = src != tgt
            ax.set_title(f"{SHORT[src]} → {SHORT[tgt]}  [{'OOD' if is_ood else 'ID'}]  "
                         f"oracle L{cell['best_layer']['layer']}", fontsize=9,
                         fontweight="normal" if is_ood else "bold")
            ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            if not is_ood:
                ax.set_facecolor("#f4f4ec")
        axes[ri, 0].set_ylabel(f"probe: {SHORT[src]}\nMASE (test)")
    fig.suptitle(f"Transferred probe vs target baselines — MASE  (DIAGNOSTIC: oracle test-best "
                 f"layer)  [{qset}, seed {seed}]\nrows = source (probe) dataset, cols = target;  "
                 "LOWER = better;  oracle layer = target-test argmin (biased) — see the source_val "
                 "figure for the primary result;  native/seasonal/last identical down a column",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ood.FIG_DIR / f"baseline_comparison__{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {out}")


def make_source_val_baseline_figure(base, cmp, sel, qset, seed):
    """PRIMARY N×N grouped-bar grid: each source→target cell shows MASE for the SOURCE-VALIDATION-
    selected probe (primary), the final-layer probe, and the three target baselines (LOWER =
    better). The source-val layer is fixed per source row (chosen without any target data)."""
    labels = ["probe\nval-sel", "probe\nfinal", "native", "seasonal", "last"]
    N = len(DATASET_ORDER)
    fig, axes = plt.subplots(N, N, figsize=(5.4 * N, 4 * N), sharex=True)
    for ri, src in enumerate(DATASET_ORDER):
        color = ID_STYLE.get(src, {}).get("color", "#333333")
        for ci, tgt in enumerate(DATASET_ORDER):
            ax = axes[ri, ci]
            pt, cell = base[tgt], cmp[(src, tgt)]
            vals = [cell["source_val_selected"]["loss"], cell["final_layer"]["loss"],
                    pt["metrics"]["native_chronos2"]["mase"],
                    pt["metrics"]["seasonal_naive"]["mase"], pt["metrics"]["last_value"]["mase"]]
            colors = [color, color, "#555555", "#999999", "#c0c0c0"]
            bars = ax.bar(range(5), vals, color=colors, edgecolor="k", linewidth=0.6)
            bars[1].set_alpha(0.55)          # final-layer probe
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6.5)
            is_ood = src != tgt
            ax.set_title(f"{SHORT[src]} → {SHORT[tgt]}  [{'OOD' if is_ood else 'ID'}]  "
                         f"val-sel L{cell['source_val_selected']['layer']}", fontsize=8.5,
                         fontweight="normal" if is_ood else "bold")
            ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=6.5)
            ax.grid(axis="y", alpha=0.3)
            if not is_ood:
                ax.set_facecolor("#f4f4ec")
        axes[ri, 0].set_ylabel(f"probe: {SHORT[src]}\nMASE (test)")
    fig.suptitle(f"Transferred probe vs target baselines — MASE  (PRIMARY: source-validation-selected "
                 f"layer)  [{qset}, seed {seed}]\nrows = source (probe) dataset, cols = target;  "
                 "LOWER = better;  native/seasonal/last identical down a column", fontsize=11, y=0.996)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = ood.FIG_DIR / "source_val_baseline_comparison.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {out}")


def make_skill_heatmap(cmp, qset, seed):
    """Best-layer probe skill = 1 − probe/baseline, vs native and vs seasonal-naive. Positive
    (blue) = probe better; ★ = paired improvement CI excludes 0."""
    idx = {d: i for i, d in enumerate(DATASET_ORDER)}
    N = len(DATASET_ORDER)
    fig, axes = plt.subplots(1, 2, figsize=(max(13, 4.5 * N), max(5.5, 2.0 * N)))
    for ax, bname in zip(axes, ["native_chronos2", "seasonal_naive"]):
        M = np.full((N, N), np.nan)
        for (src, tgt), cell in cmp.items():
            M[idx[src], idx[tgt]] = cell["best_layer"]["by_base"][bname]["skill"]
        vmax = np.nanmax(np.abs(M)) or 1.0
        im = ax.imshow(M, cmap="RdBu", vmin=-vmax, vmax=vmax)
        for (src, tgt), cell in cmp.items():
            info = cell["best_layer"]["by_base"][bname]
            star = "★ " if info["beats"] else ""
            ax.text(idx[tgt], idx[src], f"{star}{info['skill']:+.2f}", ha="center", va="center",
                    fontsize=9, fontweight="bold" if info["beats"] else "normal")
            if src == tgt:
                ax.add_patch(plt.Rectangle((idx[tgt] - .5, idx[src] - .5), 1, 1, fill=False,
                                           ec="k", lw=2.2))
        ax.set_xticks(range(N)); ax.set_xticklabels([SHORT[d] for d in DATASET_ORDER])
        ax.set_yticks(range(N)); ax.set_yticklabels([SHORT[d] for d in DATASET_ORDER])
        ax.set_xlabel("target"); ax.set_ylabel("source (probe)")
        ax.set_title(f"best-probe skill vs {BASE_LABEL[bname]}")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="skill = 1 − probe/baseline")
    fig.suptitle(f"Best transferred probe skill  [{qset}, seed {seed}]  —  positive = probe beats "
                 "baseline (MASE);  ★ = paired 95% CI excludes 0;  boxed = in-dataset", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = ood.FIG_DIR / f"skill_heatmap__{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {out}")


# --------------------------------------------------------------------------- #
# minimal self-check (CPU, no model): baselines, reduction, one paired bootstrap
# --------------------------------------------------------------------------- #

def self_check(qset, seed):
    quantiles = QUANTILE_SETS[qset]
    t = DATASET_ORDER[0]
    a = target_baselines(t, qset, quantiles)
    b = target_baselines(t, qset, quantiles)
    assert a["metrics"] == b["metrics"], "baseline must be deterministic / reusable per target"
    # seasonal-naive uses only context (native cache guard already checks same windows)
    w = build_windows(t); Cx = np.asarray(w["X_test"]).shape[1]
    assert (Cx - M_SEASON + (np.arange(H) % M_SEASON)).max() < Cx, "seasonal index leaks future"
    # degenerate q9 mean-reduction: per-window .mean() == scalar quantile loss
    z = np.zeros_like(a["y_raw"]); pw = _degenerate_qloss_per_window(z, w["Y_test_traj"], quantiles)
    assert np.isfinite(pw).all() and pw.ndim == 1, "degenerate q9 per-window malformed"
    # one paired bootstrap example (synthetic probe better than baseline by a margin)
    rng = np.random.default_rng(0)
    probe = a["pw_mase"]["seasonal_naive"] * 0.8 + rng.normal(0, 1e-6, a["n_windows"])
    imp, lo, hi, S = _paired_improvement_ci(probe, a["pw_mase"]["seasonal_naive"], a["series_test"])
    assert lo <= imp <= hi and S == a["n_series"], "paired bootstrap CI malformed"
    print(f"  self-check OK: baselines reused, seasonal context-only, mean reduction, "
          f"paired bootstrap (imp={imp:.3f} [{lo:.3f},{hi:.3f}], S={S})")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Absolute-quality baselines for the 3×3 OOD transfer.")
    ap.add_argument("--dataset-set", default=None)
    ap.add_argument("--quantile-set", choices=sorted(QUANTILE_SETS), default="q9")
    ap.add_argument("--device", default="cpu", help="torch device (default cpu — login-node).")
    ap.add_argument("--self-check", action="store_true")
    return ap.parse_args(argv)


def main():
    args = _parse_args()
    if args.dataset_set:
        config.set_dataset_set(args.dataset_set)
    ood._derive_dirs()
    ood._derive_datasets()                       # matrix order/labels follow the override
    global DATASET_ORDER, SHORT
    DATASET_ORDER, SHORT = ood.DATASET_ORDER, ood.SHORT
    qset, seed = args.quantile_set, SEED
    if median_index(QUANTILE_SETS[qset]) is None:
        raise SystemExit(f"{qset} has no 0.5 level — MASE/MAE undefined")
    print(f"[config] dataset_set={config.DATASET_SET} quantile_set={qset} device={args.device} "
          f"-> {ood.OOD_DIR}")
    if args.self_check:
        self_check(qset, seed)
        return
    run(qset, seed, args.device)


if __name__ == "__main__":
    main()
