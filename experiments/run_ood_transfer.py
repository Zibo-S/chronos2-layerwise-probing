"""Strict cross-dataset OOD forecasting-probe transfer (exploratory pilot).

Question: does the intermediate-layer advantage over the final Chronos-2 layer SURVIVE when
the linear forecasting probe is trained on one dataset and evaluated, frozen, on another?

Protocol (per source dataset, per seed, per layer):
  1. build SOURCE windows, extract SOURCE-train content-pooled features (Embed + L1..L12),
  2. train the linear quantile probe on the SOURCE train split only (fit_quantile_probe:
     per-layer wd selected on the SOURCE 80/20 carve, then refit on FULL source train),
  3. FREEZE the probe (scaler + Linear + selected wd) and save one checkpoint per layer,
  4. reuse that SAME probe to score every TARGET's test split — no target training, tuning,
     early stopping, calibration, or refitting; targets enter only as frozen features from
     target contexts + the target futures they are scored against.

Diagonal (source == target) reproduces the ordinary in-dataset evaluation; the six
off-diagonal cells are strict cross-dataset transfer. is_ood = source_dataset != target_dataset.

This experiment uses --quantile-set q9 (recorded explicitly in every output). Everything else
is inherited UNCHANGED from the per-dataset pipeline: frozen amazon/chronos-2, 13 depths
(Embed L0 + L1..L12), content-token mean pooling, C=512, H=64, arcsinh context normalization,
Chronos-2 quantile loss, in-context seasonal-naive MASE (m=24), seed 0. Feature/native caches
are the existing per-(dataset,split) files, so with a warm cache this is CPU / login-node only.

Run (one source per job — there are only THREE unique source-training configurations):
    python -m experiments.run_ood_transfer --source-dataset monash_electricity_hourly
    python -m experiments.run_ood_transfer --source-dataset monash_kdd_cup_2018
    python -m experiments.run_ood_transfer --source-dataset uber_tlc_hourly
    python -m experiments.run_ood_transfer --figure-only     # assemble the 3x3 grid + summary
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

from probing import config, id_data
from probing.config import NUM_LAYERS, LAST_LAYER, SEED
from probing.id_data import build_windows
from probing.extraction import extract_window_features
from probing.probes import (fit_quantile_probe, predict_quantile_probe, QUANTILE_SETS,
                            median_index)
from probing.stats import cluster_bootstrap_counts, cluster_bootstrap_apply
# reuse the per-dataset pipeline's MASE definition + cached native forecast + context inverse,
# so the diagonal MASE matches the committed per-dataset numbers exactly (importing this module
# only defines functions; it does not run the per-dataset main()).
from experiments.run_id_forecasting import compute_mase, _ctx_stats, M_SEASON, ID_STYLE

# ---- fixed experimental frame (inherited from the per-dataset pipeline; DO NOT diverge) ----
POOLING = "content"            # content-token mean pooling (the primary pooled readout)
C, H = 512, 64                 # context / horizon (build_windows defaults)
QUANTILE_EPOCHS = 300          # same probe training budget as run_id_forecasting
WD_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)   # same per-layer weight-decay grid
SEEDS = (SEED,)                # the repo is single-seed (SEED=0); uncertainty = series bootstrap.
                               # seed is carried as a schema column so multi-seed can be added
                               # later (would require threading the seed into the probe init).
BOOT_B = 5000                  # series-level cluster-bootstrap replicates for the CI bands
CI_LO, CI_HI = 2.5, 97.5

# 3x3 matrix order + short panel labels
DATASET_ORDER = ["monash_electricity_hourly", "monash_kdd_cup_2018", "uber_tlc_hourly"]
SHORT = {"monash_electricity_hourly": "Electricity",
         "monash_kdd_cup_2018": "KDD", "uber_tlc_hourly": "Uber"}


# --------------------------------------------------------------------------- #
# output paths (re-derived after a --dataset-set override so the namespace matches)
# --------------------------------------------------------------------------- #

def _derive_dirs():
    global OOD_DIR, CKPT_DIR, PER_SOURCE_DIR, BOOT_IN_DIR, FIG_DIR
    OOD_DIR = config.ID_OUT_DIR / "ood_transfer"
    CKPT_DIR = OOD_DIR / "checkpoints"
    PER_SOURCE_DIR = OOD_DIR / "per_source"
    BOOT_IN_DIR = OOD_DIR / "bootstrap_inputs"
    FIG_DIR = OOD_DIR / "figures"
    for d in (OOD_DIR, CKPT_DIR, PER_SOURCE_DIR, BOOT_IN_DIR, FIG_DIR):
        d.mkdir(parents=True, exist_ok=True)


_derive_dirs()


# --------------------------------------------------------------------------- #
# frozen-probe identity + checkpoints (target is NEVER part of the training identity)
# --------------------------------------------------------------------------- #

def _probe_run_id(source, qset, seed):
    """Identity of a SOURCE-trained probe. Excludes the target on purpose, so every target
    evaluation of one source reuses the SAME checkpoint (electricity->kdd and electricity->uber
    never train separate probes)."""
    return f"{source}__{POOLING}__C{C}_H{H}__{qset}__seed{seed}"


def _ckpt_dir(source, qset, seed):
    return CKPT_DIR / _probe_run_id(source, qset, seed)


def _ckpt_meta(source, qset, seed, quantiles):
    return {"source": source, "pooling": POOLING, "context_length": C, "prediction_length": H,
            "quantile_set": qset, "quantile_config": int(len(quantiles)),
            "quantiles": [float(x) for x in quantiles], "seed": int(seed),
            "model": "amazon/chronos-2",
            "normalization": "arcsinh-context (per-window, context-only) + source-fit StandardScaler",
            "epochs": QUANTILE_EPOCHS, "wd_grid": list(WD_GRID),
            "layer_scheme": f"Embed(L0)+L1..L{NUM_LAYERS - 1}"}


def save_checkpoints(fitted, source, qset, seed, quantiles):
    d = _ckpt_dir(source, qset, seed)
    d.mkdir(parents=True, exist_ok=True)
    meta = _ckpt_meta(source, qset, seed, quantiles)
    for i, f in fitted.items():
        torch.save({"linear_state": f["linear"].state_dict(), "scaler": f["scaler"],
                    "wd": f["wd"], "selection": f["selection"],
                    "in_features": f["in_features"], "out_features": f["out_features"],
                    "layer": int(i), "meta": meta}, d / f"L{i:02d}.pt")
    json.dump(meta, open(d / "run_meta.json", "w"), indent=2)
    print(f"  [saved] frozen probe -> {d}  ({NUM_LAYERS} layer checkpoints)")
    return d


def load_checkpoints(source, qset, seed, quantiles, device):
    """Resume: load a previously-frozen source probe if ALL layer checkpoints are present and
    their metadata matches the current config; a mismatch fails loudly (never silently reused)."""
    d = _ckpt_dir(source, qset, seed)
    paths = [d / f"L{i:02d}.pt" for i in range(NUM_LAYERS)]
    if not all(p.exists() for p in paths):
        return None
    want = _ckpt_meta(source, qset, seed, quantiles)
    fitted = {}
    for i, p in enumerate(paths):
        ck = torch.load(p, map_location=device, weights_only=False)
        got = ck["meta"]
        for k in ("source", "pooling", "context_length", "prediction_length",
                  "quantile_set", "quantile_config", "seed"):
            if got.get(k) != want[k]:
                raise RuntimeError(
                    f"checkpoint {p} meta[{k}]={got.get(k)!r} != expected {want[k]!r} — stale or "
                    "mismatched probe; delete the run dir and re-fit")
        lin = torch.nn.Linear(ck["in_features"], ck["out_features"])
        lin.load_state_dict(ck["linear_state"])
        lin.to(device)
        lin.eval()
        fitted[i] = {"scaler": ck["scaler"], "linear": lin, "wd": float(ck["wd"]),
                     "selection": ck["selection"], "in_features": ck["in_features"],
                     "out_features": ck["out_features"], "device": str(device)}
    print(f"  [resume] loaded frozen probe from {d} (no re-training)")
    return fitted


# --------------------------------------------------------------------------- #
# train ONCE on source; evaluate the frozen probe on each target
# --------------------------------------------------------------------------- #

def get_source_probe(source, qset, quantiles, seed, device):
    """Return the frozen SOURCE probe (loaded from checkpoints if present, else fit + saved).
    Trained on the SOURCE train split ONLY — target arrays never reach this function."""
    fitted = load_checkpoints(source, qset, seed, quantiles, device)
    if fitted is not None:
        return fitted, _ckpt_dir(source, qset, seed)
    print(f"  [fit] training source probe on {source} (seed {seed}, {qset})")
    w = build_windows(source)      # seed default = SEED -> same windows the q9 per-dataset run used
    f_tr, _ = extract_window_features(source, "train", w["X_train"], w["y_train"], pooling=POOLING)
    fitted = fit_quantile_probe(f_tr, w["Y_train_traj"], quantiles=quantiles,
                                epochs=QUANTILE_EPOCHS, wd_grid=WD_GRID, device=device)
    ckpt = save_checkpoints(fitted, source, qset, seed, quantiles)
    del w, f_tr
    gc.collect()
    return fitted, ckpt


def _mae_median_raw(w, diag):
    """Per-layer MAE of the probe's median forecast in RAW units, inverting the arcsinh label
    with each TARGET window's own context stats (mu + s*sinh(z)) — leakage-free (context only)."""
    mu, s = _ctx_stats(w["X_test"], w["meta"]["sigma_eps"])
    y_raw = mu[:, None] + s[:, None] * np.sinh(w["Y_test_traj"].astype(np.float64))
    out = []
    for i in range(NUM_LAYERS):
        yhat = mu[:, None] + s[:, None] * np.sinh(diag["test_median"][i].astype(np.float64))
        out.append(float(np.abs(y_raw - yhat).mean()))
    return out


def evaluate_target(fitted, ckpt_dir, source, target, qset, quantiles, seed, device):
    """Score the frozen SOURCE probe on TARGET's test split. Returns (rows, summary)."""
    is_ood = source != target
    tag = "OOD" if is_ood else "ID "
    print(f"  [eval {tag}] {SHORT.get(source, source)} -> {SHORT.get(target, target)}")
    w = build_windows(target)
    f_te, _ = extract_window_features(target, "test", w["X_test"], w["y_test"], pooling=POOLING)
    out, diag = predict_quantile_probe(fitted, f_te, w["Y_test_traj"], quantiles=quantiles,
                                       device=device, collect_test_median=True,
                                       collect_test_window_loss=True)
    # MASE: source probe's median on TARGET windows vs TARGET future, TARGET seasonal-naive
    # denominator + TARGET native Chronos-2 (all from the per-dataset pipeline, cached).
    mase_entry, mase_pw = compute_mase(target, w, {POOLING: diag})
    mae = _mae_median_raw(w, diag)

    sid = np.asarray(w["series_test"], np.int64)
    n_test, n_series = int(w["meta"]["n_test"]), int(len(np.unique(sid)))
    run_id = _probe_run_id(source, qset, seed)
    rows = []
    for i in range(NUM_LAYERS):
        rows.append({
            "source_dataset": source, "target_dataset": target, "layer": i, "seed": int(seed),
            "split": "test", "is_ood": is_ood, "pooling": POOLING,
            "quantile_config": int(len(quantiles)), "quantile_set": qset,
            "context_length": C, "prediction_length": H,
            "probe_run_id": run_id, "probe_checkpoint": str(ckpt_dir / f"L{i:02d}.pt"),
            "selected_source_hyperparameters": json.dumps({"weight_decay": fitted[i]["wd"]}),
            "quantile_loss": float(out[i]),
            "mean_pinball_loss": float(diag["test_mean_pinball"][i]),
            "mase": float(mase_entry["poolings"][POOLING][i]),
            "mae_median_raw": float(mae[i]),
            "n_test_windows": n_test, "n_test_series": n_series})

    ql = np.array([out[i] for i in range(NUM_LAYERS)], float)
    bi = int(ql.argmin())
    summary = {"source_dataset": source, "target_dataset": target, "seed": int(seed),
               "is_ood": is_ood, "quantile_set": qset, "quantile_config": int(len(quantiles)),
               "best_layer": bi, "min_loss": float(ql[bi]),
               "last_layer": LAST_LAYER, "last_layer_loss": float(ql[LAST_LAYER]),
               "delta_vs_last": float(ql[LAST_LAYER] - ql[bi]),   # positive = earlier layer beats last
               "native_mase": float(mase_entry["native_mase"]),
               "n_test_windows": n_test, "n_test_series": n_series}

    _save_boot_inputs(source, target, qset, seed, sid, diag, n_test)
    del w, f_te
    gc.collect()
    return rows, summary


def _save_boot_inputs(source, target, qset, seed, sid, diag, n_test):
    """Per-window loss + series ids for the CI bands (series-level cluster bootstrap, reused
    from probing.stats). Written per (source,target,seed) under an unambiguous filename so a
    source and a target evaluation can never collide."""
    wl = np.stack([diag["test_window_loss"][i] for i in range(NUM_LAYERS)]).astype(np.float64)
    assert wl.shape == (NUM_LAYERS, n_test), f"window-loss {wl.shape} != ({NUM_LAYERS}, {n_test})"
    out = BOOT_IN_DIR / f"{source}__to__{target}__{qset}__seed{seed}.npz"
    np.savez(out, window_loss=wl, series_test=sid)


def run_source(source, targets, qset, quantiles, seed, device):
    """Train the source probe once, evaluate it on every target, write the per-source JSON."""
    print(f"\n{'=' * 70}\n[source={source}  seed={seed}  {qset}] train once, eval {len(targets)} "
          f"targets\n{'=' * 70}")
    fitted, ckpt_dir = get_source_probe(source, qset, quantiles, seed, device)
    all_rows, all_summ = [], []
    for target in targets:
        rows, summ = evaluate_target(fitted, ckpt_dir, source, target, qset, quantiles, seed, device)
        all_rows += rows
        all_summ.append(summ)
    payload = {"run_meta": {"source_dataset": source, "targets": list(targets), "seed": int(seed),
                            "quantile_set": qset, "quantile_config": int(len(quantiles)),
                            "pooling": POOLING, "context_length": C, "prediction_length": H,
                            "probe_run_id": _probe_run_id(source, qset, seed),
                            "checkpoint_dir": str(ckpt_dir)},
               "rows": all_rows, "summaries": all_summ}
    out = PER_SOURCE_DIR / f"{source}__{qset}__seed{seed}.json"
    json.dump(payload, open(out, "w"), indent=2)
    print(f"  [saved] {out}  ({len(all_rows)} rows, {len(all_summ)} source->target cells)")
    return payload


# --------------------------------------------------------------------------- #
# aggregate all per-source JSONs -> combined table, summary, 3x3 figure
# --------------------------------------------------------------------------- #

def _load_per_source(qset, seed):
    payloads = {}
    for source in DATASET_ORDER:
        p = PER_SOURCE_DIR / f"{source}__{qset}__seed{seed}.json"
        if p.exists():
            payloads[source] = json.load(open(p))
    return payloads


def _boot_band(source, target, qset, seed):
    """95% series-level cluster-bootstrap CI band (per layer) for one panel, or None if the
    per-window inputs are absent. Reuses the exact cluster bootstrap of run_bootstrap."""
    p = BOOT_IN_DIR / f"{source}__to__{target}__{qset}__seed{seed}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        wl, sid = np.asarray(d["window_loss"], np.float64), np.asarray(d["series_test"], np.int64)
    uniq, inv = np.unique(sid, return_inverse=True)
    S = len(uniq)
    cnt = np.bincount(inv).astype(np.float64)
    sums = np.zeros((S, NUM_LAYERS))
    np.add.at(sums, inv, wl.T)
    counts = cluster_bootstrap_counts(S, BOOT_B, SEED)
    boot = cluster_bootstrap_apply(counts, sums, cnt)     # (B, NUM_LAYERS)
    return np.percentile(boot, CI_LO, axis=0), np.percentile(boot, CI_HI, axis=0)


def aggregate(qset, seed):
    payloads = _load_per_source(qset, seed)
    if not payloads:
        print(f"  [aggregate] no per-source results for {qset} seed {seed} yet — run the "
              "source jobs first")
        return
    # combined tidy CSV/JSON
    rows = [r for pl in payloads.values() for r in pl["rows"]]
    fields = ["source_dataset", "target_dataset", "layer", "seed", "split", "is_ood", "pooling",
              "quantile_config", "quantile_set", "context_length", "prediction_length",
              "probe_run_id", "probe_checkpoint", "selected_source_hyperparameters",
              "quantile_loss", "mean_pinball_loss", "mase", "mae_median_raw",
              "n_test_windows", "n_test_series"]
    with open(OOD_DIR / f"ood_transfer_results__{qset}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(OOD_DIR / f"ood_transfer_results__{qset}.json", "w"), indent=2)

    # summary with delta_vs_last + best_layer_shift (shift measured against the source's ID best)
    summ = {(s["source_dataset"], s["target_dataset"]): s
            for pl in payloads.values() for s in pl["summaries"]}
    id_best = {src: summ[(src, src)]["best_layer"] for src in payloads
               if (src, src) in summ}                     # each source's diagonal best layer
    out_summ = []
    for (src, tgt), s in summ.items():
        e = dict(s)
        e["best_layer_source_id"] = id_best.get(src)
        e["best_layer_shift"] = (None if id_best.get(src) is None
                                 else s["best_layer"] - id_best[src])
        out_summ.append(e)
    json.dump({"config": {"quantile_set": qset, "quantile_config": len(QUANTILE_SETS[qset]),
                          "seed": int(seed), "pooling": POOLING, "context_length": C,
                          "prediction_length": H, "last_layer": LAST_LAYER,
                          "delta_vs_last": "loss[last] - min_over_layers(loss); "
                                           "positive = an earlier layer beats the final layer",
                          "best_layer_shift": "best_layer(transfer) - best_layer(source ID)"},
               "cells": out_summ},
              open(OOD_DIR / f"ood_transfer_summary__{qset}.json", "w"), indent=2)
    print(f"  [saved] combined table + summary under {OOD_DIR} ({len(rows)} rows, "
          f"{len(out_summ)} cells)")

    make_matrix_figure(summ, qset, seed)
    make_summary_figure(out_summ, qset, seed)


def make_matrix_figure(summ, qset, seed):
    """3x3 transfer grid: rows = source probe, cols = evaluation target. Each panel: Chronos-2
    quantile loss vs depth (Embed + L1..L12), 95% cluster-bootstrap band, ★ at the best layer,
    dotted line at the final layer. Diagonal (ID) panels get a tinted background; off-diagonal
    are strict cross-dataset transfer (OOD). y-scale is SHARED WITHIN A COLUMN (same target =
    comparable loss scale) and NOT across columns — absolute height is not a distance measure."""
    xs = np.arange(NUM_LAYERS)
    xlabels = ["Embed"] + [str(i) for i in range(1, NUM_LAYERS)]
    fig, axes = plt.subplots(3, 3, figsize=(16, 13), sharex=True, sharey="col")
    for ri, src in enumerate(DATASET_ORDER):
        color = ID_STYLE.get(src, {}).get("color", "#333333")
        for ci, tgt in enumerate(DATASET_ORDER):
            ax = axes[ri, ci]
            s = summ.get((src, tgt))
            if s is None:
                ax.text(0.5, 0.5, "(not run)", ha="center", va="center", transform=ax.transAxes,
                        color="0.6", fontsize=11)
                ax.set_xticks(xs)
                continue
            is_ood = s["is_ood"]
            # the summary carries only scalars; the per-layer loss curve is read back from the
            # per-source payload's rows (the actual point estimates, not a band midpoint).
            curve = _panel_curve(src, tgt, qset, seed)
            ax.plot(xs, curve, color=color, lw=2.3, marker="o", ms=4, zorder=3)
            band = _boot_band(src, tgt, qset, seed)
            if band is not None:
                ax.fill_between(xs, band[0], band[1], color=color, alpha=0.16, lw=0,
                                label="95% cluster-bootstrap CI")
            bi = s["best_layer"]
            ax.plot(bi, curve[bi], marker="*", ms=16, color=color, mec="k", mew=0.6, zorder=5,
                    label=f"best = {xlabels[bi]}")
            ax.axvline(LAST_LAYER, color="0.4", ls=":", lw=1.1)
            if is_ood:
                ax.set_facecolor("white")
                border = "0.75"
            else:                                          # ID diagonal: tinted + bold frame
                ax.set_facecolor("#f4f4ec")
                border = "k"
                for spine in ax.spines.values():
                    spine.set_linewidth(1.8)
                    spine.set_edgecolor(border)
            tagtxt = "OOD (cross-dataset)" if is_ood else "ID (in-dataset)"
            ax.set_title(f"{SHORT[src]} → {SHORT[tgt]}   [{tagtxt}]\n"
                         f"best {xlabels[bi]}, Δ vs last = {s['delta_vs_last']:+.3f}",
                         fontsize=10, fontweight=("bold" if not is_ood else "normal"))
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, fontsize=7)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("representation (Embed + L1..L12)")
    for ri, src in enumerate(DATASET_ORDER):
        axes[ri, 0].set_ylabel(f"probe trained on {SHORT[src]}\nChronos-2 quantile loss (test)")
    fig.suptitle(f"Cross-dataset probe transfer (3×3) — Chronos-2 quantile loss by layer  "
                 f"[{qset}, Q={len(QUANTILE_SETS[qset])}, seed {seed}]\n"
                 "rows = source (training) dataset, columns = target (evaluation) dataset;  "
                 "diagonal = in-dataset, off-diagonal = strict transfer;  LOWER = better, "
                 "★ = best layer;  y shared within a column only", fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIG_DIR / f"transfer_matrix_3x3__{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def _panel_curve(src, tgt, qset, seed):
    """Per-layer quantile-loss curve for one panel, read back from the per-source payload."""
    pl = json.load(open(PER_SOURCE_DIR / f"{src}__{qset}__seed{seed}.json"))
    by_layer = {r["layer"]: r["quantile_loss"] for r in pl["rows"]
                if r["target_dataset"] == tgt}
    return np.array([by_layer[i] for i in range(NUM_LAYERS)], float)


def make_summary_figure(cells, qset, seed):
    """Compact 3x3 summary heatmap of delta_vs_last (loss[last] - best loss). Annotated with the
    best layer + delta; ID diagonal outlined. Higher delta (warmer) = a stronger earlier-layer
    advantage that survived transfer. Absolute cross-panel loss levels are NOT shown (not a
    distance)."""
    xlabels = ["Embed"] + [str(i) for i in range(1, NUM_LAYERS)]
    idx = {src: i for i, src in enumerate(DATASET_ORDER)}
    M = np.full((3, 3), np.nan)
    for c in cells:
        si, ti = idx.get(c["source_dataset"]), idx.get(c["target_dataset"])
        if si is not None and ti is not None:
            M[si, ti] = c["delta_vs_last"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0.0)
    for c in cells:
        si, ti = idx.get(c["source_dataset"]), idx.get(c["target_dataset"])
        if si is None or ti is None:
            continue
        txt = f"best {xlabels[c['best_layer']]}\nΔ={c['delta_vs_last']:+.3f}"
        ax.text(ti, si, txt, ha="center", va="center", fontsize=9,
                color="k", fontweight=("bold" if not c["is_ood"] else "normal"))
        if not c["is_ood"]:
            ax.add_patch(plt.Rectangle((ti - 0.5, si - 0.5), 1, 1, fill=False, ec="k", lw=2.5))
    ax.set_xticks(range(3)); ax.set_xticklabels([SHORT[d] for d in DATASET_ORDER])
    ax.set_yticks(range(3)); ax.set_yticklabels([SHORT[d] for d in DATASET_ORDER])
    ax.set_xlabel("target (evaluation) dataset")
    ax.set_ylabel("source (probe training) dataset")
    ax.set_title(f"delta_vs_last = loss[last] − best-layer loss  [{qset}, seed {seed}]\n"
                 "positive = an earlier layer beat the final layer;  boxed = in-dataset (ID)",
                 fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ vs last (quantile loss)")
    fig.tight_layout()
    out = FIG_DIR / f"transfer_summary_delta__{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Strict cross-dataset OOD forecasting-probe transfer. Train the linear "
                    "quantile probe on ONE source dataset, freeze it, and evaluate on every "
                    "target's test split (one source per job). Diagonal = in-dataset.")
    ap.add_argument("--dataset-set", default=None, metavar="NAME",
                    help="dataset set (see probing.id_data.ID_DATASET_SPECS); precedence "
                         "CLI > env ID_DATASET_SET > default extended_v1.")
    ap.add_argument("--source-dataset", default=None,
                    help="the dataset the probe is TRAINED on (one per job). Use the exact tag, "
                         "e.g. monash_electricity_hourly / monash_kdd_cup_2018 / uber_tlc_hourly.")
    ap.add_argument("--target-datasets", nargs="+", default=DATASET_ORDER,
                    help="the datasets to EVALUATE the frozen probe on (default: all three).")
    ap.add_argument("--quantile-set", choices=sorted(QUANTILE_SETS), default="q9",
                    help="probe-head quantile configuration (default q9 for this experiment).")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available).")
    ap.add_argument("--figure-only", action="store_true",
                    help="skip training/eval; just aggregate existing per-source results into "
                         "the combined table + 3x3 figure.")
    return ap.parse_args(argv)


def main():
    args = _parse_args()
    if args.dataset_set:
        config.set_dataset_set(args.dataset_set)
        _derive_dirs()
    qset = args.quantile_set
    quantiles = QUANTILE_SETS[qset]
    if median_index(quantiles) is None:
        raise SystemExit(f"quantile set {qset} has no 0.5 level — MASE/median metrics are "
                         "undefined; use q1/q9/q21 (all contain 0.5)")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[config] dataset_set={config.DATASET_SET}  quantile_set={qset}  "
          f"Q={len(quantiles)}  device={device}  seeds={list(SEEDS)}")
    print(f"[config] frozen amazon/chronos-2, {POOLING} pooling, {NUM_LAYERS} depths "
          f"(Embed+L1..L{LAST_LAYER}), C={C} H={H}; outputs -> {OOD_DIR}")

    known = set(id_data.ID_DATASETS)
    if args.figure_only:
        for seed in SEEDS:
            aggregate(qset, seed)
        return

    if not args.source_dataset:
        raise SystemExit("--source-dataset is required (one source per job); or pass "
                         "--figure-only to assemble the grid from existing results.")
    if args.source_dataset not in known:
        raise SystemExit(f"unknown --source-dataset {args.source_dataset!r}; known: {sorted(known)}")
    bad = [t for t in args.target_datasets if t not in known]
    if bad:
        raise SystemExit(f"unknown --target-datasets {bad}; known: {sorted(known)}")

    for seed in SEEDS:
        run_source(args.source_dataset, args.target_datasets, qset, quantiles, seed, device)
        aggregate(qset, seed)          # refresh combined outputs with whatever sources exist so far

    print(f"\n{'=' * 70}\nDONE — source {args.source_dataset}. Run the other two sources, then "
          f"`--figure-only` completes the 3x3 grid.\n{'=' * 70}")


if __name__ == "__main__":
    main()
