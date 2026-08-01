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
from probing.id_data import build_windows, ROLLING_SETS
from probing.extraction import extract_window_features
from probing.probes import (fit_quantile_probe, fit_quantile_probe_explicit_val,
                            predict_quantile_probe, QUANTILE_SETS, median_index)
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

# Matrix order + short panel labels. The order is PER SET (explicit, not list(ID_DATASETS)):
# extended_v1's OOD pilot used only 3 of its 4 datasets (pedestrian excluded), so deriving the
# order from the roster would silently change the committed 3x3. extended_v2 uses all four.
ORDER_BY_SET = {
    "extended_v1": ["monash_electricity_hourly", "monash_kdd_cup_2018", "uber_tlc_hourly"],
    "extended_v2": ["monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly"],
    # rolling-origin within-series 4×4 (same roster as extended_v2; uniform temporal split)
    "extended_v3_rolling": ["monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly"],
}
SHORT_LABELS = {
    "monash_electricity_hourly": "Electricity", "monash_kdd_cup_2018": "KDD",
    "uber_tlc_hourly": "Uber", "monash_pedestrian_counts": "Pedestrian",
    "m4_hourly": "M4", "wind_farms_hourly": "WindFarms", "solar_1h": "Solar",
}


def _derive_datasets():
    """(Re)derive the matrix order + labels for the ACTIVE dataset set. Called at import and again
    after a --dataset-set override in main(). NDATA is the matrix side length (3 or 4)."""
    global DATASET_ORDER, SHORT, NDATA
    DATASET_ORDER = ORDER_BY_SET.get(config.DATASET_SET, list(id_data.ID_DATASETS))
    SHORT = {t: SHORT_LABELS.get(t, t) for t in DATASET_ORDER}
    NDATA = len(DATASET_ORDER)


_derive_datasets()


# --------------------------------------------------------------------------- #
# output paths (re-derived after a --dataset-set override so the namespace matches)
# --------------------------------------------------------------------------- #

def _derive_dirs():
    global OOD_DIR, CKPT_DIR, PER_SOURCE_DIR, BOOT_IN_DIR, FIG_DIR, SPLIT_META_DIR
    OOD_DIR = config.ID_OUT_DIR / "ood_transfer"
    CKPT_DIR = OOD_DIR / "checkpoints"
    PER_SOURCE_DIR = OOD_DIR / "per_source"
    BOOT_IN_DIR = OOD_DIR / "bootstrap_inputs"
    FIG_DIR = OOD_DIR / "figures"
    # rolling sets only (created lazily by _dump_split_meta, so legacy result trees stay unchanged)
    SPLIT_META_DIR = OOD_DIR / "split_meta"
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
            # dataset_set/split_mode pin the checkpoint to its namespace (directory separation
            # alone would let a manually copied probe pass the meta check unnoticed)
            "dataset_set": config.DATASET_SET,
            "split_mode": ("rolling_origin_within_series" if config.DATASET_SET in ROLLING_SETS
                           else "auto_within_or_cross_series"),
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
        # dataset_set was added to the meta later: enforce only when the checkpoint carries it,
        # so the pre-existing extended_v2/v3 seed-0 checkpoints (written before the key) resume.
        if "dataset_set" in got and got["dataset_set"] != want["dataset_set"]:
            raise RuntimeError(
                f"checkpoint {p} meta[dataset_set]={got['dataset_set']!r} != active "
                f"{want['dataset_set']!r} — a probe from another dataset set was placed in this "
                "namespace; delete the run dir and re-fit")
        lin = torch.nn.Linear(ck["in_features"], ck["out_features"])
        lin.load_state_dict(ck["linear_state"])
        lin.to(device)
        lin.eval()
        fitted[i] = {"scaler": ck["scaler"], "linear": lin, "wd": float(ck["wd"]),
                     "selection": ck["selection"], "in_features": ck["in_features"],
                     "out_features": ck["out_features"], "device": str(device)}
    print(f"  [resume] loaded frozen probe from {d} (no re-training)")
    return fitted


def source_selected_layer(source, qset, seed):
    """Layer chosen on the SOURCE VALIDATION split ONLY — never touches any target data.

    Each per-layer checkpoint stores selection.val_loss_by_wd (the source 80/20 validation q9 loss
    at every weight decay) and chosen_wd (its per-layer argmin). We collapse that to ONE validation
    score per layer = min over the wd grid (== the loss at chosen_wd), then take the argmin across
    the NUM_LAYERS depths. np.argmin breaks exact ties toward the EARLIEST layer (the deterministic
    source-only tie rule). Returns (layer, validation_loss_at_that_layer)."""
    d = _ckpt_dir(source, qset, seed)
    vals = []
    for i in range(NUM_LAYERS):
        p = d / f"L{i:02d}.pt"
        if not p.exists():
            raise FileNotFoundError(f"{p} missing — cannot derive the source-validation-selected "
                                    "layer; run the source job first")
        sel = torch.load(p, map_location="cpu", weights_only=False)["selection"]
        vals.append(float(min(sel["val_loss_by_wd"].values())))
    vals = np.asarray(vals, dtype=float)
    L = int(np.argmin(vals))
    return L, float(vals[L])


def _selected_layers_by_source(qset, seed):
    """{source: (layer, val_loss)} for every source whose checkpoints are present — the fixed,
    source-validation-selected layer reused for that source's ID cell AND all its transfer cells."""
    out = {}
    for src in DATASET_ORDER:
        if (_ckpt_dir(src, qset, seed) / "L00.pt").exists():
            out[src] = source_selected_layer(src, qset, seed)
    return out


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
    if config.DATASET_SET in ROLLING_SETS:
        # rolling sets: weight-decay selection on the EXPLICIT temporal val split (no 80/20 carve).
        f_va, _ = extract_window_features(source, "val", w["X_val"], w["y_val"], pooling=POOLING)
        print(f"    [rolling] explicit temporal val: {w['meta']['n_val']} windows / "
              f"{w['meta']['n_val_series']} series")
        fitted = fit_quantile_probe_explicit_val(f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"],
                                                 quantiles=quantiles, epochs=QUANTILE_EPOCHS,
                                                 wd_grid=WD_GRID, device=device)
        del f_va
    else:
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
    _dump_split_meta(target, w)      # rolling sets: persist the split's audit trail (idempotent)
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


def _dump_split_meta(tag, w):
    """Persist the rolling split's audit trail for `tag` (rolling sets only; no-op otherwise).

    One JSON per dataset per set: split mode, the deterministic selected val/test series, the
    per-window series id and target-start origin of every RETAINED window, and the realized
    counts vs the budget. The window build is deterministic (window seed = SEED even when probe
    seeds vary), so re-dumps overwrite identical content — the file is keyed by tag alone."""
    if config.DATASET_SET not in ROLLING_SETS:
        return
    m = w["meta"]
    payload = {
        "dataset_set": config.DATASET_SET, "tag": tag, "split_mode": m["split_mode"],
        "C": m["C"], "H": m["H"], "seed": m["seed"],
        "budget": [m["target_train"], m["target_val"], m["target_test"]],
        "selected_series": m["selected_series"],
        "series": {"train": np.asarray(w["series_train"]).tolist(),
                   "val": np.asarray(w["series_val"]).tolist(),
                   "test": np.asarray(w["series_test"]).tolist()},
        "origins": m["origins"],
        "counts": {"n_train": m["n_train"], "n_val": m["n_val"], "n_test": m["n_test"],
                   "n_train_series": int(len(np.unique(np.asarray(w["series_train"])))),
                   "n_val_series": m["n_val_series"], "n_test_series": m["n_test_series"],
                   "n_eligible_series": m["n_eligible_series"]},
    }
    SPLIT_META_DIR.mkdir(parents=True, exist_ok=True)
    out = SPLIT_META_DIR / f"{tag}.json"
    json.dump(payload, open(out, "w"), indent=2)
    print(f"  [saved] split meta -> {out}")


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


def _paired_delta_bootstrap(wl, sid):
    """Series-level cluster bootstrap of the per-layer loss AND the paired Δ-vs-last.

    wl (NUM_LAYERS, n) per-window quantile loss, sid (n,) test-series ids. Resamples WHOLE test
    series (stride 64 << span 576, so windows within a series are correlated); ONE shared counts
    matrix means each replicate's loss[last] and loss[layer] come from the SAME resampled series,
    so Δ_l = loss[last] − loss[layer] is a genuinely PAIRED difference (not two independent
    bootstraps subtracted) — identical method to run_bootstrap.summarize_dataset, reusing
    probing.stats verbatim. Returns per-layer point + 95% CI for the loss and for Δ-vs-last, plus
    the CI-excludes-zero flag (positive Δ = an earlier layer beats the last layer). Δ at the last
    layer is identically 0 (its CI must bracket 0)."""
    wl = np.asarray(wl, np.float64)
    sid = np.asarray(sid, np.int64)
    uniq, inv = np.unique(sid, return_inverse=True)
    S = len(uniq)
    cnt = np.bincount(inv).astype(np.float64)              # windows per series
    sums = np.zeros((S, wl.shape[0]))
    np.add.at(sums, inv, wl.T)
    counts = cluster_bootstrap_counts(S, BOOT_B, SEED)     # shared across all layers -> paired
    boot = cluster_bootstrap_apply(counts, sums, cnt)      # (B, NUM_LAYERS) per-layer loss
    point = wl.mean(axis=1)
    delta_b = boot[:, [LAST_LAYER]] - boot                 # paired: same replicates both sides
    d_lo = np.percentile(delta_b, CI_LO, axis=0)
    d_hi = np.percentile(delta_b, CI_HI, axis=0)
    return {"point": point, "boot": boot,             # boot (B, NUM_LAYERS): paired per-replicate loss
            "ci_lo": np.percentile(boot, CI_LO, axis=0),
            "ci_hi": np.percentile(boot, CI_HI, axis=0),
            "delta_vs_last": point[LAST_LAYER] - point,    # loss[last] − loss[layer]
            "delta_ci_lo": d_lo, "delta_ci_hi": d_hi,
            "delta_above_zero": d_lo > 0,
            "n_series": int(S), "n_windows": int(wl.shape[1])}


def _boot_cell(source, target, qset, seed):
    """Paired-Δ bootstrap for one source->target cell (loads the saved per-window npz), or None
    when that cell has not been run yet."""
    p = BOOT_IN_DIR / f"{source}__to__{target}__{qset}__seed{seed}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return _paired_delta_bootstrap(d["window_loss"], d["series_test"])


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

    summ = {(s["source_dataset"], s["target_dataset"]): s
            for pl in payloads.values() for s in pl["summaries"]}

    # paired cluster-bootstrap Δ-vs-last for every present cell (post-hoc, from the saved
    # per-window npz — no probe re-fit needed), then the tidy Δ table.
    boot_cells = {}
    for (src, tgt) in summ:
        bc = _boot_cell(src, tgt, qset, seed)
        if bc is not None:
            boot_cells[(src, tgt)] = bc
    write_delta_table(boot_cells, summ, qset, seed)

    # summary with delta_vs_last + best_layer_shift (shift measured against the source's ID best),
    # enriched with the BEST layer's paired Δ CI so the heatmap can print it.
    id_best = {src: summ[(src, src)]["best_layer"] for src in payloads
               if (src, src) in summ}                     # each source's diagonal best layer
    out_summ = []
    for (src, tgt), s in summ.items():
        e = dict(s)
        e["best_layer_source_id"] = id_best.get(src)
        e["best_layer_shift"] = (None if id_best.get(src) is None
                                 else s["best_layer"] - id_best[src])
        bc = boot_cells.get((src, tgt))
        if bc is not None:
            L = s["best_layer"]
            e["delta_vs_last_ci"] = [float(bc["delta_ci_lo"][L]), float(bc["delta_ci_hi"][L])]
            e["delta_above_zero"] = bool(bc["delta_above_zero"][L])
            e["bootstrap"] = {"n_replicates": BOOT_B, "seed": int(seed),
                              "n_test_series": bc["n_series"]}
        out_summ.append(e)
    json.dump({"config": {"quantile_set": qset, "quantile_config": len(QUANTILE_SETS[qset]),
                          "seed": int(seed), "pooling": POOLING, "context_length": C,
                          "prediction_length": H, "last_layer": LAST_LAYER,
                          "bootstrap_replicates": BOOT_B, "bootstrap_seed": int(seed),
                          "delta_vs_last": "loss[last] − loss[layer] (scalar cell field uses the "
                                           "best layer); positive = an earlier layer beats the "
                                           "final layer. delta_vs_last_ci = paired 95% cluster-"
                                           "bootstrap CI on the BEST layer's Δ; the best layer is "
                                           "the test-argmin, so per-cell it is exploratory (see "
                                           "the full-layer Δ table for the selection-free view)",
                          "best_layer_shift": "best_layer(transfer) - best_layer(source ID)"},
               "cells": out_summ},
              open(OOD_DIR / f"ood_transfer_summary__{qset}.json", "w"), indent=2)
    print(f"  [saved] combined table + summary under {OOD_DIR} ({len(rows)} rows, "
          f"{len(out_summ)} cells)")

    make_summary_figure(out_summ, boot_cells, qset, seed)

    # ------------------------------------------------------------------ #
    # normalized improvement over L12 — two layer-selection views.
    #   PRIMARY  (source_val): layer chosen on SOURCE VALIDATION only, one per source row, reused
    #            for the ID cell + all transfer cells; gains may be NEGATIVE; the fair result.
    #   DIAGNOSTIC (oracle):   layer = target-test argmin — optimistically biased, kept for
    #            reference under oracle_* filenames (the layer-selection-bias correction).
    # Same saved per-window q9 losses + same paired series-cluster bootstrap for both; only the
    # (fixed, pre-bootstrap) layer differs. Raw delta_vs_last table/figure above are untouched.
    # ------------------------------------------------------------------ #
    sel_by_src = _selected_layers_by_source(qset, seed)
    sv_rows = write_relative_gain_table(boot_cells, summ, qset, seed, "source_val", sel_by_src)
    make_relative_gain_heatmap(summ, boot_cells, qset, seed, "source_val", sel_by_src)
    make_relative_gain_id_vs_ood(summ, boot_cells, qset, seed, "source_val", sel_by_src)
    make_matrix_figure(summ, boot_cells, qset, seed, "source_val", sel_by_src)
    or_rows = write_relative_gain_table(boot_cells, summ, qset, seed, "oracle", sel_by_src)
    make_relative_gain_heatmap(summ, boot_cells, qset, seed, "oracle", sel_by_src)
    make_relative_gain_id_vs_ood(summ, boot_cells, qset, seed, "oracle", sel_by_src)
    make_matrix_figure(summ, boot_cells, qset, seed, "oracle", sel_by_src)
    _print_corrected_summary(sv_rows, or_rows, sel_by_src, qset)


def write_delta_table(boot_cells, summ, qset, seed):
    """Per-cell, per-layer paired Δ-vs-last table — the requested tidy output: delta_vs_last,
    its 95% cluster-bootstrap CI, and the CI-excludes-zero flag, for every earlier layer vs the
    final layer in each source->target cell. The last layer's Δ is 0 by construction (kept as a
    row); is_best_layer marks the test-argmin layer."""
    fields = ["source_dataset", "target_dataset", "seed", "is_ood", "quantile_set", "layer",
              "delta_vs_last", "delta_ci_lo", "delta_ci_hi", "delta_above_zero",
              "is_best_layer", "is_last_layer", "n_test_series", "n_test_windows",
              "bootstrap_replicates"]
    rows = []
    for (src, tgt), bc in sorted(boot_cells.items()):
        best = summ[(src, tgt)]["best_layer"]
        for L in range(NUM_LAYERS):
            rows.append({"source_dataset": src, "target_dataset": tgt, "seed": int(seed),
                         "is_ood": src != tgt, "quantile_set": qset, "layer": L,
                         "delta_vs_last": float(bc["delta_vs_last"][L]),
                         "delta_ci_lo": float(bc["delta_ci_lo"][L]),
                         "delta_ci_hi": float(bc["delta_ci_hi"][L]),
                         "delta_above_zero": bool(bc["delta_above_zero"][L]),
                         "is_best_layer": L == best, "is_last_layer": L == LAST_LAYER,
                         "n_test_series": bc["n_series"], "n_test_windows": bc["n_windows"],
                         "bootstrap_replicates": BOOT_B})
    with open(OOD_DIR / f"ood_transfer_delta_vs_last__{qset}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(OOD_DIR / f"ood_transfer_delta_vs_last__{qset}.json", "w"), indent=2)
    print(f"  [saved] paired Δ-vs-last table ({len(rows)} rows) -> "
          f"{OOD_DIR / f'ood_transfer_delta_vs_last__{qset}.csv'}")


def make_matrix_figure(summ, boot_cells, qset, seed, mode="oracle", sel_by_src=None):
    """N×N transfer grid: rows = source probe, cols = evaluation target. Each panel: Chronos-2
    quantile loss vs depth (Embed + L1..L12), 95% cluster-bootstrap band, filled black-edged markers
    where a layer's paired Δ-vs-last CI excludes zero, dotted line at the final layer. All curves are
    kept (descriptive); the highlighted layer + the Δ / relative-gain reported per title depend on
    `mode`:
      mode='source_val' (PRIMARY): prominent ◆ at the SOURCE-VALIDATION-selected layer (one per
        source row, chosen with NO target data) + a secondary hollow ★ at the target-test oracle
        best layer, clearly labeled. Δ / gain reported for the ◆ layer and MAY be negative.
      mode='oracle' (DIAGNOSTIC): ★ at the target-test-best layer — optimistically biased.
    Diagonal (ID) panels get a tinted background. y-scale is SHARED WITHIN A COLUMN only (same target
    = comparable loss scale); absolute height is not a distance measure."""
    xs = np.arange(NUM_LAYERS)
    xlabels = ["Embed"] + [str(i) for i in range(1, NUM_LAYERS)]
    fig, axes = plt.subplots(NDATA, NDATA, figsize=(5.3 * NDATA, 4.3 * NDATA),
                             sharex=True, sharey="col")
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
            bc = boot_cells.get((src, tgt))
            if bc is not None:
                ax.fill_between(xs, bc["ci_lo"], bc["ci_hi"], color=color, alpha=0.16, lw=0,
                                label="95% cluster-bootstrap CI")
                sig = bc["delta_above_zero"].copy()
                sig[LAST_LAYER] = False                    # the last layer never beats itself
                if sig.any():
                    ax.plot(xs[sig], curve[sig], "o", color=color, ms=7, mec="k", mew=0.9,
                            zorder=4, label="Δ vs last: CI > 0")
            oracle_bi = s["best_layer"]
            if mode == "source_val":
                chosen = sel_by_src[src][0]
                ax.plot(chosen, curve[chosen], marker="D", ms=12, color=color, mec="k", mew=1.4,
                        zorder=6, label=f"source-val sel = {xlabels[chosen]}")
                ax.plot(oracle_bi, curve[oracle_bi], marker="*", ms=14, mfc="none", mec="0.25",
                        mew=1.1, zorder=5, label=f"oracle test-best = {xlabels[oracle_bi]}")
            else:
                chosen = oracle_bi
                ax.plot(chosen, curve[chosen], marker="*", ms=16, color=color, mec="k", mew=0.6,
                        zorder=5, label=f"oracle test-best = {xlabels[chosen]}")
            ax.axvline(LAST_LAYER, color="0.4", ls=":", lw=1.1)
            if is_ood:
                ax.set_facecolor("white")
            else:                                          # ID diagonal: tinted + bold frame
                ax.set_facecolor("#f4f4ec")
                for spine in ax.spines.values():
                    spine.set_linewidth(1.8)
                    spine.set_edgecolor("k")
            delta_chosen = float(curve[LAST_LAYER] - curve[chosen])
            rg_txt = ("" if bc is None
                      else f"  ({_relative_gain_cell(bc, chosen)['relative_gain_pct']:+.1f}%)")
            sel_word = "source-val sel" if mode == "source_val" else "oracle test-best"
            tagtxt = "OOD (cross-dataset)" if is_ood else "ID (in-dataset)"
            ax.set_title(f"{SHORT[src]} → {SHORT[tgt]}   [{tagtxt}]\n"
                         f"{sel_word} {xlabels[chosen]}: Δ vs last = {delta_chosen:+.3f}{rg_txt}",
                         fontsize=10, fontweight=("bold" if not is_ood else "normal"))
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, fontsize=7)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("representation (Embed + L1..L12)")
    for ri, src in enumerate(DATASET_ORDER):
        axes[ri, 0].set_ylabel(f"probe trained on {SHORT[src]}\nChronos-2 quantile loss (test)")
    if mode == "source_val":
        head = (f"Cross-dataset probe transfer ({NDATA}×{NDATA}) — PRIMARY: layer selected on SOURCE "
                f"VALIDATION only  [{qset}, Q={len(QUANTILE_SETS[qset])}, seed {seed}]")
        sub = ("rows = source, cols = target;  ◆ = source-validation-selected layer (one per row, no "
               "target data);  hollow ★ = target-test oracle best (secondary/diagnostic);  Δ / gain "
               "reported for ◆ and MAY be negative;  filled dot = paired Δ-vs-last CI>0;  LOWER = "
               "better;  y shared within a column")
        out = FIG_DIR / f"source_val_selected_layerwise_matrix_{qset}.png"
    else:
        head = (f"Cross-dataset probe transfer ({NDATA}×{NDATA}) — DIAGNOSTIC (oracle test-best layer)"
                f"  [{qset}, Q={len(QUANTILE_SETS[qset])}, seed {seed}]")
        sub = ("rows = source, cols = target;  ★ = target-test-best layer (optimistically biased — NOT "
               "the primary result);  filled dot = paired Δ-vs-last CI>0;  LOWER = better;  "
               "y shared within a column")
        out = FIG_DIR / f"oracle_test_selected_layerwise_matrix_{qset}.png"
    fig.suptitle(head + "\n" + sub, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def _panel_curve(src, tgt, qset, seed):
    """Per-layer quantile-loss curve for one panel, read back from the per-source payload."""
    pl = json.load(open(PER_SOURCE_DIR / f"{src}__{qset}__seed{seed}.json"))
    by_layer = {r["layer"]: r["quantile_loss"] for r in pl["rows"]
                if r["target_dataset"] == tgt}
    return np.array([by_layer[i] for i in range(NUM_LAYERS)], float)


def make_summary_figure(cells, boot_cells, qset, seed):
    """Compact 3x3 summary heatmap of delta_vs_last (loss[last] - best loss). Each cell shows the
    best layer, its Δ, and the paired 95% cluster-bootstrap CI of that Δ; a ★ prefix (and bold)
    marks cells whose best-layer Δ CI EXCLUDES zero (the earlier-layer advantage is significant).
    ID diagonal outlined. Warmer = a stronger earlier-layer advantage that survived transfer.
    Absolute cross-panel loss levels are NOT shown (not a distance)."""
    xlabels = ["Embed"] + [str(i) for i in range(1, NUM_LAYERS)]
    idx = {src: i for i, src in enumerate(DATASET_ORDER)}
    M = np.full((NDATA, NDATA), np.nan)
    for c in cells:
        si, ti = idx.get(c["source_dataset"]), idx.get(c["target_dataset"])
        if si is not None and ti is not None:
            M[si, ti] = c["delta_vs_last"]
    fig, ax = plt.subplots(figsize=(max(8.5, 2.6 * NDATA), max(7.0, 2.3 * NDATA)))
    im = ax.imshow(M, cmap="YlOrRd", vmin=0.0)
    for c in cells:
        si, ti = idx.get(c["source_dataset"]), idx.get(c["target_dataset"])
        if si is None or ti is None:
            continue
        L = c["best_layer"]
        bc = boot_cells.get((c["source_dataset"], c["target_dataset"]))
        lines = [f"best {xlabels[L]}", f"Δ={c['delta_vs_last']:+.3f}"]
        sig = False
        if bc is not None:
            lo, hi = bc["delta_ci_lo"][L], bc["delta_ci_hi"][L]
            lines.append(f"95% CI [{lo:+.2f}, {hi:+.2f}]")
            sig = bool(bc["delta_above_zero"][L])
        txt = ("★ " if sig else "") + "\n".join(lines)
        ax.text(ti, si, txt, ha="center", va="center", fontsize=8,
                color="k", fontweight=("bold" if sig else "normal"))
        if not c["is_ood"]:
            ax.add_patch(plt.Rectangle((ti - 0.5, si - 0.5), 1, 1, fill=False, ec="k", lw=2.5))
    ax.set_xticks(range(NDATA)); ax.set_xticklabels([SHORT[d] for d in DATASET_ORDER])
    ax.set_yticks(range(NDATA)); ax.set_yticklabels([SHORT[d] for d in DATASET_ORDER])
    ax.set_xlabel("target (evaluation) dataset")
    ax.set_ylabel("source (probe training) dataset")
    ax.set_title(f"delta_vs_last = loss[last] − best-layer loss  [{qset}, seed {seed}]\n"
                 "positive = an earlier layer beat the final layer;  ★ = best-layer Δ 95% CI "
                 "excludes 0;  boxed = in-dataset (ID)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Δ vs last (quantile loss)")
    fig.tight_layout()
    out = FIG_DIR / f"transfer_summary_delta__{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# --------------------------------------------------------------------------- #
# Normalized improvement over L12: relative_gain_pct = 100 * (loss[L12] - loss[best]) / loss[L12].
# ADDITIVE — the raw delta_vs_last outputs/plots above are untouched. Same saved per-window q9 losses,
# same full-sample best layer (test-argmin), same paired series cluster bootstrap. The CI is the
# percentile of the per-replicate RATIO (never raw-CI / a constant). The best layer is selected on the
# full test set and held FIXED during the bootstrap, so these CIs are DESCRIPTIVE (post-selection) and
# NOT adjusted for layer selection.
# --------------------------------------------------------------------------- #

REL_GAIN_LABEL = "Relative improvement over L12 (%)"


def _relative_gain_cell(bc, best_layer):
    """relative_gain_pct + paired-bootstrap percentile CI for the FIXED best layer vs L12.

    Full sample: 100 * (loss[L12] - loss[best]) / loss[L12]. Per replicate (same resampled series on
    both sides -> paired): 100 * (boot[L12] - boot[best]) / boot[L12], then the 2.5/97.5 percentiles.
    best_layer is the full-sample test-argmin held FIXED (CI is descriptive, not selection-adjusted).
    excludes_zero is True when the whole CI sits on one side of 0."""
    point, boot = bc["point"], bc["boot"]
    L = LAST_LAYER
    pct = 100.0 * (point[L] - point[best_layer]) / point[L]
    rg_b = 100.0 * (boot[:, L] - boot[:, best_layer]) / boot[:, L]
    lo, hi = float(np.percentile(rg_b, CI_LO)), float(np.percentile(rg_b, CI_HI))
    return {"relative_gain_pct": float(pct), "ci_lo": lo, "ci_hi": hi,
            "excludes_zero": bool(lo > 0 or hi < 0)}


def _split_mode_map():
    """{dataset: split_mode} read from the saved screen JSON (file-only, no dataset load). Empty when
    the screen has not been run — callers then fall back to 'unknown'."""
    for suffix in ("", "_native"):
        p = OOD_DIR / "screen" / f"dataset_screen__{config.DATASET_SET}{suffix}.json"
        if p.exists():
            d = json.load(open(p))
            return {r["dataset"]: r.get("split_mode", "unknown") for r in d.get("rows", [])}
    return {}


def write_relative_gain_table(boot_cells, summ, qset, seed, mode="oracle", sel_by_src=None):
    """Normalized improvement over L12 per source->target cell, for one layer-selection view.

    mode='source_val' (PRIMARY): layer chosen on SOURCE VALIDATION only (one per source row). Writes
      source_val_relative_gain_summary_<qset>.csv/json with the full requested schema — the selected
      layer + the validation loss used, its test loss and L12's, raw gain + 95% CI, relative gain +
      95% CI, whether each CI excludes 0, the target-test ORACLE best layer + its relative gain, the
      split mode, the test-window count and the number of bootstrap clusters. Gains may be NEGATIVE.
    mode='oracle' (DIAGNOSTIC): layer = target-test argmin (optimistically biased). Writes
      oracle_relative_gain_summary_<qset>.csv/json AND the legacy relative_gain_summary_<qset>.*
      alias (so run_ood_pretrain_transfer's overlay keeps resolving) — same oracle content.
    Both reuse the same paired series-cluster bootstrap; the layer is fixed BEFORE the bootstrap, so
    every CI here is DESCRIPTIVE (not adjusted for layer selection)."""
    smap = _split_mode_map()
    rows = []
    for (src, tgt), bc in sorted(boot_cells.items()):
        oracle_best = int(summ[(src, tgt)]["best_layer"])
        L = int(sel_by_src[src][0]) if mode == "source_val" else oracle_best
        rg = _relative_gain_cell(bc, L)
        raw_gain = float(bc["delta_vs_last"][L])                       # loss[last] − loss[L]
        raw_lo, raw_hi = float(bc["delta_ci_lo"][L]), float(bc["delta_ci_hi"][L])
        common = {"source_dataset": src, "target_dataset": tgt,
                  "id_or_ood": ("ID" if src == tgt else "OOD")}
        if mode == "source_val":
            oracle_rg = _relative_gain_cell(bc, oracle_best)["relative_gain_pct"]
            rows.append({**common,
                         "source_val_selected_layer": L,
                         "validation_loss_used": round(float(sel_by_src[src][1]), 6),
                         "selected_layer_test_q9_loss": round(float(bc["point"][L]), 6),
                         "last_layer_test_q9_loss": round(float(bc["point"][LAST_LAYER]), 6),
                         "raw_gain": round(raw_gain, 6),
                         "raw_ci_lo": round(raw_lo, 6), "raw_ci_hi": round(raw_hi, 6),
                         "raw_ci_excludes_zero": bool(raw_lo > 0 or raw_hi < 0),
                         "relative_gain_pct": round(rg["relative_gain_pct"], 4),
                         "relative_ci_lo": round(rg["ci_lo"], 4),
                         "relative_ci_hi": round(rg["ci_hi"], 4),
                         "relative_ci_excludes_zero": rg["excludes_zero"],
                         "oracle_best_layer": oracle_best,
                         "oracle_relative_gain_pct": round(float(oracle_rg), 4),
                         "split_mode": smap.get(tgt, "unknown"),
                         "n_test_windows": int(bc["n_windows"]),
                         "n_bootstrap_clusters": int(bc["n_series"])})
        else:
            rows.append({**common, "best_layer": L,
                         "best_layer_q9_loss": round(float(bc["point"][L]), 6),
                         "last_layer_q9_loss": round(float(bc["point"][LAST_LAYER]), 6),
                         "delta_vs_last": round(raw_gain, 6),
                         "relative_gain_pct": round(rg["relative_gain_pct"], 4),
                         "relative_ci_lo": round(rg["ci_lo"], 4),
                         "relative_ci_hi": round(rg["ci_hi"], 4),
                         "relative_ci_excludes_zero": rg["excludes_zero"],
                         "split_mode": smap.get(tgt, "unknown")})
    prefix = ("source_val_relative_gain_summary" if mode == "source_val"
              else "oracle_relative_gain_summary")
    stems = [prefix] + ([] if mode == "source_val" else ["relative_gain_summary"])   # legacy alias
    fields = list(rows[0].keys()) if rows else []
    for stem in stems:
        with open(OOD_DIR / f"{stem}_{qset}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        json.dump(rows, open(OOD_DIR / f"{stem}_{qset}.json", "w"), indent=2)
    print(f"  [saved] {mode} relative-gain table ({len(rows)} cells) -> "
          f"{OOD_DIR / f'{prefix}_{qset}.csv'}")
    return rows


def make_relative_gain_heatmap(summ, boot_cells, qset, seed, mode="oracle", sel_by_src=None):
    """N×N normalized-gain heatmap: relative_gain_pct = 100*(loss[L12]-loss[L])/loss[L12]. Diverging
    colormap centered at 0 (warm = layer L beats L12, cool = worse); each cell annotated with the
    chosen layer, gain %, and 95% paired-bootstrap CI; cells whose CI excludes 0 are bold + ★; ID
    diagonal boxed. `mode` picks L and the filename/caption:
      'source_val' (PRIMARY): L = SOURCE-VALIDATION-selected layer (one per source row, no target
        data); also prints the oracle layer in small text. -> source_val_relative_gain_heatmap.
      'oracle' (DIAGNOSTIC): L = target-test argmin (optimistically biased). -> oracle_*.
    The layer is fixed BEFORE the bootstrap -> CIs descriptive, not selection-adjusted."""
    idx = {d: i for i, d in enumerate(DATASET_ORDER)}
    xlabels = ["Embed"] + [str(i) for i in range(1, NUM_LAYERS)]
    M = np.full((NDATA, NDATA), np.nan)
    cell_rg = {}
    for (src, tgt), bc in boot_cells.items():
        oracle_best = summ[(src, tgt)]["best_layer"]
        L = sel_by_src[src][0] if mode == "source_val" else oracle_best
        rg = _relative_gain_cell(bc, L)
        cell_rg[(src, tgt)] = (L, oracle_best, rg)
        M[idx[src], idx[tgt]] = rg["relative_gain_pct"]
    vmax = float(np.nanmax(np.abs(M))) or 1.0
    fig, ax = plt.subplots(figsize=(max(9.0, 2.9 * NDATA), max(7.5, 2.6 * NDATA)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)          # centered at 0; warm = positive gain
    for (src, tgt), (L, oracle_best, rg) in cell_rg.items():
        si, ti = idx[src], idx[tgt]
        sig = rg["excludes_zero"]
        if mode == "source_val":
            head, tail = f"sel {xlabels[L]}", f"\n(oracle {xlabels[oracle_best]})"
        else:
            head, tail = f"oracle {xlabels[L]}", ""
        txt = (("★ " if sig else "") + f"{head}\n{rg['relative_gain_pct']:+.1f}%\n"
               f"95% CI [{rg['ci_lo']:+.1f}, {rg['ci_hi']:+.1f}]" + tail)
        ax.text(ti, si, txt, ha="center", va="center", fontsize=7.5,
                fontweight=("bold" if sig else "normal"), color="k")
        if src == tgt:                                              # ID diagonal boxed
            ax.add_patch(plt.Rectangle((ti - 0.5, si - 0.5), 1, 1, fill=False, ec="k", lw=2.5))
    ax.set_xticks(range(NDATA)); ax.set_xticklabels([SHORT[d] for d in DATASET_ORDER])
    ax.set_yticks(range(NDATA)); ax.set_yticklabels([SHORT[d] for d in DATASET_ORDER])
    ax.set_xlabel("target (evaluation) dataset")
    ax.set_ylabel("source (probe training) dataset")
    if mode == "source_val":
        title = (f"PRIMARY — relative improvement over L12 at the SOURCE-VALIDATION-selected layer  "
                 f"[{qset}, seed {seed}]\nlayer chosen on source validation ONLY (one per source row, "
                 "no target data); warm = beats L12, cool = worse; ★/bold = 95% paired CI excludes 0; "
                 "boxed = ID.\ngains may be NEGATIVE; CIs descriptive (layer fixed before the bootstrap)")
        prefix = "source_val_relative_gain_heatmap"
    else:
        title = (f"DIAGNOSTIC (oracle) — relative improvement over L12 at the TARGET-TEST-best layer  "
                 f"[{qset}, seed {seed}]\noptimistically biased (L = test argmin, ≥0 by construction); "
                 "★/bold = CI excludes 0; boxed = ID.\nNOT the primary result — see the source_val heatmap")
        prefix = "oracle_relative_gain_heatmap"
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=REL_GAIN_LABEL)
    fig.tight_layout()
    out = FIG_DIR / f"{prefix}_{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def make_relative_gain_id_vs_ood(summ, boot_cells, qset, seed, mode="oracle", sel_by_src=None):
    """Descriptive ID-vs-OOD view: the diagonal (ID) and off-diagonal (OOD) relative gains shown
    individually, with each group's MEAN (solid) and MEDIAN (dashed) overlaid. `mode` picks the
    per-cell layer + filename/caption: 'source_val' (PRIMARY, source-validation-selected layer, gains
    may be negative) -> source_val_relative_gain_id_vs_ood; 'oracle' (DIAGNOSTIC, target-test-best,
    biased) -> oracle_*. The cells are NOT statistically independent, so this is descriptive only —
    no inferential test across cells."""
    cells = {"ID": [], "OOD": []}                        # (label, relative_gain_pct) per point
    for (src, tgt), bc in boot_cells.items():
        L = sel_by_src[src][0] if mode == "source_val" else summ[(src, tgt)]["best_layer"]
        rg = _relative_gain_cell(bc, L)
        cells["ID" if src == tgt else "OOD"].append((f"{SHORT[src]}→{SHORT[tgt]}", rg["relative_gain_pct"]))
    fig, ax = plt.subplots(figsize=(9.5, 7.0))
    rng = np.random.default_rng(SEED)
    colors = {"ID": "#4c72b0", "OOD": "#dd8452"}
    allv = [v for pts in cells.values() for _, v in pts]
    min_gap = 0.034 * ((max(allv) - min(allv)) or 1.0)   # min vertical spacing between stacked labels
    n_by = {}
    for g, x0 in (("ID", 0), ("OOD", 1)):
        pts = sorted(cells[g], key=lambda p: p[1])       # ascending -> leader lines don't cross
        n_by[g] = len(pts)
        if not pts:
            continue
        vals = np.array([p[1] for p in pts], float)
        xdot = x0 + rng.uniform(-0.05, 0.05, size=len(vals))
        ax.scatter(xdot, vals, s=60, color=colors[g], edgecolor="k", lw=0.5, zorder=3)
        # keep ALL labels: stack them to the right of the column with a greedy min-gap (bottom-up)
        # so none overlap, connected to their dot by a thin leader line.
        ly = vals.astype(float).copy()
        for i in range(1, len(ly)):
            if ly[i] < ly[i - 1] + min_gap:
                ly[i] = ly[i - 1] + min_gap
        for (lab, v), yl, xd in zip(pts, ly, xdot):
            ax.annotate(lab, xy=(xd, v), xytext=(x0 + 0.30, yl), fontsize=6.5, va="center", ha="left",
                        arrowprops=dict(arrowstyle="-", color="0.65", lw=0.5), annotation_clip=False)
        mean_v, med_v = float(vals.mean()), float(np.median(vals))
        ax.hlines(mean_v, x0 - 0.22, x0 + 0.22, color=colors[g], lw=2.6, zorder=4,
                  label=f"{g} mean {mean_v:+.1f}%")
        ax.hlines(med_v, x0 - 0.22, x0 + 0.22, color=colors[g], lw=1.6, ls="--", zorder=4,
                  label=f"{g} median {med_v:+.1f}%")
    ax.axhline(0, color="0.5", lw=1.0, ls=":")
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"ID  (n={n_by['ID']})", f"OOD  (n={n_by['OOD']})"])
    ax.set_xlim(-0.5, 1.95)
    ax.set_ylabel(REL_GAIN_LABEL)
    if mode == "source_val":
        title = ("Relative improvement over L12 — ID vs OOD  (PRIMARY: source-validation-selected "
                 f"layer)  [{qset}, seed {seed}]\neach point = one source→target cell (layer fixed on "
                 "SOURCE validation only); solid = mean, dashed = median; gains may be negative.\n"
                 "Descriptive only — the 16 cells are NOT statistically independent (no cross-cell "
                 "inference).")
        prefix = "source_val_relative_gain_id_vs_ood"
    else:
        title = ("Relative improvement over L12 — ID vs OOD  (DIAGNOSTIC: oracle target-test-best "
                 f"layer)  [{qset}, seed {seed}]\noptimistically biased (layer = test argmin); "
                 "descriptive only — NOT the primary result.")
        prefix = "oracle_relative_gain_id_vs_ood"
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / f"{prefix}_{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def _print_corrected_summary(sv_rows, or_rows, sel_by_src, qset):
    """Terminal summary of the CORRECTED (source-validation-selected) result, with the OLD ORACLE
    estimates alongside for comparison. Both sets are descriptive (layer fixed before the bootstrap);
    the source-val view is the fair primary result, the oracle view is optimistically biased."""
    def ms(rows, grp):
        v = np.array([r["relative_gain_pct"] for r in rows if r["id_or_ood"] == grp], float)
        return (float(v.mean()), float(np.median(v)), len(v)) if len(v) else (float("nan"),
                                                                              float("nan"), 0)
    print(f"\n  ══ CORRECTED primary result — layer selected on SOURCE VALIDATION only  ({qset}) ══")
    print("    source-validation-selected layer (reused for the ID cell + all transfer cells):")
    for src in DATASET_ORDER:
        if src in sel_by_src:
            L, vl = sel_by_src[src]
            print(f"       {SHORT[src]:12s} -> L{L:<2d}  (source-val q9 loss {vl:.4f})")
    n_pos = sum(1 for r in sv_rows if r["relative_gain_pct"] > 0)
    n_sig = sum(1 for r in sv_rows if r["relative_ci_excludes_zero"])
    n_sig_neg = sum(1 for r in sv_rows
                    if r["relative_ci_excludes_zero"] and r["relative_gain_pct"] < 0)
    id_m, id_md, id_n = ms(sv_rows, "ID")
    ood_m, ood_md, ood_n = ms(sv_rows, "OOD")
    print(f"    positive cells: {n_pos}/{len(sv_rows)}   |   95% rel-CI excludes 0: "
          f"{n_sig}/{len(sv_rows)}  (of which significantly NEGATIVE: {n_sig_neg})")
    print(f"    ID  (n={id_n:2d}):  mean {id_m:+.2f}%   median {id_md:+.2f}%")
    print(f"    OOD (n={ood_n:2d}):  mean {ood_m:+.2f}%   median {ood_md:+.2f}%")
    oid_m, oid_md, _ = ms(or_rows, "ID")
    ood_o_m, ood_o_md, _ = ms(or_rows, "OOD")
    o_pos = sum(1 for r in or_rows if r["relative_gain_pct"] > 0)
    print("    ── vs OLD ORACLE (target-test-best layer, optimistically biased) ──")
    print(f"    ID  oracle:  mean {oid_m:+.2f}%   median {oid_md:+.2f}%")
    print(f"    OOD oracle:  mean {ood_o_m:+.2f}%   median {ood_o_md:+.2f}%   "
          f"(oracle positive cells: {o_pos}/{len(or_rows)})")
    print("    (source-val = fair primary result; oracle CIs are post-selection / optimistic)")


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
    ap.add_argument("--target-datasets", nargs="+", default=None,
                    help="the datasets to EVALUATE the frozen probe on (default: all in the set).")
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
        _derive_datasets()            # matrix order/labels must follow the override
    if args.target_datasets is None:  # default = every dataset in the active set
        args.target_datasets = list(DATASET_ORDER)
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
