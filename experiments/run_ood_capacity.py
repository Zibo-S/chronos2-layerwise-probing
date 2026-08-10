"""Higher-capacity forecasting probes as capacity controls for the OOD-transfer pilot.

Question the linear pilot leaves open: is a probe's poor forecast (vs native Chronos-2) caused
by WEAK layer representations, or by the LIMITED CAPACITY of the linear readout? And does the
layerwise "tunnel" (earlier layer beats last) survive a higher-capacity decoder?

Two nonlinear head families (probing.heads / probing.probes), trained FROM SCRATCH per layer:
  - content_mlp_head          : ResidualBlock over the (n, 768) mean-pooled content vector.
  - forecast_slot_native_head : ONE shared ResidualBlock over the K native forecast slots
                                (n, K, 768) — the nonlinear analogue of the native head.

EVERYTHING else is inherited UNCHANGED from run_ood_transfer so the comparison to the committed
linear result is apples-to-apples: frozen amazon/chronos-2, 13 depths (Embed + L1..L12), C=512,
H=64, q9, arcsinh context normalization, Chronos-2 quantile loss, in-context seasonal-naive MASE
(m=24), strict source-only training (fit once per source, freeze, score every target), seed 0,
series-level cluster bootstrap. Fixed 300-epoch + wd-grid fit (no early stopping) — identical
budget to the committed linear probe. Feature/native caches are the existing per-(dataset,split)
files, so with a warm cache this is CPU / login-node only (the nonlinear HEAD fit is heavier than
the linear one — a short GPU salloc is faster).

Outputs are namespaced under results/<set>/ood_transfer/capacity/<family>/ and NEVER touch the
committed linear ood_transfer/ artifacts. Checkpoint identity includes probe_family and EXCLUDES
the target (electricity->kdd and electricity->uber reuse one checkpoint).

Run (one source per job, per family — three unique source-training runs per family):
    python -m experiments.run_ood_capacity --probe-family content_mlp_head \
        --source-dataset monash_electricity_hourly
    python -m experiments.run_ood_capacity --probe-family forecast_slot_native_head \
        --source-dataset monash_electricity_hourly
    python -m experiments.run_ood_capacity --probe-family content_mlp_head --figure-only
    python -m experiments.run_ood_capacity --compare        # cross-family + native-gap figures
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
from probing.config import NUM_LAYERS, LAST_LAYER, SEED, OUTPUT_PATCH_SIZE
from probing.id_data import build_windows, ROLLING_SETS
from probing.extraction import extract_window_features, extract_kout_features
from probing.heads import NATIVE_D_FF
from probing.probes import (QUANTILE_SETS, median_index,
                            fit_content_mlp_head, fit_content_mlp_head_explicit_val,
                            predict_content_mlp_head,
                            fit_forecast_slot_native_head, predict_forecast_slot_native_head)
# reuse the linear pilot's frame + pure helpers (importing only defines things; no main() runs)
from experiments import run_ood_transfer as ood
from experiments.run_ood_transfer import (_paired_delta_bootstrap, _mae_median_raw,
                                          _relative_gain_cell)
from experiments.run_id_forecasting import compute_mase, ID_STYLE
from experiments.run_ood_baselines import target_baselines

# ---- fixed experimental frame (inherited; DO NOT diverge from the linear pilot) ----
C, H = ood.C, ood.H                        # 512 / 64
QUANTILE_EPOCHS = ood.QUANTILE_EPOCHS      # 300 — same budget, NO early stopping
WD_GRID = ood.WD_GRID                      # (1e-5..1e-1) — same per-layer weight-decay grid
BOOT_B, CI_LO, CI_HI = ood.BOOT_B, ood.CI_LO, ood.CI_HI
P = OUTPUT_PATCH_SIZE                       # native output patch size (16)

# capacity-head hyperparameters (recorded in every checkpoint's meta)
HIDDEN_DIM = NATIVE_D_FF                    # native d_ff = 3072
DROPOUT = 0.0                              # deterministic probe (native head uses 0.1)

# |z| threshold (arcsinh-normalized target space) above which a median prediction is flagged
# EXTREME. z is the context-standardized + arcsinh label space: typical |z| is O(1-3), so |z|>10
# (≈ sinh(10) ≈ 1.1e4 context sigmas once un-transformed) is an unambiguous explosion. Fixed and
# documented — nothing is clipped or dropped; only counted.
EXTREME_NORM_THRESH = 10.0

FAMILIES = {
    "content_mlp_head": {
        "fit": fit_content_mlp_head, "fit_explicit_val": fit_content_mlp_head_explicit_val,
        "predict": predict_content_mlp_head,
        "feature_kind": "content", "pooling_or_token_type": "content"},
    "forecast_slot_native_head": {
        "fit": fit_forecast_slot_native_head, "predict": predict_forecast_slot_native_head,
        "feature_kind": "fslot", "pooling_or_token_type": "forecast_slot"},
}
LINEAR = "linear_content"                   # label for the committed linear pilot in comparisons


def _derive_datasets():
    """(Re)derive the source/target roster + matrix side length NDATA for the ACTIVE dataset set,
    tracking run_ood_transfer so extended_v3_rolling gets its uniform 4-dataset order instead of the
    import-time default (extended_v1's 3). Called at import and again after a --dataset-set override
    in main(); also re-derives ood's output dirs so the committed-linear reads resolve to this set."""
    global DATASET_ORDER, SHORT, NDATA
    ood._derive_datasets()
    ood._derive_dirs()
    DATASET_ORDER, SHORT = ood.DATASET_ORDER, ood.SHORT
    NDATA = len(DATASET_ORDER)


_derive_datasets()


# --------------------------------------------------------------------------- #
# output paths — all under results/<set>/ood_transfer/capacity/ (never the linear dir)
# --------------------------------------------------------------------------- #

def _cap_base():
    return config.ID_OUT_DIR / "ood_transfer" / "capacity"


def _fam_dirs(family):
    base = _cap_base() / family
    d = {"base": base, "ckpt": base / "checkpoints", "per_source": base / "per_source",
         "boot_in": base / "bootstrap_inputs", "fig": base / "figures"}
    for p in d.values():
        p.mkdir(parents=True, exist_ok=True)
    return d


def _cmp_dir():
    d = _cap_base() / "comparison"
    d.mkdir(parents=True, exist_ok=True)
    return d


# --------------------------------------------------------------------------- #
# frozen-probe identity + checkpoints (probe_family in; target NEVER in)
# --------------------------------------------------------------------------- #

def _probe_run_id(source, family, qset, seed):
    """Identity of a SOURCE-trained head. Includes probe_family + token type; EXCLUDES the target
    so every target evaluation of one source reuses the SAME checkpoint."""
    ptt = FAMILIES[family]["pooling_or_token_type"]
    return f"{source}__{family}__{ptt}__C{C}_H{H}__{qset}__seed{seed}"


def _ckpt_dir(source, family, qset, seed):
    return _fam_dirs(family)["ckpt"] / _probe_run_id(source, family, qset, seed)


def _ckpt_meta(source, family, qset, seed, quantiles):
    return {"source": source, "probe_family": family,
            "pooling_or_token_type": FAMILIES[family]["pooling_or_token_type"],
            "context_length": C, "horizon": H, "patch_size": P,
            "quantile_set": qset, "quantile_config": int(len(quantiles)),
            "quantiles": [float(x) for x in quantiles], "seed": int(seed),
            "model": "amazon/chronos-2",
            "normalization": "arcsinh-context (per-window, context-only) + source-fit StandardScaler",
            "epochs": QUANTILE_EPOCHS, "wd_grid": list(WD_GRID),
            "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
            "layer_scheme": f"Embed(L0)+L1..L{NUM_LAYERS - 1}"}


def save_checkpoints(fitted, source, family, qset, seed, quantiles):
    d = _ckpt_dir(source, family, qset, seed)
    d.mkdir(parents=True, exist_ok=True)
    meta = _ckpt_meta(source, family, qset, seed, quantiles)
    for i, f in fitted.items():
        payload = {"head_state": f["head"].state_dict(), "scaler": f["scaler"], "wd": f["wd"],
                   "selection": f["selection"], "source_val_loss": f["source_val_loss"],
                   "in_features": f["in_features"], "out_features": f["out_features"],
                   "hidden_dim": f["hidden_dim"], "dropout": f["dropout"], "family": f["family"],
                   "pooling_or_token_type": f["pooling_or_token_type"],
                   "param_count": f["param_count"], "layer": int(i), "meta": meta}
        if "output_patch_size" in f:
            payload["output_patch_size"] = f["output_patch_size"]
        torch.save(payload, d / f"L{i:02d}.pt")
    json.dump(meta, open(d / "run_meta.json", "w"), indent=2)
    print(f"  [saved] frozen {family} probe -> {d}  ({NUM_LAYERS} layer checkpoints)")
    return d


def load_checkpoints(source, family, qset, seed, quantiles, device):
    """Resume a frozen source probe iff ALL layer checkpoints exist and their meta matches;
    a mismatch (incl. probe_family) fails loudly."""
    from probing.heads import build_head
    d = _ckpt_dir(source, family, qset, seed)
    paths = [d / f"L{i:02d}.pt" for i in range(NUM_LAYERS)]
    if not all(p.exists() for p in paths):
        return None
    want = _ckpt_meta(source, family, qset, seed, quantiles)
    fitted = {}
    for i, p in enumerate(paths):
        ck = torch.load(p, map_location=device, weights_only=False)
        got = ck["meta"]
        for k in ("source", "probe_family", "pooling_or_token_type", "context_length", "horizon",
                  "patch_size", "quantile_set", "quantile_config", "seed"):
            if got.get(k) != want[k]:
                raise RuntimeError(f"checkpoint {p} meta[{k}]={got.get(k)!r} != expected "
                                   f"{want[k]!r} — stale/mismatched probe; delete the run dir and re-fit")
        head = build_head(ck["in_features"], ck["out_features"], hidden_dim=ck["hidden_dim"],
                          dropout=ck["dropout"], device=device)
        head.load_state_dict(ck["head_state"])
        head.eval()
        fitted[i] = {"scaler": ck["scaler"], "head": head, "wd": float(ck["wd"]),
                     "selection": ck["selection"], "source_val_loss": ck["source_val_loss"],
                     "in_features": ck["in_features"], "out_features": ck["out_features"],
                     "hidden_dim": ck["hidden_dim"], "dropout": ck["dropout"],
                     "family": ck["family"], "pooling_or_token_type": ck["pooling_or_token_type"],
                     "param_count": ck["param_count"], "device": str(device)}
        if "output_patch_size" in ck:
            fitted[i]["output_patch_size"] = ck["output_patch_size"]
    print(f"  [resume] loaded frozen {family} probe from {d} (no re-training)")
    return fitted


# --------------------------------------------------------------------------- #
# feature extraction dispatch (content-pooled vs forecast-slot)
# --------------------------------------------------------------------------- #

def _extract(family, tag, split, w):
    """Return {layer: features} for one dataset split, in the shape the family's head consumes.
    split ∈ {train, val, test}; 'val' is the rolling sets' explicit temporal validation split."""
    Xkey, ykey = {"train": ("X_train", "y_train"), "val": ("X_val", "y_val"),
                  "test": ("X_test", "y_test")}[split]
    if FAMILIES[family]["feature_kind"] == "content":
        f, _y = extract_window_features(tag, split, w[Xkey], w[ykey], pooling="content")
        return f
    feats, _final, _y = extract_kout_features(tag, split, w[Xkey], w[ykey], horizon=H)
    return feats["fslot"]                     # {layer: (n, K, 768)}


# --------------------------------------------------------------------------- #
# train ONCE on source; evaluate the frozen probe on each target
# --------------------------------------------------------------------------- #

def get_source_probe(source, family, qset, quantiles, seed, device):
    """Frozen SOURCE probe (loaded if checkpointed, else fit + saved). Source train split only."""
    fitted = load_checkpoints(source, family, qset, seed, quantiles, device)
    if fitted is not None:
        return fitted, _ckpt_dir(source, family, qset, seed)
    print(f"  [fit] training {family} source probe on {source} (seed {seed}, {qset})")
    w = build_windows(source)
    f_tr = _extract(family, source, "train", w)
    if config.DATASET_SET in ROLLING_SETS:
        # rolling sets: weight-decay AND the downstream source-selected layer are chosen on the
        # EXPLICIT temporal val split (no 80/20 carve); the head stays trained on FULL train only.
        fit_ev = FAMILIES[family].get("fit_explicit_val")
        if fit_ev is None:
            raise SystemExit(
                f"{family} has no explicit-temporal-val fit — rolling sets {sorted(ROLLING_SETS)} "
                "require it; only content_mlp_head is supported under extended_v3_rolling")
        f_va = _extract(family, source, "val", w)
        print(f"    [rolling] explicit temporal val: {w['meta']['n_val']} windows / "
              f"{w['meta']['n_val_series']} series (no 80/20 carve)")
        fitted = fit_ev(f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"], quantiles=quantiles,
                        epochs=QUANTILE_EPOCHS, wd_grid=WD_GRID, device=device,
                        hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
        del f_va
    else:
        fit_fn = FAMILIES[family]["fit"]
        fitted = fit_fn(f_tr, w["Y_train_traj"], quantiles=quantiles, epochs=QUANTILE_EPOCHS,
                        wd_grid=WD_GRID, device=device, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    ckpt = save_checkpoints(fitted, source, family, qset, seed, quantiles)
    del w, f_tr
    gc.collect()
    return fitted, ckpt


def _source_selected_layer(fitted):
    """argmin over layers of the SOURCE-validation loss (chosen-wd 80/20 carve). NEVER uses the
    target — this is the primary layer for a fair, deployable OOD comparison. None if source-val
    loss was not recorded (only when wd_grid was omitted)."""
    svl = {i: fitted[i]["source_val_loss"] for i in range(NUM_LAYERS)}
    if any(v is None for v in svl.values()):
        return None
    return int(min(svl, key=lambda i: svl[i]))


def evaluate_target(fitted, ckpt_dir, source, target, family, qset, quantiles, seed, device):
    """Score the frozen SOURCE probe on TARGET's test split. Returns (rows, summary)."""
    is_ood = source != target
    tag = "OOD" if is_ood else "ID "
    print(f"  [eval {tag}] {family}: {SHORT.get(source, source)} -> {SHORT.get(target, target)}")
    w = build_windows(target)
    f_te = _extract(family, target, "test", w)
    predict_fn = FAMILIES[family]["predict"]
    out, diag = predict_fn(fitted, f_te, w["Y_test_traj"], quantiles=quantiles, device=device,
                           collect_test_median=True, collect_test_window_loss=True)
    # MASE via the per-dataset pipeline (target seasonal-naive denom + cached target native).
    # mase_pw[family] is (NUM_LAYERS, n) per-window MASE -> robust raw-scale summaries below.
    mase_entry, mase_pw = compute_mase(target, w, {family: diag})
    mae = _mae_median_raw(w, diag)
    pw_by_layer = mase_pw[family]                          # (NUM_LAYERS, n)

    sid = np.asarray(w["series_test"], np.int64)
    n_test, n_series = int(w["meta"]["n_test"]), int(len(np.unique(sid)))
    ptt = FAMILIES[family]["pooling_or_token_type"]
    param_count = int(fitted[0]["param_count"])
    run_id = _probe_run_id(source, family, qset, seed)
    rows = []
    for i in range(NUM_LAYERS):
        pw = np.asarray(pw_by_layer[i], np.float64)        # per-window MASE for this layer
        zmed = np.asarray(diag["test_median"][i], np.float64)   # (n, H) median pred, arcsinh space
        rows.append({
            "source_dataset": source, "target_dataset": target, "layer": i, "seed": int(seed),
            "split": "test", "is_ood": is_ood, "probe_family": family, "pooling": ptt,
            "parameter_count": param_count, "quantile_config": int(len(quantiles)),
            "quantile_set": qset, "context_length": C, "prediction_length": H,
            "probe_run_id": run_id, "probe_checkpoint": str(ckpt_dir / f"L{i:02d}.pt"),
            "selected_source_hyperparameters": json.dumps(
                {"weight_decay": fitted[i]["wd"], "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT}),
            "quantile_loss": float(out[i]),
            "mean_pinball_loss": float(diag["test_mean_pinball"][i]),
            "mase": float(mase_entry["poolings"][family][i]),         # == mase_mean (window mean)
            "mase_median": float(np.median(pw)),
            "mase_p95": float(np.percentile(pw, 95)),
            "mase_max": float(pw.max()),
            "mae_median_raw": float(mae[i]),
            "pred_norm_max": float(np.abs(zmed).max()),
            "n_extreme_norm_pred": int((np.abs(zmed) > EXTREME_NORM_THRESH).sum()),
            "n_pred_elements": int(zmed.size),
            "extreme_norm_threshold": EXTREME_NORM_THRESH,
            "n_test_windows": n_test, "n_test_series": n_series})

    ql = np.array([out[i] for i in range(NUM_LAYERS)], float)
    oracle = int(ql.argmin())
    ssl = _source_selected_layer(fitted)
    summary = {"source_dataset": source, "target_dataset": target, "seed": int(seed),
               "is_ood": is_ood, "probe_family": family, "quantile_set": qset,
               "quantile_config": int(len(quantiles)), "parameter_count": param_count,
               "oracle_best_layer": oracle, "best_layer": oracle,        # best_layer alias for figures
               "min_loss": float(ql[oracle]),
               "source_selected_layer": ssl,
               "source_selected_loss": (None if ssl is None else float(ql[ssl])),
               "last_layer": LAST_LAYER, "last_layer_loss": float(ql[LAST_LAYER]),
               "delta_vs_last": float(ql[LAST_LAYER] - ql[oracle]),
               "native_mase": float(mase_entry["native_mase"]),
               "n_test_windows": n_test, "n_test_series": n_series}

    _save_boot_inputs(source, target, family, qset, seed, sid, diag, n_test)
    del w, f_te
    gc.collect()
    return rows, summary


def _save_boot_inputs(source, target, family, qset, seed, sid, diag, n_test):
    wl = np.stack([diag["test_window_loss"][i] for i in range(NUM_LAYERS)]).astype(np.float64)
    assert wl.shape == (NUM_LAYERS, n_test), f"window-loss {wl.shape} != ({NUM_LAYERS}, {n_test})"
    out = _fam_dirs(family)["boot_in"] / f"{source}__to__{target}__{family}__{qset}__seed{seed}.npz"
    np.savez(out, window_loss=wl, series_test=sid)


def run_source(source, targets, family, qset, quantiles, seed, device):
    print(f"\n{'=' * 70}\n[{family}  source={source}  seed={seed}  {qset}] train once, eval "
          f"{len(targets)} targets\n{'=' * 70}")
    fitted, ckpt_dir = get_source_probe(source, family, qset, quantiles, seed, device)
    all_rows, all_summ = [], []
    for target in targets:
        rows, summ = evaluate_target(fitted, ckpt_dir, source, target, family, qset, quantiles,
                                     seed, device)
        all_rows += rows
        all_summ.append(summ)
    payload = {"run_meta": {"source_dataset": source, "probe_family": family,
                            "targets": list(targets), "seed": int(seed), "quantile_set": qset,
                            "quantile_config": int(len(quantiles)), "context_length": C,
                            "prediction_length": H, "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT,
                            "param_count": int(fitted[0]["param_count"]),
                            "probe_run_id": _probe_run_id(source, family, qset, seed),
                            "checkpoint_dir": str(ckpt_dir)},
               "rows": all_rows, "summaries": all_summ}
    out = _fam_dirs(family)["per_source"] / f"{source}__{family}__{qset}__seed{seed}.json"
    json.dump(payload, open(out, "w"), indent=2)
    print(f"  [saved] {out}  ({len(all_rows)} rows, {len(all_summ)} cells)")
    return payload


# --------------------------------------------------------------------------- #
# aggregate one family -> tables + per-family 3x3 figure
# --------------------------------------------------------------------------- #

def _load_per_source(family, qset, seed):
    payloads = {}
    for source in DATASET_ORDER:
        p = _fam_dirs(family)["per_source"] / f"{source}__{family}__{qset}__seed{seed}.json"
        if p.exists():
            payloads[source] = json.load(open(p))
    return payloads


def _boot_cell(source, target, family, qset, seed):
    p = _fam_dirs(family)["boot_in"] / f"{source}__to__{target}__{family}__{qset}__seed{seed}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return _paired_delta_bootstrap(d["window_loss"], d["series_test"])


def _panel_curve(family, src, tgt, qset, seed, key="quantile_loss"):
    pl = json.load(open(_fam_dirs(family)["per_source"] / f"{src}__{family}__{qset}__seed{seed}.json"))
    by_layer = {r["layer"]: r[key] for r in pl["rows"] if r["target_dataset"] == tgt}
    return np.array([by_layer[i] for i in range(NUM_LAYERS)], float)


def aggregate(family, qset, seed):
    payloads = _load_per_source(family, qset, seed)
    if not payloads:
        print(f"  [aggregate] no {family} results for {qset} seed {seed} yet — run the sources first")
        return
    fam = _fam_dirs(family)
    rows = [r for pl in payloads.values() for r in pl["rows"]]
    fields = ["source_dataset", "target_dataset", "layer", "seed", "split", "is_ood",
              "probe_family", "pooling", "parameter_count", "quantile_config", "quantile_set",
              "context_length", "prediction_length", "probe_run_id", "probe_checkpoint",
              "selected_source_hyperparameters", "quantile_loss", "mean_pinball_loss", "mase",
              "mase_median", "mase_p95", "mase_max", "mae_median_raw", "pred_norm_max",
              "n_extreme_norm_pred", "n_pred_elements", "extreme_norm_threshold",
              "n_test_windows", "n_test_series"]
    with open(fam["base"] / f"ood_capacity_results__{family}__{qset}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(fam["base"] / f"ood_capacity_results__{family}__{qset}.json", "w"), indent=2)

    summ = {(s["source_dataset"], s["target_dataset"]): s
            for pl in payloads.values() for s in pl["summaries"]}
    boot_cells = {}
    for (src, tgt) in summ:
        bc = _boot_cell(src, tgt, family, qset, seed)
        if bc is not None:
            boot_cells[(src, tgt)] = bc
    _write_delta_table(family, boot_cells, summ, qset, seed)
    _write_summary(family, summ, boot_cells, qset, seed)
    _write_param_count(family, payloads, qset, seed)
    _write_selected_layers(family, payloads, qset, seed)
    # normalized improvement over L12 (relative_gain_pct + paired-bootstrap CI), both layer views:
    #   source_val = source-validation-selected layer (PRIMARY, one per source row, no target data)
    #   oracle     = target-test argmin (DIAGNOSTIC, optimistically biased)
    _write_relative_gain_table(family, boot_cells, summ, qset, seed, "source_val")
    _write_relative_gain_table(family, boot_cells, summ, qset, seed, "oracle")
    _write_linear_comparison(family, boot_cells, summ, qset, seed)
    _make_matrix_figure(family, summ, boot_cells, qset, seed)
    _make_baseline_bars_figure(family, summ, rows, qset, seed)
    print(f"  [saved] {family} tables + figure under {fam['base']} ({len(rows)} rows)")


def _write_delta_table(family, boot_cells, summ, qset, seed):
    fields = ["source_dataset", "target_dataset", "probe_family", "seed", "is_ood", "quantile_set",
              "layer", "delta_vs_last", "delta_ci_lo", "delta_ci_hi", "delta_above_zero",
              "is_oracle_best_layer", "is_source_selected_layer", "is_last_layer",
              "n_test_series", "n_test_windows", "bootstrap_replicates"]
    rows = []
    for (src, tgt), bc in sorted(boot_cells.items()):
        oracle = summ[(src, tgt)]["oracle_best_layer"]
        ssl = summ[(src, tgt)]["source_selected_layer"]
        for L in range(NUM_LAYERS):
            rows.append({"source_dataset": src, "target_dataset": tgt, "probe_family": family,
                         "seed": int(seed), "is_ood": src != tgt, "quantile_set": qset, "layer": L,
                         "delta_vs_last": float(bc["delta_vs_last"][L]),
                         "delta_ci_lo": float(bc["delta_ci_lo"][L]),
                         "delta_ci_hi": float(bc["delta_ci_hi"][L]),
                         "delta_above_zero": bool(bc["delta_above_zero"][L]),
                         "is_oracle_best_layer": L == oracle, "is_source_selected_layer": L == ssl,
                         "is_last_layer": L == LAST_LAYER,
                         "n_test_series": bc["n_series"], "n_test_windows": bc["n_windows"],
                         "bootstrap_replicates": BOOT_B})
    base = _fam_dirs(family)["base"]
    with open(base / f"ood_capacity_delta_vs_last__{family}__{qset}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(base / f"ood_capacity_delta_vs_last__{family}__{qset}.json", "w"), indent=2)


def _write_summary(family, summ, boot_cells, qset, seed):
    out_cells = []
    for (src, tgt), s in summ.items():
        e = dict(s)
        bc = boot_cells.get((src, tgt))
        if bc is not None:
            L = s["oracle_best_layer"]
            e["delta_vs_last_ci"] = [float(bc["delta_ci_lo"][L]), float(bc["delta_ci_hi"][L])]
            e["delta_above_zero"] = bool(bc["delta_above_zero"][L])
            e["bootstrap"] = {"n_replicates": BOOT_B, "seed": int(seed), "n_test_series": bc["n_series"]}
        out_cells.append(e)
    payload = {"config": {"probe_family": family, "quantile_set": qset,
                          "quantile_config": len(QUANTILE_SETS[qset]), "seed": int(seed),
                          "context_length": C, "prediction_length": H, "last_layer": LAST_LAYER,
                          "hidden_dim": HIDDEN_DIM, "dropout": DROPOUT, "epochs": QUANTILE_EPOCHS,
                          "bootstrap_replicates": BOOT_B,
                          "oracle_best_layer": "argmin over TARGET-test loss (diagnostic, not deployable)",
                          "source_selected_layer": "argmin over SOURCE-validation loss (fair OOD layer)",
                          "delta_vs_last": "loss[last] − loss[oracle_best]; positive = earlier layer wins"},
               "cells": out_cells}
    json.dump(payload, open(_fam_dirs(family)["base"] / f"ood_capacity_summary__{family}__{qset}.json",
                            "w"), indent=2)


def _write_param_count(family, payloads, qset, seed):
    pc = int(next(iter(payloads.values()))["run_meta"]["param_count"])
    # linear-baseline param count for context: Linear(d, Q*out) with bias.
    Q, d = len(QUANTILE_SETS[qset]), 768
    lin_out = H if FAMILIES[family]["feature_kind"] == "content" else P
    lin_params = (d + 1) * Q * lin_out
    payload = {"probe_family": family, "quantile_set": qset, "hidden_dim": HIDDEN_DIM,
               "dropout": DROPOUT, "head_parameter_count": pc,
               "linear_baseline_probe": ("quantile (pooled)" if lin_out == H else "shared_forecast"),
               "linear_baseline_parameter_count": int(lin_params),
               "capacity_ratio_vs_linear": round(pc / lin_params, 2)}
    json.dump(payload, open(_fam_dirs(family)["base"] / f"parameter_counts__{family}__{qset}.json",
                            "w"), indent=2)


def _write_selected_layers(family, payloads, qset, seed):
    """One row per (source,target,layer_selection_type ∈ {oracle_best, source_selected, final})
    with q9_loss / mase / mae for the probe and the target baselines (native / seasonal / last)."""
    by_cell = {(r["source_dataset"], r["target_dataset"]): {} for pl in payloads.values() for r in pl["rows"]}
    for pl in payloads.values():
        for r in pl["rows"]:
            by_cell[(r["source_dataset"], r["target_dataset"])][r["layer"]] = r
    summ = {(s["source_dataset"], s["target_dataset"]): s
            for pl in payloads.values() for s in pl["summaries"]}
    baselines = {t: target_baselines(t, qset, QUANTILE_SETS[qset]) for t in DATASET_ORDER
                 if any(tt == t for (_s, tt) in summ)}
    fields = ["source_dataset", "target_dataset", "probe_family", "is_ood", "quantile_set",
              "layer_selection_type", "layer", "q9_loss", "mase", "mae",
              "native_mase", "native_mae", "seasonal_naive_mase", "last_value_mase"]
    rows = []
    for (src, tgt), s in summ.items():
        b = baselines[tgt]["metrics"]
        picks = {"oracle_best": s["oracle_best_layer"], "source_selected": s["source_selected_layer"],
                 "final": LAST_LAYER}
        for sel_type, L in picks.items():
            if L is None:
                continue
            r = by_cell[(src, tgt)][L]
            rows.append({"source_dataset": src, "target_dataset": tgt, "probe_family": family,
                         "is_ood": src != tgt, "quantile_set": qset, "layer_selection_type": sel_type,
                         "layer": L, "q9_loss": r["quantile_loss"], "mase": r["mase"],
                         "mae": r["mae_median_raw"],
                         "native_mase": b["native_chronos2"]["mase"], "native_mae": b["native_chronos2"]["mae"],
                         "seasonal_naive_mase": b["seasonal_naive"]["mase"],
                         "last_value_mase": b["last_value"]["mase"]})
    base = _fam_dirs(family)["base"]
    with open(base / f"ood_capacity_selected_layers__{family}__{qset}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(base / f"ood_capacity_selected_layers__{family}__{qset}.json", "w"), indent=2)


def _write_relative_gain_table(family, boot_cells, summ, qset, seed, mode):
    """Normalized improvement over L12 per source->target cell for one layer-selection view.

    mode='source_val' (PRIMARY): layer = source-validation-selected (one per source row, NO target
      data). mode='oracle' (DIAGNOSTIC): layer = target-test argmin (optimistically biased).
    relative_gain_pct = 100*(loss[L12]-loss[L])/loss[L12], with the paired series-cluster-bootstrap
    ratio CI (layer fixed BEFORE resampling -> descriptive, not selection-adjusted). Gains may be
    negative. Writes <mode>_relative_gain__<family>__<qset>.csv/json."""
    base = _fam_dirs(family)["base"]
    rows = []
    for (src, tgt), bc in sorted(boot_cells.items()):
        s = summ[(src, tgt)]
        oracle_best = int(s["oracle_best_layer"])
        ss = s.get("source_selected_layer")
        if mode == "source_val" and ss is None:
            continue
        L = int(ss) if mode == "source_val" else oracle_best
        rg = _relative_gain_cell(bc, L)
        rows.append({
            "source_dataset": src, "target_dataset": tgt,
            "id_or_ood": ("ID" if src == tgt else "OOD"),
            "probe_family": family, "layer_selection": mode, "selected_layer": L,
            "selected_layer_test_q9_loss": round(float(bc["point"][L]), 6),
            "last_layer_test_q9_loss": round(float(bc["point"][LAST_LAYER]), 6),
            "raw_gain": round(float(bc["delta_vs_last"][L]), 6),
            "raw_ci_lo": round(float(bc["delta_ci_lo"][L]), 6),
            "raw_ci_hi": round(float(bc["delta_ci_hi"][L]), 6),
            "relative_gain_pct": round(rg["relative_gain_pct"], 4),
            "relative_ci_lo": round(rg["ci_lo"], 4), "relative_ci_hi": round(rg["ci_hi"], 4),
            "relative_ci_excludes_zero": rg["excludes_zero"],
            "oracle_best_layer": oracle_best,
            "n_test_windows": int(bc["n_windows"]), "n_bootstrap_clusters": int(bc["n_series"])})
    stem = f"{mode}_relative_gain__{family}__{qset}"
    fields = list(rows[0].keys()) if rows else []
    with open(base / f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(base / f"{stem}.json", "w"), indent=2)
    print(f"  [saved] {mode} relative-gain table ({len(rows)} cells) -> {base / (stem + '.csv')}")
    return rows


def _committed_linear_source_val(qset):
    """Linear source-val relative-gain per (src,tgt) from run_ood_transfer's committed seen-matrix
    table (ood.OOD_DIR is re-derived to the active set in _derive_datasets), or {} if absent."""
    p = ood.OOD_DIR / f"source_val_relative_gain_summary_{qset}.json"
    if not p.exists():
        return {}
    return {(r["source_dataset"], r["target_dataset"]): r for r in json.load(open(p))}


def _write_linear_comparison(family, boot_cells, summ, qset, seed):
    """Per-cell nonlinear-vs-linear comparison at each head's OWN source-validation-selected layer
    (normalized quantile loss primary). Joins this family's source_val gain with the committed LINEAR
    source_val table (run_ood_transfer). nonlinear_minus_linear_* = nonlinear − linear (negative loss
    delta / positive gain delta => the nonlinear head helps). Linear absent -> linear_* = null."""
    base = _fam_dirs(family)["base"]
    lin = _committed_linear_source_val(qset)
    rows = []
    for (src, tgt), bc in sorted(boot_cells.items()):
        ss = summ[(src, tgt)].get("source_selected_layer")
        if ss is None:
            continue
        L = int(ss)
        rg = _relative_gain_cell(bc, L)
        nl_sel_loss, nl_l12 = float(bc["point"][L]), float(bc["point"][LAST_LAYER])
        lr = lin.get((src, tgt))
        lin_layer = int(lr["source_val_selected_layer"]) if lr else None
        lin_sel_loss = float(lr["selected_layer_test_q9_loss"]) if lr else None
        lin_l12 = float(lr["last_layer_test_q9_loss"]) if lr else None
        lin_rg = float(lr["relative_gain_pct"]) if lr else None
        rows.append({
            "source_dataset": src, "target_dataset": tgt,
            "id_or_ood": ("ID" if src == tgt else "OOD"), "probe_family": family,
            "nonlinear_source_selected_layer": L,
            "nonlinear_sel_q9_loss": round(nl_sel_loss, 6),
            "nonlinear_l12_q9_loss": round(nl_l12, 6),
            "nonlinear_relative_gain_pct": round(rg["relative_gain_pct"], 4),
            "nonlinear_relative_ci_lo": round(rg["ci_lo"], 4),
            "nonlinear_relative_ci_hi": round(rg["ci_hi"], 4),
            "nonlinear_relative_ci_excludes_zero": rg["excludes_zero"],
            "linear_source_selected_layer": lin_layer,
            "linear_sel_q9_loss": (None if lin_sel_loss is None else round(lin_sel_loss, 6)),
            "linear_l12_q9_loss": (None if lin_l12 is None else round(lin_l12, 6)),
            "linear_relative_gain_pct": (None if lin_rg is None else round(lin_rg, 4)),
            "nonlinear_minus_linear_sel_q9_loss": (None if lin_sel_loss is None
                                                   else round(nl_sel_loss - lin_sel_loss, 6)),
            "nonlinear_minus_linear_relative_gain_pct": (None if lin_rg is None
                                                         else round(rg["relative_gain_pct"] - lin_rg, 4))})
    stem = f"capacity_vs_linear__{family}__{qset}"
    fields = list(rows[0].keys()) if rows else []
    with open(base / f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(base / f"{stem}.json", "w"), indent=2)
    n_lin = sum(1 for r in rows if r["linear_relative_gain_pct"] is not None)
    print(f"  [saved] nonlinear-vs-linear comparison ({len(rows)} cells, {n_lin} with linear) -> "
          f"{base / (stem + '.csv')}")
    return rows


def _make_matrix_figure(family, summ, boot_cells, qset, seed):
    """Per-family transfer grid: Chronos-2 q loss by depth, cluster-bootstrap band, ★ at the
    oracle-best layer, ◆ at the source-selected layer, filled markers where paired Δ-vs-last CI>0,
    dotted line at the final layer. Diagonal (ID) tinted; y shared within a column."""
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
                        color="0.6")
                ax.set_xticks(xs)
                continue
            curve = _panel_curve(family, src, tgt, qset, seed)
            ax.plot(xs, curve, color=color, lw=2.3, marker="o", ms=4, zorder=3)
            bc = boot_cells.get((src, tgt))
            if bc is not None:
                ax.fill_between(xs, bc["ci_lo"], bc["ci_hi"], color=color, alpha=0.16, lw=0,
                                label="95% cluster-bootstrap CI")
                sig = bc["delta_above_zero"].copy()
                sig[LAST_LAYER] = False
                if sig.any():
                    ax.plot(xs[sig], curve[sig], "o", color=color, ms=7, mec="k", mew=0.9,
                            zorder=4, label="Δ vs last: CI > 0")
            o = s["oracle_best_layer"]
            ax.plot(o, curve[o], marker="*", ms=16, color=color, mec="k", mew=0.6, zorder=5,
                    label=f"oracle = {xlabels[o]}")
            ssl = s["source_selected_layer"]
            if ssl is not None:
                ax.plot(ssl, curve[ssl], marker="D", ms=8, color="white", mec=color, mew=1.8,
                        zorder=6, label=f"source-sel = {xlabels[ssl]}")
            ax.axvline(LAST_LAYER, color="0.4", ls=":", lw=1.1)
            if not s["is_ood"]:
                ax.set_facecolor("#f4f4ec")
                for sp in ax.spines.values():
                    sp.set_linewidth(1.8)
            tagtxt = "OOD (cross-dataset)" if s["is_ood"] else "ID (in-dataset)"
            ax.set_title(f"{SHORT[src]} → {SHORT[tgt]}  [{tagtxt}]\n"
                         f"oracle {xlabels[o]}, Δ vs last = {s['delta_vs_last']:+.3f}",
                         fontsize=10, fontweight=("bold" if not s["is_ood"] else "normal"))
            ax.set_xticks(xs)
            ax.set_xticklabels(xlabels, fontsize=7)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=6.5, loc="best")
    for ax in axes[-1]:
        ax.set_xlabel("representation (Embed + L1..L12)")
    for ri, src in enumerate(DATASET_ORDER):
        axes[ri, 0].set_ylabel(f"probe trained on {SHORT[src]}\nChronos-2 q loss (test)")
    pc = summ[next(iter(summ))]["parameter_count"]
    fig.suptitle(f"{family} — cross-dataset transfer ({NDATA}×{NDATA}), Chronos-2 quantile loss by layer  "
                 f"[{qset}, {pc:,} head params, seed {seed}]\n"
                 "★ = oracle-best (target test) · ◆ = source-selected (source val) · "
                 "filled dot = paired Δ-vs-last CI excludes 0 · y shared within a column",
                 fontsize=12, y=0.996)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = _fam_dirs(family)["fig"] / f"transfer_matrix_{NDATA}x{NDATA}__{family}__{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def _make_baseline_bars_figure(family, summ, rows, qset, seed):
    """Per-family clone of run_ood_baselines' baseline_comparison: 3×3 grid, each source→target
    cell shows target-space MASE for capacity-probe-best, LINEAR-probe-best, capacity-probe-final,
    LINEAR-probe-final, native Chronos-2, seasonal-naive, last-value (LOWER = better). Capacity and
    linear best layers are each the source→target quantile-loss argmin; references identical down a column."""
    by_cell = {}
    for r in rows:
        by_cell.setdefault((r["source_dataset"], r["target_dataset"]), {})[r["layer"]] = r["mase"]
    bmetrics = {t: target_baselines(t, qset, QUANTILE_SETS[qset])["metrics"]
                for t in DATASET_ORDER if any(tt == t for (_s, tt) in summ)}
    lin = _committed_linear_curves(qset) or {}     # {(src,tgt): {layer: {quantile_loss, mase}}}
    labels = ["cap\nbest", "lin\nbest", "cap\nfinal", "lin\nfinal", "native", "seasonal", "last"]
    fig, axes = plt.subplots(NDATA, NDATA, figsize=(5.6 * NDATA, 4.0 * NDATA), sharex=True)
    for ri, src in enumerate(DATASET_ORDER):
        color = ID_STYLE.get(src, {}).get("color", "#333333")
        for ci, tgt in enumerate(DATASET_ORDER):
            ax = axes[ri, ci]
            s = summ.get((src, tgt))
            if s is None:
                ax.text(0.5, 0.5, "(not run)", ha="center", va="center", transform=ax.transAxes,
                        color="0.6")
                ax.set_xticks(range(7))
                continue
            L_best, lm, b = s["oracle_best_layer"], by_cell[(src, tgt)], bmetrics[tgt]
            lc = lin.get((src, tgt))
            if lc is not None:
                lin_best_L = int(min(lc, key=lambda i: lc[i]["quantile_loss"]))
                lin_best, lin_final = lc[lin_best_L]["mase"], lc[LAST_LAYER]["mase"]
            else:
                lin_best_L, lin_best, lin_final = None, np.nan, np.nan
            vals = [lm[L_best], lin_best, lm[LAST_LAYER], lin_final,
                    b["native_chronos2"]["mase"], b["seasonal_naive"]["mase"], b["last_value"]["mase"]]
            colors = [color, "#4c72b0", color, "#4c72b0", "#555555", "#999999", "#c0c0c0"]
            bars = ax.bar(range(7), vals, color=colors, edgecolor="k", linewidth=0.6)
            bars[2].set_alpha(0.55)      # capacity final-layer
            bars[3].set_alpha(0.55)      # linear final-layer
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.2f}", ha="center",
                            va="bottom", fontsize=7)
            is_ood = src != tgt
            lin_lbl = f" / lin L{lin_best_L}" if lin_best_L is not None else ""
            ax.set_title(f"{SHORT[src]} → {SHORT[tgt]}  [{'OOD' if is_ood else 'ID'}]  "
                         f"best cap L{L_best}{lin_lbl}", fontsize=9,
                         fontweight="normal" if is_ood else "bold")
            ax.set_xticks(range(7)); ax.set_xticklabels(labels, fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            if not is_ood:
                ax.set_facecolor("#f4f4ec")
        axes[ri, 0].set_ylabel(f"probe: {SHORT[src]}\nMASE (test)")
    pc = summ[next(iter(summ))]["parameter_count"]
    fig.suptitle(f"{family} (colored) vs linear probe (blue) vs target baselines — MASE  "
                 f"[{qset}, {pc:,} head params, seed {seed}]\n"
                 "rows = source (probe) dataset, cols = target;  LOWER = better;  cap/lin = "
                 "capacity/linear probe (best & final layer);  native/seasonal/last identical down a column",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = _fam_dirs(family)["fig"] / f"baseline_comparison__{family}__{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# --------------------------------------------------------------------------- #
# cross-family comparison: linear_content vs the two capacity heads
# --------------------------------------------------------------------------- #

def _committed_linear_curves(qset):
    """Committed linear per-(source,target,layer) quantile_loss + mase from the linear pilot CSV."""
    p = ood.OOD_DIR / f"ood_transfer_results__{qset}.csv"
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        for r in csv.DictReader(f):
            key = (r["source_dataset"], r["target_dataset"])
            out.setdefault(key, {})[int(r["layer"])] = {"quantile_loss": float(r["quantile_loss"]),
                                                         "mase": float(r["mase"])}
    return out


def _committed_linear_selected(qset, seed, device="cpu"):
    """oracle-best (test argmin) + source-selected (source-val argmin) layer for the committed
    linear pilot. Source-selected is read from the frozen linear checkpoints' selection val loss
    (read-only) so all three families use one consistent definition."""
    q = QUANTILE_SETS[qset]
    curves = _committed_linear_curves(qset)
    if curves is None:
        return None
    oracle = {k: int(min(v, key=lambda i: v[i]["quantile_loss"])) for k, v in curves.items()}
    ssl = {}
    for src in DATASET_ORDER:
        fitted = ood.load_checkpoints(src, qset, seed, q, device)
        if fitted is None:
            continue
        svl = {i: fitted[i]["selection"]["val_loss_by_wd"][fitted[i]["selection"]["chosen_wd"]]
               for i in range(NUM_LAYERS) if fitted[i]["selection"] is not None}
        if len(svl) == NUM_LAYERS:
            ssl[src] = int(min(svl, key=lambda i: svl[i]))
    return {"curves": curves, "oracle": oracle, "source_selected": ssl}


def _family_summ(family, qset, seed):
    p = _fam_dirs(family)["base"] / f"ood_capacity_summary__{family}__{qset}.json"
    if not p.exists():
        return None
    return {(c["source_dataset"], c["target_dataset"]): c for c in json.load(open(p))["cells"]}


def make_comparison(qset, seed):
    """Two cross-family figures (linear_content vs content_mlp_head vs forecast_slot_native_head)
    + a tidy comparison CSV: (1) source-selected q9 loss & MASE and oracle Δ-vs-last per cell,
    (2) native-gap-closed at the source-selected layer (guarded when the denominator is small)."""
    lin = _committed_linear_selected(qset, seed)
    fam_summ = {fam: _family_summ(fam, qset, seed) for fam in FAMILIES}
    fam_curves = {fam: _committed_family_curves(fam, qset) for fam in FAMILIES}
    if lin is None or any(fam_summ[f] is None for f in FAMILIES):
        print("  [compare] need the committed linear pilot AND both family summaries first "
              "(run each family's sources + aggregate).")
        return
    families = [LINEAR] + list(FAMILIES)
    color = {LINEAR: "#4c72b0", "content_mlp_head": "#dd8452", "forecast_slot_native_head": "#55a868"}

    # ---- tidy comparison table + gap-closed ----
    rows = []
    for src in DATASET_ORDER:
        for tgt in DATASET_ORDER:
            native = fam_summ["content_mlp_head"].get((src, tgt), {}).get("native_mase")
            lin_ss = lin["source_selected"].get(src)
            lin_mase_ss = (lin["curves"][(src, tgt)][lin_ss]["mase"] if lin_ss is not None else None)
            for fam in families:
                if fam == LINEAR:
                    ss = lin["source_selected"].get(src)
                    oracle = lin["oracle"].get((src, tgt))
                    c = lin["curves"].get((src, tgt), {})
                    ql_ss = c[ss]["quantile_loss"] if ss is not None else None
                    mase_ss = c[ss]["mase"] if ss is not None else None
                    dvl = ((c[LAST_LAYER]["quantile_loss"] - c[oracle]["quantile_loss"])
                           if (c and oracle is not None) else None)
                else:
                    s = fam_summ[fam].get((src, tgt))
                    if s is None:
                        continue
                    ss = s["source_selected_layer"]
                    ql_ss = s["source_selected_loss"]
                    mase_ss = (fam_curves[fam][(src, tgt)][ss]["mase"] if ss is not None else None)
                    dvl = s["delta_vs_last"]
                gap = None
                if (fam != LINEAR and lin_mase_ss is not None and native is not None
                        and mase_ss is not None):
                    denom = lin_mase_ss - native
                    gap = (float((lin_mase_ss - mase_ss) / denom)
                           if denom > max(1e-6, 0.02 * native) else None)
                rows.append({"source_dataset": src, "target_dataset": tgt, "is_ood": src != tgt,
                             "quantile_set": qset, "probe_family": fam,
                             "source_selected_layer": ss, "source_selected_q9_loss": ql_ss,
                             "source_selected_mase": mase_ss, "oracle_delta_vs_last": dvl,
                             "native_mase": native, "linear_source_selected_mase": lin_mase_ss,
                             "gap_closed_vs_native": gap,
                             "gap_closed_valid": (gap is not None) if fam != LINEAR else None})
    cmp = _cmp_dir()
    fields = ["source_dataset", "target_dataset", "is_ood", "quantile_set", "probe_family",
              "source_selected_layer", "source_selected_q9_loss", "source_selected_mase",
              "oracle_delta_vs_last", "native_mase", "linear_source_selected_mase",
              "gap_closed_vs_native", "gap_closed_valid"]
    with open(cmp / f"family_comparison__{qset}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        wr.writerows(rows)
    json.dump(rows, open(cmp / f"family_comparison__{qset}.json", "w"), indent=2)

    _make_family_bars(rows, families, color, qset, seed, cmp)
    _make_gap_figure(rows, qset, seed, cmp)
    print(f"  [saved] cross-family comparison under {cmp}")


def _committed_family_curves(family, qset):
    p = _fam_dirs(family)["base"] / f"ood_capacity_results__{family}__{qset}.json"
    if not p.exists():
        return {}
    out = {}
    for r in json.load(open(p)):
        out.setdefault((r["source_dataset"], r["target_dataset"]), {})[int(r["layer"])] = {
            "quantile_loss": r["quantile_loss"], "mase": r["mase"]}
    return out


def _make_family_bars(rows, families, color, qset, seed, cmp):
    """N×N grid of grouped bars: source-selected MASE per family, native line for reference."""
    fig, axes = plt.subplots(NDATA, NDATA, figsize=(5.0 * NDATA, 4.0 * NDATA), sharex=True)
    idx = {f: k for k, f in enumerate(families)}
    for ri, src in enumerate(DATASET_ORDER):
        for ci, tgt in enumerate(DATASET_ORDER):
            ax = axes[ri, ci]
            cell = [r for r in rows if r["source_dataset"] == src and r["target_dataset"] == tgt]
            native = next((r["native_mase"] for r in cell if r["native_mase"] is not None), None)
            for r in cell:
                k = idx[r["probe_family"]]
                if r["source_selected_mase"] is not None:
                    ax.bar(k, r["source_selected_mase"], color=color[r["probe_family"]], width=0.7)
            if native is not None:
                ax.axhline(native, color="k", ls="--", lw=1.2, label=f"native {native:.2f}")
            ax.set_title(f"{SHORT[src]} → {SHORT[tgt]}", fontsize=9,
                         fontweight=("bold" if src == tgt else "normal"))
            ax.set_xticks(range(len(families)))
            ax.set_xticklabels(["linear", "content\nMLP", "fslot\nnative"], fontsize=7)
            ax.grid(axis="y", alpha=0.3)
            ax.legend(fontsize=7)
    for ri, src in enumerate(DATASET_ORDER):
        axes[ri, 0].set_ylabel(f"source {SHORT[src]}\nsource-selected MASE")
    fig.suptitle(f"Probe-family comparison — source-selected-layer MASE (lower = better)  "
                 f"[{qset}, seed {seed}]\nblack dashed = native Chronos-2 MASE", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = cmp / f"family_comparison_mase__{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def _make_gap_figure(rows, qset, seed, cmp):
    """N×N heatmap-style grid: fraction of the linear→native MASE gap closed by each capacity head
    at the source-selected layer. Invalid cells (denominator ≤ ~0) are hatched + labeled."""
    caps = [f for f in FAMILIES]
    fig, axes = plt.subplots(1, len(caps), figsize=(7.5 * len(caps), 6.2))
    if len(caps) == 1:
        axes = [axes]
    for ax, fam in zip(axes, caps):
        M = np.full((NDATA, NDATA), np.nan)
        valid = np.zeros((NDATA, NDATA), bool)
        for r in rows:
            if r["probe_family"] != fam:
                continue
            si, ti = DATASET_ORDER.index(r["source_dataset"]), DATASET_ORDER.index(r["target_dataset"])
            if r["gap_closed_vs_native"] is not None:
                M[si, ti] = r["gap_closed_vs_native"]
                valid[si, ti] = True
        im = ax.imshow(M, cmap="RdYlGn", vmin=-1.0, vmax=1.0)
        for si in range(NDATA):
            for ti in range(NDATA):
                if valid[si, ti]:
                    ax.text(ti, si, f"{M[si, ti]:+.0%}", ha="center", va="center", fontsize=10)
                else:
                    ax.text(ti, si, "n/a", ha="center", va="center", fontsize=9, color="0.4")
                    ax.add_patch(plt.Rectangle((ti - 0.5, si - 0.5), 1, 1, fill=False, hatch="///",
                                               ec="0.6", lw=0))
        ax.set_xticks(range(NDATA)); ax.set_xticklabels([SHORT[d] for d in DATASET_ORDER])
        ax.set_yticks(range(NDATA)); ax.set_yticklabels([SHORT[d] for d in DATASET_ORDER])
        ax.set_xlabel("target"); ax.set_ylabel("source")
        ax.set_title(fam, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="gap to native closed")
    fig.suptitle(f"Native-gap closed at the source-selected layer  "
                 f"[(linear − new) / (linear − native), {qset}, seed {seed}]\n"
                 "green = closes the linear→native gap · n/a = linear already ≈/≤ native (denom ≤ 0)",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = cmp / f"native_gap_closed__{qset}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Higher-capacity forecasting probes (capacity controls) for the OOD transfer "
                    "pilot. Train one nonlinear head family per source, freeze, score every target.")
    ap.add_argument("--dataset-set", default=None, metavar="NAME")
    ap.add_argument("--probe-family", choices=sorted(FAMILIES), default=None,
                    help="which capacity head to fit/evaluate.")
    ap.add_argument("--source-dataset", default=None, help="the dataset the head is TRAINED on.")
    ap.add_argument("--target-datasets", nargs="+", default=None,
                    help="targets to score the frozen source head on (default: the active set's "
                         "full roster, resolved AFTER --dataset-set).")
    ap.add_argument("--quantile-set", choices=sorted(QUANTILE_SETS), default="q9")
    ap.add_argument("--device", default=None, help="torch device (default: cuda if available).")
    ap.add_argument("--figure-only", action="store_true",
                    help="skip training; aggregate one family's existing per-source results.")
    ap.add_argument("--compare", action="store_true",
                    help="build the cross-family comparison + native-gap figures (needs the "
                         "committed linear pilot and both family summaries).")
    return ap.parse_args(argv)


def main():
    args = _parse_args()
    if args.dataset_set:
        config.set_dataset_set(args.dataset_set)
    _derive_datasets()          # re-derive roster + ood dirs for the active set (4x4 under rolling)
    qset = args.quantile_set
    quantiles = QUANTILE_SETS[qset]
    if median_index(quantiles) is None:
        raise SystemExit(f"quantile set {qset} has no 0.5 level — MASE/median undefined; use q1/q9/q21")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu" and args.device != "cpu":
        raise SystemExit(
            "CUDA not visible — refusing to fit nonlinear heads on CPU (~hours/fit). Check the "
            "job env (module load ... cuda/13.2; source .venv/bin/activate), or pass --device cpu "
            "to override intentionally.")
    seed = SEED
    print(f"[config] dataset_set={config.DATASET_SET}  quantile_set={qset}  Q={len(quantiles)}  "
          f"device={device}  hidden_dim={HIDDEN_DIM}  dropout={DROPOUT}")

    if args.compare:
        make_comparison(qset, seed)
        return

    if not args.probe_family:
        raise SystemExit("--probe-family is required (content_mlp_head | forecast_slot_native_head), "
                         "or pass --compare for the cross-family figures.")
    family = args.probe_family

    if args.figure_only:
        aggregate(family, qset, seed)
        return

    known = set(id_data.ID_DATASETS)
    targets = args.target_datasets or list(DATASET_ORDER)   # resolved AFTER the set override
    if not args.source_dataset:
        raise SystemExit("--source-dataset is required (one source per job); or --figure-only.")
    if args.source_dataset not in known:
        raise SystemExit(f"unknown --source-dataset {args.source_dataset!r}; known: {sorted(known)}")
    bad = [t for t in targets if t not in known]
    if bad:
        raise SystemExit(f"unknown --target-datasets {bad}; known: {sorted(known)}")

    run_source(args.source_dataset, targets, family, qset, quantiles, seed, device)
    aggregate(family, qset, seed)
    print(f"\n{'=' * 70}\nDONE — {family} source {args.source_dataset}. Run the other sources, then "
          f"`--figure-only` per family.\n{'=' * 70}")


if __name__ == "__main__":
    main()
