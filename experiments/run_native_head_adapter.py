"""ext_v5 — NATIVE-HEAD ADAPTER driver: is each layer's representation usable by Chronos-2's own
frozen Quantile Head, and can a shared linear 768->768 map make it usable?

Three conditions, ALL scored through the pretrained ``output_patch_embedding`` on identical windows
(see ``probing/native_head_adapter.py`` for the exact paths and the precise interpretation):

    1. native baseline   — final_slots -> native head                      (horizontal reference)
    2. zero-shot head     — h_l -> final RMSNorm -> native head             (no new params)
    3. linear adapter     — h_l -> A_l -> final RMSNorm -> native head      (only A_l trains, Emb..L12)

L12+RMS (the post-final-RMSNorm slots) is the native-head input, so zero-shot@L12+RMS == native and
the adapter curve TERMINATES at native there (we do NOT train an adapter at L12+RMS — that would be
dataset-specific adaptation of the native model, a different question).

Everything is isolated under ``results/ext_v5_native_head_adapter/`` and every file name carries
``native_head_adapter`` so it can never be confused with the ext_v4 shared-linear fslot results.

Modes (USER submits SLURM; adapter fits + the model load are GPU/compute-node work, NOT login node):
    python -m experiments.run_native_head_adapter --sanity                       # 1 dataset, 5 layers
    python -m experiments.run_native_head_adapter --adapt                         # all 7 datasets
    python -m experiments.run_native_head_adapter --adapt --datasets m4_hourly    # a subset
    python -m experiments.run_native_head_adapter --figures                       # CPU/login aggregate
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probing import config
from probing.config import NUM_LAYERS, OUTPUT_PATCH_SIZE, REPO_ROOT, SEED
from probing.probes import (CHRONOS2_QUANTILES, QUANTILE_SETS, WD_GRID_V2,
                            fit_shared_forecast_probe_explicit_val, median_index,
                            predict_shared_forecast_probe, validate_quantiles)
from probing.native_head_adapter import (NUM_NATIVE_QUANTILES, LinearAdapter, fit_adapter_explicit_val,
                                        native_head_modules, slots_to_normalized_quantiles)
from probing.id_data import build_ood_rolling_windows, build_windows
from probing.tunnel import PT_ID_TAGS, PT_OOD_TAGS
from probing.stats import cluster_bootstrap_apply, cluster_bootstrap_counts, ci_bounds
from experiments.run_id_forecasting import M_SEASON, _ctx_stats, _mase_denominator
from experiments.run_ptood_probing_ftok import C, H, K, SHORT, _fslot_feats, load_ptid_ckpt

OUT_ROOT = REPO_ROOT / "results" / "ext_v5_native_head_adapter"
# frozen adapter-TRANSFER outputs live in a disjoint subtree so the per-dataset --adapt results
# (bootstrap_inputs/, tables/, plots/, adapters/) are never touched.
TRANSFER_ROOT = OUT_ROOT / "transfer"
TRANSFER_BOOT_DIR = TRANSFER_ROOT / "bootstrap_inputs"
TRANSFER_TAB_DIR = TRANSFER_ROOT / "tables"
TRANSFER_PLOT_DIR = TRANSFER_ROOT / "plots"
MODEL_ID = "amazon/chronos-2"
EPOCHS = 300
LR = 1e-2
WD_GRID = WD_GRID_V2                       # existing 8-value grid (no new search)
BOOT_B = 5000
P = OUTPUT_PATCH_SIZE                       # 16
# x-axis: Emb, L1..L12 (pre-RMS block slots), then L12+RMS (post-final-RMSNorm = native-head input)
LAYER_LABELS = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)] + ["L12+RMS"]   # len NUM_LAYERS+1 = 14
REF_IDX = NUM_LAYERS                        # 13 = L12+RMS = native endpoint (not a trained adapter)
ADAPTER_LAYERS = list(range(NUM_LAYERS))   # 0..12 get trained adapters
ALL_LAYERS = list(range(NUM_LAYERS + 1))   # 0..13, for zero-shot + native
CONDITIONS = ("native", "zero_shot", "linear_adapter")
ALL_TAGS = list(PT_ID_TAGS) + list(PT_OOD_TAGS)
DATASET_KIND = {**{t: "PT-ID" for t in PT_ID_TAGS}, **{t: "PT-OOD" for t in PT_OOD_TAGS}}
RECON_REL_TOL = 5e-3                        # native reconstruction vs pipeline (float32 through the head)
SANITY_LAYERS = [0, 3, 8, 12, 13]
SANITY_EPOCHS = 60
SANITY_WD = (1e-4, 1e-2, 1.0)
Q1_SEEDS = (0, 1, 2)                        # committed ext_v4 q1 fslot-probe seeds (PT-ID) / fresh-fit seeds (PT-OOD)


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #
def _git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT)).decode().strip()
    except Exception:
        return "unknown"


def _dirs():
    for d in ("native_baseline", "zero_shot_head", "linear_adapter", "plots", "tables",
              "configs", "adapters", "bootstrap_inputs"):
        (OUT_ROOT / d).mkdir(parents=True, exist_ok=True)


def _transfer_dirs():
    for d in (TRANSFER_ROOT, TRANSFER_BOOT_DIR, TRANSFER_TAB_DIR, TRANSFER_PLOT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _windows(tag):
    if tag in PT_OOD_TAGS:
        w = build_ood_rolling_windows(tag, C=C, H=H, seed=SEED)
        return w, ("train_rolling", "val_rolling", "test_rolling")
    w = build_windows(tag)
    return w, ("train", "val", "test")


# per-window raw-scale metrics (identical implementation for all three conditions)
def _mase_pw(y_raw, yhat_raw, denom):
    return (np.abs(y_raw - yhat_raw) / denom).mean(axis=1)


def _mae_pw(y_raw, yhat_median_raw):
    return np.abs(y_raw - yhat_median_raw).mean(axis=1)


def _wql_pw_parts(y_raw, quant_raw, quantiles):
    """Per-window WQL numerator (2*sum pinball over Q,H) and denominator (sum_H |y|), kept separate so
    the bootstrap forms WQL = sum(num)/sum(den) inside each replicate. quant_raw: (n, Q, H)."""
    y = y_raw[:, None, :]
    tau = np.asarray(quantiles, np.float64)[None, :, None]
    pinball = np.where(y >= quant_raw, tau * (y - quant_raw), (1 - tau) * (quant_raw - y))
    return 2.0 * pinball.sum(axis=(1, 2)), np.abs(y_raw).sum(axis=1)


def _series_group(sid):
    uniq, inv = np.unique(np.asarray(sid, np.int64), return_inverse=True)
    return uniq.size, inv


def _per_series(vec, inv, S):
    out = np.zeros(S, np.float64)
    np.add.at(out, inv, np.asarray(vec, np.float64))
    return out


def _boot_mean(M, vec, inv, S):
    ssum = _per_series(vec, inv, S)[:, None]
    scount = _per_series(np.ones_like(vec, np.float64), inv, S)
    return cluster_bootstrap_apply(M, ssum, scount)[:, 0]


def _boot_ratio(M, num, den, inv, S):
    nsum = _per_series(num, inv, S)[:, None]
    dsum = _per_series(den, inv, S)
    return cluster_bootstrap_apply(M, nsum, dsum)[:, 0]


def _raw_quantiles(slots_L, adapter, apply_rms, final_rms, head, quantiles, mu, s, device):
    """(n, Q, H) ORIGINAL-scale quantiles: slots -> [adapter] -> [final RMSNorm] -> native head, then
    invert the model's normalization with mu + s*sinh (== instance_norm.inverse; sinh is monotone so
    quantile order is preserved). Never trains."""
    X = torch.as_tensor(np.asarray(slots_L, np.float32), device=device)
    with torch.no_grad():
        z = slots_to_normalized_quantiles(X, adapter, apply_rms, final_rms, head, quantiles, horizon=H)
    z = z.cpu().numpy().astype(np.float64)                              # (n, Q, H) arcsinh space
    return mu[:, None, None] + s[:, None, None] * np.sinh(z)


def _native_pipeline_median(tag, X_test, split_name, pipeline):
    """The REAL pipeline median forecast (predict_quantiles @0.5) on THESE exact windows — the
    independent reference for the reconstruction gate. Cached under an ext_v5-only key
    (``__<split>__nha_native_median_H<H>``) so it never collides with the legacy 650-window
    ``__ood__test`` cache or the extended_v2 median caches; the ctx-tail guard fails loud on drift.
    Single K=4 pass (max_output_patches=64 >> K, so predict_quantiles does NOT unroll for H=64)."""
    from probing.config import CACHE_DIR
    from probing.extraction import _idf_prefix
    cache = CACHE_DIR / f"{_idf_prefix(tag)}__{split_name}__nha_native_median_H{H}.npz"
    X = np.asarray(X_test, np.float32)
    if cache.exists():
        d = np.load(cache)
        if d["ctx_tail"].shape[0] == len(X) and np.allclose(d["ctx_tail"], X[:, -8:]):
            print(f"  [cache HIT]  {cache.name}")
            return d["median"].astype(np.float64)
        raise RuntimeError(f"stale {cache.name}: contexts changed — delete it")
    print(f"  [native] predict_quantiles median on {len(X)} windows (H={H})")
    qs, _mean = pipeline.predict_quantiles(list(X), prediction_length=H, quantile_levels=[0.5])
    med = np.stack([q.reshape(H).cpu().numpy() for q in qs]).astype(np.float64)
    np.savez(cache, median=med.astype(np.float32), ctx_tail=X[:, -8:])
    return med


def _metrics_from_qr(qr, y_raw, denom, quantiles, qmid):
    med = qr[:, qmid, :]
    num, den = _wql_pw_parts(y_raw, qr, quantiles)
    return {"mase": _mase_pw(y_raw, med, denom), "mae": _mae_pw(y_raw, med),
            "wql_num": num, "wql_den": den}


# --------------------------------------------------------------------------- #
# frozen-parameter gate: only the adapter may have requires_grad=True
# --------------------------------------------------------------------------- #
def _assert_only_adapter_trains(pipeline, adapter):
    n_model_train = sum(int(p.requires_grad) for p in pipeline.model.parameters())
    assert n_model_train == 0, (
        f"{n_model_train} backbone/head params have requires_grad=True — the whole Chronos-2 model "
        "must be frozen; only the adapter trains")
    assert all(p.requires_grad for p in adapter.parameters()), "adapter params must be trainable"
    print(f"  [frozen-check] model trainable params = 0 ; adapter trainable params = "
          f"{sum(p.numel() for p in adapter.parameters())}")


# --------------------------------------------------------------------------- #
# one dataset: native + zero-shot + adapter on identical test windows
# --------------------------------------------------------------------------- #
def process_dataset(tag, quantiles, device, pipeline, sanity=False):
    kind = DATASET_KIND[tag]
    print(f"\n[{'SANITY' if sanity else 'adapt'}] {SHORT.get(tag, tag)} ({kind})")
    head, final_rms = native_head_modules(pipeline)
    w, splits = _windows(tag)
    m = w["meta"]
    if m["n_test"] == 0 or m["n_train"] == 0:
        raise RuntimeError(f"{tag}: empty split (train {m['n_train']} / test {m['n_test']})")
    f_tr = _fslot_feats(tag, splits[0], w["X_train"], w["y_train"])     # 14-pt slot dicts (warm cache)
    f_va = _fslot_feats(tag, splits[1], w["X_val"], w["y_val"])
    f_te = _fslot_feats(tag, splits[2], w["X_test"], w["y_test"])

    Xte = np.asarray(w["X_test"], np.float64)
    mu, s = _ctx_stats(Xte, m["sigma_eps"])
    y_raw = mu[:, None] + s[:, None] * np.sinh(np.asarray(w["Y_test_traj"], np.float64))
    denom = np.maximum(_mase_denominator(Xte), 1e-8)[:, None]
    S, inv = _series_group(w["series_test"])
    qmid = median_index(quantiles)
    print(f"  windows: train {m['n_train']} / val {m['n_val'] if 'n_val' in m else '?'} / "
          f"test {m['n_test']} ({S} series)")

    # ---- native baseline == zero-shot @ L12+RMS (post-RMS slots, no adapter, no 2nd norm) ----
    native_qr = _raw_quantiles(f_te[REF_IDX], None, False, final_rms, head, quantiles, mu, s, device)
    native_pw = _metrics_from_qr(native_qr, y_raw, denom, quantiles, qmid)

    # GATE 1: native reconstruction must match the real pipeline forecast (predict_quantiles median)
    pipe_med = _native_pipeline_median(tag, w["X_test"], splits[2], pipeline)
    recon_max = float(np.max(np.abs(native_qr[:, qmid, :] - pipe_med)))
    recon_rel = recon_max / (float(np.abs(pipe_med).mean()) + 1e-12)
    ok = recon_rel < RECON_REL_TOL
    print(f"  [gate:native-reconstruction] max|recon-pipeline median|={recon_max:.3e}  "
          f"rel={recon_rel:.3e}  {'OK' if ok else 'FAIL'} (tol {RECON_REL_TOL:g})")
    if not ok:
        raise RuntimeError(
            f"{tag}: native reconstruction (output_patch_embedding on the extracted L12+RMS slots) "
            f"disagrees with predict_quantiles by rel {recon_rel:.3e} > {RECON_REL_TOL:g} — STOP and "
            "diagnose (extraction/normalization mismatch) before trusting any zero-shot/adapter number")

    pw = {("native", REF_IDX): native_pw}

    # ---- zero-shot per layer (0..13), no training ----
    zs_layers = SANITY_LAYERS if sanity else ALL_LAYERS
    for L in zs_layers:
        apply_rms = (L != REF_IDX)                       # L12+RMS is already post-RMS
        qr = _raw_quantiles(f_te[L], None, apply_rms, final_rms, head, quantiles, mu, s, device)
        pw[("zero_shot", L)] = _metrics_from_qr(qr, y_raw, denom, quantiles, qmid)
    # GATE 2: zero-shot @ L12+RMS must equal the native baseline exactly (same tensor path -> no 2nd norm)
    if REF_IDX in zs_layers:
        d = float(np.max(np.abs(pw[("zero_shot", REF_IDX)]["mase"] - native_pw["mase"])))
        print(f"  [gate:zeroshot@L12+RMS==native] max|Δ MASE|={d:.3e}  {'OK' if d == 0.0 else 'CHECK'}")
        assert d == 0.0, "zero-shot@L12+RMS != native — an accidental extra RMSNorm crept in"

    # ---- adapters: train on Emb..L12 only (REF is the native endpoint) ----
    adapt_layers = [L for L in (SANITY_LAYERS if sanity else ADAPTER_LAYERS) if L != REF_IDX]
    _assert_only_adapter_trains(pipeline, LinearAdapter(768).to(device))
    fitted = fit_adapter_explicit_val(
        f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"], final_rms, head, layers=adapt_layers,
        quantiles=quantiles, epochs=(SANITY_EPOCHS if sanity else EPOCHS),
        wd_grid=(SANITY_WD if sanity else WD_GRID), device=device)
    records = []
    for L in adapt_layers:
        qr = _raw_quantiles(f_te[L], fitted[L]["adapter"], True, final_rms, head, quantiles, mu, s, device)
        pw[("linear_adapter", L)] = _metrics_from_qr(qr, y_raw, denom, quantiles, qmid)
        zs = pw[("zero_shot", L)]["mase"].mean() if ("zero_shot", L) in pw else None
        ad = pw[("linear_adapter", L)]["mase"].mean()
        records.append({
            "dataset": tag, "kind": kind, "condition": "linear_adapter", "layer": L,
            "layer_label": LAYER_LABELS[L], "test_mase": round(float(ad), 6),
            "zero_shot_test_mase": (None if zs is None else round(float(zs), 6)),
            "native_test_mase": round(float(native_pw["mase"].mean()), 6),
            "selected_wd": fitted[L]["wd"], "val_qloss": fitted[L]["selection"]["val_loss_by_wd"][fitted[L]["wd"]]
            if fitted[L]["wd"] in fitted[L]["selection"]["val_loss_by_wd"] else None,
            "adapter_param_count": fitted[L]["param_count"], "epochs": (SANITY_EPOCHS if sanity else EPOCHS),
        })
    # the adapter curve terminates at native at L12+RMS (not a trained adapter)
    pw[("linear_adapter", REF_IDX)] = native_pw

    if sanity:
        print("\n  [sanity summary] MASE by layer (mean over windows):")
        print(f"    native (L12+RMS) = {native_pw['mase'].mean():.4f}")
        for L in [x for x in SANITY_LAYERS]:
            zline = pw[("zero_shot", L)]["mase"].mean()
            aline = pw[("linear_adapter", L)]["mase"].mean() if ("linear_adapter", L) in pw else float("nan")
            print(f"    {LAYER_LABELS[L]:>8}: zero-shot {zline:7.4f}   adapter {aline:7.4f}")
        return None

    # ---- persist per-window metrics (feeds the login-side bootstrap) + records + config + adapters ----
    _save_bootstrap_inputs(tag, pw, w["series_test"])
    _save_records(tag, kind, pw, records, native_pw, quantiles)
    _save_adapters(tag, fitted, quantiles)
    _save_config(tag, kind, m, quantiles)
    print(f"  [done] {SHORT.get(tag, tag)}: native/zero-shot/adapter written")
    return pw


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def _save_bootstrap_inputs(tag, pw, series_test):
    save = {"series_test": np.asarray(series_test)}
    for (cond, L), d in pw.items():
        for k in ("mase", "mae", "wql_num", "wql_den"):
            save[f"{cond}__L{L:02d}__{k}"] = np.asarray(d[k], np.float64)
    out = OUT_ROOT / "bootstrap_inputs" / f"native_head_adapter__{tag}.npz"
    np.savez(out, **save)
    print(f"  [saved] {out.name}")


def _point_regret(pw, cond, L, native_mase_mean, native_wql):
    mase = float(pw[(cond, L)]["mase"].mean())
    wql = float(pw[(cond, L)]["wql_num"].sum() / max(pw[(cond, L)]["wql_den"].sum(), 1e-12))
    return mase, wql, mase / native_mase_mean - 1.0, wql / native_wql - 1.0


def _save_records(tag, kind, pw, adapter_records, native_pw, quantiles):
    native_mase = float(native_pw["mase"].mean())
    native_wql = float(native_pw["wql_num"].sum() / max(native_pw["wql_den"].sum(), 1e-12))
    rows = []
    for (cond, L) in sorted(pw, key=lambda t: (CONDITIONS.index(t[0]), t[1])):
        mase, wql, rr_mase, rr_wql = _point_regret(pw, cond, L, native_mase, native_wql)
        rows.append({
            "dataset": tag, "kind": kind, "condition": cond, "layer": L, "layer_label": LAYER_LABELS[L],
            "test_mase": round(mase, 6), "test_median_mae": round(float(pw[(cond, L)]["mae"].mean()), 6),
            "test_wql": round(wql, 6),
            "relative_regret_mase": round(rr_mase, 6), "relative_regret_wql": round(rr_wql, 6),
            "native_test_mase": round(native_mase, 6), "native_test_wql": round(native_wql, 6),
        })
    stem = OUT_ROOT / "tables" / f"native_head_adapter__records__{tag}"
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    json.dump({"per_layer_records": rows, "adapter_fit_records": adapter_records},
              open(f"{stem}.json", "w"), indent=2)
    print(f"  [saved] {stem.name}.csv/.json")


def _save_adapters(tag, fitted, quantiles):
    for L, f in fitted.items():
        torch.save({"state_dict": f["adapter"].state_dict(), "wd": f["wd"], "selection": f["selection"],
                    "apply_rms": f["apply_rms"], "family": f["family"], "layer": L,
                    "param_count": f["param_count"]},
                   OUT_ROOT / "adapters" / f"native_head_adapter__{tag}__L{L:02d}.pt")
    print(f"  [saved] {len(fitted)} adapters -> adapters/native_head_adapter__{tag}__L*.pt")


def _save_config(tag, kind, meta, quantiles):
    cfg = {"experiment": "ext_v5_native_head_adapter", "dataset": tag, "kind": kind,
           "model_id": MODEL_ID, "git_commit": _git_hash(),
           "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
           "C": C, "H": H, "output_patch_size": P, "K": K, "num_native_quantiles": NUM_NATIVE_QUANTILES,
           "quantiles": [float(q) for q in quantiles], "wd_grid": [float(w) for w in WD_GRID],
           "epochs": EPOCHS, "lr": LR, "early_stopping": False, "adapter": "shared Linear(768,768), identity init",
           "adapter_param_count": 768 * 768 + 768, "seeds": "single deterministic fit (identity init)",
           "n_train": int(meta["n_train"]), "n_test": int(meta["n_test"]),
           "n_val": int(meta.get("n_val", -1)), "seasonal_m": M_SEASON,
           "dataset_set": config.DATASET_SET, "bootstrap_B": BOOT_B,
           "note": ("A_l is supervised by the forecast target Y through the FROZEN native head; a low "
                    "adapter loss shows a linear map of layer l is SUFFICIENT to make it usable by the "
                    "native head, NOT that it recovers the l->L12 coordinate transform.")}
    json.dump(cfg, open(OUT_ROOT / "configs" / f"native_head_adapter__{tag}__config.json", "w"), indent=2)


# --------------------------------------------------------------------------- #
# q1 LINEAR-PROBE baseline: score the ext_v4 shared-forecast fslot probe (median-only) on ext_v5's
# OWN test windows so the adapt panels can compare "adapter into the frozen native head" vs "fresh
# linear readout of the same slots". MASE is a MEDIAN-only metric, so the median-optimised q1 probe is
# the metric-matched (and conservative) linear baseline; WQL would instead call for q21 (not built).
# --------------------------------------------------------------------------- #
def _q1_fitted(tag, w, splits, seed, q1, device):
    """The q1 shared-forecast LINEAR probe for one dataset/seed, as a fitted dict
    predict_shared_forecast_probe consumes. PT-ID: reuse the committed frozen ext_v4 probe
    (load_ptid_ckpt — no re-fit, reproduces the published q1 numbers). PT-OOD: no saved probe-ID probe
    exists, so fit a fresh q1 fslot probe on the target's own train (wd on target-val) with the SAME
    estimator/grid ext_v4 used."""
    if tag in PT_ID_TAGS:
        return load_ptid_ckpt(tag, "q1", seed, device=device)
    f_tr = _fslot_feats(tag, splits[0], w["X_train"], w["y_train"])
    f_va = _fslot_feats(tag, splits[1], w["X_val"], w["y_val"])
    return fit_shared_forecast_probe_explicit_val(
        f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"], quantiles=q1,
        epochs=EPOCHS, wd_grid=WD_GRID, device=device, init_seed=seed)


def _q1_baseline_pw(tag, device):
    """Per-window test MASE of the q1 linear probe on ext_v5's OWN windows (so series_test aligns with
    the adapter's for a PAIRED bootstrap). Per-window MASE is averaged across the 3 probe seeds (the
    ext_v4 per-window seed protocol). Returns ({L: mase_pw (n,)}, series_test)."""
    q1 = validate_quantiles(QUANTILE_SETS["q1"])
    w, splits = _windows(tag)
    m = w["meta"]
    if m["n_test"] == 0 or m["n_train"] == 0:
        raise RuntimeError(f"{tag}: empty split (train {m['n_train']} / test {m['n_test']})")
    f_te = _fslot_feats(tag, splits[2], w["X_test"], w["y_test"])
    Xte = np.asarray(w["X_test"], np.float64)
    mu, s = _ctx_stats(Xte, m["sigma_eps"])
    y_raw = mu[:, None] + s[:, None] * np.sinh(np.asarray(w["Y_test_traj"], np.float64))
    denom = np.maximum(_mase_denominator(Xte), 1e-8)[:, None]
    seed_mase = {}
    for seed in Q1_SEEDS:
        fitted = _q1_fitted(tag, w, splits, seed, q1, device)
        _out, diag = predict_shared_forecast_probe(fitted, f_te, w["Y_test_traj"], quantiles=q1,
                                                   device=device, collect_test_median=True)
        for L, zmed in diag["test_median"].items():
            yhat = mu[:, None] + s[:, None] * np.sinh(np.asarray(zmed, np.float64))
            seed_mase.setdefault(L, []).append(_mase_pw(y_raw, yhat, denom))
    pw = {L: np.mean(np.stack(v), axis=0) for L, v in seed_mase.items()}
    return pw, w["series_test"]


def run_linear_baseline(tags, device):
    """Compute + save the q1 linear-probe per-window MASE sidecar (one npz per dataset). Compute-node
    work (fresh PT-OOD probe fits + OOD rolling-window builds); no Chronos-2 model load (fslot caches
    are warm). The login `--figures` pass overlays it on the adapt panels."""
    for tag in tags:
        which = "load committed probe" if tag in PT_ID_TAGS else "fresh probe-ID fit"
        print(f"\n[linear-baseline q1] {SHORT.get(tag, tag)} ({DATASET_KIND[tag]})  [{which}]")
        pw, series = _q1_baseline_pw(tag, device)
        save = {"series_test": np.asarray(series)}
        for L, v in pw.items():
            save[f"linear_q1__L{L:02d}__mase"] = np.asarray(v, np.float64)
        out = OUT_ROOT / "bootstrap_inputs" / f"native_head_adapter__linear_q1__{tag}.npz"
        np.savez(out, **save)
        print(f"  [saved] {out.name}  ({len(pw)} layers, {len(series)} windows)")


# --------------------------------------------------------------------------- #
# frozen adapter TRANSFER: apply ONE dataset's saved adapters to the other datasets
# (predict-only — no adapter is trained; the model loads only for the frozen head + RMSNorm)
# --------------------------------------------------------------------------- #
def _load_adapters(src, device):
    """Rebuild the source dataset's Emb..L12 adapters from disk (the objects `--adapt` saved with
    `_save_adapters`). Each is used FROZEN — its wd was already selected on the SOURCE's validation
    split; nothing is re-fit or re-selected. Fail loud if any layer's adapter is missing."""
    adapters = {}
    for L in ADAPTER_LAYERS:
        p = OUT_ROOT / "adapters" / f"native_head_adapter__{src}__L{L:02d}.pt"
        if not p.exists():
            raise RuntimeError(f"missing source adapter {p.name} — run `--adapt --datasets {src}` first")
        ck = torch.load(p, map_location=device)
        a = LinearAdapter(768).to(device)
        a.load_state_dict(ck["state_dict"])
        a.eval()
        adapters[L] = a
    print(f"  [loaded] {len(adapters)} {SHORT.get(src, src)} adapters (frozen, Emb..L12)")
    return adapters


def _eval_context(tag, quantiles, device, head, final_rms, pipeline):
    """Everything needed to score a target's test windows through the frozen head, plus the native
    baseline and GATE 1 (native reconstruction == pipeline forecast). No adapter fit. Duplicates the
    front half of `process_dataset` deliberately so the tested `--adapt` path stays byte-identical."""
    w, splits = _windows(tag)
    m = w["meta"]
    if m["n_test"] == 0:
        raise RuntimeError(f"{tag}: empty test split (n_test 0)")
    f_te = _fslot_feats(tag, splits[2], w["X_test"], w["y_test"])       # 14-pt slot dict (warm cache)
    Xte = np.asarray(w["X_test"], np.float64)
    mu, s = _ctx_stats(Xte, m["sigma_eps"])
    y_raw = mu[:, None] + s[:, None] * np.sinh(np.asarray(w["Y_test_traj"], np.float64))
    denom = np.maximum(_mase_denominator(Xte), 1e-8)[:, None]
    S, inv = _series_group(w["series_test"])
    qmid = median_index(quantiles)
    native_qr = _raw_quantiles(f_te[REF_IDX], None, False, final_rms, head, quantiles, mu, s, device)
    native_pw = _metrics_from_qr(native_qr, y_raw, denom, quantiles, qmid)
    pipe_med = _native_pipeline_median(tag, w["X_test"], splits[2], pipeline)   # cache HIT from --adapt
    recon_rel = float(np.max(np.abs(native_qr[:, qmid, :] - pipe_med))) / (float(np.abs(pipe_med).mean()) + 1e-12)
    if recon_rel >= RECON_REL_TOL:
        raise RuntimeError(f"{tag}: native reconstruction rel {recon_rel:.3e} > {RECON_REL_TOL:g} — STOP")
    print(f"  windows: test {m['n_test']} ({S} series)  [gate:native-reconstruction rel={recon_rel:.2e} OK]")
    return {"f_te": f_te, "mu": mu, "s": s, "y_raw": y_raw, "denom": denom, "qmid": qmid,
            "native_pw": native_pw, "series_test": w["series_test"]}


def transfer_from_source(src_tag, target_tags, quantiles, device, pipeline):
    """Apply the frozen source adapters to every target's test windows (source adapter -> final
    RMSNorm -> frozen native head), scoring the target native baseline as the reference. The curve
    terminates at native at L12+RMS (no adapter there, by construction)."""
    _transfer_dirs()
    head, final_rms = native_head_modules(pipeline)
    adapters = _load_adapters(src_tag, device)
    for tgt in target_tags:
        print(f"\n[transfer] {SHORT.get(src_tag, src_tag)} adapter -> {SHORT.get(tgt, tgt)} "
              f"({DATASET_KIND[tgt]})")
        ctx = _eval_context(tgt, quantiles, device, head, final_rms, pipeline)
        pw = {("native", REF_IDX): ctx["native_pw"]}
        for L in ADAPTER_LAYERS:                                # 0..12: source-trained adapters
            qr = _raw_quantiles(ctx["f_te"][L], adapters[L], True, final_rms, head, quantiles,
                                ctx["mu"], ctx["s"], device)
            pw[("transfer", L)] = _metrics_from_qr(qr, ctx["y_raw"], ctx["denom"], quantiles, ctx["qmid"])
        pw[("transfer", REF_IDX)] = ctx["native_pw"]           # curve terminates at native at L12+RMS
        _save_transfer_inputs(src_tag, tgt, pw, ctx["series_test"])
        _save_transfer_records(src_tag, tgt, pw, quantiles)
        print(f"  [done] {SHORT.get(tgt, tgt)}: native + {SHORT.get(src_tag, src_tag)}-transfer written")


def _save_transfer_inputs(src, tgt, pw, series_test):
    save = {"series_test": np.asarray(series_test)}
    for (cond, L), d in pw.items():
        for k in ("mase", "mae", "wql_num", "wql_den"):
            save[f"{cond}__L{L:02d}__{k}"] = np.asarray(d[k], np.float64)
    out = TRANSFER_BOOT_DIR / f"native_head_adapter__from_{src}__{tgt}.npz"
    np.savez(out, **save)
    print(f"  [saved] {out.name}")


def _save_transfer_records(src, tgt, pw, quantiles):
    native_pw = pw[("native", REF_IDX)]
    native_mase = float(native_pw["mase"].mean())
    native_wql = float(native_pw["wql_num"].sum() / max(native_pw["wql_den"].sum(), 1e-12))
    order = {"native": 0, "transfer": 1}
    rows = []
    for (cond, L) in sorted(pw, key=lambda t: (order[t[0]], t[1])):
        mase, wql, rr_mase, rr_wql = _point_regret(pw, cond, L, native_mase, native_wql)
        rows.append({
            "source_dataset": src, "target_dataset": tgt, "kind": DATASET_KIND[tgt],
            "condition": cond, "layer": L, "layer_label": LAYER_LABELS[L],
            "test_mase": round(mase, 6), "test_median_mae": round(float(pw[(cond, L)]["mae"].mean()), 6),
            "test_wql": round(wql, 6), "relative_regret_mase": round(rr_mase, 6),
            "relative_regret_wql": round(rr_wql, 6),
            "native_test_mase": round(native_mase, 6), "native_test_wql": round(native_wql, 6)})
    stem = TRANSFER_TAB_DIR / f"native_head_adapter__transfer__from_{src}__{tgt}"
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    json.dump({"per_layer_records": rows}, open(f"{stem}.json", "w"), indent=2)
    print(f"  [saved] {stem.name}.csv/.json")


# --------------------------------------------------------------------------- #
# figures (CPU/login): per-dataset panels + 2x4 overview, MASE + relative regret
# --------------------------------------------------------------------------- #
def _maybe_add_linear_q1(tag, pw, series, metric):
    """If the q1 linear-probe sidecar exists (written by --linear-baseline), merge its per-window MASE
    into pw as ('linear_q1', L) so _boot_curves resamples it with the SAME series matrix (paired with
    the adapter). MASE only — the q1 probe is median-only; WQL would need q21. Fail loud on window drift."""
    if metric != "mase":
        return pw
    p = OUT_ROOT / "bootstrap_inputs" / f"native_head_adapter__linear_q1__{tag}.npz"
    if not p.exists():
        return pw
    d = np.load(p, allow_pickle=True)
    if not np.array_equal(np.asarray(d["series_test"]), np.asarray(series)):
        raise RuntimeError(f"{tag}: q1 baseline series_test != adapter series_test — re-run "
                           "--linear-baseline and --adapt on the same split")
    merged = dict(pw)
    for key in d.files:
        if key == "series_test":
            continue
        _cond, lab, met = key.split("__")
        merged[("linear_q1", int(lab[1:]))] = {met: np.asarray(d[key])}
    return merged


def _load_pw(tag):
    p = OUT_ROOT / "bootstrap_inputs" / f"native_head_adapter__{tag}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    series = d["series_test"]
    pw = {}
    for key in d.files:
        if key == "series_test":
            continue
        cond, lab, metric = key.split("__")
        L = int(lab[1:])
        pw.setdefault((cond, L), {})[metric] = d[key]
    return pw, series


def _boot_curves(pw, series, metric="mase"):
    """Bootstrapped mean + 95% CI per (condition, layer), one shared resample -> paired conditions."""
    S, inv = _series_group(series)
    M = cluster_bootstrap_counts(S, BOOT_B, SEED)
    curves = {}
    native_boot = None
    for (cond, L), d in pw.items():
        if metric == "wql":
            b = _boot_ratio(M, d["wql_num"], d["wql_den"], inv, S)
        else:
            b = _boot_mean(M, d[metric], inv, S)
        lo, hi = ci_bounds(b)
        curves[(cond, L)] = (float(b.mean()), float(lo), float(hi))
        if cond == "native":
            native_boot = b
    return curves, native_boot, (S, inv, M)


def _panel(ax, tag, pw, series, metric="mase"):
    pw = _maybe_add_linear_q1(tag, pw, series, metric)     # q1 linear-probe overlay (MASE only, if present)
    curves, _nb, _ = _boot_curves(pw, series, metric)
    x = np.arange(NUM_LAYERS)               # Emb..L12 only — drop the L12+RMS (native-endpoint) tick
    # native horizontal band (its point + CI, drawn across the panel)
    nat = curves.get(("native", REF_IDX))
    if nat is not None:
        ax.axhspan(nat[1], nat[2], color="0.6", alpha=0.18)
        ax.axhline(nat[0], color="0.35", ls="--", lw=1.3, label="native (frozen head)")
    for cond, color, marker in (("zero_shot", "tab:orange", "o"), ("linear_adapter", "tab:blue", "s")):
        xs = [i for i in x if (cond, i) in curves]
        if cond == "linear_adapter":
            xs = [i for i in xs if i != REF_IDX]     # adapter@L12+RMS is native by construction, not a fit
        mean = np.array([curves[(cond, i)][0] for i in xs])
        lo = np.array([curves[(cond, i)][1] for i in xs])
        hi = np.array([curves[(cond, i)][2] for i in xs])
        label = "zero-shot native head" if cond == "zero_shot" else "linear adapter + native head"
        ax.plot(xs, mean, marker=marker, ms=3.5, color=color, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
    # q1 linear-probe baseline (a fresh readout with a GENUINE fit at L12+RMS -> drawn through the last tick)
    if any(c == "linear_q1" for (c, _l) in curves):
        xs = [i for i in x if ("linear_q1", i) in curves]
        mean = np.array([curves[("linear_q1", i)][0] for i in xs])
        lo = np.array([curves[("linear_q1", i)][1] for i in xs])
        hi = np.array([curves[("linear_q1", i)][2] for i in xs])
        ax.plot(xs, mean, marker="^", ms=3.5, color="tab:green", label="q1 linear probe (fslot readout)")
        ax.fill_between(xs, lo, hi, color="tab:green", alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS[:NUM_LAYERS], rotation=60, fontsize=6)
    ax.set_title(f"{SHORT.get(tag, tag)}  [{DATASET_KIND[tag]}]", fontsize=9)
    ax.grid(alpha=0.25)


def make_figures(metric="mase"):
    _dirs()
    have = [t for t in ALL_TAGS if (OUT_ROOT / "bootstrap_inputs" / f"native_head_adapter__{t}.npz").exists()]
    if not have:
        print("[figures] no bootstrap_inputs found — run `--adapt` first"); return
    # per-dataset panels
    for tag in have:
        pw, series = _load_pw(tag)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        _panel(ax, tag, pw, series, metric)
        ax.set_ylabel(f"{metric.upper()} (original scale; lower = better)")
        ax.set_xlabel("representation depth")
        ax.legend(fontsize=7)
        fig.suptitle(f"native-head adapter — {SHORT.get(tag, tag)} ({metric.upper()})\n"
                     "does a linear map of layer l make it usable by the frozen native Quantile Head?",
                     fontsize=9)
        fig.tight_layout()
        out = OUT_ROOT / "plots" / f"native_head_adapter__{tag}__{metric}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"  [saved] {out.name}")

    # 2x4 overview (7 datasets + legend in the 8th panel)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    axes = axes.ravel()
    for k, tag in enumerate(ALL_TAGS):
        if tag not in have:
            axes[k].set_visible(False); continue
        pw, series = _load_pw(tag)
        _panel(axes[k], tag, pw, series, metric)
        if k % 4 == 0:
            axes[k].set_ylabel(f"{metric.upper()}")
    axes[7].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[7].legend(h, l, loc="center", fontsize=11, title="conditions")
    fig.suptitle(f"native-head adapter overview ({metric.upper()}): native vs zero-shot vs linear-adapter "
                 "— 4 PT-ID + 3 PT-OOD\nadapter Emb..L12 through the frozen native head; dashed = native Chronos-2",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = OUT_ROOT / "plots" / f"native_head_adapter__overview_2x4__{metric}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  [saved] {out.name}")

    _write_regret_table(have)


def _write_regret_table(tags):
    rows = []
    for tag in tags:
        pw, series = _load_pw(tag)
        curves, _nb, _ = _boot_curves(pw, series, "mase")
        wql, _n2, _ = _boot_curves(pw, series, "wql")
        nat = curves[("native", REF_IDX)][0]
        natw = wql[("native", REF_IDX)][0]
        for (cond, L) in sorted(curves, key=lambda t: (CONDITIONS.index(t[0]), t[1])):
            rows.append({
                "dataset": tag, "kind": DATASET_KIND[tag], "condition": cond, "layer": L,
                "layer_label": LAYER_LABELS[L],
                "mase": round(curves[(cond, L)][0], 6), "mase_ci_lo": round(curves[(cond, L)][1], 6),
                "mase_ci_hi": round(curves[(cond, L)][2], 6),
                "relative_regret_mase": round(curves[(cond, L)][0] / nat - 1.0, 6),
                "wql": round(wql[(cond, L)][0], 6),
                "relative_regret_wql": round(wql[(cond, L)][0] / natw - 1.0, 6)})
    stem = OUT_ROOT / "tables" / "native_head_adapter__relative_regret__all"
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    json.dump(rows, open(f"{stem}.json", "w"), indent=2)
    print(f"  [saved] {stem.name}.csv/.json ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
# transfer figures (CPU/login): native reference + the source-transferred adapter curve only
# --------------------------------------------------------------------------- #
def _load_transfer_pw(src, tgt):
    p = TRANSFER_BOOT_DIR / f"native_head_adapter__from_{src}__{tgt}.npz"
    if not p.exists():
        return None
    d = np.load(p, allow_pickle=True)
    series = d["series_test"]
    pw = {}
    for key in d.files:
        if key == "series_test":
            continue
        cond, lab, metric = key.split("__")
        L = int(lab[1:])
        pw.setdefault((cond, L), {})[metric] = d[key]
    return pw, series


def _augment_transfer_pw(tgt, pw, series, metric):
    """Overlay the TARGET's own reference curves onto a transfer panel: the zero-shot 'just frozen
    head' curve (from the target's --adapt npz) and the q1 linear-probe readout (from --linear-baseline).
    Both are target-specific and share the transfer panel's series_test (deterministic windows), so they
    resample under the same paired bootstrap. Fail loud on window drift."""
    merged = dict(pw)
    idp = _load_pw(tgt)                                    # target's own --adapt curves (native/zero_shot/adapter)
    if idp is not None:
        id_pw, id_series = idp
        if not np.array_equal(np.asarray(id_series), np.asarray(series)):
            raise RuntimeError(f"{tgt}: --adapt series_test != transfer series_test — re-run on the same split")
        for (cond, L), d in id_pw.items():
            if cond == "zero_shot":                       # h_l -> frozen head, no adapter
                merged[("zero_shot", L)] = d
    return _maybe_add_linear_q1(tgt, merged, series, metric)   # q1 linear readout (MASE only; asserts series)


def _transfer_panel(ax, src, tgt, pw, series, metric="mase"):
    pw = _augment_transfer_pw(tgt, pw, series, metric)     # + target zero-shot & q1 linear readout
    curves, _nb, _ = _boot_curves(pw, series, metric)
    x = np.arange(NUM_LAYERS)                          # Emb..L12 only — L12+RMS is native, not an adapter depth
    nat = curves.get(("native", REF_IDX))              # native still read from the L12+RMS slots (kept internally)
    if nat is not None:
        ax.axhspan(nat[1], nat[2], color="0.6", alpha=0.18)
        ax.axhline(nat[0], color="0.35", ls="--", lw=1.3, label="native Chronos-2 (frozen head)")
    if any(c == "zero_shot" for (c, _l) in curves):    # target's own no-adapter baseline
        xs = [i for i in x if ("zero_shot", i) in curves]
        mean = np.array([curves[("zero_shot", i)][0] for i in xs])
        lo = np.array([curves[("zero_shot", i)][1] for i in xs])
        hi = np.array([curves[("zero_shot", i)][2] for i in xs])
        ax.plot(xs, mean, marker="o", ms=3.5, color="tab:orange", label="zero-shot native head (no adapter)")
        ax.fill_between(xs, lo, hi, color="tab:orange", alpha=0.15)
    xs = [i for i in x if ("transfer", i) in curves]   # 0..12; the by-construction L12+RMS point is not drawn
    mean = np.array([curves[("transfer", i)][0] for i in xs])
    lo = np.array([curves[("transfer", i)][1] for i in xs])
    hi = np.array([curves[("transfer", i)][2] for i in xs])
    ax.plot(xs, mean, marker="D", ms=3.5, color="tab:purple",
            label=f"{SHORT.get(src, src)} adapter (transferred) + native head")
    ax.fill_between(xs, lo, hi, color="tab:purple", alpha=0.15)
    if any(c == "linear_q1" for (c, _l) in curves):    # target's own fresh linear readout (MASE only)
        xs = [i for i in x if ("linear_q1", i) in curves]
        mean = np.array([curves[("linear_q1", i)][0] for i in xs])
        lo = np.array([curves[("linear_q1", i)][1] for i in xs])
        hi = np.array([curves[("linear_q1", i)][2] for i in xs])
        ax.plot(xs, mean, marker="^", ms=3.5, color="tab:green", label="q1 linear probe (fslot readout)")
        ax.fill_between(xs, lo, hi, color="tab:green", alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS[:NUM_LAYERS], rotation=60, fontsize=6)
    ax.set_title(f"{SHORT.get(tgt, tgt)}  [{DATASET_KIND[tgt]}]", fontsize=9)
    ax.grid(alpha=0.25)


def make_transfer_figures(src_tag, metric="mase"):
    _transfer_dirs()
    panel_tags = [t for t in ALL_TAGS if t != src_tag]
    have = [t for t in panel_tags
            if (TRANSFER_BOOT_DIR / f"native_head_adapter__from_{src_tag}__{t}.npz").exists()]
    if not have:
        print(f"[transfer-figures] no transfer inputs for source {src_tag} — run "
              f"`--transfer-source {src_tag}` first"); return
    short_src = SHORT.get(src_tag, src_tag)
    for tgt in have:
        pw, series = _load_transfer_pw(src_tag, tgt)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        _transfer_panel(ax, src_tag, tgt, pw, series, metric)
        ax.set_ylabel(f"{metric.upper()} (original scale; lower = better)")
        ax.set_xlabel("representation depth")
        ax.legend(fontsize=7)
        fig.suptitle(f"native-head adapter TRANSFER — {short_src} adapter -> {SHORT.get(tgt, tgt)} "
                     f"({metric.upper()})\ndoes {short_src}'s frozen linear alignment make "
                     f"{SHORT.get(tgt, tgt)}'s layers usable by the native Quantile Head?", fontsize=9)
        fig.tight_layout()
        out = TRANSFER_PLOT_DIR / f"native_head_adapter__from_{src_tag}__{tgt}__{metric}.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"  [saved] {out.name}")

    # 2x4 overview (up to 6 target panels + legend)
    fig, axes = plt.subplots(2, 4, figsize=(19, 8.5))
    axes = axes.ravel()
    for k, tgt in enumerate(panel_tags):
        if tgt not in have:
            axes[k].set_visible(False); continue
        pw, series = _load_transfer_pw(src_tag, tgt)
        _transfer_panel(axes[k], src_tag, tgt, pw, series, metric)
        if k % 4 == 0:
            axes[k].set_ylabel(metric.upper())
    for j in range(len(panel_tags), 8):
        axes[j].axis("off")
    h, l = axes[0].get_legend_handles_labels()
    axes[7].legend(h, l, loc="center", fontsize=11, title=f"{short_src}-adapter transfer")
    fig.suptitle(f"native-head adapter TRANSFER ({metric.upper()}): {short_src}-trained adapter frozen, "
                 f"evaluated on the other {len(panel_tags)} datasets\nadapter applied Emb..L12; per-target "
                 "references: native Chronos-2 (dashed), zero-shot (no adapter), q1 linear probe", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = TRANSFER_PLOT_DIR / f"native_head_adapter__from_{src_tag}__overview_2x4__{metric}.png"
    fig.savefig(out, dpi=130); plt.close(fig)
    print(f"  [saved] {out.name}")
    _write_transfer_regret_table(src_tag, have)


def _write_transfer_regret_table(src, tags):
    order = {"native": 0, "transfer": 1}
    rows = []
    for tgt in tags:
        pw, series = _load_transfer_pw(src, tgt)
        curves, _nb, _ = _boot_curves(pw, series, "mase")
        wql, _n2, _ = _boot_curves(pw, series, "wql")
        nat = curves[("native", REF_IDX)][0]
        natw = wql[("native", REF_IDX)][0]
        for (cond, L) in sorted(curves, key=lambda t: (order[t[0]], t[1])):
            rows.append({
                "source_dataset": src, "target_dataset": tgt, "kind": DATASET_KIND[tgt],
                "condition": cond, "layer": L, "layer_label": LAYER_LABELS[L],
                "mase": round(curves[(cond, L)][0], 6), "mase_ci_lo": round(curves[(cond, L)][1], 6),
                "mase_ci_hi": round(curves[(cond, L)][2], 6),
                "relative_regret_mase": round(curves[(cond, L)][0] / nat - 1.0, 6),
                "wql": round(wql[(cond, L)][0], 6),
                "relative_regret_wql": round(wql[(cond, L)][0] / natw - 1.0, 6)})
    stem = TRANSFER_TAB_DIR / f"native_head_adapter__transfer_relative_regret__from_{src}"
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows)
    json.dump(rows, open(f"{stem}.json", "w"), indent=2)
    print(f"  [saved] {stem.name}.csv/.json ({len(rows)} rows)")


# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sanity", action="store_true", help="1 dataset, 5 layers, gates only (compute node)")
    p.add_argument("--adapt", action="store_true", help="full native+zero-shot+adapter per dataset (GPU)")
    p.add_argument("--transfer-source", default=None, choices=ALL_TAGS,
                   help="apply THIS dataset's saved (frozen) adapters to the other 6 datasets — "
                        "predict-only, but loads the model for the frozen head (compute node)")
    p.add_argument("--linear-baseline", action="store_true",
                   help="score the q1 linear fslot probe on each dataset's own test windows and save "
                        "per-window MASE for the figure overlay (compute node; no model load)")
    p.add_argument("--figures", action="store_true", help="aggregate + plots (CPU/login)")
    p.add_argument("--transfer-figures", action="store_true",
                   help="aggregate + plot the --transfer-source transfer curves (CPU/login)")
    p.add_argument("--datasets", nargs="*", default=None, choices=ALL_TAGS,
                   help="subset of the 7 datasets (default: all for --adapt; electricity for --sanity)")
    p.add_argument("--quantile-set", default="q21", choices=["q21"],
                   help="native head is 21-quantile; only q21 is meaningful for this experiment")
    p.add_argument("--metric", default="mase", choices=["mase", "wql", "mae"])
    return p.parse_args(argv)


def main():
    args = _parse_args()
    config.set_dataset_set("extended_v3_rolling")            # PT-ID roster/windows/cache namespace
    _dirs()
    quantiles = validate_quantiles(CHRONOS2_QUANTILES)       # the head's own 21 native levels
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.figures:
        make_figures(args.metric); return
    if args.transfer_figures:
        make_transfer_figures(args.transfer_source or "m4_hourly", args.metric); return

    if args.linear_baseline:                                 # no Chronos-2 model load (fslot caches warm)
        tags = args.datasets or ALL_TAGS
        run_linear_baseline(tags, device)
        print(f"\n[linear-baseline] {len(tags)} datasets done -> {OUT_ROOT}/bootstrap_inputs. "
              "Now (login/CPU): python -m experiments.run_native_head_adapter --figures")
        return

    if not (args.sanity or args.adapt or args.transfer_source):
        print("nothing to do — pass --sanity, --adapt, --linear-baseline, --transfer-source, "
              "--figures, or --transfer-figures")
        return

    from probing.extraction import get_pipeline
    pipeline, _cfg = get_pipeline()                          # loads amazon/chronos-2 FROZEN (GPU)
    print(f"[ext_v5 native-head adapter] device={device}  git={_git_hash()[:8]}  "
          f"quantiles={len(quantiles)}  C={C} H={H} K={K} P={P}")

    if args.sanity:
        tags = args.datasets or ["monash_electricity_hourly"]
        for tag in tags:
            process_dataset(tag, quantiles, device, pipeline, sanity=True)
        print("\n[sanity] gates passed — inspect the numbers above, then launch `--adapt`.")
        return

    if args.transfer_source:
        src = args.transfer_source
        targets = [t for t in ALL_TAGS if t != src]
        transfer_from_source(src, targets, quantiles, device, pipeline)
        print(f"\n[transfer] {SHORT.get(src, src)} adapter -> {len(targets)} targets done -> "
              f"{TRANSFER_ROOT}. Now (login/CPU): python -m experiments.run_native_head_adapter "
              f"--transfer-figures --transfer-source {src}")
        return

    tags = args.datasets or ALL_TAGS
    for tag in tags:
        process_dataset(tag, quantiles, device, pipeline, sanity=False)
    print(f"\n[adapt] {len(tags)} datasets done -> {OUT_ROOT}. Now (login/CPU): "
          "python -m experiments.run_native_head_adapter --figures")


if __name__ == "__main__":
    main()
