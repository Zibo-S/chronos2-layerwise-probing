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
from probing.probes import CHRONOS2_QUANTILES, QUANTILE_SETS, WD_GRID_V2, median_index, validate_quantiles
from probing.native_head_adapter import (NUM_NATIVE_QUANTILES, LinearAdapter, fit_adapter_explicit_val,
                                        native_head_modules, slots_to_normalized_quantiles)
from probing.id_data import build_ood_rolling_windows, build_windows
from probing.tunnel import PT_ID_TAGS, PT_OOD_TAGS
from probing.stats import cluster_bootstrap_apply, cluster_bootstrap_counts, ci_bounds
from experiments.run_id_forecasting import M_SEASON, _ctx_stats, _mase_denominator
from experiments.run_ptood_probing_ftok import C, H, K, SHORT, _fslot_feats

OUT_ROOT = REPO_ROOT / "results" / "ext_v5_native_head_adapter"
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
# figures (CPU/login): per-dataset panels + 2x4 overview, MASE + relative regret
# --------------------------------------------------------------------------- #
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
    curves, _nb, _ = _boot_curves(pw, series, metric)
    x = np.arange(len(LAYER_LABELS))
    # native horizontal band (its point + CI, drawn across the panel)
    nat = curves.get(("native", REF_IDX))
    if nat is not None:
        ax.axhspan(nat[1], nat[2], color="0.6", alpha=0.18)
        ax.axhline(nat[0], color="0.35", ls="--", lw=1.3, label="native (frozen head)")
    for cond, color, marker in (("zero_shot", "tab:orange", "o"), ("linear_adapter", "tab:blue", "s")):
        xs = [i for i in x if (cond, i) in curves]
        mean = np.array([curves[(cond, i)][0] for i in xs])
        lo = np.array([curves[(cond, i)][1] for i in xs])
        hi = np.array([curves[(cond, i)][2] for i in xs])
        label = "zero-shot native head" if cond == "zero_shot" else "linear adapter + native head"
        ax.plot(xs, mean, marker=marker, ms=3.5, color=color, label=label)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS, rotation=60, fontsize=6)
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
                 "— 4 PT-ID + 3 PT-OOD\nadapter trained Emb..L12; L12+RMS is the native endpoint",
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
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--sanity", action="store_true", help="1 dataset, 5 layers, gates only (compute node)")
    p.add_argument("--adapt", action="store_true", help="full native+zero-shot+adapter per dataset (GPU)")
    p.add_argument("--figures", action="store_true", help="aggregate + plots (CPU/login)")
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

    if not (args.sanity or args.adapt):
        print("nothing to do — pass --sanity, --adapt, or --figures"); return

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

    tags = args.datasets or ALL_TAGS
    for tag in tags:
        process_dataset(tag, quantiles, device, pipeline, sanity=False)
    print(f"\n[adapt] {len(tags)} datasets done -> {OUT_ROOT}. Now (login/CPU): "
          "python -m experiments.run_native_head_adapter --figures")


if __name__ == "__main__":
    main()
