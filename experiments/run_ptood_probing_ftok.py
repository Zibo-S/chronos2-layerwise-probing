"""PT-OOD / Probe-ID (fresh target probe) DIAGNOSTIC — SHARED FORECAST-TOKEN readout (v4 future tokens).

DIAGNOSTIC, NOT A TRANSFER EXPERIMENT: the per-layer probes here are re-fit ON each PT-OOD target
(its own rolling train/val), so this measures how linearly accessible the forecast is on an
unseen-pretraining dataset — the "Probe-ID" (fresh target probe) quadrant. The genuine cross-dataset
transfer experiments (PT-ID/Probe-OOD 4×4 and PT-OOD/Probe-OOD) live in experiments/run_fslot_transfer.py
and reuse the FROZEN PT-ID source probes + sustained tunnels this driver produces. This module also
remains the producer of those shared inputs (--fit-ptid / --tunnels-only).

Sibling of experiments/run_ptood_probing.py. Same 2023 protocol and same PT-ID/PT-OOD tunnel
framing (probing.tunnel), but the headline readout is the **shared-head future-token probe**
(fit_shared_forecast_probe_explicit_val / predict_shared_forecast_probe over the K=ceil(H/P) native
forecast-slot states, extract_kout_features -> feats["fslot"]) instead of pooled content. This is
the deliberate v4 pivot: the story becomes the Chronos-native readout; the committed pooled-content
run (run_ptood_probing.py -> results/extended_v3_rolling/) stays byte-identical as the comparison.

Two structural differences from the content driver, both deliberate:
  * NAMESPACE. The datasets are unchanged (the 4 extended_v3_rolling rolling datasets), so the
    active dataset set stays "extended_v3_rolling" — that keeps the rolling WINDOWS, roster and
    feature-cache namespace identical to the content run. Only the OUTPUTS are re-rooted to
    results/ext_v4_future_tokens/ via OUT_ROOT (a documented exception to the set=output coupling;
    the separation here is by READOUT, content vs fslot, not by dataset). fslot feature caches sit
    under the v3 cache namespace but are disjoint from content by the K<K>_H<H> cache-key suffix.
  * NO SEED-0 LEGACY REUSE. The content driver treats "PT-ID seed 0" as the committed 4x4 rolling
    checkpoints — those are pooled content and do not exist for this readout. So all three PT-ID
    run seeds are fit FRESH here (RUN_SEEDS = 0, 1, 2), each writing uniform ptid_runs artifacts.

Every reported condition uses 3 INDEPENDENT PROBE RUNS (backbone frozen -> run seed = the probe's
Linear init, the only randomness in the deterministic full-batch fit; init_seed threaded through
fit_shared_forecast_probe_explicit_val). Windows/features are run-seed-independent (shared caches);
tunnels are defined from the MEAN validation curve, never per-seed indices averaged.

Stages (mirrors the content driver):
  --fit-ptid    (GPU): fit the PT-ID rolling probes for ALL run seeds, score their own test split,
                       write per-seed val/test curves + per-window losses. Idempotent (skips seeds
                       already fit). The no-flag full run does this first if artifacts are missing.
  --tunnels-only (CPU): per PT-ID source, average the 3 temporal-val curves -> sustained-plateau
                       tunnel l_start = min{l : mean_val(j) <= 1.05*mean_val(last) for all j>=l};
                       excursion M; D_ID with
                       a series-cluster-bootstrap CI on the seed-averaged window losses.
  (default, GPU): per PT-OOD target x run seed, rolling split ON the target, fresh per-layer probes
                       (wd on target-VAL only), score target-test at every layer.
  --figure-only (CPU): overlay each PT-ID tunnel on each PT-OOD mean test curve; D_OOD(s,t) at s's
                       l_start and Delta(s,t) = D_OOD - D_ID (independent-replicate bootstrap diff).

Run (USER submits SLURM):
    python -m experiments.run_ptood_probing_ftok --fit-ptid          # PT-ID all seeds (GPU)
    python -m experiments.run_ptood_probing_ftok --tunnels-only      # PT-ID tunnels (CPU)
    python -m experiments.run_ptood_probing_ftok --targets sg_carpark  # one target, all seeds (GPU)
    python -m experiments.run_ptood_probing_ftok --figure-only       # aggregate (CPU)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from probing import config
from probing.config import NUM_LAYERS, SEED, OUTPUT_PATCH_SIZE   # last index is data-driven (post-LN=13)
from probing.extraction import extract_kout_features
from probing.id_data import build_ood_rolling_windows, build_windows
from probing.probes import (QUANTILE_SETS, fit_shared_forecast_probe_explicit_val,
                            predict_shared_forecast_probe, validate_quantiles)
from probing.tunnel import (PT_ID_TAGS, PT_OOD_TAGS, TUNNEL_TOL, d_stat_boot, delta_stat,
                            domain_status, max_excursion, tunnel_record_multi,
                            val_curve_from_selection)

# fixed frame, inherited unchanged from the rolling 4x4; readout is the ONLY change vs content
PTID_SET = "extended_v3_rolling"   # active dataset set = roster + rolling windows + cache namespace
READOUT = "fslot"                  # shared forecast-token readout (tag for filenames/checkpoints)
OUT_ROOT = config.REPO_ROOT / "results" / "ext_v4_future_tokens"   # v4 output namespace (override)
C, H = 512, 64
K = math.ceil(H / OUTPUT_PATCH_SIZE)   # native slot count, e.g. H=64, P=16 -> K=4
QUANTILE_EPOCHS = 300
WD_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
RUN_SEEDS = (0, 1, 2)              # 3 independent probe-init runs; ALL fit fresh (no legacy seed 0)
RUN_TYPE = "probe_seed"
RUNS_TAG = "runs" + "-".join(str(s) for s in RUN_SEEDS)
# This driver's target eval is the PT-OOD / Probe-ID (fresh target probe) DIAGNOSTIC — the probe is
# re-fit ON the target, so it measures representation accessibility on an unseen-pretraining dataset,
# NOT cross-dataset probe transfer. The transfer experiments (PT-ID/Probe-OOD 4×4 + PT-OOD/Probe-OOD)
# live in experiments/run_fslot_transfer.py, which reuses this module's frozen PT-ID probes + tunnels.
PROBE_ID_DIAG = "PT-OOD / Probe-ID (fresh target probe)"

# The shared-head fslot line adds ONE readout point beyond the 13 block states (Emb, L1..L12):
# index NUM_LAYERS (=13) = the POST-final-LayerNorm forecast slots (the native head's ACTUAL input,
# from extract_kout_features's final["fslot"]). So fslot curves have NUM_LAYERS+1 points, and this
# post-LN point is the tunnel's "last" reference (tunnel.py is length-agnostic — uses curve[-1]).
POST_LN_LABEL = "L12+LN"           # x-axis label for the post-final-LN native-head input
LAYER_LABELS = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)] + [POST_LN_LABEL]   # len NUM_LAYERS+1

SHORT = {"monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber",
         "m4_hourly": "M4", "wind_farms_hourly": "WindFarms",
         "sg_carpark": "SG Carpark", "coastal_ts": "Coastal T-S", "boom_hourly": "BOOM"}


def _derive_dirs():
    global OUT_DIR, TUNNEL_DIR, PER_TARGET_DIR, BOOT_IN_DIR, CKPT_DIR, FIG_DIR, \
        PTID_RUN_DIR, PTID_CKPT_DIR
    OUT_DIR = OUT_ROOT / "ptood_probing"
    TUNNEL_DIR = OUT_ROOT / "tunnels"                   # PT-ID artifact, portable by source tag
    PER_TARGET_DIR = OUT_DIR / "per_target"
    BOOT_IN_DIR = OUT_DIR / "bootstrap_inputs"
    CKPT_DIR = OUT_DIR / "checkpoints"                  # per-PT-OOD-target frozen probes
    FIG_DIR = OUT_DIR / "figures"
    PTID_RUN_DIR = OUT_DIR / "ptid_runs"                # per-seed PT-ID val/test curves + windows
    PTID_CKPT_DIR = OUT_DIR / "ptid_checkpoints"        # per-source frozen PT-ID probes (all seeds)
    for d in (OUT_DIR, TUNNEL_DIR, PER_TARGET_DIR, BOOT_IN_DIR, CKPT_DIR, FIG_DIR,
              PTID_RUN_DIR, PTID_CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _fslot_feats(tag, split, X, y):
    """{layer: (n, K, 768)} forecast-slot states from the num_output_patches=K pass (GPU on a cold
    cache). Keys 0..NUM_LAYERS-1 = the PRE-final-LN block states (Emb, L1..L12); key NUM_LAYERS (=13)
    = the POST-final-LN slots (final["fslot"], the native head's actual input) — the extra readout
    point beyond L12. K is derived inside extract_kout_features from H; assert it matches the driver."""
    fk, final, _ = extract_kout_features(tag, split, X, y, horizon=H)
    feats = dict(fk["fslot"])                       # keys 0..NUM_LAYERS-1 (pre-final-LN block slots)
    feats[NUM_LAYERS] = final["fslot"]              # key NUM_LAYERS = post-final-LN native-head input
    for i, arr in feats.items():
        assert np.ndim(arr) == 3 and arr.shape[1] == K, (
            f"{tag}/{split} L{i}: expected (n, {K}, 768) forecast slots, got {np.shape(arr)}")
    return feats


def _save_ckpt(ckdir, fitted):
    """Freeze a shared-forecast probe (per layer): linear state_dict + slot-scaler + selection +
    the shape/patch metadata needed to rebuild + re-apply predict_shared_forecast_probe."""
    ckdir.mkdir(parents=True, exist_ok=True)
    for i, f in fitted.items():
        torch.save({"state_dict": f["linear"].state_dict(), "scaler_mean": f["scaler"].mean_,
                    "scaler_scale": f["scaler"].scale_, "wd": f["wd"], "selection": f["selection"],
                    "in_features": f["in_features"], "out_features": f["out_features"],
                    "output_patch_size": f["output_patch_size"], "K": f["K"],
                    "family": f["family"]}, ckdir / f"L{i:02d}.pt")


def _ptid_ckpt_dir(src, qset, seed):
    """Path of a frozen PT-ID source probe. Mirrors fit_ptid's save path but is built from the
    OUT_ROOT constant (not the _derive_dirs() global), so a SIBLING driver can resolve it without
    running this module's main()."""
    return (OUT_ROOT / "ptood_probing" / "ptid_checkpoints"
            / f"{src}__{READOUT}__C{C}_H{H}__{qset}__seed{seed}")


def _scaler_from_arrays(mean, scale):
    """Rebuild the frozen slot StandardScaler from its stored mean_/scale_ (the fslot checkpoint
    saves the arrays, not the pickled object). Setting mean_/scale_/var_/n_features_in_ makes
    .transform reproduce the original scaler exactly."""
    sc = StandardScaler()
    sc.mean_ = np.asarray(mean, dtype=np.float64)
    sc.scale_ = np.asarray(scale, dtype=np.float64)
    sc.var_ = sc.scale_ ** 2
    sc.n_features_in_ = int(sc.mean_.shape[0])
    return sc


def load_ptid_ckpt(src, qset, seed, device="cpu"):
    """Reload a frozen PT-ID source probe (saved by _save_ckpt) as a fitted dict that
    predict_shared_forecast_probe consumes directly — the FROZEN source probe reused across the
    transfer row. Rebuilds each layer's StandardScaler (from mean/scale) and nn.Linear (from
    state_dict, eval mode); NEVER trains. Keys = the 14 fslot readout points (L0..L12 + post-LN).
    Fail-loud on a missing / short checkpoint dir."""
    d = _ptid_ckpt_dir(src, qset, seed)
    paths = sorted(d.glob("L*.pt"))
    if not paths:
        raise FileNotFoundError(f"no checkpoints in {d} — run run_ptood_probing_ftok --fit-ptid first")
    fitted = {}
    for p in paths:
        i = int(p.stem[1:])                                  # "L07" -> 7
        ck = torch.load(p, map_location=device, weights_only=False)
        lin = torch.nn.Linear(ck["in_features"], ck["out_features"])
        lin.load_state_dict(ck["state_dict"])
        lin.to(device)
        lin.eval()
        fitted[i] = {"linear": lin,
                     "scaler": _scaler_from_arrays(ck["scaler_mean"], ck["scaler_scale"]),
                     "wd": float(ck["wd"]), "selection": ck["selection"],
                     "in_features": int(ck["in_features"]), "out_features": int(ck["out_features"]),
                     "output_patch_size": int(ck["output_patch_size"]), "K": int(ck["K"]),
                     "family": ck["family"], "device": str(device)}
    return fitted


# --------------------------------------------------------------------------- #
# Stage 1a — PT-ID rolling probes, ALL run seeds fit fresh (GPU)
# --------------------------------------------------------------------------- #
def fit_ptid(qset, quantiles, device):
    """Fit the PT-ID shared-forecast rolling probes for every run seed (idempotent — skips seeds
    already on disk). Features are extracted once per source (run-seed-independent) and reused
    across seeds; only the probe init differs by seed."""
    for src in PT_ID_TAGS:
        pending = [s for s in RUN_SEEDS
                   if not (PTID_RUN_DIR / f"{src}__{qset}__seed{s}.json").exists()]
        if not pending:
            print(f"  [skip] {SHORT[src]}: all seeds already fit")
            continue
        w = build_windows(src)                 # rolling split (window seed fixed at SEED)
        f_tr = _fslot_feats(src, "train", w["X_train"], w["y_train"])
        f_va = _fslot_feats(src, "val", w["X_val"], w["y_val"])
        f_te = _fslot_feats(src, "test", w["X_test"], w["y_test"])
        for seed in pending:
            print(f"\n[fit PT-ID {READOUT}] {SHORT[src]} run seed {seed} ({qset})")
            fitted = fit_shared_forecast_probe_explicit_val(
                f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"], quantiles=quantiles,
                epochs=QUANTILE_EPOCHS, wd_grid=WD_GRID, device=device, init_seed=seed)
            _save_ckpt(PTID_CKPT_DIR / f"{src}__{READOUT}__C{C}_H{H}__{qset}__seed{seed}", fitted)
            out, diag = predict_shared_forecast_probe(fitted, f_te, w["Y_test_traj"],
                                                      quantiles=quantiles, device=device,
                                                      collect_test_window_loss=True)
            wl = np.stack([diag["test_window_loss"][i]
                           for i in sorted(diag["test_window_loss"])]).astype(np.float64)  # 14 rows (fslot)
            np.savez(PTID_RUN_DIR / f"{src}__{qset}__seed{seed}.npz", window_loss=wl,
                     series_test=np.asarray(w["series_test"], np.int64))
            json.dump({"dataset": src, "quantile_set": qset, "run_seed": int(seed),
                       "run_type": RUN_TYPE, "readout": READOUT, "pooling_or_token_type": "forecast_slot",
                       "val_loss_by_layer": val_curve_from_selection(
                           {i: fitted[i]["selection"] for i in sorted(fitted)}, num_layers=len(fitted)),
                       "test_loss_by_layer": [float(out[i]) for i in sorted(out)]},
                      open(PTID_RUN_DIR / f"{src}__{qset}__seed{seed}.json", "w"), indent=2)
            print(f"  [saved] {src}__{qset}__seed{seed}.json")
        del w, f_tr, f_va, f_te
        gc.collect()


# --------------------------------------------------------------------------- #
# PT-ID per-run curves (uniform for every seed — no legacy special case)
# --------------------------------------------------------------------------- #
def _ptid_run_curves(src, qset, seed):
    """(val_curve list, window_loss (n_points, n), series_test) for ONE PT-ID probe run
    (n_points = NUM_LAYERS+1 = 14 for fslot: L0..L12 + post-final-LN)."""
    pj = PTID_RUN_DIR / f"{src}__{qset}__seed{seed}.json"
    pz = PTID_RUN_DIR / f"{src}__{qset}__seed{seed}.npz"
    if not (pj.exists() and pz.exists()):
        raise FileNotFoundError(f"missing PT-ID run artifacts for {src} seed {seed} ({pj.name}) — "
                                "run --fit-ptid first")
    z = np.load(pz)
    return json.load(open(pj))["val_loss_by_layer"], z["window_loss"], z["series_test"]


def _tunnel_path(src, qset):
    return TUNNEL_DIR / f"{src}__{READOUT}__{qset}__{RUNS_TAG}.json"


def _seed_mean_windows(curves_by_seed):
    """Seed-averaged per-window losses (same windows across runs — asserted)."""
    wls = [wl for _, wl, _ in curves_by_seed]
    sids = [sid for _, _, sid in curves_by_seed]
    assert all(w.shape == wls[0].shape for w in wls), "runs must share identical test windows"
    assert all(np.array_equal(s, sids[0]) for s in sids), "runs must share identical series ids"
    return np.mean(wls, axis=0), sids[0]


# --------------------------------------------------------------------------- #
# Stage 1b — one tunnel per PT-ID dataset from the MEAN validation curve (CPU)
# --------------------------------------------------------------------------- #
def compute_ptid_tunnels(qset):
    for src in PT_ID_TAGS:
        runs = [_ptid_run_curves(src, qset, s) for s in RUN_SEEDS]
        val_by_run = [v for v, _, _ in runs]
        test_by_run = [wl.mean(axis=1) for _, wl, _ in runs]
        wl_mean, sid = _seed_mean_windows(runs)
        rec = tunnel_record_multi(src, val_by_run, test_by_run, RUN_SEEDS, run_type=RUN_TYPE,
                                  val_split_kind="temporal_rolling",
                                  extra={"quantile_set": qset, "readout": READOUT,
                                         "pooling_or_token_type": "forecast_slot",
                                         "dataset_set": PTID_SET,
                                         "provenance": {"ptid_runs": str(PTID_RUN_DIR)}})
        last = wl_mean.shape[0] - 1        # data-driven last index (=13 for fslot) = post-LN reference
        d_id = d_stat_boot(wl_mean, sid, rec["l_start"], last=last, B=config.BOOT_B, seed=SEED)
        rec["d_id_ci"] = list(d_id["ci"])
        rec["d_id_by_run"] = [float((t[-1] - t[rec["l_start"]]) / t[rec["l_start"]])
                              for t in test_by_run]
        rec["n_clusters"] = d_id["n_clusters"]; rec["n_windows"] = d_id["n_windows"]
        out = _tunnel_path(src, qset)
        json.dump(rec, open(out, "w"), indent=2)
        print(f"  [{SHORT[src]:>12}] tunnel [{LAYER_LABELS[rec['l_start']]}, {LAYER_LABELS[last]}]  "
              f"(sustained plateau)  M_test={rec['M_test']:+.3f}  "
              f"D_ID={rec['D_ID']:+.3f} [{d_id['ci'][0]:+.3f}, {d_id['ci'][1]:+.3f}] "
              f"per-run {['%+.3f' % d for d in rec['d_id_by_run']]} -> {out.name}")
        _tunnel_figure(rec)


def _plot_runs(ax, curves_by_run, mean, std, color, label):
    """Mean curve + shaded +-1 std band + faint individual run curves."""
    x = np.arange(len(mean))               # data-driven (14 for fslot: L0..L12 + post-LN)
    for c in curves_by_run:
        ax.plot(x, c, "-", color=color, alpha=0.25, lw=0.9)
    ax.fill_between(x, np.array(mean) - np.array(std), np.array(mean) + np.array(std),
                    color=color, alpha=0.15)
    ax.plot(x, mean, "o-", color=color, label=label)


def _tunnel_figure(rec):
    src, ls = rec["dataset"], rec["l_start"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(len(rec["mean_test_loss_by_layer"]))     # 14 for fslot
    last = len(x) - 1                                       # post-LN reference (L12+LN)
    ax.axvspan(ls - 0.25, last + 0.25, color="tab:green", alpha=0.12,
               label=f"tunnel (sustained plateau) [{LAYER_LABELS[ls]}, {LAYER_LABELS[last]}]")
    _plot_runs(ax, rec["val_loss_by_run"], rec["mean_val_loss_by_layer"],
               rec["std_val_loss_by_layer"], "tab:blue",
               f"validation, mean of {len(rec['run_seeds'])} runs (defines tunnel)")
    _plot_runs(ax, rec["test_loss_by_run"], rec["mean_test_loss_by_layer"],
               rec["std_test_loss_by_layer"], "tab:orange",
               f"test, mean of {len(rec['run_seeds'])} runs (tunnel frozen)")
    ax.axhline((1 + rec["tolerance"]) * rec["mean_val_loss_by_layer"][-1], color="tab:blue",
               ls=":", lw=1, label="1.05 x final-layer mean val loss")
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS[:len(x)])
    ax.set_xlabel("layer"); ax.set_ylabel("Chronos-2 quantile loss")
    ax.set_title(f"{SHORT[src]} (PT-ID, shared forecast-token): sustained-plateau tunnel from MEAN "
                 f"validation  (M_test={rec['M_test']:+.2f}, D_ID={rec['D_ID']:+.3f})", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / f"ptid_tunnel__{src}__{READOUT}__{rec['quantile_set']}__{RUNS_TAG}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"    [saved] {out.name}")


# --------------------------------------------------------------------------- #
# Stage 2 — fresh per-layer probes ON each PT-OOD target, one fit per run seed (GPU)
# --------------------------------------------------------------------------- #
def eval_target(tag, qset, quantiles, seed, device):
    """Rolling split on the PT-OOD target; shared-forecast probes trained on target-train, wd on
    target-VAL, scored on target-test. The target's test set never tunes anything."""
    print(f"\n[eval {PROBE_ID_DIAG} — {READOUT}] {SHORT.get(tag, tag)} ({qset}, run seed {seed})")
    w = build_ood_rolling_windows(tag, C=C, H=H, seed=SEED)     # window seed FIXED across runs
    m = w["meta"]
    if m["n_test"] == 0 or m["n_train"] == 0:
        raise RuntimeError(f"{tag}: empty split (train {m['n_train']} / test {m['n_test']}) — "
                           "check the loader / rolling eligibility (run_ood_screen)")
    print(f"  windows: train {m['n_train']} / val {m['n_val']} / test {m['n_test']}  "
          f"({m['n_test_clusters']} {m['cluster_unit']} clusters)")
    # "*_rolling" split names keep these fslot caches disjoint from any legacy eval-only test cache
    f_tr = _fslot_feats(tag, "train_rolling", w["X_train"], w["y_train"])
    f_va = _fslot_feats(tag, "val_rolling", w["X_val"], w["y_val"])
    f_te = _fslot_feats(tag, "test_rolling", w["X_test"], w["y_test"])
    fitted = fit_shared_forecast_probe_explicit_val(f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"],
                                                    quantiles=quantiles, epochs=QUANTILE_EPOCHS,
                                                    wd_grid=WD_GRID, device=device, init_seed=seed)
    _save_ckpt(CKPT_DIR / f"{tag}__{READOUT}__C{C}_H{H}__{qset}__seed{seed}", fitted)

    out, diag = predict_shared_forecast_probe(fitted, f_te, w["Y_test_traj"], quantiles=quantiles,
                                              device=device, collect_test_window_loss=True)
    wl = np.stack([diag["test_window_loss"][i]
                   for i in sorted(diag["test_window_loss"])]).astype(np.float64)   # 14 rows (fslot)
    sid = np.asarray(w["series_test"], np.int64)
    np.savez(BOOT_IN_DIR / f"{tag}__{qset}__seed{seed}.npz", window_loss=wl, series_test=sid)

    val_curve = val_curve_from_selection({i: fitted[i]["selection"] for i in sorted(fitted)},
                                         num_layers=len(fitted))
    payload = {
        "dataset": tag, "domain_status": domain_status(tag), "quantile_set": qset,
        "pt_status": ("PT-ID" if domain_status(tag)["pretraining"] == "pt_id" else "PT-OOD"),
        "probe_status": "Probe-ID", "quadrant": PROBE_ID_DIAG,
        "run_seed": int(seed), "run_type": RUN_TYPE, "readout": READOUT,
        "pooling_or_token_type": "forecast_slot",
        "protocol": "fresh_probe_2023", "val_split_kind": "temporal_rolling",
        "test_loss_by_layer": [float(out[i]) for i in sorted(out)],
        "val_loss_by_layer": val_curve,
        "chosen_wd_by_layer": {str(i): float(fitted[i]["wd"]) for i in sorted(fitted)},
        "meta": {k: v for k, v in m.items() if k != "notes"}, "notes": m["notes"],
    }
    outp = PER_TARGET_DIR / f"{tag}__{qset}__seed{seed}.json"
    json.dump(payload, open(outp, "w"), indent=2)
    ql = np.array(payload["test_loss_by_layer"])
    print(f"  test loss: last {LAYER_LABELS[-1]}={ql[-1]:.3f}  "
          f"min {LAYER_LABELS[int(ql.argmin())]}={ql.min():.3f}  -> {outp.name}")
    del w, f_tr, f_va, f_te
    gc.collect()


# --------------------------------------------------------------------------- #
# Stage 3 — apply PT-ID tunnels to PT-OOD mean curves; D_OOD + Delta (CPU)
# --------------------------------------------------------------------------- #
def _load_tunnels(qset):
    recs = {}
    for src in PT_ID_TAGS:
        p = _tunnel_path(src, qset)
        if not p.exists():
            raise FileNotFoundError(f"missing tunnel record {p} — run --tunnels-only first")
        recs[src] = json.load(open(p))
    return recs


def _load_ptood_runs(tgt, qset):
    payloads, runs = [], []
    for seed in RUN_SEEDS:
        pj = PER_TARGET_DIR / f"{tgt}__{qset}__seed{seed}.json"
        pz = BOOT_IN_DIR / f"{tgt}__{qset}__seed{seed}.npz"
        if not (pj.exists() and pz.exists()):
            continue
        z = np.load(pz)
        payloads.append(json.load(open(pj)))
        runs.append((payloads[-1]["val_loss_by_layer"], z["window_loss"], z["series_test"]))
    if not runs:
        return None
    if len(runs) < len(RUN_SEEDS):
        print(f"  [warn] {tgt}: only {len(runs)}/{len(RUN_SEEDS)} runs present — "
              "aggregating what exists")
    return payloads, runs


def aggregate(qset):
    tunnels = _load_tunnels(qset)
    d_id = {}
    for src in PT_ID_TAGS:
        wl_mean, sid = _seed_mean_windows([_ptid_run_curves(src, qset, s) for s in RUN_SEEDS])
        d_id[src] = d_stat_boot(wl_mean, sid, tunnels[src]["l_start"], last=wl_mean.shape[0] - 1,
                                B=config.BOOT_B, seed=SEED)     # last=13 (post-LN) for fslot

    rows, targets = [], []
    for tgt in PT_OOD_TAGS:
        loaded = _load_ptood_runs(tgt, qset)
        if loaded is None:
            print(f"  [skip] {tgt}: no per-target results yet")
            continue
        payloads, runs = loaded
        test_by_run = [wl.mean(axis=1) for _, wl, _ in runs]
        wl_mean, sid = _seed_mean_windows(runs)
        mean_test = wl_mean.mean(axis=1)
        targets.append((tgt, payloads, runs))
        for src in PT_ID_TAGS:
            ls = tunnels[src]["l_start"]
            src_last = len(tunnels[src]["mean_test_loss_by_layer"]) - 1     # 13 (post-LN) for fslot
            d_ood = d_stat_boot(wl_mean, sid, ls, last=wl_mean.shape[0] - 1,
                                B=config.BOOT_B, seed=SEED)
            dl = delta_stat(d_ood, d_id[src])
            rows.append({
                "source_dataset": src, "target_dataset": tgt, "quantile_set": qset,
                "readout": READOUT, "run_type": RUN_TYPE,
                "run_seeds": " ".join(str(s) for s in RUN_SEEDS), "n_runs_present": len(runs),
                "tolerance": TUNNEL_TOL, "l_start": ls,
                "tunnel": f"[{LAYER_LABELS[ls]}, {LAYER_LABELS[src_last]}]",
                "ptid_m_test": round(tunnels[src]["M_test"], 6),
                "ptood_m_test": round(max_excursion(mean_test, ls), 6),
                "ptid_test_criterion_holds": tunnels[src]["test_criterion_holds"],
                "d_id": round(d_id[src]["point"], 6),
                "d_id_ci_lo": round(d_id[src]["ci"][0], 6),
                "d_id_ci_hi": round(d_id[src]["ci"][1], 6),
                "d_id_by_run": " ".join(f"{d:+.4f}" for d in tunnels[src]["d_id_by_run"]),
                "d_ood": round(d_ood["point"], 6),
                "d_ood_ci_lo": round(d_ood["ci"][0], 6),
                "d_ood_ci_hi": round(d_ood["ci"][1], 6),
                "d_ood_by_run": " ".join(f"{(t[-1] - t[ls]) / t[ls]:+.4f}" for t in test_by_run),
                "delta": round(dl["point"], 6),
                "delta_ci_lo": round(dl["ci"][0], 6),
                "delta_ci_hi": round(dl["ci"][1], 6),
                "delta_excludes_zero": bool(dl["ci"][0] > 0 or dl["ci"][1] < 0),
                "n_ood_clusters": d_ood["n_clusters"], "n_ood_windows": d_ood["n_windows"],
                "delta_ci_note": "independent-replicate bootstrap difference (disjoint test "
                                 "sets); point stats on 3-run mean curves",
            })
    if not rows:
        print("  [aggregate] no PT-OOD results found — run the GPU eval first")
        return
    base = OUT_DIR / f"tunnel_effect_stats__{READOUT}__{qset}__{RUNS_TAG}"
    with open(f"{base}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    json.dump(rows, open(f"{base}.json", "w"), indent=2)
    print(f"  [saved] {base.name}.csv/.json ({len(rows)} source x target cells)")

    for tgt, payloads, runs in targets:
        _target_overlay_figure(tgt, runs, tunnels, qset)
    _delta_heatmap(rows, qset)
    _print_summary(rows)


def _target_overlay_figure(tgt, runs, tunnels, qset):
    T = np.stack([wl.mean(axis=1) for _, wl, _ in runs])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _plot_runs(ax, T, T.mean(axis=0), T.std(axis=0), "tab:orange",
               f"PT-OOD test loss, mean of {len(runs)} fresh-probe runs")
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(PT_ID_TAGS)))
    for c, src in zip(colors, PT_ID_TAGS):
        ls = tunnels[src]["l_start"]
        ax.axvline(ls, color=c, ls="--", lw=1.2, label=f"{SHORT[src]} l_start = {LAYER_LABELS[ls]}")
    x = np.arange(T.shape[1])              # 14 for fslot
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS[:T.shape[1]])
    ax.set_xlabel("layer"); ax.set_ylabel("Chronos-2 quantile loss")
    ax.set_title(f"{SHORT.get(tgt, tgt)} [{PROBE_ID_DIAG}, shared forecast-token]: fresh layerwise "
                 "probes under the four PT-ID mean-validation tunnels", fontsize=9)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / f"ptood_curve_with_tunnels__{tgt}__{READOUT}__{qset}__{RUNS_TAG}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [saved] {out.name}")


def _delta_heatmap(rows, qset):
    tgts = sorted({r["target_dataset"] for r in rows}, key=list(PT_OOD_TAGS).index)
    M = np.full((len(PT_ID_TAGS), len(tgts)), np.nan)
    by = {(r["source_dataset"], r["target_dataset"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(1.9 * len(tgts) + 2.6, 4.2))
    for i, s in enumerate(PT_ID_TAGS):
        for j, t in enumerate(tgts):
            r = by[(s, t)]
            M[i, j] = 100 * r["delta"]
            star = "*" if r["delta_excludes_zero"] else ""
            ax.text(j, i, f"{100*r['delta']:+.1f}%{star}\n({LAYER_LABELS[r['l_start']]})",
                    ha="center", va="center", fontsize=8)
    vmax = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(tgts))); ax.set_xticklabels([SHORT.get(t, t) for t in tgts])
    ax.set_yticks(range(len(PT_ID_TAGS)))
    ax.set_yticklabels([SHORT[s] for s in PT_ID_TAGS])
    ax.set_xlabel("PT-OOD target (Probe-ID: fresh target probe)"); ax.set_ylabel("PT-ID source (tunnel)")
    ax.set_title(f"Delta(s,t) = D_OOD - D_ID at the source tunnel start  "
                 f"[{PROBE_ID_DIAG}, shared forecast-token, {qset}, {RUNS_TAG}]\n"
                 "positive = late-layer degradation stronger PT-OOD; * = 95% CI excludes 0",
                 fontsize=9)
    fig.colorbar(im, ax=ax, label="Delta (pct points)")
    fig.tight_layout()
    out = FIG_DIR / f"delta_heatmap__{READOUT}__{qset}__{RUNS_TAG}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [saved] {out.name}")


def _print_summary(rows):
    print(f"\n  == tunnel-effect summary [{PROBE_ID_DIAG}, shared forecast-token, 3-run means; "
          "D > 0: final layer worse than tunnel entrance] ==")
    for r in rows:
        print(f"    {SHORT[r['source_dataset']]:>12} ({LAYER_LABELS[r['l_start']]}) -> "
              f"{SHORT.get(r['target_dataset'], r['target_dataset']):<12} "
              f"D_ID={r['d_id']:+.3f}  D_OOD={r['d_ood']:+.3f}  "
              f"Delta={r['delta']:+.3f} [{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}]"
              f"{' *' if r['delta_excludes_zero'] else ''}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quantile-set", default="q9", choices=sorted(QUANTILE_SETS))
    p.add_argument("--targets", nargs="*", default=list(PT_OOD_TAGS), choices=list(PT_OOD_TAGS))
    p.add_argument("--fit-ptid", action="store_true",
                   help="fit the PT-ID rolling probes for ALL run seeds (GPU) and exit")
    p.add_argument("--tunnels-only", action="store_true",
                   help="compute the PT-ID tunnel records + figures from all runs (CPU) and exit")
    p.add_argument("--figure-only", action="store_true",
                   help="aggregate saved per-target results into stats + figures (CPU)")
    return p.parse_args(argv)


def main():
    args = _parse_args()
    config.set_dataset_set(PTID_SET)     # roster + rolling windows + cache namespace (outputs -> OUT_ROOT)
    _derive_dirs()
    quantiles = validate_quantiles(QUANTILE_SETS[args.quantile_set])
    print(f"[run_ptood_probing_ftok] readout={READOUT}  set={PTID_SET}  out={OUT_ROOT.name}  "
          f"qset={args.quantile_set}  {RUNS_TAG}  K={K}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fit_ptid:
        fit_ptid(args.quantile_set, quantiles, device)
        return
    if not (args.tunnels_only or args.figure_only):
        fit_ptid(args.quantile_set, quantiles, device)   # idempotent: fit any missing PT-ID seeds
    compute_ptid_tunnels(args.quantile_set)
    if args.tunnels_only:
        return
    if not args.figure_only:
        for tgt in args.targets:
            for seed in RUN_SEEDS:
                eval_target(tgt, args.quantile_set, quantiles, seed, device)
    aggregate(args.quantile_set)


if __name__ == "__main__":
    main()
