#!/usr/bin/env python
"""Training vs validation loss by layer for the shared-LINEAR fslot readout (q9).

The linear PT-ID probes only persisted val/test loss by layer (no train — _save_fit_histories
is MLP-only), so the train curve is RECOMPUTED here post-hoc: reload each frozen linear
checkpoint (load_ptid_ckpt) and apply predict_shared_forecast_probe to that dataset's cached
TRAIN forecast-slot features — the exact primitive the driver used to write test_loss_by_layer,
just on the train split. As a correctness check the val curve is ALSO recomputed the same way
and compared against the saved val_loss_by_layer (should match to numerical tolerance).

COMPUTE-NODE ONLY (reads 4x ~344 MB train caches + builds windows) — do NOT run on a login node.
Caches are warm, so it is CPU-only (no GPU / no model). Run from the repo root:

    salloc --account=def-irina --cpus-per-task=4 --mem=32G --time=0:45:00
    module load gcc python/3.11 arrow/24.0.0 && source .venv/bin/activate
    export HF_HOME=$SCRATCH/chronos2/hf_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4 MPLBACKEND=Agg
    python results/ext_v4_future_tokens/ptood_probing/figures/plot_linear_train_vs_val.py

Writes:  linear_train_vs_val_by_layer.png (this dir)  and  a per-seed
         ptood_probing/ptid_runs/<tag>__q9__seed<seed>__train_recompute.json  (fills the gap).
"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# running by file path puts THIS dir on sys.path, not the repo root -> make `probing`/`experiments` importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))

from probing import config
config.set_dataset_set("extended_v3_rolling")          # roster + rolling windows + cache prefix
from probing.id_data import build_windows
from probing.probes import QUANTILE_SETS, predict_shared_forecast_probe
from experiments.run_ptood_probing_ftok import (load_ptid_ckpt, _fslot_feats, SHORT,
                                                 LAYER_LABELS, OUT_ROOT)

PTID_RUN_DIR = OUT_ROOT / "ptood_probing" / "ptid_runs"   # linear PT-ID run artifacts (not a module const)

HERE = os.path.dirname(os.path.abspath(__file__))
DS = [("monash_electricity_hourly", "Electricity"), ("uber_tlc_hourly", "Uber"),
      ("m4_hourly", "M4"), ("wind_farms_hourly", "WindFarms")]
SEEDS = [0, 1, 2]
QSET = "q9"
C_TR, C_VA = "#D55E00", "#0072B2"                       # Okabe-Ito vermillion / blue
quantiles = QUANTILE_SETS[QSET]


def curves(tag):
    """(train[seeds,L], val[seeds,L], saved_val[seeds,L]) recomputed from frozen linear ckpts."""
    w = build_windows(tag)
    f_tr = _fslot_feats(tag, "train", w["X_train"], w["y_train"])
    f_va = _fslot_feats(tag, "val",   w["X_val"],   w["y_val"])
    tr, va, saved = [], [], []
    for s in SEEDS:
        fitted = load_ptid_ckpt(tag, QSET, s, device="cpu")
        # no collect flags -> predict returns just the {layer: scalar_loss} dict
        out_tr = predict_shared_forecast_probe(fitted, f_tr, w["Y_train_traj"],
                                               quantiles=quantiles, device="cpu")
        out_va = predict_shared_forecast_probe(fitted, f_va, w["Y_val_traj"],
                                               quantiles=quantiles, device="cpu")
        tr.append([float(out_tr[i]) for i in sorted(out_tr)])
        va.append([float(out_va[i]) for i in sorted(out_va)])
        sv = json.load(open(PTID_RUN_DIR / f"{tag}__{QSET}__seed{s}.json"))["val_loss_by_layer"]
        saved.append(sv)
        # persist the recomputed train curve so it never has to be recomputed again
        json.dump({"dataset": tag, "quantile_set": QSET, "run_seed": s, "readout": "fslot_linear",
                   "note": "post-hoc recompute of TRAIN loss via frozen ckpt + predict on train cache",
                   "train_loss_by_layer": tr[-1], "val_loss_by_layer_recomputed": va[-1],
                   "val_loss_by_layer_saved": sv},
                  open(PTID_RUN_DIR / f"{tag}__{QSET}__seed{s}__train_recompute.json", "w"), indent=2)
    return np.array(tr), np.array(va), np.array(saved)


fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
print("=== self-check: recomputed val vs saved val_loss_by_layer (max abs diff per dataset) ===")
for ax, (tag, short) in zip(axes.ravel(), DS):
    tr, va, saved = curves(tag)
    d = float(np.abs(va - saved).max())
    print(f"  {short:12} max|Δ| = {d:.2e}  {'PASS' if d < 1e-2 else 'WARN — investigate'}")
    x = np.arange(tr.shape[1]); trm, vam = tr.mean(0), va.mean(0)
    ax.fill_between(x, tr.min(0), tr.max(0), color=C_TR, alpha=.15, lw=0)
    ax.fill_between(x, va.min(0), va.max(0), color=C_VA, alpha=.15, lw=0)
    ax.plot(x, trm, "-o", color=C_TR, ms=4, lw=1.8, label="train (mean of 3 seeds, recomputed)")
    ax.plot(x, vam, "--s", color=C_VA, ms=4, lw=1.8, label="validation (mean of 3 seeds)")
    gap = vam - trm
    ax.set_title(f"{short}   (train–val gap: {gap.min():+.2f} … {gap.max():+.2f})",
                 fontsize=11, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=.25); ax.set_ylabel(f"{QSET} quantile loss")
    ax.legend(fontsize=8, loc="upper right", framealpha=.9)
fig.suptitle("Shared-LINEAR fslot readout — training vs validation loss by layer  "
             f"({QSET}; train recomputed from frozen checkpoints)", fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{HERE}/linear_train_vs_val_by_layer.png"
fig.savefig(out, dpi=140)
print("SAVED", out)
