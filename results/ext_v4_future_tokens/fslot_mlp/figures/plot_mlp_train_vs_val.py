#!/usr/bin/env python
"""Training vs validation loss by layer for the native-MLP fslot readout (q9).

Reads the per-seed fit histories written by run_ptood_probing_ftok --probe-family
native_mlp (results/ext_v4_future_tokens/fslot_mlp/ptid_runs/*__history.json) and the
per-dataset MLP tunnels (fslot_mlp/tunnels/*), and draws a 2x2 panel (one per PT-ID
dataset) of mean train vs mean validation q9 loss across the 3 probe seeds, with the
seed min-max range shaded and the val-selected tunnel entrance marked.

Login-node safe: reads only small JSONs, no feature caches, no model. Run from the repo
root with the project venv:  MPLBACKEND=Agg .venv/bin/python <this file>
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.dirname(HERE)                         # .../fslot_mlp
DS = [("monash_electricity_hourly", "Electricity"), ("uber_tlc_hourly", "Uber"),
      ("m4_hourly", "M4"), ("wind_farms_hourly", "WindFarms")]
SEEDS = [0, 1, 2]
C_TR, C_VA = "#D55E00", "#0072B2"                 # Okabe-Ito vermillion / blue (colorblind-safe)


def load(tag):
    tr, va = [], []
    labels = conv = None
    for s in SEEDS:
        h = json.load(open(f"{R}/ptid_runs/{tag}__q9__seed{s}__history.json"))
        bl = h["by_layer"]; ks = sorted(bl, key=int)
        tr.append([bl[k]["final_train_loss"] for k in ks])
        va.append([bl[k]["final_val_loss"] for k in ks])
        if labels is None:
            labels = [bl[k]["layer_label"] for k in ks]
            conv = [bl[k]["converged"] for k in ks]
    t = json.load(open(f"{R}/tunnels/{tag}__fslot_mlp__q9__runs0-1-2.json"))
    return np.array(tr), np.array(va), labels, conv, t["l_start"]


fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
for ax, (tag, short) in zip(axes.ravel(), DS):
    tr, va, labels, conv, ls = load(tag)
    x = np.arange(len(labels))
    trm, vam = tr.mean(0), va.mean(0)
    ax.fill_between(x, tr.min(0), tr.max(0), color=C_TR, alpha=.15, lw=0)
    ax.fill_between(x, va.min(0), va.max(0), color=C_VA, alpha=.15, lw=0)
    ax.plot(x, trm, "-o", color=C_TR, ms=4, lw=1.8, label="train (mean of 3 seeds)")
    ax.plot(x, vam, "--s", color=C_VA, ms=4, lw=1.8, label="validation (mean of 3 seeds)")
    ax.axvline(ls, color="0.35", ls=":", lw=1.3)
    ax.annotate(f"val-selected\ntunnel entrance\n{labels[ls]}", (ls, ax.get_ylim()[1]),
                xytext=(2, -2), textcoords="offset points", va="top", fontsize=7.5, color="0.3")
    gap = vam - trm
    ax.set_title(f"{short}   (train–val gap: {gap.min():+.2f} … {gap.max():+.2f})",
                 fontsize=11, weight="bold")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.grid(alpha=.25); ax.set_ylabel("q9 quantile loss")
    ax.legend(fontsize=8, loc="upper right", framealpha=.9)
fig.suptitle("Native-MLP fslot readout — training vs validation loss by layer  "
             "(q9, 2.92M-param head, 300 ep, dropout 0.1)", fontsize=12.5, weight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = f"{HERE}/mlp_train_vs_val_by_layer.png"
fig.savefig(out, dpi=140)
print("SAVED", out)
