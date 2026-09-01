"""BOOM-as-source frozen transfer: the missing 5th transfer source for the q1 appendix figure.

The committed transfer pipeline (`run_fslot_transfer`) only ever fits source probes on the four
PT-ID datasets (electricity / uber / m4 / wind_farms) — BOOM appears solely as a *target*. This
driver adds the symmetric cell that the appendix combined figure is missing: fit ONE shared-forecast
linear probe on BOOM's own train split (pretrained backbone, wd selected on BOOM validation, 3 probe
seeds), FREEZE it, and score it unchanged on all 7 evaluation targets. Same estimand as the other
frozen-transfer blocks (blocks 1..n of make_id_paper_figures.make_multi_source_delta_figure), NOT the
"fresh per-target probe" estimand of the old BOOM block it replaces.

Nothing is re-extracted: BOOM's pretrained forecast-slot features are already cached
(IDF_boom_hourly__ood__{train,val,test}_rolling__clean__K4_H64.npz), as are every target's test-split
fslot features. The heavy part is the probe FIT + the 7 predict passes over those warm caches, so this
is COMPUTE work (probe fitting + rolling-window builds for sg_carpark/boom) — run it on a compute node
(salloc/sbatch), never the login node. No model load, no GPU strictly required (a GPU only speeds the
linear-head AdamW fits).

Reuses verbatim, so the numbers are bit-comparable to the committed sources:
  * fit  = probes.fit_shared_forecast_probe_explicit_val   (wd grid on BOOM-val, init_seed = run seed)
  * score = probes.predict_shared_forecast_probe            (frozen; per-window loss for the bootstrap)
  * windows = id_data.build_ood_rolling_windows (BOOM + the 3 PT-OOD targets) / build_windows (4 PT-ID)
  * features = run_ptood_probing_ftok._fslot_feats          (14 points: Emb, L1..L12, L12+LN)
  * MASE / two-axis labels = run_fslot_transfer._fslot_mase / _quadrant / _relative_regret

Outputs (isolated namespace — the committed 28-cell 4×4/pt_ood data is never touched):
  results/ext_v4_future_tokens/<qset>/boom_source/
      bootstrap_inputs/boom_hourly__to__<target>__<qset>__seed{0,1,2}.npz   (window_loss, series_test)
      tables/transfer_summary__boom_src__<qset>.{csv,json}                   (one row per target)
      checkpoints/boom_hourly__fslot__C512_H64__<qset>__<proto>__seed{s}/    (frozen source probe)

Run (USER submits SLURM — [[submit-slurm-jobs-self]]; compute node, warm caches):
    python -m experiments.run_boom_source_transfer --quantile-set q1
then regenerate the figure on the login node:
    python -m experiments.make_id_paper_figures --figure transfer
"""

from __future__ import annotations

import argparse
import csv
import gc
import json

import numpy as np
import torch

from probing import config
from probing.config import NUM_LAYERS, SEED
from probing.id_data import build_ood_rolling_windows, build_windows
from probing.probes import (QUANTILE_SETS, fit_shared_forecast_probe_explicit_val,
                            predict_shared_forecast_probe, validate_quantiles)
from probing.tunnel import (PT_ID_TAGS, PT_OOD_TAGS, TUNNEL_TOL, val_curve_from_selection)
# metric + two-axis-label helpers, reused so a BOOM-source row matches the committed schema exactly.
# (importing these modules only defines functions — no main() runs on import.)
from experiments.run_fslot_transfer import _fslot_mase, _pt_label, _probe_label, _quadrant, _relative_regret
from experiments.run_ptood_probing_ftok import (C, H, K, LAYER_LABELS, OUT_ROOT, PROBE_PROTOCOL_VERSION,
                                                QUANTILE_EPOCHS, READOUT, RUN_SEEDS, SHORT, WD_GRID,
                                                _fslot_feats, _save_ckpt)

SOURCE = "boom_hourly"
N_POINTS = NUM_LAYERS + 1                       # 14 fslot readout points (Emb, L1..L12, L12+LN)
REF_LABEL = LAYER_LABELS[-1]                    # "L12+LN" — final reference
# every combined-figure block scores the SAME 7 targets; the build path differs only by pt-status group
PT_ID_TARGETS = list(PT_ID_TAGS)               # elec, uber, m4, wind -> build_windows / "test"
PT_OOD_TARGETS = list(PT_OOD_TAGS)             # sg_carpark, coastal_ts, boom_hourly -> rolling/"test_rolling"
TARGETS = PT_ID_TARGETS + PT_OOD_TARGETS       # BOOM is its own PT-OOD target (the Probe-ID diagonal cell)


def _derive_dirs(qset):
    global OUT_DIR, BOOT_IN_DIR, TAB_DIR, CKPT_DIR
    OUT_DIR = OUT_ROOT / qset / "boom_source"
    BOOT_IN_DIR = OUT_DIR / "bootstrap_inputs"
    TAB_DIR = OUT_DIR / "tables"
    CKPT_DIR = OUT_DIR / "checkpoints"
    for d in (OUT_DIR, BOOT_IN_DIR, TAB_DIR, CKPT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _sustained_entrance(val_curve, tol=TUNNEL_TOL):
    """Sustained-plateau tunnel entrance on BOOM's mean validation curve (same rule as tunnel_start):
    earliest l with val[j] <= (1+tol)*val[last] for ALL j >= l. Recorded for schema parity; the
    combined figure does not use it."""
    v = np.asarray(val_curve, dtype=np.float64)
    ref = v[-1]
    ok = [l for l in range(len(v)) if np.all(v[l:] <= (1.0 + tol) * ref)]
    return ok[0] if ok else len(v) - 1


# --------------------------------------------------------------------------- #
# fit the BOOM source probe (once per run seed) on the pretrained backbone
# --------------------------------------------------------------------------- #
def fit_boom_source(qset, quantiles, device):
    """Fit + freeze the shared-forecast probe on BOOM train (wd on BOOM val) for each run seed.
    Returns {seed: fitted} and the source-VALIDATION-selected layer ℓ_s (argmin of the mean val
    curve; earliest-layer tie-break via np.argmin) — the exact ℓ_s convention run_fslot_transfer
    reads from a source tunnel, computed here directly from the fit."""
    w = build_ood_rolling_windows(SOURCE, C=C, H=H, seed=SEED)
    m = w["meta"]
    if m["n_train"] == 0 or m["n_val"] == 0:
        raise RuntimeError(f"{SOURCE}: empty train/val split (train {m['n_train']} / val {m['n_val']})")
    print(f"[fit BOOM source — {READOUT}] windows: train {m['n_train']} / val {m['n_val']} / "
          f"test {m['n_test']}  ({m['n_test_clusters']} {m['cluster_unit']} clusters)")
    f_tr = _fslot_feats(SOURCE, "train_rolling", w["X_train"], w["y_train"])
    f_va = _fslot_feats(SOURCE, "val_rolling", w["X_val"], w["y_val"])

    fitted_by_seed, val_curves = {}, []
    for seed in RUN_SEEDS:
        print(f"  run seed {seed} ({qset})")
        fitted = fit_shared_forecast_probe_explicit_val(
            f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"], quantiles=quantiles,
            epochs=QUANTILE_EPOCHS, wd_grid=WD_GRID, device=device, init_seed=seed)
        _save_ckpt(CKPT_DIR / f"{SOURCE}__{READOUT}__C{C}_H{H}__{qset}__{PROBE_PROTOCOL_VERSION}__seed{seed}",
                   fitted)
        fitted_by_seed[seed] = fitted
        val_curves.append(val_curve_from_selection(
            {i: fitted[i]["selection"] for i in sorted(fitted)}, num_layers=len(fitted)))

    mean_val = np.mean(val_curves, axis=0)
    ell = int(np.argmin(mean_val))
    print(f"  source-validation-selected layer ℓ_s = {LAYER_LABELS[ell]} "
          f"(sustained entrance {LAYER_LABELS[_sustained_entrance(mean_val)]})")
    del f_tr, f_va
    gc.collect()
    return fitted_by_seed, ell, mean_val


# --------------------------------------------------------------------------- #
# score the frozen BOOM probe on one target's test split
# --------------------------------------------------------------------------- #
def _target_windows_and_feats(target):
    """Target test windows + 14-point fslot test features (cache HIT). PT-OOD targets use the rolling
    builder + 'test_rolling' cache name; the 4 PT-ID targets use build_windows + 'test'."""
    if target in PT_OOD_TARGETS:
        w = build_ood_rolling_windows(target, C=C, H=H, seed=SEED)
        split = "test_rolling"
    else:
        w = build_windows(target)
        split = "test"
    if int(w["meta"]["n_test"]) == 0:
        raise RuntimeError(f"{target}: empty test split — check the loader / run_ood_screen")
    feats = _fslot_feats(target, split, w["X_test"], w["y_test"])
    return w, feats


def eval_target(target, w, feats, fitted_by_seed, qset, quantiles, device):
    """Score the frozen BOOM probe on the target's test windows for all 3 run seeds. Writes per-seed
    per-window loss + series ids (the cluster-bootstrap inputs the figure reads) and returns the
    seed-mean 14-point quantile-loss + MASE curves."""
    ql_seeds, mase_seeds, sid = [], [], None
    for seed in RUN_SEEDS:
        out, diag = predict_shared_forecast_probe(
            fitted_by_seed[seed], feats, w["Y_test_traj"], quantiles=quantiles, device=device,
            collect_test_median=True, collect_test_window_loss=True)
        layers = sorted(out)
        wl = np.stack([diag["test_window_loss"][i] for i in layers]).astype(np.float64)     # (14, n)
        sid = np.asarray(w["series_test"], np.int64)
        np.savez(BOOT_IN_DIR / f"{SOURCE}__to__{target}__{qset}__seed{seed}.npz",
                 window_loss=wl, series_test=sid)
        ql_seeds.append([float(out[i]) for i in layers])
        mase = _fslot_mase(target, w, diag)
        mase_seeds.append([float(mase[i]) for i in layers])
    return {"ql": np.mean(ql_seeds, axis=0), "mase": np.mean(mase_seeds, axis=0),
            "n_test": int(len(sid)), "n_clusters": int(np.unique(sid).size)}


def _write_records(cells, ell, mean_val, qset):
    """One summary row per target, mirroring run_fslot_transfer._write_records so a BOOM-source row is
    a drop-in sibling of the committed 4×4 / pt_ood rows (l_start_sustained + val_selected_layer +
    quadrant + selected/reference losses)."""
    ls_sustained = int(_sustained_entrance(mean_val))
    summary, curves = [], {}
    for tgt in TARGETS:
        ql, mase = cells[tgt]["ql"], cells[tgt]["mase"]
        base = {
            "pt_status": _pt_label(tgt), "probe_status": _probe_label(SOURCE, tgt),
            "quadrant": _quadrant(SOURCE, tgt), "source_dataset": SOURCE, "target_dataset": tgt,
            "probe_fitted_on": SOURCE, "wd_selected_on": f"{SOURCE}:validation",
            "tunnel_defined_on": f"{SOURCE}:validation", "l_start_sustained": ls_sustained,
            "val_selected_layer": int(ell), "final_reference": REF_LABEL, "quantile_set": qset,
            "readout": READOUT, "probe_family": "shared_linear",
            "probe_protocol_version": PROBE_PROTOCOL_VERSION,
        }
        summary.append({**base, "val_selected_layer_label": LAYER_LABELS[int(ell)],
                        "quantile_loss_at_selected": round(float(ql[ell]), 6),
                        "quantile_loss_at_reference": round(float(ql[-1]), 6),
                        "mase_at_selected": round(float(mase[ell]), 6),
                        "mase_at_reference": round(float(mase[-1]), 6),
                        "transfer_gap": None,                        # no target diagonal in this pass
                        "relative_regret_at_selected_supp": round(_relative_regret(ql, ell), 6),
                        "n_test_windows": cells[tgt]["n_test"],
                        "n_test_clusters": cells[tgt]["n_clusters"]})
        curves[f"{SOURCE}__to__{tgt}"] = {"quantile_loss": [float(x) for x in ql],
                                          "mase": [float(x) for x in mase],
                                          "layer_labels": LAYER_LABELS}
    stem = TAB_DIR / f"transfer_summary__boom_src__{qset}"
    with open(f"{stem}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(summary[0]))
        wr.writeheader()
        wr.writerows(summary)
    json.dump({"source": SOURCE, "final_reference": REF_LABEL, "run_seeds": list(RUN_SEEDS),
               "val_selected_layer": int(ell), "cells": curves}, open(f"{stem}.json", "w"), indent=2)
    print(f"  [saved] {len(summary)} BOOM-source rows -> {stem}.csv")


# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quantile-set", default="q1", choices=sorted(QUANTILE_SETS))
    return p.parse_args(argv)


def main():
    args = _parse_args()
    config.set_dataset_set("extended_v3_rolling")       # roster + rolling windows + cache namespace
    _derive_dirs(args.quantile_set)
    quantiles = validate_quantiles(QUANTILE_SETS[args.quantile_set])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[run_boom_source_transfer] source={SOURCE}  readout={READOUT}  qset={args.quantile_set}  "
          f"K={K}  device={device}  targets={len(TARGETS)}  out={OUT_DIR.relative_to(OUT_ROOT.parent)}")

    fitted_by_seed, ell, mean_val = fit_boom_source(args.quantile_set, quantiles, device)
    cells = {}
    for tgt in TARGETS:
        w, feats = _target_windows_and_feats(tgt)
        cells[tgt] = eval_target(tgt, w, feats, fitted_by_seed, args.quantile_set, quantiles, device)
        print(f"  {SHORT.get(tgt, tgt):<12} n_test={cells[tgt]['n_test']:<5} "
              f"ql(ℓ_s={LAYER_LABELS[ell]})={cells[tgt]['ql'][ell]:.4f}  ql({REF_LABEL})={cells[tgt]['ql'][-1]:.4f}")
        del w, feats
        gc.collect()
    _write_records(cells, ell, mean_val, args.quantile_set)


if __name__ == "__main__":
    main()
