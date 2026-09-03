"""ONE-OFF recompute of the common-pooling comparison (Electricity, phase0_trio).

Authorized exception to the no-recompute rule, for fig_common_pooling.png only: the
backing quantile_loss_results.json for the phase0_trio set is unavailable locally.
Recomputes the four per-layer metrics (binned accuracy, ridge R^2, quantile loss, MASE)
for content vs REG pooling from the CACHED hidden states, with the frozen config and the
exact pipeline functions of experiments/run_id_forecasting.py (same windows/splits/seed,
q21 quantile set, epochs=300, wd_grid).

HARD RULES ENFORCED HERE:
  - No model forwards: every needed feature cache must already exist, else STOP before
    anything is computed. The native-Chronos-2 MASE baseline cache does not exist, so the
    native baseline is OMITTED (per-layer probe MASE only).
  - GATE before the expensive step: recomputed binned accuracy + ridge R^2 must match
    results/phase0_trio/id_probing_summary.json to numerical tolerance, else STOP.
  - Writes ONLY results/phase0_trio/quantile_loss/pooling_comparison/
    common_pooling_recompute.json (never touches id_probing_summary.json).

Run:  python -m probing.common_pooling_recompute
"""

from __future__ import annotations

import json
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import torch

from probing import config

config.set_dataset_set("phase0_trio")

from probing.config import CACHE_DIR, NUM_LAYERS, OUT_DIR
from probing.id_data import build_windows
from probing.extraction import extract_window_features
from probing.probes import (ridge_regression_probe, binned_future_probe, quantile_probe,
                            QUANTILE_SETS)
from experiments.run_id_forecasting import (_ctx_stats, _mase_denominator, M_SEASON,
                                            QUANTILE_EPOCHS, QUANTILE_WD_GRID)

TAG = "monash_electricity_hourly"
POOLINGS = ("content", "reg")
GATE_ATOL = 1e-9
OUT_JSON = OUT_DIR / "phase0_trio/quantile_loss/pooling_comparison/common_pooling_recompute.json"


def main():
    # ---- STOP rule 1: every hidden-state cache must exist (no model forwards) ----
    needed = [CACHE_DIR / f"IDF_{TAG}__{split}__clean__{pool}.npz"
              for split in ("train", "test") for pool in POOLINGS]
    missing = [p.name for p in needed if not p.exists()]
    if missing:
        sys.exit(f"STOP: hidden-state cache missing, not re-extracting: {missing}")
    print(f"[caches] all present: {[p.name for p in needed]}")

    w = build_windows(TAG)
    m = w["meta"]
    print(f"[windows] split={m['split_mode']} train={m['n_train']} test={m['n_test']}")

    summary = json.load(open(OUT_DIR / "phase0_trio/id_probing_summary.json"))
    ref = summary["id_datasets"][TAG]["poolings"]

    result = {"poolings": {}}
    diags = {}
    for pool in POOLINGS:
        f_tr, _ = extract_window_features(TAG, "train", w["X_train"], w["y_train"], pooling=pool)
        f_te, _ = extract_window_features(TAG, "test", w["X_test"], w["y_test"], pooling=pool)

        binned = binned_future_probe(f_tr, w["y_train"], f_te, w["y_test"], n_bins=5)
        ridge = ridge_regression_probe(f_tr, w["y_train"], f_te, w["y_test"])
        b = np.array([binned[i] for i in range(NUM_LAYERS)])
        r = np.array([ridge[i] for i in range(NUM_LAYERS)])

        # ---- GATE: must reproduce the committed curves before training anything ----
        db = np.abs(b - np.array(ref[pool]["binned_accuracy"])).max()
        dr = np.abs(r - np.array(ref[pool]["ridge_r2"])).max()
        print(f"[GATE {pool}] max|Δ binned_accuracy| = {db:.3e}   max|Δ ridge_r2| = {dr:.3e}")
        if db > GATE_ATOL or dr > GATE_ATOL:
            sys.exit(f"STOP: {pool} gate FAILED (atol {GATE_ATOL}): "
                     f"binned Δ={db:.3e}, ridge Δ={dr:.3e} — windows/features do not "
                     f"reproduce id_probing_summary.json; not proceeding to quantile probe")

        device = "mps" if torch.backends.mps.is_available() else None
        qloss, qdiag = quantile_probe(
            f_tr, w["Y_train_traj"], f_te, w["Y_test_traj"],
            quantiles=QUANTILE_SETS["q21"], epochs=QUANTILE_EPOCHS,
            wd_grid=QUANTILE_WD_GRID, device=device,
            collect_history=True, collect_test_median=True)
        diags[pool] = qdiag

        # per-layer probe MASE, exactly compute_mase's math minus the native baseline
        # (native forecast cache absent -> omitted rather than re-run the model)
        mu, s = _ctx_stats(w["X_test"], m["sigma_eps"])
        y_raw = mu[:, None] + s[:, None] * np.sinh(w["Y_test_traj"].astype(np.float64))
        d = np.maximum(_mase_denominator(w["X_test"]), 1e-8)[:, None]
        mase = []
        for i in range(NUM_LAYERS):
            zhat = qdiag["test_median"][i].astype(np.float64)
            yhat = mu[:, None] + s[:, None] * np.sinh(zhat)
            mase.append(float((np.abs(y_raw - yhat) / d).mean()))

        result["poolings"][pool] = {
            "binned_accuracy": b.tolist(),
            "ridge_r2": r.tolist(),
            "quantile_loss": [float(qloss[i]) for i in range(NUM_LAYERS)],
            "mean_pinball_loss": [qdiag["test_mean_pinball"][i] for i in range(NUM_LAYERS)],
            "mase_context": mase,
            "wd_selected": {str(i): float(qdiag["wd"][i]) for i in range(NUM_LAYERS)},
        }
        ql = np.array(result["poolings"][pool]["quantile_loss"])
        print(f"[{pool}] qloss argmin L{int(ql.argmin())}={ql.min():.4f}  "
              f"L0={ql[0]:.4f}  L11={ql[-1]:.4f}  | mase argmin L{int(np.argmin(mase))}"
              f"={min(mase):.4f}")

    result["provenance"] = {
        "note": "ONE-OFF recompute (authorized exception to the no-recompute rule) for "
                "fig_common_pooling.png only; backing quantile_loss_results.json for "
                "phase0_trio unavailable locally. Same pipeline functions as "
                "experiments/run_id_forecasting.py, cached hidden states only.",
        "dataset": TAG, "dataset_set": "phase0_trio", "poolings": list(POOLINGS),
        "quantile_set": "q21", "epochs": QUANTILE_EPOCHS,
        "wd_grid": list(QUANTILE_WD_GRID), "seasonal_m": M_SEASON,
        "gate": "binned_accuracy + ridge_r2 reproduced id_probing_summary.json "
                f"within atol {GATE_ATOL} for both poolings (see run log)",
        "native_mase": "OMITTED — native median-forecast cache absent; re-running the "
                       "model was forbidden",
        "windows": {k: m[k] for k in ("split_mode", "n_train", "n_test")},
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, indent=1))
    print(f"[saved] {OUT_JSON.relative_to(OUT_DIR.parent)}")


if __name__ == "__main__":
    main()
