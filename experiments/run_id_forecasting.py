"""Phase 0 Stage 4 — in-distribution forecasting probes + ID-vs-classification overlay.

Builds windowed forecasting examples from Chronos-2-SEEN datasets, extracts per-layer
features (content + REG pooling), runs the two linear ID probes (binned-future accuracy —
the primary tunnel-signature readout — and ridge R^2, secondary), and overlays the ID
curves against the existing UEA classification curves.

Additive only: reads results/perdataset_summary.json for the classification reference,
writes results/id_probing_summary.json + results/id_vs_classification_overlay.png. Does
not touch the UEA cache or results.

Run:  python -m experiments.run_id_forecasting
"""

from __future__ import annotations

import gc
import json
import warnings

warnings.filterwarnings("ignore")  # quiet Ridge ill-conditioning + HF/aeon deprecation noise

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probing.config import NUM_LAYERS, OUT_DIR, CACHE_DIR
from probing.id_data import ID_DATASETS, build_windows
from probing.extraction import extract_window_features
from probing.probes import ridge_regression_probe, binned_future_probe

N_BINS = 5
POOLINGS = ("content", "reg")
# UEA classification reference = the 6 non-saturated datasets used for Phase 0 conclusions.
UEA_REF = ["UWaveGestureLibrary", "EthanolConcentration", "SelfRegulationSCP1",
           "Handwriting", "LSST", "SelfRegulationSCP2"]


def _bytes_per_split(n, pools=3):
    return n * NUM_LAYERS * 768 * 4 * pools


def run_dataset(tag):
    print(f"\n{'='*70}\n[{tag}] building windows\n{'='*70}")
    w = build_windows(tag)
    m = w["meta"]
    print(f"  split_mode={m['split_mode']}  n_series={m['n_series']} "
          f"(support within={m['n_series_supporting_within']})")
    print(f"  windows: train={m['n_train']} (of {m['n_train_windows_before_subsample']}), "
          f"test={m['n_test']} (of {m['n_test_windows_before_subsample']}), "
          f"skipped={m['n_skipped_windows']}")
    print(f"  label y range: train[{w['y_train'].min():+.3f},{w['y_train'].max():+.3f}] "
          f"test[{w['y_test'].min():+.3f},{w['y_test'].max():+.3f}]")

    result = {"meta": m, "poolings": {}}
    for pool in POOLINGS:
        f_tr, _ = extract_window_features(tag, "train", w["X_train"], w["y_train"], pooling=pool)
        f_te, _ = extract_window_features(tag, "test", w["X_test"], w["y_test"], pooling=pool)
        binned = binned_future_probe(f_tr, w["y_train"], f_te, w["y_test"], n_bins=N_BINS)
        ridge = ridge_regression_probe(f_tr, w["y_train"], f_te, w["y_test"])
        result["poolings"][pool] = {
            "binned_accuracy": [float(binned[i]) for i in range(NUM_LAYERS)],
            "ridge_r2": [float(ridge[i]) for i in range(NUM_LAYERS)],
        }
        acc = np.array(result["poolings"][pool]["binned_accuracy"])
        r2 = np.array(result["poolings"][pool]["ridge_r2"])
        print(f"  [{pool:>7}] binned acc: argmax L{int(acc.argmax())}={acc.max():.3f} "
              f"(chance={1/N_BINS:.2f}) | ridge R^2 range [{r2.min():+.3f},{r2.max():+.3f}]")
    # free the (large) downloaded arrays before the next dataset
    del w
    gc.collect()
    return result


def _norm_to_max(a):
    a = np.asarray(a, float)
    m = a.max()
    return a / m if m > 0 else a


def make_overlay(id_results, uea_curves):
    xs = np.arange(NUM_LAYERS)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.5))

    # ---- Panel A: primary — binned-future accuracy (ID) vs UEA classification, own-max normalized ----
    for j, (name, acc) in enumerate(uea_curves.items()):
        axA.plot(xs, _norm_to_max(acc), color="steelblue", alpha=0.45, lw=1.2,
                 label="UEA classification (n=6)" if j == 0 else None)
    warm = ["#d62728", "#ff7f0e", "#8c564b"]  # 3 ID datasets
    for k, (tag, res) in enumerate(id_results.items()):
        c = warm[k % len(warm)]
        axA.plot(xs, _norm_to_max(res["poolings"]["content"]["binned_accuracy"]),
                 color=c, lw=2.4, marker="o", ms=4, label=f"{tag} (content)")
        axA.plot(xs, _norm_to_max(res["poolings"]["reg"]["binned_accuracy"]),
                 color=c, lw=1.6, ls="--", marker="s", ms=3, label=f"{tag} (REG)")
    axA.set_title("PRIMARY: binned future-mean accuracy (ID) vs UEA classification\n"
                  "(each curve normalized to its own max)")
    axA.set_xlabel("encoder layer"); axA.set_ylabel("accuracy / max"); axA.set_xticks(xs)
    axA.grid(alpha=0.3); axA.legend(fontsize=7, ncol=2, loc="lower center")

    # ---- Panel B: secondary — ridge R^2 (raw; normalizing possibly-negative R^2 is misleading) ----
    axB.axhline(0.0, color="gray", ls=":", lw=1)
    for k, (tag, res) in enumerate(id_results.items()):
        c = warm[k % len(warm)]
        axB.plot(xs, res["poolings"]["content"]["ridge_r2"], color=c, lw=2.4, marker="o", ms=4,
                 label=f"{tag} (content)")
        axB.plot(xs, res["poolings"]["reg"]["ridge_r2"], color=c, lw=1.6, ls="--", marker="s", ms=3,
                 label=f"{tag} (REG)")
    axB.set_title("SECONDARY: ridge R² of normalized future mean (ID)\n(raw R², not normalized)")
    axB.set_xlabel("encoder layer"); axB.set_ylabel("test R²"); axB.set_xticks(xs)
    axB.grid(alpha=0.3); axB.legend(fontsize=7, loc="best")

    fig.suptitle("Phase 0: in-distribution forecasting probes vs UEA transfer classification",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "id_vs_classification_overlay.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  [saved] {out}")


def main():
    # classification reference curves (content pooling) from the committed UEA summary
    summ = json.load(open(OUT_DIR / "perdataset_summary.json"))["datasets"]
    uea_curves = {name: summ[name]["per_layer_accuracy"]["ID"] for name in UEA_REF}

    id_results = {}
    for tag in ID_DATASETS:
        id_results[tag] = run_dataset(tag)

    # estimated new cache footprint (FYI; disk is not a constraint)
    total_bytes = sum(_bytes_per_split(r["meta"]["n_train"]) + _bytes_per_split(r["meta"]["n_test"])
                      for r in id_results.values())
    print(f"\n  est. new IDF_ cache footprint: ~{total_bytes/1e9:.2f} GB "
          f"(3 poolings x 12 layers x n_windows x 768 x 4B)")

    summary = {
        "config": {"C": 512, "H": 64, "n_bins": N_BINS, "poolings": list(POOLINGS),
                   "id_datasets": list(ID_DATASETS), "uea_reference": UEA_REF,
                   "binned_chance": 1.0 / N_BINS},
        "id_datasets": id_results,
        "uea_classification_reference": {name: uea_curves[name] for name in UEA_REF},
    }
    with open(OUT_DIR / "id_probing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] {OUT_DIR / 'id_probing_summary.json'}")

    make_overlay(id_results, uea_curves)

    # ---- concise report (no interpretation) ----
    print(f"\n{'='*70}\nID PROBING SUMMARY (no interpretation)\n{'='*70}")
    print(f"{'dataset':>26} {'split':>13} {'n_tr':>6} {'n_te':>6}  "
          f"{'binned argmax/max (content)':>28}  {'ridge R^2 range (content)':>26}")
    for tag, r in id_results.items():
        m = r["meta"]; acc = np.array(r["poolings"]["content"]["binned_accuracy"])
        r2 = np.array(r["poolings"]["content"]["ridge_r2"])
        print(f"{tag:>26} {m['split_mode']:>13} {m['n_train']:>6} {m['n_test']:>6}  "
              f"{'L'+str(int(acc.argmax()))+'='+format(acc.max(),'.3f'):>28}  "
              f"{'['+format(r2.min(),'+.3f')+','+format(r2.max(),'+.3f')+']':>26}")
    print(f"\n  binned chance = {1/N_BINS:.2f}   |   PAUSE: review the overlay before any interpretation.")


if __name__ == "__main__":
    main()
