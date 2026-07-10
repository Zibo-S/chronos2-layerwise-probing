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

from probing.config import NUM_LAYERS, LAST_LAYER, OUT_DIR, CACHE_DIR
from probing.id_data import ID_DATASETS, build_windows
from probing.extraction import extract_window_features
from probing.probes import ridge_regression_probe, binned_future_probe

N_BINS = 5
POOLINGS = ("content", "reg")
# UEA classification reference = the 6 non-saturated datasets used for Phase 0 conclusions.
UEA_REF = ["UWaveGestureLibrary", "EthanolConcentration", "SelfRegulationSCP1",
           "Handwriting", "LSST", "SelfRegulationSCP2"]

# UEA modality audit (see data/uea_domain_audit.md). Only genuine sensor/motion time-series
# are retained for the TS-restricted overlay; the rest are other modalities re-encoded as
# sequences (spectroscopy / EEG bio-signal / handwriting motion / astronomical light curves).
UEA_TS_APPROPRIATE = {"UWaveGestureLibrary"}
UEA_EXCLUDED_MODALITY = {
    "EthanolConcentration": "spectroscopy (axis = wavelength, not time)",
    "SelfRegulationSCP1": "EEG bio-signal",
    "SelfRegulationSCP2": "EEG bio-signal",
    "Handwriting": "handwriting / pen-trajectory motion",
    "LSST": "astronomical light curves",
}

# per-ID-dataset plot styling. solar_1h is DEMOTED (thin/low-alpha) because its
# context-normalized future-mean label is pathological on a strongly diurnal series
# (see paper/phase0_fixes.md "Known limitation"). M4 is labeled cross-series.
ID_STYLE = {
    "m4_hourly":                 {"color": "#d62728", "demoted": False, "label": "m4_hourly (cross-series)"},
    "monash_electricity_hourly": {"color": "#ff7f0e", "demoted": False, "label": "monash_electricity_hourly"},
    "solar_1h":                  {"color": "#8c564b", "demoted": True,  "label": "solar_1h (label pathology)"},
}


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


def _rel_dropoff(a):
    """(score - peak) / peak per layer: 0 at the peak layer, negative afterwards (peak > 0)."""
    a = np.asarray(a, float)
    peak = a.max()
    return (a - peak) / peak if peak > 0 else np.zeros_like(a)


def _peak_retention(a):
    """Return (peak_layer, peak_value, late_retention = score[L11] / peak)."""
    a = np.asarray(a, float)
    pk = int(a.argmax())
    peak = float(a[pk])
    ret = float(a[LAST_LAYER] / peak) if peak != 0 else float("nan")
    return pk, peak, ret


def _id_line(ax, tag, ys, pool, transform):
    """Plot one ID curve with the tag's styling (solar demoted, content solid / REG dashed)."""
    st = ID_STYLE[tag]
    if st["demoted"]:
        lw, alpha = 1.1, 0.45
    else:
        lw, alpha = (2.4 if pool == "content" else 1.6), 1.0
    ls = "-" if pool == "content" else "--"
    mk = "o" if pool == "content" else "s"
    lab = st["label"] + ("" if pool == "content" else " REG")
    ax.plot(np.arange(NUM_LAYERS), transform(ys), color=st["color"], lw=lw, alpha=alpha,
            ls=ls, marker=mk, ms=3.5, label=lab)


def make_overlay(id_results, uea_curves):
    xs = np.arange(NUM_LAYERS)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.5))

    # ---- Panel A: primary — binned-future accuracy (ID) vs UEA classification, own-max normalized ----
    for j, (name, acc) in enumerate(uea_curves.items()):
        axA.plot(xs, _norm_to_max(acc), color="steelblue", alpha=0.45, lw=1.2,
                 label="UEA classification (n=6)" if j == 0 else None)
    for tag, res in id_results.items():
        _id_line(axA, tag, res["poolings"]["content"]["binned_accuracy"], "content", _norm_to_max)
        _id_line(axA, tag, res["poolings"]["reg"]["binned_accuracy"], "reg", _norm_to_max)
    axA.set_title("PRIMARY: binned future-mean accuracy (ID) vs UEA classification\n"
                  "(each curve normalized to its own max)")
    axA.set_xlabel("encoder layer"); axA.set_ylabel("accuracy / max"); axA.set_xticks(xs)
    axA.grid(alpha=0.3); axA.legend(fontsize=7, ncol=2, loc="lower center")

    # ---- Panel B: secondary — ridge R^2 (raw; normalizing possibly-negative R^2 is misleading) ----
    axB.axhline(0.0, color="gray", ls=":", lw=1)
    for tag, res in id_results.items():
        _id_line(axB, tag, res["poolings"]["content"]["ridge_r2"], "content", lambda a: np.asarray(a, float))
        _id_line(axB, tag, res["poolings"]["reg"]["ridge_r2"], "reg", lambda a: np.asarray(a, float))
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


def make_tsonly(id_results, uea_curves):
    """Same as the primary overlay, but UEA classification curves from non-TS modalities are
    greyed out (excluded per the domain-restriction fix); genuine sensor-TS ones stay colored.
    Written to a SEPARATE file — the original overlay is left unchanged."""
    xs = np.arange(NUM_LAYERS)
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(15, 5.5))

    kept_lab_used = excl_lab_used = False
    for name, acc in uea_curves.items():
        if name in UEA_TS_APPROPRIATE:
            axA.plot(xs, _norm_to_max(acc), color="steelblue", alpha=0.9, lw=1.8,
                     label=f"UEA TS-appropriate: {name}" if not kept_lab_used else None)
            kept_lab_used = True
        else:
            axA.plot(xs, _norm_to_max(acc), color="0.6", alpha=0.35, lw=1.0,
                     label="UEA excluded modality (n=5)" if not excl_lab_used else None)
            excl_lab_used = True
    for tag, res in id_results.items():
        _id_line(axA, tag, res["poolings"]["content"]["binned_accuracy"], "content", _norm_to_max)
        _id_line(axA, tag, res["poolings"]["reg"]["binned_accuracy"], "reg", _norm_to_max)
    axA.set_title("TS-RESTRICTED: binned future-mean accuracy (ID) vs UEA classification\n"
                  "(non-TS modalities greyed; each curve normalized to its own max)")
    axA.set_xlabel("encoder layer"); axA.set_ylabel("accuracy / max"); axA.set_xticks(xs)
    axA.grid(alpha=0.3); axA.legend(fontsize=7, ncol=2, loc="lower center")

    axB.axhline(0.0, color="gray", ls=":", lw=1)
    for tag, res in id_results.items():
        _id_line(axB, tag, res["poolings"]["content"]["ridge_r2"], "content", lambda a: np.asarray(a, float))
        _id_line(axB, tag, res["poolings"]["reg"]["ridge_r2"], "reg", lambda a: np.asarray(a, float))
    axB.set_title("SECONDARY: ridge R² of normalized future mean (ID)\n(raw R², not normalized)")
    axB.set_xlabel("encoder layer"); axB.set_ylabel("test R²"); axB.set_xticks(xs)
    axB.grid(alpha=0.3); axB.legend(fontsize=7, loc="best")

    fig.suptitle("Phase 0 (TS-restricted): conclusions drawn only from genuine-TS UEA datasets",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "id_vs_classification_overlay_tsonly.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


def make_dropoff(id_results, uea_curves):
    """Relative drop from each curve's own peak: makes the late-layer LOSS readable
    (the normalize-to-max overlay compresses this away). Accuracy-scale curves only."""
    xs = np.arange(NUM_LAYERS)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for j, (name, acc) in enumerate(uea_curves.items()):
        ax.plot(xs, _rel_dropoff(acc), color="steelblue", alpha=0.45, lw=1.2,
                label="UEA classification (n=6)" if j == 0 else None)
    for tag, res in id_results.items():
        _id_line(ax, tag, res["poolings"]["content"]["binned_accuracy"], "content", _rel_dropoff)
        _id_line(ax, tag, res["poolings"]["reg"]["binned_accuracy"], "reg", _rel_dropoff)
    ax.axhline(0.0, color="gray", ls=":", lw=1)
    ax.set_title("Relative drop from peak, per layer:  (acc_layer - acc_peak) / acc_peak\n"
                 "(0 at each curve's own peak; more negative = larger post-peak loss)")
    ax.set_xlabel("encoder layer"); ax.set_ylabel("relative drop from peak"); ax.set_xticks(xs)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, ncol=2, loc="lower left")
    fig.tight_layout()
    out = OUT_DIR / "id_vs_classification_dropoff.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out}")


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

    # ---- late-layer retention = score[L11] / score[peak] (the quantity the tunnel
    #      comparison needs — how much each curve keeps after its own peak) ----
    retention = {
        "_basis": "retention_L11 = score[L11] / score[peak]. ID = binned-future accuracy "
                  "(content/reg pooling); UEA = classification accuracy (content pooling). "
                  "The ID-vs-transfer comparison should use the genuine-TS UEA subset "
                  "(ts_appropriate=true).",
        "id": {}, "uea_classification": {},
    }
    for tag, res in id_results.items():
        retention["id"][tag] = {}
        for pool in POOLINGS:
            ent = {}
            for metric in ("binned_accuracy", "ridge_r2"):
                pk, peak, ret = _peak_retention(res["poolings"][pool][metric])
                ent[metric] = {"peak_layer": pk, "peak_value": peak, "retention_L11": ret}
            retention["id"][tag][pool] = ent
    for name, acc in uea_curves.items():
        pk, peak, ret = _peak_retention(acc)
        retention["uea_classification"][name] = {
            "peak_layer": pk, "peak_value": peak, "retention_L11": ret,
            "basis": "classification_accuracy_content",
            "ts_appropriate": name in UEA_TS_APPROPRIATE,
            "modality": "genuine sensor time-series" if name in UEA_TS_APPROPRIATE
                        else UEA_EXCLUDED_MODALITY[name],
        }

    summary = {
        "config": {"C": 512, "H": 64, "n_bins": N_BINS, "poolings": list(POOLINGS),
                   "id_datasets": list(ID_DATASETS), "uea_reference": UEA_REF,
                   "binned_chance": 1.0 / N_BINS,
                   "solar_note": "solar_1h ridge R^2 is pathological (context-normalized "
                                 "future-mean label fails on strongly diurnal series); demoted "
                                 "in the overlay. See paper/phase0_fixes.md.",
                   "m4_note": "m4_hourly uses a cross-series split (series too short for a "
                              "within-series train/test span at C=512/H=64); comparable in "
                              "shape, not absolute level."},
        "id_datasets": id_results,
        "uea_classification_reference": {name: uea_curves[name] for name in UEA_REF},
        "late_layer_retention": retention,
    }
    with open(OUT_DIR / "id_probing_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  [saved] {OUT_DIR / 'id_probing_summary.json'}")

    make_overlay(id_results, uea_curves)
    make_tsonly(id_results, uea_curves)
    make_dropoff(id_results, uea_curves)

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
    # late-layer retention table = score[L11] / score[peak] — the tunnel-relevant number
    print(f"\n  late-layer retention = score[L11] / score[peak]  (content pooling):")
    print(f"    -- ID forecasting (binned-future accuracy) --")
    for tag, r in id_results.items():
        pk, peak, ret = _peak_retention(r["poolings"]["content"]["binned_accuracy"])
        print(f"    {tag:>28}:  peak L{pk}={peak:.3f}  ->  L11 retains {ret:.3f}")
    print(f"    -- UEA classification accuracy (genuine-TS subset) --")
    for name in UEA_REF:
        if name in UEA_TS_APPROPRIATE:
            pk, peak, ret = _peak_retention(uea_curves[name])
            print(f"    {name:>28}:  peak L{pk}={peak:.3f}  ->  L11 retains {ret:.3f}  [TS]")
    print(f"    -- UEA classification accuracy (excluded modalities) --")
    for name in UEA_REF:
        if name not in UEA_TS_APPROPRIATE:
            pk, peak, ret = _peak_retention(uea_curves[name])
            print(f"    {name:>28}:  peak L{pk}={peak:.3f}  ->  L11 retains {ret:.3f}  (excl.)")
    print(f"\n  binned chance = {1/N_BINS:.2f}")


if __name__ == "__main__":
    main()
