"""
Per-dataset, per-layer ID-vs-OOD picture + tunnel-shape recheck.

Reuses the validated extraction and probe-fit logic from probe_pipeline.py:
  - a forward pre-hook capturing the embedded input as L0 + forward hooks on the 12 encoder.block
    modules as L1..L12, pooling=content (drop last 2 positions),
    per-channel-through-encoder then concatenate, Chronos-2 frozen / float32 / mps / no_grad
  - StandardScaler fit on CLEAN TRAIN only + LogisticRegression(max_iter=2000, random_state=0)

Cached features under ./features_cache/ are reused; only missing (dataset, shift) combos are
extracted. Outputs: 5 small-multiple PNGs + perdataset_summary.json, under results/uea/.

Run:  python -m experiments.run_perdataset
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# Reuse the validated machinery (do NOT reimplement extraction / probe fit).
from probing.config import NUM_LAYERS, SEED, UEA_OUT_DIR
from probing.extraction import extract_features, fit_layerwise_probes

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------- #
# Config
# ----------------------------------------------------------------------- #
DATASETS = [
    "UWaveGestureLibrary", "EthanolConcentration", "SelfRegulationSCP1", "Handwriting",
    "LSST", "SelfRegulationSCP2", "Epilepsy", "Cricket",
]

# Same shift definitions/params as Test 5 (probe_harden / probe_pipeline corruptions).
SHIFTS = [
    ("gauss",    {"kind": "gauss", "alpha": 0.25, "seed": SEED}),
    ("timewarp", {"kind": "timewarp", "factor": 1.2}),
    ("drift",    {"kind": "drift", "amplitude": 0.3}),
]
SHIFT_NAMES = [s[0] for s in SHIFTS]

MIDDLE_BAND = list(range(4, 10))  # a-priori "middle" layers (same blocks as before, +1 shifted)
LAST_LAYER = NUM_LAYERS - 1       # 12 (L0 embedding + L1..L12 block outputs)
BOOT_B = 2000

# panel colors
C_ID = "C0"
C_SHIFT = {"gauss": "C3", "timewarp": "C4", "drift": "C2"}


# ----------------------------------------------------------------------- #
# Stats helpers (correctness-vector based; no refit, no re-extraction)
# ----------------------------------------------------------------------- #
def score_correctness(probes, features, y_true):
    y_true = np.asarray(y_true)
    return {i: (probes[i]["clf"].predict(probes[i]["scaler"].transform(features[i])) == y_true).astype(np.float32)
            for i in range(NUM_LAYERS)}


def bootstrap_ci(correct, rng, B=BOOT_B):
    c = np.asarray(correct, np.float64); n = c.size
    means = c[rng.integers(0, n, size=(B, n))].mean(1)
    return float(c.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff_ci(a, b, rng, B=BOOT_B):
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64); n = a.size
    idx = rng.integers(0, n, size=(B, n))
    d = a[idx].mean(1) - b[idx].mean(1)
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def excl0(lo, hi):
    return bool((lo > 0) or (hi < 0))


def band_correct(correct):
    return np.stack([correct[L] for L in MIDDLE_BAND], 0).mean(0)


def accs_with_ci(correct, rng):
    accs = np.zeros(NUM_LAYERS); lo = np.zeros(NUM_LAYERS); hi = np.zeros(NUM_LAYERS)
    for i in range(NUM_LAYERS):
        accs[i], lo[i], hi[i] = bootstrap_ci(correct[i], rng)
    return accs, lo, hi


def amplification_band_ci(correct_id, correct_ood, rng, B=BOOT_B):
    """(band-last)_OOD - (band-last)_ID, same resampled indices across all four vectors."""
    n = correct_id[LAST_LAYER].size
    idx = rng.integers(0, n, size=(B, n))
    bid, bood = band_correct(correct_id), band_correct(correct_ood)
    g_id = bid[idx].mean(1) - correct_id[LAST_LAYER][idx].mean(1)
    g_ood = bood[idx].mean(1) - correct_ood[LAST_LAYER][idx].mean(1)
    amp = g_ood - g_id
    pt = (bood.mean() - correct_ood[LAST_LAYER].mean()) - (bid.mean() - correct_id[LAST_LAYER].mean())
    return float(pt), float(np.percentile(amp, 2.5)), float(np.percentile(amp, 97.5))


def saturated_flag(accs):
    return bool((np.asarray(accs) >= 0.95).sum() >= 3)


# ----------------------------------------------------------------------- #
# Per-dataset computation
# ----------------------------------------------------------------------- #
def process_dataset(dataset, rng):
    print(f"\n=== {dataset} ===")
    f_tr, y_tr = extract_features(dataset, "train", pooling="content")
    f_te, y_te = extract_features(dataset, "test", pooling="content")

    probes = fit_layerwise_probes(f_tr, y_tr)
    correct_id = score_correctness(probes, f_te, y_te)
    id_accs, id_lo, id_hi = accs_with_ci(correct_id, rng)

    classes = np.unique(y_tr)
    n_classes = len(classes)
    chance = 1.0 / n_classes
    saturated = saturated_flag(id_accs)
    argmax = int(np.argmax(id_accs))

    # ID late_drop_band
    ld_pt, ld_lo, ld_hi = paired_diff_ci(band_correct(correct_id), correct_id[LAST_LAYER], rng)

    ood = {}        # shift -> dict(accs, lo, hi)
    amp = {}        # shift -> (pt, lo, hi, excl0)
    for sname, corr in SHIFTS:
        f_o, y_o = extract_features(dataset, "test", corruption=corr, pooling="content")
        assert np.array_equal(y_o, y_te), f"labels changed under {sname}"
        correct_ood = score_correctness(probes, f_o, y_te)
        a, l, h = accs_with_ci(correct_ood, rng)
        ood[sname] = {"accs": a, "lo": l, "hi": h}
        ap, al, ah = amplification_band_ci(correct_id, correct_ood, rng)
        amp[sname] = (ap, al, ah, excl0(al, ah))

    return {
        "dataset": dataset,
        "n_test": int(len(y_te)),
        "n_classes": int(n_classes),
        "chance": float(chance),
        "saturated": saturated,
        "argmax_layer": argmax,
        "id_accs": id_accs, "id_lo": id_lo, "id_hi": id_hi,
        "late_drop_band": (ld_pt, ld_lo, ld_hi, excl0(ld_lo, ld_hi)),
        "ood": ood,
        "amplification": amp,
    }


# ----------------------------------------------------------------------- #
# Plotting: 2x4 small multiples
# ----------------------------------------------------------------------- #
def _panel_ylim(series_list, chance):
    vals = np.concatenate([np.asarray(s) for s in series_list])
    lo = min(float(vals.min()), chance)
    hi = max(float(vals.max()), chance)
    pad = max(0.02, 0.06 * (hi - lo))
    return (max(0.0, lo - pad), min(1.02, hi + pad))


def make_grid(fig_path, suptitle, curves_for, draw_ci, draw_argmax, legend_handles):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    xs = np.arange(NUM_LAYERS)
    for idx, ds in enumerate(DATASETS):
        ax = axes.flat[idx]
        r = results.get(ds)
        if r is None:
            ax.text(0.5, 0.5, f"{ds}\n(failed to load)", ha="center", va="center", fontsize=11)
            ax.set_xticks([]); ax.set_yticks([])
            continue
        curves = curves_for(r)        # list of (label, accs, lo|None, hi|None, color)
        all_series = []
        for label, accs, lo, hi, color in curves:
            if draw_ci and lo is not None:
                ax.fill_between(xs, lo, hi, alpha=0.18, color=color, linewidth=0)
                all_series += [lo, hi]
            else:
                all_series.append(accs)
            ax.plot(xs, accs, marker="o", ms=4, color=color, label=label)
        ax.axhline(r["chance"], ls="--", color="gray", lw=1)
        if draw_argmax:
            j = r["argmax_layer"]; a0 = curves[0][1]
            ax.scatter([j], [a0[j]], marker="*", s=200, color="gold", edgecolor="black", zorder=5)
        title = ds + ("  (saturated)" if r["saturated"] else "")
        ax.set_title(title, fontsize=11)
        ax.set_xticks(xs); ax.set_xlabel("layer"); ax.set_ylabel("test acc")
        ax.set_ylim(*_panel_ylim(all_series, r["chance"]))
        ax.grid(alpha=0.3)
        # per-panel chance annotation
        ax.text(0.02, 0.03, f"chance={r['chance']:.3f}  n_test={r['n_test']}  K={r['n_classes']}",
                transform=ax.transAxes, fontsize=8, va="bottom",
                bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.7))
    fig.suptitle(suptitle, fontsize=15, y=0.99)
    fig.legend(handles=legend_handles, loc="lower center", ncol=len(legend_handles),
               fontsize=11, frameon=True, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    fig.savefig(UEA_OUT_DIR / fig_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {fig_path}")


# ----------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------- #
results = {}

def main():
    global results
    np.random.seed(SEED)
    rng = np.random.default_rng(SEED)

    for ds in DATASETS:
        try:
            results[ds] = process_dataset(ds, rng)
        except Exception as e:
            print(f"  !! {ds} FAILED: {type(e).__name__}: {e}")
            results[ds] = None

    # ---------------- figures ----------------
    chance_proxy = Line2D([0], [0], ls="--", color="gray", lw=1, label="chance (1/K)")
    argmax_proxy = Line2D([0], [0], ls="none", marker="*", ms=14, color="gold",
                          markeredgecolor="black", label="argmax layer")
    id_proxy = Line2D([0], [0], marker="o", color=C_ID, label="ID (clean test)")

    # 1) ID-only tunnel grid
    make_grid("fig_grid_id_tunnel.png",
              "Per-dataset ID layer-wise probe accuracy (tunnel shape?)  -  95% bootstrap CI",
              curves_for=lambda r: [("ID", r["id_accs"], r["id_lo"], r["id_hi"], C_ID)],
              draw_ci=True, draw_argmax=True,
              legend_handles=[id_proxy, argmax_proxy, chance_proxy])

    # 2-4) ID vs each single OOD
    for sname in SHIFT_NAMES:
        shift_proxy = Line2D([0], [0], marker="o", color=C_SHIFT[sname], label=f"OOD ({sname})")
        make_grid(f"fig_grid_idood_{sname}.png",
                  f"Per-dataset ID vs {sname}-OOD (probe/scaler frozen on clean train)  -  95% CI",
                  curves_for=(lambda s: (lambda r: [
                      ("ID", r["id_accs"], r["id_lo"], r["id_hi"], C_ID),
                      (f"OOD-{s}", r["ood"][s]["accs"], r["ood"][s]["lo"], r["ood"][s]["hi"], C_SHIFT[s]),
                  ]))(sname),
                  draw_ci=True, draw_argmax=False,
                  legend_handles=[id_proxy, shift_proxy, chance_proxy])

    # 5) ID + all 3 OOD overview (no CI)
    all_handles = [id_proxy] + [Line2D([0], [0], marker="o", color=C_SHIFT[s], label=s) for s in SHIFT_NAMES] + [chance_proxy]
    make_grid("fig_grid_idood_all.png",
              "Per-dataset ID + all OOD shifts (overview, no CI bands)",
              curves_for=lambda r: [("ID", r["id_accs"], None, None, C_ID)] +
                                   [(s, r["ood"][s]["accs"], None, None, C_SHIFT[s]) for s in SHIFT_NAMES],
              draw_ci=False, draw_argmax=False,
              legend_handles=all_handles)

    # ---------------- verdict table + json ----------------
    print("\n" + "=" * 132)
    print("VERDICT TABLE")
    print("=" * 132)
    hdr = (f"{'dataset':<22}{'n_test':>7}{'K':>4}{'chance':>8}{'sat?':>6}{'argmax':>7}"
           f"{'ID late_drop_band [lo,hi] x0':>34}"
           f"{'amp_gauss x0':>20}{'amp_timewarp x0':>20}{'amp_drift x0':>20}")
    print(hdr)
    print("-" * len(hdr))

    summary = {}
    for ds in DATASETS:
        r = results.get(ds)
        if r is None:
            print(f"{ds:<22}{'(failed to load)':>20}")
            summary[ds] = None
            continue
        ld = r["late_drop_band"]
        def amp_str(s):
            ap, al, ah, e = r["amplification"][s]
            return f"{ap:+.3f}[{al:+.3f},{ah:+.3f}]{'*' if e else ' '}"
        ld_str = f"{ld[0]:+.3f}[{ld[1]:+.3f},{ld[2]:+.3f}]{'*' if ld[3] else ' '}"
        print(f"{ds:<22}{r['n_test']:>7}{r['n_classes']:>4}{r['chance']:>8.3f}"
              f"{('YES' if r['saturated'] else 'no'):>6}{('L'+str(r['argmax_layer'])):>7}"
              f"{ld_str:>34}{amp_str('gauss'):>20}{amp_str('timewarp'):>20}{amp_str('drift'):>20}")

        summary[ds] = {
            "n_test": r["n_test"], "n_classes": r["n_classes"], "chance": r["chance"],
            "saturated": r["saturated"], "argmax_layer": r["argmax_layer"],
            "id_late_drop_band": {"point": ld[0], "lo": ld[1], "hi": ld[2], "excludes_0": ld[3]},
            "amplification": {
                s: {"point": r["amplification"][s][0], "lo": r["amplification"][s][1],
                    "hi": r["amplification"][s][2], "excludes_0": r["amplification"][s][3]}
                for s in SHIFT_NAMES
            },
            "per_layer_accuracy": {
                "ID": [float(v) for v in r["id_accs"]],
                **{s: [float(v) for v in r["ood"][s]["accs"]] for s in SHIFT_NAMES},
            },
        }
    print("\n(* = 95% CI excludes 0)")

    print("\nSATURATION FLAGS:")
    for ds in DATASETS:
        r = results.get(ds)
        if r is not None:
            n_ge = int((np.asarray(r["id_accs"]) >= 0.95).sum())
            print(f"  {ds:<22} saturated={'YES' if r['saturated'] else 'no '}  ({n_ge}/{NUM_LAYERS} layers >= 0.95)")

    out = {
        "config": {
            "datasets": DATASETS, "shifts": {n: c for n, c in SHIFTS},
            "middle_band": MIDDLE_BAND, "last_layer": LAST_LAYER,
            "bootstrap_B": BOOT_B, "seed": SEED, "pooling": "content",
        },
        "datasets": summary,
    }
    with open(UEA_OUT_DIR / "perdataset_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\n  [saved] perdataset_summary.json")


if __name__ == "__main__":
    main()
