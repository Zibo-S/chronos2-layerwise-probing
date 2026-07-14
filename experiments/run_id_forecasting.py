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
import math
import warnings

warnings.filterwarnings("ignore")  # quiet Ridge ill-conditioning + HF/aeon deprecation noise

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probing.config import NUM_LAYERS, LAST_LAYER, OUT_DIR, QUANT_DIR, CACHE_DIR, OUTPUT_PATCH_SIZE
from probing.id_data import ID_DATASETS, build_windows
from probing.extraction import extract_window_features, extract_kout_features
from probing.probes import (ridge_regression_probe, binned_future_probe, quantile_probe,
                            shared_forecast_token_probe)

N_BINS = 5
# Chronos-2-native quantile probe (forecasting currency). wd_grid=None -> fixed weight_decay
# (fast, for iteration); set to (1e-5, 1e-4, 1e-3, 1e-2, 1e-1) for final val-selected numbers.
QUANTILE_EPOCHS = 300
QUANTILE_WD_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
POOLINGS = ("content", "reg")
# Chronos-aligned shared forecast-token probe: run the encoder with K = ceil(H / OUTPUT_PATCH_SIZE)
# forecast slots (Chronos-2's own rule, pipeline.get_num_output_patches) and compare pooled vs
# shared readouts, all derived from the SAME K-slot pass (a controlled comparison — attention is
# non-causal, so K changes the context/REG states too). K is derived per-dataset from the actual
# label horizon, NOT hardcoded (H=64, output_patch_size=16 -> K=4 for the current datasets); for
# H not a multiple of 16 the shared probe predicts K whole patches and trims to H (native-style).
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

# per-ID-dataset plot styling. All four are long-series hourly datasets on the within_series
# temporal split, so none is demoted (the old solar_1h label pathology and m4_hourly
# cross-series caveat are gone with those datasets). Colors avoid steelblue (the UEA overlay).
ID_STYLE = {
    "monash_electricity_hourly": {"color": "#d62728", "demoted": False, "label": "electricity (energy)"},
    "monash_kdd_cup_2018":       {"color": "#ff7f0e", "demoted": False, "label": "kdd_cup_2018 (air quality)"},
    "monash_pedestrian_counts":  {"color": "#2ca02c", "demoted": False, "label": "pedestrian_counts (transport)"},
    "uber_tlc_hourly":           {"color": "#9467bd", "demoted": False, "label": "uber_tlc_hourly (transport)"},
}


# MASE seasonal period: all four ID datasets are HOURLY, so the seasonal-naive scale uses
# lag m=24 (same-hour-yesterday). Revisit if a non-hourly dataset is ever added.
M_SEASON = 24


def _bytes_per_split(n, pools=3):
    return n * NUM_LAYERS * 768 * 4 * pools


# --------------------------------------------------------------------------- #
# MASE: probe median forecast vs native Chronos-2, on the SAME test windows.
# --------------------------------------------------------------------------- #

def _ctx_stats(X, sigma_eps):
    """Per-window context mean and clamped std (float64) — mirrors id_data._make_examples,
    so `mu + s*sinh(z)` exactly inverts the probe's label transform."""
    X64 = np.asarray(X, np.float64)
    mu = X64.mean(axis=1)
    s = np.maximum(X64.std(axis=1), sigma_eps)
    return mu, s


def _mase_denominator(X, m=M_SEASON):
    """Seasonal-naive in-context scale d_n = mean_t |x_t - x_{t-m}| over the C-m lagged pairs."""
    X64 = np.asarray(X, np.float64)
    return np.abs(X64[:, m:] - X64[:, :-m]).mean(axis=1)


def native_median_forecast(tag, X_test, H):
    """Native Chronos-2 median (q=0.5) forecast for every test context, in RAW units
    (the pipeline inverse-instance-norms internally). Cached to features_cache; the cache
    guard compares context tails so a re-windowed dataset fails loudly instead of misaligning."""
    cache = CACHE_DIR / f"IDF_{tag}__test__native_median_H{H}.npz"
    X_test = np.asarray(X_test, dtype=np.float32)
    if cache.exists():
        d = np.load(cache)
        if d["ctx_tail"].shape[0] == len(X_test) and np.allclose(d["ctx_tail"], X_test[:, -8:]):
            print(f"  [cache HIT]  {cache.name}")
            return d["median"]
        raise RuntimeError(
            f"stale native-forecast cache {cache.name}: cached contexts do not match the "
            f"current test windows — delete it and re-run")
    from probing.extraction import get_pipeline
    pipeline, _ = get_pipeline()
    print(f"  [native] forecasting {len(X_test)} test windows (H={H}, median only)")
    quantiles, _mean = pipeline.predict_quantiles(
        list(X_test), prediction_length=H, quantile_levels=[0.5])
    # each element: (n_variates=1, H, len(levels)=1) -> (H,)
    median = np.stack([qt.reshape(H).cpu().numpy() for qt in quantiles]).astype(np.float32)
    np.savez(cache, median=median, ctx_tail=X_test[:, -8:])
    print(f"  [saved]      {cache.name}  shape={median.shape}")
    return median


def compute_mase(tag, w, diags):
    """Per-layer probe MASE (median forecast, un-transformed to raw units) + native Chronos-2
    MASE on the SAME test windows / horizon / seasonal-naive denominator. LOWER = better."""
    X_test, Y_traj = w["X_test"], w["Y_test_traj"]
    mu, s = _ctx_stats(X_test, w["meta"]["sigma_eps"])
    # raw future reconstructed from the arcsinh label with the SAME mu/s used to invert the
    # probe predictions -- keeps target and prediction exactly consistent.
    y_raw = mu[:, None] + s[:, None] * np.sinh(Y_traj.astype(np.float64))
    d = _mase_denominator(X_test)
    n_clamped = int((d < 1e-8).sum())
    d = np.maximum(d, 1e-8)[:, None]

    native = native_median_forecast(tag, X_test, Y_traj.shape[1])
    entry = {"seasonal_m": M_SEASON, "n_denominator_clamped": n_clamped,
             "native_mase": float((np.abs(y_raw - native) / d).mean()),
             "poolings": {}}
    for pool, diag in diags.items():
        curve = []
        for i in range(NUM_LAYERS):
            zhat = diag["test_median"][i].astype(np.float64)
            yhat = mu[:, None] + s[:, None] * np.sinh(zhat)
            curve.append(float((np.abs(y_raw - yhat) / d).mean()))
        entry["poolings"][pool] = curve
    return entry


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
    diags = {}
    for pool in POOLINGS:
        f_tr, _ = extract_window_features(tag, "train", w["X_train"], w["y_train"], pooling=pool)
        f_te, _ = extract_window_features(tag, "test", w["X_test"], w["y_test"], pooling=pool)
        binned = binned_future_probe(f_tr, w["y_train"], f_te, w["y_test"], n_bins=N_BINS)
        ridge = ridge_regression_probe(f_tr, w["y_train"], f_te, w["y_test"])
        # Chronos-2 quantile probe: uses the (n, H) arcsinh TRAJECTORY labels, not the scalar y.
        # collect_history=True -> also returns per-layer train/val curves + wd-selection diag,
        # kept OUTSIDE `result` so the big summary JSON stays lean (histories only feed figures).
        qloss, qdiag = quantile_probe(f_tr, w["Y_train_traj"], f_te, w["Y_test_traj"],
                                      epochs=QUANTILE_EPOCHS, wd_grid=QUANTILE_WD_GRID,
                                      collect_history=True, collect_test_median=True)
        diags[pool] = qdiag
        result["poolings"][pool] = {
            "binned_accuracy": [float(binned[i]) for i in range(NUM_LAYERS)],
            "ridge_r2": [float(ridge[i]) for i in range(NUM_LAYERS)],
            "quantile_loss": [float(qloss[i]) for i in range(NUM_LAYERS)],  # Chronos-2 loss, lower=better
        }
        acc = np.array(result["poolings"][pool]["binned_accuracy"])
        r2 = np.array(result["poolings"][pool]["ridge_r2"])
        ql = np.array(result["poolings"][pool]["quantile_loss"])
        print(f"  [{pool:>7}] binned acc: argmax L{int(acc.argmax())}={acc.max():.3f} "
              f"(chance={1/N_BINS:.2f}) | ridge R^2 range [{r2.min():+.3f},{r2.max():+.3f}] "
              f"| qloss: argmin L{int(ql.argmin())}={ql.min():.3f}")

    # ---- K CONTROLLED comparison: content / REG / shared-forecast, all from ONE K-slot pass ----
    # content_K & reg_K use the pooled quantile_probe; fslot_K uses the shared forecast-token
    # probe. Same encoder config for all three, so pooled-vs-shared is apples-to-apples. `_final_*`
    # (post-final-norm state) is extracted+cached now for the later frozen-native-head experiment.
    # K = ceil(H / output_patch_size), Chronos-2's own rule, derived from the ACTUAL label horizon;
    # for H not a multiple of 16 the shared probe predicts K whole patches and trims to H.
    H = w["Y_train_traj"].shape[1]
    K = math.ceil(H / OUTPUT_PATCH_SIZE)   # e.g. H=64 -> 4, H=48 -> 3, H=16 -> 1, H=80 -> 5
    fk_tr, _final_tr, _ = extract_kout_features(tag, "train", w["X_train"], w["y_train"], horizon=H)
    fk_te, _final_te, _ = extract_kout_features(tag, "test",  w["X_test"],  w["y_test"],  horizon=H)
    result["kslot"] = {"K": K, "H": H, "probes": {}}
    for name, probe, tr_f, te_f in [
        ("content_K", quantile_probe,               fk_tr["content"], fk_te["content"]),
        ("reg_K",     quantile_probe,               fk_tr["reg"],     fk_te["reg"]),
        ("fslot_K",   shared_forecast_token_probe,  fk_tr["fslot"],   fk_te["fslot"]),
    ]:
        loss, dg = probe(tr_f, w["Y_train_traj"], te_f, w["Y_test_traj"],
                         epochs=QUANTILE_EPOCHS, wd_grid=QUANTILE_WD_GRID,
                         collect_history=True, collect_test_median=True)
        diags[name] = dg
        result["kslot"]["probes"][name] = {"quantile_loss": [float(loss[i]) for i in range(NUM_LAYERS)]}
        c = np.asarray(result["kslot"]["probes"][name]["quantile_loss"])
        print(f"  [{name:>10}] K={K} qloss: argmin L{int(c.argmin())}={c.min():.3f}")

    # MASE comparison must run while the raw windows (X_test, Y_test_traj) are still alive
    result["mase"] = compute_mase(tag, w, diags)
    for pool in POOLINGS:
        mc = np.asarray(result["mase"]["poolings"][pool], float)
        print(f"  [{pool:>7}] MASE: probe best L{int(mc.argmin())}={mc.min():.3f} "
              f"| native Chronos-2 = {result['mase']['native_mase']:.3f}")

    # free the (large) downloaded arrays before the next dataset
    del w
    gc.collect()
    return result, diags


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
    """Plot one ID curve with the tag's styling (demoted curves thin/low-alpha, content solid / REG dashed)."""
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


def _plot_by_layer(ax, id_results, pools):
    """Shared per-layer quantile-loss plotter (content solid / REG dashed, demoted thin/low-alpha, ★ at
    each curve's argmin). LOWER = better, so the tunnel signature is a dip."""
    xs = np.arange(NUM_LAYERS)
    for tag, res in id_results.items():
        st = ID_STYLE[tag]
        for pool in pools:
            loss = np.asarray(res["poolings"][pool]["quantile_loss"], float)
            demoted = st["demoted"]
            lw = 1.1 if demoted else (2.4 if pool == "content" else 1.6)
            alpha = 0.45 if demoted else 1.0
            ls = "-" if pool == "content" else "--"
            mk = "o" if pool == "content" else "s"
            lab = st["label"] + ("" if pool == "content" else " REG")
            ax.plot(xs, loss, color=st["color"], lw=lw, alpha=alpha, ls=ls, marker=mk, ms=3.5, label=lab)
            bi = int(loss.argmin())                       # best (lowest-loss) layer
            ax.plot(bi, loss[bi], marker="*", ms=13, color=st["color"], alpha=alpha,
                    markeredgecolor="k", markeredgewidth=0.5, zorder=5)
    ax.set_xlabel("encoder layer"); ax.set_ylabel("Chronos-2 quantile loss (test)")
    ax.set_xticks(xs); ax.grid(alpha=0.3)


def make_quantile_by_layer(id_results):
    """Main results: per-layer Chronos-2 quantile loss, content and REG in SEPARATE figures
    (cleaner than overlaying both). LOWER = better; ★ = argmin (the tunnel 'dip')."""
    for pool, sub, fname, title in [
        ("content", "content", "quantile_loss_by_layer.png", "content pooling"),
        ("reg", "reg", "quantile_loss_by_layer_reg.png", "REG-token pooling")]:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        _plot_by_layer(ax, id_results, [pool])
        ax.set_title(f"Chronos-2 quantile loss per layer — {title}\n"
                     "LOWER = better; ★ = argmin (best) layer — the tunnel 'dip'")
        ax.legend(fontsize=8, loc="best")
        fig.tight_layout()
        out = QUANT_DIR / sub / fname
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  [saved] {out}")


def make_pooling_comparison(id_results):
    """How the pooling choice changes the tunnel effect: content (solid) vs REG (dashed) on one
    axes, per-layer quantile loss."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    _plot_by_layer(ax, id_results, POOLINGS)
    ax.set_title("Pooling comparison: content (solid) vs REG (dashed)\n"
                 "Chronos-2 quantile loss per layer; ★ = argmin")
    ax.legend(fontsize=7, ncol=2, loc="best")
    fig.tight_layout()
    out = QUANT_DIR / "pooling_comparison" / "content_vs_reg.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {out}")


def make_training_curves(id_diags):
    """Diagnostic: one 4×3 grid per dataset×pooling, each panel = one encoder layer's train
    (80%) and validation (20%) Chronos-2 loss vs epoch. ★ = min val loss (overfitting onset).
    Answers 'did each layer's probe converge, and where does it start to overfit?'."""
    for tag, pools in id_diags.items():
        for pool, diag in pools.items():
            if pool not in POOLINGS:                       # skip the K=4 diags (content_k4/reg_k4/fslot_k4)
                continue
            hist = diag["history"]                        # {layer(int): {"train":[...], "val":[...]}}
            fig, axes = plt.subplots(3, 4, figsize=(16, 10), sharex=True)
            for i, ax in enumerate(axes.ravel()):
                tr = np.asarray(hist[i]["train"], float)
                va = np.asarray(hist[i]["val"], float)
                ep = np.arange(len(tr))
                ax.plot(ep, tr, color="#1f77b4", lw=1.6, label="train (80%)")
                ax.plot(ep, va, color="#d62728", lw=1.6, label="val (20%)")
                vi = int(va.argmin())
                ax.plot(vi, va[vi], "k*", ms=10, zorder=5)   # overfitting onset
                ax.set_title(f"L{i}  wd={diag['wd'][i]:g}", fontsize=9)
                ax.grid(alpha=0.3)
                if i == 0:
                    ax.legend(fontsize=8)
            fig.suptitle(f"Training curves (train vs val) — {tag} / {pool}\n"
                         "★ = min val loss (overfitting onset)", fontsize=13)
            for ax in axes[-1]:   ax.set_xlabel("epoch")
            for ax in axes[:, 0]: ax.set_ylabel("quantile loss")
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            out = QUANT_DIR / "training_curves" / f"{tag}_{pool}.png"
            fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)
            print(f"  [saved] {out}")


def write_quantile_json(id_results, id_diags):
    """Focused, self-contained JSON for the quantile folder: per dataset×pooling test-loss
    curve + argmin + the wd-selection validation losses (the audit trail for tuning). Epoch
    histories are intentionally omitted (they live in the figures, not here)."""
    payload = {"config": {"epochs": QUANTILE_EPOCHS, "wd_grid": QUANTILE_WD_GRID,
                          "poolings": list(POOLINGS),
                          "note": "lower = better; argmin = tunnel dip. quantile_loss = full-train "
                                  "refit test loss; wd_selection_val = per-candidate validation "
                                  "loss from the seed-based 80/20 carve."},
               "datasets": {}}
    for tag, res in id_results.items():
        payload["datasets"][tag] = {}
        for pool in POOLINGS:
            ql = np.asarray(res["poolings"][pool]["quantile_loss"], float)
            diag = id_diags[tag][pool]
            payload["datasets"][tag][pool] = {
                "quantile_loss": [float(x) for x in ql],
                "argmin_layer": int(ql.argmin()), "min_loss": float(ql.min()),
                "L0": float(ql[0]), "L11": float(ql[LAST_LAYER]),
                "wd_selected": {str(k): v for k, v in diag["wd"].items()},
                "wd_selection_val": {str(k): v for k, v in diag["selection"].items()},
            }
    out = QUANT_DIR / "quantile_loss_results.json"
    json.dump(payload, open(out, "w"), indent=2)
    print(f"  [saved] {out}")


def make_shared_forecast_comparison(id_results):
    """Chronos-alignment under a CONTROLLED K-slot pass: pooled content_K (solid) vs pooled reg_K
    (dashed) vs the SHARED forecast-token probe fslot_K (dotted) — per-layer Chronos-2 quantile
    loss, one panel per dataset. Same units + same encoder config; ★ = each curve's argmin.
    Caveat (state in writeup): pooled vs shared differ in BOTH representation AND readout (shared
    has ~K× fewer params + enforced patch-wise weight sharing), so a lower fslot_K curve is not
    purely 'info more decodable here' — the decoding constraint differs too."""
    xs = np.arange(NUM_LAYERS)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (tag, res) in zip(axes.ravel(), id_results.items()):
        color = ID_STYLE[tag]["color"]
        H, K = res["kslot"]["H"], res["kslot"]["K"]
        styles = [("content_K", "-", "o", f"pooled content_K  Linear(768, 21·{H})"),
                  ("reg_K",     "--", "s", f"pooled REG_K  Linear(768, 21·{H})"),
                  ("fslot_K",   ":", "D", f"shared forecast-token  Linear(768, 21·{OUTPUT_PATCH_SIZE})")]
        for key, ls, mk, lab in styles:
            c = np.asarray(res["kslot"]["probes"][key]["quantile_loss"], float)
            ax.plot(xs, c, color=color, lw=2.0, ls=ls, marker=mk, ms=4, label=lab)
            bi = int(c.argmin())
            ax.plot(bi, c[bi], marker="*", ms=12, color=color, markeredgecolor="k",
                    markeredgewidth=0.5, zorder=5)
        ax.set_title(f"{ID_STYLE[tag]['label']}  (H={H}, K={K})", fontsize=11)
        ax.set_xticks(xs); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    for ax in axes[-1]:   ax.set_xlabel("encoder layer")
    for ax in axes[:, 0]: ax.set_ylabel("Chronos-2 quantile loss (test)")
    fig.suptitle("Chronos-alignment (controlled K-slot pass): pooled content / REG vs SHARED forecast-token\n"
                 "same quantile loss; LOWER = better; ★ = argmin", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = QUANT_DIR / "shared_forecast_vs_pooled.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {out}")


def make_mase_figures(id_results):
    """Per-layer probe MASE vs the native Chronos-2 head, one 2x2 figure per pooling
    (one panel per dataset). LOWER = better; ★ = best probe layer; dashed line = native.
    A probe below the line means the linear readout beat the native median forecast."""
    xs = np.arange(NUM_LAYERS)
    for pool in POOLINGS:
        fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
        for ax, (tag, res) in zip(axes.ravel(), id_results.items()):
            st = ID_STYLE[tag]
            ms = res["mase"]
            curve = np.asarray(ms["poolings"][pool], float)
            ax.plot(xs, curve, color=st["color"], lw=2.2, marker="o", ms=4,
                    label="linear probe (median)")
            ax.axhline(ms["native_mase"], color="k", ls="--", lw=1.4,
                       label=f"native Chronos-2 = {ms['native_mase']:.3f}")
            bi = int(curve.argmin())
            ax.plot(bi, curve[bi], marker="*", ms=14, color=st["color"],
                    markeredgecolor="k", markeredgewidth=0.5, zorder=5)
            ax.set_title(st["label"], fontsize=11)
            ax.set_xticks(xs); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc="best")
        for ax in axes[-1]:   ax.set_xlabel("encoder layer")
        for ax in axes[:, 0]: ax.set_ylabel("test MASE")
        fig.suptitle(f"MASE: per-layer linear probe (median forecast) vs native Chronos-2 — "
                     f"{pool} pooling\nseasonal-naive scale m={M_SEASON}; LOWER = better; "
                     "★ = best probe layer", fontsize=13)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        out = QUANT_DIR / f"mase_{pool}_vs_native.png"
        fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
        print(f"  [saved] {out}")


def make_mase_k4_figure(id_results):
    """MASE under the controlled K-slot pass: pooled content_k4 (solid) / reg_k4 (dashed) vs the
    SHARED forecast-token probe fslot_k4 (dotted), plus native Chronos-2 (black dashed). One panel
    per dataset; ★ = best probe layer; LOWER = better. Companion to make_shared_forecast_comparison
    (that one is quantile loss, this one is MASE of the median forecast). Same interpretation caveat:
    pooled vs shared differ in representation AND readout capacity (4x fewer params + weight sharing)."""
    xs = np.arange(NUM_LAYERS)
    keys = [("content_k4", "-", "o"), ("reg_k4", "--", "s"), ("fslot_k4", ":", "D")]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, (tag, res) in zip(axes.ravel(), id_results.items()):
        color = ID_STYLE[tag]["color"]; ms = res["mase"]
        for key, ls, mk in keys:
            curve = np.asarray(ms["poolings"][key], float)
            ax.plot(xs, curve, color=color, lw=2.0, ls=ls, marker=mk, ms=4, label=key)
            bi = int(curve.argmin())
            ax.plot(bi, curve[bi], marker="*", ms=12, color=color, markeredgecolor="k",
                    markeredgewidth=0.5, zorder=5)
        ax.axhline(ms["native_mase"], color="k", ls="--", lw=1.3,
                   label=f"native Chronos-2 = {ms['native_mase']:.3f}")
        ax.set_title(ID_STYLE[tag]["label"], fontsize=11)
        ax.set_xticks(xs); ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="best")
    for ax in axes[-1]:   ax.set_xlabel("encoder layer")
    for ax in axes[:, 0]: ax.set_ylabel("test MASE")
    fig.suptitle("MASE (controlled K-slot pass): pooled content/REG vs SHARED forecast-token vs native\n"
                 f"seasonal-naive m={M_SEASON}; LOWER = better; ★ = best probe layer", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = QUANT_DIR / "shared_forecast_mase.png"
    fig.savefig(out, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  [saved] {out}")


def write_mase_json(id_results):
    """Focused MASE JSON: per dataset the native Chronos-2 MASE + per-pooling per-layer probe
    MASE, argmin, and the probe-vs-native ratio (<1 = probe beat the native median)."""
    payload = {"config": {"seasonal_m": M_SEASON, "H": 64, "poolings": list(POOLINGS),
                          "note": "MASE of the q=0.5 forecast, un-transformed to raw units "
                                  "(y = mu_ctx + sigma_ctx*sinh(z)); denominator = in-context "
                                  "seasonal-naive scale, identical for probe and native. "
                                  "LOWER = better."},
               "datasets": {}}
    for tag, res in id_results.items():
        ms = res["mase"]
        ent = {"native_mase": ms["native_mase"], "seasonal_m": ms["seasonal_m"],
               "n_denominator_clamped": ms["n_denominator_clamped"], "poolings": {}}
        for pool, curve in ms["poolings"].items():
            c = np.asarray(curve, float)
            bi = int(c.argmin())
            ent["poolings"][pool] = {
                "probe_mase": [float(x) for x in c],
                "argmin_layer": bi, "min_mase": float(c[bi]),
                "L11_mase": float(c[LAST_LAYER]),
                "best_probe_over_native": float(c[bi] / ms["native_mase"]),
            }
        payload["datasets"][tag] = ent
    out = QUANT_DIR / "mase_results.json"
    json.dump(payload, open(out, "w"), indent=2)
    print(f"  [saved] {out}")


def main():
    # classification reference curves (content pooling) from the committed UEA summary
    summ = json.load(open(OUT_DIR / "perdataset_summary.json"))["datasets"]
    uea_curves = {name: summ[name]["per_layer_accuracy"]["ID"] for name in UEA_REF}

    id_results, id_diags = {}, {}
    for tag in ID_DATASETS:
        id_results[tag], id_diags[tag] = run_dataset(tag)

    # estimated new cache footprint (FYI; disk is not a constraint)
    total_bytes = sum(_bytes_per_split(r["meta"]["n_train"]) + _bytes_per_split(r["meta"]["n_test"])
                      for r in id_results.values())
    print(f"\n  est. new IDF_ cache footprint: ~{total_bytes/1e9:.2f} GB "
          f"(3 poolings x 12 layers x n_windows x 768 x 4B)")

    # ---- late-layer retention = score[L11] / score[peak] (the quantity the tunnel
    #      comparison needs — how much each curve keeps after its own peak) ----
    retention = {"id": {}, "uea_classification": {}}
    for tag, res in id_results.items():
        retention["id"][tag] = {}
        for pool in POOLINGS:
            ent = {}
            for metric in ("binned_accuracy", "ridge_r2"):
                pk, peak, ret = _peak_retention(res["poolings"][pool][metric])
                ent[metric] = {"peak_layer": pk, "peak_value": peak, "retention_L11": ret}
            # quantile loss is LOWER=better, so the tunnel-relevant number is the excess
            # of the last layer over the best layer: loss[L11]/loss[argmin] >= 1.
            ql = np.asarray(res["poolings"][pool]["quantile_loss"], float)
            bi = int(ql.argmin())
            ent["quantile_loss"] = {"best_layer": bi, "best_value": float(ql[bi]),
                                    "excess_L11": float(ql[LAST_LAYER] / ql[bi])}
            retention["id"][tag][pool] = ent
    for name, acc in uea_curves.items():
        pk, peak, ret = _peak_retention(acc)
        retention["uea_classification"][name] = {"peak_layer": pk, "peak_value": peak,
                                                 "retention_L11": ret}

    summary = {
        "config": {"C": 512, "H": 64, "n_bins": N_BINS, "poolings": list(POOLINGS),
                   "id_datasets": list(ID_DATASETS), "uea_reference": UEA_REF,
                   "binned_chance": 1.0 / N_BINS,
                   "quantile_epochs": QUANTILE_EPOCHS, "quantile_wd_grid": QUANTILE_WD_GRID,
                   "quantile_note": "Chronos-2-native probe: nn.Linear(768, 21*H) trained + scored "
                                    "with Chronos-2's quantile loss on arcsinh trajectory labels "
                                    "(AdamW, decay on weights only, bias undecayed); per-layer TEST "
                                    "loss, LOWER=better (tunnel signature = argmin). Caveat: the wd "
                                    "selection carve is random over heavily overlapping windows "
                                    "(stride 64 << span 576), so val loss is optimistic — same "
                                    "caveat applies to the ridge probe's alpha selection.",
                   "dataset_note": "All four ID datasets are long-series hourly (electricity, "
                                   "kdd_cup_2018, pedestrian_counts, uber_tlc_hourly) and use the "
                                   "within_series temporal split; the earlier solar_1h label "
                                   "pathology and m4_hourly cross-series caveat no longer apply."},
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
    make_quantile_by_layer(id_results)
    make_pooling_comparison(id_results)
    make_shared_forecast_comparison(id_results)
    make_training_curves(id_diags)
    write_quantile_json(id_results, id_diags)
    make_mase_figures(id_results)
    make_mase_k4_figure(id_results)
    write_mase_json(id_results)

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
    # late-layer retention table (binned accuracy, content) — the tunnel-relevant number
    print(f"\n  late-layer retention = binned_acc[L11] / binned_acc[peak]  (content pooling):")
    for tag, r in id_results.items():
        pk, peak, ret = _peak_retention(r["poolings"]["content"]["binned_accuracy"])
        print(f"    {tag:>26}:  peak L{pk}={peak:.3f}  ->  L11 retains {ret:.3f}")
    # Chronos-2 quantile loss (content) — the forecasting-native tunnel readout (LOWER=better)
    print(f"\n  Chronos-2 quantile loss  (content pooling; LOWER=better, tunnel=argmin):")
    for tag, r in id_results.items():
        ql = np.array(r["poolings"]["content"]["quantile_loss"])
        print(f"    {tag:>26}:  best L{int(ql.argmin())}={ql.min():.3f}  |  "
              f"L0={ql[0]:.3f}  L11={ql[LAST_LAYER]:.3f}")
    # MASE (content) — probe median forecast vs native Chronos-2 on the same test windows
    print(f"\n  MASE, median forecast  (content pooling; m={M_SEASON}; LOWER=better):")
    for tag, r in id_results.items():
        mc = np.array(r["mase"]["poolings"]["content"])
        nat = r["mase"]["native_mase"]
        print(f"    {tag:>26}:  probe best L{int(mc.argmin())}={mc.min():.3f}  "
              f"L11={mc[LAST_LAYER]:.3f}  |  native={nat:.3f}  "
              f"(best/native={mc.min()/nat:.2f})")
    print(f"\n  binned chance = {1/N_BINS:.2f}")


if __name__ == "__main__":
    main()
