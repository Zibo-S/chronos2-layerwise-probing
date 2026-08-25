"""v4 future-token TWO-AXIS transfer: frozen shared-forecast probes across pretraining × probe axes.

Separates PRETRAINING status (was the TARGET in Chronos-2's pretraining corpus?) from PROBE-TRAINING
status (was the frozen probe fit on the same dataset it is scored on?). The four quadrants:

  * PT-ID / Probe-ID    — target pretrained, probe fit + tested on the SAME set (4×4 diagonal).
  * PT-ID / Probe-OOD   — target pretrained, frozen probe fit on a DIFFERENT set (4×4 off-diagonal).
  * PT-OOD / Probe-OOD  — target NOT pretrained, frozen probe fit elsewhere (experiment 2).
  * PT-OOD / Probe-ID   — target NOT pretrained, FRESH probe fit on the target — the diagnostic
                          that lives in run_ptood_probing_ftok.py, NOT a transfer experiment.

`pt_status` describes the TARGET (never the source). This driver is PREDICT-ONLY: it reuses the
frozen PT-ID source probes already trained + checkpointed by `run_ptood_probing_ftok --fit-ptid`
(source-fit slot-scaler + Linear + source-validation-selected wd) and the first-crossing tunnels from
`--tunnels-only`. Nothing is trained here — `predict_shared_forecast_probe` applies the frozen
source scaler + weights to each target's test split; no target scaler / wd / layer selection ever
runs off-diagonal. The shared forecast-token curve has 14 points (Emb, L1..L12, L12+LN); L12+LN
(the post-final-LN native-head input) is the final reference.

Two modes:
  --experiment transfer_4x4  (default): 4 PT-ID sources × 4 PT-ID targets. Diagonal = PT-ID/Probe-ID,
                             off-diagonal = PT-ID/Probe-OOD. Headline summary = transfer gap
                             L_{s→t}(ℓ_s) / L_{t→t}(ℓ_t) − 1, ℓ_s / ℓ_t both from source/target
                             VALIDATION (never test); diagonal cells are 0 by construction.
  --experiment pt_ood        : all 4 PT-ID sources → {sg_carpark, coastal_ts, boom_hourly} targets
                             (4×3). Every cell is PT-OOD/Probe-OOD. Each source row shades ONLY its
                             own source-validation tunnel entrance (never the other sources').

Metrics per cell (both modes): raw Chronos-2 quantile loss, in-context seasonal-naive MASE (m=24,
mase_context — the project-wide definition), and (4×4 only) the transfer gap. Min-over-test
relative regret is recorded SUPPLEMENTARY only (a broken transferred probe still has a 0-regret
layer, so regret alone cannot flag catastrophic transfer).

Outputs: results/ext_v4_future_tokens/{fslot_transfer, fslot_pt_ood}/ (NEW dirs — nothing existing
is moved). Predict-only + warm caches → CPU/login-node for the 4×4; the PT-OOD targets rebuild
their rolling windows, which is heavy for sg_carpark/boom_hourly → run experiment 2 on a compute node.

Run (USER submits SLURM where a compute node is needed):
    python -m experiments.run_fslot_transfer                       # 4×4 (default)
    python -m experiments.run_fslot_transfer --experiment pt_ood   # Electricity → 3 PT-OOD targets
"""

from __future__ import annotations

import argparse
import csv
import gc
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probing import config
from probing.config import NUM_LAYERS
from probing.extraction import _cache_path, _idf_prefix       # for the §4 shared-cache preflight
from probing.id_data import build_ood_rolling_windows, build_windows
from probing.probes import QUANTILE_SETS, validate_quantiles
from probing.tunnel import PT_ID_TAGS, PT_OOD_TAGS, domain_status
# reuse the per-dataset pipeline's in-context MASE pieces (import only defines functions)
from experiments.run_id_forecasting import M_SEASON, _ctx_stats, _mase_denominator
# frozen-probe loader + shared fslot frame from the diagnostic driver (constants are module-level,
# so importing does not run its main(); _fslot_feats uses that module's H/K/NUM_LAYERS globals).
# PROBE_FAMILIES routes the readout head (shared_linear default | native_mlp) — same features/windows/
# tunnels, family-specific checkpoints/predict/output namespace.
from experiments.run_ptood_probing_ftok import (C, H, K, LAYER_LABELS, OUT_ROOT, PROBE_FAMILIES,
                                                PROBE_PROTOCOL_VERSION, RUN_SEEDS, RUNS_TAG, SHORT,
                                                _fslot_feats)

REF_LABEL = LAYER_LABELS[-1]                                # "L12+LN" — the final reference point
N_POINTS = NUM_LAYERS + 1                                   # 14 fslot readout points
FAMILY = PROBE_FAMILIES["shared_linear"]                   # module default; main() sets from --probe-family


# --------------------------------------------------------------------------- #
# terminology helpers — the two-axis labels, never a bare "ID" / "OOD"
# --------------------------------------------------------------------------- #
def _pt_label(target):
    """PT status of the TARGET (correction: target, not source)."""
    return "PT-ID" if domain_status(target)["pretraining"] == "pt_id" else "PT-OOD"


def _probe_label(source, target):
    return "Probe-ID" if source == target else "Probe-OOD"


def _quadrant(source, target):
    return f"{_pt_label(target)} / {_probe_label(source, target)}"


def _derive_dirs(experiment, qset):
    """Predict-only transfer, so nothing is FIT here — the frozen source probes + tunnels are read
    from the versioned ext-v4 ID layout (FAMILY.ckpt_dir / FAMILY.tunnel_path already carry the
    protocol version). Outputs route into the browsable per-quantile tree: the 4x4 cross-dataset grid
    -> results/ext_v4_future_tokens/<qset>/cross_dataset/, the 4x3 unseen grid -> <qset>/unseen/.
    Legacy flat fslot_transfer/ + fslot_pt_ood/ (committed q9) are left untouched. MLP unchanged."""
    global OUT_DIR, BOOT_IN_DIR, FIG_DIR, TAB_DIR
    if FAMILY.name == "native_mlp":                     # fslot_mlp/{transfer_4x4,ptood_transfer}/
        OUT_DIR = FAMILY.out_root / ("transfer_4x4" if experiment == "transfer_4x4" else "ptood_transfer")
    else:                                               # linear: browsable per-quantile tree
        OUT_DIR = OUT_ROOT / qset / ("cross_dataset" if experiment == "transfer_4x4" else "unseen")
    BOOT_IN_DIR = OUT_DIR / "bootstrap_inputs"
    FIG_DIR = OUT_DIR / "figures"
    TAB_DIR = OUT_DIR / "tables"
    for d in (OUT_DIR, BOOT_IN_DIR, FIG_DIR, TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _tunnel_path(src, qset):
    return FAMILY.tunnel_path(src, qset)     # linear -> OUT_ROOT/tunnels/... (unchanged); mlp -> fslot_mlp/


def _load_tunnel(src, qset):
    p = _tunnel_path(src, qset)
    if not p.exists():
        raise FileNotFoundError(f"missing tunnel record {p} — run "
                                "`run_ptood_probing_ftok --tunnels-only` first")
    return json.load(open(p))


def _val_selected_layer(rec):
    """ℓ = argmin over layers of the MEAN source-VALIDATION curve (earliest-layer tie-break via
    np.argmin). Never touches test; consumes the existing authoritative tunnel record."""
    return int(np.argmin(rec["mean_val_loss_by_layer"]))


# --------------------------------------------------------------------------- #
# preflight — fail loud BEFORE any warm-cache predict (correction 8)
# --------------------------------------------------------------------------- #
def preflight_checkpoints(sources, qset):
    """Every source×seed checkpoint has all 14 heads + scalers with consistent dims (family-aware:
    the MLP checkpoint stores head_state_dict, the linear one state_dict)."""
    sd_key = "head_state_dict" if FAMILY.name == "native_mlp" else "state_dict"
    for src in sources:
        for seed in RUN_SEEDS:
            d = FAMILY.ckpt_dir(src, qset, seed)
            paths = sorted(d.glob("L*.pt"))
            if len(paths) != N_POINTS:
                raise RuntimeError(f"[preflight] {d}: {len(paths)} layer files, expected {N_POINTS} "
                                   "(L00..L13 = Emb..L12 + post-LN) — re-run --fit-ptid")
            for p in paths:
                ck = torch.load(p, map_location="cpu", weights_only=False)
                for kk in (sd_key, "scaler_mean", "scaler_scale", "in_features", "out_features"):
                    if kk not in ck:
                        raise RuntimeError(f"[preflight] {p}: checkpoint missing '{kk}'")
                if int(np.asarray(ck["scaler_mean"]).shape[0]) != int(ck["in_features"]):
                    raise RuntimeError(f"[preflight] {p}: scaler dim {np.asarray(ck['scaler_mean']).shape} "
                                       f"!= in_features {ck['in_features']}")
    print(f"  [preflight] {len(sources)}×{len(RUN_SEEDS)} {FAMILY.name} source probes OK "
          f"({N_POINTS} heads+scalers each)")


def preflight_feature_cache(targets, rolling_ood):
    """§4: the MLP run MUST resolve the SAME fslot feature caches as the linear run — feature_kind is
    unchanged, only the readout head differs, so there is no re-extraction. The kout cache path depends
    solely on (tag, split, K, H) (extraction._cache_path), never on the probe family, so assert the
    resolved path keeps its K/H key and NEVER embeds the artifact tag (fslot / fslot_mlp)."""
    split = "test_rolling" if rolling_ood else "test"
    for tgt in targets:
        name = _cache_path(_idf_prefix(tgt), split, None, f"K{K}_H{H}").name
        assert f"K{K}_H{H}" in name, f"feature cache {name} lost its K/H key"
        assert FAMILY.artifact_tag not in name and "fslot" not in name, (
            f"feature cache {name} embeds a probe-family artifact tag — caches must be family-shared")
    print(f"  [preflight] fslot feature caches are family-independent "
          f"(K{K}_H{H}; {FAMILY.name} shares the linear cache)")


def _check_features(tag, feats, w):
    """Feature/label/series counts agree and the 14 fslot keys are all present (correction 8)."""
    n = int(len(w["Y_test_traj"]))
    if not (n == len(w["series_test"]) == int(w["meta"]["n_test"])):
        raise RuntimeError(f"[preflight] {tag}: label {n} / series {len(w['series_test'])} / "
                           f"meta n_test {w['meta']['n_test']} disagree")
    keys = sorted(feats)
    if keys != list(range(N_POINTS)):
        raise RuntimeError(f"[preflight] {tag}: feature keys {keys} != 0..{NUM_LAYERS}")
    for i in keys:
        if feats[i].shape[0] != n:
            raise RuntimeError(f"[preflight] {tag} L{i}: {feats[i].shape[0]} windows != {n} labels")
    return n


# --------------------------------------------------------------------------- #
# metrics — 14-point in-context MASE (compute_mase is hardwired to 13 layers, so a small twin)
# --------------------------------------------------------------------------- #
def _fslot_mase(tag, w, diag):
    """Per-layer MASE of the frozen probe's median forecast, un-transformed to raw units with each
    TARGET window's own context stats (mu + s*sinh(z)), scaled by the in-context seasonal-naive
    denominator (m=24). Same definition as run_id_forecasting.compute_mase (mase_context), extended
    to all 14 fslot points. Leakage-free (context-only inversion + denominator)."""
    mu, s = _ctx_stats(w["X_test"], w["meta"]["sigma_eps"])
    y_raw = mu[:, None] + s[:, None] * np.sinh(w["Y_test_traj"].astype(np.float64))
    d = np.maximum(_mase_denominator(w["X_test"]), 1e-8)[:, None]
    out = {}
    for i in sorted(diag["test_median"]):
        yhat = mu[:, None] + s[:, None] * np.sinh(diag["test_median"][i].astype(np.float64))
        out[i] = float((np.abs(y_raw - yhat) / d).mean())
    return out


def _relative_regret(curve, layer):
    """Supplementary: (L(layer) − min_j L(j)) / min_j L(j). 0 at the per-target argmin, so it
    CANNOT flag a uniformly-bad transfer — kept only as a secondary view."""
    c = np.asarray(curve, dtype=np.float64)
    lo = float(c.min())
    return float((c[layer] - lo) / lo)


# --------------------------------------------------------------------------- #
# one predict-only transfer cell (frozen source probe on a target test split)
# --------------------------------------------------------------------------- #
def eval_cell(source, target, w, feats, fitted, qset, seed, quantiles, device):
    """Score a FROZEN source probe on the target's test windows. Returns 14-point quantile-loss and
    MASE curves; writes per-window loss + series ids for the cluster bootstrap. No fit / no scaler
    fit / no layer selection touches the target."""
    out, diag = FAMILY.predict(fitted, feats, w["Y_test_traj"], quantiles=quantiles,
                               device=device, collect_test_median=True,
                               collect_test_window_loss=True)
    layers = sorted(out)
    mase = _fslot_mase(target, w, diag)
    wl = np.stack([diag["test_window_loss"][i] for i in layers]).astype(np.float64)   # (14, n)
    sid = np.asarray(w["series_test"], np.int64)
    np.savez(BOOT_IN_DIR / f"{source}__to__{target}__{qset}__seed{seed}.npz",
             window_loss=wl, series_test=sid)
    return {"quantile_loss": [float(out[i]) for i in layers],
            "mase": [float(mase[i]) for i in layers],
            "n_test": int(len(sid)), "n_clusters": int(np.unique(sid).size)}


def _target_windows_and_feats(target, rolling_ood):
    """Build the target's test split + load its 14-point fslot test features (cache HIT from the
    diagnostic run). rolling_ood selects the PT-OOD rolling builder + its 'test_rolling' cache name."""
    if rolling_ood:
        w = build_ood_rolling_windows(target, C=C, H=H)
        split = "test_rolling"
    else:
        w = build_windows(target)
        split = "test"
    m = w["meta"]
    if int(m["n_test"]) == 0:
        raise RuntimeError(f"{target}: empty test split — check the loader / run_ood_screen")
    feats = _fslot_feats(target, split, w["X_test"], w["y_test"])
    _check_features(target, feats, w)
    return w, feats


# --------------------------------------------------------------------------- #
# Experiment 1 — 4×4 PT-ID/Probe-* transfer
# --------------------------------------------------------------------------- #
def run_4x4(qset, quantiles, device):
    sources = list(PT_ID_TAGS)
    targets = list(PT_ID_TAGS)
    preflight_checkpoints(sources, qset)
    preflight_feature_cache(targets, rolling_ood=False)
    tunnels = {s: _load_tunnel(s, qset) for s in sources}
    ell = {s: _val_selected_layer(tunnels[s]) for s in sources}     # source-VAL-selected layer
    print("  source-validation-selected layers: "
          + ", ".join(f"{SHORT[s]}={LAYER_LABELS[ell[s]]}" for s in sources))

    # build each target's windows + features ONCE (shared across the 4 source rows)
    tgt_wf = {t: _target_windows_and_feats(t, rolling_ood=False) for t in targets}

    cells = {}                                                       # (src,tgt) -> {"ql":[S×14],"mase":[S×14]}
    meta_cell = {}
    for src in sources:
        for seed in RUN_SEEDS:
            fitted = FAMILY.load_ckpt(src, qset, seed, device=device)
            for tgt in targets:
                w, feats = tgt_wf[tgt]
                res = eval_cell(src, tgt, w, feats, fitted, qset, seed, quantiles, device)
                c = cells.setdefault((src, tgt), {"ql": [], "mase": []})
                c["ql"].append(res["quantile_loss"])
                c["mase"].append(res["mase"])
                meta_cell[(src, tgt)] = {"n_test": res["n_test"], "n_clusters": res["n_clusters"]}
            del fitted
            gc.collect()

    mean_ql = {k: np.mean(v["ql"], axis=0) for k, v in cells.items()}
    mean_mase = {k: np.mean(v["mase"], axis=0) for k, v in cells.items()}
    gap = {}                                                         # transfer gap (correction 2)
    for src in sources:
        for tgt in targets:
            Ls = float(mean_ql[(src, tgt)][ell[src]])                # L_{s→t}(ℓ_s)
            Lt = float(mean_ql[(tgt, tgt)][ell[tgt]])                # L_{t→t}(ℓ_t), both from val
            gap[(src, tgt)] = Ls / Lt - 1.0

    _write_records(sources, targets, cells, mean_ql, mean_mase, ell, tunnels, gap, qset, meta_cell)
    make_4x4_figure(sources, targets, cells, ell, tunnels, gap, qset)
    _print_gap_summary(sources, targets, gap, ell)


# --------------------------------------------------------------------------- #
# Experiment 2 — 4 PT-ID sources → 3 PT-OOD targets, all PT-OOD/Probe-OOD
# --------------------------------------------------------------------------- #
def run_pt_ood(qset, quantiles, device):
    sources = list(PT_ID_TAGS)                                       # all 4 pretrained probe sources
    targets = list(PT_OOD_TAGS)
    preflight_checkpoints(sources, qset)
    preflight_feature_cache(targets, rolling_ood=True)
    tunnels = {s: _load_tunnel(s, qset) for s in sources}
    ell = {s: _val_selected_layer(tunnels[s]) for s in sources}     # source-VAL-selected layer, per source
    print("  source-validation-selected layers: "
          + ", ".join(f"{SHORT[s]}={LAYER_LABELS[ell[s]]}" for s in sources))

    cells, meta_cell = {}, {}
    for tgt in targets:                                              # build each OOD target's windows+feats ONCE
        w, feats = _target_windows_and_feats(tgt, rolling_ood=True)
        for src in sources:
            for seed in RUN_SEEDS:
                fitted = FAMILY.load_ckpt(src, qset, seed, device=device)
                res = eval_cell(src, tgt, w, feats, fitted, qset, seed, quantiles, device)
                c = cells.setdefault((src, tgt), {"ql": [], "mase": []})
                c["ql"].append(res["quantile_loss"])
                c["mase"].append(res["mase"])
                meta_cell[(src, tgt)] = {"n_test": res["n_test"], "n_clusters": res["n_clusters"]}
                del fitted
                gc.collect()
        del w, feats
        gc.collect()

    mean_ql = {k: np.mean(v["ql"], axis=0) for k, v in cells.items()}
    mean_mase = {k: np.mean(v["mase"], axis=0) for k, v in cells.items()}
    _write_records(sources, targets, cells, mean_ql, mean_mase, ell,
                   tunnels, gap=None, qset=qset, meta_cell=meta_cell)
    make_pt_ood_figure(sources, targets, cells, ell, tunnels, qset)


# --------------------------------------------------------------------------- #
# records — every row carries the two-axis metadata (no bare ID/OOD)
# --------------------------------------------------------------------------- #
def _write_records(sources, targets, cells, mean_ql, mean_mase, ell, tunnels, gap, qset, meta_cell):
    """Per-cell summary (one row) + per-layer tidy table (cell × 14 layers), both with the full
    metadata schema; plus a curves JSON with the mean 14-point ql/mase per cell."""
    summary, tidy, curves = [], [], {}
    for src in sources:
        for tgt in targets:
            if (src, tgt) not in cells:
                continue
            rec = tunnels[src]
            ls_sustained = int(rec["l_start"])                       # existing authoritative boundary
            sel = int(ell[src])
            ql = mean_ql[(src, tgt)]
            mase = mean_mase[(src, tgt)]
            base = {
                "pt_status": _pt_label(tgt), "probe_status": _probe_label(src, tgt),
                "quadrant": _quadrant(src, tgt),
                "source_dataset": src, "target_dataset": tgt,
                "probe_fitted_on": src, "wd_selected_on": f"{src}:validation",
                "tunnel_defined_on": f"{src}:validation",
                "l_start_sustained": ls_sustained,
                "val_selected_layer": sel, "final_reference": REF_LABEL,
                "quantile_set": qset, "readout": FAMILY.artifact_tag, "probe_family": FAMILY.name,
                "probe_protocol_version": PROBE_PROTOCOL_VERSION,
            }
            summary.append({**base,
                            "val_selected_layer_label": LAYER_LABELS[sel],
                            "quantile_loss_at_selected": round(float(ql[sel]), 6),
                            "quantile_loss_at_reference": round(float(ql[-1]), 6),
                            "mase_at_selected": round(float(mase[sel]), 6),
                            "mase_at_reference": round(float(mase[-1]), 6),
                            "transfer_gap": (None if gap is None else round(float(gap[(src, tgt)]), 6)),
                            "relative_regret_at_selected_supp": round(_relative_regret(ql, sel), 6),
                            "n_test_windows": meta_cell[(src, tgt)]["n_test"],
                            "n_test_clusters": meta_cell[(src, tgt)]["n_clusters"]})
            curves[f"{src}__to__{tgt}"] = {"quantile_loss": [float(x) for x in ql],
                                           "mase": [float(x) for x in mase],
                                           "layer_labels": LAYER_LABELS}
            for i in range(N_POINTS):
                tidy.append({**base, "layer": i, "layer_label": LAYER_LABELS[i],
                             "quantile_loss": round(float(ql[i]), 6), "mase": round(float(mase[i]), 6),
                             "is_selected_layer": i == sel, "is_reference": i == NUM_LAYERS,
                             "in_sustained_tunnel": i >= ls_sustained})

    tag = "4x4" if gap is not None else "pt_ood"
    _dump_csv_json(TAB_DIR / f"transfer_summary__{tag}__{qset}", summary)
    _dump_csv_json(TAB_DIR / f"transfer_by_layer__{tag}__{qset}", tidy)
    json.dump({"mase_definition": f"mase_context in-context seasonal-naive (m={M_SEASON})",
               "final_reference": REF_LABEL, "run_seeds": list(RUN_SEEDS), "cells": curves},
              open(TAB_DIR / f"transfer_curves__{tag}__{qset}.json", "w"), indent=2)
    print(f"  [saved] {len(summary)} cell records + {len(tidy)} layer rows -> {TAB_DIR}")


def _dump_csv_json(stem, rows):
    if rows:
        with open(f"{stem}.csv", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    json.dump(rows, open(f"{stem}.json", "w"), indent=2)


def _print_gap_summary(sources, targets, gap, ell):
    print("\n  == 4×4 transfer gap  L_{s→t}(ℓ_s)/L_{t→t}(ℓ_t) − 1  (0 on the diagonal by construction) ==")
    for src in sources:
        row = "  ".join(f"{SHORT[t]}:{gap[(src, t)]*100:+6.1f}%" for t in targets)
        print(f"    {SHORT[src]:>11} (ℓ={LAYER_LABELS[ell[src]]:>6}) -> {row}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plot_cell(ax, seeds, ls, sel, color):
    """One panel: faint per-seed curves + mean ± std band + tunnel shading + val-selected marker."""
    x = np.arange(N_POINTS)
    last = NUM_LAYERS
    ax.axvspan(ls - 0.25, last + 0.25, color="tab:green", alpha=0.10)
    for c in seeds:
        ax.plot(x, c, color=color, alpha=0.25, lw=0.8)
    m, sd = seeds.mean(axis=0), seeds.std(axis=0)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18)
    ax.plot(x, m, "o-", color=color, ms=3, lw=1.6)
    ax.plot(sel, m[sel], "D", color="k", ms=8, mfc=color, mew=1.3, zorder=6)
    ax.axvline(last, color="0.4", ls=":", lw=1.0)
    return m


def make_4x4_figure(sources, targets, cells, ell, tunnels, gap, qset):
    """rows = probe-training source, cols = evaluation target. Mean-over-seed target-test loss +
    seed band; the row's source-validation first-crossing tunnel shaded; ◆ at the source-val-selected
    layer. Y-LIMITS SHARED DOWN EACH TARGET COLUMN (correction 7) so catastrophic transfer cannot
    be hidden by per-panel autoscaling. Diagonal titles PT-ID/Probe-ID (bold), off-diagonal
    PT-ID/Probe-OOD."""
    nrow, ncol = len(sources), len(targets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), sharex=True)
    col_lim = {}
    for tgt in targets:
        allv = np.concatenate([np.asarray(cells[(s, tgt)]["ql"]).ravel() for s in sources])
        pad = 0.04 * (allv.max() - allv.min() or 1.0)
        col_lim[tgt] = (allv.min() - pad, allv.max() + pad)
    for ri, src in enumerate(sources):
        ls = int(tunnels[src]["l_start"])
        for ci, tgt in enumerate(targets):
            ax = axes[ri, ci]
            _plot_cell(ax, np.asarray(cells[(src, tgt)]["ql"]), ls, int(ell[src]), "tab:orange")
            ax.set_ylim(col_lim[tgt])                                 # shared per target column
            is_diag = src == tgt
            ax.set_title(f"{SHORT[src]} → {SHORT[tgt]}   [{_quadrant(src, tgt)}]\n"
                         f"gap = {gap[(src, tgt)] * 100:+.1f}%   (ℓ_s = {LAYER_LABELS[int(ell[src])]})",
                         fontsize=8.5, fontweight=("bold" if is_diag else "normal"))
            if is_diag:
                for sp in ax.spines.values():
                    sp.set_linewidth(1.8)
                    sp.set_edgecolor("k")
            ax.grid(alpha=0.25)
            if ri == nrow - 1:
                ax.set_xticks(np.arange(N_POINTS))
                ax.set_xticklabels(LAYER_LABELS, rotation=90, fontsize=6)
        axes[ri, 0].set_ylabel(f"probe: {SHORT[src]}\nquantile loss (test)")
    fig.suptitle(f"4×4 {FAMILY.label} transfer — rows = probe source, cols = eval target  "
                 f"[{qset}, {RUNS_TAG}]\n◆ = source-validation-selected layer;  green = source-"
                 "validation first-crossing tunnel;  y shared down each column;  gap = L_{s→t}(ℓ_s)/"
                 "L_{t→t}(ℓ_t) − 1", fontsize=11, y=0.997)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIG_DIR / f"transfer_4x4__{FAMILY.artifact_tag}__{qset}__{RUNS_TAG}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.name}")


def make_pt_ood_figure(sources, targets, cells, ell, tunnels, qset):
    """rows = probe-training PT-ID source, cols = PT-OOD eval target (4×3). Every cell is PT-OOD/
    Probe-OOD: the frozen source probe scored on the target's test split. Each source row shades
    ONLY its OWN source-validation first-crossing tunnel entrance (never the other sources'), and
    marks its own source-val-selected layer with ◆. Y-LIMITS SHARED DOWN EACH TARGET COLUMN so
    catastrophic transfer cannot be hidden by per-panel autoscaling."""
    nrow, ncol = len(sources), len(targets)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.9 * nrow), sharex=True, squeeze=False)
    col_lim = {}
    for tgt in targets:
        allv = np.concatenate([np.asarray(cells[(s, tgt)]["ql"]).ravel() for s in sources])
        pad = 0.04 * (allv.max() - allv.min() or 1.0)
        col_lim[tgt] = (allv.min() - pad, allv.max() + pad)
    for ri, src in enumerate(sources):
        ls = int(tunnels[src]["l_start"])                            # ONLY this source's tunnel entrance
        for ci, tgt in enumerate(targets):
            ax = axes[ri, ci]
            _plot_cell(ax, np.asarray(cells[(src, tgt)]["ql"]), ls, int(ell[src]), "tab:purple")
            ax.set_ylim(col_lim[tgt])                                 # shared per target column
            ax.set_title(f"{SHORT[src]} → {SHORT.get(tgt, tgt)}   [{_quadrant(src, tgt)}]\n"
                         f"{SHORT[src]} tunnel [{LAYER_LABELS[ls]}, {REF_LABEL}], ℓ_s = {LAYER_LABELS[int(ell[src])]}",
                         fontsize=8.5)
            ax.grid(alpha=0.25)
            if ri == nrow - 1:
                ax.set_xticks(np.arange(N_POINTS))
                ax.set_xticklabels(LAYER_LABELS, rotation=90, fontsize=6)
        axes[ri, 0].set_ylabel(f"probe: {SHORT[src]}\nquantile loss (test)")
    fig.suptitle(f"Genuine pretraining-OOD transfer (PT-OOD / Probe-OOD): 4 frozen PT-ID "
                 f"{FAMILY.label} probes → 3 PT-OOD targets  [{qset}, {RUNS_TAG}]\n"
                 "◆ = source-validation-selected layer;  green = each row's OWN source-validation "
                 "first-crossing tunnel;  y shared down each column", fontsize=10, y=0.998)
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    out = FIG_DIR / f"transfer_pt_ood__{FAMILY.artifact_tag}__{qset}__{RUNS_TAG}.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.name}")


# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--experiment", default="transfer_4x4", choices=("transfer_4x4", "pt_ood"))
    p.add_argument("--probe-family", default="shared_linear", choices=sorted(PROBE_FAMILIES),
                   help="frozen source probe head: shared_linear (default) or native_mlp (loads the "
                        "fslot_mlp/ checkpoints + tunnels; outputs -> fslot_mlp/{transfer_4x4,ptood_transfer})")
    p.add_argument("--quantile-set", default="q9", choices=sorted(QUANTILE_SETS))
    return p.parse_args(argv)


def main():
    global FAMILY
    args = _parse_args()
    FAMILY = PROBE_FAMILIES[args.probe_family]
    config.set_dataset_set("extended_v3_rolling")     # roster + rolling windows + cache namespace
    _derive_dirs(args.experiment, args.quantile_set)
    quantiles = validate_quantiles(QUANTILE_SETS[args.quantile_set])
    device = "cpu"                                    # predict-only over cached features
    print(f"[run_fslot_transfer] experiment={args.experiment}  family={FAMILY.name}  "
          f"readout={FAMILY.artifact_tag}  qset={args.quantile_set}  {RUNS_TAG}  K={K}  out={OUT_DIR.name}")
    if args.experiment == "transfer_4x4":
        run_4x4(args.quantile_set, quantiles, device)
    else:
        run_pt_ood(args.quantile_set, quantiles, device)


if __name__ == "__main__":
    main()
