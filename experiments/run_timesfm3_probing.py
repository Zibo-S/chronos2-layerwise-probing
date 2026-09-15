"""Layer-wise shared-origin probing of a frozen TimesFM-3  (independent causal prefixes, Q=1).

The TimesFM-3 line of the project. The Chronos-2 drivers and results/ namespaces are untouched.

    At what depth does forecasting information become linearly decodable?

    Chronos-2 : shared linear head across explicit FORECAST SLOTS
    TimesFM-3 : shared linear head across causal FORECAST ORIGINS

    per representation point l in {Emb, L1..L20}:
        one W_l = Linear(1280, 64), Q=1 / tau=0.5, applied with SHARED weights to all
        J=16 origins. Origin j is its OWN forward pass on x[1:32j] -> readout token j-1,
        so its state, detrend coefficients and RevIN statistics depend on x[1:32j] alone.
        TRAIN on every valid origin; REPORT origin 16 (C=512 -> H=64), the real task.

Reused unchanged from Chronos-2: id_data.build_windows (windows, within-series split, series
ids, MASE denominators, seed); probes.chronos2_quantile_loss / _per_window / mean_pinball_loss;
probes.WD_GRID_V2 + the 80/20 carve + AdamW convention; stats' series-level cluster bootstrap;
run_id_forecasting's in-context seasonal-naive MASE (m=24, floor 1e-8).

    WHERE TO RUN: loads a 330M model and runs 16 forward passes per window batch, then fits
    21 x |wd grid| probes. sbatch/salloc on Narval -- NEVER a login node. The lightweight
    contract tests (python -m tests.test_timesfm3_probe) ARE login-node safe.

    python -m experiments.run_timesfm3_probing --help
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
M_SEASON = 24                      # hourly seasonality, as in the Chronos-2 ID pipeline
MASE_DEN_FLOOR = 1e-8


# --------------------------------------------------------------------------- #
# metrics (definitions copied from experiments/run_id_forecasting so both
# models' MASE numbers are computed identically)
# --------------------------------------------------------------------------- #

def mase_denominator(X, m=M_SEASON):
    X64 = np.asarray(X, np.float64)
    return np.abs(X64[:, m:] - X64[:, :-m]).mean(axis=1)


def per_window_mase(y_raw, yhat_raw, den):
    return (np.abs(np.asarray(y_raw, np.float64) - np.asarray(yhat_raw, np.float64))
            / den[:, None]).mean(axis=1)


def cluster_ci(per_window, sid, B, seed, ref_idx):
    """Per-layer point + 95% CI and the paired delta vs the reference layer (L20)."""
    from probing.stats import cluster_bootstrap_apply, cluster_bootstrap_counts, ci_bounds
    W = np.asarray(per_window, np.float64)
    uniq, inv = np.unique(np.asarray(sid), return_inverse=True)
    S, L = len(uniq), W.shape[0]
    per_sum = np.zeros((S, L))
    np.add.at(per_sum, inv, W.T)
    per_cnt = np.bincount(inv, minlength=S).astype(np.float64)
    M = cluster_bootstrap_counts(S, B, seed)             # ONE matrix -> deltas stay paired
    boot = cluster_bootstrap_apply(M, per_sum, per_cnt)
    lo, hi = ci_bounds(boot)
    dlo, dhi = ci_bounds(boot[:, [ref_idx]] - boot)
    point = W.mean(axis=1)
    return {"point": point.tolist(), "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
            "delta_vs_last": (point[ref_idx] - point).tolist(),
            "delta_ci_lo": dlo.tolist(), "delta_ci_hi": dhi.tolist(),
            "reference_layer_index": int(ref_idx), "n_series": int(S),
            "n_windows": int(W.shape[1]), "B": int(B)}


# --------------------------------------------------------------------------- #
# audit table (section 10 of the spec) -- 1-BASED time positions
# --------------------------------------------------------------------------- #

def print_audit(geom, active, path=None):
    rows = geom.audit_rows(active)
    hdr = (f"{'origin':>6} {'raw_prefix_len':>15} {'n_real_ctx_patches':>19} "
           f"{'sel_token_idx':>14} {'target_start':>13} {'target_end':>11} "
           f"{'detrend_active':>15} {'hidden':>9} {'target':>8}")
    print("\n  FIRST-BATCH AUDIT (time positions are 1-BASED; token index is 0-based)")
    print("  " + hdr)
    print("  " + "-" * len(hdr))
    for r in rows:
        print(f"  {r['origin']:>6} {r['raw_prefix_length']:>15} "
              f"{r['num_real_context_patches']:>19} {r['selected_token_index_0based']:>14} "
              f"{r['target_start_1based']:>13} {r['target_end_1based']:>11} "
              f"{r['detrend_active']:>14.1%} {str(r['hidden_shape']):>9} "
              f"{str(r['transformed_target_shape']):>8}")
    if path is not None:
        Path(path).write_text(json.dumps(rows, indent=2, default=str))
        print(f"  [saved]      {path.name}")
    return rows


# --------------------------------------------------------------------------- #
# one dataset
# --------------------------------------------------------------------------- #

def run_dataset(tag, args, geom, paths, model, device):
    from probing.id_data import build_windows
    from probing.timesfm3 import (LAST_LAYER, LAYER_NAMES, NUM_LAYERS, build_prefix_targets,
                                  cached_prefix_features, denormalize,
                                  prefix_vs_fullwindow_diagnostic, raw_future_from_arcsinh)
    from probing.timesfm3_probes import native_median_reference, shared_origin_layerwise

    print(f"\n{'=' * 82}\n[{tag}]\n{'=' * 82}")
    t0 = time.time()
    w = build_windows(tag, C=args.context_len, H=args.horizon, stride=args.stride,
                      seed=args.seed)
    meta = w["meta"]
    print(f"  windows: {meta['n_train']} train / {meta['n_test']} test  "
          f"({meta['split_mode']}, {meta['n_test_series']} test series)")

    layers = list(range(NUM_LAYERS)) if args.layers is None else sorted(set(args.layers))
    if LAST_LAYER not in layers:
        layers = sorted(layers + [LAST_LAYER])
        print(f"  [note] added L{LAST_LAYER} to --layers (native reference + delta baseline)")
    origins = list(geom.origins_1based) if args.origins is None else sorted(set(args.origins))
    if geom.headline_origin_1based not in origins:
        raise SystemExit(f"--origins must include the headline origin "
                         f"{geom.headline_origin_1based}")

    ex = dict(geom=geom, model=model, device=device, batch_size=args.extract_batch_size,
              layers=layers, feature_dtype=np.dtype(args.feature_dtype),
              detrend=not args.no_detrend, origins=origins,
              cache_dir=paths["cache"], checkpoint=args.checkpoint, seed=args.seed,
              force=args.force_extract)
    tr = cached_prefix_features(tag, "train", w["X_train"], **ex)
    te = cached_prefix_features(tag, "test", w["X_test"], **ex)

    Ztr = np.concatenate([w["X_train"],
                          raw_future_from_arcsinh(w["X_train"], w["Y_train_traj"],
                                                  meta["sigma_eps"])], axis=1)
    Zte = np.concatenate([w["X_test"],
                          raw_future_from_arcsinh(w["X_test"], w["Y_test_traj"],
                                                  meta["sigma_eps"])], axis=1)
    ptr = build_prefix_targets(Ztr, tr["mu"], tr["sd"], geom, detrend=not args.no_detrend)
    pte = build_prefix_targets(Zte, te["mu"], te["sd"], geom, detrend=not args.no_detrend)
    valid_tr, valid_te = ptr["valid"].copy(), pte["valid"].copy()
    keep = np.zeros(geom.J, bool)
    keep[[j - 1 for j in origins]] = True
    valid_tr &= keep[None, :]
    valid_te &= keep[None, :]

    if not args.no_audit:
        print_audit(geom, ptr["active"], paths["out"] / f"audit__{tag}.json")
    print(f"\n  detrending fires per origin (train): " +
          " ".join(f"{j}:{ptr['active'][:, j-1].mean():.0%}" for j in geom.origins_1based))
    print(f"  usable origins: train {valid_tr.mean():.4f}  test {valid_te.mean():.4f}  "
          f"(headline: {valid_tr[:, -1].mean():.4f} / {valid_te[:, -1].mean():.4f})")

    hz = geom.headline_origin_1based - 1
    rows = np.flatnonzero(valid_te[:, hz])
    # guard: inverting the TARGET must reproduce the raw future, per dataset
    rt = denormalize(pte["targets"][rows, hz, :], mu=te["mu"][rows, hz],
                     sd=te["sd"][rows, hz], trend=pte["trend"][rows, hz, :])
    y_raw = Zte[rows, geom.target_start[hz]:geom.target_end[hz]]
    rel = float(np.abs(rt - y_raw).max() / (np.abs(y_raw).mean() + 1e-12))
    if rel > 1e-4:
        raise RuntimeError(f"{tag}: inverting the headline target misses the raw future by "
                           f"relative {rel:.3e} -- the per-origin RevIN/trend algebra is wrong")
    print(f"  target -> raw future round-trip: relative max|d| = {rel:.2e}")

    scores, diag = shared_origin_layerwise(
        tr["feats"], ptr["targets"], valid_tr, te["feats"], pte["targets"], valid_te,
        H=geom.H, epochs=args.probe_epochs, lr=args.probe_lr,
        wd_grid=None if args.no_wd_grid else tuple(args.wd_grid), device=device,
        batch_size=args.probe_batch_size, layers=layers, headline_origin_idx=hz,
        collect_history=args.collect_history)

    # ---- MASE in RAW units at the headline origin + native tau=0.5 baseline ----
    den = mase_denominator(w["X_test"][rows])
    n_clamped = int((den < MASE_DEN_FLOOR).sum())
    den = np.maximum(den, MASE_DEN_FLOOR)
    mase_curve, mase_pw = [], []
    for i in layers:
        yhat = denormalize(diag["test_median_pred"][i], mu=te["mu"][rows, hz],
                           sd=te["sd"][rows, hz], trend=pte["trend"][rows, hz, :])
        pw = per_window_mase(y_raw, yhat, den)
        mase_pw.append(pw)
        mase_curve.append(float(pw.mean()))
    mase_pw = np.stack(mase_pw)

    nat_loss, nat_pw, nat_mp, nat_med = native_median_reference(
        te["native"], te["mu"], te["sd"], pte["trend"], pte["targets"], valid_te, hz)
    nat_mase_pw = per_window_mase(y_raw, nat_med, den)

    sid = np.asarray(w["series_test"], np.int64)[rows]
    loss_pw = np.stack([diag["test_window_loss"][i] for i in layers])
    ref = layers.index(LAST_LAYER)
    # keys match the entry's metric names; the bootstrap-INPUT npz keeps the Chronos-2
    # schema names (window_loss__* / reported.quantile_loss) for tooling parity
    boot = {"median_loss": cluster_ci(loss_pw, sid, args.boot_b, args.seed, ref),
            "mase_context": cluster_ci(mase_pw, sid, args.boot_b, args.seed, ref)}

    val = [diag["val_loss"][i] for i in layers]
    star = int(layers[int(np.argmin(val))]) if all(v is not None for v in val) else None
    si = layers.index(star) if star is not None else ref

    entry = {
        "tag": tag, "layers": layers, "layer_names": [LAYER_NAMES[i] for i in layers],
        "geometry": geom.as_dict(), "window_meta": meta, "origins_used": origins,
        "quantiles": [0.5], "num_quantiles": 1,
        "median_loss": [scores[i] for i in layers],
        "mean_pinball": [diag["test_mean_pinball"][i] for i in layers],
        "mase_context": mase_curve,
        "median_loss_all_origins": [diag["test_loss_all_origins"][i] for i in layers],
        "median_loss_by_origin": {str(i): diag["test_loss_by_origin"][i] for i in layers},
        "val_loss": val, "val_selected_layer": star,
        "weight_decay": [diag["wd"][i] for i in layers],
        "native": {"median_loss": nat_loss, "mean_pinball": nat_mp,
                   "mase_context": float(nat_mase_pw.mean())},
        "l20_vs_native": {"probe_median_loss": scores[LAST_LAYER],
                          "native_median_loss": nat_loss,
                          "gap": scores[LAST_LAYER] - nat_loss,
                          "probe_mase": mase_curve[ref],
                          "native_mase": float(nat_mase_pw.mean())},
        "bootstrap": boot,
        "detrend_fire_rate_by_origin": {j: float(ptr["active"][:, j - 1].mean())
                                        for j in geom.origins_1based},
        "target_roundtrip_relative_err": rel,
        "native_median_reconstruction": te["native_median_err"],
        "n_test_windows_scored": int(len(rows)), "n_denominator_clamped": n_clamped,
        "seconds": round(time.time() - t0, 1),
    }
    if args.collect_history:
        entry["history"] = {str(i): diag["history"][i] for i in layers}
    if args.diagnostic_prefix_vs_full:
        entry["prefix_vs_fullwindow"] = prefix_vs_fullwindow_diagnostic(
            model, geom, device, w["X_test"][:8], detrend=not args.no_detrend)

    save_bootstrap_inputs(tag, entry, sid, loss_pw, mase_pw, nat_mase_pw, nat_pw, layers,
                          paths, args)
    print(f"  L*={LAYER_NAMES[star] if star is not None else '?'}  "
          f"loss(L*)={entry['median_loss'][si]:.4f}  "
          f"loss(L20)={scores[LAST_LAYER]:.4f}  native={nat_loss:.4f}")
    print(f"  MASE  L*={mase_curve[si]:.4f}  L20={mase_curve[ref]:.4f}  "
          f"native={nat_mase_pw.mean():.4f}   [{entry['seconds']}s]")
    return entry


def save_bootstrap_inputs(tag, entry, sid, loss_pw, mase_pw, nat_mase_pw, nat_loss_pw,
                          layers, paths, args):
    """Per-window test metrics in the Chronos-2 bootstrap-input schema (readout
    ``origin_last``). ``num_layers`` is in meta because it is 21 here and 13 for Chronos-2."""
    arrays = {"series_test": sid,
              "window_loss__origin_last": np.asarray(loss_pw, np.float64),
              "window_mase_context__origin_last": np.asarray(mase_pw, np.float64),
              "window_loss__native": np.asarray(nat_loss_pw, np.float64),
              "window_mase_context__native": np.asarray(nat_mase_pw, np.float64)}
    for k, a in arrays.items():
        if not np.isfinite(a).all():
            raise RuntimeError(f"{tag}: non-finite values in {k}")
    meta = {"model": "timesfm-3.0", "prefix_extraction": True, "tag": tag,
            "H": entry["geometry"]["H"], "J": entry["geometry"]["J"],
            "num_layers": len(layers), "layers": layers, "layer_names": entry["layer_names"],
            "last_layer": layers[-1], "quantiles": [0.5], "num_quantiles": 1,
            "split_mode": entry["window_meta"]["split_mode"],
            "n_test_windows": int(len(sid)), "n_test_series": int(len(np.unique(sid))),
            "primary_readouts": ["origin_last"], "controlled_readouts": [],
            "headline_origin_1based": entry["geometry"]["headline_origin_1based"],
            "mase_definition": f"mase_context: in-context seasonal-naive (m={M_SEASON}, "
                               f"floor {MASE_DEN_FLOOR})",
            "seasonal_m": M_SEASON,
            "val_selected_layer": {"origin_last": entry["val_selected_layer"]},
            "val_loss_by_layer": {"origin_last": entry["val_loss"]},
            "reported": {"quantile_loss": {"origin_last": entry["median_loss"]},
                         "mase_context": {"origin_last": entry["mase_context"]},
                         "native_mase_context": entry["native"]["mase_context"],
                         "native_median_loss": entry["native"]["median_loss"]}}
    out = paths["boot"] / "inputs" / f"{tag}__H{meta['H']}_J{meta['J']}_q1.npz"
    np.savez(out, meta=json.dumps(meta), **arrays)
    print(f"  [saved]      {out.relative_to(paths['out'])}")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #

def make_figures(summary, paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = list(summary["datasets"].values())
    if not ds:
        return
    n = len(ds)
    for metric, ylab, natk in [
            ("median_loss", "median pinball loss (Q=1, tau=0.5)", "median_loss"),
            ("mase_context", f"MASE (in-context seasonal-naive, m={M_SEASON})", "mase_context")]:
        fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
        for a, e in zip(ax[0], ds):
            L, b = e["layers"], e["bootstrap"][metric]
            a.plot(L, e[metric], "o-", ms=3, color="#1f77b4", label="shared-origin probe")
            a.fill_between(L, b["ci_lo"], b["ci_hi"], alpha=0.2, color="#1f77b4")
            a.axhline(e["native"][natk], ls="--", c="crimson", lw=1.2, label="native TimesFM-3")
            if e["val_selected_layer"] is not None:
                a.axvline(e["val_selected_layer"], ls=":", c="k", lw=1,
                          label=f"L* = {e['layer_names'][L.index(e['val_selected_layer'])]}")
            a.set_title(e["tag"], fontsize=10)
            a.set_xlabel("representation point (0 = Emb, 20 = L20)")
            a.grid(alpha=0.3)
        ax[0][0].set_ylabel(ylab)
        ax[0][0].legend(fontsize=7)
        fig.suptitle("TimesFM-3 shared-origin probe (independent causal prefixes), "
                     "evaluated at origin 16: C=512 -> H=64", fontsize=11)
        fig.tight_layout()
        p = paths["fig"] / f"{metric}_by_layer.png"
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"  [figure]     {p.relative_to(paths['out'])}")

    fig, ax = plt.subplots(1, n, figsize=(4.2 * n, 3.8), squeeze=False)
    for a, e in zip(ax[0], ds):
        L, b = e["layers"], e["bootstrap"]["median_loss"]
        a.plot(L, b["delta_vs_last"], "o-", ms=3, color="#2ca02c")
        a.fill_between(L, b["delta_ci_lo"], b["delta_ci_hi"], alpha=0.2, color="#2ca02c")
        a.axhline(0, c="k", lw=0.8)
        a.set_title(e["tag"], fontsize=10)
        a.set_xlabel("representation point")
        a.grid(alpha=0.3)
    ax[0][0].set_ylabel("loss(L20) - loss(L)   (+ = earlier layer better)")
    fig.suptitle("Delta vs L20, series-level cluster bootstrap 95% CI", fontsize=11)
    fig.tight_layout()
    p = paths["fig"] / "median_loss_delta_vs_last.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  [figure]     {p.relative_to(paths['out'])}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TimesFM-3 layer-wise shared-origin probing (causal prefixes, Q=1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("model / environment (Narval: point these at $SCRATCH)")
    g.add_argument("--checkpoint", default=os.environ.get("TIMESFM3_CHECKPOINT",
                                                          "google/timesfm-3.0-pytorch"),
                   help="HF repo id, or a LOCAL dir with config.json + model.safetensors "
                        "(use the local form on offline compute nodes)")
    g.add_argument("--hf-home", default=None, help="sets HF_HOME before anything is loaded")
    g.add_argument("--cache-dir", default=os.environ.get("TFM3_CACHE_DIR", None),
                   help="feature cache root [default: <repo>/features_cache]")
    g.add_argument("--out-root", default=os.environ.get("TFM3_OUT_ROOT", None),
                   help="results root [default: <repo>/results]")
    g.add_argument("--device", default=os.environ.get("TFM3_DEVICE", None),
                   help="cuda | mps | cpu [default: auto]")

    g = p.add_argument_group("data")
    g.add_argument("--dataset-set", default=os.environ.get("ID_DATASET_SET", "extended_v1"))
    g.add_argument("--datasets", nargs="+", default=None)
    g.add_argument("--context-len", type=int, default=512)
    g.add_argument("--horizon", type=int, default=64)
    g.add_argument("--stride", type=int, default=64)

    g = p.add_argument_group("extraction")
    g.add_argument("--extract-batch-size", type=int, default=64,
                   help="windows per decode() call; prefixes are bucketed by length")
    g.add_argument("--feature-dtype", default="float16", choices=["float16", "float32"])
    g.add_argument("--force-extract", action="store_true")
    g.add_argument("--layers", type=int, nargs="+", default=None, help="default: 0..20")
    g.add_argument("--origins", type=int, nargs="+", default=None,
                   help="1-based origins to TRAIN on; must include 16. Use e.g. "
                        "--origins 8 9 10 11 12 13 14 15 16 to test sensitivity to the "
                        "short-prefix context-length shift")
    g.add_argument("--no-detrend", action="store_true",
                   help="ablation: disable TimesFM-3's linear detrending in the BACKBONE and "
                        "in the targets (makes every origin trivially causal)")

    g = p.add_argument_group("probe (Q=1, tau=0.5)")
    g.add_argument("--probe-epochs", type=int, default=300)
    g.add_argument("--probe-lr", type=float, default=1e-2)
    g.add_argument("--probe-batch-size", type=int, default=0,
                   help="0 = full batch (the Chronos-2 protocol)")
    g.add_argument("--wd-grid", type=float, nargs="+",
                   default=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0],
                   help="probes.WD_GRID_V2")
    g.add_argument("--no-wd-grid", action="store_true")
    g.add_argument("--collect-history", action="store_true")

    g = p.add_argument_group("evaluation / checks")
    g.add_argument("--boot-b", type=int, default=5000)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--skip-runtime-checks", action="store_true",
                   help="skip the causality + batching checks (NOT recommended)")
    g.add_argument("--diagnostic-prefix-vs-full", action="store_true")
    g.add_argument("--no-audit", action="store_true")
    g.add_argument("--tag", default=None, help="extra suffix on the output directory")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser())
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    import torch
    from probing import config
    from probing.id_data import ID_DATASET_SPECS
    from probing.timesfm3 import (CACHE_VERSION, LAST_LAYER, NUM_LAYERS, PrefixGeometry,
                                  assert_batching_invariant, assert_causal_prefixes,
                                  get_model, timesfm_version)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    config.set_dataset_set(args.dataset_set)
    tags = args.datasets or list(ID_DATASET_SPECS[args.dataset_set])
    unknown = [t for t in tags if t not in ID_DATASET_SPECS[args.dataset_set]]
    if unknown:
        raise SystemExit(f"unknown dataset tag(s) {unknown} for set {args.dataset_set}")

    out_root = Path(args.out_root).expanduser() if args.out_root else REPO_ROOT / "results"
    name = f"timesfm3_{args.dataset_set}_q1" + (f"_{args.tag}" if args.tag else "")
    out = out_root / name
    paths = {"out": out, "boot": out / "bootstrap", "fig": out / "figures",
             "cache": (Path(args.cache_dir).expanduser() if args.cache_dir
                       else REPO_ROOT / "features_cache")}
    (paths["boot"] / "inputs").mkdir(parents=True, exist_ok=True)
    paths["fig"].mkdir(parents=True, exist_ok=True)
    paths["cache"].mkdir(parents=True, exist_ok=True)

    geom = PrefixGeometry(args.context_len, args.horizon)
    model, device = get_model(args.checkpoint, args.device)
    print(f"TimesFM-3 shared-origin probing (independent causal prefixes, Q=1 / tau=0.5)")
    print(f"  checkpoint : {args.checkpoint}  (timesfm {timesfm_version()}, device {device})")
    print(f"  geometry   : J={geom.J} origins, prefixes {geom.prefix_len.tolist()}")
    print(f"  headline   : origin {geom.headline_origin_1based} (C={geom.C} -> H={geom.H})")
    print(f"  cache      : {paths['cache']}  ({CACHE_VERSION})")
    print(f"  out        : {out}")

    if not args.skip_runtime_checks:
        print("\n  runtime correctness checks")
        rep = assert_causal_prefixes(model, geom, device, detrend=not args.no_detrend)
        print(f"    strict causality : max|dh| = "
              f"{max(v['max_abs_dh'] for v in rep.values()):.3e}  over origins "
              f"{sorted(rep)}  [must be 0]")
        bi = assert_batching_invariant(model, geom, device, detrend=not args.no_detrend)
        print(f"    batching         : semantic {bi['semantic_max_abs']:.3e} [must be 0], "
              f"numeric {bi['numeric_max_abs']:.3e} ({bi['numeric_relative']:.1e} rel)")

    summary = {"config": {**vars(args), "num_layers": NUM_LAYERS, "last_layer": LAST_LAYER,
                          "m_season": M_SEASON, "cache_version": CACHE_VERSION,
                          "timesfm_version": timesfm_version(), "device": device,
                          "geometry": geom.as_dict()},
               "datasets": {}}
    for tag in tags:
        summary["datasets"][tag] = run_dataset(tag, args, geom, paths, model, device)

    sp = out / "timesfm3_probing_summary.json"
    sp.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[saved] {sp}")
    make_figures(summary, paths)

    print(f"\n{'=' * 82}\nSANITY CHECK - L20 probe vs the native tau=0.5 head\n"
          f"(the native median IS a linear function of h_L20, so the probe's hypothesis\n"
          f" class contains it; the trained result is reported as measured)\n{'=' * 82}")
    print(f"{'dataset':<30}{'probe loss':>12}{'native':>10}{'gap':>9}"
          f"{'probe MASE':>12}{'native':>9}")
    for tag, e in summary["datasets"].items():
        c = e["l20_vs_native"]
        print(f"{tag:<30}{c['probe_median_loss']:>12.4f}{c['native_median_loss']:>10.4f}"
              f"{c['gap']:>9.4f}{c['probe_mase']:>12.4f}{c['native_mase']:>9.4f}")
    return summary


if __name__ == "__main__":
    main()
