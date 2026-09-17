"""Can a LINEAR map put h_l into the coordinate system TimesFM-3's frozen head expects?

The fourth paper7 analysis. Three facts already established for TimesFM-3 motivate it:

    1. a learned linear probe decodes good forecasts from INTERMEDIATE layers;
    2. the frozen native head applied DIRECTLY to those same layers is catastrophic until the
       last few (e.g. Electricity at its 5% tunnel entrance L15: probe 0.167 vs head 85.79);
    3. CKA and effective rank show the representation keeps changing long after the forecast
       becomes linearly decodable.

Together those say the information is present but not expressed in the readout's basis. This
run tests that directly, by fitting -- WITHOUT EVER SEEING THE FUTURE -- an affine map

    A_l : R^1280 -> R^1280 ,    A_l h_{l,15} + b_l  ~  h_{20,15}

and then pushing the result through the pretrained frozen head:

    h_{l,15} -> A_l h_{l,15} + b_l -> W_native -> decode()'s inverse path -> y_hat

THE ADAPTER IS NEVER TRAINED ON FORECAST TARGETS. Since W_native is itself linear, an adapter
fit on forecast loss would collapse to another linear forecasting map W_native A_l -- the
learned probe in disguise. The objective here is representation reconstruction only:

    min_{A,b} ||X_l A + 1 b^T - H_20||_F^2 + lambda ||A||_F^2,   bias unpenalized, float64,
    lambda selected on VALIDATION REPRESENTATION MSE, never on any forecast quantity.

Four curves per dataset, all on the same paper7 test windows and the same Q=9 objective:

    W_native h_l                    direct frozen head            (committed reference + recomputed)
    W_native (A_l h_l + b_l)        representation-aligned head   THIS RUN
    B_l h_l                         learned linear probe          (merged from the probe run)
    W_native h_20                   the endpoint                  (two variants, see below)

TWO ENDPOINT BASELINES, BOTH REPORTED
-------------------------------------
    cached_L20_native_head   the frozen head applied to the CACHED L20 representation through
                             the SAME (N, 1280) path every aligned representation takes. This is
                             the internal, apples-to-apples endpoint for alignment, and it is
                             what A_20 = I, b_20 = 0 produces by construction.
    official_native_decode   decode()'s own forecast, cached at extraction time and scored by the
                             same ``native_reference``. The true model baseline.

They are close but NOT bit-identical, and are not required to be: the head applied to a
(N, 1280) matrix accumulates differently than inside decode()'s (b, 1, 18, 1280) pass (~1e-6
relative, measured by the native-head run's own slice-order control). A synthesized A h + b has
no token sequence to sit in, so the cached path is the only possible one for the aligned curves
-- and the endpoint is therefore defined to travel it too. Their difference is recorded per
dataset; a tiny discrepancy is a numerical provenance fact, not an adapter failure.

NO BACKBONE FORWARD PASS. Representations come from the validated float32 last-token cache; the
checkpoint is loaded for ``model.output_head`` (and value_clip / patch geometry) alone. CPU is
enough. The head's parameter checksum is taken before and after and must be identical.

This run does NOT define a tunnel entrance: the 5% entrance is read from the probe run purely as
an overlay and a table index. No Chronos-2 file, probe result or geometry output is modified.

Run:
    sbatch job_timesfm3_alignment.sh
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_timesfm3_last_token_probing import (KIND, PAPER7, SHORT,  # noqa: E402
                                                         assert_window_parity, chronos_reference,
                                                         print_roster_audit, suite_tags,
                                                         windows_for)
from experiments.run_timesfm3_native_head_transfer import (SLUG, _panel_grid,  # noqa: E402
                                                           head_checksum)
from experiments.run_timesfm3_probing import (MASE_DEN_FLOOR, M_SEASON,  # noqa: E402
                                              cluster_ci, mase_denominator, per_window_mase)
from probing.timesfm3_alignment import (ALIGNMENT_VERSION, ALIGNMENTS,  # noqa: E402
                                        RIDGE_GRID, SOLVE_DTYPE, apply_alignment,
                                        assert_no_forecast_targets, fit_layer_alignment,
                                        frozen_head_forecast, identity_alignment,
                                        load_native_context, load_last_token_reps,
                                        mean_baseline_metrics, permuted_rows,
                                        representation_metrics, select_lambda,
                                        RidgeAlignmentSolver)
from probing.timesfm3_geometry import git_commit, window_identity_hash  # noqa: E402
from probing.timesfm3_last_token import (LAST_LAYER, LAYER_NAMES, MODEL_DIMS,  # noqa: E402
                                         NATIVE_QUANTILES, NUM_LAYERS, NUM_QUANTILES,
                                         LastTokenGeometry, assert_target_roundtrip,
                                         build_last_token_targets, raw_future_from_arcsinh)

# the committed frozen-head + probe curves this run merges against (written by
# run_timesfm3_native_head_transfer.py, in the repo, on the SAME paper7 test windows)
DEFAULT_REFERENCE = (REPO_ROOT / "results" / "timesfm3_representation_geometry"
                     / "native_head_transfer_summary.json")
PAPER_OUT = REPO_ROOT / "results" / "timesfm3_representation_alignment"


def paper_out_default() -> Path:
    """Repo-local home for the lightweight paper outputs of THIS experiment line.

    One top-level directory per experiment line is the repo's convention
    (results/ext_v5_native_head_adapter, results/timesfm3_representation_geometry, ...). This
    line FITS something, which the geometry line deliberately does not, so it gets its own
    namespace rather than a subfolder of it. Heavy artifacts (adapter matrices, per-layer
    arrays) go to --out-root on $SCRATCH and never into git.
    """
    return PAPER_OUT


# --------------------------------------------------------------------------- #
# the merged reference curves
# --------------------------------------------------------------------------- #
def load_reference(path, probe_path=None) -> dict:
    """Per-dataset direct-head / learned-probe / native / tunnel curves, merged by LAYER.

    Source of truth is the committed native-head-transfer summary: it holds the frozen-head
    curve measured in memory on the GPU, the probe curve it was already joined against, the
    native decode baseline and the 5% tunnel entrance -- all on these exact test windows.
    ``--probe-results`` may additionally point at the probe summary itself, which then supplies
    the probe curve and tunnel entrance directly (and is cross-checked against the merged copy).
    """
    out = {"path": str(path), "available": False, "datasets": {}, "probe_path": str(probe_path)
           if probe_path else None}
    p = Path(path)
    if not p.exists():
        out["note"] = (f"no reference results at {p}; the aligned curve will be saved alone, "
                       "without the direct-head / probe / tunnel overlays")
        return out
    d = json.loads(p.read_text())
    for tag, e in d.get("datasets", {}).items():
        rows = {r["layer"]: r for r in (e.get("combined_with_probe", {}).get("rows") or [])}
        out["datasets"][tag] = {
            "direct_native_head_q9_loss": {int(x["layer"]): float(x["native_head_q9_pinball"])
                                           for x in e["per_layer"]},
            "direct_native_head_mase": {int(x["layer"]): float(x["native_head_mase"])
                                        for x in e["per_layer"]},
            "learned_probe_q9_loss": {int(l): float(r["probe_q9_loss"])
                                      for l, r in rows.items()},
            "native_timesfm_q9_loss": float(e["native_decode_baseline"]["q9_loss"]),
            "native_timesfm_mase": float(e["native_decode_baseline"].get("mase_context",
                                                                        float("nan"))),
            "tunnel_entrance_5pct": e.get("combined_with_probe", {}).get("tunnel_entrance_5pct"),
            "tunnel_entrance_5pct_name": e.get("combined_with_probe", {}).get(
                "tunnel_entrance_5pct_name"),
            "window_identity_hash": e.get("window_identity_hash"),
            "n_test": e.get("N")}
    out["available"] = bool(out["datasets"])
    out["source"] = ("results/timesfm3_representation_geometry/native_head_transfer_summary.json "
                     "-- the frozen-head curve measured IN MEMORY on the GPU from decode()'s own "
                     "forward pass, joined with the learned Q=9 probe curve")
    if probe_path:
        pr = json.loads(Path(probe_path).read_text())
        for tag, e in (pr.get("datasets") or {}).items():
            if tag not in out["datasets"]:
                continue
            lay = [int(x) for x in e["layers"]]
            out["datasets"][tag]["learned_probe_q9_loss"] = {
                l: float(v) for l, v in zip(lay, e["test_q9_loss"])}
            tun = (e.get("tunnel_by_tolerance") or {}).get("0.05") or e.get("tunnel") or {}
            out["datasets"][tag]["tunnel_entrance_5pct"] = tun.get("layer")
            out["datasets"][tag]["tunnel_entrance_5pct_name"] = tun.get("layer_name")
            out["datasets"][tag]["probe_source"] = "probe summary (--probe-results)"
    return out


# --------------------------------------------------------------------------- #
# forecasting evaluation of ONE (N, 1280) representation
# --------------------------------------------------------------------------- #
def score_representation(model, states, ctx, pte, geom, y_raw, den, rows, *, batch_size,
                         device) -> dict:
    """states (N, 1280) -> frozen head -> the Q=9 / median / MAE / MASE row of the tables.

    ``native_reference`` is the SAME scorer the probe run and the frozen-head run use, on the
    same normalized axis, so every curve in the figures is directly comparable.
    """
    from probing.timesfm3_last_token_probes import native_reference
    fc = frozen_head_forecast(model, states, ctx["mu"], ctx["sd"], pte["trend"], geom,
                              batch_size=batch_size, device=device)
    r = native_reference(fc, ctx["mu"], ctx["sd"], pte["trend"], pte["targets"], pte["valid"])
    if not np.array_equal(r["rows"], rows):
        raise RuntimeError("a representation was scored on different test windows than the "
                           "native reference -- the per-window arrays would not be paired")
    mp = per_window_mase(y_raw, r["median_raw"], den)
    return {"q9_loss": r["q9_loss"], "median_pinball": r["median_loss"],
            # rho_0.5(u) = 0.5|u|, so the median pinball loss is exactly half the MAE
            "mae": 2.0 * r["median_loss"], "mase": float(mp.mean()),
            "per_quantile_pinball": r["per_quantile"],
            "q9_window": r["q9_window"], "median_window": r["median_window"], "mase_window": mp}


def recovery(direct, aligned, endpoint, *, floor: float = 1e-9) -> dict:
    """R_align = (L_direct - L_aligned) / (L_direct - L_endpoint). DIAGNOSTIC ONLY.

    0 = alignment bought nothing over the direct frozen head; 1 = it reached the endpoint;
    >1 = it beat it. The denominator is guarded: where the direct head is already at the
    endpoint there is nothing to recover and the ratio is undefined, reported as None with the
    reason rather than as a large number produced by dividing by noise. Never used for model
    selection -- lambda is chosen on validation REPRESENTATION error only.
    """
    gap = float(direct) - float(endpoint)
    scale = max(abs(float(direct)), abs(float(endpoint)), 1.0)
    if not np.isfinite(gap) or abs(gap) < floor * scale:
        return {"value": None, "denominator": gap,
                "reason": "the direct frozen head is already at the endpoint (denominator ~ 0); "
                          "there is no gap to recover"}
    return {"value": (float(direct) - float(aligned)) / gap, "denominator": gap, "reason": None}


# --------------------------------------------------------------------------- #
# one dataset
# --------------------------------------------------------------------------- #
def run_dataset(tag, args, geom, model, device, ref_entry, adapter_dir) -> dict:
    from probing.timesfm3_last_token_probes import native_reference

    t0 = time.time()
    short, kind = SHORT.get(tag, tag), KIND.get(tag, "unclassified")
    print(f"\n{'=' * 84}\n[{tag}]  {short}  ({kind})\n{'=' * 84}")

    # ---- windows: the committed Chronos-2 / TimesFM-3 paper7 ones, verified ----
    w = windows_for(tag, args.suite, args)
    ident = assert_window_parity(tag, w, chronos_reference(tag),
                                 strict=not args.allow_window_mismatch)
    meta = w["meta"]
    if "X_val" not in w or len(w["X_val"]) == 0:
        raise RuntimeError(
            f"{tag} has no dedicated validation split. lambda is selected on VALIDATION "
            "representation error; an 80/20 carve of train would change the protocol relative "
            "to the probe run's explicit-val contract, so this run refuses to improvise one.")
    print(f"  windows: {ident['n_train_windows']} train / {ident['n_val_windows']} val / "
          f"{ident['n_test_windows']} test  ({ident['n_test_series']} test "
          f"{ident['cluster_unit']})   [Chronos-2 parity: {ident.get('chronos_parity')}]")

    layers = list(range(NUM_LAYERS)) if args.layers is None else sorted(set(args.layers))
    if LAST_LAYER not in layers:
        layers = sorted(layers + [LAST_LAYER])
        print(f"  [note] added L{LAST_LAYER}: it is the alignment TARGET and the endpoint")
    fit_layers = [l for l in layers if l != LAST_LAYER]

    # ---- representations: cache only, and through the loader that DISCARDS mu/sd/native ----
    # The cache's metadata records the EXACT layer list it was written with, and ``read_cache``
    # diffs EVERY metadata field -- so asking for a subset of a 21-point cache is rejected as
    # incompatible. Always read the full cache (the per-layer arrays are memmapped, so unused
    # points cost nothing) and select the requested points by position afterwards. ``--layers``
    # then means "fit and report these", never "read a different cache".
    cache_layers = list(range(NUM_LAYERS))
    lo = dict(cache_dir=args.cache_dir, geom=geom, layers=cache_layers,
              checkpoint=args.checkpoint, suite=args.suite, seed=args.seed,
              detrend=not args.no_detrend, feature_dtype=np.dtype(args.feature_dtype),
              ignore_timesfm_version=args.ignore_timesfm_version)
    R = {s: load_last_token_reps(tag, s, w[f"X_{s}"], series_ids=w.get(f"series_{s}"), **lo)
         for s in ("train", "val", "test")}
    for s, r in R.items():
        print(f"  [cache] {s:<5} {r['n']:>5} x {tuple(np.shape(r['reps'][0]))}  "
              f"{len(r['layers'])} points cached, {len(layers)} in use   "
              f"id {r['window_identity_hash']}")
    hashes = {s: r["window_identity_hash"] for s, r in R.items()}
    if len(set(hashes.values())) != 3:
        raise RuntimeError(f"two splits of {tag} have the same window identity hash {hashes} -- "
                           "train/val/test are not distinct window sets")

    pos = {l: cache_layers.index(l) for l in layers}
    Y = {s: np.asarray(R[s]["reps"][pos[LAST_LAYER]], SOLVE_DTYPE) for s in R}   # h_L20

    # ---- the scoring side of the SAME verified cache read (test split only) ----
    ctx = load_native_context(tag, "test", w["X_test"], cache_dir=args.cache_dir, geom=geom,
                              checkpoint=args.checkpoint, suite=args.suite, seed=args.seed,
                              detrend=not args.no_detrend,
                              feature_dtype=np.dtype(args.feature_dtype), layers=cache_layers,
                              ignore_timesfm_version=args.ignore_timesfm_version)

    # ---- targets, built ONLY for scoring; never handed to an adapter fit ----
    Zte = np.concatenate([w["X_test"],
                          raw_future_from_arcsinh(w["X_test"], w["Y_test_traj"],
                                                  meta["sigma_eps"])], axis=1)
    pte = build_last_token_targets(Zte, ctx["mu"], ctx["sd"], geom, detrend=not args.no_detrend)
    rt = assert_target_roundtrip(Zte, pte["targets"], pte["trend"], ctx["mu"], ctx["sd"], geom,
                                 pte["valid"], rtol=args.roundtrip_rtol)
    print(f"  target round-trip: {rt:.2e}  [< {args.roundtrip_rtol}]")

    official = native_reference(ctx["native"], ctx["mu"], ctx["sd"], pte["trend"],
                                pte["targets"], pte["valid"])
    rows = official["rows"]
    y_raw = Zte[rows, geom.target_start:geom.target_end]
    den = mase_denominator(w["X_test"][rows])
    n_clamped = int((den < MASE_DEN_FLOOR).sum())
    den = np.maximum(den, MASE_DEN_FLOOR)
    official_mase = per_window_mase(y_raw, official["median_raw"], den)

    sc = dict(batch_size=args.head_batch_size, device=device)
    # ---- curve 1: the frozen head applied DIRECTLY to each cached representation ----
    print(f"  direct frozen head on the cached representations ({len(layers)} points)...",
          flush=True)
    direct = {l: score_representation(model, np.asarray(R["test"]["reps"][pos[l]], SOLVE_DTYPE),
                                      ctx, pte, geom, y_raw, den, rows, **sc) for l in layers}
    endpoint = direct[LAST_LAYER]                      # cached_L20_native_head
    d_end = endpoint["q9_loss"] - official["q9_loss"]
    rel_end = abs(d_end) / max(abs(official["q9_loss"]), 1e-12)
    print(f"  endpoints:  cached_L20_native_head Q9 {endpoint['q9_loss']:.6f}   "
          f"official_native_decode {official['q9_loss']:.6f}   "
          f"delta {d_end:+.2e} (relative {rel_end:.2e})")
    if rel_end > args.endpoint_report_rtol:
        print(f"  [note] the two endpoints differ by more than {args.endpoint_report_rtol:g} "
              "relative. Both are reported; the ALIGNED curves are compared against "
              "cached_L20_native_head, which shares their numerical path.")

    # ---- curves 2+: the representation-aligned frozen head, one adapter family at a time ----
    out_align = {}
    for kind_align in args.alignment:
        print(f"\n  --- {kind_align.upper()} alignment  h_l -> h_L20  (target-free) ---")
        per_layer, q9_pw, med_pw, mase_pw, spectra = [], [], [], [], None
        for l in layers:
            if l == LAST_LAYER:
                idp = identity_alignment(MODEL_DIMS)
                A, b, rec = idp["A"], idp["b"], {"alignment": "identity", "fitted": False,
                                                 "lambda": None}
                rep_test = representation_metrics(Y["test"], Y["test"])
                rep_train = representation_metrics(Y["train"], Y["train"])
                rep_val = representation_metrics(Y["val"], Y["val"])
                fc = endpoint                       # identity: BY CONSTRUCTION the endpoint
                sel = {"criterion": "none -- L20 is the reference endpoint (A = I, b = 0), "
                                    "never a fitted layer", "refit_after_selection": False}
                dof = float(MODEL_DIMS)
            else:
                Xtr = np.asarray(R["train"]["reps"][pos[l]], SOLVE_DTYPE)
                Xva = np.asarray(R["val"]["reps"][pos[l]], SOLVE_DTYPE)
                Xte = np.asarray(R["test"]["reps"][pos[l]], SOLVE_DTYPE)
                rec = fit_layer_alignment(Xtr, Y["train"], Xva, Y["val"], Xte, Y["test"],
                                          alignment=kind_align, grid=args.ridge_grid,
                                          criterion=args.selection_criterion)
                A, b = rec["A"], rec["b"]
                rep_train, rep_val, rep_test = rec["train"], rec["val"], rec["test"]
                sel = rec["selection"]
                dof = rec.get("effective_dof")
                if spectra is None and kind_align == "ridge":
                    spectra = rec["solver"]["spectrum"]
                fc = score_representation(model, apply_alignment(A, b, Xte), ctx, pte, geom,
                                          y_raw, den, rows, **sc)
                if args.save_adapters:
                    save_adapter(adapter_dir / kind_align, l, A, b, rec, tag, kind_align)
            q9_pw.append(fc["q9_window"]); med_pw.append(fc["median_window"])
            mase_pw.append(fc["mase_window"])
            per_layer.append({
                "layer": l, "layer_name": LAYER_NAMES[l],
                "is_identity_endpoint": l == LAST_LAYER,
                "lambda_selected": rec.get("lambda"),
                "effective_dof": dof,
                "lambda_at_grid_edge": bool(sel.get("at_grid_edge", False)),
                "train_representation_mse": rep_train["mse"],
                "val_representation_mse": rep_val["mse"],
                "test_representation_mse": rep_test["mse"],
                "train_representation_r2": rep_train["r2"],
                "val_representation_r2": rep_val["r2"],
                "test_representation_r2": rep_test["r2"],
                "test_representation_relative_frobenius_error":
                    rep_test["relative_frobenius_error"],
                "test_per_dimension_r2": rep_test.get("per_dimension_r2"),
                "aligned_native_q9_loss": fc["q9_loss"],
                "aligned_native_median_pinball": fc["median_pinball"],
                "aligned_native_mae": fc["mae"],
                "aligned_native_mase": fc["mase"],
                "aligned_per_quantile_pinball": fc["per_quantile_pinball"],
                "direct_native_head_q9_loss": direct[l]["q9_loss"],
                "direct_native_head_mase": direct[l]["mase"],
                "learned_probe_q9_loss": (ref_entry or {}).get(
                    "learned_probe_q9_loss", {}).get(l),
                "native_timesfm_q9_loss": official["q9_loss"],
                "cached_L20_native_head_q9_loss": endpoint["q9_loss"],
                "recovery_vs_cached_L20": recovery(direct[l]["q9_loss"], fc["q9_loss"],
                                                   endpoint["q9_loss"]),
                "recovery_vs_official_decode": recovery(direct[l]["q9_loss"], fc["q9_loss"],
                                                        official["q9_loss"]),
                "lambda_selection": {k: v for k, v in sel.items() if k != "candidates"},
                "lambda_candidates": sel.get("candidates")})
            p = per_layer[-1]
            print(f"    {LAYER_NAMES[l]:>4}  lam {str(p['lambda_selected'] or '-'):>8}  "
                  f"R2_test {p['test_representation_r2']:+.4f}  "
                  f"aligned Q9 {p['aligned_native_q9_loss']:>10.5f}  "
                  f"direct {p['direct_native_head_q9_loss']:>10.5f}  "
                  f"recov {('%.3f' % p['recovery_vs_cached_L20']['value']) if p['recovery_vs_cached_L20']['value'] is not None else '  n/a'}",
                  flush=True)

        li = layers.index(LAST_LAYER)
        sid = np.asarray(w["series_test"], np.int64)[rows]
        boot = {"aligned_q9_loss": cluster_ci(np.stack(q9_pw), sid, args.boot_b, args.seed, li),
                "aligned_median_loss": cluster_ci(np.stack(med_pw), sid, args.boot_b,
                                                  args.seed, li),
                "aligned_mase": cluster_ci(np.stack(mase_pw), sid, args.boot_b, args.seed, li)}
        edges = [LAYER_NAMES[p["layer"]] for p in per_layer if p["lambda_at_grid_edge"]]
        if edges:
            print(f"    [warn] lambda hit a GRID EDGE at {edges} -- the optimum may lie outside "
                  f"[{min(args.ridge_grid):g}, {max(args.ridge_grid):g}]; widen --ridge-grid")
        out_align[kind_align] = {"per_layer": per_layer, "bootstrap": boot,
                                 "train_spectrum": spectra,
                                 "lambda_grid_edges": edges}

    # ---- controls ----
    ctrl = controls(R, Y, pos, layers, fit_layers, args, model, ctx, pte, geom, y_raw, den,
                    rows, sc, endpoint, direct, out_align)

    return {"tag": tag, "short": short, "slug": SLUG.get(tag, tag), "kind": kind,
            "domain_status": kind, "split": "test", "N": int(len(rows)),
            "n_train": int(len(w["X_train"])), "n_val": int(len(w["X_val"])),
            "layers": layers, "layer_names": [LAYER_NAMES[l] for l in layers],
            "fitted_layers": fit_layers, "window_identity": ident,
            "window_identity_hashes": hashes,
            "window_identity_hash": window_identity_hash(w["X_test"], w["series_test"]),
            "alignments": out_align, "controls": ctrl,
            "endpoints": {
                "cached_L20_native_head": {
                    "q9_loss": endpoint["q9_loss"], "median_pinball": endpoint["median_pinball"],
                    "mae": endpoint["mae"], "mase": endpoint["mase"],
                    "role": "INTERNAL alignment endpoint -- the frozen head on the CACHED L20 "
                            "representation, the same (N, 1280) numerical path every aligned "
                            "representation takes; produced by A_20 = I, b_20 = 0"},
                "official_native_decode": {
                    "q9_loss": official["q9_loss"], "median_pinball": official["median_loss"],
                    "mase": float(official_mase.mean()),
                    "per_quantile_loss": official["per_quantile"],
                    "role": "the TRUE TimesFM-3 baseline -- decode()'s own forecast, captured at "
                            "extraction time and scored by the same native_reference"},
                "difference": {
                    "q9_absolute": d_end, "q9_relative": rel_end,
                    "note": "expected to be small but NOT zero: applying the head to a "
                            "(N, 1280) matrix accumulates differently than inside decode()'s "
                            "(b, 1, 18, 1280) pass. This is numerical provenance, not an "
                            "adapter result; aligned curves are compared against "
                            "cached_L20_native_head, which shares their path."}},
            "reference": ref_entry,
            "tunnel_entrance_5pct": (ref_entry or {}).get("tunnel_entrance_5pct"),
            "tunnel_entrance_5pct_name": (ref_entry or {}).get("tunnel_entrance_5pct_name"),
            "target_roundtrip_relative_err": rt, "n_denominator_clamped": n_clamped,
            "m_season": M_SEASON, "cache_roots": {s: R[s]["cache_root"] for s in R},
            "quantile_sorting": ctx["sorting"], "cached_native_check": ctx["native_check"],
            "seconds": round(time.time() - t0, 1)}


def controls(R, Y, pos, layers, fit_layers, args, model, ctx, pte, geom, y_raw, den, rows, sc,
             endpoint, direct, out_align) -> dict:
    """The five controls the design rests on, all measured rather than asserted in prose."""
    out = {}

    # 1. L20 identity: the aligned endpoint must BE the cached-L20 direct head, exactly.
    ridge = out_align.get("ridge") or next(iter(out_align.values()))
    l20 = next(p for p in ridge["per_layer"] if p["layer"] == LAST_LAYER)
    d = abs(l20["aligned_native_q9_loss"] - endpoint["q9_loss"])
    if d != 0.0:
        raise RuntimeError(
            f"L20 IDENTITY FAILED: the aligned endpoint Q9 {l20['aligned_native_q9_loss']:.9f} "
            f"is not the cached-L20 frozen-head Q9 {endpoint['q9_loss']:.9f} (delta {d:.3e}). "
            "A_20 = I and b_20 = 0, so these are the same computation; a difference means the "
            "identity adapter was fit or the endpoint was taken from a different path.")
    out["l20_identity"] = {
        "aligned_q9_loss": l20["aligned_native_q9_loss"],
        "cached_L20_native_head_q9_loss": endpoint["q9_loss"], "absolute_difference": d,
        "adapter_was_fitted": bool(l20["is_identity_endpoint"] is False),
        "test_representation_r2": l20["test_representation_r2"],
        "verdict": "A_20 = I, b_20 = 0 reproduces the cached-L20 endpoint exactly"}

    # 2. mean baseline: predict the TRAIN mean of h_L20 for every test window
    mb = mean_baseline_metrics(Y["test"], Y["train"].mean(axis=0))
    mb_self = mean_baseline_metrics(Y["test"], Y["test"].mean(axis=0))
    fc = score_representation(model, np.broadcast_to(Y["train"].mean(axis=0),
                                                     Y["test"].shape).copy(),
                              ctx, pte, geom, y_raw, den, rows, **sc)
    out["mean_baseline"] = {
        "test_representation_r2_train_mean": mb["r2"],
        "test_representation_mse_train_mean": mb["mse"],
        "test_representation_r2_test_mean": mb_self["r2"],
        "native_head_q9_loss": fc["q9_loss"], "native_head_mase": fc["mase"],
        "note": "R^2 of the TEST split's own mean is 0 by construction; the TRAIN mean scores "
                "slightly below it, which is the honest number and is not repaired. Any layer "
                "whose alignment R^2 does not clear this has learned nothing."}

    # 3. permutation control: same X, same spectrum, correspondence destroyed
    if args.permutation_layers:
        want = ([l for l in fit_layers] if args.permutation_layers == ["all"]
                else [int(x) for x in args.permutation_layers])
        prm = []
        Yp = permuted_rows(Y["train"], seed=args.seed)
        for l in [x for x in want if x in pos and x != LAST_LAYER]:
            Xtr = np.asarray(R["train"]["reps"][pos[l]], SOLVE_DTYPE)
            Xva = np.asarray(R["val"]["reps"][pos[l]], SOLVE_DTYPE)
            Xte = np.asarray(R["test"]["reps"][pos[l]], SOLVE_DTYPE)
            solver = RidgeAlignmentSolver(Xtr, Y["train"]).with_target(Yp)
            sel = select_lambda(solver, Xva, Y["val"], grid=args.ridge_grid,
                                criterion=args.selection_criterion)
            A, b = solver.coefficients(sel["lambda"])
            m = representation_metrics(apply_alignment(A, b, Xte), Y["test"],
                                       per_dimension=False)
            f = score_representation(model, apply_alignment(A, b, Xte), ctx, pte, geom, y_raw,
                                     den, rows, **sc)
            real = next(p for p in ridge["per_layer"] if p["layer"] == l)
            prm.append({"layer": l, "layer_name": LAYER_NAMES[l], "lambda": sel["lambda"],
                        "permuted_test_representation_r2": m["r2"],
                        "real_test_representation_r2": real["test_representation_r2"],
                        "permuted_native_q9_loss": f["q9_loss"],
                        "real_aligned_native_q9_loss": real["aligned_native_q9_loss"],
                        "direct_native_q9_loss": direct[l]["q9_loss"]})
            print(f"    [control] permuted {LAYER_NAMES[l]:>4}: R2 "
                  f"{m['r2']:+.4f} (real {real['test_representation_r2']:+.4f})   Q9 "
                  f"{f['q9_loss']:.5f} (real {real['aligned_native_q9_loss']:.5f})")
        out["permutation"] = {
            "rows": prm, "seed": args.seed,
            "design": "the SAME X, the SAME SVD and the SAME lambda grid; only the "
                      "window-to-window correspondence between h_l and h_L20 is destroyed",
            "expectation": "test representation R^2 collapses toward the mean baseline and the "
                           "forecasting recovery disappears"}

    # 4. no forecast-target leakage: structural, re-asserted on the arrays that were fit
    if fit_layers:
        assert_no_forecast_targets(horizon=geom.H, X_train_last_fitted_layer=np.asarray(
            R["train"]["reps"][pos[fit_layers[-1]]]), Y_train_representation=Y["train"],
            X_val=np.asarray(R["val"]["reps"][pos[fit_layers[-1]]]), Y_val=Y["val"])
    out["no_forecast_target_leakage"] = {
        "adapter_inputs": ["h_l (N, 1280)", "h_L20 (N, 1280)"],
        "loader": "probing.timesfm3_geometry.load_last_token_reps -- verifies the cache's "
                  "mu/sd/native entries and then DISCARDS them, so the fitting path structurally "
                  "cannot see a forecast",
        "selection_criterion": args.selection_criterion,
        "targets_built_after_fitting": True,
        "shape_guard": "probing.timesfm3_alignment.assert_no_forecast_targets rejects any (n, H) "
                       "trajectory or (n, H, Q) forecast reaching the fit",
        "verdict": "no future value, forecast, quantile or loss enters the adapter objective or "
                   "the lambda selection"}

    # 5. split discipline
    out["split_discipline"] = {
        "train": "fits every adapter candidate (full train, all lambdas)",
        "val": "selects lambda on representation MSE only; the selected candidate is NOT refit",
        "test": "evaluated exactly once, after lambda is frozen",
        "n_train": int(Y["train"].shape[0]), "n_val": int(Y["val"].shape[0]),
        "n_test": int(Y["test"].shape[0])}
    return out


def save_adapter(dirpath, layer, A, b, rec, tag, alignment) -> Path:
    """One adapter -> $SCRATCH. float32 on disk (the numbers are reported from the float64 fit).

    ~6.6 MB each; 20 layers x 7 datasets x alignment. NEVER written into the repo -- the
    committed outputs are the summary JSON, the table and the figures.
    """
    dirpath = Path(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)
    p = dirpath / f"L{layer:02d}.npz"
    np.savez_compressed(
        p, A=np.asarray(A, np.float32), b=np.asarray(b, np.float64),
        muX=np.asarray(rec.get("muX", np.zeros(1)), np.float64),
        muY=np.asarray(rec.get("muY", np.zeros(1)), np.float64),
        meta=json.dumps({"alignment_version": ALIGNMENT_VERSION, "dataset": tag,
                         "alignment": alignment, "layer": int(layer),
                         "layer_name": LAYER_NAMES[layer], "lambda": rec.get("lambda"),
                         "effective_dof": rec.get("effective_dof"),
                         "solve_dtype": str(np.dtype(SOLVE_DTYPE)),
                         "storage_dtype": "float32",
                         "objective": "min_A ||Xc A - Yc||_F^2 + lambda ||A||_F^2  (target-free)",
                         "prediction": "h_hat_L20 = X @ A + b"}))
    return p


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
DS_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]
ALIGN_STYLE = {"ridge": ("#2ca02c", "^-", "ridge-aligned native head"),
               "procrustes": ("#ff7f0e", "v-", "orthogonal (Procrustes) aligned head")}


def _axis(a, rec):
    a.set_xticks(rec["layers"][::2])
    a.set_xticklabels(rec["layer_names"][::2], rotation=45, fontsize=7)
    a.grid(alpha=0.3)


def figure_forecast_panels(recs, figdir) -> Path:
    """THE headline: four curves per dataset on one Q=9 pinball axis, tunnel entrance marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow, ncol = _panel_grid(len(recs))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.1 * nrow), squeeze=False)
    axes = [a for r in ax for a in r]
    for a, r in zip(axes, recs):
        lay = r["layers"]
        first = r["alignments"][list(r["alignments"])[0]]["per_layer"]
        probe = [p["learned_probe_q9_loss"] for p in first]
        if any(v is not None for v in probe):
            a.plot(lay, [np.nan if v is None else v for v in probe], "s-", ms=3.5,
                   color="#1f77b4", label="learned linear Q=9 probe")
        a.plot(lay, [p["direct_native_head_q9_loss"] for p in first], "o-", ms=3.5,
               color="#d62728", label="direct frozen native head")
        for name, al in r["alignments"].items():
            col, mk, lab = ALIGN_STYLE.get(name, ("#7f7f7f", "d-", name))
            a.plot(lay, [p["aligned_native_q9_loss"] for p in al["per_layer"]], mk, ms=3.8,
                   color=col, label=lab)
        a.axhline(r["endpoints"]["official_native_decode"]["q9_loss"], ls="--", lw=1.0,
                  c="grey", label="native TimesFM-3 (decode)")
        ce = r["endpoints"]["cached_L20_native_head"]["q9_loss"]
        a.axhline(ce, ls=(0, (1, 2)), lw=1.0, c="black", label="cached-L20 endpoint")
        t = r.get("tunnel_entrance_5pct")
        if t is not None:
            a.axvline(t, ls=":", c="k", lw=1.2,
                      label=f"5% tunnel entrance = {r.get('tunnel_entrance_5pct_name')}")
        vals = [v for p in first for v in (p["direct_native_head_q9_loss"],) if np.isfinite(v)
                and v > 0]
        if vals and max(vals) / max(min(vals), 1e-12) > 20:
            a.set_yscale("log")
        a.set_title(f"{r['short']}  ({r['domain_status']}, test N={r['N']})", fontsize=10)
        a.set_xlabel("representation point")
        _axis(a, r)
        a.legend(fontsize=6.2, loc="best")
    for a in axes[len(recs):]:
        a.set_visible(False)
    for r in ax:
        r[0].set_ylabel("Q=9 pinball loss   (log scale)")
    fig.suptitle("TimesFM-3: does a LINEAR map into the L20 basis make the frozen native head "
                 "work?\n(adapters fit on representation reconstruction only -- never on "
                 "forecast targets)", fontsize=11)
    fig.tight_layout()
    p = figdir / "forecast" / "aligned_vs_direct_vs_probe_all_datasets.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)
    return p


def figure_representation_r2(recs, figdir) -> Path:
    """Test representation R^2 vs depth, all seven datasets on one axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [n for n in ("ridge", "procrustes") if any(n in r["alignments"] for r in recs)][:2]
    fig, ax = plt.subplots(1, max(len(names), 1), figsize=(6.9 * max(len(names), 1), 5.0),
                           squeeze=False, sharey=True)
    ax = ax[0]
    for k, name in enumerate(names):
        a = ax[k]
        for i, r in enumerate(recs):
            al = r["alignments"].get(name)
            if al is None:
                continue
            c = DS_COLORS[i % len(DS_COLORS)]
            ls = "-" if r["domain_status"] == "PT-ID" else "--"
            a.plot(r["layers"], [p["test_representation_r2"] for p in al["per_layer"]],
                   ls, marker="o", ms=3.0, color=c, lw=1.4,
                   label=f"{r['short']} ({r['domain_status']})")
            t = r.get("tunnel_entrance_5pct")
            if t is not None and t in r["layers"]:
                v = al["per_layer"][r["layers"].index(t)]["test_representation_r2"]
                a.plot([t], [v], "*", ms=13, color=c, mec="k", mew=0.5, zorder=5)
        a.axhline(0.0, lw=1.0, c="k")
        a.axhline(1.0, lw=0.8, c="grey", ls=":")
        a.set_title(f"{name} alignment   (stars = 5% forecasting-tunnel entrance)", fontsize=10)
        a.set_xlabel("representation point  $h_{\\ell,15}$")
        a.set_xticks(recs[0]["layers"][::2])
        a.set_xticklabels(recs[0]["layer_names"][::2], rotation=45, fontsize=8)
        a.grid(alpha=0.3)
    ax[0].set_ylabel("test representation $R^2$  vs $h_{20,15}$")
    ax[0].legend(fontsize=7.5, loc="lower right")
    for a in ax[len(names):]:
        a.set_visible(False)
    fig.suptitle("How linearly recoverable is the FINAL representation from each depth?\n"
                 r"$R^2 = 1 - \|\hat H_{20} - H_{20}\|_F^2 / \|H_{20} - \bar H_{20}\|_F^2$  "
                 "(global, variance-weighted; test split)", fontsize=11)
    fig.tight_layout()
    p = figdir / "representation" / "representation_r2_by_layer.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)
    return p


def figure_recovery(recs, figdir) -> Path:
    """R_align vs depth: how much of the direct-head-to-endpoint gap alignment closes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, a = plt.subplots(figsize=(9.0, 5.4))
    for i, r in enumerate(recs):
        al = r["alignments"].get("ridge") or next(iter(r["alignments"].values()))
        v = [p["recovery_vs_cached_L20"]["value"] for p in al["per_layer"]]
        c = DS_COLORS[i % len(DS_COLORS)]
        ls = "-" if r["domain_status"] == "PT-ID" else "--"
        a.plot(r["layers"], [np.nan if x is None else x for x in v], ls, marker="o", ms=3.0,
               color=c, lw=1.4, label=f"{r['short']} ({r['domain_status']})")
        t = r.get("tunnel_entrance_5pct")
        if t is not None and t in r["layers"]:
            x = v[r["layers"].index(t)]
            if x is not None:
                a.plot([t], [x], "*", ms=13, color=c, mec="k", mew=0.5, zorder=5)
    a.axhspan(0.0, 1.0, color="grey", alpha=0.10)
    a.axhline(0.0, lw=1.0, c="k")
    a.axhline(1.0, lw=1.0, c="k", ls=":")
    a.set_ylim(-0.1, 1.25)
    a.set_xticks(recs[0]["layers"][::2])
    a.set_xticklabels(recs[0]["layer_names"][::2], rotation=45, fontsize=8)
    a.set_xlabel(r"representation point  $h_{\ell,15}$")
    a.set_ylabel(r"$R^{\rm align}_\ell = (L^{\rm direct}_\ell - L^{\rm aligned}_\ell)\,/\,"
                 r"(L^{\rm direct}_\ell - L^{\rm endpoint})$")
    a.set_title("Fraction of the frozen-head gap closed by linear alignment to $h_{20}$\n"
                "(ridge; 0 = no gain over the direct head, 1 = the cached-L20 endpoint; "
                "stars = 5% tunnel entrance)", fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    p = figdir / "recovery" / "alignment_recovery_by_layer.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
CSV_HEAD = ["dataset", "display_name", "domain_status", "alignment", "layer", "layer_name",
            "lambda_selected", "effective_dof", "lambda_at_grid_edge",
            "train_representation_mse", "val_representation_mse", "test_representation_mse",
            "val_representation_r2", "test_representation_r2",
            "test_per_dimension_r2_median", "test_per_dimension_r2_iqr",
            "aligned_native_q9_loss", "aligned_native_median_pinball", "aligned_native_mae",
            "aligned_native_mase", "direct_native_head_q9_loss", "learned_probe_q9_loss",
            "native_timesfm_q9_loss", "cached_L20_native_head_q9_loss",
            "recovery_vs_cached_L20", "recovery_vs_official_decode", "tunnel_entrance_5pct"]


def _fmt(x, nd=6):
    return "" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def write_outputs(recs, meta, out_root, paper_out) -> Path:
    nums = Path(out_root) / "numerical_results"
    nums.mkdir(parents=True, exist_ok=True)
    for r in recs:
        (nums / f"representation_alignment__{r['tag']}.json").write_text(
            json.dumps({**meta, **r}, indent=2, default=str))
    paper_out = Path(paper_out)
    paper_out.mkdir(parents=True, exist_ok=True)
    sp = paper_out / "representation_alignment_summary.json"
    sp.write_text(json.dumps({**meta, "datasets": {r["tag"]: r for r in recs}}, indent=2,
                             default=str))
    with open(paper_out / "representation_alignment_table.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(CSV_HEAD)
        for r in recs:
            for name, al in r["alignments"].items():
                for p in al["per_layer"]:
                    pd = p.get("test_per_dimension_r2") or {}
                    iqr = (None if pd.get("q75") is None or pd.get("q25") is None
                           else pd["q75"] - pd["q25"])
                    wr.writerow([
                        r["tag"], r["short"], r["domain_status"], name, p["layer"],
                        p["layer_name"],
                        "" if p["lambda_selected"] is None else f"{p['lambda_selected']:g}",
                        _fmt(p["effective_dof"], 2), int(bool(p["lambda_at_grid_edge"])),
                        _fmt(p["train_representation_mse"], 8),
                        _fmt(p["val_representation_mse"], 8),
                        _fmt(p["test_representation_mse"], 8),
                        _fmt(p["val_representation_r2"]), _fmt(p["test_representation_r2"]),
                        _fmt(pd.get("median")), _fmt(iqr),
                        _fmt(p["aligned_native_q9_loss"]),
                        _fmt(p["aligned_native_median_pinball"]), _fmt(p["aligned_native_mae"]),
                        _fmt(p["aligned_native_mase"]), _fmt(p["direct_native_head_q9_loss"]),
                        _fmt(p["learned_probe_q9_loss"]), _fmt(p["native_timesfm_q9_loss"]),
                        _fmt(p["cached_L20_native_head_q9_loss"]),
                        _fmt(p["recovery_vs_cached_L20"]["value"], 4),
                        _fmt(p["recovery_vs_official_decode"]["value"], 4),
                        "" if r.get("tunnel_entrance_5pct") is None
                        else r["tunnel_entrance_5pct"]])
    return sp


def print_final_report(recs, meta, paths):
    """The specification's closing report, printed from what actually ran."""
    print(f"\n{'=' * 84}\nFINAL REPORT -- TimesFM-3 linear representation alignment\n{'=' * 84}")
    c = meta["config"]
    print(f"  adapter objective : min_A ||Xc A - Yc||_F^2 + lambda ||A||_F^2 ; b = muY - muX A "
          f"(bias unpenalized)\n"
          f"                      target Y = h_20,15 -- NO forecast target, ever\n"
          f"  ridge solver      : economy SVD per layer; A_lambda = V diag(s/(s^2+lambda)) U^T Yc\n"
          f"  lambda grid       : {[float(x) for x in c['ridge_grid']]}\n"
          f"  centering         : both X and Y centred on TRAIN means; bias restored, unpenalized\n"
          f"  dtype             : {np.dtype(SOLVE_DTYPE).name} (solve + all metrics)\n"
          f"  selection         : {c['selection_criterion']} on the dedicated VAL split; no refit\n"
          f"  representation    : h_l,15 in R^{MODEL_DIMS}, points {LAYER_NAMES[0]}..."
          f"{LAYER_NAMES[-1]} ({NUM_LAYERS} total; L20 = identity)")
    for r in recs:
        al = r["alignments"].get("ridge") or next(iter(r["alignments"].values()))
        per = {p["layer"]: p for p in al["per_layer"]}
        t = r.get("tunnel_entrance_5pct")
        picks = [("Emb", 0), (f"tunnel({LAYER_NAMES[t]})" if t is not None else "tunnel", t),
                 ("L10", 10), ("L15", 15), ("L19", 19), ("L20", 20)]
        print(f"\n  {r['short']}  ({r['domain_status']}, N train {r['n_train']} / val "
              f"{r['n_val']} / test {r['N']})   tunnel entrance "
              f"{r.get('tunnel_entrance_5pct_name')}")
        print(f"    {'point':<14}{'R2_test':>9}{'aligned Q9':>12}{'direct Q9':>12}"
              f"{'probe Q9':>10}{'recovery':>10}")
        for lab, l in picks:
            if l is None or l not in per:
                continue
            p = per[l]
            rv = p["recovery_vs_cached_L20"]["value"]
            print(f"    {lab:<14}{p['test_representation_r2']:>+9.4f}"
                  f"{p['aligned_native_q9_loss']:>12.5f}{p['direct_native_head_q9_loss']:>12.5f}"
                  f"{(p['learned_probe_q9_loss'] if p['learned_probe_q9_loss'] is not None else float('nan')):>10.5f}"
                  f"{(rv if rv is not None else float('nan')):>10.3f}")
        e = r["endpoints"]
        print(f"    endpoints: cached_L20_native_head {e['cached_L20_native_head']['q9_loss']:.6f}"
              f"   official_native_decode {e['official_native_decode']['q9_loss']:.6f}"
              f"   delta {e['difference']['q9_absolute']:+.2e}")
        mb = r["controls"]["mean_baseline"]
        print(f"    controls: mean-baseline R2 {mb['test_representation_r2_train_mean']:+.2e}"
              f" / Q9 {mb['native_head_q9_loss']:.4f}"
              f"   L20 identity delta {r['controls']['l20_identity']['absolute_difference']:.1e}")
        for row in (r["controls"].get("permutation") or {}).get("rows", []):
            print(f"              permuted {row['layer_name']:>4}: R2 "
                  f"{row['permuted_test_representation_r2']:+.4f} vs real "
                  f"{row['real_test_representation_r2']:+.4f}")
    print(f"\n  outputs (repo)   : {paths['paper_out']}")
    print(f"  outputs (SCRATCH): {paths['out_root']}")
    for k, v in paths.get("figures", {}).items():
        print(f"  figure {k:<14}: {v}")
    print(f"  runtime          : {meta['seconds']}s")


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Linear representation alignment h_l -> h_L20 into TimesFM-3's frozen "
                    "native head. The adapter is NEVER trained on forecast targets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("model / environment")
    g.add_argument("--checkpoint", default=os.environ.get("TIMESFM3_CHECKPOINT",
                                                          "google/timesfm-3.0-pytorch"))
    g.add_argument("--hf-home", default=None)
    g.add_argument("--device", default=os.environ.get("TFM3_DEVICE", "cpu"),
                   help="the frozen head is a single Linear(1280, 576); CPU is sufficient and "
                        "is the default. No backbone forward pass is ever run.")
    g.add_argument("--head-batch-size", type=int, default=512)

    g = p.add_argument_group("data / cache")
    g.add_argument("--suite", default="paper7")
    g.add_argument("--datasets", nargs="+", default=None)
    g.add_argument("--cache-dir", default=os.environ.get(
        "TFM3_LT_CACHE_DIR", str(Path(os.environ.get("SCRATCH", "/tmp"))
                                 / "timesfm3_last_token" / "features_cache")))
    g.add_argument("--layers", nargs="+", type=int, default=None)
    g.add_argument("--context-len", type=int, default=512)
    g.add_argument("--horizon", type=int, default=64)
    g.add_argument("--stride", type=int, default=64)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-detrend", action="store_true")
    g.add_argument("--feature-dtype", default="float32")
    g.add_argument("--allow-window-mismatch", action="store_true")
    g.add_argument("--ignore-timesfm-version", action="store_true")
    g.add_argument("--roundtrip-rtol", type=float, default=1e-4)

    g = p.add_argument_group("alignment")
    g.add_argument("--alignment", nargs="+", default=["ridge"], choices=list(ALIGNMENTS),
                   help="ridge is the headline; procrustes is the stricter orthogonal control")
    g.add_argument("--ridge-grid", nargs="+", type=float, default=list(RIDGE_GRID))
    g.add_argument("--selection-criterion", default="val_representation_mse",
                   choices=["val_representation_mse", "val_representation_r2"])
    g.add_argument("--endpoint-report-rtol", type=float, default=1e-4,
                   help="above this the two endpoints' difference is flagged in the log. It is "
                        "a REPORT threshold, never a gate: the cached-L20 endpoint is not "
                        "required to be bit-identical to decode()")

    g = p.add_argument_group("controls")
    g.add_argument("--permutation-layers", nargs="+", default=["0", "10", "19"],
                   help="layers to run the permuted-correspondence control on; 'all' or 'none'")
    g.add_argument("--boot-b", type=int, default=5000)

    g = p.add_argument_group("outputs")
    g.add_argument("--out-root", default=str(Path(os.environ.get("SCRATCH", "/tmp"))
                                             / "timesfm3_alignment"),
                   help="HEAVY artifacts ($SCRATCH): adapter matrices, per-dataset JSON")
    g.add_argument("--paper-out", default=str(paper_out_default()),
                   help="LIGHT artifacts (repo): summary JSON, table, figures")
    g.add_argument("--reference-results", default=str(DEFAULT_REFERENCE),
                   help="committed native-head-transfer summary: direct-head + probe curves, "
                        "native baseline and the 5%% tunnel entrance")
    g.add_argument("--probe-results", default=None,
                   help="optional probe summary; overrides the merged probe curve / tunnel")
    g.add_argument("--no-save-adapters", dest="save_adapters", action="store_false",
                   help="skip writing the ~6.6 MB float32 adapter matrices to --out-root")
    g.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    if a.permutation_layers and a.permutation_layers[0].lower() == "none":
        a.permutation_layers = []
    return a


def main(argv=None):
    args = parse_args(argv)
    t0 = time.time()
    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home
    from probing.timesfm3_last_token import assert_backbone_frozen, get_model

    geom = LastTokenGeometry(C=args.context_len, H=args.horizon)
    tags = [t for t in suite_tags(args.suite) if args.datasets is None or t in args.datasets]
    if not tags:
        raise SystemExit(f"no datasets selected from suite {args.suite!r}")
    out_root, paper_out = Path(args.out_root), Path(args.paper_out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"TimesFM-3 LINEAR REPRESENTATION ALIGNMENT  h_l -> h_L20 -> frozen native head")
    print(f"  suite {args.suite}: {len(tags)} datasets   alignment {args.alignment}   "
          f"points {LAYER_NAMES[0]}..{LAYER_NAMES[-1]}")
    print(f"  cache      {args.cache_dir}")
    print(f"  SCRATCH    {out_root}")
    print(f"  repo       {paper_out}")

    model, device = get_model(args.checkpoint, args.device)
    assert_backbone_frozen(model)
    ck_before = head_checksum(model)
    print(f"  frozen head: {ck_before['module']}({ck_before['in_features']}, "
          f"{ck_before['out_features']}), {ck_before['n_params']} params, sha256 "
          f"{ck_before['sha256']}, requires_grad=False   device {device}")

    ref = load_reference(args.reference_results, args.probe_results)
    print(f"  reference  {args.reference_results}  [{'merged' if ref['available'] else 'MISSING'}]")

    recs = []
    for tag in tags:
        recs.append(run_dataset(tag, args, geom, model, device,
                                ref["datasets"].get(tag), out_root / "adapters" / tag))
    print_roster_audit([r["window_identity"] for r in recs])

    ck_after = head_checksum(model)
    if ck_after["sha256"] != ck_before["sha256"]:
        raise RuntimeError("the native head CHANGED during the run -- it must be frozen")

    meta = {"analysis": "timesfm3_linear_representation_alignment",
            "alignment_version": ALIGNMENT_VERSION,
            "model": "timesfm-3.0", "checkpoint": args.checkpoint, "device": str(device),
            "objective": "min_{A,b} ||X_l A + 1 b^T - H_20||_F^2 + lambda ||A||_F^2",
            "adapter_target": "the FINAL-LAYER REPRESENTATION h_20,15 -- never a forecast",
            "why_not_forecast_loss": "TimesFM-3's native head W_native is itself linear, so an "
                                     "adapter trained on forecast loss would collapse to another "
                                     "linear forecasting map W_native A_l, i.e. the learned "
                                     "probe in disguise",
            "not_the_chronos_adapter": "probing/native_head_adapter.py trains Linear(768,768) on "
                                       "FORECAST loss into Chronos-2's NONLINEAR ResidualBlock "
                                       "head; that objective and this one are never mixed",
            "representation": f"h_l,15 (last real context token), (N, {MODEL_DIMS}) per point",
            "representation_points": NUM_LAYERS, "layer_names": LAYER_NAMES,
            "identity_endpoint": f"A_{LAST_LAYER} = I, b_{LAST_LAYER} = 0 (not fitted)",
            "solver": "economy SVD per layer; every lambda re-weights ONE decomposition",
            "solve_dtype": np.dtype(SOLVE_DTYPE).name,
            "selection": "validation representation error only; the selected train fit is NOT "
                         "refit",
            "tunnel_definition": "NOT defined here -- the 5% entrance is read from the probe run "
                                 "as an overlay and a table index only",
            "backbone_forward_passes": 0,
            "feature_source": "the validated float32 last-token cache (tfm3-last-token-q9-fp32-v1)",
            "native_head": ck_before, "native_head_checksum_after": ck_after,
            "native_head_unchanged": True,
            "native_quantiles": [float(x) for x in NATIVE_QUANTILES],
            "num_quantiles": NUM_QUANTILES, "m_season": M_SEASON,
            "reference_results": ref, "config": vars(args), "git_commit": git_commit(),
            "computed": datetime.datetime.now().isoformat(timespec="seconds"),
            "seconds": round(time.time() - t0, 1)}

    figs = {}
    if not args.no_figures:
        figdir = paper_out / "figures"
        figs = {"forecast": str(figure_forecast_panels(recs, figdir)),
                "representation": str(figure_representation_r2(recs, figdir)),
                "recovery": str(figure_recovery(recs, figdir))}
        meta["figures"] = figs
    sp = write_outputs(recs, meta, out_root, paper_out)
    print_final_report(recs, meta, {"paper_out": paper_out, "out_root": out_root,
                                    "figures": figs})
    print(f"\n  summary: {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
