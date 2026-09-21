"""TiRex layer-wise probing driver: Q=1 median probe + forecasting tunnel + CKA + effective rank.

    python -m experiments.run_tirex_probing --discover-only          # geometry report, no probing
    python -m experiments.run_tirex_probing --datasets monash_electricity_hourly --limit 64
    python -m experiments.run_tirex_probing                          # the full paper7 suite

ONE pass over each dataset produces everything the spec asks for:

    extraction   the PRIMARY definition is `two_pass` -- the released package's own default
                 inference pathway (max_accelerated_rollout_steps=1): K=2 forward passes, the
                 state for output patch k captured from the pass that produced it, at that pass's
                 own last token, under that pass's own (loc, scale). Kept at all 14 depths
                 {Emb, L1..L12, L12+RMS} -> (n, 2, 512) per depth. `--rollout-mode single_pass`
                 is the one-pass accelerated path, retained as a robustness check.
    probe        ONE SHARED Linear(512, 32) per depth, applied to BOTH native forecast-producing
                 states, Q=1 (tau=0.5). Train fits, validation selects the weight decay, test
                 scored once. The probe is IDENTICAL in both rollout modes.
    tunnel       probing.tunnel.tunnel_start, UNCHANGED (first crossing of 1.05 x final depth)
    geometry     probing.cka (biased AND unbiased) + probing.spectral_metrics, UNCHANGED.
                 HEADLINE for BOTH estimators: `pos0` -- readout position 0 (the last-real-context
                 state) at EVERY depth, so the observation count is CONSTANT (n) across depth.
                 COMPANION: `all_positions_from_L1` -- both readout states, 2n rows, L1 onward
                 only, where the second state is non-degenerate. A curve whose n changes with
                 depth is deliberately not produced.
    metrics      MASE in raw units against run_timesfm3_probing's denominator -- the same
                 definition run_id_forecasting and the Chronos-2 runs use

NOTHING about dataset rosters, splits, windows, MASE or the tunnel criterion is redefined here:
those are imported from the modules that already own them, so the three model lines cannot drift.

PRETRAINING PROVENANCE IS DELIBERATELY NOT LABELLED. ``probing.tunnel.domain_status`` would
return pt_id / pt_ood, but that taxonomy is defined relative to CHRONOS-2's pretraining corpus
and is being redesigned. Every record therefore carries ``pretraining_status: null`` with a note;
the dataset ROSTER is reused, the LABEL is not.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing.config import BOOT_B, SEED                                          # noqa: E402
from probing.timesfm3 import raw_future_from_arcsinh                             # noqa: E402
from probing.tirex_geometry import (CKA_ESTIMATORS, CKA_HEADLINE_SPLIT,          # noqa: E402
                                    CKA_HEADLINE_VARIANT, CROSS_MODEL_CAVEAT,
                                    EMB_POLICY, ERANK_HEADLINE_SPLIT,
                                    ERANK_HEADLINE_VARIANT, GEOMETRY_VARIANTS,
                                    HEADLINE_CKA_ESTIMATOR, VARIANT_ALL_FROM_L1,
                                    VARIANT_POS0, geometry_for_split, variant_spec)
from probing.tirex_model import (CACHE_VERSION, CONTEXT_LEN, DEFAULT_CHECKPOINT,  # noqa: E402
                                 HORIZON, MODEL_DIMS, NUM_BLOCKS, NUM_POINTS,
                                 REP_NAMES, REP_POINTS, assert_frozen,
                                 assert_native_geometry, assert_no_grads,
                                 assert_target_roundtrip,
                                 build_targets, cached_features, denormalize,
                                 discover_geometry, geometry_from_model, get_model,
                                 ROLLOUT_MODES, ROLLOUT_SINGLE, ROLLOUT_TWO,
                                 assert_rollout_appends_missing,
                                 assert_two_pass_matches_package, compare_backends,
                                 head_checksum, native_forward, native_forward_two_pass,
                                 rep_depth_table, rollout_mode_gap, tirex_version,
                                 verify_native_head, verify_native_head_two_pass)
from probing.tirex_probes import (EPOCHS, LR, TAU, WD_GRID_TIREX,                # noqa: E402
                                  assert_wd_grid, constant_forecast_floor, layerwise_probe,
                                  native_median_reference, representation_norms,
                                  tunnel_entrance, tunnel_entrances)
# ONE definition of MASE + cluster CIs for every model line (read-only import; these helpers are
# model- and geometry-agnostic, and the denominator is byte-identical to
# run_id_forecasting._mase_denominator, which the committed Chronos-2 runs use).
from experiments.run_timesfm3_probing import (MASE_DEN_FLOOR, M_SEASON, cluster_ci,  # noqa: E402
                                              mase_denominator, per_window_mase)
# Dataset roster, window construction and the committed-window parity gate: imported, never
# re-declared, so TiRex is probed on EXACTLY the windows Chronos-2 and TimesFM-3 were.
from experiments.run_timesfm3_last_token_probing import (PAPER7, SHORT,          # noqa: E402
                                                         assert_window_parity as _parity,
                                                         chronos_reference, suite_tags,
                                                         windows_for)

def assert_window_parity(tag, w, ref, *, strict=True):
    """TiRex must receive the SAME x[T-512:T] -> x[T:T+64] windows as Chronos-2 and TimesFM-3.

    The comparison itself is the ALREADY-VALIDATED one (counts, C/H/seasonal m, and the
    per-window test series ids element-wise); only the failure wording is re-pointed at TiRex,
    since the shared implementation names TimesFM-3 in its message.
    """
    try:
        return _parity(tag, w, ref, strict=strict)
    except RuntimeError as exc:
        raise RuntimeError(
            f"WINDOW PARITY FAILED for {tag}: TiRex is not being evaluated on the same windows "
            f"as Chronos-2 / TimesFM-3.\n{exc}\n  Fix the window construction -- do NOT proceed "
            "with mismatched windows (pass --allow-window-mismatch only to INSPECT the "
            "difference)." ) from exc


OUT_DEFAULT = REPO_ROOT / "results" / "tirex_probing"
PROVENANCE_NOTE = ("pretraining status (pt_id / pt_ood) is defined relative to Chronos-2's "
                   "pretraining corpus and is being redesigned; it is deliberately NOT applied "
                   "to TiRex here. The dataset roster is reused, the label is not.")


# --------------------------------------------------------------------------- #
# one dataset
# --------------------------------------------------------------------------- #
def run_dataset(tag, args, model, geom, paths):
    t0 = time.time()
    print(f"\n=== {tag} ({SHORT.get(tag, tag)}) " + "=" * (56 - len(tag)))
    w = windows_for(tag, args.suite, args)
    ident = assert_window_parity(tag, w, chronos_reference(tag), strict=not args.allow_window_mismatch)
    print(f"  windows: train {ident['n_train_windows']} / val {ident['n_val_windows']} / "
          f"test {ident['n_test_windows']}  ({ident['n_test_series']} test clusters, "
          f"parity={ident.get('chronos_parity')})")

    if "X_val" not in w:
        raise SystemExit(f"{tag}: this suite has no dedicated validation split. The TiRex probe "
                         "selects weight decay and the tunnel entrance on validation only -- use "
                         "--suite paper7 (or another rolling set) which provides X_val.")

    splits = {}
    for sp in ("train", "val", "test"):
        X = np.asarray(w[f"X_{sp}"], np.float32)
        Y = np.asarray(w[f"Y_{sp}_traj"], np.float32)
        if args.limit and len(X) > args.limit:
            X, Y = X[:args.limit], Y[:args.limit]
        splits[sp] = {"X": X, "raw_future": raw_future_from_arcsinh(X, Y).astype(np.float32)}

    # ---- targets in TiRex's own normalized space (context-only statistics) ----
    for sp, d in splits.items():
        tgt, loc, scale = build_targets(d["X"], d["raw_future"], model, geom,
                                        mode=args.rollout_mode)
        d.update(targets=tgt, loc=loc, scale=scale)
        d["roundtrip"] = assert_target_roundtrip(d["raw_future"], tgt, loc, scale, geom)
    print(f"  target round-trip (test): max|d| {splits['test']['roundtrip']['max_abs_error']:.2e}, "
          f"relative {splits['test']['roundtrip']['relative_error']:.2e}")

    # ---- extraction (cached) ----
    feats, native = {}, {}
    for sp, d in splits.items():
        f, nat, meta, hit = cached_features(tag, sp, d["X"], model, geom,
                                            cache_dir=args.cache_dir, checkpoint=args.checkpoint,
                                            backend=args.backend, points=args.points,
                                            seed=args.seed, batch_size=args.batch_size,
                                            verbose=args.verbose, mode=args.rollout_mode)
        feats[sp], native[sp] = f, nat
        print(f"  features[{sp:<5s}] {f[args.points[0]].shape}  "
              f"{'CACHE HIT' if hit else 'extracted'}")

    # ---- the decisive invariant, re-proved on THESE windows ----
    nb = min(8, len(splits["test"]["X"]))
    ctx = torch.as_tensor(splits["test"]["X"][:nb])
    if args.rollout_mode == ROLLOUT_SINGLE:
        off, reps, st = native_forward(model, ctx, geom)
        vn = verify_native_head(model, reps, off, st, geom)
        vn["readout_index_per_pass"] = list(geom.readout_indices)
    else:
        pkg = assert_two_pass_matches_package(model, ctx, geom)
        off, per_pass, states, layouts = native_forward_two_pass(model, ctx, geom)
        vn = verify_native_head_two_pass(model, per_pass, off, states, geom)
        vn["replication_matches_package"] = pkg
        vn["pass_layouts"] = layouts
        vn["rollout_appends"] = assert_rollout_appends_missing(model, ctx, geom)
    print(f"  native-head identity [{args.rollout_mode}]: tokens "
          f"{vn['readout_index_per_pass']} reproduce the native H={geom.H} forecast  "
          f"scaled={vn['max_scaled_error']:.3f} (<=1)  max|d|={vn['max_abs_error']:.2e}  "
          f"exact={vn['exact_bitwise_match']}  "
          f"(controls {min(vn['wrong_index_controls'].values()):.2e})")
    gap = rollout_mode_gap(model, ctx, geom)

    # ---- probes ----
    res = layerwise_probe(feats["train"], splits["train"]["targets"],
                          feats["val"], splits["val"]["targets"],
                          feats["test"], splits["test"]["targets"], geom,
                          points=args.points, device=args.probe_device, wd_grid=args.wd_grid,
                          epochs=args.epochs, lr=args.lr, init_seed=args.seed,
                          verbose=args.verbose)
    # spec test 20: probe training must leave the frozen backbone untouched. The features are
    # numpy by the time they reach the probe, so there is structurally no path back into the
    # model -- this asserts it held rather than assuming it.
    assert_no_grads(model)
    assert_frozen(model)
    floor = constant_forecast_floor(splits["train"]["targets"], splits["test"]["targets"],
                                    splits["val"]["targets"])
    nat_ref = native_median_reference(native["test"], splits["test"]["loc"],
                                      splits["test"]["scale"], splits["test"]["targets"], geom)

    # ---- MASE in RAW units, same denominator as every other line ----
    den = np.maximum(mase_denominator(splits["test"]["X"]), MASE_DEN_FLOOR)
    n_clamped = int((mase_denominator(splits["test"]["X"]) < MASE_DEN_FLOOR).sum())
    y_raw = splits["test"]["raw_future"].astype(np.float64)
    mase_pw = {}
    for name in args.points:
        yhat = denormalize(res[name]["pred_test_norm"], splits["test"]["loc"],
                           splits["test"]["scale"], geom)
        mase_pw[name] = per_window_mase(y_raw, yhat, den)
    nat_mase_pw = per_window_mase(y_raw, nat_ref["median_raw"], den)

    # ---- tunnel (validation only) ----
    val_curve = [res[n]["val_loss"] for n in args.points]
    test_curve = [res[n]["test_loss"] for n in args.points]
    ent = tunnel_entrance(val_curve, args.points)
    ents = tunnel_entrances(val_curve, args.points)
    ref_idx = len(args.points) - 1
    print(f"  TUNNEL: entrance {ent['point']} (block {ent['block_index']}, relative_depth "
          f"{ent['relative_depth']:.3f}) | val {ent['val_loss_at_entrance']:.5f} vs final "
          f"{ent['val_loss_at_reference']:.5f}")

    # ---- cluster-bootstrap CIs over test series ----
    sid = np.asarray(w["series_test"], np.int64)[:len(splits["test"]["X"])]
    boot = {"loss": cluster_ci(np.stack([res[n]["test_window_loss"] for n in args.points]),
                               sid, args.boot_b, args.seed, ref_idx),
            "mase": cluster_ci(np.stack([mase_pw[n] for n in args.points]),
                               sid, args.boot_b, args.seed, ref_idx)}

    # ---- representation geometry, BOTH splits, BOTH estimators ----
    # HEADLINE: the spec's construction, every readout position stacked -> (n*K, d).
    # COMPANION (`pos0`): readout position 0 only -> (n, d). It exists because (Emb, position 1)
    # is constant across windows (see tirex_geometry.DEGENERACY_CAVEAT), which distorts Emb's row
    # of the headline matrix relative to every deeper depth. Nothing is dropped from the headline.
    geo = {}
    for sp in args.geometry_splits:
        geo[sp] = geometry_for_split(feats[sp], points=args.points,
                                     estimators=args.cka_estimators,
                                     null_floor_reps=args.cka_null_floor_reps, seed=args.seed,
                                     n_subsamples=args.erank_subsamples)
        g = geo[sp]
        nf = g["cka_null_floor"][CKA_HEADLINE_VARIANT]
        er = g["effective_rank"][ERANK_HEADLINE_VARIANT]["effective_rank"]
        erc = g["effective_rank"][VARIANT_ALL_FROM_L1]["effective_rank"]
        print(f"  geometry[{sp:<5s}] HEADLINE {CKA_HEADLINE_VARIANT}: "
              f"{g['n_rows'][CKA_HEADLINE_VARIANT]} rows x {len(g['variant_points'][CKA_HEADLINE_VARIANT])} "
              f"depths  "
              + "  ".join(f"{e} null floor {nf[e]['mean']:+.3f}" for e in args.cka_estimators)
              + f"  | r_eff {er[0]:.1f} (Emb) -> {er[-1]:.1f} (final)")
        print(f"             companion {VARIANT_ALL_FROM_L1}: {g['n_rows'][VARIANT_ALL_FROM_L1]} "
              f"rows x {len(g['variant_points'][VARIANT_ALL_FROM_L1])} depths (L1 onward)  "
              f"| r_eff {erc[0]:.1f} (L1) -> {erc[-1]:.1f} (final)")
    deg = geo[args.geometry_splits[0]]["degeneracy"]
    if deg["affected"] != deg["expected"]:
        print(f"  WARNING: constant-across-window readouts are {deg['affected']}, expected "
              f"{deg['expected']} -- investigate before trusting the geometry numbers.")
    else:
        print(f"  degeneracy: {deg['affected']} constant across windows (EXPECTED: the masked "
              f"future token has no recurrence at Emb) -- which is why the HEADLINE geometry is "
              f"{CKA_HEADLINE_VARIANT} at every depth and the companion starts at L1")

    # ---- assemble ----
    layers = []
    for i, name in enumerate(args.points):
        r = res[name]
        layers.append({
            "representation": name, "index": i,
            "block_index": r["block_index"], "position_index": r["position_index"],
            "relative_depth": r["relative_depth"], "relative_position": r["relative_position"],
            "kind": r["kind"], "is_native_readout": r["is_native_readout"],
            "train_loss": r["train_loss"], "val_loss": r["val_loss"], "test_loss": r["test_loss"],
            "test_mase": float(mase_pw[name].mean()),
            "test_loss_patch0_last_context": r["test_per_patch"]["patch0_last_context"],
            "test_loss_patch1_masked_future": r["test_per_patch"]["patch1_masked_future"],
            "val_loss_patch0_last_context": r["val_per_patch"]["patch0_last_context"],
            "val_loss_patch1_masked_future": r["val_per_patch"]["patch1_masked_future"],
            "weight_decay": r["wd"], "wd_at_grid_max": r["selection"]["at_grid_max"],
            "n_probe_params": r["n_params"],
            "beats_constant_floor": bool(r["test_loss"] < floor["test_loss"]),
            "test_loss_ci": [boot["loss"]["ci_lo"][i], boot["loss"]["ci_hi"][i]],
            "test_mase_ci": [boot["mase"]["ci_lo"][i], boot["mase"]["ci_hi"][i]],
            "effective_rank": {sp: geo[sp]["effective_rank"][ERANK_HEADLINE_VARIANT]
                                   ["effective_rank"][i] for sp in geo},
            "effective_rank_companion": {sp: _companion_erank(geo[sp], name) for sp in geo},
            "constant_across_windows": geo[args.geometry_splits[0]]["degeneracy"][
                "constant_across_windows_by_point"][name],
        })

    clipped = [l["representation"] for l in layers if l["wd_at_grid_max"]]
    if clipped:
        print(f"  WARNING: weight decay hit the grid MAXIMUM at {clipped} -- the grid may be "
              f"clipping; widen --wd-grid and re-check.")
    weak = [l["representation"] for l in layers if not l["beats_constant_floor"]]
    if weak:
        print(f"  WARNING: {weak} do NOT beat the constant-forecast floor "
              f"({floor['test_loss']:.5f}) -- no linearly decodable forecast at those depths.")

    entry = {
        "dataset": tag, "short": SHORT.get(tag, tag),
        "pretraining_status": None, "pretraining_status_note": PROVENANCE_NOTE,
        "windows": ident, "n_windows": {sp: int(len(splits[sp]["X"])) for sp in splits},
        "n_test_clusters": int(np.unique(sid).size),
        "layers": layers,
        "val_loss_by_representation": val_curve, "test_loss_by_representation": test_curve,
        "test_mase_by_representation": [float(mase_pw[n].mean()) for n in args.points],
        "tunnel": ent, "tunnel_all_tolerances": ents,
        "constant_forecast_floor": floor,
        "native": {"test_loss": nat_ref["test_loss"], "test_mase": float(nat_mase_pw.mean()),
                   "test_per_patch": nat_ref["test_per_patch"],
                   "median_index": nat_ref["median_index"]},
        "native_head_identity": vn, "rollout_mode_gap": gap,
        "target_roundtrip": {sp: splits[sp]["roundtrip"] for sp in splits},
        "representation_norms": {sp: representation_norms(feats[sp]) for sp in ("train", "test")},
        "mase": {"definition": f"in-context seasonal-naive (m={M_SEASON}, floor {MASE_DEN_FLOOR})",
                 "n_denominator_clamped": n_clamped},
        "bootstrap": boot,
        "geometry": {sp: {k: v for k, v in g.items() if k != "cka"} for sp, g in geo.items()},
        "readout_degeneracy": geo[args.geometry_splits[0]]["degeneracy"],
        "emb_policy": EMB_POLICY,
        "geometry_headline": {"cka_split": CKA_HEADLINE_SPLIT,
                              "cka_estimator": HEADLINE_CKA_ESTIMATOR,
                              "effective_rank_split": ERANK_HEADLINE_SPLIT},
        "seconds": round(time.time() - t0, 1),
    }

    _save_dataset(tag, entry, geo, res, mase_pw, nat_ref, nat_mase_pw, sid, args, paths)
    print(f"  probe MASE at entrance {layers[ent['index']]['test_mase']:.4f} | final "
          f"{layers[-1]['test_mase']:.4f} | native {nat_mase_pw.mean():.4f}   "
          f"[{entry['seconds']}s]")
    return entry


def _save_dataset(tag, entry, geo, res, mase_pw, nat_ref, nat_mase_pw, sid, args, paths):
    (paths["per_dataset"] / f"{tag}.json").write_text(json.dumps(entry, indent=2, default=_json))
    from probing.cka import save_matrix_csv
    for sp, g in geo.items():
        for variant, per_est in g["cka"].items():
            labels = g["variant_points"][variant]        # each variant has its OWN depth set
            for est, M in per_est.items():
                base = paths["matrices"] / variant / est / sp
                base.mkdir(parents=True, exist_ok=True)
                np.save(base / f"cka__{tag}.npy", M)
                save_matrix_csv(M, labels, labels, base / f"cka__{tag}.csv")
    np.savez(paths["bootstrap_inputs"] / f"{tag}.npz",
             series_test=sid,
             window_loss=np.stack([res[n]["test_window_loss"] for n in args.points]),
             window_mase=np.stack([mase_pw[n] for n in args.points]),
             native_window_loss=nat_ref["test_window_loss"],
             native_window_mase=nat_mase_pw,
             representation_points=np.array(args.points))


def _companion_erank(g, name):
    """The companion (all-positions, L1 onward) effective rank at one depth, or None at Emb --
    the companion does not cover Emb, because its second readout state is degenerate there."""
    er = g["effective_rank"][VARIANT_ALL_FROM_L1]
    pts = er["representation_points"]
    return er["effective_rank"][pts.index(name)] if name in pts else None


def _json(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(f"not JSON serializable: {type(o)}")


# --------------------------------------------------------------------------- #
# outputs
# --------------------------------------------------------------------------- #
def compare_rollout_modes(summaries: dict, args) -> dict:
    """Do the two native rollout modes change the CONCLUSIONS, or only the digits?

    Compared per dataset, on exactly the quantities a reader would act on:
      * the tunnel entrance (index, point, relative depth) -- the headline claim;
      * the val/test probe curves and the MASE curve, as max absolute and max relative deltas;
      * whether the test-loss curve's ARGMIN and its ORDERING survive (Spearman over depths);
      * the headline unbiased CKA matrix, as max |dCKA|;
      * the headline effective-rank curve, as max |d r_eff| and max relative;
      * the two modes' native forecasts themselves (rollout_mode_gap), for context.

    "Material" is declared per criterion with thresholds stated in the record, not by eyeball:
    a tunnel entrance that moves at all is material; a curve delta below 1% of the curve's own
    range is not. Nothing here decides which mode is the paper's primary definition -- it reports
    the size of the choice.
    """
    a, b = ROLLOUT_SINGLE, ROLLOUT_TWO
    pts = list(args.points)
    out = {"modes": [a, b], "representation_points": pts,
           "thresholds": {"curve_relative": 0.01, "cka_abs": 0.02, "erank_relative": 0.05,
                          "tunnel": "any move is material"},
           "definition": "delta = two_pass - single_pass", "datasets": {}}
    by = {m: {e["dataset"]: e for e in summaries[m]["datasets"]} for m in (a, b)}
    for tag in by[a]:
        if tag not in by[b]:
            continue
        ea, eb = by[a][tag], by[b][tag]
        rec = {"short": ea["short"]}
        ta, tb = ea["tunnel"], eb["tunnel"]
        rec["tunnel"] = {
            a: {"point": ta["point"], "index": ta["index"],
                "relative_depth": ta["relative_depth"]},
            b: {"point": tb["point"], "index": tb["index"],
                "relative_depth": tb["relative_depth"]},
            "index_delta": tb["index"] - ta["index"],
            "same_entrance": ta["index"] == tb["index"],
            "material": ta["index"] != tb["index"]}
        rec["tunnel_all_tolerances"] = {
            k: {a: ea["tunnel_all_tolerances"][k]["point"],
                b: eb["tunnel_all_tolerances"][k]["point"],
                "same": ea["tunnel_all_tolerances"][k]["index"]
                        == eb["tunnel_all_tolerances"][k]["index"]}
            for k in ea["tunnel_all_tolerances"]}

        for key, label in (("val_loss_by_representation", "val_loss"),
                           ("test_loss_by_representation", "test_loss"),
                           ("test_mase_by_representation", "test_mase")):
            va, vb = np.asarray(ea[key], float), np.asarray(eb[key], float)
            rng = float(va.max() - va.min()) or 1.0
            d = np.abs(vb - va)
            rec[label] = {
                "max_abs_delta": float(d.max()),
                "max_delta_relative_to_curve_range": float(d.max() / rng),
                "mean_abs_delta": float(d.mean()),
                f"argmin_{a}": pts[int(va.argmin())], f"argmin_{b}": pts[int(vb.argmin())],
                "same_argmin": bool(va.argmin() == vb.argmin()),
                "spearman_over_depths": _spearman(va, vb),
                "material": bool(d.max() / rng > out["thresholds"]["curve_relative"])}

        # headline geometry: unbiased CKA (pos0) and the mixed effective rank
        rec["geometry"] = {}
        for sp in ea["geometry"]:
            if sp not in eb["geometry"]:
                continue
            ga, gb = ea["geometry"][sp], eb["geometry"][sp]
            era = np.asarray(ga["effective_rank"][ERANK_HEADLINE_VARIANT]["effective_rank"], float)
            erb = np.asarray(gb["effective_rank"][ERANK_HEADLINE_VARIANT]["effective_rank"], float)
            de = np.abs(erb - era)
            rec["geometry"][sp] = {
                "effective_rank": {
                    "variant": ERANK_HEADLINE_VARIANT,
                    "max_abs_delta": float(de.max()),
                    "max_relative_delta": float((de / np.maximum(era, 1e-12)).max()),
                    f"argmax_{a}": pts[int(era.argmax())], f"argmax_{b}": pts[int(erb.argmax())],
                    "same_argmax": bool(era.argmax() == erb.argmax()),
                    "material": bool((de / np.maximum(era, 1e-12)).max()
                                     > out["thresholds"]["erank_relative"])}}
        rec["native_forecast_gap"] = {
            "max_abs": ea["rollout_mode_gap"]["max_abs_diff"],
            "relative_to_mean_abs": ea["rollout_mode_gap"]["relative_to_mean_abs"],
            "first_patch_identical": ea["rollout_mode_gap"]["first_patch_identical"]}
        rec["native_head_identity"] = {
            m: {"max_abs_error": by[m][tag]["native_head_identity"]["max_abs_error"],
                "max_scaled_error": by[m][tag]["native_head_identity"]["max_scaled_error"],
                "passed": by[m][tag]["native_head_identity"]["passed"]} for m in (a, b)}
        rec["native_mase"] = {m: by[m][tag]["native"]["test_mase"] for m in (a, b)}
        out["datasets"][tag] = rec

    ds = out["datasets"].values()
    out["verdict"] = {
        "tunnel_entrance_identical_everywhere": all(d["tunnel"]["same_entrance"] for d in ds),
        "test_loss_argmin_identical_everywhere": all(d["test_loss"]["same_argmin"] for d in ds),
        "max_test_loss_delta_relative_to_range":
            max((d["test_loss"]["max_delta_relative_to_curve_range"] for d in ds), default=0.0),
        "max_test_mase_abs_delta": max((d["test_mase"]["max_abs_delta"] for d in ds), default=0.0),
        "any_material_curve_change": any(d["test_loss"]["material"] or d["test_mase"]["material"]
                                         for d in ds),
        "any_material_erank_change": any(v["effective_rank"]["material"]
                                         for d in ds for v in d["geometry"].values())}
    return out


def _spearman(x, y) -> float:
    """Rank correlation of two depth curves -- does the ORDERING of depths survive the mode
    change? Computed without scipy (ties broken by average rank, as scipy does)."""
    def rank(v):
        v = np.asarray(v, float)
        order = v.argsort()
        r = np.empty(len(v), float)
        r[order] = np.arange(len(v), dtype=float)
        # average ranks for ties
        for val in np.unique(v):
            m = v == val
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def print_rollout_comparison(cmp: dict) -> None:
    a, b = cmp["modes"]
    print(f"\n=== ROLLOUT-MODE COMPARISON  (delta = {b} - {a}) ===")
    print("    columns: the TUNNEL ENTRANCE under each mode, then max|d test loss| as a "
          "fraction of that\n    curve's own range, max|d MASE|, the Spearman rank correlation "
          "of the depth ordering,\n    and max|d effective rank| (headline variant).")
    print(f"{'dataset':<14}{a:<14}{b:<14}{'dtest/range':>12}{'dMASE':>9}{'rho':>7}{'d r_eff':>9}")
    for tag, d in cmp["datasets"].items():
        g = next(iter(d["geometry"].values()), {}).get("effective_rank", {})
        print(f"{d['short']:<14}{d['tunnel'][a]['point']:<14}{d['tunnel'][b]['point']:<14}"
              f"{d['test_loss']['max_delta_relative_to_curve_range']:>12.3f}"
              f"{d['test_mase']['max_abs_delta']:>9.4f}"
              f"{d['test_loss']['spearman_over_depths']:>7.3f}"
              f"{g.get('max_abs_delta', float('nan')):>9.2f}")
    v = cmp["verdict"]
    print(f"\n  tunnel entrance identical everywhere : {v['tunnel_entrance_identical_everywhere']}")
    print(f"  test-loss argmin identical everywhere: {v['test_loss_argmin_identical_everywhere']}")
    print(f"  max test-loss delta / curve range    : "
          f"{v['max_test_loss_delta_relative_to_range']:.3f}  "
          f"(material > {cmp['thresholds']['curve_relative']})")
    print(f"  any material curve change            : {v['any_material_curve_change']}")
    print(f"  any material effective-rank change   : {v['any_material_erank_change']}")


def _fmt(v):
    """CSV cell for a companion value that is absent at Emb (the companion starts at L1)."""
    return "" if v is None else f"{v:.4f}"


def write_table(summary, path):
    import csv
    with open(path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["dataset", "representation", "index", "block_index", "relative_depth",
                      "relative_position", "kind", "is_native_readout", "train_loss", "val_loss",
                      "test_loss", "test_loss_ci_lo", "test_loss_ci_hi", "test_mase",
                      "test_mase_ci_lo", "test_mase_ci_hi", "test_loss_patch0_last_context",
                      "test_loss_patch1_masked_future", "effective_rank_train",
                      "effective_rank_test", "effective_rank_companion_train",
                      "effective_rank_companion_test", "constant_readout_positions", "weight_decay", "beats_constant_floor",
                      "is_tunnel_entrance"])
        for e in summary["datasets"]:
            ti = e["tunnel"]["index"]
            for l in e["layers"]:
                wtr.writerow([e["dataset"], l["representation"], l["index"], l["block_index"],
                              f"{l['relative_depth']:.6f}", f"{l['relative_position']:.6f}",
                              l["kind"], l["is_native_readout"], f"{l['train_loss']:.6f}",
                              f"{l['val_loss']:.6f}", f"{l['test_loss']:.6f}",
                              f"{l['test_loss_ci'][0]:.6f}", f"{l['test_loss_ci'][1]:.6f}",
                              f"{l['test_mase']:.6f}", f"{l['test_mase_ci'][0]:.6f}",
                              f"{l['test_mase_ci'][1]:.6f}",
                              f"{l['test_loss_patch0_last_context']:.6f}",
                              f"{l['test_loss_patch1_masked_future']:.6f}",
                              f"{l['effective_rank'].get('train', float('nan')):.4f}",
                              f"{l['effective_rank'].get('test', float('nan')):.4f}",
                              _fmt(l["effective_rank_companion"].get("train")),
                              _fmt(l["effective_rank_companion"].get("test")),
                              sum(l["constant_across_windows"]),
                              f"{l['weight_decay']:g}", l["beats_constant_floor"],
                              l["index"] == ti])
    return path


def make_figures(summary, paths, args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from probing.cka import heatmap

    ds = summary["datasets"]
    if not ds:
        return
    pts = args.points
    x = np.arange(len(pts))
    fig_dir = paths["figures"]

    def panels(name, draw, ylab, title):
        n = len(ds)
        cols = min(4, n)
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.4 * cols, 3.6 * rows), squeeze=False)
        for ax, e in zip(axes.ravel(), ds):
            draw(ax, e)
            ax.set_xticks(x)
            ax.set_xticklabels(pts, rotation=60, ha="right", fontsize=6)
            ax.set_title(e["short"], fontsize=9)
            ax.grid(alpha=0.25)
        for ax in axes.ravel()[len(ds):]:
            ax.axis("off")
        for ax in axes[:, 0]:
            ax.set_ylabel(ylab, fontsize=8)
        fig.suptitle(title, fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for ext in ("png", "pdf"):
            fig.savefig(fig_dir / f"{name}.{ext}", dpi=190)
        plt.close(fig)

    # 1. the tunnel curve
    def draw_loss(ax, e):
        b = e["bootstrap"]["loss"]
        ax.fill_between(x, b["ci_lo"], b["ci_hi"], alpha=0.2, color="#1f77b4")
        ax.plot(x, e["val_loss_by_representation"], "s--", ms=3, color="#ff7f0e", label="val")
        ax.plot(x, e["test_loss_by_representation"], "o-", ms=3, color="#1f77b4", label="test")
        ax.axhline(e["native"]["test_loss"], ls="--", c="crimson", lw=1.1, label="TiRex native")
        ax.axhline(e["constant_forecast_floor"]["test_loss"], ls=":", c="grey", lw=1.1,
                   label="constant floor")
        ax.axvline(e["tunnel"]["index"], c="green", lw=1.1, alpha=0.7,
                   label=f"tunnel {e['tunnel']['point']}")
        ax.legend(fontsize=6)
    panels("probe_loss_by_depth", draw_loss, f"pinball (tau={TAU})",
           "TiRex Q=1 shared-patch probe: loss by depth (tunnel = first crossing of 1.05x final)")

    # 2. MASE
    def draw_mase(ax, e):
        b = e["bootstrap"]["mase"]
        ax.fill_between(x, b["ci_lo"], b["ci_hi"], alpha=0.2, color="#2ca02c")
        ax.plot(x, e["test_mase_by_representation"], "o-", ms=3, color="#2ca02c")
        ax.axhline(e["native"]["test_mase"], ls="--", c="crimson", lw=1.1, label="TiRex native")
        ax.axvline(e["tunnel"]["index"], c="green", lw=1.1, alpha=0.7)
        ax.legend(fontsize=6)
    panels("probe_mase_by_depth", draw_mase, f"MASE (m={M_SEASON})",
           "TiRex median-forecast MASE by depth, raw units")

    # 3. the per-patch diagnostic (spec section K)
    def draw_patch(ax, e):
        ax.plot(x, [l["test_loss_patch0_last_context"] for l in e["layers"]], "o-", ms=3,
                label="patch 0 (last context token)")
        ax.plot(x, [l["test_loss_patch1_masked_future"] for l in e["layers"]], "s-", ms=3,
                label="patch 1 (masked future token)")
        ax.plot(x, e["test_loss_by_representation"], "k--", lw=1, label="full H=64")
        ax.legend(fontsize=6)
    panels("per_patch_diagnostic", draw_patch, f"pinball (tau={TAU})",
           "Per-readout-position diagnostic (same shared probe, horizon split in two)")

    # 4. effective rank
    def draw_erank(ax, e):
        for sp, style in (("train", "o-"), ("test", "s--")):
            g = e["geometry"].get(sp)
            if not g:
                continue
            ax.plot(x, g["effective_rank"][ERANK_HEADLINE_VARIANT]["effective_rank"], style, ms=3,
                    label=f"{sp} headline (pos0, n rows)")
            erc = g["effective_rank"][VARIANT_ALL_FROM_L1]
            xc = [pts.index(n) for n in erc["representation_points"]]
            ax.plot(xc, erc["effective_rank"], style, ms=2, alpha=0.4,
                    label=f"{sp} companion (both pos, 2n, L1+)")
        ax.axvline(e["tunnel"]["index"], c="green", lw=1.1, alpha=0.7)
        ax.legend(fontsize=5)
    panels("effective_rank_by_depth", draw_erank, "entropy effective rank",
           f"Effective rank of the readout matrix by depth (headline: {ERANK_HEADLINE_SPLIT})")

    # 5. CKA heatmaps
    for e in ds:
        for sp in args.geometry_splits:
            for est in args.cka_estimators:
                M = np.load(paths["matrices"] / CKA_HEADLINE_VARIANT / est / sp
                            / f"cka__{e['dataset']}.npy")
                lbl = e["geometry"][sp]["variant_points"][CKA_HEADLINE_VARIANT]
                floor = e["geometry"][sp]["cka_null_floor"][CKA_HEADLINE_VARIANT][est]["mean"]
                lo = min(0.0, float(np.nanmin(M)))
                heatmap(M, lbl, lbl,
                        fig_dir / "cka" / CKA_HEADLINE_VARIANT / est / sp / f"cka__{e['dataset']}.png",
                        title=f"{e['short']} -- {est} CKA ({sp}, {CKA_HEADLINE_VARIANT}, "
                              f"null floor {floor:+.3f})",
                        vmin=lo, vmax=1.0, cbar_label=f"{est} linear CKA",
                        xaxis_label="representation", yaxis_label="representation")
    print(f"  figures -> {fig_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="paper7",
                   help="dataset roster: 'paper7' (default) or an id_data.ID_DATASET_SPECS key")
    p.add_argument("--datasets", nargs="+", default=None, help="override the suite roster")
    p.add_argument("--points", nargs="+", default=None,
                   help=f"representation points (default all {NUM_POINTS}: {' '.join(REP_NAMES)})")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--rollout-mode", default=ROLLOUT_TWO, choices=ROLLOUT_MODES,
                   help="how the H=64 horizon is produced, and therefore WHICH states are probed. "
                        f"DEFAULT '{ROLLOUT_TWO}' is the PAPER'S PRIMARY DEFINITION because it is "
                        "the released package's own default inference pathway "
                        "(max_accelerated_rollout_steps=1): K separate passes, each read at its "
                        "own last token, each with its own (loc, scale); pass k>0 is handed the "
                        "context with the horizon marked MISSING (NaN), never the previous "
                        f"forecast. '{ROLLOUT_SINGLE}' (max_accelerated_rollout_steps=K) is the "
                        "one-pass accelerated path, retained as a robustness check. The mode is "
                        "part of the cache path, so the two can never be mixed.")
    p.add_argument("--backend", default="torch", choices=("torch", "cuda"),
                   help="'torch' = the pure-PyTorch sLSTM (bfloat16 recurrence); 'cuda' = the "
                        "xLSTM custom kernels (needs the xlstm package + a GPU). The two are NOT "
                        "bit-identical; whichever is used is recorded in every cache and summary.")
    p.add_argument("--device", default=None, help="cpu / cuda (default: cuda if available)")
    p.add_argument("--probe-device", default="cpu", help="device for the tiny probe fits")
    p.add_argument("--cache-dir", default=str(REPO_ROOT / "features_cache" / "tirex"))
    p.add_argument("--out-root", default=str(OUT_DEFAULT))
    p.add_argument("--batch-size", type=int, default=64,
                   help="extraction batch size. NOT just a speed knob: it is part of the feature "
                        "cache key, because the bfloat16 sLSTM recurrence makes representations "
                        "mildly batch-size dependent on accelerator backends (see summary.json "
                        "-> extraction.why). Keep it fixed for a whole paper; prefer a LARGE "
                        "value (256), and avoid very small ones.")
    p.add_argument("--limit", type=int, default=0, help="cap windows per split (smoke runs only)")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--wd-grid", nargs="+", type=float, default=list(WD_GRID_TIREX))
    p.add_argument("--boot-b", type=int, default=BOOT_B)
    p.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    p.add_argument("--horizon", type=int, default=HORIZON)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--geometry-splits", nargs="+", default=["train", "test"],
                   choices=("train", "val", "test"))
    p.add_argument("--cka-estimators", nargs="+", default=list(CKA_ESTIMATORS),
                   choices=CKA_ESTIMATORS)
    p.add_argument("--cka-null-floor-reps", type=int, default=3)
    p.add_argument("--erank-subsamples", type=int, default=0)
    p.add_argument("--discover-only", action="store_true",
                   help="print the checkpoint geometry report and exit (loads the model, no data)")
    p.add_argument("--compare-rollout-modes", action="store_true",
                   help="run the SAME datasets under BOTH rollout modes and report whether the "
                        "probe curves, tunnel entrance, unbiased CKA and effective rank "
                        "materially differ. Writes rollout_mode_comparison.json.")
    p.add_argument("--compare-backends", action="store_true",
                   help="measure the torch vs cuda sLSTM backends on synthetic windows and exit. "
                        "Run this ONCE on the GPU before committing to a backend: the two are "
                        "different implementations of the recurrence and are not expected to "
                        "agree bit-for-bit, so the size of the gap should be recorded, not "
                        "assumed. Needs `pip install xlstm ninja` for the cuda side.")
    p.add_argument("--allow-window-mismatch", action="store_true",
                   help="downgrade the committed-window parity gate to a report")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args(argv)
    if "biased" not in a.cka_estimators:
        p.error("dropping 'biased' from --cka-estimators is REFUSED: it is the within-model "
                "parity analysis against the committed Chronos-2 CKA.")
    try:
        a.wd_grid = assert_wd_grid(a.wd_grid, a.lr)
    except ValueError as exc:
        p.error(str(exc))
    return a


def main(argv=None):
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"=== TiRex layer-wise probing (Q=1, tau={TAU}, shared patch head) ===")
    print(f"    rollout mode: {args.rollout_mode}")
    model = get_model(args.checkpoint, device=args.device, backend=args.backend)
    geom = geometry_from_model(model, C=args.context_len, H=args.horizon)
    disc = discover_geometry(model, geom)
    assert_native_geometry(model, geom)

    print(json.dumps({k: disc[k] for k in
                      ("package", "package_version", "torch_version", "checkpoint", "num_blocks",
                       "embedding_dim", "num_heads", "input_patch_size", "output_patch_size",
                       "input_ff_dim", "train_ctx_len", "quantiles", "median_index",
                       "final_norm_class", "output_head_class", "output_head_out_features",
                       "n_parameters", "head_checksum")}, indent=2))
    g = geom.as_dict()
    print(f"  C={g['C']} -> {g['n_real_context_patches']} real + {g['n_pad_patches']} NaN-pad "
          f"patches = {g['n_context_tokens']} context tokens")
    if args.rollout_mode == ROLLOUT_SINGLE:
        print(f"  [{ROLLOUT_SINGLE}] ONE pass of {g['n_tokens']} tokens "
              f"(+{g['n_forecast_patches'] - 1} rollout token); READOUT TOKENS "
              f"{g['readout_indices']}: {g['first_readout_is_last_real_context_patch']} = last "
              f"REAL context patch, {g['masked_future_readouts']} = masked future patch(es)")
    else:
        print(f"  [{ROLLOUT_TWO}] (PRIMARY -- the released package's default inference pathway) "
              f"{g['n_forecast_patches']} passes of {g['two_pass_n_tokens']} tokens, each read at "
              f"its OWN token {g['two_pass_readout_index']}:")
        for lay in g["two_pass_layout"]:
            what = ("the LAST REAL context patch" if lay["readout_is_last_real_context_patch"]
                    else "an APPENDED MISSING (NaN) patch")
            print(f"      pass {lay['pass']}: context {lay['context_len']} "
                  f"(+{lay['n_appended_missing_patches'] * g['output_patch']} NaN) -> pad "
                  f"{lay['pad_len']}, real tokens {lay['real_token_range']}, readout "
                  f"{lay['readout_index']} = {what} -> y[{lay['target_slice'][0]}:"
                  f"{lay['target_slice'][1]}]")
    print(f"  representation points ({NUM_POINTS}): {' '.join(REP_NAMES)}")
    if args.compare_backends:
        t = np.arange(args.context_len, dtype=np.float32)
        rng = np.random.default_rng(args.seed)
        ctx = np.stack([10 + 3 * np.sin(2 * np.pi * t / 24) + 0.02 * t
                        + rng.normal(0, 0.5, args.context_len).astype(np.float32)
                        for _ in range(8)]).astype(np.float32)
        rep = compare_backends(ctx, geom, checkpoint=args.checkpoint,
                               device=args.device or ("cuda" if torch.cuda.is_available()
                                                      else "cpu"))
        print(json.dumps(rep, indent=2, default=_json))
        Path(args.out_root).mkdir(parents=True, exist_ok=True)
        (Path(args.out_root) / "backend_comparison.json").write_text(
            json.dumps(rep, indent=2, default=_json))
        print(f"\n--compare-backends: wrote {args.out_root}/backend_comparison.json")
        return 0

    if args.discover_only:
        Path(args.out_root).mkdir(parents=True, exist_ok=True)
        (Path(args.out_root) / "geometry_discovery.json").write_text(
            json.dumps(disc, indent=2, default=_json))
        print(f"\n--discover-only: wrote {args.out_root}/geometry_discovery.json")
        return 0

    args.points = list(args.points) if args.points else list(REP_NAMES)
    bad = [p for p in args.points if p not in REP_NAMES]
    if bad:
        raise SystemExit(f"unknown representation points {bad}; known: {list(REP_NAMES)}")
    if args.points[-1] != REP_NAMES[-1]:
        raise SystemExit(f"the LAST probed point must be {REP_NAMES[-1]} -- the tunnel criterion "
                         "and every delta are defined relative to the native readout depth.")

    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    root = Path(args.out_root)
    modes = list(ROLLOUT_MODES) if args.compare_rollout_modes else [args.rollout_mode]
    summaries = {}
    for mode in modes:
        args.rollout_mode = mode
        summaries[mode] = run_suite(args, model, geom, disc,
                                    root / mode if len(modes) > 1 else root)
    if len(modes) > 1:
        cmp = compare_rollout_modes(summaries, args)
        (root / "rollout_mode_comparison.json").write_text(
            json.dumps(cmp, indent=2, default=_json))
        print_rollout_comparison(cmp)
        print(f"\nwrote {root}/rollout_mode_comparison.json")
    return 0


def run_suite(args, model, geom, disc, out: Path):
    out = Path(out)
    paths = {k: out / k for k in ("per_dataset", "matrices", "figures", "bootstrap_inputs")}
    for pth in [out, *paths.values()]:
        pth.mkdir(parents=True, exist_ok=True)

    tags = args.datasets if args.datasets else suite_tags(args.suite)
    print(f"\n########## ROLLOUT MODE: {args.rollout_mode} ##########")
    ck0 = head_checksum(model)
    entries = []
    for tag in tags:
        entries.append(run_dataset(tag, args, model, geom, paths))
    if head_checksum(model) != ck0:
        raise RuntimeError("the frozen native head changed during the run")
    assert_frozen(model)

    summary = {
        "experiment": "tirex_layerwise_probing", "cache_version": CACHE_VERSION,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {"checkpoint": args.checkpoint, "backend": args.backend,
                  "device": str(args.device or ("cuda" if torch.cuda.is_available() else "cpu")),
                  "tirex_version": tirex_version(), "torch_version": torch.__version__,
                  "numpy_version": np.__version__, "head_checksum": ck0,
                  "frozen": assert_frozen(model)},
        "geometry": disc["geometry"], "geometry_discovery": disc,
        "representation_points": rep_depth_table(),
        "depth_convention": {
            "relative_depth": "block_index / num_blocks -- the CROSS-MODEL coordinate; Emb=0, "
                              "Lk=k/12, L12+RMS=1.0 (the final norm adds no block, so it ties "
                              "with L12; `kind` distinguishes them)",
            "relative_position": "position_index / (n_points - 1) -- strictly monotone, safe as "
                                 "a plotting axis, NOT comparable across models"},
        "probe": {"type": "shared patch linear head", "quantiles": [TAU], "num_quantiles": 1,
                  "shape": f"Linear({MODEL_DIMS}, {geom.output_patch}) applied to all "
                           f"{geom.n_forecast_patches} readout states",
                  "objective": "mean pinball at tau=0.5 (== 0.5 * MAE)",
                  "epochs": args.epochs, "lr": args.lr, "wd_grid": list(args.wd_grid),
                  "feature_standardization": "StandardScaler fit on the STACKED train rows only",
                  "selection": "explicit validation split, no refit"},
        "tunnel_criterion": {"module": "probing.tunnel.tunnel_start",
                             "definition": "first_crossing_95",
                             "reference": REP_NAMES[-1]},
        "estimators": {"cka": "probing.cka.cka_matrix", "effective_rank":
                       "probing.spectral_metrics.spectral_metrics",
                       "cka_estimators": list(args.cka_estimators),
                       "headline_cka_estimator": HEADLINE_CKA_ESTIMATOR,
                       "cka_headline_split": CKA_HEADLINE_SPLIT,
                       "effective_rank_headline_split": ERANK_HEADLINE_SPLIT,
                       "geometry_variants": list(GEOMETRY_VARIANTS),
                       "cka_headline_variant": CKA_HEADLINE_VARIANT,
                       "effective_rank_headline_variant": ERANK_HEADLINE_VARIANT,
                       "headline_row_construction": "position 0 (the last-real-context readout "
                                                    "state) at EVERY depth -- n rows, CONSTANT "
                                                    "across depth, for BOTH estimators",
                       "companion_row_construction": "both readout states, 2n rows, L1..L12+RMS "
                                                     "only (Emb omitted: its second state is "
                                                     "constant before recurrence)"},
        "cross_model_caveat": CROSS_MODEL_CAVEAT,
        "pretraining_status_note": PROVENANCE_NOTE,
        "seeds": {"global": args.seed, "probe_init": args.seed, "bootstrap": args.seed,
                  "bootstrap_B": args.boot_b},
        "extraction": {
            "batch_size": args.batch_size,
            "batch_size_is_part_of_the_cache_key": True,
            "why": "TiRex's sLSTM recurrence runs in bfloat16; on accelerator backends a "
                   "small-batch GEMM kernel switch perturbs block 1 by ~1e-3 of its std and the "
                   "64-step x 12-block recurrence amplifies it to ~6e-2 by L6/L7. Measured on "
                   "MPS: batch 4 == batch 8 BITWISE, batch 2 differs. Extraction is bitwise "
                   "deterministic for a FIXED batch size; features are therefore never reused "
                   "across batch sizes."},
        "suite": args.suite, "rollout_mode": args.rollout_mode,
        "rollout_mode_is_primary": args.rollout_mode == ROLLOUT_TWO,
        "primary_rollout_mode": ROLLOUT_TWO,
        "primary_rollout_rationale": "the released tirex-ts package's DEFAULT inference pathway "
                                     "(max_accelerated_rollout_steps=1)",
        "rollout_mode_definition": {
            ROLLOUT_SINGLE: "max_accelerated_rollout_steps=K; one 65-token pass; readouts at "
                            "tokens 63,64; ONE (loc, scale)",
            ROLLOUT_TWO: "max_accelerated_rollout_steps=1 (package default); K passes of 64 "
                         "tokens; readout at token 63 of each; pass k is handed the context "
                         "extended by k*32 MISSING (NaN) values, never the previous forecast; "
                         "each pass has its OWN (loc, scale)"},
        "emb_policy": EMB_POLICY,
        "datasets": entries,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=_json))
    write_table(summary, out / "probe_table.csv")
    (out / "geometry_discovery.json").write_text(json.dumps(disc, indent=2, default=_json))
    if not args.no_figures:
        make_figures(summary, paths, args)

    print(f"\n=== SUMMARY [{args.rollout_mode}] ===")
    for e in entries:
        t = e["tunnel"]
        print(f"  {e['short']:<14s} tunnel {t['point']:<9s} (block {t['block_index']:>2d}, "
              f"depth {t['relative_depth']:.2f})  final-depth MASE "
              f"{e['layers'][-1]['test_mase']:.4f}  native {e['native']['test_mase']:.4f}")
    print(f"wrote {out}/summary.json, probe_table.csv, per_dataset/, matrices/, "
          f"bootstrap_inputs/")
    return summary


if __name__ == "__main__":
    raise SystemExit(main())
