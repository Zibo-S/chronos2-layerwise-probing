"""PHASE 1 — the final 14 datasets x 3 models = 42-cell run. One resumable job.

    Q1  At what depth does forecasting become RECOVERABLE?
    Q2  How does that functional transition compare with representation GEOMETRY?

    layer-wise Q=9 forecasting probes -> 5% validation tunnel entrance
    unbiased linear CKA + entropy effective rank on the SAME raw representations

No alignment, no adapters, no truncation. Those are later phases.

    python -m experiments.run_three_model_phase1 \\
        --output-root results/three_model_final \\
        --cache-root $SCRATCH/chronos2/three_model_final_cache \\
        --quantile-set q9 --device cuda --resume

WHERE TO RUN. A COMPUTE NODE with a GPU. It loads three foundation models, extracts features
for 42 model x dataset cells and fits ~700 linear probes. Only ``--audit-only`` (windows and
parity, no model) and ``--plan`` (the cell list, no data) are light enough for a login node.

EXECUTION ORDER. Dataset-major, Electricity first: the first three completed cells cover all
three model implementations on ONE dataset, so a systemic bug surfaces at cell 3 rather than at
cell 29. Windows are the expensive part per dataset, so they are built once and shared by the
three models — and cached to $SCRATCH, so a resumed run does not rebuild them.

RESUMABILITY. Every cell is built in a staging directory and moved into place atomically only
after all required artifacts validate. Re-running the SAME command skips complete cells whose
config hash matches, and refuses (loudly) to mix two protocols in one tree. A preemption, a
timeout or one dataset failing can therefore never make a later table read an incomplete run as
a finished one. See ``probing.phase1.CellStore``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, registry, window_parity as wp, window_reference as wref  # noqa: E402
from probing.phase1 import (MODELS, PHASE1_BOOT_B, PHASE1_C, PHASE1_H,                # noqa: E402
                            PHASE1_PROTOCOL_VERSION, PHASE1_TUNNEL_TOL, PHASE1_TUNNEL_TOLS,
                            TUNNEL_DEFINITION_VERSION, CellStore,
                            cell_config_hash, clean_staging, fit_config_hash, model_spec,
                            postprocess_config_hash)
from probing.phase1_cells import save_cell                                            # noqa: E402
from probing.windows import build_for                                                 # noqa: E402

CREATED_BY = "python -m experiments.run_three_model_phase1"
#: The EXPANDED-GRID / SUSTAINED-TUNNEL rerun writes here. `results/three_model_final/` holds
#: the earlier narrow-grid, first-crossing run and is deliberately NOT the default any more: it
#: is the audit trail, and a bare invocation must not be able to touch it.
DEFAULT_OUT = REPO_ROOT / "results" / "three_model_phase1_q9_expanded"


# --------------------------------------------------------------------------- #
# dataset order
# --------------------------------------------------------------------------- #
def dataset_order(tags, first: str | None) -> list[str]:
    """``first`` (default Electricity) leads, the rest keep registry order."""
    tags = list(tags)
    if first and first in tags:
        return [first] + [t for t in tags if t != first]
    return tags


# --------------------------------------------------------------------------- #
# windows, cached to scratch so a resume never rebuilds them
# --------------------------------------------------------------------------- #
def windows_for(tag: str, args) -> dict:
    """Build (or load) one dataset's canonical windows. ONE builder path for all three models.

    ``probing.windows.build_for`` dispatches on ``registry.builder(tag)`` — a DATASET fact, not
    a pretraining label — so the three models cannot disagree about which builder a dataset gets.
    C/H/seed are frozen at the committed values and cannot be overridden for a paper suite.
    """
    cache = Path(args.window_cache) if args.window_cache else None
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"windows__{args.suite}__{tag}.npz"
        if path.exists() and not args.rebuild_windows:
            with np.load(path, allow_pickle=True) as z:
                w = {k: z[k] for k in z.files if k != "meta"}
                w["meta"] = json.loads(str(z["meta"]))
            print(f"  [windows: cache HIT] {path.name}")
            return w
    t0 = time.time()
    w = build_for(tag, args.suite, C=PHASE1_C, H=PHASE1_H, seed=args.seed)
    print(f"  [windows: built] {tag} in {time.time() - t0:.1f}s")
    if cache is not None:
        arrays = {k: v for k, v in w.items() if k != "meta"}
        np.savez(cache / f"windows__{args.suite}__{tag}.npz",
                 meta=json.dumps(w["meta"], default=str), **arrays)
    return w


# --------------------------------------------------------------------------- #
# model handles, loaded once per job
# --------------------------------------------------------------------------- #
class Backbones:
    """Lazy, cached model handles. Each backbone is loaded at most once for the whole job —
    42 reloads would cost ~20 minutes for no benefit, and all three together are well under
    2 GB of parameters."""

    def __init__(self, args):
        self.args = args
        self._h = {}

    def chronos2(self):
        if "chronos2" not in self._h:
            from probing.extraction import get_pipeline
            t0 = time.time()
            pipe, cfg = get_pipeline()
            print(f"  [model] chronos-2 loaded in {time.time() - t0:.1f}s")
            self._h["chronos2"] = (pipe, cfg)
        return self._h["chronos2"]

    def timesfm3(self):
        if "timesfm3" not in self._h:
            from probing.timesfm3_last_token import get_model
            t0 = time.time()
            model, device = get_model(self.args.timesfm_checkpoint, self.args.device)
            print(f"  [model] timesfm-3 loaded in {time.time() - t0:.1f}s on {device}")
            self._h["timesfm3"] = (model, device)
        return self._h["timesfm3"]

    def tirex(self):
        if "tirex" not in self._h:
            from probing.tirex_model import geometry_from_model, get_model
            t0 = time.time()
            model = get_model(self.args.tirex_checkpoint, self.args.device,
                              backend=self.args.tirex_backend)
            geom = geometry_from_model(model, C=PHASE1_C, H=PHASE1_H)
            print(f"  [model] tirex loaded in {time.time() - t0:.1f}s")
            self._h["tirex"] = (model, geom)
        return self._h["tirex"]


# --------------------------------------------------------------------------- #
# per-cell configuration + hash
# --------------------------------------------------------------------------- #
def cell_config(model: str, tag: str, args, qvec, window_digest: str, reg_hash: str) -> dict:
    """Everything that would make two cells scientifically different.

    Included: the protocol version, the geometry (C/H/Q + the vector itself), the checkpoint,
    the extraction parameters that provably move numbers, the probe protocol, the bootstrap B,
    the dataset's window digest and the registry hash. Deliberately EXCLUDED: the git commit and
    the wall-clock — a docs commit must not invalidate 40 GPU-hours. The commit is recorded in
    the manifest instead.
    """
    import probing.phase1_chronos2 as c2
    import probing.phase1_timesfm3 as t3
    import probing.phase1_tirex as tx
    common = {
        "protocol": PHASE1_PROTOCOL_VERSION, "model": model, "dataset": tag,
        "C": PHASE1_C, "H": PHASE1_H, "quantile_set": args.quantile_set,
        "quantiles": [float(x) for x in qvec], "num_quantiles": len(qvec),
        "suite": args.suite, "seed": args.seed, "boot_b": args.boot_b,
        # POSTPROCESS half of the identity (probing.phase1.POSTPROCESS_KEYS): these are pure
        # functions of curves already saved in the cell, so changing one re-derives a tunnel
        # without refitting a probe. Everything else below is FIT half.
        "tunnel_definition": TUNNEL_DEFINITION_VERSION,
        "tunnel_tol": args.tunnel_tol,
        "tunnel_tols": [float(t) for t in args.tunnel_tols],
        "window_digest": window_digest,
        "registry_hash": reg_hash,
        "probe_epochs": args.probe_epochs, "probe_lr": args.probe_lr,
        # The probe ARCHITECTURE, explicitly in the hash rather than only implied by
        # (model, quantile_set): Linear(d, Q*P) for Chronos-2/TiRex, Linear(d, H*Q) for
        # TimesFM-3. Q=1 and Q=9 therefore differ in the hash twice over.
        "probe": model_spec(model).probe,
        "probe_out_features": model_spec(model).probe_out_features // 9 * len(qvec),
        "geometry_splits": list(args.geometry_splits),
        "cka_estimators": ["unbiased", "biased"],
    }
    # ONE grid for all three models and both quantile sets (phase1.PHASE1_WD_GRID), unless
    # --wd-grid overrides it. It is in the FIT half of the hash, so an old narrow-grid cell can
    # never satisfy this run's skip logic.
    defaults = {"chronos2": c2.WD_GRID, "timesfm3": t3.WD_GRID, "tirex": tx.WD_GRID}
    grid = [float(w) for w in (args.wd_grid if args.wd_grid else defaults[model])]
    common["wd_grid"] = grid
    if model == "chronos2":
        common.update(checkpoint="amazon/chronos-2",
                      extract_batch_size=args.chronos_batch_size,
                      readout="forecast_slots_K4")
    elif model == "timesfm3":
        common.update(checkpoint=args.timesfm_checkpoint,
                      null_wd=t3.NULL_WD, extract_batch_size=args.timesfm_batch_size,
                      feature_dtype=args.timesfm_feature_dtype,
                      detrend=not args.timesfm_no_detrend, readout="last_real_context_token")
    else:
        common.update(checkpoint=args.tirex_checkpoint,
                      extract_batch_size=args.tirex_batch_size,
                      backend=args.tirex_backend, rollout_mode=args.tirex_rollout_mode,
                      readout="two_forecast_producing_states")
    return common


# --------------------------------------------------------------------------- #
# the cell banner + footer (spec Y)
# --------------------------------------------------------------------------- #
def banner(model, tag, spec, cfg, ident, extraction=None):
    prov = registry.provenance(tag, model)
    ds = registry.spec(tag)
    print(f"\n{'=' * 86}")
    print(f"[{model}]  {ds.display_name} ({tag})")
    print(f"{'=' * 86}")
    print(f"  protocol      C={cfg['C']} H={cfg['H']} Q={cfg['num_quantiles']} "
          f"({cfg['quantile_set']})  quantiles={cfg['quantiles']}")
    print(f"  windows       {ident['n_train_windows']} train / {ident['n_val_windows']} val / "
          f"{ident['n_test_windows']} test   ({ident['n_test_series']} test "
          f"{ident['cluster_unit']}s, builder={ident['builder']})")
    print(f"  parity        {ident.get('chronos_parity')}  "
          f"[reference: {ident.get('reference_kind')}]")
    print(f"  seasonal m    {ds.seasonal_m}   role={ds.role}   domain={ds.domain}")
    print(f"  provenance    {prov.status} / {prov.evidence_scope}"
          f"{'' if prov.verified else '  (UNVERIFIED citation)'}")
    print(f"  readout       {spec.readout}")
    print(f"  probe         {spec.probe}")
    print(f"  points ({spec.n_points:2d})   {' '.join(spec.labels)}")
    if extraction is not None:
        print(f"  cache         {extraction.get('cache_hits')}")


def footer(model, tag, spec, res, tun, boot, raw, native, geom_blocks, elapsed):
    h = tun["headline"]
    head = next((b for b in geom_blocks
                 if b.get("is_headline", b["variant"] == "headline") and b["split"] == "train"),
                geom_blocks[0] if geom_blocks else None)
    _ = boot
    print(f"  ----------------------------------------------------------------------")
    ri = spec.reference_index
    print(f"  TUNNEL        {h['label']} (depth {h['depth_axis_index']}/"
          f"{spec.n_depth_points - 1}, normalized depth {h['relative_depth']:.3f}, "
          f"ratio to {spec.reference_label} {h['ratio_to_reference']:.4f})")
    print(f"  final depth   {spec.reference_label}   "
          f"val {res['val_loss'][ri]:.6f}    test {res['test_loss'][ri]:.6f}")
    if raw is not None:
        print(f"  final MASE    {float(np.asarray(raw['mase_pw'])[ri].mean()):.4f}"
              + (f"    native MASE {native['mase']:.4f}"
                 if native and native.get("available") else "    native: n/a"))
    for i in spec.diagnostic_indices:                     # reported, never on the depth axis
        print(f"  head input    {spec.labels[i]:<8s} val {res['val_loss'][i]:.6f}  "
              f"test {res['test_loss'][i]:.6f}   [diagnostic: a norm is not a model depth; "
              f"excluded from the tunnel and the depth axis]")
    if head is not None:
        # depth-axis entries only: the head-input point is not part of a depth curve
        keep = [j for j, t in enumerate(head["effective_rank"].get(
            "point_types", [phase1.BLOCK_DEPTH] * len(head["effective_rank"]["effective_rank"])))
            if t == phase1.BLOCK_DEPTH]
        er = [head["effective_rank"]["effective_rank"][j] for j in keep]
        fl = head["cka_null_floor"]["unbiased"]
        print(f"  CKA sanity    unbiased {head['variant']}/{head['split']}: diag=1, "
              f"off-diag mean {fl['mean_offdiagonal_cka']:.3f} "
              f"(null floor {fl['mean']:+.3f}, n={head['n_rows']}, d={head['feature_dim']})")
        print(f"  ERank sanity  {er[0]:.2f} (first) -> {max(er):.2f} (max) -> {er[-1]:.2f} "
              f"(final), ceiling {head['effective_rank']['max_possible_rank']:.0f}")
    nmax = int(sum(res["wd_at_grid_max"]))
    if nmax:
        print(f"  [WD CLIPPING] {nmax}/{spec.n_points} depths selected the grid MAXIMUM "
              f"({[spec.labels[i] for i in range(spec.n_points) if res['wd_at_grid_max'][i]]}) "
              "— a clipped search, not a converged selection. Widen the grid or accept "
              "deliberately.")
    print(f"  elapsed       {elapsed:.1f}s")


# --------------------------------------------------------------------------- #
# NaN / inf gate
# --------------------------------------------------------------------------- #
def assert_finite(name: str, arr) -> None:
    a = np.asarray(arr, np.float64)
    if not np.all(np.isfinite(a)):
        n = int((~np.isfinite(a)).sum())
        raise RuntimeError(f"{name}: {n}/{a.size} non-finite values. Phase 1 fails immediately "
                           "on NaN/inf where they are not explicitly allowed rather than "
                           "writing a table entry nobody can interpret.")


# --------------------------------------------------------------------------- #
# ONE CELL
# --------------------------------------------------------------------------- #
def run_cell(model: str, tag: str, w: dict, ident: dict, args, qvec, med_idx,
             backbones: Backbones, cfg: dict, config_hash: str, store: CellStore) -> dict:
    spec = model_spec(model)
    t0 = time.time()
    banner(model, tag, spec, cfg, ident)
    warnings, timings = [], {}

    # ---- extraction + targets ------------------------------------------------ #
    te0 = time.time()
    if model == "chronos2":
        import probing.phase1_chronos2 as ad
        backbones.chronos2()                       # ensure the frozen singleton is loaded
        data, extraction = ad.extract(tag, w, horizon=PHASE1_H,
                                      batch_size=args.chronos_batch_size)
        geom = None
    elif model == "timesfm3":
        import probing.phase1_timesfm3 as ad
        mdl, device = backbones.timesfm3()
        geom = ad.geometry_for(PHASE1_C, PHASE1_H)
        data, extraction = ad.extract(
            tag, w, geom=geom, cache_dir=args.cache_root, checkpoint=args.timesfm_checkpoint,
            model=mdl, device=device, batch_size=args.timesfm_batch_size,
            feature_dtype=np.dtype(args.timesfm_feature_dtype),
            detrend=not args.timesfm_no_detrend, suite=args.suite, seed=args.seed,
            force=args.force_extract)
    else:
        import probing.phase1_tirex as ad
        mdl, geom = backbones.tirex()
        data, extraction = ad.extract(
            tag, w, model=mdl, geom=geom, cache_dir=args.cache_root,
            checkpoint=args.tirex_checkpoint, backend=args.tirex_backend,
            batch_size=args.tirex_batch_size, mode=args.tirex_rollout_mode, seed=args.seed,
            verbose=args.verbose_extract)
    timings["extract_s"] = time.time() - te0
    print(f"  [extract] {timings['extract_s']:.1f}s   cache hits "
          f"{extraction.get('cache_hits')}   shapes {extraction.get('feature_shapes')}")

    # ---- probes -------------------------------------------------------------- #
    tp0 = time.time()
    common = dict(quantiles=qvec, median_idx=med_idx, device=args.device,
                  epochs=args.probe_epochs, lr=args.probe_lr, seed=args.seed,
                  wd_grid=tuple(cfg["wd_grid"]), verbose=not args.quiet_probes)
    if model == "chronos2":
        res = ad.fit_layerwise(tag, data, horizon=PHASE1_H, **common)
    elif model == "timesfm3":
        res = ad.fit_layerwise(tag, data, geom=geom, **common)
    else:
        res = ad.fit_layerwise(tag, data, geom=geom, **common)
    timings["probe_s"] = time.time() - tp0
    for k in ("train_loss", "val_loss", "test_loss"):
        assert_finite(f"{model}/{tag} {k}", res[k])

    # ---- tunnel: VALIDATION only --------------------------------------------- #
    tun = phase1.tunnel_record(spec, res["val_loss"], res["test_loss"], tol=args.tunnel_tol,
                               tols=tuple(args.tunnel_tols))

    # ---- raw-unit metrics + native baseline ---------------------------------- #
    tm0 = time.time()
    if model == "chronos2":
        raw = ad.raw_metrics(tag, data, res, qvec, med_idx)
        native = (ad.native_reference(tag, w, data, qvec, med_idx,
                                      batch_size=args.chronos_batch_size)
                  if not args.no_native else None)
    elif model == "timesfm3":
        raw = ad.raw_metrics(tag, data, res, qvec, med_idx)
        native = None if args.no_native else ad.native_reference(tag, data, res, qvec, med_idx)
    else:
        raw = ad.raw_metrics(tag, data, res, qvec, med_idx, geom)
        native = None if args.no_native else ad.native_reference(tag, data, qvec, med_idx, geom)
    timings["metrics_s"] = time.time() - tm0
    assert_finite(f"{model}/{tag} test MASE", raw["mase_pw"])

    # ---- bootstrap ----------------------------------------------------------- #
    tb0 = time.time()
    sid_test = _cluster_ids(w, "test", res)
    sid_val = _cluster_ids(w, "val", res)
    # The delta-vs-last baseline is the FINAL BLOCK (L12 / L20 / L12), not the last probed
    # point: for Chronos-2 and TiRex the last point is a final normalization, which is a
    # head-input diagnostic and not a model depth.
    ref = spec.reference_index
    boot = {
        "loss": phase1.cluster_ci(res["test_window_loss"], sid_test, args.boot_b, args.seed, ref),
        "val_loss": phase1.cluster_ci(res["val_window_loss"], sid_val, args.boot_b,
                                      args.seed, ref),
        "mase": phase1.cluster_ci(raw["mase_pw"], sid_test, args.boot_b, args.seed, ref),
        "mae": phase1.cluster_ci(raw["mae_pw"], sid_test, args.boot_b, args.seed, ref),
        # WQL is a ratio of sums, so its replicate must be formed as sum(num)/sum(den) INSIDE
        # each resample -- never as the mean of per-window ratios.
        "wql": phase1.cluster_ratio_ci(raw["wql_num_pw"], raw["wql_den_pw"], sid_test,
                                       args.boot_b, args.seed, ref),
    }
    timings["bootstrap_s"] = time.time() - tb0

    # ---- geometry: RAW representations, both estimators, both splits ---------- #
    tg0 = time.time()
    if model == "timesfm3":
        geom_blocks = ad.geometry_blocks(data, res, splits=tuple(args.geometry_splits),
                                         seed=args.seed,
                                         null_floor_reps=args.cka_null_floor_reps)
    else:
        geom_blocks = ad.geometry_blocks(data, splits=tuple(args.geometry_splits),
                                         seed=args.seed,
                                         null_floor_reps=args.cka_null_floor_reps)
    timings["geometry_s"] = time.time() - tg0

    warnings.extend(_wd_warnings(spec, res, cfg))

    # ---- persist ------------------------------------------------------------- #
    timings["total_s"] = time.time() - t0
    stage = store.begin()
    summary = save_cell(
        stage, model=model, tag=tag, spec=spec, cfg=cfg, config_hash=config_hash,
        res=res, raw=raw, native=native, boot=boot, tun=tun, geom_blocks=geom_blocks,
        ident=ident, extraction=extraction, window_meta=w["meta"],
        cluster_ids={"test": sid_test, "val": sid_val},
        contexts={"test": _rows(data, model, res, "test", "X"),
                  "val": _rows(data, model, res, "val", "X")},
        targets={"test": _targets(data, model, res, "test"),
                 "val": _targets(data, model, res, "val")},
        save_predictions=args.save_predictions,
        save_probe_artifacts=not args.no_probe_artifacts,
        timings=timings, warnings=warnings)
    final = store.commit(stage, config_hash,
                         fit_hash=phase1.fit_config_hash(cfg),
                         post_hash=phase1.postprocess_config_hash(cfg),
                         extra={"elapsed_s": timings["total_s"],
                                "tunnel_definition": tun["definition"],
                                "tunnel_layer": tun["headline"]["label"],
                                "tunnel_relative_depth": tun["headline"]["relative_depth"],
                                "first_crossing_layer": (
                                    tun.get("first_crossing", {})
                                    .get(f"first_crossing_{args.tunnel_tol:g}", {})
                                    .get("label"))})
    footer(model, tag, spec, res, tun, boot, raw, native, geom_blocks, timings["total_s"])
    print(f"  COMPLETE      {final.relative_to(args.output_root)}")
    return {"summary": summary, "path": final, "timings": timings, "warnings": warnings}


def _wd_warnings(spec, res, cfg) -> list[dict]:
    """Grid-edge warnings that distinguish a CUT-OFF search from a reached FLOOR.

    Under the expanded grid the maximum (90 at lr=1e-2) is within ~1e-3 relative of the
    bias-only no-information floor, so a grid-max selection has two very different readings.
    The closed-form ``constant_forecast_floor`` separates them, so the warning states which one
    it is instead of asserting the pessimistic one.
    """
    out = []
    n = spec.n_points
    floor = (res.get("constant_forecast_floor") or {}).get("val_loss")
    at_max = [i for i in range(n) if res["wd_at_grid_max"][i]]
    at_min = [i for i in range(n) if res["wd_at_grid_min"][i]]
    if at_max:
        ratios = ({spec.labels[i]: round(float(res["val_loss"][i]) / float(floor), 4)
                   for i in at_max} if floor else None)
        near_floor = ([spec.labels[i] for i in at_max
                       if float(res["val_loss"][i]) >= 0.98 * float(floor)] if floor else [])
        out.append({
            "kind": "wd_grid_selected_at_max",
            "layers": [spec.labels[i] for i in at_max],
            "n_layers": len(at_max), "fraction": len(at_max) / n,
            "grid_max": float(max(cfg["wd_grid"])),
            "lr_times_grid_max": float(cfg["probe_lr"]) * float(max(cfg["wd_grid"])),
            "val_loss_over_constant_forecast_floor": ratios,
            "layers_at_the_no_information_floor": near_floor,
            "meaning": (
                "the selected weight decay sits at the grid MAXIMUM. Two readings, told apart "
                "by the ratio above: ~1.0 means this depth's validation optimum IS the "
                "no-information floor (a FINDING — the probe wants to predict nothing); well "
                "below 1.0 means the search was genuinely cut off and the grid should be "
                "widened. The grid cannot be widened past lr*wd = 1 at this lr: that value is "
                "the floor itself, and beyond it AdamW diverges.")})
    if at_min:
        out.append({
            "kind": "wd_grid_selected_at_min",
            "layers": [spec.labels[i] for i in at_min],
            "n_layers": len(at_min), "fraction": len(at_min) / n,
            "grid_min": float(min(cfg["wd_grid"])),
            "meaning": ("the selected weight decay sits at the grid MINIMUM — the probe wants "
                        "less regularization than the grid offers; extend it downward if this "
                        "is widespread")})
    return out


def _cluster_ids(w, split, res):
    """Bootstrap unit id per window, restricted to the rows the probes were scored on."""
    sid = np.asarray(w[f"series_{split}"], np.int64)
    rows = res.get(f"rows_{split}")
    return sid if rows is None else sid[np.asarray(rows, np.int64)]


def _rows(data, model, res, split, key):
    a = np.asarray(data[split][key])
    rows = res.get(f"rows_{split}")
    return a if rows is None else a[np.asarray(rows, np.int64)]


def _targets(data, model, res, split):
    if model == "chronos2":
        return np.asarray(data[split]["Y"], np.float32)
    if model == "timesfm3":
        return np.asarray(res[f"target_{split}"], np.float32)
    return np.asarray(data[split]["targets"], np.float32)


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)
    out = Path(args.output_root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir(exist_ok=True)
    (out / "window_audits").mkdir(exist_ok=True)

    qvec, med_idx = phase1.quantile_set(args.quantile_set)
    qcheck = phase1.assert_canonical_quantiles()
    tags = dataset_order([t for t in registry.roster(args.roster)
                          if not args.datasets or t in args.datasets],
                         args.first_dataset)
    models = [m for m in MODELS if not args.models or m in args.models]
    if args.audit_only:
        # Windows + parity + audits, then stop. Explicit, NOT "models == []": an empty
        # --models list means "no filter" everywhere else in this driver, so relying on it
        # here would have quietly run all three backbones under a flag that promises not to.
        models = []
    reg_hash = phase1.registry_hash(args.roster)
    # Fail in seconds, not after a backbone load: the grid guard runs before any work, and the
    # tolerance set is checked for the monotonicity the sustained rule guarantees.
    wd_grid = phase1.assert_wd_grid(args.wd_grid or phase1.PHASE1_WD_GRID, args.probe_lr,
                                    null_wd=phase1.PHASE1_WD_NULL)
    if args.tunnel_tol not in set(float(t) for t in args.tunnel_tols):
        raise SystemExit(f"--tunnel-tol {args.tunnel_tol:g} is not in --tunnel-tols "
                         f"{list(args.tunnel_tols)}; the headline tolerance must be one of the "
                         "tolerances actually computed")

    removed = clean_staging(out)
    if removed:
        print(f"[resume] removed {len(removed)} leftover staging dir(s) from a killed run: "
              + ", ".join(removed[:6]))

    # ---- manifests, written BEFORE any work --------------------------------- #
    manifest = {
        "schema": "phase1_run_manifest/v1",
        "protocol": PHASE1_PROTOCOL_VERSION,
        "phase": "Phase 1 — recoverability depth + representation geometry",
        "questions": ["At what depth does forecasting become recoverable?",
                      "How does that functional transition compare with representation geometry?"],
        "measurements": ["layerwise Q=9 forecasting probes", "5% forecasting-tunnel entrance",
                         "unbiased linear CKA", "entropy effective rank"],
        "not_in_phase_1": ["alignment", "adapters", "truncation"],
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "command_line": " ".join(sys.argv),
        "output_root": str(out), "cache_root": str(args.cache_root),
        "roster": args.roster, "datasets": tags, "models": models,
        "n_cells": len(tags) * len(models),
        "C": PHASE1_C, "H": PHASE1_H, "quantile_set": args.quantile_set,
        "quantiles": [float(x) for x in qvec], "num_quantiles": len(qvec),
        "quantile_verification": qcheck,
        "median_index": med_idx,
        "tunnel": {"tolerance": args.tunnel_tol, "split": "validation",
                   "definition": TUNNEL_DEFINITION_VERSION,
                   "criterion": ("min { l in DEPTH AXIS : max_{j >= l} "
                                 "(L_val(j)/L_val(final block) - 1) <= tol }"),
                   "implementation": "probing.tunnel.sustained_tunnel_start",
                   "tolerances": [float(t) for t in args.tunnel_tols],
                   "reported_tolerances": [float(t) for t in phase1.PHASE1_REPORTED_TOLS],
                   "caveat": phase1.TUNNEL_DEFINITION_CAVEAT,
                   "first_crossing_note": ("probing.tunnel.tunnel_start is still computed and "
                                           "saved, under the name first_crossing_<tol>; it is "
                                           "a diagnostic and is never called the tunnel")},
        "weight_decay_grid": {
            "grid": [float(w) for w in wd_grid],
            "n_candidates": len(wd_grid),
            "is_project_default": list(wd_grid) == list(phase1.PHASE1_WD_GRID),
            "shared_by": "all three models, both quantile sets",
            "lr": args.probe_lr,
            "max_lr_times_wd": args.probe_lr * max(wd_grid),
            "decoupled_limit": phase1.PHASE1_WD_DECOUPLED_LIMIT,
            "null_wd_never_selectable": phase1.PHASE1_WD_NULL,
            "ceiling_reason": (
                "AdamW's decoupled decay multiplies the weight by (1 - lr*wd) each step. At "
                f"lr={args.probe_lr:g} the candidate wd={phase1.PHASE1_WD_NULL:g} gives "
                "lr*wd = 1 and zeroes the weight every step (that is the no-information NULL, "
                "not a stronger regularizer), and anything above it diverges with alternating "
                "sign. The grid therefore continues log-spaced UP TO the wall, not through it.")},
        "bootstrap_B": args.boot_b, "seed": args.seed,
        "cka_estimators": ["unbiased (headline)", "biased (diagnostic)"],
        "effective_rank": ("probing.spectral_metrics.spectral_metrics: squared singular values "
                           "of the observation-centred matrix, natural-log spectral entropy, "
                           "exp(H)"),
        "geometry_splits": list(args.geometry_splits),
        "registry_hash": reg_hash,
        "mase_definition": registry.MASE_DEFINITION,
        "seasonal_m_by_dataset": {t: registry.seasonal_m(t) for t in tags},
        "model_specs": {m: model_spec(m).as_dict() for m in models},
        "tirex": {"rollout_mode": args.tirex_rollout_mode, "backend": args.tirex_backend,
                  "batch_size": args.tirex_batch_size,
                  "batch_size_note": ("part of the cache key: bfloat16 sLSTM makes the "
                                      "representation mildly batch-size dependent")},
        "timesfm3": {"checkpoint": args.timesfm_checkpoint,
                     "feature_dtype": args.timesfm_feature_dtype,
                     "batch_size": args.timesfm_batch_size},
        "chronos2": {"checkpoint": "amazon/chronos-2",
                     "batch_size": args.chronos_batch_size},
        "probe": {"epochs": args.probe_epochs, "lr": args.probe_lr,
                  "optimizer": "AdamW, full batch, weight decay on the WEIGHT only",
                  "selection": "weight decay chosen on the DEDICATED validation split, no refit"},
        "environment": phase1.environment_record(args.device),
        "cross_model_loss_caveat": phase1.CROSS_MODEL_LOSS_CAVEAT,
        "geometry_caveat": phase1.GEOMETRY_CAVEAT,
    }
    phase1.atomic_write_json(out / "run_manifest.json", manifest)
    phase1.atomic_write_json(out / "environment.json", manifest["environment"])
    phase1.atomic_write_json(out / "dataset_registry_snapshot.json",
                             phase1.registry_snapshot(args.roster))
    phase1.write_csv(out / "provenance_matrix.csv", phase1.provenance_rows(args.roster),
                     ["model", "dataset", "display_name", "role", "domain", "freq",
                      "seasonal_m", "status", "evidence_scope", "verified", "citation"])
    unver = registry.unverified_provenance(args.roster, MODELS)
    if unver:
        print(f"[provenance] {len(unver)}/{len(tags) * len(MODELS)} cells have an UNVERIFIED "
              f"citation. Usable as a working assumption; NOT citable until checked: "
              + ", ".join(f"{t}x{m}" for t, m in unver[:8])
              + (" ..." if len(unver) > 8 else ""))

    if args.plan:
        _print_plan(tags, models, out, args, qvec, reg_hash)
        return 0

    backbones = Backbones(args)
    cells_state = _load_cells_state(out)
    failures = []

    for tag in tags:
        print(f"\n{'#' * 86}\n# DATASET {tag}  ({registry.display_name(tag)})\n{'#' * 86}")
        try:
            w = windows_for(tag, args)
        except Exception as exc:
            print(f"[FAIL] {tag}: window construction: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            for m in models:
                failures.append((m, tag, f"windows: {exc}"))
                _set_cell(cells_state, m, tag, "failed", reason=str(exc))
            _save_cells_state(out, cells_state)
            continue

        ident = wp.parity_for(tag, w, args, CREATED_BY, model_label="Phase-1")
        phase1.atomic_write_json(out / "window_audits" / f"{tag}.json", {
            "dataset": tag, **ident,
            "window_digest": wref.window_digest(w),
            "seasonal_m": registry.seasonal_m(tag),
            "series_cap": w["meta"].get("series_cap"),
            "valtest_budget": w["meta"].get("valtest_budget"),
            "split_mode": w["meta"].get("split_mode"),
            "note": ("all three models are evaluated on THESE windows; the digest covers the "
                     "test contexts themselves, not just their identifiers")})
        digest = wref.window_digest(w)

        for model in models:
            cfg = cell_config(model, tag, args, qvec, digest, reg_hash)
            chash = cell_config_hash(cfg)
            fhash, phash = fit_config_hash(cfg), postprocess_config_hash(cfg)
            store = CellStore(out, model, tag)
            status, reason = store.status(chash, fit_hash=fhash, post_hash=phash)
            if status == "stale_postprocess" and not args.force_recompute:
                # The probes are fine; only the TUNNEL DEFINITION moved. Refitting would burn
                # GPU hours to reproduce identical weights, so the driver refuses and names the
                # no-GPU repair instead of quietly doing either wrong thing.
                print(f"\n[REFUSED — no refit needed] {model}/{tag}\n    {reason}\n"
                      f"  Re-derive it without a GPU:\n"
                      f"    python -m experiments.rebuild_phase1_tunnels "
                      f"--output-root {args.output_root}\n"
                      "  or pass --force-recompute to refit the probes from scratch anyway.")
                failures.append((model, tag, "stale postprocess (tunnel definition changed)"))
                _set_cell(cells_state, model, tag, "failed",
                          reason="stale postprocess — run rebuild_phase1_tunnels")
                _save_cells_state(out, cells_state)
                continue
            if status == "complete" and args.resume:
                print(f"\n[skip] {model}/{tag}: COMPLETE and config hash matches ({reason})")
                _set_cell(cells_state, model, tag, "complete", config_hash=chash)
                continue
            if status == "incompatible" and not args.force_recompute:
                msg = (f"{model}/{tag} already exists under a DIFFERENT configuration.\n"
                       f"    {reason}\n"
                       "  Overwriting it silently would mix two protocols in one results tree; "
                       "keeping it would report the old protocol's numbers under this run's "
                       "manifest. Choose deliberately:\n"
                       "    --force-recompute      recompute and replace it, or\n"
                       "    --output-root <new>    start a separate run id.")
                print(f"\n[REFUSED] {msg}")
                failures.append((model, tag, "incompatible config hash"))
                _set_cell(cells_state, model, tag, "failed", reason="incompatible config hash")
                _save_cells_state(out, cells_state)
                continue
            if status == "corrupt":
                print(f"\n[repair] {model}/{tag}: {reason} — recomputing")

            _set_cell(cells_state, model, tag, "running", config_hash=chash)
            _save_cells_state(out, cells_state)
            try:
                r = run_cell(model, tag, w, ident, args, qvec, med_idx, backbones, cfg,
                             chash, store)
                _set_cell(cells_state, model, tag, "complete", config_hash=chash,
                          elapsed_s=r["timings"]["total_s"], timings=r["timings"],
                          tunnel=r["summary"]["tunnel"]["label"])
            except Exception as exc:                          # one cell must not kill the job
                print(f"\n[FAIL] {model}/{tag}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                store.abandon(store.staging())
                failures.append((model, tag, f"{type(exc).__name__}: {exc}"))
                _set_cell(cells_state, model, tag, "failed", reason=f"{type(exc).__name__}: {exc}")
            _save_cells_state(out, cells_state)
            if args.rebuild_tables_each_cell:
                _rebuild_tables(out, args.roster, quiet=True)

        del w

    if args.audit_only:
        print(f"\n{'=' * 86}\nWINDOW AUDIT ONLY -- no model was loaded and no cell was written."
              f"\n  audits: {out / 'window_audits'}"
              f"\n  {len(tags)} dataset(s) built and parity-checked; "
              f"{len(failures)} failed\n{'=' * 86}")
        return 1 if failures and args.fail_on_error else 0
    _rebuild_tables(out, args.roster)
    _print_summary(out, cells_state, failures, tags, models)
    return 1 if failures and args.fail_on_error else 0


def _rebuild_tables(out, roster, quiet=False):
    from experiments.make_phase1_tables import build
    r = build(out, roster)
    if not quiet:
        print(f"\n[combined] {r['n_complete']}/{r['n_total']} cells -> {r['out']}")
    return r


def _print_plan(tags, models, out, args, qvec, reg_hash):
    print(f"\nPHASE-1 PLAN  ({len(tags)} datasets x {len(models)} models = "
          f"{len(tags) * len(models)} cells)")
    print(f"  output   {out}")
    print(f"  cache    {args.cache_root}")
    print(f"  Q={len(qvec)} {list(map(float, qvec))}   C={PHASE1_C} H={PHASE1_H}   B={args.boot_b}")
    print(f"  tunnel   {TUNNEL_DEFINITION_VERSION}  headline tol {args.tunnel_tol:g}  "
          f"all tols {list(args.tunnel_tols)}")
    grid = args.wd_grid or phase1.PHASE1_WD_GRID
    print(f"  wd grid  {[float(w) for w in grid]}   lr {args.probe_lr:g}   "
          f"max lr*wd {args.probe_lr * max(grid):.2f} (< {phase1.PHASE1_WD_DECOUPLED_LIMIT:g})")
    print(f"  registry {reg_hash}")
    print(f"\n  {'#':>3}  {'model':<9} {'dataset':<28} {'m':>4} {'role':<8} status")
    i = 0
    for tag in tags:
        for m in models:
            i += 1
            st = CellStore(out, m, tag).read_marker()
            print(f"  {i:>3}  {m:<9} {tag:<28} {registry.seasonal_m(tag):>4} "
                  f"{registry.role(tag):<8} "
                  f"{'complete' if st else 'pending'}")


def _print_summary(out, state, failures, tags, models):
    done = sum(1 for v in state.get("cells", {}).values() if v.get("status") == "complete")
    total = len(tags) * len(models)
    el = [v["elapsed_s"] for v in state.get("cells", {}).values() if v.get("elapsed_s")]
    print(f"\n{'=' * 86}\nPHASE-1 RUN SUMMARY   {done}/{total} cells complete")
    if el:
        print(f"  cell wall time: median {np.median(el):.0f}s  max {max(el):.0f}s  "
              f"total {sum(el) / 3600:.2f} h")
    if failures:
        print(f"  FAILURES ({len(failures)}):")
        for m, t, why in failures:
            print(f"    {m}/{t}: {why}")
        print("  Re-submitting the SAME command retries only these; complete cells are skipped.")
    print(f"  results: {out}\n{'=' * 86}")


def _load_cells_state(out):
    p = Path(out) / "cells.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            pass
    return {"schema": "phase1_cells/v1", "cells": {}}


def _set_cell(state, model, tag, status, **kw):
    state["cells"][f"{model}/{tag}"] = {
        "model": model, "dataset": tag, "status": status,
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        **kw}


def _save_cells_state(out, state):
    state["updated_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")
    counts = {s: 0 for s in phase1.STATUSES}
    for v in state["cells"].values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    state["counts"] = counts
    phase1.atomic_write_json(Path(out) / "cells.json", state)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)

    g = p.add_argument_group("where things go")
    g.add_argument("--output-root", default=str(DEFAULT_OUT),
                   help="DURABLE results, inside the repo checkout (default: %(default)s)")
    g.add_argument("--cache-root", default=os.environ.get(
        "PHASE1_CACHE_ROOT", str(Path(os.environ.get("SCRATCH", "/tmp")) /
                                 "chronos2" / "three_model_final_cache")),
                   help="HEAVY feature caches; put this on $SCRATCH")
    g.add_argument("--window-cache", default=None,
                   help="cache built windows here (default: <cache-root>/windows) so a resumed "
                        "run does not rebuild them; window building dominates the wall time")
    g.add_argument("--rebuild-windows", action="store_true")

    g = p.add_argument_group("what to run")
    g.add_argument("--roster", default=phase1.ROSTER)
    g.add_argument("--suite", default="paper14", choices=["paper7", "paper14"])
    g.add_argument("--datasets", nargs="+", default=None)
    g.add_argument("--models", nargs="+", default=None, choices=list(MODELS))
    g.add_argument("--first-dataset", default=phase1.FIRST_DATASET,
                   help="run this dataset first so all three models are exercised at cell 3")
    g.add_argument("--quantile-set", default="q9", choices=sorted(phase1.PHASE1_QUANTILE_SETS),
                   help="Q=9 is THE Phase-1 setting; q1 is kept for later robustness work")
    g.add_argument("--plan", action="store_true", help="print the cell list and exit (no data)")
    g.add_argument("--resume", action="store_true", default=True,
                   help="skip cells already COMPLETE under the same config hash (default on)")
    g.add_argument("--no-resume", dest="resume", action="store_false")
    g.add_argument("--force-recompute", action="store_true",
                   help="replace cells whose config hash differs (otherwise REFUSED)")
    g.add_argument("--fail-on-error", action="store_true",
                   help="exit non-zero if any cell failed (default: exit 0 so a partial run "
                        "still writes its tables)")
    g.add_argument("--rebuild-tables-each-cell", action="store_true", default=True,
                   help="regenerate combined tables after every cell so the run is inspectable "
                        "while it is still going (default on; cheap)")
    g.add_argument("--no-rebuild-tables-each-cell", dest="rebuild_tables_each_cell",
                   action="store_false")

    g = p.add_argument_group("window parity")
    g.add_argument("--reference-mode", default="require", choices=list(wref.REFERENCE_MODES),
                   help="'require' aborts a dataset with no window reference (default); "
                        "'create' writes one from THIS run's windows; 'allow-missing' is "
                        "opt-in and must not be used for a reported number")
    g.add_argument("--force-reference", action="store_true")
    g.add_argument("--reference-root", default=None)
    g.add_argument("--allow-window-mismatch", action="store_true",
                   help="downgrade a parity FAILURE to a report — never for a paper number")
    g.add_argument("--audit-only", action="store_true",
                   help="build windows, check parity, write the audits, and stop (no model)")

    g = p.add_argument_group("probe protocol")
    g.add_argument("--probe-epochs", type=int, default=300)
    g.add_argument("--probe-lr", type=float, default=1e-2)
    g.add_argument("--tunnel-tol", type=float, default=PHASE1_TUNNEL_TOL,
                   help="headline tunnel tolerance (default 0.05 = the 5%% rule)")
    g.add_argument("--tunnel-tols", type=float, nargs="+", default=list(PHASE1_TUNNEL_TOLS),
                   help="every tolerance the sustained entrance is saved at; part of the "
                        "POSTPROCESS half of the cell hash, so changing it never refits a probe")
    g.add_argument("--wd-grid", type=float, nargs="+", default=None,
                   help="override the shared weight-decay grid (normally left alone: the "
                        "default is probing.phase1.PHASE1_WD_GRID, one grid for all three "
                        "models and both quantile sets). Refused if any candidate has "
                        "lr*wd >= 1 or equals the null baseline.")
    g.add_argument("--boot-b", type=int, default=PHASE1_BOOT_B)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--quiet-probes", action="store_true")

    g = p.add_argument_group("geometry")
    g.add_argument("--geometry-splits", nargs="+", default=["test", "train"],
                   choices=["train", "val", "test"],
                   help="both committed conventions are produced (CKA headline was test, "
                        "effective rank was train), so neither has to be re-run")
    g.add_argument("--cka-null-floor-reps", type=int, default=3)

    g = p.add_argument_group("models / environment")
    g.add_argument("--device", default=os.environ.get("PHASE1_DEVICE", None))
    g.add_argument("--chronos-batch-size", type=int, default=128)
    g.add_argument("--timesfm-checkpoint",
                   default=os.environ.get("TIMESFM3_CHECKPOINT", "google/timesfm-3.0-pytorch"))
    g.add_argument("--timesfm-batch-size", type=int, default=64)
    g.add_argument("--timesfm-feature-dtype", default="float32", choices=["float32", "float16"])
    g.add_argument("--timesfm-no-detrend", action="store_true")
    g.add_argument("--tirex-checkpoint", default=os.environ.get("TIREX_CHECKPOINT",
                                                                "NX-AI/TiRex"))
    g.add_argument("--tirex-backend", default="torch", choices=["torch", "cuda"],
                   help="pick ONE for the whole paper; xLSTM's cuda kernel is not bit-compatible")
    g.add_argument("--tirex-rollout-mode", default="two_pass",
                   choices=["two_pass", "single_pass"],
                   help="two_pass is the released package's own default and the PRIMARY "
                        "definition; single_pass is the documented robustness mode")
    g.add_argument("--tirex-batch-size", type=int, default=256,
                   help="a REPRODUCIBILITY parameter, not a speed knob: it is part of the "
                        "TiRex cache key (bfloat16 sLSTM batch sensitivity)")
    g.add_argument("--force-extract", action="store_true")
    g.add_argument("--verbose-extract", action="store_true")
    g.add_argument("--no-native", action="store_true",
                   help="skip the native-head baselines (they are comparison only and never "
                        "part of tunnel selection)")
    g.add_argument("--no-probe-artifacts", action="store_true",
                   help="skip saving the frozen probe weights. They are NOT needed for any "
                        "Phase-1 figure (the predictions cover those) but are ~55 MB per "
                        "TimesFM-3 cell, ~1 GB over the full run")
    g.add_argument("--save-predictions", default="val+test",
                   choices=["none", "test", "val+test"],
                   help="full (n, Q, H) forecasts + targets + contexts, so calibration and "
                        "per-window figures are regenerable without a GPU")

    a = p.parse_args(argv)
    a.output_root = Path(a.output_root)
    a.cache_root = Path(a.cache_root)
    if a.window_cache is None:
        a.window_cache = a.cache_root / "windows"
    return a


if __name__ == "__main__":
    raise SystemExit(main())
