"""Phase 2d driver: content-mean-pooled vs architecture-aware layerwise readouts.

One cell = one (model, dataset). Per cell: read the committed Phase-1 cell (native arm, never
refit) and reproduce its curves + entrances from artifacts; load the SAME windows and the same
forward pass's states; fit the Phase-2d arms (pooled for all three models; concat and
native_mean for Chronos-2 and TiRex) with the model's own Phase-1 fitter and the Phase-1 cell's
own lr / epochs / wd grid; score every arm on the same rows; paired cluster bootstrap against
native and along the one-factor decomposition. Restartable: a COMPLETE cell with the same config
hash is skipped; a different hash is refused unless --force-recompute.

    # login node (seconds): what would run, from the committed Phase-1 configs
    python -m experiments.run_phase2d_pooled --plan
    # GPU smoke (TiRex x Electricity + one-batch checks for all three models + native refit)
    python -m experiments.run_phase2d_pooled --smoke --device cuda
    # full, one model per job
    python -m experiments.run_phase2d_pooled --models tirex --device cuda

Writes ONLY: --output-root (repo, compact artifacts), --pooled-cache-root and --artifact-root
(scratch). Reads Phase-1 trees, caches and windows read-only. Never touches Phase 2c.
"""

from __future__ import annotations

import argparse
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

from probing import phase1, phase2, registry                                      # noqa: E402
from probing import phase2d_pooled as p2d                                         # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "three_model_phase2d_pooled"
SMOKE_OUT = REPO_ROOT / "results" / "three_model_phase2d_pooled_smoke"
SMOKE_MODEL, SMOKE_DATASET = "tirex", "monash_electricity_hourly"
CREATED_BY = "python -m experiments.run_phase2d_pooled"
#: trees Phase 2d must never write into (Phase 1, Phase 2 H3/H4, the running Phase 2c LODO)
PROTECTED = ("three_model_phase1_q9_expanded", "three_model_phase1_q9_loop_lowlr",
             "three_model_phase1_q1", "three_model_phase1_q1_loop_lowlr", "three_model_final",
             "three_model_phase2", "three_model_phase2_lodo", "three_model_phase2_nested",
             "three_model_phase2_smoke", "three_model_phase2_optcheck")


def _scratch(sub: str):
    s = os.environ.get("SCRATCH")
    return str(Path(s) / sub) if s else None


def assert_output_root(out: Path) -> Path:
    out = Path(out).resolve()
    for name in PROTECTED:
        p = (REPO_ROOT / "results" / name).resolve()
        if out == p or p in out.parents:
            raise SystemExit(f"refusing --output-root {out}: it is inside the protected tree "
                             f"results/{name} (Phase 1 / Phase 2 / Phase 2c)")
    return out


# --------------------------------------------------------------------------- #
# windows + Phase-1 sources (read-only)
# --------------------------------------------------------------------------- #
def windows_readonly(tag: str, window_cache, suite: str) -> dict:
    if not window_cache:
        raise SystemExit("no --window-cache (set $SCRATCH or $PHASE1_WINDOWS)")
    path = Path(window_cache) / f"windows__{suite}__{tag}.npz"
    if not path.exists():
        raise phase2.Phase1DependencyError(
            f"no Phase-1 window cache at {path}. Phase 2d never rebuilds windows: point "
            "--window-cache at $SCRATCH/chronos2/phase1_shared_windows.")
    with np.load(path, allow_pickle=True) as z:
        w = {k: z[k] for k in z.files if k != "meta"}
        w["meta"] = json.loads(str(z["meta"]))
    return w


def phase1_source(model: str, tag: str, args):
    root = phase2.resolve_phase1_root(tag, args.phase1_root,
                                      phase2.parse_overrides(args.phase1_override))
    cell = phase2.Phase1Cell(root, model, tag)
    cell.marker()
    cfg = cell.config()
    for k, want in (("quantile_set", "q9"), ("suite", args.suite),
                    ("tunnel_definition", phase1.TUNNEL_DEFINITION_VERSION)):
        if cfg.get(k) != want:
            raise phase2.Phase1DependencyError(f"{cell.path}: {k}={cfg.get(k)!r}, Phase 2d "
                                               f"compares against the {want!r} headline")
    return cell, cfg, root


class Backbones:
    """Each checkpoint loaded at most once per job."""

    def __init__(self, args):
        self.args, self._h = args, {}

    def chronos2(self):
        if "chronos2" not in self._h:
            from probing.extraction import get_pipeline
            self._h["chronos2"] = get_pipeline()[0]
        return self._h["chronos2"]

    def timesfm3(self, checkpoint):
        if "timesfm3" not in self._h:
            from probing.timesfm3_last_token import get_model
            self._h["timesfm3"] = get_model(checkpoint, self.args.device)
        return self._h["timesfm3"]

    def tirex(self, checkpoint, backend):
        if "tirex" not in self._h:
            from probing.tirex_model import geometry_from_model, get_model
            mdl = get_model(checkpoint, self.args.device, backend=backend)
            self._h["tirex"] = (mdl, geometry_from_model(mdl, C=phase1.PHASE1_C,
                                                         H=phase1.PHASE1_H))
        return self._h["tirex"]


def cell_config(model: str, tag: str, p1cfg: dict, p1root: Path, args, arms, qvec) -> dict:
    ext = {"chronos2": ("checkpoint", "extract_batch_size", "readout"),
           "timesfm3": ("checkpoint", "extract_batch_size", "feature_dtype", "detrend",
                        "null_wd", "readout"),
           "tirex": ("checkpoint", "extract_batch_size", "backend", "rollout_mode", "readout")}
    return {"protocol": p2d.PROTOCOL, "kind": p2d.KIND, "model": model, "dataset": tag,
            "arms": list(arms), "C": phase1.PHASE1_C, "H": phase1.PHASE1_H,
            "quantile_set": "q9", "quantiles": [float(x) for x in qvec],
            "suite": args.suite, "seed": int(p1cfg["seed"]), "boot_b": int(args.boot_b),
            "tunnel_definition": phase1.TUNNEL_DEFINITION_VERSION,
            "tunnel_tol": p2d.HEADLINE_TOL, "tunnel_tols": list(p2d.TUNNEL_TOLS),
            # the TRAINING PROTOCOL IS THE PHASE-1 CELL'S OWN (LOOP keeps lr 1e-3 + its grid)
            "probe_epochs": int(p1cfg["probe_epochs"]), "probe_lr": float(p1cfg["probe_lr"]),
            "wd_grid": [float(x) for x in p1cfg["wd_grid"]],
            "phase1_root": str(Path(p1root).resolve().relative_to(REPO_ROOT))
            if Path(p1root).resolve().is_relative_to(REPO_ROOT) else str(p1root),
            "phase1_config_hash": p1cfg.get("config_hash"),
            "phase1_window_digest": p1cfg.get("window_digest"),
            "phase1_extraction": {k: p1cfg.get(k) for k in ext[model]},
            "pooled_cache_version": p2d.CACHE_VERSION, "pooling": p2d.POOLING[model],
            "readout_params": {a: p2d.readout_params(model, a, len(qvec))
                               for a in (p2d.ARM_NATIVE,) + tuple(arms)}}


# --------------------------------------------------------------------------- #
# one cell
# --------------------------------------------------------------------------- #
def fit_arms(model, tag, D, arms, p1cfg, qvec, med, args) -> dict:
    """{arm: {"res", "raw", "cid_test", "cid_val"}} -- every arm on the same rows."""
    sp = D["splits"]
    common = dict(quantiles=qvec, median_idx=med, device=args.device,
                  epochs=int(p1cfg["probe_epochs"]), lr=float(p1cfg["probe_lr"]),
                  seed=int(p1cfg["seed"]), verbose=not args.quiet_probes)
    out = {}
    for arm in arms:
        t0 = time.time()
        print(f"  [arm] {arm}: {p2d.readout_params(model, arm)['description']}", flush=True)
        if model == "timesfm3":
            import probing.phase1_timesfm3 as t3
            res = t3.fit_layerwise(tag, sp, geom=D["geom"], wd_grid=tuple(p1cfg["wd_grid"]),
                                   null_wd=p1cfg.get("null_wd"), **common)
            res["probe_description"] = p2d.readout_params(model, arm)["description"]
            raw = t3.raw_metrics(tag, sp, res, qvec, med)
            cid_t = sp["test"]["series"][res["rows_test"]]
            cid_v = sp["val"]["series"][res["rows_val"]]
        else:
            data = {s: {"feats": p2d.arm_features(arm, pooled=sp[s]["pooled"],
                                                  native_states=sp[s]["native_states"]),
                        "Y": sp[s]["Y"]} for s in p2d.SPLITS}
            res = p2d.fit_shared_layerwise(model, data, wd_grid=p1cfg["wd_grid"],
                                           description=p2d.readout_params(model, arm)
                                           ["description"], **common)
            if model == "chronos2":
                import probing.phase1_chronos2 as c2
                raw = c2.raw_metrics(tag, sp, res, qvec, med)
            else:
                import probing.phase1_tirex as tx
                raw = tx.raw_metrics(tag, sp, res, qvec, med, D["geom"])
            cid_t, cid_v = sp["test"]["series"], sp["val"]["series"]
        for k in ("train_loss", "val_loss", "test_loss"):
            if not np.all(np.isfinite(res[k])):
                raise RuntimeError(f"{model}/{tag}/{arm}: non-finite {k}")
        out[arm] = {"res": res, "raw": raw, "cid_test": np.asarray(cid_t),
                    "cid_val": np.asarray(cid_v), "seconds": time.time() - t0}
        print(f"  [arm] {arm} done in {out[arm]['seconds']:.0f}s", flush=True)
    return out


def load_data(model, tag, w, p1cfg, arrays, args, bb) -> dict:
    if model == "chronos2":
        return p2d.load_chronos2(tag, w, arrays, cache_dir=args.chronos2_cache_dir)
    if model == "timesfm3":
        mdl, dev = bb.timesfm3(p1cfg["checkpoint"])
        return p2d.load_timesfm3(tag, w, p1cfg, arrays, phase1_cache_root=args.phase1_cache_root,
                                 pooled_cache_root=args.pooled_cache_root, model=mdl, device=dev,
                                 force_extract=args.force_extract)
    mdl, geom = bb.tirex(p1cfg["checkpoint"], p1cfg.get("backend", "torch"))
    return p2d.load_tirex(tag, w, p1cfg, arrays, phase1_cache_root=args.phase1_cache_root,
                          pooled_cache_root=args.pooled_cache_root, model=mdl, geom=geom,
                          force_extract=args.force_extract)


def run_cell(model, tag, args, bb, qvec, med, arms) -> dict:
    from probing.window_reference import window_digest
    t0 = time.time()
    spec = phase1.model_spec(model)
    cell, p1cfg, p1root = phase1_source(model, tag, args)
    cfg = cell_config(model, tag, p1cfg, p1root, args, arms, qvec)
    cfg["config_hash"] = p2d.config_hash(cfg)
    store = phase2.Phase2CellStore(args.output_root, p2d.KIND, model, tag, required=p2d.REQUIRED)
    st, why = store.status(cfg["config_hash"])
    if st == "complete" and not args.force_recompute:
        print(f"[skip] {model}/{tag}: COMPLETE ({why})")
        return {"status": "skipped"}
    if st in ("incompatible", "corrupt") and not args.force_recompute:
        raise RuntimeError(f"{model}/{tag}: existing cell is {st} ({why}); pass --force-recompute "
                           "or a new --output-root")
    print(f"\n{'=' * 86}\n[{model}] {registry.display_name(tag)} ({tag})   Phase-1 source "
          f"{cfg['phase1_root']}\n  lr={cfg['probe_lr']:g} epochs={cfg['probe_epochs']} "
          f"wd grid={len(cfg['wd_grid'])} values (max {max(cfg['wd_grid']):g})   arms={arms}\n"
          f"{'=' * 86}", flush=True)

    w = windows_readonly(tag, args.window_cache, args.suite)
    dig = window_digest(w)
    if dig != p1cfg.get("window_digest"):
        raise phase2.Phase1DependencyError(f"{model}/{tag}: window digest {dig} != Phase-1 "
                                           f"{p1cfg.get('window_digest')}")
    native = p2d.native_from_phase1(cell, spec)
    repro = p2d.reproduce_native(native, spec)
    print(f"  [native] reproduced from artifacts: curves max rel {repro['max_rel_val_curve']:.1e}"
          f" / {repro['max_rel_test_curve']:.1e}; entrance {repro['entrances']['tol_0.05']}")
    timings = {}
    tl = time.time()
    D = load_data(model, tag, w, p1cfg, cell.arrays(), args, bb)
    timings["load_extract_s"] = time.time() - tl
    tf = time.time()
    arm_res = fit_arms(model, tag, D, arms, p1cfg, qvec, med, args)
    timings["fit_s"] = time.time() - tf
    timings["per_arm_fit_s"] = {a: v["seconds"] for a, v in arm_res.items()}

    artifacts = {}
    if not args.no_probe_artifacts:
        if not args.artifact_root:
            raise SystemExit("no --artifact-root (set $SCRATCH); readout checkpoints are kept "
                             "out of the repo -- or pass --no-probe-artifacts")
        for a, v in arm_res.items():
            artifacts[a] = p2d.save_probe_artifacts(args.artifact_root, model, tag, a, v["res"])
    gates = {"window_digest": dig, "native_reproduction": {k: v for k, v in repro.items()
                                                            if k != "tunnel"},
             "data": D["gates"], "row_alignment": "cluster ids equal per arm (asserted)"}
    tb = time.time()
    bundle = p2d.assemble_cell(model, tag, native=native, repro=repro, arms=arm_res, cfg=cfg,
                               B=args.boot_b, seed=int(p1cfg["seed"]), gates=gates,
                               extra={"provenance": D["provenance"],
                                      "probe_artifacts": artifacts,
                                      "phase1_cell": str(cell.path)})
    timings["bootstrap_s"] = time.time() - tb
    timings["total_s"] = time.time() - t0
    bundle["summary"]["timings"] = timings
    stage = store.begin()
    try:
        p2d.write_cell(stage, bundle)
        final = store.commit(stage, cfg["config_hash"],
                             extra={"elapsed_s": timings["total_s"],
                                    "entrances": {a: s["headline_entrance"] for a, s in
                                                  bundle["summary"]["per_arm"].items()}})
    except Exception:
        store.abandon(stage)
        raise
    _print_cell(bundle["summary"])
    return {"status": "complete", "path": str(final), "timings": timings, "data": D,
            "native": native, "p1cfg": p1cfg}


def _print_cell(s):
    print(f"  {'arm':<12s} {'params':>10s} {'entrance':>9s} {'final val':>10s} "
          f"{'final test':>11s} {'final MASE':>11s}")
    for a, r in s["per_arm"].items():
        fb = r["final_block"]
        print(f"  {a:<12s} {r['params']['n_params']:>10,d} {r['headline_entrance']:>9s} "
              f"{fb['val_loss']:>10.5f} {fb['test_loss']:>11.5f} {fb['test_mase']:>11.4f}")
    for k, v in s["final_block_comparisons"].items():
        lo = v["loss"]
        if lo:
            print(f"  final block {k:<24s} test loss {lo['delta_rel']:+.1%} "
                  f"[{lo['delta_rel_ci_lo']:+.1%}, {lo['delta_rel_ci_hi']:+.1%}] {lo['direction']}")


# --------------------------------------------------------------------------- #
# smoke extras
# --------------------------------------------------------------------------- #
def refit_native_check(model, tag, info, qvec, med, args) -> dict:
    """The Phase-1 fitter, unchanged, on the Phase-1 cache, in THIS environment: must give back
    the committed validation curve and entrance (proves the harness, not just the artifacts)."""
    D, native, p1cfg = info["data"], info["native"], info["p1cfg"]
    spec = phase1.model_spec(model)
    common = dict(quantiles=qvec, median_idx=med, device=args.device,
                  epochs=int(p1cfg["probe_epochs"]), lr=float(p1cfg["probe_lr"]),
                  seed=int(p1cfg["seed"]), wd_grid=tuple(p1cfg["wd_grid"]),
                  verbose=not args.quiet_probes)
    sp = D["splits"]
    if model == "tirex":
        import probing.phase1_tirex as tx
        data = {s: {"feats": sp[s]["native_by_label"], "targets": sp[s]["targets"]}
                for s in p2d.SPLITS}
        res = tx.fit_layerwise(tag, data, geom=D["geom"], **common)
    elif model == "chronos2":
        import probing.phase1_chronos2 as c2
        data = {s: {"feats": sp[s]["native_states"], "Y": sp[s]["Y"]} for s in p2d.SPLITS}
        res = c2.fit_layerwise(tag, data, horizon=phase1.PHASE1_H, **common)
    else:
        import probing.phase1_timesfm3 as t3
        data = {s: {**sp[s], "feats": sp[s]["last_token_feats"]} for s in p2d.SPLITS}
        res = t3.fit_layerwise(tag, data, geom=D["geom"], null_wd=p1cfg.get("null_wd"), **common)
    v_new, v_old = np.asarray(res["val_loss"]), np.asarray(native["val_loss"])
    rel = float(np.max(np.abs(v_new / v_old - 1.0)))
    t_new = phase1.tunnel_record(spec, res["val_loss"], res["test_loss"], tol=p2d.HEADLINE_TOL,
                                 tols=p2d.TUNNEL_TOLS)
    ent_new = {k: e["label"] for k, e in t_new["by_tolerance"].items()}
    ent_old = {f"tol_{t:g}": native["entrances"][t]["label"] for t in p2d.TUNNEL_TOLS}
    wd_same = [float(a) == float(b) for a, b in zip(res["wd"], native["wd"])]
    ok = rel <= 1e-4 and ent_new == ent_old
    rec = {"model": model, "dataset": tag, "val_curve_max_rel_diff": rel,
           "test_curve_max_rel_diff": float(np.max(np.abs(
               np.asarray(res["test_loss"]) / np.asarray(native["test_loss"]) - 1.0))),
           "entrances_refit": ent_new, "entrances_committed": ent_old,
           "wd_identical_points": int(sum(wd_same)), "n_points": len(wd_same),
           "status": "pass" if ok else "FAIL", "criterion": "val curve within 1e-4 relative "
                                                         "AND identical entrances at all tols"}
    print(f"  [refit-native {model}/{tag}] {rec['status']}: val curve max rel diff {rel:.2e}, "
          f"wd identical at {rec['wd_identical_points']}/{rec['n_points']} points, entrances "
          f"{'identical' if ent_new == ent_old else f'{ent_new} vs {ent_old}'}")
    return rec


def one_batch_checks(models, tag, args, bb) -> dict:
    out = {}
    for model in models:
        t0 = time.time()
        cell, p1cfg, _root = phase1_source(model, tag, args)
        w = windows_readonly(tag, args.window_cache, args.suite)
        try:
            if model == "chronos2":
                rec = p2d.one_batch_chronos2(tag, w, pipeline=bb.chronos2(),
                                             cache_dir=args.chronos2_cache_dir)
            elif model == "timesfm3":
                mdl, dev = bb.timesfm3(p1cfg["checkpoint"])
                rec = p2d.one_batch_timesfm3(tag, w, p1cfg, phase1_cache_root=args.phase1_cache_root,
                                             model=mdl, device=dev)
            else:
                mdl, geom = bb.tirex(p1cfg["checkpoint"], p1cfg.get("backend", "torch"))
                rec = p2d.one_batch_tirex(tag, w, p1cfg, phase1_cache_root=args.phase1_cache_root,
                                          model=mdl, geom=geom)
        except Exception as exc:
            traceback.print_exc()
            rec = {"model": model, "dataset": tag, "passed": False,
                   "error": f"{type(exc).__name__}: {exc}"}
        rec["seconds"] = time.time() - t0
        out[model] = rec
        print(f"  [one-batch {model}] {'PASS' if rec.get('passed') else 'FAIL'} "
              f"({rec['seconds']:.0f}s)  {json.dumps({k: v for k, v in rec.items() if k in ('tokens', 'tokens_pass0', 'content_mean_vs_cache', 'token15_vs_cache', 'readouts_vs_cache', 'refuses_parent_window')}, default=str)}")
    return out


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #
def print_plan(models, tags, args):
    print(f"\nPHASE 2d PLAN  ({len(models)} models x {len(tags)} datasets = "
          f"{len(models) * len(tags)} cells)   output {args.output_root}")
    print("\nREADOUT CONFIGURATIONS (Q=9, H=64)")
    print(f"  {'model':<10s} {'arm':<12s} {'in':>6s} {'out':>5s} {'params':>11s} {'x native':>9s}"
          "  description")
    for r in p2d.param_table():
        if r["model"] in models:
            print(f"  {r['model']:<10s} {r['arm']:<12s} {r['in_features']:>6d} "
                  f"{r['out_features']:>5d} {r['n_params']:>11,d} {r['ratio_to_native']:>9.2f}  "
                  f"{r['description']}")
    print("\nCELLS (training protocol read from each Phase-1 cell)")
    bad = 0
    for m in models:
        for t in tags:
            try:
                cell, cfg, root = phase1_source(m, t, args)
                ent = cell.entrance(p2d.HEADLINE_TOL)
                spec = phase1.model_spec(m)
                st, _ = phase2.Phase2CellStore(args.output_root, p2d.KIND, m, t,
                                               required=p2d.REQUIRED).status("?")
                print(f"  {m:<9s} {t:<28s} lr={cfg['probe_lr']:<6g} grid={len(cfg['wd_grid']):>2d}"
                      f"  native entrance {spec.depth_labels[ent]:<5s}  arms "
                      f"{','.join(p2d.ARMS_BY_MODEL[m])}  [{Path(root).name}]"
                      f"{'  (exists)' if st != 'pending' else ''}")
            except Exception as exc:
                bad += 1
                print(f"  {m:<9s} {t:<28s} NOT READY: {type(exc).__name__}: {exc}")
    print(f"\n{bad} cell(s) not ready" if bad else "\nall Phase-1 sources COMPLETE")
    for name, val in (("window cache", args.window_cache),
                      ("phase-1 cache root", args.phase1_cache_root),
                      ("pooled cache root (written)", args.pooled_cache_root),
                      ("readout checkpoints (written)", args.artifact_root)):
        print(f"  {name:<30s} {val}")
    return bad


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-root", default=None)
    ap.add_argument("--phase1-root", default=str(phase2.DEFAULT_PHASE1_ROOT))
    ap.add_argument("--phase1-override", nargs="*", default=[], metavar="TAG=ROOT")
    ap.add_argument("--window-cache", default=os.environ.get("PHASE1_WINDOWS")
                    or _scratch("chronos2/phase1_shared_windows"))
    ap.add_argument("--phase1-cache-root", default=_scratch("chronos2/phase1_shared_cache"),
                    help="Phase-1 TimesFM-3 / TiRex feature caches (read-only)")
    ap.add_argument("--chronos2-cache-dir", default=None,
                    help="Phase-1 Chronos-2 K4 caches (default: probing.config.CACHE_DIR)")
    ap.add_argument("--pooled-cache-root", default=_scratch("chronos2/phase2d_pooled_cache"))
    ap.add_argument("--artifact-root", default=_scratch("chronos2/phase2d_pooled_artifacts"))
    ap.add_argument("--no-probe-artifacts", action="store_true")
    ap.add_argument("--models", nargs="*", default=None, choices=phase1.MODELS)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--arms", nargs="*", default=None,
                    choices=(p2d.ARM_POOLED, p2d.ARM_CONCAT, p2d.ARM_NATIVE_MEAN))
    ap.add_argument("--suite", default="paper14", choices=["paper14"])
    ap.add_argument("--device", default=None)
    ap.add_argument("--boot-b", type=int, default=p2d.BOOT_B)
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help=f"{SMOKE_MODEL} x {SMOKE_DATASET} + one-batch checks for all three "
                         "models + native refit check, into the smoke tree")
    ap.add_argument("--one-batch-checks", nargs="*", default=None, choices=phase1.MODELS)
    ap.add_argument("--checks-only", action="store_true")
    ap.add_argument("--check-dataset", default=SMOKE_DATASET)
    ap.add_argument("--refit-native-check", action="store_true")
    ap.add_argument("--force-recompute", action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--quiet-probes", action="store_true")
    args = ap.parse_args(argv)
    if args.smoke:
        args.models = args.models or [SMOKE_MODEL]
        args.datasets = args.datasets or [SMOKE_DATASET]
        args.output_root = args.output_root or str(SMOKE_OUT)
        if args.one_batch_checks is None:
            args.one_batch_checks = list(phase1.MODELS)
        args.refit_native_check = True
    args.output_root = assert_output_root(Path(args.output_root or DEFAULT_OUT))
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    models = list(args.models or phase1.MODELS)
    roster = list(registry.roster("paper14"))
    unknown = [t for t in (args.datasets or []) if t not in roster]
    if unknown:
        raise SystemExit(f"unknown datasets {unknown}; roster: {roster}")
    tags = [t for t in roster if not args.datasets or t in args.datasets]
    if args.plan:
        return 1 if print_plan(models, tags, args) else 0

    qvec, med = phase1.quantile_set("q9")
    args.output_root.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    manifest = {"schema": "phase2d_run_manifest/v1", "protocol": p2d.PROTOCOL,
                "command_line": " ".join(sys.argv), "started": started, "models": models,
                "datasets": tags, "git": phase1.git_state(),
                "environment": phase1.environment_record(args.device),
                "paths": {k: str(getattr(args, k)) for k in
                          ("output_root", "phase1_root", "window_cache", "phase1_cache_root",
                           "chronos2_cache_dir", "pooled_cache_root", "artifact_root")}}
    tag_suffix = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    run_dir = args.output_root / "runs" / tag_suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    phase1.atomic_write_json(run_dir / "run_manifest.json", manifest)
    phase2.clean_staging(args.output_root, p2d.KIND, cells=[(m, t) for m in models for t in tags])
    bb = Backbones(args)

    if args.one_batch_checks:
        print(f"\nONE-BATCH CHECKS on {args.check_dataset}: {args.one_batch_checks}")
        chk = one_batch_checks(args.one_batch_checks, args.check_dataset, args, bb)
        phase1.atomic_write_json(run_dir / "one_batch_checks.json", chk)
        if not all(r.get("passed") for r in chk.values()):
            print("ONE-BATCH CHECKS FAILED -- not running any cell")
            return 1
        if args.checks_only:
            return 0

    state, failures = {}, []
    for m in models:
        arms = tuple(a for a in p2d.ARMS_BY_MODEL[m] if not args.arms or a in args.arms)
        for t in tags:
            try:
                info = run_cell(m, t, args, bb, qvec, med, arms)
                state[f"{m}/{t}"] = {k: v for k, v in info.items()
                                     if k in ("status", "path", "timings")}
                if args.refit_native_check and info["status"] == "complete":
                    rec = refit_native_check(m, t, info, qvec, med, args)
                    phase1.atomic_write_json(run_dir / f"native_refit__{m}__{t}.json", rec)
                    state[f"{m}/{t}"]["native_refit"] = rec["status"]
            except Exception as exc:
                traceback.print_exc()
                failures.append(f"{m}/{t}")
                state[f"{m}/{t}"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            phase1.atomic_write_json(run_dir / "cells.json", state)
    print(f"\n{'=' * 86}\nPHASE 2d: {sum(v['status'] == 'complete' for v in state.values())} built"
          f", {sum(v['status'] == 'skipped' for v in state.values())} skipped, "
          f"{len(failures)} failed {failures if failures else ''}\n  run record: {run_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
