"""H3 — dataset-specific native-pathway compatibility, every block depth, all 14 datasets.

    Once forecasting is recoverable from an intermediate representation, can a lightweight
    transformation make that representation usable by the model's ORIGINAL frozen forecasting
    pathway, bringing performance close to the full native model?

Per model x dataset x depth: the hard cut, the label-free native-output alignment (primary), the
supervised forecast-loss alignment (Chronos-2, TiRex), a hidden-space ridge diagnostic, and the
REUSED Phase-1 probe -- all on the Phase-1 windows, rows and depth axis. Train fits, validation
selects, test is read once.

    # SMOKE (one completed dataset, 4 depths incl. the Phase-1 entrance, the real 300 epochs):
    python -m experiments.run_phase2_h3 --smoke --models chronos2 --device cuda

    # FULL (resumable; re-submitting the identical line skips COMPLETE cells):
    python -m experiments.run_phase2_h3 --device cuda

WHERE TO RUN. A COMPUTE NODE with a GPU (iterative Chronos-2 / TiRex fits). TimesFM-3's alignment
is closed-form and runs on CPU, but its native head is loaded from the checkpoint either way.
``--plan`` (no data, no model) is the only login-node-safe mode.

PHASE 1 IS READ-ONLY. Windows come from the Phase-1 window cache, features from the Phase-1
feature caches; a cache miss is an ERROR (Phase 2 never extracts and never writes a Phase-1 file).
Every cell records the Phase-1 cell's fit hash and sha256s, and refuses to run on a Phase-1 cell
that is not COMPLETE.
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

from probing import phase1, phase2, registry                                      # noqa: E402
from probing.phase2 import (DEFAULT_OUT_ROOT, DEFAULT_PHASE1_ROOT, H3_FAMILIES,     # noqa: E402
                            PHASE2_PROTOCOL_VERSION, Phase1Cell, Phase2CellStore,
                            clean_staging, parse_overrides, resolve_phase1_root)

KIND = "h3"


# --------------------------------------------------------------------------- #
# read-only inputs
# --------------------------------------------------------------------------- #
def windows_readonly(tag: str, args) -> dict:
    """Load the Phase-1 window cache. Rebuild only with --allow-window-rebuild (never written)."""
    path = Path(args.window_cache) / f"windows__{args.suite}__{tag}.npz"
    if path.exists():
        with np.load(path, allow_pickle=True) as z:
            w = {k: z[k] for k in z.files if k != "meta"}
            w["meta"] = json.loads(str(z["meta"]))
        return w
    if not args.allow_window_rebuild:
        raise phase2.Phase1DependencyError(
            f"no Phase-1 window cache at {path}. Point --window-cache at the Phase-1 "
            "$SCRATCH/chronos2/phase1_shared_windows, or pass --allow-window-rebuild (slow; the "
            "rebuilt windows are digest-checked against the Phase-1 cell and never written).")
    from probing.windows import build_for
    return build_for(tag, args.suite, C=phase1.PHASE1_C, H=phase1.PHASE1_H, seed=args.seed)


class Backbones:
    """Each checkpoint is loaded at most once per job; only its head/norm modules are used."""

    def __init__(self, args):
        self.args, self._h = args, {}

    def pathway(self, model: str, p1cfg: dict):
        from probing.phase2_pathways import Chronos2Pathway, TiRexPathway, TimesFM3Pathway
        if model not in self._h:
            t0 = time.time()
            if model == "chronos2":
                from probing.extraction import get_pipeline
                pipe, _ = get_pipeline()
                self._h[model] = {"pathway": Chronos2Pathway.from_pipeline(pipe)}
            elif model == "timesfm3":
                from probing.timesfm3_last_token import get_model
                mdl, _dev = get_model(p1cfg["checkpoint"], self.args.device)
                self._h[model] = {"pathway": TimesFM3Pathway.from_model(mdl)}
            else:
                from probing.tirex_model import geometry_from_model, get_model
                mdl = get_model(p1cfg["checkpoint"], self.args.device,
                                backend=p1cfg.get("backend", "torch"))
                geom = geometry_from_model(mdl, C=phase1.PHASE1_C, H=phase1.PHASE1_H)
                self._h[model] = {"pathway": TiRexPathway.from_model(mdl, geom), "model": mdl,
                                  "geom": geom}
            print(f"  [model] {model} pathway ready in {time.time() - t0:.1f}s")
        return self._h[model]


def load_cell_data(model, tag, w, arrays, p1cfg, handle, args):
    from probing.phase2_pathways import load_chronos2, load_timesfm3, load_tirex
    if model == "chronos2":
        return load_chronos2(tag, w, arrays, cache_dir=args.chronos_features_dir)
    if model == "timesfm3":
        return load_timesfm3(tag, w, arrays, p1cfg, cache_dir=args.cache_root)
    return load_tirex(tag, w, arrays, p1cfg, cache_dir=args.cache_root, model=handle["model"],
                      geom=handle["geom"])


# --------------------------------------------------------------------------- #
# identity
# --------------------------------------------------------------------------- #
def cell_config(model, tag, dep, families, depths, fit_cfg, args) -> dict:
    """FIT half = everything that would change an adapter or a per-window metric; POSTPROCESS half
    = tunnel tolerances / budget / bootstrap (re-derivable from saved per-window arrays)."""
    return {"protocol": PHASE2_PROTOCOL_VERSION, "kind": KIND, "model": model, "dataset": tag,
            "phase1_fit_hash": dep["fit_hash"], "phase1_config_hash": dep["config_hash"],
            "window_digest": dep["window_digest"], "families": list(families),
            "depths": None if depths is None else [int(d) for d in depths],
            "epochs": fit_cfg["epochs"], "lr": fit_cfg["lr"], "wd_grid": fit_cfg["wd_grid"],
            "eval_every": fit_cfg["eval_every"], "kappas": fit_cfg["kappas"],
            "seed": fit_cfg["seed"], "native_gate_rtol": fit_cfg["native_gate_rtol"],
            "quantiles": [float(x) for x in phase2.QUANTILES],
            # postprocess
            "tunnel_tols": [float(t) for t in phase2.TUNNEL_TOLS],
            "headline_tol": phase2.HEADLINE_TOL, "budget": phase2.BUDGET,
            "frontier_budgets": [float(e) for e in phase2.FRONTIER_BUDGETS],
            "frontier_rule": phase2.BUDGET_RULE["version"],
            "boot_b": fit_cfg["boot_b"], "boot_seed": fit_cfg["boot_seed"]}


def smoke_depths(entrance: int, L: int) -> list[int]:
    """Emb, the Phase-1 entrance, one depth between it and the end, and the final block."""
    return sorted({0, int(entrance), int((entrance + L) // 2), int(L)})


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    import torch
    from probing.phase2_env import environment_record, set_precision_flags
    from probing.phase2_h3 import DEFAULT_FIT_CFG, run_h3_cell, save_h3_cell
    numerics = set_precision_flags(deterministic=True)
    out = Path(args.output_root)
    (out / KIND).mkdir(parents=True, exist_ok=True)
    overrides = parse_overrides(args.phase1_override)
    tags = [t for t in registry.roster("paper14") if not args.datasets or t in args.datasets]
    if args.smoke and not args.datasets:
        tags = [phase1.FIRST_DATASET]
    models = [m for m in phase1.MODELS if not args.models or m in args.models]
    fit_cfg = {**DEFAULT_FIT_CFG, "epochs": args.epochs, "lr": args.lr,
               "wd_grid": list(phase2.assert_adapter_wd_grid(args.wd_grid, args.lr)),
               "eval_every": args.eval_every, "seed": args.seed,
               "native_gate_rtol": args.native_gate_rtol, "boot_b": args.boot_b,
               "boot_seed": args.seed}

    if args.plan:
        print(f"\nH3 PLAN  {len(tags)} datasets x {len(models)} models -> {out / KIND}")
        for tag in tags:
            for m in models:
                root = resolve_phase1_root(tag, args.phase1_root, overrides)
                c = Phase1Cell(root, m, tag)
                try:
                    c.marker()
                    st = f"phase1 COMPLETE, entrance {c.entrances()[0.05]['label']}"
                except Exception as e:
                    st = f"phase1 NOT READY: {type(e).__name__}"
                print(f"  {m:<9} {tag:<28} {st}   [{root}]")
        return 0

    # ONLY this job's cells: other jobs (other models, or other dataset halves) may be building
    # their own cells in the same output root right now.
    removed = clean_staging(out, KIND, cells=[(m, t) for m in models for t in tags])
    if removed:
        print(f"[resume] removed stale staging dirs: {removed[:5]}")
    manifest = {"schema": "phase2_h3_run/v1", "protocol": PHASE2_PROTOCOL_VERSION,
                "question": "H3: functional replaceability of intermediate representations",
                "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds"),
                "phase1_root": str(args.phase1_root), "phase1_overrides":
                    {k: str(v) for k, v in overrides.items()},
                "cache_root": str(args.cache_root), "window_cache": str(args.window_cache),
                "adapter_root": str(args.adapter_root), "smoke": bool(args.smoke),
                "models": models, "datasets": tags, "families": list(args.families),
                "fit_cfg": fit_cfg, "numerics": numerics,
                "environment": environment_record()}
    # One record directory PER JOB: jobs sharing this output root never overwrite each other's
    # manifest or cell log. The cells themselves are the durable record (COMPLETE markers).
    job = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    run_dir = out / KIND / "runs" / job
    manifest["job"] = job
    phase2.atomic_write_json(run_dir / "run_manifest.json", manifest)
    state_path = run_dir / "cells.json"
    state = {"cells": {}}
    bb = Backbones(args)
    failures = []

    for tag in tags:
        print(f"\n{'#' * 86}\n# {tag}  ({registry.display_name(tag)})\n{'#' * 86}")
        root = resolve_phase1_root(tag, args.phase1_root, overrides)
        try:
            w = windows_readonly(tag, args)
            from probing.window_reference import window_digest
            digest = window_digest(w)
        except Exception as exc:
            print(f"[FAIL] {tag}: windows: {type(exc).__name__}: {exc}")
            failures += [(m, tag, str(exc)) for m in models]
            continue
        for model in models:
            key = f"{model}/{tag}"
            try:
                p1 = Phase1Cell(root, model, tag)
                dep = p1.dependency_record()
                if dep["window_digest"] != digest:
                    raise phase2.Phase1DependencyError(
                        f"{key}: window digest {digest} != Phase-1 cell's {dep['window_digest']}")
                ent = p1.entrances()
                L = phase1.model_spec(model).reference_index
                depths = (smoke_depths(ent[0.05]["depth_axis_index"], L) if args.smoke
                          else (None if not args.depths else [int(d) for d in args.depths]))
                ccfg = cell_config(model, tag, dep, args.families, depths, fit_cfg, args)
                chash = phase1._digest(ccfg)
                store = Phase2CellStore(out, KIND, model, tag)
                status, reason = store.status(chash)
                if status == "complete" and args.resume:
                    print(f"[skip] {key}: COMPLETE ({reason})")
                    state["cells"][key] = {"status": "complete", "config_hash": chash}
                    continue
                if status == "incompatible" and not args.force_recompute:
                    raise RuntimeError(f"{key} exists under a different configuration "
                                       f"({reason}); pass --force-recompute or a new "
                                       "--output-root")
                print(f"\n[{key}] Phase-1 entrance {ent[0.05]['label']}  "
                      f"(fit hash {dep['fit_hash']})")
                handle = bb.pathway(model, p1.config())
                arrays = p1.arrays()
                p1_preds = {"val": p1.predictions("val"), "test": p1.predictions("test")}
                t0 = time.time()
                data = load_cell_data(model, tag, w, arrays, p1.config(), handle, args)
                print(f"  [data] loaded in {time.time() - t0:.1f}s; rows train/val/test = "
                      f"{[len(data.splits[s].target) for s in ('train', 'val', 'test')]}")
                res = run_h3_cell(
                    handle["pathway"], data, arrays, entrances=ent, families=args.families,
                    depths=depths, device=args.device or None, fit_cfg=fit_cfg,
                    adapter_dir=Path(args.adapter_root),
                    floor_val_loss=p1.floor_val_loss(),
                    save_prediction_depths=[ent[t]["depth_axis_index"] for t in ent],
                    phase1_predictions=p1_preds, log=print)
                stage = store.begin()
                try:
                    save_h3_cell(stage, res, ccfg, chash, dep, data.provenance,
                                 extra={"smoke": bool(args.smoke)})
                    final = store.commit(stage, chash, extra={
                        "phase1_entrance": ent[0.05]["label"], "smoke": bool(args.smoke)})
                except Exception:
                    store.abandon(stage)
                    raise
                h = res["ladder"]["headline"]["rungs"]
                print(f"  [ladder @ {ent[0.05]['label']}] " + "  ".join(
                    f"{f}: MASE x{v['mase']['ratio']:.3f}" for f, v in h.items()))
                print(f"  COMPLETE {final}")
                state["cells"][key] = {"status": "complete", "config_hash": chash,
                                       "timings": res["timings"]}
                del data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as exc:
                print(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
                traceback.print_exc()
                failures.append((model, tag, f"{type(exc).__name__}: {exc}"))
                state["cells"][key] = {"status": "failed", "reason": str(exc)[:500]}
            phase2.atomic_write_json(state_path, state)

    print(f"\n{'=' * 86}\nH3 SUMMARY  "
          f"{sum(v['status'] == 'complete' for v in state['cells'].values())} complete, "
          f"{len(failures)} failed this run")
    for m, t, why in failures:
        print(f"  {m}/{t}: {why}")
    return 1 if failures and args.fail_on_error else 0


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    scratch = Path(os.environ.get("SCRATCH", "/tmp"))
    project = os.environ.get("PROJECT")
    g = p.add_argument_group("inputs (READ-ONLY Phase 1)")
    g.add_argument("--phase1-root", default=str(DEFAULT_PHASE1_ROOT))
    g.add_argument("--phase1-override", action="append", default=[],
                   help="TAG=ROOT, e.g. LOOP_SEATTLE_5T=results/three_model_phase1_q9_loop_lowlr")
    g.add_argument("--cache-root", default=os.environ.get(
        "PHASE1_CACHE", str(scratch / "chronos2" / "phase1_shared_cache")))
    g.add_argument("--window-cache", default=os.environ.get(
        "PHASE1_WINDOWS", str(scratch / "chronos2" / "phase1_shared_windows")))
    g.add_argument("--chronos-features-dir", default=None,
                   help="Chronos-2 K-slot caches (default: probing.config.CACHE_DIR = "
                        "features_cache/, the Phase-1 location)")
    g.add_argument("--allow-window-rebuild", action="store_true")
    g.add_argument("--suite", default="paper14")
    g = p.add_argument_group("outputs")
    g.add_argument("--output-root", default=None,
                   help="default results/three_model_phase2 (results/three_model_phase2_smoke "
                        "under --smoke, so a smoke cell can never be mistaken for a real one)")
    g.add_argument("--adapter-root", default=None,
                   help="adapter weights (large): default $PROJECT/chronos2_phase2/adapters, "
                        "else $SCRATCH/chronos2_phase2/adapters")
    g = p.add_argument_group("what to run")
    g.add_argument("--models", nargs="+", choices=list(phase1.MODELS), default=None)
    g.add_argument("--datasets", nargs="+", default=None)
    g.add_argument("--families", nargs="+", default=list(H3_FAMILIES), choices=list(H3_FAMILIES))
    g.add_argument("--depths", nargs="+", type=int, default=None,
                   help="depth-axis indices (default: every block depth)")
    g.add_argument("--smoke", action="store_true",
                   help="Electricity, depths {Emb, entrance, midpoint, final}, B=500, separate tree")
    g.add_argument("--plan", action="store_true")
    g.add_argument("--resume", action="store_true", default=True)
    g.add_argument("--no-resume", dest="resume", action="store_false")
    g.add_argument("--force-recompute", action="store_true")
    g.add_argument("--fail-on-error", action="store_true")
    g = p.add_argument_group("protocol")
    g.add_argument("--epochs", type=int, default=phase2.ADAPTER_EPOCHS)
    g.add_argument("--lr", type=float, default=phase2.ADAPTER_LR)
    g.add_argument("--wd-grid", type=float, nargs="+", default=list(phase2.ADAPTER_WD_GRID))
    g.add_argument("--eval-every", type=int, default=phase2.ADAPTER_EVAL_EVERY)
    g.add_argument("--boot-b", type=int, default=None)
    g.add_argument("--seed", type=int, default=phase2.SEED)
    g.add_argument("--native-gate-rtol", type=float, default=1e-4)
    g.add_argument("--device", default=os.environ.get("PHASE2_DEVICE", None))
    a = p.parse_args(argv)
    if a.output_root is None:
        a.output_root = str(DEFAULT_OUT_ROOT) + ("_smoke" if a.smoke else "")
    if a.adapter_root is None:
        base = Path(project) if project else scratch
        a.adapter_root = str(base / "chronos2_phase2" / ("adapters_smoke" if a.smoke
                                                         else "adapters"))
    if a.boot_b is None:
        a.boot_b = 500 if a.smoke else phase2.BOOT_B
    return a


if __name__ == "__main__":
    raise SystemExit(main())
