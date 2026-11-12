"""Phase 2c -- pooled leave-one-dataset-out (LODO) adapter transfer, Chronos-2, L4 / L6 / L9.

    Can ONE affine adapter, fitted on the other datasets, align Chronos-2's intermediate states
    well enough to truncate on an UNSEEN dataset with no target-domain calibration?

Per (variant, target, depth): the H3 label-free adapter fit, unchanged, on the pooled sources
(equal-weighted per source), then ONE read of the target's test split: full model | hard cut |
pooled transfer | the committed dataset-specific H3 adapter. See ``probing.phase2_lodo``.

    # PLAN (login node: JSON reads only, no model, no arrays)
    python -m experiments.run_phase2_lodo --plan

    # SMOKE (compute node, GPU): Electricity hourly x L9, main protocol, the REAL 1500 epochs
    python -m experiments.run_phase2_lodo --smoke --device cuda

    # FULL (resumable; re-submitting the identical line skips COMPLETE cells)
    python -m experiments.run_phase2_lodo --depths 4 --chunk 1/2 --device cuda

WHERE TO RUN. A COMPUTE NODE with a GPU for everything except ``--plan``. Every job loads the
forecast-slot caches of all 14 datasets (each one is a source in 13 folds), ~5 GB of reads.

READ-ONLY INPUTS. Phase-1 windows/caches/cells and the committed H3 cells + adapters are never
written. Outputs: ``results/three_model_phase2_lodo/`` and the pooled adapters under
``$SCRATCH/chronos2_phase2/adapters_lodo``.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2, registry                                      # noqa: E402
from probing import phase2_lodo as lodo                                           # noqa: E402
from probing.phase2 import (DEFAULT_OUT_ROOT, DEFAULT_PHASE1_ROOT, Phase1Cell,      # noqa: E402
                            Phase2CellStore, clean_staging, parse_overrides,
                            resolve_phase1_root)

DEFAULT_LODO_ROOT = REPO_ROOT / "results" / "three_model_phase2_lodo"
SMOKE_TARGET = "monash_electricity_hourly"     # exercises the sibling-exclusion path (12 sources)
SMOKE_DEPTH = 9


def _store(out, cell) -> Phase2CellStore:
    return Phase2CellStore(out, f"{lodo.KIND}/{cell['variant']}", cell["target"],
                           lodo.depth_label(cell["depth"]), required=lodo.LODO_REQUIRED)


def select_cells(args, roster) -> list[dict]:
    if args.smoke:
        return lodo.plan_cells(roster, depths=[SMOKE_DEPTH], targets=[SMOKE_TARGET],
                               variants=[lodo.MAIN])
    cells = []
    for v in args.variants:
        tg = [t for t in roster if (not args.targets or t in args.targets)
              and (v == lodo.MAIN or t in lodo.SERIES_SIBLINGS)]
        tg = lodo.chunk(tg, args.chunk)
        cells += lodo.plan_cells(roster, depths=args.depths, targets=tg, variants=[v])
    return cells


def main(argv=None) -> int:
    args = parse_args(argv)
    roster = list(registry.roster("paper14"))
    cells = select_cells(args, roster)
    out = Path(args.output_root)
    overrides = parse_overrides(args.phase1_override)
    fit_cfg = {**_default_fit_cfg(), "epochs": args.epochs, "lr": args.lr,
               "wd_grid": list(phase2.assert_adapter_wd_grid(args.wd_grid, args.lr)),
               "eval_every": args.eval_every, "seed": args.seed,
               "native_gate_rtol": args.native_gate_rtol, "boot_b": args.boot_b,
               "boot_seed": args.seed}
    targets = sorted({c["target"] for c in cells}, key=roster.index)
    needed = sorted({c["target"] for c in cells} | {s for c in cells for s in c["sources"]},
                    key=roster.index)
    depths = sorted({c["depth"] for c in cells})

    # ------------------------------------------------------------ dependencies (JSON only)
    deps, p1cells, h3refs = {}, {}, {}
    for t in needed:
        p1 = Phase1Cell(resolve_phase1_root(t, args.phase1_root, overrides), lodo.MODEL, t)
        p1cells[t] = p1
        deps[t] = p1.dependency_record()
    for t in targets:
        h3refs[t] = lodo.load_h3_reference(args.h3_root, t)

    status = {}
    for c in cells:
        ccfg = lodo.cell_config(c, deps, h3refs[c["target"]], fit_cfg)
        c["config"], c["hash"] = ccfg, phase1._digest(ccfg)
        status[_key(c)] = _store(out, c).status(c["hash"])

    print(f"\nPHASE 2c LODO  {len(cells)} cells -> {out / lodo.KIND}")
    for c in cells:
        st, why = status[_key(c)]
        print(f"  {c['variant']:<17} {c['target']:<28} {lodo.depth_label(c['depth']):<4} "
              f"{len(c['sources']):>2} sources  [{st}]")
    if args.plan:
        return 0
    pending = [c for c in cells if not (status[_key(c)][0] == "complete" and args.resume)]
    bad = [c for c in pending if status[_key(c)][0] == "incompatible"]
    if bad and not args.force_recompute:
        raise RuntimeError(f"{len(bad)} cell(s) exist under a different configuration, e.g. "
                           f"{_key(bad[0])}: {status[_key(bad[0])][1]}. Pass --force-recompute "
                           "or a new --output-root.")
    if not pending:
        print("\nall requested cells are COMPLETE; nothing to do")
        return 0

    # ------------------------------------------------------------ heavy part (compute node)
    import torch
    from experiments.run_phase2_h3 import windows_readonly
    from probing.extraction import get_pipeline
    from probing.phase2_align import load_adapter, save_adapter
    from probing.phase2_env import environment_record, set_precision_flags
    from probing.phase2_pathways import Chronos2Pathway, load_chronos2
    from probing.window_reference import window_digest

    numerics = set_precision_flags(deterministic=True)
    removed = clean_staging(out, f"{lodo.KIND}/{lodo.MAIN}",
                            cells=[(c["target"], lodo.depth_label(c["depth"]))
                                   for c in pending if c["variant"] == lodo.MAIN])
    removed += clean_staging(out, f"{lodo.KIND}/{lodo.SENSITIVITY}",
                             cells=[(c["target"], lodo.depth_label(c["depth"]))
                                    for c in pending if c["variant"] == lodo.SENSITIVITY])
    if removed:
        print(f"[resume] removed stale staging dirs: {removed[:5]}")
    job = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    run_dir = out / lodo.KIND / "runs" / job
    env = environment_record()
    phase2.atomic_write_json(run_dir / "run_manifest.json", {
        "schema": "phase2c_lodo_run/v1", "protocol": lodo.LODO_PROTOCOL_VERSION, "job": job,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "cells": [_key(c) for c in pending], "datasets_loaded": needed, "depths": depths,
        "fit_cfg": fit_cfg, "numerics": numerics, "phase1_root": str(args.phase1_root),
        "h3_root": str(args.h3_root), "h3_adapter_root": str(args.h3_adapter_root),
        "adapter_root": str(args.adapter_root), "window_cache": str(args.window_cache),
        "smoke": bool(args.smoke), "environment": env})

    t0 = time.time()
    pipe, _ = get_pipeline()
    pathway = Chronos2Pathway.from_pipeline(pipe)
    device = args.device or pathway.device
    print(f"  [model] Chronos-2 pathway ready on {pathway.device} in {time.time() - t0:.1f}s")

    wargs = SimpleNamespace(window_cache=args.window_cache, suite=args.suite,
                            allow_window_rebuild=False, seed=args.seed)
    prepared, p1arrays = {}, {}
    for t in needed:
        t1 = time.time()
        w = windows_readonly(t, wargs)
        digest = window_digest(w)
        if digest != deps[t]["window_digest"]:
            raise phase2.Phase1DependencyError(
                f"{t}: window digest {digest} != Phase-1 cell's {deps[t]['window_digest']}")
        arrays = p1cells[t].arrays()
        data = load_chronos2(t, w, arrays, cache_dir=args.chronos_features_dir)
        prepared[t] = lodo.prepare_dataset(pathway, data, depths)
        if t in targets:
            p1arrays[t] = arrays
        n = [len(data.splits[s].target) for s in ("train", "val", "test")]
        print(f"  [data] {t:<28} train/val/test {n}  ({time.time() - t1:.1f}s)")
        del data, w

    state = {"cells": {}}
    failures = []
    for c in pending:
        key = _key(c)
        lab = lodo.depth_label(c["depth"])
        print(f"\n{'#' * 86}\n# {key}  ({len(c['sources'])} sources; excluded "
              f"{c['excluded']})\n{'#' * 86}")
        try:
            ref = h3refs[c["target"]]
            sp = Path(args.h3_adapter_root) / lodo.MODEL / c["target"] / f"noa__{lab}.npz"
            spec_ad, spec_meta = load_adapter(sp, device)
            want = ref["fits"]["noa"][lab].get("adapter_sha256")
            if spec_meta.get("sha256") != want:
                raise RuntimeError(f"{sp}: sha256 {spec_meta.get('sha256')} != the H3 cell's "
                                   f"recorded {want}")
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            res = lodo.run_lodo_cell(pathway, prepared, c, phase1_arrays=p1arrays[c["target"]],
                                     h3_ref=ref, specific_adapter=spec_ad, fit_cfg=fit_cfg,
                                     device=device, log=print)
            arec = save_adapter(
                Path(args.adapter_root) / c["variant"] / c["target"] / f"noa__{lab}.npz",
                res["adapter"], {"model": lodo.MODEL, "family": "noa_lodo",
                                 "variant": c["variant"], "target": c["target"],
                                 "depth_index": c["depth"], "label": lab,
                                 "sources": c["sources"],
                                 "selected_wd": res["fit"]["fit"]["selected_wd"],
                                 "selected_epoch": res["fit"]["fit"]["selected_epoch"],
                                 "config_hash": c["hash"]})
            store = _store(out, c)
            stage = store.begin()
            try:
                lodo.save_lodo_cell(stage, res, c["config"], c["hash"], arec,
                                    {"path": str(sp), "sha256": spec_meta.get("sha256")},
                                    environment={"job": job})
                final = store.commit(stage, c["hash"], extra={
                    "protocol": lodo.LODO_PROTOCOL_VERSION, "smoke": bool(args.smoke)})
            except Exception:
                store.abandon(stage)
                raise
            s = res["stats"]
            dm = {a: 100 * s["vs_native"][a]["mase"]["degradation"]
                  for a in ("hard", "pooled", "specific")}
            f = res["fit"]["fit"]
            print(f"  [{key}] dMASE  hard {dm['hard']:+.1f}%  pooled {dm['pooled']:+.1f}%  "
                  f"specific {dm['specific']:+.1f}%   (wd {f['selected_wd']}, epoch "
                  f"{f['selected_epoch']}, flags {res['fit']['flags']})")
            print(f"  [{key}] timings {res['timings']}")
            print(f"  COMPLETE {final}")
            state["cells"][key] = {"status": "complete", "config_hash": c["hash"],
                                   "timings": res["timings"]}
            del res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            print(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append((key, f"{type(exc).__name__}: {exc}"))
            state["cells"][key] = {"status": "failed", "reason": str(exc)[:500]}
        phase2.atomic_write_json(run_dir / "cells.json", state)

    print(f"\n{'=' * 86}\nLODO SUMMARY  "
          f"{sum(v['status'] == 'complete' for v in state['cells'].values())} complete, "
          f"{len(failures)} failed this run")
    for k, why in failures:
        print(f"  {k}: {why}")
    return 1 if failures and args.fail_on_error else 0


def _key(c) -> str:
    return f"{c['variant']}/{c['target']}/{lodo.depth_label(c['depth'])}"


def _default_fit_cfg() -> dict:
    from probing.phase2_h3 import DEFAULT_FIT_CFG
    return dict(DEFAULT_FIT_CFG)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    scratch = Path(os.environ.get("SCRATCH", "/tmp"))
    g = p.add_argument_group("inputs (READ-ONLY)")
    g.add_argument("--phase1-root", default=str(DEFAULT_PHASE1_ROOT))
    g.add_argument("--phase1-override", action="append", default=[],
                   help="TAG=ROOT (LOOP Seattle is rerouted automatically, as in H3)")
    g.add_argument("--window-cache", default=os.environ.get(
        "PHASE1_WINDOWS", str(scratch / "chronos2" / "phase1_shared_windows")))
    g.add_argument("--chronos-features-dir", default=None,
                   help="Chronos-2 K-slot caches (default: features_cache/, as in H3)")
    g.add_argument("--suite", default="paper14")
    g.add_argument("--h3-root", default=str(DEFAULT_OUT_ROOT),
                   help="the committed dataset-specific H3 tree (results/three_model_phase2)")
    g.add_argument("--h3-adapter-root", default=str(scratch / "chronos2_phase2" / "adapters"),
                   help="the committed H3 adapter weights (<root>/chronos2/<tag>/noa__L*.npz)")
    g = p.add_argument_group("outputs")
    g.add_argument("--output-root", default=None,
                   help="default results/three_model_phase2_lodo (_smoke under --smoke)")
    g.add_argument("--adapter-root", default=None,
                   help="pooled adapters: default $SCRATCH/chronos2_phase2/adapters_lodo")
    g = p.add_argument_group("what to run")
    g.add_argument("--depths", nargs="+", type=int, default=list(lodo.LODO_DEPTHS))
    g.add_argument("--targets", nargs="+", default=None)
    g.add_argument("--variants", nargs="+", default=list(lodo.VARIANTS),
                   choices=list(lodo.VARIANTS))
    g.add_argument("--chunk", default=None, help="i/n: the i-th of n contiguous target chunks "
                                                  "(roster order, per variant)")
    g.add_argument("--smoke", action="store_true",
                   help=f"{SMOKE_TARGET} x L{SMOKE_DEPTH}, main protocol, real epochs, B=500, "
                        "separate output tree")
    g.add_argument("--plan", action="store_true")
    g.add_argument("--resume", action="store_true", default=True)
    g.add_argument("--no-resume", dest="resume", action="store_false")
    g.add_argument("--force-recompute", action="store_true")
    g.add_argument("--fail-on-error", action="store_true")
    g = p.add_argument_group("protocol (defaults = the frozen H3 optimizer)")
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
        a.output_root = str(DEFAULT_LODO_ROOT) + ("_smoke" if a.smoke else "")
    if a.adapter_root is None:
        a.adapter_root = str(scratch / "chronos2_phase2" /
                             ("adapters_lodo_smoke" if a.smoke else "adapters_lodo"))
    if a.boot_b is None:
        a.boot_b = 500 if a.smoke else phase2.BOOT_B
    return a


if __name__ == "__main__":
    raise SystemExit(main())
