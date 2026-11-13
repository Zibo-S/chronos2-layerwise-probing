"""Phase 2f -- source-dataset scaling of zero-shot adapter transfer (Chronos-2, L6), ONE job.

    Does the label-free adapter, fitted on 1 / 4 / 8 / 13 OTHER datasets, transfer better to an
    unseen target as the source pool grows? See ``probing.phase2f_source_scaling``.

    # PLAN (login node: JSON reads only, no model, no arrays)
    python -m experiments.run_phase2f_source_scaling --plan [--check-inputs]

    # FULL (compute node, GPU; resumable: re-submitting the identical line skips COMPLETE cells)
    python -m experiments.run_phase2f_source_scaling --device cuda --fail-on-error

ORDER OF WORK.
  1. Load the 14 datasets once (every dataset is a source for some target): window digests checked
     against the Phase-1 cells, forecast-slot states, the full model's outputs as teacher targets.
  2. PREFLIGHT, before any fit: the Phase 2c reproduction gates for each target at L6 (full model
     vs Phase 1 and H3, hard cut and dataset-specific adapter vs H3, test rows, identity adapter)
     plus the dataset-specific adapter's sha256 (``lodo.preflight_gates``). Any failure: exit 3,
     ``runs/<job>/preflight.json``, no fit started.
  3. The 40 cells in order of N. The N = 1 cells run first and each is a FIT-REPRODUCTION gate:
     with one source the pooled fit IS that source's dataset-specific H3 fit, so it must reproduce
     the committed H3 fit (``ss.fit_reproduction``). A failure stops the job (exit 4,
     ``runs/<job>/fit_reproduction_failure.json``) before the costly N = 4 / 8 / 13 fits. Every
     cell re-runs all Phase 2c gates inside ``lodo.run_lodo_cell`` (authoritative).

READ-ONLY INPUTS: Phase-1 windows / caches / cells, the committed H3 cells and adapters. Outputs:
``results/three_model_phase2f_source_scaling/`` and ``$SCRATCH/chronos2/phase2f_source_scaling/``.
Nothing of Phase 2c / 2d / 2e is read or written (``ss.assert_isolated`` refuses overlapping roots).
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
from probing import phase2f_source_scaling as ss                                  # noqa: E402
from probing.phase2 import (DEFAULT_OUT_ROOT, DEFAULT_PHASE1_ROOT, Phase1Cell,      # noqa: E402
                            Phase2CellStore, clean_staging, parse_overrides,
                            resolve_phase1_root)

DEFAULT_ROOT = REPO_ROOT / "results" / "three_model_phase2f_source_scaling"


def _store(out, cell) -> Phase2CellStore:
    return Phase2CellStore(out, ss.KIND, cell["target"], cell["cell_id"], required=ss.REQUIRED)


def _key(c) -> str:
    return f"{c['target']}/{c['cell_id']}"


def _run_order(cells, roster):
    """N = 1 first (the fit-reproduction gates, ~1 min each), the 13-source fits last."""
    return sorted(cells, key=lambda c: (c["n_sources"], roster.index(c["target"]),
                                        -1 if c["order_seed"] is None else c["order_seed"]))


def _fit_cfg(args) -> dict:
    from probing.phase2_h3 import DEFAULT_FIT_CFG
    return {**DEFAULT_FIT_CFG, "epochs": args.epochs, "lr": args.lr,
            "wd_grid": list(phase2.assert_adapter_wd_grid(args.wd_grid, args.lr)),
            "eval_every": args.eval_every, "seed": args.seed,
            "native_gate_rtol": args.native_gate_rtol, "boot_b": args.boot_b,
            "boot_seed": args.seed}


# --------------------------------------------------------------------------- #
# the cell loop (the model-free contracts drive it with a mock pathway)
# --------------------------------------------------------------------------- #
def run_cells(pathway, prepared, cells, *, out, adapter_root, h3refs, p1arrays, specific,
              fit_cfg, roster, device=None, job="local", run_dir=None, fit_repro=True,
              resume=True, log=print) -> dict:
    """Run every not-yet-COMPLETE cell. Returns {"rc", "cells", "failures", "fit_reproduction"}.

    ``specific`` = {target: (dataset-specific H3 adapter, its file record)}. ``h3refs`` must hold
    the targets and every N = 1 source. ``c["config"]`` / ``c["hash"]`` must be set."""
    import torch
    from probing.phase2_align import adapter_arrays_sha256, save_adapter
    out, adapter_root = Path(out), Path(adapter_root)
    run_dir = Path(run_dir) if run_dir is not None else out / ss.KIND / "runs" / job
    order = _run_order(cells, roster)
    todo = [c for c in order if not (resume and _store(out, c).status(c["hash"])[0] == "complete")]
    todo_keys = {_key(c) for c in todo}
    units_total = sum(c["n_sources"] for c in todo)
    units_done, t_start = 0, time.time()
    state = {"rc": 0, "cells": {}, "failures": [], "fit_reproduction": []}
    for c in order:
        if _key(c) not in todo_keys:
            state["cells"][_key(c)] = {"status": "complete", "skipped": "already COMPLETE"}
    for k, c in enumerate(todo, 1):
        key, t, lab = _key(c), c["target"], lodo.depth_label(c["depth"])
        log(f"\n{'#' * 86}\n# [{k}/{len(todo)}] {key}  N={c['n_sources']}  sources "
            f"{c['selection_order']}\n{'#' * 86}")
        try:
            spec_ad, spec_rec = specific[t]
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            res = lodo.run_lodo_cell(pathway, prepared, c, phase1_arrays=p1arrays[t],
                                     h3_ref=h3refs[t], specific_adapter=spec_ad, fit_cfg=fit_cfg,
                                     device=device, log=log)
            ad = res["adapter"]
            sha = adapter_arrays_sha256(ad)
            rep = None
            if c["n_sources"] == 1 and fit_repro:
                s = c["sources"][0]
                te_s = prepared[s].test
                pw_s = lodo.score_arm(pathway, te_s, s, te_s.feats[c["depth"]], ad)
                src_test = lodo._repro(pw_s["mase"],
                                       h3refs[s]["arrays"]["noa__test_mase"][c["depth"]],
                                       ss.FIT_REPRO_MASE_RTOL)
                rep = ss.fit_reproduction(res["fit"]["fit"], sha, h3refs[s]["fits"]["noa"][lab],
                                          source=s, label=lab, source_test=src_test)
                state["fit_reproduction"].append({"cell": key, **rep})
                log(f"  [fit-repro] {s} {lab}: {'PASS' if rep['passed'] else 'FAIL'}  bitwise "
                    f"adapter {rep['bitwise_adapter']}  selected (wd, epoch) "
                    f"({rep['checks']['selected_wd']['ours']}, "
                    f"{rep['checks']['selected_epoch']['ours']}) vs H3 "
                    f"({rep['checks']['selected_wd']['h3']}, "
                    f"{rep['checks']['selected_epoch']['h3']})  val curve max rel diff "
                    f"{rep['val_curve_max_rel_diff']}")
                if not rep["passed"]:
                    diag = save_adapter(adapter_root / "_fit_reproduction_failures" / job / t /
                                        f"{c['cell_id']}__noa__{lab}.npz", ad,
                                        {"cell": key, "diagnostic": True})
                    phase2.atomic_write_json(run_dir / "fit_reproduction_failure.json", {
                        "schema": "phase2f_fit_reproduction_failure/v1", "cell": key,
                        "reproduction": rep, "fit": res["fit"]["fit"], "adapter": diag,
                        "note": "the one-source pooled fit did not reproduce the committed H3 "
                                "fit of its source; no further fit was started"})
                    state["rc"] = 4
                    state["cells"][key] = {"status": "failed", "reason": "fit reproduction"}
                    log(f"\nFIT REPRODUCTION FAILED for {key}: stopping before any further fit. "
                        f"Diagnostics: {run_dir / 'fit_reproduction_failure.json'}")
                    break
            fc = ss.pooled_forecasts(pathway, prepared, t, c["depth"], ad)
            rows_pw = int(prepared[c["sources"][0]].train_states[c["depth"]].shape[1])
            work = ss.training_work(res["fit"], rows_pw)
            if torch.cuda.is_available():
                work["cuda_max_memory_allocated_gb"] = res["timings"].get(
                    "cuda_max_memory_allocated_gb")
            arec = save_adapter(adapter_root / t / c["cell_id"] / f"noa__{lab}.npz", ad, {
                "model": ss.MODEL, "family": "noa_source_scaling", "target": t,
                "depth_index": c["depth"], "label": lab, "n_sources": c["n_sources"],
                "order_seed": c["order_seed"], "sources": c["sources"],
                "selection_order": c["selection_order"],
                "selected_wd": res["fit"]["fit"]["selected_wd"],
                "selected_epoch": res["fit"]["fit"]["selected_epoch"], "config_hash": c["hash"]})
            store = _store(out, c)
            stage = store.begin()
            try:
                lodo.save_lodo_cell(stage, res, c["config"], c["hash"], arec, spec_rec,
                                    environment={"job": job})
                ss.save_extras(stage, c, res, fc, work=work, fit_repro=rep, job=job)
                final = store.commit(stage, c["hash"], extra={
                    "protocol": ss.PROTOCOL_VERSION, "phase2c_protocol": lodo.LODO_PROTOCOL_VERSION,
                    "n_sources": c["n_sources"], "order_seed": c["order_seed"]})
            except Exception:
                store.abandon(stage)
                raise
            s_ = res["stats"]
            dm = {a: 100 * s_["vs_native"][a]["mase"]["degradation"]
                  for a in ("hard", "pooled", "specific")}
            f = res["fit"]["fit"]
            units_done += c["n_sources"]
            el = time.time() - t_start
            eta = el / units_done * (units_total - units_done) if units_done else None
            log(f"  [{key}] dMASE hard {dm['hard']:+.1f}%  pooled {dm['pooled']:+.1f}%  "
                f"specific {dm['specific']:+.1f}%   (wd {f['selected_wd']}, epoch "
                f"{f['selected_epoch']}, {work['n_train_windows']} train windows, fit "
                f"{work['fit_seconds']:.0f}s)")
            log(f"  [progress] {k}/{len(todo)} cells this run, elapsed {el / 60:.1f} min, ETA "
                f"{'?' if eta is None else f'{eta / 60:.0f} min'}   COMPLETE {final}")
            state["cells"][key] = {"status": "complete", "config_hash": c["hash"],
                                   "timings": res["timings"], "fit_seconds": work["fit_seconds"]}
            phase2.atomic_write_json(run_dir / "progress.json", {
                "done_this_run": k, "todo_this_run": len(todo), "units_done": units_done,
                "units_total": units_total, "elapsed_s": round(el, 1),
                "eta_s": None if eta is None else round(eta, 1), "last": key,
                "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                    timespec="seconds")})
            del res
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            log(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            state["failures"].append((key, f"{type(exc).__name__}: {exc}"))
            state["cells"][key] = {"status": "failed", "reason": str(exc)[:500]}
            if state["rc"] == 0:
                state["rc"] = 1
        phase2.atomic_write_json(run_dir / "cells.json", state)
    phase2.atomic_write_json(run_dir / "cells.json", state)
    return state


# --------------------------------------------------------------------------- #
def check_inputs(args, cells, targets, n1_sources) -> list[str]:
    """Cheap existence / checksum checks of everything the job reads (no model, no states)."""
    from probing.phase2_align import load_adapter
    problems = []
    for p in (Path(args.window_cache), Path(args.chronos_features_dir or REPO_ROOT /
                                            "features_cache")):
        if not p.exists():
            problems.append(f"missing input directory {p}")
    for t in sorted(set(targets) | set(n1_sources)):
        d = Path(args.h3_root) / "h3" / ss.MODEL / t
        if not (d / "COMPLETE").exists():
            problems.append(f"no COMPLETE H3 cell {d}")
    for t in targets:
        p = Path(args.h3_adapter_root) / ss.MODEL / t / f"noa__{lodo.depth_label(ss.DEPTH)}.npz"
        try:
            _, meta = load_adapter(p)
            want = lodo.load_h3_reference(args.h3_root, t)["fits"]["noa"][
                lodo.depth_label(ss.DEPTH)].get("adapter_sha256")
            if meta.get("sha256") != want:
                problems.append(f"{p}: sha256 {meta.get('sha256')} != H3 record {want}")
        except Exception as exc:
            problems.append(f"{p}: {type(exc).__name__}: {exc}")
    return problems


def main(argv=None) -> int:
    args = parse_args(argv)
    roster = list(registry.roster("paper14"))
    out, adapter_root = Path(args.output_root), Path(args.adapter_root)
    ss.assert_isolated(out, adapter_root, REPO_ROOT)
    picked, audit = ss.select_targets(roster)
    if set(picked) != set(ss.TARGETS):
        raise RuntimeError(f"the frozen TARGETS {ss.TARGETS} no longer follow the selection rule "
                           f"({picked}); the design changed")
    targets = [t for t in ss.TARGETS if not args.targets or t in args.targets]
    cells = ss.plan_cells(roster, targets=targets)
    if args.n:
        cells = [c for c in cells if c["n_sources"] in args.n]
    fit_cfg = _fit_cfg(args)
    needed = sorted({c["target"] for c in cells} | {s for c in cells for s in c["sources"]},
                    key=roster.index)
    n1_sources = sorted({c["sources"][0] for c in cells if c["n_sources"] == 1},
                        key=roster.index)
    overrides = parse_overrides(args.phase1_override)

    # ------------------------------------------------------------ dependencies (JSON only)
    deps, p1cells, h3refs = {}, {}, {}
    for t in needed:
        p1 = Phase1Cell(resolve_phase1_root(t, args.phase1_root, overrides), lodo.MODEL, t)
        p1cells[t] = p1
        deps[t] = p1.dependency_record()
    for t in sorted(set(targets) | set(n1_sources), key=roster.index):
        h3refs[t] = lodo.load_h3_reference(args.h3_root, t)
    status = {}
    for c in cells:
        c["config"] = ss.cell_config(c, deps, h3refs[c["target"]], fit_cfg)
        c["hash"] = phase1._digest(c["config"])
        status[_key(c)] = _store(out, c).status(c["hash"])

    print(f"\nPHASE 2f SOURCE SCALING  {len(cells)} cells (adapter fits) -> {out / ss.KIND}")
    print(f"  targets {targets}  depth {lodo.depth_label(ss.DEPTH)}  N {list(ss.N_GRID)}  "
          f"order seeds {list(ss.ORDER_SEEDS)}")
    for c in _run_order(cells, roster):
        st, _ = status[_key(c)]
        print(f"  {c['target']:<28} {c['cell_id']:<11} {c['n_families']:>2} fam.  "
              f"{c['selection_order']}  [{st}]")
    if args.check_inputs:
        problems = check_inputs(args, cells, targets, n1_sources)
        for p in problems:
            print(f"  [inputs] {p}")
        print(f"  [inputs] {'all present' if not problems else f'{len(problems)} problem(s)'}")
        if problems:
            return 2
    if args.plan:
        return 0
    pending = [c for c in cells if not (status[_key(c)][0] == "complete" and args.resume)]
    bad = [c for c in pending if status[_key(c)][0] in ("incompatible", "corrupt")]
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
    from probing.phase2_align import load_adapter
    from probing.phase2_env import environment_record, set_precision_flags
    from probing.phase2_pathways import Chronos2Pathway, load_chronos2
    from probing.window_reference import window_digest

    numerics = set_precision_flags(deterministic=True)
    removed = []
    for t in sorted({c["target"] for c in pending}):
        removed += clean_staging(out, ss.KIND,
                                 cells=[(t, c["cell_id"]) for c in pending if c["target"] == t])
    if removed:
        print(f"[resume] removed stale staging dirs: {removed[:5]}")
    job = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    run_dir = out / ss.KIND / "runs" / job
    design = {"schema": "phase2f_design/v1", "protocol": ss.PROTOCOL_VERSION,
              "phase2c_protocol": lodo.LODO_PROTOCOL_VERSION, "depth": ss.DEPTH,
              "label": lodo.depth_label(ss.DEPTH), "n_grid": list(ss.N_GRID),
              "order_seeds": list(ss.ORDER_SEEDS), "order_rule": ss.ORDER_RULE,
              "targets": list(ss.TARGETS), "target_selection": audit,
              "target_rationale": ss.TARGET_RATIONALE,
              "orders": {t: ss.source_orders(t, roster) for t in ss.TARGETS},
              "cells": [{k: c[k] for k in ("target", "cell_id", "n_sources", "order_seed",
                                           "selection_order", "sources", "n_families",
                                           "series_family_members", "shared_across_seeds")}
                        for c in ss.plan_cells(roster)],
              "notes": {"series_families": "Electricity hourly and Electricity 15-minute share "
                                           "meters: the N = 13 pool holds 12 dataset families",
                        "data_volume": "each source contributes its full 1,394 training windows: "
                                       "a larger N also means more training data and more "
                                       "optimization work per step"}}
    phase2.atomic_write_json(out / ss.KIND / "design.json", design)
    env = environment_record()
    phase2.atomic_write_json(run_dir / "run_manifest.json", {
        "schema": "phase2f_source_scaling_run/v1", "protocol": ss.PROTOCOL_VERSION, "job": job,
        "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
            timespec="seconds"),
        "cells": [_key(c) for c in pending], "datasets_loaded": needed, "fit_cfg": fit_cfg,
        "numerics": numerics, "phase1_root": str(args.phase1_root),
        "h3_root": str(args.h3_root), "h3_adapter_root": str(args.h3_adapter_root),
        "adapter_root": str(adapter_root), "window_cache": str(args.window_cache),
        "environment": env})

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
        prepared[t] = lodo.prepare_dataset(pathway, data, [ss.DEPTH])
        if t in targets:
            p1arrays[t] = arrays
        n = [len(data.splits[s].target) for s in ("train", "val", "test")]
        print(f"  [data] {t:<28} train/val/test {n}  ({time.time() - t1:.1f}s)")
        del data, w

    # ------------------------------------------------------------ PREFLIGHT (before ANY fit)
    lab = lodo.depth_label(ss.DEPTH)
    specific, pf, bad_pf = {}, [], []
    t_pf = time.time()
    for t in sorted({c["target"] for c in pending}, key=roster.index):
        sp = Path(args.h3_adapter_root) / lodo.MODEL / t / f"noa__{lab}.npz"
        try:
            spec_ad, spec_meta = load_adapter(sp, device)
            want = h3refs[t]["fits"]["noa"][lab].get("adapter_sha256")
            rec = lodo.preflight_gates(pathway, prepared, t, ss.DEPTH, phase1_arrays=p1arrays[t],
                                       h3_ref=h3refs[t], specific_adapter=spec_ad,
                                       fit_cfg=fit_cfg, device=device)
            rec["specific_adapter"] = {"path": str(sp), "sha256": spec_meta.get("sha256"),
                                       "expected_sha256": want,
                                       "passed": spec_meta.get("sha256") == want}
            if not rec["specific_adapter"]["passed"]:
                rec["failed"].append("specific_adapter_sha256")
                rec["all_passed"] = False
            specific[t] = (spec_ad, {"path": str(sp), "sha256": spec_meta.get("sha256")})
            te = prepared[t].test
            L = prepared[t].reference_index
            base = {}
            for arm, feats, ad in (("native", te.feats[L], None), ("hard", te.feats[ss.DEPTH], None),
                                   ("specific", te.feats[ss.DEPTH], spec_ad)):
                _, z = pathway.run(feats, ad)
                base[f"{arm}_q_norm"] = z.astype("float32")
                base[f"{arm}_q_raw"] = pathway.to_raw(z, te).astype("float32")
            ss._atomic_savez(out / ss.KIND / "baselines" / t / "forecasts.npz", **base,
                             rows_test=te.rows, cluster_ids_test=te.cluster_ids)
        except Exception as exc:
            traceback.print_exc()
            rec = {"target": t, "depth": ss.DEPTH, "label": lab, "all_passed": False,
                   "failed": ["exception"], "error": f"{type(exc).__name__}: {exc}"}
        pf.append(rec)
        m = rec.get("mean_test_mase", {})
        print(f"  [preflight] {t:<28} {lab:<4} {'PASS' if rec['all_passed'] else 'FAIL'}"
              + (f"  mean test MASE native {m['native']:.4f} hard {m['hard']:.4f} "
                 f"specific {m['specific']:.4f}" if m else "")
              + ("" if rec["all_passed"] else f"  failed: {rec['failed']}"))
        if not rec["all_passed"]:
            bad_pf.append(f"{t}/{lab}")
    phase2.atomic_write_json(run_dir / "preflight.json", {
        "schema": "phase2f_preflight/v1", "protocol": ss.PROTOCOL_VERSION,
        "gates": list(lodo.PREFLIGHT_GATES) + ["specific_adapter_sha256"],
        "n_targets": len(pf), "n_failed": len(bad_pf), "failed": bad_pf,
        "seconds": round(time.time() - t_pf, 1), "targets": pf})
    if bad_pf:
        print(f"\nPREFLIGHT FAILED for {bad_pf}: NO adapter fit was started. Diagnostics: "
              f"{run_dir / 'preflight.json'}")
        return 3
    print(f"  [preflight] {len(pf)}/{len(pf)} targets pass ({time.time() - t_pf:.0f}s)")

    state = run_cells(pathway, prepared, pending, out=out, adapter_root=adapter_root,
                      h3refs=h3refs, p1arrays=p1arrays, specific=specific, fit_cfg=fit_cfg,
                      roster=roster, device=device, job=job, run_dir=run_dir,
                      resume=args.resume, log=print)
    n_ok = sum(v["status"] == "complete" for v in state["cells"].values())
    print(f"\n{'=' * 86}\nSOURCE SCALING SUMMARY  {n_ok} complete, {len(state['failures'])} "
          f"failed this run, fit reproductions {sum(r['passed'] for r in state['fit_reproduction'])}"
          f"/{len(state['fit_reproduction'])} pass")
    for k, why in state["failures"]:
        print(f"  {k}: {why}")
    if state["rc"] == 4:
        return 4
    return 1 if state["failures"] and args.fail_on_error else 0


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
    g.add_argument("--h3-adapter-root", default=str(scratch / "chronos2_phase2" / "adapters"))
    g = p.add_argument_group("outputs")
    g.add_argument("--output-root", default=str(DEFAULT_ROOT))
    g.add_argument("--adapter-root", default=str(scratch / "chronos2" / "phase2f_source_scaling"
                                                 / "adapters"))
    g = p.add_argument_group("what to run")
    g.add_argument("--targets", nargs="+", default=None, choices=list(ss.TARGETS))
    g.add_argument("--n", nargs="+", type=int, default=None, choices=list(ss.N_GRID))
    g.add_argument("--plan", action="store_true")
    g.add_argument("--check-inputs", action="store_true")
    g.add_argument("--resume", action="store_true", default=True)
    g.add_argument("--no-resume", dest="resume", action="store_false")
    g.add_argument("--force-recompute", action="store_true")
    g.add_argument("--fail-on-error", action="store_true")
    g = p.add_argument_group("protocol (defaults = the frozen H3 / Phase 2c optimizer)")
    g.add_argument("--epochs", type=int, default=phase2.ADAPTER_EPOCHS)
    g.add_argument("--lr", type=float, default=phase2.ADAPTER_LR)
    g.add_argument("--wd-grid", type=float, nargs="+", default=list(phase2.ADAPTER_WD_GRID))
    g.add_argument("--eval-every", type=int, default=phase2.ADAPTER_EVAL_EVERY)
    g.add_argument("--boot-b", type=int, default=phase2.BOOT_B)
    g.add_argument("--seed", type=int, default=phase2.SEED)
    g.add_argument("--native-gate-rtol", type=float, default=1e-4)
    g.add_argument("--device", default=os.environ.get("PHASE2_DEVICE", None))
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
