"""Phase 2e -- non-contiguous block selection for Chronos-2, all 14 paper datasets.

    At the SAME number of retained blocks (4 / 6 / 9), does keeping selected middle and late
    blocks beat the paper's prefix truncation?

Per dataset: 12 matched-budget masks (prefix | uniform | early_late | middle) x {hard, NOA}, where
the 9 non-prefix masks are run through the SKIPPED-BLOCK backbone and get a fresh H3 NOA adapter
(the H3 fit, unchanged), and the 3 prefix masks reuse the committed H3 cells after reproduction
gates; then the single-block skipping diagnostic (12 masks, hard only) and the paired per-dataset
comparison vs prefix. See ``probing.phase2e_noncontig`` for the protocol.

    # PLAN (login node: JSON reads only, no model, no arrays; seconds)
    python -m experiments.run_phase2e_noncontig --plan
    python -m experiments.run_phase2e_noncontig --plan --check-inputs     # + stat() every input

    # FULL (compute node, ONE A100; resumable -- re-running the identical line skips COMPLETE cells)
    python -m experiments.run_phase2e_noncontig --device cuda

WHERE TO RUN. Everything except ``--plan`` loads Chronos-2 and the forecast-slot caches: a COMPUTE
NODE with a GPU (job_phase2e_noncontig.sh), never the login node.

READ-ONLY INPUTS. Phase-1 windows / K=4 caches / cells, the committed H3 cells
(results/three_model_phase2/h3) and H3 adapters ($SCRATCH/chronos2_phase2/adapters), the committed
H4 physical metrics. WRITES ONLY: results/three_model_phase2e_noncontig/ (compact artifacts) and
$SCRATCH/chronos2/phase2e_noncontig/{adapters,states}/. Phase 2c (LODO) and 2d (pooled) trees,
caches and jobs are never read or written; the driver refuses output roots inside any of them.
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
from types import SimpleNamespace

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2, registry                                      # noqa: E402
from probing import phase2e_noncontig as P                                        # noqa: E402
from probing.phase2 import (DEFAULT_OUT_ROOT, DEFAULT_PHASE1_ROOT, Phase1Cell,      # noqa: E402
                            Phase1DependencyError, Phase2CellStore, clean_staging,
                            parse_overrides, resolve_phase1_root)

DEFAULT_OUT = REPO_ROOT / "results" / "three_model_phase2e_noncontig"
CHECKPOINT = "amazon/chronos-2"
SPLITS = ("train", "val", "test")
#: kinds (cell trees) inside the output root
KIND_MASK, KIND_SKIP, KIND_CMP = "masks", "single_skip", "comparison"
MASK_REQUIRED = ("cell_config.json", "metrics.json", "per_window.npz", "fit.json",
                 "verification.json")
SKIP_REQUIRED = ("cell_config.json", "metrics.json", "per_window.npz", "verification.json")
CMP_REQUIRED = ("cell_config.json", "comparison.json", "verification.json")
#: result trees Phase 2e never writes into (Phase 1, Phase 2 H3/H4, 2b, 2c LODO, 2d pooled)
PROTECTED = ("three_model_phase1_q9_expanded", "three_model_phase1_q9_loop_lowlr",
             "three_model_phase1_q1", "three_model_phase1_q1_loop_lowlr", "three_model_final",
             "three_model_phase2", "three_model_phase2_smoke", "three_model_phase2_optcheck",
             "three_model_phase2_nested", "three_model_phase2_nonlinear_smoke",
             "three_model_phase2_lodo", "three_model_phase2_lodo_smoke",
             "three_model_phase2d_pooled", "three_model_phase2d_pooled_smoke")
#: $SCRATCH sub-trees owned by other phases (read-only or untouchable for Phase 2e)
PROTECTED_SCRATCH = ("chronos2_phase2", "chronos2/features_cache", "chronos2/phase1_shared_cache",
                     "chronos2/phase1_shared_windows", "chronos2/phase2d_pooled_cache",
                     "chronos2/phase2d_pooled_artifacts", "chronos2/hf_cache")
#: measured on the committed H3 cells: hard + NOA + FL + RA ~124 s per depth on an A100, of which
#: one NOA fit (6 decays x 1500 full-batch epochs) is ~half. Latency: 27 configurations per repeat,
#: each ~60 s (fresh load + 120 calls at B=1 and at B=256 ~ 0.35 s, both levels; committed Phase-2
#: Chronos-2 timings). Used for --plan's time estimate and the progress ETA only.
EST_FIT_S, EST_DATASET_OVERHEAD_S, EST_LATENCY_S = 65.0, 90.0, 3 * 27 * 60.0


def _scratch() -> Path:
    return Path(os.environ.get("SCRATCH", "/tmp"))


def assert_output_root(out) -> Path:
    out = Path(out).resolve()
    for name in PROTECTED:
        p = (REPO_ROOT / "results" / name).resolve()
        if out == p or p in out.parents:
            raise SystemExit(f"refusing --output-root {out}: inside the protected tree "
                             f"results/{name} (Phase 1 / Phase 2 / 2b / 2c / 2d)")
    return out


def assert_scratch_root(path, what: str) -> Path:
    p = Path(path).resolve()
    for sub in PROTECTED_SCRATCH:
        q = (_scratch() / sub).resolve()
        if p == q or q in p.parents:
            raise SystemExit(f"refusing {what} {p}: inside {q}, owned by another phase")
    if (REPO_ROOT / "features_cache").resolve() in [p, *p.parents]:
        raise SystemExit(f"refusing {what} {p}: inside the Phase-1 feature cache")
    return p


# --------------------------------------------------------------------------- #
# read-only references
# --------------------------------------------------------------------------- #
def load_h3_reference(h3_root, tag: str, *, arrays: bool = True) -> dict:
    """The committed dataset-specific H3 Chronos-2 cell. READ-ONLY; incomplete = error."""
    d = Path(h3_root) / "h3" / P.MODEL / tag
    mk = d / "COMPLETE"
    if not mk.exists():
        raise Phase1DependencyError(f"no COMPLETE H3 cell at {d}: Phase 2e reuses its prefix arms")
    marker = json.loads(mk.read_text())
    cfg = json.loads((d / "cell_config.json").read_text())
    fits = json.loads((d / "fits.json").read_text())
    for k in P.BUDGETS:
        if f"L{k}" not in fits.get("noa", {}) or not fits["noa"][f"L{k}"].get("adapter_sha256"):
            raise Phase1DependencyError(f"{d}: no saved NOA adapter at L{k}")
    ref = {"dir": str(d), "config_hash": marker.get("config_hash"), "config": cfg,
           "fits": fits, "protocol": marker.get("protocol")}
    if arrays:
        with np.load(d / "bootstrap_inputs.npz", allow_pickle=True) as z:
            ref["arrays"] = {k: z[k] for k in z.files}
    return ref


def load_h4_physical(h3_root, tag: str) -> dict:
    """Committed physical H4 per-window test MASE of Chronos-2 at the depths where it exists:
    {(arm, depth): (n,)} from the entrance cell (h4/) and the budget cells (h4_budget/)."""
    out = {}
    root = Path(h3_root)
    ent = root / "h4" / P.MODEL / tag
    if (ent / "COMPLETE").exists():
        tr = json.loads((ent / "truncation.json").read_text())
        dep = int(tr["depth_index"])
        with np.load(ent / "physical_metrics.npz", allow_pickle=True) as z:
            for arm in ("hard", "noa"):
                if f"{arm}__mase" in z.files:
                    out[(arm, dep)] = np.asarray(z[f"{arm}__mase"], np.float64)
            out["cluster_ids_entrance"] = np.asarray(z["cluster_ids_test"])
    bud = root / "h4_budget" / P.MODEL / tag
    if (bud / "COMPLETE").exists():
        with np.load(bud / "physical_metrics.npz", allow_pickle=True) as z:
            for k in z.files:
                parts = k.split("__")
                if len(parts) == 3 and parts[2] == "mase" and parts[0] in ("hard", "noa"):
                    lab = parts[1]
                    if lab.startswith("L"):
                        out.setdefault((parts[0], int(lab[1:])), np.asarray(z[k], np.float64))
            out["cluster_ids_budget"] = np.asarray(z["cluster_ids_test"])
    return out


# --------------------------------------------------------------------------- #
# identities
# --------------------------------------------------------------------------- #
def mask_cell_config(tag, cfg_entry, dep, h3, fit_cfg) -> dict:
    """Everything that would change a reported number of one (dataset, mask) cell."""
    return {"protocol": P.PHASE2E_PROTOCOL_VERSION, "kind": KIND_MASK, "model": P.MODEL,
            "dataset": tag, "strategy": cfg_entry["strategy"], "budget": cfg_entry["budget"],
            "mask": list(cfg_entry["mask"]), "mask_table_digest": P.mask_table_digest(),
            "prefix_policy": "H3 reuse after gates; refit if the state gate fails",
            "phase1_fit_hash": dep["fit_hash"], "window_digest": dep["window_digest"],
            "h3_config_hash": h3["config_hash"], "checkpoint": CHECKPOINT,
            "extract_batch_size": P.EXTRACT_BATCH, "K": P.K_SLOTS,
            "epochs": fit_cfg["epochs"], "lr": fit_cfg["lr"], "wd_grid": list(fit_cfg["wd_grid"]),
            "eval_every": fit_cfg["eval_every"], "seed": fit_cfg["seed"],
            "noa_adapter": "affine", "native_gate_rtol": fit_cfg["native_gate_rtol"],
            "quantiles": [float(x) for x in phase2.QUANTILES], "boot_b": fit_cfg["boot_b"],
            "boot_seed": fit_cfg["boot_seed"], "state_gate_tol": P.STATE_GATE_TOL,
            "h3_repro_rtol": P.H3_REPRO_RTOL, "v3_mean_rtol": P.V3_MEAN_RTOL}


def skip_cell_config(tag, dep, h3, fit_cfg) -> dict:
    return {"protocol": P.PHASE2E_PROTOCOL_VERSION, "kind": KIND_SKIP, "model": P.MODEL,
            "dataset": tag, "skip_masks": {str(i): list(m) for i, m in P.SKIP_MASKS.items()},
            "phase1_fit_hash": dep["fit_hash"], "window_digest": dep["window_digest"],
            "h3_config_hash": h3["config_hash"], "checkpoint": CHECKPOINT,
            "extract_batch_size": P.EXTRACT_BATCH, "K": P.K_SLOTS, "arms": ["native", "hard"],
            "quantiles": [float(x) for x in phase2.QUANTILES], "boot_b": fit_cfg["boot_b"],
            "boot_seed": fit_cfg["boot_seed"], "native_gate_rtol": fit_cfg["native_gate_rtol"]}


def cmp_cell_config(tag, mask_hashes: dict, fit_cfg) -> dict:
    return {"protocol": P.PHASE2E_PROTOCOL_VERSION, "kind": KIND_CMP, "model": P.MODEL,
            "dataset": tag, "mask_cells": dict(sorted(mask_hashes.items())),
            "boot_b": fit_cfg["boot_b"], "boot_seed": fit_cfg["boot_seed"],
            "selection_rule": "secondary: per budget x arm, the strategy with the lowest mean "
                              "VALIDATION MASE (ties -> prefix); test read once afterwards"}


def mask_store(out, tag, label) -> Phase2CellStore:
    return Phase2CellStore(out, KIND_MASK, tag, label, required=MASK_REQUIRED)


def skip_store(out, tag) -> Phase2CellStore:
    return Phase2CellStore(out, KIND_SKIP, P.MODEL, tag, required=SKIP_REQUIRED)


def cmp_store(out, tag) -> Phase2CellStore:
    return Phase2CellStore(out, KIND_CMP, P.MODEL, tag, required=CMP_REQUIRED)


def build_plan(tags, deps, h3refs, fit_cfg, out, *, masks=None, single_skip=True) -> dict:
    """{tag: {"masks": [entry + config + hash + status], "skip": ..., "cmp": ...}} (JSON only)."""
    masks = masks or P.plan_masks()
    plan = {}
    for t in tags:
        rows = []
        for e in masks:
            ccfg = mask_cell_config(t, e, deps[t], h3refs[t], fit_cfg)
            h = phase1._digest(ccfg)
            rows.append({**e, "config": ccfg, "hash": h,
                         "status": mask_store(out, t, e["label"]).status(h)})
        sk = None
        if single_skip:
            scfg = skip_cell_config(t, deps[t], h3refs[t], fit_cfg)
            sh = phase1._digest(scfg)
            sk = {"config": scfg, "hash": sh, "status": skip_store(out, t).status(sh)}
        ccfg = cmp_cell_config(t, {r["label"]: r["hash"] for r in rows}, fit_cfg)
        ch = phase1._digest(ccfg)
        plan[t] = {"masks": rows, "skip": sk,
                   "cmp": {"config": ccfg, "hash": ch, "status": cmp_store(out, t).status(ch)}}
    return plan


def _pending(st, resume: bool) -> bool:
    return not (st[0] == "complete" and resume)


# --------------------------------------------------------------------------- #
# the heavy part
# --------------------------------------------------------------------------- #
class Runner:
    """Holds the loaded model, the state cache and the per-job bookkeeping."""

    def __init__(self, args, fit_cfg, numerics, out: Path, job: str):
        from probing import extraction
        from probing import phase2_truncate as T
        self.args, self.fit_cfg, self.numerics, self.out, self.job = args, fit_cfg, numerics, \
            out, job
        t0 = time.time()
        self.pipe, _ = extraction.get_pipeline()
        self.core = self.pipe.model
        # the H3 pathway class (Chronos2Pathway) on the checkpoint's OWN frozen head + final norm,
        # with d read from the model (768 for amazon/chronos-2; the tiny contract model is 32)
        self.pathway = T.pathway_for(self.pipe, P.MODEL)
        if not (self.pathway.head is self.core.output_patch_embedding
                and self.pathway.norm is self.core.encoder.final_layer_norm):
            raise RuntimeError("the pathway does not wrap the model's own head / final norm")
        if int(self.core.config.d_model) == 768 and self.pathway.d != 768:
            raise RuntimeError(f"pathway width {self.pathway.d} != 768")
        self.device = args.device or str(self.pathway.device)
        self.cache = P.StateCache(args.state_root)
        self.weights0 = P.weights_digest(self.pipe)
        self.fit_seconds, self.cells_done = [], 0
        print(f"  [model] Chronos-2 on {self.pathway.device} in {time.time() - t0:.1f}s; "
              f"weights {self.weights0[:23]}...")

    # ---------------------------------------------------------------- states
    def states(self, tag, mask, split, X, window_digest) -> np.ndarray:
        meta = P.StateCache.meta(tag, mask, split, X, window_digest=window_digest,
                                 checkpoint=CHECKPOINT, batch_size=P.EXTRACT_BATCH,
                                 numerics=self.numerics)
        got = self.cache.read(tag, mask, split, meta)
        if got is not None:
            return got
        s = P.extract_mask_states(self.pipe, X, mask, batch_size=P.EXTRACT_BATCH)
        self.cache.write(tag, mask, split, meta, s)
        return s

    def check_weights(self, where: str) -> str:
        w = P.weights_digest(self.pipe)
        if w != self.weights0:
            raise RuntimeError(f"Chronos-2 weights CHANGED ({where}): {w} != {self.weights0}")
        return w


def verify_masks(runner, mask_list, n: int = 8, seed: int = 0) -> dict:
    """Model-level checks on synthetic contexts, BEFORE any dataset (no labels, no test data):
       M1 the retained blocks execute exactly once each, in the original order (trace == mask);
          skipped blocks never execute
       M2 the encoder's block list and final norm are restored after every use
       M3 identity adapter spliced at the mask == the hard cut at the mask (public API, bitwise)
       M4 physical (public API) == offline (extracted states -> frozen pathway -> inverse)
       M5 the all-12 mask == the unmodified model (public API, bitwise)
       M6 no pretrained weight changes (state_dict sha256)"""
    from probing import phase2_truncate as T
    from probing.phase2_align import ResidualAdapter
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-50, 50, size=(n, 1))
    sig = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=(n, 1)))
    X = (mu + sig * np.cumsum(rng.normal(size=(n, phase1.PHASE1_C)), axis=1)).astype(np.float32)
    pipe, pw, core = runner.pipe, runner.pathway, runner.core
    ids0 = [id(b) for b in core.encoder.block]
    norm0 = core.encoder.final_layer_norm
    unmod = np.ascontiguousarray(T.public_forecast(pipe, P.MODEL, X, batch_size=n)
                                 .transpose(0, 2, 1))
    inv = T.inverse_split(pipe, P.MODEL, X)
    recs = {}
    for mask in mask_list:
        key = P.mask_string(mask)
        if key in recs:
            continue
        r = {"mask": list(mask)}
        trace = P.block_call_trace(pipe, X, mask)
        r["M1_trace"] = trace
        r["M1_passed"] = trace == list(mask)
        r["M2_passed"] = ([id(b) for b in core.encoder.block] == ids0
                          and core.encoder.final_layer_norm is norm0)
        hard = P.physical_forecast(pipe, X, mask, batch_size=n)
        ident = P.physical_forecast(pipe, X, mask, batch_size=n,
                                    adapter=ResidualAdapter(pw.d).to(runner.device).eval())
        r["M3_passed"] = bool(np.array_equal(hard, ident))
        st = P.extract_mask_states(pipe, X, mask, batch_size=n)
        _, z = pw.run(st)
        off = pw.to_raw(z, inv)
        g = T.elementwise_gate(hard, off, atol=1e-4 * float(np.abs(off).mean()), rtol=1e-4)
        r["M4"] = g
        r["M4_passed"] = bool(g["passed"])
        if tuple(mask) == P.FULL_MASK:
            r["M5_bitwise"] = bool(np.array_equal(hard, unmod))
            r["M5_passed"] = r["M5_bitwise"]
        r["M2_passed"] = r["M2_passed"] and ([id(b) for b in core.encoder.block] == ids0
                                             and core.encoder.final_layer_norm is norm0)
        r["passed"] = all(v for k, v in r.items() if k.endswith("_passed"))
        recs[key] = r
    w = runner.check_weights("after the model-level mask verification")
    return {"protocol": P.PHASE2E_PROTOCOL_VERSION, "n_synthetic_contexts": n,
            "weights_digest": w, "M6_passed": True, "masks": recs,
            "passed_masks": sorted(k for k, v in recs.items() if v["passed"]),
            "failed_masks": sorted(k for k, v in recs.items() if not v["passed"]),
            "written_utc": _now()}


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def _repro(ours, ref, rtol) -> dict:
    a, b = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    if a.shape != b.shape:
        return {"passed": False, "reason": f"rows {a.shape} vs {b.shape}"}
    rel = abs(a.mean() - b.mean()) / max(abs(b.mean()), 1e-12)
    return {"mean_ours": float(a.mean()), "mean_ref": float(b.mean()), "mean_rel_diff": float(rel),
            "window_max_abs_diff": float(np.abs(a - b).max()), "bitwise": bool(np.array_equal(a, b)),
            "rtol": float(rtol), "passed": bool(rel <= rtol)}


def _arm_rows(sc: dict) -> dict:
    return {k: np.asarray(sc[k], np.float64) for k in ("loss", "mase", "mae", "wql_num")}


def _raw_mase(tag, split, raw) -> np.ndarray:
    from probing.phase1_metrics import raw_window_metrics
    return raw_window_metrics(tag, split.X, split.y_raw, raw, phase2.QUANTILES,
                              phase2.MEDIAN_INDEX)["mase_pw"]


def _save_cell(store, cfg_hash, ccfg, files_json: dict, npz: dict | None, extra: dict) -> Path:
    stage = store.begin()
    try:
        phase2.atomic_write_json(stage / "cell_config.json", {**ccfg, "config_hash": cfg_hash})
        for name, obj in files_json.items():
            phase2.atomic_write_json(stage / name, obj)
        if npz is not None:
            np.savez_compressed(stage / "per_window.npz", **npz)
        return store.commit(stage, cfg_hash, extra={"protocol": P.PHASE2E_PROTOCOL_VERSION,
                                                    **extra})
    except Exception:
        store.abandon(stage)
        raise


def process_dataset(runner, tag, plan_t, dep, p1cell, h3, h4phys, verif, progress) -> list:
    """Every pending cell of one dataset. Returns [(cell key, error)] for failed cells."""
    import torch
    from experiments.run_phase2_h3 import windows_readonly
    from probing.phase2_align import load_adapter, save_adapter
    from probing.phase2_h3 import native_gate
    from probing.phase2_pathways import load_chronos2
    from probing.window_reference import window_digest
    args, fit_cfg, pw = runner.args, runner.fit_cfg, runner.pathway
    resume = args.resume
    fails = []
    t_ds = time.time()

    # ------------------------------------------------------------ data (read-only)
    w = windows_readonly(tag, SimpleNamespace(window_cache=args.window_cache, suite=args.suite,
                                              allow_window_rebuild=False, seed=fit_cfg["seed"]))
    wd = window_digest(w)
    if wd != dep["window_digest"]:
        raise Phase1DependencyError(f"{tag}: window digest {wd} != Phase-1 {dep['window_digest']}")
    if h3["config"].get("phase1_fit_hash") != dep["fit_hash"]:
        raise Phase1DependencyError(f"{tag}: the H3 cell was built on Phase-1 fit hash "
                                    f"{h3['config'].get('phase1_fit_hash')}, not {dep['fit_hash']}")
    p1arrays = p1cell.arrays()
    data = load_chronos2(tag, w, p1arrays, cache_dir=args.chronos_features_dir)
    del w
    L = P.N_BLOCKS
    keep = set(P.BUDGETS) | {L - 1, L}
    for s in SPLITS:
        data.splits[s].feats = {l: f for l, f in data.splits[s].feats.items() if l in keep}
    tr, va, te = (data.splits[s] for s in SPLITS)
    H3A = h3["arrays"]
    rows_ok = (np.array_equal(np.asarray(te.cluster_ids), H3A["cluster_ids_test"])
               and np.array_equal(np.asarray(va.cluster_ids), H3A["cluster_ids_val"]))
    if not rows_ok:
        raise Phase1DependencyError(f"{tag}: val/test rows differ from the H3 cell's")
    print(f"  [data] {tag}: train/val/test {[len(data.splits[s].target) for s in SPLITS]} "
          f"({time.time() - t_ds:.1f}s)")

    # ------------------------------------------------------------ teacher + native (as H3)
    teacher_tr, _ = pw.run(tr.feats[L])
    teacher_va, _ = pw.run(va.feats[L])
    nat_te = P.score_states(pw, te, te.feats[L], tag)
    nat_va = P.score_states(pw, va, va.feats[L], tag)
    gate = native_gate(nat_te["loss"], nat_te["mase"], p1arrays, fit_cfg["native_gate_rtol"])
    if gate.get("passed") is False:
        raise RuntimeError(f"{tag}: native pathway does not reproduce Phase 1: {gate}")
    nat_vs_h3 = _repro(nat_te["mase"], H3A["native_test_mase"], P.H3_REPRO_RTOL)
    if not nat_vs_h3["passed"]:
        raise RuntimeError(f"{tag}: native test MASE != the H3 cell's: {nat_vs_h3}")
    teacher_sha = phase2.array_sha256(teacher_tr, teacher_va)
    wql_den = nat_te["wql_den"]
    cid = np.asarray(te.cluster_ids)
    B, bs = int(fit_cfg["boot_b"]), int(fit_cfg["boot_seed"])

    # ------------------------------------------------------------ dataset-level gates
    gates = {"native_gate_vs_phase1": gate, "native_vs_h3": nat_vs_h3,
             "teacher_sha256": teacher_sha}
    full = {s: runner.states(tag, P.FULL_MASK, s, data.splits[s].X, wd) for s in SPLITS}
    gates["full_mask_states_vs_phase1_L12"] = {s: P.state_gate(full[s], data.splits[s].feats[L])
                                               for s in SPLITS}
    from probing import phase2_truncate as T
    unmod = np.ascontiguousarray(T.public_forecast(runner.pipe, P.MODEL, te.X,
                                                   batch_size=P.EXTRACT_BATCH).transpose(0, 2, 1))
    phys_full = P.physical_forecast(runner.pipe, te.X, P.FULL_MASK)
    gates["full_mask_physical_equals_unmodified_bitwise"] = bool(np.array_equal(phys_full, unmod))
    gates["full_mask_physical_vs_offline_native"] = P.v3_check(_raw_mase(tag, te, phys_full),
                                                              nat_te["mase"])
    del full
    gates["all_passed"] = bool(all(g["passed"] for g in
                                   gates["full_mask_states_vs_phase1_L12"].values())
                               and gates["full_mask_physical_equals_unmodified_bitwise"]
                               and gates["full_mask_physical_vs_offline_native"]["passed"])
    phase2.atomic_write_json(runner.out / "gates" / f"{tag}.json", gates)
    print(f"  [gates] {tag}: full-mask states vs L12 "
          f"{[round(g['max_abs_over_std'], 9) for g in gates['full_mask_states_vs_phase1_L12'].values()]}"
          f" /std; physical full == unmodified {gates['full_mask_physical_equals_unmodified_bitwise']}")
    if not gates["all_passed"]:
        raise RuntimeError(f"{tag}: the all-12-block backbone does NOT reproduce the full model "
                           f"(gates/{tag}.json) -- no mask is evaluated on top of it")

    native_rows = _arm_rows(nat_te)
    common_npz = {"native__test_" + k: v for k, v in native_rows.items()}
    common_npz.update({"native__val_loss": nat_va["loss"], "native__val_mase": nat_va["mase"],
                       "wql_den": np.asarray(wql_den, np.float64), "cluster_ids_test": cid,
                       "cluster_ids_val": np.asarray(va.cluster_ids), "rows_test": te.rows})

    # ------------------------------------------------------------ the 12 mask cells
    for e in plan_t["masks"]:
        if not _pending(e["status"], resume):
            continue
        if e["status"][0] == "incompatible" and not args.force_recompute:
            fails.append((f"{tag}/{e['label']}", f"incompatible: {e['status'][1]}"))
            continue
        key = f"{tag}/{e['label']}"
        mask = tuple(e["mask"])
        ms = P.mask_string(mask)
        print(f"\n  ---- {key}  blocks {ms}  (skipped {P.mask_string(e['skipped']) if e['skipped'] else '-'})")
        t_cell = time.time()
        try:
            if ms not in verif["passed_masks"]:
                raise RuntimeError(f"mask {ms} failed the model-level verification "
                                   f"({verif['masks'].get(ms)})")
            states = {s: runner.states(tag, mask, s, data.splits[s].X, wd) for s in SPLITS}
            ver = {"model_level": verif["masks"][ms], "dataset_gates": f"gates/{tag}.json"}
            ad, fitrec, source = None, None, "fit"
            k = e["budget"]
            if e["is_prefix"]:
                sg = {s: P.state_gate(states[s], data.splits[s].feats[k]) for s in SPLITS}
                ver["prefix_states_vs_phase1_cache"] = sg
                ap = Path(args.h3_adapter_root) / P.MODEL / tag / f"noa__L{k}.npz"
                h3ad, h3meta = load_adapter(ap, runner.device)
                want = h3["fits"]["noa"][f"L{k}"]["adapter_sha256"]
                if h3meta.get("sha256") != want:
                    raise RuntimeError(f"{ap}: sha256 {h3meta.get('sha256')} != H3's {want}")
                # our scorer on the CACHED states == the H3 cell (same inputs, same code path)
                c_h = P.score_states(pw, te, te.feats[k], tag)
                c_n = P.score_states(pw, te, te.feats[k], tag, adapter=h3ad)
                ver["scorer_on_cached_states_vs_h3"] = {
                    "hard": _repro(c_h["mase"], H3A["hard__test_mase"][k], P.H3_REPRO_RTOL),
                    "noa": _repro(c_n["mase"], H3A["noa__test_mase"][k], P.H3_REPRO_RTOL)}
                if not all(v["passed"] for v in ver["scorer_on_cached_states_vs_h3"].values()):
                    raise RuntimeError(f"our scorer does not reproduce H3 on H3's own inputs: "
                                       f"{ver['scorer_on_cached_states_vs_h3']}")
                if all(g["passed"] for g in sg.values()):
                    source = "h3"
                    ad = h3ad
                    fitrec = {**h3["fits"]["noa"][f"L{k}"], "source": "committed H3 cell",
                              "h3_cell": h3["dir"], "adapter_path": str(ap)}
                else:
                    source = "refit"
                    ver["h3_reference_mean_test_mase"] = {
                        a: float(np.mean(H3A[f"{a}__test_mase"][k])) for a in ("hard", "noa")}
                    print(f"  [prefix] {key}: re-extracted states differ from the Phase-1 cache "
                          f"({ {s: g['max_abs_over_std'] for s, g in sg.items()} }) -> REFIT on "
                          "the re-extracted states, labelled 'refit'")
            if ad is None:                                  # non-prefix, or prefix fallback
                t_fit = time.time()
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                res = P.fit_noa(pw, states["train"], states["val"], teacher_tr, teacher_va,
                                fit_cfg=fit_cfg, device=runner.device, log=print,
                                label=f"noa {e['label']}")
                ad = res.pop("adapter")
                fit_s = round(time.time() - t_fit, 1)
                runner.fit_seconds.append(fit_s)
                arec = save_adapter(Path(args.adapter_root) / tag / f"noa__{e['label']}.npz", ad,
                                    {"model": P.MODEL, "dataset": tag, "family": "noa_phase2e",
                                     "strategy": e["strategy"], "budget": k,
                                     "mask": list(mask), "selected_wd": res["selected_wd"],
                                     "selected_epoch": res["selected_epoch"],
                                     "config_hash": e["hash"]})
                fitrec = {**res, "source": source, "fit_seconds": fit_s, "adapter_file": arec,
                          "teacher_sha256": teacher_sha,
                          "cuda_max_memory_allocated_gb": round(
                              torch.cuda.max_memory_allocated() / 1e9, 2)
                          if torch.cuda.is_available() else None}
            # ------------------------------------------------ evaluation (test read once)
            sv_h = P.score_states(pw, va, states["val"], tag)
            sv_n = P.score_states(pw, va, states["val"], tag, adapter=ad)
            st_h = P.score_states(pw, te, states["test"], tag)
            st_n = P.score_states(pw, te, states["test"], tag, adapter=ad)
            from probing.phase2_align import ResidualAdapter
            _, z_id = pw.run(states["test"], ResidualAdapter(pw.d).to(runner.device).eval())
            ver["identity_adapter_equals_hard_offline_bitwise"] = bool(
                np.array_equal(z_id, st_h["z"]))
            ph_h = P.physical_forecast(runner.pipe, te.X, mask)
            ph_n = P.physical_forecast(runner.pipe, te.X, mask, adapter=ad)
            ver["V3_hard"] = P.v3_check(_raw_mase(tag, te, ph_h), st_h["mase"])
            ver["V3_noa"] = P.v3_check(_raw_mase(tag, te, ph_n), st_n["mase"])
            if source == "h3":
                hard_rows = {m: np.asarray(H3A[f"hard__test_{m}"][k], np.float64)
                             for m in ("loss", "mase", "mae", "wql_num")}
                noa_rows = {m: np.asarray(H3A[f"noa__test_{m}"][k], np.float64)
                            for m in ("loss", "mase", "mae", "wql_num")}
                val_h = {m: np.asarray(H3A[f"hard__val_{m}"][k], np.float64)
                         for m in ("loss", "mase")}
                val_n = {m: np.asarray(H3A[f"noa__val_{m}"][k], np.float64)
                         for m in ("loss", "mase")}
                ver["reextracted_vs_h3"] = {"hard": _repro(st_h["mase"], hard_rows["mase"],
                                                           P.V3_MEAN_RTOL),
                                            "noa": _repro(st_n["mase"], noa_rows["mase"],
                                                          P.V3_MEAN_RTOL)}
                ver["physical_prefix_vs_h3_offline"] = {
                    "hard": P.v3_check(_raw_mase(tag, te, ph_h), hard_rows["mase"]),
                    "noa": P.v3_check(_raw_mase(tag, te, ph_n), noa_rows["mase"])}
                h4 = {}
                for arm, phys in (("hard", ph_h), ("noa", ph_n)):
                    ref = h4phys.get((arm, k))
                    if ref is not None:
                        h4[arm] = P.v3_check(_raw_mase(tag, te, phys), ref)
                ver["physical_prefix_vs_committed_h4_physical"] = h4 or {
                    "note": f"no committed physical H4 result for Chronos-2 at L{k} on {tag}"}
                if not all(v["passed"] for v in ver["reextracted_vs_h3"].values()):
                    raise RuntimeError(f"re-extracted prefix states pass the state gate but do not "
                                       f"reproduce H3: {ver['reextracted_vs_h3']}")
            else:
                hard_rows, noa_rows = _arm_rows(st_h), _arm_rows(st_n)
                val_h = {m: np.asarray(sv_h[m], np.float64) for m in ("loss", "mase")}
                val_n = {m: np.asarray(sv_n[m], np.float64) for m in ("loss", "mase")}
            v3_ok = bool(ver["V3_hard"]["passed"] and ver["V3_noa"]["passed"]
                         and ver["identity_adapter_equals_hard_offline_bitwise"])
            status = "ok" if v3_ok else "invalid_v3"
            pw_te = {"native": native_rows, "hard": hard_rows, "noa": noa_rows}
            stats = P.mask_cell_stats(pw_te, wql_den, cid, B=B, seed=bs)
            flags = P.fit_flags(fitrec)
            deg = {a: 100 * stats["vs_native"][a]["mase"]["degradation"] for a in ("hard", "noa")}
            metrics = {
                "schema": "phase2e_mask_cell/v1", "protocol": P.PHASE2E_PROTOCOL_VERSION,
                "dataset": tag, "strategy": e["strategy"], "budget": k, "label": e["label"],
                "mask": list(mask), "mask_string": ms, "skipped": list(e["skipped"]),
                "is_prefix": e["is_prefix"], "contiguous": e["contiguous"], "source": source,
                "status": status, "stats": stats,
                "test_mase": {a: stats["absolute"][a]["mase"] for a in pw_te},
                "test_mase_degradation_pct": deg,
                "val_mase": {"native": float(np.mean(nat_va["mase"])),
                             "hard": float(np.mean(val_h["mase"])),
                             "noa": float(np.mean(val_n["mase"]))},
                "selected_wd": fitrec.get("selected_wd"),
                "selected_epoch": fitrec.get("selected_epoch"),
                "selected_val_distillation_mse": fitrec.get("selected_val"),
                "hard_cut_val_distillation_mse": fitrec.get("hard_cut_val"),
                "flags": flags, "n_test": int(len(cid)),
                "timings": {"cell_s": round(time.time() - t_cell, 1),
                            "fit_s": fitrec.get("fit_seconds")},
                "provenance": {"job": runner.job, "phase1_fit_hash": dep["fit_hash"],
                               "h3_config_hash": h3["config_hash"]}}
            npz = {**common_npz,
                   **{f"hard__test_{m}": v for m, v in hard_rows.items()},
                   **{f"noa__test_{m}": v for m, v in noa_rows.items()},
                   **{f"hard__val_{m}": v for m, v in val_h.items()},
                   **{f"noa__val_{m}": v for m, v in val_n.items()},
                   "physical_hard__test_mase": _raw_mase(tag, te, ph_h),
                   "physical_noa__test_mase": _raw_mase(tag, te, ph_n)}
            fitrec = {k2: v for k2, v in fitrec.items() if k2 != "adapter"}
            final = _save_cell(mask_store(runner.out, tag, e["label"]), e["hash"], e["config"],
                               {"metrics.json": metrics, "fit.json": fitrec,
                                "verification.json": ver}, npz,
                               {"dataset": tag, "label": e["label"], "status": status,
                                "source": source})
            runner.cells_done += 1
            print(f"  [{key}] dMASE vs full  hard {deg['hard']:+.1f}%  aligned {deg['noa']:+.1f}%"
                  f"   (source {source}, wd {fitrec.get('selected_wd')}, epoch "
                  f"{fitrec.get('selected_epoch')}, flags {flags}, V3 {status}) -> {final}")
            e["status"] = ("complete", "this job")
        except Exception as exc:
            print(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            fails.append((key, f"{type(exc).__name__}: {exc}"))
        progress(tag, e["label"])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------ single-block skipping
    sk = plan_t["skip"]
    if sk is not None and _pending(sk["status"], resume):
        key = f"{tag}/single_skip"
        try:
            t0 = time.time()
            per, val, npz, ver = {}, {"native": {"mase": float(np.mean(nat_va["mase"]))}}, {}, {}
            for i, mask in P.SKIP_MASKS.items():
                ms = P.mask_string(mask)
                if ms not in verif["passed_masks"]:
                    raise RuntimeError(f"skip mask {ms} failed model-level verification")
                sv = P.score_states(pw, va, runner.states(tag, mask, "val", va.X, wd), tag)
                stt = runner.states(tag, mask, "test", te.X, wd)
                s_te = P.score_states(pw, te, stt, tag)
                per[i] = _arm_rows(s_te)
                val[i] = {"mase": float(np.mean(sv["mase"]))}
                for m, v in per[i].items():
                    npz[f"{P.skip_label(i)}__test_{m}"] = v
                npz[f"{P.skip_label(i)}__val_mase"] = np.asarray(sv["mase"], np.float64)
                if i == L:            # (1..11) is the prefix L11: must be the H3 hard cut at L11
                    ver["skip_L12_states_vs_phase1_L11"] = P.state_gate(stt, te.feats[L - 1])
                    ver["skip_L12_vs_h3_hard_L11"] = _repro(per[i]["mase"],
                                                            H3A["hard__test_mase"][L - 1],
                                                            P.V3_MEAN_RTOL)
            rows = P.single_skip_stats(per, native_rows, wql_den, cid, val=val, B=B, seed=bs)
            ok = bool(ver["skip_L12_vs_h3_hard_L11"]["passed"])
            metrics = {"schema": "phase2e_single_skip/v1", "dataset": tag, "rows": rows,
                       "native_test_mase": float(np.mean(nat_te["mase"])),
                       "status": "ok" if ok else "gate_failed",
                       "timings": {"cell_s": round(time.time() - t0, 1)},
                       "note": "hard cut only (no adapter): sensitivity of the frozen pathway to "
                               "removing ONE block, not a measure of intrinsic block importance"}
            _save_cell(skip_store(runner.out, tag), sk["hash"], sk["config"],
                       {"metrics.json": metrics, "verification.json": ver},
                       {**common_npz, **npz}, {"dataset": tag})
            worst = max(rows, key=lambda r: r["mase_degradation_pct"])
            print(f"  [{key}] worst single skip {worst['label']} "
                  f"{worst['mase_degradation_pct']:+.1f}%  ({time.time() - t0:.0f}s)")
        except Exception as exc:
            print(f"[FAIL] {key}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            fails.append((key, f"{type(exc).__name__}: {exc}"))

    # ------------------------------------------------------------ per-dataset comparison
    write_comparison(runner.out, tag, plan_t, fit_cfg, resume=resume, fails=fails)
    runner.check_weights(f"after dataset {tag}")
    print(f"  [dataset] {tag} done in {(time.time() - t_ds) / 60:.1f} min")
    return fails


def write_comparison(out, tag, plan_t, fit_cfg, *, resume=True, fails=None) -> bool:
    """Paired per-dataset comparison of every strategy vs prefix. Needs all 12 mask cells."""
    ccfg = cmp_cell_config(tag, {e["label"]: e["hash"] for e in plan_t["masks"]}, fit_cfg)
    ch = phase1._digest(ccfg)
    st = cmp_store(out, tag).status(ch)
    if not _pending(st, resume):
        return True
    per, valm, nat, cid, den, ver = {}, {}, None, None, None, {"cells": {}}
    for e in plan_t["masks"]:
        store = mask_store(out, tag, e["label"])
        if store.status(e["hash"])[0] != "complete":
            print(f"  [comparison] {tag}: {e['label']} not complete; comparison deferred")
            return False
        with np.load(store.final / "per_window.npz", allow_pickle=True) as z:
            a = {k: z[k] for k in z.files}
        m = json.loads((store.final / "metrics.json").read_text())
        ver["cells"][e["label"]] = {"status": m["status"], "source": m["source"]}
        rows = {arm: {mm: a[f"{arm}__test_{mm}"] for mm in ("loss", "mase", "mae", "wql_num")}
                for arm in ("hard", "noa")}
        per[(e["strategy"], e["budget"])] = rows
        valm[(e["strategy"], e["budget"])] = {"hard": float(a["hard__val_mase"].mean()),
                                               "noa": float(a["noa__val_mase"].mean())}
        n_rows = {mm: a[f"native__test_{mm}"] for mm in ("loss", "mase", "mae", "wql_num")}
        if nat is None:
            nat, cid, den = n_rows, a["cluster_ids_test"], a["wql_den"]
        elif not (all(np.array_equal(nat[mm], n_rows[mm]) for mm in nat)
                  and np.array_equal(cid, a["cluster_ids_test"])):
            raise RuntimeError(f"{tag}: the native arm / test rows differ across mask cells")
    ver["native_identical_across_cells"] = True
    stats = P.comparison_stats(per, nat, den, cid, val_mase=valm, B=int(fit_cfg["boot_b"]),
                               seed=int(fit_cfg["boot_seed"]))
    out_json = {"schema": "phase2e_comparison/v1", "dataset": tag, **stats,
                "val_mase": {f"{s}|{k}": v for (s, k), v in valm.items()},
                "native_test_mase": float(np.mean(nat["mase"]))}
    _save_cell(cmp_store(out, tag), ch, ccfg, {"comparison.json": out_json,
                                               "verification.json": ver}, None, {"dataset": tag})
    print(f"  [comparison] {tag}: written")
    return True


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    out = assert_output_root(args.output_root)
    roster = list(registry.roster("paper14"))
    tags = [t for t in roster if not args.datasets or t in args.datasets]
    if args.datasets and set(args.datasets) - set(roster):
        raise SystemExit(f"unknown datasets {sorted(set(args.datasets) - set(roster))}")
    overrides = parse_overrides(args.phase1_override)
    fit_cfg = make_fit_cfg(args)
    masks = [e for e in P.plan_masks()
             if (not args.strategies or e["strategy"] in args.strategies)
             and (not args.budgets or e["budget"] in args.budgets)]

    # ------------------------------------------------------------ dependencies (JSON only)
    deps, p1cells, h3refs = {}, {}, {}
    for t in tags:
        p1 = Phase1Cell(resolve_phase1_root(t, args.phase1_root, overrides), P.MODEL, t)
        p1cells[t] = p1
        deps[t] = p1.dependency_record()
        h3refs[t] = load_h3_reference(args.h3_root, t, arrays=False)
    plan = build_plan(tags, deps, h3refs, fit_cfg, out, masks=masks,
                      single_skip=not args.no_single_skip)
    n_fit = sum(1 for t in tags for e in plan[t]["masks"]
                if not e["is_prefix"] and _pending(e["status"], args.resume))
    n_cells = sum(1 for t in tags for e in plan[t]["masks"] if _pending(e["status"], args.resume))
    print(f"\nPHASE 2e  non-contiguous block selection (Chronos-2)  -> {out}")
    print(f"  masks frozen: {P.MASKS_FROZEN}\n  mask table digest {P.mask_table_digest()}")
    for e in masks:
        print(f"    {e['label']:<15} k={e['budget']}  blocks {e['mask_string']:<22} "
              f"{'PREFIX (H3 reuse)' if e['is_prefix'] else ('contiguous control' if e['contiguous'] else 'non-contiguous')}")
    print(f"\n  {'dataset':<28} {'mask cells (done/12)':<22} single-skip  comparison")
    for t in tags:
        done = sum(e["status"][0] == "complete" for e in plan[t]["masks"])
        bad = [e["label"] for e in plan[t]["masks"] if e["status"][0] == "incompatible"]
        sk = plan[t]["skip"]["status"][0] if plan[t]["skip"] else "off"
        print(f"  {t:<28} {done:>2}/{len(plan[t]['masks'])}{'  INCOMPATIBLE ' + str(bad) if bad else '':<18} "
              f"{sk:<12} {plan[t]['cmp']['status'][0]}")
    n_ds = sum(1 for t in tags if any(_pending(e["status"], args.resume) for e in plan[t]["masks"])
               or (plan[t]["skip"] and _pending(plan[t]["skip"]["status"], args.resume)))
    est = n_fit * EST_FIT_S + n_ds * EST_DATASET_OVERHEAD_S
    print(f"\n  pending: {n_cells} mask cells ({n_fit} new adapter fits), {n_ds} datasets to load"
          f"\n  estimate (A100): accuracy ~{est / 3600:.1f} h + latency ~{EST_LATENCY_S / 3600:.1f} h"
          f" (3 repeats) -- measured H3 fit cost, not a guarantee")
    if args.check_inputs:
        check_inputs(args, tags)
    if args.plan:
        return 0
    heavy = any(_pending(e["status"], args.resume) for t in tags for e in plan[t]["masks"]) or \
        any(plan[t]["skip"] and _pending(plan[t]["skip"]["status"], args.resume) for t in tags)
    if not heavy:                         # resume with nothing to fit: never load the model
        for t in tags:
            write_comparison(out, t, plan[t], fit_cfg, resume=args.resume)
        print("\nall mask and single-skip cells are COMPLETE (comparisons refreshed); nothing "
              "to fit")
        return 0

    # ------------------------------------------------------------ heavy part (compute node)
    assert_scratch_root(args.adapter_root, "--adapter-root")
    assert_scratch_root(args.state_root, "--state-root")
    import torch
    from probing.phase2_env import environment_record, set_precision_flags
    numerics = set_precision_flags(deterministic=True)
    job = os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"
    run_dir = out / "runs" / job
    removed = []
    for t in tags:
        removed += clean_staging(out, f"{KIND_MASK}", cells=[(t, e["label"]) for e in masks])
    removed += clean_staging(out, KIND_SKIP, cells=[(P.MODEL, t) for t in tags])
    removed += clean_staging(out, KIND_CMP, cells=[(P.MODEL, t) for t in tags])
    if removed:
        print(f"[resume] removed stale staging dirs: {removed[:5]}{' ...' if len(removed) > 5 else ''}")
    env = environment_record()
    phase2.atomic_write_json(run_dir / "run_manifest.json", {
        "schema": "phase2e_run/v1", "protocol": P.PHASE2E_PROTOCOL_VERSION, "job": job,
        "started_utc": _now(), "datasets": tags, "masks": [e["label"] for e in masks],
        "mask_table": {e["label"]: list(e["mask"]) for e in masks},
        "mask_table_digest": P.mask_table_digest(), "masks_frozen": P.MASKS_FROZEN,
        "fit_cfg": fit_cfg, "numerics": numerics, "phase1_root": str(args.phase1_root),
        "phase1_overrides": {k: str(v) for k, v in overrides.items()},
        "h3_root": str(args.h3_root), "h3_adapter_root": str(args.h3_adapter_root),
        "adapter_root": str(args.adapter_root), "state_root": str(args.state_root),
        "window_cache": str(args.window_cache), "environment": env})

    runner = Runner(args, fit_cfg, numerics, out, job)
    all_masks = [e["mask"] for e in P.plan_masks()] + list(P.SKIP_MASKS.values()) + [P.FULL_MASK]
    t0 = time.time()
    verif = verify_masks(runner, all_masks)
    phase2.atomic_write_json(out / "verification" / "masks.json", verif)
    phase2.atomic_write_json(run_dir / "mask_verification.json", verif)
    print(f"  [verify] {len(verif['passed_masks'])}/{len(verif['masks'])} masks pass M1-M6 on "
          f"synthetic contexts ({time.time() - t0:.0f}s); failed: {verif['failed_masks'] or 'none'}")

    t_job = time.time()
    state = {"job": job, "started_utc": _now(), "datasets": {}, "failures": []}

    def progress(tag, label):
        n = runner.cells_done
        fit_left = sum(1 for t in tags for e in plan[t]["masks"]
                       if not e["is_prefix"] and _pending(e["status"], args.resume))
        mean_fit = float(np.mean(runner.fit_seconds)) if runner.fit_seconds else EST_FIT_S
        eta = fit_left * mean_fit + EST_LATENCY_S
        state["progress"] = {"cells_done_this_job": n, "new_fits_left": fit_left,
                             "mean_fit_s": round(mean_fit, 1),
                             "elapsed_min": round((time.time() - t_job) / 60, 1),
                             "eta_remaining_min_incl_latency": round(eta / 60, 1),
                             "last": f"{tag}/{label}", "updated_utc": _now()}
        phase2.atomic_write_json(run_dir / "progress.json", state)
        print(f"  [progress] {n} cells this job | {fit_left} fits left x ~{mean_fit:.0f}s "
              f"+ latency ~{EST_LATENCY_S / 60:.0f} min -> ETA ~{eta / 60:.0f} min "
              f"| elapsed {(time.time() - t_job) / 60:.0f} min")

    failures = []
    for i, t in enumerate(tags, 1):
        pt = plan[t]
        need = (any(_pending(e["status"], args.resume) for e in pt["masks"])
                or (pt["skip"] and _pending(pt["skip"]["status"], args.resume)))
        print(f"\n{'#' * 90}\n# [{i}/{len(tags)}] {t}  ({registry.display_name(t)})\n{'#' * 90}")
        if not need:
            print("  all cells COMPLETE")
            write_comparison(out, t, pt, fit_cfg, resume=args.resume, fails=failures)
            continue
        try:
            h3 = load_h3_reference(args.h3_root, t, arrays=True)
            h4p = load_h4_physical(args.h3_root, t)
            f = process_dataset(runner, t, pt, deps[t], p1cells[t], h3, h4p, verif, progress)
            failures += f
            state["datasets"][t] = {"failed_cells": [k for k, _ in f]}
        except Exception as exc:
            print(f"[FAIL] {t}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append((t, f"{type(exc).__name__}: {exc}"))
            state["datasets"][t] = {"failed": str(exc)[:500]}
        state["failures"] = failures
        phase2.atomic_write_json(run_dir / "progress.json", state)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    runner.check_weights("end of job")
    state["finished_utc"] = _now()
    state["weights_unchanged"] = True
    phase2.atomic_write_json(run_dir / "progress.json", state)
    print(f"\n{'=' * 90}\nPHASE 2e SUMMARY  {runner.cells_done} cells written this job, "
          f"{len(failures)} failed; weights unchanged ({runner.weights0[:23]}...)")
    for k, why in failures:
        print(f"  {k}: {why}")
    return 1 if failures and args.fail_on_error else 0


def check_inputs(args, tags) -> None:
    """stat() every array input (no array is opened). Login-node safe."""
    from probing.phase2_pathways import chronos2_cache_candidates
    print("\n  INPUT CHECK (stat only)")
    bad = 0
    for t in tags:
        miss = []
        wp = Path(args.window_cache) / f"windows__{args.suite}__{t}.npz"
        if not wp.exists():
            miss.append(str(wp))
        for s in SPLITS:
            if not any(p.exists() for p in chronos2_cache_candidates(t, s, args.chronos_features_dir)):
                miss.append(f"K4 cache {t}/{s}")
        for k in P.BUDGETS:
            ap = Path(args.h3_adapter_root) / P.MODEL / t / f"noa__L{k}.npz"
            if not ap.exists():
                miss.append(str(ap))
        bad += bool(miss)
        print(f"    {t:<28} {'OK' if not miss else 'MISSING ' + '; '.join(miss)}")
    print(f"  -> {'all inputs present' if not bad else f'{bad} dataset(s) with missing inputs'}")


def make_fit_cfg(args) -> dict:
    from probing.phase2_h3 import DEFAULT_FIT_CFG
    cfg = {**DEFAULT_FIT_CFG, "epochs": args.epochs, "lr": args.lr,
           "wd_grid": list(phase2.assert_adapter_wd_grid(args.wd_grid, args.lr)),
           "eval_every": args.eval_every, "seed": args.seed,
           "native_gate_rtol": args.native_gate_rtol, "boot_b": args.boot_b,
           "boot_seed": args.seed, "noa_adapter": "affine"}
    h3 = {**DEFAULT_FIT_CFG}
    drift = {k: (cfg[k], h3[k]) for k in ("epochs", "lr", "wd_grid", "eval_every", "seed")
             if list(np.atleast_1d(cfg[k])) != list(np.atleast_1d(h3[k]))}
    if drift and not args.allow_protocol_change:
        raise SystemExit(f"the adapter protocol differs from the frozen H3 protocol: {drift}. "
                         "Phase 2e compares masks under the SAME fit; pass "
                         "--allow-protocol-change (and a new --output-root) to deviate.")
    return cfg


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    scratch = _scratch()
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
                   help="the committed Phase-2 tree (h3/ cells + h4/ physical metrics)")
    g.add_argument("--h3-adapter-root", default=str(scratch / "chronos2_phase2" / "adapters"))
    g = p.add_argument_group("outputs")
    g.add_argument("--output-root", default=str(DEFAULT_OUT))
    g.add_argument("--adapter-root", default=str(scratch / "chronos2" / "phase2e_noncontig" /
                                                 "adapters"))
    g.add_argument("--state-root", default=str(scratch / "chronos2" / "phase2e_noncontig" /
                                               "states"),
                   help="skipped-block forecast-slot states (atomic, verified on read)")
    g = p.add_argument_group("what to run")
    g.add_argument("--datasets", nargs="+", default=None)
    g.add_argument("--strategies", nargs="+", default=None, choices=list(P.STRATEGIES))
    g.add_argument("--budgets", nargs="+", type=int, default=None, choices=list(P.BUDGETS))
    g.add_argument("--no-single-skip", action="store_true")
    g.add_argument("--plan", action="store_true")
    g.add_argument("--check-inputs", action="store_true")
    g.add_argument("--resume", action="store_true", default=True)
    g.add_argument("--no-resume", dest="resume", action="store_false")
    g.add_argument("--force-recompute", action="store_true")
    g.add_argument("--fail-on-error", action="store_true")
    g = p.add_argument_group("protocol (defaults = the frozen H3 protocol; deviations refused)")
    g.add_argument("--epochs", type=int, default=phase2.ADAPTER_EPOCHS)
    g.add_argument("--lr", type=float, default=phase2.ADAPTER_LR)
    g.add_argument("--wd-grid", type=float, nargs="+", default=list(phase2.ADAPTER_WD_GRID))
    g.add_argument("--eval-every", type=int, default=phase2.ADAPTER_EVAL_EVERY)
    g.add_argument("--boot-b", type=int, default=phase2.BOOT_B)
    g.add_argument("--seed", type=int, default=phase2.SEED)
    g.add_argument("--native-gate-rtol", type=float, default=1e-4)
    g.add_argument("--allow-protocol-change", action="store_true")
    g.add_argument("--device", default=os.environ.get("PHASE2_DEVICE", None))
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())
