"""Phase 2e -- non-contiguous block selection for Chronos-2 (ICLR rebuttal experiment).

    "Have you tried non-contiguous layer selection or skipping specific intermediate layers,
    rather than always truncating from the end? If so, how does the performance compare?"

MATCHED-BUDGET COMPARISON. At k in {4, 6, 9} RETAINED blocks, four masks, frozen 2026-11-12 from
the rebuttal spec BEFORE any Phase-2e number existed (contract 01 pins them literally):

    strategy            k = 4          k = 6                k = 9
    prefix (paper)      1,2,3,4        1-6                  1-9
    uniform             1,4,8,12       1,3,5,8,10,12        1,2,4,5,7,8,10,11,12
    early_late          1,2,11,12      1,2,3,10,11,12       1-5,9-12
    middle              4-7            4-9                  2-10        (contiguous CONTROL)

MECHANISM. The retained blocks run in their ORIGINAL order; a skipped block is simply not in the
encoder's block list, so it performs no computation and the residual stream passes it unchanged
(the identity). This is exact for Chronos-2 2.3.1 because every ``Chronos2EncoderBlock`` is
self-contained: RoPE is computed inside each attention module and ``Chronos2Encoder.forward`` hands
the SAME masks / position ids to every block (no T5-style position bias computed in block 0).
Input embedding, ``encoder.final_layer_norm``, ``output_patch_embedding`` and the instance-norm
inverse are untouched; no pretrained parameter is reinitialized, fine-tuned or copied.

ARMS per (dataset, mask), all on the Phase-1 windows / rows and the K = 4 forecast slots:
    native   f(h_L12) of the FULL model (the reference; identical to H3's)
    hard     f(h_mask): the skipped-block backbone through the frozen native pathway, no adapter
    noa      f(a(h_mask)): the H3 label-free residual affine adapter, fitted with the H3 fit
             UNCHANGED (``phase2_align.fit_residual_adapter``: AdamW lr 1e-3, 1500 full-batch
             epochs, decoupled decay on Delta only over the H3 grid, validation every 25 epochs,
             epoch 0 = the hard cut, the same tie-breaking) on the states the SKIPPED-BLOCK
             backbone produces -- never on cached full-model states. Target = the full model's own
             normalized head outputs; no ground-truth future enters the fit or the selection.
Prefix masks reuse the committed H3 numbers after reproduction gates (states, H3 per-window, the
physical H4 per-window where it exists); if the state gate fails the prefix is REFIT and labelled.

DIAGNOSTIC (kept apart): single-block skipping, all blocks but one, hard cut only.

Nothing here loads a checkpoint or touches a cache by itself: the pipeline, pathway and arrays are
injected, so every contract runs on tiny random-weight models (tests.test_phase2e_noncontig).
"""

from __future__ import annotations

import contextlib
import hashlib
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from probing import phase1
from probing.phase1_metrics import raw_window_metrics
from probing.phase2 import (BOOT_B, MEDIAN_INDEX, QUANTILES, SEED, array_sha256,
                            cluster_replicates, ratio_summary)
from probing.phase2_align import fit_residual_adapter, mse_loss

__all__ = [
    "PHASE2E_PROTOCOL_VERSION", "MODEL", "N_BLOCKS", "K_SLOTS", "EXTRACT_BATCH", "BUDGETS",
    "STRATEGIES", "PREFIX", "CONTROL", "STRATEGY_LABELS", "MASKS", "MASKS_FROZEN", "FULL_MASK",
    "SKIP_MASKS", "ARMS", "METRICS", "STATE_GATE_TOL", "H3_REPRO_RTOL", "V3_MEAN_RTOL",
    "LATE_EPOCH_FRACTION", "validate_mask", "is_prefix", "is_contiguous", "skipped_blocks_of",
    "mask_label", "mask_string", "plan_masks", "mask_table_digest", "skip_label",
    "skipped_blocks", "spliced_adapter", "weights_digest", "extract_mask_states",
    "block_call_trace", "physical_forecast", "state_gate", "score_states", "fit_noa",
    "fit_flags", "mask_cell_stats", "comparison_stats", "single_skip_stats", "v3_check",
    "atomic_savez", "StateCache",
]

PHASE2E_PROTOCOL_VERSION = "phase2e/noncontig-v1"
MODEL = "chronos2"
N_BLOCKS = 12
K_SLOTS = 4                    # ceil(H / 16) native forecast slots (the Phase-1 readout)
EXTRACT_BATCH = 128            # the Phase-1 extraction batch (cell_config.extract_batch_size)
BUDGETS = (4, 6, 9)
STRATEGIES = ("prefix", "uniform", "early_late", "middle")
PREFIX = "prefix"
CONTROL = "middle"             # contiguous, but not a prefix: an additional control
STRATEGY_LABELS = {"prefix": "Prefix (paper)", "uniform": "Uniform",
                   "early_late": "Early + late", "middle": "Middle contiguous"}

#: The 12 matched-budget masks (1-indexed blocks L1..L12). FROZEN: changing one changes
#: ``mask_table_digest`` and therefore every cell hash, and contract 01 pins the literal table.
MASKS = {
    ("prefix", 4): (1, 2, 3, 4),
    ("prefix", 6): (1, 2, 3, 4, 5, 6),
    ("prefix", 9): (1, 2, 3, 4, 5, 6, 7, 8, 9),
    ("uniform", 4): (1, 4, 8, 12),
    ("uniform", 6): (1, 3, 5, 8, 10, 12),
    ("uniform", 9): (1, 2, 4, 5, 7, 8, 10, 11, 12),
    ("early_late", 4): (1, 2, 11, 12),
    ("early_late", 6): (1, 2, 3, 10, 11, 12),
    ("early_late", 9): (1, 2, 3, 4, 5, 9, 10, 11, 12),
    ("middle", 4): (4, 5, 6, 7),
    ("middle", 6): (4, 5, 6, 7, 8, 9),
    ("middle", 9): (2, 3, 4, 5, 6, 7, 8, 9, 10),
}
MASKS_FROZEN = ("2026-11-12, transcribed from the rebuttal specification before any Phase-2e "
                "number was computed; never selected on test (or on any) data")
FULL_MASK = tuple(range(1, N_BLOCKS + 1))
#: Single-block skipping diagnostic: every block but block i.
SKIP_MASKS = {i: tuple(b for b in FULL_MASK if b != i) for i in FULL_MASK}

ARMS = ("native", "hard", "noa")
METRICS = ("loss", "mase", "mae", "wql")
#: Prefix-mask states vs the Phase-1 cache: max |d| / std(cache) per depth. The 2d identity bar.
STATE_GATE_TOL = 1e-3
#: Same states + same code path (our scorer on the H3 inputs) -> H3: mean relative difference.
H3_REPRO_RTOL = 1e-6
#: Physical (public API) vs offline per-window MASE: the V3 rule (option A, 2026-09-26).
V3_MEAN_RTOL = 1e-4
LATE_EPOCH_FRACTION = 0.9      # the 2026-09-23 convergence rule (flag only)


# --------------------------------------------------------------------------- #
# the mask table
# --------------------------------------------------------------------------- #
def validate_mask(mask, n_blocks: int = N_BLOCKS) -> tuple:
    """A retained-block mask: non-empty, 1-indexed, strictly increasing (= original order, no
    repeats), inside 1..n_blocks."""
    m = tuple(int(b) for b in mask)
    if not m:
        raise ValueError("a mask must retain at least one block")
    if any(b < 1 or b > n_blocks for b in m):
        raise ValueError(f"mask {m} has blocks outside 1..{n_blocks}")
    if list(m) != sorted(set(m)):
        raise ValueError(f"mask {m} must be strictly increasing: retained blocks keep their "
                         "original order and appear once")
    return m


def is_prefix(mask) -> bool:
    m = tuple(mask)
    return m == tuple(range(1, len(m) + 1))


def is_contiguous(mask) -> bool:
    m = tuple(mask)
    return m == tuple(range(m[0], m[-1] + 1))


def skipped_blocks_of(mask, n_blocks: int = N_BLOCKS) -> tuple:
    return tuple(b for b in range(1, n_blocks + 1) if b not in set(mask))


def mask_label(strategy: str, k: int) -> str:
    return f"{strategy}_k{int(k)}"


def skip_label(i: int) -> str:
    return f"skip_L{int(i)}"


def mask_string(mask) -> str:
    """(1,2,3,4,5,9,10,11,12) -> '1-5,9-12'."""
    m = list(mask)
    runs, start = [], m[0]
    for a, b in zip(m, m[1:] + [None]):
        if b != a + 1:
            runs.append(f"{start}" if start == a else f"{start}-{a}")
            start = b
    return ",".join(runs)


def plan_masks() -> list[dict]:
    """The 12 matched-budget configurations, in a fixed (budget, strategy) order."""
    out = []
    for k in BUDGETS:
        for s in STRATEGIES:
            m = validate_mask(MASKS[(s, k)])
            if len(m) != k:
                raise RuntimeError(f"{s} k={k} retains {len(m)} blocks")
            out.append({"strategy": s, "budget": int(k), "mask": m, "label": mask_label(s, k),
                        "mask_string": mask_string(m), "skipped": skipped_blocks_of(m),
                        "is_prefix": is_prefix(m), "contiguous": is_contiguous(m),
                        "prefix_depth": len(m) if is_prefix(m) else None})
    return out


def mask_table_digest() -> str:
    return phase1._digest({f"{s}/{k}": list(v) for (s, k), v in sorted(MASKS.items())})


# --------------------------------------------------------------------------- #
# the backbone modification -- always restored
# --------------------------------------------------------------------------- #
def _core(handle) -> nn.Module:
    """pipeline.model for a Chronos2Pipeline, else the module itself."""
    m = getattr(handle, "model", None)
    return m if isinstance(m, nn.Module) else handle


@contextlib.contextmanager
def skipped_blocks(handle, mask):
    """Run Chronos-2 with ONLY the retained blocks, in their original order.

    The encoder's ``block`` ModuleList is swapped for one holding the SAME block objects (no copy:
    the pretrained parameters are shared, never duplicated or modified) and is restored on exit,
    also after an exception. Nesting is refused: a mask always indexes the ORIGINAL 12 blocks.
    """
    core = _core(handle)
    enc = core.encoder
    if getattr(enc, "_phase2e_original_blocks", None) is not None:
        raise RuntimeError("skipped_blocks is already active on this model (nesting refused)")
    original = enc.block
    m = validate_mask(mask, len(original))
    kept = nn.ModuleList([original[b - 1] for b in m])
    enc._phase2e_original_blocks = original
    enc.block = kept
    try:
        yield kept
    finally:
        enc.block = original
        enc._phase2e_original_blocks = None


@contextlib.contextmanager
def spliced_adapter(handle, adapter):
    """Insert ``adapter`` on the final retained block's output, immediately before the native
    final norm -- the H4 splice point (``phase2_truncate.splice_adapter``) -- and restore."""
    if adapter is None:
        yield
        return
    enc = _core(handle).encoder
    norm = enc.final_layer_norm
    enc.final_layer_norm = nn.Sequential(adapter, norm)
    try:
        yield
    finally:
        enc.final_layer_norm = norm


def weights_digest(handle) -> str:
    """sha256 over every state_dict tensor's exact bytes (sorted keys)."""
    h = hashlib.sha256()
    for k, v in sorted(_core(handle).state_dict().items()):
        a = v.detach().cpu().contiguous()
        h.update(k.encode())
        h.update(str(a.dtype).encode())
        h.update(str(tuple(a.shape)).encode())
        h.update(a.reshape(-1).view(torch.uint8).numpy().tobytes())
    return "sha256:" + h.hexdigest()


# --------------------------------------------------------------------------- #
# forward passes under a mask
# --------------------------------------------------------------------------- #
@torch.no_grad()
def extract_mask_states(handle, X, mask, *, batch_size: int = EXTRACT_BATCH,
                        K: int = K_SLOTS) -> np.ndarray:
    """(n, K, d) forecast-slot states at the OUTPUT OF THE FINAL RETAINED BLOCK (= the input of the
    native final norm) of the skipped-block backbone.

    The call mirrors the Phase-1 extraction exactly (``extraction.extract_kout_features``): float32
    contexts in batches of 128, ``model.encode(context, num_output_patches=K)``, eval mode, the last
    K tokens. For a prefix mask this IS the Phase-1 ``fslot_L<k>`` state (contract 05).
    """
    core = _core(handle)
    X = np.asarray(X, np.float32)
    if X.ndim != 2:
        raise ValueError(f"contexts must be (n, C), got {X.shape}")
    dev = next(core.parameters()).device
    cap, out = [], []

    def pre(_m, args):
        cap.append(args[0][:, -K:, :].detach().float().cpu())

    with skipped_blocks(core, mask):
        hk = core.encoder.final_layer_norm.register_forward_pre_hook(pre)
        was = core.training
        core.eval()
        try:
            for s in range(0, len(X), batch_size):
                cap.clear()
                ctx = torch.from_numpy(np.ascontiguousarray(X[s:s + batch_size])).to(
                    device=dev, dtype=torch.float32)
                core.encode(context=ctx, num_output_patches=K)
                if len(cap) != 1:
                    raise RuntimeError(f"final-norm hook fired {len(cap)} times for one batch")
                out.append(cap[0])
        finally:
            hk.remove()
            core.train(was)
    return torch.cat(out).numpy()


@torch.no_grad()
def block_call_trace(handle, X, mask, K: int = K_SLOTS) -> list[int]:
    """The 1-indexed ORIGINAL blocks that execute, in execution order, for one forward of X.
    Hooks sit on the block modules themselves, so a skipped block cannot run unseen."""
    core = _core(handle)
    blocks = list(core.encoder.block)
    order = []
    hs = [b.register_forward_pre_hook(lambda _m, _a, i=i: order.append(i))
          for i, b in enumerate(blocks, start=1)]
    dev = next(core.parameters()).device
    try:
        with skipped_blocks(core, mask):
            core.encode(context=torch.as_tensor(np.asarray(X, np.float32), device=dev),
                        num_output_patches=K)
    finally:
        for h in hs:
            h.remove()
    return order


@torch.no_grad()
def physical_forecast(handle, X, mask, *, adapter=None,
                      batch_size: int = EXTRACT_BATCH) -> np.ndarray:
    """The PUBLIC API (``Chronos2Pipeline.predict_quantiles``, the 9 levels selected exactly) of
    the skipped-block model with ``adapter`` spliced before the final norm. (n, 9, H) raw units."""
    from probing import phase2_truncate as T
    with skipped_blocks(handle, mask), spliced_adapter(handle, adapter):
        q = T.public_forecast(handle, MODEL, X, batch_size=batch_size)
    return np.ascontiguousarray(np.asarray(q, np.float64).transpose(0, 2, 1))


def state_gate(ours, ref, tol: float = STATE_GATE_TOL) -> dict:
    """Re-extracted states vs a cached reference: bitwise flag, max |d|, and max |d| / std(ref)."""
    a, b = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    if a.shape != b.shape:
        return {"passed": False, "reason": f"shape {a.shape} vs {b.shape}"}
    d = np.abs(a - b)
    sd = float(b.std()) or 1.0
    return {"bitwise": bool(np.array_equal(a, b)), "max_abs": float(d.max()),
            "max_abs_over_std": float(d.max() / sd), "ref_std": sd, "tol": float(tol),
            "passed": bool(d.max() / sd <= tol)}


# --------------------------------------------------------------------------- #
# scoring + the fit -- the H3 call chain, verbatim
# --------------------------------------------------------------------------- #
def score_states(pathway, split, states, tag: str, adapter=None) -> dict:
    """Per-window metrics of one arm: EXACTLY H3's ``score`` chain (pathway.run -> pinball in the
    normalized space; pathway.to_raw -> raw_window_metrics). Contract 07 pins equality with H3."""
    _, z = pathway.run(states, adapter)
    raw = pathway.to_raw(z, split)
    m = raw_window_metrics(tag, split.X, split.y_raw, raw, QUANTILES, MEDIAN_INDEX)
    return {"loss": phase1.mean_pinball_per_window(z, split.target, QUANTILES),
            "mase": m["mase_pw"], "mae": m["mae_pw"], "wql_num": m["wql_num_pw"],
            "wql_den": m["wql_den_pw"], "z": z, "raw": raw}


def fit_noa(pathway, train_states, val_states, teacher_train, teacher_val, *, fit_cfg: dict,
            device=None, log=None, label: str = "") -> dict:
    """The H3 label-free NOA fit, UNCHANGED. Same function, same arguments as
    ``phase2_h3.run_h3_cell``'s NOA branch for an iterative (non-linear-head) pathway: target = the
    full model's normalized head outputs, MSE train loss, MSE validation criterion. There is no
    test argument and no ground-truth argument. Contract 07: bitwise the H3 adapter."""
    if fit_cfg.get("noa_adapter", "affine") != "affine":
        raise ValueError("Phase 2e uses the committed H3 affine adapter only")
    device = device or pathway.device

    def T(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    return fit_residual_adapter(
        d=pathway.d, pathway_fn=pathway.outputs, train_x=T(train_states), train_y=T(teacher_train),
        val_x=T(val_states), val_y=T(teacher_val), train_loss=mse_loss, val_criterion=mse_loss,
        wd_grid=fit_cfg["wd_grid"], epochs=fit_cfg["epochs"], lr=fit_cfg["lr"],
        eval_every=fit_cfg["eval_every"], device=device, seed=fit_cfg["seed"], log=log,
        label=label)


def fit_flags(fit: dict) -> dict:
    return {"selected_hard_cut": bool(fit.get("selected_hard_cut")),
            "late_checkpoint": bool(fit.get("selected_epoch") is not None and fit.get("epochs")
                                    and fit["selected_epoch"] >= LATE_EPOCH_FRACTION
                                    * fit["epochs"]),
            "grid_max_wd": bool(fit.get("selected_wd") is not None and fit.get("grid")
                                and fit["selected_wd"] == max(fit["grid"]))}


# --------------------------------------------------------------------------- #
# statistics -- paired series-cluster bootstrap (one count matrix per dataset)
# --------------------------------------------------------------------------- #
def _points_and_reps(rows: dict, wql_den, cluster_ids, B: int, seed: int):
    """rows: {arm: {"loss","mase","mae","wql_num"}} -> points[metric][arm], reps[metric][arm].
    ``cluster_replicates`` is a pure function of (n_clusters, B, seed), so every call for one
    dataset resamples the SAME clusters: every comparison is paired, across calls and files."""
    names = list(rows)
    q = len(QUANTILES)
    points, reps = {}, {}
    den = np.asarray(wql_den, np.float64)
    for metric in METRICS:
        if metric == "wql":
            W = np.stack([np.asarray(rows[a]["wql_num"], np.float64) for a in names])
            num = cluster_replicates(W, cluster_ids, B, seed, reduce="sum")
            dd = cluster_replicates(den[None, :], cluster_ids, B, seed, reduce="sum")[:, 0]
            R = num / (q * dd[:, None])
            P = W.sum(axis=1) / (q * den.sum())
        else:
            W = np.stack([np.asarray(rows[a][metric], np.float64) for a in names])
            R = cluster_replicates(W, cluster_ids, B, seed, reduce="mean")
            P = W.mean(axis=1)
        points[metric] = {a: float(P[i]) for i, a in enumerate(names)}
        reps[metric] = {a: R[:, i] for i, a in enumerate(names)}
    return points, reps


def mask_cell_stats(pw: dict, wql_den, cluster_ids, *, B: int = BOOT_B,
                    seed: int = SEED) -> dict:
    """Absolute metrics + ratio vs the full model (with paired CIs) for native / hard / noa, the
    noa-vs-hard ratio, and the H3 gap closure of the alignment."""
    from probing.phase2_h3 import gap_closure_stats
    P, R = _points_and_reps(pw, wql_den, cluster_ids, B, seed)
    out = {"absolute": {a: {m: P[m][a] for m in METRICS} for a in pw},
           "vs_native": {a: {m: ratio_summary(P[m][a], P[m]["native"], R[m][a], R[m]["native"])
                             for m in METRICS} for a in pw if a != "native"},
           "noa_vs_hard": {}, "gap_closure": {}}
    if "noa" in pw and "hard" in pw:
        for m in METRICS:
            r = ratio_summary(P[m]["noa"], P[m]["hard"], R[m]["noa"], R[m]["hard"])
            r.pop("within_budget", None)
            r.pop("budget_fragile", None)
            out["noa_vs_hard"][m] = r
        for m in ("mase", "wql"):
            out["gap_closure"][m] = gap_closure_stats(P[m]["hard"], P[m]["noa"], P[m]["native"],
                                                      R[m]["hard"], R[m]["noa"], R[m]["native"])
    return out


def _paired_vs_prefix(P, R, m, arm_s, arm_p, nat="native") -> dict:
    """strategy vs prefix on one metric: ratio, and the DIFFERENCE OF DEGRADATIONS in percentage
    points of the full model, (M_s - M_p) / M_native, with paired percentile CIs."""
    r = ratio_summary(P[m][arm_s], P[m][arm_p], R[m][arm_s], R[m][arm_p])
    r.pop("within_budget", None)
    r.pop("budget_fragile", None)
    dd = 100.0 * (R[m][arm_s] - R[m][arm_p]) / R[m][nat]
    lo, hi = np.percentile(dd, [2.5, 97.5])
    diff = 100.0 * (P[m][arm_s] - P[m][arm_p]) / P[m][nat]
    return {**r, "diff_pp": float(diff), "diff_pp_ci": [float(lo), float(hi)],
            "better": bool(diff < 0), "better_ci": bool(hi < 0), "worse_ci": bool(lo > 0)}


def comparison_stats(per_config: dict, native: dict, wql_den, cluster_ids, *,
                     val_mase: dict | None = None, B: int = BOOT_B, seed: int = SEED) -> dict:
    """Per dataset: every strategy vs prefix at the same budget, for the hard and aligned arms.

    ``per_config[(strategy, k)] = {"hard": {...per-window test...}, "noa": {...}}``;
    ``native`` = the full model's per-window test arrays. ``val_mase[(strategy, k)][arm]`` = the
    mean VALIDATION MASE, used for the secondary, validation-only strategy choice (ties -> prefix).
    """
    rows = {"native": native}
    for (s, k), arms in per_config.items():
        for a, v in arms.items():
            rows[f"{s}|{k}|{a}"] = v
    P, R = _points_and_reps(rows, wql_den, cluster_ids, B, seed)
    out = {"by_budget": {}, "val_selected": {}}
    for k in BUDGETS:
        if (PREFIX, k) not in per_config:
            continue
        bk = {}
        for s in STRATEGIES:
            if (s, k) not in per_config:
                continue
            for a in ("hard", "noa"):
                key = f"{s}|{k}|{a}"
                if key not in rows:
                    continue
                vs_nat = {m: ratio_summary(P[m][key], P[m]["native"], R[m][key], R[m]["native"])
                          for m in ("mase", "wql")}
                rec = {"strategy": s, "budget": k, "arm": a, "vs_native": vs_nat}
                if s != PREFIX:
                    rec["vs_prefix"] = {m: _paired_vs_prefix(P, R, m, key, f"{PREFIX}|{k}|{a}")
                                        for m in ("mase", "wql")}
                bk[f"{s}|{a}"] = rec
        out["by_budget"][f"k{k}"] = bk
        if val_mase:                          # secondary: pick the strategy on VALIDATION only
            for a in ("hard", "noa"):
                cand = [(float(val_mase[(s, k)][a]), STRATEGIES.index(s), s) for s in STRATEGIES
                        if (s, k) in val_mase and a in val_mase[(s, k)]
                        and np.isfinite(val_mase[(s, k)][a])]
                if not cand:
                    continue
                _, _, s_sel = min(cand)
                key, pkey = f"{s_sel}|{k}|{a}", f"{PREFIX}|{k}|{a}"
                rec = {"selected": s_sel, "val_mase": {s: float(val_mase[(s, k)][a])
                                                      for _, _, s in cand},
                       "vs_native": ratio_summary(P["mase"][key], P["mase"]["native"],
                                                  R["mase"][key], R["mase"]["native"])}
                rec["vs_prefix"] = (None if s_sel == PREFIX
                                    else _paired_vs_prefix(P, R, "mase", key, pkey))
                out["val_selected"][f"k{k}|{a}"] = rec
    return out


def single_skip_stats(per_skip: dict, native: dict, wql_den, cluster_ids, *,
                      val: dict | None = None, B: int = BOOT_B, seed: int = SEED) -> list[dict]:
    """Per skipped block i: the hard-cut forecast of the 11-block backbone vs the full model."""
    rows = {"native": native, **{skip_label(i): v for i, v in per_skip.items()}}
    P, R = _points_and_reps(rows, wql_den, cluster_ids, B, seed)
    out = []
    for i in sorted(per_skip):
        key = skip_label(i)
        rec = {"skipped_block": int(i), "label": key}
        for m in ("mase", "wql", "loss"):
            s = ratio_summary(P[m][key], P[m]["native"], R[m][key], R[m]["native"])
            rec[f"{m}_ratio"] = s["ratio"]
            rec[f"{m}_degradation_pct"] = 100.0 * s["degradation"]
            rec[f"{m}_degradation_ci_pct"] = [100.0 * s["degradation_ci_lo"],
                                              100.0 * s["degradation_ci_hi"]]
        if val:
            rec["val_mase_ratio"] = float(val[i]["mase"] / val["native"]["mase"])
        out.append(rec)
    return out


def v3_check(physical_mase, offline_mase, rtol: float = V3_MEAN_RTOL) -> dict:
    """Physical (public API) vs offline per-window MASE. Pass = mean MASE agrees to ``rtol``
    relative (the V3 rule adopted 2026-09-26); the pre-specified element-wise check is recorded
    beside it (atol = rtol * mean|ref|, rtol), never dropped."""
    from probing.phase2_truncate import elementwise_gate
    a, b = np.asarray(physical_mase, np.float64), np.asarray(offline_mase, np.float64)
    if a.shape != b.shape:
        return {"passed": False, "reason": f"rows {a.shape} vs {b.shape}"}
    rel = abs(a.mean() - b.mean()) / max(abs(b.mean()), 1e-12)
    ew = elementwise_gate(a, b, atol=rtol * float(np.abs(b).mean()), rtol=rtol)
    return {"rule": "mean test MASE agrees to rtol (V3, option A)", "rtol": float(rtol),
            "mase_mean_physical": float(a.mean()), "mase_mean_offline": float(b.mean()),
            "mase_mean_rel_diff": float(rel), "elementwise": ew,
            "elementwise_passed": bool(ew["passed"]), "passed": bool(rel <= rtol)}


# --------------------------------------------------------------------------- #
# persistence of large arrays (states) -- atomic, verified on read
# --------------------------------------------------------------------------- #
def atomic_savez(path, **arrays) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


class StateCache:
    """Skipped-block forecast-slot states on $SCRATCH, one file per (dataset, mask, split).

    The key is everything that moves the bits: the mask, the window digest, a digest of THESE
    contexts, the checkpoint, the extraction batch size and the precision flags. A file whose
    metadata differs is a MISS (re-extracted), never silently reused. Written atomically, so a
    killed job leaves no half file. Never a Phase-1 cache: a different root, refused if equal.
    """

    def __init__(self, root):
        self.root = Path(root)

    def path(self, tag: str, mask, split: str) -> Path:
        return self.root / tag / f"mask_{mask_string(mask).replace(',', '_')}__{split}.npz"

    @staticmethod
    def meta(tag, mask, split, X, *, window_digest, checkpoint, batch_size, numerics) -> dict:
        return {"protocol": PHASE2E_PROTOCOL_VERSION, "dataset": tag, "mask": list(mask),
                "split": split, "n": int(len(X)), "contexts_sha256": array_sha256(X),
                "window_digest": window_digest, "checkpoint": checkpoint,
                "batch_size": int(batch_size),
                "numerics": {k: numerics.get(k) for k in ("matmul_allow_tf32",
                                                          "float32_matmul_precision",
                                                          "cudnn_allow_tf32")}}

    def read(self, tag, mask, split, meta: dict):
        p = self.path(tag, mask, split)
        if not p.exists():
            return None
        import json
        try:
            with np.load(p, allow_pickle=False) as z:
                got = json.loads(str(z["meta"]))
                if got != meta:
                    return None
                return np.asarray(z["states"], np.float32)
        except Exception:
            return None

    def write(self, tag, mask, split, meta: dict, states) -> Path:
        import json
        return atomic_savez(self.path(tag, mask, split), states=np.asarray(states, np.float32),
                            meta=np.array(json.dumps(meta, sort_keys=True)))

