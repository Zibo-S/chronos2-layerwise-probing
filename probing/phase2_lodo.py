"""Phase 2c -- pooled leave-one-dataset-out (LODO) transfer of the H3 label-free adapter.

    Can ONE affine adapter, fitted across heterogeneous source datasets, align Chronos-2's
    intermediate forecast-slot states well enough to truncate on an UNSEEN dataset with no
    target-domain calibration?

One cell = (variant, target t, depth l), l in {L4, L6, L9}:

    sources   every paper-14 dataset except t. In the MAIN protocol (``sibling_excluded``) t's
              series sibling is excluded too: Electricity hourly and Electricity 15-minute share
              321 meters, so pooling one while testing the other would put the target's own series
              into the fit. ``sibling_included`` (all 13 others) exists ONLY for those two targets,
              as a sensitivity analysis -- never part of the unseen-series result.
    fit       the H3 label-free NOA fit, UNCHANGED (``phase2_align.fit_residual_adapter``): residual
              affine adapter a(h) = h + Delta h + b from zero, AdamW lr 1e-3, 1500 FULL-BATCH
              epochs, decoupled decay on Delta only over the H3 grid, validation every 25 epochs,
              epoch 0 (the hard cut) a candidate, the same tie-breaking. Only the data differ: the
              objective and the selection criterion are the EQUAL-WEIGHTED mean over sources of
              each source's mean squared error to the full model's normalized head outputs.
    test      the frozen adapter on t's TEST split, read once after selection, beside the full
              model, the hard cut and the committed dataset-specific H3 adapter (re-scored and
              gated against the H3 per-window numbers).

What never reaches the fit: t's train/val splits (contract: replacing them by noise leaves the
adapter bitwise unchanged), any test split, and any ground-truth future (only states and the full
model's own outputs are passed). The depth is FIXED; choosing a truncation depth on an unseen
dataset is a separate question this experiment does not answer.
"""

from __future__ import annotations

import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from probing import phase1
from probing.phase1_metrics import raw_window_metrics
from probing.phase2 import (BOOT_B, MEDIAN_INDEX, QUANTILES, SEED, Phase1DependencyError,
                            array_sha256, cluster_replicates, ratio_summary)
from probing.phase2_align import ResidualAdapter, fit_residual_adapter
from probing.phase2_h3 import DEFAULT_FIT_CFG, gap_closure_stats, native_gate

__all__ = ["LODO_PROTOCOL_VERSION", "KIND", "MODEL", "LODO_DEPTHS", "MAIN", "SENSITIVITY",
           "VARIANTS", "SERIES_SIBLINGS", "CHRONOS2_PROVENANCE", "PROVENANCE_GROUPS",
           "LODO_REQUIRED", "ARMS", "fold_sources", "plan_cells", "chunk", "depth_label",
           "PreparedDataset", "prepare_dataset", "pooled_split", "equal_weight_mse",
           "per_source_mse", "fit_pooled_adapter", "score_arm", "load_h3_reference",
           "gap_sign", "cell_statistics", "PREFLIGHT_GATES", "preflight_gates", "run_lodo_cell",
           "cell_config", "save_lodo_cell"]

LODO_PROTOCOL_VERSION = "phase2c/lodo-v1"
KIND = "lodo"
MODEL = "chronos2"
LODO_DEPTHS = (4, 6, 9)
MAIN, SENSITIVITY = "sibling_excluded", "sibling_included"
VARIANTS = (MAIN, SENSITIVITY)
OBJECTIVE = "noa_equal_weight_mse/v1"
#: Datasets that share SERIES (not just a domain): UCI ElectricityLoadDiagrams (Trindade 2015).
#: The hourly set is the 321-client LSTNet subset of the 370 clients in the 15-minute set.
SERIES_SIBLINGS = {"monash_electricity_hourly": ("electricity_15min",),
                   "electricity_15min": ("monash_electricity_hourly",)}
#: The H3 reproduction gates compare the SAME code path on the same rows, so they are tight.
H3_REPRO_RTOL = 1e-6
#: A selected checkpoint at or beyond this fraction of the epochs is flagged as possibly unconverged
#: (the convergence rule of the 2026-09-23 optimizer check).
LATE_EPOCH_FRACTION = 0.9
ARMS = ("native", "hard", "pooled", "specific")
METRICS = ("loss", "mase", "mae", "wql")

#: Chronos-2 column of the paper's provenance table (notes/help.tex, tab:provenance), as
#: (status, evidence scope). Transcribed from the PAPER, not from probing/registry.py, whose
#: Chronos-2 cells predate the 2026-09-25 primary-source re-derivation (notes/PLAN.md).
#: "not_listed" means only that the documentation does not identify the dataset as training data.
CHRONOS2_PROVENANCE = {
    "m4_hourly": ("listed", "exact"),
    "monash_electricity_hourly": ("listed", "exact"),
    "uber_tlc_hourly": ("listed", "source"),
    "wind_farms_hourly": ("listed", "exact"),
    "boom_hourly": ("not_listed", "exact"),        # fev-bench has BOOM-derived tasks; overlap unknown
    "sg_carpark": ("not_listed", "exact"),
    "coastal_ts": ("not_listed", "exact"),
    "LOOP_SEATTLE_5T": ("held_out", "exact"),      # fev-bench
    "electricity_15min": ("listed", "exact"),
    "SZ_TAXI_15T": ("held_out", "exact"),          # fev-bench
    "monash_london_smart_meters": ("listed", "exact"),
    "m5": ("held_out", "exact"),                   # Chronos Benchmark II + fev-bench
    "wiki_daily_100k": ("listed", "exact"),
    "monash_traffic": ("held_out", "exact"),       # Chronos Benchmark II
}
PROVENANCE_GROUPS = ("held_out", "not_listed", "listed")

LODO_REQUIRED = ("cell_config.json", "fold.json", "fit.json", "per_source_val.csv",
                 "metrics.json", "per_window.npz", "verification.json")


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #
def fold_sources(target: str, roster, variant: str = MAIN) -> list[str]:
    """The source datasets of one fold, in roster order (the pooled data order)."""
    roster = list(roster)
    if target not in roster:
        raise ValueError(f"target {target!r} is not in the roster")
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; known: {VARIANTS}")
    if variant == SENSITIVITY and target not in SERIES_SIBLINGS:
        raise ValueError(f"{SENSITIVITY!r} exists only for targets with a series sibling "
                         f"({sorted(SERIES_SIBLINGS)}); for {target!r} it equals {MAIN!r}")
    excluded = {target} | (set(SERIES_SIBLINGS.get(target, ())) if variant == MAIN else set())
    return [t for t in roster if t not in excluded]


def plan_cells(roster, *, depths=LODO_DEPTHS, targets=None, variants=VARIANTS) -> list[dict]:
    """Every (variant, target, depth) cell. The sensitivity variant only exists for the targets
    with a series sibling; for every other target it would duplicate the main cell."""
    roster = list(roster)
    cells = []
    for v in variants:
        for t in roster:
            if targets and t not in targets:
                continue
            if v == SENSITIVITY and t not in SERIES_SIBLINGS:
                continue
            src = fold_sources(t, roster, v)
            for l in depths:
                cells.append({"variant": v, "target": t, "depth": int(l), "sources": src,
                              "excluded": [x for x in roster if x not in src]})
    return cells


def chunk(items, spec: str | None) -> list:
    """``"i/n"`` -> the i-th of n contiguous, near-equal chunks (1-based); None -> everything."""
    items = list(items)
    if not spec:
        return items
    i, n = (int(x) for x in spec.split("/"))
    if not 1 <= i <= n:
        raise ValueError(f"chunk {spec!r}: need 1 <= i <= n")
    bounds = np.linspace(0, len(items), n + 1).round().astype(int)
    return items[bounds[i - 1]:bounds[i]]


def depth_label(depth: int) -> str:
    return phase1.model_spec(MODEL).depth_labels[int(depth)]


# --------------------------------------------------------------------------- #
# data: the arrays a pooled fit may read, kept apart from the test split
# --------------------------------------------------------------------------- #
@dataclass
class PreparedDataset:
    tag: str
    train_states: dict              # depth -> (n, K, d) float32
    val_states: dict
    train_teacher: np.ndarray       # (n, K, n_out) float32: f(h_L), the full model's own outputs
    val_teacher: np.ndarray
    test: object                    # phase2_pathways.SplitData, feats pruned to {depths, L}
    reference_index: int
    fingerprints: dict              # sha256 of everything a fit may read, per split


def prepare_dataset(pathway, data, depths) -> PreparedDataset:
    """Teacher outputs exactly as H3 computes them (``pathway.run`` on the final block, the same
    batching), and only the depths this job needs."""
    L = data.reference_index
    tr, va, te = data.splits["train"], data.splits["val"], data.splits["test"]
    keep = sorted({int(l) for l in depths})
    t_out, _ = pathway.run(tr.feats[L])
    v_out, _ = pathway.run(va.feats[L])
    test = dataclasses.replace(te, feats={l: te.feats[l] for l in sorted(set(keep) | {L})})
    trs = {l: tr.feats[l] for l in keep}
    vas = {l: va.feats[l] for l in keep}
    fp = {"train": array_sha256(*(trs[l] for l in keep), t_out),
          "val": array_sha256(*(vas[l] for l in keep), v_out)}
    return PreparedDataset(data.tag, trs, vas, t_out, v_out, test, L, fp)


def pooled_split(prepared: dict, sources, split: str, depth: int):
    """Concatenate the sources' states and teacher outputs in source order.
    Returns (x, y, segments) with segments = [(tag, start, stop), ...]."""
    if split not in ("train", "val"):
        raise ValueError(f"a pooled fit reads train or val only, not {split!r}")
    xs, ys, segs, s = [], [], [], 0
    for t in sources:
        p = prepared[t]
        x = (p.train_states if split == "train" else p.val_states)[int(depth)]
        y = p.train_teacher if split == "train" else p.val_teacher
        if len(x) != len(y):
            raise RuntimeError(f"{t}/{split}: {len(x)} states vs {len(y)} teacher rows")
        xs.append(x)
        ys.append(y)
        segs.append((t, s, s + len(x)))
        s += len(x)
    return np.concatenate(xs), np.concatenate(ys), segs


def equal_weight_mse(segments):
    """mean over sources of [mean over that source's rows of (pred - target)^2]: every source
    weighs 1/|S| whatever its window count. With equal counts this IS the plain pooled mean
    (``phase2_align.mse_loss``), the H3 objective (contract 2)."""
    bounds = [(int(a), int(b)) for _, a, b in segments]
    if not bounds or any(b <= a for a, b in bounds):
        raise ValueError(f"empty source segment in {segments}")

    def crit(pred, target):
        return torch.stack([((pred[a:b] - target[a:b]) ** 2).mean() for a, b in bounds]).mean()
    return crit


def per_source_mse(pathway, x, y, segments, adapter=None) -> dict:
    """Diagnostic only: each source's mean squared error at a fixed adapter (float64 means)."""
    out, _ = pathway.run(x, adapter)
    return {t: float(((out[a:b].astype(np.float64) - y[a:b].astype(np.float64)) ** 2).mean())
            for t, a, b in segments}


def fit_pooled_adapter(pathway, prepared: dict, sources, depth: int, *, fit_cfg=None,
                       device=None, log=print) -> dict:
    """The H3 NOA fit on the pooled sources. ``prepared`` may hold the target too: only
    ``sources`` are read (contract 3)."""
    cfg = {**DEFAULT_FIT_CFG, **(fit_cfg or {})}
    device = device or pathway.device
    xtr, ytr, seg_tr = pooled_split(prepared, sources, "train", depth)
    xva, yva, seg_va = pooled_split(prepared, sources, "val", depth)

    def T(a):
        return torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=device)
    t0 = time.time()
    res = fit_residual_adapter(
        d=pathway.d, pathway_fn=pathway.outputs, train_x=T(xtr), train_y=T(ytr),
        val_x=T(xva), val_y=T(yva), train_loss=equal_weight_mse(seg_tr),
        val_criterion=equal_weight_mse(seg_va), wd_grid=cfg["wd_grid"], epochs=cfg["epochs"],
        lr=cfg["lr"], eval_every=cfg["eval_every"], device=device, seed=cfg["seed"], log=log,
        label=f"lodo {depth_label(depth)}")
    fit_s = time.time() - t0
    ad = res.pop("adapter")
    hard = per_source_mse(pathway, xva, yva, seg_va, None)
    sel = per_source_mse(pathway, xva, yva, seg_va, ad)
    sh, ss = sum(hard.values()), sum(sel.values())
    rows = [{"source": t, "n_train": b - a, "n_val": vb - va_,
             "val_mse_hard_cut": hard[t], "val_mse_selected": sel[t],
             "share_hard_cut": hard[t] / sh if sh > 0 else float("nan"),
             "share_selected": sel[t] / ss if ss > 0 else float("nan"),
             "relative_reduction": 1.0 - sel[t] / hard[t] if hard[t] > 0 else float("nan")}
            for (t, a, b), (_, va_, vb) in zip(seg_tr, seg_va)]
    counts = [r["n_train"] for r in rows]
    flags = {"selected_hard_cut": bool(res["selected_hard_cut"]),
             "late_checkpoint": bool(res["selected_epoch"] >= LATE_EPOCH_FRACTION * res["epochs"]),
             "grid_max_wd": bool(res["selected_wd"] is not None
                                 and res["selected_wd"] == max(res["grid"]))}
    return {"adapter": ad, "fit": res, "per_source_val": rows, "flags": flags,
            "equal_train_counts": len(set(counts)) == 1, "n_train_rows": int(len(xtr)),
            "n_val_rows": int(len(xva)), "fit_seconds": round(fit_s, 1),
            "criterion_check": {"equal_weight_mean_of_per_source_hard": float(np.mean(
                list(hard.values()))), "fit_hard_cut_val": float(res["hard_cut_val"])}}


# --------------------------------------------------------------------------- #
# evaluation on the target's test split
# --------------------------------------------------------------------------- #
def score_arm(pathway, te, tag: str, feats, adapter=None) -> dict:
    """Per-window test metrics of one arm -- the exact H3 ``score`` call chain."""
    _, z = pathway.run(feats, adapter)
    m = raw_window_metrics(tag, te.X, te.y_raw, pathway.to_raw(z, te), QUANTILES, MEDIAN_INDEX)
    return {"loss": phase1.mean_pinball_per_window(z, te.target, QUANTILES),
            "mase": m["mase_pw"], "mae": m["mae_pw"], "wql_num": m["wql_num_pw"],
            "wql_den": m["wql_den_pw"]}


def load_h3_reference(h3_root, tag: str) -> dict:
    """The committed dataset-specific H3 cell: its config hash, per-window test arrays and fits.
    READ-ONLY; a missing or incomplete cell is an error."""
    import json
    d = Path(h3_root) / "h3" / MODEL / tag
    mk = d / "COMPLETE"
    if not mk.exists():
        raise Phase1DependencyError(f"no COMPLETE dataset-specific H3 cell at {d}")
    marker = json.loads(mk.read_text())
    with np.load(d / "bootstrap_inputs.npz", allow_pickle=True) as z:
        arrays = {k: z[k] for k in z.files}
    fits = json.loads((d / "fits.json").read_text())
    return {"dir": str(d), "config_hash": marker.get("config_hash"), "arrays": arrays,
            "fits": fits}


def _repro(ours, ref, rtol: float) -> dict:
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    if ours.shape != ref.shape:
        return {"passed": False, "reason": f"rows {ours.shape} vs {ref.shape}"}
    rel = abs(ours.mean() - ref.mean()) / max(abs(ref.mean()), 1e-12)
    return {"mean_ours": float(ours.mean()), "mean_ref": float(ref.mean()),
            "mean_rel_diff": float(rel), "window_max_abs_diff": float(np.abs(ours - ref).max()),
            "bitwise": bool(np.array_equal(ours, ref)), "rtol": rtol, "passed": bool(rel <= rtol)}


def gap_sign(hard_gap_ci) -> str:
    """Direction of the hard-cut MASE gap, from its paired 95% CI."""
    lo, hi = hard_gap_ci
    return "positive" if lo > 0 else ("negative" if hi < 0 else "indistinguishable")


def cell_statistics(pw: dict, wql_den, cluster_ids, *, B: int = BOOT_B, seed: int = SEED) -> dict:
    """Paired series-level cluster bootstrap over the four arms (ONE count matrix, as in H3)."""
    q = len(QUANTILES)
    idx = {a: i for i, a in enumerate(ARMS)}
    points, reps = {}, {}
    for metric in METRICS:
        if metric == "wql":
            W = np.stack([pw[a]["wql_num"] for a in ARMS])
            num = cluster_replicates(W, cluster_ids, B, seed, reduce="sum")
            den = cluster_replicates(np.asarray(wql_den)[None, :], cluster_ids, B, seed,
                                     reduce="sum")[:, 0]
            reps[metric] = num / (q * den[:, None])
            points[metric] = W.sum(axis=1) / (q * np.sum(wql_den))
        else:
            W = np.stack([pw[a][metric] for a in ARMS])
            reps[metric] = cluster_replicates(W, cluster_ids, B, seed, reduce="mean")
            points[metric] = W.mean(axis=1)

    def P(m, a):
        return float(points[m][idx[a]])

    def R(m, a):
        return reps[m][:, idx[a]]

    out = {"absolute": {a: {m: P(m, a) for m in METRICS} for a in ARMS},
           "vs_native": {a: {m: ratio_summary(P(m, a), P(m, "native"), R(m, a), R(m, "native"))
                             for m in METRICS} for a in ("hard", "pooled", "specific")},
           "pooled_vs_hard": {}, "pooled_vs_specific": {}, "gap": {}}
    for m in METRICS:
        r = ratio_summary(P(m, "pooled"), P(m, "hard"), R(m, "pooled"), R(m, "hard"))
        r["improves"] = bool(r["ratio"] < 1.0)
        r["improves_ci"] = bool(r["ratio_ci_hi"] < 1.0)
        out["pooled_vs_hard"][m] = r
        s = ratio_summary(P(m, "pooled"), P(m, "specific"), R(m, "pooled"), R(m, "specific"))
        # the difference of the two degradations vs the full model, in percentage points
        dd = (R(m, "pooled") - R(m, "specific")) / R(m, "native")
        lo, hi = np.percentile(dd, [2.5, 97.5])
        s["degradation_difference_pp"] = 100 * (P(m, "pooled") - P(m, "specific")) / P(m, "native")
        s["degradation_difference_pp_ci"] = [float(100 * lo), float(100 * hi)]
        out["pooled_vs_specific"][m] = s
    for m in ("mase", "wql"):
        g = {}
        for arm in ("pooled", "specific"):
            gc = gap_closure_stats(P(m, "hard"), P(m, arm), P(m, "native"), R(m, "hard"),
                                   R(m, arm), R(m, "native"))
            sign = gap_sign(gc["hard_gap_ci"])
            # a closure is only REPORTED when there is a gap to close: the hard cut must be
            # significantly WORSE than the full model. H3's ``valid`` also admits a significantly
            # negative gap, where "recovering the degradation" has no meaning.
            gc["gap_sign"] = sign
            gc["reported"] = gc["gap_closure"] if sign == "positive" else None
            gc["note"] = {"positive": "hard cut significantly worse than the full model",
                          "negative": "hard cut significantly BETTER than the full model: "
                                      "no degradation to recover",
                          "indistinguishable": "hard-cut gap CI contains 0: nothing measurable "
                                               "to recover"}[sign]
            g[arm] = gc
        out["gap"][m] = g
    return out


#: the gates of ``run_lodo_cell`` that do not need the pooled fit (``preflight_gates`` runs them)
PREFLIGHT_GATES = ("native_gate_vs_phase1", "native_vs_h3", "hard_vs_h3", "specific_vs_h3",
                   "test_rows_vs_h3", "identity_adapter_equals_hard_cut")


def preflight_gates(pathway, prepared: dict, target: str, depth: int, *, phase1_arrays: dict,
                    h3_ref: dict, specific_adapter, fit_cfg=None, device=None,
                    h3_rtol: float = H3_REPRO_RTOL) -> dict:
    """The reproduction gates of ``run_lodo_cell`` for one (target, depth), WITHOUT a pooled fit.

    Same functions, same arrays, same tolerances as the cell: the full model vs Phase 1 and vs H3,
    the hard cut and the committed dataset-specific adapter vs H3, the test rows, and identity
    adapter == hard cut. The driver runs it for every pending (target, depth) BEFORE the first fit,
    so a systematic mismatch stops a job in minutes instead of after hours of fitting. The cell
    still re-runs every gate (that copy is authoritative). It reads no pooled adapter and no
    train/val array, so it cannot influence any fit or selection."""
    cfg = {**DEFAULT_FIT_CFG, **(fit_cfg or {})}
    l = int(depth)
    device = device or pathway.device
    pathway.assert_frozen()
    te = prepared[target].test
    L = prepared[target].reference_index
    pw = {"native": score_arm(pathway, te, target, te.feats[L]),
          "hard": score_arm(pathway, te, target, te.feats[l]),
          "specific": score_arm(pathway, te, target, te.feats[l], specific_adapter)}
    ref = h3_ref["arrays"]
    ver = {"native_gate_vs_phase1": native_gate(pw["native"]["loss"], pw["native"]["mase"],
                                                phase1_arrays, cfg["native_gate_rtol"]),
           "native_vs_h3": _repro(pw["native"]["mase"], ref["native_test_mase"], h3_rtol),
           "hard_vs_h3": _repro(pw["hard"]["mase"], ref["hard__test_mase"][l], h3_rtol),
           "specific_vs_h3": _repro(pw["specific"]["mase"], ref["noa__test_mase"][l], h3_rtol),
           "test_rows_vs_h3": {"passed": bool(np.array_equal(np.asarray(te.cluster_ids),
                                                             ref["cluster_ids_test"]))}}
    _, z_id = pathway.run(te.feats[l], ResidualAdapter(pathway.d).to(device).eval())
    _, z_h = pathway.run(te.feats[l], None)
    ver["identity_adapter_equals_hard_cut"] = {"passed": bool(np.array_equal(z_id, z_h))}
    failed = [k for k in PREFLIGHT_GATES if ver[k].get("passed") is False]
    return {"target": target, "depth": l, "label": depth_label(l), "gates": ver,
            "failed": failed, "all_passed": not failed,
            "mean_test_mase": {a: float(np.mean(v["mase"])) for a, v in pw.items()}}


def run_lodo_cell(pathway, prepared: dict, cell: dict, *, phase1_arrays: dict, h3_ref: dict,
                  specific_adapter, fit_cfg=None, device=None, h3_rtol: float = H3_REPRO_RTOL,
                  log=print) -> dict:
    """Fit on the sources, then read the target's test split ONCE. Raises (no cell is written)
    if any reproduction gate fails."""
    cfg = {**DEFAULT_FIT_CFG, **(fit_cfg or {})}
    target, l, variant = cell["target"], int(cell["depth"]), cell["variant"]
    sources = list(cell["sources"])
    lab = depth_label(l)
    device = device or pathway.device
    t_cell = time.time()
    pathway.assert_frozen()

    # ---------------------------------------------------------------- isolation (structural)
    if target in sources:
        raise RuntimeError(f"{target} is among its own sources")
    if variant == MAIN and set(SERIES_SIBLINGS.get(target, ())) & set(sources):
        raise RuntimeError(f"{target}: its series sibling is a source under {MAIN}")
    tgt_fp = prepared[target].fingerprints
    clash = [t for t in sources if set(prepared[t].fingerprints.values())
             & set(tgt_fp.values())]
    if clash:
        raise RuntimeError(f"{target}: train/val arrays identical to source(s) {clash}")

    # ---------------------------------------------------------------- fit (sources only)
    fit = fit_pooled_adapter(pathway, prepared, sources, l, fit_cfg=cfg, device=device, log=log)
    ad = fit.pop("adapter")

    # ---------------------------------------------------------------- test, read once
    t_eval = time.time()
    te = prepared[target].test
    L = prepared[target].reference_index
    pw = {"native": score_arm(pathway, te, target, te.feats[L]),
          "hard": score_arm(pathway, te, target, te.feats[l]),
          "pooled": score_arm(pathway, te, target, te.feats[l], ad),
          "specific": score_arm(pathway, te, target, te.feats[l], specific_adapter)}
    wql_den = pw["native"]["wql_den"]

    # ---------------------------------------------------------------- gates
    ref = h3_ref["arrays"]
    ver = {"native_gate_vs_phase1": native_gate(pw["native"]["loss"], pw["native"]["mase"],
                                                phase1_arrays, cfg["native_gate_rtol"]),
           "native_vs_h3": _repro(pw["native"]["mase"], ref["native_test_mase"], h3_rtol),
           "hard_vs_h3": _repro(pw["hard"]["mase"], ref["hard__test_mase"][l], h3_rtol),
           "specific_vs_h3": _repro(pw["specific"]["mase"], ref["noa__test_mase"][l], h3_rtol),
           "test_rows_vs_h3": {"passed": bool(np.array_equal(np.asarray(te.cluster_ids),
                                                             ref["cluster_ids_test"]))}}
    _, z_id = pathway.run(te.feats[l], ResidualAdapter(pathway.d).to(device).eval())
    _, z_h = pathway.run(te.feats[l], None)
    ver["identity_adapter_equals_hard_cut"] = {"passed": bool(np.array_equal(z_id, z_h))}
    ver["target_isolation"] = {
        "passed": True, "target": target, "sources": sources, "excluded": cell["excluded"],
        "target_fingerprints": tgt_fp,
        "source_fingerprints": {t: prepared[t].fingerprints for t in sources},
        "note": "the pooled tensors are built from `sources` only; the target's train/val "
                "fingerprints match no source block"}
    ver["equal_train_counts"] = {"passed": bool(fit["equal_train_counts"]),
                                 "counts": {r["source"]: r["n_train"]
                                            for r in fit["per_source_val"]},
                                 "note": "informational: the criterion weighs sources equally "
                                         "whatever the counts"}
    gated = ("native_gate_vs_phase1", "native_vs_h3", "hard_vs_h3", "specific_vs_h3",
             "test_rows_vs_h3", "identity_adapter_equals_hard_cut")
    failed = [k for k in gated if ver[k].get("passed") is False]
    ver["all_passed"] = not failed
    if failed:
        raise RuntimeError(f"{variant}/{target}/{lab}: reproduction gate(s) failed: "
                           f"{ {k: ver[k] for k in failed} }")

    stats = cell_statistics(pw, wql_den, te.cluster_ids, B=int(cfg["boot_b"]),
                            seed=int(cfg["boot_seed"]))
    timings = {"fit_s": fit["fit_seconds"], "eval_s": round(time.time() - t_eval, 1),
               "cell_total_s": round(time.time() - t_cell, 1)}
    if torch.cuda.is_available():
        timings["cuda_max_memory_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    return {"cell": cell, "label": lab, "adapter": ad, "fit": fit, "per_window": pw,
            "verification": ver, "stats": stats, "timings": timings, "fit_cfg": cfg,
            "test_cluster_ids": np.asarray(te.cluster_ids), "test_rows": np.asarray(te.rows)}


# --------------------------------------------------------------------------- #
# identity + persistence
# --------------------------------------------------------------------------- #
def cell_config(cell: dict, deps: dict, h3_ref: dict, fit_cfg: dict,
                h3_rtol: float = H3_REPRO_RTOL) -> dict:
    """Everything that would change the adapter or a reported number."""
    tags = list(cell["sources"]) + [cell["target"]]
    return {"protocol": LODO_PROTOCOL_VERSION, "kind": KIND, "model": MODEL,
            "objective": OBJECTIVE, "variant": cell["variant"], "target": cell["target"],
            "depth": int(cell["depth"]), "depth_label": depth_label(cell["depth"]),
            "sources": list(cell["sources"]), "excluded": list(cell["excluded"]),
            "dependencies": {t: {"phase1_fit_hash": deps[t]["fit_hash"],
                                 "window_digest": deps[t]["window_digest"]} for t in tags},
            "h3_reference_config_hash": h3_ref["config_hash"],
            "epochs": fit_cfg["epochs"], "lr": fit_cfg["lr"], "wd_grid": list(fit_cfg["wd_grid"]),
            "eval_every": fit_cfg["eval_every"], "seed": fit_cfg["seed"],
            "native_gate_rtol": fit_cfg["native_gate_rtol"], "h3_repro_rtol": h3_rtol,
            "quantiles": [float(x) for x in QUANTILES], "boot_b": fit_cfg["boot_b"],
            "boot_seed": fit_cfg["boot_seed"]}


def save_lodo_cell(stage, res: dict, ccfg: dict, config_hash: str, adapter_record: dict | None,
                   specific_record: dict, environment: dict | None = None) -> dict:
    """Write every required artifact into ``stage`` (a Phase2CellStore staging dir)."""
    from probing.phase2 import atomic_write_json, write_csv
    stage = Path(stage)
    cell, fit = res["cell"], res["fit"]
    tgt = cell["target"]
    status, scope = CHRONOS2_PROVENANCE.get(tgt, ("unknown", "unknown"))
    atomic_write_json(stage / "cell_config.json", {**ccfg, "config_hash": config_hash})
    atomic_write_json(stage / "fold.json", {
        "variant": cell["variant"], "target": tgt, "depth": cell["depth"], "label": res["label"],
        "sources": cell["sources"], "excluded": cell["excluded"],
        "n_sources": len(cell["sources"]),
        "series_siblings_of_target": list(SERIES_SIBLINGS.get(tgt, ())),
        "target_series_in_adapter_training": bool(
            cell["variant"] == SENSITIVITY and set(SERIES_SIBLINGS.get(tgt, ()))
            & set(cell["sources"])),
        "n_train_rows": fit["n_train_rows"], "n_val_rows": fit["n_val_rows"],
        "equal_train_counts": fit["equal_train_counts"],
        "chronos2_pretraining_provenance": {"status": status, "scope": scope,
                                            "source": "notes/help.tex tab:provenance"}})
    atomic_write_json(stage / "fit.json", {
        **fit["fit"], "flags": fit["flags"], "criterion_check": fit["criterion_check"],
        "fit_seconds": fit["fit_seconds"], "objective": OBJECTIVE,
        "adapter": adapter_record, "dataset_specific_adapter": specific_record})
    write_csv(stage / "per_source_val.csv", fit["per_source_val"])
    atomic_write_json(stage / "verification.json", res["verification"])
    pw = {f"{a}__{k}": np.asarray(v, np.float64) for a, d in res["per_window"].items()
          for k, v in d.items() if k != "wql_den"}
    np.savez_compressed(stage / "per_window.npz", **pw,
                        wql_den=np.asarray(res["per_window"]["native"]["wql_den"], np.float64),
                        cluster_ids_test=res["test_cluster_ids"], rows_test=res["test_rows"])
    summary = {"schema": "phase2c_lodo_cell/v1", "protocol": LODO_PROTOCOL_VERSION,
               "variant": cell["variant"], "target": tgt, "depth": cell["depth"],
               "label": res["label"], "n_sources": len(cell["sources"]),
               "chronos2_pretraining_provenance": {"status": status, "scope": scope},
               "stats": res["stats"], "flags": fit["flags"],
               "selected_wd": fit["fit"]["selected_wd"],
               "selected_epoch": fit["fit"]["selected_epoch"],
               "selected_val": fit["fit"]["selected_val"],
               "hard_cut_val": fit["fit"]["hard_cut_val"],
               "timings": res["timings"], "fit_cfg": res["fit_cfg"],
               "verification_all_passed": res["verification"]["all_passed"],
               "environment": environment or {}}
    atomic_write_json(stage / "metrics.json", summary)
    return summary
