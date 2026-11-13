"""Phase 2f tables, figures, statistics and macros, rebuilt from the cell artifacts ONLY.

    python -m experiments.make_phase2f_tables          # seconds (small paired bootstraps)

Reads   results/three_model_phase2f_source_scaling/source_scaling/<target>/<cell>/{COMPLETE,
        metrics.json, fold.json, selection.json, per_window.npz}
        (read-only, optional) the Phase 2c main cell of each target at L6, as a cross-check of the
        N = 13 cell, and the measured Phase-2 speedup at L6
Writes  results/three_model_phase2f_source_scaling/combined/   CSV + JSON
        three_models_paper/figures/source_scaling/            .tex tables, .pdf/.png figures, macros

Statistics. Every per-cell number (MASE change vs the full model, pooled minus dataset-specific,
gap recovery) is the Phase 2c paired series-cluster bootstrap computed inside the cell. Changes
BETWEEN two source pools of the same target are paired the same way here (same test rows, same
cluster resampling). The three source orders are repeated source-set choices within a target, not
extra targets: aggregates average the orders within a target first, and the across-target interval
resamples targets, then orders within a target (descriptive, four targets).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2, registry                                      # noqa: E402
from probing import phase2_lodo as lodo                                           # noqa: E402
from probing import phase2f_source_scaling as ss                                  # noqa: E402
from experiments.make_phase2_lodo_tables import _csv, flat as lodo_flat, load_speedups  # noqa: E402
from experiments.make_phase2_paper_tables import (RC, TEXTWIDTH_IN, _dname, _num,    # noqa: E402
                                                  _pct, _save, _write)

DEFAULT_ROOT = REPO_ROOT / "results" / "three_model_phase2f_source_scaling"
DEFAULT_PAPER_DIR = REPO_ROOT / "three_models_paper" / "figures" / "source_scaling"
DEFAULT_LATENCY = REPO_ROOT / "results" / "three_model_phase2" / "combined" / "latency_lookup.csv"
DEFAULT_LODO_ROOT = REPO_ROOT / "results" / "three_model_phase2_lodo"
POOL = 13
STEPS = ((1, 4), (4, 8), (8, 13))
TARGET_COLORS = {"LOOP_SEATTLE_5T": "#0072B2", "monash_london_smart_meters": "#D55E00",
                 "boom_hourly": "#009E73", "m5": "#CC79A7"}
SEED_STYLE = {0: "-", 1: (0, (4, 1.5)), 2: (0, (1.2, 1.2))}
PROV = {"held_out": "held out", "not_listed": "not listed", "listed": "listed"}


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def load_cells(root: Path) -> list[dict]:
    """Every COMPLETE cell. A cell whose Phase 2c gates or whose fit reproduction failed is
    refused (it can never be tabulated)."""
    base = Path(root) / ss.KIND
    out = []
    for d in sorted(base.glob("*/N*")):
        if not d.is_dir() or ".building-" in d.name or not (d / "COMPLETE").exists():
            continue
        m = json.loads((d / "metrics.json").read_text())
        sel = json.loads((d / "selection.json").read_text())
        if not m.get("verification_all_passed"):
            raise RuntimeError(f"{d}: a Phase 2c reproduction gate failed; refusing to tabulate")
        if sel.get("fit_reproduction") is not None and not sel["fit_reproduction"]["passed"]:
            raise RuntimeError(f"{d}: the N = 1 fit did not reproduce H3; refusing to tabulate")
        out.append({"dir": str(d), "metrics": m, "selection": sel,
                    "fold": json.loads((d / "fold.json").read_text())})
    return out


def per_window(cell: dict) -> dict:
    with np.load(Path(cell["dir"]) / "per_window.npz") as z:
        return {k: z[k] for k in z.files}


def flat(cell: dict) -> dict:
    r = lodo_flat(cell)
    sel = cell["selection"]
    w = sel["training_work"]
    rep = sel.get("fit_reproduction")
    r.update({
        "target": sel["target"], "cell_id": sel["cell_id"], "n_sources": sel["n_sources"],
        "order_seed": "all" if sel["order_seed"] is None else int(sel["order_seed"]),
        "selection_order": ";".join(sel["selection_order"]),
        "sources": ";".join(sel["sources_fit_order"]), "n_families": sel["n_families"],
        "both_series_siblings": sel["both_series_siblings"],
        "n_train_windows": w["n_train_windows"], "n_val_windows": w["n_val_windows"],
        "n_train_slot_states": w["n_train_slot_states"],
        "steps_completed": w["steps_completed"], "window_passes": w["window_passes"],
        "fit_seconds": w["fit_seconds"], "fit_minutes": round(w["fit_seconds"] / 60, 2),
        "cuda_max_memory_allocated_gb": w.get("cuda_max_memory_allocated_gb"),
        "fit_reproduction": "" if rep is None else ("pass" if rep["passed"] else "fail"),
        "fit_reproduction_bitwise": "" if rep is None else rep["bitwise_adapter"],
        "negative_transfer_vs_hard_ci": bool(r["pooled_over_hard_ci_lo"] > 1.0),
        "worse_than_hard_point": bool(r["pooled_over_hard"] > 1.0)})
    return r


def _ordered(rows):
    tix = {t: i for i, t in enumerate(ss.TARGETS)}
    return sorted(rows, key=lambda r: (tix.get(r["target"], 99), r["n_sources"],
                                       -1 if r["order_seed"] == "all" else r["order_seed"]))


# --------------------------------------------------------------------------- #
# analysis
# --------------------------------------------------------------------------- #
def paired_change(a: dict, b: dict, B: int, seed: int) -> dict:
    """Change of the pooled MASE degradation from source pool a to source pool b (b minus a), in
    percentage points of the full model, with a paired cluster bootstrap over the target's test
    series. Negative = the larger pool transfers better."""
    for k in ("native__mase", "cluster_ids_test", "rows_test"):
        if not np.array_equal(a[k], b[k]):
            raise RuntimeError(f"cells do not share the target's test rows ({k})")
    W = np.stack([a["native__mase"], a["pooled__mase"], b["pooled__mase"]])
    R = phase2.cluster_replicates(W, a["cluster_ids_test"], B, seed)
    rep = 100.0 * (R[:, 2] - R[:, 1]) / R[:, 0]
    pt = 100.0 * (W[2].mean() - W[1].mean()) / W[0].mean()
    lo, hi = np.percentile(rep, [2.5, 97.5])
    return {"change_pp": float(pt), "ci_lo": float(lo), "ci_hi": float(hi),
            "direction": "improves" if hi < 0 else ("worsens" if lo > 0 else "indistinguishable")}


def _cell_map(cells):
    return {(c["selection"]["target"], c["selection"]["cell_id"]): c for c in cells}


def steps_table(cells, B: int = phase2.BOOT_B, seed: int = phase2.SEED) -> list[dict]:
    """Per (target, order seed): every consecutive step 1->4->8->13 and the overall 1->13."""
    cm = _cell_map(cells)
    cache = {}

    def pw(key):
        if key not in cache:
            cache[key] = per_window(cm[key])
        return cache[key]
    out = []
    for t in ss.TARGETS:
        for s in ss.ORDER_SEEDS:
            ids = {n: (t, ss.cell_id(n, None if n == POOL else s)) for n in ss.N_GRID}
            for a, b in STEPS + ((1, POOL),):
                ka, kb = ids[a], ids[b]
                if ka not in cm or kb not in cm:
                    continue
                r = paired_change(pw(ka), pw(kb), B, seed)
                out.append({"target": t, "display": _dname(t), "order_seed": s,
                            "from_n": a, "to_n": b, "step": f"{a}->{b}",
                            "added_sources": b - a, **r,
                            "gain_pp_per_added_source": -r["change_pp"] / (b - a)})
    return out


def by_target_n(rows) -> list[dict]:
    """Per target x N: the three orders (one shared cell at N = 13)."""
    out = []
    for t in ss.TARGETS:
        rt = [r for r in rows if r["target"] == t]
        if not rt:
            continue
        for n in ss.N_GRID:
            g = [r for r in rt if r["n_sources"] == n]
            if not g:
                continue
            d = [100 * r["pooled_dmase"] for r in g]
            gap = [r["pooled_minus_specific_pp"] for r in g]
            best = min(g, key=lambda r: r["pooled_dmase"])
            worst = max(g, key=lambda r: r["pooled_dmase"])
            out.append({"target": t, "display": _dname(t), "n_sources": n, "n_cells": len(g),
                        "dmase_pct_mean": float(np.mean(d)), "dmase_pct_min": float(min(d)),
                        "dmase_pct_max": float(max(d)), "dmase_pct_range": float(max(d) - min(d)),
                        "gap_to_specific_pp_mean": float(np.mean(gap)),
                        "gap_to_specific_pp_min": float(min(gap)),
                        "gap_to_specific_pp_max": float(max(gap)),
                        "best_selection": best["selection_order"],
                        "worst_selection": worst["selection_order"],
                        "hard_dmase_pct": 100 * g[0]["hard_dmase"],
                        "specific_dmase_pct": 100 * g[0]["specific_dmase"],
                        "full_mase": g[0]["full_mase"],
                        "n_train_windows": g[0]["n_train_windows"],
                        "fit_minutes_mean": float(np.mean([r["fit_minutes"] for r in g]))})
    return out


def hierarchical_interval(vals: dict, B: int = 5000, seed: int = 0) -> tuple:
    """Mean over targets of the within-target mean, and a 95% interval from resampling targets,
    then source orders within each resampled target. Descriptive (four targets)."""
    keys = [k for k, v in vals.items() if len(v)]
    if not keys:
        return None, None, None
    point = float(np.mean([np.mean(vals[k]) for k in keys]))
    if len(keys) < 2:
        return point, None, None
    rng = np.random.default_rng(seed)
    reps = np.empty(B)
    for b in range(B):
        m = []
        for i in rng.integers(0, len(keys), len(keys)):
            v = np.asarray(vals[keys[i]], float)
            m.append(v[rng.integers(0, len(v), len(v))].mean())
        reps[b] = np.mean(m)
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return point, float(lo), float(hi)


def aggregate(rows) -> list[dict]:
    out = []
    for n in ss.N_GRID:
        g = [r for r in rows if r["n_sources"] == n]
        if not g:
            continue
        d = {t: [100 * r["pooled_dmase"] for r in g if r["target"] == t] for t in ss.TARGETS}
        gp = {t: [r["pooled_minus_specific_pp"] for r in g if r["target"] == t]
              for t in ss.TARGETS}
        dm, dlo, dhi = hierarchical_interval(d)
        gm, glo, ghi = hierarchical_interval(gp)
        out.append({"n_sources": n, "n_targets": sum(1 for v in d.values() if v),
                    "n_cells": len(g), "dmase_pct_mean": dm, "dmase_pct_lo": dlo,
                    "dmase_pct_hi": dhi, "gap_to_specific_pp_mean": gm,
                    "gap_to_specific_pp_lo": glo, "gap_to_specific_pp_hi": ghi,
                    "n_train_windows": g[0]["n_train_windows"],
                    "steps_per_fit": g[0]["steps_completed"],
                    "window_passes_mean": float(np.mean([r["window_passes"] for r in g])),
                    "fit_minutes_mean": float(np.mean([r["fit_minutes"] for r in g])),
                    "n_negative_transfer_vs_hard_ci": sum(r["negative_transfer_vs_hard_ci"]
                                                          for r in g),
                    "n_within_5pct_of_full": sum(r["pooled_dmase"] <= 0.05 for r in g)})
    return out


def analysis(rows, steps, bt) -> dict:
    """The six questions of the design, answered with counts and numbers (no test over targets)."""
    pairs = [(t, s) for t in ss.TARGETS for s in ss.ORDER_SEEDS]
    seq = {}
    for t, s in pairs:
        v = {}
        for n in ss.N_GRID:
            cid = ss.cell_id(n, None if n == POOL else s)
            r = next((x for x in rows if x["target"] == t and x["cell_id"] == cid), None)
            if r is not None:
                v[n] = 100 * r["pooled_dmase"]
        if len(v) == len(ss.N_GRID):
            seq[(t, s)] = v
    overall = [x for x in steps if x["step"] == f"1->{POOL}"]
    consec = [x for x in steps if x["step"] != f"1->{POOL}"]
    monotone = [k for k, v in seq.items()
                if all(v[a] >= v[b] for a, b in STEPS)]
    rng = {}
    for r in bt:
        if r["n_sources"] < POOL:
            rng.setdefault(r["target"], {})[r["n_sources"]] = r["dmase_pct_range"]
    n13 = [r for r in rows if r["n_sources"] == POOL]
    gains = {}
    for x in consec:
        gains.setdefault(x["target"], {}).setdefault(x["step"], []).append(
            x["gain_pp_per_added_source"])
    gain_mean = {t: {k: float(np.mean(v)) for k, v in g.items()} for t, g in gains.items()}
    diminishing = [t for t, g in gain_mean.items()
                   if all(k in g for k in ("1->4", "4->8", "8->13"))
                   and g["1->4"] > g["4->8"] > g["8->13"]]

    def cat(r):
        lo, hi = r["pooled_minus_specific_pp_ci_lo"], r["pooled_minus_specific_pp_ci_hi"]
        return "worse" if lo > 0 else ("better" if hi < 0 else "indistinguishable")
    return {
        "n_complete_sequences": len(seq), "n_sequences_expected": len(pairs),
        "q1_larger_pool_improves": {
            "overall_1_to_13": {d: sum(x["direction"] == d for x in overall)
                                for d in ("improves", "worsens", "indistinguishable")},
            "overall_1_to_13_point_negative": sum(x["change_pp"] < 0 for x in overall),
            "n_pairs": len(overall),
            "per_target_mean_change_pp": {t: float(np.mean([x["change_pp"] for x in overall
                                                            if x["target"] == t]))
                                          for t in ss.TARGETS
                                          if any(x["target"] == t for x in overall)}},
        "q2_monotone": {"n_monotone_nonincreasing_point": len(monotone),
                        "monotone_pairs": [f"{t}/seed{s}" for t, s in monotone],
                        "consecutive_steps": {d: sum(x["direction"] == d for x in consec)
                                              for d in ("improves", "worsens",
                                                        "indistinguishable")},
                        "n_consecutive_steps": len(consec)},
        "q3_source_identity": {"seed_range_pp": rng,
                               "note": "range of the MASE change over the three source orders "
                                       "at a fixed N (percentage points of the full model)"},
        "q4_negative_transfer": {
            "cells_worse_than_hard_cut_ci": [f"{r['target']}/{r['cell_id']}" for r in rows
                                             if r["negative_transfer_vs_hard_ci"]],
            "cells_worse_than_hard_cut_point": [f"{r['target']}/{r['cell_id']}" for r in rows
                                                if r["worse_than_hard_point"]],
            "steps_worsening_ci": [f"{x['target']}/seed{x['order_seed']}/{x['step']}"
                                   for x in consec if x["direction"] == "worsens"]},
        "q5_n13_vs_dataset_specific": {
            r["target"]: {"pooled_minus_specific_pp": r["pooled_minus_specific_pp"],
                          "ci": [r["pooled_minus_specific_pp_ci_lo"],
                                 r["pooled_minus_specific_pp_ci_hi"]],
                          "category": cat(r), "pooled_dmase_pct": 100 * r["pooled_dmase"],
                          "specific_dmase_pct": 100 * r["specific_dmase"],
                          "hard_dmase_pct": 100 * r["hard_dmase"]}
            for t in ss.TARGETS for r in n13 if r["target"] == t},
        "q6_diminishing_returns": {"gain_pp_per_added_source_mean_over_orders": gain_mean,
                                   "targets_strictly_diminishing": diminishing}}


def phase2c_crosscheck(rows, lodo_root: Path) -> dict:
    """READ-ONLY: if the Phase 2c main cell (target, L6) exists, the N = 13 cell must equal it
    (same sources, same code; bitwise expected). Absence is recorded, never waited for."""
    out = {}
    for r in rows:
        if r["n_sources"] != POOL:
            continue
        d = Path(lodo_root) / lodo.KIND / lodo.MAIN / r["target"] / lodo.depth_label(ss.DEPTH)
        if not (d / "COMPLETE").exists():
            out[r["target"]] = {"available": False}
            continue
        try:
            m = json.loads((d / "metrics.json").read_text())
            fit = json.loads((d / "fit.json").read_text())
            mine = json.loads((Path(r["_dir"]) / "fit.json").read_text())
            a = m["stats"]["absolute"]["pooled"]["mase"]
            b = r["pooled_mase"]
            out[r["target"]] = {"available": True, "phase2c_dir": str(d),
                                "phase2c_pooled_mase": a, "phase2f_pooled_mase": b,
                                "rel_diff": abs(a - b) / max(abs(a), 1e-12),
                                "same_sources": sorted(json.loads((d / "fold.json").read_text())
                                                       ["sources"]) == sorted(
                                    r["sources"].split(";")),
                                "same_adapter_sha256": (fit.get("adapter") or {}).get("sha256")
                                == (mine.get("adapter") or {}).get("sha256")}
        except Exception as exc:                      # never let a cross-check break the tables
            out[r["target"]] = {"available": True, "error": f"{type(exc).__name__}: {exc}"}
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
XPOS = {1: 1.0, 4: 4.0, 8: 8.0, 13: 13.0}
X_HARD, X_SPEC = -3.5, 17.5       # the two references sit OFF the N axis, in shaded margins


def _xaxis(ax, compact=False):
    ax.set_xlim(X_HARD - 1.2, X_SPEC + 1.2)
    ax.set_xticks([X_HARD, 1, 4, 8, 13, X_SPEC])
    ax.set_xticklabels(["hard", "1", "4", "8", "13", "spec."] if compact else
                       ["hard cut\n(no adapter)", "1", "4", "8", "13", "target-\nspecific"])
    for a, b in ((X_HARD - 1.2, -0.5), (14.5, X_SPEC + 1.2)):
        ax.axvspan(a, b, color="#F2F2F2", zorder=0, lw=0)


def _sym(ax, values):
    import matplotlib.ticker as mtk
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    lo = min(-3.0, float(v.min()) * 1.15 if len(v) else -3.0)
    hi = max(10.0, float(v.max()) * 1.15 if len(v) else 10.0)
    ax.set_yscale("symlog", linthresh=20, linscale=1.6)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (-50, -20, -10, 0, 5, 10, 20, 50, 100, 200, 500, 1000) if lo <= t <= hi]
    ax.yaxis.set_major_locator(mtk.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mtk.FuncFormatter(
        lambda x, _: "0" if x == 0 else (f"+{x:g}" if x > 0 else f"{x:g}")))
    ax.yaxis.set_minor_locator(mtk.NullLocator())
    ax.axhline(0, color="black", lw=0.6, zorder=1)


def _err(y, lo, hi):
    """Error-bar half-widths, clipped at 0 (a percentile CI need not contain the point)."""
    y, lo, hi = (np.asarray(v, float) for v in (y, lo, hi))
    return [np.maximum(0.0, y - lo), np.maximum(0.0, hi - y)]


def _series(rows, t, key, seed=None):
    """(x, y) of one target: the seed's own cells at N = 1, 4, 8 and the shared N = 13 cell."""
    xs, ys = [], []
    for n in ss.N_GRID:
        cid = ss.cell_id(n, None if n == POOL else seed)
        r = next((x for x in rows if x["target"] == t and x["cell_id"] == cid), None)
        if r is not None:
            xs.append(XPOS[n])
            ys.append(key(r))
    return xs, ys


def main_figure(rows, bt, out_dir: Path, *, gap: bool):
    """Per target: the mean over the three orders (line), their min-max (band); hard cut at N = 0,
    the dataset-specific H3 adapter on the right."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    stem = "source_scaling_gap_to_specific" if gap else "source_scaling_main"
    with plt.rc_context(RC):
        fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN * 0.62, 3.0))
        vals = []
        for t in ss.TARGETS:
            g = [r for r in bt if r["target"] == t]
            if not g:
                continue
            col = TARGET_COLORS[t]
            key = "gap_to_specific_pp" if gap else "dmase_pct"
            xs = [XPOS[r["n_sources"]] for r in g]
            m = [r[f"{key}_mean"] for r in g]
            lo = [r[f"{key}_min"] for r in g]
            hi = [r[f"{key}_max"] for r in g]
            hard = (g[0]["hard_dmase_pct"] - g[0]["specific_dmase_pct"]) if gap else \
                g[0]["hard_dmase_pct"]
            spec = 0.0 if gap else g[0]["specific_dmase_pct"]
            ax.fill_between(xs, lo, hi, color=col, alpha=0.15, lw=0)
            ax.plot(xs, m, color=col, marker="o", ms=3, lw=1.1,
                    label=f"{_dname(t)} ({PROV[lodo.CHRONOS2_PROVENANCE[t][0]]})")
            ax.plot(X_HARD, hard, ls="none", marker="x", ms=3.6, color=col, mew=1.0)
            ax.plot(X_SPEC, spec, ls="none", marker="s", ms=3.4, color=col, mfc="white", mew=1.0)
            vals += m + lo + hi + [hard, spec]
        _sym(ax, vals)
        _xaxis(ax)
        ax.set_xlabel("Number of source datasets (N)")
        ax.set_ylabel("Pooled minus target-specific\n(pp of full-model MASE)" if gap else
                      "Test MASE change vs full\nmodel (%), truncated at L6")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.3), ncol=2, fontsize=6.0,
                  frameon=False)
        fig.tight_layout()
        return _save(fig, out_dir, stem)


def per_target_figure(rows, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tg = [t for t in ss.TARGETS if any(r["target"] == t for r in rows)]
    if not tg:
        return []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(tg), figsize=(TEXTWIDTH_IN, 2.3), squeeze=False)
        for ax, t in zip(axes[0], tg):
            col = TARGET_COLORS[t]
            rt = [r for r in rows if r["target"] == t]
            vals = []
            ref = rt[0]
            h, hlo, hhi = (100 * ref["hard_dmase"], 100 * ref["hard_dmase_ci_lo"],
                           100 * ref["hard_dmase_ci_hi"])
            sp, slo, shi = (100 * ref["specific_dmase"], 100 * ref["specific_dmase_ci_lo"],
                            100 * ref["specific_dmase_ci_hi"])
            ax.errorbar([X_HARD], [h], yerr=_err([h], [hlo], [hhi]), color="#555555", marker="x",
                        ms=3.5, capsize=1.5, lw=0.8)
            ax.errorbar([X_SPEC], [sp], yerr=_err([sp], [slo], [shi]), color=col, marker="s",
                        mfc="white", ms=3.2, capsize=1.5, lw=0.8)
            ax.axhline(sp, color=col, lw=0.5, ls=(0, (1, 1.5)), zorder=0)
            vals += [h, hlo, hhi, sp, slo, shi]
            for k, s in enumerate(ss.ORDER_SEEDS):
                xs, ys = _series(rt, t, lambda r: 100 * r["pooled_dmase"], s)
                _, lo = _series(rt, t, lambda r: 100 * r["pooled_dmase_ci_lo"], s)
                _, hi = _series(rt, t, lambda r: 100 * r["pooled_dmase_ci_hi"], s)
                dx = (k - 1) * 0.22
                xx = [x + (dx if x != XPOS[POOL] else 0.0) for x in xs]
                ax.errorbar(xx, ys, yerr=_err(ys, lo, hi),
                            color=col, ls=SEED_STYLE[s], marker="o", ms=2.6, lw=0.9,
                            capsize=1.2, elinewidth=0.6, label=f"order {s}")
                vals += ys + lo + hi
            _sym(ax, vals)
            _xaxis(ax, compact=True)
            ax.set_title(f"{_dname(t)}\n({PROV[lodo.CHRONOS2_PROVENANCE[t][0]]}, "
                         f"{ss.TARGET_RATIONALE[t]['frequency']})", fontsize=7.5)
            ax.tick_params(axis="x", labelsize=5.5)
        axes[0][0].set_ylabel("Test MASE change vs full model (%)")
        h_, l_ = axes[0][0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        return _save(fig, out_dir, "source_scaling_per_target")


def aggregate_figure(rows, agg, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not agg:
        return []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.4))
        for ax, key, ylab in ((axes[0], "dmase_pct", "MASE change vs full (%)"),
                              (axes[1], "gap_to_specific_pp", "Pooled minus specific (pp)")):
            vals = []
            for t in ss.TARGETS:
                rt = [r for r in rows if r["target"] == t]
                if not rt:
                    continue
                pts = [(XPOS[r["n_sources"]] + 0.12 * (ss.TARGETS.index(t) - 1.5),
                        100 * r["pooled_dmase"] if key == "dmase_pct"
                        else r["pooled_minus_specific_pp"]) for r in rt]
                ax.plot(*zip(*pts), ls="none", marker="o", ms=2.0, alpha=0.55,
                        color=TARGET_COLORS[t], label=_dname(t))
                vals += [p[1] for p in pts]
            x = [XPOS[a["n_sources"]] for a in agg]
            m = [a[f"{key}_mean"] for a in agg]
            lo = [a[f"{key}_lo"] if a[f"{key}_lo"] is not None else a[f"{key}_mean"] for a in agg]
            hi = [a[f"{key}_hi"] if a[f"{key}_hi"] is not None else a[f"{key}_mean"] for a in agg]
            ax.fill_between(x, lo, hi, color="black", alpha=0.10, lw=0)
            ax.plot(x, m, color="black", lw=1.4, marker="D", ms=3, label="mean over targets")
            vals += m + lo + hi
            _sym(ax, vals)
            ax.set_xlim(0.2, 13.8)
            ax.set_xticks([1, 4, 8, 13])
            ax.set_xlabel("Number of source datasets (N)")
            ax.set_ylabel(ylab)
        h_, l_ = axes[0].get_legend_handles_labels()
        fig.legend(h_, l_, loc="lower center", ncol=5, frameon=False, fontsize=6.0)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        return _save(fig, out_dir, "source_scaling_aggregate")


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def _p(x):
    return "n/a" if x is None else _pct(x)


def _ppf(x, digits=1):
    if x is None:
        return "n/a"
    t = f"{x:+.{digits}f}"
    return "$-$" + t[1:] if t.startswith("-") else t


def cells_tex(rows, out: Path, partial: str):
    L = ["% Phase 2f: every target x N x source order (Chronos-2 truncated at L6). The target is",
         "% excluded from adapter fitting and selection. Generated by",
         "% experiments/make_phase2f_tables.py; do not edit." + partial,
         r"\begin{tabular}{@{}lrlrrrrr@{}}", r"\toprule",
         r"Target & $N$ & Order & $\Delta$MASE [95\% CI] & Pooled $-$ spec. (pp) & "
         r"Gap rec. & wd / epoch & Train win. \\", r"\midrule"]
    for t in ss.TARGETS:
        rt = [r for r in _ordered(rows) if r["target"] == t]
        if not rt:
            continue
        r0 = rt[0]
        L.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{{_dname(t)}}} "
                 rf"({PROV[r0['provenance_status']]}): full MASE {r0['full_mase']:.3f}, "
                 rf"hard {_p(r0['hard_dmase'])}, target-specific {_p(r0['specific_dmase'])}}} \\")
        for r in rt:
            gc = "n/a" if r["pooled_gap_closure"] is None else \
                f"{100 * r['pooled_gap_closure']:.0f}\\%"
            wd = "hard" if r["selected_epoch"] == 0 else f"{r['selected_wd']:g} / " \
                                                         f"{r['selected_epoch']}"
            nwin = f"{r['n_train_windows']:,}".replace(",", "{,}")
            L.append(f" & {r['n_sources']} & {r['order_seed']} & {_p(r['pooled_dmase'])} "
                     f"[{_num(r['pooled_dmase_ci_lo'])}, {_num(r['pooled_dmase_ci_hi'])}] & "
                     f"{_ppf(r['pooled_minus_specific_pp'])} & {gc} & {wd} & {nwin} \\\\")
        L.append(r"\midrule" if t != ss.TARGETS[-1] else "")
    L = [x for x in L if x]
    L += [r"\bottomrule", r"\end{tabular}"]
    _write(out, L)


def orders_tex(roster, out: Path):
    L = ["% Phase 2f source orders (fixed before any fit): N = 1, 4, 8 are the first 1, 4, 8",
         "% datasets of each order (marked |); N = 13 is the whole pool. Generated; do not edit.",
         r"\begin{tabular}{@{}llp{0.72\linewidth}@{}}", r"\toprule",
         r"Target & Order & Sources in order \\", r"\midrule"]
    for t in ss.TARGETS:
        for o in ss.source_orders(t, roster):
            names = [_dname(x) for x in o["order"]]
            parts = [names[0], ", ".join(names[1:4]), ", ".join(names[4:8]), ", ".join(names[8:])]
            L.append(f"{_dname(t)} & {o['seed']} & {' $|$ '.join(parts)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _write(out, L)


def macros_tex(agg, ana, out: Path, partial: str):
    L = ["% Phase 2f macros. Generated by experiments/make_phase2f_tables.py." + partial]

    def put(name, val):
        L.append(f"\\newcommand{{\\ssc{name}}}{{{val}}}")
    word = {1: "One", 4: "Four", 8: "Eight", 13: "Thirteen"}
    for a in agg:
        w = word[a["n_sources"]]
        put(f"DmaseMean{w}", "n/a" if a["dmase_pct_mean"] is None
            else f"{a['dmase_pct_mean']:+.1f}\\%")
        put(f"GapMean{w}", "n/a" if a["gap_to_specific_pp_mean"] is None
            else _ppf(a["gap_to_specific_pp_mean"]))
    q1 = ana["q1_larger_pool_improves"]
    put("PairsImprove", q1["overall_1_to_13"]["improves"])
    put("PairsWorsen", q1["overall_1_to_13"]["worsens"])
    put("Pairs", q1["n_pairs"])
    put("Monotone", ana["q2_monotone"]["n_monotone_nonincreasing_point"])
    _write(out, L)


# --------------------------------------------------------------------------- #
def build(root: Path, paper_dir: Path, *, latency: Path = DEFAULT_LATENCY,
          lodo_root: Path = DEFAULT_LODO_ROOT, figures: bool = True,
          boot_b: int = phase2.BOOT_B) -> dict:
    roster = list(registry.roster("paper14"))
    cells = load_cells(root)
    rows = []
    for c in cells:
        r = flat(c)
        r["_dir"] = c["dir"]
        rows.append(r)
    expected = ss.expected_cell_ids()
    have = {(r["target"], r["cell_id"]) for r in rows}
    missing = sorted(expected - have)
    extra = sorted(have - expected)
    partial = "" if not missing else f"  INCOMPLETE: {len(have & expected)}/{len(expected)} cells"
    steps = steps_table(cells, B=boot_b)
    bt = by_target_n(rows)
    agg = aggregate(rows)
    ana = analysis(rows, steps, bt)
    cross = phase2c_crosscheck(rows, lodo_root)
    speed = load_speedups(latency)
    comb = Path(root) / "combined"
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in _ordered(rows)]
    _csv(comb / "source_scaling_cells.csv", clean)
    _csv(comb / "source_scaling_by_target_n.csv", bt)
    _csv(comb / "source_scaling_steps.csv", steps)
    _csv(comb / "source_scaling_aggregate.csv", agg)
    files = []
    if rows:
        cells_tex(rows, paper_dir / "source_scaling_cells_table.tex", partial)
        orders_tex(roster, paper_dir / "source_scaling_orders_table.tex")
        macros_tex(agg, ana, paper_dir / "source_scaling_macros.tex", partial)
        files += ["source_scaling_cells_table.tex", "source_scaling_orders_table.tex",
                  "source_scaling_macros.tex"]
        if figures:
            files += main_figure(rows, bt, paper_dir, gap=False)
            files += main_figure(rows, bt, paper_dir, gap=True)
            files += per_target_figure(rows, paper_dir)
            files += aggregate_figure(rows, agg, paper_dir)
    design = Path(root) / ss.KIND / "design.json"
    stats = {"schema": "phase2f_source_scaling_stats/v1", "protocol": ss.PROTOCOL_VERSION,
             "complete": not missing, "n_cells": len(have & expected),
             "n_expected": len(expected), "missing": [list(m) for m in missing],
             "unexpected_cells": [list(e) for e in extra],
             "targets": list(ss.TARGETS), "target_rationale": ss.TARGET_RATIONALE,
             "target_selection_rule": ss.TARGET_SELECTION_RULE,
             "depth": ss.DEPTH, "label": lodo.depth_label(ss.DEPTH), "n_grid": list(ss.N_GRID),
             "order_seeds": list(ss.ORDER_SEEDS),
             "orders": {t: ss.source_orders(t, roster) for t in ss.TARGETS},
             "design_file": str(design) if design.exists() else None,
             "provenance": {t: lodo.CHRONOS2_PROVENANCE[t][0] for t in ss.TARGETS},
             "speedup_L6": {"B1": speed.get((ss.DEPTH, 1)), "B256": speed.get((ss.DEPTH, 256)),
                            "source": str(latency)},
             "fit_reproduction": [{"cell": f"{r['target']}/{r['cell_id']}",
                                   "status": r["fit_reproduction"],
                                   "bitwise": r["fit_reproduction_bitwise"]}
                                  for r in _ordered(rows) if r["n_sources"] == 1],
             "phase2c_crosscheck": cross, "aggregate": agg, "by_target_n": bt,
             "analysis": ana,
             "caveats": {
                 "four_targets": "four targets: counts are descriptive, not a population test",
                 "data_volume": "each source adds 1,394 training windows: N also scales the "
                                "training data and the optimization work per step",
                 "series_families": "N = 13 contains both Electricity datasets (shared meters): "
                                    "12 dataset families, not 13",
                 "no_extrapolation": "no trend is fitted or extrapolated beyond N = 13",
                 "specific_reference": "the dataset-specific H3 adapter is calibrated on the "
                                       "target's unlabeled training inputs: a practical "
                                       "reference, not a guaranteed upper bound"},
             "paper_files": files}
    phase1.atomic_write_json(comb / "source_scaling_stats.json", stats)
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--paper-dir", default=None,
                   help="default three_models_paper/figures/source_scaling for the default root, "
                        "<root>/paper otherwise")
    p.add_argument("--latency", default=str(DEFAULT_LATENCY))
    p.add_argument("--lodo-root", default=str(DEFAULT_LODO_ROOT))
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    root = Path(a.root)
    paper = Path(a.paper_dir) if a.paper_dir else (
        DEFAULT_PAPER_DIR if root.resolve() == DEFAULT_ROOT.resolve() else root / "paper")
    s = build(root, paper, latency=Path(a.latency), lodo_root=Path(a.lodo_root),
              figures=not a.no_figures)
    print(f"Phase 2f tables: {s['n_cells']}/{s['n_expected']} cells "
          f"({'complete' if s['complete'] else 'INCOMPLETE'}) -> {root / 'combined'}, {paper}")
    for g in s["aggregate"]:
        print(f"  N={g['n_sources']:>2}  targets {g['n_targets']}  mean dMASE "
              f"{g['dmase_pct_mean']:+.1f}%  mean pooled-minus-specific "
              f"{g['gap_to_specific_pp_mean']:+.1f} pp  ({g['n_train_windows']} train windows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
