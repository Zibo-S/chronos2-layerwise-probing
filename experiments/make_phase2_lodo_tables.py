"""Phase 2c tables, figures and macros, rebuilt from the LODO cell artifacts ONLY.

    python -m experiments.make_phase2_lodo_tables            # login node: seconds, no model

Reads   results/three_model_phase2_lodo/lodo/<variant>/<target>/<L>/{COMPLETE, metrics.json,
        fit.json, fold.json}  and the measured Phase-2 speedups (combined/latency_lookup.csv).
Writes  results/three_model_phase2_lodo/combined/   machine-readable CSV + JSON
        three_models_paper/figures/lodo/            .tex tables, .pdf/.png figures, macros

Three readings are kept apart everywhere, because they answer different questions:

    unseen DATASET     every main-protocol fold: the adapter never saw the target dataset
    unseen SERIES      ... and never saw the target's series: true for every main-protocol fold
                       (the Electricity sibling is excluded); FALSE in the sibling_included
                       sensitivity cells, reported separately and never pooled with the main ones
    unseen by the      the target is documented as HELD OUT from Chronos-2 pretraining (LOOP
    BACKBONE           Seattle, SZ Taxi, M5, Monash Traffic). "listed" targets were documented
                       pretraining data, so for them a successful transfer is transfer to an
                       unseen dataset whose series the frozen backbone may already know;
                       "not listed" is undocumented, NOT evidence of being unseen.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, registry                                              # noqa: E402
from probing import phase2_lodo as lodo                                           # noqa: E402
from experiments.make_phase2_paper_tables import (ARMS, RC, TEXTWIDTH_IN, Y_CAP,    # noqa: E402
                                                  _dname, _num, _pct, _save, _write)

DEFAULT_ROOT = REPO_ROOT / "results" / "three_model_phase2_lodo"
DEFAULT_PAPER_DIR = REPO_ROOT / "three_models_paper" / "figures" / "lodo"
DEFAULT_LATENCY = REPO_ROOT / "results" / "three_model_phase2" / "combined" / "latency_lookup.csv"

#: (table label, colour, marker) -- hard and dataset-specific reuse the H3 paper colours
LODO_ARMS = {"hard": ("Hard truncation", ARMS["hard"][2], "x"),
             "pooled": ("Pooled transfer", "#D55E00", "D"),
             "specific": ("Dataset-specific alignment", ARMS["noa"][2], "o")}
GROUP_NAMES = {"held_out": "Held out from pretraining", "not_listed": "Not listed",
               "listed": "Listed in pretraining"}
DEPTH_WORD = {4: "Lfour", 6: "Lsix", 9: "Lnine"}


# --------------------------------------------------------------------------- #
# reading
# --------------------------------------------------------------------------- #
def load_cells(root: Path) -> list[dict]:
    base = Path(root) / lodo.KIND
    out = []
    for v in lodo.VARIANTS:
        for d in sorted((base / v).glob("*/*")):
            if not d.is_dir() or ".building-" in d.name or not (d / "COMPLETE").exists():
                continue
            m = json.loads((d / "metrics.json").read_text())
            fold = json.loads((d / "fold.json").read_text())
            if not m.get("verification_all_passed"):
                raise RuntimeError(f"{d}: a reproduction gate failed; refusing to tabulate")
            out.append({"dir": str(d), "metrics": m, "fold": fold})
    return out


def flat(cell: dict) -> dict:
    m, fold = cell["metrics"], cell["fold"]
    s = m["stats"]
    tag = m["target"]
    status, scope = lodo.CHRONOS2_PROVENANCE[tag]
    r = {"variant": m["variant"], "dataset": tag, "display": _dname(tag), "depth": m["depth"],
         "label": m["label"], "n_sources": m["n_sources"],
         "provenance_status": status, "provenance_scope": scope,
         "target_series_in_adapter_training": fold["target_series_in_adapter_training"],
         "unseen_dataset": True,
         "unseen_series_by_adapter": not fold["target_series_in_adapter_training"],
         "held_out_from_backbone_pretraining": status == "held_out",
         "full_mase": s["absolute"]["native"]["mase"],
         "full_wql": s["absolute"]["native"]["wql"]}
    for a in ("hard", "pooled", "specific"):
        v = s["vs_native"][a]
        r[f"{a}_mase"] = s["absolute"][a]["mase"]
        r[f"{a}_dmase"] = v["mase"]["degradation"]
        r[f"{a}_dmase_ci_lo"] = v["mase"]["degradation_ci_lo"]
        r[f"{a}_dmase_ci_hi"] = v["mase"]["degradation_ci_hi"]
        r[f"{a}_dwql"] = v["wql"]["degradation"]
    ph, ps = s["pooled_vs_hard"]["mase"], s["pooled_vs_specific"]["mase"]
    r.update({"pooled_over_hard": ph["ratio"], "pooled_over_hard_ci_lo": ph["ratio_ci_lo"],
              "pooled_over_hard_ci_hi": ph["ratio_ci_hi"], "improves_over_hard": ph["improves"],
              "improves_over_hard_ci": ph["improves_ci"],
              "pooled_minus_specific_pp": ps["degradation_difference_pp"],
              "pooled_minus_specific_pp_ci_lo": ps["degradation_difference_pp_ci"][0],
              "pooled_minus_specific_pp_ci_hi": ps["degradation_difference_pp_ci"][1]})
    gp, gs = s["gap"]["mase"]["pooled"], s["gap"]["mase"]["specific"]
    r.update({"hard_gap_sign": gp["gap_sign"],
              "hard_gap_ci_lo": gp["hard_gap_ci"][0], "hard_gap_ci_hi": gp["hard_gap_ci"][1],
              "pooled_gap_closure": gp["reported"],
              "pooled_gap_closure_ci_lo": gp["ci_lo"] if gp["reported"] is not None else None,
              "pooled_gap_closure_ci_hi": gp["ci_hi"] if gp["reported"] is not None else None,
              "specific_gap_closure": gs["reported"],
              "selected_wd": m["selected_wd"], "selected_epoch": m["selected_epoch"],
              **{k: m["flags"][k] for k in ("selected_hard_cut", "late_checkpoint",
                                            "grid_max_wd")}})
    return r


def load_speedups(path: Path) -> dict:
    """{(depth, batch): measured end-to-end adapter speedup} for Chronos-2 (median of 3 jobs)."""
    out = {}
    if not Path(path).exists():
        return out
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if (r["model"] == lodo.MODEL and r["kind"] == "adapter" and r["level"] == "api_e2e"
                    and int(r["depth"]) in lodo.LODO_DEPTHS):
                out[(int(r["depth"]), int(r["batch"]))] = float(r["speedup_vs_native"])
    return out


# --------------------------------------------------------------------------- #
# aggregation
# --------------------------------------------------------------------------- #
def _q(vals):
    v = np.asarray([x for x in vals if x is not None], float)
    if not len(v):
        return None, None, None
    return tuple(float(x) for x in np.percentile(v, [50, 25, 75]))


def aggregate(rows: list[dict], speed: dict) -> list[dict]:
    """Per depth x provenance group, over the MAIN-protocol rows only."""
    main = [r for r in rows if r["variant"] == lodo.MAIN]
    out = []
    for depth in lodo.LODO_DEPTHS:
        rd = [r for r in main if r["depth"] == depth]
        for group in ("all",) + lodo.PROVENANCE_GROUPS:
            g = rd if group == "all" else [r for r in rd if r["provenance_status"] == group]
            if not g:
                continue
            rec = {"depth": depth, "label": lodo.depth_label(depth), "group": group, "n": len(g),
                   "speedup_B1": speed.get((depth, 1)), "speedup_B256": speed.get((depth, 256))}
            for a in ("hard", "pooled", "specific"):
                med, q1, q3 = _q([r[f"{a}_dmase"] for r in g])
                rec.update({f"{a}_median": med, f"{a}_q1": q1, f"{a}_q3": q3,
                            f"{a}_within_5": sum(r[f"{a}_dmase"] <= 0.05 for r in g),
                            f"{a}_within_10": sum(r[f"{a}_dmase"] <= 0.10 for r in g)})
            pos = [r for r in g if r["hard_gap_sign"] == "positive"]
            rec.update({
                "pooled_improves_over_hard": sum(r["improves_over_hard"] for r in g),
                "pooled_improves_over_hard_ci": sum(r["improves_over_hard_ci"] for r in g),
                "pooled_minus_specific_pp_median": _q([r["pooled_minus_specific_pp"]
                                                       for r in g])[0],
                "n_positive_hard_gap": len(pos),
                "pooled_gap_closure_median": _q([r["pooled_gap_closure"] for r in pos])[0],
                "specific_gap_closure_median": _q([r["specific_gap_closure"] for r in pos])[0],
                "n_late_checkpoint": sum(r["late_checkpoint"] for r in g),
                "n_grid_max_wd": sum(r["grid_max_wd"] for r in g),
                "n_selected_hard_cut": sum(r["selected_hard_cut"] for r in g)})
            out.append(rec)
    return out


def consistency(rows: list[dict]) -> list[dict]:
    """Per dataset across the three depths (main protocol): is a failure depth-consistent or
    depth-specific?"""
    main = [r for r in rows if r["variant"] == lodo.MAIN]
    out = []
    for tag in sorted({r["dataset"] for r in main}, key=_roster_index):
        g = sorted((r for r in main if r["dataset"] == tag), key=lambda r: r["depth"])
        out.append({"dataset": tag, "display": _dname(tag),
                    "provenance_status": g[0]["provenance_status"], "n_depths": len(g),
                    "depths_improving_over_hard": sum(r["improves_over_hard"] for r in g),
                    "depths_within_10": sum(r["pooled_dmase"] <= 0.10 for r in g),
                    "worst_depth": max(g, key=lambda r: r["pooled_dmase"])["label"],
                    "max_pooled_dmase": max(r["pooled_dmase"] for r in g),
                    "max_pooled_minus_specific_pp": max(r["pooled_minus_specific_pp"]
                                                        for r in g)})
    return out


def sensitivity(rows: list[dict]) -> list[dict]:
    """Electricity targets: 12-source unseen-series fit vs the 13-source fit that contains the
    target's series sibling."""
    out = []
    for tag in lodo.SERIES_SIBLINGS:
        for depth in lodo.LODO_DEPTHS:
            m = [r for r in rows if r["dataset"] == tag and r["depth"] == depth]
            a = next((r for r in m if r["variant"] == lodo.MAIN), None)
            b = next((r for r in m if r["variant"] == lodo.SENSITIVITY), None)
            if a is None or b is None:
                continue
            out.append({"dataset": tag, "display": _dname(tag), "label": a["label"],
                        "depth": depth, "full_mase": a["full_mase"],
                        "unseen_series_dmase": a["pooled_dmase"],
                        "sibling_included_dmase": b["pooled_dmase"],
                        "difference_pp": 100 * (b["pooled_dmase"] - a["pooled_dmase"]),
                        "specific_dmase": a["specific_dmase"], "hard_dmase": a["hard_dmase"],
                        "n_sources_unseen_series": a["n_sources"],
                        "n_sources_sibling_included": b["n_sources"]})
    return out


def _roster_index(tag):
    r = list(registry.tags("paper14"))
    return r.index(tag) if tag in r else len(r)


def _group_order(rows):
    return sorted(rows, key=lambda r: (lodo.PROVENANCE_GROUPS.index(r["provenance_status"]),
                                       _roster_index(r["dataset"])))


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def _csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _ci(lo, hi):
    return f"[{_num(lo)}, {_num(hi)}]"


def _pp(x, digits=1):
    if x is None:
        return "--"
    t = f"{x:+.{digits}f}"
    return "$-$" + t[1:] if t.startswith("-") else t


def per_dataset_tex(rows, out: Path, partial: str):
    main = [r for r in rows if r["variant"] == lodo.MAIN]
    L = ["% Phase 2c: per-dataset LODO transfer (main protocol: target and its series sibling",
         "% excluded from adapter fitting and selection). Generated by",
         "% experiments/make_phase2_lodo_tables.py; do not edit." + partial,
         r"\begin{tabular}{@{}llrrrrrr@{}}", r"\toprule",
         r"Dataset & Prov. & Full & Hard & Pooled transfer & Dataset-specific & "
         r"Pooled $-$ spec. & Gap rec. \\",
         r" & & MASE & $\Delta$ & $\Delta$ [95\% CI] & $\Delta$ [95\% CI] & (pp) & (pooled) \\"]
    prov = {"held_out": "held out", "not_listed": "not listed", "listed": "listed"}
    for depth in lodo.LODO_DEPTHS:
        rd = _group_order([r for r in main if r["depth"] == depth])
        if not rd:
            continue
        L += [r"\midrule", rf"\multicolumn{{8}}{{@{{}}l}}{{\textit{{Truncation at "
                           rf"{lodo.depth_label(depth)}}}}} \\"]
        for r in rd:
            gc = "--" if r["pooled_gap_closure"] is None else f"{100 * r['pooled_gap_closure']:.0f}\\%"
            L.append(f"{r['display']} & {prov[r['provenance_status']]} & {r['full_mase']:.3f} & "
                     f"{_pct(r['hard_dmase'])} & {_pct(r['pooled_dmase'])} "
                     f"{_ci(r['pooled_dmase_ci_lo'], r['pooled_dmase_ci_hi'])} & "
                     f"{_pct(r['specific_dmase'])} "
                     f"{_ci(r['specific_dmase_ci_lo'], r['specific_dmase_ci_hi'])} & "
                     f"{_pp(r['pooled_minus_specific_pp'])} & {gc} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _write(out, L)


def aggregate_tex(agg, out: Path, partial: str):
    L = ["% Phase 2c: aggregate LODO transfer, median [IQR] of the MASE change vs the full model",
         "% over datasets; counts are datasets. Generated; do not edit." + partial,
         r"\begin{tabular}{@{}llrcccccc@{}}", r"\toprule",
         r"Datasets & Depth & Speedup & Hard & Pooled transfer & Dataset-specific & "
         r"Pooled $<$ hard & Pooled $\le$5\% / $\le$10\% & Pooled $-$ spec. (pp) \\", r"\midrule"]
    for group in ("all", "held_out"):
        g = [a for a in agg if a["group"] == group]
        for a in g:
            name = (f"All ({a['n']})" if group == "all" else f"Held out ({a['n']})")
            sp = "--" if a["speedup_B1"] is None else f"{a['speedup_B1']:.2f}$\\times$"

            def med(arm):
                return (f"{_pct(a[f'{arm}_median'])} "
                        f"[{_num(a[f'{arm}_q1'])}, {_num(a[f'{arm}_q3'])}]")
            L.append(f"{name} & {a['label']} & {sp} & {med('hard')} & {med('pooled')} & "
                     f"{med('specific')} & {a['pooled_improves_over_hard']}/{a['n']} & "
                     f"{a['pooled_within_5']} / {a['pooled_within_10']} & "
                     f"{_pp(a['pooled_minus_specific_pp_median'])} \\\\")
        if group == "all" and g:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    _write(out, L)


def sensitivity_tex(sens, out: Path, partial: str):
    L = ["% Phase 2c sensitivity: Electricity targets with vs without the series sibling in the",
         "% adapter's sources. The 12-source column is the main (unseen-series) result.",
         "% Generated; do not edit." + partial,
         r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
         r"Target & Depth & Unseen series (12 src.) & Sibling included (13 src.) & "
         r"Difference (pp) & Dataset-specific \\", r"\midrule"]
    for s in sens:
        L.append(f"{s['display']} & {s['label']} & {_pct(s['unseen_series_dmase'])} & "
                 f"{_pct(s['sibling_included_dmase'])} & {_pp(s['difference_pp'])} & "
                 f"{_pct(s['specific_dmase'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    _write(out, L)


def macros_tex(agg, out: Path, partial: str):
    L = ["% Phase 2c LODO macros. Generated by experiments/make_phase2_lodo_tables.py." + partial]

    def put(name, val):
        L.append(f"\\newcommand{{\\lodo{name}}}{{{val}}}")
    for a in agg:
        w = DEPTH_WORD[a["depth"]]
        grp = {"all": "", "held_out": "HeldOut", "not_listed": "NotListed",
               "listed": "Listed"}[a["group"]]
        put(f"{grp}N{w}", a["n"])
        for arm, nm in (("hard", "Hard"), ("pooled", "Pooled"), ("specific", "Specific")):
            put(f"{grp}{nm}Median{w}", _pct(a[f"{arm}_median"]))
        put(f"{grp}PooledImproves{w}", a["pooled_improves_over_hard"])
        put(f"{grp}PooledWithinFive{w}", a["pooled_within_5"])
        put(f"{grp}PooledWithinTen{w}", a["pooled_within_10"])
        put(f"{grp}PooledMinusSpecific{w}", _pp(a["pooled_minus_specific_pp_median"]))
        if a["group"] == "all" and a["speedup_B1"] is not None:
            put(f"Speedup{w}", f"{a['speedup_B1']:.2f}")
    _write(out, L)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _xaxis(ax, lo=-15.0, cap=Y_CAP):
    """The paper's signed-percent degradation axis (linear to 20%, log beyond), horizontal."""
    import matplotlib.ticker as mt
    ax.set_xscale("symlog", linthresh=20, linscale=1.5)
    ax.set_xlim(lo, cap * 1.2)
    ticks = [t for t in (-10, 0, 10, 100) if lo <= t <= cap]       # narrow panels: sparse ticks
    ax.xaxis.set_major_locator(mt.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(mt.FuncFormatter(
        lambda v, _: "0" if v == 0 else (f"+{v:g}%" if v > 0 else f"{v:g}%")))
    ax.xaxis.set_minor_locator(mt.NullLocator())
    for b in (5, 10):
        ax.axvline(b, color="#D2D2D2", lw=0.5, ls=(0, (1, 1.5)), zorder=0)
    ax.axvline(0, color="black", lw=0.6, zorder=1)


def per_dataset_figure(rows, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main = [r for r in rows if r["variant"] == lodo.MAIN]
    first = {}
    for r in main:                                   # every dataset present at ANY depth
        first.setdefault(r["dataset"], r)
    order = [r["dataset"] for r in _group_order(list(first.values()))]
    if not order:
        return []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 0.21 * len(order) + 0.9),
                                 sharey=True)
        y = {t: len(order) - 1 - i for i, t in enumerate(order)}
        for ax, depth in zip(axes, lodo.LODO_DEPTHS):
            rd = {r["dataset"]: r for r in main if r["depth"] == depth}
            _xaxis(ax)
            for arm, dy in (("hard", 0.22), ("specific", 0.0), ("pooled", -0.22)):
                name, col, mk = LODO_ARMS[arm]
                for t in order:
                    r = rd.get(t)
                    if r is None:
                        continue
                    v = 100 * r[f"{arm}_dmase"]
                    yy = y[t] + dy
                    if arm != "hard":
                        lo, hi = 100 * r[f"{arm}_dmase_ci_lo"], 100 * r[f"{arm}_dmase_ci_hi"]
                        ax.plot([lo, min(hi, Y_CAP)], [yy, yy], color=col, lw=0.7, zorder=2)
                    if v > Y_CAP:
                        ax.plot(Y_CAP, yy, ls="none", marker=">", ms=3.2, color=col, zorder=3,
                                clip_on=False)
                    else:
                        ax.plot(v, yy, ls="none", marker=mk, ms=2.8 if mk != "x" else 3.2,
                                color=col, mfc=col if mk != "x" else None, mew=0.8, zorder=3)
            ax.set_title(f"Truncated at {lodo.depth_label(depth)}")
            ax.set_xlabel("MASE change vs full model")
            # provenance group separators
            prev = None
            for t in order:
                g = lodo.CHRONOS2_PROVENANCE[t][0]
                if prev is not None and g != prev:
                    ax.axhline(y[t] + 0.5, color="#BBBBBB", lw=0.5, zorder=0)
                prev = g
        axes[0].set_yticks([y[t] for t in order])
        axes[0].set_yticklabels([_dname(t) for t in order])
        handles = [plt.Line2D([], [], ls="none", marker=LODO_ARMS[a][2], color=LODO_ARMS[a][1],
                              label=LODO_ARMS[a][0], ms=3.5) for a in ("hard", "pooled",
                                                                         "specific")]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.55, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        return _save(fig, out_dir, "lodo_per_dataset")


def aggregate_figure(agg, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from experiments.make_phase2_paper_tables import _degradation_axis
    panels = [g for g in ("all", "held_out") if any(a["group"] == g for a in agg)]
    if not panels:
        return []
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, len(panels), figsize=(TEXTWIDTH_IN * 0.8, 2.4),
                                 sharey=True, squeeze=False)
        for ax, group in zip(axes[0], panels):
            g = sorted((a for a in agg if a["group"] == group), key=lambda a: a["depth"])
            _degradation_axis(ax, lo=-5.0, cap=Y_CAP)
            x0 = np.arange(len(g))
            for k, arm in enumerate(("hard", "pooled", "specific")):
                name, col, mk = LODO_ARMS[arm]
                xs = x0 + (k - 1) * 0.15
                med = np.array([100 * a[f"{arm}_median"] for a in g])
                lo = np.array([100 * a[f"{arm}_q1"] for a in g])
                hi = np.array([100 * a[f"{arm}_q3"] for a in g])
                ax.errorbar(xs, np.minimum(med, Y_CAP),
                            yerr=[np.minimum(med, Y_CAP) - np.minimum(lo, Y_CAP),
                                  np.minimum(hi, Y_CAP) - np.minimum(med, Y_CAP)],
                            ls="-" if arm != "hard" else (0, (3, 1.6)), lw=0.9, color=col,
                            marker=mk, ms=3.2, capsize=1.5, elinewidth=0.7, label=name,
                            mfc=col if mk != "x" else None)
            ax.set_xticks(x0)
            ax.set_xticklabels([f"{a['label']}\n" + ("" if a["speedup_B1"] is None else
                                                    f"{a['speedup_B1']:.2f}x")
                                for a in g])
            ax.set_xlim(-0.5, len(g) - 0.5)
            n = g[0]["n"] if g else 0
            ax.set_title(f"All datasets (n={n})" if group == "all"
                         else f"Held out from pretraining (n={n})")
        axes[0][0].set_ylabel("Median MASE change\n(IQR) vs full model")
        h, lab = axes[0][0].get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=6.5)
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        return _save(fig, out_dir, "lodo_aggregate")


# --------------------------------------------------------------------------- #
def build(root: Path, paper_dir: Path, latency: Path, figures: bool = True) -> dict:
    cells = load_cells(root)
    rows = [flat(c) for c in cells]
    roster = list(registry.tags("paper14"))
    expected = {(c["variant"], c["target"], c["depth"]) for c in lodo.plan_cells(roster)}
    have = {(r["variant"], r["dataset"], r["depth"]) for r in rows}
    missing = sorted(expected - have)
    n_main = sum(1 for e in expected if e[0] == lodo.MAIN)
    partial = "" if not missing else (f"  PARTIAL: {len(expected) - len(missing)}/"
                                      f"{len(expected)} cells")
    speed = load_speedups(latency)
    agg = aggregate(rows, speed)
    cons = consistency(rows)
    sens = sensitivity(rows)
    comb = Path(root) / "combined"
    _csv(comb / "lodo_per_dataset.csv", _group_order(rows) if rows else [])
    _csv(comb / "lodo_aggregate.csv", agg)
    _csv(comb / "lodo_depth_consistency.csv", cons)
    _csv(comb / "lodo_sibling_sensitivity.csv", sens)
    files = []
    if rows:
        per_dataset_tex(rows, paper_dir / "lodo_per_dataset_table.tex", partial)
        aggregate_tex(agg, paper_dir / "lodo_aggregate_table.tex", partial)
        sensitivity_tex(sens, paper_dir / "lodo_sibling_sensitivity_table.tex", partial)
        macros_tex(agg, paper_dir / "lodo_macros.tex", partial)
        files += ["lodo_per_dataset_table.tex", "lodo_aggregate_table.tex",
                  "lodo_sibling_sensitivity_table.tex", "lodo_macros.tex"]
        if figures:
            files += per_dataset_figure(rows, paper_dir) + aggregate_figure(agg, paper_dir)
    stats = {"schema": "phase2c_lodo_stats/v1", "protocol": lodo.LODO_PROTOCOL_VERSION,
             "complete": not missing, "n_cells": len(rows), "n_expected": len(expected),
             "n_expected_main": n_main, "missing": [list(m) for m in missing],
             "provenance_source": "notes/help.tex tab:provenance (Chronos-2 column)",
             "provenance_counts": {g: sum(1 for v in lodo.CHRONOS2_PROVENANCE.values()
                                          if v[0] == g) for g in lodo.PROVENANCE_GROUPS},
             "readings": {
                 "unseen_dataset": "all main-protocol folds",
                 "unseen_series_by_adapter": "all main-protocol folds; NOT the sibling_included "
                                             "sensitivity cells",
                 "held_out_from_backbone_pretraining": [t for t, v in
                                                        lodo.CHRONOS2_PROVENANCE.items()
                                                        if v[0] == "held_out"],
                 "caveat": "a 'listed' target is unseen by the adapter but documented pretraining "
                           "data for the frozen backbone; 'not listed' is undocumented, not "
                           "evidence of being unseen"},
             "aggregate": agg, "depth_consistency": cons, "sibling_sensitivity": sens,
             "speedups_measured": {f"L{d}_B{b}": v for (d, b), v in sorted(speed.items())},
             "paper_files": files}
    phase1.atomic_write_json(comb / "lodo_stats.json", stats)
    return stats


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    p.add_argument("--paper-dir", default=None,
                   help="default three_models_paper/figures/lodo (for the default root only; "
                        "<root>/paper otherwise, so a smoke can never overwrite paper files)")
    p.add_argument("--latency", default=str(DEFAULT_LATENCY))
    p.add_argument("--no-figures", action="store_true")
    a = p.parse_args(argv)
    root = Path(a.root)
    paper = Path(a.paper_dir) if a.paper_dir else (
        DEFAULT_PAPER_DIR if root.resolve() == DEFAULT_ROOT.resolve() else root / "paper")
    s = build(root, paper, Path(a.latency), figures=not a.no_figures)
    print(f"LODO tables: {s['n_cells']}/{s['n_expected']} cells "
          f"({'complete' if s['complete'] else 'PARTIAL'}) -> {root / 'combined'}, {paper}")
    for g in s["aggregate"]:
        if g["group"] in ("all", "held_out"):
            print(f"  {g['label']} {g['group']:<9} n={g['n']:>2}  pooled median "
                  f"{100 * g['pooled_median']:+.1f}%  specific {100 * g['specific_median']:+.1f}%"
                  f"  hard {100 * g['hard_median']:+.1f}%  pooled<hard {g['pooled_improves_over_hard']}"
                  f"  <=5% {g['pooled_within_5']}  <=10% {g['pooled_within_10']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
