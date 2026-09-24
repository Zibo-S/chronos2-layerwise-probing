"""Results §H3 / §H4 ("from recoverability to acceleration"): figures, tables and statistics.

    python -m experiments.make_phase2_tables          # first: rebuild <root>/combined/ (data layer)
    python -m experiments.make_phase2_paper_tables    # then:  this script (paper layer)

Reads ONLY <output-root>/combined/*.csv, which make_phase2_tables rebuilds from the saved Phase-2
cells. No model, no GPU, no refit; every value below is read from those files, nothing is typed.

WHERE TO RUN: login node is fine -- a few MB of CSV and ~10 matplotlib figures, seconds.

WHAT IT PRODUCES (``--out-dir``, default ``three_models_paper/figures/truncation/``; paths inside
the paper assume its root is ``three_models_paper/``, as for the Phase-1 figures)

  MAIN BODY
    h4_frontier.{pdf,png}             the accuracy-compute frontier: median [IQR] test MASE increase
                                      vs MEASURED end-to-end speedup (B=1), every block depth,
                                      hard / label-free aligned / supervised aligned / probe head
    tables/h3_ladder_table.tex        the compatibility ladder at the frozen Phase-1 entrance
    tables/h4_budget_table.tex        validation-budgeted truncation, eps = 2 / 5 / 10 / 20 %

  APPENDIX
    h4_frontier_appendix.{pdf,png}    the frontier vs throughput (B=256), forward-pass speedup
                                      (B=1) and parameters removed
    h4_frontier_wql.{pdf,png}         the frontier in WQL (probabilistic accuracy)
    h3_ladder_at_entrance.{pdf,png}   every dataset's ladder at its entrance
    h3_depth_curves.{pdf,png}         three representative datasets x three models, every depth
    h3_depth_{model}.{pdf,png}        all 14 datasets at every depth, one figure per model
    tables/h3_ladder_full.tex         per dataset at the entrance; label-free arm with its CI
    tables/h3_compatibility.tex       compatibility entrance and lag per readout (validation)
    tables/h3_adapter_selection.tex   what validation selected (epoch, decay, ridge strength)
    tables/h4_budget_points.tex       per dataset: the validation-selected cut, label-free arm
    tables/h4_latency_table.tex       absolute latency / throughput / memory / parameters
    tables/h4_truncation_table.tex    physically truncated models at the entrance (H4 evaluate)
    tables/h4_outcome_table.tex       the frozen-entrance verdict, secondary test (H4 evaluate)
    phase2_macros.tex                 \\newcommand macros (and \\PhaseTwoMissing)
    truncation_stats.json             every number the prose quotes, with its source

MISSING / INVALID INPUTS. A table whose inputs do not exist yet is written as a visible
``\\PhaseTwoMissing{...}`` box, never as a stale number. Both entrance tables are replaced by such a
box while any H4 cell is invalid (its physical model failed V3), and the budget table while any
physically instantiated operating point failed V3. Before submission, grep the output directory for
PhaseTwoMissing.
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

from probing import phase1, phase2, registry                                      # noqa: E402

OUT_DIR = REPO_ROOT / "three_models_paper" / "figures" / "truncation"
MODELS = tuple(phase1.MODELS)
MODEL_NAMES = {m: phase1.model_spec(m).display_name for m in MODELS}
REPRESENTATIVE = ("monash_electricity_hourly", "uber_tlc_hourly", "monash_london_smart_meters")

# ---- the four readouts of a depth-l state, in the order every figure/table uses -------------
#      (label in tables, label in figures, colour, line style, line width)
ARMS = {
    "hard": ("Hard cut", "Hard truncation", "#8C8C8C", (0, (3, 1.6)), 1.0),
    "noa": ("Label-free alignment", "Label-free alignment + native head", "#009E73", "-", 1.9),
    "fl": ("Supervised alignment", "Supervised alignment + native head", "#0072B2", "-", 1.0),
    "probe": ("Probe head", "Probe head (replaces native head)", "#E69F00", "-", 1.15),
    "ra": ("Hidden-state alignment", "Hidden-state alignment (diagnostic)", "#CC79A7", "-", 0.9),
}
LADDER_RUNGS = ("hard", "noa", "fl", "probe")          # RA is a diagnostic: appendix only
BUDGET_WORD = {0.02: "Two", 0.05: "Five", 0.1: "Ten", 0.2: "Twenty"}

# ---- style: the Phase-1 figure scripts' settings (5.5in ICLR \textwidth, 1:1 point sizes) ----
TEXTWIDTH_IN = 5.5
RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
      "ytick.labelsize": 6.5, "legend.fontsize": 7.0, "axes.linewidth": 0.7,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
      "lines.linewidth": 1.3, "pdf.fonttype": 42, "ps.fonttype": 42}
Y_CAP = 300.0             # % MASE increase: larger values are drawn as triangles at the cap
X_CAP = 16.0              # x speedup cap: TiRex with no blocks kept (~340x) is annotated instead
ENTRANCE_C = "#2E7D32"    # the Phase-1 entrance marker, as in the Phase-1 figures


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _rows(p: Path):
    if not p.exists():
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        v = float(x)
        return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _true(x) -> bool:
    return str(x) == "True"


def _minus(txt: str) -> str:
    """A typeset minus sign instead of a text hyphen for negative numbers."""
    return "$-$" + txt[1:] if txt.startswith("-") else txt


def _pct(x, digits=1):
    """A degradation (ratio - 1) as signed percent; thousands get a LaTeX thin separator."""
    if x is None:
        return "--"
    v = 100 * x
    if abs(v) >= 1000:
        return _minus(("+" if v >= 0 else "-") + f"{abs(v):,.0f}".replace(",", "{,}") + "\\%")
    return _minus(f"{v:+.{digits}f}\\%")


def _num(x, digits=1):
    """A bare signed percent number (no % sign), for bracketed intervals."""
    if x is None:
        return "--"
    v = 100 * x
    if abs(v) >= 1000:
        return _minus(("+" if v >= 0 else "-") + f"{abs(v):,.0f}".replace(",", "{,}"))
    return _minus(f"{v:+.{digits}f}")


def _sig(v, n=2):
    """Round to n significant digits (for figure annotations)."""
    if v == 0:
        return 0.0
    return float(f"{v:.{n}g}")


def _disp(label) -> str:
    """The input-embedding output is depth 0 of the block axis; the paper calls it L0."""
    return "L0" if str(label) == "Emb" else str(label)


def _names():
    return {t: registry.display_name(t) for t in registry.tags("paper14")}


def _order(present):
    """Datasets in the paper's roster order; anything unknown to the roster goes last."""
    roster = list(registry.tags("paper14"))
    return [t for t in roster if t in present] + sorted(t for t in present if t not in roster)


def _dname(tag):
    try:
        return registry.display_name(tag)
    except KeyError:
        return tag


def _missing(path: Path, what: str,
             why: str = "NO DATA YET -- run the Phase-2 jobs, then make_phase2_tables"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"% {what}: {why}\n\\PhaseTwoMissing{{{what}}}\n")


def _write(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _save(fig, out_dir: Path, stem: str, dpi: int = 300):
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf")
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi)
    plt.close(fig)
    return [f"{stem}.pdf", f"{stem}.png"]


def _final_index(model, rows=()):
    try:
        return int(phase1.model_spec(model).reference_index)
    except Exception:                                    # noqa: BLE001 -- mock models in tests
        return max(int(r["depth_index"]) for r in rows) if rows else 12


# --------------------------------------------------------------------------- #
# the shared degradation axis: signed percent, linear to 20 %, logarithmic beyond
# --------------------------------------------------------------------------- #
def _degradation_axis(ax, lo=-3.0, cap=Y_CAP):
    import matplotlib.ticker as mt
    ax.set_yscale("symlog", linthresh=20, linscale=2.0)
    ax.set_ylim(min(lo, -3.0), cap * 1.2)
    ticks = [t for t in (-20, -10, 0, 5, 10, 20, 50, 100, 300) if min(lo, -3.0) <= t <= cap]
    ax.yaxis.set_major_locator(mt.FixedLocator(ticks))
    ax.yaxis.set_major_formatter(mt.FuncFormatter(
        lambda v, _: "0" if v == 0 else (f"+{v:g}%" if v > 0 else f"{v:g}%")))
    ax.yaxis.set_minor_locator(mt.NullLocator())
    for b in phase2.FRONTIER_BUDGETS:                    # the Table H4 budgets, as faint guides
        ax.axhline(100 * b, color="#D2D2D2", lw=0.5, ls=(0, (1, 1.5)), zorder=0)
    ax.axhline(0, color="black", lw=0.6, zorder=1)


def _plot_capped(ax, x, y, color, ls, lw, ms=1.9, zorder=3, cap=Y_CAP):
    """Line in the given order; points above the cap are drawn AT the cap as triangles."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ax.plot(x, np.minimum(y, cap), color=color, ls=ls, lw=lw, marker="o", ms=ms, zorder=zorder)
    off = y > cap
    if off.any():
        ax.plot(x[off], np.full(off.sum(), cap), ls="none", marker="^", ms=3.0, color=color,
                zorder=zorder + 1, clip_on=False)


# --------------------------------------------------------------------------- #
# H3 tables
# --------------------------------------------------------------------------- #
def ladder_table(summary, out: Path):
    """MAIN H3 table: at the frozen Phase-1 entrance, how far each readout of h_l is from the
    full model. Aggregates only datasets whose entrance lies BEFORE the final block."""
    rows = [r for r in summary if _true(r["includes_low_skill"])]
    if not rows:
        return _missing(out, "h3_ladder_table")
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             "% n = datasets whose Phase-1 entrance lies BEFORE the final block; a dataset whose "
             "entrance IS the final block has no intermediate state to test and is counted in the "
             "model's header row instead",
             "\\begin{tabular}{lrrrr}", "\\toprule",
             " & \\multicolumn{2}{c}{Test increase vs.\\ full model} & & \\\\",
             "\\cmidrule(lr){2-3}",
             "Readout at the entrance & $\\Delta$MASE median [IQR] & "
             "$\\Delta$WQL median & $\\le$5\\% & Gap closed \\\\", "\\midrule"]
    for m in MODELS:
        rm = [r for r in rows if r["model"] == m]
        if not rm:
            continue
        n = rm[0]["n_datasets"]
        nf = int(rm[0]["n_entrance_is_final_block"] or 0)
        extra = f"; {nf} with entrance at the final block, not counted" if nf else ""
        lines.append(f"\\multicolumn{{5}}{{l}}{{\\textbf{{{MODEL_NAMES[m]}}} "
                     f"($n={n}$ datasets{extra})}} \\\\")
        for key in LADDER_RUNGS:
            r = next((x for x in rm if x["rung"] == key), None)
            if r is None:
                if key == "fl" and m == "timesfm3":
                    lines.append(f"\\quad {ARMS[key][0]} & \\multicolumn{{4}}{{l}}"
                                 "{\\textit{identical to the probe head (linear native head)}} \\\\")
                continue
            q = [_f(r["mase_degradation_q25"]), _f(r["mase_degradation_median"]),
                 _f(r["mase_degradation_q75"])]
            gc = _f(r["gap_closure_median"])
            lines.append(f"\\quad {ARMS[key][0]} & {_pct(q[1])} [{_num(q[0])}, {_num(q[2])}] & "
                         f"{_pct(_f(r['wql_degradation_median']))} & "
                         f"{r['n_within_5pct_mase']}/{r['n_datasets']} & "
                         f"{'--' if gc is None else f'{100 * gc:.0f}' + chr(92) + '%'} \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write(out, lines)


def ladder_full_table(ladder, out: Path):
    """APPENDIX: every model x dataset at its entrance. The label-free arm (the arm the frozen
    entrance test decides on) carries its paired 95% cluster-bootstrap interval."""
    head = [r for r in ladder if _true(r["is_headline"])]
    if not head:
        return _missing(out, "h3_ladder_full")
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             "% the hidden-state diagnostic is summarized in the text (per dataset: h3_ladder.csv)",
             "\\begin{tabular}{llrrrrr}", "\\toprule",
             " & & \\multicolumn{5}{c}{Test $\\Delta$MASE vs.\\ full model (\\%) at the entrance} \\\\",
             "\\cmidrule(lr){3-7}",
             "Dataset & Entr. & Hard cut & Label-free [95\\% CI] & Gap closed & Supervised & "
             "Probe head \\\\", "\\midrule"]
    for m in MODELS:
        hm = [r for r in head if r["model"] == m]
        if not hm:
            continue
        lines.append(f"\\multicolumn{{7}}{{l}}{{\\textbf{{{MODEL_NAMES[m]}}} "
                     f"(final block L{_final_index(m, hm)})}} \\\\")
        for tag in _order({r["dataset"] for r in hm}):
            rr = {r["rung"]: r for r in hm if r["dataset"] == tag}
            any_r = next(iter(rr.values()))
            name = _dname(tag) + ("$^{*}$" if _true(any_r["low_skill"]) else "")
            lab = _disp(any_r["label"])
            if int(any_r["blocks_removed"]) == 0:
                lines.append(f"{name} & {lab} & \\multicolumn{{5}}{{c}}{{\\textit{{entrance is the "
                             "final block: nothing to remove}} \\\\")
                continue

            def pt(key):
                r = rr.get(key)
                return "--" if r is None else _num(_f(r["mase_ratio"]) - 1)
            n = rr.get("noa")
            noa = "--" if n is None else (
                f"{_num(_f(n['mase_ratio']) - 1)} [{_num(_f(n['mase_ci_lo']) - 1)}, "
                f"{_num(_f(n['mase_ci_hi']) - 1)}]")
            gc = None if n is None else _f(n["gap_closure_mase"])
            fl = pt("fl") if "fl" in rr else ("= probe" if m == "timesfm3" else "--")
            lines.append(f"{name} & {lab} & {pt('hard')} & {noa} & "
                         f"{'--' if gc is None else f'{100 * gc:.0f}'} & {fl} & {pt('probe')} "
                         "\\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write(out, lines)


def compatibility_table(tunnels, out: Path, tol="tol_0.05"):
    """APPENDIX: the compatibility entrance (Phase-1's sustained operator on VALIDATION loss, with
    the native model as the reference) and its lag behind the recoverability entrance."""
    rows = [r for r in tunnels if r["tolerance"] == tol]
    if not rows:
        return _missing(out, "h3_compatibility")
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             "% validation only; lag = compatibility entrance - Phase-1 recoverability entrance "
             "(blocks); 'before final' = datasets whose readout stays within 5% of the native "
             "model's validation loss from some block before the final one onwards",
             "\\begin{tabular}{llrrr}", "\\toprule",
             "Model & Readout & Before final & Lag median [IQR] & Lag $\\le 1$ \\\\", "\\midrule"]
    for m in MODELS:
        rm = [r for r in rows if r["model"] == m]
        if not rm:
            continue
        first = True
        for key in ("hard", "noa", "fl", "probe", "ra"):
            rk = [r for r in rm if r["family"] == key]
            if not rk:
                continue
            lag = np.array([int(r["compatibility_lag"]) for r in rk], float)
            before = sum(not _true(r["no_truncation"]) for r in rk)
            q = np.percentile(lag, [25, 50, 75])
            q1, q0, q2 = (_minus(f"{v:+.1f}") for v in (q[1], q[0], q[2]))
            lines.append(f"{MODEL_NAMES[m] if first else ''} & {ARMS[key][0]} & {before}/{len(rk)} & "
                         f"{q1} [{q0}, {q2}] & {int((lag <= 1).sum())}/{len(rk)} \\\\")
            first = False
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write(out, lines)


def selection_table(depth_rows, out: Path):
    """APPENDIX: what the validation split selected for every truncation depth (< final)."""
    if not depth_rows:
        return _missing(out, "h3_adapter_selection")
    last = phase2.ADAPTER_EPOCHS
    wd_max = max(phase2.ADAPTER_WD_GRID)
    kmin, kmax = min(phase2.RIDGE_KAPPAS), max(phase2.RIDGE_KAPPAS)
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             f"% iterative fits: AdamW lr {phase2.ADAPTER_LR:g}, {last} epochs, decay grid "
             f"{list(phase2.ADAPTER_WD_GRID)}; closed form: kappa in [{kmin:g}, {kmax:g}]",
             "% 'At grid edge': iterative = decay at the grid maximum; closed form = kappa at "
             "either end of its grid",
             "\\begin{tabular}{llrrrrrr}", "\\toprule",
             "Model & Adapter & Fits & Params & Med.\\ epoch & Late & Grid edge & Hard cut \\\\",
             "\\midrule"]
    for m in MODELS:
        L = _final_index(m, [r for r in depth_rows if r["model"] == m])
        first = True
        for key in ("noa", "fl", "ra"):
            rr = [r for r in depth_rows if r["model"] == m and r["family"] == key
                  and int(r["depth_index"]) < L]
            if not rr:
                continue
            eps = [_f(r["selected_epoch"]) for r in rr if _f(r["selected_epoch"]) is not None]
            wds = [_f(r["selected_wd"]) for r in rr if _f(r["selected_wd"]) is not None]
            kap = [_f(r["selected_kappa"]) for r in rr if _f(r["selected_kappa"]) is not None]
            hc = sum(_true(r["selected_hard_cut"]) for r in rr)
            npar = int(float(rr[0]["n_adapter_params"] or 0))
            if eps:
                med = f"{np.median(eps):.0f}"
                late = f"{sum(e >= 0.9 * last for e in eps)}/{len(eps)}"
                atmax = f"{sum(w >= wd_max for w in wds)}/{len(wds)}"
            else:                                         # closed form: report the ridge edge
                med = "closed form"
                late = "--"
                atmax = (f"{sum(k in (kmin, kmax) for k in kap)}/{len(kap)}"
                         if kap else "--")
            short = {"noa": "Label-free", "fl": "Supervised", "ra": "Hidden-state"}[key]
            lines.append(f"{MODEL_NAMES[m] if first else ''} & {short} & {len(rr)} & "
                         f"{npar / 1e6:.2f}M & {med} & {late} & {atmax} & {hc}/{len(rr)} \\\\")
            first = False
        if not first:
            lines.append("\\midrule")
    if lines[-1] != "\\midrule":
        return _missing(out, "h3_adapter_selection")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write(out, lines)


# --------------------------------------------------------------------------- #
# H4 tables
# --------------------------------------------------------------------------- #
def budget_table(budget_summary, out: Path, physical=()):
    """MAIN H4 table: per accuracy budget, the VALIDATION-selected cut of every arm -- median
    measured speedup over ALL datasets (no qualifying cut = the full model, 1x), and how many of
    the datasets that were cut stay within the budget on held-out test."""
    rows = [r for r in budget_summary if _true(r["includes_low_skill"])]
    if not rows:
        return _missing(out, "h4_budget_table")
    eps = sorted({float(r["budget"]) for r in rows})
    arms = ("hard", "noa", "fl", "probe")
    x = "$\\times$"
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             "% per budget: median measured speedup (end-to-end, B=1) over ALL datasets (no cut = "
             "1x) ; within/cut = cut datasets within the budget on held-out test / datasets cut. "
             "The cut depth is chosen on VALIDATION MASE only (probing.phase2.BUDGET_RULE)",
             f"% physically instantiated + V3-verified operating points: "
             f"{sum(_true(r['V3_passed']) for r in physical)}/{len(physical)}"
             + ("" if physical else " (physical evaluation pending: numbers are the offline H3 "
                                    "computation, V3-verified at every depth by h4 verify)"),
             "\\begin{tabular}{l" + "rr" + "rrr" + "rr" + "rr" + "}", "\\toprule",
             " & \\multicolumn{2}{c}{Hard truncation} & "
             "\\multicolumn{3}{c}{Label-free alignment} & "
             "\\multicolumn{2}{c}{Supervised alignment} & \\multicolumn{2}{c}{Probe head} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-6}\\cmidrule(lr){7-8}\\cmidrule(lr){9-10}",
             "Budget & speedup & within/cut & speedup & within/cut & test $\\Delta$ & "
             "speedup & within/cut & speedup & within/cut \\\\", "\\midrule"]
    for m in MODELS:
        rm = [r for r in rows if r["model"] == m]
        if not rm:
            continue
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textbf{{{MODEL_NAMES[m]}}}}} \\\\")
        for e in eps:
            cells = []
            for arm in arms:
                r = next((y for y in rm if y["arm"] == arm and float(y["budget"]) == e), None)
                width = 3 if arm == "noa" else 2
                if r is None:
                    cells.append(" & ".join(["--"] * width))
                    continue
                sp = _f(r["median_speedup_e2e_b1"])
                c = [("--" if sp is None else f"{sp:.2f}{x}"),
                     f"{r['n_within_budget_test_cut']}/{r['n_cut']}"]
                if arm == "noa":
                    c.append(_pct(_f(r["cut_degradation_median"])) if int(r["n_cut"]) else "--")
                cells.append(" & ".join(c))
            lines.append(f"\\quad {100 * e:g}\\% & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write(out, lines)


def budget_points_table(points, out: Path, arm="noa"):
    """APPENDIX: per dataset, the depth each budget selected (validation) and its test MASE
    increase; a dagger marks a cut that exceeds its budget on test."""
    pts = [p for p in points if p["arm"] == arm]
    if not pts:
        return _missing(out, "h4_budget_points")
    eps = sorted({float(p["budget"]) for p in pts})
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             f"% arm: {arm}; 'full' = no depth satisfies the budget on validation, so the operating "
             "point is the full model; dagger = test increase above the budget",
             "\\begin{tabular}{l" + "lr" * len(eps) + "}", "\\toprule",
             " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\epsilon={100 * e:g}\\%$}}"
                                for e in eps) + " \\\\",
             "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(eps))),
             "Dataset" + " & cut & test $\\Delta$" * len(eps) + " \\\\", "\\midrule"]
    for m in MODELS:
        pm = [p for p in pts if p["model"] == m]
        if not pm:
            continue
        lines.append(f"\\multicolumn{{{1 + 2 * len(eps)}}}{{l}}{{\\textbf{{{MODEL_NAMES[m]}}}}} \\\\")
        for tag in _order({p["dataset"] for p in pm}):
            cells = []
            for e in eps:
                p = next((q for q in pm if q["dataset"] == tag and float(q["budget"]) == e), None)
                if p is None or not _true(p["truncated"]):
                    cells.append("full & --")
                    continue
                d = _f(p["test_mase_ratio"]) - 1
                dag = "$^{\\dagger}$" if not _true(p["within_budget_test"]) else ""
                cells.append(f"{_disp(p['label'])} & {_num(d)}{dag}")
            low = any(_true(p["low_skill"]) for p in pm if p["dataset"] == tag)
            lines.append(f"{_dname(tag)}{'$^{*}$' if low else ''} & " + " & ".join(cells) + " \\\\")
        lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write(out, lines)


def latency_table(lookup, out: Path):
    """APPENDIX: absolute cost of the full models (and Chronos-2-small), and the no-block floor
    (the model with every block removed: tokenizer + head + the package's own pre/post-processing)."""
    if not lookup:
        return _missing(out, "h4_latency_table")

    def get(model, kind, depth, level, b, field="median_ms"):
        for r in lookup:
            if (r["model"] == model and r["kind"] == kind and r["level"] == level
                    and int(r["batch"]) == b and (depth is None or str(r["depth"]) == str(depth))):
                return _f(r[field])
        return None
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             "% medians of the per-job medians (3 independent jobs x 100 timed calls); A100-SXM4-40GB,"
             " float32, TF32 off, eager PyTorch",
             "\\begin{tabular}{lrrrrrrr}", "\\toprule",
             " & & & \\multicolumn{2}{c}{Batch size 1 (ms)} & \\multicolumn{2}{c}{Batch size 256} "
             "& No-block floor \\\\",
             "\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
             "Model & Blocks & Params & end-to-end & forward & ms & series/s & B=1 (ms) \\\\",
             "\\midrule"]
    for m in MODELS:
        rn = [r for r in lookup if r["model"] == m and r["kind"] == "native"]
        if not rn:
            continue
        L = int(rn[0]["depth"])
        e1, f1 = get(m, "native", None, "api_e2e", 1), get(m, "native", None, "device_forward", 1)
        e256 = get(m, "native", None, "api_e2e", 256)
        tp = get(m, "native", None, "api_e2e", 256, "throughput")
        par = get(m, "native", None, "api_e2e", 1, "active_params")
        floor = get(m, "hard", 0, "api_e2e", 1)
        lines.append(f"{MODEL_NAMES[m]} & {L} & {par / 1e6:.1f}M & {e1:.1f} & {f1:.1f} & "
                     f"{e256:.1f} & {tp:,.0f} & {'--' if floor is None else f'{floor:.1f}'} \\\\"
                     .replace(",", "{,}"))
        if m == "chronos2" and get(m, "chronos2_small", None, "api_e2e", 1) is not None:
            s1, sf = get(m, "chronos2_small", None, "api_e2e", 1), get(m, "chronos2_small", None,
                                                                       "device_forward", 1)
            s256 = get(m, "chronos2_small", None, "api_e2e", 256)
            stp = get(m, "chronos2_small", None, "api_e2e", 256, "throughput")
            sp = get(m, "chronos2_small", None, "api_e2e", 1, "active_params")
            lines.append(f"\\quad Chronos-2-small & 6 & {sp / 1e6:.1f}M & {s1:.1f} & {sf:.1f} & "
                         f"{s256:.1f} & {stp:,.0f} & -- \\\\".replace(",", "{,}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out, lines)


def truncation_table(joined, out: Path):
    """APPENDIX (needs H4 evaluate): the physically truncated models at the frozen entrance."""
    if not joined:
        return _missing(out, "h4_truncation_table")
    arms = [("hard", "Hard-cut truncation"), ("noa", "Label-free aligned truncation"),
            ("fl", "Supervised aligned truncation"), ("probe_head", "Probe-head truncation"),
            ("chronos2_small", "Chronos-2-small")]
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             "% truncation rows aggregate the datasets whose Phase-1 entrance lies BEFORE the "
             "final block (a truncated model exists); Chronos-2-small: every dataset",
             "\\begin{tabular}{llrrrrrr}", "\\toprule",
             "Model & Model variant & blocks removed & params removed & speedup B=1 & "
             "speedup B=256 & $\\Delta$MASE median & within 5\\% \\\\", "\\midrule"]
    for m in MODELS:
        for key, name in arms:
            rr = [r for r in joined if r["model"] == m and r["arm"] == key]
            if key != "chronos2_small":
                rr = [r for r in rr if (_f(r["blocks_removed"]) or 0) > 0]
            if not rr:
                continue

            def med(k):
                v = [_f(r[k]) for r in rr if _f(r[k]) is not None]
                return float(np.median(v)) if v else None
            s1, s256 = med("api_e2e_B1_speedup"), med("api_e2e_B256_speedup")
            x = "$\\times$"                       # no backslash inside an f-string expression
            sp1 = "--" if s1 is None else f"{s1:.2f}{x}"
            sp256 = "--" if s256 is None else f"{s256:.2f}{x}"
            mr = med("mase_ratio")
            nwin = sum(_true(r["within_budget_mase"]) for r in rr)
            lines.append(
                f"{MODEL_NAMES[m]} & {name} & {med('blocks_removed') or 0:.0f} & "
                f"{_pct(med('fraction_params_removed'))} & {sp1} & {sp256} & "
                f"{_pct(None if mr is None else mr - 1)} & {nwin}/{len(rr)} " + "\\\\")
        if lines[-1] != "\\midrule":          # a model with no rows adds no rule
            lines.append("\\midrule")
    lines[-1] = "\\bottomrule"
    lines.append("\\end{tabular}")
    _write(out, lines)


def outcome_table(outcome_summary, out: Path):
    """APPENDIX (needs H4 evaluate): the pre-registered verdict at the frozen Phase-1 entrance.
    Failures are counted, never hidden; a cell that could not be truncated is its own column."""
    rows = [r for r in outcome_summary if _true(r["includes_low_skill"])]
    if not rows:
        return _missing(out, "h4_outcome_table")
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             "% rule (probing.phase2.H4_OUTCOME_RULE): candidate depth = the frozen Phase-1 5% "
             "entrance; success = label-free aligned truncation dMASE <= 5%; a failure is "
             "reported, never replaced by another depth",
             "\\begin{tabular}{lrrrrr}", "\\toprule",
             "Model & datasets & truncatable & success & failure (fragile) & "
             "entrance = final \\\\", "\\midrule"]
    for m in MODELS:
        r = next((x for x in rows if x["model"] == m), None)
        if r is None:
            continue
        lines.append(f"{MODEL_NAMES[m]} & {r['n_datasets']} & {r['n_truncatable']} & "
                     f"{r['n_success']} & {r['n_failure']} ({r['n_failure_fragile']}) & "
                     f"{r['n_no_truncation_possible']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    _write(out, lines)


# --------------------------------------------------------------------------- #
# H4 figures: the accuracy-compute frontier
# --------------------------------------------------------------------------- #
XLABEL = {"speedup_e2e_b1": "Measured end-to-end speedup, batch size 1 (log scale)",
          "speedup_e2e_b256": "Measured end-to-end throughput gain, batch size 256 (log scale)",
          "speedup_device_b1": "Measured forward-pass speedup, batch size 1 (log scale)",
          "params_removed_fraction": "Fraction of parameters removed",
          "fraction_blocks_removed": "Fraction of blocks removed (latency not measured yet)"}
FRONTIER_ARM_ORDER = ("hard", "probe", "fl", "noa")      # draw order: primary arm on top


def _frontier_panel(ax, model, rows, xkey, ykey=("degradation_q25", "degradation_median",
                                                 "degradation_q75"), small=None):
    """One model: median test increase (IQR band for the label-free arm) against the cost axis,
    points joined in DEPTH order (final block -> L0). A cost not measured yet falls back to the
    fraction of blocks removed. Returns the annotations it had to make (off-axis points)."""
    import matplotlib.ticker as mt
    rm = [r for r in rows if r["model"] == model]
    ax.set_title(MODEL_NAMES.get(model, model), fontweight="bold", pad=3)
    if not rm:
        ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center", va="center",
                fontsize=7, color="#777777")
        ax.set_xticks([])
        ax.set_yticks([])
        return {}
    use = xkey if all(_f(r.get(xkey)) is not None for r in rm) else "fraction_blocks_removed"
    log_x = use.startswith("speedup")
    x0 = 1.0 if log_x else 0.0
    ys, xs, notes = [0.0], [x0], {}
    by_depth_hard = {}
    for arm in FRONTIER_ARM_ORDER:
        ra = sorted((r for r in rm if r["arm"] == arm), key=lambda r: -int(r["depth_index"]))
        if not ra:
            continue
        _, _, col, ls, lw = ARMS[arm]
        x = np.array([x0] + [_f(r[use]) for r in ra], float)
        q = {k: np.array([0.0] + [np.nan if _f(r[k]) is None else 100 * _f(r[k]) for r in ra],
                         float) for k in ykey}
        lo, med, hi = (q[k] for k in ykey)
        if arm == "hard":
            by_depth_hard = {int(r["depth_index"]): _f(r[use]) for r in ra}
        keep = x <= X_CAP if log_x else np.ones_like(x, bool)
        if not keep.all():                                     # e.g. TiRex L0 at ~340x
            notes[arm] = (float(x[~keep].min()), float(x[~keep].max()),
                          float(med[~keep].min()), float(med[~keep].max()))
        if arm == "noa":
            o = np.argsort(x[keep])
            ax.fill_between(x[keep][o], np.minimum(lo[keep], Y_CAP)[o],
                            np.minimum(hi[keep], Y_CAP)[o], color=col, alpha=0.16, lw=0,
                            zorder=1)
        _plot_capped(ax, x[keep], med[keep], col, ls, lw)
        if (med[keep] > Y_CAP).any():
            notes[f"{arm}_ycap"] = (float(med[keep][med[keep] > Y_CAP].min()),
                                    float(med[keep][med[keep] > Y_CAP].max()))
        ys += list(np.minimum(lo[keep], Y_CAP))
        xs += list(x[keep])
    ax.plot([x0], [0.0], marker="*", ms=8, color="black", ls="none", zorder=6, clip_on=False)
    if small and small.get(use) is not None and small.get("degradation") is not None:
        ax.plot([small[use]], [100 * small["degradation"]], marker="D", ms=4.2, color="#2166AC",
                mec="white", mew=0.5, ls="none", zorder=6)
        xs.append(small[use])
    _degradation_axis(ax, lo=float(np.nanmin(ys)) - 2)
    if log_x:
        ax.set_xscale("log")
        xmax = max(xs) * 1.1
        ax.set_xlim(0.96, xmax)
        ticks = ([1, 1.2, 1.4] if xmax < 1.8 else [1, 1.5, 2, 2.5] if xmax < 3 else
                 [1, 2, 3, 4, 6] if xmax < 7 else [1, 2, 4, 8, 16])
        ax.xaxis.set_major_locator(mt.FixedLocator([t for t in ticks if t <= xmax]))
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:g}×"))
        ax.xaxis.set_minor_locator(mt.NullLocator())
        if by_depth_hard:                      # which block each position on the axis is
            L = max(by_depth_hard) + 1
            pick = [9, 6, 3, 1] if L <= 12 else [18, 12, 6, 0]
            pick = [d for d in pick if d in by_depth_hard and by_depth_hard[d] <= min(X_CAP, xmax)]
            top = ax.secondary_xaxis("top")
            top.set_xticks([by_depth_hard[d] for d in pick], [f"L{d}" for d in pick])
            top.set_xticks([], minor=True)
            top.tick_params(labelsize=5.8, length=2, pad=1, colors="#666666")
    else:
        ax.set_xlim(-0.02, 1.02)
        ax.xaxis.set_major_locator(mt.FixedLocator([0, 0.25, 0.5, 0.75, 1.0]))
        ax.xaxis.set_major_formatter(mt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    for s in ("right",):
        ax.spines[s].set_visible(False)
    return {"axis": use, **notes}


def _annotate_offaxis(ax, notes):
    """Say in the panel what could not be drawn on its axes."""
    ycap = [v for k, v in notes.items() if k.endswith("_ycap") and k.startswith("hard")]
    if ycap:
        lo, hi = ycap[0]
        ax.text(0.97, 0.13, f"\u25b2 hard truncation, clipped\n(+{_sig(lo):,.0f}% to "
                f"+{_sig(hi):,.0f}%)", transform=ax.transAxes, ha="right", va="bottom",
                fontsize=5.6, color="#6E6E6E", linespacing=0.95)
    offx = [v for k, v in notes.items() if k in ARMS]
    if offx:
        xlo, xhi = min(v[0] for v in offx), max(v[1] for v in offx)
        rng = f"{xlo:.0f}×" if round(xlo) == round(xhi) else f"{xlo:.0f}–{xhi:.0f}×"
        ax.text(0.97, 0.13, f"L0 (no blocks kept)\nat {rng}, off axis",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6, color="#6E6E6E",
                linespacing=0.95)


def _frontier_legend(fig, small=None, ncol=2):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Line2D([], [], marker="*", ms=7, color="black", ls="none", label="Full model")]
    for arm in ("noa", "fl", "probe", "hard"):
        _, name, col, ls, lw = ARMS[arm]
        h.append(Line2D([], [], color=col, ls=ls, lw=lw, label=name))
    h.append(Patch(facecolor=ARMS["noa"][2], alpha=0.16, lw=0,
                   label="IQR over datasets (label-free)"))
    if small:
        h.append(Line2D([], [], marker="D", ms=4, color="#2166AC", ls="none",
                        label="Chronos-2-small"))
    fig.legend(handles=h, loc="outside lower center", ncol=ncol, frameon=False,
               handlelength=2.0, columnspacing=1.0, borderpad=0.1)


def frontier_figure(summary, out_dir: Path, small=None, xkey="speedup_e2e_b1",
                    stem="h4_frontier"):
    """MAIN H4 figure: accuracy-compute frontier, one panel per model."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summary if _true(r["includes_low_skill"])]
    if not rows:
        return _missing(out_dir / f"{stem}.tex", stem)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.35), layout="constrained")
        axis = xkey
        for ax, m in zip(axes, MODELS):
            notes = _frontier_panel(ax, m, rows, xkey, small=small if m == "chronos2" else None)
            _annotate_offaxis(ax, notes)
            axis = notes.get("axis", axis)
        axes[0].set_ylabel("Test MASE increase\nover full model (median)")
        axes[1].set_xlabel(XLABEL[axis], labelpad=3)
        _frontier_legend(fig, small=small)
        return _save(fig, out_dir, stem)


def frontier_appendix(summary, out_dir: Path, stem="h4_frontier_appendix"):
    """APPENDIX: the same frontier against throughput (B=256), forward-pass speedup (B=1) and the
    fraction of parameters removed (linear axis, the full model at 0)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [r for r in summary if _true(r["includes_low_skill"])]
    if not rows:
        return _missing(out_dir / f"{stem}.tex", stem)
    keys = ("speedup_e2e_b256", "speedup_device_b1", "params_removed_fraction")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(3, 3, figsize=(TEXTWIDTH_IN, 6.6), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.06, hspace=0.08)
        for i, key in enumerate(keys):
            axis = key
            for ax, m in zip(axes[i], MODELS):
                notes = _frontier_panel(ax, m, rows, key)
                _annotate_offaxis(ax, notes)
                axis = notes.get("axis", axis)
                if i:
                    ax.set_title("")
            axes[i][0].set_ylabel("Test MASE increase (median)")
            axes[i][1].set_xlabel(XLABEL[axis], fontsize=7.5)
        _frontier_legend(fig)
        return _save(fig, out_dir, stem)


def wql_frontier_rows(depth_rows, summary):
    """The frontier in WQL: per model x arm x depth, median [IQR] of the test WQL ratio - 1 over
    datasets (from the H3 depth metrics), at the SAME measured cost as the MASE frontier."""
    cost = {(r["model"], r["arm"], int(r["depth_index"])): r for r in summary
            if _true(r["includes_low_skill"])}
    out = []
    for (m, arm, d), c in sorted(cost.items()):
        v = [_f(r["wql_ratio_vs_native"]) for r in depth_rows
             if r["model"] == m and r["family"] == arm and int(r["depth_index"]) == d
             and _f(r["wql_ratio_vs_native"]) is not None]
        if not v:
            continue
        q = np.percentile(np.asarray(v) - 1.0, [25, 50, 75])
        out.append({**c, "degradation_q25": q[0], "degradation_median": q[1],
                    "degradation_q75": q[2], "n_datasets": len(v)})
    return out


def frontier_wql(depth_rows, summary, out_dir: Path, stem="h4_frontier_wql"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = wql_frontier_rows(depth_rows, summary)
    if not rows:
        return _missing(out_dir / f"{stem}.tex", stem)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.35), layout="constrained")
        axis = "speedup_e2e_b1"
        for ax, m in zip(axes, MODELS):
            notes = _frontier_panel(ax, m, rows, "speedup_e2e_b1")
            _annotate_offaxis(ax, notes)
            axis = notes.get("axis", axis)
        axes[0].set_ylabel("Test WQL increase\nover full model (median)")
        axes[1].set_xlabel(XLABEL[axis], labelpad=3)
        _frontier_legend(fig)
        return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# H3 figures
# --------------------------------------------------------------------------- #
def _depth_panel(ax, rr, entrance, L, show_x=True):
    """One model x dataset: test MASE increase of every readout at every depth (final block
    included), the label-free arm with its 95% band, the Phase-1 entrance marked."""
    import matplotlib.ticker as mt
    lows = []
    for arm in ("hard", "probe", "fl", "noa"):
        pts = sorted((int(r["depth_index"]), _f(r["mase_ratio_vs_native"]), _f(r["mase_ratio_ci_lo"]),
                      _f(r["mase_ratio_ci_hi"])) for r in rr
                     if r["family"] == arm and _f(r["mase_ratio_vs_native"]) is not None)
        if not pts:
            continue
        d, v, lo, hi = (np.array(a, float) for a in zip(*pts))
        _, _, col, ls, lw = ARMS[arm]
        if arm == "noa":
            ax.fill_between(d, np.minimum(100 * (lo - 1), Y_CAP), np.minimum(100 * (hi - 1), Y_CAP),
                            color=col, alpha=0.18, lw=0, zorder=1)
        _plot_capped(ax, d, 100 * (v - 1), col, ls, 0.8 if arm != "noa" else 1.3, ms=1.3)
        lows.append(np.nanmin(100 * (np.minimum(v, lo) - 1)))
    if entrance is not None:
        ax.axvline(entrance, color=ENTRANCE_C, lw=0.9, ls=(0, (3, 2)), zorder=0.5)
    _degradation_axis(ax, lo=(min(lows) - 2) if lows else -3)
    step = 4 if L > 12 else 2
    ax.set_xticks(range(0, L + 1, step))
    ax.set_xticklabels([f"L{i}" for i in range(0, L + 1, step)] if show_x else [])
    ax.set_xticks(range(L + 1), minor=True)
    ax.set_xlim(-0.5, L + 0.5)
    ax.tick_params(which="major", length=2.5, pad=1.5)
    ax.tick_params(which="minor", length=1.5)
    bottom = ax.get_ylim()[0]
    ax.yaxis.set_major_locator(mt.FixedLocator([t for t in (-20, -10, 0, 10, 20, 100, 300)
                                                if bottom <= t <= Y_CAP]))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _depth_legend_handles(models=MODELS):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = []
    for arm in ("noa", "fl", "probe", "hard"):
        if arm == "fl" and not any(m in phase2.FL_MODELS for m in models):
            continue                                   # TimesFM-3: supervised == probe
        _, name, col, ls, lw = ARMS[arm]
        h.append(Line2D([], [], color=col, ls=ls, lw=1.3 if arm == "noa" else 0.9, marker="o",
                        ms=2, label=name))
    h.append(Patch(facecolor=ARMS["noa"][2], alpha=0.18, lw=0, label="95% bootstrap CI"))
    h.append(Line2D([], [], color=ENTRANCE_C, lw=0.9, ls=(0, (3, 2)),
                    label="Phase-1 entrance (5%)"))
    return h


def _entrances(ladder):
    return {(r["model"], r["dataset"]): int(r["depth_index"]) for r in ladder
            if _true(r["is_headline"])}


def depth_curves(depth_rows, ladder, out_dir: Path, stem="h3_depth_curves"):
    """Three representative datasets x three models (the Phase-1 main-figure datasets)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not depth_rows:
        return _missing(out_dir / f"{stem}.tex", stem)
    ent = _entrances(ladder)
    with plt.rc_context({**RC, "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, axes = plt.subplots(len(REPRESENTATIVE), 3, figsize=(TEXTWIDTH_IN, 5.2),
                                 squeeze=False, layout="constrained")
        for i, tag in enumerate(REPRESENTATIVE):
            for j, m in enumerate(MODELS):
                ax = axes[i][j]
                rr = [r for r in depth_rows if r["model"] == m and r["dataset"] == tag]
                if not rr:
                    ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                            va="center", fontsize=7, color="#777777")
                    ax.set_xticks([])
                    ax.set_yticks([])
                    continue
                _depth_panel(ax, rr, ent.get((m, tag)), _final_index(m, rr),
                             show_x=i == len(REPRESENTATIVE) - 1)
                if i == 0:
                    ax.set_title(MODEL_NAMES[m], fontweight="bold", pad=2)
                if j == 0:
                    ax.set_ylabel(f"{_dname(tag)}\nTest MASE increase", fontsize=7)
        fig.legend(handles=_depth_legend_handles(), loc="outside lower center", ncol=2,
                   frameon=False, handlelength=2.0, columnspacing=1.2)
        return _save(fig, out_dir, stem)


def depth_figure_model(model, depth_rows, ladder, out_dir: Path):
    """APPENDIX: all datasets of one model, 4 x 4 grid in roster order (as the Phase-1 appendix
    figures); the last slots hold the legend."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rm = [r for r in depth_rows if r["model"] == model]
    stem = f"h3_depth_{model}"
    if not rm:
        return _missing(out_dir / f"{stem}.tex", stem)
    ent = _entrances(ladder)
    order = _order({r["dataset"] for r in rm})
    ncol = 4
    nrow = max(1, -(-(len(order) + 2) // ncol))
    L = _final_index(model, rm)
    with plt.rc_context({**RC, "axes.titlesize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, axes = plt.subplots(nrow, ncol, figsize=(TEXTWIDTH_IN, 1.4 * nrow),
                                 layout="constrained", squeeze=False)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.13)
        flat = axes.ravel()
        for i, tag in enumerate(order):
            ax = flat[i]
            show = i + ncol >= len(order)
            _depth_panel(ax, [r for r in rm if r["dataset"] == tag], ent.get((model, tag)), L,
                         show_x=show)
            ax.set_title(_dname(tag), fontweight="bold", pad=2).set_in_layout(False)
            if show:
                ax.set_xlabel("Block depth", fontsize=7, labelpad=3)
        for ax in flat[len(order):]:
            ax.remove()
        lax = fig.add_subplot(axes[0, 0].get_gridspec()[nrow - 1, len(order) % ncol:])
        lax.axis("off")
        lax.legend(handles=_depth_legend_handles((model,)), loc="center", ncol=1, frameon=False,
                   handlelength=1.8, fontsize=6.5, borderaxespad=0.0)
        fig.supylabel("Test MASE increase over the full model", fontsize=7.5)
        return _save(fig, out_dir, stem)


def ladder_figure(ladder, out_dir: Path, stem="h3_ladder_at_entrance"):
    """APPENDIX: every dataset's ladder at its entrance (thin lines) and the median (black)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    head = [r for r in ladder if _true(r["is_headline"]) and int(r["blocks_removed"]) > 0]
    if not head:
        return _missing(out_dir / f"{stem}.tex", stem)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.3), layout="constrained")
        for ax, m in zip(axes, MODELS):
            ax.set_title(MODEL_NAMES[m], fontweight="bold", pad=2)
            rungs = [k for k in LADDER_RUNGS if any(r["model"] == m and r["rung"] == k
                                                   for r in head)]
            if not rungs:
                ax.text(0.5, 0.5, "no data yet", transform=ax.transAxes, ha="center",
                        va="center", fontsize=7, color="#777777")
                continue
            lows = [0.0]
            for tag in _order({r["dataset"] for r in head if r["model"] == m}):
                ys = [next((100 * (_f(r["mase_ratio"]) - 1) for r in head if r["model"] == m
                            and r["dataset"] == tag and r["rung"] == k), np.nan) for k in rungs]
                _plot_capped(ax, range(len(rungs)), ys, "#B5B5B5", "-", 0.6, ms=1.2, zorder=2)
                lows.append(np.nanmin(ys))
            med = [np.nanmedian([100 * (_f(r["mase_ratio"]) - 1) for r in head
                                 if r["model"] == m and r["rung"] == k]) for k in rungs]
            _plot_capped(ax, range(len(rungs)), med, "black", "-", 1.8, ms=3.5, zorder=5)
            _degradation_axis(ax, lo=float(np.nanmin(lows)) - 2)
            ax.set_xticks(range(len(rungs)))
            short = {"hard": "Hard\ncut", "noa": "Label-\nfree", "fl": "Super-\nvised",
                     "probe": "Probe\nhead"}
            ax.set_xticklabels([short[k] for k in rungs], fontsize=6)
            ax.set_xlim(-0.3, len(rungs) - 0.7)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel("Test MASE increase at the\nPhase-1 entrance")
        return _save(fig, out_dir, stem)


# --------------------------------------------------------------------------- #
# macros + the statistics the prose quotes
# --------------------------------------------------------------------------- #
def macros(summary, outcome_summary, out: Path, budget_summary=()):
    lines = ["% generated by experiments.make_phase2_paper_tables -- do not edit",
             "% a missing Phase-2 artifact renders as a visible box (grep for PhaseTwoMissing)",
             "\\providecommand{\\PhaseTwoMissing}[1]{\\fbox{\\textbf{MISSING Phase-2 artifact: "
             "\\detokenize{#1}}}\\typeout{WARNING: Phase-2 artifact missing: \\detokenize{#1}}}"]
    for r in summary:
        if not _true(r["includes_low_skill"]):
            continue
        name = f"phtwo{r['model']}{r['rung']}".replace("chronos2", "chronos").replace(
            "timesfm3", "timesfm").replace("_", "")
        med = _f(r["mase_degradation_median"])
        lines.append(f"\\newcommand{{\\{name}MaseMedian}}{{{_pct(med)}}}")
        lines.append(f"\\newcommand{{\\{name}WithinFive}}{{{r['n_within_5pct_mase']}/"
                     f"{r['n_datasets']}}}")
    for r in outcome_summary:
        if not _true(r["includes_low_skill"]):
            continue
        base = f"phtwo{r['model']}".replace("chronos2", "chronos").replace("timesfm3", "timesfm")
        lines.append(f"\\newcommand{{\\{base}HfourSuccess}}{{{r['n_success']}/{r['n_truncatable']}}}")
        lines.append(f"\\newcommand{{\\{base}HfourFailure}}{{{r['n_failure']}/{r['n_truncatable']}}}")
        lines.append(f"\\newcommand{{\\{base}HfourNoTruncation}}{{{r['n_no_truncation_possible']}}}")
    # the budget sentence: "with a 10% validation budget, <arm> removes a median of X% of the
    # backbone for a Y x median measured speedup; Z of the n cut datasets stay within budget on test"
    for r in budget_summary:
        if not _true(r["includes_low_skill"]):
            continue
        word = BUDGET_WORD.get(round(float(r["budget"]), 4))
        if word is None:
            continue
        base = (f"phtwo{r['model']}{r['arm']}".replace("chronos2", "chronos")
                .replace("timesfm3", "timesfm").replace("_", "") + f"Budget{word}")
        fr, sp = _f(r["median_fraction_blocks_removed"]), _f(r["median_speedup_e2e_b1"])
        rem = "--" if fr is None else f"{100 * fr:.0f}" + "\\%"   # no backslash in an f-expr
        spd = "--" if sp is None else f"{sp:.2f}"
        lines.append(f"\\newcommand{{\\{base}Removed}}{{{rem}}}")
        lines.append(f"\\newcommand{{\\{base}Speedup}}{{{spd}}}")
        lines.append(f"\\newcommand{{\\{base}Within}}{{{r['n_within_budget_test_cut']}/{r['n_cut']}}}")
        lines.append(f"\\newcommand{{\\{base}Cut}}{{{r['n_cut']}/{r['n_datasets']}}}")
    _write(out, lines)


def stats(comb: Path, summary, fr_summ, bud_summ, points, lookup, depth_rows, tunnels):
    """Every number the Phase-2 prose quotes, keyed so a sentence can cite its source."""
    def fr(m, arm, label):
        r = next((x for x in fr_summ if x["model"] == m and x["arm"] == arm
                  and _disp(x["label"]) == label and _true(x["includes_low_skill"])), None)
        if r is None:
            return None
        return {k: _f(r[k]) for k in ("degradation_q25", "degradation_median", "degradation_q75",
                                      "speedup_e2e_b1", "speedup_e2e_b256", "speedup_device_b1",
                                      "params_removed_fraction", "fraction_blocks_removed")}

    by_point = {}
    for p in _rows(comb / "h4_frontier_points.csv"):
        v = _f(p["test_mase_degradation"])
        if v is not None:
            by_point.setdefault((p["model"], p["arm"], _disp(p["label"])), []).append(v)

    def within_counts(m, arm, label, eps=(0.05, 0.10)):
        v = by_point.get((m, arm, label), [])
        return {f"n_within_{e:g}": int(sum(x <= e for x in v)) for e in eps} | {"n": len(v)}

    out = {"schema": "phase2_paper_stats/v1",
           "source": "experiments.make_phase2_paper_tables from <output-root>/combined/*.csv",
           "h3_ladder": [{k: r[k] for k in r} for r in summary if _true(r["includes_low_skill"])],
           "h3_ladder_without_low_skill": [{k: r[k] for k in r} for r in summary
                                           if not _true(r["includes_low_skill"])],
           "frontier_points_quoted": {}, "budget": [{k: r[k] for k in r} for r in bud_summ],
           "latency_native": [], "selection": {}, "final_block_probe_gap": {}}
    for m in MODELS:
        for arm in ("hard", "noa", "fl", "probe", "ra"):
            for d in range(0, 21):
                lab = f"L{d}"
                v = fr(m, arm, lab)
                if v is not None:
                    out["frontier_points_quoted"][f"{m}/{arm}/{lab}"] = {
                        **v, **within_counts(m, arm, lab)}
        L = _final_index(m, [r for r in depth_rows if r["model"] == m])
        pr = [_f(r["mase_ratio_vs_native"]) for r in depth_rows if r["model"] == m
              and r["family"] == "probe" and int(r["depth_index"]) == L
              and _f(r["mase_ratio_vs_native"]) is not None]
        if pr:
            out["final_block_probe_gap"][m] = {
                "median": float(np.median(pr) - 1), "q25": float(np.percentile(pr, 25) - 1),
                "q75": float(np.percentile(pr, 75) - 1), "n": len(pr)}
        for key in ("noa", "fl", "ra"):
            rr = [r for r in depth_rows if r["model"] == m and r["family"] == key
                  and int(r["depth_index"]) < L]
            eps = [_f(r["selected_epoch"]) for r in rr if _f(r["selected_epoch"]) is not None]
            wds = [_f(r["selected_wd"]) for r in rr if _f(r["selected_wd"]) is not None]
            if rr:
                out["selection"][f"{m}/{key}"] = {
                    "n_fits": len(rr),
                    "median_epoch": float(np.median(eps)) if eps else None,
                    "n_epoch_in_last_10pct": int(sum(e >= 0.9 * phase2.ADAPTER_EPOCHS
                                                     for e in eps)) if eps else None,
                    "n_wd_at_grid_max": int(sum(w >= max(phase2.ADAPTER_WD_GRID)
                                                for w in wds)) if wds else None,
                    "n_hard_cut_selected": int(sum(_true(r["selected_hard_cut"]) for r in rr))}
    for r in lookup:
        if r["kind"] in ("native", "chronos2_small"):
            out["latency_native"].append({k: r[k] for k in ("model", "kind", "level", "batch",
                                                            "median_ms", "median_ms_min_job",
                                                            "median_ms_max_job", "throughput",
                                                            "peak_allocated_bytes",
                                                            "active_params", "n_jobs")})
    for r in lookup:
        if r["kind"] == "hard" and str(r["depth"]) == "0" and r["level"] == "api_e2e":
            out.setdefault("no_block_floor_ms", {})[f"{r['model']}/B{r['batch']}"] = _f(
                r["median_ms"])
    lag = {}
    for t in tunnels:
        if t["tolerance"] == "tol_0.05":
            lag.setdefault(f"{t['model']}/{t['family']}", []).append(int(t["compatibility_lag"]))
    out["compatibility_lag_5pct"] = {k: {"median": float(np.median(v)),
                                         "q25": float(np.percentile(v, 25)),
                                         "q75": float(np.percentile(v, 75)), "n": len(v)}
                                     for k, v in lag.items()}
    out["budget_points_noa"] = [{k: p[k] for k in ("model", "dataset", "budget", "truncated",
                                                   "label", "val_mase_ratio", "test_mase_ratio",
                                                   "within_budget_test", "speedup_e2e_b1")}
                                for p in points if p["arm"] == "noa"]
    return out


# --------------------------------------------------------------------------- #
def build(output_root, gen_dir=OUT_DIR) -> dict:
    comb = Path(output_root) / "combined"
    gen = Path(gen_dir)
    tab = gen / "tables"
    gen.mkdir(parents=True, exist_ok=True)
    summary = _rows(comb / "h3_summary_by_model.csv")
    ladder = _rows(comb / "h3_ladder.csv")
    depth_rows = _rows(comb / "h3_depth_metrics.csv")
    tunnels = _rows(comb / "h3_tunnels.csv")
    joined = _rows(comb / "h4_with_latency.csv")
    outcomes = _rows(comb / "h4_outcome_summary.csv")
    fr_summ = _rows(comb / "h4_frontier_summary.csv")
    bud_summ = _rows(comb / "h4_budget_summary.csv")
    points = _rows(comb / "h4_budget_points.csv")
    lookup = _rows(comb / "latency_lookup.csv")
    small = None
    srows = [r for r in joined if r.get("arm") == "chronos2_small"]
    if srows:
        deg = [_f(r["mase_ratio"]) - 1 for r in srows if _f(r.get("mase_ratio")) is not None]
        sp = [_f(r.get("api_e2e_B1_speedup")) for r in srows]
        tp = [_f(r.get("api_e2e_B256_speedup")) for r in srows]
        small = {"degradation": float(np.median(deg)) if deg else None,
                 "speedup_e2e_b1": float(np.median(sp)) if sp and None not in sp else None,
                 "speedup_e2e_b256": float(np.median(tp)) if tp and None not in tp else None}
        if small["degradation"] is None:
            small = None
    # ---- tables
    ladder_table(summary, tab / "h3_ladder_table.tex")
    ladder_full_table(ladder, tab / "h3_ladder_full.tex")
    compatibility_table(tunnels, tab / "h3_compatibility.tex")
    selection_table(depth_rows, tab / "h3_adapter_selection.tex")
    budget_points_table(points, tab / "h4_budget_points.tex")
    latency_table(lookup, tab / "h4_latency_table.tex")
    invalid = [f"{r['model']}: {r['invalid_datasets']}" for r in outcomes
               if _true(r["includes_low_skill"]) and int(r["n_invalid_v3"]) > 0]
    if invalid:
        why = ("invalid H4 cells (physical model failed V3) " + "; ".join(invalid)).replace("_", "-")
        fix = "INVALID H4 CELLS -- fix V3 and re-run evaluate before any H4 table is built"
        _missing(tab / "h4_truncation_table.tex", "h4_truncation_table: " + why, fix)
        _missing(tab / "h4_outcome_table.tex", "h4_outcome_table: " + why, fix)
    else:
        truncation_table(joined, tab / "h4_truncation_table.tex")
        outcome_table(outcomes, tab / "h4_outcome_table.tex")
    phys = _rows(comb / "h4_budget_physical.csv")
    bad = [f"{r['model']}/{r['dataset']}/{r['arm']}@{r['label']}" for r in phys
           if not _true(r["V3_passed"])]
    if bad:
        _missing(tab / "h4_budget_table.tex",
                 ("h4_budget_table: operating points failed V3 " + "; ".join(bad)).replace("_", "-"),
                 "A PHYSICAL OPERATING POINT DOES NOT REPRODUCE ITS OFFLINE METRICS -- fix before "
                 "any H4 table is built")
    else:
        budget_table(bud_summ, tab / "h4_budget_table.tex", physical=phys)
    # ---- figures
    frontier_figure(fr_summ, gen, small=small)
    frontier_appendix(fr_summ, gen)
    frontier_wql(depth_rows, fr_summ, gen)
    ladder_figure(ladder, gen)
    depth_curves(depth_rows, ladder, gen)
    for m in MODELS:
        depth_figure_model(m, depth_rows, ladder, gen)
    macros(summary, outcomes, gen / "phase2_macros.tex", budget_summary=bud_summ)
    st = stats(comb, summary, fr_summ, bud_summ, points, lookup, depth_rows, tunnels)
    st["h4_physical_operating_points"] = {"n": len(phys),
                                          "n_V3_passed": sum(_true(r["V3_passed"]) for r in phys)}
    st["chronos2_small"] = small
    (gen / "truncation_stats.json").write_text(json.dumps(st, indent=1, default=str))
    return {"out": str(gen), "files": sorted(str(p.relative_to(gen)) for p in gen.rglob("*")
                                             if p.is_file())}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default=str(phase2.DEFAULT_OUT_ROOT))
    p.add_argument("--out-dir", "--generated-dir", dest="out_dir", default=str(OUT_DIR))
    a = p.parse_args(argv)
    r = build(a.output_root, a.out_dir)
    print(f"[phase2 paper tables] {r['out']}: {len(r['files'])} files")
    for f in r["files"]:
        print("   ", f)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
