"""Phase 2c auto-generated report + draft rebuttal paragraph, from the LODO TABLES ONLY.

    python -m experiments.make_phase2_lodo_report     # login node, seconds; after make_phase2_lodo_tables

Reads   <root>/combined/{lodo_stats.json, lodo_per_dataset.csv}   (make_phase2_lodo_tables)
        <root>/lodo/runs/<job>/preflight.json                    (the driver's pre-fit gates)
Writes  <root>/report/report.md, <root>/report/rebuttal_draft.md

Every number is copied from the tables, which are rebuilt from cell artifacts whose reproduction
gates all passed (the table builder refuses a gate-failed cell). Interpretation and the final
rebuttal wording are written by hand. A partial run is stamped INCOMPLETE at the top of both files,
and the draft then states how many datasets it covers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase2_lodo as lodo                                           # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "results" / "three_model_phase2_lodo"
GROUP_NAMES = {"all": "all datasets", "held_out": "held out from Chronos-2 pretraining",
               "not_listed": "not listed in the pretraining documentation",
               "listed": "listed as Chronos-2 pretraining data"}
BOOL = {"True": True, "False": False, "true": True, "false": False}


def _num(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if v in BOOL:
        return BOOL[v]
    try:
        return float(v)
    except ValueError:
        return v


def _pct(x, digits=1) -> str:
    return "n/a" if x is None else f"{100 * float(x):+.{digits}f}%"


def _pp(x, digits=1) -> str:
    return "n/a" if x is None else f"{float(x):+.{digits}f} pp"


def _x(x) -> str:
    return "n/a" if x is None else f"{float(x):.2f}x"


def _md_table(rows, cols, head) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def load(root: Path) -> dict:
    root = Path(root)
    comb = root / "combined"
    sp, rp = comb / "lodo_stats.json", comb / "lodo_per_dataset.csv"
    if not sp.exists():
        raise SystemExit(f"no {sp}: run `python -m experiments.make_phase2_lodo_tables` first")
    stats = json.loads(sp.read_text())
    rows = []
    if rp.exists() and rp.stat().st_size:
        with open(rp) as fh:
            rows = [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]
    pre = []
    for p in sorted((root / lodo.KIND / "runs").glob("*/preflight.json")):
        try:
            pre.append({"job": p.parent.name, **json.loads(p.read_text())})
        except Exception:                                        # pragma: no cover
            continue
    return {"stats": stats, "rows": rows, "preflight": pre}


def _agg(stats, depth, group):
    return next((a for a in stats.get("aggregate", []) if a["depth"] == depth
                 and a["group"] == group), None)


def _iqr(a, arm) -> str:
    if a is None or a.get(f"{arm}_median") is None:
        return "n/a"
    return (f"{_pct(a[f'{arm}_median'])} [{_pct(a[f'{arm}_q1'])}, {_pct(a[f'{arm}_q3'])}]")


def report_lines(data) -> tuple[list[str], list[str]]:
    stats, rows, pre = data["stats"], data["rows"], data["preflight"]
    main = [r for r in rows if r.get("variant") == lodo.MAIN]
    sens = [r for r in rows if r.get("variant") == lodo.SENSITIVITY]
    n_exp_main = int(stats.get("n_expected_main", 42))
    n_exp_sens = int(stats.get("n_expected", 48)) - n_exp_main
    complete = bool(stats.get("complete"))
    targets = sorted({r["dataset"] for r in main})
    banner = [] if complete else [
        f"> **INCOMPLETE RESULTS.** {stats.get('n_cells', 0)} of {stats.get('n_expected', 48)} "
        f"cells are written ({len(main)} of {n_exp_main} main-protocol, {len(sens)} of "
        f"{n_exp_sens} sensitivity) over {len(targets)} of 14 target datasets. Every number below "
        "covers only what exists; do not quote it as the full experiment.", ""]
    missing = stats.get("missing", [])
    last_pf = pre[-1] if pre else None
    flags = {k: sum(bool(r.get(k)) for r in main)
             for k in ("late_checkpoint", "grid_max_wd", "selected_hard_cut")}

    agg_rows = []
    for group in ("all",) + lodo.PROVENANCE_GROUPS:
        for depth in lodo.LODO_DEPTHS:
            a = _agg(stats, depth, group)
            if a is None:
                continue
            agg_rows.append({
                "group": GROUP_NAMES[group], "depth": a["label"], "n": a["n"],
                "speed": _x(a.get("speedup_B1")),
                "hard": _iqr(a, "hard"), "pooled": _iqr(a, "pooled"),
                "specific": _iqr(a, "specific"),
                "imp": f"{a['pooled_improves_over_hard']} / {a['pooled_improves_over_hard_ci']}",
                "within": f"{a['pooled_within_5']} / {a['pooled_within_10']}",
                "minus": _pp(a.get("pooled_minus_specific_pp_median")),
                "gap": ("n/a" if a.get("pooled_gap_closure_median") is None else
                        f"{100 * a['pooled_gap_closure_median']:.0f}% (n={a['n_positive_hard_gap']})")})

    per = []
    for r in sorted(main, key=lambda r: (lodo.PROVENANCE_GROUPS.index(r["provenance_status"]),
                                         r["dataset"], r["depth"])):
        per.append({"ds": r["display"], "prov": r["provenance_status"].replace("_", " "),
                    "depth": r["label"], "full": f"{r['full_mase']:.3f}",
                    "hard": _pct(r["hard_dmase"]),
                    "pooled": f"{_pct(r['pooled_dmase'])} [{_pct(r['pooled_dmase_ci_lo'])}, "
                              f"{_pct(r['pooled_dmase_ci_hi'])}]",
                    "specific": f"{_pct(r['specific_dmase'])} [{_pct(r['specific_dmase_ci_lo'])}, "
                                f"{_pct(r['specific_dmase_ci_hi'])}]",
                    "ratio": f"{r['pooled_over_hard']:.3f} [{r['pooled_over_hard_ci_lo']:.3f}, "
                             f"{r['pooled_over_hard_ci_hi']:.3f}]",
                    "minus": f"{_pp(r['pooled_minus_specific_pp'])} "
                             f"[{_pp(r['pooled_minus_specific_pp_ci_lo'])}, "
                             f"{_pp(r['pooled_minus_specific_pp_ci_hi'])}]",
                    "flags": ",".join(k for k in ("late_checkpoint", "grid_max_wd",
                                                  "selected_hard_cut") if r.get(k)) or "-"})
    cons = [{"ds": c["display"], "prov": c["provenance_status"].replace("_", " "),
             "imp": f"{c['depths_improving_over_hard']}/{c['n_depths']}",
             "w10": f"{c['depths_within_10']}/{c['n_depths']}", "worst": c["worst_depth"],
             "max": _pct(c["max_pooled_dmase"])}
            for c in stats.get("depth_consistency", [])]
    sen = [{"ds": s["display"], "depth": s["label"], "a": _pct(s["unseen_series_dmase"]),
            "b": _pct(s["sibling_included_dmase"]), "d": _pp(s["difference_pp"]),
            "spec": _pct(s["specific_dmase"]), "hard": _pct(s["hard_dmase"])}
           for s in stats.get("sibling_sensitivity", [])]

    lines = [
        "# Phase 2c: pooled leave-one-dataset-out adapter transfer (Chronos-2), auto-generated "
        "report", "", *banner,
        "Generated by `python -m experiments.make_phase2_lodo_report` from the LODO tables "
        "(`combined/`), which are rebuilt from cell artifacts only. Numbers are facts; "
        "interpretation and the final rebuttal wording are written by hand.", "",
        "## Coverage and gates", "",
        f"* cells written: {stats.get('n_cells', 0)} of {stats.get('n_expected', 48)} "
        f"(main protocol {len(main)} of {n_exp_main}; Electricity sibling sensitivity "
        f"{len(sens)} of {n_exp_sens}); target datasets with a main cell: {len(targets)} of 14",
        f"* missing cells: {len(missing)}" + (f" (first: {missing[:6]})" if missing else ""),
        "* every tabulated cell passed all of its reproduction gates (full model vs Phase 1 and "
        "H3, hard cut and dataset-specific adapter vs H3, test rows, identity adapter); the table "
        "builder refuses a cell that did not",
        ("* pre-fit preflight (latest job " + str(last_pf["job"]) + "): "
         f"{last_pf['n_pairs'] - last_pf['n_failed']} of {last_pf['n_pairs']} (target, depth) "
         f"pairs passed{'; FAILED: ' + str(last_pf['failed']) if last_pf['n_failed'] else ''}")
        if last_pf else "* pre-fit preflight: no record found",
        f"* fit diagnostics over the main cells: late checkpoint (selected at or after 90% of the "
        f"epochs) {flags['late_checkpoint']}, decay at the grid maximum {flags['grid_max_wd']}, "
        f"hard cut selected (epoch 0) {flags['selected_hard_cut']}", "",
        "## Main protocol: test MASE change vs the full model (median [IQR] over datasets)", "",
        "Speedup = the measured end-to-end Chronos-2 adapter speedup at batch size 1 from the "
        "committed Phase-2 latency runs (same adapter shape, so it carries over). `pooled < hard` "
        "= datasets where pooled transfer beats the hard cut (point estimate / paired 95% CI "
        "excludes no improvement). Gap closure = median share of the hard-cut MASE gap recovered, "
        "over datasets whose hard-cut gap CI is above 0.", "",
        _md_table(agg_rows, ["group", "depth", "n", "speed", "hard", "pooled", "specific", "imp",
                             "within", "minus", "gap"],
                  ["datasets", "depth", "n", "speedup B=1", "hard cut", "pooled transfer",
                   "dataset-specific", "pooled < hard (point / CI)", "pooled within 5% / 10%",
                   "pooled minus specific (median)", "pooled gap closure"]), "",
        "## Per dataset (main protocol; 95% paired series-cluster bootstrap CIs)", "",
        _md_table(per, ["ds", "prov", "depth", "full", "hard", "pooled", "specific", "ratio",
                        "minus", "flags"],
                  ["dataset", "Chronos-2 pretraining", "depth", "full-model MASE", "hard cut",
                   "pooled transfer", "dataset-specific", "pooled / hard MASE",
                   "pooled minus specific", "fit flags"]) if per else "(no main cell yet)", "",
        "## Depth consistency per dataset (main protocol)", "",
        _md_table(cons, ["ds", "prov", "imp", "w10", "worst", "max"],
                  ["dataset", "pretraining", "depths where pooled beats hard",
                   "depths within 10%", "worst depth", "worst pooled change"])
        if cons else "(none yet)", "",
        "## Electricity sibling sensitivity (13 sources incl. the series sibling vs 12 without)",
        "",
        _md_table(sen, ["ds", "depth", "a", "b", "d", "spec", "hard"],
                  ["target", "depth", "unseen series (main)", "sibling included",
                   "difference", "dataset-specific", "hard cut"])
        if sen else "(sensitivity cells not written yet)", "",
        "## Fixed limitations (stated independently of the outcome)", "",
        "* The truncation depth (L4, L6, L9) is fixed in advance. Choosing a depth on an unseen "
        "dataset without calibration data is a different question, not answered here.",
        "* Unseen DATASET is true for every main cell; unseen SERIES is true for every main cell "
        "(the Electricity sibling is excluded) and false for the sensitivity cells. For targets "
        "listed as Chronos-2 pretraining data the frozen backbone may already know the series; "
        "'not listed' means undocumented, not unseen. Only the held-out group is unseen by the "
        "backbone as documented.",
        "* One adapter fit per cell (seed 0); variation across seeds is not measured. Sources "
        "weigh equally in the objective and in the selection criterion, whatever their size.",
        "* Chronos-2 only; the label-free objective matches the full model's own head outputs, so "
        "a pooled adapter cannot exceed what the full model forecasts.",
    ]

    def med(depth, group, arm):
        a = _agg(stats, depth, group)
        return None if a is None else a.get(f"{arm}_median")
    depths = [d for d in lodo.LODO_DEPTHS if _agg(stats, d, "all")]
    labels = [lodo.depth_label(d) for d in depths]
    held = sorted({r["display"] for r in main if r["provenance_status"] == "held_out"})
    scope = ("for each of the 14 datasets" if complete else
             f"for {len(targets)} of the 14 datasets (the remaining cells are still running)")

    def series(group, arm):
        return " / ".join(_pct(med(d, group, arm)) for d in depths)
    imp = "; ".join(f"{_agg(stats, d, 'all')['pooled_improves_over_hard']} of "
                    f"{_agg(stats, d, 'all')['n']} datasets at {lodo.depth_label(d)} "
                    f"({_agg(stats, d, 'all')['pooled_improves_over_hard_ci']} with a paired 95% "
                    "CI below 1 for the MASE ratio)" for d in depths)
    speed = " / ".join(_x(_agg(stats, d, "all").get("speedup_B1")) for d in depths)
    reb = ["# Draft rebuttal paragraph (auto-filled numbers; finalize the wording by hand)", "",
           *banner]
    if not depths:
        reb.append("No LODO cell has been written yet; nothing to report.")
    else:
        txt = (
            "We tested whether one alignment adapter transfers to a dataset it was never fitted "
            f"on. Leaving one dataset out at a time, {scope} we fitted a single label-free affine "
            "adapter on the remaining datasets (13 sources, or 12 when an Electricity dataset is "
            "the target, because the hourly and 15-minute Electricity datasets share meters) with "
            "the same objective, optimizer and checkpoint selection as the dataset-specific "
            "adapters, and evaluated it once on the held-out dataset's test windows with no "
            "target-specific fitting or selection. Truncating at " + ", ".join(labels) +
            f" (measured end-to-end speedups {speed} at batch size 1), the median test MASE "
            f"change relative to the full model was {series('all', 'hard')} for hard truncation, "
            f"{series('all', 'pooled')} for the pooled adapter and {series('all', 'specific')} "
            "for the dataset-specific adapter. The pooled adapter beat hard truncation on " + imp +
            ".")
        if held and any(_agg(stats, d, "held_out") for d in depths):
            txt += (f" On the datasets documented as held out from Chronos-2 pretraining "
                    f"({', '.join(held)}), the pooled adapter's median change was "
                    f"{series('held_out', 'pooled')} (hard truncation "
                    f"{series('held_out', 'hard')}).")
        txt += (" The truncation depth was fixed in advance; selecting a depth for an unseen "
                "dataset without calibration data is a separate question that this experiment "
                "does not address.")
        reb.append(txt)
    return lines, reb


def build(root=DEFAULT_ROOT) -> dict:
    root = Path(root)
    data = load(root)
    lines, reb = report_lines(data)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "rebuttal_draft.md").write_text("\n".join(reb) + "\n")
    return {"report": str(out / "report.md"), "rebuttal_draft": str(out / "rebuttal_draft.md"),
            "complete": bool(data["stats"].get("complete")),
            "n_cells": data["stats"].get("n_cells", 0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    s = build(a.root)
    print(f"LODO report: {s['n_cells']} cells ({'complete' if s['complete'] else 'INCOMPLETE'}) "
          f"-> {s['report']}, {s['rebuttal_draft']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
