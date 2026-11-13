"""Phase 2d auto-generated report + draft rebuttal paragraph, from the Phase-2d TABLES ONLY.

    python -m experiments.make_phase2d_report        # login node, seconds; after make_phase2d_tables

Reads   <root>/combined/{stats.json, entrances.csv, final_block.csv}   (make_phase2d_tables)
        <root>/runs/<job>/{one_batch_checks.json, preflight_native_refit__*.json}
Writes  <root>/report/report.md, <root>/report/rebuttal_draft.md

Every number is copied from the tables (rebuilt from cell artifacts only). Entrances are relative to
each readout's OWN final block, so they are always printed next to the final-block losses. The
readout-structure confound (different head shapes / parameter counts for Chronos-2 and TiRex) is
stated next to the results. A partial run is stamped INCOMPLETE at the top of both files.
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

from probing import phase1                                                         # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "results" / "three_model_phase2d_pooled"
MODEL_NAME = {"chronos2": "Chronos-2", "timesfm3": "TimesFM-3", "tirex": "TiRex"}
ARM_LABEL = {"native": "architecture-aware (native)", "concat": "native states, concatenated",
             "pooled": "content mean (generic)", "native_mean": "native states, averaged"}
STEP_LABEL = {"readout_structure": "readout structure (native to concat)",
              "pooling": "pooling + capacity (concat to native mean)",
              "token_selection": "token selection (native mean to content mean; same head)",
              "representation": "representation (native to content mean; same head, TimesFM-3)"}


def _num(v):
    if v is None or v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


def _read_csv(p: Path) -> list[dict]:
    if not p.exists() or not p.stat().st_size:
        return []
    with open(p) as fh:
        return [{k: _num(v) for k, v in r.items()} for r in csv.DictReader(fh)]


def _pct(x, digits=1) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if not np.isfinite(x) else f"{100 * x:+.{digits}f}%"


def _md_table(rows, cols, head) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def load(root: Path) -> dict:
    root = Path(root)
    comb = root / "combined"
    sp = comb / "stats.json"
    if not sp.exists():
        raise SystemExit(f"no {sp}: run `python -m experiments.make_phase2d_tables` first")
    gates = {"one_batch": None, "refit": {}}
    for run in sorted((root / "runs").glob("*")):
        ob = run / "one_batch_checks.json"
        if ob.exists():
            gates["one_batch"] = {"job": run.name, **json.loads(ob.read_text())}
        for f in sorted(run.glob("preflight_native_refit__*.json")):
            r = json.loads(f.read_text())
            gates["refit"][r.get("model", f.stem)] = {"job": run.name, **r}
    return {"stats": json.loads(sp.read_text()), "ent": _read_csv(comb / "entrances.csv"),
            "fin": _read_csv(comb / "final_block.csv"), "gates": gates}


def report_lines(data) -> tuple[list[str], list[str]]:
    st, ent, fin, gates = data["stats"], data["ent"], data["fin"], data["gates"]
    n_c, n_p = int(st.get("n_complete", 0)), int(st.get("n_planned", 42))
    complete = n_c == n_p and not st.get("pending")
    agg = st.get("aggregate", [])
    models = [m for m in phase1.MODELS if any(r["model"] == m for r in agg)]
    done = {m: sum(1 for r in ent if r["model"] == m and r.get("status") == "complete")
            for m in phase1.MODELS}
    banner = [] if complete else [
        f"> **INCOMPLETE RESULTS.** {n_c} of {n_p} model-dataset cells are written ("
        + ", ".join(f"{MODEL_NAME[m]} {done[m]}/14" for m in phase1.MODELS) + "). Every number "
        "below covers only what exists; do not quote it as the full experiment.", ""]

    def A(m, arm):
        return next((r for r in agg if r["model"] == m and r["arm"] == arm), None)

    ob = gates.get("one_batch")
    gl = []
    if ob:
        res = {k: v for k, v in ob.items() if isinstance(v, dict) and "passed" in v}
        gl.append(f"* one-batch extraction checks (job {ob['job']}): "
                  + ", ".join(f"{MODEL_NAME.get(k, k)} {'PASS' if v.get('passed') else 'FAIL'}"
                              for k, v in res.items()))
    else:
        gl.append("* one-batch extraction checks: no record found")
    for m in phase1.MODELS:
        r = gates["refit"].get(m)
        if r:
            gl.append(f"* native-refit gate {MODEL_NAME[m]} on {r.get('dataset')} (job "
                      f"{r['job']}): {r.get('status')}, validation-curve max relative "
                      f"difference {r.get('val_curve_max_rel_diff', 'n/a')}"
                      + (f", error {r['error']}" if r.get("error") else ""))

    arm_rows, step_rows = [], []
    for m in models:
        for arm in ("native", "concat", "native_mean", "pooled"):
            a = A(m, arm)
            if not a:
                continue
            fm = [r["final_test_mase"] for r in fin if r["model"] == m and r["arm"] == arm
                  and isinstance(r.get("final_test_mase"), float)]
            row = {"model": MODEL_NAME[m], "arm": ARM_LABEL[arm], "params": f"{int(a['params']):,}",
                   "n": f"{a['n_cells']}/{a['n_planned']}",
                   "pre": f"{a['n_pre_final_entrance']}/{a['n_cells']}",
                   "rd": f"{a['median_entrance_relative_depth']:.2f}",
                   "loss": f"{a['median_final_test_loss']:.4f}",
                   "mase": f"{np.median(fm):.3f}" if fm else "n/a"}
            if arm != "native":
                row.update({
                    "same": f"{a['entrance_same_as_native']} / {a['entrance_within_1_block']} / "
                            f"{a['entrance_within_2_blocks']}",
                    "ea": f"{a['entrance_earlier_than_native']} / {a['entrance_later_than_native']}",
                    "dl": _pct(a["median_final_loss_delta_rel_vs_native"]),
                    "dir": f"{a['n_final_worse_than_native_ci']} / "
                           f"{a['n_final_better_than_native_ci']} / {a['n_final_inconclusive']}",
                    "rho": f"{a['median_spearman_val_curve_vs_native']:.2f}"})
            arm_rows.append(row)
        for step in STEP_LABEL:
            a = A(m, f"step:{step}")
            if a:
                step_rows.append({"model": MODEL_NAME[m], "step": STEP_LABEL[step],
                                  "n": a["n_cells"],
                                  "dl": _pct(a["median_final_loss_delta_rel_step"]),
                                  "dir": f"{a['n_step_worse_ci']} / {a['n_step_better_ci']} / "
                                         f"{a['n_step_inconclusive']}"})
    per = []
    for r in ent:
        if r.get("status") != "complete":
            per.append({"model": MODEL_NAME[r["model"]], "ds": r["display_name"],
                        "nat": "pending"})
            continue
        row = {"model": MODEL_NAME[r["model"]], "ds": r["display_name"],
               "fb": r.get("final_block", "")}
        for arm, key in (("native", "nat"), ("concat", "con"), ("native_mean", "nm"),
                         ("pooled", "poo")):
            if r.get(f"{arm}_entrance") is not None:
                row[key] = (f"{r[f'{arm}_entrance']} "
                            f"({r[f'{arm}_final_test_loss']:.4f})")
        per.append(row)
    params = [{"model": MODEL_NAME[r["model"]], "arm": ARM_LABEL.get(r["arm"], r["arm"]),
               "shape": f"{r['in_features']} to {r['out_features']}",
               "n": f"{int(r['n_params']):,}", "ratio": f"{r['ratio_to_native']:.1f}x"}
              for r in st.get("param_table", [])]

    lines = [
        "# Phase 2d: content-mean-pooled vs architecture-aware layerwise readouts, auto-generated "
        "report", "", *banner,
        "Generated by `python -m experiments.make_phase2d_report` from the Phase-2d tables "
        "(`combined/`), which are rebuilt from cell artifacts only. Numbers are facts; "
        "interpretation and the final rebuttal wording are written by hand.", "",
        "## Coverage and gates", "",
        f"* cells written: {n_c} of {n_p} (" + ", ".join(
            f"{MODEL_NAME[m]} {done[m]}/14" for m in phase1.MODELS) + ")",
        "* the native arm is READ from the committed Phase-1 cells and reproduced from their "
        "artifacts in every cell (curves and entrances); a cell is only written if that and its "
        "data gates (window digest, valid-token counts, state identity vs the Phase-1 caches) "
        "pass", *gl, "",
        "## Readout configurations (Q=9, H=64)", "",
        _md_table(params, ["model", "arm", "shape", "n", "ratio"],
                  ["model", "readout", "head", "parameters", "vs native"]), "",
        "## Per model and readout (median over datasets)", "",
        "Entrance = sustained 5% tunnel entrance on VALIDATION loss, relative to that readout's "
        "own final block (so read it with the final-block loss). `same / within 1 / within 2` "
        "counts datasets whose entrance matches the native one exactly, within one block, within "
        "two. Final change = test loss at the final block vs the native readout; `worse / better / "
        "inconclusive` uses the paired 95% CI. rho = median Spearman correlation of the "
        "validation curve with the native one.", "",
        _md_table(arm_rows, ["model", "arm", "params", "n", "pre", "rd", "loss", "mase", "same",
                             "ea", "dl", "dir", "rho"],
                  ["model", "readout", "params", "cells", "entrance before final", "median "
                   "entrance (relative depth)", "final test loss", "final test MASE",
                   "entrance same / within 1 / within 2", "earlier / later", "final change vs "
                   "native", "worse / better / inconclusive", "rho"]) if arm_rows
        else "(no cell yet)", "",
        "## One-factor decomposition (final-block test loss, median over datasets)", "",
        _md_table(step_rows, ["model", "step", "n", "dl", "dir"],
                  ["model", "step", "cells", "median change", "worse / better / inconclusive"])
        if step_rows else "(no cell yet)", "",
        "## Per dataset: entrance (final-block test loss)", "",
        _md_table(per, ["model", "ds", "fb", "nat", "con", "nm", "poo"],
                  ["model", "dataset", "final block", "native", "concat", "native mean",
                   "content mean"]) if per else "(none)", "",
        "## Fixed limitations (stated independently of the outcome)", "",
        "* Readout-structure confound: for Chronos-2 and TiRex the content-mean readout also "
        "changes the head (one vector to the whole horizon, no patch-wise sharing, 4x and 2x the "
        "native parameters). TimesFM-3 is the controlled case (same head, same parameters); "
        "the native-mean control isolates token selection with the head fixed.",
        "* An earlier entrance can come from a weaker final block; entrances are never read "
        "without the final-block losses.",
        "* Losses are comparable across readouts within a model (same windows, same normalized "
        "targets) but not across models.",
        "* One probe fit per depth and readout (the Phase-1 protocol, seed 0, weight decay chosen "
        "on validation); variation across seeds is not measured.",
    ]

    def same(m, arm="pooled"):
        a = A(m, arm)
        return None if a is None else (a["entrance_same_as_native"], a["entrance_within_1_block"],
                                       a["n_cells"])
    parts = []
    for m in models:
        s, a = same(m), A(m, "pooled")
        if s and a:
            parts.append(f"{MODEL_NAME[m]}: same entrance on {s[0]} of {s[2]} datasets "
                         f"({s[1]} within one block), final-block test loss "
                         f"{_pct(a['median_final_loss_delta_rel_vs_native'])} vs the native "
                         f"readout (worse on {a['n_final_worse_than_native_ci']}, better on "
                         f"{a['n_final_better_than_native_ci']}, paired CI)")
    tsel = []
    for m in models:
        a = A(m, "step:token_selection") or A(m, "step:representation")
        if a:
            tsel.append(f"{MODEL_NAME[m]} {_pct(a['median_final_loss_delta_rel_step'])} "
                        f"(worse on {a['n_step_worse_ci']}, better on {a['n_step_better_ci']})")
    scope = ("all 14 datasets for each of the three models" if complete else
             f"{n_c} of the 42 model-dataset pairs; the rest are still running")
    reb = ["# Draft rebuttal paragraph (auto-filled numbers; finalize the wording by hand)", "",
           *banner]
    if not parts:
        reb.append("No Phase-2d cell has been written yet; nothing to report.")
    else:
        reb.append(
            "To test whether the forecasting tunnel depends on reading each model where its own "
            "head reads, we repeated the layerwise probing with a generic readout: the mean of "
            "the valid content-token states at each depth, followed by one linear head to the "
            "full horizon, fitted and selected exactly as the native probes on the same windows "
            f"and targets ({scope}). Compared with the architecture-aware readout, the sustained "
            "5% entrance and the final-block test loss were: " + "; ".join(parts) + ". Holding the "
            "head fixed and changing only which token states are read (the native readout tokens "
            "versus the mean of all content tokens) changed the final-block test loss by " +
            "; ".join(tsel) + ". For Chronos-2 and TiRex the generic readout also has a different "
            "head shape and parameter count, so we report it next to the native-mean control "
            "rather than as a pure representation comparison.")
    return lines, reb


def build(root=DEFAULT_ROOT) -> dict:
    root = Path(root)
    data = load(root)
    lines, reb = report_lines(data)
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(lines) + "\n")
    (out / "rebuttal_draft.md").write_text("\n".join(reb) + "\n")
    st = data["stats"]
    return {"report": str(out / "report.md"), "rebuttal_draft": str(out / "rebuttal_draft.md"),
            "n_complete": st.get("n_complete", 0), "n_planned": st.get("n_planned", 42)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", default=str(DEFAULT_ROOT))
    a = ap.parse_args(argv)
    s = build(a.output_root)
    print(f"Phase 2d report: {s['n_complete']}/{s['n_planned']} cells -> {s['report']}, "
          f"{s['rebuttal_draft']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
