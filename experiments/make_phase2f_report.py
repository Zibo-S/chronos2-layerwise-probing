"""Phase 2f technical report + rebuttal draft, written from the tables ONLY.

    python -m experiments.make_phase2f_report          # after make_phase2f_tables; seconds

Reads   <root>/combined/source_scaling_stats.json (+ source_scaling_cells.csv), runs/*/preflight.json
Writes  <root>/report/report.md, <root>/report/rebuttal_draft.md

Every sentence that states a direction (improves / does not improve / approaches) is chosen from
the measured numbers; nothing presumes that transfer improves with N. A partial run is stamped
INCOMPLETE and the draft states its coverage.
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

from probing import phase2f_source_scaling as ss                                  # noqa: E402
from probing import registry                                                      # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "results" / "three_model_phase2f_source_scaling"
NAMES = {"LOOP_SEATTLE_5T": "Loop Seattle", "monash_london_smart_meters": "London Smart Meters",
         "boom_hourly": "BOOM", "m5": "M5"}
PROV = {"held_out": "held out from Chronos-2 pretraining", "not_listed": "not listed",
        "listed": "listed as Chronos-2 pretraining data"}


def _f(x, d=1, unit="%"):
    if x is None:
        return "n/a"
    return f"{x:+.{d}f}{unit}"


def _pp(x, d=1):
    return _f(x, d, " pp")


def _rows(path: Path):
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def _preflight(root: Path) -> dict:
    out = {"files": 0, "targets": 0, "failed": 0}
    for p in sorted((root / ss.KIND / "runs").glob("*/preflight.json")):
        d = json.loads(p.read_text())
        out["files"] += 1
        out["targets"] = max(out["targets"], int(d.get("n_targets", 0)))
        out["failed"] += int(d.get("n_failed", 0))
    return out


def _join(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def build(root: Path) -> dict:
    root = Path(root)
    st = json.loads((root / "combined" / "source_scaling_stats.json").read_text())
    cells = _rows(root / "combined" / "source_scaling_cells.csv")
    agg = {a["n_sources"]: a for a in st["aggregate"]}
    bt = st["by_target_n"]
    ana = st["analysis"]
    complete = bool(st["complete"])
    n_have, n_exp = st["n_cells"], st["n_expected"]
    tg_have = [t for t in ss.TARGETS if any(r["target"] == t for r in bt)]
    pf = _preflight(root)
    fr = st["fit_reproduction"]
    sp = st.get("speedup_L6") or {}
    banner = ([] if complete else [
        f"> **INCOMPLETE RESULTS.** {n_have} of the {n_exp} planned adapter fits are written "
        f"({len(tg_have)} of the 4 targets have at least one). Every number below covers only "
        "what exists; do not quote it as the full experiment.", ""])

    L = ["# Phase 2f: source-dataset scaling of zero-shot adapter transfer", ""] + banner
    L += ["## Question", "",
          "Does the label-free adapter that aligns Chronos-2's truncated L6 states transfer better "
          "to an unseen dataset when it is fitted on more source datasets, and does it approach "
          "the dataset-specific H3 adapter? A testable hypothesis: the analysis below does not "
          "presume a monotone trend.", "",
          "## Design", "",
          f"- Model and depth: Chronos-2 truncated at {st['label']} (6 of 12 blocks kept)"
          + (f"; measured end-to-end speedup of the adapter pathway {sp['B1']:.2f}x at batch 1"
             + (f", {sp['B256']:.2f}x at batch 256" if sp.get("B256") else "")
             if sp.get("B1") else "") + ".",
          "- Adapter fit: the Phase 2c pooled fit, unchanged (the H3 label-free residual affine "
          "adapter: AdamW lr 1e-3, 1500 full-batch epochs, decay on Delta only over "
          "{0, 0.1, 1, 10, 30, 100}, validation every 25 epochs, epoch 0 = the hard cut, "
          "equal-weighted per-source objective and selection). The only variable is which source "
          "datasets the fit reads. The target is never used for fitting or selection; its test "
          "split is read once, after selection.",
          f"- Source pools: N in {st['n_grid']}. For N < 13, three fixed orders per target "
          "(seeds 0, 1, 2), nested prefixes: a larger N adds datasets. N = 13 is the whole pool, "
          "one fit per target shared by the three orders.",
          "- Order restrictions (fixed before any fit, blind to performance): the two Electricity "
          "datasets (shared meters) never sit in the same N <= 8 prefix; the N = 1 source differs "
          "across the three orders of a target.",
          f"- {n_exp} adapter fits in total (4 targets x (3 orders x 3 sizes + 1)).", "",
          "### Targets (selected from metadata before any result)", "",
          st["target_selection_rule"], "",
          "| Target | Frequency | Domain | Chronos-2 provenance | Signal | Why |",
          "|---|---|---|---|---|---|"]
    for t in ss.TARGETS:
        r = st["target_rationale"][t]
        L.append(f"| {NAMES[t]} | {r['frequency']} (m = {r['seasonal_period']}) | "
                 f"{r['domain_family']} | {r['provenance'].replace('_', ' ')} | {r['signal']} | "
                 f"{r['why']} |")
    L += ["", "### Source orders", "", "| Target | Order | N = 1 | added for N = 4 | "
          "added for N = 8 | added for N = 13 |", "|---|---|---|---|---|---|"]
    for t in ss.TARGETS:
        for o in st["orders"][t]:
            v = [registry.display_name(x) for x in o["order"]]
            L.append(f"| {NAMES[t]} | {o['seed']} | {v[0]} | {', '.join(v[1:4])} | "
                     f"{', '.join(v[4:8])} | {', '.join(v[8:])} |")

    L += ["", "## Coverage and gates", "",
          f"- Cells: {n_have} of {n_exp} written"
          + ("" if complete else f"; missing: {', '.join('/'.join(m) for m in st['missing'])}")
          + ".",
          f"- Preflight (Phase 2c reproduction gates per target before any fit): "
          + (f"{pf['targets'] - pf['failed']} of {pf['targets']} targets passed"
             if pf["files"] else "no preflight record found") + ".",
          f"- Fit reproduction (each N = 1 fit vs the committed H3 fit of its single source): "
          f"{sum(x['status'] == 'pass' for x in fr)} of {len(fr)} pass; bitwise-identical "
          f"adapters: {sum(str(x['bitwise']) == 'True' for x in fr)} of {len(fr)}.",
          "- Every cell re-ran the Phase 2c gates (full model vs Phase 1 and H3, hard cut and "
          "dataset-specific adapter vs H3, test rows, identity adapter); a failing cell is never "
          "tabulated."]
    cross = st.get("phase2c_crosscheck") or {}
    if any(v.get("available") for v in cross.values()):
        for t, v in cross.items():
            if v.get("available") and "rel_diff" in v:
                L.append(f"- Phase 2c cross-check ({NAMES.get(t, t)}, N = 13 vs the Phase 2c main "
                         f"cell at L6): pooled MASE relative difference {v['rel_diff']:.2e}, same "
                         f"adapter checksum: {v['same_adapter_sha256']}.")
    else:
        L.append("- Phase 2c cross-check of the N = 13 cells: no Phase 2c L6 cell available when "
                 "the tables were built (the N = 13 fits were made independently).")

    L += ["", "## Results per target", "",
          "MASE change of the truncated model vs the full model on the target's test split "
          "(paired series-cluster bootstrap, B = 5000). 'Pooled - specific' is the difference of "
          "the two degradations in percentage points of the full model.", "",
          "| Target | N | Mean over orders | Min | Max | Pooled - specific (mean) | Hard cut | "
          "Target-specific | Train windows |", "|---|---|---|---|---|---|---|---|---|"]
    for r in bt:
        L.append(f"| {NAMES.get(r['target'], r['target'])} | {r['n_sources']} | "
                 f"{_f(r['dmase_pct_mean'])} | {_f(r['dmase_pct_min'])} | "
                 f"{_f(r['dmase_pct_max'])} | {_pp(r['gap_to_specific_pp_mean'])} | "
                 f"{_f(r['hard_dmase_pct'])} | {_f(r['specific_dmase_pct'])} | "
                 f"{r['n_train_windows']:,} |")
    if cells:
        L += ["", "Every cell:", "", "| Target | N | Order | dMASE [95% CI] | Pooled - specific "
              "[95% CI] | Gap recovery | wd | epoch | fit (min) | sources |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in cells:
            gc = r["pooled_gap_closure"]
            gc = "n/a" if gc in ("", "None") else f"{100 * float(gc):.0f}%"
            L.append(f"| {NAMES.get(r['target'], r['target'])} | {r['n_sources']} | "
                     f"{r['order_seed']} | {_f(100 * float(r['pooled_dmase']))} "
                     f"[{_f(100 * float(r['pooled_dmase_ci_lo']))}, "
                     f"{_f(100 * float(r['pooled_dmase_ci_hi']))}] | "
                     f"{_pp(float(r['pooled_minus_specific_pp']))} "
                     f"[{_pp(float(r['pooled_minus_specific_pp_ci_lo']))}, "
                     f"{_pp(float(r['pooled_minus_specific_pp_ci_hi']))}] | {gc} | "
                     f"{r['selected_wd']} | {r['selected_epoch']} | {r['fit_minutes']} | "
                     f"{', '.join(registry.display_name(x) for x in r['selection_order'].split(';'))} |")

    q1, q2 = ana["q1_larger_pool_improves"], ana["q2_monotone"]
    q4, q5, q6 = ana["q4_negative_transfer"], ana["q5_n13_vs_dataset_specific"], \
        ana["q6_diminishing_returns"]
    o = q1["overall_1_to_13"]
    L += ["", "## Scaling analysis", "",
          f"Counts are over (target, order) pairs: {q1['n_pairs']} pairs with both N = 1 and "
          "N = 13 written. With four targets they describe these datasets; they are not a "
          "population test.", "",
          f"1. **Larger pools.** From N = 1 to N = 13 the degradation decreased with a paired "
          f"95% CI excluding zero in {o['improves']} pairs, increased in {o['worsens']}, and "
          f"the CI contained zero in {o['indistinguishable']} "
          f"(point estimate lower at N = 13 in {q1['overall_1_to_13_point_negative']}). Mean "
          "change per target (pp): " + _join(f"{NAMES[t]} {_pp(v)}" for t, v in
                                             q1["per_target_mean_change_pp"].items()) + ".",
          f"2. **Monotonicity.** {q2['n_monotone_nonincreasing_point']} of "
          f"{ana['n_complete_sequences']} complete sequences (1, 4, 8, 13) never increase "
          f"(point estimates). Consecutive steps: {q2['consecutive_steps']['improves']} improve, "
          f"{q2['consecutive_steps']['worsens']} worsen, "
          f"{q2['consecutive_steps']['indistinguishable']} indistinguishable (paired CI), of "
          f"{q2['n_consecutive_steps']}.",
          "3. **Dependence on which datasets.** Range over the three orders at a fixed N "
          "(pp): " + "; ".join(
              f"{NAMES[t]} " + ", ".join(f"N = {n}: {float(v):.1f}" for n, v in sorted(
                  ana["q3_source_identity"]["seed_range_pp"][t].items(), key=lambda kv: int(kv[0])))
              for t in ss.TARGETS if t in ana["q3_source_identity"]["seed_range_pp"]) + ".",
          "4. **Negative transfer.** Cells worse than the hard cut (CI of pooled / hard above 1): "
          + (", ".join(q4["cells_worse_than_hard_cut_ci"]) or "none") + ". Steps that "
          "significantly increased the degradation: "
          + (", ".join(q4["steps_worsening_ci"]) or "none") + ".",
          "5. **N = 13 vs the dataset-specific adapter.** " + "; ".join(
              f"{NAMES[t]}: {_pp(v['pooled_minus_specific_pp'])} [{_pp(v['ci'][0])}, "
              f"{_pp(v['ci'][1])}], {v['category']}" for t, v in q5.items()) + "." if q5 else
          "5. **N = 13 vs the dataset-specific adapter.** No N = 13 cell written yet.",
          "6. **Diminishing returns.** Mean gain per added source dataset (pp of degradation "
          "removed per dataset, mean over orders): " + "; ".join(
              f"{NAMES[t]} " + ", ".join(f"{k}: {v:+.2f}" for k, v in g.items())
              for t, g in q6["gain_pp_per_added_source_mean_over_orders"].items()) +
          f". Strictly decreasing gains: {', '.join(NAMES[t] for t in q6['targets_strictly_diminishing']) or 'none'}."]

    L += ["", "## Aggregate (descriptive)", "",
          "Mean over targets of the mean over orders; the interval resamples targets, then orders "
          "within a target (four targets: descriptive, not a test).", "",
          "| N | Targets | dMASE mean [interval] | Pooled - specific mean [interval] | "
          "Train windows | Steps per fit | Window passes | Fit (min) |",
          "|---|---|---|---|---|---|---|---|"]
    for n in ss.N_GRID:
        a = agg.get(n)
        if not a:
            continue
        L.append(f"| {n} | {a['n_targets']} | {_f(a['dmase_pct_mean'])} [{_f(a['dmase_pct_lo'])}, "
                 f"{_f(a['dmase_pct_hi'])}] | {_pp(a['gap_to_specific_pp_mean'])} "
                 f"[{_pp(a['gap_to_specific_pp_lo'])}, {_pp(a['gap_to_specific_pp_hi'])}] | "
                 f"{a['n_train_windows']:,} | {a['steps_per_fit']:,} | "
                 f"{a['window_passes_mean']:,.0f} | {a['fit_minutes_mean']:.1f} |")
    L += ["", "## Fixed limitations", "",
          "- Four targets. Counts are descriptive; no claim is made about datasets in general.",
          "- Data volume grows with N: every source adds its full 1,394 training windows (the "
          "Phase 2c per-source budget), so N also scales the training data and the work per "
          "optimization step (table above). This measures the practical effect of a larger "
          "source pool, not diversity at a fixed amount of data.",
          "- The two Electricity datasets share meters: N = 13 contains 12 dataset families, "
          "not 13 independent ones.",
          "- No trend is fitted or extrapolated beyond N = 13. These results cannot show that "
          "more source datasets would make zero-shot alignment match dataset-specific alignment.",
          "- The dataset-specific H3 adapter is calibrated on the target's unlabeled training "
          "inputs; it is a practical reference, not a guaranteed upper bound.",
          "- One model (Chronos-2) and one depth (L6).",
          "- Optional follow-up, not run: a fixed-total-window control (each of the N sources "
          "subsampled to 1,394 / N training windows, so every fit sees 1,394 windows) would "
          "separate the number of datasets from the amount of data; at about one minute per fit "
          "it would cost well under an hour of GPU time.", ""]
    out = root / "report"
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text("\n".join(L) + "\n")
    reb = rebuttal(st, tg_have, complete, n_have, n_exp)
    (out / "rebuttal_draft.md").write_text(reb)
    return {"report": str(out / "report.md"), "rebuttal_draft": str(out / "rebuttal_draft.md"),
            "complete": complete}


def rebuttal(st: dict, tg_have, complete: bool, n_have: int, n_exp: int) -> str:
    """One paragraph block for the reviewer concern 'dataset-specific calibration weakens the
    zero-shot acceleration claim', from the measured numbers only."""
    agg = {a["n_sources"]: a for a in st["aggregate"]}
    ana = st["analysis"]
    q1, q5, q2 = ana["q1_larger_pool_improves"], ana["q5_n13_vs_dataset_specific"], \
        ana["q2_monotone"]
    sp = st.get("speedup_L6") or {}
    head = [] if complete else [
        f"> **INCOMPLETE RESULTS.** This draft covers {n_have} of the {n_exp} planned adapter "
        f"fits ({len(tg_have)} of the 4 targets). Do not send it until the run is complete.", ""]
    scope = ("for four target datasets" if complete else
             f"for {len(tg_have)} of the four planned target datasets")
    s1 = ("To test whether the remaining gap between zero-shot and dataset-specific alignment "
          "reflects the small number of source datasets, we fitted the same label-free adapter "
          f"to Chronos-2 truncated after block 6 of 12"
          + (f" (measured end-to-end speedup {sp['B1']:.2f}x at batch size 1)" if sp.get("B1")
             else "")
          + " on N = 1, 4, 8 and 13 source datasets, never using the target dataset for fitting "
          f"or model selection, {scope} chosen in advance from metadata alone: Loop Seattle "
          "(5-minute), London Smart Meters (30-minute), BOOM (hourly) and M5 (daily), which "
          "cover all three of our pretraining-provenance groups. For N below 13 we used three "
          "nested random orders of the source datasets per target.")
    parts = [s1]
    a13, a1 = agg.get(13), agg.get(1)
    if a13 and q5:
        per = "; ".join(f"{NAMES[t]} {v['pooled_dmase_pct']:+.1f}% vs {v['specific_dmase_pct']:+.1f}%"
                        for t, v in q5.items())
        spec_mean = sum(v["specific_dmase_pct"] for v in q5.values()) / len(q5)
        hard_mean = sum(v["hard_dmase_pct"] for v in q5.values()) / len(q5)
        parts.append(
            f"With all 13 other datasets as sources, the test MASE of the truncated model was "
            f"{a13['dmase_pct_mean']:+.1f}% relative to the full model (mean over "
            f"{a13['n_targets']} targets), against {spec_mean:+.1f}% for the dataset-specific "
            f"adapter and {hard_mean:+.1f}% for truncation without an adapter (per target, "
            f"pooled vs dataset-specific: {per}).")
        cats = [v["category"] for v in q5.values()]
        n_ind = cats.count("indistinguishable") + cats.count("better")
        parts.append(
            f"The 13-source adapter was statistically indistinguishable from, or better than, the "
            f"dataset-specific adapter for {n_ind} of {len(cats)} targets and worse for "
            f"{cats.count('worse')} (paired 95% bootstrap intervals over test series).")
    if a1 and a13 and q1["n_pairs"]:
        o = q1["overall_1_to_13"]
        trend = ("decreased" if a13["dmase_pct_mean"] < a1["dmase_pct_mean"] else
                 "did not decrease")
        seq = ", ".join(f"{agg[n]['dmase_pct_mean']:+.1f}%" for n in ss.N_GRID if n in agg)
        parts.append(
            f"Across source counts, the mean degradation {trend} from N = 1 to N = 13 ({seq} at "
            f"N = {', '.join(str(n) for n in ss.N_GRID if n in agg)}); going from one source to "
            f"13 reduced it with a paired interval excluding zero in {o['improves']} of the "
            f"{q1['n_pairs']} target and order combinations and increased it in {o['worsens']}, "
            f"and {q2['n_monotone_nonincreasing_point']} of the {ana['n_complete_sequences']} "
            "target and order combinations never got worse as sources were added.")
    parts.append(
        "These results describe our 14-dataset collection only: each added source also adds "
        "training windows (1,394 per dataset), the two Electricity datasets share meters (so "
        "N = 13 spans 12 dataset families), and we do not extrapolate beyond 13 sources. In "
        "particular, they do not show that a larger and more diverse source collection would "
        "make zero-shot alignment match dataset-specific alignment.")
    return "\n".join(head + ["**Dataset-specific calibration and zero-shot acceleration.** "
                             + " ".join(parts), ""])


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(DEFAULT_ROOT))
    a = p.parse_args(argv)
    s = build(Path(a.root))
    print(f"Phase 2f report ({'complete' if s['complete'] else 'INCOMPLETE'}): {s['report']}, "
          f"{s['rebuttal_draft']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
