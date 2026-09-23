"""Combined Phase-2 tables, rebuilt from saved artifacts ONLY (no model, no GPU).

    python -m experiments.make_phase2_tables                      # login node OK: seconds

Reads every COMPLETE cell under <output-root>/h3 and <output-root>/h4 and every HEADLINE-protocol
latency job under <output-root>/latency, and writes <output-root>/combined/:

    h3_depth_metrics.csv       every model x dataset x family x depth row (all H3 cells)
    h3_ladder.csv              the compatibility ladder at the frozen Phase-1 entrance
    h3_tunnels.csv             compatibility entrances + compatibility lag (secondary diagnostic)
    h3_summary_by_model.csv    per model x rung: median [IQR] degradation, counts within 5%, ...
    h4_truncation.csv          physically truncated models at the entrance (+ V3 flags)
    h4_outcomes.csv            THE pre-registered H4 verdict per model x dataset (success /
                               failure / no_truncation_possible / invalid_v3) -- failures kept
    h4_outcome_summary.csv     those verdicts counted per model, with / without low-skill cells
    latency_jobs.csv           every (job, config, level, batch) median, raw-vector stats
    latency_lookup.csv         per (model, kind, depth, level, batch): median of per-job medians,
                               across-job range, and the within-job speedup vs native
    h4_with_latency.csv        h4_truncation joined with the latency lookup at each arm's depth
    phase2_stats.json          every number the prose may quote, with its provenance

Smoke cells (``summary.json.smoke``) are never mixed into a combined table, and neither is a
latency job that is not headline-eligible (smoke / tiny / unverified / CPU / killed); both are
listed in phase2_stats.json with the reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase2                                                         # noqa: E402

RUNG_ORDER = ("hard", "noa", "fl", "probe", "ra")
KIND_FOR_ARM = {"native": "native", "hard": "hard", "noa": "adapter", "fl": "adapter",
                "probe_head": "probe_head", "chronos2_small": "chronos2_small"}


def _cells(root: Path, kind: str):
    base = root / kind
    if not base.exists():
        return
    for marker in sorted(base.glob("*/*/COMPLETE")):
        cell = marker.parent
        if ".building-" in cell.name:
            continue
        yield cell


def _read_csv(p: Path) -> list[dict]:
    with open(p) as fh:
        return list(csv.DictReader(fh))


def _q(xs, qs=(25, 50, 75)):
    xs = [float(x) for x in xs if x is not None and np.isfinite(float(x))]
    if not xs:
        return [None] * len(qs)
    return [float(np.percentile(xs, q)) for q in qs]


# --------------------------------------------------------------------------- #
# H3
# --------------------------------------------------------------------------- #
def build_h3(root: Path):
    depth_rows, ladder_rows, tunnel_rows, smoke = [], [], [], []
    for cell in _cells(root, "h3"):
        summ = json.loads((cell / "summary.json").read_text())
        if summ.get("smoke"):
            smoke.append(str(cell))
            continue
        model, tag = summ["model"], summ["dataset"]
        depth_rows += _read_csv(cell / "depth_metrics.csv")
        lad = json.loads((cell / "ladder_at_tunnel.json").read_text())
        low = (summ.get("low_skill") or {}).get("flag")
        for tol_key, entry in lad["by_tolerance"].items():
            for rung, v in entry["rungs"].items():
                gc = v.get("gap_closure_mase") or {}
                ladder_rows.append({
                    "model": model, "dataset": tag, "tolerance": tol_key,
                    "is_headline": tol_key == f"tol_{phase2.HEADLINE_TOL:g}",
                    "label": entry["label"], "depth_index": entry["depth_index"],
                    "relative_depth": entry["relative_depth"],
                    "blocks_removed": entry["blocks_removed"], "rung": rung,
                    "mase_ratio": v["mase"]["ratio"], "mase_ci_lo": v["mase"]["ratio_ci_lo"],
                    "mase_ci_hi": v["mase"]["ratio_ci_hi"],
                    "wql_ratio": v["wql"]["ratio"], "wql_ci_lo": v["wql"]["ratio_ci_lo"],
                    "wql_ci_hi": v["wql"]["ratio_ci_hi"],
                    "within_budget_mase": v["mase"]["within_budget"],
                    "budget_fragile_mase": v["mase"]["budget_fragile"],
                    "gap_closure_mase": gc.get("gap_closure"), "gap_closure_valid": gc.get("valid"),
                    "compatibility_gap_exists": entry["compatibility_gap_exists"],
                    "low_skill": low})
        tun = json.loads((cell / "tunnels.json").read_text())
        for fam in [k for k in tun if isinstance(tun[k], dict) and "by_tolerance" in tun[k]]:
            for tol_key, ent in tun[fam]["by_tolerance"].items():
                tunnel_rows.append({"model": model, "dataset": tag, "family": fam,
                                    "tolerance": tol_key, "compatibility_entrance": ent,
                                    "phase1_entrance": tun["phase1_recoverability"][tol_key],
                                    "compatibility_lag": tun[fam]["compatibility_lag_in_depth"][
                                        tol_key],
                                    "no_truncation": tol_key in tun[fam]["no_truncation_at"]})
    return depth_rows, ladder_rows, tunnel_rows, smoke


def summarize_h3(ladder_rows, tunnel_rows):
    """Aggregates over the cells whose Phase-1 entrance lies BEFORE the final block. A cell whose
    entrance IS the final block has no intermediate state to test (the hard cut and the aligned
    arms are the native model there), so it would count as a vacuous 'within 5%'; it is counted
    separately as ``n_entrance_is_final_block`` instead."""
    out = []
    head = [r for r in ladder_rows if r["is_headline"]]
    for model in sorted({r["model"] for r in head}):
        for include_low in (True, False):
            rows_m = [r for r in head if r["model"] == model and (include_low or not r["low_skill"])]
            for rung in RUNG_ORDER:
                every = [r for r in rows_m if r["rung"] == rung]
                rr = [r for r in every if int(r["blocks_removed"]) > 0]
                if not every:
                    continue
                dm = [r["mase_ratio"] - 1 for r in rr]
                dw = [r["wql_ratio"] - 1 for r in rr]
                gcs = [r["gap_closure_mase"] for r in rr if r["gap_closure_valid"]]
                # the lag is meaningful at every cell (it can be negative when the entrance is
                # the final block), so it is taken over all of this model's (filtered) cells
                ds = {r["dataset"] for r in every}
                lags = [t["compatibility_lag"] for t in tunnel_rows
                        if t["model"] == model and t["family"] == rung
                        and t["tolerance"] == f"tol_{phase2.HEADLINE_TOL:g}"
                        and t["dataset"] in ds]
                q_m, q_w, q_g, q_l = _q(dm), _q(dw), _q(gcs), _q(lags)
                out.append({
                    "model": model, "rung": rung, "includes_low_skill": include_low,
                    "n_datasets": len(rr), "n_entrance_is_final_block": len(every) - len(rr),
                    "mase_degradation_q25": q_m[0], "mase_degradation_median": q_m[1],
                    "mase_degradation_q75": q_m[2],
                    "wql_degradation_q25": q_w[0], "wql_degradation_median": q_w[1],
                    "wql_degradation_q75": q_w[2],
                    "n_within_5pct_mase": sum(bool(r["within_budget_mase"]) for r in rr),
                    "n_fragile": sum(bool(r["budget_fragile_mase"]) for r in rr),
                    "n_compatibility_gap": (sum(bool(r["compatibility_gap_exists"]) for r in rr)
                                            if rung == "hard" else ""),
                    "n_gap_closure_valid": len(gcs), "gap_closure_median": q_g[1],
                    "compatibility_lag_median": q_l[1], "n_lag": len(lags)})
    return out


# --------------------------------------------------------------------------- #
# H4 + latency
# --------------------------------------------------------------------------- #
def build_h4(root: Path):
    rows = []
    for cell in _cells(root, "h4"):
        t = json.loads((cell / "truncation.json").read_text())
        for arm, a in t["arms"].items():
            rows.append({"model": t["model"], "dataset": t["dataset"], "arm": arm,
                         "kind": KIND_FOR_ARM.get(arm, arm),
                         "depth_index": a.get("depth_index", t["depth_index"])
                         if arm != "native" else t["blocks_total"],
                         "label": a.get("label") or ("native" if arm == "native" else arm),
                         "blocks_removed": a.get("blocks_removed", 0),
                         "active_params": a.get("active_params"),
                         "fraction_params_removed": a.get("fraction_params_removed"),
                         "adapter_params": a.get("adapter_params", 0),
                         "test_mase": a["test_mase"], "test_wql": a["test_wql"],
                         "mase_ratio": a["mase_vs_native"]["ratio"],
                         "mase_ci_lo": a["mase_vs_native"]["ratio_ci_lo"],
                         "mase_ci_hi": a["mase_vs_native"]["ratio_ci_hi"],
                         "wql_ratio": a["wql_vs_native"]["ratio"],
                         "within_budget_mase": a["mase_vs_native"]["within_budget"],
                         "V3_passed": a.get("V3_passed")})
    return rows


def build_h4_outcomes(root: Path):
    """One row per model x dataset: the pre-registered verdict (``phase2.H4_OUTCOME_RULE``).
    A failure is a row like any other -- never dropped, never replaced by another depth."""
    rows = []
    for cell in _cells(root, "h4"):
        t = json.loads((cell / "truncation.json").read_text())
        o = t.get("h4_outcome")
        if not o or o["rule"]["version"] != phase2.H4_OUTCOME_RULE["version"]:
            raise RuntimeError(f"{cell}: no H4 outcome under rule "
                               f"{phase2.H4_OUTCOME_RULE['version']}; re-run evaluate there")
        sec = o["secondary_not_deciding"]
        rows.append({"model": t["model"], "dataset": t["dataset"],
                     "candidate_label": o["candidate_depth"]["label"],
                     "candidate_depth_index": o["candidate_depth"]["depth_index"],
                     "blocks_total": o["blocks_total"], "blocks_removed": o["blocks_removed"],
                     "status": o["status"], "successful_truncation": o["successful_truncation"],
                     "aligned_mase_degradation": o["aligned_mase_degradation"],
                     "aligned_mase_ci_lo": o["aligned_mase_degradation_ci"][0],
                     "aligned_mase_ci_hi": o["aligned_mase_degradation_ci"][1],
                     "aligned_wql_degradation": o["aligned_wql_degradation"],
                     "fragile": o["fragile"], "low_skill": o["low_skill"],
                     **{f"{a}_within_budget": (sec.get(a) or {}).get("within_budget")
                        for a in ("fl", "hard", "probe_head")}})
    return rows


def summarize_h4_outcomes(rows):
    """Per model (with / without low-skill cells): how many cells succeeded, failed, could not be
    truncated (entrance = final block) or are invalid (V3 failed -- a pipeline error)."""
    out = []
    for model in sorted({r["model"] for r in rows}):
        for include_low in (True, False):
            rr = [r for r in rows if r["model"] == model and (include_low or not r["low_skill"])]
            n = {s: sum(r["status"] == s for r in rr) for s in phase2.H4_STATUSES}
            out.append({"model": model, "includes_low_skill": include_low,
                        "n_datasets": len(rr),
                        "n_truncatable": len(rr) - n["no_truncation_possible"],
                        **{f"n_{s}": n[s] for s in phase2.H4_STATUSES},
                        "n_success_fragile": sum(r["status"] == "success" and bool(r["fragile"])
                                                 for r in rr),
                        "n_failure_fragile": sum(r["status"] == "failure" and bool(r["fragile"])
                                                 for r in rr),
                        "failed_datasets": ";".join(sorted(r["dataset"] for r in rr
                                                           if r["status"] == "failure")),
                        "invalid_datasets": ";".join(sorted(r["dataset"] for r in rr
                                                            if r["status"] == "invalid_v3"))})
    return out


def build_latency(root: Path):
    """Only HEADLINE-protocol jobs enter the lookup. A smoke / tiny / unverified / CPU job
    (``index.json.headline_eligible`` false) or a killed one (no index.json) is excluded and
    returned in ``excluded`` with its reasons -- never averaged into a speedup."""
    job_rows, excluded = [], []
    base = root / "latency"
    if not base.exists():
        return job_rows, [], excluded
    for jdir in sorted(p for p in base.iterdir() if p.is_dir()):
        job, idx = jdir.name, jdir / "index.json"
        if not idx.exists():
            excluded.append({"job": job, "reasons": ["incomplete: no index.json (killed or "
                                                     "still running)"]})
            continue
        meta = json.loads(idx.read_text())
        if meta.get("headline_eligible") is not True:
            excluded.append({"job": job, "reasons": meta.get("ineligible_reasons")
                             or ["index.json carries no headline_eligible flag"]})
            continue
        for cdir in sorted(jdir.iterdir()):
            sp = cdir / "summary.json"
            if not sp.exists():
                continue
            s = json.loads(sp.read_text())
            cfg = s["config"]
            for key, lv in s.get("levels", {}).items():
                level, b = key.split("__B")
                job_rows.append({"job": job, "config_id": cdir.name, "model": cfg["model"],
                                 "kind": cfg["kind"], "depth": cfg.get("depth"),
                                 "drift": cfg.get("drift"), "level": level, "batch": int(b),
                                 "status": lv.get("status"), "median_ms": lv.get("median_ms"),
                                 "p95_ms": lv.get("p95_ms"), "iqr_ms": lv.get("iqr_ms"),
                                 "throughput": lv.get("throughput_series_per_s"),
                                 "peak_allocated_bytes": lv.get("peak_allocated_bytes"),
                                 "active_params": (s.get("params") or {}).get("active_params"),
                                 "weights_bytes": s.get("weights_bytes"),
                                 "gpu": s.get("environment", {}).get("gpu", {}).get("name")})
    # within-job speedup vs that job's native (median over the start/middle/end repeats)
    nat = defaultdict(list)
    for r in job_rows:
        if r["kind"] == "native" and r["status"] == "ok":
            nat[(r["job"], r["model"], r["level"], r["batch"])].append(r["median_ms"])
    for r in job_rows:
        n = nat.get((r["job"], r["model"], r["level"], r["batch"]))
        r["native_median_ms_same_job"] = float(np.median(n)) if n else None
        r["speedup_vs_native_same_job"] = (r["native_median_ms_same_job"] / r["median_ms"]
                                           if n and r["median_ms"] else None)
        if n and len(n) > 1:
            r["native_drift_range_ms"] = float(max(n) - min(n))
    groups = defaultdict(list)
    for r in job_rows:
        if r["status"] == "ok":
            groups[(r["model"], r["kind"], r["depth"], r["level"], r["batch"])].append(r)
    lookup = []
    for (m, k, d, lvl, b), rr in sorted(groups.items(), key=lambda x: str(x[0])):
        per_job = defaultdict(list)
        for r in rr:
            per_job[r["job"]].append(r)
        med = [float(np.median([x["median_ms"] for x in v])) for v in per_job.values()]
        spd = [float(np.median([x["speedup_vs_native_same_job"] for x in v
                                if x["speedup_vs_native_same_job"]])) for v in per_job.values()
               if any(x["speedup_vs_native_same_job"] for x in v)]
        lookup.append({"model": m, "kind": k, "depth": d, "level": lvl, "batch": b,
                       "n_jobs": len(per_job), "median_ms": float(np.median(med)),
                       "median_ms_min_job": min(med), "median_ms_max_job": max(med),
                       "speedup_vs_native": float(np.median(spd)) if spd else None,
                       "speedup_min_job": min(spd) if spd else None,
                       "speedup_max_job": max(spd) if spd else None,
                       "throughput": float(b / (np.median(med) / 1e3)),
                       "peak_allocated_bytes": float(np.median([x["peak_allocated_bytes"]
                                                                for x in rr
                                                                if x["peak_allocated_bytes"]]))
                       if any(x["peak_allocated_bytes"] for x in rr) else None,
                       "active_params": rr[0]["active_params"]})
    return job_rows, lookup, excluded


def join_h4_latency(h4_rows, lookup):
    idx = {(r["model"], r["kind"], r["depth"], r["level"], r["batch"]): r for r in lookup}
    out = []
    for r in h4_rows:
        row = dict(r)
        for lvl in ("api_e2e", "device_forward"):
            for b in (1, 32, 256):
                depth = None if r["kind"] == "chronos2_small" else r["depth_index"]
                L = idx.get((r["model"], r["kind"], depth, lvl, b))
                row[f"{lvl}_B{b}_median_ms"] = None if L is None else L["median_ms"]
                row[f"{lvl}_B{b}_speedup"] = None if L is None else L["speedup_vs_native"]
                row[f"{lvl}_B{b}_throughput"] = None if L is None else L["throughput"]
                row[f"{lvl}_B{b}_peak_bytes"] = None if L is None else L["peak_allocated_bytes"]
        out.append(row)
    return out


def build(output_root) -> dict:
    root = Path(output_root)
    comb = root / "combined"
    comb.mkdir(parents=True, exist_ok=True)
    depth_rows, ladder_rows, tunnel_rows, smoke = build_h3(root)
    summ = summarize_h3(ladder_rows, tunnel_rows)
    h4 = build_h4(root)
    h4_out = build_h4_outcomes(root)
    h4_summ = summarize_h4_outcomes(h4_out)
    jobs, lookup, lat_excluded = build_latency(root)
    joined = join_h4_latency(h4, lookup)
    for name, rows in (("h3_depth_metrics", depth_rows), ("h3_ladder", ladder_rows),
                       ("h3_tunnels", tunnel_rows), ("h3_summary_by_model", summ),
                       ("h4_truncation", h4), ("h4_outcomes", h4_out),
                       ("h4_outcome_summary", h4_summ), ("latency_jobs", jobs),
                       ("latency_lookup", lookup), ("h4_with_latency", joined)):
        if rows:
            fields = list(dict.fromkeys(k for r in rows for k in r))
            phase2.write_csv(comb / f"{name}.csv", rows, fields)
    stats = {"schema": "phase2_stats/v1", "n_h3_cells": len({(r["model"], r["dataset"])
                                                             for r in ladder_rows}),
             "n_h4_cells": len({(r["model"], r["dataset"]) for r in h4}),
             "n_latency_jobs": len({r["job"] for r in jobs}),
             "smoke_cells_excluded": smoke,
             "latency_jobs_excluded": lat_excluded,
             "h3_summary_by_model": summ,
             "h4_rule": phase2.H4_OUTCOME_RULE,
             "h4_outcome_summary": h4_summ,
             "h4_invalid_cells": [f"{r['model']}/{r['dataset']}" for r in h4_out
                                  if r["status"] == "invalid_v3"],
             "provenance": "rebuilt by experiments.make_phase2_tables from saved artifacts only"}
    phase2.atomic_write_json(comb / "phase2_stats.json", stats)
    return {"out": str(comb), **{k: v for k, v in stats.items() if k.startswith("n_")}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default=str(phase2.DEFAULT_OUT_ROOT))
    a = p.parse_args(argv)
    r = build(a.output_root)
    print(f"[phase2 tables] {r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
