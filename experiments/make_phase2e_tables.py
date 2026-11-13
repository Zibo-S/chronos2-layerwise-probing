"""Phase 2e tables, figures, macros and the auto-generated report -- from cell ARTIFACTS ONLY.

No model, no cache, no bootstrap (every CI was computed inside the GPU job from per-window arrays
under one cluster count matrix per dataset): this script only aggregates and plots numbers that
already exist, so it is login-node safe (seconds, low CPU).

    python -m experiments.make_phase2e_tables

READS   results/three_model_phase2e_noncontig/{masks,comparison,single_skip,latency,verification,
        gates}/ and, for a context curve only, results/three_model_phase2/combined/
        h4_frontier_points.csv (the committed prefix frontier at every depth; read-only).
WRITES  results/three_model_phase2e_noncontig/{tables,figures,report}/ and
        three_models_paper/figures/noncontig/ (paper-ready .tex / .pdf / .png).

Only cells whose physical model reproduces the offline forecast (V3, status "ok") enter a median;
an invalid_v3 cell is counted and listed, never averaged in.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase2e_noncontig as P                                        # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "results" / "three_model_phase2e_noncontig"
PAPER_DIR = REPO_ROOT / "three_models_paper" / "figures" / "noncontig"
PHASE2_FRONTIER = REPO_ROOT / "results" / "three_model_phase2" / "combined" / \
    "h4_frontier_points.csv"
ARM_NAME = {"hard": "Hard cut (no adapter)", "noa": "Aligned (dataset-specific NOA adapter)"}
LAT_KIND = {"hard": "hard", "noa": "adapter"}
#: strategy -> (colour, marker, line style). Validated with the dataviz validator (all pairs,
#: light): CVD dE >= 11.0, normal-vision dE >= 18.7, contrast >= 3:1. The control is a neutral
#: near-black ON PURPOSE (outside the categorical lightness band: it must not read as a 4th hue).
STYLE = {"prefix": ("#0072B2", "o", "-"), "uniform": ("#D55E00", "s", "-"),
         "early_late": ("#009E73", "^", "-"), "middle": ("#333333", "D", (0, (3, 1.6)))}
TEXTWIDTH_IN = 5.5
RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
      "ytick.labelsize": 6.5, "legend.fontsize": 7.0, "axes.linewidth": 0.7,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
      "lines.linewidth": 1.3, "pdf.fonttype": 42, "ps.fonttype": 42}
WORD = {4: "Four", 6: "Six", 9: "Nine"}
SWORD = {"prefix": "Prefix", "uniform": "Uniform", "early_late": "EarlyLate", "middle": "Middle"}


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _json(p: Path):
    return json.loads(Path(p).read_text())


def _complete(d: Path) -> bool:
    return (d / "COMPLETE").exists()


def load_cells(root: Path) -> dict:
    out = {"masks": {}, "comparison": {}, "single_skip": {}, "gates": {}}
    base = root / "masks"
    if base.exists():
        for ds in sorted(p for p in base.iterdir() if p.is_dir()):
            for c in sorted(p for p in ds.iterdir() if p.is_dir() and _complete(p)):
                m = _json(c / "metrics.json")
                out["masks"][(m["dataset"], m["strategy"], m["budget"])] = m
    for kind in ("comparison", "single_skip"):
        b = root / kind / P.MODEL
        if b.exists():
            for c in sorted(p for p in b.iterdir() if p.is_dir() and _complete(p)):
                name = "comparison.json" if kind == "comparison" else "metrics.json"
                out[kind][c.name] = _json(c / name)
    g = root / "gates"
    if g.exists():
        for p in sorted(g.glob("*.json")):
            out["gates"][p.stem] = _json(p)
    v = root / "verification" / "masks.json"
    out["verification"] = _json(v) if v.exists() else None
    return out


def load_latency(root: Path) -> dict:
    """Within-repeat speedups vs that repeat's native, summarized over complete repeats.
    Returns {(label, kind, level, B): {"median_ms", "speedup", "speedup_min", "speedup_max", ...}}."""
    base = root / "latency"
    per = defaultdict(list)
    reps = []
    if not base.exists():
        return {"lookup": {}, "repeats": []}
    for rdir in sorted(p for p in base.iterdir() if p.is_dir() and ".building-" not in p.name):
        idx = rdir / "index.json"
        if not idx.exists():
            continue
        meta = _json(idx)
        reps.append({"repeat": meta.get("repeat"), "host": meta.get("host"),
                     "eligible": meta.get("headline_eligible"),
                     "reasons": meta.get("ineligible_reasons")})
        if meta.get("headline_eligible") is not True:
            continue
        nat = defaultdict(list)
        rows = []
        for c in meta["configs"]:
            if c.get("status") != "ok":
                continue
            for key, ms in (c.get("median_ms") or {}).items():
                if ms is None:
                    continue
                level, b = key.split("__B")
                if c["kind"] == "native":
                    nat[(level, int(b))].append(ms)
                else:
                    rows.append((c["label"], c["kind"], level, int(b), ms))
        for lab, kind, level, b, ms in rows:
            n = nat.get((level, b))
            if n:
                per[(lab, kind, level, b)].append({"repeat": meta.get("repeat"), "ms": ms,
                                                   "native_ms": float(np.median(n)),
                                                   "speedup": float(np.median(n)) / ms})
        for (level, b), n in nat.items():
            per[("native", "native", level, b)].append({"repeat": meta.get("repeat"),
                                                        "ms": float(np.median(n)),
                                                        "native_ms": float(np.median(n)),
                                                        "speedup": 1.0,
                                                        "drift_range_ms": float(max(n) - min(n))})
    lookup = {}
    for k, v in per.items():
        sp = [x["speedup"] for x in v]
        lookup[k] = {"n_repeats": len(v), "median_ms": float(np.median([x["ms"] for x in v])),
                     "speedup": float(np.median(sp)), "speedup_min": float(min(sp)),
                     "speedup_max": float(max(sp))}
    return {"lookup": lookup, "repeats": reps}


def _speed(lat, label, arm, level="api_e2e", B=1):
    r = lat["lookup"].get((label, LAT_KIND[arm], level, B))
    return None if r is None else r


# --------------------------------------------------------------------------- #
# summaries
# --------------------------------------------------------------------------- #
def _q(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return (None, None, None)
    a = np.percentile(xs, [25, 50, 75])
    return float(a[0]), float(a[1]), float(a[2])


def sign_test_p(n_better: int, n_worse: int) -> float | None:
    """Exact two-sided binomial sign test on the direction of the point differences (ties
    dropped)."""
    n = n_better + n_worse
    if n == 0:
        return None
    k = min(n_better, n_worse)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return float(min(1.0, 2 * p))


def per_dataset_rows(cells, lat) -> list[dict]:
    rows = []
    for (tag, s, k), m in sorted(cells["masks"].items(), key=lambda x: (
            x[0][2], P.STRATEGIES.index(x[0][1]), x[0][0])):
        cmp = cells["comparison"].get(tag, {}).get("by_budget", {}).get(f"k{k}", {})
        r = {"dataset": tag, "budget": k, "strategy": s, "label": m["label"],
             "blocks": m["mask_string"], "source": m["source"], "status": m["status"],
             "native_test_mase": m["test_mase"]["native"]}
        for a in ("hard", "noa"):
            v = m["stats"]["vs_native"][a]["mase"]
            r[f"{a}_test_mase"] = m["test_mase"][a]
            r[f"{a}_dmase_pct"] = 100 * v["degradation"]
            r[f"{a}_dmase_ci_lo"] = 100 * v["degradation_ci_lo"]
            r[f"{a}_dmase_ci_hi"] = 100 * v["degradation_ci_hi"]
            w = m["stats"]["vs_native"][a]["wql"]
            r[f"{a}_dwql_pct"] = 100 * w["degradation"]
            r[f"{a}_val_mase"] = m["val_mase"][a]
            vp = (cmp.get(f"{s}|{a}") or {}).get("vs_prefix", {}).get("mase")
            r[f"{a}_vs_prefix_pp"] = None if vp is None else vp["diff_pp"]
            r[f"{a}_vs_prefix_ci_lo"] = None if vp is None else vp["diff_pp_ci"][0]
            r[f"{a}_vs_prefix_ci_hi"] = None if vp is None else vp["diff_pp_ci"][1]
            r[f"{a}_better_ci"] = None if vp is None else vp["better_ci"]
            r[f"{a}_worse_ci"] = None if vp is None else vp["worse_ci"]
            sp = _speed(lat, m["label"], a)
            r[f"{a}_speedup_b1"] = None if sp is None else sp["speedup"]
        r["noa_gap_closure"] = (m["stats"].get("gap_closure", {}).get("mase") or {}).get(
            "gap_closure")
        r["selected_wd"] = m.get("selected_wd")
        r["selected_epoch"] = m.get("selected_epoch")
        r["selected_hard_cut"] = m["flags"].get("selected_hard_cut")
        r["late_checkpoint"] = m["flags"].get("late_checkpoint")
        r["grid_max_wd"] = m["flags"].get("grid_max_wd")
        rows.append(r)
    return rows


def summary_rows(cells, lat, rows) -> list[dict]:
    out = []
    by = defaultdict(list)
    for r in rows:
        by[(r["budget"], r["strategy"])].append(r)
    for k in P.BUDGETS:
        for s in P.STRATEGIES:
            rr = by.get((k, s), [])
            if not rr:
                continue
            ok = [r for r in rr if r["status"] == "ok"]
            for a in ("hard", "noa"):
                d = _q([r[f"{a}_dmase_pct"] for r in ok])
                rec = {"budget": k, "strategy": s, "arm": a, "blocks": rr[0]["blocks"],
                       "n_datasets": len(ok), "n_invalid_v3": len(rr) - len(ok),
                       "dmase_q25": d[0], "dmase_median": d[1], "dmase_q75": d[2],
                       "dwql_median": _q([r[f"{a}_dwql_pct"] for r in ok])[1],
                       "n_within_5pct": sum(r[f"{a}_dmase_pct"] <= 5 for r in ok),
                       "n_within_10pct": sum(r[f"{a}_dmase_pct"] <= 10 for r in ok)}
                if s != P.PREFIX:
                    pp = [r for r in ok if r[f"{a}_vs_prefix_pp"] is not None]
                    q = _q([r[f"{a}_vs_prefix_pp"] for r in pp])
                    nb = sum(r[f"{a}_vs_prefix_pp"] < 0 for r in pp)
                    nw = sum(r[f"{a}_vs_prefix_pp"] > 0 for r in pp)
                    rec.update({"vs_prefix_q25": q[0], "vs_prefix_median": q[1],
                                "vs_prefix_q75": q[2], "n_compared": len(pp), "n_better": nb,
                                "n_worse": nw, "n_better_ci": sum(bool(r[f"{a}_better_ci"])
                                                                  for r in pp),
                                "n_worse_ci": sum(bool(r[f"{a}_worse_ci"]) for r in pp),
                                "sign_test_p": sign_test_p(nb, nw)})
                for lvl, B, key in (("api_e2e", 1, "speedup_b1"), ("api_e2e", 256, "speedup_b256"),
                                    ("device_forward", 1, "speedup_device_b1")):
                    sp = _speed(lat, rr[0]["label"], a, lvl, B)
                    rec[key] = None if sp is None else sp["speedup"]
                    if key == "speedup_b1":
                        rec["speedup_b1_min"] = None if sp is None else sp["speedup_min"]
                        rec["speedup_b1_max"] = None if sp is None else sp["speedup_max"]
                        rec["latency_b1_ms"] = None if sp is None else sp["median_ms"]
                out.append(rec)
    return out


def val_selected_rows(cells) -> list[dict]:
    """Secondary: per budget x arm, the strategy chosen on VALIDATION MASE per dataset."""
    out = []
    for k in P.BUDGETS:
        for a in ("hard", "noa"):
            picks, diffs, nb, nw, nbc, nwc = defaultdict(int), [], 0, 0, 0, 0
            n = 0
            for tag, c in cells["comparison"].items():
                v = c.get("val_selected", {}).get(f"k{k}|{a}")
                if not v:
                    continue
                n += 1
                picks[v["selected"]] += 1
                vp = v.get("vs_prefix")
                d = 0.0 if vp is None else vp["diff_pp"]
                diffs.append(d)
                nb += d < 0
                nw += d > 0
                nbc += bool(vp and vp["better_ci"])
                nwc += bool(vp and vp["worse_ci"])
            if n:
                q = _q(diffs)
                out.append({"budget": k, "arm": a, "n_datasets": n,
                            "picked": dict(picks), "vs_prefix_median": q[1],
                            "vs_prefix_q25": q[0], "vs_prefix_q75": q[2], "n_better": nb,
                            "n_worse": nw, "n_better_ci": nbc, "n_worse_ci": nwc})
    return out


def single_skip_tables(cells) -> tuple[list, list]:
    per, summ = [], []
    by = defaultdict(list)
    for tag, m in sorted(cells["single_skip"].items()):
        for r in m["rows"]:
            rec = {"dataset": tag, "skipped_block": r["skipped_block"],
                   "dmase_pct": r["mase_degradation_pct"],
                   "dmase_ci_lo": r["mase_degradation_ci_pct"][0],
                   "dmase_ci_hi": r["mase_degradation_ci_pct"][1],
                   "dwql_pct": r["wql_degradation_pct"], "val_mase_ratio": r.get("val_mase_ratio")}
            per.append(rec)
            by[r["skipped_block"]].append(rec)
    for i in sorted(by):
        q = _q([r["dmase_pct"] for r in by[i]])
        summ.append({"skipped_block": i, "n_datasets": len(by[i]), "dmase_q25": q[0],
                     "dmase_median": q[1], "dmase_q75": q[2],
                     "n_worse_ci": sum(r["dmase_ci_lo"] > 0 for r in by[i]),
                     "n_better_ci": sum(r["dmase_ci_hi"] < 0 for r in by[i])})
    return per, summ


def latency_rows(lat) -> list[dict]:
    rows = []
    for (lab, kind, level, b), v in sorted(lat["lookup"].items(), key=lambda x: str(x[0])):
        rows.append({"label": lab, "kind": kind, "level": level, "batch": b, **v})
    return rows


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v)
                        for k, v in r.items()})


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def _num(x, nd=1, sign=True) -> str:
    if x is None or not np.isfinite(x):
        return "n/a"
    t = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return "$-$" + t[1:] if t.startswith("-") else t


def _miqr(med, lo, hi, nd=1) -> str:
    if med is None:
        return "n/a"
    return f"{_num(med, nd)} [{_num(lo, nd)}, {_num(hi, nd)}]"


def mask_tex(mask) -> str:
    """Paper-safe block list (no dashes): runs of >= 3 in words, others listed.
    (1,2,3,4,5,9,10,11,12) -> '1 to 5, 9 to 12'; (1,2,11,12) -> '1, 2, 11, 12'."""
    m, runs, i = list(mask), [], 0
    while i < len(m):
        j = i
        while j + 1 < len(m) and m[j + 1] == m[j] + 1:
            j += 1
        runs.append(f"{m[i]} to {m[j]}" if j - i >= 2 else ", ".join(str(b) for b in m[i:j + 1]))
        i = j + 1
    return ", ".join(runs)


def latex_main(summ, path: Path):
    by = {(r["budget"], r["strategy"], r["arm"]): r for r in summ}
    L = [r"\begin{tabular}{lllccccc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Test MASE increase over full model (\%)} & "
         r"\multicolumn{2}{c}{Aligned vs prefix} & Speedup \\",
         r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"$k$ & Strategy & Blocks kept & Hard cut & Aligned & $\Delta$ (pp) & better / worse & "
         r"($B{=}1$) \\", r"\midrule"]
    for k in P.BUDGETS:
        first = True
        for s in P.STRATEGIES:
            h, n = by.get((k, s, "hard")), by.get((k, s, "noa"))
            if not n:
                continue
            vs = ("ref." if s == P.PREFIX else
                  _miqr(n.get("vs_prefix_median"), n.get("vs_prefix_q25"), n.get("vs_prefix_q75")))
            bw = ("ref." if s == P.PREFIX else f"{n['n_better_ci']} / {n['n_worse_ci']}")
            sp = "n/a" if n.get("speedup_b1") is None else f"{n['speedup_b1']:.2f}$\\times$"
            name = f"{P.STRATEGY_LABELS[s]} & {mask_tex(P.MASKS[(s, k)])}"
            L.append(f"{k if first else ''} & {name} & "
                     f"{_miqr(h['dmase_median'], h['dmase_q25'], h['dmase_q75'])} & "
                     f"{_miqr(n['dmase_median'], n['dmase_q25'], n['dmase_q75'])} & {vs} & {bw} & "
                     f"{sp} \\\\")
            first = False
        L.append(r"\midrule" if k != P.BUDGETS[-1] else r"\bottomrule")
    L.append(r"\end{tabular}")
    path.write_text("\n".join(L) + "\n")


def latex_per_dataset(rows, path: Path):
    from probing import registry
    by = {(r["dataset"], r["budget"], r["strategy"]): r for r in rows}
    tags = sorted({r["dataset"] for r in rows},
                  key=lambda t: registry.roster("paper14").index(t))
    cols = [(k, s) for k in P.BUDGETS for s in P.STRATEGIES]
    L = [r"\begin{tabular}{l" + "rrrr" * len(P.BUDGETS) + "}", r"\toprule",
         "Dataset & " + " & ".join(rf"\multicolumn{{4}}{{c}}{{$k={k}$}}" for k in P.BUDGETS)
         + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(len(P.BUDGETS))),
         " & " + " & ".join({"prefix": "P", "uniform": "U", "early_late": "E+L",
                             "middle": "M"}[s] for _, s in cols) + r" \\", r"\midrule"]
    for t in tags:
        cells = []
        for k, s in cols:
            r = by.get((t, k, s))
            if r is None:
                cells.append("n/a")
                continue
            v = _num(r["noa_dmase_pct"], 1)
            if r["status"] != "ok":
                v += r"$^\dagger$"
            elif s != P.PREFIX and r.get("noa_better_ci"):
                v = r"\textbf{" + v + "}"
            cells.append(v)
        L.append(f"{registry.display_name(t)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def latex_single_skip(summ, path: Path):
    L = [r"\begin{tabular}{lccc}", r"\toprule",
         r"Skipped block & Median $\Delta$MASE (\%) [IQR] & worse (CI) & better (CI) \\",
         r"\midrule"]
    for r in summ:
        L.append(f"L{r['skipped_block']} & {_miqr(r['dmase_median'], r['dmase_q25'], r['dmase_q75'])}"
                 f" & {r['n_worse_ci']}/{r['n_datasets']} & {r['n_better_ci']}/{r['n_datasets']} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    path.write_text("\n".join(L) + "\n")


def macros(summ, ss_summ, path: Path):
    L = ["% Phase 2e (non-contiguous block selection, Chronos-2): generated by "
         "experiments.make_phase2e_tables, do not edit"]
    for r in summ:
        stem = f"\\phtwoe{SWORD[r['strategy']]}K{WORD[r['budget']]}" + \
               ("Aligned" if r["arm"] == "noa" else "Hard")
        L.append(f"\\newcommand{{{stem}Median}}{{{_num(r['dmase_median'])}}}")
        if r["strategy"] != P.PREFIX and r.get("n_compared"):
            L.append(f"\\newcommand{{{stem}BetterCI}}{{{r['n_better_ci']}}}")
            L.append(f"\\newcommand{{{stem}WorseCI}}{{{r['n_worse_ci']}}}")
            L.append(f"\\newcommand{{{stem}VsPrefix}}{{{_num(r['vs_prefix_median'])}}}")
        if r.get("speedup_b1") is not None:
            L.append(f"\\newcommand{{{stem}Speedup}}{{{r['speedup_b1']:.2f}}}")
    if ss_summ:
        worst = max(ss_summ, key=lambda r: r["dmase_median"] or -1e9)
        L.append(f"\\newcommand{{\\phtwoeSkipWorstBlock}}{{L{worst['skipped_block']}}}")
        L.append(f"\\newcommand{{\\phtwoeSkipWorstMedian}}{{{_num(worst['dmase_median'])}}}")
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC)
    return plt


SYMLOG_TICKS = (-500, -100, -50, -20, -10, -5, 0, 5, 10, 20, 50, 100, 500, 1000, 5000)


def _plain_symlog(axis, lo, hi):
    """Plain-number ticks (no 10^k labels) on a symlog axis, limited to the data range."""
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    t = [v for v in SYMLOG_TICKS if lo - 1e-9 <= v <= hi + 1e-9]
    if len(t) > 7:                                   # thin out: keep 0 and every other tick
        t = [v for i, v in enumerate(t) if v == 0 or i % 2 == 0]
    axis.set_major_locator(FixedLocator(t))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace("-", "\u2212")))


def _phase2_prefix_curve(arm: str):
    """Committed Phase-2 prefix frontier at every depth (Chronos-2): per depth, the median over
    datasets of test MASE increase, at the committed (separate-job) B=1 speedup. Context only."""
    if not PHASE2_FRONTIER.exists():
        return None
    by = defaultdict(list)
    sp = {}
    with open(PHASE2_FRONTIER) as fh:
        for r in csv.DictReader(fh):
            if r["model"] != P.MODEL or r["arm"] != arm:
                continue
            try:
                d = int(r["depth_index"])
                by[d].append(100 * float(r["test_mase_degradation"]))
                sp[d] = float(r["speedup_e2e_b1"])
            except (TypeError, ValueError):
                continue
    if not by:
        return None
    ds = sorted(d for d in by if 1 <= d <= P.N_BLOCKS - 1)
    return np.array([sp[d] for d in ds]), np.array([np.median(by[d]) for d in ds]), ds


def figure_frontier(summ, lat, paths, title_note=""):
    """x = measured speedup (B=1), y = median test MASE increase with IQR bars. Masks with the
    same k have almost the same speedup, so markers are dodged by a few POINTS (the data x is
    unchanged); the caption must say so."""
    plt = _plt()
    from matplotlib.lines import Line2D
    from matplotlib.transforms import offset_copy
    have_lat = any(r.get("speedup_b1") for r in summ)
    fig, axes = plt.subplots(1, 2, figsize=(TEXTWIDTH_IN, 2.45), layout="constrained")
    for ax, arm in zip(axes, ("noa", "hard")):
        ctx = _phase2_prefix_curve(arm)
        if ctx is not None and have_lat:
            ax.plot(ctx[0], ctx[1], color="#BDBDBD", lw=0.8, marker=".", ms=3, zorder=1)
        ax.axhline(0, color="#9E9E9E", lw=0.5, zorder=0)
        for s in P.STRATEGIES:
            col, mk, ls = STYLE[s]
            rr = sorted((r for r in summ if r["strategy"] == s and r["arm"] == arm
                         and r["dmase_median"] is not None), key=lambda r: r["budget"])
            if not rr:
                continue
            if have_lat:
                x = np.array([r["speedup_b1"] or np.nan for r in rr])
            else:
                x = np.array([r["budget"] + 0.12 * (P.STRATEGIES.index(s) - 1.5) for r in rr])
            y = np.array([r["dmase_median"] for r in rr])
            lo = y - np.array([r["dmase_q25"] for r in rr])
            hi = np.array([r["dmase_q75"] for r in rr]) - y
            dodge = offset_copy(ax.transData, fig=fig, x=3.2 * (P.STRATEGIES.index(s) - 1.5),
                                y=0, units="points")
            ax.errorbar(x, y, yerr=[lo, hi], color=col, marker=mk, ms=4.2, ls=ls, lw=1.1,
                        elinewidth=0.6, capsize=1.5, markeredgecolor="white",
                        markeredgewidth=0.5, zorder=3, transform=dodge)
            # a dodged (non-data-transform) artist does not autoscale: register its extent
            fin = np.isfinite(x)
            ax.update_datalim(np.column_stack([np.r_[x[fin], x[fin]],
                                               np.r_[(y - lo)[fin], (y + hi)[fin]]]))
            ax.autoscale_view()
            if s == P.PREFIX:
                for xi, r in zip(x, rr):
                    if np.isfinite(xi):
                        ax.text(xi, 1.0, f"$k={r['budget']}$", transform=ax.get_xaxis_transform(),
                                ha="center", va="bottom", fontsize=6, color="#4D4D4D")
        ax.plot([1.0], [0.0], marker="*", ms=8, color="black", ls="none", zorder=6,
                clip_on=False)
        ax.set_title(ARM_NAME[arm], fontsize=7.5, pad=10)
        ax.set_xlabel("Measured speedup, end to end, $B{=}1$" if have_lat else
                      "Blocks retained (latency not yet measured)")
        if arm == "hard":
            ax.set_yscale("symlog", linthresh=10)
            lo_, hi_ = ax.get_ylim()
            _plain_symlog(ax.yaxis, lo_, hi_)
        ax.grid(axis="y", color="#EEEEEE", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Test MASE increase (%)\nmedian over datasets, IQR bars")
    h = [Line2D([], [], color=STYLE[s][0], marker=STYLE[s][1], ls=STYLE[s][2], ms=4,
                label=P.STRATEGY_LABELS[s]) for s in P.STRATEGIES]
    h.append(Line2D([], [], marker="*", ms=7, color="black", ls="none", label="Full model"))
    if have_lat and _phase2_prefix_curve("noa") is not None:
        h.append(Line2D([], [], color="#BDBDBD", marker=".", lw=0.8,
                        label="Prefix, every depth (Phase 2)"))
    fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=6.5)
    for p in paths:
        fig.savefig(p, dpi=300)
    plt.close(fig)


def figure_single_skip(per, summ, paths):
    if not per:
        return
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap, SymLogNorm
    from probing import registry
    tags = sorted({r["dataset"] for r in per}, key=lambda t: registry.roster("paper14").index(t))
    M = np.full((len(tags), P.N_BLOCKS), np.nan)
    for r in per:
        M[tags.index(r["dataset"]), r["skipped_block"] - 1] = r["dmase_pct"]
    fig, (a0, a1) = plt.subplots(2, 1, figsize=(TEXTWIDTH_IN, 4.6), layout="constrained",
                                 height_ratios=(1, 2.4), sharex=True)
    xb = np.array([r["skipped_block"] for r in summ])
    med = np.array([r["dmase_median"] for r in summ])
    a0.fill_between(xb, [r["dmase_q25"] for r in summ], [r["dmase_q75"] for r in summ],
                    color="#0072B2", alpha=0.16, lw=0)
    a0.plot(xb, med, color="#0072B2", marker="o", ms=3.5, lw=1.3)
    a0.axhline(0, color="#9E9E9E", lw=0.5)
    a0.set_yscale("symlog", linthresh=5)
    _plain_symlog(a0.yaxis, *a0.get_ylim())
    a0.set_ylabel("Median test MASE\nincrease (%), IQR")
    a0.grid(axis="y", color="#EEEEEE", lw=0.5)
    for sp in ("top", "right"):
        a0.spines[sp].set_visible(False)
    cmap = LinearSegmentedColormap.from_list("div", ["#0072B2", "#EDEDED", "#B2182B"])
    vmax = float(np.nanmax(np.abs(M))) if np.isfinite(M).any() else 1.0
    im = a1.imshow(M, aspect="auto", cmap=cmap,
                   norm=SymLogNorm(linthresh=2.0, vmin=-vmax, vmax=vmax),
                   extent=(0.5, P.N_BLOCKS + 0.5, len(tags) - 0.5, -0.5), interpolation="nearest")
    a1.set_yticks(range(len(tags)))
    a1.set_yticklabels([registry.display_name(t) for t in tags], fontsize=6)
    a1.set_xticks(range(1, P.N_BLOCKS + 1))
    a1.set_xticklabels([f"L{i}" for i in range(1, P.N_BLOCKS + 1)])
    a1.set_xlabel("Skipped block (all other 11 blocks run; no adapter)")
    cb = fig.colorbar(im, ax=a1, shrink=0.9, pad=0.01)
    cb.set_label("Test MASE increase (%)", fontsize=6.5)
    _plain_symlog(cb.ax.yaxis, -vmax, vmax)
    cb.ax.tick_params(labelsize=5.5)
    for p in paths:
        fig.savefig(p, dpi=300)
    plt.close(fig)


def figure_paired(rows, paths, arm="noa"):
    rr = [r for r in rows if r["strategy"] != P.PREFIX and r[f"{arm}_vs_prefix_pp"] is not None
          and r["status"] == "ok"]
    if not rr:
        return
    plt = _plt()
    from matplotlib.lines import Line2D
    from probing import registry
    tags = sorted({r["dataset"] for r in rr}, key=lambda t: registry.roster("paper14").index(t))
    fig, axes = plt.subplots(1, len(P.BUDGETS), figsize=(TEXTWIDTH_IN, 3.6), layout="constrained",
                             sharey=True)
    for ax, k in zip(axes, P.BUDGETS):
        ax.axvline(0, color="black", lw=0.6, zorder=1)
        for j, s in enumerate([s for s in P.STRATEGIES if s != P.PREFIX]):
            col, mk, _ = STYLE[s]
            for r in (r for r in rr if r["budget"] == k and r["strategy"] == s):
                y = tags.index(r["dataset"]) + (j - 1) * 0.22
                x, lo, hi = r[f"{arm}_vs_prefix_pp"], r[f"{arm}_vs_prefix_ci_lo"], \
                    r[f"{arm}_vs_prefix_ci_hi"]
                ax.plot([lo, hi], [y, y], color=col, lw=0.7, zorder=2)
                ax.plot([x], [y], marker=mk, ms=3.4, color=col, markeredgecolor="white",
                        markeredgewidth=0.4, ls="none", zorder=3)
        ax.set_xscale("symlog", linthresh=5)
        _plain_symlog(ax.xaxis, *ax.get_xlim())
        ax.set_title(f"$k={k}$ blocks retained", fontsize=7.5)
        ax.grid(axis="x", color="#EEEEEE", lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_yticks(range(len(tags)))
    axes[0].set_yticklabels([registry.display_name(t) for t in tags], fontsize=6)
    axes[0].invert_yaxis()
    fig.supxlabel("Aligned test MASE minus prefix, percentage points of the full model\n"
                  "(negative = better than prefix; bars = paired 95% cluster-bootstrap CI)",
                  fontsize=7)
    h = [Line2D([], [], color=STYLE[s][0], marker=STYLE[s][1], ls="-", lw=0.7, ms=3.4,
                label=P.STRATEGY_LABELS[s]) for s in P.STRATEGIES if s != P.PREFIX]
    fig.legend(handles=h, loc="outside upper center", ncol=3, frameon=False, fontsize=6.5)
    for p in paths:
        fig.savefig(p, dpi=300)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# the auto report (facts only; the final rebuttal wording is written by a human)
# --------------------------------------------------------------------------- #
def _md_table(rows, cols, head):
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def _f1(x):
    return "n/a" if x is None else f"{x:+.1f}"


def report(cells, summ, vsel, ss_summ, lat, rows, path: Path, rebuttal_path: Path):
    n_ds = len({r["dataset"] for r in rows})
    n_cells = len(rows)
    n_bad = sum(r["status"] != "ok" for r in rows)
    src = defaultdict(int)
    for r in rows:
        if r["strategy"] == P.PREFIX:
            src[r["source"]] += 1
    gates = cells["gates"]
    g_ok = sum(bool(g.get("all_passed")) for g in gates.values())
    v = cells["verification"] or {}
    tab = []
    for r in summ:
        if r["arm"] != "noa":
            continue
        h = next((x for x in summ if x["arm"] == "hard" and x["budget"] == r["budget"]
                  and x["strategy"] == r["strategy"]), {})
        tab.append({"k": r["budget"], "strategy": P.STRATEGY_LABELS[r["strategy"]],
                    "blocks": r["blocks"],
                    "hard": f"{_f1(h.get('dmase_median'))} [{_f1(h.get('dmase_q25'))}, {_f1(h.get('dmase_q75'))}]",
                    "aligned": f"{_f1(r['dmase_median'])} [{_f1(r['dmase_q25'])}, {_f1(r['dmase_q75'])}]",
                    "vs_prefix": "" if r["strategy"] == P.PREFIX else
                    f"{_f1(r.get('vs_prefix_median'))} [{_f1(r.get('vs_prefix_q25'))}, {_f1(r.get('vs_prefix_q75'))}]",
                    "better_worse_ci": "" if r["strategy"] == P.PREFIX else
                    f"{r.get('n_better_ci')}/{r.get('n_worse_ci')} of {r.get('n_compared')}",
                    "sign_p": "" if r.get("sign_test_p") is None else f"{r['sign_test_p']:.3g}",
                    "speedup": "n/a" if r.get("speedup_b1") is None else f"{r['speedup_b1']:.2f}x",
                    "n": r["n_datasets"]})
    best = {}
    for k in P.BUDGETS:
        cand = [r for r in summ if r["arm"] == "noa" and r["budget"] == k
                and r["dmase_median"] is not None]
        if cand:
            best[k] = min(cand, key=lambda r: r["dmase_median"])
    n_ok_cells = n_cells - n_bad
    complete = n_ds == 14 and n_cells == 168 and n_bad == 0
    banner = ([] if complete else [
        f"> **INCOMPLETE RESULTS.** {n_cells} of 168 mask cells over {n_ds} of 14 datasets are "
        f"written ({n_bad} flagged invalid_v3). Every number below covers only what exists; do "
        "not quote it as the full experiment.", ""])
    lines = [
        "# Phase 2e: non-contiguous block selection (Chronos-2), auto-generated report", "",
        *banner,
        "Generated by `python -m experiments.make_phase2e_tables` from cell artifacts only. "
        "Numbers are facts; interpretation and the final rebuttal wording are written by hand.", "",
        "## Coverage and gates", "",
        f"* datasets with mask cells: {n_ds} of 14; mask cells: {n_cells} of 168; "
        f"cells failing physical-vs-offline (V3): {n_bad} (never averaged in)",
        f"* prefix cells by source: {dict(src)} (`h3` = committed H3 numbers after the "
        "reproduction gates; `refit` = state gate failed, refit on re-extracted states)",
        f"* dataset gates (all-12-block backbone == full model; states vs Phase-1 L12): "
        f"{g_ok}/{len(gates)} passed",
        f"* model-level mask checks (trace / restore / identity adapter / physical vs offline / "
        f"weights): {len(v.get('passed_masks', []))} passed, failed {v.get('failed_masks', 'n/a')}",
        f"* latency repeats (headline protocol): "
        f"{sum(bool(r['eligible']) for r in lat['repeats'])} of {len(lat['repeats'])}", "",
        "## Matched-budget comparison (test MASE increase over the full model, %; median [IQR] "
        "over datasets)", "",
        _md_table(tab, ["k", "strategy", "blocks", "hard", "aligned", "vs_prefix",
                        "better_worse_ci", "sign_p", "speedup", "n"],
                  ["k", "strategy", "blocks", "hard cut", "aligned", "aligned vs prefix (pp)",
                   "better/worse (paired CI)", "sign test p", "speedup B=1", "n"]), "",
        "Best aligned strategy by median, per budget: " + "; ".join(
            f"k={k}: {P.STRATEGY_LABELS[r['strategy']]} ({_f1(r['dmase_median'])}%)"
            for k, r in best.items()), "",
        "## Secondary: strategy chosen on VALIDATION MASE per dataset", "",
        _md_table([{**r, "picked": json.dumps(r["picked"]),
                    "vs": f"{_f1(r['vs_prefix_median'])} [{_f1(r['vs_prefix_q25'])}, {_f1(r['vs_prefix_q75'])}]",
                    "bw": f"{r['n_better_ci']}/{r['n_worse_ci']}"}
                   for r in vsel], ["budget", "arm", "picked", "vs", "bw", "n_datasets"],
                  ["k", "arm", "picks", "test vs prefix (pp)", "better/worse (CI)", "n"]), "",
        "## Single-block skipping (hard cut, no adapter)", "",
        _md_table([{**r, "med": f"{_f1(r['dmase_median'])} [{_f1(r['dmase_q25'])}, {_f1(r['dmase_q75'])}]"}
                   for r in ss_summ], ["skipped_block", "med", "n_worse_ci", "n_better_ci",
                                       "n_datasets"],
                  ["skipped block", "median dMASE % [IQR]", "worse (CI)", "better (CI)", "n"]), "",
        "## Limitations (fixed, independent of the outcome)", "",
        "* Chronos-2 only, 14 datasets, three budgets and four PREDEFINED masks: this is not a "
        "search over masks, so it bounds what simple non-contiguous rules achieve, not what the "
        "best possible mask achieves.",
        "* Skipping changes the inputs a retained block receives: a non-prefix mask feeds block j a "
        "state it never saw in pretraining. Poor accuracy of a mask is therefore a joint effect of "
        "the removed computation and this distribution mismatch, and does not show that the "
        "skipped blocks are unimportant.",
        "* The adapter is the H3 residual affine map at the final retained block only. Mismatch "
        "INSIDE the retained stack is not corrected; per-block adapters or fine-tuning could "
        "change the ranking.",
        "* Single-block skipping is a sensitivity diagnostic of the frozen pathway to removing one "
        "block, not a measure of intrinsic block importance (interactions between blocks are "
        "not measured).",
        "* Latency is measured on one node, three in-job repeats, synthetic contexts; prefix and "
        "non-prefix configurations are timed identically in the same job.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")

    # data-driven rebuttal skeleton (numbers filled in; wording to be finalized by hand)
    def get(k, s, a="noa"):
        return next((r for r in summ if r["budget"] == k and r["strategy"] == s and r["arm"] == a),
                    None)
    parts = []
    for k in P.BUDGETS:
        p = get(k, P.PREFIX)
        if not p:
            continue
        seg = [f"with {k} of 12 blocks the prefix degrades MASE by a median "
               f"{_f1(p['dmase_median'])}%"]
        for s in ("uniform", "early_late", "middle"):
            r = get(k, s)
            if r and r.get("n_compared"):
                seg.append(f"{P.STRATEGY_LABELS[s].lower()} {_f1(r['dmase_median'])}% "
                           f"(better than prefix on {r['n_better_ci']} and worse on "
                           f"{r['n_worse_ci']} of {r['n_compared']} datasets, paired CI)")
        parts.append(", ".join(seg))
    sk = ""
    if ss_summ:
        worst = max(ss_summ, key=lambda r: r["dmase_median"] or -1e9)
        mild = min(ss_summ, key=lambda r: r["dmase_median"] or 1e9)
        sk = (f" Removing a single block without any adapter is most damaging for "
              f"L{worst['skipped_block']} (median {_f1(worst['dmase_median'])}%) and least for "
              f"L{mild['skipped_block']} (median {_f1(mild['dmase_median'])}%).")
    speeds = []
    for k in P.BUDGETS:
        sp = [f"{get(k, s)['speedup_b1']:.2f}x" for s in P.STRATEGIES
              if get(k, s) and get(k, s).get("speedup_b1")]
        if sp:
            speeds.append(f"k={k}: " + ", ".join(sp))
    lat_txt = (" Latency was measured, not assumed; end-to-end speedups at batch size 1 "
               "(prefix, uniform, early plus late, middle) were " + "; ".join(speeds) + "."
               if speeds else " (Latency not yet measured.)")
    scope = ("on all 14 datasets" if complete else
             f"on {n_ds} of the 14 datasets ({n_ok_cells} of 168 configurations available)")
    reb = [
        "# Draft rebuttal paragraph (auto-filled numbers; finalize the wording by hand)", "",
        *banner,
        f"We ran this experiment for Chronos-2 {scope}. At a fixed number of retained "
        "blocks (4, 6 or 9) we compared the prefix used in the paper with three predefined masks "
        "fixed before evaluation: uniform spacing, early plus late blocks, and a middle "
        "contiguous window as a control. Skipped blocks are removed from the pretrained backbone "
        "(retained blocks keep their order and weights), and every configuration receives the "
        "same dataset-specific label-free adapter, objective and selection rule as in the paper. "
        "Test MASE relative to the full model, median over datasets: " + "; ".join(parts) + "." +
        sk + lat_txt + " Because skipping feeds retained blocks inputs they never saw in "
        "pretraining, a weaker non-contiguous mask does not imply that the skipped blocks are "
        "unimportant.", ""]
    rebuttal_path.write_text("\n".join(reb) + "\n")


# --------------------------------------------------------------------------- #
def build(root: Path, paper_dir: Path | None) -> dict:
    cells = load_cells(root)
    lat = load_latency(root)
    rows = per_dataset_rows(cells, lat)
    summ = summary_rows(cells, lat, rows)
    vsel = val_selected_rows(cells)
    ss_per, ss_summ = single_skip_tables(cells)
    lrows = latency_rows(lat)
    T, F, R = root / "tables", root / "figures", root / "report"
    for d in (T, F, R):
        d.mkdir(parents=True, exist_ok=True)
    write_csv(T / "per_dataset.csv", rows)
    write_csv(T / "summary_by_budget.csv", summ)
    write_csv(T / "val_selected.csv", vsel)
    write_csv(T / "single_skip_per_dataset.csv", ss_per)
    write_csv(T / "single_skip_summary.csv", ss_summ)
    write_csv(T / "latency.csv", lrows)
    outs = [T]
    if paper_dir is not None:
        paper_dir.mkdir(parents=True, exist_ok=True)
        (paper_dir / "tables").mkdir(exist_ok=True)
        outs.append(paper_dir / "tables")
    for d in outs:
        if summ:
            latex_main(summ, d / "noncontig_budget_table.tex")
        if rows:
            latex_per_dataset(rows, d / "noncontig_per_dataset_table.tex")
        if ss_summ:
            latex_single_skip(ss_summ, d / "single_skip_table.tex")
    macro_dirs = [T] + ([paper_dir] if paper_dir is not None else [])
    for d in macro_dirs:
        macros(summ, ss_summ, d / "phase2e_macros.tex")
    figs = [F] + ([paper_dir] if paper_dir is not None else [])
    if summ:
        figure_frontier(summ, lat, [d / f"noncontig_frontier.{e}" for d in figs
                                    for e in ("pdf", "png")])
        figure_paired(rows, [d / f"noncontig_paired_vs_prefix.{e}" for d in figs
                             for e in ("pdf", "png")])
    if ss_per:
        figure_single_skip(ss_per, ss_summ, [d / f"single_skip_sensitivity.{e}" for d in figs
                                             for e in ("pdf", "png")])
    report(cells, summ, vsel, ss_summ, lat, rows, R / "report.md", R / "rebuttal_draft.md")
    stats = {"n_mask_cells": len(rows), "n_datasets": len({r["dataset"] for r in rows}),
             "summary": summ, "val_selected": vsel, "single_skip_summary": ss_summ,
             "latency_repeats": lat["repeats"]}
    (T / "phase2e_stats.json").write_text(json.dumps(stats, indent=1, default=str))
    print(f"[phase2e tables] {len(rows)} mask cells over {stats['n_datasets']} datasets; "
          f"{len(ss_per)} single-skip rows; {len(lat['repeats'])} latency repeats -> {T}, {F}, {R}"
          + (f", {paper_dir}" if paper_dir else ""))
    return stats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(DEFAULT_ROOT))
    ap.add_argument("--paper-dir", default=str(PAPER_DIR))
    ap.add_argument("--no-paper", action="store_true")
    a = ap.parse_args(argv)
    build(Path(a.root), None if a.no_paper else Path(a.paper_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
