"""Compare the Q=9 headline Phase-1 run against the Q=1 robustness run. Reads BOTH trees, writes
a THIRD. No GPU, no model, no feature cache.

    python -m experiments.make_phase1_q1_q9_comparison \\
        --q9-root results/three_model_phase1_q9_expanded \\
        --q1-root results/three_model_phase1_q1 \\
        --out-root results/phase1_q1_q9_comparison

WHAT THE COMPARISON IS FOR, stated so a table is not over-read. Q=1 is NOT a replacement for
Q=9 and a Q1/Q9 disagreement is NOT by itself evidence that Q=9 overfits. Q=9 remains the
headline probabilistic experiment; Q=1 asks one question:

    does the layerwise forecast-recoverability conclusion persist when the readout is much
    lower capacity and predicts only the median?

The three tables therefore report DESCRIPTIVE agreement, not a test:
    tunnel_comparison.csv          per model x dataset: both runs' sustained entrances at 2/5/10%
    wd_comparison.csv              per model x dataset: grid-edge behaviour and median wd, both Q
    generalization_diagnostics.csv per model x dataset x depth: train/val/test loss and the
                                   val/test ratio to the final block, in BOTH runs, side by side

Signs that WOULD support a probe-capacity problem at Q=9 -- strong train improvement with poor
validation and test, persistent grid-max clipping, a much earlier Q9 entrance whose test ratio
does not hold -- are all readable from these three files. The tool computes them and never
labels them.

WHAT IT REFUSES. Two runs are comparable only if they probed the SAME windows with the SAME
protocol apart from Q. The tool checks the window digest, C/H, the suite, the seed, the wd grid,
epochs/lr, the tunnel definition and the registry hash per cell, and reports every mismatch in
``comparability.csv``. A cell that differs in anything other than the quantile set is EXCLUDED
from the agreement statistics and named in the summary -- never silently averaged in.

GEOMETRY IS Q-INDEPENDENT. CKA and effective rank are computed from raw representations and
never see a quantile vector, so the two runs' matrices must be identical. ``--check-geometry``
verifies that element-wise rather than asserting it, and records the worst deviation.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, registry                                      # noqa: E402
from probing.phase1 import MODELS, ROSTER, write_csv                      # noqa: E402
from probing.phase1_cells import cell_dirs                                # noqa: E402

DEFAULT_Q9 = REPO_ROOT / "results" / "three_model_phase1_q9_expanded"
DEFAULT_Q1 = REPO_ROOT / "results" / "three_model_phase1_q1"
DEFAULT_OUT = REPO_ROOT / "results" / "phase1_q1_q9_comparison"

#: Everything that must match for two cells to be comparable. The quantile set is the ONE thing
#: allowed to differ -- that is the experiment.
COMPARABLE_KEYS = ("C", "H", "suite", "seed", "window_digest", "registry_hash", "wd_grid",
                   "probe_epochs", "probe_lr", "tunnel_definition", "tunnel_tol", "protocol",
                   "checkpoint", "boot_b")


def _json(p: Path):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def _csv(p: Path) -> list[dict]:
    if not Path(p).exists():
        return []
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def _cell(root: Path, model: str, tag: str):
    d = cell_dirs(root, model, tag)
    if not (d / "COMPLETE").exists():
        return None
    return {"dir": d, "summary": _json(d / "summary.json"), "tunnel": _json(d / "tunnel.json"),
            "cfg": _json(d / "cell_config.json"), "layers": _csv(d / "layer_metrics.csv"),
            "wd": _csv(d / "wd_selection.csv"), "hparams": _json(d / "probe_hparams.json")}


def _entrance(tun, tol, first_crossing=False):
    if tun is None:
        return None
    src = tun.get("first_crossing" if first_crossing else "by_tolerance", {}) or {}
    return src.get(f"first_crossing_{tol:g}" if first_crossing else f"tol_{tol:g}")


def _num(x, default=""):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def comparability(c9, c1) -> tuple[bool, list[str]]:
    """Same windows, same protocol, differing ONLY in the quantile set?"""
    if c9 is None or c1 is None:
        return False, ["one of the two cells is not complete"]
    a, b = c9["cfg"] or {}, c1["cfg"] or {}
    bad = [f"{k}: q9={a.get(k)!r} vs q1={b.get(k)!r}"
           for k in COMPARABLE_KEYS if a.get(k) != b.get(k)]
    if a.get("quantile_set") == b.get("quantile_set"):
        bad.append(f"both runs carry quantile_set={a.get('quantile_set')!r} — these are not a "
                   "Q1/Q9 pair")
    return (not bad), bad


def geometry_identity(c9, c1) -> dict:
    """CKA and effective rank are Q-independent: VERIFY, do not assume."""
    out = {"checked": 0, "max_abs_cka_diff": 0.0, "max_abs_erank_diff": 0.0, "missing": []}
    d9, d1 = c9["dir"] / "cka", c1["dir"] / "cka"
    for f in sorted(d9.glob("*.npy")):
        g = d1 / f.name
        if not g.exists():
            out["missing"].append(f.name)
            continue
        A, B = np.load(f), np.load(g)
        if A.shape != B.shape:
            out["missing"].append(f"{f.name} (shape {A.shape} vs {B.shape})")
            continue
        out["max_abs_cka_diff"] = max(out["max_abs_cka_diff"],
                                      float(np.abs(A - B).max()) if A.size else 0.0)
        out["checked"] += 1
    e9 = {(r["geometry_variant"], r["split"], r["layer"]): _num(r["effective_rank"])
          for r in _csv(c9["dir"] / "effective_rank.csv")}
    e1 = {(r["geometry_variant"], r["split"], r["layer"]): _num(r["effective_rank"])
          for r in _csv(c1["dir"] / "effective_rank.csv")}
    both = set(e9) & set(e1)
    if both:
        out["max_abs_erank_diff"] = max(abs(e9[k] - e1[k]) for k in both)
    out["n_erank_points"] = len(both)
    return out


def build(q9_root: Path, q1_root: Path, out_root: Path, roster=ROSTER,
          check_geometry: bool = True) -> dict:
    out_root = Path(out_root)
    out = out_root / "combined_q1_q9"
    out.mkdir(parents=True, exist_ok=True)
    tags = registry.roster(roster)
    tols = list(phase1.PHASE1_REPORTED_TOLS)
    head = phase1.PHASE1_TUNNEL_TOL

    tunnel_rows, wd_rows, gen_rows, comp_rows, geom_rows = [], [], [], [], []
    for model in MODELS:
        for tag in tags:
            c9, c1 = _cell(q9_root, model, tag), _cell(q1_root, model, tag)
            ok, why = comparability(c9, c1)
            comp_rows.append({"model": model, "dataset": tag,
                              "display_name": registry.display_name(tag),
                              "q9_complete": c9 is not None, "q1_complete": c1 is not None,
                              "comparable": ok, "mismatches": " ; ".join(why)})
            if c9 is None and c1 is None:
                continue

            # ---- tunnel comparison ------------------------------------------------- #
            row = {"model": model, "dataset": tag,
                   "display_name": registry.display_name(tag),
                   "comparable": ok,
                   "n_depth_points": len((c9 or c1)["tunnel"]["depth_axis_labels"]),
                   "reference_layer": (c9 or c1)["tunnel"]["reference_point"]}
            for q, c in (("q9", c9), ("q1", c1)):
                for t in tols:
                    e = _entrance((c or {}).get("tunnel"), t)
                    key = f"{q}_tunnel_{int(round(t * 100)):02d}"
                    row[key] = "" if e is None else e["label"]
                    row[f"{key}_depth_index"] = "" if e is None else e["depth_axis_index"]
                    row[f"{key}_normalized"] = "" if e is None else e["relative_depth"]
                fc = _entrance((c or {}).get("tunnel"), head, first_crossing=True)
                row[f"{q}_first_crossing_05"] = "" if fc is None else fc["label"]
            # agreement at the HEADLINE tolerance, in depth-axis blocks
            k = f"_tunnel_{int(round(head * 100)):02d}_depth_index"
            a, b = row.get(f"q9{k}", ""), row.get(f"q1{k}", "")
            if ok and a != "" and b != "":
                row["delta_depth_blocks_q1_minus_q9"] = int(b) - int(a)
                row["abs_delta_depth_blocks"] = abs(int(b) - int(a))
                row["same_tunnel_05"] = int(a) == int(b)
                na = _num(row.get(f"q9_tunnel_{int(round(head*100)):02d}_normalized"))
                nb = _num(row.get(f"q1_tunnel_{int(round(head*100)):02d}_normalized"))
                if na != "" and nb != "":
                    row["delta_normalized_q1_minus_q9"] = nb - na
            tunnel_rows.append(row)

            # ---- weight-decay comparison -------------------------------------------- #
            wrow = {"model": model, "dataset": tag,
                    "display_name": registry.display_name(tag), "comparable": ok}
            for q, c in (("q9", c9), ("q1", c1)):
                gc = (((c or {}).get("hparams") or {}).get("grid_clipping") or {})
                wrow[f"{q}_n_at_grid_max"] = gc.get("n_layers_at_grid_max", "")
                wrow[f"{q}_n_at_grid_min"] = gc.get("n_layers_at_grid_min", "")
                wrow[f"{q}_fraction_at_grid_max"] = gc.get("fraction_at_grid_max", "")
                wrow[f"{q}_median_selected_wd"] = gc.get("median_selected_wd", "")
                wrow[f"{q}_grid_max"] = gc.get("grid_max", "")
                wrow[f"{q}_layers_at_grid_max"] = "|".join(gc.get("layers_at_grid_max", []) or [])
                wrow[f"{q}_floor_val_loss"] = gc.get("constant_forecast_floor_val_loss", "")
            wd_rows.append(wrow)

            # ---- per-depth generalization diagnostics --------------------------------- #
            l9 = {r["layer"]: r for r in (c9 or {}).get("layers", [])}
            l1 = {r["layer"]: r for r in (c1 or {}).get("layers", [])}
            wd9 = {r["layer"]: r for r in (c9 or {}).get("wd", [])}
            wd1 = {r["layer"]: r for r in (c1 or {}).get("wd", [])}
            for lab in ((c9 or c1)["summary"]["model_spec"]["labels"]):
                r9, r1 = l9.get(lab, {}), l1.get(lab, {})
                base = r9 or r1
                if not base:
                    continue
                g = {"model": model, "dataset": tag,
                     "display_name": registry.display_name(tag), "layer": lab,
                     "point_index": base.get("point_index", ""),
                     "point_type": base.get("point_type", ""),
                     "include_in_main_depth_axis": base.get("include_in_main_depth_axis", ""),
                     "relative_depth": base.get("relative_depth", "")}
                for q, r, wr in (("q9", r9, wd9.get(lab, {})), ("q1", r1, wd1.get(lab, {}))):
                    g[f"{q}_train_loss"] = r.get("train_loss", "")
                    g[f"{q}_val_loss"] = r.get("val_loss", "")
                    g[f"{q}_test_loss"] = r.get("test_loss", "")
                    g[f"{q}_val_ratio_to_final"] = r.get("val_ratio_to_final", "")
                    g[f"{q}_test_ratio_to_final"] = r.get("test_ratio_to_final", "")
                    g[f"{q}_selected_wd"] = r.get("weight_decay", "")
                    g[f"{q}_wd_at_grid_max"] = r.get("wd_at_grid_max", "")
                    g[f"{q}_test_mase"] = r.get("test_mase", "")
                    g[f"{q}_val_over_floor"] = wr.get("val_over_floor", "")
                    tr, va = _num(r.get("train_loss")), _num(r.get("val_loss"))
                    # train-vs-val gap: the quantity a capacity worry is actually about. It is
                    # reported, never interpreted here.
                    g[f"{q}_val_minus_train"] = (va - tr) if "" not in (tr, va) else ""
                    g[f"{q}_val_over_train"] = (va / tr) if "" not in (tr, va) and tr else ""
                gen_rows.append(g)

            # ---- geometry identity ---------------------------------------------------- #
            if check_geometry and ok:
                gi = geometry_identity(c9, c1)
                geom_rows.append({"model": model, "dataset": tag, **gi,
                                  "missing": "|".join(gi["missing"])})

    write_csv(out / "tunnel_comparison.csv", tunnel_rows,
              list(tunnel_rows[0]) if tunnel_rows else [])
    write_csv(out / "wd_comparison.csv", wd_rows, list(wd_rows[0]) if wd_rows else [])
    write_csv(out / "generalization_diagnostics.csv", gen_rows,
              list(gen_rows[0]) if gen_rows else [])
    write_csv(out / "comparability.csv", comp_rows, list(comp_rows[0]) if comp_rows else [])
    if geom_rows:
        write_csv(out / "geometry_identity.csv", geom_rows, list(geom_rows[0]))

    # ---- descriptive agreement, per model and overall --------------------------------- #
    usable = [r for r in tunnel_rows
              if r.get("comparable") and r.get("abs_delta_depth_blocks") != ""
              and "abs_delta_depth_blocks" in r]
    def agree(rows):
        if not rows:
            return {"n_cells": 0}
        d = [int(r["abs_delta_depth_blocks"]) for r in rows]
        dn = [float(r["delta_normalized_q1_minus_q9"]) for r in rows
              if r.get("delta_normalized_q1_minus_q9") not in ("", None)]
        return {"n_cells": len(rows),
                "n_exact_same_tunnel_05": sum(1 for x in d if x == 0),
                "fraction_exact": sum(1 for x in d if x == 0) / len(d),
                "n_within_1_block": sum(1 for x in d if x <= 1),
                "fraction_within_1_block": sum(1 for x in d if x <= 1) / len(d),
                "n_within_2_blocks": sum(1 for x in d if x <= 2),
                "fraction_within_2_blocks": sum(1 for x in d if x <= 2) / len(d),
                "median_abs_delta_blocks": float(statistics.median(d)),
                "mean_signed_delta_blocks_q1_minus_q9": float(
                    statistics.fmean(int(r["delta_depth_blocks_q1_minus_q9"]) for r in rows)),
                "median_abs_delta_normalized_depth": (float(statistics.median([abs(x)
                                                                               for x in dn]))
                                                      if dn else None)}
    summary = {
        "schema": "phase1_q1_q9_comparison/v1",
        "q9_root": str(q9_root), "q1_root": str(q1_root),
        "headline_tolerance": head, "reported_tolerances": tols,
        "tunnel_definition": phase1.TUNNEL_DEFINITION_VERSION,
        "agreement_overall": agree(usable),
        "agreement_by_model": {m: agree([r for r in usable if r["model"] == m]) for m in MODELS},
        "n_cells_not_comparable": sum(1 for r in comp_rows if not r["comparable"]),
        "cells_not_comparable": [f"{r['model']}/{r['dataset']}: {r['mismatches']}"
                                 for r in comp_rows if not r["comparable"]],
        "geometry_identity": {
            "checked_cells": len(geom_rows),
            "max_abs_cka_diff": max([r["max_abs_cka_diff"] for r in geom_rows], default=None),
            "max_abs_erank_diff": max([r["max_abs_erank_diff"] for r in geom_rows], default=None),
            "expectation": ("CKA and effective rank never see a quantile vector, so both should "
                            "be 0.0 exactly when the two runs shared a feature cache; any "
                            "non-zero value means the two runs did NOT read the same "
                            "representations and the comparison must be re-examined")},
        "interpretation_caveat": (
            "Q=9 is the headline probabilistic experiment; Q=1 (tau=0.5) is a probe-capacity / "
            "objective robustness check. A Q1/Q9 difference is NOT by itself evidence of "
            "overfitting at Q=9 — the two optimize different objectives with different readout "
            "capacity, and the losses are not on one scale. These tables are descriptive."),
        "loss_scale_caveat": (
            "q9 and q1 losses are BOTH mean pinball, but over different quantile sets, so their "
            "absolute values are not comparable. What is comparable: the tunnel entrance (a "
            "within-run ratio against that run's own final block), MASE and MAE (raw units)."),
    }
    phase1.atomic_write_json(out / "comparison_summary.json", summary)
    return {"out": out, "n_tunnel_rows": len(tunnel_rows), "n_comparable": len(usable),
            "summary": summary}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--q9-root", default=str(DEFAULT_Q9))
    ap.add_argument("--q1-root", default=str(DEFAULT_Q1))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT))
    ap.add_argument("--roster", default=ROSTER)
    ap.add_argument("--no-check-geometry", action="store_true")
    a = ap.parse_args(argv)
    r = build(Path(a.q9_root), Path(a.q1_root), Path(a.out_root), a.roster,
              check_geometry=not a.no_check_geometry)
    s = r["summary"]
    print(f"\n[q1 vs q9] {r['n_comparable']} comparable cells -> {r['out']}")
    ov = s["agreement_overall"]
    if ov.get("n_cells"):
        print(f"  sustained {int(s['headline_tolerance'] * 100)}% entrance: "
              f"{ov['n_exact_same_tunnel_05']}/{ov['n_cells']} exactly equal, "
              f"{ov['n_within_1_block']} within 1 block, {ov['n_within_2_blocks']} within 2; "
              f"median |delta| = {ov['median_abs_delta_blocks']:g} blocks")
    if s["n_cells_not_comparable"]:
        print(f"  {s['n_cells_not_comparable']} cell(s) NOT comparable and excluded:")
        for line in s["cells_not_comparable"][:8]:
            print(f"    {line}")
    g = s["geometry_identity"]
    if g["checked_cells"]:
        print(f"  geometry identity over {g['checked_cells']} cells: "
              f"max |dCKA| = {g['max_abs_cka_diff']:.3g}, "
              f"max |d effective rank| = {g['max_abs_erank_diff']:.3g}  (expected 0.0)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
