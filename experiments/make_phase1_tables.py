"""Rebuild the Phase-1 combined tables FROM CELL ARTIFACTS ONLY. No model, no GPU, no cache.

    python -m experiments.make_phase1_tables --output-root results/three_model_final

This is deliberately a separate, re-runnable step with exactly one input — the per-cell files
written by the orchestrator — so that:

  * combined tables can be regenerated after any change to their layout without touching a GPU;
  * the tables can be inspected WHILE the 42-cell job is still running: a missing cell is simply
    absent from the table and listed in ``model_dataset_manifest.csv`` with its status, never
    silently filled in or averaged over;
  * the reconstruction is itself a check — if a cell's artifacts cannot rebuild its rows, the
    cell is not durable, which is a bug in the writer, not in the plotting code.

WHERE TO RUN: the login node is fine. It reads a few MB of CSV/JSON and finishes in seconds.

OUTPUTS (under ``<output-root>/combined/``)
    tunnel_matrix.csv          models x datasets, normalized tunnel entrance (+ the absolute
                               representation label, which a normalized number cannot replace)
    layer_metrics_all.csv      one row per model x dataset x layer, every layerwise metric
    geometry_summary.csv       one row per model x dataset x variant x split x layer
    cka_index.csv              where every CKA matrix lives, with its null floor
    provenance_matrix.csv      all 42 model x dataset provenance cells (status x evidence_scope)
    dataset_metadata.csv       frequency, domain, seasonal m, role, realized window yields
    model_dataset_manifest.csv per-cell status, timings, tunnel entrance, key numbers
    plot_data.csv              the long-format frame most Phase-1 figures are drawn from
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

from probing import phase1, registry                                    # noqa: E402
from probing.phase1 import MODELS, ROSTER, write_csv                    # noqa: E402
from probing.phase1_cells import cell_dirs                              # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "results" / "three_model_final"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def collect(root: Path, roster=ROSTER) -> dict:
    """Every COMPLETE cell's artifacts, keyed (model, dataset). Incomplete cells are recorded
    as such and contribute no rows — never a partial row, never an imputed one."""
    tags = registry.roster(roster)
    cells, manifest = {}, []
    for model in MODELS:
        for tag in tags:
            d = cell_dirs(root, model, tag)
            complete = (d / "COMPLETE").exists()
            rec = {"model": model, "dataset": tag,
                   "display_name": registry.display_name(tag),
                   "status": "complete" if complete else "missing",
                   "path": str(d.relative_to(root)) if d.exists() else ""}
            if complete:
                summary = _read_json(d / "summary.json")
                tun = _read_json(d / "tunnel.json")
                cells[(model, tag)] = {
                    "summary": summary, "tunnel": tun,
                    "layers": _read_csv(d / "layer_metrics.csv"),
                    "erank": _read_csv(d / "effective_rank.csv"),
                    "cka": (_read_json(d / "cka" / "cka_metadata.json") or {}).get("index", []),
                    "marker": _read_json(d / "COMPLETE")}
                h = tun["headline"]
                nat = (summary or {}).get("native_baseline") or {}
                rec.update(
                    tunnel_layer=h["label"], tunnel_index=h["index"],
                    tunnel_relative_depth=h["relative_depth"],
                    tunnel_relative_position=h["relative_position"],
                    n_points=summary["model_spec"]["n_points"],
                    n_depth_points=summary["model_spec"]["n_depth_points"],
                    reference_layer=summary["model_spec"]["tunnel_reference_point"],
                    # final DEPTH loss = the final BLOCK, never a final normalization
                    final_val_loss=summary["final_depth_loss"]["val"],
                    final_test_loss=summary["final_depth_loss"]["test"],
                    native_readout_layer=summary["model_spec"]["native_readout_point"],
                    native_loss=nat.get("loss"), native_mase=nat.get("mase"),
                    n_train=summary["window_identity"].get("n_train_windows"),
                    n_val=summary["window_identity"].get("n_val_windows"),
                    n_test=summary["window_identity"].get("n_test_windows"),
                    n_test_clusters=summary["window_identity"].get("n_test_series"),
                    seasonal_m=summary.get("seasonal_m"),
                    provenance_status=summary["pretraining_provenance"]["status"],
                    provenance_scope=summary["pretraining_provenance"]["evidence_scope"],
                    parity_ok=summary["window_identity"].get("parity_ok"),
                    completed_utc=(rec.get("completed_utc")
                                   or (_read_json(d / "COMPLETE") or {}).get("completed_utc")),
                    elapsed_s=summary.get("timings", {}).get("total_s"),
                    n_layers_at_wd_grid_max=sum(
                        1 for r in _read_csv(d / "layer_metrics.csv")
                        if str(r.get("wd_at_grid_max", "")).lower() == "true"))
                final_mase = [r for r in cells[(model, tag)]["layers"]
                              if r["layer"] == rec["reference_layer"]
                              and r.get("point_type", "block_depth") == "block_depth"]
                if final_mase and final_mase[0].get("test_mase"):
                    rec["final_test_mase"] = float(final_mase[0]["test_mase"])
            manifest.append(rec)
    return {"cells": cells, "manifest": manifest, "tags": tags}


def build(root: Path, roster=ROSTER) -> dict:
    root = Path(root)
    out = root / "combined"
    out.mkdir(parents=True, exist_ok=True)
    got = collect(root, roster)
    cells, tags = got["cells"], got["tags"]

    # ---- 1. tunnel matrix: rows = models, columns = the 14 datasets ---------- #
    tm_rows = []
    for model in MODELS:
        norm = {"model": model, "quantity": "normalized_tunnel_entrance (block_index/num_blocks)"}
        lab = {"model": model, "quantity": "tunnel_entrance_representation_label"}
        pos = {"model": model, "quantity": "tunnel_entrance_relative_position"}
        for tag in tags:
            c = cells.get((model, tag))
            h = c["tunnel"]["headline"] if c else None
            norm[tag] = "" if h is None else h["relative_depth"]
            lab[tag] = "" if h is None else h["label"]
            pos[tag] = "" if h is None else h["relative_position"]
        tm_rows += [norm, lab, pos]
    write_csv(out / "tunnel_matrix.csv", tm_rows, ["model", "quantity"] + tags)

    # ---- 2/3/4. long tables -------------------------------------------------- #
    layers = [r for c in cells.values() for r in c["layers"]]
    if layers:
        lead = ["model", "dataset", "display_name", "point_index", "layer", "kind",
                "point_type", "include_in_main_depth_axis",
                "block_index", "relative_depth", "relative_position",
                "is_tunnel_entrance", "is_reference_point", "is_native_readout"]
        rest = sorted({k for r in layers for k in r} - set(lead))
        write_csv(out / "layer_metrics_all.csv", layers,
                  [k for k in lead if any(k in r for r in layers)] + rest)
    erank = [r for c in cells.values() for r in c["erank"]]
    if erank:
        write_csv(out / "geometry_summary.csv", erank, list(erank[0]))
    cka = [r for c in cells.values() for r in c["cka"]]
    if cka:
        write_csv(out / "cka_index.csv", cka, list(cka[0]))

    # ---- 5. provenance: all 42 cells, always complete ------------------------ #
    write_csv(out / "provenance_matrix.csv", phase1.provenance_rows(roster),
              ["model", "dataset", "display_name", "role", "domain", "freq", "seasonal_m",
               "status", "evidence_scope", "verified", "citation"])

    # ---- 6. dataset metadata + realized yields ------------------------------- #
    meta_rows = []
    for tag in tags:
        s = registry.spec(tag)
        row = {**s.as_dict(), "is_frequency_control": registry.is_control(tag)}
        for model in MODELS:
            c = cells.get((model, tag))
            wi = (c["summary"]["window_identity"] if c else {})
            row[f"{model}_n_train"] = wi.get("n_train_windows", "")
            row[f"{model}_n_val"] = wi.get("n_val_windows", "")
            row[f"{model}_n_test"] = wi.get("n_test_windows", "")
            row[f"{model}_n_test_clusters"] = wi.get("n_test_series", "")
        meta_rows.append(row)
    write_csv(out / "dataset_metadata.csv", meta_rows, list(meta_rows[0]) if meta_rows else [])

    # ---- 7. per-cell manifest ------------------------------------------------ #
    lead = ["model", "dataset", "display_name", "status", "tunnel_layer",
            "tunnel_relative_depth", "reference_layer", "final_val_loss", "final_test_loss",
            "final_test_mase", "native_mase", "elapsed_s"]
    man_keys = ([k for k in lead if any(k in r for r in got["manifest"])]
                + sorted({k for r in got["manifest"] for k in r} - set(lead)))
    write_csv(out / "model_dataset_manifest.csv", got["manifest"], man_keys)

    # ---- 8. the long plotting frame ------------------------------------------ #
    # plot_data.csv is the DEPTH AXIS ONLY. A head-input diagnostic (a final normalization) is
    # not a model depth, so it is not in the frame the main figures are drawn from -- it goes
    # to its own file. That is what makes accidental inclusion impossible rather than merely
    # discouraged: a script would have to open a differently-named file to get one.
    plot_rows, diag_rows = [], []
    for (model, tag), c in cells.items():
        prov = c["summary"]["pretraining_provenance"]
        by_layer = {r["layer"]: r for r in c["layers"]}
        erank_head = {r["layer"]: r for r in c["erank"]
                      if str(r.get("is_headline", "")).lower() == "true"
                      and r["split"] == "train"}
        for r in c["layers"]:
            on_axis = str(r.get("include_in_main_depth_axis", "True")).lower() == "true"
            er = erank_head.get(r["layer"], {})
            (plot_rows if on_axis else diag_rows).append({
                "model": model, "dataset": tag,
                "display_name": registry.display_name(tag),
                "role": registry.role(tag), "domain": registry.spec(tag).domain,
                "freq": registry.spec(tag).freq, "seasonal_m": registry.seasonal_m(tag),
                "provenance_status": prov["status"],
                "provenance_evidence_scope": prov["evidence_scope"],
                "layer": r["layer"], "point_index": r["point_index"],
                "point_type": r.get("point_type", "block_depth"),
                "relative_depth": r["relative_depth"],
                "relative_position": r["relative_position"],
                "kind": r["kind"],
                "train_loss": r["train_loss"], "val_loss": r["val_loss"],
                "test_loss": r["test_loss"],
                "test_loss_ci_lo": r.get("test_loss_ci_lo", ""),
                "test_loss_ci_hi": r.get("test_loss_ci_hi", ""),
                "test_mase": r.get("test_mase", ""), "test_mae": r.get("test_mae", ""),
                "test_wql": r.get("test_wql", ""),
                "weight_decay": r["weight_decay"],
                "is_tunnel_entrance": r["is_tunnel_entrance"],
                "is_reference_point": r["is_reference_point"],
                "effective_rank_train_headline": er.get("effective_rank", ""),
                "normalized_effective_rank_train_headline":
                    er.get("normalized_effective_rank", ""),
                "native_loss": (c["summary"].get("native_baseline") or {}).get("loss", ""),
                "native_mase": (c["summary"].get("native_baseline") or {}).get("mase", ""),
            })
        _ = by_layer
    if plot_rows:
        write_csv(out / "plot_data.csv", plot_rows, list(plot_rows[0]))
    if diag_rows:
        write_csv(out / "plot_data_head_input.csv", diag_rows, list(diag_rows[0]))

    n_done = len(cells)
    n_total = len(MODELS) * len(tags)
    phase1.atomic_write_json(out / "combined_index.json", {
        "schema": "phase1_combined/v1",
        "generated_from": "cell artifacts only (no model, no GPU, no feature cache)",
        "roster": roster, "n_cells_complete": n_done, "n_cells_total": n_total,
        "complete": sorted(f"{m}/{t}" for m, t in cells),
        "missing": sorted(f"{r['model']}/{r['dataset']}" for r in got["manifest"]
                          if r["status"] != "complete"),
        "files": sorted(p.name for p in out.glob("*.csv")),
        "main_depth_axis": {
            "frame": "plot_data.csv",
            "contains": "block_depth points only (Emb, L1..L_N)",
            "head_input_frame": "plot_data_head_input.csv",
            "note": ("a final normalization (Chronos-2 L12+LN, TiRex L12+RMS) is a HEAD-INPUT "
                     "DIAGNOSTIC, not a model depth. It is excluded from plot_data.csv, from "
                     "the depth-axis CKA matrices and from every normalized depth. "
                     "layer_metrics_all.csv and geometry_summary.csv keep it, tagged with "
                     "point_type / include_in_main_depth_axis.")},
        "head_input_caveat": phase1.HEAD_INPUT_CAVEAT,
        "cross_model_loss_caveat": phase1.CROSS_MODEL_LOSS_CAVEAT,
        "geometry_caveat": phase1.GEOMETRY_CAVEAT,
        "partial_run_note": ("a missing cell contributes NO rows to any table and is listed in "
                             "model_dataset_manifest.csv with status != complete; nothing is "
                             "imputed or averaged over")})
    return {"n_complete": n_done, "n_total": n_total, "out": out,
            "missing": [f"{r['model']}/{r['dataset']}" for r in got["manifest"]
                        if r["status"] != "complete"]}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", default=str(DEFAULT_ROOT))
    p.add_argument("--roster", default=ROSTER)
    a = p.parse_args(argv)
    r = build(Path(a.output_root), a.roster)
    print(f"[combined] {r['n_complete']}/{r['n_total']} cells -> {r['out']}")
    if r["missing"]:
        print(f"[combined] not yet complete ({len(r['missing'])}): "
              + ", ".join(r["missing"][:12]) + (" ..." if len(r["missing"]) > 12 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
