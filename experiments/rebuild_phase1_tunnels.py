"""Re-derive every Phase-1 cell's TUNNEL from artifacts already on disk. No GPU, no model.

    python -m experiments.rebuild_phase1_tunnels --output-root results/three_model_phase1_q9_expanded

WHY THIS EXISTS. A cell's identity has two halves (``probing.phase1.split_cell_config``):

    FIT          the checkpoint, the extraction parameters, the probe protocol (wd grid /
                 epochs / lr / seed / architecture), the windows, the bootstrap B.
                 Changing any of these means refitting probes -- GPU hours.
    POSTPROCESS  the tunnel definition, the headline tolerance and the tolerance set.
                 These are PURE FUNCTIONS of the validation and test curves the cell already
                 saved, so changing one costs seconds and touches no probe.

Without the split, changing the tunnel definition would invalidate 42 completed cells and
re-burn a day of A100 time to reproduce bit-identical probe weights. With it, the driver
REFUSES such a cell with status ``stale_postprocess`` and names this tool, which re-derives the
tunnel in place and re-stamps the cell.

WHAT IT REWRITES, and nothing else:
    tunnel.json          entirely regenerated from the saved val/test loss curves
    layer_metrics.csv    ONLY the tunnel-derived columns (the entrance flags, the val/test
                         ratios and the suffix excursions); every other column is copied
                         through byte for byte
    summary.json         only its tunnel block and the ratio curves
    cell_config.json     its postprocess keys, plus the recomputed hashes
    COMPLETE             config_hash / fit_hash / postprocess_hash / tunnel_definition

It NEVER touches predictions_*.npz, bootstrap_inputs.npz, probe_artifacts/, cka/ or
spectral_metrics.npz -- none of which a tunnel definition can affect.

WHERE TO RUN: the login node. It reads a few MB of JSON/CSV per cell and finishes in seconds.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, registry                                          # noqa: E402
from probing.phase1 import (MODELS, PHASE1_TUNNEL_TOL, PHASE1_TUNNEL_TOLS,    # noqa: E402
                            ROSTER, TUNNEL_DEFINITION_VERSION, model_spec)
from probing.phase1_cells import cell_dirs                                    # noqa: E402

DEFAULT_ROOT = REPO_ROOT / "results" / "three_model_phase1_q9_expanded"

#: Columns of layer_metrics.csv that the tunnel definition owns. Everything else in that file is
#: a probe or bootstrap quantity and is copied through unchanged.
TUNNEL_OWNED_PREFIXES = ("is_tunnel_entrance", "is_first_crossing")
TUNNEL_OWNED = ("val_ratio_to_final", "test_ratio_to_final",
                "val_suffix_excursion", "test_suffix_excursion")


def _read_json(p: Path):
    return json.loads(Path(p).read_text()) if Path(p).exists() else None


def _read_csv(p: Path) -> list[dict]:
    with open(p, newline="") as fh:
        r = csv.DictReader(fh)
        return list(r), list(r.fieldnames or [])


def rebuild_cell(d: Path, *, tol: float, tols, dry_run: bool = False) -> dict:
    """Re-derive one cell's tunnel. Returns a report; raises only on a genuinely broken cell."""
    summary = _read_json(d / "summary.json")
    cfg = _read_json(d / "cell_config.json")
    marker = _read_json(d / "COMPLETE")
    if summary is None or cfg is None or marker is None:
        raise RuntimeError(f"{d}: missing summary.json / cell_config.json / COMPLETE")
    model, tag = summary["model"], summary["dataset"]
    spec = model_spec(model)
    curves = summary.get("loss_by_layer") or {}
    if "val" not in curves or "test" not in curves:
        raise RuntimeError(f"{d}: summary.json has no saved val/test loss curves, so the tunnel "
                           "cannot be re-derived without refitting -- refusing to guess")

    old = _read_json(d / "tunnel.json") or {}
    tun = phase1.tunnel_record(spec, curves["val"], curves["test"], tol=tol, tols=tuple(tols))
    changed = (old.get("definition") != tun["definition"]
               or (old.get("headline") or {}).get("index") != tun["headline"]["index"]
               or (old.get("headline") or {}).get("tolerance") != tun["headline"]["tolerance"])

    # the two halves of the identity, recomputed with the CURRENT postprocess config
    new_cfg = {k: v for k, v in cfg.items() if k != "config_hash"}
    new_cfg.update(tunnel_definition=TUNNEL_DEFINITION_VERSION, tunnel_tol=float(tol),
                   tunnel_tols=[float(t) for t in tols])
    fit_hash = phase1.fit_config_hash(new_cfg)
    if marker.get("fit_hash") not in (None, fit_hash):
        raise RuntimeError(
            f"{d}: the FIT half of this cell's config does not match the current code "
            f"({marker.get('fit_hash')} vs {fit_hash}). That is not a postprocessing change — "
            "the probes themselves would differ. Re-run the driver instead of re-deriving.")
    post_hash = phase1.postprocess_config_hash(new_cfg)
    chash = phase1.cell_config_hash(new_cfg)

    rep = {"model": model, "dataset": tag, "path": str(d),
           "old_definition": old.get("definition"), "new_definition": tun["definition"],
           "old_tunnel": (old.get("headline") or {}).get("label"),
           "new_tunnel": tun["headline"]["label"],
           "first_crossing": (tun.get("first_crossing", {})
                              .get(f"first_crossing_{tol:g}", {}).get("label")),
           "changed": bool(changed), "config_hash": chash, "dry_run": bool(dry_run)}
    if dry_run:
        return rep

    # ---- tunnel.json: fully regenerated -------------------------------------------- #
    phase1.atomic_write_json(d / "tunnel.json", {
        "model": model, "dataset": tag, "display_name": registry.display_name(tag), **tun,
        "representation_points": [p.as_dict() for p in spec.points],
        "rederived_by": "experiments.rebuild_phase1_tunnels",
        "rederived_from": "summary.json loss_by_layer (val/test), the cell's own saved curves"})

    # ---- layer_metrics.csv: ONLY the tunnel-owned columns --------------------------- #
    rows, fields = _read_csv(d / "layer_metrics.csv")
    dep = tun["depth_axis_indices"]
    ratios = {"val_ratio_to_final": dict(zip(dep, tun.get("val_ratio_by_depth", []))),
              "test_ratio_to_final": dict(zip(dep, tun.get("test_ratio_by_depth", []))),
              "val_suffix_excursion": dict(zip(dep, tun.get("val_suffix_excursion_by_depth", []))),
              "test_suffix_excursion": dict(zip(dep,
                                                tun.get("test_suffix_excursion_by_depth", [])))}
    ent = {e["tolerance"]: e["index"] for e in tun["by_tolerance"].values()}
    fcs = {e["tolerance"]: e["index"] for e in tun.get("first_crossing", {}).values()}
    new_fields = [f for f in fields
                  if not f.startswith(TUNNEL_OWNED_PREFIXES) and f not in TUNNEL_OWNED]
    added = (list(TUNNEL_OWNED) + [f"is_tunnel_entrance_{t:g}" for t in sorted(ent)]
             + [f"is_first_crossing_{t:g}" for t in sorted(fcs)])
    for r in rows:
        i = int(r["point_index"])
        r["is_tunnel_entrance"] = str(i == tun["headline"]["index"])
        for key, m in ratios.items():
            r[key] = m.get(i, "")
        for t, idx in ent.items():
            r[f"is_tunnel_entrance_{t:g}"] = str(i == idx)
        for t, idx in fcs.items():
            r[f"is_first_crossing_{t:g}"] = str(i == idx)
    ordered = new_fields + [f for f in added if f not in new_fields]
    if "is_tunnel_entrance" not in ordered:
        ordered.insert(min(len(ordered), 12), "is_tunnel_entrance")
    phase1.write_csv(d / "layer_metrics.csv", rows, ordered)

    # ---- summary.json: the tunnel block only ---------------------------------------- #
    summary["tunnel"] = tun["headline"]
    summary["tunnel_definition"] = tun["definition"]
    summary["tunnel_definition_caveat"] = tun.get("definition_caveat")
    summary["tunnel_all_tolerances"] = {
        k: {"tolerance": v["tolerance"], "label": v["label"], "index": v["index"],
            "depth_axis_index": v["depth_axis_index"], "relative_depth": v["relative_depth"]}
        for k, v in tun["by_tolerance"].items()}
    summary["first_crossing_diagnostic"] = {
        k: {"tolerance": v["tolerance"], "label": v["label"], "index": v["index"],
            "depth_axis_index": v["depth_axis_index"], "relative_depth": v["relative_depth"]}
        for k, v in tun.get("first_crossing", {}).items()}
    summary["generalization_at_entrance"] = tun.get("generalization_at_entrance")
    summary["val_ratio_by_depth"] = tun.get("val_ratio_by_depth")
    summary["test_ratio_by_depth"] = tun.get("test_ratio_by_depth")
    summary.setdefault("rederivations", []).append(
        {"tool": "experiments.rebuild_phase1_tunnels", "definition": tun["definition"],
         "tolerance": float(tol), "tolerances": [float(t) for t in tols]})
    phase1.atomic_write_json(d / "summary.json", summary)

    # ---- cell_config.json + COMPLETE ------------------------------------------------- #
    phase1.atomic_write_json(d / "cell_config.json", {**new_cfg, "config_hash": chash})
    marker.update(config_hash=chash, fit_hash=fit_hash, postprocess_hash=post_hash,
                  tunnel_definition=tun["definition"], tunnel_layer=tun["headline"]["label"],
                  tunnel_relative_depth=tun["headline"]["relative_depth"],
                  rederived_by="experiments.rebuild_phase1_tunnels")
    tmp = (d / "COMPLETE").with_suffix(".tmp")
    tmp.write_text(json.dumps(marker, indent=2, default=str))
    shutil.move(str(tmp), str(d / "COMPLETE"))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-root", default=str(DEFAULT_ROOT))
    ap.add_argument("--roster", default=ROSTER)
    ap.add_argument("--models", nargs="+", default=None, choices=list(MODELS))
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--tunnel-tol", type=float, default=PHASE1_TUNNEL_TOL)
    ap.add_argument("--tunnel-tols", type=float, nargs="+", default=list(PHASE1_TUNNEL_TOLS))
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change and write nothing")
    ap.add_argument("--no-rebuild-tables", action="store_true")
    a = ap.parse_args(argv)

    root = Path(a.output_root)
    tags = [t for t in registry.roster(a.roster) if not a.datasets or t in a.datasets]
    models = [m for m in MODELS if not a.models or m in a.models]
    print(f"\nRE-DERIVING PHASE-1 TUNNELS   definition={TUNNEL_DEFINITION_VERSION}   "
          f"tol={a.tunnel_tol:g}   tols={list(a.tunnel_tols)}")
    print(f"  root {root}" + ("   (DRY RUN — nothing is written)" if a.dry_run else ""))

    done, skipped, failed = [], [], []
    for model in models:
        for tag in tags:
            d = cell_dirs(root, model, tag)
            if not (d / "COMPLETE").exists():
                skipped.append(f"{model}/{tag}")
                continue
            try:
                r = rebuild_cell(d, tol=a.tunnel_tol, tols=a.tunnel_tols, dry_run=a.dry_run)
                done.append(r)
                flag = "CHANGED" if r["changed"] else "same   "
                print(f"  [{flag}] {model:<9} {tag:<28} {r['old_tunnel'] or '-':>8} -> "
                      f"{r['new_tunnel']:<8} (first_crossing {r['first_crossing']})")
            except Exception as exc:
                failed.append((model, tag, f"{type(exc).__name__}: {exc}"))
                print(f"  [FAIL  ] {model}/{tag}: {type(exc).__name__}: {exc}")

    print(f"\n  {len(done)} re-derived ({sum(1 for r in done if r['changed'])} moved), "
          f"{len(skipped)} not complete, {len(failed)} failed")
    if not a.dry_run and done and not a.no_rebuild_tables:
        from experiments.make_phase1_tables import build
        res = build(root, a.roster)
        print(f"  [combined] {res['n_complete']}/{res['n_total']} cells -> {res['out']}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
