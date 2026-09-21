"""Window-only smoke for the frozen 14-dataset roster: build every dataset's windows, report the
EXACT realized train/val/test counts, and stop.

NO MODEL. NO GPU. NO PROBE FIT. NO FEATURE EXTRACTION. This runs the real window builders
through the real dispatch (``probing.windows.build_for``) on the real data, and nothing else.
It is the last gate before any GPU time is spent: if a dataset's windowing is wrong, or the
deterministic series cap misfires, or a seasonal period is missing, this is where it surfaces —
for the price of reading some parquet, not the price of an A100.

WHAT IT REPORTS PER DATASET
  * realized train / val / test window counts and the number of distinct bootstrap units;
  * the builder, the bootstrap cluster unit, the seasonal period and the primary/control role;
  * the deterministic series-cap audit (fired or not, and from how many eligible series);
  * the window digest, so two runs — or two machines — can be compared without re-deriving;
  * whether a window reference exists, and (with --create-references) writes the missing ones.

WHERE TO RUN -- a COMPUTE NODE, not the login node.
  It reads multi-GB of raw series (wiki_daily_100k alone is ~2.2 GB of float64) and sustains a
  core for minutes while it walks millions of origins. Only the ``--dry-run`` roster print is
  cheap enough for a login node.

      salloc --account=def-irina --cpus-per-task=2 --mem=32G --time=2:00:00
      module load gcc python/3.11 arrow/24.0.0 && source .venv/bin/activate
      export HF_HOME=$SCRATCH/chronos2/hf_cache HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
      export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets
      python -m experiments.window_smoke14 --json $SCRATCH/window_smoke14.json

Usage:
    python -m experiments.window_smoke14                      # all 14
    python -m experiments.window_smoke14 --only m5 wiki_daily_100k
    python -m experiments.window_smoke14 --dry-run            # roster only, no data read
    python -m experiments.window_smoke14 --create-references  # write missing window references
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import registry                                       # noqa: E402
from probing import window_reference as wref                       # noqa: E402
from probing.windows import build_for                              # noqa: E402

CREATED_BY = "python -m experiments.window_smoke14"


def smoke_one(tag: str, suite: str, create_references: bool, force: bool) -> dict:
    t0 = time.time()
    w = build_for(tag, suite)
    meta = w["meta"]
    sid = np.asarray(w["series_test"], np.int64)
    vid = np.asarray(w.get("series_val", np.zeros(0)), np.int64)
    cap = meta.get("series_cap") or {}

    ref = wref.resolve_reference(tag, lambda _t: None)
    ref_state = "present" if ref else "MISSING"
    if ref is None and create_references:
        wref.write_reference(tag, w, created_by=CREATED_BY, force=force)
        ref_state = "created"
    elif ref is not None:
        fails = wref.compare_to_reference(tag, w, ref)
        ref_state = "match" if not fails else f"MISMATCH ({len(fails)})"

    return {
        "dataset": tag,
        "display_name": registry.display_name(tag),
        "role": registry.role(tag),
        "builder": registry.builder(tag),
        "cluster_unit": registry.cluster_unit(tag),
        "seasonal_m": registry.seasonal_m(tag),
        "freq": registry.spec(tag).freq,
        "split_mode": meta.get("split_mode"),
        "n_train": int(len(w["X_train"])),
        "n_val": int(len(w["X_val"])) if "X_val" in w else None,
        "n_test": int(len(sid)),
        "n_val_units": int(np.unique(vid).size) if vid.size else 0,
        "n_test_units": int(np.unique(sid).size),
        "n_series_total": meta.get("n_series", meta.get("n_series_total")),
        "n_eligible_series": meta.get("n_eligible_series"),
        "cap_applied": bool(cap.get("applied", False)),
        "cap_from": cap.get("n_eligible_before"),
        "cap_to": cap.get("n_eligible_after"),
        "n_denominator_invalid": meta.get("n_denominator_invalid"),
        "window_digest": wref.window_digest(w),
        "reference": ref_state,
        "seconds": round(time.time() - t0, 1),
    }


def print_roster(tags):
    hdr = (f"{'dataset':<28}{'name':<22}{'role':<9}{'freq':>8}{'m':>5}  "
           f"{'builder':<22}{'unit':<14}{'cap':>6}")
    print("\nROSTER (probing/registry.py)\n" + hdr + "\n" + "-" * len(hdr))
    for t in tags:
        s = registry.spec(t)
        print(f"{t:<28}{s.display_name:<22}{s.role:<9}{s.freq:>8}{s.seasonal_m:>5}  "
              f"{s.builder:<22}{s.cluster_unit:<14}{str(s.max_series or '-'):>6}")
    n_control = len(registry.control_datasets())
    print(f"\n  {len(tags)} datasets: {len(tags) - n_control} primary + {n_control} control "
          f"({', '.join(registry.control_datasets()) or 'none'})")
    periods = sorted({registry.seasonal_m(t) for t in tags})
    print(f"  seasonal periods in use: {periods}   (a single global 24 would be wrong for "
          f"{sum(1 for t in tags if registry.seasonal_m(t) != 24)} of them)")


def print_results(rows):
    hdr = (f"{'dataset':<28}{'train':>7}{'val':>6}{'test':>6}{'units':>7}{'m':>5}"
           f"{'elig':>8}{'cap':>14}  {'reference':<12}{'s':>6}")
    print("\nREALIZED WINDOW COUNTS\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        if "error" in r:
            print(f"{r['dataset']:<28}  ERROR: {r['error'][:80]}")
            continue
        cap = (f"{r['cap_from']}->{r['cap_to']}" if r["cap_applied"] else "-")
        print(f"{r['dataset']:<28}{r['n_train']:>7}{str(r['n_val']):>6}{r['n_test']:>6}"
              f"{r['n_test_units']:>7}{r['seasonal_m']:>5}"
              f"{str(r['n_eligible_series']):>8}{cap:>14}  {r['reference']:<12}{r['seconds']:>6}")

    ok = [r for r in rows if "error" not in r]
    bad = [r for r in rows if "error" in r]
    print(f"\n  {len(ok)}/{len(rows)} datasets built windows successfully")
    if bad:
        print("  FAILED: " + ", ".join(r["dataset"] for r in bad))

    print("\nFLAGS")
    flagged = False
    for r in ok:
        f = []
        if r["n_val"] is not None and r["n_val"] != r["n_test"]:
            f.append(f"val ({r['n_val']}) != test ({r['n_test']})")
        if r["n_test_units"] < 100:
            f.append(f"only {r['n_test_units']} bootstrap units -> wide CIs")
        if r["cap_applied"]:
            f.append(f"deterministic series cap fired: {r['cap_from']} -> {r['cap_to']}")
        if r["reference"].startswith("MISMATCH"):
            f.append("WINDOWS DIFFER FROM THE COMMITTED REFERENCE")
        if r["reference"] == "MISSING":
            f.append("no window reference (run with --create-references, deliberately)")
        if f:
            flagged = True
            print(f"  {r['dataset']}:")
            for x in f:
                print(f"      - {x}")
    if not flagged:
        print("  (none)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="paper14", help="paper14 (default) | paper7")
    p.add_argument("--only", nargs="*", default=None, help="restrict to these tags")
    p.add_argument("--dry-run", action="store_true",
                   help="print the roster and exit; reads no data (login-node safe)")
    p.add_argument("--create-references", action="store_true",
                   help="write a window reference for every dataset that has none. This is the "
                        "deliberate act that makes later cross-model parity checkable.")
    p.add_argument("--force-reference", action="store_true",
                   help="allow --create-references to OVERWRITE an existing reference")
    p.add_argument("--json", type=str, default=None, help="write the full records here")
    args = p.parse_args()

    tags = args.only or registry.roster(args.suite)
    unknown = [t for t in tags if not registry.known(t)]
    if unknown:
        raise SystemExit(f"unknown dataset tag(s): {unknown}")

    print_roster(tags)
    if args.dry_run:
        print("\n[dry-run] roster only; no data was read.")
        return

    rows = []
    for tag in tags:
        try:
            r = smoke_one(tag, args.suite, args.create_references, args.force_reference)
            print(f"  [done] {tag:<28} train {r['n_train']:>5} / val {str(r['n_val']):>4} / "
                  f"test {r['n_test']:>4}   ({r['seconds']}s)")
        except Exception as exc:
            r = {"dataset": tag, "error": f"{type(exc).__name__}: {exc}",
                 "traceback": traceback.format_exc()}
            print(f"  [FAIL] {tag}: {r['error']}")
        rows.append(r)

    print_results(rows)

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"suite": args.suite, "roster": tags, "rows": rows,
             "mase_definition": registry.MASE_DEFINITION}, indent=2))
        print(f"\n  wrote {args.json}")

    if any("error" in r for r in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
