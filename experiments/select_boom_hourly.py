"""Build the committed BOOM hourly-variate selection manifest (metadata/quality only).

BOOM ships 378 native-hourly metric queries (dirs ``ds-<N>-H``); each is ONE multivariate query
(target shape (V, T)). This picks, per query, the FIRST variate that passes a fixed data-quality
gate — chosen on metadata BEFORE any layerwise result is seen — and writes the pinned list to
``data/boom_hourly_selection.json`` (query id + variate index + per-variate stats). The OOD loader
(``id_data.load_ood_target_series('boom_hourly')``) reads exactly this manifest, so the evaluation
set is fully reproducible.

Quality gate (per variate, at C=512 / H=64, stride 64):
  * native-hourly (the ``-H`` dir suffix; BOOM stores a regular grid with gaps as in-place NaN);
  * missing fraction <= --missing-cap (default 0.20 — matches the wind_farms tolerance);
  * yields >= 1 fully-finite, non-constant 576-step window (EXACT ``_make_examples`` contract, so a
    selected query is guaranteed >=1 usable window; incomplete windows are dropped, never filled).
The FIRST variate (ascending index) that passes is chosen -> one variate per query. Queries with no
passing variate are recorded as dropped (with the reason) and excluded.

Run (login node, needs `module load arrow`; reads $OOD_TARGET_ROOT/boom_hourly):
    python -m experiments.select_boom_hourly                 # writes data/boom_hourly_selection.json
    python -m experiments.select_boom_hourly --missing-cap 0.20
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from probing.config import SEED
from probing.id_data import (OOD_TARGET_ROOT, BOOM_MANIFEST, _read_arrow,
                             _make_examples, _all_starts, _boom_variates)

C, H, STRIDE, SIGMA_EPS = 512, 64, 64, 1e-6


def _variate_stats(series):
    """(missing_fraction, n_valid_windows) for one univariate variate under the EXACT window
    contract used at build time (_make_examples: drops any non-finite 576-span or near-constant
    context)."""
    a = np.asarray(series, dtype=np.float64)
    miss = float(1.0 - np.isfinite(a).mean()) if a.size else 1.0
    starts = _all_starts(len(a), C, H, STRIDE)
    ctxs, _ys, _yv, _sk = _make_examples(a, starts, C, H, SIGMA_EPS)
    return miss, len(ctxs)


def _query_dirs():
    root = OOD_TARGET_ROOT / "boom_hourly"
    if not root.exists():
        raise SystemExit(f"{root} not found — download Datadog/BOOM ds-*-H first "
                         "(snapshot_download allow_patterns=['ds-*-H/*'])")
    dirs = [d for d in root.iterdir() if d.is_dir() and re.match(r"ds-\d+-H$", d.name)]
    # deterministic order: ascending query number
    return sorted(dirs, key=lambda d: int(re.match(r"ds-(\d+)-H$", d.name).group(1)))


def select(missing_cap):
    selected, dropped = [], []
    miss_all, nvalid_all = [], []
    for d in _query_dirs():
        t = _read_arrow(d / "data-00000-of-00001.arrow").to_pydict()
        item_id = str(t["item_id"][0])
        variates = _boom_variates(t["target"][0])                      # list of 1-D arrays
        V = len(variates)
        chosen = None
        for v in range(V):                                             # first passing variate
            miss, nvalid = _variate_stats(variates[v])
            if v == 0:
                miss_all.append(miss); nvalid_all.append(nvalid)
            if miss <= missing_cap and nvalid >= 1:
                chosen = {"query_dir": d.name, "item_id": item_id, "variate_index": int(v),
                          "n_variates": int(V), "length": int(len(variates[v])),
                          "missing_fraction": round(miss, 5), "n_valid_windows": int(nvalid)}
                break
        if chosen is not None:
            selected.append(chosen)
        else:
            dropped.append({"query_dir": d.name, "item_id": item_id, "n_variates": int(V),
                            "reason": f"no variate with missing<= {missing_cap} and >=1 valid window"})
    return selected, dropped, np.array(miss_all), np.array(nvalid_all)


def main():
    ap = argparse.ArgumentParser(description="Build the committed BOOM hourly-variate manifest.")
    ap.add_argument("--missing-cap", type=float, default=0.20,
                    help="max per-variate missing fraction (default 0.20 = wind_farms tolerance).")
    ap.add_argument("--out", default=str(BOOM_MANIFEST))
    args = ap.parse_args()

    print(f"[select_boom_hourly] root={OOD_TARGET_ROOT / 'boom_hourly'}  missing_cap={args.missing_cap}")
    selected, dropped, miss0, nvalid0 = select(args.missing_cap)
    n_total = len(selected) + len(dropped)
    total_valid = int(sum(s["n_valid_windows"] for s in selected))

    manifest = {
        "dataset": "Datadog/BOOM", "subset": "native_hourly", "frequency": "H",
        "cluster_unit": "metric_query", "seed": SEED,
        "window": {"C": C, "H": H, "stride": STRIDE, "sigma_eps": SIGMA_EPS},
        "criteria": {
            "native_hourly_only": True, "missing_fraction_max": args.missing_cap,
            "min_valid_windows": 1, "variate_pick": "first_passing_ascending_index",
            "window_contract": "id_data._make_examples (drop any non-finite 576-span / near-constant "
                               "context; NO fill)",
            "note": "one quality-passing variate per query -> maximize independent clusters; "
                    "windows later drawn query-balanced (round-robin) in build_ood_windows"},
        "n_queries_total": n_total, "n_queries_selected": len(selected),
        "n_queries_dropped": len(dropped),
        "total_valid_windows_available": total_valid,
        "selected": selected, "dropped": dropped,
    }
    json.dump(manifest, open(args.out, "w"), indent=2)

    # report
    print(f"  queries: total={n_total}  selected={len(selected)}  dropped={len(dropped)}")
    if miss0.size:
        print(f"  variate-0 missing fraction: p50={np.percentile(miss0,50):.4f} "
              f"p90={np.percentile(miss0,90):.4f} max={miss0.max():.4f}")
    if nvalid0.size:
        print(f"  valid-windows/query (variate 0): p10={int(np.percentile(nvalid0,10))} "
              f"p50={int(np.percentile(nvalid0,50))} p90={int(np.percentile(nvalid0,90))}")
    print(f"  total valid windows available across selected queries: {total_valid} "
          f"(target eval set = 650, query-balanced)")
    if dropped:
        print(f"  example dropped: {dropped[0]}")
    print(f"  [saved] committed manifest -> {args.out}")


if __name__ == "__main__":
    main()
