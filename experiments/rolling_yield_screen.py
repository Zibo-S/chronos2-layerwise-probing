"""Count-only rolling-origin within-series window yield (extended_v3_rolling design probe).

No model, no feature extraction, no arcsinh, no training — this script only loads each series
and COUNTS how many train/validation/test windows the strict non-overlapping-target
rolling-origin protocol yields, so we can choose the largest CLEAN common budget BEFORE
modifying build_windows. It never writes to any results/ or features_cache/ path.

Protocol enforced here (identical to the intended extended_v3_rolling split):
  * C = 512, H = 64.
  * Forecast origin t: context = series[t-C : t], target = series[t : t+H].
  * Origins are H-spaced  ->  t in {C, C+H, C+2H, ...},  t <= L-H  ->  targets never overlap.
  * A window is valid only if its context AND target are fully finite and the context is
    non-constant (std >= SIGMA_EPS) — the SAME validity rule as id_data._make_examples, so a
    dataset's missing values / constant series are dropped leakage-free.
  * Per eligible series (>= 3 valid origins): last valid origin -> TEST, second-to-last -> VAL,
    all earlier valid origins -> TRAIN. This makes every train target strictly earlier than the
    val target, which is strictly earlier than the test target — no target timestamp is shared
    across splits within a series.

Reports, per dataset and as a common-budget summary:
  total series, eligible series, exclusion reasons, train/val/test window supply, train
  windows-per-series (min/med/max), the cross-split target-overlap count (MUST be 0), and the
  largest clean common budget = the per-split minimum across all four datasets.

Light enough for a Narval login node (a few seconds; a couple hundred MB of series in RAM).

Run (repo root, modules first then venv):
    module load gcc python/3.11 arrow/24.0.0 && source .venv/bin/activate
    export HF_HOME=$SCRATCH/chronos2/hf_cache HF_HUB_OFFLINE=1
    python -m experiments.rolling_yield_screen
"""

from __future__ import annotations

import numpy as np

from probing import config
from probing.id_data import load_seen_series

C, H, SIGMA_EPS = 512, 64, 1e-6
MIN_LEN = C + 3 * H          # 704: shortest series that can supply >=3 non-overlapping origins
TAGS = ["monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly"]


def valid_origins(s: np.ndarray) -> list[int]:
    """H-spaced target origins t whose context [t-C, t) and target [t, t+H) are finite and whose
    context is non-constant — the same validity rule as id_data._make_examples. Stepping by H
    means the kept origins are a subset of the H-grid, so their targets are pairwise disjoint."""
    L = len(s)
    out: list[int] = []
    for t in range(C, L - H + 1, H):
        ctx = s[t - C:t]
        fut = s[t:t + H]
        if np.all(np.isfinite(ctx)) and np.all(np.isfinite(fut)) and ctx.std() >= SIGMA_EPS:
            out.append(t)
    return out


def targets_overlap(train: list[int], val: int, test: int) -> bool:
    """True if ANY target interval [t, t+H) intersects across the train/val/test splits.

    An explicit interval-intersection audit (not merely an ordering check), so the
    'zero overlap between train, validation and test targets' guarantee is verified directly."""
    groups = [[(t, t + H) for t in train], [(val, val + H)], [(test, test + H)]]
    for gi in range(3):
        for gj in range(gi + 1, 3):
            for a0, a1 in groups[gi]:
                for b0, b1 in groups[gj]:
                    if a0 < b1 and b0 < a1:          # half-open intervals intersect
                        return True
    return False


def screen_tag(tag: str) -> dict:
    series = load_seen_series(tag)
    n_train = n_val = n_test = 0
    excl = {"too_short": 0, "insufficient_valid": 0}
    overlaps = 0
    tr_per_series: list[int] = []
    tr_origins: list[int] = []
    va_origins: list[int] = []
    te_origins: list[int] = []
    for s in series:
        s = np.asarray(s, dtype=np.float64)
        if len(s) < MIN_LEN:
            excl["too_short"] += 1
            continue
        o = valid_origins(s)
        if len(o) < 3:                               # need >=1 train, 1 val, 1 test
            excl["insufficient_valid"] += 1
            continue
        train, val, test = o[:-2], o[-2], o[-1]
        if targets_overlap(train, val, test):
            overlaps += 1
        n_train += len(train)
        n_val += 1
        n_test += 1
        tr_per_series.append(len(train))
        tr_origins += train
        va_origins.append(val)
        te_origins.append(test)
    tp = np.array(tr_per_series) if tr_per_series else np.array([0])
    return {
        "tag": tag,
        "n_series": len(series),
        "eligible": len(tr_per_series),
        "excluded": excl,
        "supply": {"train": n_train, "val": n_val, "test": n_test},
        "train_windows_per_series_min_med_max": [int(tp.min()), int(np.median(tp)), int(tp.max())],
        "cross_split_target_overlaps": overlaps,
        "target_origin_step_range": {
            "train": [min(tr_origins), max(tr_origins)] if tr_origins else None,
            "val": [min(va_origins), max(va_origins)] if va_origins else None,
            "test": [min(te_origins), max(te_origins)] if te_origins else None,
        },
    }


def main() -> None:
    config.set_dataset_set("extended_v2")            # all 4 tags resolve here (m4 + wind live in extended_v2)
    rows = [screen_tag(t) for t in TAGS]
    print(f"\n{'dataset':28s}{'series':>7}{'elig':>6}{'train':>8}{'val':>5}{'test':>5}"
          f"{'  tr/series[min,med,max]':>26}{'  overlaps':>10}")
    for r in rows:
        print(f"{r['tag']:28s}{r['n_series']:7d}{r['eligible']:6d}{r['supply']['train']:8d}"
              f"{r['supply']['val']:5d}{r['supply']['test']:5d}"
              f"{str(r['train_windows_per_series_min_med_max']):>26}"
              f"{r['cross_split_target_overlaps']:10d}")
        print(f"    excluded {r['excluded']}   (val/test = 1 window per eligible series)")
        print(f"    target-origin step ranges {r['target_origin_step_range']}")
    common_train = min(r["supply"]["train"] for r in rows)
    common_val = min(r["supply"]["val"] for r in rows)
    common_test = min(r["supply"]["test"] for r in rows)
    total_overlaps = sum(r["cross_split_target_overlaps"] for r in rows)
    print(f"\n  LARGEST CLEAN COMMON BUDGET  ->  train={common_train}  val={common_val}  test={common_test}")
    print("  (H-spaced non-overlapping targets; test/val = 1 window/series; overlaps MUST be 0)")
    print(f"  total cross-split target overlaps across all datasets: {total_overlaps}  "
          f"({'OK' if total_overlaps == 0 else 'PROTOCOL VIOLATION — investigate'})")


if __name__ == "__main__":
    main()
