"""Rolling-origin window-yield screen for the proposed 14-dataset paper roster (+ controls).

COUNT-ONLY. No model, no feature extraction, no probe fitting, no training. This script loads
each dataset's raw series and measures what the EXISTING rolling-origin protocol would yield,
so the roster can be frozen on measured numbers instead of inferred ones. It never writes to
results/ or features_cache/.

WHY THIS EXISTS (three questions it answers)
  1. realized supply  -- exact train/val/test counts under the committed budget, plus the
     fraction of candidate windows the finite/non-constant filter rejects;
  2. MASE viability   -- the per-dataset seasonal-naive denominator failure rate (a NaN denom
     means that test window is EXCLUDED from MASE, not divided by a fabricated floor);
  3. degeneracy       -- for intermittent datasets (M5 above all), whether the Q=1 forecasting
     task survives at all, measured in the probe's OWN target space.

PROTOCOL -- imported, never re-implemented
  Validity and the target transform come from `probing.id_data` itself
  (`_rolling_valid_starts`, `_seasonal_naive_scale`), so this screen cannot drift from what
  `_build_rolling_windows` will actually do:
    * C = 512, H = 64; context = s[st : st+C], target = s[st+C : st+C+H];
    * origins are H-spaced  ->  targets are pairwise non-overlapping;
    * a window is valid iff context AND target are fully finite and ctx.std() >= SIGMA_EPS;
    * per eligible series (>= 3 valid origins): LAST -> test, 2nd-last -> val, earlier -> train;
    * the MASE denominator is the seasonal-naive scale of that series' history STRICTLY BEFORE
      its test target (`s[:last_start + C]`), matching `_build_rolling_windows` exactly.

DEGENERACY DIAGNOSTIC (the M5 decision rule)
  The probe target is `yvec = arcsinh((future - mu_ctx) / sd_ctx)` (`_make_examples`). The Q=1
  (tau=0.5) pinball loss of the best CONSTANT per-window forecast is `0.5 * mad_const` where
    mad_const(window) = mean_t | yvec_t - median_t(yvec) |.
  That is the floor any probe reaches by reading only the context's level. If it collapses to
  ~0 the tunnel question is vacuous on that dataset. We report its median across windows plus
  the fraction of windows with mad_const == 0 (a literally constant future).

KNOWN BUG THIS SCREEN PREDICTS (does not fix)
  `_build_rolling_windows` draws the 262 val/test series from the FULL eligible pool but the
  cluster-balanced round robin only reaches `target_train` (1394) distinct series. When
  n_eligible > 1394 the fail-loud check `missing = sel_set - set(tr_sid)` almost surely trips
  and the builder RAISES. Each row reports `would_raise_fail_loud` so we know which datasets
  need the deterministic series cap before anything can run.

WHERE TO RUN -- a COMPUTE NODE, never the login node.
  It holds multi-GB of raw series, sustains a full core for minutes, and loops millions of
  origins. See the header of the __main__ block for the exact salloc/sbatch invocation.

Usage:
    python -m experiments.rolling_yield_screen14                       # everything
    python -m experiments.rolling_yield_screen14 --only m5 wiki_daily_100k
    python -m experiments.rolling_yield_screen14 --max-series 200      # fast smoke ONLY
    python -m experiments.rolling_yield_screen14 --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# THE protocol primitives -- imported so this screen and the production builder cannot disagree.
from probing.id_data import _rolling_valid_starts, _seasonal_naive_scale   # noqa: E402

C, H, SIGMA_EPS = 512, 64, 1e-6
MIN_LEN = C + 3 * H                  # 704: shortest series that can supply >= 3 origins
BUDGET_TRAIN, BUDGET_VALTEST = 1394, 262     # BUDGET_BY_SET["extended_v3_rolling"]


# --------------------------------------------------------------------------- #
# roster under screen: the proposed 14 + the two designated alternates
# --------------------------------------------------------------------------- #
# kind:     "hf"   -> load_dataset(repo, config)["train"], one univariate series per row
#           "ood"  -> probing.id_data.load_ood_target_series (staged arrow shards)
# m_season: the CENTRALIZED seasonal-naive period (this table is the prototype of the registry
#           field). For every dataset we share with fev-bench it EQUALS fev-bench's published
#           `seasonality`, so the choice has an external citation rather than our assertion.
SPECS = {
    # ---- the current seven (re-screened so candidates are compared like-for-like) ----
    "m4_hourly":                  dict(kind="hf", repo="autogluon/chronos_datasets",
                                       config="m4_hourly", target="target",
                                       freq="1H", m_season=24, group="current"),
    "monash_electricity_hourly":  dict(kind="hf", repo="autogluon/chronos_datasets",
                                       config="monash_electricity_hourly", target="target",
                                       freq="1H", m_season=24, group="current"),
    "uber_tlc_hourly":            dict(kind="hf", repo="autogluon/chronos_datasets",
                                       config="uber_tlc_hourly", target="target",
                                       freq="1H", m_season=24, group="current"),
    "wind_farms_hourly":          dict(kind="hf", repo="autogluon/chronos_datasets",
                                       config="wind_farms_hourly", target="target",
                                       freq="1H", m_season=24, group="current"),
    "sg_carpark":                 dict(kind="ood", freq="1H(agg)", m_season=24, group="current"),
    "coastal_ts":                 dict(kind="ood", freq="1H", m_season=24, group="current"),
    "boom_hourly":                dict(kind="ood", freq="1H", m_season=24, group="current"),

    # ---- the seven proposed additions ----
    "LOOP_SEATTLE_5T":            dict(kind="hf", repo="autogluon/fev_datasets",
                                       config="LOOP_SEATTLE_5T", target="target",
                                       freq="5min", m_season=288, group="new"),
    # NOTE: no "target" column -- fev's own task declares `target: Patv` (active power).
    "kdd_cup_2022_10T":           dict(kind="hf", repo="autogluon/fev_datasets",
                                       config="kdd_cup_2022_10T", target="Patv",
                                       freq="10min", m_season=144, group="new"),
    "SZ_TAXI_15T":                dict(kind="hf", repo="autogluon/fev_datasets",
                                       config="SZ_TAXI_15T", target="target",
                                       freq="15min", m_season=96, group="new"),
    "monash_london_smart_meters": dict(kind="hf", repo="autogluon/chronos_datasets",
                                       config="monash_london_smart_meters", target="target",
                                       freq="30min", m_season=48, group="new"),
    "m5":                         dict(kind="hf", repo="autogluon/chronos_datasets",
                                       config="m5", target="target",
                                       freq="1D", m_season=7, group="new"),
    "wiki_daily_100k":            dict(kind="hf", repo="autogluon/chronos_datasets",
                                       config="wiki_daily_100k", target="target",
                                       freq="1D", m_season=7, group="new"),
    "monash_traffic":             dict(kind="hf", repo="autogluon/chronos_datasets",
                                       config="monash_traffic", target="target",
                                       freq="1H", m_season=24, group="new"),

    # ---- designated alternates (screened, not in the roster unless a trigger fires) ----
    "rossmann_1D":                dict(kind="hf", repo="autogluon/fev_datasets",
                                       config="rossmann_1D", target="Sales",
                                       freq="1D", m_season=7, group="alternate"),
    "electricity_15min":          dict(kind="hf", repo="autogluon/chronos_datasets",
                                       config="electricity_15min", target="consumption_kW",
                                       freq="15min", m_season=96, group="alternate"),
}


# --------------------------------------------------------------------------- #
# lazy series iteration -- never materializes the whole dataset
# --------------------------------------------------------------------------- #
def _iter_hf(repo: str, config: str, target: str, max_series: int | None):
    """Yield one 1-D float64 series per row, memory-mapped, without building a 100k-element list.

    `load_seen_series` materializes EVERY series as float64 (wiki_daily_100k ~= 2.2 GB,
    london_smart_meters ~= 1.3 GB). For a count-only screen that is pure waste, so we walk the
    Arrow column row by row instead. The VALUES are identical -- only the materialization
    strategy differs.
    """
    from datasets import load_dataset
    ds = load_dataset(repo, config, split="train")
    if target not in ds.column_names:
        raise KeyError(f"{repo}:{config} has no column {target!r}; columns = {ds.column_names}")
    n = ds.num_rows if max_series is None else min(ds.num_rows, max_series)
    try:                                              # fast path: Arrow, zero-copy-ish
        col = ds.data.column(target)
        for i in range(n):
            yield np.asarray(col[i].values.to_numpy(zero_copy_only=False), dtype=np.float64)
    except (AttributeError, TypeError):               # portable fallback
        fmt = ds.with_format("numpy")
        for i in range(n):
            yield np.asarray(fmt[i][target], dtype=np.float64)


def _iter_ood(tag: str, max_series: int | None):
    """Yield (series, cluster_id) for a staged OOD target. Cluster != series for coastal_ts
    (2 variates per station), which is why the cluster id is carried through: the bootstrap
    unit -- and therefore the effective test support -- is the CLUSTER, not the row."""
    from probing.id_data import load_ood_target_series
    loaded = load_ood_target_series(tag)
    pairs = list(zip(loaded["series"], loaded["cluster_ids"]))
    if max_series is not None:
        pairs = pairs[:max_series]
    for s, cid in pairs:
        yield np.asarray(s, dtype=np.float64), int(cid)


# --------------------------------------------------------------------------- #
# per-dataset screen
# --------------------------------------------------------------------------- #
def screen(tag: str, spec: dict, max_series: int | None = None) -> dict:
    t0 = time.time()
    m = spec["m_season"]

    n_series = 0
    excl = {"too_short": 0, "insufficient_valid": 0}
    lengths: list[int] = []
    cand_origins = 0            # H-spaced origins offered by series of length >= MIN_LEN
    valid_origins = 0           # ... surviving the finite + non-constant filter
    train_supply = 0
    eligible_clusters: set[int] = set()
    n_eligible_series = 0
    denom_fail = 0              # test windows whose seasonal-naive denominator is non-finite
    mads: list[float] = []      # mad_const per (sampled) window, in the probe's target space
    zero_mad = 0
    n_mad = 0

    if spec["kind"] == "ood":
        source = _iter_ood(tag, max_series)
    else:
        source = ((s, i) for i, s in
                  enumerate(_iter_hf(spec["repo"], spec["config"], spec["target"], max_series)))

    for s, cid in source:
        n_series += 1
        L = len(s)
        lengths.append(L)
        if L < MIN_LEN:
            excl["too_short"] += 1
            continue
        # every H-spaced origin this series OFFERS, before the validity filter
        cand_origins += max(0, (L - (C + H)) // H + 1)
        starts = _rolling_valid_starts(s, C, H, SIGMA_EPS)
        if len(starts) < 3:
            excl["insufficient_valid"] += 1
            valid_origins += len(starts)
            continue

        n_eligible_series += 1
        eligible_clusters.add(cid)
        valid_origins += len(starts)
        train_supply += len(starts) - 2                     # last -> test, 2nd-last -> val

        # ---- MASE denominator, exactly as _build_rolling_windows computes it ----
        te_st = starts[-1]
        if not np.isfinite(_seasonal_naive_scale(s[:te_st + C], m)):
            denom_fail += 1

        # ---- degeneracy, in the probe's own target space, on the val+test windows ----
        for st in (starts[-2], starts[-1]):
            ctx, fut = s[st:st + C], s[st + C:st + C + H]
            sd = max(float(ctx.std()), SIGMA_EPS)
            yv = np.arcsinh((fut - float(ctx.mean())) / sd)
            mad = float(np.abs(yv - np.median(yv)).mean())
            mads.append(mad)
            zero_mad += (mad == 0.0)
            n_mad += 1

    n_clusters = len(eligible_clusters)
    realized_valtest = min(BUDGET_VALTEST, n_clusters)
    realized_train = min(BUDGET_TRAIN, train_supply)
    lens = np.asarray(lengths) if lengths else np.zeros(1)
    mad_arr = np.asarray(mads) if mads else np.zeros(1)

    return {
        "tag": tag,
        "group": spec["group"],
        "freq": spec["freq"],
        "m_season": m,
        "n_series": n_series,
        "series_len_min_med_max": [int(lens.min()), int(np.median(lens)), int(lens.max())],
        "excluded_series": excl,
        "n_eligible_series": n_eligible_series,
        "n_eligible_clusters": n_clusters,
        # ---- window supply + rejection ----
        "candidate_origins": cand_origins,
        "valid_origins": valid_origins,
        "frac_windows_rejected": (1.0 - valid_origins / cand_origins) if cand_origins else None,
        "train_supply": train_supply,
        "realized": {"train": realized_train, "val": realized_valtest, "test": realized_valtest},
        "train_budget_satisfied": train_supply >= BUDGET_TRAIN,
        "valtest_budget_satisfied": n_clusters >= BUDGET_VALTEST,
        # ---- MASE ----
        "denominator_fail_rate": (denom_fail / n_eligible_series) if n_eligible_series else None,
        "n_denominator_fail": denom_fail,
        # ---- degeneracy (probe target space; Q=1 constant-forecast floor = 0.5 * mad) ----
        "mad_const_median": float(np.median(mad_arr)),
        "mad_const_p10": float(np.percentile(mad_arr, 10)),
        "frac_windows_zero_mad": (zero_mad / n_mad) if n_mad else None,
        # ---- the known builder bug ----
        "would_raise_fail_loud": n_eligible_series > BUDGET_TRAIN,
        "seconds": round(time.time() - t0, 1),
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", nargs="*", default=None, help="screen just these tags")
    p.add_argument("--skip", nargs="*", default=[], help="skip these tags")
    p.add_argument("--max-series", type=int, default=None,
                   help="cap series per dataset. SMOKE TEST ONLY -- it biases every count.")
    p.add_argument("--json", type=str, default=None, help="write the full records here")
    args = p.parse_args()

    tags = args.only or [t for t in SPECS if t not in args.skip]
    if args.max_series:
        print(f"!! --max-series {args.max_series}: counts are a SMOKE TEST, not the screen.\n")

    rows = []
    for tag in tags:
        try:
            r = screen(tag, SPECS[tag], args.max_series)
        except Exception as exc:                      # a missing OOD shard must not kill the run
            print(f"  [SKIP] {tag}: {type(exc).__name__}: {exc}")
            continue
        rows.append(r)
        print(f"  [done] {tag:28s} {r['seconds']:7.1f}s  "
              f"eligible={r['n_eligible_series']:6d}  clusters={r['n_eligible_clusters']:5d}")

    hdr = (f"\n{'dataset':28s}{'freq':>8}{'m':>5}{'series':>8}{'elig':>7}{'clust':>7}"
           f"{'train':>8}{'val':>5}{'test':>5}{'rej%':>7}{'denomfail%':>11}{'madMed':>8}{'0mad%':>7}")
    print(hdr + "\n" + "-" * len(hdr))
    for r in rows:
        rej = "  n/a" if r["frac_windows_rejected"] is None else f"{100 * r['frac_windows_rejected']:6.2f}"
        dfr = "  n/a" if r["denominator_fail_rate"] is None else f"{100 * r['denominator_fail_rate']:10.3f}"
        zmd = "  n/a" if r["frac_windows_zero_mad"] is None else f"{100 * r['frac_windows_zero_mad']:6.2f}"
        print(f"{r['tag']:28s}{r['freq']:>8}{r['m_season']:5d}{r['n_series']:8d}"
              f"{r['n_eligible_series']:7d}{r['n_eligible_clusters']:7d}"
              f"{r['realized']['train']:8d}{r['realized']['val']:5d}{r['realized']['test']:5d}"
              f"{rej}{dfr}{r['mad_const_median']:8.3f}{zmd}")

    print("\nFLAGS")
    for r in rows:
        flags = []
        if not r["valtest_budget_satisfied"]:
            flags.append(f"val/test capped at {r['n_eligible_clusters']} (< {BUDGET_VALTEST})")
        if not r["train_budget_satisfied"]:
            flags.append(f"train supply {r['train_supply']} < {BUDGET_TRAIN}")
        if r["would_raise_fail_loud"]:
            flags.append(f"_build_rolling_windows WOULD RAISE (eligible {r['n_eligible_series']} "
                         f"> target_train {BUDGET_TRAIN}) -- needs the deterministic series cap")
        if (r["denominator_fail_rate"] or 0) > 0.05:
            flags.append(f"seasonal-MASE denominator fails on "
                         f"{100 * r['denominator_fail_rate']:.1f}% of test series")
        if (r["frac_windows_rejected"] or 0) > 0.30:
            flags.append(f"{100 * r['frac_windows_rejected']:.1f}% of candidate windows rejected")
        if r["mad_const_median"] < 0.05:
            flags.append(f"DEGENERACY RISK: median constant-forecast MAD {r['mad_const_median']:.4f}")
        if flags:
            print(f"  {r['tag']}:")
            for f in flags:
                print(f"      - {f}")
    if not any(r for r in rows):
        print("  (no rows)")

    if args.json:
        Path(args.json).write_text(json.dumps(
            {"C": C, "H": H, "sigma_eps": SIGMA_EPS, "budget": [BUDGET_TRAIN, BUDGET_VALTEST],
             "max_series": args.max_series, "rows": rows}, indent=2))
        print(f"\n  wrote {args.json}")


if __name__ == "__main__":
    main()
