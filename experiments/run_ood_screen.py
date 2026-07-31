"""Dataset-inclusion screen for the OOD-transfer matrix (data + task quality, no probing).

Purpose: decide whether a candidate dataset belongs in the matrix on DATA quality and TASK
quality — never on whether the layerwise result supports the hypothesis. This is the guard that
demoted KDD (persistence-dominated: last-value ≤ seasonal-naive) and rejected WeatherBench
2m_temperature. It runs BEFORE any layerwise probe.

For every dataset in the active set (or --datasets) it reports, at C=512 / H=64:
  * data diagnostics  — n_series, length distribution, missing-value rate, constant-series rate,
    per-series scale distribution;
  * window supply     — the split mode the existing build_windows selects, windows before/after
    the matched-budget uniform subsample, series contributing, and the windows-per-series spread
    (so a few long series can't silently dominate);
  * naive baselines   — last-value and seasonal-naive (m=24, and m=168 weekly when C≥168) MASE of
    the MEDIAN forecast, using the EXACT in-context seasonal-naive denominator compute_mase uses;
  * (opt) native      — native Chronos-2 median MASE with --native (GPU / warm cache), the check
    the persistence proxy cannot substitute for.

`last-value < seasonal-naive` is reported as a PERSISTENCE-DOMINATED WARNING, not proof the task
is trivial; the native gap is the confirming signal.

CPU / login-node by default (diagnostics + naive baselines). Add --native for the Chronos-2 pass.

Run:
    python -m experiments.run_ood_screen --dataset-set extended_v2                 # CPU screen
    python -m experiments.run_ood_screen --dataset-set extended_v2 --datasets m4_hourly wind_farms_hourly
    python -m experiments.run_ood_screen --dataset-set extended_v2 --native        # + native (GPU)
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from probing import config, id_data
from probing.config import SEED
from probing.id_data import load_seen_series, build_windows
from experiments.run_id_forecasting import _ctx_stats, _mase_denominator, M_SEASON

M_WEEK = 168   # weekly seasonal period for hourly data (optional secondary naive baseline)


# --------------------------------------------------------------------------- #
# data diagnostics (raw series) + window supply (exact build_windows)
# --------------------------------------------------------------------------- #

def data_diagnostics(tag, C=512, H=64):
    series = load_seen_series(tag)
    L = np.array([len(s) for s in series])
    span = C + H
    n_pts = int(L.sum())
    nonfin = 0
    n_const = 0
    stds = []
    per_series_miss = np.empty(len(series))
    for k, s in enumerate(series):
        a = np.asarray(s, dtype=np.float64)
        fin = np.isfinite(a)
        nonfin += int((~fin).sum())
        per_series_miss[k] = 1.0 - fin.mean() if a.size else 1.0
        av = a[fin]
        if av.size == 0 or av.std() < 1e-6:
            n_const += 1
        else:
            stds.append(float(av.std()))
    stds = np.array(stds) if stds else np.array([np.nan])
    qs = [0, 10, 25, 50, 75, 90, 100]
    return {
        "tag": tag, "n_series": len(series),
        "length_quantiles": {f"p{q}": int(np.percentile(L, q)) for q in qs},
        "series_ge_2span": int((L >= 2 * span).sum()), "series_ge_span": int((L >= span).sum()),
        "missing_fraction_overall": round(nonfin / n_pts, 5) if n_pts else float("nan"),
        "missing_fraction_per_series_p50": round(float(np.percentile(per_series_miss, 50)), 4),
        "missing_fraction_per_series_max": round(float(per_series_miss.max()), 4),
        "constant_series": n_const, "constant_series_rate": round(n_const / len(series), 4),
        "scale_std_p10_p50_p90": [round(float(np.percentile(stds, q)), 4) for q in (10, 50, 90)],
    }


def window_supply(tag):
    """Exact current build_windows + uniform _subsample at the active set's matched budget."""
    w = build_windows(tag)                       # budget resolves from the active dataset set
    m = w["meta"]
    st = np.asarray(w["series_train"]); se = np.asarray(w["series_test"])
    def spread(arr):
        if not arr.size:
            return {"distinct_series": 0, "per_series_min_med_max": [0, 0, 0], "top_series_share": None}
        c = np.bincount(arr); c = c[c > 0]
        return {"distinct_series": int(len(c)),
                "per_series_min_med_max": [int(c.min()), int(np.median(c)), int(c.max())],
                "top_series_share": round(float(c.max() / c.sum()), 4)}
    supply = {
        "split_mode": m["split_mode"], "target_budget": [m["target_train"], m["target_test"]],
        "before_subsample": [m["n_train_windows_before_subsample"], m["n_test_windows_before_subsample"]],
        "after_subsample": [m["n_train"], m["n_test"]], "n_skipped_windows": m["n_skipped_windows"],
        "train_spread": spread(st), "test_spread": spread(se),
    }
    return w, supply


# --------------------------------------------------------------------------- #
# baselines on the TEST windows (median forecast; in-context m-seasonal MASE)
# --------------------------------------------------------------------------- #

def _mase(y_raw, f_raw, d):
    return float((np.abs(y_raw - f_raw) / d).mean())


def naive_baselines(w, want_native=False):
    X = np.asarray(w["X_test"], np.float64)
    n, C = X.shape
    Y = w["Y_test_traj"]; H = Y.shape[1]
    mu, s = _ctx_stats(X, w["meta"]["sigma_eps"])
    y_raw = mu[:, None] + s[:, None] * np.sinh(Y.astype(np.float64))
    d_raw = _mase_denominator(X)                              # in-context seasonal-naive scale (m=24)
    n_zero = int((d_raw < 1e-8).sum()); n_nan = int((~np.isfinite(d_raw)).sum())
    d = np.maximum(d_raw, 1e-8)[:, None]
    out = {"n_test_windows": n, "denominator_clamped": n_zero, "denominator_non_finite": n_nan,
           "denominator_p5_p50": [round(float(np.percentile(d_raw, 5)), 4),
                                  round(float(np.percentile(d_raw, 50)), 4)]}
    out["last_value_mase"] = round(_mase(y_raw, np.repeat(X[:, -1:], H, axis=1), d), 4)
    out["seasonal_naive_m24_mase"] = round(
        _mase(y_raw, X[:, C - M_SEASON + (np.arange(H) % M_SEASON)], d), 4)
    if C >= M_WEEK:
        out["seasonal_naive_m168_mase"] = round(
            _mase(y_raw, X[:, C - M_WEEK + (np.arange(H) % M_WEEK)], d), 4)
    if want_native:
        from experiments.run_id_forecasting import native_median_forecast
        native = native_median_forecast(w["meta"]["tag"], w["X_test"], H).astype(np.float64)
        out["native_chronos2_mase"] = round(_mase(y_raw, native, d), 4)
    # persistence read (WARNING signal, not a triviality verdict)
    r = out["last_value_mase"] / out["seasonal_naive_m24_mase"]
    out["last_over_seasonal_ratio"] = round(r, 3)
    out["persistence_warning"] = bool(r < 1.0)   # last-value beats same-hour-yesterday
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def screen(tags, want_native):
    rows = []
    for tag in tags:
        print(f"\n{'=' * 70}\n[screen] {tag}\n{'=' * 70}")
        diag = data_diagnostics(tag)
        w, supply = window_supply(tag)
        base = naive_baselines(w, want_native=want_native)
        print(f"  n_series={diag['n_series']}  length p50={diag['length_quantiles']['p50']}  "
              f"missing={diag['missing_fraction_overall']:.3f}  const_rate={diag['constant_series_rate']}")
        print(f"  split={supply['split_mode']}  supply(before)={supply['before_subsample']} "
              f"-> budget {supply['after_subsample']}  test_series={supply['test_spread']['distinct_series']} "
              f"top_series_share={supply['test_spread']['top_series_share']}")
        line = (f"  last-value MASE={base['last_value_mase']}  seasonal-m24={base['seasonal_naive_m24_mase']}"
                f"  ratio={base['last_over_seasonal_ratio']} "
                f"({'PERSISTENCE WARNING' if base['persistence_warning'] else 'seasonal stronger'})")
        if "native_chronos2_mase" in base:
            line += f"  native={base['native_chronos2_mase']}"
        print(line)
        rows.append({"dataset": tag, **diag, **supply, **base})

    out_dir = config.ID_OUT_DIR / "ood_transfer" / "screen"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag_native = "_native" if want_native else ""
    jpath = out_dir / f"dataset_screen__{config.DATASET_SET}{tag_native}.json"
    json.dump({"dataset_set": config.DATASET_SET, "seed": SEED, "with_native": want_native,
               "note": "last-value < seasonal-naive => persistence-dominated WARNING (not a "
                       "triviality proof); confirm with native. Inclusion on data+task quality only.",
               "rows": rows}, open(jpath, "w"), indent=2, default=float)
    # flat CSV of the headline columns
    flat_fields = ["dataset", "n_series", "missing_fraction_overall", "constant_series_rate",
                   "split_mode", "last_value_mase", "seasonal_naive_m24_mase",
                   "last_over_seasonal_ratio", "persistence_warning"]
    if want_native:
        flat_fields.insert(-1, "native_chronos2_mase")
    with open(out_dir / f"dataset_screen__{config.DATASET_SET}{tag_native}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=flat_fields, extrasaction="ignore")
        wr.writeheader(); wr.writerows(rows)
    print(f"\n  [saved] {jpath}")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Dataset-inclusion screen (data + task quality; no probing).")
    ap.add_argument("--dataset-set", default=None)
    ap.add_argument("--datasets", nargs="+", default=None,
                    help="datasets to screen (default: all in the active set).")
    ap.add_argument("--native", action="store_true",
                    help="also compute native Chronos-2 MASE (GPU / warm cache).")
    return ap.parse_args(argv)


def main():
    args = _parse_args()
    if args.dataset_set:
        config.set_dataset_set(args.dataset_set)
    tags = args.datasets or list(id_data.ID_DATASETS)
    known = set(id_data.ID_DATASETS)
    bad = [t for t in tags if t not in known]
    if bad:
        raise SystemExit(f"unknown --datasets {bad}; known in {config.DATASET_SET}: {sorted(known)}")
    print(f"[config] dataset_set={config.DATASET_SET}  datasets={tags}  native={args.native}")
    screen(tags, args.native)


if __name__ == "__main__":
    main()
