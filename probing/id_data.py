"""In-distribution (ID) forecasting windows from Chronos-2-SEEN datasets.

Phase 0's genuine ID condition. The UEA classification probing is, from Chronos-2's
perspective, a foreign/transfer task; here we build a forecasting-shaped probing task on
data Chronos-2 was actually PRETRAINED on (Table 6 of arXiv:2510.15821 — see
``data/chronos2_seen_manifest.md``).

Each probing example is a length-``C`` univariate context window. Its label is the
normalized ``H``-step future mean

    y = ( mean(target[t+1 : t+H]) - mu_context ) / sigma_context

i.e. future dynamics RELATIVE TO the context's own mean/std (matching Chronos-2's
instance-scaling philosophy). Without this normalization the hidden states trivially
encode series scale and every layer saturates.

Train/test split is ALONG TIME with no leakage:
  - within_series (primary): for a series long enough (L >= 2*(C+H)), early windows -> train,
    late windows -> test; every train context+horizon span ends at/before the split point and
    every test span starts at/after it, so no train span overlaps any test span.
  - cross_series (documented fallback): for datasets whose series are too short to fit both a
    train span and a test span (e.g. M4-Hourly at C=512), series are partitioned into disjoint
    train/test series. There is still no within-series leakage (a series is wholly in one
    split). This does NOT change C, H, or the label; it only changes how the split is drawn,
    and it is recorded per dataset in the run summary.

Nothing here touches the UEA classification pipeline, its cache, or its results.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from probing.config import SEED

# Chronos-2-seen ID dataset sets (HF config + target column per tag). The active set is
# selected by probing.config.DATASET_SET (env ID_DATASET_SET, or the --dataset-set CLI
# override via config.set_dataset_set), the same value that namespaces the results
# directories — so the dataset list a run uses and the directory its outputs land in can
# never disagree. ID_DATASETS is resolved DYNAMICALLY (module __getattr__) so an override
# applied after this module was imported is still honored.
ID_DATASET_SPECS = {
    # the original Phase 0 run, tags/targets exactly as published (m4_hourly is the
    # cross_series fallback case; solar_1h carries the documented label pathology).
    "phase0_trio": {
        "m4_hourly":                 {"hf_config": "m4_hourly",                 "target": "target"},
        "monash_electricity_hourly": {"hf_config": "monash_electricity_hourly", "target": "target"},
        "solar_1h":                  {"hf_config": "solar_1h",                  "target": "power_mw"},
    },
    # four long hourly series -> all support the within_series temporal split. All four
    # verified against autogluon/chronos_datasets: target column == "target", min sampled
    # series length >> 2*(C+H).
    "extended_v1": {
        "monash_electricity_hourly": {"hf_config": "monash_electricity_hourly", "target": "target"},  # Energy
        "monash_kdd_cup_2018":       {"hf_config": "monash_kdd_cup_2018",       "target": "target"},  # Nature / air quality
        "monash_pedestrian_counts":  {"hf_config": "monash_pedestrian_counts",  "target": "target"},  # Transport / foot traffic
        "uber_tlc_hourly":           {"hf_config": "uber_tlc_hourly",           "target": "target"},  # Transport / ride demand
    },
    # 4×4 OOD-transfer set (KDD dropped as persistence-dominated). All hourly (m=24).
    # m4_hourly is short -> cross_series; the other three -> within_series.
    "extended_v2": {
        "monash_electricity_hourly": {"hf_config": "monash_electricity_hourly", "target": "target"},  # Energy consumption
        "uber_tlc_hourly":           {"hf_config": "uber_tlc_hourly",           "target": "target"},  # Transport / ride demand
        "m4_hourly":                 {"hf_config": "m4_hourly",                 "target": "target"},  # Mixed hourly (cross_series)
        "wind_farms_hourly":         {"hf_config": "wind_farms_hourly",         "target": "target"},  # Renewable generation
    },
    # 4×4 ROLLING-ORIGIN WITHIN-SERIES set. Same tags/targets as extended_v2, but ALL FOUR
    # datasets (including m4_hourly, which the length-derived auto split forces onto cross_series)
    # use ONE uniform temporal split — dedicated train/val/test forecast origins per series,
    # H-spaced non-overlapping targets. See _build_rolling_windows + ROLLING_SETS. extended_v2 is
    # left untouched as a cross-series sensitivity comparison.
    "extended_v3_rolling": {
        "monash_electricity_hourly": {"hf_config": "monash_electricity_hourly", "target": "target"},  # Energy consumption
        "uber_tlc_hourly":           {"hf_config": "uber_tlc_hourly",           "target": "target"},  # Transport / ride demand
        "m4_hourly":                 {"hf_config": "m4_hourly",                 "target": "target"},  # Mixed hourly (rolling)
        "wind_farms_hourly":         {"hf_config": "wind_farms_hourly",         "target": "target"},  # Renewable generation
    },
}
_HF_REPO = "autogluon/chronos_datasets"

# Matched TOTAL train/test window budget per dataset set (applied by the existing uniform
# _subsample — NOT a per-series cap). extended_v2 = 1500/650 (M4 is the floor); the others keep
# the historical 3000/1500 so their committed numbers are unchanged.
BUDGET_BY_SET = {
    "phase0_trio": (3000, 1500),
    "extended_v1": (3000, 1500),
    "extended_v2": (1500, 650),
    # rolling-origin sets carry a (train, VAL, test) 3-tuple: val is a dedicated temporal split,
    # not a carve of train. build_windows dispatches these to _build_rolling_windows.
    "extended_v3_rolling": (1394, 262, 262),
}

# Dataset sets that use the uniform rolling-origin WITHIN-SERIES split (explicit temporal
# train/val/test, H-spaced non-overlapping targets) instead of the length-derived
# within_series/cross_series auto split. Their BUDGET_BY_SET entry is a (train, val, test)
# 3-tuple and build_windows routes them through _build_rolling_windows.
ROLLING_SETS = {"extended_v3_rolling"}


def _active_specs() -> dict:
    """The active set's specs, re-read from probing.config on every call so a CLI override
    (config.set_dataset_set) is honored even after this module was imported."""
    from probing import config
    return ID_DATASET_SPECS[config.DATASET_SET]


def __getattr__(name):  # PEP 562: `id_data.ID_DATASETS` / from-imports stay dynamic
    if name == "ID_DATASETS":
        return _active_specs()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --------------------------------------------------------------------------- #
# raw series loading
# --------------------------------------------------------------------------- #
def load_seen_series(tag: str) -> list[np.ndarray]:
    """Download an ID dataset and return its target series as a list of 1-D float64 arrays."""
    from datasets import load_dataset

    spec = _active_specs()[tag]
    ds = load_dataset(_HF_REPO, spec["hf_config"], split="train")
    col = spec["target"]
    return [np.asarray(r[col], dtype=np.float64) for r in ds]


# --------------------------------------------------------------------------- #
# window -> (context, normalized-future-mean label)
# --------------------------------------------------------------------------- #
def _make_examples(series, starts, C, H, sigma_eps):
    """Build (context, scalar future-mean, arcsinh future TRAJECTORY) per start.

    Skips constant/non-finite windows. The trajectory ``yvec`` is the H-step future
    context-standardized THEN arcsinh'd -- Chronos-2's own target space (config
    ``use_arcsinh=True``; see chronos2 InstanceNorm). The legacy scalar ``y`` is the mean of
    the pre-arcsinh (linear) normalized future, kept UNCHANGED for the ridge/binned probes.
    Consistency: ``sinh(yvec).mean() == y`` since ``sinh(arcsinh(x)) == x``.
    """
    ctxs, ys, yvecs = [], [], []
    n_skipped = 0
    for st in starts:
        ctx = series[st:st + C]
        fut = series[st + C:st + C + H]
        if not (np.all(np.isfinite(ctx)) and np.all(np.isfinite(fut))):
            n_skipped += 1
            continue
        mu = float(ctx.mean())
        sd = float(ctx.std())
        if sd < sigma_eps:                      # near-constant context -> label ill-defined
            n_skipped += 1
            continue
        s = max(sd, sigma_eps)
        lin_vec = (fut - mu) / s                        # (H,) linear context-standardized future
        yvec = np.arcsinh(lin_vec).astype(np.float32)   # arcsinh -> Chronos-2 target space (use_arcsinh=True)
        ctxs.append(ctx.astype(np.float32))
        ys.append(np.float32(lin_vec.mean()))           # legacy scalar UNCHANGED (linear space)
        yvecs.append(yvec)
    return ctxs, ys, yvecs, n_skipped


def _within_series_starts(L, C, H, stride, test_frac):
    """Train/test start positions for one series under the within-series temporal split."""
    span = C + H
    if L < 2 * span:
        return [], []                           # cannot fit a train span AND a test span
    p = int(round(L * (1 - test_frac)))
    tr = list(range(0, p - span + 1, stride))   # window fully inside [0, p)
    te = list(range(p, L - span + 1, stride))   # window fully inside [p, L)
    return tr, te


def _all_starts(L, C, H, stride):
    span = C + H
    return list(range(0, L - span + 1, stride))


def _rolling_valid_starts(s, C, H, sigma_eps):
    """Chronological, H-SPACED context-start positions st (context [st, st+C), target
    [st+C, st+C+H)) whose window is finite AND non-constant — the SAME validity rule as
    _make_examples. Stepping by H (not an arbitrary stride) guarantees the kept origins' targets
    are pairwise NON-OVERLAPPING (the rolling-origin invariant). Non-finite / constant spans are
    simply skipped, so missing data drops windows leakage-free. Returned in time order."""
    span = C + H
    out = []
    for st in range(0, len(s) - span + 1, H):
        ctx = s[st:st + C]
        fut = s[st + C:st + C + H]
        if np.all(np.isfinite(ctx)) and np.all(np.isfinite(fut)) and ctx.std() >= sigma_eps:
            out.append(st)
    return out


def _seasonal_naive_scale(x, m):
    """In-sample seasonal-naive MASE denominator d = mean_t |x_t - x_{t-m}| over `x`.

    `x` must be data available BEFORE the forecast origin (a series' TRAINING portion) — never
    the test span, which would leak future values into the scale. Returns NaN (NOT a floor) when
    `x` is too short for a lagged pair or the scale is non-finite / non-positive, so downstream
    MASE can EXCLUDE those windows instead of dividing by a fabricated tiny number."""
    x = np.asarray(x, dtype=np.float64)
    if x.size <= m:
        return float("nan")
    scale = float(np.abs(x[m:] - x[:-m]).mean())
    if not np.isfinite(scale) or scale <= 0:
        return float("nan")
    return scale


def _subsample(ctxs, ys, yvecs, sids, target, rng):
    """Deterministically subsample (ctx, y, yvec, series_id) tuples down to `target`.

    All FOUR lists are sliced by the SAME indices so the trajectory AND the series id stay
    row-aligned with the context (and hence with the cached hidden-state features). The single
    rng.choice call is unchanged from the 3-list version, so X_train/X_test rows are byte-
    identical and the existing feature cache still aligns."""
    n = len(ys)
    if n <= target:
        return ctxs, ys, yvecs, sids
    idx = np.sort(rng.choice(n, size=target, replace=False))
    return ([ctxs[i] for i in idx], [ys[i] for i in idx],
            [yvecs[i] for i in idx], [sids[i] for i in idx])


def _build_rolling_windows(tag, C, H, stride, sigma_eps, m_season, seed,
                           target_train=None, target_val=None, target_test=None):
    """Uniform rolling-origin WITHIN-SERIES windows for the extended_v3_rolling set.

    ALL datasets (including m4_hourly) use the SAME protocol — no cross_series fallback:
      * H-spaced non-overlapping forecast targets; a window is valid iff its context and target
        are finite and the context is non-constant (`_rolling_valid_starts`).
      * Per eligible series (>= 3 valid origins): LAST valid origin -> test, 2nd-last -> val, all
        earlier -> train. Within a series every train target precedes the val target, which
        precedes the test target, so no target timestamp is shared across splits.
      * The SAME deterministic `target_val` (== `target_test`) series carry both the val and the
        test window (one each). Train windows are drawn from EVERY eligible series and
        cluster-balanced (round-robin) down to `target_train`, so a few long series can't dominate
        and every selected val/test series keeps >= 1 train window (enforced, fail-loud).

    Returns the build_windows dict shape PLUS X_val/y_val/Y_val_traj/series_val. The reported MASE
    still uses the in-context seasonal-naive scale (run_id_forecasting._mase_denominator); the
    canonical `test_denominator` here is the seasonal-naive scale of each test series' history
    STRICTLY BEFORE its test target (leakage-free), so `mase_canonical` is True for all four."""
    from probing import config
    b_tr, b_va, b_te = BUDGET_BY_SET[config.DATASET_SET]
    target_train = b_tr if target_train is None else target_train
    target_val = b_va if target_val is None else target_val
    target_test = b_te if target_test is None else target_test
    if target_val != target_test:
        raise ValueError(f"rolling split uses the SAME series for val and test; "
                         f"target_val ({target_val}) must equal target_test ({target_test})")

    series = load_seen_series(tag)
    rng = np.random.default_rng(seed)
    min_len = C + 3 * H                                  # 704: shortest series giving 3 origins

    # 1) eligible series -> their valid H-spaced context starts (>= 3 each)
    eligible: dict[int, list[int]] = {}
    excl = {"too_short": 0, "insufficient_valid": 0}
    for i, s in enumerate(series):
        s = np.asarray(s, dtype=np.float64)
        if len(s) < min_len:
            excl["too_short"] += 1
            continue
        starts = _rolling_valid_starts(s, C, H, sigma_eps)
        if len(starts) < 3:
            excl["insufficient_valid"] += 1
            continue
        eligible[i] = starts
    if len(eligible) < target_val:
        raise RuntimeError(f"{tag}: only {len(eligible)} eligible series (need {target_val} for "
                           "val/test) — lower the val/test budget or check the data")

    # 2) the SAME deterministic series carry val AND test (one window each)
    elig_idx = np.array(sorted(eligible))
    sel = np.sort(rng.permutation(elig_idx)[:target_val])
    sel_set = {int(i) for i in sel}

    # 3) candidate windows: train from EVERY eligible series (valid[:-2]); val/test from the
    #    selected series (valid[-2] / valid[-1]). origins recorded as the target-start index.
    tr_ctx, tr_y, tr_yv, tr_sid, tr_org = [], [], [], [], []
    va_ctx, va_y, va_yv, va_sid, va_org = [], [], [], [], []
    te_ctx, te_y, te_yv, te_sid, te_org = [], [], [], [], []
    den_by_series: dict[int, float] = {}
    for i in sorted(eligible):
        s = np.asarray(series[i], dtype=np.float64)
        starts = eligible[i]
        c, y, v, _ = _make_examples(s, starts[:-2], C, H, sigma_eps)
        tr_ctx += c; tr_y += y; tr_yv += v; tr_sid += [i] * len(c)
        tr_org += [st + C for st in starts[:-2]]
        if i in sel_set:
            va_st, te_st = starts[-2], starts[-1]
            c, y, v, _ = _make_examples(s, [va_st], C, H, sigma_eps)
            va_ctx += c; va_y += y; va_yv += v; va_sid += [i]; va_org += [va_st + C]
            c, y, v, _ = _make_examples(s, [te_st], C, H, sigma_eps)
            te_ctx += c; te_y += y; te_yv += v; te_sid += [i]; te_org += [te_st + C]
            den_by_series[i] = _seasonal_naive_scale(s[:te_st + C], m_season)  # history before test

    n_tr_full = len(tr_y)
    # 4) cluster-balanced (round-robin) subsample of TRAIN to target_train — broad series coverage.
    order = _cluster_balanced_order(np.asarray(tr_sid, np.int64), target_train, rng)
    tr_ctx = [tr_ctx[j] for j in order]; tr_y = [tr_y[j] for j in order]
    tr_yv = [tr_yv[j] for j in order]; tr_sid = [tr_sid[j] for j in order]
    tr_org = [tr_org[j] for j in order]

    # 5) fail-loud: every selected val/test series MUST keep >= 1 retained train window.
    missing = sel_set - {int(i) for i in tr_sid}
    if missing:
        raise RuntimeError(f"{tag}: {len(missing)} selected val/test series kept no train window "
                           f"after subsample (e.g. {sorted(missing)[:5]}) — target_train "
                           f"({target_train}) must be >= n_eligible_series ({len(eligible)})")

    def _stack(ctxs, dim):
        return np.stack(ctxs).astype(np.float32) if ctxs else np.zeros((0, dim), np.float32)

    X_train, X_val, X_test = _stack(tr_ctx, C), _stack(va_ctx, C), _stack(te_ctx, C)
    Y_train_traj, Y_val_traj, Y_test_traj = _stack(tr_yv, H), _stack(va_yv, H), _stack(te_yv, H)
    series_train = np.asarray(tr_sid, np.int64)
    series_val = np.asarray(va_sid, np.int64)
    series_test = np.asarray(te_sid, np.int64)
    test_denominator = np.array([den_by_series[int(i)] for i in te_sid], np.float64)

    meta = {
        "tag": tag, "hf_config": _active_specs()[tag]["hf_config"],
        "split_mode": "rolling_origin_within_series",
        "n_series": len(series), "n_eligible_series": len(eligible), "excluded_series": excl,
        "C": C, "H": H, "stride": H, "test_frac": None,
        "target_train": target_train, "target_val": target_val, "target_test": target_test,
        "sigma_eps": sigma_eps, "seed": seed, "m_season": m_season,
        "mase_denominator": "per_series_history_before_test_seasonal_naive",
        "mase_canonical": True,
        "selected_series": [int(i) for i in sel],                 # the val/test series (req: metadata)
        "origins": {"train": [int(o) for o in tr_org],            # per-window target-start index
                    "val": [int(o) for o in va_org],
                    "test": [int(o) for o in te_org]},
        "n_train_windows_before_subsample": n_tr_full,
        "n_test_windows_before_subsample": len(te_y),
        "n_train": int(len(tr_y)), "n_val": int(len(va_y)), "n_test": int(len(te_y)),
        "n_val_series": int(len(np.unique(series_val))) if series_val.size else 0,
        "n_test_series": int(len(np.unique(series_test))) if series_test.size else 0,
        "n_denominator_invalid": int((~np.isfinite(test_denominator)).sum()),
        "n_skipped_windows": 0,
    }
    return {"X_train": X_train, "y_train": np.asarray(tr_y, np.float32),
            "X_val": X_val, "y_val": np.asarray(va_y, np.float32),
            "X_test": X_test, "y_test": np.asarray(te_y, np.float32),
            "Y_train_traj": Y_train_traj, "Y_val_traj": Y_val_traj, "Y_test_traj": Y_test_traj,
            "series_train": series_train, "series_val": series_val, "series_test": series_test,
            "test_denominator": test_denominator, "meta": meta}


def build_windows(
    tag: str,
    C: int = 512,
    H: int = 64,
    stride: int = 64,
    test_frac: float = 0.25,
    target_train: int | None = None,
    target_test: int | None = None,
    sigma_eps: float = 1e-6,
    m_season: int = 24,
    seed: int = SEED,
):
    """Build ID probing windows for one seen dataset.

    Returns a dict with:
        X_train, X_test           : float32 arrays (n, C)   (univariate context windows)
        y_train, y_test           : float32 arrays (n,)     (normalized future-MEAN labels; legacy)
        Y_train_traj, Y_test_traj : float32 arrays (n, H)   (normalized future TRAJECTORY; quantile probe)
        meta                      : split_mode, counts, skip counts, params
    """
    from probing import config
    # rolling-origin sets use a uniform explicit temporal train/val/test split for ALL datasets;
    # dispatch before the length-derived within_series/cross_series logic below (legacy path).
    if config.DATASET_SET in ROLLING_SETS:
        return _build_rolling_windows(tag, C, H, stride, sigma_eps, m_season, seed,
                                      target_train=target_train, target_test=target_test)
    # Resolve the matched budget from the ACTIVE dataset set unless explicitly overridden.
    if target_train is None or target_test is None:
        from probing import config
        _bt, _bte = BUDGET_BY_SET.get(config.DATASET_SET, (3000, 1500))
        target_train = _bt if target_train is None else target_train
        target_test = _bte if target_test is None else target_test

    series = load_seen_series(tag)
    rng = np.random.default_rng(seed)
    span = C + H

    n_series_ok = sum(1 for s in series if len(s) >= 2 * span)
    split_mode = "within_series" if n_series_ok >= 1 else "cross_series"

    tr_ctx, tr_y, tr_yv, tr_sid = [], [], [], []
    te_ctx, te_y, te_yv, te_sid = [], [], [], []
    # series_idx -> seasonal-naive MASE denominator computed from that series' TRAIN data only.
    # NaN sentinel = "no canonical in-sample scale" (cross_series test series have no train
    # portion; we do NOT compute it from the test span, which would leak future values).
    den_by_series: dict[int, float] = {}
    skipped = 0

    if split_mode == "within_series":
        for i, s in enumerate(series):
            tr_s, te_s = _within_series_starts(len(s), C, H, stride, test_frac)
            if te_s:                                   # only series yielding test windows need a denom
                p = int(round(len(s) * (1 - test_frac)))   # forecast origin == train/test cut
                den_by_series[i] = _seasonal_naive_scale(s[:p], m_season)  # TRAIN portion only
            c, y, v, k = _make_examples(s, tr_s, C, H, sigma_eps)
            tr_ctx += c; tr_y += y; tr_yv += v; tr_sid += [i] * len(c); skipped += k
            c, y, v, k = _make_examples(s, te_s, C, H, sigma_eps)
            te_ctx += c; te_y += y; te_yv += v; te_sid += [i] * len(c); skipped += k
    else:
        # cross_series fallback: disjoint train/test series (leakage-free), all windows each.
        order = rng.permutation(len(series))
        n_tr_series = int(round(0.7 * len(series)))
        tr_series = set(order[:n_tr_series].tolist())
        for i, s in enumerate(series):
            starts = _all_starts(len(s), C, H, stride)
            c, y, v, k = _make_examples(s, starts, C, H, sigma_eps)
            if i in tr_series:
                tr_ctx += c; tr_y += y; tr_yv += v; tr_sid += [i] * len(c)
            else:
                # a wholly-held-out test series has no pre-origin history of its own -> canonical
                # MASE is undefined here; mark NaN rather than leak the test span into the scale.
                den_by_series[i] = float("nan")
                te_ctx += c; te_y += y; te_yv += v; te_sid += [i] * len(c)
            skipped += k

    n_tr_full, n_te_full = len(tr_y), len(te_y)
    tr_ctx, tr_y, tr_yv, tr_sid = _subsample(tr_ctx, tr_y, tr_yv, tr_sid, target_train, rng)
    te_ctx, te_y, te_yv, te_sid = _subsample(te_ctx, te_y, te_yv, te_sid, target_test, rng)

    X_train = np.stack(tr_ctx).astype(np.float32) if tr_ctx else np.zeros((0, C), np.float32)
    X_test = np.stack(te_ctx).astype(np.float32) if te_ctx else np.zeros((0, C), np.float32)
    y_train = np.asarray(tr_y, dtype=np.float32)
    y_test = np.asarray(te_y, dtype=np.float32)
    Y_train_traj = np.stack(tr_yv).astype(np.float32) if tr_yv else np.zeros((0, H), np.float32)
    Y_test_traj = np.stack(te_yv).astype(np.float32) if te_yv else np.zeros((0, H), np.float32)
    series_train = np.asarray(tr_sid, dtype=np.int64)
    series_test = np.asarray(te_sid, dtype=np.int64)
    # per-test-window MASE denominator = its series' train-data seasonal-naive scale (built AFTER
    # subsample so it stays row-aligned). NaN for cross_series (canonical MASE undefined there).
    test_denominator = np.array([den_by_series[int(i)] for i in te_sid], dtype=np.float64)
    n_denominator_invalid = int((~np.isfinite(test_denominator)).sum())

    meta = {
        "tag": tag,
        "hf_config": _active_specs()[tag]["hf_config"],
        "split_mode": split_mode,
        "n_series": len(series),
        "n_series_supporting_within": n_series_ok,
        "C": C, "H": H, "stride": stride, "test_frac": test_frac,
        "target_train": target_train, "target_test": target_test,
        "sigma_eps": sigma_eps, "seed": seed,
        "m_season": m_season,
        "mase_denominator": "per_series_train_seasonal_naive",
        "mase_canonical": split_mode == "within_series",   # False under cross_series (denom is NaN)
        "n_test_series": int(len(np.unique(series_test))) if series_test.size else 0,
        "n_denominator_invalid": n_denominator_invalid,
        "n_train_windows_before_subsample": n_tr_full,
        "n_test_windows_before_subsample": n_te_full,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_skipped_windows": int(skipped),
    }
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test,
            "Y_train_traj": Y_train_traj, "Y_test_traj": Y_test_traj,
            "series_train": series_train, "series_test": series_test,
            "test_denominator": test_denominator, "meta": meta}


# --------------------------------------------------------------------------- #
# Documented pretraining-OOD targets (EVALUATION-ONLY — never used to TRAIN a probe).
# These are NOT in autogluon/chronos_datasets and NOT in the Chronos-2 seen manifest
# (data/chronos2_seen_manifest.md). Each has a bespoke loader returning univariate series +
# a PARENT-CLUSTER id (carpark / station / metric-query) for the series-level cluster
# bootstrap. Raw arrow shards are pre-downloaded on the login node to OOD_TARGET_ROOT
# (compute nodes are offline); the BOOM subset is pinned by a COMMITTED manifest so the exact
# variates are reproducible and were chosen on metadata/quality BEFORE any layerwise look.
# --------------------------------------------------------------------------- #

_REPO_ROOT = Path(__file__).resolve().parent.parent
# Where the pre-downloaded OOD-target arrow shards live (large; kept off the repo). Default is
# $SCRATCH/chronos2/ood_targets (Narval), overridable via OOD_TARGET_ROOT for tests / laptops.
OOD_TARGET_ROOT = Path(os.environ.get(
    "OOD_TARGET_ROOT",
    Path(os.environ.get("SCRATCH", _REPO_ROOT)) / "chronos2" / "ood_targets"))
# committed pin of the BOOM hourly variate subset (one quality-passing variate per query)
BOOM_MANIFEST = _REPO_ROOT / "data" / "boom_hourly_selection.json"

OOD_TARGET_TAGS = ("sg_carpark", "coastal_ts", "boom_hourly")
OOD_CLUSTER_UNIT = {"sg_carpark": "carpark", "coastal_ts": "station",
                    "boom_hourly": "metric_query"}
COASTAL_VARIATES = ("TEMP", "PSAL")     # LOCKED: use temperature + salinity (drop PRES_REL)


def _read_arrow(path):
    """Read a HuggingFace-datasets .arrow shard -> pyarrow Table (IPC file, else stream).
    pyarrow is imported lazily (needs `module load arrow`) so importing id_data stays light for
    the no-dependency tests (test_quantile_sets)."""
    import pyarrow as pa
    path = str(path)
    try:
        with pa.memory_map(path, "r") as src:
            return pa.ipc.open_file(src).read_all()
    except pa.lib.ArrowInvalid:
        with pa.memory_map(path, "r") as src:
            return pa.ipc.open_stream(src).read_all()


MIN_SG_SAMPLES_PER_HOUR = 3    # SG Carpark: require >=3 of the 4 fifteen-min samples present (tolerate
                               # <=1 missing). "All 4 required" zeroed the target: SG has a systematic
                               # single missing 15-min sample at ONE clock hour EVERY day, so that hour
                               # was NaN daily -> longest clean run 23 h << C+H=576 -> zero windows.
                               # Mean of the PRESENT samples only; still NO forward-fill / cross-hour
                               # interpolation (see notes/PLAN.md + data/ood_targets_manifest.md).


def _aggregate_15min_to_hourly(series, start_minute):
    """SG Carpark 15min -> hourly: mean of the AVAILABLE 15-min samples per clock hour, requiring at
    least MIN_SG_SAMPLES_PER_HOUR (=3) of the 4 to be finite (else NaN — NO forward-fill, NO cross-hour
    interpolation). The 15-min grid is regular with gaps encoded in place as NaN, so align to the first
    :00 sample and reshape into (n_hours, 4)."""
    if start_minute not in (0, 15, 30, 45):
        raise ValueError(f"SG Carpark start minute {start_minute} is off the 15-min grid")
    off = ((60 - start_minute) // 15) % 4          # steps from `start` to the first :00 sample
    usable = np.asarray(series, dtype=np.float64)[off:]
    n_full = usable.size // 4
    block = usable[:n_full * 4].reshape(n_full, 4)
    finite = np.isfinite(block)
    k = finite.sum(axis=1)                         # number of present 15-min samples in the hour
    keep = k >= MIN_SG_SAMPLES_PER_HOUR            # tolerate <=1 missing; require >=3 present
    hourly = np.full(n_full, np.nan, dtype=np.float64)
    summed = np.where(finite, block, 0.0).sum(axis=1)
    hourly[keep] = summed[keep] / k[keep]          # mean of the AVAILABLE samples (not a fixed /4)
    return hourly


def _load_sg_carpark():
    t = _read_arrow(OOD_TARGET_ROOT / "sg_carpark" / "data-00000-of-00001.arrow").to_pydict()
    series, cluster_ids, names = [], [], []
    for c, (item, st, tgt) in enumerate(zip(t["item_id"], t["start"], t["target"])):
        series.append(_aggregate_15min_to_hourly(tgt, st.minute))
        cluster_ids.append(c)
        names.append(str(item))
    return {"series": series, "cluster_ids": cluster_ids, "cluster_unit": "carpark",
            "cluster_names": names,
            "notes": "SG Carpark; target = available-lot count; 15min->hourly mean of available "
                     "samples (>=3 of 4 required; <=1 missing tolerated, no fill)"}


def _load_coastal_ts():
    t = _read_arrow(OOD_TARGET_ROOT / "coastal_ts" / "data-00000-of-00001.arrow").to_pydict()
    series, cluster_ids, names = [], [], []
    for c, (item, tgt, vnames) in enumerate(zip(t["item_id"], t["target"], t["variate_names"])):
        arr = np.asarray(tgt, dtype=np.float64)                     # (V, T)
        vmap = {str(v): j for j, v in enumerate(vnames)}
        for vn in COASTAL_VARIATES:                                 # fixed order -> deterministic
            if vn not in vmap:
                raise ValueError(f"Coastal station {item} lacks variate {vn}; has {list(vmap)}")
            series.append(arr[vmap[vn]])
            cluster_ids.append(c)                                   # cluster = parent STATION
            names.append(f"{item}:{vn}")
    return {"series": series, "cluster_ids": cluster_ids, "cluster_unit": "station",
            "cluster_names": names,
            "notes": f"Coastal T-S; variates={list(COASTAL_VARIATES)}; cluster by station"}


def _boom_variates(target_row):
    """Univariate variate arrays for one BOOM query row (used by the selection screen, which needs
    every variate). BOOM stores a MULTIVARIATE query as (V, T) but a UNIVARIATE query as a flat
    (T,) list — normalize both to a list of 1-D float64 arrays so variate indexing is uniform."""
    row = target_row
    if len(row) == 0:
        return []
    first = row[0]
    if isinstance(first, (list, tuple, np.ndarray)):               # multivariate (V, T)
        return [np.asarray(r, dtype=np.float64) for r in row]
    return [np.asarray(row, dtype=np.float64)]                     # univariate stored flat (T,)


def _boom_read_variate(path, vidx):
    """Memory-lean read of ONE variate from a BOOM query arrow. Avoids materializing all V
    variates (the ~100-variate queries make a full to_pydict() heavy) — reads the columnar table
    and converts only variate ``vidx`` to numpy. Handles both the multivariate (list<list>) and
    univariate-flat (list<double>) storage layouts BOOM uses."""
    import pyarrow as pa
    vals = _read_arrow(path).column("target")[0].values           # single row -> inner values
    inner = vals[int(vidx)].values if pa.types.is_list(vals.type) else vals   # MV slice / UV flat
    return np.asarray(inner.to_numpy(zero_copy_only=False), dtype=np.float64)


def _load_boom_hourly():
    import json
    if not BOOM_MANIFEST.exists():
        raise FileNotFoundError(
            f"BOOM selection manifest {BOOM_MANIFEST} not found — run the BOOM screen/selection "
            "step first; it writes the committed manifest (one quality-passing hourly variate per "
            "query, chosen on metadata only).")
    sel = json.load(open(BOOM_MANIFEST))["selected"]                # [{query_dir, variate_index}, ..]
    series, cluster_ids, names = [], [], []
    for c, e in enumerate(sel):
        q = e["query_dir"]
        p = OOD_TARGET_ROOT / "boom_hourly" / q / "data-00000-of-00001.arrow"
        series.append(_boom_read_variate(p, e["variate_index"]))    # only the selected variate
        cluster_ids.append(c)                                       # cluster = parent metric query
        names.append(f"{q}:v{e['variate_index']}")
    return {"series": series, "cluster_ids": cluster_ids, "cluster_unit": "metric_query",
            "cluster_names": names,
            "notes": "BOOM native-hourly; one quality-passing variate per query (manifest-pinned)"}


_OOD_LOADERS = {"sg_carpark": _load_sg_carpark, "coastal_ts": _load_coastal_ts,
                "boom_hourly": _load_boom_hourly}


def load_ood_target_series(tag):
    """Load a documented pretraining-OOD target as univariate series + parent-cluster ids.

    Returns {series: list[1-D float64], cluster_ids: list[int], cluster_unit: str,
             cluster_names: list[str], notes: str}. EVALUATION-ONLY — nothing here is ever used
    to train a probe."""
    if tag not in _OOD_LOADERS:
        raise ValueError(f"unknown OOD target {tag!r}; known: {sorted(_OOD_LOADERS)}")
    return _OOD_LOADERS[tag]()


def _cluster_balanced_order(cluster_ids, target, rng):
    """Cluster-balanced (query-balanced) round-robin selection indices over parent clusters.

    Sweep clusters in rounds taking ONE candidate window per cluster per round, until `target`
    (or all candidates) are chosen: every cluster contributes its 1st window before any
    contributes a 2nd, and so on. This stops a few long series from dominating the 650 and keeps
    broad independent-cluster coverage (what the series-level cluster bootstrap rests on).
    Deterministic given `rng`: the cluster visiting order AND each cluster's within-candidate
    order are permuted once, so which clusters receive the extra final-round window is unbiased
    rather than id-ordered."""
    by: dict[int, list[int]] = {}
    for idx, c in enumerate(np.asarray(cluster_ids).tolist()):
        by.setdefault(int(c), []).append(idx)
    clusters = np.array(sorted(by))
    rng.shuffle(clusters)                                   # unbiased cluster visiting order
    clusters = clusters.tolist()
    for c in clusters:
        arr = np.array(by[c]); rng.shuffle(arr); by[c] = arr.tolist()   # unbiased within-cluster
    if target is None:
        target = len(cluster_ids)
    order, k = [], 0
    maxlen = max((len(v) for v in by.values()), default=0)
    while len(order) < target and k < maxlen:
        for c in clusters:
            if k < len(by[c]):
                order.append(by[c][k])
                if len(order) >= target:
                    break
        k += 1
    return order


def build_ood_windows(tag, C: int = 512, H: int = 64, stride: int = 64,
                      target_test: int = 650, sigma_eps: float = 1e-6,
                      m_season: int = 24, seed: int = SEED):
    """Evaluation-only windows for a pretraining-OOD target.

    Same window geometry + label transform as ``build_windows`` (context-standardize -> arcsinh
    trajectory in Chronos-2's own target space), but there is NO train split: the FROZEN source
    probe is only SCORED here. ``series_test`` carries the PARENT-CLUSTER id (carpark / station /
    metric-query) so the paired series-level cluster bootstrap resamples whole clusters. The
    reported MASE uses the in-context seasonal-naive denominator (run_id_forecasting._mase_
    denominator, context-only), so ``test_denominator`` is NaN here (the canonical train-series
    denominator is undefined for an eval-only target). Returns the SAME dict shape as
    build_windows with empty train arrays, so extraction / predict / compute_mase / the bootstrap
    all work unchanged."""
    loaded = load_ood_target_series(tag)
    rng = np.random.default_rng(seed)
    te_ctx, te_y, te_yv, te_cid = [], [], [], []
    skipped = 0
    for s, cid in zip(loaded["series"], loaded["cluster_ids"]):
        starts = _all_starts(len(s), C, H, stride)
        c, y, v, k = _make_examples(s, starts, C, H, sigma_eps)
        te_ctx += c; te_y += y; te_yv += v; te_cid += [cid] * len(c); skipped += k
    n_full = len(te_y)
    cand_cid = np.asarray(te_cid, dtype=np.int64)
    # cluster-balanced (query-balanced) round-robin down to target_test: every cluster gets its
    # 1st window before any gets a 2nd, and so on — so a few long series can't dominate the 650
    # and independent-cluster coverage stays broad. Deterministic (seed).
    sel = _cluster_balanced_order(cand_cid, target_test, rng)
    te_ctx = [te_ctx[i] for i in sel]
    te_yv = [te_yv[i] for i in sel]
    te_y = [te_y[i] for i in sel]
    te_cid = [int(cand_cid[i]) for i in sel]

    X_test = np.stack(te_ctx).astype(np.float32) if te_ctx else np.zeros((0, C), np.float32)
    y_test = np.asarray(te_y, dtype=np.float32)
    Y_test_traj = np.stack(te_yv).astype(np.float32) if te_yv else np.zeros((0, H), np.float32)
    series_test = np.asarray(te_cid, dtype=np.int64)
    test_denominator = np.full(len(y_test), np.nan, dtype=np.float64)

    if series_test.size:
        _u, _c = np.unique(series_test, return_counts=True)
        wpc = [int(_c.min()), int(np.median(_c)), int(_c.max())]
        n_contrib = int(len(_u))
    else:
        wpc, n_contrib = [0, 0, 0], 0

    meta = {
        "tag": tag, "ood_target": True, "split_mode": "ood_eval_only",
        "cluster_unit": loaded["cluster_unit"], "sampling": "cluster_balanced_round_robin",
        "C": C, "H": H, "stride": stride, "target_test": target_test,
        "sigma_eps": sigma_eps, "seed": seed, "m_season": m_season,
        "mase_denominator": "in_context_seasonal_naive (compute_mase); train-series denom undefined",
        "mase_canonical": False,
        "n_clusters_total": int(len(set(loaded["cluster_ids"]))),
        "n_test_clusters": n_contrib, "windows_per_cluster_min_med_max": wpc,
        "n_test_windows_before_subsample": int(n_full),
        "n_test": int(len(y_test)), "n_train": 0, "n_skipped_windows": int(skipped),
        "notes": loaded["notes"],
    }
    return {"X_train": np.zeros((0, C), np.float32), "y_train": np.zeros((0,), np.float32),
            "X_test": X_test, "y_test": y_test,
            "Y_train_traj": np.zeros((0, H), np.float32), "Y_test_traj": Y_test_traj,
            "series_train": np.zeros((0,), np.int64), "series_test": series_test,
            "test_denominator": test_denominator, "meta": meta}
