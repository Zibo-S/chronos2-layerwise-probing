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

import numpy as np

from probing.config import SEED

# Chronos-2-seen ID datasets (long hourly series -> all support the within_series temporal
# split), with the HF config + target column name. All four verified against
# autogluon/chronos_datasets: target column == "target", min sampled series length >> 2*(C+H).
ID_DATASETS = {
    "monash_electricity_hourly": {"hf_config": "monash_electricity_hourly", "target": "target"},  # Energy
    "monash_kdd_cup_2018":       {"hf_config": "monash_kdd_cup_2018",       "target": "target"},  # Nature / air quality
    "monash_pedestrian_counts":  {"hf_config": "monash_pedestrian_counts",  "target": "target"},  # Transport / foot traffic
    "uber_tlc_hourly":           {"hf_config": "uber_tlc_hourly",           "target": "target"},  # Transport / ride demand
}

_HF_REPO = "autogluon/chronos_datasets"


# --------------------------------------------------------------------------- #
# raw series loading
# --------------------------------------------------------------------------- #
def load_seen_series(tag: str) -> list[np.ndarray]:
    """Download an ID dataset and return its target series as a list of 1-D float64 arrays."""
    from datasets import load_dataset

    spec = ID_DATASETS[tag]
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


def build_windows(
    tag: str,
    C: int = 512,
    H: int = 64,
    stride: int = 64,
    test_frac: float = 0.25,
    target_train: int = 3000,
    target_test: int = 1500,
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
        "hf_config": ID_DATASETS[tag]["hf_config"],
        "split_mode": split_mode,
        "n_series": len(series),
        "n_series_supporting_within": n_series_ok,
        "C": C, "H": H, "stride": stride, "test_frac": test_frac,
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
