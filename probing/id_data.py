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

# Chronos-2-seen ID datasets (hourly variants), with the HF config + target column name.
ID_DATASETS = {
    "m4_hourly":                 {"hf_config": "m4_hourly",                 "target": "target"},
    "monash_electricity_hourly": {"hf_config": "monash_electricity_hourly", "target": "target"},
    "solar_1h":                  {"hf_config": "solar_1h",                  "target": "power_mw"},
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
    """Build (context_float32, label_float32) for each start; skip constant/non-finite windows."""
    ctxs, ys = [], []
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
        y = (float(fut.mean()) - mu) / max(sd, sigma_eps)
        ctxs.append(ctx.astype(np.float32))
        ys.append(np.float32(y))
    return ctxs, ys, n_skipped


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


def _subsample(ctxs, ys, target, rng):
    """Deterministically subsample a list of (ctx, y) down to `target` (keep all if fewer)."""
    n = len(ys)
    if n <= target:
        return ctxs, ys
    idx = np.sort(rng.choice(n, size=target, replace=False))
    return [ctxs[i] for i in idx], [ys[i] for i in idx]


def build_windows(
    tag: str,
    C: int = 512,
    H: int = 64,
    stride: int = 64,
    test_frac: float = 0.25,
    target_train: int = 3000,
    target_test: int = 1500,
    sigma_eps: float = 1e-6,
    seed: int = SEED,
):
    """Build ID probing windows for one seen dataset.

    Returns a dict with:
        X_train, X_test : float32 arrays (n, C)   (univariate context windows)
        y_train, y_test : float32 arrays (n,)     (normalized future-mean labels)
        meta            : split_mode, counts, skip counts, params
    """
    series = load_seen_series(tag)
    rng = np.random.default_rng(seed)
    span = C + H

    n_series_ok = sum(1 for s in series if len(s) >= 2 * span)
    split_mode = "within_series" if n_series_ok >= 1 else "cross_series"

    tr_ctx, tr_y, te_ctx, te_y = [], [], [], []
    skipped = 0

    if split_mode == "within_series":
        for s in series:
            tr_s, te_s = _within_series_starts(len(s), C, H, stride, test_frac)
            c, y, k = _make_examples(s, tr_s, C, H, sigma_eps); tr_ctx += c; tr_y += y; skipped += k
            c, y, k = _make_examples(s, te_s, C, H, sigma_eps); te_ctx += c; te_y += y; skipped += k
    else:
        # cross_series fallback: disjoint train/test series (leakage-free), all windows each.
        order = rng.permutation(len(series))
        n_tr_series = int(round(0.7 * len(series)))
        tr_series = set(order[:n_tr_series].tolist())
        for i, s in enumerate(series):
            starts = _all_starts(len(s), C, H, stride)
            c, y, k = _make_examples(s, starts, C, H, sigma_eps)
            if i in tr_series:
                tr_ctx += c; tr_y += y
            else:
                te_ctx += c; te_y += y
            skipped += k

    n_tr_full, n_te_full = len(tr_y), len(te_y)
    tr_ctx, tr_y = _subsample(tr_ctx, tr_y, target_train, rng)
    te_ctx, te_y = _subsample(te_ctx, te_y, target_test, rng)

    X_train = np.stack(tr_ctx).astype(np.float32) if tr_ctx else np.zeros((0, C), np.float32)
    X_test = np.stack(te_ctx).astype(np.float32) if te_ctx else np.zeros((0, C), np.float32)
    y_train = np.asarray(tr_y, dtype=np.float32)
    y_test = np.asarray(te_y, dtype=np.float32)

    meta = {
        "tag": tag,
        "hf_config": ID_DATASETS[tag]["hf_config"],
        "split_mode": split_mode,
        "n_series": len(series),
        "n_series_supporting_within": n_series_ok,
        "C": C, "H": H, "stride": stride, "test_frac": test_frac,
        "sigma_eps": sigma_eps, "seed": seed,
        "n_train_windows_before_subsample": n_tr_full,
        "n_test_windows_before_subsample": n_te_full,
        "n_train": int(len(y_train)), "n_test": int(len(y_test)),
        "n_skipped_windows": int(skipped),
    }
    return {"X_train": X_train, "y_train": y_train, "X_test": X_test, "y_test": y_test, "meta": meta}
