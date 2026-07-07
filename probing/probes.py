"""Pluggable probe methods — the extension point for new probing techniques.

A *probe* answers: "given the per-layer representations of a dataset, how much
task-relevant / structural information does each layer carry?" The reference probe is a
linear (logistic-regression) classifier; collaborators add alternative probes
(effective-rank, entropy, epiplexity, ...) here WITHOUT touching extraction or the
experiment loops.

--------------------------------------------------------------------------------------
PROBE CONTRACT
--------------------------------------------------------------------------------------
    probe(train_feats, train_labels, test_feats, test_labels) -> dict[int, float]

    train_feats / test_feats : {layer_idx: np.ndarray of shape (n_samples, d)}
                               d = n_channels * 768, produced by extract_features(...)
    train_labels / test_labels : 1-D array of labels aligned with the feature rows
    returns : {layer_idx: float}  — one scalar score per layer
              (higher = "more information", by convention)

- SUPERVISED probes (e.g. the linear probe) fit on the train split and evaluate on test.
- UNSUPERVISED measures (effective-rank, entropy) may ignore the labels and/or the train
  split and compute directly on `test_feats`. They still receive all four arguments so
  every probe has one uniform signature and slots into the same driver loop.

To register a new probe: implement it below and add it to the PROBES dict at the bottom.
See README > "Adding a new probe".
--------------------------------------------------------------------------------------
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from probing.config import NUM_LAYERS, SEED
from probing.extraction import fit_layerwise_probes


def score_layerwise_correctness(probes, features, y_true):
    """Returns dict {layer_idx: float32 correctness array of shape (n_test,)}."""
    y_true = np.asarray(y_true)
    out = {}
    for i in range(NUM_LAYERS):
        Xs = probes[i]["scaler"].transform(features[i])
        y_pred = probes[i]["clf"].predict(Xs)
        out[i] = (y_pred == y_true).astype(np.float32)
    return out


# ============================ reference probe (linear) ============================ #

def linear_probe(train_feats, train_labels, test_feats, test_labels):
    """Reference probe: per-layer StandardScaler + LogisticRegression, test accuracy.

    Numerically identical to the original pipeline: it simply composes the unchanged
    ``fit_layerwise_probes`` (fit on train only) and ``score_layerwise_correctness``,
    then reduces each layer's per-sample correctness to a mean accuracy.

    Returns {layer_idx: test_accuracy}.
    """
    probes = fit_layerwise_probes(train_feats, train_labels)
    correct = score_layerwise_correctness(probes, test_feats, test_labels)
    # reduce in float64 (matches how the pipeline's bootstrap_ci computes point accuracy)
    return {i: float(np.mean(correct[i], dtype=np.float64)) for i in range(NUM_LAYERS)}


# ===================== ID forecasting probes (Phase 0, linear) ==================== #
# These consume windowed forecasting features (see probing.id_data / extract_window_features)
# whose labels are the normalized H-step future mean. Both stay strictly LINEAR so the
# tunnel-effect diagnostic (linear readout) is preserved.

def ridge_regression_probe(train_feats, train_labels, test_feats, test_labels):
    """Per-layer linear Ridge regression on the (continuous) normalized future-mean label.

    Alpha is chosen per layer from a small grid on a held-out validation split carved from
    train (fixed-seed random 80/20). Score = R^2 on the test split. R^2 can be negative
    (worse than predicting the mean); that is reported as-is, never clipped.
    """
    y_tr = np.asarray(train_labels, dtype=np.float64)
    y_te = np.asarray(test_labels, dtype=np.float64)
    alphas = [0.1, 1.0, 10.0, 100.0]
    rng = np.random.default_rng(SEED)
    n = len(y_tr)
    perm = rng.permutation(n)
    n_val = max(1, int(0.2 * n))
    va, tr = perm[:n_val], perm[n_val:]

    out = {}
    for i in range(NUM_LAYERS):
        Xtr = train_feats[i]
        # alpha selection on the carved validation split
        sc = StandardScaler().fit(Xtr[tr])
        best_a, best_r2 = alphas[0], -np.inf
        for a in alphas:
            m = Ridge(alpha=a).fit(sc.transform(Xtr[tr]), y_tr[tr])
            r2v = r2_score(y_tr[va], m.predict(sc.transform(Xtr[va])))
            if r2v > best_r2:
                best_r2, best_a = r2v, a
        # refit on FULL train with the selected alpha, score on test
        sc_full = StandardScaler().fit(Xtr)
        m = Ridge(alpha=best_a).fit(sc_full.transform(Xtr), y_tr)
        out[i] = float(r2_score(y_te, m.predict(sc_full.transform(test_feats[i]))))
    return out


def binned_future_probe(train_feats, train_labels, test_feats, test_labels, n_bins=5):
    """Per-layer logistic regression on the future-mean label discretized into K quantile bins.

    Bin edges are computed on TRAIN LABELS ONLY. Score = accuracy per layer — an accuracy-scale
    curve directly comparable to the UEA classification curves, and robust to any last-layer
    triviality of raw regression. This is the PRIMARY tunnel-signature readout for the ID
    condition; ridge R^2 is secondary.
    """
    y_tr = np.asarray(train_labels, dtype=np.float64)
    y_te = np.asarray(test_labels, dtype=np.float64)
    edges = np.quantile(y_tr, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])  # K-1 interior edges
    ytr_b = np.digitize(y_tr, edges)
    yte_b = np.digitize(y_te, edges)

    out = {}
    for i in range(NUM_LAYERS):
        scaler = StandardScaler().fit(train_feats[i])
        clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(
            scaler.transform(train_feats[i]), ytr_b)
        pred = clf.predict(scaler.transform(test_feats[i]))
        out[i] = float(np.mean((pred == yte_b), dtype=np.float64))
    return out


# ======================= NEW PROBES GO HERE (collaborators) ======================= #
# Implement each with the standard 4-arg signature and register it in PROBES below.
# Raise NotImplementedError until filled in so the registry stays importable.

def effective_rank(train_feats, train_labels, test_feats, test_labels):
    """TODO: effective rank (e.g. entropy of normalized singular values) of each
    layer's TEST feature matrix. Unsupervised — ignore the labels."""
    raise NotImplementedError("effective_rank probe not implemented yet")


def entropy(train_feats, train_labels, test_feats, test_labels):
    """TODO: representation entropy per layer (unsupervised)."""
    raise NotImplementedError("entropy probe not implemented yet")


def epiplexity(train_feats, train_labels, test_feats, test_labels):
    """TODO: epiplexity per layer."""
    raise NotImplementedError("epiplexity probe not implemented yet")


# ================================ probe registry ================================= #
# name -> callable(train_feats, train_labels, test_feats, test_labels) -> {layer: score}
PROBES = {
    "linear": linear_probe,                       # UEA classification (reference)
    "ridge_regression": ridge_regression_probe,   # ID forecasting, R^2
    "binned_future": binned_future_probe,         # ID forecasting, accuracy (primary readout)
    # "effective_rank": effective_rank,           # uncomment once implemented
    # "entropy": entropy,
    # "epiplexity": epiplexity,
}
