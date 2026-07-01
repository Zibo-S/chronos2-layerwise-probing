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

from probing.config import NUM_LAYERS
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
    "linear": linear_probe,
    # "effective_rank": effective_rank,   # uncomment once implemented
    # "entropy": entropy,
    # "epiplexity": epiplexity,
}
