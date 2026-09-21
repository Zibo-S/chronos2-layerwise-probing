"""Representation geometry of the TiRex native readout states: linear CKA + effective rank.

Label-free, model-free, CPU-only. It consumes the SAME (n, K, d) readout features the Q=1 shared
patch probe consumes and hands them to the project's EXISTING estimators. **Nothing here
re-implements a similarity or a rank measure** -- the three helpers below are imported verbatim
from the TimesFM-3 geometry module, which itself only wraps:

    linear CKA (biased + unbiased)   probing.cka.cka_matrix / linear_cka       [UNCHANGED]
    entropy effective rank           probing.spectral_metrics.spectral_metrics [UNCHANGED]

Importing rather than re-writing them is the point: it is what guarantees Chronos-2, TimesFM-3
and TiRex are compared under ONE estimator instead of three look-alikes. The only model-specific
choice is the representation matrix:

    Chronos-2 : forecast slots     (n, K=4, 768)  -> (4n, 768)   [cka.stack_slots]
    TimesFM-3 : one readout token  (n, 1280)      -> (n, 1280)   [no reshape, K=1]
    TiRex     : ONE readout patch  (n, 512)       -> (n, 512)    [headline: `pos0`]
                both readout patches (n, K=2, 512)  -> (2n, 512)   [companion, L1 onward]

Row ordering is row-major and IDENTICAL at every depth: row i*K + k is window i, readout position
k. It is the same ordering ``tirex_probes._stack`` uses for the shared probe's feature matrix, so
"the probe's rows" and "the CKA rows" are provably the same objects in the same order.

RAW REPRESENTATIONS ONLY. ``assert_not_standardized`` refuses a matrix that looks like it has
been through the probe's train-fit StandardScaler; CKA and effective rank see the hidden states
as the model produced them, with only the centering the estimators themselves apply.

WHY THE NULL FLOOR IS REPORTED. The biased estimator has an O(1/n) UPWARD bias (probing/cka.py
says so), and it MOVES WITH n -- which is the second reason the headline keeps n constant across
depth. The headline `pos0` construction gives 1 row per window at d=512 (262 rows on a test
split, like TimesFM-3's 262 at d=1280); the companion gives 2n. Chronos-2 has 4n at d=768. The
floors therefore differ across the three models, which is exactly why:

    NEVER compare ABSOLUTE biased CKA across Chronos-2, TimesFM-3 and TiRex. The finite-sample
    baselines differ. Compare patterns within a model, or compare the unbiased numbers.

The unbiased estimator is the cross-model one; it is NOT bounded in [0, 1] and small negative
values are left exactly as computed (clipping them would destroy the property that makes it
unbiased). Both estimators are produced for every dataset and split.
"""

from __future__ import annotations

import numpy as np

from probing import cka as cka_mod
# The three estimator wrappers, imported UNCHANGED so all three model lines share one
# implementation. They are model-agnostic: cka_layer_matrix/cka_null_floor take only matrices,
# and effective_rank_curve takes `hidden_dim` as an argument (we pass TiRex's 512).
from probing.timesfm3_geometry import (cka_layer_matrix, cka_null_floor,  # noqa: F401
                                       effective_rank_curve)
from probing.tirex_model import MODEL_DIMS, REP_NAMES

__all__ = ["CKA_ESTIMATORS", "HEADLINE_CKA_ESTIMATOR", "CKA_HEADLINE_SPLIT",
           "ERANK_HEADLINE_SPLIT", "rep_matrix", "rep_matrices", "assert_not_standardized",
           "cka_layer_matrix", "cka_null_floor", "effective_rank_curve", "geometry_for_split",
           "degenerate_readouts", "CROSS_MODEL_CAVEAT", "DEGENERACY_CAVEAT",
           "EMB_POLICY", "EMB_POINT", "variant_spec", "GEOMETRY_VARIANTS",
           "VARIANT_POS0", "VARIANT_ALL_FROM_L1",
           "CKA_HEADLINE_VARIANT", "ERANK_HEADLINE_VARIANT"]

# Emitted beside every headline geometry record; see degenerate_readouts().
DEGENERACY_CAVEAT = (
    "At the Emb depth ONLY, readout position 1 (the first masked-future token) is bit-identical "
    "across windows: its patch-embedding input is (values=0, mask=0) before any recurrence, so "
    "the embedding is a pure bias. This is why the HEADLINE geometry is `pos0` (position 0 only, "
    "at every depth) and why the `all_positions_from_L1` companion starts at L1: a 2n-row matrix "
    "at Emb would be n varying rows plus n copies of one point, which deflates its spectrum and "
    "distorts its CKA row relative to every other depth. Neither produced curve contains that "
    "matrix. The degeneracy is still measured and reported every run.")

# Both always run. "unbiased" is the cross-model headline (the spec's instruction); "biased" is
# retained for within-model parity with the committed Chronos-2 CKA analysis
# (experiments/run_cka_analysis.py uses probing.cka's default, which is "biased").
CKA_ESTIMATORS = ("biased", "unbiased")
HEADLINE_CKA_ESTIMATOR = "unbiased"

# Split conventions, inherited per-analysis from the committed Chronos-2 runs, which disagree
# with each other -- so parity is per-analysis, not global (the TimesFM-3 line made the same
# call, for the same reason):
#   run_cka_analysis.py --fslot-split -> test      run_spectral.py --split -> train
# Both analyses are nevertheless computed on BOTH splits and saved, so the spec's "effective rank
# on the SAME matrix as CKA" is available as the test-split entry.
CKA_HEADLINE_SPLIT = "test"
ERANK_HEADLINE_SPLIT = "train"

CROSS_MODEL_CAVEAT = (
    "Biased CKA is retained for parity with the committed Chronos-2 analysis, but its "
    "finite-sample baseline differs across the three models because the number of readout rows "
    "per window and the representation dimension differ (Chronos-2: 4 rows/window at d=768; "
    "TimesFM-3: 1 row/window at d=1280; TiRex headline `pos0`: 1 row/window at d=512, companion "
    "`all_positions_from_L1`: 2 rows/window). Absolute biased CKA is "
    "therefore NOT comparable across models -- use the unbiased estimator for cross-model "
    "statements, or compare patterns within a model. Per-(n, d, estimator) null floors are "
    "measured and reported beside every matrix; nothing is subtracted from the matrices.")


# --------------------------------------------------------------------------- #
# the representation matrix
# --------------------------------------------------------------------------- #
def rep_matrix(F, positions=None) -> np.ndarray:
    """(n, K, d) readout features -> (n*K', d) CKA/effective-rank observation matrix.

    Delegates the reshape to ``probing.cka.stack_slots``, the same helper the Chronos-2 forecast
    slots go through, so the row ordering is provably identical across the two model lines.

    ``positions`` optionally restricts to a subset of readout positions BEFORE stacking (it must
    be the same subset at every depth a variant covers -- ``variant_spec`` is the single source of
    truth for that pairing). See ``EMB_POLICY`` for why the headline uses positions (0,).
    """
    A = np.asarray(F)
    if A.ndim != 3:
        raise ValueError(f"expected (n, K, d) readout features, got shape {A.shape}; a 2-D array "
                         "would silently skip the position stacking")
    if A.shape[-1] != MODEL_DIMS:
        raise ValueError(f"expected d={MODEL_DIMS} (TiRex embedding_dim), got {A.shape[-1]}")
    if positions is not None:
        pos = list(positions)
        if not pos or min(pos) < 0 or max(pos) >= A.shape[1]:
            raise ValueError(f"positions {pos} outside 0..{A.shape[1] - 1}")
        A = A[:, pos, :]
    M = cka_mod.stack_slots(A)
    assert M.shape == (A.shape[0] * A.shape[1], A.shape[2]), "stack_slots changed its layout"
    return np.asarray(M, dtype=np.float64)


def degenerate_readouts(feats: dict, points=None) -> dict:
    """Which (representation point, readout position) pairs are CONSTANT across windows.

    There is exactly one, and it is a true property of TiRex rather than an extraction fault:

        (Emb, position 1) -- the first MASKED FUTURE token enters the patch embedding as
        (values=0, mask=0), because no recurrence has touched it yet. ``input_patch_embedding``
        of a zero vector is a pure bias term, so that row is bit-identical for every window.

    Why it must be reported for the geometry analyses: the headline (n*K, d) matrix at Emb is
    then n varying rows PLUS n copies of one point. That block of identical rows deflates the
    spectrum and distorts Emb's CKA row relative to every other depth, which are window-dependent
    at both positions. The driver therefore also computes a ``pos0`` companion (position 0 only,
    n rows, clean at EVERY depth) so Emb can be read on a like-for-like basis. Nothing is
    dropped from the headline matrix -- the spec's (2N, d) construction is preserved and the
    caveat travels with it.
    """
    names = list(points) if points is not None else list(REP_NAMES)
    flags, affected = {}, []
    for n in names:
        A = np.asarray(feats[n])
        const = [bool(np.all(A[:, k] == A[0:1, k])) for k in range(A.shape[1])]
        flags[n] = const
        affected += [(n, k) for k, c in enumerate(const) if c]
    return {"constant_across_windows_by_point": flags,
            "affected": [{"point": n, "position": k} for n, k in affected],
            "n_affected": len(affected),
            "expected": [{"point": "Emb", "position": 1}],
            "note": degenerate_readouts.__doc__.strip().splitlines()[0]}


def assert_not_standardized(M, name: str = "representation") -> None:
    """Guard against handing CKA / effective rank the PROBE's standardized features.

    A ``StandardScaler``-transformed train matrix has per-column mean ~0 and std ~1 by
    construction. Raw TiRex states do not. This is a heuristic, deliberately loose enough never
    to fire on genuine raw states, and it exists because silently z-scoring before an effective
    rank would change the spectrum and make the curve incomparable with Chronos-2's.
    """
    A = np.asarray(M, dtype=np.float64)
    if A.ndim != 2 or A.shape[0] < 4:
        return
    mu, sd = np.abs(A.mean(axis=0)).max(), A.std(axis=0)
    if mu < 1e-6 and np.abs(sd - 1.0).max() < 1e-3:
        raise RuntimeError(
            f"{name} looks dimension-wise standardized (max|mean| {mu:.2e}, std within 1e-3 of 1). "
            "CKA and effective rank must see RAW hidden states -- pass the extracted features, "
            "not the probe's scaler output.")


def rep_matrices(feats: dict, points=None, positions=None) -> tuple[list[np.ndarray], list[str]]:
    """Ordered (matrices, names) for one split. Every matrix shares the same rows in the same
    order -- the precondition ``probing.cka.require_matched_rows`` enforces downstream."""
    names = list(points) if points is not None else list(REP_NAMES)
    mats = []
    for n in names:
        M = rep_matrix(feats[n], positions=positions)
        assert_not_standardized(M, f"representation {n}")
        mats.append(M)
    cka_mod.require_matched_rows(mats)
    return mats, names


# --------------------------------------------------------------------------- #
# the per-split analysis
# --------------------------------------------------------------------------- #
EMB_POINT = "Emb"

# The two row constructions. BOTH CKA and effective rank use the SAME one as headline, so the
# similarity curve and the rank curve describe the same matrices.
VARIANT_POS0 = "pos0"
VARIANT_ALL_FROM_L1 = "all_positions_from_L1"
GEOMETRY_VARIANTS = (VARIANT_POS0, VARIANT_ALL_FROM_L1)
CKA_HEADLINE_VARIANT = VARIANT_POS0
ERANK_HEADLINE_VARIANT = VARIANT_POS0

EMB_POLICY = (
    "The Emb masked-future state is constant before recurrence, so it is excluded from the "
    "headline geometry entirely: the HEADLINE construction is `pos0` -- the last-real-context "
    "readout state at EVERY depth, n rows throughout -- for BOTH linear CKA and effective rank. "
    "The observation count is therefore CONSTANT across depth, which is what makes a rank curve "
    "comparable from one depth to the next; a curve whose n jumps from n at Emb to 2n afterwards "
    "is deliberately NOT produced, because min(n-1, d) and the O(1/n) CKA bias both move with n "
    "and the jump would be indistinguishable from a representational change. It is also the only "
    "construction CKA admits at all: CKA is pairwise and requires MATCHED ROWS "
    "(probing.cka.require_matched_rows refuses otherwise), so an n-row Emb cannot be compared "
    "with a 2n-row L1. The COMPANION construction is `all_positions_from_L1`: both "
    "forecast-producing states, 2n rows, over L1..L12+RMS ONLY -- Emb is omitted because its "
    "second state is degenerate there. The companion is self-consistent (constant n across the "
    "depths it covers) and is where the second forecast state's geometry is read."
)


def variant_spec(variant: str, points, K: int) -> tuple[list[str], tuple[int, ...]]:
    """(depths, readout positions) for a geometry variant. ONE source of truth for both
    estimators, so the CKA matrix and the effective-rank curve can never disagree about which
    rows they describe."""
    names = list(points)
    if variant == VARIANT_POS0:
        return names, (0,)
    if variant == VARIANT_ALL_FROM_L1:
        return [n for n in names if n != EMB_POINT], tuple(range(K))
    raise ValueError(f"unknown geometry variant {variant!r}; known: {GEOMETRY_VARIANTS}")


def _cka_block(mats, names, estimators, null_floor_reps, seed) -> tuple[dict, dict]:
    n, d = mats[0].shape
    cka, floors = {}, {}
    for est in estimators:
        cka[est] = cka_layer_matrix(mats, estimator=est)
        floors[est] = cka_null_floor(n, d, reps=null_floor_reps, seed=seed, estimator=est)
        off = cka[est][~np.eye(len(names), dtype=bool)]
        floors[est]["min_offdiagonal_cka"] = float(off.min())
        floors[est]["mean_offdiagonal_cka"] = float(off.mean())
        floors[est]["mean_offdiagonal_above_null_floor"] = float(off.mean() - floors[est]["mean"])
    return cka, floors


def geometry_for_split(feats: dict, *, points=None, estimators=CKA_ESTIMATORS,
                       null_floor_reps: int = 3, seed: int = 0, n_subsamples: int = 0) -> dict:
    """CKA (every estimator) + effective rank on ONE split's readout features.

    Produces BOTH row constructions, with CKA and effective rank always describing the SAME
    matrices within a variant (see ``EMB_POLICY``):

        pos0                  n rows, ALL 14 depths      -- last-real-context state  [HEADLINE]
        all_positions_from_L1 n*K rows, L1..L12+RMS only -- both forecast states     [companion]

    Within each variant the observation count is CONSTANT across the depths it covers, so a rank
    curve is comparable along it and the CKA null floor is a single number.
    """
    names = list(points) if points is not None else list(REP_NAMES)
    K = int(np.asarray(feats[names[0]]).shape[1])

    out = {"representation_points": names, "feature_dim": int(MODEL_DIMS),
           "n_windows": int(np.asarray(feats[names[0]]).shape[0]), "positions_per_window": K,
           "row_order": "row-major: row i*K + k is window i, readout position k",
           "emb_policy": EMB_POLICY, "variants": list(GEOMETRY_VARIANTS),
           "degeneracy": degenerate_readouts(feats, names),
           "degeneracy_caveat": DEGENERACY_CAVEAT,
           "cka_headline_variant": CKA_HEADLINE_VARIANT,
           "effective_rank_headline_variant": ERANK_HEADLINE_VARIANT,
           "cka": {}, "cka_null_floor": {}, "n_rows": {}, "variant_points": {},
           "effective_rank": {}, "cross_model_caveat": CROSS_MODEL_CAVEAT}

    for vname in GEOMETRY_VARIANTS:
        vnames, pos = variant_spec(vname, names, K)
        vfeats = {n: feats[n] for n in vnames}
        mats, _ = rep_matrices(vfeats, vnames, pos)
        out["variant_points"][vname] = vnames
        out["n_rows"][vname] = int(mats[0].shape[0])
        out["cka"][vname], out["cka_null_floor"][vname] = _cka_block(
            mats, vnames, estimators, null_floor_reps, seed)
        er = effective_rank_curve(mats, vnames, hidden_dim=MODEL_DIMS,
                                  n_subsamples=n_subsamples, seed=seed)
        er.update(variant=vname, positions_used=list(pos),
                  constant_observation_count=True,
                  representation_points=vnames)
        out["effective_rank"][vname] = er
    return out
