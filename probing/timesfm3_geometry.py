"""Representation geometry of the TimesFM-3 LAST-CONTEXT-TOKEN states (CKA + effective rank).

Cache-only, model-free, CPU-only. Loads the validated ``tfm3-last-token-q9-fp32-v1`` features
written by the paper7 probe run and hands them to the project's EXISTING estimators. Nothing
here re-implements a similarity or a rank measure:

    biased linear CKA   probing.cka.cka_matrix(..., estimator="biased")      [UNCHANGED]
    effective rank      probing.spectral_metrics.spectral_metrics(...)       [UNCHANGED]

Both are the exact functions the Chronos-2 analyses call, so the two models are compared under
one estimator, not two look-alikes. The ONLY model-specific choice is the representation matrix:

    Chronos-2 : forecast slots (n, K, 768) -> (n*K, 768)   [cka.stack_slots]
    TimesFM-3 : the last real context token h_{l,15}, directly (N, 1280)  -- no reshape,
                because H=64 gives exactly ONE native readout token per window (geom.K == 1).

SCIENTIFIC RULE (inherited from probing/cka.py and enforced by ``require_matched_rows``): CKA
rows must be the SAME examples. Every layer of one (dataset, split) goes through the same
windows, and there is no code path that pairs two datasets' rows by position.

PROBE INDEPENDENCE (structural, not a promise): ``load_last_token_reps`` returns ONLY the
per-layer feature matrices. The cache's ``mu``/``sd``/``native`` arrays -- the RevIN statistics
and decode()'s own forecast -- are read by the loader's integrity check and then DROPPED, so no
probe weight, prediction, target or native-head output can reach a CKA or an effective rank.

WINDOW IDENTITY: the cache is only accepted when every metadata field agrees AND the stored
context tails match the freshly built paper7 windows element-wise (``read_cache``). A cache from
another dataset, split, suite, checkpoint, geometry or window build is REJECTED, never repaired.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from probing import cka as cka_mod
from probing.spectral_metrics import spectral_metrics, subsample_metrics
from probing.timesfm3_last_token import (DEFAULT_CHECKPOINT, LAYER_NAMES, MODEL_DIMS,
                                         NUM_LAYERS, cache_metadata, cache_root, read_cache)

__all__ = ["repo_root", "paper_out_default", "LAYER_NAMES", "NUM_LAYERS", "MODEL_DIMS",
           "SELECTED_TOKEN_INDEX", "CKA_ESTIMATOR", "estimator_provenance",
           "assert_last_token_matrix", "window_identity_hash", "load_last_token_reps",
           "cka_null_floor",
           "cka_layer_matrix", "effective_rank_curve", "git_commit"]

SELECTED_TOKEN_INDEX = 15          # the last REAL context token for C=512, P=32 (geom asserts it)
CKA_ESTIMATOR = "biased"           # probing.cka's default; stated explicitly so it is in the JSON


# --------------------------------------------------------------------------- #
# paths -- resolved from THIS file, never from a hardcoded home directory
# --------------------------------------------------------------------------- #
def repo_root() -> Path:
    """The chronos2-layerwise-probing checkout root.

    ``probing/`` always sits one level under it, so ``__file__`` locates the repo wherever it is
    cloned (laptop, $HOME on Narval, a compute node's bind mount). ``git rev-parse
    --show-toplevel`` is consulted only as a cross-check and is ignored when git is unavailable
    or the tree is not a repo -- the file-relative answer is authoritative.
    """
    root = Path(__file__).resolve().parent.parent
    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=5)
        if top.returncode == 0:
            g = Path(top.stdout.strip()).resolve()
            if g != root:
                print(f"  [note] git toplevel {g} != file-relative repo root {root}; using {root}")
    except Exception:
        pass
    return root


def paper_out_default() -> Path:
    """Repo-local home for the lightweight paper outputs (summary JSON + final figures).

    Heavy artifacts (feature caches, full CKA matrices, spectra) belong on $SCRATCH via the
    driver's --out-root; only what goes into the paper lands here.
    """
    return repo_root() / "results" / "timesfm3_representation_geometry"


def git_commit() -> str:
    try:
        r = subprocess.run(["git", "-C", str(repo_root()), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------- #
# estimator provenance -- read off the imported implementations, not retyped
# --------------------------------------------------------------------------- #
def estimator_provenance() -> dict:
    """The exact definitions this analysis runs under, for the summary JSON.

    Every field describes ``probing.cka`` / ``probing.spectral_metrics`` as imported here, so the
    record cannot drift from the code: if the shared helper ever changed, the Chronos-2 numbers
    would change with it and this text would still describe what actually ran.
    """
    return {
        "cka": {
            "estimator": CKA_ESTIMATOR,
            "name": "biased linear CKA (Kornblith et al. 2019, biased HSIC)",
            "formula": "||Xc^T Yc||_F^2 / (||Xc^T Xc||_F * ||Yc^T Yc||_F)",
            "centering": "feature-centred across the N observations (axis=0), per representation",
            "dtype": "float64 (probing.cka._as_2d_f64 casts the float32 cache before centering)",
            "degenerate_denominator": "non-finite or <= 0 -> NaN (never silently 0)",
            "implementation": "probing.cka.cka_matrix / linear_cka  [shared with Chronos-2, "
                              "UNCHANGED]",
            "not_used": ["unbiased/debiased HSIC", "RBF CKA", "cosine similarity",
                         "Gram-space variants with other finite-sample corrections"],
        },
        "effective_rank": {
            "name": "effective rank = exp(spectral entropy)  (Roy & Vetterli 2007)",
            "definition": "s = svdvals(X - mean(X, axis=0)); lam = s**2; p = lam / lam.sum(); "
                          "H = -sum p log p (natural log); effective_rank = exp(H)",
            "uses_squared_singular_values": True,
            "centering": "across examples (axis=0) before the SVD",
            "log_base": "natural",
            "epsilon": 1e-12,
            "dtype": "float64 (probing.spectral_metrics._as_2d_float64)",
            "degenerate": "zero total energy -> effective_rank = 0.0",
            "implementation": "probing.spectral_metrics.spectral_metrics  [shared with "
                              "Chronos-2, UNCHANGED]",
            "normalized_effective_rank": "effective_rank / hidden_dim -- an ADDED diagnostic in "
                                         "this driver only; the raw metric is unchanged and is "
                                         "the one comparable to the Chronos-2 records",
        },
    }


# --------------------------------------------------------------------------- #
# representation matrices
# --------------------------------------------------------------------------- #
def assert_last_token_matrix(arr, n: int, name: str) -> np.ndarray:
    """One representation point must be (N, 1280) -- the token-15 state, nothing else.

    Rejects the 16-prefix ablation's (N, 16, 1280) shared-origin array on RANK, so that cache can
    never be analysed here even if its directory were renamed by hand.
    """
    a = np.asarray(arr)
    if a.ndim == 3:
        raise ValueError(
            f"{name}: got a rank-3 array {a.shape}. This is the shared-origin (N, 16, 1280) "
            "representation of the 16-prefix ABLATION; the last-token analysis reads exactly "
            f"one token per window and needs ({n}, {MODEL_DIMS}).")
    if a.ndim != 2:
        raise ValueError(f"{name}: expected a 2-D (N, {MODEL_DIMS}) matrix, got {a.shape}")
    if a.shape != (n, MODEL_DIMS):
        raise ValueError(f"{name}: expected ({n}, {MODEL_DIMS}), got {a.shape}")
    return a


def window_identity_hash(X, series_ids=None) -> str:
    """A 16-char digest of the exact windows an analysis consumed.

    Hashes the raw float32 contexts (and the per-window series/cluster ids when given), so two
    runs that report the same hash provably saw the same rows in the same order.
    """
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(np.asarray(X, np.float32)).tobytes())
    if series_ids is not None:
        h.update(np.ascontiguousarray(np.asarray(series_ids, np.int64)).tobytes())
    return h.hexdigest()[:16]


def load_last_token_reps(tag: str, split: str, X, *, cache_dir, geom, layers=None,
                         checkpoint: str | None = None, suite: str = "paper7", seed: int = 0,
                         detrend: bool = True, feature_dtype=np.float32,
                         ignore_timesfm_version: bool = False, series_ids=None) -> dict:
    """The 21 token-15 representation matrices for one (dataset, split), identity-verified.

    ``X`` is the freshly rebuilt (N, C) paper7 CONTEXT array for that split. It is what makes the
    correspondence a CHECK: ``read_cache`` compares the cached context tails against it
    element-wise and refuses a cache built from any other windowing, on top of a full metadata
    match (dataset, split, suite, checkpoint, C/H/P, token index, seed, dtype, detrending, N).

    Returns ONLY representations + provenance. The cache's ``mu``/``sd``/``native`` entries are
    verified by the read and then discarded here, so this function structurally cannot leak a
    native-head output or any probe quantity into a geometry number.
    """
    layers = list(range(NUM_LAYERS)) if layers is None else sorted(set(int(l) for l in layers))
    bad = [l for l in layers if not 0 <= l < NUM_LAYERS]
    if bad:
        raise ValueError(f"representation points {bad} outside 0..{NUM_LAYERS - 1}")
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != geom.C:
        raise ValueError(f"X must be the (N, {geom.C}) context array, got {X.shape}")
    if geom.selected_token_index != SELECTED_TOKEN_INDEX:
        raise RuntimeError(f"geometry reads token {geom.selected_token_index}, this analysis is "
                           f"specified for token {SELECTED_TOKEN_INDEX}")

    meta = cache_metadata(tag, split, geom, checkpoint=checkpoint, detrend=detrend,
                          layers=layers, seed=seed, feature_dtype=feature_dtype, suite=suite,
                          n_windows=len(X))
    if ignore_timesfm_version:
        meta.pop("timesfm_version", None)
    root = cache_root(cache_dir, tag, split, geom, detrend, suite)
    hit = read_cache(root, meta, X, layers)          # raises on ANY disagreement
    if hit is None:
        raise FileNotFoundError(
            f"no usable last-token feature cache at {root} for {tag}/{split} "
            f"(layers {layers[0]}..{layers[-1]}).\n"
            "  This analysis is CACHE-ONLY: it never loads TimesFM-3 and never extracts. Run the "
            "paper7 probe driver first (it writes train/val/test caches), or point --cache-dir at "
            "the cache that run produced.")

    reps = [assert_last_token_matrix(hit["feats"][l], len(X), LAYER_NAMES[l]) for l in layers]
    cka_mod.require_matched_rows(reps)               # same examples through every layer
    return {"tag": tag, "split": split, "layers": layers,
            "layer_names": [LAYER_NAMES[l] for l in layers], "reps": reps,
            "n": int(len(X)), "hidden_dim": MODEL_DIMS,
            "selected_token_index": int(geom.selected_token_index),
            "cache_root": str(root), "cache_version": meta["cache_version"],
            "checkpoint": checkpoint, "suite": suite, "detrending": bool(detrend),
            "feature_dtype": np.dtype(feature_dtype).name,
            "timesfm_version": meta.get("timesfm_version", "not-checked"),
            "window_identity_hash": window_identity_hash(X, series_ids),
            "context_len": geom.C, "horizon": geom.H, "patch_size": geom.P, "seed": int(seed)}


# --------------------------------------------------------------------------- #
# the two measures -- thin, auditable wrappers over the shared estimators
# --------------------------------------------------------------------------- #
def cka_layer_matrix(reps, *, estimator: str = CKA_ESTIMATOR) -> np.ndarray:
    """Symmetric (L, L) linear-CKA matrix over ordered representation points.

    Delegates to ``probing.cka.cka_matrix`` -- the same call the Chronos-2 analysis makes.
    ``estimator`` defaults to "biased" (that analysis's estimator, and the parity headline);
    "unbiased" selects the Song et al. (2012) companion, which is NOT bounded in [0, 1] and may
    return slightly negative values for unrelated representations -- that is exactly what makes
    it unbiased, and it is not repaired here. Post-conditions asserted here (the estimator itself
    returns NaN for a degenerate representation rather than a silent 0):
      * finite everywhere -- a NaN means a constant/degenerate layer and is reported, not hidden;
      * symmetric to 1e-10;
      * unit diagonal to 1e-8 (the ratio of two large Frobenius norms; 1e-12 is below float64
        rounding for d=1280 and would fail spuriously).
    """
    M = cka_mod.cka_matrix(reps, estimator=estimator)
    if not np.all(np.isfinite(M)):
        n_bad = int((~np.isfinite(M)).sum())
        idx = sorted({int(i) for i in np.argwhere(~np.isfinite(M)).ravel()})
        raise RuntimeError(
            f"the CKA matrix has {n_bad} non-finite entries at representation points {idx}. "
            "probing.cka returns NaN when ||Xc^T Xc||_F is zero or non-finite, i.e. a layer is "
            "constant across the selected windows. Investigate that layer -- this is NOT repaired "
            "and NOT replaced by 0.")
    asym = float(np.abs(M - M.T).max())
    if asym > 1e-10:
        raise RuntimeError(f"CKA matrix is not symmetric (max |M - M^T| = {asym:.3e})")
    d = float(np.abs(np.diag(M) - 1.0).max())
    if d > 1e-8:
        raise RuntimeError(f"CKA diagonal departs from 1 by {d:.3e}; CKA(X, X) must be exactly 1 "
                           "up to float64 rounding")
    return M


def cka_null_floor(n: int, d: int, *, reps: int = 3, seed: int = 0,
                   estimator: str = CKA_ESTIMATOR) -> dict:
    """The value this estimator returns for INDEPENDENT representations at this exact (n, d).

    Calibration, not a correction. The biased estimator carries an O(1/n) UPWARD bias that
    grows as n falls relative to the representations' effective rank (probing/cka.py says so in
    its own docstring), and the TimesFM-3 last-token setting is squarely in that regime: one
    readout token per window means n = the window count, where Chronos-2's K=4 forecast slots
    gave it 4n rows at d=768. Measuring the floor on i.i.d. Gaussian matrices of the SAME shape
    turns "0.83 is high" into "0.83 is the floor here", so a heatmap can be read honestly.

    Nothing is subtracted from the reported matrix; this number is saved beside it.
    """
    rng = np.random.default_rng(seed)
    vals = [cka_mod.linear_cka(rng.normal(size=(n, d)), rng.normal(size=(n, d)),
                               estimator=estimator) for _ in range(max(1, int(reps)))]
    a = np.asarray(vals, dtype=np.float64)
    return {"estimator": estimator, "n": int(n), "d": int(d), "n_repeats": len(vals),
            "mean": float(a.mean()), "std": float(a.std()), "values": a.tolist(), "seed": int(seed),
            "meaning": "biased-linear-CKA value for two INDEPENDENT Gaussian representations of "
                       "this shape; the true CKA there is 0, so this is the estimator's own "
                       "upward bias floor at this (n, d). NOT subtracted from the reported matrix"}


def effective_rank_curve(reps, layer_names, *, hidden_dim: int = MODEL_DIMS, n_subsamples: int = 0,
                         frac: float = 0.8, seed: int = 0) -> dict:
    """Layerwise effective rank via the UNCHANGED Chronos-2 helper, plus added normalizations.

    ``spectral_metrics`` supplies effective_rank / spectral_entropy / pc1_fraction /
    numerical_rank exactly as it does for Chronos-2. Two ADDED diagnostics (they do not replace
    or alter the raw metric):
      normalized_effective_rank        r_eff / d               (d = 1280)
      effective_rank_over_max_possible r_eff / min(N - 1, d)   -- the attainable ceiling after
                                       centering removes one dimension; the honest denominator
                                       when N is small (Coastal T-S test has N=48 -> ceiling 47).

    ``n_subsamples > 0`` adds the Chronos-2 uncertainty protocol (``subsample_metrics``:
    WITHOUT replacement, frac=0.8) -- off by default because it costs n_subsamples SVDs per layer.
    """
    n = int(np.asarray(reps[0]).shape[0])
    ceiling = float(min(n - 1, hidden_dim))
    out = {"layer_names": list(layer_names), "n_samples": n, "hidden_dim": int(hidden_dim),
           "max_possible_rank": ceiling, "effective_rank": [], "normalized_effective_rank": [],
           "effective_rank_over_max_possible": [], "spectral_entropy": [], "pc1_fraction": [],
           "numerical_rank": [], "spectrum": {}}
    for name, R in zip(layer_names, reps):
        m = spectral_metrics(R, return_spectrum=True)
        out["effective_rank"].append(m["effective_rank"])
        out["normalized_effective_rank"].append(m["effective_rank"] / hidden_dim)
        out["effective_rank_over_max_possible"].append(m["effective_rank"] / ceiling)
        out["spectral_entropy"].append(m["spectral_entropy"])
        out["pc1_fraction"].append(m["pc1_fraction"])
        out["numerical_rank"].append(m["numerical_rank"])
        out["spectrum"][name] = m["spectrum"]
    if n_subsamples:
        out["subsample"] = {name: subsample_metrics(R, n_subsamples=n_subsamples, frac=frac,
                                                    seed=seed)
                            for name, R in zip(layer_names, reps)}
        out["subsample_protocol"] = {"method": "without_replacement",
                                     "n_subsamples": int(n_subsamples), "frac": float(frac),
                                     "seed": int(seed)}
    return out
