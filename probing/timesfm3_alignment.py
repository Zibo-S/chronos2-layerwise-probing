"""Frozen TimesFM-3: LINEAR REPRESENTATION ALIGNMENT  h_l -> h_L20 -> the frozen native head.

The fourth analysis on the paper7 windows, and the one that separates "the information is
there" from "the information is in the right coordinate system":

    learned probe        B_l : h_l -> y            fits the FUTURE.  Is the forecast linearly
                                                   DECODABLE from h_l?
    frozen native head   W_native h_l              fits NOTHING.     Is h_l already in the
                                                   pretrained readout's basis?
    THIS analysis        A_l : h_l -> h_L20        fits the FINAL REPRESENTATION, never the
                         then W_native (A_l h_l + b_l)              future. Can a single affine
                                                   map put h_l into that basis?

    identity / direct native head
            v
    orthogonal alignment (Procrustes, optional)
            v
    general ridge linear alignment          <- this module
            v
    free linear forecasting probe

THE OBJECTIVE IS TARGET-FREE, AND THAT IS THE WHOLE POINT
--------------------------------------------------------
TimesFM-3's native head is itself linear, W_native : R^1280 -> R^576. So an adapter trained on
FORECAST loss would only ever produce another effective linear forecasting map W_native A_l --
i.e. a reparameterized copy of the learned probe, which answers nothing new. The adapter here is
therefore fit by

    min_{A,b}  || X_l A + 1 b^T - H_20 ||_F^2 + lambda ||A||_F^2      (bias NOT regularized)

and lambda is selected on VALIDATION REPRESENTATION error alone. No future value, no forecast
loss and no quantile ever enters the fit or the selection -- structurally, not by convention:
the fitting path loads representations through ``probing.timesfm3_geometry.load_last_token_reps``,
which verifies the cache's mu/sd/native entries and then DISCARDS them, and
``assert_no_forecast_targets`` refuses any array shaped like a target or a quantile forecast.

NOT the Chronos-2 adapter. ``probing/native_head_adapter.py`` trains Linear(768,768) on the
FORECAST loss into a NONLINEAR ResidualBlock head; its own docstring parks
``min_A ||RMS(A h_l) - h_L12||^2`` as "a PARKED future direction, not built here". This module is
that parked direction, for TimesFM-3, where the native head happens to be linear. The two
objectives are never mixed, in code or in the writeup.

SOLVER
------
N = 1394 train windows against d = 1280, so A has 1.64M coefficients on barely more rows than
columns: ridge is mandatory and conditioning must be watched. One economy SVD per layer,

    Xc = U S V^T          A_lambda = V diag(s / (s^2 + lambda)) U^T Yc       b = muY - muX A

makes every lambda a re-weighting of ONE decomposition -- no refit, no iterative solver, no SGD,
and the whole grid costs one SVD plus a handful of matmuls. Everything is float64. Reported with
it: the spectrum's scale (so the absolute lambda grid is interpretable), the condition number and
the effective degrees of freedom  df(lambda) = sum_i s_i^2/(s_i^2 + lambda).

THE ENDPOINT, AND WHY THERE ARE TWO BASELINES
---------------------------------------------
A_20 = I, b_20 = 0: L20 is the reference, not a fitted layer. Its forecast is the frozen head
applied to the CACHED L20 representation through the SAME numerical path every aligned
representation takes -- that is the apples-to-apples endpoint for alignment
(``cached_L20_native_head``). decode()'s own forecast is reported separately as the true model
baseline (``official_native_decode``), and the two are recorded together with their difference.
They are NOT required to be bit-identical: the head applied to a (N, 1280) matrix accumulates
differently than inside decode()'s (b, 1, 18, 1280) pass, a ~1e-6 relative effect measured by the
native-head transfer run's own slice-order control. A synthesized representation A h + b has no
full-sequence form, so the cached path is not a shortcut here -- it is the only possible path,
and the endpoint is defined to travel it too.
"""

from __future__ import annotations

import numpy as np

from probing.timesfm3_geometry import (SELECTED_TOKEN_INDEX, load_last_token_reps,
                                       window_identity_hash)
from probing.timesfm3_last_token import (DEFAULT_CHECKPOINT, LAST_LAYER, LAYER_NAMES,
                                         MODEL_DIMS, NUM_LAYERS, NUM_QUANTILES,
                                         cache_metadata, cache_root, read_cache)

__all__ = ["ALIGNMENT_VERSION", "RIDGE_GRID", "ALIGNMENTS", "SOLVE_DTYPE",
           "assert_no_forecast_targets", "spectral_scale_report", "RidgeAlignmentSolver",
           "procrustes_alignment", "identity_alignment", "apply_alignment",
           "representation_metrics", "select_lambda", "fit_layer_alignment",
           "permuted_rows", "mean_baseline_metrics", "frozen_head_forecast",
           "load_native_context", "load_last_token_reps", "window_identity_hash",
           "LAYER_NAMES", "NUM_LAYERS", "LAST_LAYER", "MODEL_DIMS", "SELECTED_TOKEN_INDEX"]

ALIGNMENT_VERSION = "tfm3-repr-align-v1"
SOLVE_DTYPE = np.float64          # every ridge / Procrustes / metric computation

# The lambda grid. Deliberately NOT the probe's WD_GRID_LAST_TOKEN: that parameterizes AdamW's
# DECOUPLED weight decay on a forecasting objective (a per-step multiplicative shrink of the
# weights, scale-free in the data), whereas lambda here is an ADDITIVE penalty competing with
# ||Xc A - Yc||_F^2 and therefore lives on the scale of the representation's own spectrum. The
# repo's other ridge convention (probes.ridge_regression_probe, alphas 0.1..100) does not
# transfer either: it standardizes features and has a 1-D target. So the grid below is the
# specification's log grid, extended upward to 1e6 because UNSTANDARDIZED 1280-d hidden states at
# N=1394 can put s_max^2 near ~1e6 -- and a grid that cannot reach the optimum silently reports
# the edge as the answer. Every run records where in the grid each layer landed and WARNS at an
# edge (the same convention as the probe's wd_at_grid_edge).
RIDGE_GRID = (1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
              1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6)
ALIGNMENTS = ("ridge", "procrustes")


# --------------------------------------------------------------------------- #
# leakage guard -- the contract this whole experiment rests on
# --------------------------------------------------------------------------- #
def assert_no_forecast_targets(horizon: int = 64, num_quantiles: int = NUM_QUANTILES,
                               **arrays) -> None:
    """Refuse any array that is shaped like a forecast target or a quantile forecast.

    The adapter objective is || X A + 1b^T - H_20 ||_F^2. Nothing with an H axis or a Q axis has
    any business reaching it, so this is checked on the ARRAYS rather than trusted to argument
    names. A (n, 1280) representation is accepted; a (n, 64) trajectory, a (n, 64, 9) forecast
    and anything rank-3 are rejected with the name of the offending argument.
    """
    for name, a in arrays.items():
        if a is None:
            continue
        a = np.asarray(a)
        if a.ndim == 3:
            raise ValueError(
                f"{name}: rank-3 array {a.shape} reached the representation-alignment fit. "
                f"A (n, {horizon}, {num_quantiles}) quantile forecast is a FORECAST TARGET; this "
                "adapter is trained only to reconstruct h_L20 and must never see one.")
        if a.ndim != 2:
            raise ValueError(f"{name}: expected a 2-D (n, {MODEL_DIMS}) representation, "
                             f"got {a.shape}")
        if a.shape[1] != MODEL_DIMS:
            raise ValueError(
                f"{name}: second axis is {a.shape[1]}, not the hidden width {MODEL_DIMS}"
                + (f" -- that is the horizon H={horizon}, i.e. a forecast TRAJECTORY. The "
                   "adapter objective is target-free." if a.shape[1] == horizon else ""))


# --------------------------------------------------------------------------- #
# conditioning diagnostics -- so the absolute lambda grid is interpretable
# --------------------------------------------------------------------------- #
def spectral_scale_report(s, n: int, d: int, grid=RIDGE_GRID) -> dict:
    """What the singular values of the CENTERED train representation actually are.

    ``lambda`` is added to s^2, so a grid is only meaningful next to the s^2 it competes with.
    Recorded per layer and printed once per dataset; it is what justifies (or condemns) the grid.
    """
    s = np.asarray(s, SOLVE_DTYPE)
    s2 = s ** 2
    tot = float(s2.sum())
    if tot > 0:                                   # exp(-sum p log p) on the squared spectrum,
        pr = s2 / tot                             # the repo's spectral_metrics convention
        erank = float(np.exp(-(pr * np.log(pr + 1e-300)).sum()))
    else:
        erank = float("nan")
    return {"n_train": int(n), "d": int(d), "rank_max": int(min(n - 1, d)),
            "s_max": float(s.max()), "s_min": float(s.min()),
            "s_max_squared": float(s2.max()), "s_min_squared": float(s2.min()),
            "mean_s_squared": float(s2.mean()),
            "trace_XtX": tot,
            "condition_number": float(s.max() / s.min()) if s.min() > 0 else float("inf"),
            "spectral_effective_rank": erank,
            "grid_min_over_s_max_squared": float(min(grid) / s2.max()) if s2.max() > 0 else None,
            "grid_max_over_s_max_squared": float(max(grid) / s2.max()) if s2.max() > 0 else None,
            "note": "lambda is ADDED to s^2; compare the grid against s_max_squared and "
                    "mean_s_squared. A selected lambda far below s_min_squared is effectively "
                    "unregularized; far above s_max_squared is effectively the mean predictor."}


# --------------------------------------------------------------------------- #
# ridge, solved once per layer for the WHOLE grid
# --------------------------------------------------------------------------- #
class RidgeAlignmentSolver:
    """Multi-output affine ridge  X A + 1 b^T ~ Y, every lambda from ONE economy SVD.

        Xc = X - muX,  Yc = Y - muY
        A_lambda = argmin_A ||Xc A - Yc||_F^2 + lambda ||A||_F^2
                 = V diag(s / (s^2 + lambda)) U^T Yc        [Xc = U diag(s) V^T]
        b_lambda = muY - muX A_lambda                        (the bias is NEVER penalized)

    Centering is what makes the unpenalized bias exact: for any A the optimal b is
    muY - muX A, and substituting it back leaves exactly the centered problem above. So this is
    not an approximation of the stated objective -- it IS the stated objective.

    float64 throughout. ``lambda <= 0`` is refused: at N ~ d the unregularized solution is the
    thing the specification excludes as a headline, and a pseudo-inverse silently substituted for
    it would be a different estimator reported under the same name.
    """

    def __init__(self, X, Y, *, check_targets: bool = True):
        if check_targets:
            assert_no_forecast_targets(X_train=X, Y_train_representation=Y)
        X = np.asarray(X, SOLVE_DTYPE)
        Y = np.asarray(Y, SOLVE_DTYPE)
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows, Y has {Y.shape[0]} -- the adapter is "
                             "fit on MATCHED windows (row i of both is the same window)")
        if X.shape[0] < 2:
            raise ValueError("at least 2 training rows are needed to centre and solve")
        self.n, self.d = X.shape
        self.d_out = Y.shape[1]
        self.muX = X.mean(axis=0)
        self.muY = Y.mean(axis=0)
        Xc = X - self.muX
        self.Yc = Y - self.muY
        # full_matrices=False -> U (n, r), s (r,), Vt (r, d) with r = min(n, d)
        self.U, self.s, self.Vt = np.linalg.svd(Xc, full_matrices=False)
        self.UtY = self.U.T @ self.Yc                     # (r, d_out), computed ONCE
        self.ss = self.s ** 2
        self.ss_tot_Y = float((self.Yc ** 2).sum())
        self.spectrum = spectral_scale_report(self.s, self.n, self.d)

    # -- internals -------------------------------------------------------- #
    def _filter(self, lam: float) -> np.ndarray:
        lam = float(lam)
        if not np.isfinite(lam) or lam <= 0.0:
            raise ValueError(
                f"ridge lambda must be finite and > 0, got {lam!r}. Unregularized least squares "
                "is excluded by specification at N ~ d, and a pseudo-inverse is a DIFFERENT "
                "estimator -- it will not be reported under the ridge name.")
        return self.s / (self.ss + lam)

    def coefficients(self, lam: float):
        """(A, b) for one lambda. A is (d, d_out); b is (d_out,). float64."""
        A = self.Vt.T @ (self._filter(lam)[:, None] * self.UtY)
        return A, self.muY - self.muX @ A

    def predict(self, X_new, lam: float) -> np.ndarray:
        """(X_new - muX) A_lambda + muY, without ever forming A.

        (X_new - muX) V is computed once per split and re-weighted per lambda, so a 15-point grid
        costs one (n_new, r) projection instead of 15 (d, d) matrix products.
        """
        G = (np.asarray(X_new, SOLVE_DTYPE) - self.muX) @ self.Vt.T     # (n_new, r)
        return G @ (self._filter(lam)[:, None] * self.UtY) + self.muY

    def projector(self, X_new) -> np.ndarray:
        """The reusable (n_new, r) projection, for scanning a grid on one split."""
        return (np.asarray(X_new, SOLVE_DTYPE) - self.muX) @ self.Vt.T

    def predict_from_projection(self, G, lam: float) -> np.ndarray:
        return np.asarray(G, SOLVE_DTYPE) @ (self._filter(lam)[:, None] * self.UtY) + self.muY

    def effective_dof(self, lam: float) -> float:
        """df(lambda) = sum_i s_i^2 / (s_i^2 + lambda): how many directions survive the penalty."""
        return float((self.ss / (self.ss + float(lam))).sum())

    def with_target(self, Y_new) -> "RidgeAlignmentSolver":
        """A solver for the SAME X against a different target, reusing the existing SVD.

        Only ``U^T Yc`` depends on the target, so the permutation control costs one matmul
        instead of a second decomposition -- and, more importantly, it is provably the SAME
        design matrix and the SAME spectrum, so the control differs from the real fit in exactly
        one thing: the window-to-window correspondence.
        """
        Y_new = np.asarray(Y_new, SOLVE_DTYPE)
        if Y_new.shape != (self.n, self.d_out):
            raise ValueError(f"replacement target must be ({self.n}, {self.d_out}), got "
                             f"{Y_new.shape}")
        clone = object.__new__(RidgeAlignmentSolver)
        clone.__dict__.update(self.__dict__)
        clone.muY = Y_new.mean(axis=0)
        clone.Yc = Y_new - clone.muY
        clone.UtY = clone.U.T @ clone.Yc
        clone.ss_tot_Y = float((clone.Yc ** 2).sum())
        return clone

    def as_dict(self) -> dict:
        return {"solver": "economy SVD of the centred train representation; every lambda is a "
                          "re-weighting of ONE decomposition (no refit, no iterative solver)",
                "dtype": str(np.dtype(SOLVE_DTYPE)), "n_train": self.n,
                "d_in": self.d, "d_out": self.d_out, "svd_rank": int(self.s.size),
                "bias_regularized": False, "centred": True, "spectrum": self.spectrum}


def procrustes_alignment(X, Y, *, check_targets: bool = True) -> dict:
    """Orthogonal alignment: min_{R^T R = I} ||Xc R - Yc||_F^2, from the SVD of Xc^T Yc.

    The stricter control: is a pure rotation/reflection of h_l enough, or does the native head
    need the general linear map's rescaling too? Same centering and the same unpenalized bias as
    the ridge path, so the two are read on one axis.
    """
    if check_targets:
        assert_no_forecast_targets(X_train=X, Y_train_representation=Y)
    X = np.asarray(X, SOLVE_DTYPE)
    Y = np.asarray(Y, SOLVE_DTYPE)
    if X.shape != Y.shape:
        raise ValueError(f"orthogonal alignment needs square-compatible representations, got "
                         f"{X.shape} -> {Y.shape}")
    muX, muY = X.mean(axis=0), Y.mean(axis=0)
    M = (X - muX).T @ (Y - muY)
    U, sv, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    orth = float(np.abs(R.T @ R - np.eye(R.shape[0])).max())
    if orth > 1e-8:
        raise RuntimeError(f"Procrustes solution is not orthogonal: max|R^T R - I| = {orth:.3e}")
    return {"A": R, "b": muY - muX @ R, "muX": muX, "muY": muY,
            "orthogonality_max_abs_error": orth,
            "determinant": float(np.linalg.det(R)),
            "nuclear_norm": float(sv.sum()), "lambda": None,
            "alignment": "procrustes"}


def identity_alignment(d: int = MODEL_DIMS) -> dict:
    """A = I, b = 0. L20's adapter is NOT fit -- it is the reference endpoint."""
    return {"A": np.eye(d, dtype=SOLVE_DTYPE), "b": np.zeros(d, SOLVE_DTYPE),
            "muX": np.zeros(d, SOLVE_DTYPE), "muY": np.zeros(d, SOLVE_DTYPE),
            "lambda": None, "alignment": "identity", "fitted": False}


def apply_alignment(A, b, X) -> np.ndarray:
    """X A + b, in float64. The one place a representation is mapped."""
    return np.asarray(X, SOLVE_DTYPE) @ np.asarray(A, SOLVE_DTYPE) + np.asarray(b, SOLVE_DTYPE)


# --------------------------------------------------------------------------- #
# representation metrics
# --------------------------------------------------------------------------- #
def representation_metrics(Yhat, Y, *, per_dimension: bool = True) -> dict:
    """MSE and R^2 of a reconstruction of H_20, plus per-dimension diagnostics.

    PRIMARY metric, exactly as specified -- global, variance-weighted over the whole matrix:

        MSE = ||Yhat - Y||_F^2 / (N d)
        R^2 = 1 - ||Yhat - Y||_F^2 / ||Y - mean_0(Y)||_F^2

    The denominator uses the EVALUATION split's own column means, so R^2 = 0 is exactly "no
    better than the best constant vector on this split" and R^2 < 0 is reported as-is, never
    clipped. Per-dimension R^2 (same formula, one hidden unit at a time) is summarized as
    median/IQR as a DIAGNOSTIC: it answers "is the reconstruction uniform across the 1280
    directions or carried by a few high-variance ones", which the Frobenius number cannot. It
    never replaces the global metric.
    """
    Yhat = np.asarray(Yhat, SOLVE_DTYPE)
    Y = np.asarray(Y, SOLVE_DTYPE)
    if Yhat.shape != Y.shape:
        raise ValueError(f"reconstruction {Yhat.shape} does not match the target {Y.shape}")
    n, d = Y.shape
    resid = Yhat - Y
    ss_res_j = (resid ** 2).sum(axis=0)
    ss_tot_j = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
    ss_res, ss_tot = float(ss_res_j.sum()), float(ss_tot_j.sum())
    out = {"n": int(n), "d": int(d),
           "mse": ss_res / (n * d),
           "rmse": float(np.sqrt(ss_res / (n * d))),
           "sum_squared_error": ss_res, "total_sum_squares": ss_tot,
           "r2": (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
           "relative_frobenius_error": (float(np.sqrt(ss_res / ss_tot)) if ss_tot > 0
                                        else float("nan")),
           "max_abs_error": float(np.abs(resid).max())}
    if per_dimension:
        ok = ss_tot_j > 0
        r2j = np.full(d, np.nan, SOLVE_DTYPE)
        r2j[ok] = 1.0 - ss_res_j[ok] / ss_tot_j[ok]
        fin = r2j[np.isfinite(r2j)]
        out["per_dimension_r2"] = {
            "median": float(np.median(fin)) if fin.size else float("nan"),
            "q25": float(np.percentile(fin, 25)) if fin.size else float("nan"),
            "q75": float(np.percentile(fin, 75)) if fin.size else float("nan"),
            "min": float(fin.min()) if fin.size else float("nan"),
            "max": float(fin.max()) if fin.size else float("nan"),
            "fraction_positive": float((fin > 0).mean()) if fin.size else float("nan"),
            "n_degenerate_dimensions": int((~ok).sum()),
            "note": "DIAGNOSTIC ONLY -- the primary metric is the global variance-weighted r2"}
    return out


def mean_baseline_metrics(Y_eval, Y_train_mean) -> dict:
    """CONTROL: predict the TRAIN mean of h_L20 for every window.

    Scored by the same ``representation_metrics``, whose denominator is the EVALUATION split's
    own mean -- so this returns R^2 = 0 exactly when the two means coincide and a small NEGATIVE
    value otherwise. That small negative number is the honest one (a train-fit constant cannot
    beat the evaluation split's own optimum) and it is not repaired.
    """
    Y_eval = np.asarray(Y_eval, SOLVE_DTYPE)
    pred = np.broadcast_to(np.asarray(Y_train_mean, SOLVE_DTYPE), Y_eval.shape)
    m = representation_metrics(pred, Y_eval, per_dimension=False)
    m["predictor"] = "constant = mean of the TRAIN h_L20"
    return m


def permuted_rows(Y, seed: int = 0) -> np.ndarray:
    """CONTROL: break the window-to-window correspondence between h_l and h_L20.

    A derangement-ish shuffle of the target rows. If a layer's R^2 survives this, the alignment
    is fitting marginal structure, not the correspondence -- so it must collapse.
    """
    Y = np.asarray(Y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Y))
    if len(Y) > 1:                                   # never return the identity permutation
        fixed = np.flatnonzero(idx == np.arange(len(Y)))
        for i in fixed:
            j = int(rng.integers(0, len(Y)))
            idx[i], idx[j] = idx[j], idx[i]
    return Y[idx]


# --------------------------------------------------------------------------- #
# lambda selection -- VALIDATION REPRESENTATION ERROR ONLY
# --------------------------------------------------------------------------- #
def select_lambda(solver: RidgeAlignmentSolver, X_val, Y_val, grid=RIDGE_GRID,
                  criterion: str = "val_representation_mse",
                  check_targets: bool = True) -> dict:
    """Pick lambda by validation reconstruction of h_L20. No forecast quantity is admissible.

    ``criterion`` exists only to make the selection rule an explicit, recorded field; the two
    accepted values (representation MSE, representation R^2) are monotone transforms of each
    other on a FIXED validation split, so they always agree -- which is exactly the point of
    offering both and nothing else. Anything forecast-shaped is rejected upstream by
    ``assert_no_forecast_targets``, and the adapter is NOT refit after selection: the candidate
    fit on FULL train at the chosen lambda is the one evaluated, matching
    ``probes.fit_quantile_probe_explicit_val``'s explicit-val contract.
    """
    if criterion not in ("val_representation_mse", "val_representation_r2"):
        raise ValueError(
            f"lambda selection criterion {criterion!r} is not allowed. The adapter is trained "
            "to reconstruct h_L20, so lambda is chosen on VALIDATION REPRESENTATION error only; "
            "selecting on forecast loss would leak the downstream task into a deliberately "
            "target-free objective.")
    if check_targets:
        assert_no_forecast_targets(X_val=X_val, Y_val_representation=Y_val)
    grid = [float(g) for g in grid]
    if not grid:
        raise ValueError("the lambda grid is empty")
    G = solver.projector(X_val)
    Yv = np.asarray(Y_val, SOLVE_DTYPE)
    rows, nonfinite = [], []
    for lam in grid:
        pred = solver.predict_from_projection(G, lam)
        if not np.isfinite(pred).all():
            nonfinite.append(lam)
            continue
        m = representation_metrics(pred, Yv, per_dimension=False)
        rows.append({"lambda": lam, "val_representation_mse": m["mse"],
                     "val_representation_r2": m["r2"],
                     "effective_dof": solver.effective_dof(lam)})
    if not rows:
        raise RuntimeError(
            f"every lambda in {grid} produced a non-finite validation reconstruction. The "
            "centred train representation is degenerate; no adapter can be selected.")
    best = (min(rows, key=lambda r: r["val_representation_mse"])
            if criterion == "val_representation_mse"
            else max(rows, key=lambda r: r["val_representation_r2"]))
    lam = best["lambda"]
    return {"lambda": lam, "criterion": criterion, "grid": grid, "candidates": rows,
            "nonfinite_candidates": nonfinite,
            "at_grid_edge": bool(lam == min(grid) or lam == max(grid)),
            "at_grid_min": bool(lam == min(grid)), "at_grid_max": bool(lam == max(grid)),
            "effective_dof": best["effective_dof"],
            "val_representation_mse": best["val_representation_mse"],
            "val_representation_r2": best["val_representation_r2"],
            "refit_after_selection": False,
            "selection_inputs": "validation representations only (X_val, H_20^val); no future "
                                "value, forecast, quantile or loss enters this function"}


def fit_layer_alignment(X_train, Y_train, X_val, Y_val, X_test, Y_test, *,
                        alignment: str = "ridge", grid=RIDGE_GRID,
                        criterion: str = "val_representation_mse",
                        per_dimension: bool = True, check_targets: bool = True) -> dict:
    """One layer, end to end: fit on TRAIN, select on VAL, evaluate ONCE on TEST.

    Every Y here is a REPRESENTATION (h_L20), never a forecast target -- enforced on the arrays.
    Returns the adapter (A, b), the selection record and the train/val/test representation
    metrics. The caller pushes ``A X + b`` through the frozen native head; nothing in this
    function knows that a forecasting head exists.
    """
    if check_targets:
        assert_no_forecast_targets(X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val,
                                   X_test=X_test, Y_test=Y_test)
    if alignment == "ridge":
        solver = RidgeAlignmentSolver(X_train, Y_train, check_targets=False)
        sel = select_lambda(solver, X_val, Y_val, grid=grid, criterion=criterion,
                            check_targets=False)
        A, b = solver.coefficients(sel["lambda"])
        rec = {"alignment": "ridge", "lambda": sel["lambda"], "selection": sel,
               "solver": solver.as_dict(), "muX": solver.muX, "muY": solver.muY,
               "effective_dof": sel["effective_dof"], "fitted": True}
    elif alignment == "procrustes":
        pr = procrustes_alignment(X_train, Y_train, check_targets=False)
        A, b = pr["A"], pr["b"]
        rec = {"alignment": "procrustes", "lambda": None,
               "selection": {"criterion": "none -- the orthogonal solution is closed-form and "
                                          "has no hyperparameter",
                             "refit_after_selection": False},
               "orthogonality_max_abs_error": pr["orthogonality_max_abs_error"],
               "determinant": pr["determinant"], "muX": pr["muX"], "muY": pr["muY"],
               "fitted": True}
    elif alignment == "identity":
        idp = identity_alignment(np.asarray(X_train).shape[1])
        A, b, rec = idp["A"], idp["b"], {"alignment": "identity", "lambda": None,
                                         "selection": {"criterion": "none -- L20 is the "
                                                                   "reference endpoint, not a "
                                                                   "fitted layer"},
                                         "fitted": False}
    else:
        raise ValueError(f"unknown alignment {alignment!r}; known: {ALIGNMENTS + ('identity',)}")

    rec["A"], rec["b"] = A, b
    rec["train"] = representation_metrics(apply_alignment(A, b, X_train), Y_train,
                                          per_dimension=False)
    rec["val"] = representation_metrics(apply_alignment(A, b, X_val), Y_val,
                                        per_dimension=False)
    rec["test"] = representation_metrics(apply_alignment(A, b, X_test), Y_test,
                                         per_dimension=per_dimension)
    rec["coefficient_norms"] = {"frobenius_A": float(np.linalg.norm(A)),
                                "max_abs_A": float(np.abs(A).max()),
                                "frobenius_b": float(np.linalg.norm(b)),
                                "n_coefficients": int(A.size + b.size)}
    return rec


# --------------------------------------------------------------------------- #
# the frozen native head, applied to a SYNTHESIZED (N, 1280) representation
# --------------------------------------------------------------------------- #
def frozen_head_forecast(model, states, mu, sd, trend, geom, *, batch_size: int = 512,
                         device=None) -> np.ndarray:
    """(N, 1280) -> W_native -> decode()'s EXACT inverse path -> (N, H, Q) raw-unit forecast.

    The step sequence is byte-for-byte the one in ``timesfm3_last_token.verify_native_head`` and
    ``run_timesfm3_native_head_transfer._head_stages``:

        output_head -> revin(reverse, token-15 stats) -> clamp(+-value_clip)
                    -> stitch_patches -> + context trend

    with ONE unavoidable difference: those functions feed the head the full (b, 1, n_tokens, d)
    hidden state and slice token 15 afterwards, because decode() does. An aligned representation
    A h_l + b_l is SYNTHESIZED and has no token sequence to sit in, so the head is applied to a
    (b, 1, 1, d) tensor here. That changes the matmul shape and hence the accumulation order --
    a ~1e-6 relative effect on TF32 hardware, measured by the native-head run's own slice-order
    control. It is therefore applied to EVERY curve in this experiment, including the L20
    endpoint, so the comparison is exact even though the endpoint is not bit-identical to
    decode(). ``model.output_head`` is used verbatim and never modified.
    """
    import torch
    from timesfm3.torch import util as tfm_util

    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != model.output_head.in_features:
        raise ValueError(f"the head consumes (N, {model.output_head.in_features}) token states, "
                         f"got {states.shape}")
    n, Q = len(states), model.num_quantiles
    P, opl = model.input_patch_len, model.output_patch_len
    if model.output_head.out_features != opl * Q:
        raise RuntimeError(f"native head is {model.output_head.in_features} -> "
                           f"{model.output_head.out_features}, expected {opl}*{Q}={opl * Q}")
    for name, p in model.output_head.named_parameters():
        if p.requires_grad:
            raise RuntimeError(f"native head parameter {name} has requires_grad=True; this "
                               "analysis applies a FROZEN head only")
    mu = np.asarray(mu, np.float64).reshape(-1)
    sd = np.asarray(sd, np.float64).reshape(-1)
    trend = np.asarray(trend, np.float64)
    if mu.shape != (n,) or sd.shape != (n,) or trend.shape != (n, geom.H):
        raise ValueError(f"mu/sd must be ({n},) and trend ({n}, {geom.H}); got {mu.shape}, "
                         f"{sd.shape}, {trend.shape}")

    w = model.output_head.weight
    dev = device or w.device
    out = np.zeros((n, geom.H, Q), np.float64)
    with torch.no_grad():
        for s0 in range(0, n, batch_size):
            e = min(s0 + batch_size, n)
            b = e - s0
            h = torch.as_tensor(np.ascontiguousarray(states[s0:e]), dtype=w.dtype,
                                device=dev).reshape(b, 1, 1, -1)
            raw = model.output_head(h)                                    # (b, 1, 1, opl*Q)
            m = torch.as_tensor(mu[s0:e], dtype=w.dtype, device=dev).reshape(b, 1, 1)
            v = torch.as_tensor(sd[s0:e], dtype=w.dtype, device=dev).reshape(b, 1, 1)
            den = tfm_util.revin(raw, m, v, reverse=True)
            den = torch.clamp(den, -model.value_clip, model.value_clip)
            view5 = den.reshape(b, 1, 1, opl, Q)[:, :, :, :geom.extract_len, :]
            rec = tfm_util.stitch_patches(view5, P)[:, :, :geom.H, :][:, 0]
            out[s0:e] = rec.float().cpu().numpy().astype(np.float64)
    return out + trend[:, :, None]


# --------------------------------------------------------------------------- #
# cache access -- representations for fitting, mu/sd/native for scoring, kept apart
# --------------------------------------------------------------------------- #
def load_native_context(tag: str, split: str, X, *, cache_dir, geom, checkpoint=None,
                        suite: str = "paper7", seed: int = 0, detrend: bool = True,
                        feature_dtype=np.float32, layers=(LAST_LAYER,),
                        ignore_timesfm_version: bool = False) -> dict:
    """The SCORING side of the same verified cache read: mu, sd and decode()'s own forecast.

    Deliberately a separate function from ``load_last_token_reps`` (which returns
    representations and discards exactly these fields). The adapter-fitting path calls that one;
    only the TEST-split evaluation calls this one. The split is structural: no code path can
    hand a native forecast to the adapter fit.
    """
    checkpoint = checkpoint or DEFAULT_CHECKPOINT
    layers = sorted(set(int(l) for l in layers))
    X = np.asarray(X, np.float32)
    meta = cache_metadata(tag, split, geom, checkpoint=checkpoint, detrend=detrend,
                          layers=layers, seed=seed, feature_dtype=feature_dtype, suite=suite,
                          n_windows=len(X))
    if ignore_timesfm_version:
        meta.pop("timesfm_version", None)
    root = cache_root(cache_dir, tag, split, geom, detrend, suite)
    hit = read_cache(root, meta, X, layers)
    if hit is None:
        raise FileNotFoundError(
            f"no usable last-token cache at {root} for {tag}/{split}. This experiment is "
            "CACHE-ONLY for representations; run the paper7 Q=9 probe driver first.")
    nat = np.asarray(hit["native"])
    if nat.shape != (len(X), geom.H, NUM_QUANTILES):
        raise RuntimeError(f"cached native forecast is {nat.shape}, expected "
                           f"({len(X)}, {geom.H}, {NUM_QUANTILES})")
    return {"mu": np.asarray(hit["mu"], np.float64), "sd": np.asarray(hit["sd"], np.float64),
            "native": nat, "native_check": hit["native_check"], "sorting": hit["sorting"],
            "cache_root": str(root)}
