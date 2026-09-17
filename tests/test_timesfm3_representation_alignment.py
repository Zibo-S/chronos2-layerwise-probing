"""Contract tests for the TimesFM-3 LINEAR REPRESENTATION ALIGNMENT experiment.

    python -m tests.test_timesfm3_representation_alignment               # no model, no GPU
    python -m tests.test_timesfm3_representation_alignment --with-model  # + checkpoint checks

The model-free block is login-node safe: synthetic arrays and temporary caches only, torch
pinned to 2 threads, a few seconds. The --with-model block loads the TimesFM-3 checkpoint and
belongs in salloc/sbatch.

Numbered to the specification's list.

     1 X and Y are (N, 1280)                     11 the mean predictor gives R^2 ~ 0
     2 exactly matching window identities         12 a permuted correspondence gives low R^2
     3 only token-15 representations are used     13 the frozen head's checksum is unchanged
     4 train/val/test cannot cross-load           14 the aligned L20 equals the cached endpoint
     5 ridge reproduces a known linear map        15 all 20 non-final layers + the L20 identity
     6 the bias is recovered                      16 direct / aligned / probe align BY LAYER
     7 lambda is selected on VAL representation   17 the paper7 seven-dataset roster is preserved
       error only                                 18 no backbone forward pass is needed
     8 forecast targets are not required          19 heavy adapter matrices go to $SCRATCH
     9 the representation R^2 formula is correct  20 final figures go inside the repository
    10 a perfect synthetic map gives R^2 ~ 1

    21 the identity endpoint is NOT fitted        24 Procrustes is orthogonal and is the WEAKER
    22 the selected fit is NOT refit                 hypothesis class
    23 the solve is float64                       25 lambda <= 0 is refused
    26 the recovery metric and its zero-denominator guard

NOTHING here is weakened to make a run pass.
"""

from __future__ import annotations

import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)          # never grab every core of a shared login node

from probing.timesfm3_alignment import (ALIGNMENT_VERSION, ALIGNMENTS,  # noqa: E402
                                        RIDGE_GRID, SOLVE_DTYPE, RidgeAlignmentSolver,
                                        apply_alignment, assert_no_forecast_targets,
                                        fit_layer_alignment, frozen_head_forecast,
                                        identity_alignment, load_last_token_reps,
                                        load_native_context, mean_baseline_metrics,
                                        permuted_rows, procrustes_alignment,
                                        representation_metrics, select_lambda,
                                        spectral_scale_report)
from probing.timesfm3_geometry import repo_root  # noqa: E402
from probing.timesfm3_last_token import (DEFAULT_CHECKPOINT, LAST_LAYER,  # noqa: E402
                                         LAYER_NAMES, MODEL_DIMS, NUM_LAYERS, NUM_QUANTILES,
                                         LastTokenGeometry, cache_metadata, cache_root)

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fake_cache(cache_dir, tag, split, X, geom, layers, *, rank3=False, suite="paper7",
                checkpoint=DEFAULT_CHECKPOINT, seed=0, detrend=True, dtype=np.float32,
                meta_override=None, seed_offset=100):
    """A minimal on-disk twin of a real last-token cache, built with the REAL metadata writer."""
    meta = cache_metadata(tag, split, geom, checkpoint=checkpoint, detrend=detrend,
                          layers=layers, seed=seed, feature_dtype=dtype, suite=suite,
                          n_windows=len(X))
    meta.update(meta_override or {})
    root = cache_root(cache_dir, tag, split, geom, detrend, suite)
    root.mkdir(parents=True, exist_ok=True)
    n = len(X)
    for L in layers:
        shape = (n, 16, MODEL_DIMS) if rank3 else (n, MODEL_DIMS)
        np.save(root / f"L{L:02d}.npy",
                np.random.default_rng(seed_offset + L).normal(size=shape).astype(dtype))
    np.savez(root / "meta.npz", meta=json.dumps(meta), ctx_tail=X[:, -8:],
             mu=np.zeros(n), sd=np.ones(n),
             native=np.zeros((n, geom.H, NUM_QUANTILES), np.float32),
             native_check=json.dumps({}), dtype_check=json.dumps({}), sorting=json.dumps({}))
    return root


def _ctx(n, C=512, seed=7):
    return np.random.default_rng(seed).normal(size=(n, C)).astype(np.float32)


def _linear_problem(n=240, d=40, k=40, noise=0.0, seed=1):
    """Y = X W + c (+ noise). The ground truth a ridge fit must recover at small lambda."""
    r = np.random.default_rng(seed)
    X = r.normal(size=(n, d))
    W = r.normal(size=(d, k)) / np.sqrt(d)
    c = r.normal(size=k) * 3.0
    Y = X @ W + c + (noise * r.normal(size=(n, k)) if noise else 0.0)
    return X, Y, W, c


class _StubModel:
    """Just enough of TimesFM-3 for the head PLUMBING: a real frozen Linear(1280, 576).

    This is NOT a stand-in for decode(): it validates shapes, dtype, batching, the frozen-head
    refusal and the identity-adapter bit-equality. Whether the inverse path reproduces the model
    is checked only under --with-model, against the real checkpoint.
    """

    def __init__(self, d=MODEL_DIMS, opl=64, Q=NUM_QUANTILES, seed=3):
        torch.manual_seed(seed)
        self.output_head = torch.nn.Linear(d, opl * Q)
        for p in self.output_head.parameters():
            p.requires_grad_(False)
        self.num_quantiles = Q
        self.input_patch_len = 32
        self.output_patch_len = opl
        self.value_clip = 1e9


def _ensure_tfm_util():
    """Use the installed timesfm3 util when present; otherwise a documented stub.

    ``frozen_head_forecast`` calls ``tfm_util.revin`` and ``tfm_util.stitch_patches``. On a
    machine without timesfm3 installed the plumbing contracts are still worth testing, so a stub
    implementing the documented single-patch semantics is injected -- and announced, so no reader
    can mistake a stub run for a validation of the native path.
    """
    try:
        from timesfm3.torch import util  # noqa: F401
        return "installed timesfm3.torch.util"
    except Exception:
        import types
        pkg = sys.modules.setdefault("timesfm3", types.ModuleType("timesfm3"))
        tor = sys.modules.setdefault("timesfm3.torch", types.ModuleType("timesfm3.torch"))
        util = types.ModuleType("timesfm3.torch.util")

        def revin(x, mu, sd, reverse=False):
            return x * sd[..., None] + mu[..., None] if reverse else (x - mu[..., None]) / sd[..., None]

        def stitch_patches(x5, P):
            if x5.shape[2] != 1:
                raise NotImplementedError("the stub only covers the single-patch layout (K=1)")
            return x5[:, :, 0]

        util.revin, util.stitch_patches = revin, stitch_patches
        tor.util = util
        pkg.torch = tor
        sys.modules["timesfm3.torch.util"] = util
        return "STUB timesfm3.torch.util (plumbing only -- NOT a native-path validation)"


# ------------------------------------------------------------------ 1, 3, 4, 2
def test_representation_matrices_and_cache_isolation():
    """(N, 1280) per point, token 15 only, and no split/suite/rank can cross-load."""
    g = LastTokenGeometry()
    with tempfile.TemporaryDirectory() as td:
        Xtr, Xva, Xte = _ctx(40, seed=1), _ctx(17, seed=2), _ctx(23, seed=3)
        layers = list(range(NUM_LAYERS))
        for s, X in (("train", Xtr), ("val", Xva), ("test", Xte)):
            _fake_cache(td, "m4_hourly", s, X, g, layers, seed_offset=100 + 50 * len(s))
        lo = dict(cache_dir=td, geom=g, layers=layers, suite="paper7")
        R = {s: load_last_token_reps("m4_hourly", s, X, **lo)
             for s, X in (("train", Xtr), ("val", Xva), ("test", Xte))}

        # 1: every representation point is (N, 1280)
        for s, X in (("train", Xtr), ("val", Xva), ("test", Xte)):
            assert len(R[s]["reps"]) == NUM_LAYERS == 21
            for a in R[s]["reps"]:
                assert a.shape == (len(X), MODEL_DIMS), a.shape
        print(f"     1         X and Y are (N, {MODEL_DIMS}) at all {NUM_LAYERS} points      OK")

        # 3: the cache metadata pins the readout token, and it is 15
        meta = cache_metadata("m4_hourly", "train", g, checkpoint=DEFAULT_CHECKPOINT,
                              detrend=True, layers=layers, seed=0, suite="paper7")
        assert meta["selected_token_index"] == 15 and meta["last_token_only"] is True
        assert meta["feature_shape_rank"] == 2
        print("     3         only token-15 (rank-2) representations are accepted       OK")

        # 2: window identity -- a cache built on other windows is REFUSED, and the three
        #    splits have three distinct identity hashes
        try:
            load_last_token_reps("m4_hourly", "train", _ctx(40, seed=99), **lo)
            raise AssertionError("a cache built on different windows was accepted")
        except RuntimeError as e:
            assert "do not match" in str(e), e
        h = {s: R[s]["window_identity_hash"] for s in R}
        assert len(set(h.values())) == 3, h
        print(f"     2         window identity is verified element-wise; 3 distinct hashes OK")

        # 4: train/val/test cannot cross-load (the split is in the metadata AND the path)
        for a, b in (("train", "val"), ("val", "test"), ("test", "train")):
            try:
                load_last_token_reps("m4_hourly", a, {"train": Xtr, "val": Xva,
                                                      "test": Xte}[b], **lo)
                raise AssertionError(f"{b}'s windows were accepted as {a}'s")
            except (RuntimeError, FileNotFoundError):
                pass
        print("     4         train / val / test cannot cross-load                      OK")

        # 3b: a rank-3 shared-origin cache is rejected on shape, even in the right directory
        with tempfile.TemporaryDirectory() as td3:
            _fake_cache(td3, "m4_hourly", "train", Xtr, g, layers, rank3=True)
            try:
                load_last_token_reps("m4_hourly", "train", Xtr, cache_dir=td3, geom=g,
                                     layers=layers, suite="paper7")
                raise AssertionError("a rank-3 (N, 16, 1280) cache was accepted")
            except RuntimeError as e:
                assert "rank 3" in str(e) or "rank-3" in str(e), e
        print("     3b        the (N, 16, 1280) shared-origin cache is REJECTED          OK")


# ------------------------------------------------------------------ 5, 6, 23, 25
def test_ridge_recovers_a_known_map():
    """A ridge fit at small lambda must reproduce a known (W, c), in float64."""
    X, Y, W, c = _linear_problem(n=240, d=40, k=40, noise=0.0)
    s = RidgeAlignmentSolver(X, Y, check_targets=False)

    # 23: everything is float64
    assert s.U.dtype == s.s.dtype == s.Vt.dtype == np.dtype(SOLVE_DTYPE)
    A, b = s.coefficients(1e-10)
    assert A.dtype == b.dtype == np.dtype(SOLVE_DTYPE)
    print(f"     23        solve dtype is {np.dtype(SOLVE_DTYPE).name}                                  OK")

    # 5: the coefficient matrix
    err = float(np.abs(A - W).max())
    assert err < 1e-6, err
    print(f"     5         ridge recovers the known map: max|A - W| {err:.2e}          OK")

    # 6: the unregularized bias
    berr = float(np.abs(b - c).max())
    assert berr < 1e-8, berr
    # and the two prediction forms agree exactly
    p1 = X @ A + b
    p2 = apply_alignment(A, b, X)
    assert np.allclose(p1, p2, rtol=0, atol=1e-12)
    print(f"     6         the bias is recovered: max|b - c| {berr:.2e}                 OK")

    # 6b: the bias is NOT regularized -- at an enormous lambda A -> 0 but b -> mean(Y)
    A0, b0 = s.coefficients(1e14)
    assert float(np.abs(A0).max()) < 1e-6, float(np.abs(A0).max())
    assert np.allclose(b0, Y.mean(axis=0), atol=1e-8)
    print("     6b        lambda -> inf drives A -> 0 and b -> mean(Y), never shrinking b OK")

    # 25: unregularized least squares is refused, not silently substituted
    for bad in (0.0, -1.0, float("nan"), float("inf")):
        try:
            s.coefficients(bad)
            raise AssertionError(f"lambda={bad} was accepted")
        except ValueError as e:
            assert "must be finite and > 0" in str(e), e
    print("     25        lambda <= 0 / non-finite is REFUSED                           OK")

    # the reported spectrum is the one that was solved with
    rep = spectral_scale_report(s.s, s.n, s.d)
    assert rep["condition_number"] >= 1.0 and rep["s_max_squared"] >= rep["mean_s_squared"]
    assert abs(s.effective_dof(1e-12) - min(s.n - 1, s.d)) < 1e-6
    print(f"     5b        df(lambda->0) = rank {s.effective_dof(1e-12):.2f}, condition "
          f"{rep['condition_number']:.1f}         OK")


# ------------------------------------------------------------------ 7, 8, 22
def test_lambda_selection_is_validation_representation_only():
    """lambda comes from VAL representation error. Forecast quantities cannot reach the fit."""
    # a problem where TRAIN error and VAL error disagree: noisy train, clean val
    Xtr, Ytr, W, c = _linear_problem(n=120, d=60, k=60, noise=2.0, seed=5)
    r = np.random.default_rng(6)
    Xva = r.normal(size=(80, 60)); Yva = Xva @ W + c
    Xte = r.normal(size=(90, 60)); Yte = Xte @ W + c

    solver = RidgeAlignmentSolver(Xtr, Ytr, check_targets=False)
    sel = select_lambda(solver, Xva, Yva, grid=RIDGE_GRID, check_targets=False)
    # the chosen lambda must be the argmin of the RECORDED val curve -- nothing else
    best = min(sel["candidates"], key=lambda x: x["val_representation_mse"])
    assert sel["lambda"] == best["lambda"], (sel["lambda"], best["lambda"])
    # TRAIN error would pick the grid minimum (it decreases monotonically as lambda -> 0 on the
    # data the fit saw); VAL error, with this much train noise, must not.
    tr_curve = {g: representation_metrics(solver.predict(Xtr, g), Ytr,
                                          per_dimension=False)["mse"] for g in RIDGE_GRID}
    assert tr_curve[min(RIDGE_GRID)] == min(tr_curve.values()), (
        f"this fixture no longer discriminates: train error is not minimized at the smallest "
        f"lambda ({tr_curve})")
    assert sel["lambda"] > min(RIDGE_GRID), (
        f"val selection returned the grid minimum {sel['lambda']}, which is exactly what TRAIN "
        f"error would choose; the selection is not discriminating (train MSE {tr_curve})")
    print(f"     7         lambda selected on VAL representation MSE = {sel['lambda']:g} "
          f"(> grid min)   OK")

    # 7b: a forecast-based criterion is refused by name
    try:
        select_lambda(RidgeAlignmentSolver(Xtr, Ytr, check_targets=False), Xva, Yva,
                      criterion="test_q9_loss", check_targets=False)
        raise AssertionError("a forecast criterion was accepted for lambda selection")
    except ValueError as e:
        assert "VALIDATION REPRESENTATION" in str(e), e
    print("     7b        a forecast-loss selection criterion is REFUSED               OK")

    # 8: anything shaped like a target or a quantile forecast is refused on the ARRAY
    n = 32
    for name, arr, why in (("traj", np.zeros((n, 64)), "horizon"),
                           ("forecast", np.zeros((n, 64, NUM_QUANTILES)), "rank-3"),
                           ("scalar", np.zeros(n), "rank-1")):
        try:
            assert_no_forecast_targets(**{name: arr})
            raise AssertionError(f"a {why} array was accepted by the adapter fit")
        except ValueError:
            pass
    assert_no_forecast_targets(ok=np.zeros((n, MODEL_DIMS)))       # the legitimate shape passes
    print("     8         (n,64) trajectories and (n,64,9) forecasts are REFUSED        OK")

    # 8b: the fitting entry points take no forecast argument at all
    for fn in (fit_layer_alignment, select_lambda, RidgeAlignmentSolver.__init__,
               procrustes_alignment):
        params = set(inspect.signature(fn).parameters)
        bad = {p for p in params if any(k in p.lower() for k in
                                        ("target_traj", "future", "quantile", "forecast",
                                         "y_traj", "pinball", "loss"))}
        assert not bad, f"{fn.__name__} exposes forecast-shaped parameters {bad}"
    # and the loader used for fitting returns no mu / sd / native
    g = LastTokenGeometry()
    with tempfile.TemporaryDirectory() as td:
        X = _ctx(24)
        _fake_cache(td, "m4_hourly", "train", X, g, [0, LAST_LAYER])
        got = load_last_token_reps("m4_hourly", "train", X, cache_dir=td, geom=g,
                                   layers=[0, LAST_LAYER], suite="paper7")
        leaked = {k for k in got if k in ("mu", "sd", "native", "targets", "native_check")}
        assert not leaked, f"the fitting loader leaked forecast quantities {leaked}"
        # the scoring side is a SEPARATE function, and it is the only one that returns them
        ctx = load_native_context("m4_hourly", "train", X, cache_dir=td, geom=g, suite="paper7",
                                  layers=[0, LAST_LAYER])
        assert {"mu", "sd", "native"} <= set(ctx)
    print("     8b        the fitting loader returns representations only; mu/sd/native "
          "live in a separate function  OK")

    # 22: the selected candidate is NOT refit
    assert sel["refit_after_selection"] is False
    rec = fit_layer_alignment(Xtr, Ytr, Xva, Yva, Xte, Yte, grid=RIDGE_GRID,
                              check_targets=False)
    s2 = RidgeAlignmentSolver(Xtr, Ytr, check_targets=False)
    A2, b2 = s2.coefficients(rec["lambda"])
    assert np.allclose(rec["A"], A2, rtol=0, atol=0) and np.allclose(rec["b"], b2, rtol=0,
                                                                     atol=0)
    print("     22        the selected FULL-TRAIN fit is used unchanged (no refit)      OK")


# ------------------------------------------------------------------ 9, 10, 11, 12
def test_representation_r2_and_controls():
    """The R^2 formula, and the three sanity points it must hit."""
    r = np.random.default_rng(11)
    Y = r.normal(size=(60, 12)) * np.arange(1, 13)
    Yhat = Y + r.normal(size=Y.shape) * 0.3

    # 9: against an explicit hand computation
    m = representation_metrics(Yhat, Y)
    ss_res = float(((Yhat - Y) ** 2).sum())
    ss_tot = float(((Y - Y.mean(axis=0)) ** 2).sum())
    assert abs(m["r2"] - (1.0 - ss_res / ss_tot)) < 1e-14
    assert abs(m["mse"] - ss_res / Y.size) < 1e-14
    assert abs(m["relative_frobenius_error"] - np.sqrt(ss_res / ss_tot)) < 1e-14
    print(f"     9         R^2 = 1 - SSres/SStot and MSE = SSres/(N d) verified exactly  OK")

    # 10: a perfect reconstruction
    assert abs(representation_metrics(Y, Y)["r2"] - 1.0) < 1e-12
    assert representation_metrics(Y, Y)["mse"] == 0.0
    X, Yl, W, c = _linear_problem(n=200, d=30, k=30, noise=0.0, seed=12)
    rec = fit_layer_alignment(X, Yl, X[:60], Yl[:60], X[60:], Yl[60:], grid=(1e-10, 1e-8),
                              check_targets=False)
    assert rec["test"]["r2"] > 1 - 1e-9, rec["test"]["r2"]
    print(f"     10        a perfect synthetic map gives test R^2 = {rec['test']['r2']:.10f}    OK")

    # 11: the mean predictor
    self_mean = mean_baseline_metrics(Y, Y.mean(axis=0))
    assert abs(self_mean["r2"]) < 1e-12, self_mean["r2"]
    train_mean = mean_baseline_metrics(Y, Y[:20].mean(axis=0))
    assert train_mean["r2"] <= 1e-12, train_mean["r2"]
    print(f"     11        mean predictor: own-mean R^2 {self_mean['r2']:+.1e}, train-mean "
          f"{train_mean['r2']:+.4f} (<= 0)  OK")

    # 12: a permuted correspondence collapses to ~ the mean baseline
    Yp = permuted_rows(Yl[:140], seed=0)
    assert not np.array_equal(Yp, Yl[:140])
    s = RidgeAlignmentSolver(X[:140], Yl[:140], check_targets=False)
    sp = s.with_target(Yp)
    assert np.allclose(sp.s, s.s) and np.allclose(sp.Vt, s.Vt)      # same design, same spectrum
    sel = select_lambda(sp, X[140:170], Yl[140:170], grid=RIDGE_GRID, check_targets=False)
    A, b = sp.coefficients(sel["lambda"])
    r2p = representation_metrics(apply_alignment(A, b, X[170:]), Yl[170:])["r2"]
    r2r = rec["test"]["r2"]
    assert r2p < 0.2 and r2p < r2r - 0.5, (r2p, r2r)
    print(f"     12        permuted correspondence: R^2 {r2p:+.4f} vs real {r2r:+.4f}     OK")


# ------------------------------------------------------------------ 14, 21, 15
def test_identity_endpoint():
    """A_20 = I, b_20 = 0 -- not fitted, and bit-identical to the representation it maps."""
    idp = identity_alignment(MODEL_DIMS)
    assert idp["fitted"] is False and idp["lambda"] is None
    assert np.array_equal(idp["A"], np.eye(MODEL_DIMS))
    assert not idp["b"].any()
    print("     21        the L20 adapter is the identity and is NOT fitted             OK")

    # 14: mapping through it changes nothing, bit for bit -- so the frozen head applied to
    #     A h + b IS the frozen head applied to h, and the endpoint is exact by construction
    X = np.random.default_rng(13).normal(size=(37, MODEL_DIMS)).astype(np.float32)
    out = apply_alignment(idp["A"], idp["b"], X)
    assert out.dtype == np.dtype(SOLVE_DTYPE)
    assert np.array_equal(out, X.astype(SOLVE_DTYPE)), float(np.abs(out - X).max())
    print("     14        the identity adapter reproduces h_20 BIT-IDENTICALLY          OK")

    # 15: the driver fits every non-final point and only the non-final points
    from experiments import run_timesfm3_representation_alignment as drv
    src = inspect.getsource(drv.run_dataset)
    assert "fit_layers = [l for l in layers if l != LAST_LAYER]" in src
    layers = list(range(NUM_LAYERS))
    fit_layers = [l for l in layers if l != LAST_LAYER]
    assert len(fit_layers) == 20 and LAST_LAYER not in fit_layers
    assert fit_layers[0] == 0 and LAYER_NAMES[0] == "Emb" and fit_layers[-1] == 19
    # L20 short-circuits to the endpoint object itself, so no second head application exists
    assert "fc = endpoint" in src, "L20 must reuse the cached-L20 endpoint, not recompute it"
    assert "absolute_difference" in inspect.getsource(drv.controls)
    print(f"     15        adapters fit for Emb..L19 ({len(fit_layers)} points); L20 is the "
          f"identity endpoint  OK")


# ------------------------------------------------------------------ 16, 17
def test_layer_alignment_and_roster():
    """The four curves are joined BY LAYER, and the paper7 roster is the seven."""
    from experiments.run_timesfm3_last_token_probing import KIND, PAPER7, SHORT
    from experiments.run_timesfm3_representation_alignment import CSV_HEAD, load_reference
    from probing.tunnel import PT_ID_TAGS, PT_OOD_TAGS

    # 17: the roster comes from probing/tunnel.py and is never re-declared
    assert list(PAPER7) == list(PT_ID_TAGS) + list(PT_OOD_TAGS)
    assert len(PAPER7) == 7 and len(set(PAPER7)) == 7
    assert all(KIND[t] == "PT-ID" for t in PT_ID_TAGS)
    assert all(KIND[t] == "PT-OOD" for t in PT_OOD_TAGS)
    assert all(t in SHORT for t in PAPER7)
    print(f"     17        paper7 = {len(PT_ID_TAGS)} PT-ID + {len(PT_OOD_TAGS)} PT-OOD = 7, "
          f"from probing/tunnel.py     OK")

    # 16: the merged reference is keyed by INTEGER layer for every curve
    ref_path = (repo_root() / "results" / "timesfm3_representation_geometry"
                / "native_head_transfer_summary.json")
    if ref_path.exists():
        ref = load_reference(ref_path)
        assert ref["available"], ref
        assert set(ref["datasets"]) == set(PAPER7), set(ref["datasets"]) ^ set(PAPER7)
        for tag, e in ref["datasets"].items():
            for key in ("direct_native_head_q9_loss", "learned_probe_q9_loss"):
                ks = sorted(e[key])
                assert ks == list(range(NUM_LAYERS)), (tag, key, ks[:3], ks[-3:])
                assert all(isinstance(k, int) for k in ks)
            assert isinstance(e["native_timesfm_q9_loss"], float)
            assert e["tunnel_entrance_5pct"] in range(NUM_LAYERS)
        print(f"     16        direct / probe / native / tunnel merge by INTEGER layer for "
              f"all 7  OK")
    else:
        print(f"     16        [skipped] no committed reference at {ref_path}")

    # the CSV carries every metric the specification names, in one row per (dataset, layer)
    need = {"layer", "lambda_selected", "train_representation_mse", "val_representation_mse",
            "test_representation_mse", "val_representation_r2", "test_representation_r2",
            "aligned_native_q9_loss", "aligned_native_median_pinball", "aligned_native_mae",
            "aligned_native_mase", "direct_native_head_q9_loss", "learned_probe_q9_loss",
            "native_timesfm_q9_loss", "tunnel_entrance_5pct"}
    missing = need - set(CSV_HEAD)
    assert not missing, missing
    print("     16b       the table carries every specified metric column               OK")


# ------------------------------------------------------------------ 18
def test_no_backbone_forward_pass():
    """Adapter fitting reads the cache and nothing else -- no decode(), no extraction."""
    import probing.timesfm3_alignment as mod
    src = inspect.getsource(mod)
    for forbidden in ("model.decode(", "extract_last_token_features", "cached_last_token_features",
                      "native_head_transfer_pass"):
        assert forbidden not in src, f"{forbidden} appears in the alignment module"
    # the ONLY torch/model contact is the frozen head application
    assert "model.output_head" in src
    from experiments import run_timesfm3_representation_alignment as drv
    dsrc = inspect.getsource(drv)
    assert "model.decode(" not in dsrc and "extract_last_token_features" not in dsrc
    assert "backbone_forward_passes" in dsrc
    # fitting a layer needs no model argument at all
    assert "model" not in inspect.signature(fit_layer_alignment).parameters
    print("     18        no backbone forward pass: fitting is cache-only, the model is "
          "touched only for output_head  OK")


# ------------------------------------------------------------------ 13, 24, 26
def test_head_plumbing_orthogonality_and_recovery():
    """The frozen-head application contract, Procrustes, and the recovery metric."""
    which = _ensure_tfm_util()
    from experiments.run_timesfm3_native_head_transfer import head_checksum
    from experiments.run_timesfm3_representation_alignment import recovery

    g = LastTokenGeometry()
    m = _StubModel()
    n = 12
    states = np.random.default_rng(21).normal(size=(n, MODEL_DIMS)).astype(np.float64)
    mu, sd = np.zeros(n), np.ones(n)
    trend = np.zeros((n, g.H))
    before = head_checksum(m)
    fc = frozen_head_forecast(m, states, mu, sd, trend, g, batch_size=5)
    assert fc.shape == (n, g.H, NUM_QUANTILES), fc.shape
    # 13: the head is unchanged by being applied
    assert head_checksum(m)["sha256"] == before["sha256"]
    assert before["requires_grad"] is False
    print(f"     13        head checksum unchanged after application  [{which}]          OK")

    # batching cannot change a result
    assert np.array_equal(fc, frozen_head_forecast(m, states, mu, sd, trend, g, batch_size=n))
    print(f"     13b       the forecast is batch-size invariant, shape {fc.shape}         OK")

    # a trainable head is refused
    for p in m.output_head.parameters():
        p.requires_grad_(True)
    try:
        frozen_head_forecast(m, states, mu, sd, trend, g)
        raise AssertionError("a head with requires_grad=True was accepted")
    except RuntimeError as e:
        assert "FROZEN" in str(e), e
    for p in m.output_head.parameters():
        p.requires_grad_(False)
    # and a wrong-width representation is refused
    try:
        frozen_head_forecast(m, states[:, :64], mu, sd, trend, g)
        raise AssertionError("a (n, 64) array was accepted as a token state")
    except ValueError:
        pass
    print("     13c       a trainable head and a wrong-width state are REFUSED          OK")

    # 24: Procrustes is orthogonal, and is strictly the weaker hypothesis class
    r = np.random.default_rng(22)
    X = r.normal(size=(200, 20))
    Q_, _ = np.linalg.qr(r.normal(size=(20, 20)))
    Yrot = X @ Q_ + 5.0
    pr = procrustes_alignment(X, Yrot, check_targets=False)
    assert pr["orthogonality_max_abs_error"] < 1e-8
    assert representation_metrics(apply_alignment(pr["A"], pr["b"], X), Yrot)["r2"] > 1 - 1e-12
    # a NON-orthogonal map: ridge must beat Procrustes
    Yscaled = X @ (Q_ * np.linspace(0.1, 8.0, 20)) + 5.0
    ps = procrustes_alignment(X, Yscaled, check_targets=False)
    r2_pro = representation_metrics(apply_alignment(ps["A"], ps["b"], X), Yscaled)["r2"]
    rr = fit_layer_alignment(X, Yscaled, X, Yscaled, X, Yscaled, grid=(1e-10, 1e-8),
                             check_targets=False)
    assert rr["test"]["r2"] > r2_pro + 0.1, (rr["test"]["r2"], r2_pro)
    print(f"     24        Procrustes exact on a rotation; on a rescaling ridge "
          f"{rr['test']['r2']:.4f} > orthogonal {r2_pro:.4f}  OK")

    # 26: the recovery metric and its guard
    assert abs(recovery(10.0, 10.0, 1.0)["value"] - 0.0) < 1e-12
    assert abs(recovery(10.0, 1.0, 1.0)["value"] - 1.0) < 1e-12
    assert recovery(10.0, 0.5, 1.0)["value"] > 1.0
    z = recovery(1.0, 0.9, 1.0)
    assert z["value"] is None and "denominator" in z["reason"]
    print("     26        R_align: 0 at the direct head, 1 at the endpoint, guarded at 0/0 OK")


# ------------------------------------------------------------------ 19, 20
def test_output_locations():
    """Heavy adapters on $SCRATCH; only light, versionable artifacts inside the repo."""
    from experiments.run_timesfm3_representation_alignment import (parse_args, paper_out_default,
                                                                   save_adapter)
    root = repo_root()
    scratch = os.environ.get("SCRATCH")
    a = parse_args([])

    # 20: the paper outputs live inside the repository, under their own experiment namespace
    po = Path(a.paper_out).resolve()
    assert po == paper_out_default().resolve()
    assert str(po).startswith(str(root.resolve())), (po, root)
    assert po.name == "timesfm3_representation_alignment" and po.parent.name == "results"
    print(f"     20        figures/tables land in {po.relative_to(root)}   OK")

    # 19: the heavy artifacts default to $SCRATCH and are never inside the repo
    orr = Path(a.out_root).resolve()
    assert not str(orr).startswith(str(root.resolve())), orr
    if scratch:
        assert str(orr).startswith(str(Path(scratch).resolve())), (orr, scratch)
    with tempfile.TemporaryDirectory() as td:
        A = np.eye(8); b = np.zeros(8)
        p = save_adapter(Path(td) / "ridge", 3, A, b, {"lambda": 1.0, "muX": np.zeros(8),
                                                       "muY": np.zeros(8)}, "m4_hourly", "ridge")
        assert p.exists() and p.suffix == ".npz"
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z["meta"]))
            assert z["A"].dtype == np.float32 and meta["alignment_version"] == ALIGNMENT_VERSION
            assert meta["layer"] == 3 and meta["layer_name"] == "L3"
    # nothing with a heavy extension is written to the repo side
    src = inspect.getsource(
        __import__("experiments.run_timesfm3_representation_alignment",
                   fromlist=["write_outputs"]).write_outputs)
    assert ".npy" not in src and ".npz" not in src
    print(f"     19        adapter matrices go to {orr}"
          f"{' ($SCRATCH)' if scratch else ''}   OK")


# --------------------------------------------------------------------------- #
# --with-model: the real checkpoint (COMPUTE NODE)
# --------------------------------------------------------------------------- #
def model_tests(argv):
    """13 / 14 against the real frozen head, and the cached-L20 endpoint vs decode()."""
    from experiments.run_timesfm3_native_head_transfer import head_checksum
    from probing.timesfm3_last_token import (assert_backbone_frozen, extract_last_token_features,
                                             get_model)
    from probing.timesfm3_last_token_probes import native_reference

    print("\n  --with-model (real checkpoint)")
    g = LastTokenGeometry()
    model, device = get_model(os.environ.get("TIMESFM3_CHECKPOINT", DEFAULT_CHECKPOINT), None)
    assert_backbone_frozen(model)
    ck = head_checksum(model)
    assert ck["in_features"] == MODEL_DIMS and ck["out_features"] == 64 * NUM_QUANTILES
    assert ck["requires_grad"] is False
    print(f"     13        real head: {ck['module']}({ck['in_features']}, {ck['out_features']}), "
          f"sha256 {ck['sha256']}, frozen   OK")

    # one small real batch, extracted the way the experiment's cache was built
    n = 8
    X = np.random.default_rng(5).normal(size=(n, g.C)).astype(np.float32).cumsum(axis=1)
    out = extract_last_token_features(X, geom=g, model=model, device=device, batch_size=n,
                                      layers=[LAST_LAYER], verify=True, progress=False)
    h20 = np.asarray(out["feats"][LAST_LAYER], SOLVE_DTYPE)
    assert h20.shape == (n, MODEL_DIMS)

    # 14: the identity adapter, then the frozen head, vs decode()'s own forecast
    Z = np.concatenate([X, np.zeros((n, g.H), np.float32)], axis=1)
    from probing.timesfm3_last_token import build_last_token_targets
    p = build_last_token_targets(Z, out["mu"], out["sd"], g, detrend=True)
    idp = identity_alignment(MODEL_DIMS)
    aligned = apply_alignment(idp["A"], idp["b"], h20)
    assert np.array_equal(aligned, h20)
    fc = frozen_head_forecast(model, aligned, out["mu"], out["sd"], p["trend"], g,
                              batch_size=n, device=device)
    ref = np.asarray(out["native"], np.float64)
    d = np.abs(fc - ref)
    rel = float(d.max() / (np.abs(ref).mean() + 1e-12))
    a = native_reference(fc, out["mu"], out["sd"], p["trend"], p["targets"], p["valid"])
    b = native_reference(out["native"], out["mu"], out["sd"], p["trend"], p["targets"],
                         p["valid"])
    lrel = abs(a["q9_loss"] - b["q9_loss"]) / max(abs(b["q9_loss"]), 1e-12)
    print(f"     14        cached-L20 endpoint vs decode(): max|d| {d.max():.3e} "
          f"(relative {rel:.2e}); Q9 loss relative {lrel:.2e}")
    assert lrel < 1e-3, (
        f"the cached-L20 endpoint's Q=9 loss differs from decode()'s by {lrel:.3e} relative. "
        "A ~1e-6 difference is the expected (N,1280)-vs-(b,1,18,1280) accumulation effect; "
        "1e-3 is structural.")
    print(f"     14b       the two endpoints agree to {lrel:.1e} of the loss -- the "
          "(N, 1280) path is sound  OK")
    assert head_checksum(model)["sha256"] == ck["sha256"], "the head CHANGED during evaluation"
    print("     13b       head checksum unchanged after the real application            OK")


if __name__ == "__main__":
    print(f"TimesFM-3 LINEAR REPRESENTATION ALIGNMENT contracts   [{ALIGNMENT_VERSION}, "
          f"alignments {ALIGNMENTS}]")
    for fn in (test_representation_matrices_and_cache_isolation,
               test_ridge_recovers_a_known_map,
               test_lambda_selection_is_validation_representation_only,
               test_representation_r2_and_controls,
               test_identity_endpoint,
               test_layer_alignment_and_roster,
               test_no_backbone_forward_pass,
               test_head_plumbing_orthogonality_and_recovery,
               test_output_locations):
        fn()
    if "--with-model" in sys.argv:
        model_tests(sys.argv)
        print("\nall contracts hold (model-backed checks included)")
    else:
        print("\nall model-free contracts hold "
              "(run with --with-model for 13/14 on a COMPUTE node)")
