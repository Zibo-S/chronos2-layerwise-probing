"""Contract tests for the TimesFM-3 representation-geometry and frozen-native-head analyses.

    python -m tests.test_timesfm3_representation_geometry               # no model, no GPU
    python -m tests.test_timesfm3_representation_geometry --with-model  # + checkpoint checks

The model-free block is login-node safe: synthetic arrays and temporary caches only, torch
pinned to 2 threads, a few seconds. The --with-model block loads the TimesFM-3 checkpoint, so it
belongs in salloc/sbatch.

Numbered to the two specification lists.

REPRESENTATION GEOMETRY (CKA + effective rank)
     1 input matrices are (N, 1280)            11 effective rank == the Chronos-2 helper
     2 exactly 21 representation points         12 rank-1 matrix -> effective rank 1
     3 only token-15 caches are accepted        13 isotropic matrix -> effective rank ~ d
     4 (N, 16, 1280) shared-origin is REJECTED  14 the paper7 roster is the seven
     5 biased linear CKA vs a hand computation  15 cached N / window identity is verified
     6 CKA symmetry                             16 no probe weights/targets/predictions needed
     7 CKA diagonal == 1                        17 PT-ID / PT-OOD labels preserved
     8 CKA is isotropic-rescale invariant       18 result matrices/vectors are correctly shaped
     9 centering is across OBSERVATIONS         19 a split cannot silently reuse another's cache
    38 the CKA estimator grid: biased = required parity headline, unbiased = required companion
    40 the frozen-head transfer is cache-free and in-memory, and runs as its own GPU job
    10 CKA accumulates in float64

FROZEN NATIVE-HEAD TRANSFER
    20 the head IS model.output_head            29 no optimizer/fitting is instantiated
    21 head parameters unchanged before/after   30 L20 reproduces decode() to < 1e-4
    22 requires_grad=False                      31 the Q=9 loss uses the existing helper
    23 head input is (N, 1280)                  32 the median is native index 4
    24 head output is (N, 576)                  33 MASE is the paper7 convention (m=24)
    25 the verified (N, 64, 9) reshape          34 all seven datasets are evaluated
    26 exact native quantile ordering           35 paper7 test-window identity is preserved
    27 only token-15 features are accepted      36 the alignment gap A_l is correct
    28 all 21 representation points evaluated   37 head and probe arrays align by dataset+layer

NOTHING here is weakened to make a run pass.
"""

from __future__ import annotations

import inspect
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

torch.set_num_threads(2)          # never grab every core of a shared login node

from probing import cka as cka_mod  # noqa: E402
from probing.spectral_metrics import spectral_metrics  # noqa: E402
from probing.timesfm3_last_token import (DEFAULT_CHECKPOINT, LAYER_NAMES,  # noqa: E402
                                         MODEL_DIMS, NATIVE_MEDIAN_IDX, NATIVE_QUANTILES,
                                         NUM_LAYERS, NUM_QUANTILES, LastTokenGeometry,
                                         cache_metadata, cache_root)
from probing.timesfm3_geometry import (CKA_ESTIMATOR, SELECTED_TOKEN_INDEX,  # noqa: E402
                                       assert_last_token_matrix, cka_layer_matrix,
                                       cka_null_floor, effective_rank_curve,
                                       estimator_provenance, load_last_token_reps,
                                       paper_out_default, repo_root)

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _fake_cache(cache_dir, tag, split, X, geom, layers, *, rank3=False, suite="paper7",
                checkpoint=DEFAULT_CHECKPOINT, seed=0, detrend=True, dtype=np.float32,
                meta_override=None):
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
                np.random.default_rng(100 + L).normal(size=shape).astype(dtype))
    np.savez(root / "meta.npz", meta=json.dumps(meta), ctx_tail=X[:, -8:],
             mu=np.zeros(n), sd=np.ones(n),
             native=np.zeros((n, geom.H, NUM_QUANTILES), np.float32),
             native_check=json.dumps({}), dtype_check=json.dumps({}), sorting=json.dumps({}))
    return root


def _ctx(n, C=512):
    return np.random.default_rng(7).normal(size=(n, C)).astype(np.float32)


# ------------------------------------------------------------------ 1, 2, 3, 18
def test_input_matrices_and_points():
    """(N, 1280) per point, exactly 21 points, token 15, correctly shaped results."""
    g = LastTokenGeometry()
    assert NUM_LAYERS == 21 and len(LAYER_NAMES) == 21, (NUM_LAYERS, len(LAYER_NAMES))
    assert LAYER_NAMES[0] == "Emb" and LAYER_NAMES[-1] == "L20"
    assert LAYER_NAMES[1:] == [f"L{i}" for i in range(1, 21)]
    assert SELECTED_TOKEN_INDEX == 15 == g.selected_token_index
    assert g.C == 512 and g.H == 64 and g.P == 32 and g.n_real_context_patches == 16
    assert g.K == 1, "H=64 gives ONE native readout token, so there is NO (n,K,d) reshape"
    assert MODEL_DIMS == 1280

    n = 40
    reps = [RNG.normal(size=(n, MODEL_DIMS)).astype(np.float32) for _ in range(NUM_LAYERS)]
    for name, R in zip(LAYER_NAMES, reps):
        a = assert_last_token_matrix(R, n, name)
        assert a.shape == (n, 1280)
    M = cka_layer_matrix(reps)
    assert M.shape == (21, 21), M.shape
    er = effective_rank_curve(reps, LAYER_NAMES)
    for k in ("effective_rank", "normalized_effective_rank",
              "effective_rank_over_max_possible", "spectral_entropy", "pc1_fraction"):
        assert len(er[k]) == 21, (k, len(er[k]))
    assert er["hidden_dim"] == 1280 and er["n_samples"] == n
    assert er["max_possible_rank"] == min(n - 1, 1280)
    print("  1,2,3,18  (N,1280) x 21 points at token 15; 21x21 CKA + length-21 curves   OK")


# ------------------------------------------------------------------ 4, 27
def test_shared_origin_rank3_rejected():
    """The 16-prefix ablation's (N, 16, 1280) representation can never be analysed here."""
    bad = np.zeros((5, 16, MODEL_DIMS), np.float32)
    try:
        assert_last_token_matrix(bad, 5, "L20")
    except ValueError as e:
        assert "rank-3" in str(e) and "16" in str(e), str(e)
    else:
        raise AssertionError("a (N,16,1280) shared-origin array must be REFUSED")
    for wrong in (np.zeros((5, 768), np.float32), np.zeros((4, MODEL_DIMS), np.float32),
                  np.zeros(MODEL_DIMS, np.float32)):
        try:
            assert_last_token_matrix(wrong, 5, "L20")
        except ValueError:
            pass
        else:
            raise AssertionError(f"shape {wrong.shape} must be refused")

    # and the loader refuses it on disk too, through the real read_cache shape guard
    g = LastTokenGeometry()
    X = _ctx(6)
    with tempfile.TemporaryDirectory() as d:
        _fake_cache(d, "m4_hourly", "test", X, g, list(range(NUM_LAYERS)), rank3=True)
        try:
            load_last_token_reps("m4_hourly", "test", X, cache_dir=d, geom=g)
        except RuntimeError as e:
            assert "rank 3" in str(e) or "shared-origin" in str(e), str(e)
        else:
            raise AssertionError("a rank-3 cache on disk must be REFUSED")
    print("  4,27      rank-3 shared-origin features rejected in memory AND on disk    OK")


# ------------------------------------------------------------------ 5, 6, 7, 8, 9, 10
def test_biased_linear_cka():
    """The estimator IS the Chronos-2 biased linear CKA, with its invariances."""
    assert CKA_ESTIMATOR == "biased"
    assert inspect.signature(cka_mod.linear_cka).parameters["estimator"].default == "biased"

    # 5a: the literal formula, computed independently of probing.cka
    X = RNG.normal(size=(30, 5)).astype(np.float32)
    Y = RNG.normal(size=(30, 8)).astype(np.float32)
    Xc = X.astype(np.float64) - X.astype(np.float64).mean(axis=0)
    Yc = Y.astype(np.float64) - Y.astype(np.float64).mean(axis=0)
    hand = (np.linalg.norm(Xc.T @ Yc, "fro") ** 2
            / (np.linalg.norm(Xc.T @ Xc, "fro") * np.linalg.norm(Yc.T @ Yc, "fro")))
    got = cka_mod.linear_cka(X, Y)
    assert abs(got - hand) < 1e-12, (got, hand)

    # 5b: for 1-D representations biased linear CKA is exactly the squared correlation
    a = RNG.normal(size=(50, 1)); b = 2.5 * a + 0.4 * RNG.normal(size=(50, 1))
    r2 = float(np.corrcoef(a.ravel(), b.ravel())[0, 1] ** 2)
    assert abs(cka_mod.linear_cka(a, b) - r2) < 1e-10, (cka_mod.linear_cka(a, b), r2)

    # 6, 7: symmetry and unit diagonal on a full 21-point set
    reps = [RNG.normal(size=(35, MODEL_DIMS)).astype(np.float32) for _ in range(NUM_LAYERS)]
    reps[3] = reps[2] @ np.linalg.qr(RNG.normal(size=(MODEL_DIMS, MODEL_DIMS)))[0]  # rotation
    M = cka_layer_matrix(reps)
    assert np.abs(M - M.T).max() < 1e-10
    assert np.abs(np.diag(M) - 1.0).max() < 1e-8, np.abs(np.diag(M) - 1.0).max()
    assert abs(M[2, 3] - 1.0) < 1e-8, "CKA must be invariant to an orthogonal transform"
    assert (M >= -1e-9).all() and (M <= 1 + 1e-9).all(), "biased CKA lives in [0, 1]"

    # 8: isotropic rescaling of ONE representation changes nothing
    base = cka_mod.linear_cka(X, Y)
    for s in (1e-3, 3.7, 1e3):
        assert abs(cka_mod.linear_cka(X, s * Y) - base) < 1e-8, s

    # 9: centering is across the N observations, not across the 1280 hidden dimensions
    A = RNG.normal(size=(20, 6)) + 5.0
    Ac = cka_mod._center(A)
    assert np.abs(Ac.mean(axis=0)).max() < 1e-12, "columns must be centred"
    assert np.abs(Ac.mean(axis=1)).max() > 1e-3, "rows must NOT be centred"
    # a per-feature offset is removed by the centering. Checked in float64, where this is the
    # exact algebraic statement; in float32 the SHIFT ITSELF loses ~1e-6 of the stored values
    # (that is the input's dtype, not the estimator), so it gets its own realistic bar.
    X64 = X.astype(np.float64)
    base64 = cka_mod.linear_cka(X64, Y)
    for shift in (17.0, 1e4):
        assert abs(cka_mod.linear_cka(X64 + shift, Y) - base64) < 1e-14, shift
    assert abs(cka_mod.linear_cka(X + np.float32(17.0), Y) - base) < 1e-6

    # 10: the float32 cache is cast to float64 BEFORE any accumulation
    assert cka_mod._as_2d_f64(np.zeros((3, 2), np.float32)).dtype == np.float64
    assert cka_mod.linear_cka(X, Y) == cka_mod.linear_cka(X.astype(np.float64),
                                                          Y.astype(np.float64))
    assert isinstance(got, float)

    # the null floor is a MEASUREMENT of this estimator at this shape, not a correction: it must
    # be large at the TimesFM test N and small at the train N, and it must never alter a matrix
    lo = cka_null_floor(1394, MODEL_DIMS, reps=2, seed=0)["mean"]
    hi = cka_null_floor(262, MODEL_DIMS, reps=2, seed=0)["mean"]
    tiny = cka_null_floor(48, MODEL_DIMS, reps=2, seed=0)["mean"]
    assert 0.0 < lo < hi < tiny < 1.0, (lo, hi, tiny)
    assert hi > 0.5, f"the biased floor at N=262, d=1280 is {hi:.3f}; the figures must show it"
    assert cka_null_floor(30, 5, reps=2)["estimator"] == "biased"
    assert cka_null_floor(30, 5, reps=2)["d"] == 5
    # unbiased at the same shapes sits at ~0, which is what makes it the robustness companion
    assert abs(cka_null_floor(262, MODEL_DIMS, reps=2, estimator="unbiased")["mean"]) < 0.05
    print(f"  extra     biased null floor: N=1394 {lo:.3f} < N=262 {hi:.3f} < N=48 "
          f"{tiny:.3f}  (true CKA = 0)                        OK")
    print("  5-10      biased linear CKA: hand-checked, symmetric, unit diagonal, scale- "
          "and rotation-invariant, observation-centred, float64                        OK")


# ------------------------------------------------------------------ 11, 12, 13
def test_effective_rank_definition():
    """Effective rank is the Chronos-2 helper, unmodified, on squared singular values."""
    A = RNG.normal(size=(120, 25)) @ np.diag(np.linspace(1.0, 0.02, 25))

    # 11: our curve calls spectral_metrics and reports exactly its number
    ref = spectral_metrics(A)["effective_rank"]
    mine = effective_rank_curve([A], ["Emb"], hidden_dim=25)["effective_rank"][0]
    assert mine == ref, (mine, ref)

    # the definition itself: exp(H) of the SQUARED spectrum, centred across examples
    s = np.linalg.svd(A - A.mean(axis=0), compute_uv=False)
    def er(w):
        p = w / w.sum(); p = p[p > 1e-12]; return float(np.exp(-(p * np.log(p)).sum()))
    assert abs(ref - er(s ** 2)) < 1e-9, "must use SQUARED singular values"
    assert abs(ref - er(s)) > 1e-3, "must NOT use raw singular values"
    assert estimator_provenance()["effective_rank"]["uses_squared_singular_values"] is True

    # 12: a rank-1 matrix has effective rank exactly 1
    r1 = np.outer(RNG.normal(size=200), RNG.normal(size=40))
    assert abs(spectral_metrics(r1)["effective_rank"] - 1.0) < 1e-8

    # 13: an isotropic cloud fills its dimensions
    iso = RNG.normal(size=(4000, 40))
    e_iso = spectral_metrics(iso)["effective_rank"]
    assert 0.9 * 40 < e_iso <= 40 + 1e-6, e_iso

    # the ADDED diagnostics normalize, they do not replace
    c = effective_rank_curve([iso], ["Emb"], hidden_dim=40)
    assert c["effective_rank"][0] == e_iso
    assert abs(c["normalized_effective_rank"][0] - e_iso / 40) < 1e-12
    assert abs(c["effective_rank_over_max_possible"][0] - e_iso / min(3999, 40)) < 1e-12
    print(f"  11,12,13  exp(H) of s**2 (rank-1 -> 1.000, isotropic 40-d -> {e_iso:.1f}); "
          f"normalized rank is ADDED, raw metric unchanged                             OK")


# ------------------------------------------------------------------ 14, 17
def test_paper7_roster_and_labels():
    """The roster is the Chronos-2 seven and keeps its PT-ID / PT-OOD labels and names."""
    from experiments.run_timesfm3_last_token_probing import KIND, PAPER7, SHORT, suite_tags
    from experiments.run_timesfm3_native_head_transfer import SLUG as NH_SLUG
    from experiments.run_timesfm3_representation_geometry import SLUG
    from probing.tunnel import PT_ID_TAGS, PT_OOD_TAGS

    seven = suite_tags("paper7")
    assert seven == list(PT_ID_TAGS) + list(PT_OOD_TAGS) == list(PAPER7)
    assert set(seven) == {"m4_hourly", "monash_electricity_hourly", "uber_tlc_hourly",
                          "wind_farms_hourly", "sg_carpark", "coastal_ts", "boom_hourly"}
    assert len(seven) == 7 == len(set(seven))
    assert [SHORT[t] for t in PT_ID_TAGS] == ["Electricity", "Uber TLC", "M4", "Wind Farms"]
    assert [SHORT[t] for t in PT_OOD_TAGS] == ["SG Carpark", "Coastal T-S", "BOOM"]
    assert {KIND[t] for t in PT_ID_TAGS} == {"PT-ID"}
    assert {KIND[t] for t in PT_OOD_TAGS} == {"PT-OOD"}
    # SLUG now comes from probing.registry, which knows every registered dataset -- not just
    # these seven. The contract is COVERAGE plus agreement, not an exact set equality that
    # would break every time a dataset is added to the registry.
    assert set(seven) <= set(SLUG) and set(seven) <= set(NH_SLUG), "both drivers must cover "\
        "all seven"
    assert SLUG == NH_SLUG, "the two drivers must name files identically"
    for t in ("monash_kdd_cup_2018", "monash_pedestrian_counts"):
        assert t not in seven
    print("  14,17     roster == the Chronos-2 seven; PT-ID/PT-OOD labels + names kept   OK")


# ------------------------------------------------------------------ 15, 19
def test_window_identity_and_split_isolation():
    """The cache is accepted only when it provably belongs to THESE windows and THIS split."""
    g = LastTokenGeometry()
    layers = list(range(NUM_LAYERS))
    Xtr, Xte = _ctx(11), _ctx(9) + 3.0
    with tempfile.TemporaryDirectory() as d:
        _fake_cache(d, "m4_hourly", "train", Xtr, g, layers)
        _fake_cache(d, "m4_hourly", "test", Xte, g, layers)

        ok = load_last_token_reps("m4_hourly", "test", Xte, cache_dir=d, geom=g)
        assert ok["n"] == 9 and len(ok["reps"]) == 21
        assert all(r.shape == (9, MODEL_DIMS) for r in ok["reps"])
        assert ok["selected_token_index"] == 15 and ok["split"] == "test"
        assert len(ok["window_identity_hash"]) == 16

        # 15: different windows of the right length -> the context-tail check fires
        try:
            load_last_token_reps("m4_hourly", "test", _ctx(9) - 11.0, cache_dir=d, geom=g)
        except RuntimeError as e:
            assert "stale" in str(e) or "do not match" in str(e), str(e)
        else:
            raise AssertionError("a cache from different windows must be REFUSED")

        # 15b: a different N -> refused
        try:
            load_last_token_reps("m4_hourly", "test", Xte[:5], cache_dir=d, geom=g)
        except (RuntimeError, FileNotFoundError):
            pass
        else:
            raise AssertionError("a row-count mismatch must be REFUSED")

        # 19: asking for test must never land on the train cache (distinct roots + meta.split)
        assert cache_root(d, "m4_hourly", "train", g, True, "paper7") != \
               cache_root(d, "m4_hourly", "test", g, True, "paper7")
        try:
            load_last_token_reps("m4_hourly", "val", Xte, cache_dir=d, geom=g)
        except FileNotFoundError as e:
            assert "CACHE-ONLY" in str(e).upper()
        else:
            raise AssertionError("a split with no cache must fail loud, not fall back")

        # 19b: even a hand-renamed directory is caught by the metadata's own split field
        mis = cache_root(d, "m4_hourly", "test", g, True, "paper7")
        bad_dir = Path(d) / "renamed"
        _fake_cache(d, "m4_hourly", "test", Xte, g, layers,
                    meta_override={"split": "train"})
        try:
            load_last_token_reps("m4_hourly", "test", Xte, cache_dir=d, geom=g)
        except RuntimeError as e:
            assert "split" in str(e) and "INCOMPATIBLE" in str(e), str(e)
        else:
            raise AssertionError("a relabelled cache must be REFUSED")
        del mis, bad_dir

        # a different dataset / suite / checkpoint is equally refused
        for kw in ({"checkpoint": "someone/else"}, {"suite": "extended_v1"}):
            try:
                load_last_token_reps("m4_hourly", "test", Xte, cache_dir=d, geom=g, **kw)
            except (RuntimeError, FileNotFoundError):
                pass
            else:
                raise AssertionError(f"{kw} must be REFUSED")
    print("  15,19     window identity verified element-wise; splits/suites/checkpoints "
          "cannot cross-load                                                           OK")


# ------------------------------------------------------------------ 16
def test_probe_independence():
    """No probe weight, target, prediction or native-head output can reach a geometry number."""
    g = LastTokenGeometry()
    X = _ctx(8)
    with tempfile.TemporaryDirectory() as d:
        _fake_cache(d, "boom_hourly", "test", X, g, list(range(NUM_LAYERS)))
        out = load_last_token_reps("boom_hourly", "test", X, cache_dir=d, geom=g)
    forbidden = {"mu", "sd", "native", "targets", "predictions", "probe", "weights"}
    assert not (forbidden & set(out)), f"the loader leaked {forbidden & set(out)}"
    assert set(out["reps"][0].shape) == {8, MODEL_DIMS}

    import experiments.run_timesfm3_representation_geometry as geo
    src = inspect.getsource(geo)
    for banned in ("torch.optim", "AdamW", "backward()", "fit_last_token_probe",
                   "test_q9_loss\"]", "median_pred"):
        assert banned not in src, f"the geometry driver must not reference {banned}"
    assert "probe_results" in src, "it may still READ the tunnel entrance for the overlay"
    print("  16        geometry consumes representations only; mu/sd/native dropped by "
          "the loader                                                                  OK")


# ------------------------------------------------------------------ 20-29, 31-33, 36, 37 (model-free parts)
def test_native_head_transfer_contracts():
    """Everything about the frozen-head experiment that needs no checkpoint."""
    import experiments.run_timesfm3_native_head_transfer as nh
    src = inspect.getsource(nh)

    # 20: the head is the checkpoint's own module, addressed directly, and applied to the FULL
    # (b, 1, n_tokens, d) state with the readout token sliced AFTERWARDS -- the order decode()
    # uses. Slicing first changes the matmul shape, which is not bit-identical under TF32.
    assert "model.output_head(states)[:, :, [tok], :]" in src, \
        "the head must be applied to the full token sequence, then sliced (decode()'s order)"
    assert src.count("model.output_head") >= 2
    # 21/22/29: frozen, no optimizer, no fitting, no gradients
    for banned in ("torch.optim", "AdamW", ".backward(", "requires_grad_(True)",
                   "Linear(1280", "torch.nn.Linear", "fit_last_token_probe", "wd_grid",
                   "weight_decay", "train()"):
        assert banned not in src, f"the frozen-head driver must not contain {banned}"
    assert "torch.no_grad()" in src
    assert "head_checksum" in src and "sha256" in src
    assert "native_head_checksum_after" in src

    # 26/32: the quantile vector and the median index come from the model's own constants
    # NATIVE_QUANTILES is stored float32, so compare on value (float32(0.1) != float(0.1))
    assert np.allclose(NATIVE_QUANTILES, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
                       atol=1e-7), NATIVE_QUANTILES
    assert NUM_QUANTILES == 9 and NATIVE_MEDIAN_IDX == 4
    assert abs(float(NATIVE_QUANTILES[NATIVE_MEDIAN_IDX]) - 0.5) < 1e-12
    assert np.all(np.diff(NATIVE_QUANTILES) > 0), "native order is ascending and NOT sorted again"

    # 25: the verified horizon-major reshape, (B, 576) -> (B, 64, 9), flat index t*Q + q
    from probing.timesfm3_last_token_probes import reshape_prediction
    B, H, Q = 3, 64, 9
    flat = np.arange(B * H * Q, dtype=np.float32).reshape(B, H * Q)
    out = reshape_prediction(torch.from_numpy(flat), H, Q)
    assert tuple(out.shape) == (B, Q, H)
    for t in (0, 5, 63):
        for q in (0, 4, 8):
            assert out[1, q, t].item() == flat[1, t * Q + q], (t, q)
    assert H * Q == 576

    # 31/33: the scoring helpers are IMPORTED, not reimplemented
    # The seasonal period moved from a module global to probing.registry, per dataset. The
    # contract is now that every one of the paper7 datasets still resolves to the hourly 24,
    # which is what keeps the committed MASE numbers reproducible.
    from probing.registry import PAPER7, seasonal_m
    assert {seasonal_m(t) for t in PAPER7} == {24}, {t: seasonal_m(t) for t in PAPER7}
    assert "M_SEASON" not in src, "driver must not reintroduce a global seasonal period"
    assert "from experiments.run_timesfm3_probing import" in src
    assert "per_window_mase" in src and "mase_denominator" in src
    assert "native_reference" in src and "def native_reference" not in src
    assert "def pinball" not in src and "def _mase" not in src

    # 31b: native_reference really is the Q=9 pinball on the probe's axis
    from probing.timesfm3_last_token_probes import native_reference, pinball_loss_per_window
    n = 12
    raw = RNG.normal(size=(n, H, Q))
    mu, sd, trend = RNG.normal(size=n), np.abs(RNG.normal(size=n)) + 1.0, RNG.normal(size=(n, H))
    tgt = RNG.normal(size=(n, H)).astype(np.float32)
    valid = np.ones(n, bool)
    r = native_reference(raw, mu, sd, trend, tgt, valid)
    z = (raw - trend[:, :, None] - mu[:, None, None]) / sd[:, None, None]
    pred = torch.as_tensor(np.ascontiguousarray(z.transpose(0, 2, 1)), dtype=torch.float32)
    want = pinball_loss_per_window(pred, torch.as_tensor(tgt),
                                   torch.as_tensor(NATIVE_QUANTILES)).numpy()
    assert np.allclose(r["q9_window"], want, atol=1e-6)
    assert abs(r["median_loss"] - 0.5 * np.abs(tgt - z[:, :, NATIVE_MEDIAN_IDX]).mean()) < 1e-6

    # 36/37: the alignment gap and the dataset+layer alignment
    layers = list(range(NUM_LAYERS))
    head = [{"layer": l, "native_head_q9_pinball": 1.0 + 0.1 * l} for l in layers]
    probe = {"layers": layers, "test_q9_loss": [0.5 + 0.01 * l for l in layers],
             "tunnel_by_tolerance": {"0.05": {"layer": 7, "layer_name": "L7"}}}
    c = nh._combine_with_probe(head, layers, probe, NUM_LAYERS - 1, LAYER_NAMES)
    assert len(c["rows"]) == 21
    for row, h in zip(c["rows"], head):
        assert row["layer"] == h["layer"] and row["layer_name"] == LAYER_NAMES[h["layer"]]
        assert abs(row["alignment_gap"]
                   - (row["native_head_q9_loss"] - row["probe_q9_loss"])) < 1e-12
    assert abs(c["rows"][0]["alignment_gap"] - (1.0 - 0.5)) < 1e-12
    assert c["tunnel_entrance_5pct"] == 7
    assert abs(c["at_tunnel_entrance"]["alignment_gap"] - (1.7 - 0.57)) < 1e-9
    assert abs(c["rows"][-1]["native_head_relative_to_L20"] - 1.0) < 1e-12
    assert abs(c["rows"][-1]["probe_relative_to_L20"] - 1.0) < 1e-12

    # a probe curve that does not cover every point must FAIL, never be padded
    try:
        nh._combine_with_probe(head, layers, {"layers": layers[:10],
                                              "test_q9_loss": [0.5] * 10}, 20, LAYER_NAMES)
    except RuntimeError as e:
        assert "align by dataset AND layer" in str(e)
    else:
        raise AssertionError("a misaligned probe curve must be REFUSED")
    # and with no probe results at all, the gap is simply absent
    none = nh._combine_with_probe(head, layers, None, 20, LAYER_NAMES)
    assert none["available"] is False and none["rows"] == []
    print("  20-29,31-33,36,37  frozen head (no optimizer/fitting), native quantile order, "
          "t*Q+q reshape, shared scoring helpers, A_l                                   OK")


# ------------------------------------------------------------------ 28, 34, 35
def test_all_points_and_datasets_are_covered():
    """Every run covers 21 points x 7 datasets, and L20 can never be dropped."""
    import experiments.run_timesfm3_native_head_transfer as nh
    import experiments.run_timesfm3_representation_geometry as geo

    for mod in (nh, geo):
        a = mod.parse_args([])
        assert a.suite == "paper7", mod.__name__
        assert a.layers is None, "the default must be all 21 representation points"
        assert a.datasets is None, "the default must be the whole seven-dataset roster"
        assert a.seed == 0 and a.context_len == 512 and a.horizon == 64
    assert "added L" in inspect.getsource(nh.run_dataset), \
        "L20 must be re-added when --layers omits it (the endpoint identity)"
    assert nh.parse_args([]).recon_rtol == 2e-6
    assert nh.parse_args([]).recon_atol == 1e-5
    assert nh.parse_args([]).loss_identity_rtol == 1e-5
    for gone in ("allow_extraction", "cache_dir", "cache_checkpoint", "feature_dtype"):
        assert not hasattr(nh.parse_args([]), gone), \
            f"--{gone.replace('_', '-')} must be GONE: the native-head transfer no longer uses "
    assert nh.parse_args([]).extract_batch_size == 64
    g = geo.parse_args([])
    assert g.erank_split == "train", \
        "effective rank keeps the committed Chronos-2 spectral protocol (train)"
    assert g.headline_cka_split == "test", "the CKA headline is the Chronos-parity test split"
    assert geo.parse_args(["--split", "test"]).split == "test"
    print("  28,34,35  21 points x 7 datasets by default; L20 always kept; CKA headline "
          "test / erank train                                                          OK")


# ------------------------------------------------------------------ 36
def test_endpoint_identity_is_elementwise():
    """L20 gate is |d| <= atol + rtol*|ref| PER ELEMENT, not max|d|/mean|ref|.

    Reproduces the BOOM pathology: one ~162.8 element among ~0.7 values. Float32 rounding on
    the big element must PASS, the old global-mean ratio would have false-flagged it, and a
    genuine structural error on ANY element must FAIL -- no dataset special-cased.
    """
    from experiments.run_timesfm3_native_head_transfer import _endpoint_identity, parse_args
    a = parse_args([]); atol, rtol = a.recon_atol, a.recon_rtol
    rng = np.random.default_rng(0)
    ref = np.abs(rng.normal(0.0, 0.7, size=(354, 64, 9)))     # BOOM-like small values
    ref[10, 20, 8] = 162.8                                    # one heavy-tail element
    recon = ref + rng.uniform(-1, 1, ref.shape) * np.abs(ref) * 3e-7   # ~float32 rounding
    recon[10, 20, 8] = ref[10, 20, 8] + 7.63e-5               # BOOM's real worst |d| (5 ULP)

    chk = _endpoint_identity(recon, ref, 9, atol, rtol)
    assert chk["max_scaled_error"] <= 1.0, chk                # heavy tail passes on merit
    assert set(chk) >= {"max_abs_error", "max_scaled_error", "worst_index",
                        "worst_ref", "worst_recon", "atol", "rtol"}
    old = np.abs(recon - ref).max() / (np.abs(ref).mean() + 1e-12)   # the retired metric
    assert old > 1e-4, f"fixture must reproduce the BOOM false-flag; old metric {old:.2e}"

    bad = recon.copy()
    bad[0, 0, 0] = ref[0, 0, 0] + 0.01 + 0.5 * abs(ref[0, 0, 0])     # 50% element error
    try:
        _endpoint_identity(bad, ref, 9, atol, rtol)
        raise AssertionError("a 50% element error must trip the elementwise gate")
    except RuntimeError as e:
        assert "RECONSTRUCTION FAILED" in str(e)
    print(f"  36        elementwise |d|<=atol+rtol|ref|: heavy tail passes "
          f"(scaled {chk['max_scaled_error']:.3f}); old max/mean {old:.1e} would false-flag; "
          f"50% error fails                                   OK")


# ------------------------------------------------------------------ 38: the estimator grid
def test_cka_estimator_grid():
    """Biased is the required parity headline; unbiased is a required companion, not optional."""
    import experiments.run_timesfm3_representation_geometry as geo

    g = geo.parse_args([])
    assert g.cka_estimators == ["biased", "unbiased"], g.cka_estimators
    assert g.cka_splits == ["test", "train"], g.cka_splits
    assert geo.HEADLINE_ESTIMATOR == "biased" and geo.HEADLINE_CKA_SPLIT == "test"
    assert set(geo.ROLE) == {"biased", "unbiased"}
    assert "parity" in geo.ROLE["biased"].lower()
    assert "bias-corrected" in geo.ROLE["unbiased"].lower() or \
           "bias" in geo.ROLE["unbiased"].lower()

    # the exact note the summary must carry, verbatim
    assert geo.SUMMARY_NOTE.startswith("Biased CKA is retained for direct parity with Chronos-2")
    assert "finite-sample baseline is elevated" in geo.SUMMARY_NOTE
    assert "robustness analysis for absolute interpretability" in geo.SUMMARY_NOTE
    assert "Do NOT compare absolute biased-CKA values" in geo.CROSS_MODEL_CAVEAT

    # dropping the parity estimator is REFUSED; dropping the companion is allowed but explicit
    try:
        geo.main(["--cka-estimators", "unbiased", "--no-figures"])
    except SystemExit as e:
        assert "parity" in str(e), str(e)
    else:
        raise AssertionError("--cka-estimators without 'biased' must be REFUSED")
    # a headline split outside the computed set is refused too
    try:
        geo.main(["--cka-splits", "train", "--no-figures"])
    except SystemExit as e:
        assert "headline" in str(e).lower(), str(e)
    else:
        raise AssertionError("a headline split outside --cka-splits must be REFUSED")

    # both estimators really produce a 21x21 with unit diagonal; only biased is bounded below
    reps = [RNG.normal(size=(60, MODEL_DIMS)).astype(np.float32) for _ in range(NUM_LAYERS)]
    B = cka_layer_matrix(reps, estimator="biased")
    U = cka_layer_matrix(reps, estimator="unbiased")
    assert B.shape == U.shape == (21, 21)
    assert np.abs(np.diag(B) - 1).max() < 1e-8 and np.abs(np.diag(U) - 1).max() < 1e-8
    assert np.abs(B - B.T).max() < 1e-10 and np.abs(U - U.T).max() < 1e-10
    off = ~np.eye(21, dtype=bool)
    assert B[off].min() > 0.5, "biased sits on its floor for independent representations"
    assert abs(U[off].mean()) < 0.05, "unbiased is centred on 0 there"
    assert U[off].min() < 0, "the unbiased estimator is NOT bounded below -- not repaired"
    print(f"  38        biased (parity, floor {B[off].mean():.3f}) + unbiased (companion, "
          f"mean {U[off].mean():+.4f}) grid                    OK")


# ------------------------------------------------------------------ 40: no cache, in memory
def test_native_head_is_cache_free_and_in_memory():
    """The frozen-head transfer reads NO feature cache; it runs the backbone itself.

    Its validity rests on reproducing decode() exactly at L20, which only holds when the head is
    applied to the states decode() just produced. A cached round-trip re-applies the head under a
    different matmul shape (and possibly a different device); on TF32 hardware that is not
    bit-identical, which is why the cached implementation could not reproduce decode(). The fix
    was to remove the cache, NOT to loosen the tolerance.
    """
    import experiments.run_timesfm3_native_head_transfer as nh
    src = inspect.getsource(nh)

    for banned in ("cached_last_token_features", "read_cache", "cache_root", "cache_metadata",
                   "--cache-dir", "allow_extraction"):
        assert banned not in src, f"the native-head driver must not reference {banned}"
    assert "native_head_transfer_pass" in src and "model.decode(" in src
    assert "register_layer_hooks" in src, "it must hook the layers itself"
    assert "verify_native_head" in src, "the validated check must run verbatim on the same pass"

    # tolerances were NOT relaxed when the cache was removed
    a = nh.parse_args([])
    assert (a.recon_atol, a.recon_rtol) == (1e-5, 2e-6), (a.recon_atol, a.recon_rtol)
    assert a.loss_identity_rtol == 1e-5

    # three named stages, and the slice-order control that measures the retired approach's error
    st = inspect.getsource(nh._staged_l20_identity)
    for stage in ("raw_output_head", "pre_trend", "final_inverse_transformed"):
        assert stage in st, stage
    assert "slice_before_head_max_abs_delta" in st
    assert "states[:, :, [tok], :]" in st, "the slice-order control must apply the head to the "\
                                           "pre-sliced token, as the retired cache path did"

    # test split only -- no train/val representations are built
    rd = inspect.getsource(nh.run_dataset)
    assert 'w["X_test"]' in rd and 'w["X_train"]' not in rd and 'w["X_val"]' not in rd

    # and the GPU job exists, requesting a GPU
    job = (repo_root() / "job_timesfm3_native_head.sh").read_text()
    assert "--gres=gpu:1" in job, "the native-head job must request a GPU: it runs the backbone"
    assert "run_timesfm3_native_head_transfer" in job
    geo_job = (repo_root() / "job_timesfm3_geometry.sh").read_text()
    assert "job_timesfm3_native_head.sh" in geo_job, \
        "the CPU geometry job must point --head-only at the GPU job instead of running it"
    print("  40        frozen-head transfer is cache-free and in-memory; GPU job present; "
          "tolerances unchanged                                                         OK")


# ------------------------------------------------------------------ output locations
def test_output_locations():
    """Final paper outputs resolve INSIDE the repo; heavy artifacts default beside them."""
    root = repo_root()
    assert (root / "probing" / "cka.py").exists(), root
    assert (root / ".git").exists() or (root / "README.md").exists()
    po = paper_out_default()
    assert po == root / "results" / "timesfm3_representation_geometry"
    assert str(po).startswith(str(root / "results"))
    assert "Users" not in str(po.relative_to(root))        # never a hardcoded home path
    import experiments.run_timesfm3_representation_geometry as geo
    src = inspect.getsource(geo)
    assert "Final paper outputs" in src and "Numerical/intermediate outputs" in src
    print(f"  extra     paper outputs -> {po.relative_to(root)}                          OK")


# --------------------------------------------------------------------------- #
# model-backed: 20, 21, 22, 23, 24, 30
# --------------------------------------------------------------------------- #
def model_tests():
    from probing.timesfm3_last_token import (assert_backbone_frozen, assert_native_geometry,
                                             assert_native_quantiles, get_model)
    from experiments.run_timesfm3_native_head_transfer import (head_checksum,
                                                               native_head_transfer_pass,
                                                               parse_args)
    print("\n  [model] loading TimesFM-3 ...")
    model, device = get_model(None, None)
    assert_backbone_frozen(model)
    g = LastTokenGeometry()
    assert_native_quantiles(model)
    assert_native_geometry(model, g)

    # 20/22/24: the head IS model.output_head, frozen, Linear(1280, 576)
    ck = head_checksum(model)
    assert ck["in_features"] == MODEL_DIMS == 1280
    assert ck["out_features"] == 64 * NUM_QUANTILES == 576, ck["out_features"]
    assert ck["requires_grad"] is False
    assert all(not p.requires_grad for p in model.output_head.parameters())
    print(f"     20,22,24  model.output_head = Linear({ck['in_features']}, "
          f"{ck['out_features']}), frozen, sha256[:32] {ck['sha256']}            OK")

    # 23: the head consumes (N, 1280) and emits (N, 576) -> (N, 64, 9)
    n = 6
    h = np.random.default_rng(0).normal(size=(n, MODEL_DIMS)).astype(np.float32)
    with torch.no_grad():
        raw = model.output_head(torch.as_tensor(h, device=device))
    assert tuple(raw.shape) == (n, 576), raw.shape
    assert raw.grad_fn is None, "the frozen head must be evaluated under no_grad"
    assert tuple(raw.reshape(n, 64, 9).shape) == (n, 64, 9)
    print("     23,25     (N,1280) -> (N,576) -> (N,64,9), no grad_fn                OK")

    # 30: the endpoint identity, IN MEMORY on real states from a real decode() pass.
    # This is the contract the whole experiment rests on, and it must be EXACT here: the head
    # is applied to the states decode() just produced, with no cache round-trip in between.
    a = parse_args([])
    X = np.cumsum(np.random.default_rng(1).normal(size=(4, g.C)).astype(np.float32),
                  axis=1) + 50.0
    out = native_head_transfer_pass(model, X, geom=g, device=device,
                                    layers=list(range(NUM_LAYERS)), batch_size=4, detrend=True,
                                    recon_atol=a.recon_atol, recon_rtol=a.recon_rtol,
                                    progress=False)
    idn = out["identity"]
    st = {x["stage"]: x for x in idn["stages"]}
    assert idn["max_scaled_error"] <= 1.0, idn
    assert st["pre_trend"]["max_scaled_error"] <= 1.0, st["pre_trend"]
    assert out["native_check"]["max_abs"] == 0.0, out["native_check"]
    assert set(out["forecast"]) == set(range(NUM_LAYERS))
    assert out["forecast"][NUM_LAYERS - 1].shape == (4, g.H, 9)
    print(f"     30        L20 in-memory vs decode(): stage3 max|d| "
          f"{st['final_inverse_transformed']['max_abs_error']:.2e} (scaled "
          f"{idn['max_scaled_error']:.3g}), stage2 {st['pre_trend']['max_abs_error']:.2e}"
          f"{'  EXACT' if idn['exact'] else ''}      OK")
    print(f"     30b       slice-order control (the retired cache path's error): "
          f"{st['raw_output_head']['slice_before_head_max_abs_delta']:.2e}          OK")

    # 21: the head is byte-identical after all of that
    assert head_checksum(model)["sha256"] == ck["sha256"], "the head CHANGED during evaluation"
    print("     21        head checksum unchanged before/after                       OK")


if __name__ == "__main__":
    print("TimesFM-3 representation geometry (CKA + effective rank) and frozen-native-head "
          "contracts")
    for fn in (test_input_matrices_and_points, test_shared_origin_rank3_rejected,
               test_biased_linear_cka, test_effective_rank_definition,
               test_paper7_roster_and_labels, test_window_identity_and_split_isolation,
               test_probe_independence, test_native_head_transfer_contracts,
               test_all_points_and_datasets_are_covered, test_endpoint_identity_is_elementwise,
               test_cka_estimator_grid, test_native_head_is_cache_free_and_in_memory,
               test_output_locations):
        fn()
    if "--with-model" in sys.argv:
        model_tests()
        print("\nall contracts hold (model-backed checks included)")
    else:
        print("\nall model-free contracts hold "
              "(run with --with-model for 20-25,30 on a COMPUTE node)")
