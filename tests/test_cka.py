"""CPU/synthetic contracts for the linear-CKA utility (probing/cka.py).

No model, no GPU, no cache, no torch — pure numpy, so this runs on a login node:

    OMP_NUM_THREADS=2 python -m tests.test_cka

Covers the CKA invariants (identity / orthogonal / isotropic-scale ~ 1; unrelated substantially
lower), the matched-row guard (the structural block against cross-dataset row pairing), the
matrix/diagonal helpers' ordering + shapes, deterministic fslot slot-stacking, fail-loud cache/
manifest loading, and a figure smoke test.
"""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import tempfile
from pathlib import Path

import numpy as np

from probing import cka


def _rng(seed=0):
    return np.random.default_rng(seed)


def test_cka_identity_and_unrelated():
    X = _rng(1).standard_normal((200, 32))
    assert abs(cka.linear_cka(X, X) - 1.0) < 1e-8
    # two independent random matrices are substantially less than identity
    Y = _rng(2).standard_normal((200, 40))
    assert cka.linear_cka(X, Y) < 0.5


def test_orthogonal_invariance():
    X = _rng(3).standard_normal((150, 24))
    Q, _ = np.linalg.qr(_rng(4).standard_normal((24, 24)))     # orthogonal 24x24
    assert abs(cka.linear_cka(X, X @ Q) - 1.0) < 1e-8


def test_isotropic_scale_invariance():
    X = _rng(5).standard_normal((120, 16))
    assert abs(cka.linear_cka(X, 3.7 * X) - 1.0) < 1e-8
    # a per-example additive shift is also removed by centering
    assert abs(cka.linear_cka(X, X + 5.0) - 1.0) < 1e-8


def test_row_mismatch_fails():
    X = _rng(6).standard_normal((100, 8))
    Y = _rng(7).standard_normal((99, 8))
    try:
        cka.linear_cka(X, Y)
    except ValueError:
        return
    raise AssertionError("expected ValueError on row-count mismatch")


def test_deterministic():
    X = _rng(8).standard_normal((80, 12))
    Y = _rng(9).standard_normal((80, 20))
    assert cka.linear_cka(X, Y) == cka.linear_cka(X, Y)
    reps = [_rng(k).standard_normal((80, 12)) for k in range(4)]
    assert np.array_equal(cka.cka_matrix(reps), cka.cka_matrix(reps))


def test_correct_layer_ordering():
    A, B, C = (_rng(k).standard_normal((60, 10)) for k in (10, 11, 12))
    M = cka.cka_matrix([A, B, C])
    assert M.shape == (3, 3)
    assert np.allclose(M, M.T)                                    # symmetric
    assert np.allclose(np.diag(M), 1.0, atol=1e-8)               # diagonal = 1
    assert abs(M[0, 1] - cka.linear_cka(A, B)) < 1e-10           # index maps to input order
    assert abs(M[1, 2] - cka.linear_cka(B, C)) < 1e-10
    # rectangular ordering
    D = _rng(13).standard_normal((60, 10))
    R = cka.cka_matrix(rows=[A, B], cols=[C, D])
    assert abs(R[0, 1] - cka.linear_cka(A, D)) < 1e-10


def test_fslot_stacking_row_order_across_stages():
    n, Kk, d = 3, 2, 4
    # value encodes (window, slot) so we can check the exact fold order
    arrA = np.zeros((n, Kk, d)); arrB = np.zeros((n, Kk, d))
    for w in range(n):
        for k in range(Kk):
            arrA[w, k, :] = 100 * w + k
            arrB[w, k, :] = 100 * w + k + 0.5      # different values, SAME (w,k) layout
    SA, SB = cka.stack_slots(arrA), cka.stack_slots(arrB)
    assert SA.shape == (n * Kk, d) and SB.shape == (n * Kk, d)
    for i in range(n * Kk):
        w, k = divmod(i, Kk)                        # row-major fold: row i == window w, slot k
        assert np.all(SA[i] == 100 * w + k)
        assert np.all(SB[i] == 100 * w + k + 0.5)   # identical positional mapping across "stages"
    # a 2-D array (content pooling) passes through unchanged
    flat = _rng(14).standard_normal((7, 4))
    assert np.array_equal(cka.stack_slots(flat), flat)


def test_cross_stage_matrix_shape():
    L = 14
    rows = [_rng(k).standard_normal((90, 18)) for k in range(L)]
    cols = [_rng(100 + k).standard_normal((90, 18)) for k in range(L)]
    M = cka.cka_matrix(rows=rows, cols=cols)
    assert M.shape == (L, L)


def test_same_layer_diagonal_extraction():
    rows = [_rng(k).standard_normal((70, 9)) for k in range(5)]
    cols = [_rng(50 + k).standard_normal((70, 9)) for k in range(5)]
    diag = cka.same_layer_diagonal(rows, cols)
    M = cka.cka_matrix(rows=rows, cols=cols)
    assert np.allclose(diag, np.diagonal(M), atol=1e-10)
    assert diag.shape == (5,)


def test_no_cross_dataset_pairing_guard():
    # different datasets => different n => require_matched_rows must refuse (no positional pairing)
    a = _rng(1).standard_normal((262, 32))
    b = _rng(2).standard_normal((1320, 32))
    for fn in (lambda: cka.require_matched_rows([a, b]),
               lambda: cka.cka_matrix(rows=[a], cols=[b])):
        try:
            fn(); raise AssertionError("expected ValueError pairing two different-n datasets")
        except ValueError:
            pass


def test_cache_and_manifest_fail_loud():
    with tempfile.TemporaryDirectory() as td:
        missing = Path(td) / "nope.npz"
        try:
            cka.load_npz_reps(missing, ["layer_0"]); raise AssertionError("expected FileNotFoundError")
        except FileNotFoundError:
            pass
        good = Path(td) / "ok.npz"
        np.savez(good, layer_0=np.zeros((4, 3)), layer_1=np.ones((4, 3)))
        reps = cka.load_npz_reps(good, ["layer_1", "layer_0"])
        assert len(reps) == 2 and np.all(reps[0] == 1) and np.all(reps[1] == 0)
        try:
            cka.load_npz_reps(good, ["layer_9"]); raise AssertionError("expected KeyError")
        except KeyError:
            pass
    manifest = {"checkpoints": {"stage1_ft_early": {"checkpoint_hash": "abc12345"}}}
    assert cka.stage_hash_from_manifest(manifest, "stage1_ft_early") == "abc12345"
    try:
        cka.stage_hash_from_manifest(manifest, "stage2_ft_late"); raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass


def test_figure_smoke():
    with tempfile.TemporaryDirectory() as td:
        M = _rng(0).random((6, 6))
        p1 = cka.heatmap(M, list("abcdef"), list("abcdef"), Path(td) / "hm.png", title="smoke")
        assert p1.exists() and p1.with_suffix(".pdf").exists()
        p2 = cka.drift_curve([("early", np.linspace(1, 0.5, 6), "tab:orange"),
                              ("late", np.linspace(1, 0.3, 6), "tab:red")],
                             list("abcdef"), Path(td) / "curve.png", title="smoke")
        assert p2.exists() and p2.with_suffix(".pdf").exists()
        p3 = cka.save_matrix_csv(M, list("abcdef"), list("abcdef"), Path(td) / "m.csv")
        assert p3.exists() and p3.read_text().count("\n") == 7   # header + 6 rows


# --------------------------------------------------------------------------- #
# ext_v4 forecast-slot branch (reproducible replacement for the ad-hoc matrices)
# --------------------------------------------------------------------------- #
def test_extv4_fslot_split_names():
    """PT-ID caches are bare 'train'/'test'; PT-OOD rolling caches carry the '_rolling' suffix."""
    from experiments.run_cka_analysis import _fslot_split
    assert _fslot_split("m4_hourly", "test") == "test"
    assert _fslot_split("m4_hourly", "train") == "train"
    assert _fslot_split("boom_hourly", "test") == "test_rolling"
    assert _fslot_split("sg_carpark", "train") == "train_rolling"
    for bad in ("val", "TEST", ""):
        try:
            _fslot_split("m4_hourly", bad)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for split={bad!r}")
    print("  PASS test_extv4_fslot_split_names")


def test_extv4_fslot_roster_covers_all_seven():
    from experiments.run_cka_analysis import EXTV4_TAGS, PT_ID_TAGS
    assert len(EXTV4_TAGS) == 7 and len(set(EXTV4_TAGS)) == 7
    assert PT_ID_TAGS.issubset(set(EXTV4_TAGS))
    assert {"sg_carpark", "coastal_ts", "boom_hourly"}.issubset(set(EXTV4_TAGS))
    print("  PASS test_extv4_fslot_roster_covers_all_seven")


def test_extv4_fslot_matches_stage0_reader():
    """The new reader must address the SAME cache as read_fslot_reps(stage0_pretrained), so the
    replacement matrices are the same representation the fslot probes and the FT stage0 use."""
    from experiments.run_cka_analysis import _fslot_split, _fcast_prefix, _fcast_split, FSLOT_POOL
    from probing.extraction import _cache_path, _idf_prefix
    for tag in ("m4_hourly", "boom_hourly"):
        new = _cache_path(_idf_prefix(tag), _fslot_split(tag, "test"), None, FSLOT_POOL)
        old = _cache_path(_fcast_prefix(tag, "boom", "stage0_pretrained", None),
                          _fcast_split(tag), None, FSLOT_POOL)
        assert new == old, f"{tag}: {new} != {old}"
    print("  PASS test_extv4_fslot_matches_stage0_reader")


def test_extv4_fslot_end_to_end_synthetic(tmp_root=None):
    """Write synthetic 14-key fslot caches, run the branch, check every artifact + provenance."""
    import json as _json, pathlib, tempfile, numpy as _np
    from unittest import mock
    import experiments.run_cka_analysis as R
    tags = ["m4_hourly", "boom_hourly"]
    n, K, d = 12, 4, 9
    rng = _np.random.default_rng(0)
    reps = {t: [rng.normal(size=(n, K, d)) for _ in range(14)] for t in tags}
    with tempfile.TemporaryDirectory() as td:
        with mock.patch.object(R, "OUT", pathlib.Path(td)), \
             mock.patch.object(R, "read_extv4_fslot_reps",
                               lambda tag, split: [R.cka.stack_slots(a) for a in reps[tag]]):
            R.run_extv4_fslot(max_rows=None, seed=0, split="test", tags=tags)
            root = pathlib.Path(td) / "ext_v4_future_tokens_fslot"
            for t in tags:
                M = _np.load(root / "matrices" / f"{t}__fslot__layerxlayer.npy")
                assert M.shape == (14, 14), M.shape
                assert _np.allclose(_np.diag(M), 1.0), "unit diagonal"
                assert _np.allclose(M, M.T), "symmetry"
                assert (M >= -1e-9).all() and (M <= 1 + 1e-9).all(), "CKA in [0,1]"
                assert (root / "tables" / f"{t}__fslot__layerxlayer.csv").exists()
                assert (root / "figures" / f"{t}__fslot__layerxlayer.png").exists()
            prov = _json.load(open(root / "provenance.json"))
            assert set(prov["per_dataset"]) == set(tags)
            assert prov["per_dataset"]["m4_hourly"]["cache_split"] == "test"
            assert prov["per_dataset"]["boom_hourly"]["cache_split"] == "test_rolling"
            assert prov["per_dataset"]["m4_hourly"]["rows_used"] == n * K
            assert prov["seed"] == 0 and prov["requested_split"] == "test"
    print("  PASS test_extv4_fslot_end_to_end_synthetic")


if __name__ == "__main__":
    tests = [
        test_cka_identity_and_unrelated,
        test_orthogonal_invariance,
        test_isotropic_scale_invariance,
        test_row_mismatch_fails,
        test_deterministic,
        test_correct_layer_ordering,
        test_fslot_stacking_row_order_across_stages,
        test_cross_stage_matrix_shape,
        test_same_layer_diagonal_extraction,
        test_no_cross_dataset_pairing_guard,
        test_cache_and_manifest_fail_loud,
        test_extv4_fslot_split_names,
        test_extv4_fslot_roster_covers_all_seven,
        test_extv4_fslot_matches_stage0_reader,
        test_extv4_fslot_end_to_end_synthetic,
        test_figure_smoke,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} CKA tests passed.")
