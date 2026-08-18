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
        test_figure_smoke,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} CKA tests passed.")
