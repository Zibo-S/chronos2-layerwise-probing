"""PRE-RUN CONTRACT SUITE for Phase-2 H3 (functional replaceability).

Model-free: no checkpoint, no dataset, no feature cache. Runs on a login node in well under a
minute with 2 threads:

    python -m tests.test_phase2_h3

Numbered so a failure names the requirement it broke. The real-architecture checks (the three
pathways against the real package forward passes on tiny random-weight models) live in
``tests.test_phase2_h4`` because they share the tiny-model builders with the truncation tests.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, phase2                                              # noqa: E402
from probing.phase2 import (QUANTILES, TUNNEL_TOLS, Phase1Cell, Phase2CellStore,  # noqa: E402
                            cluster_replicates, compatibility_tunnel, standard_wql)
from probing.phase2_align import (ResidualAdapter, affine_to_adapter,           # noqa: E402
                                  anchored_ridge_path, fit_residual_adapter, load_adapter,
                                  mse_loss, save_adapter)

torch.set_num_threads(2)


# =========================================================================== #
# ADAPTERS (1-6)
# =========================================================================== #
def test_01_residual_adapter_at_init_is_bitwise_identity():
    torch.manual_seed(0)
    x = torch.randn(7, 4, 24)
    ad = ResidualAdapter(24)
    assert torch.equal(ad(x), x), "Delta=0, b=0 must return x exactly"
    A, b = ad.affine()
    assert np.array_equal(A, np.eye(24)) and np.array_equal(b, np.zeros(24))
    assert ad.n_params == 24 * 24 + 24
    print(" 1   residual adapter at init is BITWISE the hard cut (x + 0 == x)              OK")


def test_02_affine_round_trip_and_serialization():
    rng = np.random.default_rng(1)
    A, b = np.eye(8) + 0.1 * rng.normal(size=(8, 8)), rng.normal(size=8)
    ad = affine_to_adapter(A, b)
    x = torch.randn(5, 8)
    want = x.double() @ torch.as_tensor(A).T + torch.as_tensor(b)
    assert torch.allclose(ad(x).double(), want, atol=1e-5)
    tmp = Path(tempfile.mkdtemp())
    try:
        rec = save_adapter(tmp / "a.npz", ad, {"family": "noa"})
        ad2, meta = load_adapter(rec["path"])
        assert torch.equal(ad2(x), ad(x)) and meta["sha256"] == rec["sha256"]
        # a corrupted file must fail its own checksum
        with np.load(rec["path"]) as z:
            d = {k: z[k] for k in z.files}
        d["delta"] = d["delta"] + 1e-3
        np.savez(tmp / "bad.npz", **d)
        try:
            load_adapter(tmp / "bad.npz")
            raise AssertionError("a modified adapter file passed its checksum")
        except RuntimeError:
            pass
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 2   (A, b) <-> ResidualAdapter round trip; files are checksummed + refused if edited OK")


def test_03_decay_shrinks_toward_identity_and_only_adapter_trains():
    """Decoupled decay on Delta: with a zero-gradient objective Delta -> 0, i.e. A -> I (NOT 0)."""
    torch.manual_seed(0)
    head = nn.Linear(6, 5)
    for p in head.parameters():
        p.requires_grad_(False)
    ad = ResidualAdapter(6)
    with torch.no_grad():
        ad.delta.copy_(torch.randn(6, 6))
    opt = torch.optim.AdamW([{"params": [ad.delta], "weight_decay": 10.0},
                             {"params": [ad.bias], "weight_decay": 0.0}], lr=1e-2)
    x = torch.randn(10, 6)
    for _ in range(300):
        opt.zero_grad()
        loss = (head(ad(x)) * 0.0).sum()           # zero gradient: only the decay acts
        loss.backward()
        opt.step()
    A, _ = ad.affine()
    assert np.abs(A - np.eye(6)).max() < 1e-6, np.abs(A - np.eye(6)).max()
    assert all(p.grad is None for p in head.parameters()), "the frozen head received a gradient"
    print(" 3   decay shrinks Delta so A -> I (identity anchor); frozen head gets no gradient OK")


def test_04_closed_form_equals_brute_force():
    """The eigenbasis solution solves X^T X B M + lam B = X^T Y M + lam I, for M = I and W^T W.

    vec(G B M) = (M^T (x) G) vec(B) in column-major order, so the brute force is one dense solve.
    """
    rng = np.random.default_rng(2)
    for n, d, p in ((5, 6, 3), (40, 6, 4)):               # n < d and n > d
        X = rng.normal(size=(n, d))
        Y = X @ rng.normal(size=(d, d)) * 0.3 + rng.normal(size=(n, d))
        W = rng.normal(size=(p, d))
        for metric in (None, W.T @ W):
            M = np.eye(d) if metric is None else metric
            Xc, Yc = X - X.mean(0), Y - Y.mean(0)
            G = Xc.T @ Xc
            for kappa in (1e-3, 1e-1, 10.0):
                r = anchored_ridge_path(X, Y, X, Y, metric=metric, kappas=(kappa,),
                                        include_hard_cut=False)
                lam = r["selected_lambda"]
                K = np.kron(M.T, G) + lam * np.eye(d * d)
                rhs = (Xc.T @ Yc @ M + lam * np.eye(d)).reshape(-1, order="F")
                B = np.linalg.solve(K, rhs).reshape(d, d, order="F")
                assert np.allclose(r["A"], B.T, atol=1e-8), np.abs(r["A"] - B.T).max()
                assert np.allclose(r["b"], Y.mean(0) - X.mean(0) @ B, atol=1e-8)
    print(" 4   closed form == brute-force Kronecker solve (M = I and M = W^T W; n<d, n>d)  OK")


def test_05_closed_form_limits_and_hard_cut_nesting():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(50, 5))
    # X = Y: A = I, b = 0 for every lambda (the final-depth identity)
    for k in (1e-9, 1e-3, 1.0):
        rr = anchored_ridge_path(X, X, X[:10], X[:10], kappas=(k,), include_hard_cut=False)
        assert np.abs(rr["A"] - np.eye(5)).max() < 1e-8 and np.abs(rr["b"]).max() < 1e-8
    # lambda -> infinity: A -> I (the identity anchor), b -> muY - muX
    Y = rng.normal(size=(50, 5)) + 3.0
    rr = anchored_ridge_path(X, Y, X[:10], Y[:10] * 1e6, kappas=(1e12,))
    assert np.abs(rr["A"] - np.eye(5)).max() < 1e-6 or rr["selected_hard_cut"]
    # pure-noise target: the explicit hard-cut candidate must be selectable and selected
    Z = rng.normal(size=(50, 5))
    Xv = rng.normal(size=(20, 5))
    rz = anchored_ridge_path(X, Z, Xv, Xv, kappas=(1e-6, 1e-3))
    assert rz["selected_hard_cut"], "validation Y = X: the hard cut (identity) must win"
    print(" 5   closed form: X=Y -> A=I,b=0; lam->inf -> A->I; the hard cut is a selectable "
          "candidate  OK")


def test_06_iterative_fit_nests_the_hard_cut_and_is_deterministic():
    torch.manual_seed(0)
    head = nn.Linear(8, 6)
    for p in head.parameters():
        p.requires_grad_(False)
    x_tr, x_va = torch.randn(40, 8), torch.randn(20, 8)
    # (a) targets = the head applied to the SAME states: the hard cut is already optimal
    kw = dict(d=8, pathway_fn=head, train_x=x_tr, train_y=head(x_tr), val_x=x_va,
              val_y=head(x_va), train_loss=mse_loss, val_criterion=mse_loss,
              wd_grid=(0.0, 1e-2), epochs=20, lr=1e-2, eval_every=5, device="cpu")
    r = fit_residual_adapter(**kw)
    assert r["selected_hard_cut"] and r["selected_epoch"] == 0 and r["hard_cut_val"] == 0.0
    # (b) a learnable rotation: the fit must move off the hard cut and improve validation
    Rm = torch.linalg.qr(torch.randn(8, 8))[0]
    kw2 = {**kw, "train_y": head(x_tr @ Rm.T), "val_y": head(x_va @ Rm.T), "epochs": 150}
    r2 = fit_residual_adapter(**kw2)
    r3 = fit_residual_adapter(**kw2)
    assert not r2["selected_hard_cut"] and r2["selected_val"] < r2["hard_cut_val"]
    assert torch.equal(r2["adapter"].delta, r3["adapter"].delta), "fits are not deterministic"
    print(" 6   iterative fit: epoch 0 = hard cut (selected when optimal); deterministic "
          "(bitwise)  OK")


def test_07_wd_grid_guard():
    for bad in ((100.0,), (-1.0,), (0.1, 0.1), ()):
        try:
            phase2.assert_adapter_wd_grid(bad, 1e-2)
            raise AssertionError(f"grid {bad} was accepted")
        except ValueError:
            pass
    # the frozen pairing (optimizer check, 2026-09-23): lr 1e-3 with a grid up to 100 ...
    assert phase2.assert_adapter_wd_grid(phase2.ADAPTER_WD_GRID, phase2.ADAPTER_LR)[-1] == 100.0
    # ... and the guard ties the grid to the lr: the same grid at the old lr 1e-2 hits lr*wd = 1
    try:
        phase2.assert_adapter_wd_grid(phase2.ADAPTER_WD_GRID, 1e-2)
        raise AssertionError("the frozen grid was accepted at lr 1e-2 (lr * wd = 1)")
    except ValueError:
        pass
    print(" 7   decay grid refuses lr*wd >= 1, negatives, duplicates, empty                 OK")


# =========================================================================== #
# METRICS / TUNNELS / BOOTSTRAP (8-12)
# =========================================================================== #
def test_08_standard_wql_is_the_mean_over_quantiles():
    from probing.phase1_metrics import wql_parts
    rng = np.random.default_rng(4)
    y = rng.gamma(2.0, size=(30, 16))
    qq = np.sort(rng.gamma(2.0, size=(30, 9, 16)), axis=1)
    num, den = wql_parts(y, qq, QUANTILES)
    per_q = []
    for k, tau in enumerate(QUANTILES):
        u = y - qq[:, k, :]
        per_q.append(2 * np.maximum(tau * u, (tau - 1) * u).sum() / np.abs(y).sum())
    assert np.isclose(standard_wql(num, den), np.mean(per_q))
    assert np.isclose(num.sum() / den.sum(), 9 * np.mean(per_q)), "shared parts are 9x (F5)"
    print(" 8   standard WQL = mean over the 9 levels; the shared parts sum them (9x)        OK")


def test_09_timesfm3_fl_is_the_probe_theorem():
    """rank(W) = p (full row rank)  =>  {W A} = all p x d matrices  =>  FL == probe."""
    rng = np.random.default_rng(5)
    p, d = 12, 30
    W = rng.normal(size=(p, d))
    assert np.linalg.matrix_rank(W) == p
    for _ in range(5):
        T = rng.normal(size=(p, d))                    # an arbitrary probe weight
        A = np.linalg.pinv(W) @ T                      # an adapter realizing it
        assert np.allclose(W @ A, T, atol=1e-9)
    from probing.phase2_h3 import families_for
    fams, dropped = families_for("timesfm3", ["hard", "noa", "fl", "ra"])
    assert "fl" not in fams and "full row rank" in dropped["fl"]
    fams, dropped = families_for("chronos2", ["hard", "noa", "fl"])
    assert fams == ["hard", "noa", "fl"] and not dropped
    print(" 9   TimesFM-3: W A spans every probe weight (rank theorem); FL dropped with reason OK")


def test_10_compatibility_tunnel_uses_the_phase1_operator_with_native_reference():
    from probing.tunnel import sustained_tunnel_start
    curve = [3.0, 1.2, 1.04, 1.07, 1.01, 999.0]          # last entry is REPLACED by native
    ct = compatibility_tunnel(curve, 1.0)
    assert ct["curve"][-1] == 1.0
    for t in TUNNEL_TOLS:
        assert ct["by_tolerance"][f"tol_{t:g}"] == sustained_tunnel_start(
            curve[:-1] + [1.0], t)
    assert ct["by_tolerance"]["tol_0.05"] == 4    # 1.07 at index 3 breaks the 5% band
    assert ct["by_tolerance"]["tol_0.1"] == 2     # 1.2 at index 1 breaks the 10% band
    ents = [ct["by_tolerance"][f"tol_{t:g}"] for t in sorted(TUNNEL_TOLS)]
    assert ents == sorted(ents, reverse=True), "entrance must be monotone in the tolerance"
    ct2 = compatibility_tunnel([5.0, 4.0, 1.0], 1.0, tols=(0.01,))
    assert ct2["no_truncation_at"] == ["tol_0.01"]
    print("10   compatibility tunnel = sustained operator on [V_arm(<L), V_native]; monotone OK")


def test_11_bootstrap_is_paired_and_matches_the_phase1_primitives():
    from probing.stats import cluster_bootstrap_apply, cluster_bootstrap_counts
    rng = np.random.default_rng(6)
    n, S = 60, 12
    cid = np.repeat(np.arange(S), n // S)
    Wm = rng.normal(size=(3, n))
    r1 = cluster_replicates(Wm, cid, 200, 0)
    r2 = cluster_replicates(Wm[[2, 0]], cid, 200, 0)      # different call, same dataset
    # Pairing = every call resamples the SAME clusters: an integer count matrix, bitwise.
    assert np.array_equal(cluster_bootstrap_counts(S, 200, 0), cluster_bootstrap_counts(S, 200, 0))
    # A row's replicates then agree across calls up to BLAS summation order only: a GEMM over 3
    # columns and one over 2 may use different kernels (bitwise equal on the Mac, last-ULP
    # different on Narval's FlexiBLAS) -- float noise, not unpairing.
    np.testing.assert_allclose(r1[:, 0], r2[:, 1], rtol=1e-12, atol=0)
    np.testing.assert_allclose(r1[:, 2], r2[:, 0], rtol=1e-12, atol=0)
    # ...and the check has teeth: a different resampling (seed) is NOT paired.
    r3 = cluster_replicates(Wm, cid, 200, 1)
    assert not np.allclose(r1[:, 0], r3[:, 0], rtol=1e-6, atol=0)
    M = cluster_bootstrap_counts(S, 200, 0)
    sums = np.zeros((S, 3))
    np.add.at(sums, cid, Wm.T)
    ref = cluster_bootstrap_apply(M, sums, np.bincount(cid).astype(float))
    assert np.allclose(r1, ref)
    rs = cluster_replicates(Wm, cid, 200, 0, reduce="sum")
    assert np.allclose(rs, M @ sums)
    print("11   bootstrap: one count matrix per dataset (paired across calls); == Phase-1 "
          "primitives OK")


def test_12_ratio_summary_budget_flags():
    rep_ref = np.full(1000, 1.0)
    rep = np.linspace(1.0, 1.10, 1000)
    s = phase2.ratio_summary(1.04, 1.0, rep, rep_ref)
    assert s["within_budget"] and s["budget_fragile"]
    s2 = phase2.ratio_summary(1.2, 1.0, rep + 0.2, rep_ref)
    assert not s2["within_budget"] and not s2["budget_fragile"]
    print("12   ratio summary: within-5% and 'fragile' (CI crosses the budget) flags         OK")


# =========================================================================== #
# PHASE-1 DEPENDENCY + STORE (13-15)
# =========================================================================== #
def _fake_phase1_cell(root: Path, model="tirex", tag="monash_electricity_hourly",
                      entrance=6, complete=True, qset="q9", definition=None):
    d = root / model / tag
    d.mkdir(parents=True, exist_ok=True)
    by_tol = {f"tol_{t:g}": {"depth_axis_index": entrance, "label": f"L{entrance}",
                             "test_loss_at_entrance": float("nan")} for t in TUNNEL_TOLS}
    (d / "tunnel.json").write_text(json.dumps({
        "definition": definition or phase1.TUNNEL_DEFINITION_VERSION, "by_tolerance": by_tol,
        "test_loss_by_depth": ["POISON"] * 3, "test_ratio_by_depth": None}))
    (d / "cell_config.json").write_text(json.dumps({"window_digest": "sha256:abc",
                                                    "checkpoint": "x", "quantile_set": qset}))
    np.savez(d / "bootstrap_inputs.npz", cluster_ids_test=np.arange(5))
    if complete:
        (d / "COMPLETE").write_text(json.dumps({"model": model, "dataset": tag,
                                                "config_hash": "h", "fit_hash": "f"}))
    return Phase1Cell(root, model, tag)


def test_13_phase1_reads_are_whitelisted_and_refuse_incomplete_cells():
    tmp = Path(tempfile.mkdtemp())
    try:
        c = _fake_phase1_cell(tmp)
        ent = c.entrances()
        assert ent[0.05] == {"depth_axis_index": 6, "label": "L6"}
        dep = c.dependency_record()
        assert dep["fit_hash"] == "f" and dep["sha256_tunnel.json"].startswith("sha256:")
        for kw, what in (({"complete": False}, "no COMPLETE"), ({"qset": "q1"}, "q1 cell"),
                         ({"definition": "first_crossing_v1"}, "old tunnel definition")):
            sub = tmp / what.replace(" ", "_")
            cc = _fake_phase1_cell(sub, **kw)
            try:
                cc.dependency_record() if what != "old tunnel definition" else cc.entrances()
                raise AssertionError(f"{what} was accepted")
            except phase2.Phase1DependencyError:
                pass
        # the TEST fields of tunnel.json are poison: nothing Phase 2 selects may read them
        import inspect
        src = inspect.getsource(Phase1Cell.entrances)
        assert "test_" not in src.replace("test_fields", "")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("13   Phase-1 reads: whitelisted tunnel fields only; incomplete / q1 / old-definition "
          "refused OK")


def test_14_phase2_store_is_atomic():
    tmp = Path(tempfile.mkdtemp())
    try:
        st = Phase2CellStore(tmp, "h3", "chronos2", "x", required=("a.json",))
        assert st.status("h")[0] == "pending"
        stage = st.begin()
        try:
            st.commit(stage, "h")
            raise AssertionError("commit without required artifacts succeeded")
        except RuntimeError:
            pass
        (stage / "a.json").write_text("{}")
        st.commit(stage, "h")
        assert st.status("h") == ("complete", "validated")
        assert st.status("other")[0] == "incompatible"
        assert not stage.exists() and (st.final / "COMPLETE").exists()
        st2 = st.begin()
        assert phase2.clean_staging(tmp, "h3") and not st2.exists()
        # a job cleans ONLY its own cells: another job's in-progress cell must survive
        mine = Phase2CellStore(tmp, "h3", "chronos2", "x").begin()
        other_model = tmp / "h3" / "tirex" / "x.building-99999"
        other_half = tmp / "h3" / "chronos2" / "y.building-99999"
        other_model.mkdir(parents=True)
        other_half.mkdir(parents=True)
        phase2.clean_staging(tmp, "h3", cells=[("chronos2", "x")])
        assert not mine.exists() and other_model.exists() and other_half.exists()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("14   Phase-2 store: staging -> validate -> COMPLETE -> os.replace; stale staging "
          "cleaned OK")


def test_15_overrides_parse():
    ov = phase2.parse_overrides(["LOOP_SEATTLE_5T=results/x"])
    assert phase2.resolve_phase1_root("LOOP_SEATTLE_5T", "d", ov) == Path("results/x")
    assert phase2.resolve_phase1_root("m5", "d", ov) == Path("d")
    # the superseded LOOP cells are rerouted to the lr=1e-3 rerun by default
    assert phase2.resolve_phase1_root("LOOP_SEATTLE_5T", phase2.DEFAULT_PHASE1_ROOT) == \
        phase2.PHASE1_REROUTES["LOOP_SEATTLE_5T"]
    assert phase2.resolve_phase1_root("m5", phase2.DEFAULT_PHASE1_ROOT) == \
        phase2.DEFAULT_PHASE1_ROOT
    print("15   --phase1-override routes a dataset; superseded LOOP cells reroute to the rerun OK")


# =========================================================================== #
# PATHWAY LAYOUTS (16-18)
# =========================================================================== #
def test_16_chronos2_pathway_layout_and_inverse():
    from probing.phase2_pathways import Chronos2Pathway, SplitData
    from probing.probes import CHRONOS2_QUANTILES, _apply_shared_head
    torch.manual_seed(0)
    head = nn.Linear(768, 21 * 16)
    pw = Chronos2Pathway(head, nn.Identity(), CHRONOS2_QUANTILES)
    assert pw.q9_index == [2, 4, 6, 8, 10, 12, 14, 16, 18]
    h = torch.randn(3, 4, 768)
    with torch.no_grad():
        z_all = _apply_shared_head(head, h, 21, 16, 64)          # the Phase-1/native layout
        z9 = pw.q9(pw.outputs(h))
    assert torch.equal(z9, z_all[:, pw.q9_index, :])
    X = np.random.default_rng(0).normal(size=(3, 512))
    sd = SplitData({}, None, None, X, {"mu": X.mean(1), "sd": X.std(1)}, None, None)
    raw = pw.to_raw(z9.double().numpy(), sd)
    assert np.allclose(raw, X.mean(1)[:, None, None] + X.std(1)[:, None, None]
                       * np.sinh(z9.double().numpy()))
    print("16   Chronos-2 pathway: 'b n (q p) -> b q (n p)', Q9 = native rows 2,4..18; "
          "mu+sd*sinh OK")


def test_17_timesfm3_pathway_layout_and_inverse():
    from probing.phase2_pathways import SplitData, TimesFM3Pathway
    torch.manual_seed(0)
    head = nn.Linear(1280, 576)
    pw = TimesFM3Pathway(head, value_clip=5.0, output_patch_len=64, num_quantiles=9)
    h = torch.randn(4, 1, 1280)
    with torch.no_grad():
        out = pw.outputs(h)
        z9 = pw.q9(out)
    flat = out[:, 0, :]
    for t in (0, 17, 63):
        for q in (0, 4, 8):
            assert torch.equal(z9[:, q, t], flat[:, t * 9 + q]), "not horizon-major"
    rng = np.random.default_rng(1)
    inv = {"mu": rng.normal(size=4), "sd": rng.uniform(1, 2, size=4),
           "trend": rng.normal(size=(4, 64))}
    sd = SplitData({}, None, None, None, inv, None, None)
    z = z9.double().numpy() * 10
    raw = pw.to_raw(z, sd)
    want = np.clip(z * inv["sd"][:, None, None] + inv["mu"][:, None, None], -5, 5) \
        + inv["trend"][:, None, :]
    assert np.allclose(raw, want)
    assert pw.linear_head()[0].shape == (576, 1280)
    print("17   TimesFM-3 pathway: horizon-major (t*Q+q); revin reverse -> clamp -> + trend  OK")


def test_18_tirex_pathway_layout_and_inverse():
    from probing.phase2_pathways import SplitData, TiRexPathway
    from probing.tirex_model import TiRexGeometry, denormalize
    geom = TiRexGeometry(C=512, H=64, input_patch=32, output_patch=32, train_ctx_len=2048,
                         num_quantiles=9, median_index=4)
    torch.manual_seed(0)
    head = nn.Linear(512, 288)
    pw = TiRexPathway(head, nn.Identity(), geom)
    h = torch.randn(3, 2, 512)
    with torch.no_grad():
        out = pw.outputs(h)
        z9 = pw.q9(out)
    for k in range(2):
        for q in (0, 4, 8):
            for t in (0, 31):
                assert torch.equal(z9[:, q, k * 32 + t], out[:, k, q * 32 + t]), "not q-major"
    rng = np.random.default_rng(2)
    inv = {"loc": rng.normal(size=(3, 2)), "scale": rng.uniform(1, 2, size=(3, 2))}
    sd = SplitData({}, None, None, None, inv, None, None)
    raw = pw.to_raw(z9.double().numpy(), sd)
    for q in range(9):
        assert np.allclose(raw[:, q, :], denormalize(z9[:, q, :].double().numpy(), inv["loc"],
                                                     inv["scale"], geom))
    assert not np.allclose(raw[:, :, :32] - inv["loc"][:, None, 0:1],
                           (raw[:, :, 32:] - inv["loc"][:, None, 1:2]) * 0 + 1e9)
    print("18   TiRex pathway: quantile-major (q*P+t), passes concatenated, per-pass inverse  OK")


def test_19_chronos2_cache_candidates_match_the_phase1_names():
    from probing.extraction import _cache_path
    from probing.phase2_pathways import chronos2_cache_candidates
    from probing.windows import PAPER14_SET
    c = chronos2_cache_candidates("m4_hourly", "test")
    assert c[0] == _cache_path(f"IDF_m4_hourly__{PAPER14_SET}", "test", None, "K4_H64")
    assert c[1] == _cache_path("IDF_m4_hourly", "test", None, "K4_H64")
    assert chronos2_cache_candidates("sg_carpark", "val")[0].name == \
        "IDF_sg_carpark__ood__val__clean__K4_H64.npz"
    print("19   Chronos-2 K-slot cache candidates reproduce extraction._cache_path names     OK")


# =========================================================================== #
# THE WHOLE CELL, END TO END ON A MOCK PATHWAY (20-22)
# =========================================================================== #
class _RMS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.weight


def _mock_cell(seed=0, poison_test=False):
    """A TiRex-shaped cell at d=16: h_l = rotated, noisier copies of h_L; the native pathway is a
    frozen RMS + MLP head. Alignment can recover h_L linearly, the hard cut cannot."""
    from probing.phase2_pathways import CellData, SplitData, TiRexPathway
    from probing.tirex_model import TiRexGeometry
    geom = TiRexGeometry(C=512, H=64, input_patch=32, output_patch=32, train_ctx_len=2048,
                         num_quantiles=9, median_index=4)
    torch.manual_seed(seed)
    d = 16
    head = nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Linear(32, 288))

    class MockPW(TiRexPathway):
        pass
    MockPW.d = d
    pw = MockPW(head, _RMS(d), geom)
    rng = np.random.default_rng(seed)
    spec = phase1.model_spec("tirex")
    L = spec.reference_index
    rots = [np.linalg.qr(rng.normal(size=(d, d)))[0] for _ in range(L + 1)]
    sizes = {"train": 120, "val": 40, "test": 40}
    splits = {}
    for s, n in sizes.items():
        hL = rng.normal(size=(n, 2, d)).astype(np.float32)
        feats = {}
        for l in range(L + 1):
            noise = 0.6 * (L - l) / L
            feats[l] = (hL @ rots[l].T + noise * rng.normal(size=hL.shape)).astype(np.float32) \
                if l < L else hL
        with torch.no_grad():
            z = pw.q9(pw.outputs(torch.as_tensor(hL))).numpy()
        target = (z[:, 4, :] + 0.2 * rng.normal(size=(n, 64))).astype(np.float32)
        X = np.cumsum(rng.normal(size=(n, 512)), axis=1).astype(np.float32)
        inv = {"loc": np.zeros((n, 2)), "scale": np.ones((n, 2))}
        y_raw = target.astype(np.float64)
        if poison_test and s == "test":
            target = rng.normal(size=target.shape).astype(np.float32) * 100
            y_raw = target.astype(np.float64)
        splits[s] = SplitData(feats, target, y_raw, X, inv, np.arange(n) // 4, np.arange(n))
    data = CellData("tirex", "monash_electricity_hourly", splits, spec.depth_labels, L, {})
    # Phase-1 arrays: native baseline computed by the SAME pathway, probe rows random
    from probing.phase1_metrics import raw_window_metrics
    te = splits["test"]
    with torch.no_grad():
        zt = pw.q9(pw.outputs(torch.as_tensor(te.feats[L]))).double().numpy()
    m = raw_window_metrics("monash_electricity_hourly", te.X, te.y_raw, zt, QUANTILES, 4)
    npts = spec.n_points
    arrays = {"native_loss_window": phase1.mean_pinball_per_window(zt, te.target, QUANTILES),
              "native_mase_window": m["mase_pw"],
              "test_loss_window": rng.uniform(0.2, 0.4, size=(npts, 40)),
              "val_loss_window": rng.uniform(0.2, 0.4, size=(npts, 40)),
              "test_mase_window": rng.uniform(0.8, 1.2, size=(npts, 40)),
              "test_mae_window": rng.uniform(0.8, 1.2, size=(npts, 40)),
              "test_wql_num_window": rng.uniform(1, 2, size=(npts, 40)),
              "test_wql_den_window": m["wql_den_pw"],
              "cluster_ids_test": te.cluster_ids, "cluster_ids_val": splits["val"].cluster_ids}
    ent = {t: {"depth_axis_index": 6, "label": "L6"} for t in TUNNEL_TOLS}
    return pw, data, arrays, ent


FAST = {"epochs": 40, "eval_every": 10, "wd_grid": [0.0, 1e-2], "kappas": [1e-6, 1e-3, 1e-1],
        "boot_b": 200}


def _mock_probe_predictions(pw, data, arrays):
    """Phase-1-style probe predictions for the mock cell (normalized (n, 9, H) per depth label),
    made CONSISTENT with the mock Phase-1 arrays: the probe's test MASE rows are recomputed from
    the same predictions through the same inverse, as the real Phase-1 cells were."""
    from probing.phase1_metrics import raw_window_metrics
    spec = phase1.model_spec("tirex")
    out = {}
    for s in ("val", "test"):
        sd = data.splits[s]
        pr = {"target": sd.target, "cluster_ids": sd.cluster_ids}
        for l, lab in enumerate(data.depth_labels):
            with torch.no_grad():
                _, z = pw.run(sd.feats[l])
            pr[f"pred__{lab}"] = np.asarray(z, np.float32)
            if s == "test":
                m = raw_window_metrics(data.tag, sd.X, sd.y_raw, pw.to_raw(pr[f"pred__{lab}"], sd),
                                       QUANTILES, 4)
                arrays["test_mase_window"][spec.depth_indices[l]] = m["mase_pw"]
        out[s] = pr
    return out


def test_20_whole_cell_end_to_end_on_a_mock_pathway():
    from probing.phase2_h3 import run_h3_cell, save_h3_cell
    pw, data, arrays, ent = _mock_cell()
    res = run_h3_cell(pw, data, arrays, entrances=ent, families=["hard", "noa", "fl", "ra"],
                      device="cpu", fit_cfg=FAST, adapter_dir=None, floor_val_loss=1.0,
                      save_prediction_depths=[6], log=lambda *a: None)
    v = res["verification"]
    assert v["native_gate"]["passed"] and v["hard_cut_at_final_depth_equals_native"]
    assert v["identity_adapter_equals_hard_cut_bitwise"]["bitwise"]
    rungs = res["ladder"]["headline"]["rungs"]
    assert set(rungs) == {"hard", "noa", "fl", "ra", "probe"}, set(rungs)
    # alignment must beat the hard cut at a deep layer of this construction
    row = {(r["family"], r["label"]): r for r in res["rows"]}
    assert row[("noa", "L11")]["test_loss"] < row[("hard", "L11")]["test_loss"]
    assert res["tunnels"]["complete_depth_sweep"] and "noa" in res["tunnels"]
    assert row[("noa", "L12")]["mase_ratio_vs_native"] == 1.0
    tmp = Path(tempfile.mkdtemp())
    try:
        st = Phase2CellStore(tmp, "h3", "tirex", "monash_electricity_hourly")
        stage = st.begin()
        save_h3_cell(stage, res, {"x": 1}, "hash", {"fit_hash": "f"}, {})
        st.commit(stage, "hash")
        assert st.status("hash") == ("complete", "validated")
        with np.load(st.final / "bootstrap_inputs.npz", allow_pickle=True) as z:
            assert z["noa__test_mase"].shape == (13, 40)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("20   whole H3 cell on a mock pathway: gate, identities, ladder, tunnels, atomic save OK")


def test_21_no_test_leakage_into_any_selection():
    """Replacing the TEST split with noise must leave every selected hyperparameter unchanged."""
    from probing.phase2_h3 import run_h3_cell
    sel = []
    for poison in (False, True):
        pw, data, arrays, ent = _mock_cell(poison_test=poison)
        if poison:
            arrays["native_loss_window"] = None                  # gate unavailable, not failed
            arrays["native_mase_window"] = None
        res = run_h3_cell(pw, data, arrays, entrances=ent, families=["noa", "fl", "ra"],
                          device="cpu", fit_cfg=FAST, log=lambda *a: None)
        sel.append({(f, lab): (r.get("selected_wd"), r.get("selected_epoch"),
                               r.get("selected_kappa"), r.get("adapter_sha256"))
                    for f, byl in res["fits"].items() for lab, r in byl.items()})
    assert sel[0] == sel[1], "a test-split change moved a selection"
    print("21   poisoning the TEST split changes no selected wd / epoch / lambda / adapter  OK")


def test_22_native_gate_fails_on_a_wrong_pathway():
    from probing.phase2_h3 import native_gate
    rng = np.random.default_rng(0)
    a = {"native_loss_window": rng.uniform(size=50), "native_mase_window": rng.uniform(1, 2, 50)}
    ok = native_gate(a["native_loss_window"] * (1 + 1e-7), a["native_mase_window"], a, 1e-4)
    bad = native_gate(a["native_loss_window"] * 1.01, a["native_mase_window"], a, 1e-4)
    rows = native_gate(a["native_loss_window"][:40], a["native_mase_window"][:40], a, 1e-4)
    assert ok["passed"] and not bad["passed"] and not rows["passed"]
    print("22   native gate: passes at float noise, fails on a 1% shift or a row mismatch   OK")


def test_23_tables_and_paper_outputs_regenerate_from_artifacts_only():
    """Two mock H3 cells -> combined CSVs -> LaTeX tables / figures / macros, no model needed."""
    from experiments import make_phase2_paper_tables as MP
    from experiments import make_phase2_tables as MT
    from probing.phase2_h3 import run_h3_cell, save_h3_cell
    tmp = Path(tempfile.mkdtemp())
    try:
        for tag in ("monash_electricity_hourly", "uber_tlc_hourly"):
            pw, data, arrays, ent = _mock_cell(seed=1)
            data.tag = tag
            res = run_h3_cell(pw, data, arrays, entrances=ent, families=["hard", "noa", "fl"],
                              device="cpu", fit_cfg=FAST, log=lambda *a: None)
            st = Phase2CellStore(tmp, "h3", "tirex", tag)
            stage = st.begin()
            save_h3_cell(stage, res, {"x": 1}, "h", {"fit_hash": "f"}, {})
            st.commit(stage, "h")
        r = MT.build(tmp)
        assert r["n_h3_cells"] == 2 and r["n_h4_cells"] == 0
        comb = tmp / "combined"
        for f in ("h3_depth_metrics.csv", "h3_ladder.csv", "h3_tunnels.csv",
                  "h3_summary_by_model.csv", "phase2_stats.json"):
            assert (comb / f).exists(), f
        gen = tmp / "generated"
        MP.build(tmp, gen)
        tab = (gen / "h3_ladder_table.tex").read_text()
        assert "TiRex" in tab and "Hard cut" in tab and "\\toprule" in tab
        assert "PhaseTwoMissing" in (gen / "h4_truncation_table.tex").read_text()
        assert (gen / "h3_depth_curves.pdf").stat().st_size > 1000
        assert (gen / "h3_ladder_at_entrance.pdf").stat().st_size > 1000
        assert "newcommand" in (gen / "phase2_macros.tex").read_text()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("23  combined tables + LaTeX/figures/macros regenerate from artifacts; missing H4 "
          "fails loudly OK")


def test_24_latency_tables_use_only_headline_protocol_jobs():
    """A smoke / killed latency job never enters the combined lookup; job tags are one-shot."""
    from experiments import make_phase2_tables as MT
    from experiments import run_phase2_latency as RL
    tmp = Path(tempfile.mkdtemp())
    try:
        def job(tag, eligible, hard_ms, reasons=()):
            jd = tmp / "latency" / tag
            for cid, kind, depth, med in (("000_tirex_native_L12_start", "native", 12, 10.0),
                                          ("001_tirex_hard_L06", "hard", 6, hard_ms)):
                (jd / cid).mkdir(parents=True)
                (jd / cid / "summary.json").write_text(json.dumps({
                    "config": {"model": "tirex", "kind": kind, "depth": depth},
                    "levels": {"api_e2e__B1": {"status": "ok", "median_ms": med}}}))
            (jd / "index.json").write_text(json.dumps({
                "job_tag": tag, "headline_eligible": eligible,
                "ineligible_reasons": list(reasons), "configs": []}))
        job("job1", True, 5.0)
        job("job2", True, 4.0)
        job("smoke", False, 1e-3, ["--allow-unverified", "warmup/reps 2/5 != 20/100"])
        (tmp / "latency" / "job3" / "000_tirex_native_L12_start").mkdir(parents=True)  # killed
        rows, lookup, excluded = MT.build_latency(tmp)
        hard = [r for r in lookup if r["kind"] == "hard"]
        assert len(hard) == 1 and hard[0]["n_jobs"] == 2
        assert abs(hard[0]["speedup_vs_native"] - np.median([10 / 5, 10 / 4])) < 1e-12
        assert {e["job"] for e in excluded} == {"smoke", "job3"}
        assert {r["job"] for r in rows} == {"job1", "job2"}
        try:
            RL.main(["--job-tag", "job1", "--output-root", str(tmp), "--device", "cpu"])
            raise AssertionError("a reused job tag was accepted")
        except SystemExit as exc:
            assert "already holds a job" in str(exc)
        why = RL.headline_eligibility(RL.parse_args(["--reps", "5", "--warmup", "2",
                                                     "--allow-unverified"]))
        assert any("allow-unverified" in w for w in why) and any("warmup/reps" in w for w in why)
        assert RL.headline_eligibility(RL.parse_args([])) == []
        assert RL.headline_eligibility(RL.parse_args(["--device", "cpu"])) == ["device cpu"]
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("24  latency tables use headline-protocol jobs only; smoke/killed jobs excluded + "
          "reported; a job tag is one-shot OK")


def test_25_h4_outcome_is_preregistered_and_failures_are_reported():
    """candidate depth = the frozen Phase-1 entrance; success = aligned dMASE <= 5%; a failure is
    written, counted and tabled as a failure; nothing searches another depth."""
    import csv
    from experiments import make_phase2_paper_tables as MP
    from experiments import make_phase2_tables as MT
    from experiments import run_phase2_truncation as RT
    b = phase2.BUDGET

    def rs(d, lo, hi):
        return {"ratio": 1 + d, "ratio_ci_lo": 1 + lo, "ratio_ci_hi": 1 + hi, "degradation": d,
                "degradation_ci_lo": lo, "degradation_ci_hi": hi, "within_budget": d <= b,
                "budget_fragile": lo <= b < hi}

    def arms(d, lo, hi, v3=True):
        return {"native": {"test_mase": 1.0, "test_wql": 0.1, "V3_passed": True,
                           "mase_vs_native": rs(0, 0, 0), "wql_vs_native": rs(0, 0, 0)},
                "noa": {"test_mase": 1 + d, "test_wql": 0.1, "V3_passed": v3,
                        "mase_vs_native": rs(d, lo, hi), "wql_vs_native": rs(d, lo, hi)},
                "hard": {"test_mase": 1.3, "test_wql": 0.2, "V3_passed": True,
                         "mase_vs_native": rs(0.3, 0.2, 0.4), "wql_vs_native": rs(0.3, 0.2, 0.4)}}

    kw = dict(candidate_depth=9, phase1_entrance=9, n_blocks=12, label="L9")
    o = phase2.h4_outcome(arms(0.02, 0.01, 0.04), **kw)
    assert (o["status"], o["successful_truncation"], o["fragile"]) == ("success", True, False)
    assert o["rule"]["version"] == phase2.H4_OUTCOME_RULE["version"]
    assert o["depth_search_performed"] is False and o["blocks_removed"] == 3
    o = phase2.h4_outcome(arms(0.04, 0.02, 0.07), **kw)
    assert (o["status"], o["fragile"]) == ("success", True)
    o = phase2.h4_outcome(arms(0.08, 0.06, 0.11), **kw)
    assert (o["status"], o["successful_truncation"]) == ("failure", False)
    o = phase2.h4_outcome(arms(0.01, 0.0, 0.02, v3=False), **kw)
    assert (o["status"], o["successful_truncation"]) == ("invalid_v3", None)
    o = phase2.h4_outcome(arms(0.0, 0.0, 0.0), candidate_depth=12, phase1_entrance=12,
                          n_blocks=12, label="L12")
    assert (o["status"], o["successful_truncation"]) == ("no_truncation_possible", None)
    for bad in ({"candidate_depth": 8, "phase1_entrance": 9},       # not the Phase-1 entrance
                {"candidate_depth": 9, "phase1_entrance": 9, "no_noa": True}):
        a = arms(0.01, 0.0, 0.02)
        if bad.pop("no_noa", False):
            del a["noa"]
        try:
            phase2.h4_outcome(a, n_blocks=12, label="Lx", **bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    try:
        RT.main(["evaluate", "--depths", "3"])
        raise AssertionError("evaluate accepted --depths")
    except SystemExit as exc:
        assert "exactly one operating point" in str(exc)

    tmp = Path(tempfile.mkdtemp())
    try:
        def cell(tag, a, depth, L=12):
            o = phase2.h4_outcome(a, candidate_depth=depth, phase1_entrance=depth, n_blocks=L,
                                  label=f"L{depth}", low_skill=False)
            for name, r in a.items():
                r.setdefault("blocks_removed", 0 if name == "native" else L - depth)
                r.setdefault("depth_index", None if name == "native" else depth)
            d = tmp / "h4" / "tirex" / tag
            d.mkdir(parents=True)
            (d / "truncation.json").write_text(json.dumps({
                "model": "tirex", "dataset": tag, "depth_index": depth, "blocks_total": L,
                "label": f"L{depth}", "arms": a, "h4_outcome": o}))
            (d / "COMPLETE").write_text("{}")
        cell("monash_electricity_hourly", arms(0.02, 0.01, 0.04), 9)
        cell("uber_tlc_hourly", arms(0.08, 0.06, 0.11), 9)                 # a FAILURE
        cell("m4_hourly", arms(0.0, 0.0, 0.0), 12)                         # entrance = final
        MT.build(tmp)
        comb = tmp / "combined"
        with open(comb / "h4_outcomes.csv") as fh:
            got = {r["dataset"]: r["status"] for r in csv.DictReader(fh)}
        assert got == {"monash_electricity_hourly": "success", "uber_tlc_hourly": "failure",
                       "m4_hourly": "no_truncation_possible"}, got
        st = json.loads((comb / "phase2_stats.json").read_text())
        s = next(x for x in st["h4_outcome_summary"] if x["includes_low_skill"])
        assert (s["n_success"], s["n_failure"], s["n_no_truncation_possible"],
                s["n_truncatable"]) == (1, 1, 1, 2) and s["failed_datasets"] == "uber_tlc_hourly"
        gen = tmp / "generated"
        MP.build(tmp, gen)
        tab = (gen / "h4_outcome_table.tex").read_text()
        assert "TiRex & 3 & 2 & 1 & 1 (0) & 1" in tab, tab
        noa_row = next(x for x in (gen / "h4_truncation_table.tex").read_text().splitlines()
                       if "Label-free aligned truncation" in x)
        assert noa_row.rstrip(" \\").endswith("1/2"), noa_row      # no vacuous 3rd 'success'
        assert "HfourFailure}{1/2}" in (gen / "phase2_macros.tex").read_text()
        cell("coastal_ts", arms(0.01, 0.0, 0.02, v3=False), 1)             # physical != offline
        MT.build(tmp)
        MP.build(tmp, gen)
        assert json.loads((comb / "phase2_stats.json").read_text())["h4_invalid_cells"] == [
            "tirex/coastal_ts"]
        for f in ("h4_truncation_table.tex", "h4_outcome_table.tex"):
            t = (gen / f).read_text()
            assert "PhaseTwoMissing" in t and "INVALID" in t, t
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("25  H4 outcome: one frozen candidate depth, success = aligned dMASE <= 5%, failures "
          "reported and counted, no depth search, invalid cells block the tables OK")


def test_26_frontier_budget_rule_is_validation_only_and_sustained():
    """The H4 main result's operating points: shallowest SUSTAINED depth within the validation
    MASE budget, the full model when none qualifies, never a test number."""
    from probing.phase2_h3 import run_h3_cell
    bd = phase2.budget_depth
    # sustained: an early lucky depth does not open the budget if a deeper cut breaks it
    r = [1.02, 1.30, 1.04, 1.03, 1.01, 1.0]           # depths 0..4 are cuts, 5 = the full model
    assert bd(r, 0.05) == 2 and bd(r, 0.35) == 0 and bd(r, 0.005) == 5
    assert bd([1.5, np.nan, 1.01, 1.0], 0.05) == 2 and bd([1.0, 1.0, np.nan, 1.0], 0.5) == 3
    assert bd([1.2, 1.1, 1.0], 0.0) == 2               # no cut fits a zero budget: full model
    assert bd([1.2, 1.0, 1.0], 0.0) == 1               # a ratio of exactly 1 + eps qualifies
    rng = np.random.default_rng(0)
    for _ in range(300):                               # monotone: a bigger budget never cuts later
        v = np.r_[1.0 + np.abs(rng.normal(0, 0.2, 9)), 1.0]
        ds = [bd(v, e) for e in phase2.FRONTIER_BUDGETS]
        assert ds == sorted(ds, reverse=True), (v, ds)

    # the whole cell: every arm gets a frontier; the probe's validation MASE comes through the
    # SAME inverse that reproduces Phase 1's probe test MASE (the probe inverse gate)
    pw, data, arrays, ent = _mock_cell()
    preds = _mock_probe_predictions(pw, data, arrays)
    res = run_h3_cell(pw, data, arrays, entrances=ent, families=["hard", "noa", "fl"],
                      device="cpu", fit_cfg=FAST, phase1_predictions=preds, log=lambda *a: None)
    assert res["verification"]["probe_inverse_gate"]["passed"]
    fr = res["frontier"]
    assert fr["complete_depth_sweep"] and set(fr["arms"]) == {"hard", "noa", "fl", "probe"}
    L = fr["reference_index"]
    for f, rec in fr["arms"].items():
        v = rec["val_mase_ratio_by_depth"]
        for key, op in rec["budgets"].items():
            eps = float(key.split("_")[1])
            if op["truncated"]:
                assert all(v[j] is not None and v[j] <= 1 + eps for j in range(op["depth_index"], L))
                assert np.isclose(op["test_mase_ratio"],
                                  rec["test_mase_ratio_by_depth"][op["depth_index"]])
            else:
                assert op["depth_index"] == L and op["operating_model"] == "native"
    # validation-only: poisoning the TEST split moves no budget depth
    pw2, data2, arrays2, ent2 = _mock_cell(poison_test=True)
    arrays2["native_loss_window"] = arrays2["native_mase_window"] = None
    preds2 = _mock_probe_predictions(pw2, data2, arrays2)
    res2 = run_h3_cell(pw2, data2, arrays2, entrances=ent2, families=["hard", "noa", "fl"],
                       device="cpu", fit_cfg=FAST, phase1_predictions=preds2, log=lambda *a: None)
    sel = lambda rr: {(f, k): op["depth_index"] for f, rec in rr["frontier"]["arms"].items()  # noqa
                      for k, op in rec["budgets"].items()}
    assert sel(res) == sel(res2), "a test-split change moved a budget operating point"
    # the inverse gate has teeth: a 1% scale error between the pathway inverse and Phase 1 aborts
    arrays_bad = dict(arrays)
    arrays_bad["test_mase_window"] = arrays["test_mase_window"] * 1.01
    try:
        run_h3_cell(pw, data, arrays_bad, entrances=ent, families=["hard"], device="cpu",
                    fit_cfg=FAST, phase1_predictions=preds, log=lambda *a: None)
        raise AssertionError("a 1% probe-MASE mismatch passed the inverse gate")
    except RuntimeError as exc:
        assert "inverse" in str(exc)
    # misaligned Phase-1 rows are refused, never silently re-aligned
    bad = {s: dict(p) for s, p in preds.items()}
    bad["val"]["target"] = bad["val"]["target"][::-1].copy()
    try:
        run_h3_cell(pw, data, arrays, entrances=ent, families=["hard"], device="cpu",
                    fit_cfg=FAST, phase1_predictions=bad, log=lambda *a: None)
        raise AssertionError("misaligned probe predictions were accepted")
    except phase2.Phase1DependencyError:
        pass
    print("26  frontier budgets: shallowest SUSTAINED depth within the validation-MASE budget, "
          "full-model fallback, monotone in the budget, test-blind; probe via the gated inverse OK")


def test_27_frontier_tables_and_figure_from_artifacts():
    """The H4 main outputs rebuild from saved cells only: frontier points / summary, budget
    points / summary, Figure H4-1, Table H4-1 and the budget macros -- first with no latency yet
    (the axis falls back to blocks removed), then joined with a headline latency job."""
    import csv
    from experiments import make_phase2_paper_tables as MP
    from experiments import make_phase2_tables as MT
    from probing.phase2_h3 import run_h3_cell, save_h3_cell
    tmp = Path(tempfile.mkdtemp())
    try:
        for tag in ("monash_electricity_hourly", "uber_tlc_hourly"):
            pw, data, arrays, ent = _mock_cell(seed=1)
            data.tag = tag
            preds = _mock_probe_predictions(pw, data, arrays)
            res = run_h3_cell(pw, data, arrays, entrances=ent, families=["hard", "noa", "fl"],
                              device="cpu", fit_cfg=FAST, phase1_predictions=preds,
                              log=lambda *a: None)
            st = Phase2CellStore(tmp, "h3", "tirex", tag)
            stage = st.begin()
            save_h3_cell(stage, res, {"x": 1}, "h", {"fit_hash": "f"}, {})
            st.commit(stage, "h")
        comb, gen = tmp / "combined", tmp / "generated"
        MT.build(tmp)
        for f in ("h4_frontier_points.csv", "h4_frontier_summary.csv", "h4_budget_points.csv",
                  "h4_budget_summary.csv"):
            assert (comb / f).exists(), f
        MP.build(tmp, gen)
        assert (gen / "h4_frontier.pdf").stat().st_size > 1000
        assert (gen / "h4_frontier_appendix.pdf").stat().st_size > 1000
        tab = (gen / "h4_budget_table.tex").read_text()
        assert "TiRex" in tab and "10\\% budget" in tab and "Probe head" in tab, tab
        assert "\\phtwotirexnoaBudgetTenCut" in (gen / "phase2_macros.tex").read_text()

        # the mock's arms never fit a budget; give one cell two real cuts so the join is exercised
        fp = tmp / "h3" / "tirex" / "uber_tlc_hourly" / "frontier.json"
        fr = json.loads(fp.read_text())
        for key, d in (("eps_0.1", 9), ("eps_0.2", 6)):
            fr["arms"]["noa"]["budgets"][key] = {
                "depth_index": d, "label": f"L{d}", "truncated": True, "blocks_removed": 12 - d,
                "operating_model": "noa", "val_mase_ratio": 1.05, "test_mase_ratio": 1.07,
                "test_mase_ci": [1.03, 1.11], "test_wql_ratio": 1.06, "within_budget_test": True}
        fp.write_text(json.dumps(fr))
        # a headline latency job: time = 10 ms x (0.2 + 0.8 d / L) for every truncated config
        L, jd = 12, tmp / "latency" / "job1"

        def config(cid, kind, depth, ms):
            (jd / cid).mkdir(parents=True)
            lv = {k: {"status": "ok", "median_ms": ms, "p95_ms": ms, "iqr_ms": 0.0,
                      "throughput_series_per_s": 1.0, "peak_allocated_bytes": 1}
                  for k in ("api_e2e__B1", "api_e2e__B256", "device_forward__B1")}
            (jd / cid / "summary.json").write_text(json.dumps({
                "config": {"model": "tirex", "kind": kind, "depth": depth}, "levels": lv,
                "params": {"active_params": int(1000 * ms / 10)}}))
        config("000_native", "native", L, 10.0)
        for d in range(L + 1):
            for kind in ("hard", "adapter", "probe_head"):
                config(f"{kind}_{d:02d}", kind, d, 10.0 * (0.2 + 0.8 * d / L))
        (jd / "index.json").write_text(json.dumps({"job_tag": "job1", "headline_eligible": True,
                                                   "ineligible_reasons": [], "configs": []}))
        MT.build(tmp)
        with open(comb / "h4_budget_points.csv") as fh:
            rows = list(csv.DictReader(fh))
        assert sum(r["truncated"] == "True" for r in rows) == 2
        for r in rows:
            sp = float(r["speedup_e2e_b1"])
            if r["truncated"] == "True":                 # a cut is measured, and it is faster
                assert sp > 1.0 and np.isclose(
                    sp, 1.0 / (0.2 + 0.8 * int(r["depth_index"]) / L)), r
            else:                                        # no qualifying cut: the full model, 1x
                assert sp == 1.0 and int(r["depth_index"]) == L
        MP.build(tmp, gen)
        assert (gen / "h4_frontier.pdf").stat().st_size > 1000

        # the physical stage instantiates exactly the frozen truncated points, nothing else
        ops = phase2.budget_operating_points(json.loads(fp.read_text()))
        assert [(o["arm"], o["depth_index"], o["budgets"]) for o in ops] == [
            ("noa", 6, [0.2]), ("noa", 9, [0.1])], ops
        try:
            phase2.budget_operating_points({"complete_depth_sweep": False, "arms": {}})
            raise AssertionError("a smoke (depth-subset) frontier yielded operating points")
        except ValueError:
            pass
        from experiments import run_phase2_truncation as RT
        for mode in ("entrance", "budget"):
            try:
                RT.main(["evaluate", "--operating-points", mode, "--depths", "3"])
                raise AssertionError("evaluate accepted --depths")
            except SystemExit as exc:
                assert "no depth option" in str(exc)

        # physical verification gates the main table: one failed V3 point blocks it
        def phys_cell(ok):
            d = tmp / "h4_budget" / "tirex" / "uber_tlc_hourly"
            shutil.rmtree(d, ignore_errors=True)
            d.mkdir(parents=True)
            pts = [{**o, "physical_test_mase": 1.0, "offline_test_mase": 1.0,
                    "V3": {"mase_mean_rel_diff": 0.0 if ok else 0.02}, "V3_passed": ok,
                    "active_params": 800, "fraction_params_removed": 0.2} for o in ops]
            (d / "operating_points.json").write_text(json.dumps({
                "model": "tirex", "dataset": "uber_tlc_hourly", "operating_points": pts}))
            (d / "COMPLETE").write_text("{}")
        phys_cell(True)
        MT.build(tmp)
        MP.build(tmp, gen)
        assert "verified operating points: 2/2" in (gen / "h4_budget_table.tex").read_text()
        phys_cell(False)
        MT.build(tmp)
        MP.build(tmp, gen)
        t = (gen / "h4_budget_table.tex").read_text()
        assert "PhaseTwoMissing" in t and "failed V3" in t, t
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("27  H4 frontier + budget tables, Figure H4-1 and macros rebuild from artifacts; "
          "speedups join by (model, configuration, depth); no-cut = the full model at 1x; the "
          "physical stage takes only the frozen points and a V3 failure blocks the table OK")


# =========================================================================== #
def main(argv=None):
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"\nPHASE-2 H3 CONTRACTS  ({len(tests)} model-free groups)\n" + "=" * 78)
    failed = []
    for t in tests:
        try:
            t()
        except Exception:                 # report EVERY failing contract, not only the first
            failed.append(t.__name__)
            print(f"FAIL {t.__name__}")
            traceback.print_exc()
    print("=" * 78)
    if failed:
        print(f"{len(failed)} of {len(tests)} H3 contracts FAILED: {', '.join(failed)}")
        return 1
    print(f"all {len(tests)} H3 contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
