"""CONTRACTS for the expanded weight-decay search and the SUSTAINED forecasting tunnel.

Lettered A-N to match the specification that asked for them, so a failure names the requirement
it broke. Model-free: runs on a login node in seconds, needs no checkpoint, no dataset and no
feature cache.

    python -m tests.test_phase1_wd_and_sustained_tunnel

These sit BESIDE tests/test_three_model_phase1.py (57 numbered contracts), which is unchanged
except for 33 and 35 -- the two that encoded the old first-crossing entrance and now pin both
statistics.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1                                                      # noqa: E402
from probing.phase1 import MODELS, model_spec                                   # noqa: E402
from probing.tunnel import (assert_tolerance_monotone, suffix_excursion,        # noqa: E402
                            sustained_tunnel_start, tunnel_start)

LR = 1e-2


# =========================================================================== #
# A-C   THE EXPANDED WEIGHT-DECAY GRID
# =========================================================================== #
def test_A_grid_contains_every_old_candidate_plus_stronger_ones():
    """The expanded grid RETAINS all three legacy grids and adds only stronger candidates."""
    from probing.probes import WD_GRID_V2
    from probing.timesfm3_last_token_probes import WD_GRID_LAST_TOKEN
    from probing.tirex_probes import WD_GRID_TIREX

    g = phase1.PHASE1_WD_GRID
    for name, legacy in (("WD_GRID_V2 (chronos2)", WD_GRID_V2),
                         ("WD_GRID_LAST_TOKEN (timesfm3)", WD_GRID_LAST_TOKEN),
                         ("WD_GRID_TIREX (tirex)", WD_GRID_TIREX)):
        assert phase1.wd_grid_is_superset_of(g, legacy), f"{name} is not retained by {g}"
    added = sorted(set(map(float, g)) - set(map(float, WD_GRID_TIREX)))
    assert added, "the grid was not expanded at all"
    assert min(added) > max(WD_GRID_TIREX), \
        f"the new candidates {added} must be STRONGER than the old maximum {max(WD_GRID_TIREX)}"
    assert list(g) == sorted(g), "the grid must be sorted"
    assert len(set(g)) == len(g), "duplicate candidates"
    # the legacy grids themselves are UNTOUCHED, so no committed non-Phase-1 result moves
    assert tuple(WD_GRID_V2) == (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0)
    assert tuple(WD_GRID_LAST_TOKEN) == tuple(WD_GRID_V2) + (10.0, 30.0)
    assert tuple(WD_GRID_TIREX) == tuple(WD_GRID_V2) + (10.0, 30.0)
    # and ALL THREE Phase-1 adapters now use the ONE grid
    import probing.phase1_chronos2 as c2
    import probing.phase1_timesfm3 as t3
    import probing.phase1_tirex as tx
    assert tuple(c2.WD_GRID) == tuple(t3.WD_GRID) == tuple(tx.WD_GRID) == tuple(g), \
        "the three models must search the SAME grid, or a model difference could be a grid one"
    print(f" A   expanded grid {list(g)} retains all 3 legacy grids, adds only {added}, "
          "and is shared by all three models  OK")


def test_B_validation_can_select_a_newly_added_high_wd_candidate():
    """A newly added candidate is REACHABLE: when validation prefers it, selection returns it.

    Run against the real TimesFM-3 fitting path on synthetic features, not a stub: the point is
    that nothing downstream silently caps the search at the old maximum.
    """
    from probing.timesfm3_last_token_probes import last_token_layerwise

    rng = np.random.default_rng(0)
    d, H, n = 1280, 8, 60
    q = np.array([0.5])
    # Pure-noise targets: no linear map helps, so the pinball-optimal fit is (near) bias-only
    # and validation prefers the STRONGEST candidate available -- which is now a new one.
    F = {s_: {0: rng.normal(size=(n, d)).astype(np.float32)} for s_ in ("tr", "va", "te")}
    y = {s_: rng.normal(size=(n, H)).astype(np.float32) for s_ in ("tr", "va", "te")}
    ok = np.ones(n, bool)
    grid = phase1.PHASE1_WD_GRID
    _scores, diag = last_token_layerwise(
        F["tr"], y["tr"], ok, F["te"], y["te"], ok,
        val_feats=F["va"], val_targets=y["va"], val_valid=ok,
        H=H, epochs=60, lr=LR, quantiles=q, wd_grid=grid, device="cpu",
        null_wd=None, batch_size=0, layers=[0], verbose=False)
    out = {"wd": diag["wd"][0]}
    sel = diag["selection"][0]["val_loss_by_wd"]
    assert set(map(float, sel)) == set(map(float, grid)), \
        f"every candidate must be scored; got {sorted(sel)}"
    chosen = float(out["wd"])
    assert chosen in set(map(float, grid))
    new = sorted(set(map(float, grid)) - {1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0})
    assert chosen in new, (
        f"on pure-noise targets the strongest candidate should win, but wd={chosen:g} was "
        f"selected and the new candidates {new} were not reachable")
    print(f" B   a NEW high-wd candidate is selectable and was selected (wd={chosen:g}) on a "
          "target with no signal  OK")


def test_C_wd_at_grid_max_refers_to_the_new_maximum():
    """``wd_at_grid_max`` tracks the CURRENT grid maximum, in every adapter and every table."""
    g = list(phase1.PHASE1_WD_GRID)
    gmax, gmin = max(g), min(g)
    assert gmax == 90.0 and gmin == 1e-5, (gmax, gmin)
    spec = model_spec("tirex")
    n = spec.n_points
    # a synthetic result whose selections sit at the two edges and at an old maximum
    res = _fake_res(spec, wds=[gmax] * 2 + [30.0] * (n - 3) + [gmin])
    rows = phase1.wd_selection_rows("tirex", "m4_hourly", spec, res, g, lr=LR)
    assert [r["wd_at_grid_max"] for r in rows] == [True, True] + [False] * (n - 3) + [False]
    assert [r["wd_at_grid_min"] for r in rows] == [False] * (n - 1) + [True]
    assert all(r["wd_grid_max"] == gmax and r["wd_grid_min"] == gmin for r in rows)
    # 30 -- the OLD maximum -- must NOT count as the grid maximum any more
    assert not any(r["wd_at_grid_max"] for r in rows if r["selected_wd"] == 30.0)
    # and every candidate's validation loss is saved, in both conventions
    got = json.loads(rows[0]["val_loss_by_wd"])
    assert set(map(float, got)) == set(map(float, g))
    assert "val_loss_by_wd_mean_pinball" in rows[0]
    print(f" C   wd_at_grid_max refers to the NEW maximum {gmax:g}; the old 30 no longer "
          "counts, and every candidate's val loss is saved  OK")


def test_C2_the_grid_ceiling_is_enforced_and_the_null_is_never_selectable():
    """``assert_wd_grid`` refuses lr*wd >= 1 and refuses the null as a candidate."""
    ok = phase1.assert_wd_grid(phase1.PHASE1_WD_GRID, LR, null_wd=phase1.PHASE1_WD_NULL)
    assert tuple(ok) == tuple(phase1.PHASE1_WD_GRID)
    assert max(phase1.PHASE1_WD_GRID) * LR < phase1.PHASE1_WD_DECOUPLED_LIMIT
    for bad, why in ((list(phase1.PHASE1_WD_GRID) + [100.0], "lr*wd == 1 (the bias-only null)"),
                     (list(phase1.PHASE1_WD_GRID) + [300.0], "lr*wd == 3 (diverges)"),
                     ([], "empty"), ([1.0, 1.0], "duplicates"), ([-1.0], "negative")):
        try:
            phase1.assert_wd_grid(bad, LR, null_wd=phase1.PHASE1_WD_NULL)
        except ValueError:
            pass
        else:
            raise AssertionError(f"a grid with {why} must be refused: {bad}")
    # the project's existing guard agrees, candidate for candidate
    from probing.tirex_probes import WD_DECOUPLED_LIMIT, assert_wd_grid as legacy
    assert WD_DECOUPLED_LIMIT == phase1.PHASE1_WD_DECOUPLED_LIMIT
    for cand in [(1e-3,), (30.0,), (90.0,), (99.0,)]:
        legacy(cand, LR); phase1.assert_wd_grid(cand, LR)
    for cand in [(100.0,), (300.0,)]:
        for fn in (lambda c: legacy(c, LR), lambda c: phase1.assert_wd_grid(c, LR)):
            try:
                fn(cand)
            except ValueError:
                continue
            raise AssertionError(f"{cand} must be refused by both guards")
    print(" C2  the lr*wd < 1 ceiling is enforced, agrees with tirex_probes.assert_wd_grid, "
          "and the null wd=100 can never be a candidate  OK")


# =========================================================================== #
# D-F   Q=1 vs Q=9
# =========================================================================== #
def test_D_q1_and_q9_output_dimensions_for_all_three_models():
    """The Q=1 probes are the validated model-specific forms, reached by passing Q -- not
    reimplemented. Shapes are read off the LIVE constructors."""
    from probing.probes import _apply_shared_head
    from probing.timesfm3_last_token_probes import make_probe as tfm_probe
    from probing.tirex_probes import make_shared_patch_probe, probe_forward

    expect = {"chronos2": {9: (768, 144), 1: (768, 16)},
              "timesfm3": {9: (1280, 576), 1: (1280, 64)},
              "tirex": {9: (512, 288), 1: (512, 32)}}
    # timesfm3: Linear(1280, H*Q)
    for Q in (9, 1):
        lin = tfm_probe(H=64, Q=Q)
        assert (lin.in_features, lin.out_features) == expect["timesfm3"][Q], lin
        out = lin(torch.zeros(5, 1280))
        from probing.timesfm3_last_token_probes import reshape_prediction
        assert tuple(reshape_prediction(out, 64, Q).shape) == (5, Q, 64)
    # tirex: ONE Linear(512, Q*32) shared across BOTH readout states
    for Q in (9, 1):
        lin = make_shared_patch_probe(d=512, out_patch=32, num_quantiles=Q)
        assert (lin.in_features, lin.out_features) == expect["tirex"][Q], lin
        assert tuple(probe_forward(lin, torch.zeros(5, 2, 512), num_quantiles=Q).shape) \
            == (5, Q, 64)
    # chronos2: ONE Linear(768, Q*16) shared across K=4 slots
    for Q in (9, 1):
        lin = torch.nn.Linear(768, Q * 16)
        assert (lin.in_features, lin.out_features) == expect["chronos2"][Q]
        assert tuple(_apply_shared_head(lin, torch.zeros(5, 4, 768), Q, 16, 64).shape) \
            == (5, Q, 64)
    # and the Phase-1 spec's declared width matches, for both sets
    for m in MODELS:
        s = model_spec(m)
        assert s.probe_out_features == expect[m][9][1], (m, s.probe_out_features)
    print(" D   Q=9 and Q=1 probe shapes correct for all three models; every one yields "
          "(B, Q, 64)  OK")


def test_E_q1_median_target_construction():
    """Q=1 is tau=0.5 exactly: median index 0, and its pinball loss IS 0.5 * MAE."""
    q1, mid = phase1.quantile_set("q1")
    assert list(q1) == [0.5] and mid == 0
    q9, mid9 = phase1.quantile_set("q9")
    assert mid9 == 4 and q9[mid9] == 0.5
    rng = np.random.default_rng(3)
    pred = rng.normal(size=(17, 1, 64)).astype(np.float32)
    targ = rng.normal(size=(17, 64)).astype(np.float32)
    pw = phase1.mean_pinball_per_window(pred, targ, q1)
    mae = np.abs(targ - pred[:, 0, :]).mean(axis=1)
    assert np.allclose(pw, 0.5 * mae, atol=1e-6), (pw[:3], 0.5 * mae[:3])
    # the target is the SAME (n, H) trajectory in both sets -- Q never touches it
    p9 = np.repeat(pred, 9, axis=1)
    per_q = phase1.per_quantile_mean_loss(p9, targ, q9)
    assert abs(per_q[4] - float((0.5 * np.abs(targ - pred[:, 0, :])).mean())) < 1e-6
    # the project's own tau=0.5 identity agrees
    from probing.tirex_probes import tau_half_is_half_mae
    assert tau_half_is_half_mae(torch.as_tensor(pred), torch.as_tensor(targ))
    print(" E   q1 is tau=0.5 with median index 0; its pinball loss == 0.5 * MAE, and the "
          "target is the same trajectory Q=9 uses  OK")


def test_F_q1_and_q9_cells_cannot_collide():
    """Q=1 and Q=9 differ in the FIT half of the hash, so they can never share a COMPLETE."""
    from experiments.run_three_model_phase1 import cell_config, parse_args
    a = parse_args([])
    hashes, fits = {}, {}
    for qs in ("q9", "q1"):
        qv, _ = phase1.quantile_set(qs)
        a.quantile_set = qs
        for m in MODELS:
            cfg = cell_config(m, "m4_hourly", a, qv, "digest", "reg")
            hashes[(m, qs)] = phase1.cell_config_hash(cfg)
            fits[(m, qs)] = phase1.fit_config_hash(cfg)
    for m in MODELS:
        assert hashes[(m, "q9")] != hashes[(m, "q1")], m
        assert fits[(m, "q9")] != fits[(m, "q1")], f"{m}: Q must be in the FIT half"
    assert len(set(hashes.values())) == len(hashes), "two cells share a hash"
    # a narrow-grid q9 cell can never satisfy an expanded-grid q9 run either
    a.quantile_set = "q9"
    qv, _ = phase1.quantile_set("q9")
    wide = cell_config("timesfm3", "m4_hourly", a, qv, "digest", "reg")
    a.wd_grid = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0]
    narrow = cell_config("timesfm3", "m4_hourly", a, qv, "digest", "reg")
    assert phase1.cell_config_hash(wide) != phase1.cell_config_hash(narrow)
    assert phase1.fit_config_hash(wide) != phase1.fit_config_hash(narrow)
    # ... while a TUNNEL-DEFINITION change moves only the POSTPROCESS half
    a.wd_grid = None
    base = cell_config("timesfm3", "m4_hourly", a, qv, "digest", "reg")
    moved = {**base, "tunnel_tols": [0.05]}
    assert phase1.fit_config_hash(base) == phase1.fit_config_hash(moved), \
        "changing a tolerance must NOT invalidate the fitted probes"
    assert phase1.postprocess_config_hash(base) != phase1.postprocess_config_hash(moved)
    assert phase1.cell_config_hash(base) != phase1.cell_config_hash(moved)
    print(" F   q1/q9 and narrow/expanded-grid cells have disjoint hashes; a tolerance change "
          "moves ONLY the postprocess half  OK")


# =========================================================================== #
# G-L   THE SUSTAINED TUNNEL
# =========================================================================== #
def test_G_sustained_tunnel_rejects_an_isolated_crossing():
    """One depth dips inside the band, the next climbs back out: NOT a tunnel entrance."""
    #        0    1     2    3    4    5     6
    v = np.array([9.0, 1.01, 5.0, 4.0, 3.0, 1.02, 1.0])
    assert tunnel_start(v, 0.05) == 1, "first crossing still reports the isolated dip"
    assert sustained_tunnel_start(v, 0.05) == 5, suffix_excursion(v).tolist()
    for m in MODELS:
        spec = model_spec(m)
        dep, n, nd = spec.depth_indices, spec.n_points, spec.n_depth_points
        curve = np.full(n, 3.0)
        curve[dep] = np.concatenate([[9.0, 1.01], np.full(nd - 4, 5.0), [1.02, 1.0]])
        r = phase1.tunnel_record(spec, curve)
        assert r["headline"]["depth_axis_index"] == nd - 2, (m, r["headline"])
        assert r["first_crossing"]["first_crossing_0.05"]["depth_axis_index"] == 1, m
    print(" G   an isolated early crossing does NOT open a sustained tunnel; first crossing "
          "still reports it, under its own name  OK")


def test_H_sustained_tunnel_accepts_a_true_persistent_entry():
    """Once inside and staying inside, the entrance is exactly the first such depth."""
    for m in MODELS:
        spec = model_spec(m)
        dep, n, nd = spec.depth_indices, spec.n_points, spec.n_depth_points
        for entrance in (0, 1, nd // 2, nd - 1):
            v = np.full(n, 7.0)
            v[dep] = [10.0 if p < entrance else 1.0 for p in range(nd)]
            r = phase1.tunnel_record(spec, v)
            assert r["headline"]["depth_axis_index"] == entrance, (m, entrance, r["headline"])
            # inside the tunnel the excursion is bounded by tol BY CONSTRUCTION
            e = r["val_suffix_excursion_by_depth"][entrance]
            assert e <= 0.05 + 1e-12, (m, entrance, e)
        # a gentle monotone approach: entrance where the ratio first stays inside
        v = np.full(n, 7.0)
        v[dep] = np.linspace(2.0, 1.0, nd)
        r = phase1.tunnel_record(spec, v)
        pos = r["headline"]["depth_axis_index"]
        ratios = np.asarray(r["val_ratio_by_depth"])
        assert np.all(ratios[pos:] <= 1.05 + 1e-12)
        assert pos == 0 or ratios[pos - 1] > 1.05
    print(" H   a genuine persistent entry is found at exactly the first sustained depth, and "
          "the excursion inside is bounded by tol  OK")


def test_I_tolerance_monotonicity():
    """depth(10%) <= depth(5%) <= depth(2%), on every curve -- a theorem, asserted anyway."""
    rng = np.random.default_rng(7)
    tols = (0.10, 0.05, 0.02, 0.01)
    for _ in range(3000):
        v = np.abs(rng.normal(1.0, 0.5, size=int(rng.integers(3, 22)))) + 1e-3
        ent = [sustained_tunnel_start(v, t) for t in tols]
        assert ent == sorted(ent), (v.tolist(), ent)
        assert_tolerance_monotone(dict(zip(tols, ent)))
        # and sustained is never EARLIER than first crossing
        for t in tols:
            assert sustained_tunnel_start(v, t) >= tunnel_start(v, t)
    for m in MODELS:
        spec = model_spec(m)
        n, dep = spec.n_points, spec.depth_indices
        for _ in range(300):
            v = np.abs(rng.normal(1.0, 0.5, size=n)) + 1e-3
            r = phase1.tunnel_record(spec, v)
            got = [r["by_tolerance"][f"tol_{t:g}"]["depth_axis_index"] for t in tols]
            assert got == sorted(got), (m, got)
        _ = dep
    # a deliberately violating map is REJECTED, so the guard is not vacuous
    try:
        assert_tolerance_monotone({0.10: 5, 0.05: 3})
    except RuntimeError:
        pass
    else:
        raise AssertionError("a non-monotone entrance map must raise")
    print(" I   10% <= 5% <= 2% <= 1% on 3300 random curves + all three model specs; the "
          "monotonicity guard rejects a violating map  OK")


def test_J_head_input_diagnostics_stay_excluded():
    """A final normalization can neither be an entrance nor move one, under the new rule."""
    for m in MODELS:
        spec = model_spec(m)
        n, dep = spec.n_points, spec.depth_indices
        base = np.full(n, 5.0)
        base[dep] = np.linspace(3.0, 1.0, spec.n_depth_points)
        want = phase1.tunnel_record(spec, base)["headline"]["index"]
        for factor in (1e-9, 1e9):
            v = base.copy()
            for i in spec.diagnostic_indices:
                v[i] = factor
            r = phase1.tunnel_record(spec, v, v)
            assert r["headline"]["index"] == want, (m, factor)
            assert r["headline"]["point_type"] == phase1.BLOCK_DEPTH
            assert r["reference_point"] == spec.reference_label
            for e in list(r["by_tolerance"].values()) + list(r["first_crossing"].values()):
                assert e["index"] in dep and e["point_type"] == phase1.BLOCK_DEPTH
            assert len(r["val_suffix_excursion_by_depth"]) == spec.n_depth_points
            for lab in spec.diagnostic_labels:
                assert lab in r["head_input_diagnostic"] and lab not in r["depth_axis_labels"]
    print(" J   a 1e-9x / 1e9x head-input loss cannot be selected and cannot move the "
          "sustained entrance at ANY tolerance  OK")


def test_K_tunnel_uses_validation_only():
    """The entrance is a function of the validation curve alone -- proved by varying test."""
    rng = np.random.default_rng(11)
    for m in MODELS:
        spec = model_spec(m)
        n = spec.n_points
        val = np.abs(rng.normal(1.0, 0.4, size=n)) + 0.1
        base = phase1.tunnel_record(spec, val)["headline"]["index"]
        for _ in range(25):
            test = np.abs(rng.normal(1.0, 3.0, size=n)) + 0.1
            r = phase1.tunnel_record(spec, val, test)
            assert r["headline"]["index"] == base, (m, r["headline"]["index"], base)
            assert r["split_used"] == "validation" and r["test_never_used_for_selection"]
            # the test-side numbers are reported DESCRIPTIVELY, and they do exist
            assert "test_ratio_at_tunnel" in r["generalization_at_entrance"]
    # the criterion functions themselves never see a test curve
    import inspect
    for fn in (sustained_tunnel_start, tunnel_start, suffix_excursion):
        assert "test" not in inspect.signature(fn).parameters
    print(" K   25 random test curves per model never move the entrance; the criterion "
          "functions take no test argument  OK")


def test_L_test_loss_cannot_alter_the_selected_tunnel():
    """Even a test curve engineered to make ANOTHER depth look best changes nothing."""
    for m in MODELS:
        spec = model_spec(m)
        n, dep, nd = spec.n_points, spec.depth_indices, spec.n_depth_points
        val = np.full(n, 9.0)
        val[dep] = [9.0] * (nd - 2) + [1.0, 1.0]          # val says: enter at nd-2
        want = phase1.tunnel_record(spec, val)["headline"]["depth_axis_index"]
        assert want == nd - 2
        adversarial = np.full(n, 9.0)
        adversarial[dep] = [1.0] * nd                      # test says: everything is perfect
        r = phase1.tunnel_record(spec, val, adversarial)
        assert r["headline"]["depth_axis_index"] == want, (m, r["headline"])
        # ... and the reverse
        adversarial2 = np.full(n, 1.0)
        adversarial2[dep] = [1.0] * (nd - 1) + [99.0]
        r2 = phase1.tunnel_record(spec, val, adversarial2)
        assert r2["headline"]["depth_axis_index"] == want, (m, r2["headline"])
    print(" L   an adversarial test curve (all-perfect, or final-depth catastrophic) cannot "
          "move the validation-selected entrance  OK")


def test_L2_delegation_and_no_second_implementation():
    """Both criteria live in probing.tunnel and nowhere else."""
    import inspect
    src = inspect.getsource(phase1.tunnel_record)
    assert "sustained_tunnel_start(" in src and "tunnel_start(" in src
    for f in ("probing/phase1.py", "probing/phase1_cells.py", "probing/phase1_chronos2.py",
              "probing/phase1_timesfm3.py", "probing/phase1_tirex.py",
              "experiments/run_three_model_phase1.py", "experiments/make_phase1_tables.py",
              "experiments/rebuild_phase1_tunnels.py",
              "experiments/make_phase1_q1_q9_comparison.py"):
        text = (REPO_ROOT / f).read_text()
        for name in ("def tunnel_start", "def sustained_tunnel_start", "def suffix_excursion"):
            assert name not in text, f"{f} re-implements {name}"
    print(" L2  neither criterion is re-implemented anywhere in the Phase-1 code  OK")


# =========================================================================== #
# M-N   REUSE
# =========================================================================== #
def test_M_cached_feature_reuse_validates_dataset_and_window_identity():
    """A cache key must pin the dataset, split, windows and checkpoint -- and NOT Q."""
    import inspect
    from probing.timesfm3_last_token import cache_metadata as tfm_meta
    from probing.tirex_model import cache_metadata as tx_meta

    for fn in (tfm_meta, tx_meta):
        params = set(inspect.signature(fn).parameters)
        assert {"tag", "split", "geom", "checkpoint"} <= params, (fn, params)
        assert not ({"q", "Q", "quantiles", "quantile_set", "num_quantiles"} & params), \
            f"{fn.__name__} must not key on the quantile set -- Q1 and Q9 SHARE the cache"
    # Chronos-2's K-slot cache: keyed by tag/split/K/H, and it FAILS LOUDLY on stale labels
    src = inspect.getsource(
        __import__("probing.extraction", fromlist=["x"]).extract_kout_features)
    assert "K{K}_H{horizon}" in src.replace('f"', '"')
    assert "cached labels do not match the current" in src   # the message wraps in the source
    # the Phase-1 cell config puts the WINDOW DIGEST in the fit half, so a re-windowing
    # can never be reused, while Q lives there too and separates the two runs
    from experiments.run_three_model_phase1 import cell_config, parse_args
    a = parse_args([])
    qv, _ = phase1.quantile_set("q9")
    h1 = phase1.fit_config_hash(cell_config("tirex", "m4_hourly", a, qv, "digest-A", "reg"))
    h2 = phase1.fit_config_hash(cell_config("tirex", "m4_hourly", a, qv, "digest-B", "reg"))
    assert h1 != h2, "the window digest must be part of the fit identity"
    print(" M   feature caches key on dataset/split/windows/checkpoint and NOT on Q, so q1 and "
          "q9 share them; a re-windowing still fails loudly  OK")


def test_N_q1_run_reuses_representations_and_geometry_is_q_independent():
    """The Q=1 run must not re-extract, and geometry must not depend on Q.

    Proved three ways: the extraction-relevant half of the cell config is IDENTICAL for q1 and
    q9 (so the same cache entries are hit); no geometry entry point takes a quantile argument;
    and ``geometry_block`` returns bit-identical matrices for two calls that differ in nothing
    but a quantile vector that it never receives.
    """
    import inspect
    from experiments.run_three_model_phase1 import cell_config, parse_args

    extraction_keys = ("checkpoint", "extract_batch_size", "feature_dtype", "detrend",
                       "backend", "rollout_mode", "readout", "C", "H", "suite", "seed",
                       "window_digest", "registry_hash")
    a = parse_args([])
    for m in MODELS:
        cfgs = {}
        for qs in ("q9", "q1"):
            a.quantile_set = qs
            qv, _ = phase1.quantile_set(qs)
            cfgs[qs] = cell_config(m, "m4_hourly", a, qv, "digest", "reg")
        for k in extraction_keys:
            assert cfgs["q9"].get(k) == cfgs["q1"].get(k), (m, k)
        assert cfgs["q9"]["quantile_set"] != cfgs["q1"]["quantile_set"]

    # no geometry entry point takes a quantile vector
    import probing.phase1_chronos2 as c2
    import probing.phase1_timesfm3 as t3
    import probing.phase1_tirex as tx
    for fn in (c2.geometry_blocks, t3.geometry_blocks, tx.geometry_blocks,
               phase1.geometry_block):
        params = set(inspect.signature(fn).parameters)
        assert not ({"q", "quantiles", "quantile_set", "median_idx"} & params), (fn, params)

    # and the matrices are bit-identical across two "runs" whose only difference is Q
    rng = np.random.default_rng(2)
    mats = [rng.normal(size=(50, 16)) for _ in range(4)]
    labs = ["Emb", "L1", "L2", "L3"]
    b9 = phase1.geometry_block(mats, labs, d=16, split="test", variant="headline",
                               null_floor_reps=1, seed=0)
    b1 = phase1.geometry_block([m.copy() for m in mats], labs, d=16, split="test",
                               variant="headline", null_floor_reps=1, seed=0)
    for est in b9["cka"]:
        assert np.array_equal(b9["cka"][est], b1["cka"][est]), est
    assert (b9["effective_rank"]["effective_rank"]
            == b1["effective_rank"]["effective_rank"])
    print(" N   the q1 run's extraction config is identical to q9's (same caches, no backbone "
          "re-run); geometry takes no quantile argument and is bit-identical  OK")


# =========================================================================== #
# the postprocess-repair path
# =========================================================================== #
def test_O_stale_postprocess_is_detected_and_repairable_without_a_refit():
    """A changed tunnel definition is ``stale_postprocess``, not ``incompatible``."""
    tmp = Path(tempfile.mkdtemp(prefix="phase1_post_"))
    try:
        store = phase1.CellStore(tmp, "tirex", "m4_hourly")
        stage = store.begin()
        for a in phase1.REQUIRED_ARTIFACTS:
            p = stage / a
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        cfg = {"protocol": "phase1/v1", "model": "tirex", "wd_grid": [1e-3, 1.0],
               "tunnel_definition": "sustained_suffix_v1", "tunnel_tol": 0.05,
               "tunnel_tols": [0.01, 0.02, 0.05, 0.10]}
        store.commit(stage, phase1.cell_config_hash(cfg),
                     fit_hash=phase1.fit_config_hash(cfg),
                     post_hash=phase1.postprocess_config_hash(cfg))
        assert store.status(phase1.cell_config_hash(cfg),
                            phase1.fit_config_hash(cfg),
                            phase1.postprocess_config_hash(cfg))[0] == "complete"
        # only the POSTPROCESS half moves -> repairable
        moved = {**cfg, "tunnel_tols": [0.05]}
        st, why = store.status(phase1.cell_config_hash(moved),
                               phase1.fit_config_hash(moved),
                               phase1.postprocess_config_hash(moved))
        assert st == "stale_postprocess", (st, why)
        assert "rebuild_phase1_tunnels" in why
        # the FIT half moves -> a real refit is required
        refit = {**cfg, "wd_grid": [1e-3, 1.0, 90.0]}
        st2, _ = store.status(phase1.cell_config_hash(refit),
                              phase1.fit_config_hash(refit),
                              phase1.postprocess_config_hash(refit))
        assert st2 == "incompatible", st2
        # ... and the driver REFUSES rather than silently refitting or silently keeping
        import inspect
        from experiments import run_three_model_phase1 as drv
        src = inspect.getsource(drv.main)
        assert 'status == "stale_postprocess"' in src
        assert "rebuild_phase1_tunnels" in src
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" O   a changed tunnel definition is stale_postprocess (repairable, no GPU); a "
          "changed wd grid is incompatible (refit)  OK")


# =========================================================================== #
# helpers
# =========================================================================== #
def _fake_res(spec, wds):
    """A minimal ``res`` dict of the shape the adapters produce, for table-level contracts."""
    n = spec.n_points
    grid = list(phase1.PHASE1_WD_GRID)
    rng = np.random.default_rng(0)
    return {"labels": spec.labels, "quantiles": [0.1 * (i + 1) for i in range(9)],
            "median_index": 4,
            "train_loss": list(rng.uniform(0.1, 0.2, n)),
            "val_loss": list(rng.uniform(0.2, 0.3, n)),
            "test_loss": list(rng.uniform(0.2, 0.3, n)),
            "wd": [float(w) for w in wds],
            "wd_at_grid_max": [float(w) == max(grid) for w in wds],
            "wd_at_grid_min": [float(w) == min(grid) for w in wds],
            "selection": [{str(g): float(rng.uniform(0.2, 0.3)) for g in grid}
                          for _ in range(n)],
            "n_params": [16416] * n}


def main(argv=None):
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"\nEXPANDED-WD + SUSTAINED-TUNNEL CONTRACTS  ({len(tests)} groups)\n" + "=" * 78)
    for t in tests:
        t()
    print("=" * 78)
    print(f"all {len(tests)} contracts hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
