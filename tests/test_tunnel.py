"""No-GPU contract tests for probing.tunnel (tunnel criterion + D statistics).

Run: python -m tests.test_tunnel   (pytest-compatible)
"""

from __future__ import annotations

import numpy as np

from probing.config import LAST_LAYER, NUM_LAYERS
from probing.tunnel import (PT_ID_TAGS, PT_OOD_TAGS, check_tunnel_on_test, d_stat_boot,
                            delta_stat, domain_status, m_stat_boot, max_excursion,
                            tunnel_record, tunnel_record_multi, tunnel_start,
                            val_curve_from_selection)


def test_tunnel_start_basic():
    # monotone descent: first-crossing = the first layer within 5% (1.05x) of the last layer
    v = [2.0, 1.6, 1.3, 1.04, 1.0, 1.01, 1.0]
    assert tunnel_start(v) == 3          # v[2]=1.3 > 1.05x last; v[3]=1.04 <= 1.05x is the first crossing


def test_tunnel_one_sided_better_than_last():
    # layers BETTER than last satisfy the criterion (one-sided)
    v = [2.0, 0.8, 0.9, 1.0]
    assert tunnel_start(v) == 1


def test_first_crossing_opens_before_a_hump():
    # U-shaped curve (the Electricity pattern): first-crossing opens the tunnel at the FIRST dip,
    # even though a later hump re-crosses the threshold. The hump then lives INSIDE the tunnel and
    # is flagged by max_excursion (which is NOT bounded by tol on validation under first-crossing).
    v = [1.0, 3.0, 1.01, 1.02, 1.0]
    assert tunnel_start(v) == 0                        # first crossing: v[0]=1.0 <= 1.05x last
    assert np.isclose(max_excursion(v, 0), 2.0)       # the hump (3.0/1.0 - 1) lives inside the tunnel


def test_tunnel_degenerate_last_only():
    v = [3.0, 2.5, 2.0, 1.0]            # every earlier layer >5% worse
    assert tunnel_start(v) == 3


def test_tunnel_tol_boundary_inclusive():
    v = [1.05, 1.0]
    assert tunnel_start(v, tol=0.05) == 0       # <= is inclusive
    assert tunnel_start([1.0500001, 1.0], tol=0.05) == 1


def test_check_tunnel_on_test():
    holds, margins = check_tunnel_on_test([2.0, 1.02, 1.0], l_start=1)
    assert holds and np.isclose(margins[0], 1.0) and np.isclose(margins[-1], 0.0)
    holds, _ = check_tunnel_on_test([2.0, 1.2, 1.0], l_start=1)
    assert not holds


def test_tunnel_record_fields_and_no_test_leak():
    val = [2.0, 1.0, 1.0]
    test = [0.9, 5.0, 1.0]              # test would give a different boundary — must not matter
    rec = tunnel_record("m4_hourly", val, test, val_split_kind="temporal_rolling")
    assert rec["l_start"] == 1 and rec["tunnel"] == [1, 2]
    assert rec["tunnel_definition"] == "first_crossing_95"
    assert "l_start_sustained" not in rec and "tunnel_sustained" not in rec   # single definition now
    assert rec["final_layer_val_loss"] == 1.0
    assert np.isclose(rec["max_excursion_val"], 0.0)     # flat val suffix here (NOT guaranteed under first-crossing)
    assert np.isclose(rec["max_excursion_test"], 4.0)    # test hump 5.0/1.0 - 1
    assert rec["test_criterion_holds"] is False          # test[1]=5.0 breaks it
    assert rec["domain_status"] == {"pretraining": "pt_id", "adaptation": None}
    assert len(rec["val_loss_by_layer"]) == len(rec["test_margins"]) == 3


def test_domain_status_axes():
    assert domain_status(PT_ID_TAGS[0])["pretraining"] == "pt_id"
    assert domain_status(PT_OOD_TAGS[0])["pretraining"] == "pt_ood"
    for t in PT_ID_TAGS + PT_OOD_TAGS:
        assert domain_status(t)["adaptation"] is None    # reserved for the FT block
    try:
        domain_status("nope")
        assert False, "unknown tag must raise"
    except ValueError:
        pass


def _synth_windows(rng, n=120, n_clusters=12, shift=0.3):
    """(NUM_LAYERS, n) window losses where the last layer is `shift` worse than layer 2."""
    base = rng.uniform(1.0, 2.0, size=n)
    wl = np.tile(base, (NUM_LAYERS, 1)) + rng.normal(0, 0.01, size=(NUM_LAYERS, n))
    wl[LAST_LAYER] += shift
    cid = np.arange(n) % n_clusters
    return wl, cid


def test_d_stat_boot_sign_shape_determinism():
    rng = np.random.default_rng(0)
    wl, cid = _synth_windows(rng)
    d1 = d_stat_boot(wl, cid, l_start=2, B=200, seed=0)
    d2 = d_stat_boot(wl, cid, l_start=2, B=200, seed=0)
    assert d1["point"] > 0                               # last worse -> D positive
    assert d1["boot"].shape == (200,)
    assert d1["ci"][0] <= d1["point"] <= d1["ci"][1]
    assert np.array_equal(d1["boot"], d2["boot"])        # deterministic
    assert d1["n_clusters"] == 12 and d1["n_windows"] == 120
    # exact point value: ratio of full-sample layer means
    m = wl.mean(axis=1)
    assert np.isclose(d1["point"], (m[LAST_LAYER] - m[2]) / m[2])


def test_m_stat_boot_point_and_boundary_fixed():
    rng = np.random.default_rng(2)
    wl, cid = _synth_windows(rng, shift=0.0)
    wl[5] += 0.4                                        # a mid-tunnel hump above the last layer
    m = m_stat_boot(wl, cid, l_start=2, B=200, seed=0)
    means = wl.mean(axis=1)
    expect = (means[2:] / means[LAST_LAYER] - 1.0).max()
    assert np.isclose(m["point"], expect) and m["point"] > 0.2
    assert m["boot"].shape == (200,)
    # starting AFTER the hump excludes it: M collapses toward 0
    m2 = m_stat_boot(wl, cid, l_start=6, B=200, seed=0)
    assert m2["point"] < 0.05


def test_delta_stat_is_replicate_difference():
    rng = np.random.default_rng(1)
    wl_id, cid_id = _synth_windows(rng, shift=0.1)
    wl_ood, cid_ood = _synth_windows(rng, shift=0.5)
    d_id = d_stat_boot(wl_id, cid_id, l_start=2, B=200, seed=0)
    d_ood = d_stat_boot(wl_ood, cid_ood, l_start=2, B=200, seed=0)
    dl = delta_stat(d_ood, d_id)
    assert np.isclose(dl["point"], d_ood["point"] - d_id["point"])
    assert np.allclose(dl["boot"], d_ood["boot"] - d_id["boot"])
    assert dl["point"] > 0                               # stronger OOD degradation


def test_tunnel_record_multi_mean_defines_boundary():
    # the tunnel must be defined from the MEAN validation curve, never an average of per-seed
    # first-crossings (which would be 0, 1, 1 here). Mean L0 = 1.20 > 1.05*last, but mean
    # L1 = 1.047 <= 1.05, so the first-crossing boundary from the MEAN curve is L1.
    val = [[1.00, 1.20, 1.00, 1.0],
           [1.30, 1.00, 1.00, 1.0],
           [1.30, 0.94, 1.00, 1.0]]
    test = [[2.0, 1.5, 1.0, 1.0]] * 3
    rec = tunnel_record_multi("m4_hourly", val, test, run_seeds=(0, 1, 2))
    assert rec["mean_val_loss_by_layer"][0] == 1.2      # mean L0 = 1.20 > 1.05 -> tunnel can't start at L0
    assert rec["l_start"] == 1 and rec["tunnel"] == [1, 3]
    assert rec["tunnel_definition"] == "first_crossing_95"
    assert rec["run_seeds"] == [0, 1, 2] and rec["run_type"] == "probe_seed"
    # all runs + means + stds retained, layerwise
    assert np.array(rec["val_loss_by_run"]).shape == (3, 4)
    assert np.allclose(rec["std_test_loss_by_layer"], 0.0)
    # D_ID and M_test on the MEAN test curve: (1.0 - 1.5)/1.5 and max(1.5/1.0 - 1)
    assert np.isclose(rec["D_ID"], -1 / 3)
    assert np.isclose(rec["M_test"], 0.5)
    # mismatched runs/seeds must raise
    try:
        tunnel_record_multi("m4_hourly", val[:2], test, run_seeds=(0, 1, 2))
        assert False, "shape mismatch must raise"
    except ValueError:
        pass


def test_val_curve_from_selection():
    sel = {i: {"val_loss_by_wd": {1e-4: 2.0 + i, 1e-3: 1.0 + i}, "chosen_wd": 1e-3}
           for i in range(NUM_LAYERS)}
    curve = val_curve_from_selection(sel)
    assert curve == [1.0 + i for i in range(NUM_LAYERS)]
    sel[3] = None
    try:
        val_curve_from_selection(sel)
        assert False, "missing selection must raise"
    except ValueError:
        pass


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"\nAll {len(fns)} tunnel tests passed.")
