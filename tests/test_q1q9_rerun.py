"""Contracts for the q1/q9 forecasting rerun (wide WD grid, protocol v2, browsable q-folders).

CPU / synthetic only — no model, no GPU, no cache, no download. Verifies the quantile-config routing,
the wide-grid + protocol-version identity, that q1 and q9 land in disjoint namespaces, that a legacy
narrow-grid q9 result can never satisfy a new q9 skip, and that the shared-linear fslot probe fits at
both output widths. Run:  python -m tests.test_q1q9_rerun
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from probing.config import NUM_LAYERS, OUTPUT_PATCH_SIZE
from probing.probes import (QUANTILE_SETS, WD_GRID_V2, PROBE_PROTOCOL_VERSION,
                            fit_shared_forecast_probe_explicit_val, predict_shared_forecast_probe)

import experiments.run_ptood_probing_ftok as ftok
import experiments.run_fslot_transfer as tr
import experiments.run_ft_specialization as ft
import experiments.run_task_shift as ts
import experiments.run_q1q9_compare as cmp

P, K, D, H = OUTPUT_PATCH_SIZE, 4, 768, 64


def _synth_slot(n, seed=0, n_points=NUM_LAYERS + 1):
    rng = np.random.default_rng(seed)
    feats = {i: rng.normal(size=(n, K, D)).astype(np.float32) for i in range(n_points)}
    y = rng.normal(size=(n, H)).astype(np.float32)
    return feats, y


# 1-2. Quantile vectors are exactly the intended sets.
def test_q9_exact_quantiles():
    assert np.allclose(QUANTILE_SETS["q9"], [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], atol=1e-6)
    assert len(QUANTILE_SETS["q9"]) == 9


def test_q1_is_median_only():
    assert np.allclose(QUANTILE_SETS["q1"], [0.5]) and len(QUANTILE_SETS["q1"]) == 1


# 3-4, 19-20. Output width = Q*P; both configs fit + predict end-to-end.
def _fit_width(qset):
    f_tr, y_tr = _synth_slot(24, seed=1)
    f_va, y_va = _synth_slot(10, seed=2)
    f_te, y_te = _synth_slot(12, seed=3)
    fitted = fit_shared_forecast_probe_explicit_val(
        f_tr, y_tr, f_va, y_va, quantiles=QUANTILE_SETS[qset], epochs=3, wd_grid=(1e-2,), device="cpu")
    out = predict_shared_forecast_probe(fitted, f_te, y_te, quantiles=QUANTILE_SETS[qset], device="cpu")
    return fitted, out


def test_q9_output_dim_is_9P():
    fitted, out = _fit_width("q9")
    assert fitted[0]["out_features"] == 9 * P == 144
    assert len(out) == NUM_LAYERS + 1                       # 14 fslot readout points


def test_q1_output_dim_is_P():
    fitted, out = _fit_width("q1")
    assert fitted[0]["out_features"] == P == 16
    assert all(np.isfinite(v) for v in out.values())


# 5. The wide WD grid is exactly the intended eight values, shared by every driver.
def test_wide_wd_grid_value_and_sharing():
    assert WD_GRID_V2 == (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0)
    assert ftok.WD_GRID == ft.WD_GRID == ts.WD_GRID == WD_GRID_V2


# 6. Weight decay is selected on validation only (chosen_wd recorded, inside the grid).
def test_wd_selected_on_val():
    f_tr, y_tr = _synth_slot(24, seed=1)
    f_va, y_va = _synth_slot(10, seed=7)
    fitted = fit_shared_forecast_probe_explicit_val(
        f_tr, y_tr, f_va, y_va, quantiles=QUANTILE_SETS["q9"], epochs=3,
        wd_grid=(1e-3, 1e-1, 1.0), device="cpu")
    sel = fitted[0]["selection"]
    assert set(map(float, sel["val_loss_by_wd"])) == {1e-3, 1e-1, 1.0}
    assert sel["chosen_wd"] in (1e-3, 1e-1, 1.0)


# 7-8. q1 and q9 result + checkpoint paths are disjoint (and both carry the v2 token).
def test_q1_q9_paths_disjoint():
    assert ftok._run_qtag("q9") == "q9__v2" != ftok._run_qtag("q1") == "q1__v2"
    c9, c1 = ftok._ptid_ckpt_dir("m4_hourly", "q9", 0), ftok._ptid_ckpt_dir("m4_hourly", "q1", 0)
    assert c9 != c1 and "q9__v2" in c9.name and "q1__v2" in c1.name


# 9, 23. A legacy narrow-grid q9 record can never satisfy a v2 q9 skip.
def test_legacy_q9_cannot_satisfy_v2_skip():
    ftok.FAMILY = ftok.PROBE_FAMILIES["shared_linear"]
    tmp = Path(tempfile.mkdtemp())
    # (a) different filename: the v2 run stub is not the legacy stub -> skip never even looks at legacy
    assert "q9__v2" in ftok._run_qtag("q9") and ftok._run_qtag("q9") != "q9"
    # (b) a file with legacy metadata AT the v2 path is rejected fail-loud (not silently skipped)
    legacy = tmp / "m4__q9__v2__seed0.json"
    legacy.write_text(json.dumps({"quantile_set": "q9", "wd_grid": [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]}))
    raised = False
    try:
        ftok._run_compatible(legacy, "q9")
    except RuntimeError:
        raised = True
    assert raised, "incompatible (old-grid / no-protocol) result must fail loud, not satisfy skip"


# 10. Protocol metadata is complete + machine-readable.
def test_protocol_metadata_complete():
    ftok.FAMILY = ftok.PROBE_FAMILIES["shared_linear"]
    m = ftok._protocol_meta(QUANTILE_SETS["q9"])
    assert m["probe_protocol_version"] == "v2" == PROBE_PROTOCOL_VERSION
    assert m["wd_grid"] == list(WD_GRID_V2) and m["Q"] == 9 and m["P"] == P
    assert m["out_features"] == 144 and m["K"] == K


# 22. A matching v2 record satisfies the skip (resume).
def test_matching_v2_record_satisfies_skip():
    ftok.FAMILY = ftok.PROBE_FAMILIES["shared_linear"]
    tmp = Path(tempfile.mkdtemp())
    p = tmp / "m4__q9__v2__seed0.json"
    p.write_text(json.dumps({"quantile_set": "q9", "probe_protocol_version": "v2",
                             "wd_grid": list(WD_GRID_V2)}))
    assert ftok._run_compatible(p, "q9") is True
    assert ftok._run_compatible(tmp / "absent.json", "q9") is False


# 11. ext-v4 ID q-routing -> <qset>/id + <qset>/tunnels.
def test_ev4_id_routing():
    ftok.FAMILY = ftok.PROBE_FAMILIES["shared_linear"]
    ftok._derive_dirs("q1")
    assert ftok.FIG_DIR == ftok.OUT_ROOT / "q1" / "id" / "figures"
    assert ftok.TUNNEL_DIR == ftok.OUT_ROOT / "q1" / "tunnels"


# 12-13. transfer q-routing -> cross_dataset / unseen under the q-folder.
def test_transfer_routing():
    tr.FAMILY = tr.PROBE_FAMILIES["shared_linear"]
    tr._derive_dirs("transfer_4x4", "q9")
    assert tr.FIG_DIR == tr.OUT_ROOT / "q9" / "cross_dataset" / "figures"
    tr._derive_dirs("pt_ood", "q1")
    assert tr.FIG_DIR == tr.OUT_ROOT / "q1" / "unseen" / "figures"


# 14. domain-FT q-routing + versioned stem.
def test_domain_ft_routing():
    ft._configure_qset("q1")
    assert ft.FIG_DIR == ft.DOMAIN_SHIFT_ROOT / "q1" / "figures"
    assert ft._run_stem("stage1_ft_early", "boom_hourly", 0).endswith("__q1__v2__seed0")
    ft._configure_qset("q9")     # restore default


# 15. task-FT q-routing + versioned stem; classification dir stays q-independent.
def test_task_ft_routing():
    ts.configure("uwave"); ts._configure_qset("q1")
    assert ts.FCAST_FIG_DIR == ts.SRC_OUT / "forecasting" / "q1" / "figures"
    assert ts.FIG_DIR == ts.SRC_OUT / "figures"            # Exp-A classification untouched by q
    assert ts._fcast_probe_stem("stage1_cls_early", "boom_hourly", 0).endswith("__q1__v2__seed0")
    ts.configure("forda"); ts._configure_qset("q9")        # restore defaults


# 16. No MLP anywhere: every fslot driver defaults shared_linear + the overnight never names native_mlp.
def test_no_mlp_in_rerun():
    assert ftok.FAMILY.name == "shared_linear"
    # driver argparse defaults to shared_linear (MLP is opt-in and never named by the overnight)
    ns = ftok._parse_args(["--fit-ptid"])
    assert ns.probe_family == "shared_linear"
    ns2 = tr._parse_args([])
    assert ns2.probe_family == "shared_linear"
    # the overnight never PASSES the MLP family on any invocation (comments may mention it in prose)
    job = Path("job_full_q1_q9_rerun.sh").read_text()
    cmds = [l for l in job.splitlines() if l.lstrip().startswith(("run python", "python -m"))]
    assert cmds, "no command lines found in overnight job"
    assert not any(("native_mlp" in l or "--probe-family" in l) for l in cmds), \
        "overnight job must never invoke the MLP family"


# 17. Tunnels are q-specific (independent per quantile config).
def test_tunnels_q_specific():
    t9 = ftok._linear_tunnel_path("uber_tlc_hourly", "q9")
    t1 = ftok._linear_tunnel_path("uber_tlc_hourly", "q1")
    assert t9 != t1 and "q9__v2" in t9.name and "q1__v2" in t1.name
    assert t9.parent == ftok.OUT_ROOT / "q9" / "tunnels"


# 18. Comparison reader targets only the requested q namespace.
def test_compare_reads_requested_q():
    assert "q9__v2" in cmp._run_json("q9", "m4_hourly", 0).name
    assert "q1__v2" in cmp._recompute_json("q1", "m4_hourly", 0).name
    assert cmp._relative_regret([2.0, 1.0]).tolist() == [1.0, 0.0]


# 21. Overnight preflight enumerates the required caches + FT checkpoints and would exit on a miss.
def test_overnight_preflight_asserts_prereqs():
    job = Path("job_full_q1_q9_rerun.sh").read_text()
    assert "sys.exit(1)" in job and "MISSING required fslot caches" in job
    for tok in ("__ft__boom__", "__ft__forda_cls__", "__ft__uwave_cls__", "__ft__handwriting_cls__",
                "boom/stage1_ft_early", "extraction NOT required"):
        assert tok in job, f"preflight missing check for {tok}"


TESTS = [
    test_q9_exact_quantiles, test_q1_is_median_only,
    test_q9_output_dim_is_9P, test_q1_output_dim_is_P,
    test_wide_wd_grid_value_and_sharing, test_wd_selected_on_val,
    test_q1_q9_paths_disjoint, test_legacy_q9_cannot_satisfy_v2_skip,
    test_protocol_metadata_complete, test_matching_v2_record_satisfies_skip,
    test_ev4_id_routing, test_transfer_routing, test_domain_ft_routing, test_task_ft_routing,
    test_no_mlp_in_rerun, test_tunnels_q_specific, test_compare_reads_requested_q,
    test_overnight_preflight_asserts_prereqs,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(TESTS)} q1/q9-rerun tests passed.")
