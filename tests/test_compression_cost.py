"""CPU tests for the truncation cost accounting (probing/model_size.py) and the compression driver.

No GPU, no model download, no feature cache. Two groups:

  * pure arithmetic / contract tests, which always run: parameter families, the adapter convention,
    the FLOP ratios, the encoder token layout, and the parameter bucketing used to verify against a
    real model (checked against a fake model whose module names match Chronos-2's);
  * provenance gates against COMMITTED artifacts, which skip loudly if the artifacts are absent:
    the saturation depth must equal the committed l_start, the effective-rank depth must equal the
    argmax of the committed spectral record, and the driver's paired bootstrap must reproduce the
    committed gap-recovery CIs bit-for-bit.

Run:  OMP_NUM_THREADS=2 python -m tests.test_compression_cost
"""

from __future__ import annotations

import csv
import json

import numpy as np
import torch
import torch.nn as nn

from probing import model_size


# --------------------------------------------------------------------------- #
# parameter accounting
# --------------------------------------------------------------------------- #
def test_families_sum_to_the_recorded_total():
    """The decisive check: the constants must sum to a number measured on the real model."""
    assert sum(model_size.param_breakdown().values()) == model_size.TOTAL_PARAMS
    assert model_size.TOTAL_PARAMS == model_size.RECORDED_TOTAL_PARAMS == 119_477_664


def test_block_and_head_recomputed_from_the_architecture():
    """Recompute independently of the module's own expressions, from d_model/d_ff/quantiles."""
    d, dff, q, p = 768, 3072, 21, 16
    mha = 4 * d * d                                  # q,k,v,o, bias=False, inner_dim == d_model
    mlp = d * dff + dff * d                          # NON-gated (layers.py asserts not is_gated_act)
    block = 2 * (mha + d) + (mlp + d)                # 2 attentions + FF, each with one RMSNorm
    assert block == model_size.BLOCK_PARAMS == 9_439_488
    head = (d * dff + dff) + (dff * q * p + q * p) + (d * q * p + q * p)   # ResidualBlock, biases on
    assert head == model_size.NATIVE_HEAD_PARAMS == 3_653_280
    emb = (48 * dff + dff) + (dff * d + d) + (48 * d + d)                  # in_dim = patch_size * 3
    assert emb == model_size.INPUT_EMBEDDING_PARAMS == 2_548_224
    assert model_size.REG_EMBEDDING_PARAMS == 1_536 and model_size.FINAL_RMSNORM_PARAMS == 768
    assert model_size.ADAPTER_PARAMS == 768 * 768 + 768 == 590_592


def test_active_params_step_is_exactly_one_block():
    for l in range(model_size.NUM_BLOCKS):
        step = model_size.active_params(l + 1) - model_size.active_params(l)
        assert step == model_size.BLOCK_PARAMS, (l, step)


def test_adapter_convention_makes_depth_12_exceed_the_stock_model():
    """Numerator includes the adapter, denominator does not -> depth 12 is 100.5%, not 100%."""
    assert model_size.active_params(12, include_adapter=False) == model_size.TOTAL_PARAMS
    assert (model_size.active_params(12) - model_size.TOTAL_PARAMS) == model_size.ADAPTER_PARAMS
    assert model_size.active_fraction(12) > 1.0


def test_active_params_rejects_out_of_range_depth():
    for bad in (-1, 13):
        try:
            model_size.active_params(bad)
        except ValueError:
            continue
        raise AssertionError(f"depth {bad} should have been rejected")


def test_published_percentages_are_reproduced():
    """The exact figures used in the paper table."""
    for depth, pct in ((1, 13.6), (3, 29.4), (6, 53.1), (8, 68.9), (10, 84.7), (11, 92.6)):
        assert round(100 * model_size.active_fraction(depth), 1) == pct, depth


# --------------------------------------------------------------------------- #
# FLOPs
# --------------------------------------------------------------------------- #
def test_block_flops_fraction_is_the_exact_ratio():
    for depth in range(model_size.NUM_BLOCKS + 1):
        assert model_size.block_flops_fraction(depth) == depth / 12


def test_encoder_token_layout_matches_encode():
    ncp, k, ntok = model_size.num_encoder_tokens(512, 64)
    assert (ncp, k, ntok) == (32, 4, 37)             # ceil(C/16) content + 1 REG + ceil(H/16) slots


def test_end_to_end_flops_exceed_block_only_because_of_fixed_costs():
    """The embedding and head do not shrink, so end-to-end is always >= depth/12, and the gap is
    largest at shallow depth. This is exactly why depth/12 must be labelled 'block FLOPs'."""
    for depth in range(1, model_size.NUM_BLOCKS):
        assert model_size.end_to_end_flops_fraction(depth) > model_size.block_flops_fraction(depth)
    gaps = [model_size.end_to_end_flops_fraction(d) - model_size.block_flops_fraction(d)
            for d in (1, 6, 11)]
    assert gaps[0] > gaps[1] > gaps[2] > 0


def test_block_stack_dominates_forward_macs():
    m = model_size.forward_macs(model_size.NUM_BLOCKS, include_adapter=False)
    assert 0.95 < m["blocks"] / m["total"] < 0.99    # ~97.5%: why depth/12 approximates end-to-end


# --------------------------------------------------------------------------- #
# verification against a real model (checked here against a FAKE one)
# --------------------------------------------------------------------------- #
class _FakeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(model_size.BLOCK_PARAMS))


class _FakeChronos(nn.Module):
    """Module names mirror Chronos2Model so group_parameters exercises the real bucketing rules."""

    def __init__(self, n_blocks=12, extra=False):
        super().__init__()
        self.input_patch_embedding = nn.Parameter(torch.zeros(model_size.INPUT_EMBEDDING_PARAMS))
        self.shared = nn.Parameter(torch.zeros(model_size.REG_EMBEDDING_PARAMS))
        self.output_patch_embedding = nn.Parameter(torch.zeros(model_size.NATIVE_HEAD_PARAMS))
        self.encoder = nn.Module()
        self.encoder.block = nn.ModuleList([_FakeBlock() for _ in range(n_blocks)])
        self.encoder.final_layer_norm = nn.Parameter(torch.zeros(model_size.FINAL_RMSNORM_PARAMS))
        if extra:
            self.mystery_module = nn.Parameter(torch.zeros(7))


def test_verify_against_model_passes_on_a_faithful_model():
    r = model_size.verify_against_model(_FakeChronos())
    assert r["ok"], {k: v for k, v in r["checks"].items() if not v[0]}
    assert r["checks"]["total"][2] == model_size.TOTAL_PARAMS


def test_verify_against_model_catches_an_unaccounted_module():
    """A module the accounting does not know about must surface, not be silently absorbed."""
    r = model_size.verify_against_model(_FakeChronos(extra=True))
    assert not r["ok"]
    assert not r["checks"]["no_unmatched_parameters"][0]
    assert any(k.startswith("other::") for k in r["groups"])


def test_verify_against_model_catches_a_wrong_block_count():
    r = model_size.verify_against_model(_FakeChronos(n_blocks=11))
    assert not r["ok"] and not r["checks"]["num_blocks"][0] and not r["checks"]["total"][0]


# --------------------------------------------------------------------------- #
# driver: LaTeX + provenance gates against committed artifacts
# --------------------------------------------------------------------------- #
def _driver():
    from experiments import run_compression_cost as rc
    return rc


def test_latex_emitters_render_the_expected_cells():
    rc = _driver()
    row = {"depth_label": "L3", "active_fraction": 0.294, "block_flops_fraction": 0.25,
           "relative_mase": 0.086, "relative_mase_ci_lo": 0.072, "relative_mase_ci_hi": 0.101,
           "truncated_mase": 0.875, "native_mase": 0.806}
    tags = ["uber_tlc_hourly"]
    one = rc.latex_single_rule({tags[0]: row}, tags, "saturation")
    assert r"\begin{tabular}" in one and r"\bottomrule" in one
    assert r"\(29.4\%\)" in one and r"\(0.25\times\)" in one and r"\(+8.6\%\)" in one
    two = rc.latex_two_rule({"saturation": {tags[0]: row}, "erank": {tags[0]: row}}, tags)
    assert two.count(r"\(29.4\%\)") == 2 and r"\cmidrule(lr){6-9}" in two


def test_saturation_depth_equals_the_committed_l_start():
    rc = _driver()
    checked = 0
    for tag in rc.PT_ID_TAGS:
        p = rc.TUNNEL_DIR / f"{tag}__fslot__{rc.QSET}__{rc.PROTO}__{rc.RUNS_TAG}.json"
        if not p.exists():
            continue
        depth, prov = rc.saturation_depth(tag)       # re-derives and asserts internally
        assert depth == int(json.load(open(p))["l_start"])
        assert "verified" in prov
        checked += 1
    print(f"  (checked {checked}/{len(rc.PT_ID_TAGS)} committed tunnel records)"
          if checked else "  (skipped: no committed tunnel records)")


def test_erank_depth_equals_argmax_of_the_committed_spectral_record():
    rc = _driver()
    checked = 0
    for tag in rc.ALL_TAGS:
        p = rc.SPEC_DIR / f"spectral__{tag}__fslot__probe_input__train.json"
        if not p.exists():
            continue
        er = np.array([x["effective_rank"] for x in json.load(open(p))["layers"]])
        assert rc.erank_depth(tag)[0] == int(er.argmax())
        checked += 1
    print(f"  (checked {checked}/{len(rc.ALL_TAGS)} spectral records)"
          if checked else "  (skipped: no committed spectral records)")


def test_paired_bootstrap_reproduces_the_committed_gap_recovery_cis():
    """The driver's bootstrap must be the same estimator run_native_head_adapter already committed.
    Reproducing its CI bounds proves the resampling matrix, clustering and percentiles all agree."""
    rc = _driver()
    p = rc.NHA_TAB / "native_head_adapter__gap_recovery__all.csv"
    if not p.exists():
        print("  (skipped: no committed gap_recovery table)")
        return
    checked = 0
    for row in csv.DictReader(open(p)):
        tag, depth = row["dataset"], int(row["layer"])
        if depth not in (3, 6, 8, 10, 11) or not (rc.NHA_BOOT /
                                                  f"native_head_adapter__{tag}.npz").exists():
            continue
        z, _lin, series = rc._load_window_metrics(tag)
        b = rc._Boot(series)
        lab = f"L{depth:02d}"
        nat, ada = b.mean(z["native__L13__mase"]), b.mean(z[f"linear_adapter__{lab}__mase"])
        zs = b.mean(z[f"zero_shot__{lab}__mase"])
        from probing.stats import ci_bounds
        lo, hi = ci_bounds((zs - ada) / (zs - nat))
        assert np.isclose(lo, float(row["R_boot_lo"]), atol=1e-6), (tag, depth, lo, row["R_boot_lo"])
        assert np.isclose(hi, float(row["R_boot_hi"]), atol=1e-6), (tag, depth, hi, row["R_boot_hi"])
        checked += 1
    print(f"  (reproduced {checked} committed gap-recovery CIs)")


def test_evaluate_gates_on_the_committed_relative_regret():
    rc = _driver()
    tag = "uber_tlc_hourly"
    if not (rc.NHA_BOOT / f"native_head_adapter__{tag}.npz").exists():
        print("  (skipped: no committed ext_v5 bootstrap inputs)")
        return
    depth, prov = rc.saturation_depth(tag)
    r = rc.evaluate(tag, depth, "saturation", prov)          # raises if it disagrees
    assert r["depth"] == depth and r["n_series"] > 0
    assert r["active_params"] == model_size.active_params(depth)
    assert r["block_flops_fraction"] == depth / 12
    assert r["relative_mase_ci_lo"] < r["relative_mase"] < r["relative_mase_ci_hi"]
    assert "identical" in r["window_provenance"]             # selection and scoring share windows


def test_readout_index_13_is_the_full_encoder_not_a_13th_block():
    """L12+RMS is the post-final-RMSNorm state the native head consumes: 12 blocks, no truncation,
    and no adapter (the head reads it directly). Costing it as "depth 13" would be wrong."""
    assert model_size.blocks_for_readout(13) == model_size.NUM_BLOCKS
    for i in range(model_size.NUM_BLOCKS + 1):
        assert model_size.blocks_for_readout(i) == i
    c = model_size.cost_of_readout(13)
    assert c["n_blocks"] == 12 and c["needs_adapter"] is False
    assert c["active_params"] == model_size.TOTAL_PARAMS          # exactly the stock model
    assert c["active_fraction"] == 1.0 and c["block_flops_fraction"] == 1.0
    assert model_size.cost_of_readout(12)["needs_adapter"] is True
    for bad in (-1, 14):
        try:
            model_size.blocks_for_readout(bad)
        except ValueError:
            continue
        raise AssertionError(f"readout index {bad} should have been rejected")


def test_stricter_tolerance_never_selects_a_shallower_depth():
    """The criterion is a first-crossing of (1+eps)*final, so shrinking eps can only push the
    selected depth later. A violation would mean tunnel_start is not monotone in the tolerance."""
    rc = _driver()
    tags = [t for t in rc.ALL_TAGS
            if (rc.TUNNEL_DIR / f"{t}__fslot__{rc.QSET}__{rc.PROTO}__{rc.RUNS_TAG}.json").exists()
            or (rc.PTOOD_DIR / "per_target" / f"{t}__{rc.QSET}__seed0.json").exists()]
    if not tags:
        print("  (skipped: no committed validation curves)")
        return
    eps = sorted(rc.DEFAULT_EPS, reverse=True)
    rows = {(r["dataset"], r["epsilon"]): r for r in rc.threshold_rows(tags, eps)}
    for t in tags:
        depths = [rows[(t, e)]["depth"] for e in eps]
        assert depths == sorted(depths), (t, eps, depths)
    print(f"  (checked monotonicity on {len(tags)} datasets x {len(eps)} tolerances)")


def test_threshold_figures_render(tmpdir=None):
    rc = _driver()
    import tempfile
    from pathlib import Path
    tags = [t for t in rc.ALL_TAGS
            if (rc.TUNNEL_DIR / f"{t}__fslot__{rc.QSET}__{rc.PROTO}__{rc.RUNS_TAG}.json").exists()]
    if not tags:
        print("  (skipped: no committed validation curves)")
        return
    with tempfile.TemporaryDirectory() as d:
        out = Path(d)
        rows = rc.make_threshold_figures(tags, sorted(rc.DEFAULT_EPS, reverse=True), out)
        assert len(rows) == len(tags) * len(rc.DEFAULT_EPS)
        for stem in ("threshold_sensitivity_curves", "threshold_sensitivity_depths"):
            for ext in ("png", "pdf"):
                f = out / f"{stem}.{ext}"
                assert f.exists() and f.stat().st_size > 5000, f
    print(f"  (rendered both panels for {len(tags)} datasets to a tempdir)")


def _latency_fixture():
    """A minimal latency record with the shape run_latency writes."""
    def row(d, b, ms):
        return {"depth": d, "depth_label": ("Emb" if d == 0 else f"L{d}"), "batch_size": b,
                "predict_median_ms": ms, "predict_p95_ms": ms * 1.02, "encode_median_ms": ms * 0.9,
                "throughput_series_per_s": b / (ms / 1e3), "peak_mib": 100.0 + 30 * d,
                "weights_mib": 60.0 + 30 * d, "warmup": 20, "reps": 100}
    return {"environment": {"gpu_name": "TestGPU", "gpu_total_mib": 40960, "gpu_capability": "8.0",
                            "driver_version": "1.2.3", "python": "3.11.4", "torch": "2.12.1",
                            "torch_cuda": "13.2", "cudnn": 91002, "cpu_model": "Test CPU",
                            "slurm_cpus_per_task": "2", "model_id": "amazon/chronos-2",
                            "context_length": 512, "horizon": 64, "forecast_slots": 4,
                            "encoder_tokens": 37, "tf32_matmul": False, "tf32_cudnn": False,
                            "verification": {"L3": 0.0, "L12": 0.0}},
            "rows": [row(3, 1, 6.0), row(12, 1, 18.0), row(3, 256, 55.0), row(12, 256, 180.0)]}


def test_latency_lookup_is_referenced_to_the_full_model():
    rc = _driver()
    lut = rc.latency_by_depth(_latency_fixture(), 256)
    assert lut[12]["speedup"] == 1.0                      # the reference is the full 12-block model
    assert abs(lut[3]["speedup"] - 180.0 / 55.0) < 1e-12
    assert lut[3]["memory_ratio"] == lut[3]["peak_mib"] / lut[12]["peak_mib"]


def test_latency_lookup_fails_loud_on_a_missing_batch_or_reference():
    rc = _driver()
    lat = _latency_fixture()
    for mutate, needle in ((lambda d: d, "batch size 7"),
                           (lambda d: {**d, "rows": [r for r in d["rows"] if r["depth"] != 12]},
                            "full 12-block")):
        try:
            rc.latency_by_depth(mutate(lat), 7 if "batch" in needle else 256)
        except ValueError as e:
            assert needle in str(e), (needle, str(e))
            continue
        raise AssertionError(f"expected a ValueError mentioning {needle!r}")


def test_full_table_carries_the_measured_speedup_not_the_flop_ratio():
    rc = _driver()
    row = {"depth": 3, "depth_label": "L3", "active_fraction": 0.294, "block_flops_fraction": 0.25,
           "relative_mase": 0.086, "relative_mase_ci_lo": 0.072, "relative_mase_ci_hi": 0.101,
           "truncated_mase": 0.875, "native_mase": 0.806}
    tags = ["uber_tlc_hourly"]
    tex = rc.latex_full_table({"saturation": {tags[0]: row}, "erank": {tags[0]: row}}, tags,
                              _latency_fixture(), 256)
    assert r"\(3.27\times\)" in tex, tex          # 180/55, the MEASURED ratio
    assert r"\(0.25\times\)" not in tex           # not the FLOP ratio
    assert "TestGPU" in tex and "batch 256" in tex
    assert tex.count(r"\bottomrule") == 1


def test_appendix_states_the_protocol_and_never_invents_missing_fields():
    rc = _driver()
    lat = _latency_fixture()
    tex = rc.latex_appendix_methodology(lat, 256)
    for needle in ("20 iterations", "100 repetitions", "torch.cuda.synchronize",
                   "max\_memory\_allocated", "float32", "TestGPU", "1.2.3", "Test CPU",
                   "bit-identical", r"\label{app:latency-methodology}", "Series per second"):
        assert needle in tex, needle
    assert "not recorded" not in tex              # every field present -> no markers

    stripped = {**lat, "environment": {k: v for k, v in lat["environment"].items()
                                       if k not in ("cpu_model", "cudnn")}}
    tex2 = rc.latex_appendix_methodology(stripped, 256)
    assert tex2.count("not recorded") == 2        # marked, not fabricated
    assert "Test CPU" not in tex2


def test_appendix_reports_the_equivalence_gate_or_says_it_was_skipped():
    rc = _driver()
    lat = _latency_fixture()
    assert "bit-identical" in rc.latex_appendix_methodology(lat, 256)
    no_gate = {**lat, "environment": {k: v for k, v in lat["environment"].items()
                                      if k != "verification"}}
    tex = rc.latex_appendix_methodology(no_gate, 256)
    assert "gate not run" in tex and "bit-identical" not in tex


def test_committed_latency_run_renders_end_to_end():
    rc = _driver()
    lat = rc.load_latency()
    if lat is None:
        print("  (skipped: no latency run on disk — sbatch -J lat job_latency.sh --verify)")
        return
    tex = rc.latex_appendix_methodology(lat, 256)
    assert tex.count(r"\begin{tabular}") == tex.count(r"\end{tabular}") == 1
    assert tex.count("{") == tex.count("}")
    n = len({(r["batch_size"], r["depth"]) for r in lat["rows"]})
    assert sum(1 for l in tex.splitlines() if l.strip().endswith(r"\\")) >= n
    print(f"  (rendered the appendix from the committed {lat['environment'].get('gpu_name')} run, "
          f"{n} configurations)")


def test_env_backfill_fills_only_missing_fields_and_discloses_itself():
    rc = _driver()
    import json, tempfile, pathlib as _pl
    lat = _latency_fixture()
    lat["environment"] = {k: v for k, v in lat["environment"].items()
                          if k not in ("cpu_model", "cudnn")}
    probe = _latency_fixture()
    probe["environment"]["cpu_model"] = "Probe CPU"
    probe["environment"]["driver_version"] = "SHOULD-NOT-BE-COPIED"   # already recorded upstream
    with tempfile.TemporaryDirectory() as d:
        f = _pl.Path(d) / "latency__env_probe.json"
        f.write_text(json.dumps(probe))
        merged = rc.backfill_environment(lat, f)
        assert merged["environment"]["cpu_model"] == "Probe CPU"
        assert merged["environment"]["cudnn"] == probe["environment"]["cudnn"]
        # a field the timing run DID record must never be overwritten by the probe
        assert merged["environment"]["driver_version"] == "1.2.3"
        assert set(merged["_backfilled"]) == {"cpu_model", "cudnn"}
        tex = rc.latex_appendix_methodology(merged, 256)
        assert "not recorded" not in tex
        assert "separate probe run" in tex and "latency\\_\\_env\\_probe.json" in tex
        # the disclosure must ride on real text, not sit as an orphaned \footnote
        assert "\n\\footnote{" not in tex


def test_env_backfill_refuses_a_probe_from_different_hardware():
    rc = _driver()
    import json, tempfile, pathlib as _pl
    lat = _latency_fixture()
    probe = _latency_fixture()
    probe["environment"]["gpu_name"] = "SomeOtherGPU"
    with tempfile.TemporaryDirectory() as d:
        f = _pl.Path(d) / "probe.json"
        f.write_text(json.dumps(probe))
        try:
            rc.backfill_environment(lat, f)
        except ValueError as e:
            assert "does not describe this run" in str(e)
            return
    raise AssertionError("a probe from a different GPU must be refused")


def test_load_latency_picks_the_sweep_not_a_newer_probe():
    """Regression: an env probe is written AFTER the sweep, so newest-by-mtime picks the probe and
    every later lookup fails with a confusing 'no batch size 256'."""
    rc = _driver()
    import json, os, tempfile, time, pathlib as _pl
    sweep = _latency_fixture()                                   # 4 configurations
    probe = {**_latency_fixture(), "rows": _latency_fixture()["rows"][:1]}
    with tempfile.TemporaryDirectory() as d:
        dd = _pl.Path(d)
        (dd / "latency__A100.json").write_text(json.dumps(sweep))
        pf = dd / "latency__env_probe.json"
        pf.write_text(json.dumps(probe))
        old = time.time() - 3600
        os.utime(dd / "latency__A100.json", (old, old))           # sweep is OLDER on disk
        got = rc.load_latency(lat_dir=dd)
        assert got["_source"] == "latency__A100.json", got["_source"]
        assert len(got["rows"]) == len(sweep["rows"])
        # and the probe is skipped outright when it is the --env-from file
        assert rc.load_latency(exclude=pf, lat_dir=dd)["_source"] == "latency__A100.json"
        # an explicit path still wins
        assert rc.load_latency(path=pf, lat_dir=dd)["_source"] == "latency__env_probe.json"


def test_missing_batch_error_names_the_run_and_the_way_out():
    rc = _driver()
    lat = {**_latency_fixture(), "_source": "latency__env_probe.json"}
    lat["rows"] = [r for r in lat["rows"] if r["batch_size"] == 1]
    try:
        rc.latency_by_depth(lat, 256)
    except ValueError as e:
        for needle in ("latency__env_probe.json", "--speedup-batch", "--latency-json"):
            assert needle in str(e), (needle, str(e))
        return
    raise AssertionError("expected a ValueError naming the run and the remedy")


if __name__ == "__main__":
    tests = [test_families_sum_to_the_recorded_total,
             test_block_and_head_recomputed_from_the_architecture,
             test_active_params_step_is_exactly_one_block,
             test_adapter_convention_makes_depth_12_exceed_the_stock_model,
             test_active_params_rejects_out_of_range_depth,
             test_published_percentages_are_reproduced,
             test_block_flops_fraction_is_the_exact_ratio,
             test_encoder_token_layout_matches_encode,
             test_end_to_end_flops_exceed_block_only_because_of_fixed_costs,
             test_block_stack_dominates_forward_macs,
             test_verify_against_model_passes_on_a_faithful_model,
             test_verify_against_model_catches_an_unaccounted_module,
             test_verify_against_model_catches_a_wrong_block_count,
             test_latex_emitters_render_the_expected_cells,
             test_saturation_depth_equals_the_committed_l_start,
             test_erank_depth_equals_argmax_of_the_committed_spectral_record,
             test_paired_bootstrap_reproduces_the_committed_gap_recovery_cis,
             test_evaluate_gates_on_the_committed_relative_regret,
             test_readout_index_13_is_the_full_encoder_not_a_13th_block,
             test_stricter_tolerance_never_selects_a_shallower_depth,
             test_threshold_figures_render,
             test_latency_lookup_is_referenced_to_the_full_model,
             test_latency_lookup_fails_loud_on_a_missing_batch_or_reference,
             test_full_table_carries_the_measured_speedup_not_the_flop_ratio,
             test_appendix_states_the_protocol_and_never_invents_missing_fields,
             test_appendix_reports_the_equivalence_gate_or_says_it_was_skipped,
             test_committed_latency_run_renders_end_to_end,
             test_env_backfill_fills_only_missing_fields_and_discloses_itself,
             test_env_backfill_refuses_a_probe_from_different_hardware,
             test_load_latency_picks_the_sweep_not_a_newer_probe,
             test_missing_batch_error_names_the_run_and_the_way_out]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} compression-cost tests passed.")
