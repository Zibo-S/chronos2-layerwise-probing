"""Stage-A contracts for the FT-specialization full fine-tuning (probing/finetune.py).

No model, no GPU, no dataset download — synthetic series + tiny torch modules, so this runs on a
login node:

    OMP_NUM_THREADS=2 python -m tests.test_ft_specialization

Stage A produces the INTERVENTION (a full fine-tuning of Chronos-2 on one source, checkpointed at
300/1000 steps). The two fixed-1073-window pilots overfit (the corpus wrongly reused the probe
subsample), so the FT data was REDESIGNED to Chronos-2's own random-cut-point sampler over the
COMPLETE, leakage-truncated source histories. The runtime acceptance criteria that need a GPU
(weights move in early/middle/late blocks + head; stage hashes differ; ft_val finite; frozen
singleton untouched; 14-point FT extraction) are verified in the pilot. What is checkable WITHOUT a
model lives here:
  * build_ft_data: per-series leakage truncation at the ft_val target start (starts[-3]+C), the fixed
    starts[-3] ft_val window, unique-training-window count (>> the rejected 1073), <3-origin skip, and
    fail-loud when nothing samplable survives;
  * the collision-proof FT feature-cache prefix vs the pretrained namespace;
  * extract_kout_features's dependency-injection kwargs (pipeline / cache_prefix) default to
    BYTE-IDENTICAL legacy behavior and route FT caches to a disjoint path;
  * parameter-drift bucketing + math (per-block / head, vs the pretrained reference);
  * the checkpoint identity hash (sha256(model.safetensors)[:8]);
  * the official-defaults record.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from probing import config as _config
_config.set_dataset_set("extended_v3_rolling")   # the FT geometry's active set; build_ft_data's guard
                                                 # becomes a no-op so these synthetic tests stay hermetic

import probing.finetune as ft
import experiments.run_ft_specialization as rfs
from probing.extraction import extract_kout_features, _idf_prefix, _cache_path
from probing.config import OUTPUT_PATCH_SIZE, NUM_LAYERS
from probing.probes import QUANTILE_SETS, validate_quantiles

C, H = ft.FT_C, ft.FT_H          # 512, 64


def _series(length, seed=0):
    """A smooth, finite, non-constant series so EVERY H-spaced rolling origin is valid — lets us
    predict _rolling_valid_starts exactly. `length` sets the origin count."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    return (10.0 + np.sin(2 * np.pi * t / 24) + 0.01 * rng.standard_normal(length)).astype(np.float64)


def _n_starts(length):
    """Number of H-spaced valid origins for a full-finite series of this length (== len(_rolling_valid_starts))."""
    return len(range(0, length - (C + H) + 1, H))


def _patch_series(series_list):
    """Monkeypatch probing.finetune.load_seen_series to return `series_list` (returns the original)."""
    orig = ft.load_seen_series
    ft.load_seen_series = lambda _tag: series_list
    return orig


# 1. Leakage truncation + corpus counts: each eligible (>=3 origins) series is truncated to its
#    ft_val target start (starts[-3]+C); series with <3 origins are skipped; the reported unique-window
#    count is the summed samplable cut points (>> the rejected 1073).
def test_build_ft_data_truncation_and_counts():
    L = C + H * 8                    # 1024 -> 8 origins; starts[-3]=320, cutoff=832
    series = [_series(L, 1), _series(L, 2), _series(C + H, 3)]   # third = 1 origin -> skipped
    starts = list(range(0, L - (C + H) + 1, H))
    ftv, cutoff = starts[-3], starts[-3] + C
    orig = _patch_series(series)
    try:
        data = ft.build_ft_data("fake", min_past=C)
    finally:
        ft.load_seen_series = orig
    m = data["meta"]
    assert m["n_series_total"] == 3 and m["n_eligible_series"] == 2, m
    assert m["n_ft_train_series"] == 2 and m["n_ft_val"] == 2
    assert m["rejected_fixed_window_corpus"] == 1073
    per_series_windows = (cutoff - C - H + 1)            # samplable cut points in one truncated history
    assert m["n_unique_train_windows"] == 2 * per_series_windows > 1073 // 100  # >> a tiny handful
    # every training history is truncated exactly at the ft_val target start
    for hist in data["train_histories"]:
        assert len(hist) == cutoff, f"history length {len(hist)} != cutoff {cutoff}"
    # cutoff precedes the preserved probe-val (starts[-2]+C) and test (starts[-1]+C) targets
    assert cutoff < starts[-2] + C < starts[-1] + C


# 2. The fixed ft_val window is exactly each series' starts[-3] window (raw context + raw future).
def test_build_ft_data_ftval_is_starts_minus_3():
    L = C + H * 8
    s = _series(L, 7)
    starts = list(range(0, L - (C + H) + 1, H))
    ftv = starts[-3]
    orig = _patch_series([s])
    try:
        data = ft.build_ft_data("fake", min_past=C)
    finally:
        ft.load_seen_series = orig
    np.testing.assert_allclose(data["X_ft_val"][0], s[ftv:ftv + C].astype(np.float32), rtol=0, atol=0)
    np.testing.assert_allclose(data["y_ft_val"][0], s[ftv + C:ftv + C + H].astype(np.float32), rtol=0, atol=0)


# 3. Series with < 3 rolling origins never enter ft_val or the training corpus.
def test_build_ft_data_skips_short_series():
    assert _n_starts(C + H) == 1 and _n_starts(C + 2 * H) == 2   # both < 3 origins
    short = [_series(C + H, 1), _series(C + 2 * H, 2)]
    orig = _patch_series(short)
    try:
        ft.build_ft_data("fake", min_past=C)
        raise AssertionError("all-short series should have raised (no eligible/samplable data)")
    except RuntimeError:
        pass
    finally:
        ft.load_seen_series = orig


# 4. Fail-loud when series are eligible for ft_val but too short to yield a full-context train window
#    (cutoff history shorter than min_past+H) — no silent empty corpus.
def test_build_ft_data_fail_loud_no_samplable_windows():
    # 704 -> 3 origins (eligible), ftv=starts[0]=0, cutoff=512 < min_past+H=576 -> 0 train windows
    L = C + 3 * H
    assert _n_starts(L) == 3
    orig = _patch_series([_series(L, 1), _series(L, 2)])
    try:
        ft.build_ft_data("fake", min_past=C)
        raise AssertionError("no samplable training window should raise RuntimeError")
    except RuntimeError as e:
        assert "no leakage-safe FT-train history" in str(e)
    finally:
        ft.load_seen_series = orig


# 4b. PT-OOD source path: OOD tags load via load_ood_target_series (not load_seen_series) and take the
#     SAME leakage truncation. This is the active pilot source (BOOM) — the model has room to specialize.
def test_build_ft_data_pt_ood_source():
    from probing.id_data import OOD_TARGET_TAGS
    assert "boom_hourly" in OOD_TARGET_TAGS
    L = C + H * 8
    series = [_series(L, 1), _series(L, 2)]
    starts = list(range(0, L - (C + H) + 1, H))
    orig = ft.load_ood_target_series
    ft.load_ood_target_series = lambda _tag: {          # stand in for the OOD arrow loader
        "series": series, "cluster_ids": [0, 1], "cluster_unit": "metric_query",
        "cluster_names": ["q0", "q1"], "notes": "synthetic"}
    orig_seen = ft.load_seen_series
    ft.load_seen_series = lambda _tag: (_ for _ in ()).throw(   # must NOT be used for an OOD source
        AssertionError("PT-OOD source must not call load_seen_series"))
    try:
        data = ft.build_ft_data("boom_hourly", min_past=C)
    finally:
        ft.load_ood_target_series = orig
        ft.load_seen_series = orig_seen
    assert data["meta"]["source_kind"] == "pt_ood:metric_query"
    assert data["meta"]["n_ft_train_series"] == 2 and data["meta"]["n_ft_val"] == 2
    for hist in data["train_histories"]:
        assert len(hist) == starts[-3] + C, "OOD source not truncated at the ft_val target start"


# 5. FT cache prefix carries source/stage/hash and cannot collide with the pretrained namespace.
def test_ft_cache_prefix_disjoint_from_pretrained():
    tag = "monash_electricity_hourly"
    pref = ft.ft_cache_prefix(tag, "electricity", "stage1_ft_early", "deadbeef")
    assert pref == f"IDF_{tag}__ft__electricity__stage1_ft_early__deadbeef"
    ft_path = _cache_path(pref, "train", None, f"K4_H{H}")
    pt_path = _cache_path(_idf_prefix(tag), "train", None, f"K4_H{H}")
    assert ft_path != pt_path
    assert "__ft__" in ft_path.name and "__ft__" not in pt_path.name


# 6. extract_kout_features gained pipeline / cache_prefix kwargs, BOTH default None (legacy-safe),
#    and cache_prefix routes to the injected namespace.
def test_extract_kout_injection_signature_and_routing():
    sig = inspect.signature(extract_kout_features)
    assert "pipeline" in sig.parameters and sig.parameters["pipeline"].default is None
    assert "cache_prefix" in sig.parameters and sig.parameters["cache_prefix"].default is None
    assert list(sig.parameters)[:5] == ["tag", "split", "contexts", "y", "horizon"]
    pref = ft.ft_cache_prefix("monash_electricity_hourly", "electricity", "stage2_ft_late", "0badf00d")
    assert _cache_path(pref, "test", None, f"K4_H{H}").name.startswith(pref)


# 7. Parameter-drift bucketing: names map to input-embed / REG / per-block / final-LN / head groups.
def test_param_group_buckets():
    cases = {
        "input_patch_embedding.hidden_layer.weight": "input_patch_embedding",
        "shared.weight": "reg_embedding",
        "encoder.final_layer_norm.weight": "final_layer_norm",
        "output_patch_embedding.output_layer.weight": "native_head",
        "encoder.block.0.time_self_attention.self_attention.q.weight": "block_00",
        "encoder.block.11.mlp.wo.weight": "block_11",
    }
    for name, grp in cases.items():
        assert ft._param_group(name) == grp, f"{name} -> {ft._param_group(name)} != {grp}"


# 8. param_drift measures per-group L2 / relative drift vs the pretrained reference and flags only
#    the groups whose weights actually moved (an EARLY block, a LATE block, and the head here).
def test_param_drift_detects_changed_groups():
    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.input_patch_embedding = torch.nn.Linear(4, 4)
            self.shared = torch.nn.Embedding(2, 4)
            self.output_patch_embedding = torch.nn.Linear(4, 4)
            enc = torch.nn.Module()
            enc.final_layer_norm = torch.nn.LayerNorm(4)
            enc.block = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
            self.encoder = enc

    torch.manual_seed(0)
    model = _Tiny()
    ref = ft.snapshot_reference_state(model)
    with torch.no_grad():                       # perturb an early block, a late block, and the head
        model.encoder.block[0].weight += 0.1
        model.encoder.block[2].weight += 0.2
        model.output_patch_embedding.weight += 0.3
    drift = ft.param_drift(model, ref)
    for g in ("block_00", "block_02", "native_head"):
        assert drift[g]["changed"] and drift[g]["l2"] > 0, f"{g} should register drift"
    for g in ("block_01", "input_patch_embedding", "reg_embedding", "final_layer_norm"):
        assert not drift[g]["changed"] and drift[g]["l2"] == 0.0, f"{g} should be unchanged"
    assert 0 < drift["native_head"]["relative"] < float("inf")


# 9. checkpoint_hash = sha256(model.safetensors)[:8]; missing file fails loud.
def test_checkpoint_hash():
    import hashlib
    import tempfile
    from pathlib import Path

    d = Path(tempfile.mkdtemp())
    payload = b"not-really-safetensors-but-hashing-is-over-bytes"
    (d / "model.safetensors").write_bytes(payload)
    assert ft.checkpoint_hash(d) == hashlib.sha256(payload).hexdigest()[:8]
    assert len(ft.checkpoint_hash(d)) == 8
    empty = Path(tempfile.mkdtemp())
    try:
        ft.checkpoint_hash(empty)
        raise AssertionError("missing model.safetensors should raise FileNotFoundError")
    except FileNotFoundError:
        pass


# 10. The official-defaults record matches the installed 2.3.1 fit() (batch 256 is the OFFICIAL value;
#     the run default is the locked-reduced 64).
def test_ft_defaults_match_official():
    d = ft.FT_DEFAULTS
    assert d["finetune_mode"] == "full"
    assert d["learning_rate"] == 1e-6
    assert d["num_steps"] == 1000
    assert d["batch_size"] == 256           # official record; runs use --batch-size 64
    assert d["lr_scheduler_type"] == "linear"
    assert d["warmup_steps"] == 0
    assert d["max_grad_norm"] == 1.0
    assert d["gradient_accumulation_steps"] == 1
    assert d["adam_betas"] == (0.9, 0.999) and d["adam_eps"] == 1e-8
    assert d["weight_decay"] == 0.0 and d["seed"] == 0
    assert ft.DEFAULT_CHECKPOINT_STEPS == {300: "stage1_ft_early", 1000: "stage2_ft_late"}
    assert ft.SOURCE_TAGS["electricity"] == "monash_electricity_hourly"
    assert (ft.FT_C, ft.FT_H) == (512, 64)
    import math
    assert math.ceil(H / OUTPUT_PATCH_SIZE) == 4      # K=4 at H=64
    # the CLI run default is the locked-reduced batch 64, min_past = full context
    sig = inspect.signature(ft.finetune)
    assert sig.parameters["batch_size"].default == 64
    assert sig.parameters["min_past"].default == ft.FT_C


# ============================ Stage B (run_ft_specialization) ============================ #
# The B1 GPU extraction is verified in the run; what is checkable WITHOUT the caches lives here:
# the fslot cache namespacing (FT vs pretrained) + fail-loud, and the B2 probe -> tunnel -> figure
# wiring driven end-to-end on synthetic 14-point forecast-slot features (no model, no windows).
NPTS = NUM_LAYERS + 1            # 14 fslot points: Emb, L1..L12, post-final-LN native-head input


def _syn_feats(n, seed):
    rng = np.random.default_rng(seed)
    return {i: rng.standard_normal((n, rfs.K, 768)).astype(np.float32) for i in range(NPTS)}


def _syn_traj(n, seed):
    return np.random.default_rng(seed + 999).standard_normal((n, rfs.H)).astype(np.float32)


def _setup_stageB(tmp):
    """Point the B2 output dirs at a tmp tree and shrink the fit/bootstrap so the synthetic run is fast."""
    from pathlib import Path
    from probing import config as _c
    tmp = Path(tmp)
    rfs.PROBE_DIR = tmp / "probes"; rfs.TUNNEL_DIR = tmp / "tunnels"
    rfs.FIG_DIR = tmp / "figures"; rfs.TABLE_DIR = tmp / "tables"
    rfs.QUANTILE_EPOCHS = 6; rfs.WD_GRID = (1e-3, 1e-2)
    _c.BOOT_B = 40


# B1. The fslot cache path is FT-namespaced for the FT stages (matches the committed smoke report) and
#     falls back to the default committed namespace (__ood) for stage0; a missing FT cache fails loud.
def test_stageB_cache_path_and_fail_loud():
    early = rfs.Stage("stage1_ft_early", "/scratch/ck", "18c93f86")
    p = rfs._fslot_cache_path(early, "boom_hourly", "test_rolling")
    assert p.name == ("IDF_boom_hourly__ft__boom__stage1_ft_early__18c93f86"
                      "__test_rolling__clean__K4_H64.npz"), p.name
    s0 = rfs.Stage(rfs.STAGE0, None, None)
    assert rfs._fslot_cache_path(s0, "boom_hourly", "test_rolling").name == \
        "IDF_boom_hourly__ood__test_rolling__clean__K4_H64.npz"
    # fail-loud: a genuinely-missing FT cache -> FileNotFoundError. Use a hash no checkpoint produced
    # so this never collides with a real on-disk smoke/B1 cache (the B0 smoke DID write 18c93f86).
    missing = rfs.Stage("stage1_ft_early", "/scratch/ck", "ffffffff")
    try:
        rfs._load_fslot(missing, "boom_hourly", "test_rolling",
                        np.zeros((1, C), np.float32), np.zeros((1, H), np.float32))
        raise AssertionError("missing FT cache must fail loud")
    except FileNotFoundError as e:
        assert "run B1" in str(e)


# B2. End-to-end probe -> tunnel -> figures on synthetic features: 3 stages x 2 targets x 3 seeds.
def test_stageB_probe_tunnel_figures_synthetic():
    import tempfile
    _setup_stageB(tempfile.mkdtemp())
    stages = [rfs.Stage(l, None, None)
              for l in (rfs.STAGE0, "stage1_ft_early", "stage2_ft_late")]
    targets = ["boom_hourly", "monash_electricity_hourly"]      # 1 FT-ID + 1 FT-OOD
    n = 18
    quantiles = validate_quantiles(QUANTILE_SETS[rfs.QSET])
    sids = np.repeat(np.arange(n // 3), 3)                      # 6 clusters, 3 windows each
    for si, stage in enumerate(stages):
        for ti, tag in enumerate(targets):
            base = 10 * si + ti
            f_tr, f_va, f_te = _syn_feats(n, base), _syn_feats(n, base + 1), _syn_feats(n, base + 2)
            Ytr, Yva, Yte = _syn_traj(n, base), _syn_traj(n, base + 1), _syn_traj(n, base + 2)
            for seed in rfs.PROBE_SEEDS:
                rec, wl = rfs._fit_one(stage.label, tag, f_tr, Ytr, f_va, Yva, f_te, Yte,
                                       sids, seed, quantiles, "cpu")
                assert len(rec["test_loss_by_layer"]) == NPTS
                assert len(rec["val_loss_by_layer"]) == NPTS
                assert wl.shape == (NPTS, n)
                assert rec["probe_status"] == "probe-ID"
                assert (rec["pt_status"], rec["ft_status"]) == rfs.target_status(tag)

    tunnels = rfs.run_tunnels(stages)                           # BOOM (FT-ID) tunnel per stage
    assert set(tunnels) == {s.label for s in stages}
    for lbl, rec in tunnels.items():
        assert 0 <= rec["l_start"] <= NPTS - 1
        assert len(rec["mean_val_loss_by_layer"]) == NPTS
        assert rec["dataset"] == rfs.FT_ID_TAG                  # tunnel defined on BOOM only
        assert rec["ft_status"] == "FT-ID" and rec["probe_status"] == "probe-ID"
        assert "d_id_ci" in rec and rec["n_windows"] == n
        assert (rfs.TUNNEL_DIR / f"tunnel__{lbl}__q9.json").exists()
        assert (rfs.FIG_DIR / f"tunnel__{lbl}__q9.png").exists()

    rows = rfs.run_figures(stages, targets)
    assert len(rows) == len(stages) * len(targets)
    for r in rows:                                             # two-axis status on every row, no bare ID/OOD
        assert r["pt_status"] in ("PT-ID", "PT-OOD") and r["ft_status"] in ("FT-ID", "FT-OOD")
        assert r["probe_status"] == "probe-ID"
        assert r["l_start"] == tunnels[r["stage"]]["l_start"]  # FT-OOD cell uses the stage's BOOM lens
        assert np.isfinite(r["D_last_vs_lstart"])
    assert (rfs.TABLE_DIR / "stageB_layerwise__q9.csv").exists()
    for tag in targets:
        assert (rfs.FIG_DIR / f"layerwise__{tag}__q9.png").exists()


if __name__ == "__main__":
    tests = [
        test_build_ft_data_truncation_and_counts,
        test_build_ft_data_ftval_is_starts_minus_3,
        test_build_ft_data_skips_short_series,
        test_build_ft_data_fail_loud_no_samplable_windows,
        test_build_ft_data_pt_ood_source,
        test_ft_cache_prefix_disjoint_from_pretrained,
        test_extract_kout_injection_signature_and_routing,
        test_param_group_buckets,
        test_param_drift_detects_changed_groups,
        test_checkpoint_hash,
        test_ft_defaults_match_official,
        test_stageB_cache_path_and_fail_loud,
        test_stageB_probe_tunnel_figures_synthetic,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} FT-specialization Stage-A tests passed.")
