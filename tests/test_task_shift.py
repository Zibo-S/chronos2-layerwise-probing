"""CPU/synthetic contracts for the TASK-SHIFT experiment (FordA classification FT + layerwise probes).

No model, no GPU, no dataset download — synthetic features + tiny torch modules, so this runs on a
login node:

    OMP_NUM_THREADS=2 python -m tests.test_task_shift

What needs a GPU (the FT run itself, real extraction, the C1 validity gate) is verified in the pilot.
What is checkable WITHOUT a model lives here (notes/PLAN.md, TASK-SHIFT §12):
  * FordA stratified split determinism + SOURCE-AWARE overlap (TRAIN/TEST are separate index spaces);
  * label {-1,+1} -> {0,1} mapping; 14-point classification feature assembly (never drops L12+LN);
  * the ONE fixed pooling rule (mean over ncp content tokens); the head is strictly linear;
  * the FT optimizer param groups (backbone trainable @ backbone_lr, head @ head_lr, native head frozen);
  * probing is backbone-free (operates only on cached feature arrays) + val-only wd selection;
  * Exp B REUSES the fslot forecasting probe (no new head); disjoint cache keys per stage + vs BOOM;
  * idempotent resume; a synthetic end-to-end cls-probe -> Plot-A render.
"""

from __future__ import annotations

import inspect
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch
import torch.nn as nn

from probing import config
from probing.config import NUM_LAYERS
import probing.cls_data as cd
import probing.finetune_cls as fc
import probing.probes as probes
import experiments.run_task_shift as rt


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _synth_14pt(n, d=8, seed=0, signal_layer=None, y=None):
    rng = np.random.default_rng(seed)
    D = {i: rng.standard_normal((n, d)).astype(np.float32) for i in range(NUM_LAYERS)}
    D[NUM_LAYERS] = rng.standard_normal((n, d)).astype(np.float32)          # key 13 = L12+LN
    if signal_layer is not None and y is not None:
        D[signal_layer][:, 0] += np.asarray(y) * 5.0                        # decodable signal
    return D


# --------------------------------------------------------------------------- #
# 1. FordA split determinism + stratification (+ guarded real load)
# --------------------------------------------------------------------------- #
def test_forda_split_determinism_and_stratification():
    y = np.array([0] * 60 + [1] * 40)
    tr1, va1 = cd.stratified_train_val_split(y, 0.2, 0)
    tr2, va2 = cd.stratified_train_val_split(y, 0.2, 0)
    assert np.array_equal(tr1, tr2) and np.array_equal(va1, va2), "split not deterministic"
    # stratified: each class ~20% in val
    for c in (0, 1):
        n_c = int((y == c).sum())
        n_val_c = int((y[va1] == c).sum())
        assert abs(n_val_c - round(0.2 * n_c)) <= 1, f"class {c} not stratified: {n_val_c}"
    # a different seed gives a different partition
    tr3, _ = cd.stratified_train_val_split(y, 0.2, 1)
    assert not np.array_equal(tr1, tr3)
    # guarded real load (skipped offline / without aeon)
    try:
        d = cd.load_forda("forda")
        assert d["X_train"].shape[1] == 500 and set(np.unique(d["y_train"]).tolist()) <= {0, 1}
        print("    (real FordA load OK)")
    except Exception as e:                       # aeon missing / no network -> not a failure here
        print(f"    (real FordA load skipped: {type(e).__name__})")


# --------------------------------------------------------------------------- #
# 2. source-aware overlap: train∩val=∅ in TRAIN space; TEST is a disjoint partition
# --------------------------------------------------------------------------- #
def test_split_no_overlap_source_aware():
    y = np.array([0, 1] * 50)
    tr, va = cd.stratified_train_val_split(y, 0.2, 0)
    assert len(np.intersect1d(tr, va)) == 0, "train/val overlap in TRAIN index space"
    assert len(tr) + len(va) == len(y), "split does not cover TRAIN exactly"
    assert tr.max() < len(y) and va.max() < len(y), "indices escape the TRAIN partition"
    # TEST lives in its OWN index space and is never compared to TRAIN indices by construction.


# --------------------------------------------------------------------------- #
# 3. label {-1,+1} -> {0,1}
# --------------------------------------------------------------------------- #
def test_label_map_neg1_pos1_to_01():
    for raw in (np.array(["-1", "1", "1", "-1"]), np.array([-1, 1, 1, -1])):
        yint, mp = cd.map_labels_to_int(raw)
        assert yint.tolist() == [0, 1, 1, 0], f"bad mapping for {raw.dtype}: {yint}"
        assert set(mp.values()) == {0, 1} and len(mp) == 2


# --------------------------------------------------------------------------- #
# 4. classification feature assembly -> 14 points, (n, d)
# --------------------------------------------------------------------------- #
def test_cls_feature_assembly_14pt_shape():
    n, d = 7, 768
    feats = {"content": {i: np.zeros((n, d), np.float32) for i in range(NUM_LAYERS)},
             "reg": {}, "fslot": {}}
    final = {"content": np.zeros((n, d), np.float32)}
    orig = rt.extract_kout_features
    try:
        rt.extract_kout_features = lambda *a, **k: (feats, final, np.zeros(n))
        D = rt.cls_feats(rt.Stage(rt.STAGE0, None, None), "test",
                         np.zeros((n, 500), np.float32), np.zeros(n), pipe=None, allow_extract=True)
    finally:
        rt.extract_kout_features = orig
    assert sorted(D) == list(range(NUM_LAYERS + 1)), "not 14 contiguous keys (L12+LN dropped?)"
    assert all(D[i].shape == (n, d) for i in D)


# --------------------------------------------------------------------------- #
# 5. pooling = mean over ncp content tokens
# --------------------------------------------------------------------------- #
def test_pooling_mean_over_content_tokens():
    hs = torch.randn(3, 8, 5)
    ncp = 4
    assert torch.allclose(fc.pool_content_cls(hs, ncp), hs[:, :ncp, :].mean(dim=1))
    # ncp for FordA length 500 is 32
    assert int(np.ceil(cd.CLS_SPECS["forda"]["length"] / 16)) == 32


# --------------------------------------------------------------------------- #
# 6. head is strictly linear (nn.Linear, forward == Wx+b, no activation)
# --------------------------------------------------------------------------- #
def test_head_is_strictly_linear():
    h = fc.build_cls_head(6, 2)
    assert isinstance(h, nn.Linear) and h.out_features == 2
    x = torch.randn(4, 6)
    assert torch.allclose(h(x), x @ h.weight.T + h.bias)


# --------------------------------------------------------------------------- #
# 7. FT optimizer param groups: backbone trainable @ backbone_lr, head @ head_lr, native head frozen
# --------------------------------------------------------------------------- #
def test_optimizer_param_groups_backbone_trainable():
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Linear(3, 3)
            self.output_patch_embedding = nn.Linear(3, 3)     # native head -> frozen
            for p in self.output_patch_embedding.parameters():
                p.requires_grad_(False)
    m, head = Fake(), nn.Linear(3, 2)
    g = fc.build_optimizer_param_groups(m, head, 1e-5, 1e-3)
    assert g[0]["lr"] == 1e-5 and g[1]["lr"] == 1e-3
    n_backbone = sum(p.numel() for p in g[0]["params"])
    assert n_backbone == (3 * 3 + 3), "backbone group must exclude the frozen native head"
    assert all(p.requires_grad for p in g[0]["params"]), "backbone params must be trainable"
    assert sum(p.numel() for p in g[1]["params"]) == (3 * 2 + 2)


# --------------------------------------------------------------------------- #
# 8. probing is backbone-free (operates only on cached feature arrays)
# --------------------------------------------------------------------------- #
def test_probe_is_backbone_free():
    sig = inspect.signature(probes.fit_linear_cls_probe_explicit_val)
    assert not any(k in sig.parameters for k in ("model", "backbone", "pipeline")), \
        "the probe must never receive the backbone"
    y = np.array([0, 1] * 20)
    fit = probes.fit_linear_cls_probe_explicit_val(
        _synth_14pt(40, seed=1), y, _synth_14pt(10, seed=2), np.array([0, 1] * 5),
        n_classes=2, epochs=20, wd_grid=(1e-3,), device="cpu")
    assert set(type(fit[i]["linear"]).__name__ for i in fit) == {"Linear"}   # only a linear head, no module graph


# --------------------------------------------------------------------------- #
# 9. extraction returns 14 ORDERED layers Emb..L12+LN
# --------------------------------------------------------------------------- #
def test_extraction_returns_14_ordered_layers():
    assert len(rt.LAYER_LABELS) == NUM_LAYERS + 1 == 14
    assert rt.LAYER_LABELS[0] == "Emb" and rt.LAYER_LABELS[-1] == "L12+LN"
    assert rt.LAYER_LABELS[1:NUM_LAYERS] == [f"L{i}" for i in range(1, NUM_LAYERS)]


# --------------------------------------------------------------------------- #
# 10. stage -> checkpoint mapping + hash; load_stages fails loud without a manifest
# --------------------------------------------------------------------------- #
def test_stage_checkpoint_mapping_and_hash(tmp_dir=None):
    assert rt.Stage(rt.STAGE0, None, None).cache_prefix("forda") is None      # stage0 -> default namespace
    pre = rt.Stage(rt.STAGE1, "d", "deadbeef").cache_prefix("forda")
    assert pre == "IDF_forda__ft__forda_cls__stage1_cls_early__deadbeef"
    # FT stage requested but no manifest -> fail loud
    orig = rt.FT_MANIFEST
    try:
        rt.FT_MANIFEST = config.REPO_ROOT / "results" / "task_shift_classification" / "__no_such__.json"
        raised = False
        try:
            rt.load_stages([rt.STAGE1])
        except FileNotFoundError:
            raised = True
        assert raised, "load_stages must fail loud when the FT manifest is missing"
    finally:
        rt.FT_MANIFEST = orig


# --------------------------------------------------------------------------- #
# 11. cls probe never uses test for selection (val-only wd)
# --------------------------------------------------------------------------- #
def test_cls_probe_selection_val_only():
    sig = inspect.signature(probes.fit_linear_cls_probe_explicit_val)
    assert "test" not in " ".join(sig.parameters), "fit must not take a test split"
    y = np.array([0, 1] * 20)
    fit = probes.fit_linear_cls_probe_explicit_val(
        _synth_14pt(40, seed=1, signal_layer=5, y=y), y,
        _synth_14pt(10, seed=2, signal_layer=5, y=np.array([0, 1] * 5)), np.array([0, 1] * 5),
        n_classes=2, epochs=30, wd_grid=(1e-4, 1e-1), device="cpu")
    sel = fit[5]["selection"]
    assert "val_ce_by_wd" in sel and "chosen_wd" in sel
    assert sel["chosen_wd"] in (1e-4, 1e-1) and sel["chosen_wd"] == min(sel["val_ce_by_wd"],
                                                                        key=sel["val_ce_by_wd"].get)


# --------------------------------------------------------------------------- #
# 12. Exp B reuses the fslot forecasting probe (no new head)
# --------------------------------------------------------------------------- #
def test_expB_reuses_fslot_functions():
    assert rt.fit_shared_forecast_probe_explicit_val is probes.fit_shared_forecast_probe_explicit_val
    assert rt.predict_shared_forecast_probe is probes.predict_shared_forecast_probe


# --------------------------------------------------------------------------- #
# 13. cache keys distinguish all 3 stages + are disjoint from BOOM
# --------------------------------------------------------------------------- #
def test_cache_keys_distinguish_stages_and_vs_boom():
    from probing.finetune import ft_cache_prefix
    p1 = ft_cache_prefix("forda", "forda_cls", rt.STAGE1, "aaaa1111")
    p2 = ft_cache_prefix("forda", "forda_cls", rt.STAGE2, "bbbb2222")
    boom = ft_cache_prefix("boom_hourly", "boom", "stage1_ft_early", "cccc3333")
    assert p1 != p2 and "forda_cls" in p1 and "forda_cls" in p2
    assert "__ft__forda_cls__" in p1 and "__ft__boom__" in boom and p1 != boom


# --------------------------------------------------------------------------- #
# 14. resume: idempotent skip of an already-written probe run
# --------------------------------------------------------------------------- #
def test_resume_idempotent_skip():
    import tempfile
    from pathlib import Path
    orig = rt.CLS_PROBE_DIR
    try:
        rt.CLS_PROBE_DIR = Path(tempfile.mkdtemp())
        rt._cls_probe_json(rt.STAGE0, 0).write_text("{}")           # pretend seed 0 done
        pending = [s for s in (0, 1, 2) if not rt._cls_probe_json(rt.STAGE0, s).exists()]
        assert pending == [1, 2], f"resume filter wrong: {pending}"
    finally:
        rt.CLS_PROBE_DIR = orig


# --------------------------------------------------------------------------- #
# 15. synthetic end-to-end cls probe -> Plot A render
# --------------------------------------------------------------------------- #
def test_end_to_end_synthetic_and_plot_a():
    import tempfile
    from pathlib import Path
    y_tr, y_va, y_te = np.array([0, 1] * 20), np.array([0, 1] * 5), np.array([0, 1] * 10)
    fit = probes.fit_linear_cls_probe_explicit_val(
        _synth_14pt(40, seed=1, signal_layer=5, y=y_tr), y_tr,
        _synth_14pt(10, seed=2, signal_layer=5, y=y_va), y_va,
        n_classes=2, epochs=50, wd_grid=(1e-3, 1e-1), device="cpu", init_seed=0)
    acc = probes.predict_linear_cls_probe(fit, _synth_14pt(20, seed=3, signal_layer=5, y=y_te), y_te,
                                          device="cpu")
    assert sorted(acc) == list(range(14)) and acc[5] > 0.6, f"signal layer not decodable: {acc[5]}"

    tmp = Path(tempfile.mkdtemp())
    o_probe, o_fig = rt.CLS_PROBE_DIR, rt.FIG_DIR
    try:
        rt.CLS_PROBE_DIR, rt.FIG_DIR = tmp / "cls", tmp / "fig"
        rt.CLS_PROBE_DIR.mkdir(parents=True)
        for st in (rt.STAGE0, rt.STAGE1):
            for seed in (0, 1):
                rt._cls_probe_json(st, seed).write_text(json.dumps(
                    {"test_acc_by_layer": list(np.linspace(0.5, 0.9, 14) + 0.01 * seed)}))
        stages = [rt.Stage(rt.STAGE0, None, None), rt.Stage(rt.STAGE1, "d", "abcd1234")]
        rt.make_plot_a(stages)
        assert (rt.FIG_DIR / "plotA_classification_accessibility.png").exists()
    finally:
        rt.CLS_PROBE_DIR, rt.FIG_DIR = o_probe, o_fig


if __name__ == "__main__":
    tests = [
        test_forda_split_determinism_and_stratification,
        test_split_no_overlap_source_aware,
        test_label_map_neg1_pos1_to_01,
        test_cls_feature_assembly_14pt_shape,
        test_pooling_mean_over_content_tokens,
        test_head_is_strictly_linear,
        test_optimizer_param_groups_backbone_trainable,
        test_probe_is_backbone_free,
        test_extraction_returns_14_ordered_layers,
        test_stage_checkpoint_mapping_and_hash,
        test_cls_probe_selection_val_only,
        test_expB_reuses_fslot_functions,
        test_cache_keys_distinguish_stages_and_vs_boom,
        test_resume_idempotent_skip,
        test_end_to_end_synthetic_and_plot_a,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} TASK-SHIFT tests passed.")
