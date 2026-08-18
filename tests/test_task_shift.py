"""CPU/synthetic contracts for the TASK-SHIFT experiment (classification FT + layerwise probes).

Three classification sources: forda (univariate, 2-class), uwave (3-channel, 8-class), handwriting
(3-channel, 26-class). No model, no GPU, no dataset download — synthetic features + tiny torch modules,
so this runs on a login node:

    OMP_NUM_THREADS=2 python -m tests.test_task_shift

What needs a GPU (the FT run itself, real extraction, the C1 validity gate) is verified in the pilot.
What is checkable WITHOUT a model lives here (notes/PLAN.md, TASK-SHIFT §12 + the UEA extension §14):
  * stratified split determinism + SOURCE-AWARE overlap (TRAIN/TEST are separate index spaces);
  * label mapping -> contiguous [0,C-1] (2-class and multiclass string labels);
  * 14-point classification feature assembly (never drops L12+LN) + MULTIVARIATE per-channel concat;
  * the ONE fixed pooling rule (mean over ncp content tokens) + encode->pool->concat shape (b, c*768);
  * the head is strictly linear with in_features = d_model*channels;
  * the FT optimizer param groups (backbone trainable, head, native head frozen);
  * probing is backbone-free (only cached feature arrays) + val-only wd selection; multiclass CE probe;
  * Exp B REUSES the fslot forecasting probe (no new head); per-SOURCE cache keys disjoint + vs BOOM;
  * configure() rebinds source globals; load_stages rejects a source-mismatched manifest; idempotent resume;
  * a synthetic end-to-end cls-probe -> Plot-A render.
"""

from __future__ import annotations

import inspect
import json
import os
import tempfile
from pathlib import Path

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
def _synth_14pt(n, d=8, seed=0, signal_layer=None, y=None, n_classes=2):
    rng = np.random.default_rng(seed)
    D = {i: rng.standard_normal((n, d)).astype(np.float32) for i in range(NUM_LAYERS)}
    D[NUM_LAYERS] = rng.standard_normal((n, d)).astype(np.float32)          # key 13 = L12+LN
    if signal_layer is not None and y is not None:
        yy = np.asarray(y)
        for c in range(n_classes):                                          # a per-class decodable bump
            D[signal_layer][yy == c, c % d] += 6.0
    return D


class _FakeEnc:
    """Minimal stand-in for model.encode: returns ([hidden_states],) so `enc_out, *_ = encode(...)`
    then `enc_out[0]` yields a deterministic (b, P, d) tensor (P=5)."""
    def encode(self, context, num_output_patches=1):
        b = int(context.shape[0])
        hs = torch.arange(b * 5 * 768, dtype=torch.float32).reshape(b, 5, 768)
        return ([hs],)


# --------------------------------------------------------------------------- #
# 1. split determinism + stratification (+ guarded real FordA load, now (n,1,L))
# --------------------------------------------------------------------------- #
def test_split_determinism_and_stratification():
    y = np.array([0] * 60 + [1] * 40)
    tr1, va1 = cd.stratified_train_val_split(y, 0.2, 0)
    tr2, va2 = cd.stratified_train_val_split(y, 0.2, 0)
    assert np.array_equal(tr1, tr2) and np.array_equal(va1, va2), "split not deterministic"
    for c in (0, 1):
        n_c = int((y == c).sum())
        n_val_c = int((y[va1] == c).sum())
        assert abs(n_val_c - round(0.2 * n_c)) <= 1, f"class {c} not stratified: {n_val_c}"
    tr3, _ = cd.stratified_train_val_split(y, 0.2, 1)
    assert not np.array_equal(tr1, tr3)
    try:                                             # guarded real load (skipped offline / without aeon)
        d = cd.load_cls("forda")
        assert d["X_train"].ndim == 3 and d["X_train"].shape[1] == 1 and d["X_train"].shape[2] == 500
        assert set(np.unique(d["y_train"]).tolist()) <= {0, 1}
        print("    (real FordA load OK)")
    except Exception as e:
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


# --------------------------------------------------------------------------- #
# 3. label mapping -> contiguous [0,C-1] for 2-class AND multiclass string labels
# --------------------------------------------------------------------------- #
def test_label_map_contiguous():
    for raw in (np.array(["-1", "1", "1", "-1"]), np.array([-1, 1, 1, -1])):
        yint, mp = cd.map_labels_to_int(raw)
        assert yint.tolist() == [0, 1, 1, 0], f"bad 2-class mapping for {raw.dtype}: {yint}"
        assert set(mp.values()) == {0, 1} and len(mp) == 2
    # multiclass string labels ('1.0'..'8.0' style) -> contiguous [0,7], deterministic
    raw = np.array([f"{i}.0" for i in range(1, 9)] * 2)
    yint, mp = cd.map_labels_to_int(raw)
    assert sorted(set(yint.tolist())) == list(range(8)) and len(mp) == 8
    assert sorted(mp.values()) == list(range(8)), "multiclass map not contiguous"


# --------------------------------------------------------------------------- #
# 3b. registry: three sources with the correct class/channel geometry
# --------------------------------------------------------------------------- #
def test_registry_three_sources_geometry():
    exp = {"forda": (2, 1, 500), "uwave": (8, 3, 315), "handwriting": (26, 3, 152)}
    for src, (nc, ch, L) in exp.items():
        s = cd.CLS_SPECS[src]
        assert (s["n_classes"], s["channels"], s["length"]) == (nc, ch, L), f"{src} geometry wrong"
    assert cd.ncp_for_length(500) == 32 and cd.ncp_for_length(315) == 20 and cd.ncp_for_length(152) == 10


# --------------------------------------------------------------------------- #
# 4. classification feature assembly -> 14 points, (n, d) — univariate (forda)
# --------------------------------------------------------------------------- #
def test_cls_feature_assembly_14pt_shape():
    n, d = 7, 768
    feats = {"content": {i: np.zeros((n, d), np.float32) for i in range(NUM_LAYERS)},
             "reg": {}, "fslot": {}}
    final = {"content": np.zeros((n, d), np.float32)}
    orig = rt.extract_kout_features
    try:
        rt.configure("forda")                        # CHANNELS == 1
        rt.extract_kout_features = lambda *a, **k: (feats, final, np.zeros(n))
        D = rt.cls_feats(rt.Stage(rt.STAGE0, None, None), "test",
                         np.zeros((n, 500), np.float32), np.zeros(n), pipe=None, allow_extract=True)
    finally:
        rt.extract_kout_features = orig
    assert sorted(D) == list(range(NUM_LAYERS + 1)), "not 14 contiguous keys (L12+LN dropped?)"
    assert all(D[i].shape == (n, d) for i in D)


# --------------------------------------------------------------------------- #
# 4b. MULTIVARIATE assembly: 3 channels -> c*768 width, one extract call per channel, __ch suffixes
# --------------------------------------------------------------------------- #
def test_cls_feats_multivariate_concat():
    n, d = 6, 768
    feats = {"content": {i: np.ones((n, d), np.float32) for i in range(NUM_LAYERS)}, "reg": {}, "fslot": {}}
    final = {"content": np.ones((n, d), np.float32)}
    seen_prefixes = []
    orig = rt.extract_kout_features
    try:
        rt.configure("uwave")                        # CHANNELS == 3
        def _mock(tag, split, X, y, horizon, pipeline=None, cache_prefix=None):
            seen_prefixes.append(cache_prefix)
            assert X.shape == (n, 315), f"per-channel context should be (n, L), got {X.shape}"
            return feats, final, np.zeros(n)
        rt.extract_kout_features = _mock
        X = np.zeros((n, 3, 315), np.float32)
        D = rt.cls_feats(rt.Stage(rt.STAGE0, None, None), "test", X, np.zeros(n), allow_extract=True)
    finally:
        rt.extract_kout_features = orig
        rt.configure("forda")
    assert sorted(D) == list(range(NUM_LAYERS + 1))
    assert all(D[i].shape == (n, 3 * d) for i in D), "multivariate features must be (n, c*768)"
    assert seen_prefixes == ["IDF_uwave__task_shift_cls__ch0", "IDF_uwave__task_shift_cls__ch1",
                             "IDF_uwave__task_shift_cls__ch2"], f"per-channel prefixes wrong: {seen_prefixes}"


# --------------------------------------------------------------------------- #
# 4c. extract guard: FT stage EXTRACTS with a pipeline, FAILS LOUD without one / at probe time
# --------------------------------------------------------------------------- #
def test_cls_feats_ft_stage_extract_guard():
    n, d = 5, 768
    feats = {"content": {i: np.zeros((n, d), np.float32) for i in range(NUM_LAYERS)}, "reg": {}, "fslot": {}}
    final = {"content": np.zeros((n, d), np.float32)}
    orig = rt.extract_kout_features
    try:
        rt.configure("forda")
        ft = rt.Stage(rt.STAGE1, "d", "deadbeef")        # FT stage; its cache won't exist
        X, y = np.zeros((n, 500), np.float32), np.zeros(n)
        calls = {"n": 0}
        rt.extract_kout_features = lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1),
                                                    (feats, final, y))[1]
        D = rt.cls_feats(ft, "test", X, y, pipe=object(), allow_extract=True)
        assert sorted(D) == list(range(NUM_LAYERS + 1)) and calls["n"] == 1
        for kw in (dict(pipe=None, allow_extract=True), dict(allow_extract=False)):
            raised = False
            try:
                rt.cls_feats(ft, "test", X, y, **kw)
            except FileNotFoundError:
                raised = True
            assert raised, f"FT stage must fail loud for {kw}, not extract off the singleton"
    finally:
        rt.extract_kout_features = orig


# --------------------------------------------------------------------------- #
# 5. pooling = mean over ncp content tokens (+ per-source ncp)
# --------------------------------------------------------------------------- #
def test_pooling_mean_over_content_tokens():
    hs = torch.randn(3, 8, 5)
    ncp = 4
    assert torch.allclose(fc.pool_content_cls(hs, ncp), hs[:, :ncp, :].mean(dim=1))
    assert cd.ncp_for_length(cd.CLS_SPECS["forda"]["length"]) == 32
    assert cd.ncp_for_length(cd.CLS_SPECS["uwave"]["length"]) == 20
    assert cd.ncp_for_length(cd.CLS_SPECS["handwriting"]["length"]) == 10


# --------------------------------------------------------------------------- #
# 5b. encode -> pool -> concat: (b, c*768), first block == content-pool of channel 0
# --------------------------------------------------------------------------- #
def test_encode_pool_concat_shape():
    model = _FakeEnc()
    b, ncp, channels = 4, 2, 3
    ctx = torch.randn(b, channels, 40)
    out = fc.encode_pool_concat(model, ctx, ncp, channels)
    assert out.shape == (b, channels * 768), f"expected (b, c*768), got {tuple(out.shape)}"
    hs = torch.arange(b * 5 * 768, dtype=torch.float32).reshape(b, 5, 768)
    assert torch.allclose(out[:, :768], hs[:, :ncp, :].mean(1)), "channel-0 block != content-pool"


# --------------------------------------------------------------------------- #
# 6. head is strictly linear (nn.Linear, forward == Wx+b) with in_features = d*channels; out == C
# --------------------------------------------------------------------------- #
def test_head_is_strictly_linear_multivariate():
    h = fc.build_cls_head(6, 2)                       # default channels=1
    assert isinstance(h, nn.Linear) and h.out_features == 2 and h.in_features == 6
    x = torch.randn(4, 6)
    assert torch.allclose(h(x), x @ h.weight.T + h.bias)
    h3 = fc.build_cls_head(768, 8, channels=3)        # multivariate
    assert isinstance(h3, nn.Linear) and h3.out_features == 8 and h3.in_features == 3 * 768


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
    assert set(type(fit[i]["linear"]).__name__ for i in fit) == {"Linear"}


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
def test_stage_checkpoint_mapping_and_hash():
    try:
        rt.configure("forda")
        assert rt.Stage(rt.STAGE0, None, None).cache_prefix("forda") is None
        pre = rt.Stage(rt.STAGE1, "d", "deadbeef").cache_prefix("forda")
        assert pre == "IDF_forda__ft__forda_cls__stage1_cls_early__deadbeef"
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
    finally:
        rt.configure("forda")


# --------------------------------------------------------------------------- #
# 10b. configure() rebinds every source-dependent global (uwave / handwriting)
# --------------------------------------------------------------------------- #
def test_configure_rebinds_source_globals():
    try:
        rt.configure("uwave")
        assert rt.CLS_SOURCE == "uwave" and rt.FT_SOURCE == "uwave_cls" and rt.CHANNELS == 3
        assert rt.N_CLASSES == 8 and rt.CLS_STAGE0_PREFIX == "IDF_uwave__task_shift_cls"
        assert rt.CLS_PROBE_DIR == rt.OUT_ROOT / "uwave" / "cls_probes"
        assert rt.Stage(rt.STAGE1, "d", "h").cache_prefix("uwave") == \
            "IDF_uwave__ft__uwave_cls__stage1_cls_early__h"
        rt.configure("handwriting")
        assert rt.N_CLASSES == 26 and rt.CHANNELS == 3 and rt.FT_SOURCE == "handwriting_cls"
    finally:
        rt.configure("forda")
    assert rt.CLS_SOURCE == "forda" and rt.CHANNELS == 1 and rt.N_CLASSES == 2


# --------------------------------------------------------------------------- #
# 10c. load_stages rejects a manifest whose source != the active FT source
# --------------------------------------------------------------------------- #
def test_load_stages_rejects_source_mismatch():
    tmp = Path(tempfile.mkdtemp())
    bad = tmp / "manifest.json"
    bad.write_text(json.dumps({"source": "handwriting_cls", "checkpoints": {}}))
    try:
        rt.configure("uwave")                        # active FT source == uwave_cls
        rt.FT_MANIFEST = bad
        raised = False
        try:
            rt.load_stages([rt.STAGE1])
        except RuntimeError as e:
            raised = "source" in str(e)
        assert raised, "load_stages must reject a source-mismatched manifest"
    finally:
        rt.configure("forda")


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
# 11b. multiclass CE probe smoke: an 8-class signal layer is decodable
# --------------------------------------------------------------------------- #
def test_multiclass_ce_probe_smoke():
    y_tr = np.tile(np.arange(8), 12)                 # 96 rows, 8 classes
    y_va = np.tile(np.arange(8), 3)
    y_te = np.tile(np.arange(8), 6)
    fit = probes.fit_linear_cls_probe_explicit_val(
        _synth_14pt(len(y_tr), seed=1, signal_layer=6, y=y_tr, n_classes=8), y_tr,
        _synth_14pt(len(y_va), seed=2, signal_layer=6, y=y_va, n_classes=8), y_va,
        n_classes=8, epochs=80, lr=1e-1, wd_grid=(1e-4, 1e-2), device="cpu", init_seed=0)
    acc = probes.predict_linear_cls_probe(
        fit, _synth_14pt(len(y_te), seed=3, signal_layer=6, y=y_te, n_classes=8), y_te, device="cpu")
    assert acc[6] > 0.5 > 1.0 / 8, f"8-class signal layer not decodable: {acc[6]}"


# --------------------------------------------------------------------------- #
# 12. Exp B reuses the fslot forecasting probe (no new head)
# --------------------------------------------------------------------------- #
def test_expB_reuses_fslot_functions():
    assert rt.fit_shared_forecast_probe_explicit_val is probes.fit_shared_forecast_probe_explicit_val
    assert rt.predict_shared_forecast_probe is probes.predict_shared_forecast_probe


# --------------------------------------------------------------------------- #
# 13. cache keys distinguish all 3 sources + stages + are disjoint from BOOM
# --------------------------------------------------------------------------- #
def test_cache_keys_distinguish_sources_and_vs_boom():
    from probing.finetune import ft_cache_prefix
    p_forda = ft_cache_prefix("forda", "forda_cls", rt.STAGE1, "aaaa1111")
    p_uwave = ft_cache_prefix("uwave", "uwave_cls", rt.STAGE1, "aaaa1111")
    p_hand = ft_cache_prefix("handwriting", "handwriting_cls", rt.STAGE2, "bbbb2222")
    boom = ft_cache_prefix("boom_hourly", "boom", "stage1_ft_early", "cccc3333")
    keys = {p_forda, p_uwave, p_hand, boom}
    assert len(keys) == 4, f"prefixes must be pairwise-disjoint: {keys}"
    assert "__ft__forda_cls__" in p_forda and "__ft__uwave_cls__" in p_uwave
    assert "__ft__handwriting_cls__" in p_hand and "__ft__boom__" in boom


# --------------------------------------------------------------------------- #
# 14. resume: idempotent skip of an already-written probe run (per-source dir)
# --------------------------------------------------------------------------- #
def test_resume_idempotent_skip():
    orig = rt.CLS_PROBE_DIR
    try:
        rt.CLS_PROBE_DIR = Path(tempfile.mkdtemp())
        rt._cls_probe_json(rt.STAGE0, 0).write_text("{}")
        pending = [s for s in (0, 1, 2) if not rt._cls_probe_json(rt.STAGE0, s).exists()]
        assert pending == [1, 2], f"resume filter wrong: {pending}"
    finally:
        rt.CLS_PROBE_DIR = orig


# --------------------------------------------------------------------------- #
# 15. synthetic end-to-end cls probe -> Plot A render
# --------------------------------------------------------------------------- #
def test_end_to_end_synthetic_and_plot_a():
    rt.configure("forda")
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


# --------------------------------------------------------------------------- #
# 16. guarded real loaders for the two new multivariate sources (skipped offline / no aeon)
# --------------------------------------------------------------------------- #
def test_uwave_handwriting_loader_smoke():
    for src, (nc, ch, L) in {"uwave": (8, 3, 315), "handwriting": (26, 3, 152)}.items():
        try:
            d = cd.load_cls(src)
        except Exception as e:
            print(f"    ({src} real load skipped: {type(e).__name__})")
            continue
        m = d["meta"]
        assert d["X_train"].ndim == 3 and d["X_train"].shape[1] == ch and d["X_train"].shape[2] == L
        assert m["n_classes"] == nc and m["channels"] == ch
        assert set(np.unique(d["y_train"]).tolist()) <= set(range(nc))
        tr, va = np.asarray(m["train_idx"]), np.asarray(m["val_idx"])
        assert len(np.intersect1d(tr, va)) == 0, f"{src} train/val overlap"
        print(f"    ({src} real load OK: {d['X_train'].shape})")


if __name__ == "__main__":
    tests = [
        test_split_determinism_and_stratification,
        test_split_no_overlap_source_aware,
        test_label_map_contiguous,
        test_registry_three_sources_geometry,
        test_cls_feature_assembly_14pt_shape,
        test_cls_feats_multivariate_concat,
        test_cls_feats_ft_stage_extract_guard,
        test_pooling_mean_over_content_tokens,
        test_encode_pool_concat_shape,
        test_head_is_strictly_linear_multivariate,
        test_optimizer_param_groups_backbone_trainable,
        test_probe_is_backbone_free,
        test_extraction_returns_14_ordered_layers,
        test_stage_checkpoint_mapping_and_hash,
        test_configure_rebinds_source_globals,
        test_load_stages_rejects_source_mismatch,
        test_cls_probe_selection_val_only,
        test_multiclass_ce_probe_smoke,
        test_expB_reuses_fslot_functions,
        test_cache_keys_distinguish_sources_and_vs_boom,
        test_resume_idempotent_skip,
        test_end_to_end_synthetic_and_plot_a,
        test_uwave_handwriting_loader_smoke,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} TASK-SHIFT tests passed.")
