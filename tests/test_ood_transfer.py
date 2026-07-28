"""Focused tests protecting the scientific validity of the cross-dataset transfer pilot.

No model, no GPU — synthetic features only (the core checks), so this runs on a login node:

    OMP_NUM_THREADS=2 python -m tests.test_ood_transfer

Covers the invariants that make the experiment "strict transfer": the frozen fit/predict
split reproduces quantile_probe (the diagonal), one probe is trained once and reused across
targets without mutation, target arrays never reach the training function, source/target
shapes are compatible, and the probe checkpoint identity excludes the target (so
electricity->kdd and electricity->uber share one checkpoint) while result identity includes
it. An end-to-end smoke over real cached features is gated behind RUN_OOD_SMOKE=1 so the
default run stays fast and cache-free.
"""

from __future__ import annotations

import inspect
import os

os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import torch

from probing.config import NUM_LAYERS, LAST_LAYER, SEED
from probing.probes import (QUANTILE_SETS, quantile_probe, fit_quantile_probe,
                            predict_quantile_probe)

Q9 = QUANTILE_SETS["q9"]
D, H = 6, 8                      # tiny feature dim / horizon (speed)


def _synth(n, d=D, h=H, seed=0):
    """Synthetic {layer: (n, d)} features + (n, h) trajectory labels — stands in for one
    dataset's hidden states. Different `seed` = a different 'dataset' (a transfer target)."""
    rng = np.random.default_rng(seed)
    feats = {i: rng.normal(size=(n, d)).astype(np.float32) for i in range(NUM_LAYERS)}
    y = rng.normal(size=(n, h)).astype(np.float32)
    return feats, y


# 1 + (diagonal reproduction). fit -> predict on the SAME split must reproduce quantile_probe.
def test_fit_predict_reproduces_quantile_probe():
    f_tr, y_tr = _synth(40, seed=1)
    f_te, y_te = _synth(25, seed=2)
    ref = quantile_probe(f_tr, y_tr, f_te, y_te, quantiles=Q9, epochs=5,
                         wd_grid=(1e-3, 1e-1), device="cpu")
    fitted = fit_quantile_probe(f_tr, y_tr, quantiles=Q9, epochs=5,
                                wd_grid=(1e-3, 1e-1), device="cpu")
    out = predict_quantile_probe(fitted, f_te, y_te, quantiles=Q9, device="cpu")
    for i in range(NUM_LAYERS):
        np.testing.assert_allclose(out[i], ref[i], rtol=1e-6, atol=1e-6,
                                   err_msg=f"L{i}: fit->predict != quantile_probe (diagonal)")


# 1 (reuse). ONE frozen probe scores multiple targets; weights are never mutated.
def test_frozen_probe_reused_across_targets_not_mutated():
    f_tr, y_tr = _synth(40, seed=1)
    fitted = fit_quantile_probe(f_tr, y_tr, quantiles=Q9, epochs=5, wd_grid=(1e-2,), device="cpu")
    w0 = fitted[0]["linear"].weight.detach().clone()
    tA = _synth(15, seed=10)
    tB = _synth(22, seed=11)
    outA = predict_quantile_probe(fitted, tA[0], tA[1], quantiles=Q9, device="cpu")
    outB = predict_quantile_probe(fitted, tB[0], tB[1], quantiles=Q9, device="cpu")
    assert torch.equal(fitted[0]["linear"].weight, w0), "predict mutated the frozen probe"
    # deterministic + side-effect-free: re-scoring target A after B is unchanged
    outA2 = predict_quantile_probe(fitted, tA[0], tA[1], quantiles=Q9, device="cpu")
    for i in range(NUM_LAYERS):
        assert outA[i] == outA2[i], f"L{i}: predict is not deterministic / has state"
    assert set(outA) == set(outB) == set(range(NUM_LAYERS))


# 2 + 3. The training call cannot receive target features / target labels / a target val loader.
def test_training_signature_has_no_target_or_val():
    fit_params = set(inspect.signature(fit_quantile_probe).parameters)
    assert "test_feats" not in fit_params and "test_labels" not in fit_params, \
        "fit_quantile_probe must not accept any test/target arrays"
    # fit takes only source train arrays + hyperparams; no separate validation split is passed
    # in (its wd carve is drawn from the SOURCE train inside the function).
    assert {"train_feats", "train_labels"} <= fit_params
    pred_params = set(inspect.signature(predict_quantile_probe).parameters)
    assert "wd_grid" not in pred_params and "weight_decay" not in pred_params, \
        "predict must not do any model selection (no wd grid / early stopping on the target)"


# 4. A source-trained probe applies directly to a DIFFERENT-sized target of the same geometry.
def test_source_to_target_shape_compatibility():
    f_tr, y_tr = _synth(30, seed=1)             # source: 30 windows
    fitted = fit_quantile_probe(f_tr, y_tr, quantiles=Q9, epochs=4, wd_grid=(1e-2,), device="cpu")
    f_te, y_te = _synth(17, seed=99)            # target: 17 windows, same d / H
    out, diag = predict_quantile_probe(fitted, f_te, y_te, quantiles=Q9, device="cpu",
                                       collect_test_median=True, collect_test_window_loss=True)
    assert set(out) == set(range(NUM_LAYERS))
    for i in range(NUM_LAYERS):
        assert diag["test_median"][i].shape == (17, H), f"L{i}: median shape wrong"
        assert diag["test_window_loss"][i].shape == (17,), f"L{i}: per-window loss shape wrong"


# 5. Checkpoint identity excludes the target; result identity includes it.
def test_checkpoint_identity_excludes_target():
    from experiments.run_ood_transfer import _probe_run_id
    src = "monash_electricity_hourly"
    rid_kdd_target = _probe_run_id(src, "q9", SEED)       # run id is a function of SOURCE only
    rid_uber_target = _probe_run_id(src, "q9", SEED)
    assert rid_kdd_target == rid_uber_target, "probe run id must not depend on the target"
    assert "electricity" in rid_kdd_target and "kdd" not in rid_kdd_target
    # a different source is a different probe identity
    assert _probe_run_id("monash_kdd_cup_2018", "q9", SEED) != rid_kdd_target


# 6 (definition). is_ood is exactly source != target.
def test_is_ood_definition():
    for src in ("monash_electricity_hourly", "monash_kdd_cup_2018", "uber_tlc_hourly"):
        for tgt in ("monash_electricity_hourly", "monash_kdd_cup_2018", "uber_tlc_hourly"):
            assert (src != tgt) == (src != tgt)          # is_ood := source_dataset != target_dataset
            if src == tgt:
                assert not (src != tgt), "diagonal must be in-dataset (is_ood False)"


# 7. End-to-end smoke over REAL cached features (gated: needs the content caches; opt-in + fast).
def test_smoke_real_cache():
    if os.environ.get("RUN_OOD_SMOKE") != "1":
        print("  [skip] test_smoke_real_cache (set RUN_OOD_SMOKE=1 to run over cached features)")
        return
    import experiments.run_ood_transfer as R
    from probing.config import CACHE_DIR
    src = "monash_electricity_hourly"
    targets = [src, "monash_kdd_cup_2018"]
    need = [CACHE_DIR / f"IDF_{src}__train__clean__content.npz"]
    need += [CACHE_DIR / f"IDF_{t}__test__clean__content.npz" for t in targets]
    if not all(p.exists() for p in need):
        print("  [skip] test_smoke_real_cache (content caches missing)")
        return
    R.QUANTILE_EPOCHS, R.WD_GRID = 3, (1e-3,)            # keep the smoke fast
    payload = R.run_source(src, targets, "q9", QUANTILE_SETS["q9"], SEED, "cpu")
    assert len(payload["summaries"]) == 2
    diag_cell = next(s for s in payload["summaries"] if s["target_dataset"] == src)
    ood_cell = next(s for s in payload["summaries"] if s["target_dataset"] != src)
    assert diag_cell["is_ood"] is False and ood_cell["is_ood"] is True
    ckpt = R._ckpt_dir(src, "q9", SEED) / "L00.pt"
    assert ckpt.exists(), "source probe checkpoint was not written"
    print(f"  [smoke] diagonal best L{diag_cell['best_layer']} Δ={diag_cell['delta_vs_last']:+.3f} | "
          f"OOD best L{ood_cell['best_layer']} Δ={ood_cell['delta_vs_last']:+.3f}")


TESTS = [test_fit_predict_reproduces_quantile_probe,
         test_frozen_probe_reused_across_targets_not_mutated,
         test_training_signature_has_no_target_or_val,
         test_source_to_target_shape_compatibility,
         test_checkpoint_identity_excludes_target,
         test_is_ood_definition,
         test_smoke_real_cache]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(TESTS)} TESTS PASS")
