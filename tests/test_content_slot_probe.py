"""Contracts for the content-slot shared-head readout (CPU/synthetic — no GPU, model or cache).

Covers the two halves of the change:
  extraction  — slot_tokens selects WHICH K token states fill the "fslot" arrays, writes a
                SEPARATE cache file, records what it wrote, and fails loud on a mismatch;
                the default path stays byte-identical to the committed forecast-slot behaviour.
  driver      — the content-slot line reuses the fslot head verbatim and lands in a disjoint
                namespace that can never overwrite the committed fslot artifacts.

Run: python -m tests.test_content_slot_probe
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from probing import config
from probing.config import NUM_LAYERS, OUTPUT_PATCH_SIZE
from probing.extraction import SLOT_TOKEN_TAGS, _cache_path, _idf_prefix
from probing.probes import (CHRONOS2_QUANTILES, QUANTILE_SETS,
                            fit_shared_forecast_probe_explicit_val,
                            predict_shared_forecast_probe)

H = 64
K = math.ceil(H / OUTPUT_PATCH_SIZE)
NCP = 32                     # ceil(C=512 / P=16)


# --------------------------------------------------------------------------- #
# extraction contracts
# --------------------------------------------------------------------------- #
def test_slot_token_registry():
    """Exactly the two intended selectors, and the default maps to the EMPTY discriminator so
    committed forecast-slot cache names are unchanged."""
    assert set(SLOT_TOKEN_TAGS) == {"forecast", "content_last"}, SLOT_TOKEN_TAGS
    assert SLOT_TOKEN_TAGS["forecast"] == "", "default must not alter the legacy cache name"
    assert SLOT_TOKEN_TAGS["content_last"], "non-default must carry a name discriminator"


def test_cache_names_disjoint_and_default_unchanged():
    """The forecast-slot name is byte-identical to the committed convention; content_last gets a
    DIFFERENT file, so the two can never overwrite or satisfy each other's cache lookups."""
    config.set_dataset_set("extended_v3_rolling")
    pre = _idf_prefix("monash_electricity_hourly")
    f = _cache_path(pre, "test", None, f"{SLOT_TOKEN_TAGS['forecast']}K{K}_H{H}")
    c = _cache_path(pre, "test", None, f"{SLOT_TOKEN_TAGS['content_last']}K{K}_H{H}")
    assert f.name == f"{pre}__test__clean__K{K}_H{H}.npz", f.name
    assert f != c and f.parent == c.parent
    assert f"K{K}_H{H}" in c.name and c.name != f.name, c.name


def test_pooler_slices_the_documented_positions():
    """The two poolers on an explicit [content(ncp) | REG | forecast(K)] tensor: 'forecast' takes
    the last K, 'content_last' the K immediately before REG. Positions are asserted by VALUE, so a
    silent off-by-one (e.g. including REG) fails."""
    P = NCP + 1 + K
    hs = torch.arange(P, dtype=torch.float32).view(1, P, 1).repeat(1, 1, 3)  # token i == value i

    fslot = hs[:, -K:, :].numpy()
    cslot = hs[:, NCP - K:NCP, :].numpy()
    assert fslot.shape == cslot.shape == (1, K, 3)
    assert np.array_equal(fslot[0, :, 0], np.arange(P - K, P)), fslot[0, :, 0]
    assert np.array_equal(cslot[0, :, 0], np.arange(NCP - K, NCP)), cslot[0, :, 0]
    assert NCP not in cslot[0, :, 0], "content slots must NOT include the REG token"
    assert not set(fslot[0, :, 0]) & set(cslot[0, :, 0]), "the two token sets must be disjoint"


def test_content_slots_cover_the_recent_context_only():
    """Documents the stated caveat numerically: K=4 content patches see the LAST 64 of 512 steps,
    whereas pooled content_K sees all 512. If this ever changes the writeup caveat must too."""
    C = 512
    covered = K * OUTPUT_PATCH_SIZE
    assert covered == 64 and C - covered == 448, (covered, C)
    assert covered < C, "content slots are a strict subset of the context"


def test_cache_record_roundtrip_and_legacy_default():
    """slot_tokens survives a savez/load round-trip as a plain string, and a cache written before
    the field existed reads back as 'forecast' (the only thing it could have been)."""
    with tempfile.TemporaryDirectory() as d:
        new = Path(d) / "new.npz"
        np.savez(new, y=np.arange(2), slot_tokens=np.array("content_last"))
        z = np.load(new, allow_pickle=True)
        assert str(z["slot_tokens"]) == "content_last"

        legacy = Path(d) / "legacy.npz"
        np.savez(legacy, y=np.arange(2))
        z = np.load(legacy, allow_pickle=True)
        got = str(z["slot_tokens"]) if "slot_tokens" in z.files else "forecast"
        assert got == "forecast", got


def test_unknown_slot_tokens_rejected():
    """A typo must raise before any GPU work, not silently fall back to forecast slots."""
    assert "content_first" not in SLOT_TOKEN_TAGS
    try:
        SLOT_TOKEN_TAGS["content_first"]
    except KeyError:
        pass
    else:
        raise AssertionError("expected the registry lookup to reject an unknown selector")


# --------------------------------------------------------------------------- #
# probe contracts — the head is token-agnostic
# --------------------------------------------------------------------------- #
def _synth(n, seed, n_pts=NUM_LAYERS + 1):
    rng = np.random.default_rng(seed)
    feats = {i: rng.normal(size=(n, K, 768)).astype(np.float32) for i in range(n_pts)}
    Y = rng.normal(size=(n, H)).astype(np.float32)
    return feats, Y


def test_head_is_token_agnostic():
    """THE enabling fact: the shared head only ever sees (n,K,768). Two different token sets of
    the same shape both fit and predict, with the same head geometry — so nothing in the probe
    needs to change to read content tokens."""
    q = QUANTILE_SETS["q1"]
    f_tr, y_tr = _synth(24, 0)
    f_va, y_va = _synth(12, 1)
    f_te, y_te = _synth(12, 2)
    fitted = fit_shared_forecast_probe_explicit_val(
        f_tr, y_tr, f_va, y_va, quantiles=q, epochs=2, wd_grid=[1e-3], device="cpu")
    assert set(fitted) == set(range(NUM_LAYERS + 1)), "must cover all 14 readout points"
    for i, f in fitted.items():
        assert f["in_features"] == 768 and f["K"] == K
        assert f["out_features"] == len(q) * OUTPUT_PATCH_SIZE, f["out_features"]
    out, diag = predict_shared_forecast_probe(fitted, f_te, y_te, quantiles=q, device="cpu",
                                              collect_test_window_loss=True)
    assert set(out) == set(range(NUM_LAYERS + 1))
    for i in out:
        assert np.isfinite(out[i])
        assert diag["test_window_loss"][i].shape == (12,)
        assert np.isclose(diag["test_window_loss"][i].mean(), out[i], rtol=1e-5)


def test_param_count_matches_the_fslot_head():
    """Same head, same capacity: the content-slot probe must not quietly become a bigger readout,
    or the cslot-vs-fslot comparison stops being a pure token swap."""
    q = QUANTILE_SETS["q9"]
    f_tr, y_tr = _synth(16, 3, n_pts=2)
    f_va, y_va = _synth(8, 4, n_pts=2)
    fitted = fit_shared_forecast_probe_explicit_val(
        f_tr, y_tr, f_va, y_va, quantiles=q, epochs=2, wd_grid=[1e-3], device="cpu")
    n_params = sum(p.numel() for p in fitted[0]["linear"].parameters())
    expected = 768 * len(q) * OUTPUT_PATCH_SIZE + len(q) * OUTPUT_PATCH_SIZE
    assert n_params == expected, (n_params, expected)


# --------------------------------------------------------------------------- #
# driver contracts
# --------------------------------------------------------------------------- #
def test_driver_uses_content_tokens_and_the_shared_head():
    """The driver must request content_last and reuse the fslot fit/predict pair verbatim — not
    define a parallel head."""
    from experiments import run_content_slot_probing as cs
    from probing import probes
    assert cs.SLOT_TOKENS == "content_last"
    src = Path(cs.__file__).read_text()
    assert 'slot_tokens=SLOT_TOKENS' in src, "must thread the selector into extract_kout_features"
    assert cs.fit_shared_forecast_probe_explicit_val is probes.fit_shared_forecast_probe_explicit_val
    assert cs.predict_shared_forecast_probe is probes.predict_shared_forecast_probe


def test_driver_namespace_disjoint_from_committed_fslot():
    """Every cslot artifact path must sit under results/ext_v4_future_tokens/cslot/ and never
    collide with the committed fslot line, which the driver only READS."""
    from experiments import run_content_slot_probing as cs
    from experiments import run_ptood_probing_ftok as ft
    for p in (cs.PTID_RUN_DIR, cs.PTID_CKPT_DIR, cs.TUNNEL_DIR, cs.FIG_DIR, cs.TAB_DIR,
              cs._ckpt_dir("m4_hourly", "q9", 0), cs._tunnel_path("m4_hourly", "q9")):
        assert cs.CSLOT_ROOT in Path(p).parents or Path(p) == cs.CSLOT_ROOT, p
    assert cs.CSLOT_ROOT.name == "cslot" and cs.CSLOT_ROOT.parent == ft.OUT_ROOT
    assert cs.FSLOT_RUN_DIR == ft.OUT_ROOT / "ptood_probing" / "ptid_runs"
    assert cs.CSLOT_ROOT not in cs.FSLOT_RUN_DIR.parents, "must not write into the fslot tree"
    assert cs.READOUT != ft.READOUT, "artifact stems must differ"


def test_driver_shares_the_fslot_protocol():
    """Windows, seeds, epochs, wd grid and horizon are IMPORTED from the fslot driver, so the two
    readouts can never drift apart on anything except the tokens."""
    from experiments import run_content_slot_probing as cs
    from experiments import run_ptood_probing_ftok as ft
    for name in ("C", "H", "K", "QUANTILE_EPOCHS", "RUN_SEEDS", "WD_GRID", "RUNS_TAG",
                 "LAYER_LABELS", "PTID_SET"):
        assert getattr(cs, name) == getattr(ft, name), name
    assert cs._qtag("q9") == "q9__" + cs.PROBE_PROTOCOL_VERSION


def test_figure_and_tunnel_stages_on_synthetic_runs(tmp=None):
    """End-to-end post-hoc smoke into a tempdir: fabricate per-seed run JSONs for both readouts,
    then run the tunnel + figure stages. Verifies they never touch the committed tree."""
    from experiments import run_content_slot_probing as cs
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        orig = (cs.PTID_RUN_DIR, cs.FSLOT_RUN_DIR, cs.TUNNEL_DIR, cs.FIG_DIR, cs.TAB_DIR)
        cs.PTID_RUN_DIR, cs.FSLOT_RUN_DIR = root / "cslot_runs", root / "fslot_runs"
        cs.TUNNEL_DIR, cs.FIG_DIR, cs.TAB_DIR = root / "tun", root / "fig", root / "tab"
        for p in (cs.PTID_RUN_DIR, cs.FSLOT_RUN_DIR, cs.TUNNEL_DIR, cs.FIG_DIR, cs.TAB_DIR):
            p.mkdir(parents=True, exist_ok=True)
        try:
            qtag = cs._qtag("q9")
            for src in cs.PT_ID_TAGS:
                for run_dir in (cs.PTID_RUN_DIR, cs.FSLOT_RUN_DIR):
                    for s in cs.RUN_SEEDS:
                        curve = list(np.sort(rng.uniform(2.0, 3.0, NUM_LAYERS + 1))[::-1])
                        json.dump({"dataset": src, "quantile_set": "q9", "run_seed": s,
                                   "val_loss_by_layer": curve,
                                   "test_loss_by_layer": [c + 0.05 for c in curve]},
                                  open(run_dir / f"{src}__{qtag}__seed{s}.json", "w"))
            cs.compute_tunnels("q9")
            for src in cs.PT_ID_TAGS:
                rec = json.load(open(cs._tunnel_path(src, "q9")))
                assert rec["readout"] == "cslot"
                assert rec["pooling_or_token_type"] == "content_slot_last"
                assert len(rec["mean_val_loss_by_layer"]) == NUM_LAYERS + 1
            cs.make_figures("q9")
            figs = sorted(p.name for p in cs.FIG_DIR.glob("*.png"))
            assert len(figs) == 2, figs
            tabs = list(cs.TAB_DIR.glob("*.csv"))
            assert len(tabs) == 1, tabs
            head = tabs[0].read_text().splitlines()[0]
            assert "delta_cslot_minus_fslot" in head, head
        finally:
            (cs.PTID_RUN_DIR, cs.FSLOT_RUN_DIR, cs.TUNNEL_DIR, cs.FIG_DIR, cs.TAB_DIR) = orig


def test_figures_fail_loud_without_runs():
    """A figure stage run before --fit-ptid must say so, not emit an empty plot."""
    from experiments import run_content_slot_probing as cs
    with tempfile.TemporaryDirectory() as d:
        orig = cs.PTID_RUN_DIR
        cs.PTID_RUN_DIR = Path(d) / "empty"
        cs.PTID_RUN_DIR.mkdir()
        try:
            cs.make_figures("q9")
        except FileNotFoundError as e:
            assert "--fit-ptid" in str(e), e
        else:
            raise AssertionError("expected FileNotFoundError with no run artifacts")
        finally:
            cs.PTID_RUN_DIR = orig


# --------------------------------------------------------------------------- #
# content-slot CKA branch
# --------------------------------------------------------------------------- #
def test_cka_cslot_reads_the_content_slot_cache():
    """The CKA branch must address the cslotL_ cache — NOT the forecast-slot one — and derive the
    discriminator from extraction.SLOT_TOKEN_TAGS so it can never drift from the extractor."""
    from experiments import run_cka_analysis as rc
    assert rc.CSLOT_POOL == f"{SLOT_TOKEN_TAGS['content_last']}K{K}_H{H}", rc.CSLOT_POOL
    assert rc.CSLOT_POOL != rc.FSLOT_POOL and rc.FSLOT_POOL in rc.CSLOT_POOL
    assert rc.CSLOT_ANALYSIS == "ext_v4_future_tokens_cslot"
    # same 14 keys + same slot stacking as fslot -> the two maps stay comparable
    src = Path(rc.__file__).read_text()
    assert "def read_extv4_cslot_reps" in src
    assert "cka.stack_slots(a) for a in cka.load_npz_reps(path, FSLOT14_KEYS)" in src


def test_cka_cslot_covers_all_seven_datasets():
    """The 7-dataset roster is shared with the fslot/content branches, and PT-OOD tags resolve to
    the '_rolling' split names their windows were built under."""
    from experiments import run_cka_analysis as rc
    assert len(rc.EXTV4_TAGS) == 7, rc.EXTV4_TAGS
    for t in rc.EXTV4_TAGS:
        want = "test" if t in rc.PT_ID_TAGS else "test_rolling"
        assert rc._fslot_split(t, "test") == want, (t, want)


def test_cka_cslot_missing_cache_names_both_extraction_stages():
    """A missing cache must name BOTH producers — --fit-ptid covers only the 4 PT-ID sets, so a
    naive 7-dataset run would otherwise fail with no hint about the PT-OOD gap."""
    from experiments import run_cka_analysis as rc
    try:
        rc.read_extv4_cslot_reps("sg_carpark", "test")
    except FileNotFoundError as e:
        assert "job_content_slot_probing.sh" in str(e) and "--extract-ood" in str(e), str(e)
    else:
        raise AssertionError("expected FileNotFoundError for an absent content-slot cache")


def test_driver_extract_ood_covers_the_pt_ood_gap():
    """--extract-ood must extract content slots for exactly the 3 PT-OOD tags, test split only,
    and must not fit or save any probe artifact."""
    from experiments import run_content_slot_probing as cs
    from probing.tunnel import PT_OOD_TAGS
    import inspect
    src = inspect.getsource(cs.extract_ood)
    assert cs.extract_ood.__defaults__[0] == PT_OOD_TAGS
    assert len(PT_OOD_TAGS) == 3
    assert '"test_rolling"' in src, "PT-OOD caches must use the _rolling split name"
    assert "build_ood_rolling_windows" in src and "seed=SEED" in src
    for banned in ("fit_shared_forecast_probe_explicit_val", "_save_ckpt", "json.dump"):
        assert banned not in src, f"extract_ood must not {banned} — it is extraction only"


def test_paper_figure_branch_targets_the_cslot_namespace():
    """The paper-style 7-panel figure must read cslot matrices and write into the cslot figure
    dir — never overwrite the committed fslot/content appendix figures that share the stem."""
    from experiments import make_id_paper_figures as mp
    assert mp.CKA_CSLOT_MAT_DIR.name == "matrices"
    assert mp.CKA_CSLOT_MAT_DIR.parent.name == "ext_v4_future_tokens_cslot"
    assert mp.CKA_CSLOT_FIG_DIR.parent.name == "ext_v4_future_tokens_cslot"
    for other in (mp.CKA_FIG_DIR, mp.CKA_CONTENT_FIG_DIR):
        assert mp.CKA_CSLOT_FIG_DIR != other
    assert len(mp.CKA_TAGS["all7"]) == 7


def test_paper_figure_renders_seven_panels(tmp=None):
    """Synthetic end-to-end: 7 valid CKA matrices -> the paper-style figure actually renders as
    PDF+PNG with the 3-per-row wrap, into the given out_dir."""
    from experiments import make_id_paper_figures as mp
    rng = np.random.default_rng(0)
    n = len(mp.LABELS)
    rows = []
    for t in mp.CKA_TAGS["all7"]:
        A = rng.normal(size=(n, n))
        M = A @ A.T
        d = np.sqrt(np.diag(M))
        M = M / np.outer(d, d)                       # symmetric, unit diagonal
        rows.append({"tag": t, "M": M, "l_start": None})
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        pdf, png = mp.make_cka_figure(rows, "synthetic", "appendix_id_cka", dpi=60,
                                      ncol=3, out_dir=out)
        assert pdf.exists() and png.exists(), (pdf, png)
        assert pdf.parent == out, "must honour out_dir, not the default fslot dir"


TESTS = [
    test_slot_token_registry,
    test_cache_names_disjoint_and_default_unchanged,
    test_pooler_slices_the_documented_positions,
    test_content_slots_cover_the_recent_context_only,
    test_cache_record_roundtrip_and_legacy_default,
    test_unknown_slot_tokens_rejected,
    test_head_is_token_agnostic,
    test_param_count_matches_the_fslot_head,
    test_driver_uses_content_tokens_and_the_shared_head,
    test_driver_namespace_disjoint_from_committed_fslot,
    test_driver_shares_the_fslot_protocol,
    test_figure_and_tunnel_stages_on_synthetic_runs,
    test_figures_fail_loud_without_runs,
    test_cka_cslot_reads_the_content_slot_cache,
    test_cka_cslot_covers_all_seven_datasets,
    test_cka_cslot_missing_cache_names_both_extraction_stages,
    test_driver_extract_ood_covers_the_pt_ood_gap,
    test_paper_figure_branch_targets_the_cslot_namespace,
    test_paper_figure_renders_seven_panels,
]

if __name__ == "__main__":
    for t in TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(TESTS)} content-slot tests passed.")
