"""CONTRACT SUITE for Phase 2e (non-contiguous block selection, Chronos-2) -- model-free.

No checkpoint, no dataset, no feature cache, no GPU. The real-architecture contracts run on a TINY
random-weight Chronos-2 built from the installed package's own classes with all 12 blocks (d = 32),
so every mask indexes the real block list; the H3-equivalence contracts reuse H3's own mock cell.

    python -m tests.test_phase2e_noncontig        # login node OK: ~1 min, 2 threads, CPU only

Numbered so a failure names the requirement it broke (spec section 7, items 1-10, plus 11-14):
     1  every mask retains exactly k blocks; the table is pinned literally
     2  retained blocks execute in their original order; skipped blocks are bypassed (identity)
     3  all-12 execution == the unmodified model; the block list is always restored
     4  prefix masks == the Phase-1 extraction states and the H4 physical truncation
     5  prefix reuse: our scorer reproduces the H3 per-window arrays on H3's own inputs
     6  no backbone weight is modified (extraction, physical forecasts, fits)
     7  the H3 objective and selection: fit_noa == the H3 NOA fit, bitwise; drift refused
     8  train/val never see the test split; no ground-truth future reaches the fit
     9  output paths are isolated from Phase 1 / H3 / 2c / 2d
    10  atomic saves + resume (cells, state cache, latency repeats)
    11  physical (public API) == offline at every non-prefix mask, with an adapter
    12  paired statistics: identity -> zero difference; a real improvement is detected
    13  tables / figures / report regenerate from artifacts only; invalid_v3 never averaged
    14  latency configuration: blocks really removed and released, adapter spliced
"""

from __future__ import annotations

import copy
import gc
import importlib
import inspect
import json
import os
import shutil
import sys
import tempfile
import weakref
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase2_truncate as T                                         # noqa: E402
from probing import phase2e_noncontig as P                                        # noqa: E402
from probing.phase2 import Phase2CellStore, clean_staging                         # noqa: E402
from probing.phase2_align import ResidualAdapter                                  # noqa: E402

torch.set_num_threads(2)
D, C, H = 32, 512, 64
CHRONOS_Q21 = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
               0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


def _chronos_available() -> bool:
    try:
        import chronos  # noqa: F401
        return True
    except Exception:
        return False


def tiny_chronos2_12(seed=0):
    """The H4 tiny Chronos-2 (tests.test_phase2_h4.tiny_chronos2) with ALL 12 blocks."""
    from chronos import Chronos2Pipeline
    from chronos.chronos2 import Chronos2Model
    from chronos.chronos2.config import Chronos2CoreConfig
    cfg = Chronos2CoreConfig(d_model=D, d_kv=8, d_ff=64, num_layers=P.N_BLOCKS, num_heads=4,
                             dropout_rate=0.0, reg_token_id=1, attn_implementation="eager",
                             chronos_config={"context_length": 8192, "input_patch_size": 16,
                                             "input_patch_stride": 16, "output_patch_size": 16,
                                             "max_output_patches": 64, "quantiles": CHRONOS_Q21,
                                             "use_reg_token": True, "use_arcsinh": True,
                                             "time_encoding_scale": 8192})
    torch.manual_seed(seed)
    m = Chronos2Model(cfg)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return Chronos2Pipeline(m)


def contexts(n=6, seed=0):
    rng = np.random.default_rng(seed)
    mu = rng.uniform(-50, 50, size=(n, 1))
    sig = np.exp(rng.uniform(np.log(0.1), np.log(10.0), size=(n, 1)))
    return (mu + sig * np.cumsum(rng.normal(size=(n, C)), axis=1)).astype(np.float32)


def _all_masks():
    return [e["mask"] for e in P.plan_masks()] + list(P.SKIP_MASKS.values()) + [P.FULL_MASK]


def _skip(name):
    print(f"{name:<4} SKIPPED: the chronos package is not importable here")


# =========================================================================== #
# 1  the mask table
# =========================================================================== #
def test_01_mask_table_is_frozen_and_counts_are_exact():
    pinned = {
        ("prefix", 4): (1, 2, 3, 4), ("prefix", 6): (1, 2, 3, 4, 5, 6),
        ("prefix", 9): (1, 2, 3, 4, 5, 6, 7, 8, 9),
        ("uniform", 4): (1, 4, 8, 12), ("uniform", 6): (1, 3, 5, 8, 10, 12),
        ("uniform", 9): (1, 2, 4, 5, 7, 8, 10, 11, 12),
        ("early_late", 4): (1, 2, 11, 12), ("early_late", 6): (1, 2, 3, 10, 11, 12),
        ("early_late", 9): (1, 2, 3, 4, 5, 9, 10, 11, 12),
        ("middle", 4): (4, 5, 6, 7), ("middle", 6): (4, 5, 6, 7, 8, 9),
        ("middle", 9): (2, 3, 4, 5, 6, 7, 8, 9, 10)}
    assert P.MASKS == pinned, "the frozen mask table changed"
    plan = P.plan_masks()
    assert len(plan) == 12 and len({e["label"] for e in plan}) == 12
    for e in plan:
        assert len(e["mask"]) == e["budget"], e
        assert list(e["mask"]) == sorted(set(e["mask"])) and 1 <= e["mask"][0] \
            and e["mask"][-1] <= 12
        assert e["is_prefix"] == (e["strategy"] == "prefix"), e
        assert set(e["mask"]) | set(e["skipped"]) == set(range(1, 13))
        if e["strategy"] == "middle":
            assert e["contiguous"] and not e["is_prefix"], "middle is a contiguous CONTROL"
        if e["strategy"] in ("uniform", "early_late"):
            assert not e["contiguous"], e
    assert len(P.SKIP_MASKS) == 12
    for i, m in P.SKIP_MASKS.items():
        assert len(m) == 11 and i not in m
    assert P.mask_string((1, 2, 3, 4, 5, 9, 10, 11, 12)) == "1-5,9-12"
    assert P.mask_string((1, 4, 8, 12)) == "1,4,8,12"
    for bad in ((), (0, 1), (1, 13), (3, 2), (2, 2, 3)):
        try:
            P.validate_mask(bad)
            raise AssertionError(f"accepted {bad}")
        except ValueError:
            pass
    assert P.mask_table_digest() == P.mask_table_digest()
    print(" 1   12 frozen masks, exactly k blocks each, original order; 12 skip masks       OK")


# =========================================================================== #
# 2  order + bypass
# =========================================================================== #
def test_02_retained_blocks_run_in_order_and_skipped_blocks_are_identity():
    if not _chronos_available():
        return _skip(" 2")
    pipe = tiny_chronos2_12()
    core = pipe.model
    X = contexts()
    for mask in _all_masks():
        trace = P.block_call_trace(pipe, X, mask)
        assert trace == list(mask), (mask, trace)
    # the skipped blocks are the IDENTITY on the residual stream: the mask's state equals the
    # retained blocks applied by hand, in order, to the embedding, with the encoder's own kwargs
    cap = {}

    def grab(_m, args, kwargs):
        cap.setdefault("h0", args[0].detach().clone())
        cap.setdefault("kw", dict(kwargs))
    hk = core.encoder.block[0].register_forward_pre_hook(grab, with_kwargs=True)
    with torch.no_grad():
        core.encode(context=torch.as_tensor(X), num_output_patches=P.K_SLOTS)
    hk.remove()
    for mask in ((1, 4, 8, 12), (1, 2, 11, 12), (4, 5, 6, 7)):
        h = cap["h0"]
        with torch.no_grad():
            for b in mask:
                h = core.encoder.block[b - 1](h, **cap["kw"])[0]
        got = P.extract_mask_states(pipe, X, mask)
        assert np.array_equal(got, h[:, -P.K_SLOTS:, :].numpy()), mask
    print(" 2   execution trace == mask for 25 masks; skipped blocks are exactly the identity OK")


# =========================================================================== #
# 3  all-12 == unmodified, restoration
# =========================================================================== #
def test_03_all_blocks_reproduce_the_model_and_the_block_list_is_restored():
    if not _chronos_available():
        return _skip(" 3")
    pipe = tiny_chronos2_12()
    core = pipe.model
    X = contexts()
    blocks0, ids0, norm0 = core.encoder.block, [id(b) for b in core.encoder.block], \
        core.encoder.final_layer_norm
    ref = np.ascontiguousarray(T.public_forecast(pipe, "chronos2", X, batch_size=4)
                               .transpose(0, 2, 1))
    got = P.physical_forecast(pipe, X, P.FULL_MASK, batch_size=4)
    assert np.array_equal(got, ref), "all-12 mask != the unmodified public forecast"
    full_states = T.readout_states(pipe, "chronos2", X, full_depth=12)
    assert np.array_equal(P.extract_mask_states(pipe, X, P.FULL_MASK), full_states)
    assert core.encoder.block is blocks0 and [id(b) for b in core.encoder.block] == ids0
    assert core.encoder.final_layer_norm is norm0
    try:                                                  # restoration after an exception
        with P.skipped_blocks(pipe, (1, 4, 8, 12)), P.spliced_adapter(pipe, ResidualAdapter(D)):
            assert len(core.encoder.block) == 4
            raise KeyError("boom")
    except KeyError:
        pass
    assert core.encoder.block is blocks0 and core.encoder.final_layer_norm is norm0
    try:
        with P.skipped_blocks(pipe, (1, 2)):
            with P.skipped_blocks(pipe, (1,)):
                pass
        raise AssertionError("nested skipped_blocks accepted")
    except RuntimeError:
        pass
    assert core.encoder.block is blocks0
    print(" 3   all-12 mask == unmodified model (bitwise); block list/norm always restored  OK")


# =========================================================================== #
# 4  prefix == Phase-1 extraction and H4 truncation
# =========================================================================== #
def test_04_prefix_masks_reproduce_phase1_extraction_and_h4_truncation():
    if not _chronos_available():
        return _skip(" 4")
    from probing import extraction
    pipe = tiny_chronos2_12()
    X = contexts(n=10)
    tmp = Path(tempfile.mkdtemp())
    saved = extraction.CACHE_DIR
    extraction.CACHE_DIR = tmp                       # the Phase-1 cache is never touched
    try:
        feats, final, _ = extraction.extract_kout_features(
            "tiny", "test", X, np.zeros((len(X), H), np.float32), H, batch_size=4,
            pipeline=pipe, cache_prefix="IDF_tiny_phase2e_contract")
    finally:
        extraction.CACHE_DIR = saved
        shutil.rmtree(tmp, ignore_errors=True)
    for k in (1, 4, 6, 9, 11, 12):
        ours = P.extract_mask_states(pipe, X, tuple(range(1, k + 1)), batch_size=4)
        assert np.array_equal(ours, feats["fslot"][k]), f"prefix {k} != Phase-1 fslot_L{k}"
    for k in (4, 6, 9):
        h4 = copy.deepcopy(pipe)
        T.truncate(h4, "chronos2", k)
        ref = T.public_forecast(h4, "chronos2", X, batch_size=4).transpose(0, 2, 1)
        got = P.physical_forecast(pipe, X, tuple(range(1, k + 1)), batch_size=4)
        assert np.array_equal(got, ref), f"prefix {k} != H4 truncate({k})"
    print(" 4   prefix k == Phase-1 extract_kout_features fslot_L<k> and H4 truncate(k), bitwise OK")


# =========================================================================== #
# 5 + 7  H3 equivalence on H3's own mock cell
# =========================================================================== #
def _h3_mock_run(tmp: Path, poison_test=False):
    from probing.phase2_h3 import run_h3_cell
    from tests.test_phase2_h3 import FAST, _mock_cell
    pw, data, arrays, ent = _mock_cell(poison_test=poison_test)
    if poison_test:
        arrays["native_loss_window"] = None
        arrays["native_mase_window"] = None
    res = run_h3_cell(pw, data, arrays, entrances=ent, families=["hard", "noa"], depths=[6, 9],
                      device="cpu", fit_cfg=FAST, adapter_dir=tmp, log=lambda *a: None)
    return pw, data, res, FAST


def test_05_prefix_reuse_scorer_reproduces_h3_per_window_arrays():
    from probing.phase2_align import load_adapter
    tmp = Path(tempfile.mkdtemp())
    try:
        pw, data, res, _ = _h3_mock_run(tmp)
        te = data.splits["test"]
        bi = res["bootstrap_inputs"]
        for l in (6, 9):
            h = P.score_states(pw, te, te.feats[l], data.tag)
            ad, _ = load_adapter(tmp / data.model / data.tag / f"noa__L{l}.npz")
            n = P.score_states(pw, te, te.feats[l], data.tag, adapter=ad)
            for m, key in (("mase", "test_mase"), ("loss", "test_loss"), ("mae", "test_mae"),
                           ("wql_num", "test_wql_num")):
                assert np.array_equal(h[m], bi[f"hard__{key}"][l]), (l, m, "hard")
                assert np.array_equal(n[m], bi[f"noa__{key}"][l]), (l, m, "noa")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print(" 5   our scorer on H3's inputs == the H3 per-window arrays (hard + NOA), bitwise  OK")


def test_07_fit_is_the_h3_noa_fit_and_protocol_drift_is_refused():
    from probing import phase2
    from probing.phase2_align import adapter_arrays_sha256
    from probing.phase2_h3 import DEFAULT_FIT_CFG
    tmp = Path(tempfile.mkdtemp())
    try:
        pw, data, res, FAST = _h3_mock_run(tmp)
        L = data.reference_index
        tr, va = data.splits["train"], data.splits["val"]
        t_tr, _ = pw.run(tr.feats[L])
        t_va, _ = pw.run(va.feats[L])
        cfg = {**DEFAULT_FIT_CFG, **FAST}
        for l in (6, 9):
            r = P.fit_noa(pw, tr.feats[l], va.feats[l], t_tr, t_va, fit_cfg=cfg, device="cpu")
            want = res["fits"]["noa"][f"L{l}"]
            assert adapter_arrays_sha256(r["adapter"]) == want["adapter_sha256"], l
            assert (r["selected_wd"], r["selected_epoch"]) == (want["selected_wd"],
                                                               want["selected_epoch"])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    assert DEFAULT_FIT_CFG["epochs"] == phase2.ADAPTER_EPOCHS == 1500
    assert DEFAULT_FIT_CFG["lr"] == phase2.ADAPTER_LR == 1e-3
    assert tuple(DEFAULT_FIT_CFG["wd_grid"]) == phase2.ADAPTER_WD_GRID
    assert DEFAULT_FIT_CFG["eval_every"] == phase2.ADAPTER_EVAL_EVERY == 25
    from experiments import run_phase2e_noncontig as R
    a = R.parse_args([])
    cfg = R.make_fit_cfg(a)
    assert (cfg["epochs"], cfg["lr"], cfg["eval_every"], tuple(cfg["wd_grid"])) == \
        (1500, 1e-3, 25, phase2.ADAPTER_WD_GRID)
    for argv in (["--epochs", "300"], ["--lr", "1e-2"], ["--wd-grid", "0", "1"]):
        try:
            R.make_fit_cfg(R.parse_args(argv))
            raise AssertionError(f"protocol drift {argv} accepted")
        except (SystemExit, ValueError):     # the drift guard, or H3's own lr*wd < 1 guard
            pass
    try:
        P.fit_noa(pw, tr.feats[6], va.feats[6], t_tr, t_va,
                  fit_cfg={**cfg, "noa_adapter": "nested"})
        raise AssertionError("a non-H3 adapter was accepted")
    except ValueError:
        pass
    print(" 7   fit_noa == the H3 NOA fit (adapter sha256, wd, epoch); drift refused         OK")


# =========================================================================== #
# 6  weights never change
# =========================================================================== #
def test_06_no_backbone_weight_is_modified():
    if not _chronos_available():
        return _skip(" 6")
    from probing.phase2_h3 import DEFAULT_FIT_CFG
    pipe = tiny_chronos2_12()
    w0 = P.weights_digest(pipe)
    X = contexts(n=12)
    pw = T.pathway_for(pipe, "chronos2")
    for mask in ((1, 4, 8, 12), (4, 5, 6, 7), P.SKIP_MASKS[6]):
        P.extract_mask_states(pipe, X, mask)
        P.block_call_trace(pipe, X, mask)
        ad = ResidualAdapter(D)
        with torch.no_grad():
            ad.delta.normal_(0, 0.05)
        P.physical_forecast(pipe, X, mask, adapter=ad, batch_size=4)
    st = P.extract_mask_states(pipe, X, (1, 2, 11, 12))
    teach, _ = pw.run(P.extract_mask_states(pipe, X, P.FULL_MASK))
    cfg = {**DEFAULT_FIT_CFG, "epochs": 30, "eval_every": 10, "wd_grid": [0.0, 1.0]}
    P.fit_noa(pw, st[:8], st[8:], teach[:8], teach[8:], fit_cfg=cfg, device="cpu")
    assert P.weights_digest(pipe) == w0, "a pretrained weight changed"
    assert not any(p.requires_grad for p in pipe.model.parameters())
    pw.assert_frozen()
    print(" 6   extraction / physical forecasts / an adapter fit leave every weight unchanged OK")


# =========================================================================== #
# 8  no test access, no ground truth in the fit
# =========================================================================== #
def test_08_fit_reads_no_test_split_and_no_ground_truth():
    sig = set(inspect.signature(P.fit_noa).parameters)
    assert sig == {"pathway", "train_states", "val_states", "teacher_train", "teacher_val",
                   "fit_cfg", "device", "log", "label"}, sig
    for bad in ("test", "target", "y_raw", "y", "labels"):
        assert not any(bad == s or s.startswith(bad + "_") for s in sig), bad
    # poisoning the TEST split (states AND targets) moves no H3/2e selection and no adapter
    tmp1, tmp2 = Path(tempfile.mkdtemp()), Path(tempfile.mkdtemp())
    try:
        _, _, r1, _ = _h3_mock_run(tmp1)
        _, _, r2, _ = _h3_mock_run(tmp2, poison_test=True)
        for lab in ("L6", "L9"):
            assert r1["fits"]["noa"][lab]["adapter_sha256"] == \
                r2["fits"]["noa"][lab]["adapter_sha256"]
    finally:
        shutil.rmtree(tmp1, ignore_errors=True)
        shutil.rmtree(tmp2, ignore_errors=True)
    # the driver hands the fit states + the FULL MODEL's outputs only (static check of the call)
    src = inspect.getsource(importlib.import_module("experiments.run_phase2e_noncontig"))
    call = src[src.index("P.fit_noa("):src.index("P.fit_noa(") + 200]
    assert 'states["train"], states["val"], teacher_tr, teacher_va' in call, call
    assert "test" not in call.split("fit_cfg")[0], call
    print(" 8   fit signature has no test/target input; poisoning test moves no adapter     OK")


# =========================================================================== #
# 9  isolation
# =========================================================================== #
def test_09_outputs_are_isolated_from_other_phases():
    from experiments import run_phase2e_noncontig as R
    res = REPO_ROOT / "results"
    for name in R.PROTECTED:
        for p in (res / name, res / name / "x"):
            try:
                R.assert_output_root(p)
                raise AssertionError(f"accepted {p}")
            except SystemExit:
                pass
    assert R.assert_output_root(R.DEFAULT_OUT) == R.DEFAULT_OUT.resolve()
    for must in ("three_model_phase2_lodo", "three_model_phase2d_pooled", "three_model_phase2",
                 "three_model_phase1_q9_expanded"):
        assert must in R.PROTECTED, must
    tmp = Path(tempfile.mkdtemp())
    old = os.environ.get("SCRATCH")
    os.environ["SCRATCH"] = str(tmp)
    try:
        for sub in R.PROTECTED_SCRATCH:
            try:
                R.assert_scratch_root(tmp / sub / "y", "x")
                raise AssertionError(f"accepted {sub}")
            except SystemExit:
                pass
        a = R.parse_args([])
        R.assert_scratch_root(a.adapter_root, "adapter")
        R.assert_scratch_root(a.state_root, "state")
        assert "phase2e_noncontig" in a.adapter_root and "phase2e_noncontig" in a.state_root
        assert a.adapter_root != a.h3_adapter_root
    finally:
        if old is None:
            os.environ.pop("SCRATCH", None)
        else:
            os.environ["SCRATCH"] = old
        shutil.rmtree(tmp, ignore_errors=True)
    for mod in ("probing/phase2e_noncontig.py", "experiments/run_phase2e_noncontig.py",
                "experiments/run_phase2e_latency.py", "experiments/make_phase2e_tables.py"):
        txt = (REPO_ROOT / mod).read_text()
        for other in ("phase2_lodo", "phase2d_pooled", "run_phase2_lodo", "run_phase2d"):
            assert f"import {other}" not in txt and f"from probing.{other}" not in txt \
                and f"experiments.{other}" not in txt, (mod, other)
    print(" 9   every 2e output root/scratch root refused inside Phase 1/H3/2b/2c/2d trees  OK")


# =========================================================================== #
# 10  atomic + resume
# =========================================================================== #
def test_10_atomic_cells_state_cache_and_resume():
    from experiments import run_phase2e_noncontig as R
    tmp = Path(tempfile.mkdtemp())
    try:
        st = R.mask_store(tmp, "ds", "uniform_k4")
        assert st.status("h1")[0] == "pending"
        R._save_cell(st, "h1", {"a": 1}, {"metrics.json": {}, "fit.json": {},
                                          "verification.json": {}}, {"x": np.zeros(3)}, {})
        assert st.status("h1") == ("complete", "validated")
        assert st.status("h2")[0] == "incompatible"
        stale = tmp / "masks" / "ds" / "middle_k4.building-123"
        stale.mkdir(parents=True)
        other = tmp / "masks" / "ds2" / "middle_k4.building-9"
        other.mkdir(parents=True)
        removed = clean_staging(tmp, "masks", cells=[("ds", "middle_k4")])
        assert not stale.exists() and other.exists() and len(removed) == 1, removed
        try:                                    # a failing writer leaves NO final cell
            R._save_cell(R.mask_store(tmp, "ds", "early_late_k4"), "h", {}, {"metrics.json": {}},
                         None, {})
            raise AssertionError("an incomplete cell was committed")
        except RuntimeError:
            pass
        assert not (tmp / "masks" / "ds" / "early_late_k4").exists()
        assert not list((tmp / "masks" / "ds").glob("early_late_k4.building-*"))
        # state cache: atomic, verified on read
        cache = P.StateCache(tmp / "states")
        X = contexts(n=3)
        meta = P.StateCache.meta("ds", (1, 4), "val", X, window_digest="w", checkpoint="c",
                                 batch_size=128, numerics={"matmul_allow_tf32": False})
        assert cache.read("ds", (1, 4), "val", meta) is None
        s = np.random.default_rng(0).normal(size=(3, 4, D)).astype(np.float32)
        cache.write("ds", (1, 4), "val", meta, s)
        assert np.array_equal(cache.read("ds", (1, 4), "val", meta), s)
        assert cache.read("ds", (1, 4), "val", {**meta, "batch_size": 64}) is None
        assert cache.read("ds", (1, 4), "val", {**meta, "contexts_sha256": "other"}) is None
        cache.path("ds", (1, 4), "val").write_bytes(b"garbage")
        assert cache.read("ds", (1, 4), "val", meta) is None
        assert not list((tmp / "states").rglob("*.tmp*"))
        # resume: a COMPLETE cell is not pending, an incompatible one needs --force-recompute
        assert not R._pending(("complete", ""), True) and R._pending(("complete", ""), False)
        assert R._pending(("incompatible", ""), True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # latency repeats: natives at start/middle/end, a different seeded order per repeat
    from experiments import run_phase2e_latency as LT
    a, b = LT.config_list(1, P.plan_masks(), LT.KINDS, 0), LT.config_list(2, P.plan_masks(),
                                                                          LT.KINDS, 0)
    assert len(a) == 27 and [c["drift"] for c in a if c["kind"] == "native"] == \
        ["start", "middle", "end"]
    assert a == LT.config_list(1, P.plan_masks(), LT.KINDS, 0)
    assert [c["label"] for c in a] != [c["label"] for c in b]
    src = inspect.getsource(LT.run_repeat)
    assert "os.replace(stage, final)" in src and src.index('"index.json"') < \
        src.index("os.replace(stage, final)"), "a repeat must be renamed only after index.json"
    print("10   atomic cells, scoped cleanup, verified state cache, atomic latency repeats  OK")


# =========================================================================== #
# 11  physical == offline
# =========================================================================== #
def test_11_physical_equals_offline_with_an_adapter():
    if not _chronos_available():
        return _skip("11")
    pipe = tiny_chronos2_12()
    pw = T.pathway_for(pipe, "chronos2")
    X = contexts(n=8)
    inv = T.inverse_split(pipe, "chronos2", X)
    g = torch.Generator().manual_seed(1)
    for e in P.plan_masks():
        ad = ResidualAdapter(D)
        with torch.no_grad():
            ad.delta.copy_(torch.randn(D, D, generator=g) * 0.05)
            ad.bias.copy_(torch.randn(D, generator=g) * 0.05)
        ad.eval()
        st = P.extract_mask_states(pipe, X, e["mask"], batch_size=4)
        _, z = pw.run(st, ad)
        off = pw.to_raw(z, inv)
        phys = P.physical_forecast(pipe, X, e["mask"], adapter=ad, batch_size=4)
        gte = T.elementwise_gate(phys, off, atol=1e-4 * float(np.abs(off).mean()), rtol=1e-4)
        assert gte["passed"], (e["label"], gte)
        hard = P.physical_forecast(pipe, X, e["mask"], batch_size=4)
        ident = P.physical_forecast(pipe, X, e["mask"], batch_size=4,
                                    adapter=ResidualAdapter(D).eval())
        assert np.array_equal(hard, ident), e["label"]
    m = np.abs(np.random.default_rng(0).normal(size=40)) + 0.5
    assert P.v3_check(m, m)["passed"] and not P.v3_check(m * 1.01, m)["passed"]
    print("11   physical public API == offline pathway at all 12 masks; identity == hard   OK")


# =========================================================================== #
# 12  statistics
# =========================================================================== #
def _fake_rows(rng, n, scale=1.0, base=None):
    m = (base if base is not None else rng.uniform(0.6, 1.4, n)) * scale
    return {"loss": m * 0.1, "mase": m, "mae": m * 2, "wql_num": m * 3}


def test_12_paired_statistics():
    rng = np.random.default_rng(0)
    n = 60
    cid = np.arange(n) // 3
    den = rng.uniform(1, 2, n)
    base = rng.uniform(0.6, 1.4, n)
    nat = _fake_rows(rng, n, base=base)
    per = {}
    for k in P.BUDGETS:
        pref = _fake_rows(rng, n, 1.10, base)
        per[("prefix", k)] = {"hard": _fake_rows(rng, n, 1.5, base), "noa": pref}
        per[("uniform", k)] = {"hard": _fake_rows(rng, n, 2.0, base),
                               "noa": {kk: v.copy() for kk, v in pref.items()}}
        per[("early_late", k)] = {"hard": _fake_rows(rng, n, 1.4, base),
                                  "noa": _fake_rows(rng, n, 1.02, base)}
        per[("middle", k)] = {"hard": _fake_rows(rng, n, 3.0, base),
                              "noa": _fake_rows(rng, n, 1.30, base)}
    val = {key: {"hard": 1.0, "noa": {"prefix": 1.1, "uniform": 1.1, "early_late": 1.02,
                                      "middle": 1.3}[key[0]]} for key in per}
    c = P.comparison_stats(per, nat, den, cid, val_mase=val, B=400, seed=0)
    u = c["by_budget"]["k4"]["uniform|noa"]["vs_prefix"]["mase"]
    assert u["diff_pp"] == 0.0 and u["diff_pp_ci"] == [0.0, 0.0] and u["ratio"] == 1.0
    assert not u["better_ci"] and not u["worse_ci"]
    e = c["by_budget"]["k4"]["early_late|noa"]["vs_prefix"]["mase"]
    assert e["diff_pp"] < 0 and e["better_ci"], e
    m = c["by_budget"]["k4"]["middle|noa"]["vs_prefix"]["mase"]
    assert m["diff_pp"] > 0 and m["worse_ci"], m
    p = c["by_budget"]["k4"]["prefix|noa"]["vs_native"]["mase"]
    assert abs(p["degradation"] - 0.10) < 1e-12
    assert c["val_selected"]["k4|noa"]["selected"] == "early_late"
    val_tie = {key: {"hard": 1.0, "noa": 1.0} for key in per}
    c2 = P.comparison_stats(per, nat, den, cid, val_mase=val_tie, B=50, seed=0)
    assert c2["val_selected"]["k6|noa"]["selected"] == "prefix", "ties must go to prefix"
    sk = P.single_skip_stats({i: _fake_rows(rng, n, 1 + 0.01 * i, base) for i in range(1, 13)},
                             nat, den, cid, B=100, seed=0)
    assert [r["skipped_block"] for r in sk] == list(range(1, 13))
    assert abs(sk[4]["mase_degradation_pct"] - 5.0) < 1e-9
    cs = P.mask_cell_stats({"native": nat, "hard": per[("prefix", 4)]["hard"],
                            "noa": per[("prefix", 4)]["noa"]}, den, cid, B=100, seed=0)
    assert abs(cs["vs_native"]["hard"]["mase"]["degradation"] - 0.5) < 1e-12
    print("12   identical masks -> 0 pp [0, 0]; real gains/losses flagged; ties -> prefix  OK")


# =========================================================================== #
# 13  tables from artifacts only
# =========================================================================== #
def _write_fake_tree(root: Path):
    from experiments import run_phase2e_noncontig as R
    rng = np.random.default_rng(1)
    tags = ["monash_electricity_hourly", "uber_tlc_hourly", "m5"]
    n = 30
    cid = np.arange(n) // 2
    den = rng.uniform(1, 2, n)
    eff = {"prefix": 1.05, "uniform": 1.2, "early_late": 1.0, "middle": 1.4}
    for t in tags:
        base = rng.uniform(0.6, 1.4, n)
        nat = _fake_rows(rng, n, base=base)
        per, valm = {}, {}
        for e in P.plan_masks():
            s, k = e["strategy"], e["budget"]
            hard, noa = _fake_rows(rng, n, eff[s] * 1.5, base), _fake_rows(rng, n, eff[s], base)
            pwt = {"native": nat, "hard": hard, "noa": noa}
            stats = P.mask_cell_stats(pwt, den, cid, B=50, seed=0)
            status = "invalid_v3" if (t == "m5" and e["label"] == "uniform_k9") else "ok"
            metrics = {"dataset": t, "strategy": s, "budget": k, "label": e["label"],
                       "mask": list(e["mask"]), "mask_string": e["mask_string"],
                       "source": "h3" if e["is_prefix"] else "fit", "status": status,
                       "stats": stats, "test_mase": {a: stats["absolute"][a]["mase"] for a in pwt},
                       "val_mase": {"native": 1.0, "hard": 1.5 * eff[s], "noa": eff[s]},
                       "selected_wd": 10.0, "selected_epoch": 900,
                       "flags": {"selected_hard_cut": False, "late_checkpoint": False,
                                 "grid_max_wd": False}}
            npz = {f"native__test_{m}": v for m, v in nat.items()}
            npz.update({f"hard__test_{m}": v for m, v in hard.items()})
            npz.update({f"noa__test_{m}": v for m, v in noa.items()})
            npz.update({"hard__val_mase": hard["mase"], "noa__val_mase": noa["mase"],
                        "wql_den": den, "cluster_ids_test": cid})
            R._save_cell(R.mask_store(root, t, e["label"]), "h", {}, {
                "metrics.json": metrics, "fit.json": {}, "verification.json": {}}, npz, {})
            per[(s, k)] = {"hard": hard, "noa": noa}
            valm[(s, k)] = {"hard": float(hard["mase"].mean()), "noa": float(noa["mase"].mean())}
        cmp = P.comparison_stats(per, nat, den, cid, val_mase=valm, B=50, seed=0)
        R._save_cell(R.cmp_store(root, t), "h", {}, {"comparison.json": {"dataset": t, **cmp},
                                                     "verification.json": {}}, None, {})
        sk = P.single_skip_stats({i: _fake_rows(rng, n, 1 + 0.03 * i, base) for i in
                                  range(1, 13)}, nat, den, cid, B=50, seed=0)
        R._save_cell(R.skip_store(root, t), "h", {}, {"metrics.json": {"rows": sk},
                                                      "verification.json": {}}, {"x": den}, {})
        phase_gates = root / "gates"
        phase_gates.mkdir(parents=True, exist_ok=True)
        (phase_gates / f"{t}.json").write_text(json.dumps({"all_passed": True}))
    for r in (1, 2):
        rep = root / "latency" / f"rep{r}"
        rep.mkdir(parents=True)
        cfgs = [{"kind": "native", "label": "native", "status": "ok",
                 "median_ms": {"api_e2e__B1": 19.4 + 0.1 * r, "api_e2e__B256": 182.0}}]
        for e in P.plan_masks():
            for kind in ("hard", "adapter"):
                ms = 19.4 * (e["budget"] + 1) / 13
                cfgs.append({"kind": kind, "label": e["label"], "status": "ok",
                             "median_ms": {"api_e2e__B1": ms, "api_e2e__B256": 10 * ms}})
        (rep / "index.json").write_text(json.dumps({"repeat": r, "headline_eligible": True,
                                                    "configs": cfgs}))
    (root / "latency" / "rep3.building-77").mkdir(parents=True)        # a killed repeat
    return tags


def test_13_tables_regenerate_from_artifacts_only():
    try:
        import matplotlib  # noqa: F401
    except Exception:
        print("13   SKIPPED: matplotlib not importable")
        return
    from experiments import make_phase2e_tables as M
    tmp = Path(tempfile.mkdtemp())
    try:
        _write_fake_tree(tmp / "out")
        st = M.build(tmp / "out", tmp / "paper")
        assert st["n_mask_cells"] == 36 and st["n_datasets"] == 3
        for f in ("tables/per_dataset.csv", "tables/summary_by_budget.csv",
                  "tables/single_skip_summary.csv", "tables/latency.csv",
                  "tables/noncontig_budget_table.tex", "tables/phase2e_macros.tex",
                  "figures/noncontig_frontier.pdf", "figures/single_skip_sensitivity.pdf",
                  "figures/noncontig_paired_vs_prefix.pdf", "report/report.md",
                  "report/rebuttal_draft.md"):
            assert (tmp / "out" / f).exists(), f
        for f in ("noncontig_frontier.pdf", "tables/noncontig_budget_table.tex",
                  "phase2e_macros.tex"):
            assert (tmp / "paper" / f).exists(), f
        summ = {(r["budget"], r["strategy"], r["arm"]): r for r in st["summary"]}
        assert summ[(9, "uniform", "noa")]["n_datasets"] == 2, "invalid_v3 must be excluded"
        assert summ[(9, "uniform", "noa")]["n_invalid_v3"] == 1
        assert summ[(4, "early_late", "noa")]["n_better_ci"] == 3
        assert summ[(4, "middle", "noa")]["n_worse_ci"] == 3
        assert abs(summ[(4, "prefix", "noa")]["speedup_b1"] - 13 / 5) < 0.05
        assert len(st["latency_repeats"]) == 2, "a killed repeat must not be read"
        rep = (tmp / "out" / "report" / "report.md").read_text()
        assert "Limitations" in rep and "--" not in rep.replace("|---", "")
        reb = (tmp / "out" / "report" / "rebuttal_draft.md").read_text()
        assert "--" not in reb, "no double dash in paper-facing text"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("13   tables/figures/macros/report from artifacts; invalid_v3 and killed repeats out OK")


# =========================================================================== #
# 14  latency configuration
# =========================================================================== #
def test_14_latency_configuration_removes_and_releases_blocks():
    if not _chronos_available():
        return _skip("14")
    from experiments import run_phase2e_latency as LT
    for kind in ("hard", "adapter"):
        pipe = tiny_chronos2_12()
        core = pipe.model
        orig = list(core.encoder.block)
        refs = {i: weakref.ref(b) for i, b in enumerate(orig, start=1)}
        keep_ids = {i: id(orig[i - 1]) for i in (1, 2, 11, 12)}
        del orig
        rec = LT.configure_mask(pipe, {"kind": kind, "mask": [1, 2, 11, 12],
                                       "label": "early_late_k4"})
        gc.collect()
        assert [id(b) for b in core.encoder.block] == [keep_ids[i] for i in (1, 2, 11, 12)]
        assert all(refs[i]() is None for i in range(3, 11)), "skipped blocks still alive"
        assert rec["blocks_kept"] == 4 and rec["skipped"] == list(range(3, 11))
        if kind == "adapter":
            assert isinstance(core.encoder.final_layer_norm, torch.nn.Sequential)
            assert isinstance(core.encoder.final_layer_norm[0], ResidualAdapter)
        q = T.device_forward(pipe, "chronos2", contexts(n=2))
        assert tuple(q.shape) == (2, 21, 64), q.shape
    print("14   latency config: kept blocks are the originals, skipped ones freed, adapter in OK")


# =========================================================================== #
# 15  the DRIVER end to end, on a synthetic Phase-1 + H3 + H4 tree built by the REAL code
# =========================================================================== #
def _tree_sha(root: Path) -> dict:
    from probing.phase2 import file_sha256
    return {str(p.relative_to(root)): file_sha256(p) for p in sorted(Path(root).rglob("*"))
            if p.is_file()}


def test_15_driver_end_to_end_on_a_tiny_model():
    """No GPU smoke exists for Phase 2e, so the whole driver runs here: the real Phase-1
    extractor writes the K=4 caches, the real H3 cell code writes the H3 cell + adapters, a
    physical H4 point is written, then ``run_phase2e_noncontig.main`` runs every stage on a tiny
    12-block Chronos-2 and is re-run to prove resume. Read-only inputs must be byte-identical."""
    if not _chronos_available():
        return _skip("15")
    from experiments import make_phase2e_tables as M
    from experiments import run_phase2e_noncontig as R
    from probing import extraction, phase1
    from probing.phase1_metrics import raw_window_metrics
    from probing.phase2 import QUANTILES, TUNNEL_TOLS
    from probing.phase2_align import load_adapter
    from probing.phase2_h3 import run_h3_cell, save_h3_cell
    from probing.phase2_pathways import load_chronos2
    from probing.window_reference import window_digest
    from probing.windows import PAPER14_SET
    tmp = Path(tempfile.mkdtemp())
    pipe = tiny_chronos2_12()
    pw = T.pathway_for(pipe, "chronos2")
    tags = ["monash_electricity_hourly", "uber_tlc_hourly"]
    sizes = {"train": 40, "val": 16, "test": 16}
    scratch = tmp / "scratch"
    p1r, h3r, win, fc, out = tmp / "p1", tmp / "p2", tmp / "win", tmp / "fc", tmp / "out"
    h3ad = scratch / "chronos2_phase2" / "adapters"
    for d in (p1r, h3r, win, fc, h3ad):
        d.mkdir(parents=True, exist_ok=True)
    saved_cache, saved_get = extraction.CACHE_DIR, extraction.get_pipeline
    saved_spec = phase1.model_spec
    old_scratch = os.environ.get("SCRATCH")
    try:
        import dataclasses
        # the Phase-1 loader checks the 768-wide checkpoint shape; the tiny model is 32 wide
        phase1.model_spec = (lambda name: dataclasses.replace(saved_spec(name), d=D)
                             if name == "chronos2" else saved_spec(name))
        extraction.CACHE_DIR = fc
        for j, tag in enumerate(tags):
            rng = np.random.default_rng(10 + j)
            w = {}
            for s, n in sizes.items():
                w[f"X_{s}"] = (contexts(n, seed=100 * j + n) + 60.0).astype(np.float32)
                w[f"Y_{s}_traj"] = rng.normal(0, 0.5, (n, H)).astype(np.float32)
                w[f"series_{s}"] = (np.arange(n) // 2).astype(np.int64)
            meta = {"sigma_eps": 1e-6, "origins": {"test": list(range(sizes["test"]))}}
            np.savez(win / f"windows__paper14__{tag}.npz", **w, meta=json.dumps(meta))
            wd = window_digest({**w, "meta": meta})
            for s in sizes:                              # the REAL Phase-1 extractor
                extraction.extract_kout_features(tag, s, w[f"X_{s}"], w[f"Y_{s}_traj"], H,
                                                 batch_size=128, pipeline=pipe,
                                                 cache_prefix=f"IDF_{tag}__{PAPER14_SET}")
            data = load_chronos2(tag, {**w, "meta": meta}, {}, cache_dir=fc)
            te, va = data.splits["test"], data.splits["val"]
            _, z = pw.run(te.feats[12])
            m = raw_window_metrics(tag, te.X, te.y_raw, pw.to_raw(z, te), QUANTILES, 4)
            arrays = {"native_loss_window": phase1.mean_pinball_per_window(z, te.target, QUANTILES),
                      "native_mase_window": m["mase_pw"], "cluster_ids_test": te.cluster_ids,
                      "cluster_ids_val": va.cluster_ids}
            cell = p1r / "chronos2" / tag
            cell.mkdir(parents=True)
            np.savez(cell / "bootstrap_inputs.npz", **arrays)
            fit_hash = f"sha256:fit-{tag}"
            (cell / "cell_config.json").write_text(json.dumps(
                {"quantile_set": "q9", "window_digest": wd, "checkpoint": "amazon/chronos-2"}))
            (cell / "COMPLETE").write_text(json.dumps({"model": "chronos2", "dataset": tag,
                                                       "fit_hash": fit_hash, "config_hash": "c"}))
            ent = {t: {"depth_axis_index": 6, "label": "L6"} for t in TUNNEL_TOLS}
            res = run_h3_cell(pw, data, arrays, entrances=ent, families=["hard", "noa"],
                              device="cpu", fit_cfg={"epochs": 30, "eval_every": 10,
                                                     "wd_grid": [0.0, 1.0], "boot_b": 50},
                              adapter_dir=h3ad, log=lambda *a: None)
            st = Phase2CellStore(h3r, "h3", "chronos2", tag)
            stage = st.begin()
            save_h3_cell(stage, res, {"phase1_fit_hash": fit_hash, "window_digest": wd},
                         "h3hash", {"fit_hash": fit_hash}, {})
            st.commit(stage, "h3hash")
            if j == 0:                                   # a committed physical H4 point at L9
                ad9, _ = load_adapter(h3ad / "chronos2" / tag / "noa__L9.npz")
                h4 = h3r / "h4" / "chronos2" / tag
                h4.mkdir(parents=True)
                mm = {}
                for arm, a in (("hard", None), ("noa", ad9)):
                    q = P.physical_forecast(pipe, te.X, tuple(range(1, 10)), adapter=a)
                    mm[f"{arm}__mase"] = raw_window_metrics(tag, te.X, te.y_raw, q, QUANTILES,
                                                            4)["mase_pw"]
                np.savez(h4 / "physical_metrics.npz", **mm, cluster_ids_test=te.cluster_ids)
                (h4 / "truncation.json").write_text(json.dumps({"depth_index": 9}))
                (h4 / "COMPLETE").write_text("{}")
        extraction.CACHE_DIR = saved_cache
        before = {"p1": _tree_sha(p1r), "h3": _tree_sha(h3r), "fc": _tree_sha(fc),
                  "h3ad": _tree_sha(h3ad), "win": _tree_sha(win)}
        os.environ["SCRATCH"] = str(scratch)
        extraction.get_pipeline = lambda: (pipe, pipe.model.config)
        argv = ["--datasets", *tags, "--phase1-root", str(p1r), "--h3-root", str(h3r),
                "--h3-adapter-root", str(h3ad), "--window-cache", str(win),
                "--chronos-features-dir", str(fc), "--output-root", str(out),
                "--adapter-root", str(scratch / "chronos2" / "phase2e_noncontig" / "adapters"),
                "--state-root", str(scratch / "chronos2" / "phase2e_noncontig" / "states"),
                "--epochs", "30", "--eval-every", "10", "--wd-grid", "0", "1",
                "--allow-protocol-change", "--boot-b", "100", "--device", "cpu",
                "--fail-on-error"]
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = R.main(argv)
        log = buf.getvalue()
        assert rc == 0, log[-4000:]
        for tag in tags:
            for e in P.plan_masks():
                d = out / "masks" / tag / e["label"]
                assert (d / "COMPLETE").exists(), (tag, e["label"], log[-3000:])
                mt = json.loads((d / "metrics.json").read_text())
                assert mt["status"] == "ok", (tag, e["label"], mt["status"])
                if e["is_prefix"]:
                    assert mt["source"] == "h3", mt["source"]
                    ver = json.loads((d / "verification.json").read_text())
                    assert all(g["bitwise"] for g in ver["prefix_states_vs_phase1_cache"].values())
                    with np.load(d / "per_window.npz") as z, \
                            np.load(h3r / "h3" / "chronos2" / tag / "bootstrap_inputs.npz",
                                    allow_pickle=True) as hz:
                        k = e["budget"]
                        assert np.array_equal(z["noa__test_mase"], hz["noa__test_mase"][k])
                        assert np.array_equal(z["hard__test_mase"], hz["hard__test_mase"][k])
                    if tag == tags[0] and k == 9:
                        h4v = ver["physical_prefix_vs_committed_h4_physical"]
                        assert h4v["hard"]["passed"] and h4v["noa"]["passed"], h4v
            assert (out / "single_skip" / "chronos2" / tag / "COMPLETE").exists(), log[-3000:]
            assert (out / "comparison" / "chronos2" / tag / "COMPLETE").exists(), log[-3000:]
            assert json.loads((out / "gates" / f"{tag}.json").read_text())["all_passed"]
        v = json.loads((out / "verification" / "masks.json").read_text())
        assert len(v["passed_masks"]) == 25 and not v["failed_masks"], v["failed_masks"]
        ads = sorted((scratch / "chronos2" / "phase2e_noncontig" / "adapters").rglob("*.npz"))
        assert len(ads) == 9 * len(tags), len(ads)
        after = {"p1": _tree_sha(p1r), "h3": _tree_sha(h3r), "fc": _tree_sha(fc),
                 "h3ad": _tree_sha(h3ad), "win": _tree_sha(win)}
        assert after == before, "a read-only input tree changed"
        markers = {p: p.stat().st_mtime_ns for p in out.rglob("COMPLETE")}

        def refuse():
            raise AssertionError("resume loaded the model although nothing was pending")
        extraction.get_pipeline = refuse
        with contextlib.redirect_stdout(io.StringIO()):
            assert R.main(argv) == 0
        assert {p: p.stat().st_mtime_ns for p in out.rglob("COMPLETE")} == markers
        st = M.build(out, None)
        assert st["n_mask_cells"] == 24 and st["n_datasets"] == 2
        assert (out / "figures" / "noncontig_frontier.pdf").exists()
    finally:
        extraction.CACHE_DIR, extraction.get_pipeline = saved_cache, saved_get
        phase1.model_spec = saved_spec
        if old_scratch is None:
            os.environ.pop("SCRATCH", None)
        else:
            os.environ["SCRATCH"] = old_scratch
        shutil.rmtree(tmp, ignore_errors=True)
    print("15   DRIVER end to end (2 datasets, 24 cells): gates, H3 reuse, H4 gate, resume, "
          "read-only inputs unchanged OK")


def main(argv=None) -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print(f"\nall {len(tests)} Phase-2e contract groups passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
