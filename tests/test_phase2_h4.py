"""CONTRACT SUITE for the Phase-2 pathways (H3) and the physical truncation machinery (H4),
on TINY RANDOM-WEIGHT instances of the three REAL architectures.

No checkpoint and no dataset: each model is built from the installed package's own classes at a
tiny width (d = 32, 3 blocks), so every layout, splice, hook and inverse transform is exercised on
the real module structure in seconds on a CPU:

    python -m tests.test_phase2_h4              # login node OK: ~1 min, 2 threads, CPU only

A model whose package is not importable is SKIPPED with the reason (never silently passed).

What is pinned, per model:
    P   the offline pathway (h_L -> norm -> head -> inverse) == the model's OWN forecast
    V1  truncated readout states == hooked full-model states, at depth 0, 1 and L-1
    V2  identity adapter at full depth through the PUBLIC API == the unmodified model (bitwise)
    V3  physical truncated + adapter model == the OFFLINE H3 computation (the H3 -> H4 bridge)
    V4  removed blocks never execute and are freed
    V5  identity adapter at depth l == the hard cut at depth l (bitwise)
    V6  (TiRex) every retained block and the adapter run once per pass
    PH  the folded probe head == scaler + probe, through the public API
    AP  active parameters == full - removed blocks + adapter (measured == analytic)
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase2_truncate as T                                         # noqa: E402
from probing.phase2 import QUANTILES                                              # noqa: E402
from probing.phase2_align import ResidualAdapter                                  # noqa: E402

torch.set_num_threads(2)
D, NB, C, H = 32, 3, 512, 64
CHRONOS_Q21 = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65,
               0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99]


# --------------------------------------------------------------------------- #
# tiny real models
# --------------------------------------------------------------------------- #
def _freeze(m):
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def tiny_chronos2(seed=0):
    from chronos import Chronos2Pipeline
    from chronos.chronos2 import Chronos2Model
    from chronos.chronos2.config import Chronos2CoreConfig
    cfg = Chronos2CoreConfig(d_model=D, d_kv=8, d_ff=64, num_layers=NB, num_heads=4,
                             dropout_rate=0.0, reg_token_id=1, attn_implementation="eager",
                             chronos_config={"context_length": 8192, "input_patch_size": 16,
                                             "input_patch_stride": 16, "output_patch_size": 16,
                                             "max_output_patches": 64, "quantiles": CHRONOS_Q21,
                                             "use_reg_token": True, "use_arcsinh": True,
                                             "time_encoding_scale": 8192})
    torch.manual_seed(seed)
    return Chronos2Pipeline(_freeze(Chronos2Model(cfg)))


def tiny_timesfm3(seed=0):
    from timesfm3.torch import TimesFM3Forecaster, TimesFM3Torch, configs
    from timesfm3.torch.timesfm3_forecaster import _ModelConfig
    torch.manual_seed(seed)
    m = TimesFM3Torch(
        residual_block_config=configs.ResidualBlockConfig(hidden_dims=D, output_dims=D,
                                                          use_bias=False, activation="relu"),
        transformer_config=configs.StackedTransformersConfig(
            num_layers=NB, transformer=configs.TransformerConfig(
                model_dims=D, hidden_dims=D, num_heads=4, attention_norm="rms",
                feedforward_norm="rms", qk_norm="rms", use_rope_seq=True, use_rope_var=False,
                use_bias=False, ff_activation="relu", deterministic=True)))
    fc = object.__new__(TimesFM3Forecaster)            # no checkpoint: bypass the weight loading
    fc.config = _ModelConfig(device="cpu")
    fc.model, fc.device = _freeze(m), torch.device("cpu")
    return fc


def tiny_tirex(seed=0):
    from tirex.models.tirex import TiRexZero
    torch.manual_seed(seed)
    m = TiRexZero(backend="torch", train_ctx_len=2048, model_config={
        "input_patch_size": 32, "output_patch_size": 32, "quantiles": list(QUANTILES),
        "input_ff_dim": 64, "block_kwargs": {"num_blocks": NB, "embedding_dim": D,
                                             "num_heads": 4}})
    # The sLSTM gate kernels are allocated with torch.empty (the package always loads a
    # checkpoint over them), so a weight-less instance holds garbage/NaN. Initialize every
    # parameter deterministically: norms -> 1, biases -> 0, weights -> N(0, 0.05).
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if name.endswith("norm.weight") or name.endswith("norm_slstm.weight") \
                    or name.endswith("norm_ffn.weight") or "group_norm" in name:
                p.fill_(1.0)
            elif "bias" in name:
                p.zero_()
            else:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    return _freeze(m)


BUILDERS = {"chronos2": tiny_chronos2, "timesfm3": tiny_timesfm3, "tirex": tiny_tirex}


def _available(model):
    try:
        BUILDERS[model]()
        return True, ""
    except Exception as e:                                          # pragma: no cover
        return False, f"{type(e).__name__}: {e}"


def _contexts(n=6, seed=1):
    rng = np.random.default_rng(seed)
    return (np.cumsum(rng.normal(size=(n, C)), axis=1) * rng.uniform(0.5, 3.0, size=(n, 1))
            + rng.uniform(-20, 20, size=(n, 1))).astype(np.float32)


# --------------------------------------------------------------------------- #
# offline pathways on the tiny models (the H3 view of the same weights) -- LIBRARY functions
# --------------------------------------------------------------------------- #
def _pathway(model, handle):
    return T.pathway_for(handle, model)


def _inverse_stats(model, handle, X):
    return T.inverse_split(handle, model, X)


def _offline_raw(model, handle, X, depth, adapter=None):
    return T.offline_forecast(handle, model, X, depth, adapter=adapter)


def _as_nqh(raw_nhq):
    return np.ascontiguousarray(np.asarray(raw_nhq).transpose(0, 2, 1))


# --------------------------------------------------------------------------- #
# contracts
# --------------------------------------------------------------------------- #
def contract_P(model):
    h = BUILDERS[model]()
    X = _contexts()
    off = _offline_raw(model, h, X, NB)
    own = _as_nqh(T.public_forecast(h, model, X, batch_size=4, sort_quantiles=False)
                  if model == "timesfm3" else T.public_forecast(h, model, X, batch_size=4))
    g = T.elementwise_gate(off, own, atol=1e-4, rtol=1e-4)
    assert g["passed"], (model, g)
    return f"offline f(h_L) == the model's own forecast (max|d| {g['max_abs']:.1e})"


def contract_V1(model):
    X = _contexts(4)
    out = []
    for depth in (0, 1, NB - 1):
        r = T.verify_state_equivalence(BUILDERS[model], model, depth, X)
        assert r["passed"], (model, depth, r)
        out.append(r["truncated_vs_hooked_full"]["bitwise"])
    return f"truncated states == hooked full model at depths 0,1,{NB - 1} (bitwise {out})"


def contract_V2(model):
    r = T.verify_identity_splice(BUILDERS[model], model, _contexts(4), batch_size=4, d=D)
    assert r["passed"] and r["bitwise"], (model, r)
    return "identity adapter at full depth through the public API == native (bitwise)"


def contract_V3(model):
    """A NON-trivial adapter: physical truncated model == offline H3 computation."""
    X = _contexts(5, seed=7)
    torch.manual_seed(3)
    ad = ResidualAdapter(D)
    with torch.no_grad():
        ad.delta.copy_(0.05 * torch.randn(D, D))
        ad.bias.copy_(0.05 * torch.randn(D))
    ad.eval()
    depth = 1
    off = _offline_raw(model, BUILDERS[model](), X, depth, adapter=ad)
    h = BUILDERS[model]()
    T.truncate(h, model, depth, adapter=copy.deepcopy(ad))
    kw = {"sort_quantiles": False} if model == "timesfm3" else {}
    phys = _as_nqh(T.public_forecast(h, model, X, batch_size=5, **kw))
    g = T.elementwise_gate(phys, off, atol=1e-4, rtol=1e-4)
    assert g["passed"], (model, g)
    return f"physical(depth 1 + adapter) == offline H3 path (max|d| {g['max_abs']:.1e})"


def contract_V4(model):
    r = T.verify_removed_blocks_never_run(BUILDERS[model], model, 1, _contexts(3))
    assert r["passed"], (model, r)
    return f"{r['removed_blocks']} removed blocks: 0 calls, 0 alive after gc"


def contract_V5(model):
    r = T.verify_identity_equals_hard_cut(BUILDERS[model], model, 1, _contexts(4),
                                          batch_size=4, d=D)
    assert r["passed"], (model, r)
    return "identity adapter at depth 1 == hard cut at depth 1 (bitwise)"


def contract_V6(model):
    if model != "tirex":
        return None
    r = T.verify_tirex_passes(BUILDERS[model], 2, _contexts(3), d=D)
    assert r["passed"], r
    return f"adapter ran {r['adapter']}x, blocks {r['blocks']} per forecast (2 passes)"


def contract_PH(model):
    """Folded probe head through the public API == scaler + probe applied offline."""
    rng = np.random.default_rng(11)
    out = {"chronos2": 9 * 16, "timesfm3": 64 * 9, "tirex": 9 * 32}[model]
    probe = {"weight": rng.normal(size=(out, D)) * 0.05, "bias": rng.normal(size=out) * 0.05,
             "scaler_mean": rng.normal(size=D), "scaler_scale": rng.uniform(0.5, 2.0, D)}
    r = T.verify_offline_equals_physical(BUILDERS[model], model, 2, _contexts(4, seed=5),
                                         probe=probe, batch_size=4)
    assert r["passed"], (model, r)
    return f"folded probe head == scaler+probe offline (max|d| {r['max_abs']:.1e})"


def contract_AP(model):
    full = T.param_breakdown(BUILDERS[model](), model)
    h = BUILDERS[model]()
    T.truncate(h, model, 1, adapter=ResidualAdapter(D))
    got = T.param_breakdown(h, model)
    want = full["active_params"] - (NB - 1) * full["per_block_params"] + D * D + D
    assert got["active_params"] == want, (model, got, want)
    return f"active params {got['active_params']} == full - {NB - 1} blocks + adapter"


def contract_LH(model):
    """The latency child on a tiny model: raw vector length == reps, every level/batch keyed,
    throughput = B / median, parameters measured; an OOM is RECORDED, never substituted."""
    from experiments import run_phase2_latency as LAT
    cfg = {"model": model, "kind": "adapter", "depth": 1, "batch_sizes": [1, 3],
           "levels": list(LAT.LEVELS), "warmup": 1, "reps": 3, "device": "cpu", "tiny": True}
    res = LAT.run_child(cfg)
    raw = res["raw_timings_ns"]
    for B in (1, 3):
        for lvl in LAT.LEVELS:
            k = f"{lvl}__B{B}"
            assert raw[k].shape == (3,) and raw[k].dtype == np.int64, k
            r = res["levels"][k]
            assert np.isclose(r["throughput_series_per_s"], B / (r["median_ms"] / 1e3))
    assert res["params"]["n_blocks"] == 1
    orig = T.device_forward

    def boom(*a, **k):
        raise torch.cuda.OutOfMemoryError("synthetic OOM")
    T.device_forward = boom
    try:
        res2 = LAT.run_child({**cfg, "levels": ["device_forward"], "batch_sizes": [3]})
    finally:
        T.device_forward = orig
    assert res2["levels"]["device_forward__B3"]["status"] == "oom"
    assert "device_forward__B3" not in res2["raw_timings_ns"]
    return "latency child: raw ns vectors, throughput = B/median, OOM recorded not substituted"


CONTRACTS = [("P ", contract_P), ("V1", contract_V1), ("V2", contract_V2), ("V3", contract_V3),
             ("V4", contract_V4), ("V5", contract_V5), ("V6", contract_V6), ("PH", contract_PH),
             ("AP", contract_AP), ("LH", contract_LH)]


def main(argv=None):
    print("\nPHASE-2 PATHWAY + TRUNCATION CONTRACTS on tiny real architectures\n" + "=" * 78)
    import traceback
    n_ok, n_skip, failed = 0, 0, []
    for model in ("chronos2", "timesfm3", "tirex"):
        ok, why = _available(model)
        if not ok:
            print(f"[SKIP] {model}: package not importable here ({why})")
            n_skip += 1
            continue
        for name, fn in CONTRACTS:
            try:
                msg = fn(model)
            except Exception:             # report EVERY failing contract, not only the first
                failed.append(f"{name}/{model}")
                print(f"{name} {model:<9} FAIL")
                traceback.print_exc()
                continue
            if msg is None:
                continue
            print(f"{name} {model:<9} {msg}  OK")
            n_ok += 1
    print("=" * 78 + f"\n{n_ok} contracts hold" + (f"; {n_skip} model(s) SKIPPED" if n_skip else ""))
    if failed:
        print(f"{len(failed)} contract(s) FAILED: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
