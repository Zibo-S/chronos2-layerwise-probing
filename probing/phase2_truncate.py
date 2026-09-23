"""H4 — physical truncation: build the SHORTENED model, prove it is the offline H3 model, count it.

H3 is evaluated offline from cached hidden states; H4 instantiates the actual shortened model and
tests REAL acceleration. Everything here operates on the packages' own module objects:

    model      block list                              adapter splice (before the native norm/head)
    Chronos-2  model.encoder.block (ModuleList)        encoder.final_layer_norm = Sequential(a, norm)
    TimesFM-3  model.transformer_stack.layers          output_head = Sequential(a, output_head)
    TiRex      model.blocks                            out_norm = Sequential(a, out_norm)
                                                       (BOTH inference passes use the truncated model)

The adapter runs on every token (37 / 18 / 64 per pass); that is harmless because the norm and the
head act per token and only the readout tokens reach the forecast.

VERIFICATION (nothing is timed unless it passes; nothing assumes "0.0e+00" -- every run measures):
    V1  block-l readout states of the TRUNCATED model == hooked FULL model (== Phase-1 cache when
        given), at every depth incl. Emb
    V2  identity adapter spliced at FULL depth, through the public API == the unmodified model
    V3  physical == offline H3 predictions at every reported operating point  (driver-level)
    V4  removed blocks never execute (hooks registered on them before truncation fire 0 times) and
        are released (weakrefs die)
    V5  identity adapter at depth l == the hard cut at depth l (physical), bitwise
    V6  TiRex: each retained block and the adapter run exactly once per pass, i.e. twice per forecast

Active parameters are MEASURED: unique parameters (deduplicated by storage) of the module that
actually runs, never a formula -- an analytic cross-check is kept beside it.
"""

from __future__ import annotations

import gc
import weakref

import numpy as np
import torch
import torch.nn as nn

__all__ = ["BLOCK_PATHS", "core_of", "blocks_of", "set_blocks", "truncate", "splice_adapter",
           "FoldedLinear", "install_probe_head", "unique_param_count", "param_breakdown",
           "readout_states", "public_forecast", "device_forward", "verify_state_equivalence",
           "verify_identity_splice", "verify_removed_blocks_never_run",
           "verify_identity_equals_hard_cut", "verify_tirex_passes", "elementwise_gate",
           "pathway_for", "inverse_split", "offline_forecast", "verify_offline_equals_physical",
           "Q9", "ATOL", "RTOL"]

Q9 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ATOL, RTOL = 1e-5, 2e-6            # the repository's elementwise standard


# --------------------------------------------------------------------------- #
# the module layout of each package
# --------------------------------------------------------------------------- #
BLOCK_PATHS = {"chronos2": ("encoder", "block"), "timesfm3": ("transformer_stack", "layers"),
               "tirex": (None, "blocks")}


def core_of(handle, model: str) -> nn.Module:
    """The nn.Module that owns the blocks: pipeline.model (Chronos-2), forecaster.model
    (TimesFM-3), or the handle itself."""
    if model in ("chronos2", "timesfm3") and hasattr(handle, "model") and \
            isinstance(handle.model, nn.Module):
        return handle.model
    return handle


def blocks_of(core: nn.Module, model: str) -> nn.ModuleList:
    parent, name = BLOCK_PATHS[model]
    owner = core if parent is None else getattr(core, parent)
    return getattr(owner, name)


def set_blocks(core: nn.Module, model: str, blocks) -> None:
    parent, name = BLOCK_PATHS[model]
    owner = core if parent is None else getattr(core, parent)
    setattr(owner, name, nn.ModuleList(list(blocks)))


def splice_adapter(core: nn.Module, model: str, adapter: nn.Module) -> None:
    """Insert ``adapter`` immediately before the native final norm (Chronos-2, TiRex) or head
    (TimesFM-3, which has no final norm)."""
    if model == "chronos2":
        core.encoder.final_layer_norm = nn.Sequential(adapter, core.encoder.final_layer_norm)
    elif model == "timesfm3":
        core.output_head = nn.Sequential(adapter, core.output_head)
    elif model == "tirex":
        core.out_norm = nn.Sequential(adapter, core.out_norm)
    else:
        raise ValueError(model)


def truncate(handle, model: str, depth: int, adapter: nn.Module | None = None) -> dict:
    """Keep blocks 1..depth, drop the rest, splice ``adapter`` (None = the hard cut).

    Returns a record with WEAK references to the removed blocks: once the caller holds no strong
    reference, ``gc`` must free them (V4). depth = 0 keeps no block (Emb -> norm -> head).
    """
    core = core_of(handle, model)
    blocks = list(blocks_of(core, model))
    L = len(blocks)
    if not 0 <= int(depth) <= L:
        raise ValueError(f"{model}: depth must be in 0..{L}, got {depth}")
    removed = blocks[int(depth):]
    rec = {"model": model, "depth": int(depth), "blocks_before": L, "blocks_kept": int(depth),
           "blocks_removed": L - int(depth), "adapter": adapter is not None,
           "removed_weakrefs": [weakref.ref(b) for b in removed]}
    set_blocks(core, model, blocks[:int(depth)])
    if adapter is not None:
        splice_adapter(core, model, adapter)
    del removed, blocks
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rec


# --------------------------------------------------------------------------- #
# the Phase-1 probe as a drop-in head (the "why keep the native pathway?" control)
# --------------------------------------------------------------------------- #
class FoldedLinear(nn.Linear):
    """A Linear whose weights already contain the probe's StandardScaler:
    W' = W diag(1/s), b' = b - W' mu, so scaler + probe == one matmul (same outputs)."""


def _fold(weight, bias, mean, scale):
    W = np.asarray(weight, np.float64)
    s = np.asarray(scale, np.float64)
    Wf = W / s[None, :]
    bf = np.asarray(bias, np.float64) - Wf @ np.asarray(mean, np.float64)
    return Wf, bf


def install_probe_head(handle, model: str, probe: dict, native_quantiles=None) -> dict:
    """Replace the native (norm + head) by the Phase-1 probe, scaler folded in.

    ``probe`` holds weight, bias, scaler_mean, scaler_scale (a Phase-1 probe artifact). Layouts
    were checked against the source: the Chronos-2 probe is per-slot (q p) like the native head,
    the TimesFM-3 probe is horizon-major like the native head, the TiRex probe is quantile-major
    like the native head -- so the drop-in needs row PLACEMENT only, never a permutation.
    """
    core = core_of(handle, model)
    Wf, bf = _fold(probe["weight"], probe["bias"], probe["scaler_mean"], probe["scaler_scale"])
    dev = next(core.parameters()).device
    if model == "chronos2":
        P = 16
        nq = list(native_quantiles)
        idx = [int(np.flatnonzero(np.abs(np.asarray(nq) - q) < 1e-6)[0]) for q in Q9]
        d = Wf.shape[1]
        W_full = np.zeros((len(nq) * P, d))
        b_full = np.zeros(len(nq) * P)
        for j, level in enumerate(nq):              # every native row gets the NEAREST Q9 row
            k = int(np.argmin(np.abs(np.asarray(Q9) - level)))
            W_full[j * P:(j + 1) * P] = Wf[k * P:(k + 1) * P]
            b_full[j * P:(j + 1) * P] = bf[k * P:(k + 1) * P]
        for k, j in enumerate(idx):                 # the nine evaluated rows are EXACT
            assert np.array_equal(W_full[j * P:(j + 1) * P], Wf[k * P:(k + 1) * P])
        lin = _linear(W_full, b_full, dev)
        core.encoder.final_layer_norm = nn.Identity()
        core.output_patch_embedding = lin
    elif model == "timesfm3":
        core.output_head = _linear(Wf, bf, dev)
    elif model == "tirex":
        core.out_norm = nn.Identity()
        core.output_patch_embedding = _linear(Wf, bf, dev)
    else:
        raise ValueError(model)
    return {"model": model, "head": "folded Phase-1 probe", "params": int(Wf.size + bf.size),
            "out_features": int(Wf.shape[0])}


def _linear(W, b, dev):
    lin = FoldedLinear(W.shape[1], W.shape[0], bias=True)
    with torch.no_grad():
        lin.weight.copy_(torch.as_tensor(W, dtype=torch.float32))
        lin.bias.copy_(torch.as_tensor(b, dtype=torch.float32))
    lin.eval()
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin.to(dev)


# --------------------------------------------------------------------------- #
# parameter accounting -- measured
# --------------------------------------------------------------------------- #
def unique_param_count(module: nn.Module) -> int:
    seen, n = set(), 0
    for p in module.parameters():
        key = (p.data_ptr(), p.numel(), str(p.device)) if p.numel() else (id(p), 0, "")
        if key in seen:
            continue
        seen.add(key)
        n += p.numel()
    return int(n)


def param_breakdown(handle, model: str) -> dict:
    core = core_of(handle, model)
    blocks = blocks_of(core, model)
    per_block = [unique_param_count(b) for b in blocks]
    total = unique_param_count(core)
    return {"active_params": total, "n_blocks": len(blocks), "block_params": int(sum(per_block)),
            "per_block_params": per_block[0] if per_block else None,
            "non_block_params": total - int(sum(per_block))}


# --------------------------------------------------------------------------- #
# forward passes: public API, device-resident forward, readout-state capture
# --------------------------------------------------------------------------- #
def _readout_hook_target(core, model, full_depth: int | None):
    """The module whose OUTPUT (or input) carries the block-l state.

    Full model at depth l: block l-1's output (l >= 1) or the input of block 0 (l = 0).
    Truncated model: the INPUT of the final norm / head (whatever the last kept block produced).
    """
    if full_depth is None:
        target = {"chronos2": lambda: core.encoder.final_layer_norm,
                  "timesfm3": lambda: core.output_head,
                  "tirex": lambda: core.out_norm}[model]()
        return target, "input"
    blocks = blocks_of(core, model)
    if full_depth == 0:
        return blocks[0], "input"
    return blocks[full_depth - 1], "output"


def _select_readout(h, model, geom):
    if model == "chronos2":
        return h[:, -4:, :]                                   # the K = 4 forecast slots
    if model == "timesfm3":
        if h.dim() == 4:                                      # (b, v, n_tokens, d)
            h = h[:, 0]
        return h[:, geom["timesfm_readout_token"]:geom["timesfm_readout_token"] + 1, :]
    return h[:, -1:, :]                                       # TiRex: the last token of the pass


@torch.no_grad()
def readout_states(handle, model: str, X, *, full_depth: int | None = None, geom=None,
                   batch_size: int = 64) -> np.ndarray:
    """(n, R, d) readout states: at block ``full_depth`` of a full model, or at the input of the
    final norm / head of a (truncated) model when ``full_depth`` is None. TiRex's two passes are
    concatenated along R (pass 0, pass 1), matching the Phase-1 cache layout."""
    core = core_of(handle, model)
    geom = geom or {"timesfm_readout_token": 15}
    module, where = _readout_hook_target(core, model, full_depth)
    captured = []

    def pre(_m, args):
        captured.append(_select_readout(args[0], model, geom).detach().float().cpu())

    def post(_m, _a, out):
        h = out[0] if isinstance(out, tuple) else getattr(out, "hidden_states", out)
        captured.append(_select_readout(h, model, geom).detach().float().cpu())

    hk = (module.register_forward_pre_hook(pre) if where == "input"
          else module.register_forward_hook(post))
    outs = []
    try:
        for s in range(0, len(X), batch_size):
            captured.clear()
            device_forward(handle, model, np.asarray(X[s:s + batch_size], np.float32))
            if model == "tirex":
                if len(captured) != 2:
                    raise RuntimeError(f"TiRex readout hook fired {len(captured)} times per "
                                       "forecast; the two-pass default path expects 2")
                outs.append(torch.cat(captured, dim=1))
            else:
                if len(captured) < 1:
                    raise RuntimeError(f"{model}: the readout hook never fired")
                outs.append(captured[-1])
    finally:
        hk.remove()
    return torch.cat(outs).numpy()


@torch.no_grad()
def device_forward(handle, model: str, X, horizon: int = 64):
    """Device-resident raw context -> the model's own forward (with its in-model normalization)
    -> device quantiles. The uniform-semantics latency level (``device_forward``)."""
    core = core_of(handle, model)
    dev = next(core.parameters()).device
    x = torch.as_tensor(np.asarray(X, np.float32), device=dev)
    if model == "chronos2":
        return core(context=x, num_output_patches=int(np.ceil(horizon / 16))).quantile_preds
    if model == "timesfm3":
        return core.decode(target=x[:, None, :], horizon=horizon)
    if model == "tirex":
        q, _ = core._forecast_quantiles(x, prediction_length=horizon, output_device=str(dev),
                                        max_accelerated_rollout_steps=1)
        return q
    raise ValueError(model)


@torch.no_grad()
def public_forecast(handle, model: str, X, *, batch_size: int, horizon: int = 64,
                    sort_quantiles: bool | None = None) -> np.ndarray:
    """The package's PUBLIC forecasting API, host numpy in -> (n, H, 9) host numpy out.

    Chronos-2 ``predict_quantiles`` (Q9 selected exactly from the 21 native levels); TimesFM-3
    ``predict_batch`` with ``per_core_batch_size = batch_size`` (the default 4 would silently split
    a batch of 256 into 64 decode calls); TiRex ``forecast`` (default two-pass).
    """
    X = np.asarray(X, np.float32)
    if model == "chronos2":
        qs, _ = handle.predict_quantiles(list(X), prediction_length=horizon,
                                         quantile_levels=list(Q9), batch_size=batch_size)
        return np.stack([np.asarray(q.cpu() if hasattr(q, "cpu") else q, np.float64)[0]
                         for q in qs])
    if model == "timesfm3":
        import dataclasses
        handle.config = dataclasses.replace(handle.config, per_core_batch_size=int(batch_size))
        kw = {} if sort_quantiles is None else {"sort_quantiles": bool(sort_quantiles)}
        outs = list(handle.predict_batch(contexts=list(X), horizon=horizon, return_quantiles=True,
                                         **kw))
        return np.stack([np.asarray(o.quantiles, np.float64) for o in outs])
    if model == "tirex":
        out = handle.forecast(X, prediction_length=horizon, batch_size=batch_size,
                              output_type="numpy")
        q = out[0] if isinstance(out, tuple) else out         # forecast() -> (quantiles, mean)
        return np.asarray(q, np.float64)
    raise ValueError(model)


# --------------------------------------------------------------------------- #
# verification gates
# --------------------------------------------------------------------------- #
def elementwise_gate(got, ref, atol: float = ATOL, rtol: float = RTOL) -> dict:
    got, ref = np.asarray(got, np.float64), np.asarray(ref, np.float64)
    if got.shape != ref.shape:
        return {"passed": False, "reason": f"shape {got.shape} vs {ref.shape}"}
    d = np.abs(got - ref)
    scaled = d / (atol + rtol * np.abs(ref))
    return {"bitwise": bool(np.array_equal(got, ref)), "max_abs": float(d.max()),
            "max_scaled": float(scaled.max()), "atol": atol, "rtol": rtol,
            "n": int(got.shape[0]), "passed": bool(scaled.max() <= 1.0)}


def verify_state_equivalence(load_fn, model: str, depth: int, X, *, cached=None, geom=None,
                             batch_size: int = 64) -> dict:
    """V1: truncated-model readout states == hooked full-model states (== Phase-1 cache)."""
    full = load_fn()
    ref = readout_states(full, model, X, full_depth=depth, geom=geom, batch_size=batch_size)
    del full
    gc.collect()
    trunc = load_fn()
    truncate(trunc, model, depth, adapter=None)
    got = readout_states(trunc, model, X, full_depth=None, geom=geom, batch_size=batch_size)
    del trunc
    gc.collect()
    rec = {"check": "V1", "depth": int(depth), "truncated_vs_hooked_full": elementwise_gate(got, ref)}
    if cached is not None:
        rec["truncated_vs_phase1_cache"] = elementwise_gate(got, cached)
    rec["passed"] = all(v.get("passed", True) for v in rec.values() if isinstance(v, dict))
    return rec


def verify_identity_splice(load_fn, model: str, X, *, batch_size: int, d: int) -> dict:
    """V2: identity adapter at FULL depth through the public API == the unmodified model."""
    from probing.phase2_align import ResidualAdapter
    ref = public_forecast(load_fn(), model, X, batch_size=batch_size)
    h = load_fn()
    L = len(blocks_of(core_of(h, model), model))
    dev = next(core_of(h, model).parameters()).device
    truncate(h, model, L, adapter=ResidualAdapter(d).to(dev).eval())
    got = public_forecast(h, model, X, batch_size=batch_size)
    return {"check": "V2", **elementwise_gate(got, ref)}


def verify_removed_blocks_never_run(load_fn, model: str, depth: int, X) -> dict:
    """V4: hooks on the removed blocks fire 0 times; the removed blocks are freed."""
    h = load_fn()
    core = core_of(h, model)
    blocks = list(blocks_of(core, model))
    fired = {"n": 0}
    handles = [b.register_forward_hook(lambda *_: fired.__setitem__("n", fired["n"] + 1))
               for b in blocks[depth:]]
    del blocks
    rec = truncate(h, model, depth, adapter=None)
    device_forward(h, model, X)
    for hk in handles:
        hk.remove()
    del handles
    gc.collect()
    alive = sum(r() is not None for r in rec["removed_weakrefs"])
    return {"check": "V4", "depth": int(depth), "removed_blocks": rec["blocks_removed"],
            "removed_block_calls": fired["n"], "removed_blocks_still_alive": int(alive),
            "passed": bool(fired["n"] == 0 and alive == 0)}


def verify_identity_equals_hard_cut(load_fn, model: str, depth: int, X, *, batch_size: int,
                                    d: int) -> dict:
    """V5: identity adapter at depth l == the hard cut at depth l, physical, bitwise."""
    from probing.phase2_align import ResidualAdapter
    a = load_fn()
    truncate(a, model, depth, adapter=None)
    ref = public_forecast(a, model, X, batch_size=batch_size)
    b = load_fn()
    dev = next(core_of(b, model).parameters()).device
    truncate(b, model, depth, adapter=ResidualAdapter(d).to(dev).eval())
    got = public_forecast(b, model, X, batch_size=batch_size)
    g = elementwise_gate(got, ref)
    g["passed"] = bool(g.get("bitwise"))
    return {"check": "V5", "depth": int(depth), **g}


def verify_tirex_passes(load_fn, depth: int, X, d: int) -> dict:
    """V6: every retained block and the adapter run once per pass (two passes per forecast)."""
    from probing.phase2_align import ResidualAdapter
    h = load_fn()
    dev = next(h.parameters()).device
    ad = ResidualAdapter(d).to(dev).eval()
    truncate(h, "tirex", depth, adapter=ad)
    counts = {"adapter": 0, "blocks": [0] * depth}
    hs = [ad.register_forward_hook(lambda *_: counts.__setitem__("adapter", counts["adapter"] + 1))]
    for i, b in enumerate(h.blocks):
        hs.append(b.register_forward_hook(
            lambda *_, i=i: counts["blocks"].__setitem__(i, counts["blocks"][i] + 1)))
    device_forward(h, "tirex", X)
    for x in hs:
        x.remove()
    ok = counts["adapter"] == 2 and all(c == 2 for c in counts["blocks"])
    return {"check": "V6", "depth": int(depth), **counts, "passed": bool(ok)}


# --------------------------------------------------------------------------- #
# the OFFLINE H3 computation on a live handle -- the reference every physical model must match
# --------------------------------------------------------------------------- #
def model_width(handle, model: str) -> int:
    core = core_of(handle, model)
    if model == "chronos2":
        return int(core.config.d_model)
    if model == "timesfm3":
        head = core.output_head[-1] if isinstance(core.output_head, nn.Sequential) \
            else core.output_head
        return int(head.in_features)
    norm = core.out_norm[-1] if isinstance(core.out_norm, nn.Sequential) else core.out_norm
    return int(norm.weight.numel())


def pathway_for(handle, model: str):
    """The H3 native pathway built from THIS handle's own (unmodified) norm/head modules."""
    from probing.phase2_pathways import Chronos2Pathway, TiRexPathway, TimesFM3Pathway
    core = core_of(handle, model)
    d = model_width(handle, model)
    if model == "chronos2":
        cls = type("Chronos2PathwayD", (Chronos2Pathway,), {"d": d})
        return cls(core.output_patch_embedding, core.encoder.final_layer_norm, handle.quantiles)
    if model == "timesfm3":
        cls = type("TimesFM3PathwayD", (TimesFM3Pathway,), {"d": d})
        return cls(core.output_head, core.value_clip, core.output_patch_len, core.num_quantiles)
    from probing.tirex_model import geometry_from_model
    cls = type("TiRexPathwayD", (TiRexPathway,), {"d": d})
    return cls(core.output_patch_embedding, core.out_norm, geometry_from_model(core, 512, 64))


@torch.no_grad()
def inverse_split(handle, model: str, X, horizon: int = 64):
    """What each model's own inverse transform needs for contexts X, computed exactly as the H3
    loaders compute it from the Phase-1 caches (Chronos-2 context mean/std; TimesFM-3 decode()'s
    token-15 RevIN stats + the context detrend; TiRex per-pass loc/scale)."""
    from probing.phase2_pathways import SplitData
    X = np.asarray(X, np.float32)
    core = core_of(handle, model)
    if model == "chronos2":
        X64 = X.astype(np.float64)
        inv = {"mu": X64.mean(1), "sd": np.maximum(X64.std(1), 1e-6)}
    elif model == "timesfm3":
        from probing.timesfm3_last_token import context_detrend_params, context_trend
        dev = next(core.parameters()).device
        _, aux = core.decode(target=torch.as_tensor(X, device=dev)[:, None, :],
                             horizon=horizon, return_aux_outputs=True)
        mu, sd = aux["revin_stats"]
        C = X.shape[1]
        tok = C // core.input_patch_len - 1
        a, b, act = context_detrend_params(X.astype(np.float64))
        inv = {"mu": mu[:, 0, tok].double().cpu().numpy(),
               "sd": sd[:, 0, tok].double().cpu().numpy(),
               "trend": context_trend(a, b, act, C, np.arange(C, C + horizon))}
    else:
        from probing.tirex_model import geometry_from_model, scaler_states_for_mode
        loc, scale = scaler_states_for_mode(core, torch.as_tensor(X),
                                            geometry_from_model(core, X.shape[1], horizon),
                                            "two_pass")
        inv = {"loc": np.asarray(loc, np.float64), "scale": np.asarray(scale, np.float64)}
    return SplitData({}, None, None, X, inv, None, None)


@torch.no_grad()
def offline_forecast(handle, model: str, X, depth: int, *, adapter=None, probe=None,
                     batch_size: int = 64) -> np.ndarray:
    """The H3 OFFLINE forecast at ``depth`` on an UNMODIFIED handle: hooked block-l readout states
    -> [adapter] -> native norm + head -> the model's own inverse; or, with ``probe``, the Phase-1
    probe (scaler + Linear, the Phase-1 layout) instead of the native head. (n, 9, H) raw."""
    X = np.asarray(X, np.float32)
    states = readout_states(handle, model, X, full_depth=depth, batch_size=batch_size)
    pw = pathway_for(handle, model)
    if probe is None:
        _, z9 = pw.run(states, adapter)
    else:
        Z = (states - np.asarray(probe["scaler_mean"])) / np.asarray(probe["scaler_scale"])
        y = torch.as_tensor(Z @ np.asarray(probe["weight"]).T + np.asarray(probe["bias"]),
                            dtype=torch.float32)
        n = len(X)
        if model == "chronos2":
            from probing.probes import _apply_shared_head
            z9 = _apply_shared_head(lambda t: t, y, 9, 16, 64).double().numpy()
        elif model == "timesfm3":
            z9 = y.reshape(n, 64, 9).permute(0, 2, 1).double().numpy()
        else:
            z9 = y.view(n, 2, 9, 32).permute(0, 2, 1, 3).reshape(n, 9, 64).double().numpy()
    return pw.to_raw(z9, inverse_split(handle, model, X))


def verify_offline_equals_physical(load_fn, model: str, depth: int, X, *, adapter=None,
                                   probe=None, batch_size: int = 64, atol_rel: float = 1e-4,
                                   rtol: float = 1e-4) -> dict:
    """V3 on synthetic contexts: the physically truncated model (adapter or probe head) through
    the PUBLIC API == the offline H3 computation on the full model. Not bitwise (different GEMM
    shapes); gated elementwise at atol = atol_rel * mean|ref|, rtol."""
    import copy as _copy
    full = load_fn()
    ref = offline_forecast(full, model, X, depth, adapter=adapter, probe=probe,
                           batch_size=batch_size)
    h = load_fn()
    if probe is None:
        truncate(h, model, depth, adapter=None if adapter is None else _copy.deepcopy(adapter))
    else:
        truncate(h, model, depth)
        install_probe_head(h, model, probe, native_quantiles=getattr(h, "quantiles", None))
    kw = {"sort_quantiles": False} if model == "timesfm3" else {}
    got = np.ascontiguousarray(public_forecast(h, model, X, batch_size=batch_size, **kw)
                               .transpose(0, 2, 1))
    scale = float(np.abs(ref).mean())
    g = elementwise_gate(got, ref, atol=atol_rel * scale, rtol=rtol)
    return {"check": "V3-synthetic" if probe is None else "PH", "depth": int(depth), **g}
