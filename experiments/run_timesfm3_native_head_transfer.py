"""Frozen native TimesFM-3 head across depth: is h_l already in the pretrained readout's basis?

The third analysis alongside the learned probe and the representation geometry, and the one that
separates two things the probe alone conflates:

    learned Q=9 probe    is forecast information linearly DECODABLE from h_l ?
    frozen native head   is h_l already expressed in the COORDINATE SYSTEM the pretrained
                         forecasting head expects ?

For every representation point Emb, L1..L20 the pretrained ``model.output_head`` -- the exact
module from the checkpoint, unmodified, untrained, requires_grad=False -- is applied directly to
the last real context token h_{l,15} and scored on the paper7 test windows:

    h_{l,15} (N, 1280) --output_head--> (N, 576) --reshape--> (N, 64, 9)
        --revin(reverse, token-15 stats)--> clamp(+-value_clip) --stitch_patches--> + trend

There is NO fitting: no probe, no optimizer, no validation search, no weight decay, no adapter,
no cross-layer map. The ONLY variable is the depth l.

THE ENDPOINT IDENTITY (aborts the run if it fails): at L20 this path IS the native forecasting
pathway, so it must reproduce decode()'s own 9-quantile output. Two levels are checked --
    raw forecast   elementwise |recon - decode| <= recon_atol + recon_rtol*|decode| over the
                   full (N,64,9) tensor (atol 1e-5, rtol 2e-6); aborts iff the worst SCALED
                   error > 1. Per element, never vs the global mean -- fair to heavy tails;
    scalar loss    the L20 Q=9 pinball loss equals the committed native baseline,
                   because both are scored by the SAME probing.timesfm3_last_token_probes
                   .native_reference call.

The L20 identity is checked at THREE stages (raw head output, pre-trend forecast, final
inverse-transformed forecast) against decode()'s own output from the SAME forward pass, and the
validated ``verify_native_head`` is additionally run verbatim on those same tensors.

Scoring reuses the probe run's helpers unchanged, so the two curves are directly comparable:
``native_reference`` for the Q=9 / median pinball losses on the probe's normalized target axis,
``per_window_mase`` + ``mase_denominator`` for MASE in raw units, ``cluster_ci`` for the
series-level bootstrap.

This experiment does NOT define a tunnel entrance. The forecasting tunnel stays defined by the
trained probe's validation criterion; what is measured here is native-head compatibility
through depth, reported next to it.

NO FEATURE CACHE. The hidden states go straight from decode()'s forward pass into
``model.output_head`` on the same device, in the same dtype, inside the same ``torch.no_grad()``
region -- which is the only way the L20 identity can be EXACT. A cached round-trip re-applies the
head under a different matmul shape (and possibly a different device), and on TF32 hardware that
is not bit-identical; the cached implementation this replaces could not reproduce decode() for
exactly that reason. The learned Q=9 probes, CKA and effective rank remain cache-based and are
untouched.

This therefore needs a GPU (it runs the backbone), and only the SEVEN TEST splits.

Run:
    sbatch job_timesfm3_native_head.sh
    # or interactively, inside an salloc with --gres=gpu:1:
    python -m experiments.run_timesfm3_native_head_transfer \
        --probe-results $SCRATCH/timesfm3_last_token/results/timesfm3_last_token_paper7_q9/timesfm3_last_token_summary.json \
        --out-root $SCRATCH/timesfm3_geometry
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_timesfm3_last_token_probing import (KIND, PAPER7, SHORT,  # noqa: E402
                                                         assert_window_parity, chronos_reference,
                                                         suite_tags, windows_for)
from experiments.run_timesfm3_probing import (MASE_DEN_FLOOR, M_SEASON,  # noqa: E402
                                              cluster_ci, mase_denominator, per_window_mase)
from probing.timesfm3_geometry import (SELECTED_TOKEN_INDEX, git_commit,  # noqa: E402
                                       paper_out_default, window_identity_hash)

SLUG = {"m4_hourly": "m4", "monash_electricity_hourly": "electricity",
        "uber_tlc_hourly": "uber_tlc", "wind_farms_hourly": "wind_farms",
        "sg_carpark": "sg_carpark", "coastal_ts": "coastal_ts", "boom_hourly": "boom"}


# --------------------------------------------------------------------------- #
# the frozen head
# --------------------------------------------------------------------------- #
def head_checksum(model) -> dict:
    """A digest of every native-head parameter, so "frozen" is a MEASUREMENT, not a claim.

    Taken before and after the whole analysis and compared; any difference aborts.
    """
    import torch
    h = hashlib.sha256()
    shapes, n = {}, 0
    with torch.no_grad():
        for name, p in sorted(model.output_head.named_parameters()):
            a = p.detach().cpu().numpy()
            h.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
            shapes[name] = list(a.shape)
            n += a.size
            if p.requires_grad:
                raise RuntimeError(f"native head parameter {name} has requires_grad=True; this "
                                   "analysis evaluates a FROZEN head only")
    return {"sha256": h.hexdigest()[:32], "param_shapes": shapes, "n_params": int(n),
            "module": type(model.output_head).__name__,
            "in_features": int(model.output_head.in_features),
            "out_features": int(model.output_head.out_features),
            "requires_grad": False}


def _head_stages(model, states, run_mu, run_sd, geom, *, tfm_util):
    """Frozen head + decode()'s EXACT inverse path, with every intermediate stage exposed.

    Byte-for-byte the sequence inside ``probing.timesfm3_last_token.verify_native_head`` -- the
    one that reproduced decode() with ``max_abs == 0.0`` at extraction time. Two details are
    load-bearing and are the reason this is not re-derived here:

      * the head is applied to the FULL (b, 1, n_tokens, d) hidden state and the readout token
        is sliced AFTERWARDS, exactly as decode() does. Slicing first changes the matmul shape
        and therefore its accumulation order, which on a TF32-enabled GPU is NOT bit-identical;
      * the RevIN statistics are the model's own tensors for this batch, never a round-tripped
        copy.

    Returns the raw head output, the denormalized+clamped tensor, and the stitched (b, H, Q)
    forecast BEFORE the trend add-back (all on-device except ``rec``, which is float64 numpy).
    """
    import torch
    Q = model.num_quantiles
    P, opl = model.input_patch_len, model.output_patch_len
    tok = geom.selected_token_index
    b = states.shape[0]
    with torch.no_grad():
        raw = model.output_head(states)[:, :, [tok], :]            # (b, 1, 1, opl*Q)
        if raw.shape[-1] != opl * Q:
            raise RuntimeError(f"native head emitted {raw.shape[-1]} values, expected "
                               f"{opl}*{Q}={opl * Q}")
        den = tfm_util.revin(raw, run_mu[:, :, [tok]], run_sd[:, :, [tok]], reverse=True)
        clipped = torch.clamp(den, -model.value_clip, model.value_clip)
        view5 = clipped.reshape(b, 1, 1, opl, Q)[:, :, :, :geom.extract_len, :]
        rec = tfm_util.stitch_patches(view5, P)[:, :, :geom.H, :][:, 0]
    return {"raw": raw, "denormalized": den, "clipped": clipped,
            "rec": rec.float().cpu().numpy().astype(np.float64)}


def _staged_l20_identity(model, states, official, run_mu, run_sd, trend, geom, stages, *,
                         tfm_util, atol, rtol) -> dict:
    """The L20 identity, checked at THREE stages of the native inverse path.

    stage 3 (final)     g_native(h_L20,15) fully inverse-transformed  vs  decode()'s output
    stage 2 (pre-trend) the same, before the context trend is added back  vs  decode() - trend
    stage 1 (raw head)  the head output itself -- no external reference exists, so what is
                        recorded is its magnitude plus the SLICE-ORDER control below.

    Stages 2 and 3 differ only by the trend, so a stage-3 failure with a clean stage 2 localizes
    the fault to the trend add-back rather than the head/RevIN/clamp/stitch chain.

    SLICE-ORDER CONTROL: the same head applied to the pre-sliced (b, 1, 1, d) token instead of
    the full token sequence. Mathematically identical, numerically not on TF32 hardware. This is
    exactly what the retired cache-based implementation did, and the recorded delta is the
    measurement of why it could not reproduce decode().
    """
    import torch
    tok = geom.selected_token_index
    ref_final = official[:, 0].float().cpu().numpy().astype(np.float64)       # (b, H, Q)
    got_final = stages["rec"] + np.asarray(trend, np.float64)[:, :, None]
    got_pre = stages["rec"]
    ref_pre = ref_final - np.asarray(trend, np.float64)[:, :, None]

    # ONE comparator for every stage: the same elementwise gate the endpoint check uses, with
    # raising deferred so all three stages are measured before any of them aborts the run.
    Q = int(official.shape[-1])
    cmp = lambda got, ref, name: _endpoint_identity(got, ref, Q, atol, rtol, stage=name,
                                                    raise_on_fail=False)
    st3 = cmp(got_final, ref_final, "final_inverse_transformed")
    st2 = cmp(got_pre, ref_pre, "pre_trend")
    with torch.no_grad():
        sliced = model.output_head(states[:, :, [tok], :])
        slice_delta = float((sliced - stages["raw"]).abs().max().item())
    st1 = {"stage": "raw_output_head", "shape": list(stages["raw"].shape),
           "dtype": str(stages["raw"].dtype),
           "max_abs": float(stages["raw"].abs().max().item()),
           "slice_before_head_max_abs_delta": slice_delta,
           "note": "no external reference exists for the raw head output; the slice-order delta "
                   "is the control -- it is the numerical difference the retired cache-based "
                   "implementation introduced by applying the head to a pre-sliced token"}
    rep = {"stages": [st1, st2, st3], "atol": float(atol), "rtol": float(rtol),
           "max_scaled_error": st3["max_scaled_error"],
           "max_abs_error": st3["max_abs_error"], "worst_index": st3["worst_index"],
           "worst_ref": st3["worst_ref"], "worst_recon": st3["worst_recon"],
           "exact": st3["exact"], "n_windows": int(ref_final.shape[0]),
           "n_quantiles": Q}
    if st3["max_scaled_error"] > 1.0:
        raise RuntimeError(
            f"L20 NATIVE RECONSTRUCTION FAILED (in-memory, same forward pass as decode()): "
            f"stage-3 max scaled error {st3['max_scaled_error']:.3f} > 1 under |d| <= "
            f"atol({atol:g}) + rtol({rtol:g})*|ref|.\n"
            f"    stage 2 (pre-trend) max scaled error {st2['max_scaled_error']:.3f} "
            f"(max|d| {st2['max_abs_error']:.3e})\n"
            f"    stage 3 (final)     worst element {st3['worst_index']}: decode()="
            f"{st3['worst_ref']:.6g}, reconstructed={st3['worst_recon']:.6g}\n"
            f"    raw-head slice-order control delta {slice_delta:.3e}\n"
            "  These states came straight from decode()'s own forward pass with no cache in "
            "between, so this is NOT a serialization or device artifact: the head input, the "
            "inverse RevIN/clamp/stitch or the trend add-back is genuinely wrong, and NO layer's "
            "number can be trusted. If stage 2 is clean and stage 3 is not, the trend add-back "
            "is the fault.")
    return rep


def native_head_transfer_pass(model, X, *, geom, device, layers, batch_size, detrend,
                              allow_sorted_reference=False, bypass_sorting=True,
                              recon_atol=1e-5, recon_rtol=2e-6, progress=True) -> dict:
    """ONE decode() pass per batch; the frozen head applied to token 15 of EVERY layer IN MEMORY.

    NO FEATURE CACHE is read or written. The hidden states go straight from decode()'s forward
    pass into ``model.output_head`` on the same device, in the same dtype, within the same
    ``torch.no_grad()`` region. That is the only way the L20 identity can be exact -- a cached
    round-trip re-runs the head under a different kernel shape (and possibly a different device),
    which is precisely why the cached implementation could not reproduce decode().

    The learned-probe, CKA and effective-rank analyses are unaffected: they remain cache-based.
    """
    import torch

    from probing.timesfm3_last_token import (LAST_LAYER, context_detrend_params, context_trend,
                                             detrending, no_quantile_sorting,
                                             register_layer_hooks, verify_native_head)
    from timesfm3.torch import util as tfm_util

    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2 or X.shape[1] != geom.C:
        raise ValueError(f"X must be (n, {geom.C}) raw CONTEXTS (no future values), got {X.shape}")
    n, H, Q = len(X), geom.H, model.num_quantiles
    tok = geom.selected_token_index
    fc = {l: np.zeros((n, H, Q), np.float64) for l in layers}
    mu = np.full(n, np.nan, np.float64)
    sd = np.full(n, np.nan, np.float64)
    native = np.zeros((n, H, Q), np.float32)
    checks: dict = {}
    caps: dict = {}
    hs = register_layer_hooks(model, caps)
    try:
        with detrending(model, detrend), no_quantile_sorting(model, bypass_sorting) as srt:
            for s in range(0, n, batch_size):
                e = min(s + batch_size, n)
                tgt = torch.from_numpy(X[s:e]).to(device).unsqueeze(1)          # (b, 1, C)
                caps.clear()
                with torch.no_grad():
                    official, aux = model.decode(target=tgt, horizon=H,
                                                 return_aux_outputs=True, **srt.decode_kwargs)
                if official.shape[1:] != (1, H, Q):
                    raise RuntimeError(f"decode() returned {tuple(official.shape)}, expected "
                                       f"(b, 1, {H}, {Q})")
                run_mu, run_sd = aux["revin_stats"]
                mu[s:e] = run_mu[:, 0, tok].float().cpu().numpy()
                sd[s:e] = run_sd[:, 0, tok].float().cpu().numpy()
                native[s:e] = official[:, 0].float().cpu().numpy()
                a, b_, act = context_detrend_params(np.asarray(X[s:e], np.float64),
                                                    enabled=detrend)
                trend = context_trend(a, b_, act, geom.C,
                                      np.arange(geom.target_start, geom.target_end))   # (b, H)
                st_last = None
                for l in layers:
                    st = _head_stages(model, caps[f"L{l}"], run_mu, run_sd, geom,
                                      tfm_util=tfm_util)
                    fc[l][s:e] = st["rec"] + trend[:, :, None]
                    if l == LAST_LAYER:
                        st_last = st
                if "identity" not in checks:
                    if st_last is None:
                        raise RuntimeError(f"L{LAST_LAYER} must be among the evaluated layers "
                                           "-- it is the endpoint identity")
                    checks["sorting"] = srt.as_dict()
                    # the VALIDATED check, run verbatim on the same states/forward pass
                    checks["native_check"] = verify_native_head(
                        model, caps[f"L{LAST_LAYER}"], official, aux["revin_stats"], X[s:e],
                        geom, detrend=detrend,
                        allow_sorted_reference=allow_sorted_reference,
                        sorting=checks["sorting"])
                    # plus the staged localization, on the same tensors
                    checks["identity"] = _staged_l20_identity(
                        model, caps[f"L{LAST_LAYER}"], official, run_mu, run_sd, trend, geom,
                        st_last, tfm_util=tfm_util, atol=recon_atol, rtol=recon_rtol)
                if progress and (s // max(batch_size, 1)) % 5 == 0:
                    print(f"    [pass] {e}/{n} windows x {len(layers)} points "
                          f"(1 decode() + {len(layers)} frozen-head applications per batch)",
                          flush=True)
    finally:
        for h in hs:
            h.remove()
    if np.isnan(mu).any() or np.isnan(sd).any():
        raise RuntimeError("the forward pass left NaN RevIN statistics -- incomplete pass")
    return {"forecast": fc, "mu": mu, "sd": sd, "native": native,
            "native_check": checks.get("native_check"), "identity": checks.get("identity"),
            "sorting": checks.get("sorting"), "layers": list(layers)}


# --------------------------------------------------------------------------- #
# per-dataset
# --------------------------------------------------------------------------- #
def run_dataset(tag, args, geom, model, device, probe_entry) -> dict:
    import torch  # noqa: F401  (device tensors are created inside the helpers)
    from probing.timesfm3_last_token import (LAST_LAYER, LAYER_NAMES, NUM_LAYERS, NUM_QUANTILES,
                                             assert_target_roundtrip, build_last_token_targets,
                                             raw_future_from_arcsinh)
    from probing.timesfm3_last_token_probes import native_reference

    t0 = time.time()
    short, kind = SHORT.get(tag, tag), KIND.get(tag, "unclassified")
    print(f"\n{'=' * 82}\n[{tag}]  {short}  ({kind})\n{'=' * 82}")

    w = windows_for(tag, args.suite, args)
    ident = assert_window_parity(tag, w, chronos_reference(tag),
                                 strict=not args.allow_window_mismatch)
    meta = w["meta"]
    print(f"  windows: {ident['n_test_windows']} test  ({ident['n_test_series']} "
          f"{ident['cluster_unit']})   [Chronos-2 parity: {ident.get('chronos_parity')}]")

    layers = list(range(NUM_LAYERS)) if args.layers is None else sorted(set(args.layers))
    if LAST_LAYER not in layers:
        layers = sorted(layers + [LAST_LAYER])
        print(f"  [note] added L{LAST_LAYER} (the mandatory native endpoint identity)")

    # TEST SPLIT ONLY: the head is already pretrained, so no train/val representations are
    # needed. ONE in-memory pass -- decode() + the frozen head on every layer, no cache.
    print(f"  forward pass: {len(w['X_test'])} windows x {len(layers)} points, in memory "
          f"(no feature cache)", flush=True)
    te = native_head_transfer_pass(
        model, w["X_test"], geom=geom, device=device, layers=layers,
        batch_size=args.extract_batch_size, detrend=not args.no_detrend,
        allow_sorted_reference=args.allow_sorted_reference,
        bypass_sorting=not args.no_sorting_bypass,
        recon_atol=args.recon_atol, recon_rtol=args.recon_rtol)

    Zte = np.concatenate([w["X_test"],
                          raw_future_from_arcsinh(w["X_test"], w["Y_test_traj"],
                                                  meta["sigma_eps"])], axis=1)
    pte = build_last_token_targets(Zte, te["mu"], te["sd"], geom, detrend=not args.no_detrend)
    rt = assert_target_roundtrip(Zte, pte["targets"], pte["trend"], te["mu"], te["sd"], geom,
                                 pte["valid"], rtol=args.roundtrip_rtol)
    print(f"  target round-trip: {rt:.2e}  [< {args.roundtrip_rtol}]")

    # decode()'s own forecast, scored by the SAME function every layer will be scored by
    ref = native_reference(te["native"], te["mu"], te["sd"], pte["trend"], pte["targets"],
                           pte["valid"])
    rows = ref["rows"]
    y_raw = Zte[rows, geom.target_start:geom.target_end]
    den = mase_denominator(w["X_test"][rows])
    n_clamped = int((den < MASE_DEN_FLOOR).sum())
    den = np.maximum(den, MASE_DEN_FLOOR)
    ref_mase_pw = per_window_mase(y_raw, ref["median_raw"], den)

    l20_check = te["identity"]
    idn = l20_check["stages"]
    print(f"  L20 identity (same forward pass, no cache):")
    print(f"      stage 1 raw head      {idn[0]['shape']} {idn[0]['dtype']}  "
          f"max|h| {idn[0]['max_abs']:.4g}   slice-order control delta "
          f"{idn[0]['slice_before_head_max_abs_delta']:.3e}")
    print(f"      stage 2 pre-trend     max scaled {idn[1]['max_scaled_error']:.3g}  "
          f"max|d| {idn[1]['max_abs_error']:.3e}"
          f"{'   EXACT' if idn[1]['exact'] else ''}")
    print(f"      stage 3 final vs decode()  max scaled {idn[2]['max_scaled_error']:.3g}  "
          f"max|d| {idn[2]['max_abs_error']:.3e}"
          f"{'   EXACT' if idn[2]['exact'] else ''}")
    nc = te["native_check"]
    if nc is not None:
        print(f"      validated verify_native_head: relative {nc['relative']:.3e}, "
              f"max_abs {nc['max_abs']:.3e}, wrong-layout control "
              f"{nc['transposed_layout_relative']:.2e}")

    per_layer, q9_pw, med_pw, mase_pw = [], [], [], []
    for l in layers:
        r = native_reference(te["forecast"][l], te["mu"], te["sd"], pte["trend"],
                             pte["targets"], pte["valid"])
        if not np.array_equal(r["rows"], rows):
            raise RuntimeError("a layer was scored on different test windows than the reference")
        m_pw = per_window_mase(y_raw, r["median_raw"], den)
        q9_pw.append(r["q9_window"]); med_pw.append(r["median_window"]); mase_pw.append(m_pw)
        per_layer.append({
            "layer": l, "layer_name": LAYER_NAMES[l],
            "native_head_q9_pinball": r["q9_loss"],
            "native_head_median_pinball": r["median_loss"],
            "native_head_median_mae": 2.0 * r["median_loss"],   # rho_0.5(u) = 0.5|u|
            "native_head_mase": float(m_pw.mean()),
            "per_quantile_pinball": r["per_quantile"]})
        print(f"    {LAYER_NAMES[l]:>4}  Q9 {r['q9_loss']:.5f}   median {r['median_loss']:.5f}   "
              f"MASE {m_pw.mean():.4f}", flush=True)

    li = layers.index(LAST_LAYER)
    l20 = per_layer[li]["native_head_q9_pinball"]
    d_scalar = abs(l20 - ref["q9_loss"])
    rel_scalar = d_scalar / max(abs(ref["q9_loss"]), 1e-12)
    # RELATIVE, not absolute: decode()'s forecast is stored float32 in the feature cache while
    # this path reconstructs it in float64, so the two agree to ~1e-9 of the loss on large-N
    # datasets and ~2e-6 on small-N Coastal T-S. 1e-5 still rejects any STRUCTURAL divergence
    # (a wrong layer, the transposed layout, a missed inverse step) by many orders of magnitude.
    if rel_scalar > args.loss_identity_rtol:
        raise RuntimeError(
            f"L20 SCALAR IDENTITY FAILED for {tag}: the transferred-head Q=9 loss {l20:.9f} "
            f"differs from decode()'s own native baseline {ref['q9_loss']:.9f} by {d_scalar:.3e} "
            f"(relative {rel_scalar:.3e} > {args.loss_identity_rtol}). At L20 the two are the "
            "same computation; a difference this large means the head input, the inverse "
            "transform or the scoring diverged.")
    print(f"  L20 scalar identity: Q9 loss relative {rel_scalar:.2e} "
          f"(< {args.loss_identity_rtol}) vs decode()'s own baseline")

    for p in per_layer:                       # relative / absolute vs the L20 native-head row
        p["native_head_relative_to_L20"] = p["native_head_q9_pinball"] / l20 if l20 else float("nan")
        p["native_head_delta_vs_L20"] = p["native_head_q9_pinball"] - l20

    sid = np.asarray(w["series_test"], np.int64)[rows]
    boot = {"q9_loss": cluster_ci(np.stack(q9_pw), sid, args.boot_b, args.seed, li),
            "median_loss": cluster_ci(np.stack(med_pw), sid, args.boot_b, args.seed, li),
            "mase_context": cluster_ci(np.stack(mase_pw), sid, args.boot_b, args.seed, li)}

    combined = _combine_with_probe(per_layer, layers, probe_entry, LAST_LAYER, LAYER_NAMES)
    return {"tag": tag, "short": short, "slug": SLUG.get(tag, tag), "domain_status": kind,
            "kind": kind, "split": "test", "N": int(len(rows)),
            "window_identity": ident,
            "window_identity_hash": window_identity_hash(w["X_test"], w["series_test"]),
            "layers": layers, "layer_names": [LAYER_NAMES[l] for l in layers],
            "per_layer": per_layer, "bootstrap": boot,
            "native_decode_baseline": {"q9_loss": ref["q9_loss"],
                                       "median_loss": ref["median_loss"],
                                       "mase_context": float(ref_mase_pw.mean()),
                                       "per_quantile_loss": ref["per_quantile"]},
            "l20_identity": {**{k: v for k, v in l20_check.items()}, "q9_loss_abs_diff_vs_native_baseline": d_scalar,
                             "q9_loss_relative_diff_vs_native_baseline": rel_scalar,
                             "q9_loss_rtol": args.loss_identity_rtol,
                             "native_reference_storage_dtype": "float32 (the feature cache "
                                                               "stores decode()'s output as "
                                                               "float32; this path is float64)"},
            "combined_with_probe": combined,
            "target_roundtrip_relative_err": rt, "n_denominator_clamped": n_clamped,
            "m_season": M_SEASON,
            "feature_cache_used": False,
            "provenance": "representations taken in memory from decode()'s own forward pass; "
                          "no feature cache is read or written by this experiment",
            "quantile_sorting": te["sorting"],
            "validated_native_head_check": te["native_check"],
            "seconds": round(time.time() - t0, 1)}


def _endpoint_identity(raw, native, Q, atol, rtol, *, stage="final",
                       raise_on_fail=True) -> dict:
    """At L20 the frozen head must reproduce decode()'s own nine quantiles up to float32.

    Elementwise mixed absolute/relative tolerance over the FULL (N, H, Q) forecast tensor:
    each reconstruction error is compared to the magnitude of the element it belongs to,
    |recon - decode| <= atol + rtol*|decode|, never to the global mean magnitude (which is
    unfair to heavy-tailed forecasts). The gate fails iff the worst SCALED error exceeds 1.
    The comparison is done in float64. No dataset is special-cased.
    """
    ref = np.asarray(native, np.float64)
    got = np.asarray(raw, np.float64)
    if got.shape != ref.shape:
        raise RuntimeError(f"L20 reconstruction {got.shape} != decode() output {ref.shape}")
    d = np.abs(got - ref)
    scaled = d / (atol + rtol * np.abs(ref))          # elementwise: error vs its OWN magnitude
    w = tuple(int(i) for i in np.unravel_index(int(np.argmax(scaled)), scaled.shape))
    rep = {"max_abs_error": float(d.max()),
           "max_scaled_error": float(scaled.max()),
           "atol": float(atol), "rtol": float(rtol),
           "worst_index": list(w),
           "worst_ref": float(ref[w]), "worst_recon": float(got[w]),
           "worst_abs_error": float(d[w]),
           "per_quantile_max_abs": d.max(axis=(0, 1)).tolist(),
           "n_windows": int(ref.shape[0]), "n_quantiles": int(Q),
           "stage": stage, "exact": bool(d.max() == 0.0)}
    if rep["max_scaled_error"] > 1.0 and raise_on_fail:
        raise RuntimeError(
            f"L20 NATIVE RECONSTRUCTION FAILED: max scaled error {rep['max_scaled_error']:.3f} "
            f"> 1 under |d| <= atol({atol:g}) + rtol({rtol:g})*|ref|. Worst element {w}: "
            f"decode()={ref[w]:.6g}, reconstructed={got[w]:.6g}, |d|={d[w]:.3e} "
            f"(max|d| over the tensor {d.max():.3e}).\n"
            "  At L20 the frozen-head transfer IS the native forecasting pathway; a scaled error "
            "above 1 means the head input, the inverse RevIN/clamp/stitch or the trend add-back "
            "is wrong and NO layer's number can be trusted.")
    return rep


def _combine_with_probe(per_layer, layers, probe_entry, last_layer, names) -> dict:
    """The key per-layer diagnostic table: probe vs frozen head, and the alignment gap A_l.

    A_l = L_l(native head) - L_l(probe). Descriptive only -- it is neither normalized nor
    interpreted here.
    """
    head = {p["layer"]: p["native_head_q9_pinball"] for p in per_layer}
    out = {"available": probe_entry is not None, "rows": [],
           "definition": {"alignment_gap": "A_l = L_l^{native-head} - L_l^{probe}  (same test "
                                           "windows, same Q=9 objective, same normalized axis)",
                          "relative_to_L20": "L_l / L_L20, within each curve separately"}}
    if probe_entry is None:
        out["note"] = ("--probe-results not given: the frozen-head curve is saved alone and no "
                       "alignment gap is computed")
        return out
    p_layers = list(probe_entry["layers"])
    p_loss = {l: v for l, v in zip(p_layers, probe_entry["test_q9_loss"])}
    missing = [l for l in layers if l not in p_loss]
    if missing:
        raise RuntimeError(
            f"the probe summary has no test loss for representation points {missing}; the "
            "native-head and probe arrays must align by dataset AND layer. Re-run the probe "
            "over the full Emb..L20 curve, or restrict --layers to what it covers.")
    pl20, hl20 = p_loss[last_layer], head[last_layer]
    for l in layers:
        out["rows"].append({
            "layer": l, "layer_name": names[l],
            "probe_q9_loss": p_loss[l], "native_head_q9_loss": head[l],
            "final_probe_q9_loss": pl20, "native_L20_q9_loss": hl20,
            "probe_relative_to_L20": p_loss[l] / pl20 if pl20 else float("nan"),
            "native_head_relative_to_L20": head[l] / hl20 if hl20 else float("nan"),
            "alignment_gap": head[l] - p_loss[l]})
    tun = (probe_entry.get("tunnel_by_tolerance") or {}).get("0.05") or probe_entry.get("tunnel")
    out["tunnel_entrance_5pct"] = (tun or {}).get("layer")
    out["tunnel_entrance_5pct_name"] = (tun or {}).get("layer_name")
    if out["tunnel_entrance_5pct"] in head:
        t = out["tunnel_entrance_5pct"]
        out["at_tunnel_entrance"] = {
            "layer": t, "layer_name": names[t], "probe_q9_loss": p_loss[t],
            "native_head_q9_loss": head[t], "alignment_gap": head[t] - p_loss[t],
            "probe_relative_to_L20": p_loss[t] / pl20 if pl20 else float("nan"),
            "native_head_relative_to_L20": head[t] / hl20 if hl20 else float("nan")}
    return out


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _panel_grid(n):
    import math
    ncol = min(4, n)
    return math.ceil(n / ncol), ncol


def figure_head_vs_probe(recs, figdir) -> Path:
    """Learned Q=9 probe vs the frozen native head, per dataset, on one pinball-loss axis."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nrow, ncol = _panel_grid(len(recs))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.0 * nrow), squeeze=False)
    axes = [a for r in ax for a in r]
    for a, r in zip(axes, recs):
        lay = r["layers"]
        head = [p["native_head_q9_pinball"] for p in r["per_layer"]]
        a.plot(lay, head, "o-", ms=3.5, color="#d62728", label="frozen native head")
        rows = r["combined_with_probe"].get("rows") or []
        if rows:
            a.plot(lay, [x["probe_q9_loss"] for x in rows], "s-", ms=3.5, color="#1f77b4",
                   label="learned Q=9 linear probe")
        a.axhline(r["native_decode_baseline"]["q9_loss"], ls="--", lw=1.0, c="grey",
                  label="native TimesFM-3 forecast (decode)")
        t = r["combined_with_probe"].get("tunnel_entrance_5pct")
        if t is not None:
            a.axvline(t, ls=":", c="k", lw=1.1,
                      label=f"5% tunnel entrance = "
                            f"{r['combined_with_probe']['tunnel_entrance_5pct_name']}")
        finite = [v for v in head if np.isfinite(v) and v > 0]
        if finite and max(finite) / min(finite) > 20:
            a.set_yscale("log")
        a.set_title(f"{r['short']}  ({r['domain_status']}, test N={r['N']})", fontsize=10)
        a.set_xlabel("representation point")
        a.set_xticks(lay[::2])
        a.set_xticklabels(r["layer_names"][::2], rotation=45, fontsize=7)
        a.grid(alpha=0.3)
        a.legend(fontsize=6.5)
    for a in axes[len(recs):]:
        a.set_visible(False)
    for r in ax:
        r[0].set_ylabel("Q=9 pinball loss  (log scale where marked)")
    fig.suptitle(f"TimesFM-3: linear DECODABILITY vs native-head ALIGNMENT at h_l,"
                 f"{SELECTED_TOKEN_INDEX}   (paper7 test windows)", fontsize=11)
    fig.tight_layout()
    p = figdir / "native_head" / "native_head_vs_probe_all_datasets.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)
    return p


def figure_alignment_gap(recs, figdir) -> Path:
    """A_l = L_l(native head) - L_l(probe) vs representation point."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    usable = [r for r in recs if r["combined_with_probe"].get("rows")]
    if not usable:
        return None
    nrow, ncol = _panel_grid(len(usable))
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow), squeeze=False)
    axes = [a for r in ax for a in r]
    for a, r in zip(axes, usable):
        rows = r["combined_with_probe"]["rows"]
        gap = [x["alignment_gap"] for x in rows]
        col = "#9467bd" if r["domain_status"] == "PT-ID" else "#8c564b"
        a.plot(r["layers"], gap, "o-", ms=3.5, color=col)
        a.axhline(0.0, lw=0.9, c="k")
        t = r["combined_with_probe"].get("tunnel_entrance_5pct")
        if t is not None:
            a.axvline(t, ls=":", c="k", lw=1.1,
                      label=f"5% tunnel entrance = "
                            f"{r['combined_with_probe']['tunnel_entrance_5pct_name']}")
            a.legend(fontsize=6.5)
        finite = [abs(v) for v in gap if np.isfinite(v) and v != 0]
        if finite and max(finite) / min(finite) > 50:
            a.set_yscale("symlog", linthresh=max(1e-4, min(finite)))
        a.set_title(f"{r['short']}  ({r['domain_status']})", fontsize=10)
        a.set_xlabel("representation point")
        a.set_xticks(r["layers"][::2])
        a.set_xticklabels(r["layer_names"][::2], rotation=45, fontsize=7)
        a.grid(alpha=0.3)
    for a in axes[len(usable):]:
        a.set_visible(False)
    for r in ax:
        r[0].set_ylabel(r"$A_\ell$ = native head $-$ probe   (Q=9 pinball)")
    fig.suptitle(r"TimesFM-3 alignment gap $A_\ell = \mathcal{L}^{\rm native-head}_\ell - "
                 r"\mathcal{L}^{\rm probe}_\ell$   (descriptive diagnostic)", fontsize=11)
    fig.tight_layout()
    p = figdir / "native_head" / "native_head_alignment_gap_all_datasets.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
def write_outputs(recs, meta, out_root, paper_out):
    nums = out_root / "numerical_results"
    nums.mkdir(parents=True, exist_ok=True)
    for r in recs:
        (nums / f"native_head_transfer__{r['tag']}.json").write_text(
            json.dumps({**meta, **r}, indent=2, default=str))
    paper_out.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "datasets": {r["tag"]: r for r in recs}}
    sp = paper_out / "native_head_transfer_summary.json"
    sp.write_text(json.dumps(payload, indent=2, default=str))
    head = ["dataset", "display_name", "domain_status", "layer", "layer_name", "probe_q9_loss",
            "native_head_q9_loss", "probe_relative_to_L20", "native_head_relative_to_L20",
            "alignment_gap", "native_head_mase", "native_head_median_pinball"]
    with open(paper_out / "native_head_transfer_table.csv", "w", newline="") as fh:
        wr = csv.writer(fh); wr.writerow(head)
        for r in recs:
            rows = {x["layer"]: x for x in (r["combined_with_probe"].get("rows") or [])}
            for p in r["per_layer"]:
                c = rows.get(p["layer"], {})
                wr.writerow([r["tag"], r["short"], r["domain_status"], p["layer"],
                             p["layer_name"],
                             f"{c['probe_q9_loss']:.6f}" if c else "",
                             f"{p['native_head_q9_pinball']:.6f}",
                             f"{c['probe_relative_to_L20']:.6f}" if c else "",
                             f"{p['native_head_relative_to_L20']:.6f}",
                             f"{c['alignment_gap']:.6f}" if c else "",
                             f"{p['native_head_mase']:.6f}",
                             f"{p['native_head_median_pinball']:.6f}"])
    return sp


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Frozen pretrained TimesFM-3 output head applied to every representation "
                    "point (no training of any kind).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("model / environment")
    g.add_argument("--checkpoint", default=os.environ.get("TIMESFM3_CHECKPOINT",
                                                          "google/timesfm-3.0-pytorch"))
    g.add_argument("--hf-home", default=None)
    g.add_argument("--device", default=os.environ.get("TFM3_DEVICE", None))
    g.add_argument("--probe-results", default=None,
                   help="the paper7 Q=9 probe summary. REQUIRED for the alignment gap and the "
                        "comparison figures; omit to save the frozen-head curve alone")

    g = p.add_argument_group("data")
    g.add_argument("--suite", default="paper7")
    g.add_argument("--datasets", nargs="+", default=None)
    g.add_argument("--layers", type=int, nargs="+", default=None)
    g.add_argument("--allow-window-mismatch", action="store_true")
    g.add_argument("--context-len", type=int, default=512)
    g.add_argument("--horizon", type=int, default=64)
    g.add_argument("--stride", type=int, default=64)
    g.add_argument("--no-detrend", action="store_true")
    g.add_argument("--extract-batch-size", type=int, default=64,
                   help="windows per decode() forward pass; each batch also runs the frozen head "
                        "once per representation point, in memory")
    g.add_argument("--allow-sorted-reference", action="store_true")
    g.add_argument("--no-sorting-bypass", action="store_true")

    g = p.add_argument_group("checks")
    g.add_argument("--recon-rtol", type=float, default=2e-6,
                   help="elementwise RELATIVE tolerance for the L20 all-quantile native "
                        "reconstruction: |recon - decode| <= recon_atol + recon_rtol*|decode|, "
                        "PER ELEMENT (never vs the global mean). ~16 float32 ULP; the observed "
                        "worst-element error is 2-7 ULP across all seven datasets. Fails iff the "
                        "worst SCALED error exceeds 1")
    g.add_argument("--recon-atol", type=float, default=1e-5,
                   help="elementwise ABSOLUTE tolerance floor for the same check, covering "
                        "near-zero forecast elements where the relative term vanishes")
    g.add_argument("--loss-identity-rtol", type=float, default=1e-5,
                   help="abort threshold for the RELATIVE difference between the L20 "
                        "transferred-head Q=9 loss and decode()'s own native baseline. Relative "
                        "because the cache stores decode()'s forecast in float32 while this path "
                        "reconstructs it in float64: large-N datasets agree to ~1e-9 of the loss, "
                        "small-N Coastal T-S to ~2e-6, so 1e-5 rejects any structural divergence")
    g.add_argument("--roundtrip-rtol", type=float, default=1e-4)
    g.add_argument("--boot-b", type=int, default=5000)
    g.add_argument("--seed", type=int, default=0)

    g = p.add_argument_group("outputs")
    g.add_argument("--out-root", "--work-dir", dest="out_root",
                   default=os.environ.get("TFM3_GEOM_OUT_ROOT", None),
                   help="HEAVY outputs ($SCRATCH) [default: <paper-out>/_work]")
    g.add_argument("--paper-out", default=None,
                   help="FINAL paper outputs (repo-local) [default: "
                        "<repo>/results/timesfm3_representation_geometry]")
    g.add_argument("--no-figures", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser())
    from probing.timesfm3_last_token import (LAYER_NAMES, NUM_LAYERS, LastTokenGeometry,
                                             assert_backbone_frozen, assert_native_geometry,
                                             assert_native_quantiles, get_model, timesfm_version)

    geom = LastTokenGeometry(args.context_len, args.horizon)
    roster = suite_tags(args.suite)
    tags = args.datasets or roster
    unknown = [t for t in tags if t not in roster]
    if unknown:
        raise SystemExit(f"unknown dataset tag(s) {unknown}; roster = {roster}")
    paper_out = Path(args.paper_out).expanduser() if args.paper_out else paper_out_default()
    out_root = Path(args.out_root).expanduser() if args.out_root else paper_out / "_work"
    figdir = paper_out / "figures"

    probes = {}
    if args.probe_results:
        pp = Path(args.probe_results).expanduser()
        if not pp.exists():
            raise FileNotFoundError(f"--probe-results {pp} does not exist")
        probes = json.loads(pp.read_text()).get("datasets", {})

    model, device = get_model(args.checkpoint, args.device)
    n_par = assert_backbone_frozen(model)
    qinfo = assert_native_quantiles(model)
    ginfo = assert_native_geometry(model, geom)
    ck0 = head_checksum(model)

    print("TimesFM-3 FROZEN NATIVE-HEAD TRANSFER across depth")
    print(f"  checkpoint : {args.checkpoint}  (timesfm {timesfm_version()}, device {device})")
    print(f"  backbone   : FROZEN, {n_par} parameter tensors")
    print(f"  native head: model.output_head = {ck0['module']}({ck0['in_features']}, "
          f"{ck0['out_features']}), {ck0['n_params']} params, requires_grad=False, "
          f"sha256[:32] {ck0['sha256']}")
    print(f"  quantiles  : {[round(x, 4) for x in qinfo['quantiles']]} (median index "
          f"{qinfo['median_index']}), NOT sorted before the headline loss")
    print(f"  readout    : h_l,{geom.selected_token_index} for {NUM_LAYERS} points "
          f"{LAYER_NAMES[0]}..{LAYER_NAMES[-1]}   (decode()'s own index "
          f"{ginfo['native_forecast_indices']})")
    print("  NO fitting: no probe, no optimizer, no validation search, no adapter")
    print("  split      : TEST only (the head is pretrained; no train/val representations)")
    print("  NO feature cache: representations go straight from decode()'s forward pass "
          "into the frozen head, in memory")

    meta = {"analysis": "timesfm3_native_head_transfer", "model": "timesfm-3.0",
            "checkpoint": args.checkpoint, "timesfm_version": timesfm_version(),
            "device": device, "backbone_frozen": True, "n_backbone_param_tensors": n_par,
            "native_head": ck0, "native_head_checksum_before": ck0["sha256"],
            "native_quantiles": qinfo, "native_geometry": ginfo, "geometry": geom.as_dict(),
            "representation": f"last real context token h_l,{SELECTED_TOKEN_INDEX} (N, 1280)",
            "representation_points": LAYER_NAMES, "split": "test",
            "trained_anything": False,
            "tunnel_definition": "unchanged -- the forecasting tunnel is defined by the TRAINED "
                                 "probe's validation criterion; this experiment defines no "
                                 "entrance of its own",
            "probe_results": (str(args.probe_results) if args.probe_results else None),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "git_commit": git_commit(),
            "computed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}

    t0 = time.time()
    recs = [run_dataset(t, args, geom, model, device, probes.get(t)) for t in tags]
    order = {"PT-ID": 0, "PT-OOD": 1}
    recs.sort(key=lambda r: (order.get(r["domain_status"], 2),
                             list(PAPER7).index(r["tag"]) if r["tag"] in PAPER7 else 99))

    ck1 = head_checksum(model)
    if ck1["sha256"] != ck0["sha256"]:
        raise RuntimeError(f"the native head CHANGED during the analysis: {ck0['sha256']} -> "
                           f"{ck1['sha256']}. It must be evaluated frozen and untouched.")
    meta["native_head_checksum_after"] = ck1["sha256"]
    meta["native_head_unchanged"] = True
    meta["seconds"] = round(time.time() - t0, 1)
    print(f"\n  native head checksum before == after: {ck1['sha256']}  (unchanged)")

    sp = write_outputs(recs, meta, out_root, paper_out)
    figs = []
    if not args.no_figures:
        figs.append(figure_head_vs_probe(recs, figdir))
        g = figure_alignment_gap(recs, figdir)
        if g:
            figs.append(g)

    print(f"\n{'=' * 104}\nSUMMARY — frozen native head vs learned probe (Q=9 pinball, paper7 "
          f"test)\n{'=' * 104}")
    print(f"{'dataset':<16}{'kind':<8}{'N':>5}{'tun5%':>7}{'head@Emb':>11}{'head@tun':>11}"
          f"{'head@L20':>11}{'probe@tun':>11}{'A(tun)':>10}{'L20 rel':>9}")
    last = None
    for r in recs:
        c = r["combined_with_probe"]
        at = c.get("at_tunnel_entrance") or {}
        head = {p["layer"]: p["native_head_q9_pinball"] for p in r["per_layer"]}
        if last is not None and r["domain_status"] != last:
            print("-" * 104)
        last = r["domain_status"]
        fmt = lambda v, w=11, p=4: (f"{v:>{w}.{p}f}" if isinstance(v, (int, float)) else f"{'-':>{w}}")
        print(f"{r['short']:<16}{r['domain_status']:<8}{r['N']:>5}"
              f"{(c.get('tunnel_entrance_5pct_name') or '-'):>7}"
              f"{fmt(head[r['layers'][0]])}{fmt(at.get('native_head_q9_loss'))}"
              f"{fmt(head[max(head)])}{fmt(at.get('probe_q9_loss'))}"
              f"{fmt(at.get('alignment_gap'), 10)}"
              f"{r['l20_identity']['max_scaled_error']:>9.3f}")

    print(f"\nNumerical/intermediate outputs:\n  {out_root}")
    print(f"\nFinal paper outputs:\n  {paper_out}")
    print(f"  {sp.name}")
    print(f"  native_head_transfer_table.csv")
    for f in figs:
        print(f"  figures/{f.relative_to(figdir)}   (+ .pdf)")
    print(f"\n[{meta['seconds']}s]")
    return {"meta": meta, "datasets": recs, "figures": [str(f) for f in figs]}


if __name__ == "__main__":
    main()
