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

Scoring reuses the probe run's helpers unchanged, so the two curves are directly comparable:
``native_reference`` for the Q=9 / median pinball losses on the probe's normalized target axis,
``per_window_mase`` + ``mase_denominator`` for MASE in raw units, ``cluster_ci`` for the
series-level bootstrap.

This experiment does NOT define a tunnel entrance. The forecasting tunnel stays defined by the
trained probe's validation criterion; what is measured here is native-head compatibility
through depth, reported next to it.

Run (compute node -- this loads the checkpoint for its output head; no backbone forward pass is
needed when the feature cache is warm):
    python -m experiments.run_timesfm3_native_head_transfer \
        --cache-dir $SCRATCH/timesfm3_last_token/features_cache \
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


def native_head_predict_raw(model, h, mu, sd, trend, geom, *, device="cpu", batch=512):
    """h_{l,15} -> the pretrained head -> RAW-units (n, H, Q) forecast.

    Byte-for-byte the inverse path of ``timesfm3_last_token.verify_native_head``: reverse RevIN
    with the SAME token-15 statistics of that window, the native value clamp, decode()'s patch
    stitching, then the context-fit linear trend added back. Nothing is refitted per layer --
    the preprocessing belongs to the WINDOW, not to the representation point.
    """
    import torch
    from timesfm3.torch import util as tfm_util
    P, Q = model.input_patch_len, model.num_quantiles
    opl = model.output_patch_len
    n = len(h)
    out = np.empty((n, geom.H, Q), np.float64)
    with torch.no_grad():
        for s in range(0, n, batch):
            e = min(s + batch, n)
            b = e - s
            x = torch.as_tensor(np.asarray(h[s:e], dtype=np.float32), device=device)
            if x.shape != (b, model.output_head.in_features):
                raise RuntimeError(f"native-head input is {tuple(x.shape)}, expected "
                                   f"({b}, {model.output_head.in_features})")
            raw = model.output_head(x)                                  # (b, opl*Q)
            if raw.shape != (b, opl * Q):
                raise RuntimeError(f"native head emitted {tuple(raw.shape)}, expected "
                                   f"({b}, {opl}*{Q}={opl * Q})")
            raw = raw.reshape(b, 1, 1, opl * Q)                          # decode()'s own layout
            m = torch.as_tensor(np.asarray(mu[s:e], np.float32), device=device).reshape(b, 1, 1)
            v = torch.as_tensor(np.asarray(sd[s:e], np.float32), device=device).reshape(b, 1, 1)
            den = tfm_util.revin(raw, m, v, reverse=True)
            den = torch.clamp(den, -model.value_clip, model.value_clip)
            # horizon-major (output_patch_len, Q): flat index t*Q + q -- the layout proven by
            # verify_native_head (which also shows the transposed one does NOT reproduce decode())
            view5 = den.reshape(b, 1, 1, opl, Q)[:, :, :, :geom.extract_len, :]
            rec = tfm_util.stitch_patches(view5, P)[:, :, :geom.H, :][:, 0]
            out[s:e] = rec.float().cpu().numpy().astype(np.float64)
    return out + np.asarray(trend, np.float64)[:, :, None]


# --------------------------------------------------------------------------- #
# per-dataset
# --------------------------------------------------------------------------- #
def run_dataset(tag, args, geom, model, device, probe_entry) -> dict:
    import torch  # noqa: F401  (device tensors are created inside the helpers)
    from probing.timesfm3_last_token import (LAST_LAYER, LAYER_NAMES, NUM_LAYERS, NUM_QUANTILES,
                                             assert_target_roundtrip, build_last_token_targets,
                                             cached_last_token_features, raw_future_from_arcsinh)
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

    # test split ONLY: the head is already pretrained, so no training data is required.
    # Pre-flight the cache first: this job normally runs on a GPU-less node, where a MISS would
    # silently start a full CPU backbone pass over every window. Abort instead, unless asked.
    if not args.allow_extraction:
        from probing.timesfm3_last_token import cache_metadata, cache_root, read_cache
        _m = cache_metadata(tag, "test", geom, checkpoint=args.checkpoint,
                            detrend=not args.no_detrend, layers=layers, seed=args.seed,
                            feature_dtype=np.dtype(args.feature_dtype), suite=args.suite,
                            n_windows=len(w["X_test"]))
        _r = cache_root(args.cache_dir, tag, "test", geom, not args.no_detrend, args.suite)
        if read_cache(_r, _m, np.asarray(w["X_test"], np.float32), layers) is None:
            raise SystemExit(
                f"no cached test features for {tag} at {_r}.\n"
                "  This analysis is meant to run on the WARM cache the paper7 Q=9 probe run "
                "wrote, with no backbone forward pass (the job requests no GPU). Extracting here "
                "would run the full backbone on CPU over every window.\n"
                "  Point --cache-dir at that cache, or pass --allow-extraction to accept the "
                "cost deliberately (then request a GPU).")
    te = cached_last_token_features(
        tag, "test", w["X_test"], geom=geom, model=model, device=device,
        batch_size=args.extract_batch_size, layers=layers,
        feature_dtype=np.dtype(args.feature_dtype), detrend=not args.no_detrend,
        cache_dir=args.cache_dir, suite=args.suite, checkpoint=args.checkpoint, seed=args.seed,
        force=False, allow_sorted_reference=args.allow_sorted_reference,
        bypass_sorting=not args.no_sorting_bypass)
    print(f"  features: {len(layers)} points, each "
          f"{tuple(np.shape(te['feats'][layers[-1]]))} at token {geom.selected_token_index}"
          f"   [cache {'HIT -- no backbone pass' if te.get('cache_hit') else 'MISS -- extracted'}]")

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

    per_layer, q9_pw, med_pw, mase_pw, l20_check = [], [], [], [], None
    for l in layers:
        raw = native_head_predict_raw(model, te["feats"][l], te["mu"], te["sd"], pte["trend"],
                                      geom, device=device, batch=args.head_batch_size)
        if l == LAST_LAYER:
            l20_check = _endpoint_identity(raw, te["native"], NUM_QUANTILES,
                                           args.recon_atol, args.recon_rtol)
        r = native_reference(raw, te["mu"], te["sd"], pte["trend"], pte["targets"], pte["valid"])
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
    print(f"  L20 endpoint: max scaled recon error {l20_check['max_scaled_error']:.3f} (< 1) | "
          f"max|d| {l20_check['max_abs_error']:.2e} at {l20_check['worst_index']} "
          f"(decode {l20_check['worst_ref']:.4g}, recon {l20_check['worst_recon']:.4g}) | "
          f"Q9 loss identity relative {rel_scalar:.2e} (< {args.loss_identity_rtol})")

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
            "l20_identity": {**l20_check, "q9_loss_abs_diff_vs_native_baseline": d_scalar,
                             "q9_loss_relative_diff_vs_native_baseline": rel_scalar,
                             "q9_loss_rtol": args.loss_identity_rtol,
                             "native_reference_storage_dtype": "float32 (the feature cache "
                                                               "stores decode()'s output as "
                                                               "float32; this path is float64)"},
            "combined_with_probe": combined,
            "target_roundtrip_relative_err": rt, "n_denominator_clamped": n_clamped,
            "m_season": M_SEASON, "cache_hit": bool(te.get("cache_hit")),
            "seconds": round(time.time() - t0, 1)}


def _endpoint_identity(raw, native, Q, atol, rtol) -> dict:
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
           "n_windows": int(ref.shape[0]), "n_quantiles": int(Q)}
    if rep["max_scaled_error"] > 1.0:
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
    g.add_argument("--cache-dir", default=os.environ.get("TFM3_LT_CACHE_DIR", None),
                   help="last-token feature cache [default: <repo>/features_cache]. A warm cache "
                        "means NO backbone forward pass")
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
    g.add_argument("--feature-dtype", default="float32", choices=["float32", "float16"])
    g.add_argument("--no-detrend", action="store_true")
    g.add_argument("--extract-batch-size", type=int, default=64)
    g.add_argument("--allow-extraction", action="store_true",
                   help="permit a feature-cache MISS to trigger a backbone forward pass. Off by "
                        "default: this analysis is designed to run GPU-less on the warm cache, "
                        "where extracting would silently cost a full CPU backbone pass")
    g.add_argument("--head-batch-size", type=int, default=512)
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
    args.cache_dir = (Path(args.cache_dir).expanduser() if args.cache_dir
                      else REPO_ROOT / "features_cache")
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
    print(f"  split      : test only (the head is pretrained; no training data is needed)")
    print(f"  cache      : {args.cache_dir}")

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
