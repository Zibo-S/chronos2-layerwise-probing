"""Native-head readout of intermediate layers (authorized follow-up).

Applies Chronos-2's OWN frozen quantile head to every layer's states at the FORECAST-SLOT
positions — no fitting, no calibration, nothing trained. Tests whether probe-flatness is a
readout artifact.

Interface credit: extends the K-slot encode interface of
``probing.extraction.extract_kout_features`` (model.encode/forward with
num_output_patches=K, position layout [content(ncp) | REG | K slots], fslot pooler
``hs[:, -K:]``). This module adds the two taps that function lacks (Embed at the
forecast slots = input_patch_embedding's future-patch call; L12_postln = final_layer_norm
output) plus per-window loc_scale capture, and runs the frozen head.

Pipeline per layer k in {Embed, L1..L12, L12_postln}:
    fslot states (n, K, 768) -> frozen encoder.final_layer_norm (SKIPPED for L12_postln,
    already normed) -> frozen output_patch_embedding -> rearrange 'b n (q p) -> b q (n p)'
    -> trim to H -> q9 pinball loss (arcsinh-normalized target space, 9 deciles) and
    MASE (real units, median forecast un-scaled via InstanceNorm.inverse with the SAME
    window's loc_scale; seasonal-naive m=24 denominator from the context window).

SELF-CONSISTENT GATE (run before any other layer): the L12_postln pathway must reproduce
the model's OWN quantile_preds from the same forward (Chronos2Output.quantile_preds,
trimmed to H): relative Frobenius delta and q9-loss relative delta both < 0.1%.

Windows: ONE loss-bearing window per series — context = seg[-(C+H):-H] (C=512), future =
seg[-H:] (H=64), where seg = the longest contiguous finite segment (Wind-Farms rule; a
no-op for NaN-free series). <=200 series, seed 0 (same sampler as repr-metrics).

Outputs: results/repr_metrics/native_head/{run}/native_head.json + fig_native_head_by_layer.png
Run:  python -m probing.repr_metrics_nativehead [--datasets electricity m4 ...]
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.config import OUT_DIR, OUTPUT_PATCH_SIZE
from probing.repr_metrics import RM_DIR, SEED, C
from probing.repr_metrics_overnight import longest_finite_segment

NH_DIR = RM_DIR / "native_head"
H = 64
K = math.ceil(H / OUTPUT_PATCH_SIZE)                 # = 4, Chronos-2's own slot rule
AXIS14 = ["Embed"] + [f"L{i}" for i in range(1, 13)] + ["L12_postln"]
DECILES = [round(0.1 * q, 1) for q in range(1, 10)]  # q9 = the nine deciles
GATE_REL_TOL = 1e-3                                  # 0.1%

# ID runs -> (dataset_set, tag); OOD runs -> loader tag via id_data.load_ood_target_series
ID_RUNS = {"electricity": ("phase0_trio", "monash_electricity_hourly"),
           "m4":          ("phase0_trio", "m4_hourly")}
OOD_RUNS = ("sg_carpark", "coastal_ts", "boom_hourly")


# --------------------------------------------------------------------------- #
# windows
# --------------------------------------------------------------------------- #

def build_windows(series: list[np.ndarray], max_series: int = 200, seed: int = SEED):
    """One loss-bearing window per series: ctx = seg[-(C+H):-H], fut = seg[-H:]."""
    ctxs, futs = [], []
    for s in series:
        seg = longest_finite_segment(s)
        if len(seg) < C + H:
            continue
        ctx, fut = seg[-(C + H):-H], seg[-H:]
        if not (np.all(np.isfinite(ctx)) and np.all(np.isfinite(fut))) or ctx.std() < 1e-12:
            continue
        ctxs.append(ctx.astype(np.float32)); futs.append(fut.astype(np.float32))
    if len(ctxs) > max_series:
        rng = np.random.default_rng(seed)
        sel = np.sort(rng.choice(len(ctxs), size=max_series, replace=False))
        ctxs = [ctxs[i] for i in sel]; futs = [futs[i] for i in sel]
    return np.stack(ctxs), np.stack(futs)


def load_series_for(run: str) -> list[np.ndarray]:
    if run in ID_RUNS:
        from probing import config
        dset, tag = ID_RUNS[run]
        config.set_dataset_set(dset)
        from probing.id_data import load_seen_series
        return load_seen_series(tag)
    os.environ.setdefault("OOD_TARGET_ROOT",
                          str((OUT_DIR / "distance" / "raw_cache").resolve()))
    if run == "boom_hourly":
        # locally staged shards cover the FIRST 200 manifest entries by file order — the
        # same restriction the distance ladder used (probing.distance_ladder.BOOM_FIRST_N);
        # reuse id_data's verbatim per-variate reader on exactly those entries.
        import json as _json
        from probing import id_data
        sel = _json.load(open(id_data.BOOM_MANIFEST))["selected"][:200]
        root = Path(os.environ["OOD_TARGET_ROOT"]) / "boom_hourly"
        return [id_data._boom_read_variate(root / e["query_dir"] / "data-00000-of-00001.arrow",
                                           e["variate_index"]) for e in sel]
    from probing.id_data import load_ood_target_series
    return [np.asarray(s, np.float64) for s in load_ood_target_series(run)["series"]]


# --------------------------------------------------------------------------- #
# one-forward extraction: 14 fslot taps + loc_scale + native quantile_preds
# --------------------------------------------------------------------------- #

def extract_fslot(run: str, ctxs: np.ndarray, batch_size: int = 64):
    """Cache: {fs_<tap>: (n,K,768) fp32}, loc (n,1), scale (n,1), native_q (n,Q,H)."""
    cache = NH_DIR / run / "cache" / f"fslot14_K{K}_H{H}.npz"
    if cache.exists():
        d = np.load(cache, allow_pickle=True)
        print(f"  [cache HIT]  {cache.relative_to(OUT_DIR)}")
        return d

    import torch
    from probing.extraction import get_pipeline
    pipe, _ = get_pipeline()
    model = pipe.model
    device = next(model.parameters()).device
    ncp = math.ceil(C / model.chronos_config.input_patch_size)
    P_expected = ncp + 1 + K                          # [content | REG | K slots]
    n = len(ctxs)
    print(f"  [extract] {run}: n={n} windows, K={K}, one forward/batch, device={device}")

    taps = {ln: [] for ln in AXIS14}
    locs, scales, natives = [], [], []

    for b0 in range(0, n, batch_size):
        batch = torch.from_numpy(np.ascontiguousarray(ctxs[b0:b0 + batch_size])).to(
            device=device, dtype=torch.float32)
        emb_hits, pln_hits, in_hits = [], [], []
        blk_hits = {i: [] for i in range(12)}

        def emb_hook(_m, _i, out):
            if out.shape[1] == K:                     # future-patch call ONLY (context call = ncp)
                emb_hits.append(out.detach())

        def pln_hook(_m, _i, out):
            pln_hits.append(out.detach())

        def in_hook(_m, _i, out):                     # InstanceNorm returns (x, (loc, scale))
            in_hits.append((out[1][0].detach(), out[1][1].detach()))

        def mk(i):
            def h(_m, _i, out):
                hs = out.hidden_states if getattr(out, "hidden_states", None) is not None else out[0]
                blk_hits[i].append(hs.detach())
            return h

        handles = [model.input_patch_embedding.register_forward_hook(emb_hook),
                   model.encoder.final_layer_norm.register_forward_hook(pln_hook),
                   model.instance_norm.register_forward_hook(in_hook)]
        handles += [b.register_forward_hook(mk(i)) for i, b in enumerate(model.encoder.block)]
        try:
            with torch.no_grad():
                out = model(context=batch, num_output_patches=K)   # NATIVE forward incl. head
        finally:
            for h in handles:
                h.remove()

        nq = out.quantile_preds.detach()              # (b, Q, K*16) — the native reference
        natives.append(nq[:, :, :H].cpu().numpy())
        loc, scale = in_hits[0]                       # first instance_norm call = context loc/scale
        locs.append(loc.cpu().numpy()); scales.append(scale.cpu().numpy())

        emb = torch.cat(emb_hits, 0)
        assert emb.shape[1] == K, f"Embed future-call shape {tuple(emb.shape)}"
        taps["Embed"].append(emb.cpu().numpy())
        pln = torch.cat(pln_hits, 0)
        assert pln.shape[1] == P_expected
        taps["L12_postln"].append(pln[:, -K:, :].cpu().numpy())
        for i in range(12):
            hs = torch.cat(blk_hits[i], 0)
            assert hs.shape[1] == P_expected
            taps[f"L{i + 1}"].append(hs[:, -K:, :].cpu().numpy())

    cache.parent.mkdir(parents=True, exist_ok=True)
    save = {f"fs_{ln}": np.concatenate(taps[ln], 0).astype(np.float32) for ln in AXIS14}
    save["loc"] = np.concatenate(locs, 0); save["scale"] = np.concatenate(scales, 0)
    save["native_q"] = np.concatenate(natives, 0)
    np.savez_compressed(cache, **save)
    print(f"  [saved] {cache.relative_to(OUT_DIR)}  ({cache.stat().st_size/1e6:.0f} MB)")
    return np.load(cache, allow_pickle=True)


# --------------------------------------------------------------------------- #
# frozen readout + metrics
# --------------------------------------------------------------------------- #

def head_forward(states: np.ndarray, loc: np.ndarray, scale: np.ndarray,
                 apply_ln: bool) -> np.ndarray:
    """(n,K,768) fslot states -> [LN] -> frozen head -> UN-SCALE -> (n, Q, H) REAL-UNIT preds.

    The un-scale step replicates the model's own forward verbatim (Chronos2Model.forward,
    "# Unscale predictions": rearrange b q h -> b (q h), instance_norm.inverse(loc_scale),
    rearrange back) — it is part of the native head pathway, not an adapter."""
    import torch
    from einops import rearrange
    from probing.extraction import get_pipeline
    model = get_pipeline()[0].model
    device = next(model.parameters()).device
    with torch.no_grad():
        x = torch.from_numpy(np.asarray(states, np.float32)).to(device)
        if apply_ln:
            x = model.encoder.final_layer_norm(x)
        qp = model.output_patch_embedding(x)          # (n, K, Q*16)
        qp = rearrange(qp, "b n (q p) -> b q (n p)",
                       n=K, q=model.num_quantiles, p=model.chronos_config.output_patch_size)
        ls = (torch.from_numpy(np.asarray(loc, np.float32)).to(device),
              torch.from_numpy(np.asarray(scale, np.float32)).to(device))
        flat = rearrange(qp, "b q h -> b (q h)")
        flat = model.instance_norm.inverse(flat, ls)  # model's own final step (real units)
        qp = rearrange(flat, "b (q h) -> b q h", q=model.num_quantiles)
        return qp[:, :, :H].cpu().numpy()             # trim to horizon, native rule


def metrics_q9_mase(pred_q_real: np.ndarray, fut: np.ndarray, loc: np.ndarray,
                    scale: np.ndarray, qlevels: list[float]) -> tuple[float, np.ndarray]:
    """q9 pinball in the model's arcsinh-normalized space (both sides re-normalized with
    the SAME window loc_scale) + the real-unit median forecast for MASE. `pred_q_real`
    is in REAL units (both the native forward output and our pathway output are)."""
    import torch
    from probing.extraction import get_pipeline
    model = get_pipeline()[0].model
    dec_idx = [qlevels.index(q) for q in DECILES]
    with torch.no_grad():
        ls = (torch.from_numpy(loc), torch.from_numpy(scale))
        y_norm = model.instance_norm(torch.from_numpy(fut.astype(np.float32)),
                                     loc_scale=ls)[0].numpy()
        n, Q, Hh = pred_q_real.shape
        p_norm = model.instance_norm(
            torch.from_numpy(pred_q_real.reshape(n, Q * Hh).astype(np.float32)),
            loc_scale=ls)[0].numpy().reshape(n, Q, Hh)
    losses = []
    for qi, q in zip(dec_idx, DECILES):
        diff = y_norm - p_norm[:, qi, :]
        losses.append(np.where(diff >= 0, q * diff, (q - 1) * diff))
    q9 = float(np.mean(losses))
    pred_real_med = pred_q_real[:, qlevels.index(0.5), :]
    return q9, pred_real_med


def run_readout(run: str) -> dict:
    print(f"\n=== native-head {run} ===")
    series = load_series_for(run)
    ctxs, futs = build_windows(series)
    n = len(ctxs)
    print(f"  windows: n={n} (ctx {ctxs.shape}, fut {futs.shape}); series raw={len(series)}")
    d = extract_fslot(run, ctxs)

    from probing.extraction import get_pipeline
    model = get_pipeline()[0].model
    qlevels = [round(float(q), 3) for q in model.chronos_config.quantiles]
    assert all(q in qlevels for q in DECILES + [0.5]), f"deciles missing from {qlevels}"
    loc, scale = d["loc"], d["scale"]
    mase_den = np.array([np.abs(c[24:] - c[:-24]).mean() for c in ctxs])
    ok_den = np.isfinite(mase_den) & (mase_den > 0)

    def full_metrics(pred_q_real):
        q9, pred_real_med = metrics_q9_mase(pred_q_real, futs, loc, scale, qlevels)
        mae = np.abs(futs - pred_real_med).mean(axis=1)
        mase = float((mae[ok_den] / mase_den[ok_den]).mean())
        return q9, mase

    # ---- native reference (the model's own forward output on these windows) ----
    native_q = d["native_q"]
    nat_q9, nat_mase = full_metrics(native_q)
    print(f"  native forward: q9={nat_q9:.6f}  MASE={nat_mase:.4f}  "
          f"(windows with valid MASE denom: {int(ok_den.sum())}/{n})")

    # ---- SELF-CONSISTENT GATE: L12_postln pathway must reproduce the native output ----
    path_q = head_forward(d["fs_L12_postln"], loc, scale, apply_ln=False)
    fro = float(np.linalg.norm(path_q - native_q) / np.linalg.norm(native_q))
    g_q9, _ = full_metrics(path_q)
    rel_loss = abs(g_q9 - nat_q9) / abs(nat_q9)
    print(f"  [GATE] L12_postln pathway vs native: rel Frobenius Δ = {fro:.3e}, "
          f"q9 rel Δ = {rel_loss:.3e}  (< {GATE_REL_TOL:.0e}: {fro < GATE_REL_TOL and rel_loss < GATE_REL_TOL})")
    assert fro < GATE_REL_TOL and rel_loss < GATE_REL_TOL, \
        f"{run}: wiring gate FAILED — STOP (no other layers evaluated)"

    # ---- per-layer readout ----
    per_layer = {}
    for ln in AXIS14:
        pred = head_forward(d[f"fs_{ln}"], loc, scale, apply_ln=(ln != "L12_postln"))
        q9, mase = full_metrics(pred)
        per_layer[ln] = {"q9_loss": q9, "mase": mase}
        print(f"    {ln:>11}: q9={q9:9.5f}  MASE={mase:9.4f}")

    q9s = np.array([per_layer[ln]["q9_loss"] for ln in AXIS14])
    argmin = AXIS14[int(q9s.argmin())]
    spread = float(q9s.max() / q9s.min())
    out = {
        "provenance": {
            "run": run, "n_windows": int(n), "C": C, "H": H, "K": K, "seed": SEED,
            "window_rule": "ctx=seg[-(C+H):-H], fut=seg[-H:], longest-finite-segment, "
                           "<=200 series seed-0 sample",
            "readout": "fslot states -> frozen final LN (skipped for L12_postln) -> frozen "
                       "output_patch_embedding -> rearrange -> instance_norm.inverse "
                       "(the model's own final forward step) -> trim H; ZERO fitting",
            "metrics": "q9 = mean pinball over the 9 deciles in the model's arcsinh-"
                       "normalized space; MASE = real units, median forecast un-scaled via "
                       "InstanceNorm.inverse(loc_scale), seasonal-naive m=24 denominator "
                       "from the context window",
            "interface_credit": "extends probing.extraction.extract_kout_features "
                                "(K-slot encode, position layout, fslot pooler)",
            "quantile_levels": qlevels,
        },
        "native": {"q9_loss": nat_q9, "mase": nat_mase},
        "gate": {"rel_frobenius": fro, "q9_rel_delta": rel_loss, "tol": GATE_REL_TOL,
                 "passed": True},
        "layer_axis": AXIS14,
        "per_layer": per_layer,
        "summary": {"argmin_layer": argmin, "q9_max_over_min": spread,
                    "shape": "FLAT" if spread < 1.05 else "STRUCTURED"},
    }
    dd = NH_DIR / run
    dd.mkdir(parents=True, exist_ok=True)
    (dd / "native_head.json").write_text(json.dumps(out, indent=1))
    print(f"  [saved] {(dd / 'native_head.json').relative_to(OUT_DIR)}  "
          f"argmin={argmin}  max/min={spread:.3f}  -> {out['summary']['shape']}")
    return out


# --------------------------------------------------------------------------- #
# figure (reads native_head.json + the committed probe summary)
# --------------------------------------------------------------------------- #

def plot_run(run: str) -> None:
    m = json.loads((NH_DIR / run / "native_head.json").read_text())
    axis = m["layer_axis"]
    xs = np.arange(len(axis))
    q9 = [m["per_layer"][ln]["q9_loss"] for ln in axis]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(xs[:-1], q9[:-1], "o-", color="C3", label="frozen native head on layer-k fslot states")
    ax.plot([xs[-1]], [q9[-1]], marker="s", mfc="none", mec="C3", ms=10, ls="none",
            label="L12_postln (native input)")
    ax.axhline(m["native"]["q9_loss"], ls="--", color="black", lw=1,
               label=f"native forward baseline (q9={m['native']['q9_loss']:.4f})")
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels(axis, rotation=45)
    ax.set_ylabel("q9 pinball loss (arcsinh space, log scale)")

    probe_tag = {"electricity": "monash_electricity_hourly", "m4": "m4_hourly"}.get(run)
    if probe_tag:
        summ = json.loads((OUT_DIR / "phase0_trio" / "id_probing_summary.json").read_text())
        acc = summ["id_datasets"][probe_tag]["poolings"]["content"]["binned_accuracy"]
        ax2 = ax.twinx()
        ax2.plot(xs[1:13], acc, "-", color="grey", alpha=0.35, lw=2,
                 label="linear-probe accuracy (existing, faint)")
        ax2.set_ylabel("probe accuracy (faint grey)", color="grey")
        ax2.tick_params(axis="y", labelcolor="grey")
    ax.set_title(f"{run}: frozen native-head readout by layer (argmin={m['summary']['argmin_layer']}, "
                 f"{m['summary']['shape']})", fontsize=10)
    ax.legend(fontsize=7, loc="upper center")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = NH_DIR / run / "fig_native_head_by_layer.png"
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {out.relative_to(OUT_DIR)}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(ID_RUNS))
    args = ap.parse_args(argv)
    NH_DIR.mkdir(parents=True, exist_ok=True)
    for run in args.datasets:
        run_readout(run)
        plot_run(run)


if __name__ == "__main__":
    main()
