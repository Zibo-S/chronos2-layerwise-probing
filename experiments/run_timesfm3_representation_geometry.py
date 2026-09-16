"""TimesFM-3 representation geometry: linear CKA + effective rank, per paper7 dataset.

Cache-only, model-free, CPU-only. Consumes the validated last-context-token features written by
``experiments/run_timesfm3_last_token_probing.py`` and measures how the representation ITSELF
evolves with depth -- the complement to "is the forecast linearly decodable" (the learned probe)
and "is the representation already aligned with the pretrained readout" (the native-head
transfer, ``run_timesfm3_native_head_transfer.py``).

ESTIMATORS -- the Chronos-2 ones, imported, never reimplemented:
  CKA             probing.cka.cka_matrix(..., estimator=...)
                  biased   ||Xc^T Yc||_F^2 / (||Xc^T Xc||_F ||Yc^T Yc||_F)
                  unbiased the Song et al. (2012) unbiased-HSIC variant, same module
                  both centred across the N observations, float64
  effective rank  probing.spectral_metrics.spectral_metrics
                  s = svdvals(X - mean_0(X)); p = s**2 / sum(s**2); exp(-sum p log p)

TWO CKA ESTIMATORS, BOTH REQUIRED -- and they play different roles:

  biased   THE HEADLINE / PARITY analysis. Byte-for-byte the estimator and split of the
           committed Chronos-2 CKA (run_cka_analysis.py), so the two models' analyses are
           methodologically identical.
  unbiased THE INTERPRETATION analysis. TimesFM-3's last-token geometry is hostile to the
           biased estimator: d=1280 against only N=262/354 test windows (N=48 for Coastal T-S),
           because ONE native readout token per window means N = the window count, where
           Chronos-2's K=4 slot stacking gave it 4N rows at d=768. Measured on INDEPENDENT
           Gaussian representations, where the true CKA is 0, the biased estimator returns
           ~0.83 at N=262, ~0.78 at N=354 and ~0.96 at N=48 (Chronos-2 sat at ~0.42). Absolute
           biased values are therefore NOT directly readable as representational similarity
           here, and the unbiased companion is the informative one for absolute structure.

  ABSOLUTE BIASED CKA VALUES MUST NOT BE COMPARED ACROSS CHRONOS-2 AND TIMESFM-3 as though they
  were on one scale: their finite-sample baselines differ (~0.42 vs ~0.83). Compare PATTERNS
  within a model, or compare the unbiased numbers.

REPRESENTATION -- the only model-specific choice:
  Chronos-2 : forecast slots (n, K, 768) -> (n*K, 768)
  TimesFM-3 : h_{l,15}, the last REAL context token, directly (N, 1280) for Emb, L1..L20.
              No reshape: at C=512/H=64/P=32 the native head reads exactly ONE token per window.

SPLITS:
  CKA            test (headline, Chronos-parity) AND train (robustness -- it is the only
                 reading with real statistical power for Coastal T-S, whose test split has 48
                 windows against d=1280)
  effective rank train, mirroring run_spectral.py's committed protocol, UNCHANGED
  --split forces everything onto one split for a single-split pass.

OUTPUTS are split by weight, as the project's general rule:
  --out-root  ($SCRATCH)  the full 21x21 matrices and labelled CSVs, one namespace per
                          <estimator>/<split>, plus per-dataset JSON with full spectra
  --paper-out (repo)      summary.json + every final figure, small enough to version and to
                          drop straight into LaTeX next to the Chronos-2 panels

Run (compute node; CPU is fine, no GPU and no model -- see CLAUDE.md's login-node rule):
    python -m experiments.run_timesfm3_representation_geometry \
        --cache-dir $SCRATCH/timesfm3_last_token/features_cache \
        --probe-results $SCRATCH/timesfm3_last_token/results/timesfm3_last_token_paper7_q9/timesfm3_last_token_summary.json \
        --out-root $SCRATCH/timesfm3_geometry
"""

from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import cka as cka_mod  # noqa: E402
from probing.timesfm3_geometry import (LAYER_NAMES, MODEL_DIMS, NUM_LAYERS,  # noqa: E402
                                       SELECTED_TOKEN_INDEX, cka_layer_matrix, cka_null_floor,
                                       effective_rank_curve, estimator_provenance, git_commit,
                                       load_last_token_reps, paper_out_default)
# roster / windows / display names / Chronos-2 parity: ONE source of truth, the probe driver
from experiments.run_timesfm3_last_token_probing import (KIND, PAPER7, SHORT,  # noqa: E402
                                                         assert_window_parity, chronos_reference,
                                                         suite_tags, windows_for)

SLUG = {"m4_hourly": "m4", "monash_electricity_hourly": "electricity",
        "uber_tlc_hourly": "uber_tlc", "wind_farms_hourly": "wind_farms",
        "sg_carpark": "sg_carpark", "coastal_ts": "coastal_ts", "boom_hourly": "boom"}

HEADLINE_ESTIMATOR = "biased"     # parity with run_cka_analysis.py
HEADLINE_CKA_SPLIT = "test"       # parity with run_cka_analysis.py --fslot-split
ERANK_SPLIT_DEFAULT = "train"     # parity with run_spectral.py --split

ROLE = {"biased": "Chronos-parity analysis (biased linear CKA, the committed Chronos-2 "
                  "estimator)",
        "unbiased": "finite-sample-bias-corrected analysis (unbiased HSIC; the informative "
                    "estimator for ABSOLUTE similarity at this N and d)"}
SHORT_ROLE = {"biased": "Chronos-parity", "unbiased": "bias-corrected"}

SUMMARY_NOTE = (
    "Biased CKA is retained for direct parity with Chronos-2, but its finite-sample baseline is "
    "elevated for TimesFM-3 because the representation dimension is large relative to the number "
    "of test observations. Unbiased CKA is therefore used as a robustness analysis for absolute "
    "interpretability.")
CROSS_MODEL_CAVEAT = (
    "Do NOT compare absolute biased-CKA values across Chronos-2 and TimesFM-3: their "
    "finite-sample baselines differ (~0.42 at Chronos-2's n=1048/d=768 vs ~0.83 at TimesFM-3's "
    "N=262/d=1280), so the same number does not mean the same similarity. Compare patterns "
    "within a model, or compare the unbiased estimates.")


# --------------------------------------------------------------------------- #
# the probe run's validation-selected tunnel entrance (overlay only, never an input)
# --------------------------------------------------------------------------- #
def load_tunnel_entrances(path) -> dict:
    """{tag: {"5pct": {...}, "2pct": {...}}} from the Q=9 probe summary, or {} when absent.

    The entrance is read ONLY to draw a vertical marker and to index the diagnostic table. It
    never enters a CKA or an effective-rank computation, and the tunnel stays defined by the
    probe's validation criterion -- nothing here redefines it.
    """
    if not path:
        return {}
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"--probe-results {p} does not exist. Omit the flag to run "
                                "without the tunnel overlay, or point it at the paper7 Q=9 "
                                "run's timesfm3_last_token_summary.json.")
    s = json.loads(p.read_text())
    out = {}
    for tag, e in s.get("datasets", {}).items():
        by_tol = e.get("tunnel_by_tolerance") or {}
        out[tag] = {"5pct": by_tol.get("0.05", e.get("tunnel")), "2pct": by_tol.get("0.02"),
                    "probe_summary": str(p)}
    if not out:
        raise RuntimeError(f"{p} has no 'datasets' entries -- is this the probe summary JSON?")
    return out


# --------------------------------------------------------------------------- #
# per-dataset analysis
# --------------------------------------------------------------------------- #
def analyse_dataset(tag, args, geom, tunnels, floors) -> dict:
    """CKA over every (estimator, split) requested, and effective rank on --erank-split."""
    t0 = time.time()
    short, kind = SHORT.get(tag, tag), KIND.get(tag, "unclassified")
    print(f"\n{'=' * 78}\n[{tag}]  {short}  ({kind})\n{'=' * 78}")

    w = windows_for(tag, args.suite, args)
    ident = assert_window_parity(tag, w, chronos_reference(tag),
                                 strict=not args.allow_window_mismatch)
    print(f"  windows: {ident['n_train_windows']} train / {ident['n_val_windows']} val / "
          f"{ident['n_test_windows']} test   [Chronos-2 parity: {ident.get('chronos_parity')}]")

    split_key = {"train": ("X_train", None), "val": ("X_val", None),
                 "test": ("X_test", "series_test")}

    def reps_for(split):
        xk, sk = split_key[split]
        if xk not in w or len(w[xk]) == 0:
            raise RuntimeError(f"{tag}: no {split} windows to analyse")
        return load_last_token_reps(
            tag, split, w[xk], cache_dir=args.cache_dir, geom=geom, layers=args.layers,
            checkpoint=args.checkpoint, suite=args.suite, seed=args.seed,
            detrend=not args.no_detrend, feature_dtype=np.dtype(args.feature_dtype),
            ignore_timesfm_version=args.ignore_timesfm_version,
            series_ids=(w[sk] if sk else None))

    needed = list(dict.fromkeys(list(args.cka_splits) + [args.erank_split]))
    loaded = {sp: reps_for(sp) for sp in needed}
    for sp, L in loaded.items():
        print(f"  [{sp}] {len(L['layers'])} points {L['layer_names'][0]}..{L['layer_names'][-1]}, "
              f"each ({L['n']}, {L['hidden_dim']}) at token {L['selected_token_index']}   "
              f"window-id {L['window_identity_hash']}")

    # ---- CKA: every (estimator, split), identical layer ordering throughout ----
    grid = {}
    for est in args.cka_estimators:
        grid[est] = {}
        for sp in args.cka_splits:
            L = loaded[sp]
            M = cka_layer_matrix(L["reps"], estimator=est)
            floor = (_floor(floors, L["n"], args, est) if args.cka_null_floor_reps else None)
            head = "  <== HEADLINE (Chronos parity)" if (est == HEADLINE_ESTIMATOR
                                                         and sp == args.headline_cka_split) else ""
            print(f"  CKA  {est:<8} {sp:<5} N={L['n']:<5} 21x21   Emb-L20 {M[0, -1]:+.4f}   "
                  f"L19-L20 {M[-2, -1]:+.4f}   min {M.min():+.4f}"
                  + (f"   null floor {floor['mean']:.3f}" if floor else "") + head)
            grid[est][sp] = {"matrix": M.tolist(), "N": L["n"], "estimator": est, "split": sp,
                             "min": float(M.min()), "max": float(M.max()),
                             "null_floor": floor, "role": ROLE[est],
                             "provenance": _prov(L)}

    # ---- effective rank: train, the committed Chronos-2 spectral protocol, UNCHANGED ----
    E = loaded[args.erank_split]
    print(f"  rank ({args.erank_split} split, N={E['n']}, ceiling "
          f"{min(E['n'] - 1, MODEL_DIMS)}) ...", end="", flush=True)
    er = effective_rank_curve(E["reps"], E["layer_names"], hidden_dim=MODEL_DIMS,
                              n_subsamples=args.erank_subsamples, frac=args.erank_subsample_frac,
                              seed=args.seed)
    print(f" Emb={er['effective_rank'][0]:.1f}  L20={er['effective_rank'][-1]:.1f}")

    layers = loaded[args.cka_splits[0]]["layers"]
    tun = tunnels.get(tag) or {}
    combined = _combined_summary(grid, er, layers, tun, args)
    if combined.get("tunnel_entrance_5pct") is not None:
        print(f"  tunnel (from probe run): 5% {combined['tunnel_entrance_5pct_name']} / "
              f"2% {combined.get('tunnel_entrance_2pct_name')}   |  r_eff at Emb "
              f"{combined['effective_rank_at_emb']:.1f} -> tunnel "
              f"{combined['effective_rank_at_tunnel']:.1f} -> L20 "
              f"{combined['effective_rank_at_L20']:.1f}")

    return {"tag": tag, "short": short, "slug": SLUG.get(tag, tag), "domain_status": kind,
            "kind": kind, "window_identity": ident,
            "representation_points": loaded[args.cka_splits[0]]["layer_names"], "layers": layers,
            "hidden_dim": MODEL_DIMS, "selected_token_index": SELECTED_TOKEN_INDEX,
            "cka": {"headline": {"estimator": HEADLINE_ESTIMATOR,
                                 "split": args.headline_cka_split},
                    "estimators": list(args.cka_estimators), "splits": list(args.cka_splits),
                    "N": {sp: loaded[sp]["n"] for sp in args.cka_splits},
                    "grid": grid, "note": SUMMARY_NOTE,
                    "cross_model_caveat": CROSS_MODEL_CAVEAT},
            "effective_rank": {"split": args.erank_split, "N": E["n"],
                               **{k: v for k, v in er.items() if k != "spectrum"},
                               "provenance": _prov(E)},
            "spectrum": er["spectrum"],
            "tunnel": tun, "combined_summary": combined,
            "seconds": round(time.time() - t0, 1)}


def _floor(cache, n, args, estimator):
    """Null floor memoized on (n, d, estimator): it depends on the SHAPE, not the dataset."""
    key = (int(n), MODEL_DIMS, estimator)
    if key not in cache:
        cache[key] = cka_null_floor(n, MODEL_DIMS, reps=args.cka_null_floor_reps,
                                    seed=args.seed, estimator=estimator)
    return cache[key]


def _prov(L) -> dict:
    """The reproducibility block for one loaded (dataset, split) representation set."""
    return {k: L[k] for k in ("cache_root", "cache_version", "checkpoint", "timesfm_version",
                              "suite", "detrending", "feature_dtype", "window_identity_hash",
                              "context_len", "horizon", "patch_size", "seed",
                              "selected_token_index", "n")}


def _cka_points(M, layers, t5) -> dict:
    """The three reported cells of one CKA matrix: Emb-L20, tunnel-L20, (tunnel-1)-L20."""
    idx = {l: i for i, l in enumerate(layers)}
    out = {"Emb_L20": float(M[0, -1]), "tunnel_L20": None, "tunnel_minus_1_L20": None}
    if t5 is not None and t5 in idx:
        i = idx[t5]
        out["tunnel_L20"] = float(M[i, -1])
        if i > 0:
            out["tunnel_minus_1_L20"] = float(M[i - 1, -1])
    return out


def _combined_summary(grid, er, layers, tun, args) -> dict:
    """The compact per-dataset diagnostic table. Measurements only -- no interpretation."""
    t5 = (tun.get("5pct") or {}).get("layer")
    t2 = (tun.get("2pct") or {}).get("layer")
    idx = {l: i for i, l in enumerate(layers)}
    out = {
        "tunnel_entrance_5pct": t5,
        "tunnel_entrance_5pct_name": (tun.get("5pct") or {}).get("layer_name"),
        "tunnel_entrance_2pct": t2,
        "tunnel_entrance_2pct_name": (tun.get("2pct") or {}).get("layer_name"),
        "effective_rank_split": er.get("split", args.erank_split),
        "effective_rank_at_emb": er["effective_rank"][0],
        "effective_rank_at_L20": er["effective_rank"][-1],
        "normalized_effective_rank_at_emb": er["normalized_effective_rank"][0],
        "normalized_effective_rank_at_L20": er["normalized_effective_rank"][-1],
        "effective_rank_at_tunnel": None, "normalized_effective_rank_at_tunnel": None,
        "cka": {}, "note": SUMMARY_NOTE,
    }
    if t5 is not None and t5 in idx:
        i = idx[t5]
        out["effective_rank_at_tunnel"] = er["effective_rank"][i]
        out["normalized_effective_rank_at_tunnel"] = er["normalized_effective_rank"][i]
    for est, by_split in grid.items():
        for sp, g in by_split.items():
            pts = _cka_points(np.asarray(g["matrix"], float), layers, t5)
            fl = (g["null_floor"] or {}).get("mean")
            out["cka"][f"{est}/{sp}"] = {
                **pts, "N": g["N"], "null_floor": fl,
                # how far the headline cells sit ABOVE this estimator's own floor at this shape
                "Emb_L20_above_null_floor": (None if fl is None else pts["Emb_L20"] - fl),
                "tunnel_L20_above_null_floor": (None if fl is None or pts["tunnel_L20"] is None
                                                else pts["tunnel_L20"] - fl)}
    h = f"{HEADLINE_ESTIMATOR}/{args.headline_cka_split}"
    if h in out["cka"]:                       # flat aliases for the headline, for quick tables
        out["CKA_Emb_L20"] = out["cka"][h]["Emb_L20"]
        out["CKA_tunnel_L20"] = out["cka"][h]["tunnel_L20"]
        out["CKA_tunnel_minus_1_L20"] = out["cka"][h]["tunnel_minus_1_L20"]
        out["CKA_null_floor"] = out["cka"][h]["null_floor"]
    u = f"unbiased/{args.headline_cka_split}"
    if u in out["cka"]:
        out["CKA_unbiased_Emb_L20"] = out["cka"][u]["Emb_L20"]
        out["CKA_unbiased_tunnel_L20"] = out["cka"][u]["tunnel_L20"]
    return out


# --------------------------------------------------------------------------- #
# figures -- Chronos-2 style (cka.heatmap: viridis, [0,1], square cells, png + pdf)
# --------------------------------------------------------------------------- #
def _caption(est, sp, g) -> str:
    fl = (g.get("null_floor") or {}).get("mean")
    bits = [f"{est} linear CKA", f"{sp} split", f"N={g['N']}"]
    if fl is not None:
        bits.append(f"independent-representation floor {fl:.3f}")
    return "  |  ".join(bits)


def _tunnel_marks(ax, tun, layers, *, both_axes=True):
    t = (tun.get("5pct") or {}).get("layer")
    if t is None or t not in layers:
        return None
    p = layers.index(t)
    ax.axvline(p, ls=":", c="w", lw=1.2, alpha=0.9)
    if both_axes:
        ax.axhline(p, ls=":", c="w", lw=1.2, alpha=0.9)
    return p


def figure_cka_per_dataset(recs, figdir, est, sp) -> list[Path]:
    """One 21x21 heatmap per dataset on the shared [0, 1] scale, for one (estimator, split)."""
    d = figdir / "cka" / est / sp
    made = []
    for r in recs:
        g = r["cka"]["grid"][est][sp]
        M = np.asarray(g["matrix"], float)
        names = r["representation_points"]
        made.append(cka_mod.heatmap(
            M, names, names, d / f"cka_{r['slug']}.png",
            title=f"TimesFM-3 {r['short']} ({r['domain_status']}) — token "
                  f"{SELECTED_TOKEN_INDEX}\n{_caption(est, sp, g)}\n{SHORT_ROLE[est]}",
            vmin=0.0, vmax=1.0, cbar_label=f"Linear CKA ({est})",
            xaxis_label="representation point", yaxis_label="representation point"))
    return made


def figure_cka_panel(recs, figdir, tunnels, est, sp, *, mark_tunnel=False) -> Path:
    """Combined panel: the 4 PT-ID datasets on the first row, the 3 PT-OOD on the second."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [[r for r in recs if r["domain_status"] == k] for k in ("PT-ID", "PT-OOD")]
    rows = [r for r in rows if r]
    ncol = max(len(r) for r in rows)
    fig, ax = plt.subplots(len(rows), ncol, figsize=(3.5 * ncol, 4.0 * len(rows)), squeeze=False)
    im = None
    for ri, group in enumerate(rows):
        for ci in range(ncol):
            a = ax[ri][ci]
            if ci >= len(group):
                a.set_visible(False)
                continue
            r = group[ci]
            g = r["cka"]["grid"][est][sp]
            M = np.asarray(g["matrix"], float)
            names = r["representation_points"]
            im = a.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="equal", origin="upper")
            step = 4
            a.set_xticks(np.arange(0, len(names), step))
            a.set_xticklabels(names[::step], rotation=45, ha="right", fontsize=6)
            a.set_yticks(np.arange(0, len(names), step))
            a.set_yticklabels(names[::step], fontsize=6)
            if mark_tunnel:
                _tunnel_marks(a, tunnels.get(r["tag"]) or {}, r["layers"])
            fl = (g.get("null_floor") or {}).get("mean")
            a.set_title(f"{r['short']}  ({r['domain_status']}, N={g['N']})"
                        + (f"\nfloor {fl:.3f}" if fl is not None else ""), fontsize=8.5)
    head = " — HEADLINE" if (est == HEADLINE_ESTIMATOR and sp == HEADLINE_CKA_SPLIT) else ""
    fig.suptitle(f"TimesFM-3 last-context-token (h_l,{SELECTED_TOKEN_INDEX}) — {est} linear CKA, "
                 f"{sp} split{head}\n{ROLE[est]}"
                 + ("   |   dotted = 5% tunnel entrance" if mark_tunnel else ""), fontsize=10)
    fig.tight_layout(rect=(0, 0, 0.92, 0.95))
    cax = fig.add_axes((0.935, 0.12, 0.015, 0.74))
    fig.colorbar(im, cax=cax).set_label(f"Linear CKA ({est})", fontsize=9)
    p = (figdir / "cka" / est / sp /
         ("cka_all_datasets__tunnel.png" if mark_tunnel else "cka_all_datasets.png"))
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)
    return p


def figure_cka_floor_comparison(recs, figdir, args) -> Path:
    """One panel that makes the estimator question visible: CKA(Emb,L20) vs its own null floor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    combos = [(e, s) for e in args.cka_estimators for s in args.cka_splits]
    x = np.arange(len(recs))
    fig, ax = plt.subplots(1, len(combos), figsize=(3.4 * len(combos), 4.4), squeeze=False)
    for a, (est, sp) in zip(ax[0], combos):
        vals = [r["combined_summary"]["cka"][f"{est}/{sp}"]["Emb_L20"] for r in recs]
        fls = [r["combined_summary"]["cka"][f"{est}/{sp}"]["null_floor"] for r in recs]
        a.bar(x, vals, color=["#1f77b4" if r["domain_status"] == "PT-ID" else "#ff7f0e"
                              for r in recs])
        if all(f is not None for f in fls):
            a.plot(x, fls, "k_", ms=22, mew=2, label="independent-representation floor")
            a.legend(fontsize=6.5)
        a.set_xticks(x)
        a.set_xticklabels([r["short"] for r in recs], rotation=45, ha="right", fontsize=7)
        a.axhline(0, lw=0.8, c="k")
        a.set_title(f"{est} / {sp}", fontsize=10)
        a.grid(alpha=0.3, axis="y")
    ax[0][0].set_ylabel("CKA(Emb, L20)")
    fig.suptitle("CKA(Emb, L20) against each estimator's own finite-sample floor\n"
                 "biased sits near its floor at small N; unbiased is floor-free", fontsize=10)
    fig.tight_layout()
    p = figdir / "cka" / "cka_vs_null_floor.png"
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)
    return p


def figure_effective_rank(recs, figdir, tunnels, *, key="effective_rank", group=None,
                          name=None) -> Path:
    """Effective-rank (or normalized) curves vs representation point, one panel per dataset."""
    import math

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ds = [r for r in recs if group is None or r["domain_status"] == group]
    if not ds:
        return None
    ncol = min(4, len(ds))
    nrow = math.ceil(len(ds) / ncol)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.3 * ncol, 3.9 * nrow), squeeze=False)
    axes = [a for row in ax for a in row]
    ylab = {"effective_rank": "effective rank  exp(H)",
            "normalized_effective_rank": "effective rank / d   (d = 1280)",
            "effective_rank_over_max_possible": "effective rank / min(N-1, d)"}[key]
    for a, r in zip(axes, ds):
        er, lay = r["effective_rank"], r["layers"]
        a.plot(lay, er[key], "o-", ms=3.5, color="#2ca02c", label="effective rank")
        if key == "effective_rank":
            a.axhline(er["max_possible_rank"], ls="--", lw=0.9, c="grey",
                      label=f"ceiling min(N-1,d)={er['max_possible_rank']:.0f}")
        sub = er.get("subsample")
        if sub and key == "effective_rank":
            # the Chronos-2 protocol's own interval: subsampling WITHOUT replacement at
            # frac*N rows, so it describes variability at the reduced size, not a CI for the
            # full-sample estimate (rank estimates grow with N).
            names = r["representation_points"]
            lo = [sub[n]["effective_rank"]["ci"][0] for n in names]
            hi = [sub[n]["effective_rank"]["ci"][1] for n in names]
            a.fill_between(lay, lo, hi, alpha=0.2, color="#2ca02c",
                           label="95% subsample interval")
        t = (tunnels.get(r["tag"]) or {}).get("5pct") or {}
        if t.get("layer") is not None:
            a.axvline(t["layer"], ls=":", c="k", lw=1.1,
                      label=f"5% tunnel entrance = {t['layer_name']}")
        a.set_title(f"{r['short']}  ({r['domain_status']}, {er['split']} N={er['N']})",
                    fontsize=10)
        a.set_xlabel("representation point")
        a.set_xticks(lay[::2])
        a.set_xticklabels([r["representation_points"][i] for i in range(0, len(lay), 2)],
                          rotation=45, fontsize=7)
        a.grid(alpha=0.3)
        a.legend(fontsize=6.5)
    for a in axes[len(ds):]:
        a.set_visible(False)
    for row in ax:
        row[0].set_ylabel(ylab)
    fig.suptitle(f"TimesFM-3 last-context-token (h_l,{SELECTED_TOKEN_INDEX}) — {ylab}"
                 + (f"   [{group}]" if group else ""), fontsize=11)
    fig.tight_layout()
    p = figdir / "effective_rank" / (name or f"{key}_all_datasets.png")
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=200); fig.savefig(p.with_suffix(".pdf")); plt.close(fig)
    return p


# --------------------------------------------------------------------------- #
# writing
# --------------------------------------------------------------------------- #
def write_heavy(recs, out_root, meta) -> None:
    """Full matrices / spectra / labelled CSVs -> $SCRATCH (never the repo).

    One namespace per <estimator>/<split>, so biased/test, unbiased/test, biased/train and
    unbiased/train can never be confused for one another.
    """
    mats, nums = out_root / "matrices", out_root / "numerical_results"
    nums.mkdir(parents=True, exist_ok=True)
    (out_root / "temporary").mkdir(parents=True, exist_ok=True)
    for r in recs:
        names = r["representation_points"]
        for est, by_split in r["cka"]["grid"].items():
            for sp, g in by_split.items():
                d = mats / est / sp
                d.mkdir(parents=True, exist_ok=True)
                M = np.asarray(g["matrix"], float)
                np.save(d / f"cka__{r['tag']}.npy", M)
                cka_mod.save_matrix_csv(M, names, names, d / f"cka__{r['tag']}.csv")
        er = r["effective_rank"]
        with open(nums / f"effective_rank__{r['tag']}__{er['split']}.csv", "w",
                  newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["representation_point", "layer_index", "effective_rank",
                         "normalized_effective_rank", "effective_rank_over_max_possible",
                         "spectral_entropy", "pc1_fraction", "numerical_rank"])
            for i, l in enumerate(r["layers"]):
                wr.writerow([names[i], l, f"{er['effective_rank'][i]:.6f}",
                             f"{er['normalized_effective_rank'][i]:.8f}",
                             f"{er['effective_rank_over_max_possible'][i]:.8f}",
                             f"{er['spectral_entropy'][i]:.6f}",
                             f"{er['pc1_fraction'][i]:.6f}", er["numerical_rank"][i]])
        (nums / f"geometry__{r['tag']}.json").write_text(
            json.dumps({**meta, **r}, indent=2, default=str))
    print(f"\n[heavy]  {out_root}  (matrices/<estimator>/<split>/, numerical_results/, "
          f"temporary/)")


def write_paper(recs, paper_out, meta) -> Path:
    """Lightweight, versionable summary -> the repo. Full spectra stay on $SCRATCH."""
    paper_out.mkdir(parents=True, exist_ok=True)
    payload = {**meta, "datasets": {}}
    for r in recs:
        payload["datasets"][r["tag"]] = {k: v for k, v in r.items() if k != "spectrum"}
    p = paper_out / "summary.json"
    p.write_text(json.dumps(payload, indent=2, default=str))

    combos = [(e, s) for e in meta["cka_estimators"] for s in meta["cka_splits"]]
    head = ["dataset", "display_name", "domain_status", "erank_split", "erank_N",
            "tunnel_5pct", "tunnel_2pct", "erank_emb", "erank_tunnel", "erank_L20",
            "nerank_emb", "nerank_tunnel", "nerank_L20"]
    for e, s in combos:
        head += [f"CKA_{e}_{s}_N", f"CKA_{e}_{s}_Emb_L20", f"CKA_{e}_{s}_tunnel_L20",
                 f"CKA_{e}_{s}_tunnel_minus_1_L20", f"CKA_{e}_{s}_null_floor",
                 f"CKA_{e}_{s}_Emb_L20_above_floor"]
    rows = [head]
    fmt = lambda v: ("" if v is None else f"{v:.6g}")
    for r in recs:
        c = r["combined_summary"]
        row = [r["tag"], r["short"], r["domain_status"], r["effective_rank"]["split"],
               r["effective_rank"]["N"], c["tunnel_entrance_5pct_name"],
               c["tunnel_entrance_2pct_name"],
               *[fmt(c[k]) for k in ("effective_rank_at_emb", "effective_rank_at_tunnel",
                                     "effective_rank_at_L20",
                                     "normalized_effective_rank_at_emb",
                                     "normalized_effective_rank_at_tunnel",
                                     "normalized_effective_rank_at_L20")]]
        for e, s in combos:
            g = c["cka"][f"{e}/{s}"]
            row += [g["N"], *[fmt(g[k]) for k in ("Emb_L20", "tunnel_L20",
                                                  "tunnel_minus_1_L20", "null_floor",
                                                  "Emb_L20_above_null_floor")]]
        rows.append(row)
    with open(paper_out / "geometry_summary_table.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    return p


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="TimesFM-3 representation geometry (linear CKA + effective rank) on the "
                    "cached last-context-token states. CPU-only, no model load.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("inputs")
    g.add_argument("--cache-dir", default=os.environ.get("TFM3_LT_CACHE_DIR", None),
                   help="last-token feature cache root [default: <repo>/features_cache]")
    g.add_argument("--probe-results", default=None,
                   help="the paper7 Q=9 run's timesfm3_last_token_summary.json; supplies the "
                        "validation-selected tunnel entrance for the OVERLAY only. Omit to run "
                        "without it")
    g.add_argument("--suite", default="paper7", help="dataset roster (headline: paper7)")
    g.add_argument("--datasets", nargs="+", default=None, help="subset of the suite's roster")
    g.add_argument("--layers", type=int, nargs="+", default=None,
                   help="representation points [default: all 21, Emb..L20]")
    g.add_argument("--checkpoint", default=os.environ.get("TIMESFM3_CHECKPOINT",
                                                          "google/timesfm-3.0-pytorch"),
                   help="must match the checkpoint the cache was extracted with")
    g.add_argument("--feature-dtype", default="float32", choices=["float32", "float16"])
    g.add_argument("--no-detrend", action="store_true",
                   help="select the no-detrending cache (matches the probe run's ablation flag)")
    g.add_argument("--ignore-timesfm-version", action="store_true",
                   help="skip ONLY the timesfm-version field of the cache metadata check (for "
                        "analysing a cache built under a different install); every other field "
                        "and the element-wise window check still apply")
    g.add_argument("--allow-window-mismatch", action="store_true",
                   help="report instead of aborting on a disagreement with the committed "
                        "Chronos-2 windows (NOT recommended)")

    g = p.add_argument_group("CKA: two estimators (parity + interpretation) x two splits")
    g.add_argument("--cka-estimators", nargs="+", default=["biased", "unbiased"],
                   choices=("biased", "unbiased"),
                   help="BOTH by default. 'biased' is the headline/parity estimator, identical "
                        "to the committed Chronos-2 CKA. 'unbiased' is the REQUIRED companion: "
                        "at d=1280 with N=262/354 (48 for Coastal T-S) the biased estimator's "
                        "finite-sample floor is ~0.78-0.96, so absolute biased values are not "
                        "readable as similarity; the unbiased one is floor-free")
    g.add_argument("--cka-splits", nargs="+", default=["test", "train"],
                   choices=("train", "val", "test"),
                   help="BOTH by default. 'test' is the headline (Chronos parity); 'train' "
                        "(N=1394) is the robustness reading, and the only one with real power "
                        "for Coastal T-S")
    g.add_argument("--headline-cka-split", default=HEADLINE_CKA_SPLIT,
                   choices=("train", "val", "test"),
                   help="which split the flat CKA_* summary aliases refer to")
    g.add_argument("--cka-null-floor-reps", type=int, default=3,
                   help="measure each estimator's value for INDEPENDENT representations at each "
                        "dataset's (N, 1280) and save it beside the matrix. Calibration only -- "
                        "nothing is subtracted. 0 disables")

    g = p.add_argument_group("effective rank (committed Chronos-2 spectral protocol)")
    g.add_argument("--erank-split", default=ERANK_SPLIT_DEFAULT, choices=("train", "val", "test"),
                   help="Chronos-2's run_spectral.py uses train; unchanged here")
    g.add_argument("--erank-subsamples", type=int, default=0,
                   help="Chronos-2's subsampling uncertainty (without replacement). 0 = off; "
                        "200 reproduces run_spectral.py's protocol at n_subsamples SVDs per layer")
    g.add_argument("--erank-subsample-frac", type=float, default=0.8)

    g = p.add_argument_group("outputs")
    g.add_argument("--split", default=None, choices=("train", "val", "test"),
                   help="single-split mode: force BOTH the CKA splits and the effective-rank "
                        "split onto this one")
    g.add_argument("--out-root", "--work-dir", dest="out_root",
                   default=os.environ.get("TFM3_GEOM_OUT_ROOT", None),
                   help="HEAVY outputs ($SCRATCH): matrices, spectra, per-dataset JSON "
                        "[default: <repo>/results/timesfm3_representation_geometry/_work]")
    g.add_argument("--paper-out", default=None,
                   help="FINAL paper outputs (repo-local): summary.json + figures/ "
                        "[default: <repo>/results/timesfm3_representation_geometry]")
    g.add_argument("--no-figures", action="store_true")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--context-len", type=int, default=512)
    g.add_argument("--horizon", type=int, default=64)
    g.add_argument("--stride", type=int, default=64)
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.split:
        args.cka_splits, args.erank_split = [args.split], args.split
        args.headline_cka_split = args.split
    args.cka_estimators = list(dict.fromkeys(args.cka_estimators))
    args.cka_splits = list(dict.fromkeys(args.cka_splits))
    if args.headline_cka_split not in args.cka_splits:
        raise SystemExit(f"--headline-cka-split {args.headline_cka_split!r} is not in "
                         f"--cka-splits {args.cka_splits}")
    if HEADLINE_ESTIMATOR not in args.cka_estimators:
        raise SystemExit(f"--cka-estimators must include {HEADLINE_ESTIMATOR!r}: it is the "
                         "parity analysis against the committed Chronos-2 CKA")
    from probing.timesfm3_last_token import LastTokenGeometry
    geom = LastTokenGeometry(args.context_len, args.horizon)   # strict: asserts 512/64/16/15

    roster = suite_tags(args.suite)
    tags = args.datasets or roster
    unknown = [t for t in tags if t not in roster]
    if unknown:
        raise SystemExit(f"unknown dataset tag(s) {unknown} for suite {args.suite!r}; "
                         f"roster = {roster}")
    args.cache_dir = (Path(args.cache_dir).expanduser() if args.cache_dir
                      else REPO_ROOT / "features_cache")
    paper_out = Path(args.paper_out).expanduser() if args.paper_out else paper_out_default()
    out_root = (Path(args.out_root).expanduser() if args.out_root else paper_out / "_work")
    figdir = paper_out / "figures"

    print("TimesFM-3 REPRESENTATION GEOMETRY — linear CKA + effective rank")
    print(f"  representation : h_l,{SELECTED_TOKEN_INDEX} (last REAL context token), "
          f"(N, {MODEL_DIMS}) per point, {NUM_LAYERS} points {LAYER_NAMES[0]}..{LAYER_NAMES[-1]}")
    print(f"  CKA estimators : {args.cka_estimators}   splits: {args.cka_splits}")
    print(f"     HEADLINE    : {HEADLINE_ESTIMATOR}/{args.headline_cka_split}  — "
          f"{ROLE[HEADLINE_ESTIMATOR]}")
    if "unbiased" in args.cka_estimators:
        print(f"     COMPANION   : unbiased/{args.headline_cka_split}  — {ROLE['unbiased']}")
    print(f"  effective rank : {args.erank_split} split   [probing.spectral_metrics, the "
          f"committed Chronos-2 protocol, unchanged]")
    print(f"  datasets       : {len(tags)}  {tags}")
    print(f"  cache          : {args.cache_dir}")
    print("  no model is loaded; no probe weight, prediction or target is read")
    print(f"\n  NOTE: {SUMMARY_NOTE}")
    print(f"  CAVEAT: {CROSS_MODEL_CAVEAT}\n")

    tunnels = load_tunnel_entrances(args.probe_results)
    overlay = (f"from {args.probe_results}" if tunnels else
               "NONE (--probe-results not given; figures omit the marker)")
    print(f"  tunnel overlay : {overlay}")

    meta = {"analysis": "timesfm3_representation_geometry",
            "model": "timesfm-3.0", "backbone": "frozen, never loaded (cache-only analysis)",
            "representation": f"last real context token h_l,{SELECTED_TOKEN_INDEX}, (N, "
                              f"{MODEL_DIMS}) per representation point; NO reshape (native K=1)",
            "representation_points": LAYER_NAMES,
            "num_representation_points": NUM_LAYERS,
            "hidden_dim": MODEL_DIMS, "selected_token_index": SELECTED_TOKEN_INDEX,
            "cka_estimators": list(args.cka_estimators), "cka_splits": list(args.cka_splits),
            "cka_headline": {"estimator": HEADLINE_ESTIMATOR, "split": args.headline_cka_split,
                             "role": ROLE[HEADLINE_ESTIMATOR]},
            "cka_companion": ({"estimator": "unbiased", "split": args.headline_cka_split,
                               "role": ROLE["unbiased"]}
                              if "unbiased" in args.cka_estimators else None),
            "cka_estimator_roles": ROLE,
            "note": SUMMARY_NOTE,
            "cross_model_caveat": CROSS_MODEL_CAVEAT,
            "erank_split": args.erank_split,
            "estimators": estimator_provenance(),
            "probe_independent": "no probe weights, predictions, targets or native-head outputs "
                                 "enter either measure",
            "tunnel_entrance_source": (args.probe_results or None),
            "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "git_commit": git_commit(),
            "computed": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "geometry": geom.as_dict()}

    t0 = time.time()
    floors: dict = {}
    recs = [analyse_dataset(t, args, geom, tunnels, floors) for t in tags]
    order = {"PT-ID": 0, "PT-OOD": 1}
    recs.sort(key=lambda r: (order.get(r["domain_status"], 2),
                             list(PAPER7).index(r["tag"]) if r["tag"] in PAPER7 else 99))
    meta["null_floors"] = {f"{k[2]}/n={k[0]}/d={k[1]}": v["mean"] for k, v in floors.items()}
    meta["seconds"] = round(time.time() - t0, 1)

    write_heavy(recs, out_root, meta)
    sp_path = write_paper(recs, paper_out, meta)

    figs = []
    if not args.no_figures:
        for est in args.cka_estimators:
            for sp in args.cka_splits:
                figs += figure_cka_per_dataset(recs, figdir, est, sp)
                figs.append(figure_cka_panel(recs, figdir, tunnels, est, sp))
                if tunnels and sp == args.headline_cka_split:
                    figs.append(figure_cka_panel(recs, figdir, tunnels, est, sp,
                                                 mark_tunnel=True))
        if args.cka_null_floor_reps:
            figs.append(figure_cka_floor_comparison(recs, figdir, args))
        for key in ("effective_rank", "normalized_effective_rank"):
            figs.append(figure_effective_rank(recs, figdir, tunnels, key=key))
        for grp, nm in (("PT-ID", "effective_rank_pt_id.png"),
                        ("PT-OOD", "effective_rank_pt_ood.png")):
            figs.append(figure_effective_rank(recs, figdir, tunnels, group=grp, name=nm))
        figs = [f for f in figs if f]

    print(f"\n{'=' * 126}\nSUMMARY — representation geometry ({NUM_LAYERS} points, token "
          f"{SELECTED_TOKEN_INDEX}).  CKA cells are CKA(Emb, L20).\n{'=' * 126}")
    hdr = (f"{'dataset':<14}{'kind':<8}{'rkN':>6}{'tun5%':>7}{'r_eff Emb':>11}{'r_eff tun':>11}"
           f"{'r_eff L20':>11}")
    combos = [(e, s) for e in args.cka_estimators for s in args.cka_splits]
    for e, s in combos:
        hdr += f"{e + '/' + s:>15}"
    print(hdr)
    print(f"{'':<14}{'':<8}{'':>6}{'':>7}{'':>11}{'':>11}{'':>11}"
          + "".join(f"{'(null floor)':>15}" for _ in combos))
    last = None
    for r in recs:
        c = r["combined_summary"]
        if last is not None and r["domain_status"] != last:
            print("-" * 126)
        last = r["domain_status"]
        f = lambda v, w=11, p=2: (f"{v:>{w}.{p}f}" if isinstance(v, float) else f"{'-':>{w}}")
        line = (f"{r['short']:<14}{r['domain_status']:<8}{r['effective_rank']['N']:>6}"
                f"{(c['tunnel_entrance_5pct_name'] or '-'):>7}"
                f"{f(c['effective_rank_at_emb'])}{f(c['effective_rank_at_tunnel'])}"
                f"{f(c['effective_rank_at_L20'])}")
        floor_line = f"{'':<14}{'':<8}{'':>6}{'':>7}{'':>11}{'':>11}{'':>11}"
        for e, s in combos:
            g = c["cka"][f"{e}/{s}"]
            line += f"{g['Emb_L20']:>+15.4f}"
            floor_line += (f"{'(' + format(g['null_floor'], '.3f') + ')':>15}"
                           if g["null_floor"] is not None else f"{'':>15}")
        print(line)
        print(floor_line)

    print(f"\nNumerical/intermediate outputs:\n  {out_root}")
    print(f"\nFinal paper outputs:\n  {paper_out}")
    print(f"  {sp_path.relative_to(paper_out)}")
    print("  geometry_summary_table.csv")
    for fp in figs:
        print(f"  figures/{fp.relative_to(figdir)}   (+ .pdf)")
    print(f"\n[{meta['seconds']}s]")
    return {"meta": meta, "datasets": recs, "figures": [str(f) for f in figs]}


if __name__ == "__main__":
    main()
