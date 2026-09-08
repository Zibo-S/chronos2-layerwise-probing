"""Content-slot shared-head probing — the missing cell of the readout 2x2 (ext_v4 sibling).

The committed ext_v4 line reads Chronos-2's K native FORECAST slots with one shared
Linear(768, Q*P) head (run_ptood_probing_ftok, `fslot`), and the pooled line reads a mean over
the ncp CONTENT patches with a Linear(768, Q*H) head (run_id_forecasting, `content_K`). Those
two differ in TWO things at once — which tokens are read AND how they are read — so the
committed finding ("pooled readouts favour intermediate layers, the shared forecast-slot readout
does not") cannot attribute the effect to either. run_id_forecasting:718 says as much in its own
docstring: the shared head "has ~Kx fewer params + enforced patch-wise weight sharing".

This driver adds the fourth cell:

    readout \\ tokens     content tokens          forecast slots
    pooled Linear(768,Q*H)  content_K (committed)   --
    shared Linear(768,Q*P)  THIS DRIVER             fslot (committed)

The head, the fit protocol, the wd grid, the seeds, the windows and the labels are IDENTICAL to
the fslot line — literally the same `fit_shared_forecast_probe_explicit_val`. The ONLY change is
which K token states are fed in: `extract_kout_features(..., slot_tokens="content_last")` slices
the K content patches immediately before the REG token instead of the K forecast slots. Sequence
layout is [content(ncp=32) | REG | forecast(K=4)], so these are the exact positional analogue.

  fslot  -> hs[:, -K:, :]            forecast slots
  cslot  -> hs[:, ncp-K:ncp, :]      the K content patches before REG   <-- this driver

STATED CAVEAT (do not drop it from any writeup): at C=512, P=16, K=4 the content slots cover
context steps 448..512 — the last 64 of 512. Pooled `content_K` sees all 512. So a cslot-vs-
content_K gap mixes readout structure with information coverage, and only the cslot-vs-fslot
comparison (same K, same head, same coverage-free question of WHICH tokens) is fully controlled.
A stride variant (K evenly spaced content patches) would close that gap; deliberately NOT built
until the last-K result says whether it is worth the GPU pass.

Namespace: results/ext_v4_future_tokens/cslot/ — disjoint from the committed fslot artifacts,
which this driver only ever READS (for the head-to-head figure).

Stages:
  --fit-ptid      GPU. Extract content-slot features (cold cache -> a real forward pass) and fit
                  the 4 PT-ID sources x 3 run seeds x 14 layers. Idempotent (skips finished seeds).
  --tunnels-only  CPU. Sustained-plateau tunnel per source from the MEAN validation curve.
  --figures       CPU. Per-source val/test curves + the cslot-vs-fslot head-to-head.

Compute discipline: --fit-ptid loads the model and fits 4 x 3 x 14 x |wd grid| heads -> sbatch,
NOT the login node. The two post-hoc stages are cache-only aggregation (login-node safe).
"""

from __future__ import annotations

import argparse
import csv
import gc
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probing import config
from probing.config import NUM_LAYERS, OUTPUT_PATCH_SIZE, SEED
from probing.extraction import extract_kout_features
from probing.id_data import build_ood_rolling_windows, build_windows
from probing.probes import (QUANTILE_SETS, PROBE_PROTOCOL_VERSION,
                            fit_shared_forecast_probe_explicit_val,
                            predict_shared_forecast_probe, validate_quantiles)
from probing.tunnel import (PT_ID_TAGS, PT_OOD_TAGS, tunnel_record_multi,
                            val_curve_from_selection)

# Import the fslot line's frozen constants + helpers rather than restating them: the whole point
# is that everything except the token slice is identical. Importing this module is side-effect
# free (it only defines constants at import time; set_dataset_set happens in ITS main()).
from experiments.run_ptood_probing_ftok import (
    C, H, K, LAYER_LABELS, OUT_ROOT, PTID_SET, QUANTILE_EPOCHS, RUN_SEEDS, RUN_TYPE,
    RUNS_TAG, SHORT, WD_GRID, _protocol_meta, _run_compatible, _save_ckpt)

SLOT_TOKENS = "content_last"      # extract_kout_features selector -> hs[:, ncp-K:ncp, :]
READOUT = "cslot"                 # filename / artifact stem (fslot's sibling)
CSLOT_ROOT = OUT_ROOT / READOUT   # results/ext_v4_future_tokens/cslot/
FSLOT_RUN_DIR = OUT_ROOT / "ptood_probing" / "ptid_runs"   # committed reference, READ-ONLY

PTID_RUN_DIR = CSLOT_ROOT / "ptid_runs"
PTID_CKPT_DIR = CSLOT_ROOT / "ptid_checkpoints"
TUNNEL_DIR = CSLOT_ROOT / "tunnels"
FIG_DIR = CSLOT_ROOT / "figures"
TAB_DIR = CSLOT_ROOT / "tables"


def _qtag(qset):
    """Filename quantile tag. Deliberately the SAME shape as the fslot line's (`q9__v2`) so a
    cslot artifact and its fslot counterpart differ only by directory — the two are meant to be
    read side by side. Protocol compatibility is still enforced by _run_compatible."""
    return f"{qset}__{PROBE_PROTOCOL_VERSION}"


def _ckpt_dir(src, qset, seed):
    return PTID_CKPT_DIR / f"{src}__{READOUT}__C{C}_H{H}__{qset}__{PROBE_PROTOCOL_VERSION}__seed{seed}"


def _tunnel_path(src, qset):
    return TUNNEL_DIR / f"{src}__{READOUT}__{qset}__{PROBE_PROTOCOL_VERSION}__{RUNS_TAG}.json"


def _mkdirs():
    for d in (PTID_RUN_DIR, PTID_CKPT_DIR, TUNNEL_DIR, FIG_DIR, TAB_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _cslot_feats(tag, split, X, y):
    """{layer: (n, K, 768)} CONTENT-slot states — the exact twin of run_ptood_probing_ftok's
    _fslot_feats, with slot_tokens='content_last'. Keys 0..NUM_LAYERS-1 = PRE-final-LN block
    states (Emb, L1..L12); key NUM_LAYERS (=13) = the POST-final-LN content slots, the analogue
    of fslot's native-head-input readout point. Writes/reads the SEPARATE cslotL_K4_H64 cache."""
    fk, final, _ = extract_kout_features(tag, split, X, y, horizon=H, slot_tokens=SLOT_TOKENS)
    feats = dict(fk["fslot"])              # "fslot" key, content tokens (see extraction docstring)
    feats[NUM_LAYERS] = final["fslot"]
    for i, arr in feats.items():
        assert np.ndim(arr) == 3 and arr.shape[1] == K, (
            f"{tag}/{split} L{i}: expected (n, {K}, 768) content slots, got {np.shape(arr)}")
    return feats


# --------------------------------------------------------------------------- #
# Stage 1 — fit the PT-ID content-slot probes (GPU)
# --------------------------------------------------------------------------- #
def fit_ptid(qset, quantiles, device):
    """Fit the 4 PT-ID sources x RUN_SEEDS. Same windows (build_windows, window seed fixed at
    SEED), same head, same wd grid, same epochs as the fslot line — only the tokens differ.
    Idempotent: a seed whose run JSON is present and protocol-compatible is skipped."""
    qtag = _qtag(qset)
    for src in PT_ID_TAGS:
        pending = [s for s in RUN_SEEDS
                   if not _run_compatible(PTID_RUN_DIR / f"{src}__{qtag}__seed{s}.json", qset)]
        if not pending:
            print(f"  [skip] {SHORT[src]}: all seeds already fit (protocol {PROBE_PROTOCOL_VERSION}, {qset})")
            continue
        w = build_windows(src)
        f_tr = _cslot_feats(src, "train", w["X_train"], w["y_train"])
        f_va = _cslot_feats(src, "val", w["X_val"], w["y_val"])
        f_te = _cslot_feats(src, "test", w["X_test"], w["y_test"])
        for seed in pending:
            print(f"\n[fit PT-ID {READOUT}] {SHORT[src]} run seed {seed} ({qset})")
            fitted = fit_shared_forecast_probe_explicit_val(
                f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"], quantiles=quantiles,
                epochs=QUANTILE_EPOCHS, wd_grid=WD_GRID, device=device, init_seed=seed)
            _save_ckpt(_ckpt_dir(src, qset, seed), fitted)
            out, diag = predict_shared_forecast_probe(
                fitted, f_te, w["Y_test_traj"], quantiles=quantiles, device=device,
                collect_test_window_loss=True)
            wl = np.stack([diag["test_window_loss"][i]
                           for i in sorted(diag["test_window_loss"])]).astype(np.float64)
            np.savez(PTID_RUN_DIR / f"{src}__{qtag}__seed{seed}.npz", window_loss=wl,
                     series_test=np.asarray(w["series_test"], np.int64))
            json.dump({"dataset": src, "quantile_set": qset, "run_seed": int(seed),
                       "run_type": RUN_TYPE, "readout": READOUT,
                       "probe_family": "shared_linear",
                       "pooling_or_token_type": "content_slot_last",
                       "slot_tokens": SLOT_TOKENS,
                       "token_positions": f"hs[:, ncp-{K}:ncp, :] (the {K} content patches before REG)",
                       "context_coverage_steps": [C - K * OUTPUT_PATCH_SIZE, C],
                       **_protocol_meta(quantiles),
                       "val_loss_by_layer": val_curve_from_selection(
                           {i: fitted[i]["selection"] for i in sorted(fitted)}, num_layers=len(fitted)),
                       "test_loss_by_layer": [float(out[i]) for i in sorted(out)]},
                      open(PTID_RUN_DIR / f"{src}__{qtag}__seed{seed}.json", "w"), indent=2)
            print(f"  [saved] {src}__{qtag}__seed{seed}.json")
        del w, f_tr, f_va, f_te
        gc.collect()


# --------------------------------------------------------------------------- #
# Stage 1b — PT-OOD content-slot TEST features (GPU, extraction only — no probe)
# --------------------------------------------------------------------------- #
def extract_ood(tags=PT_OOD_TAGS):
    """Extract the content-slot TEST features for the 3 PT-OOD targets.

    --fit-ptid covers only the 4 PT-ID sources (it fits probes, and the PT-OOD fresh-probe
    diagnostic is not part of this line). A 7-dataset CKA of the content-slot representation
    additionally needs SG Carpark / Coastal T-S / BOOM, so this stage extracts JUST their test
    split — no probe is fit and no probe artifact is written. Windows come from
    build_ood_rolling_windows with the window seed FIXED at SEED, and the '_rolling' split name
    keeps these caches disjoint from any legacy eval-only cache, exactly as the fslot line does.

    Test-only is deliberate: CKA reads one split, and the PT-OOD rolling train build (SG/BOOM,
    354 clusters) is the expensive part. Add train/val here only if a PT-OOD probe is ever wanted.
    """
    for tag in tags:
        w = build_ood_rolling_windows(tag, C=C, H=H, seed=SEED)
        m = w["meta"]
        if m["n_test"] == 0:
            raise RuntimeError(f"{tag}: empty test split — check the loader (run_ood_screen)")
        print(f"[extract-ood] {SHORT.get(tag, tag):<12} test {m['n_test']} windows "
              f"({m['n_test_clusters']} {m['cluster_unit']} clusters)")
        f = _cslot_feats(tag, "test_rolling", w["X_test"], w["y_test"])
        print(f"  [ok] {len(f)} readout points, {f[0].shape[0]} rows x K={f[0].shape[1]}")
        del w, f
        gc.collect()


# --------------------------------------------------------------------------- #
# Stage 2 — tunnels (CPU, post-hoc)
# --------------------------------------------------------------------------- #
def _runs(run_dir, src, qtag, required=True):
    """(val (n_runs,n_pts), test (n_runs,n_pts)) from a directory of per-seed run JSONs."""
    val, test = [], []
    for s in RUN_SEEDS:
        p = run_dir / f"{src}__{qtag}__seed{s}.json"
        if not p.exists():
            if required:
                raise FileNotFoundError(f"missing {p} — run --fit-ptid first")
            return None, None
        r = json.load(open(p))
        val.append(r["val_loss_by_layer"])
        test.append(r["test_loss_by_layer"])
    return np.asarray(val, float), np.asarray(test, float)


def compute_tunnels(qset):
    """Sustained-plateau tunnel per source, defined from the MEAN VALIDATION curve only (test is
    never consulted for the boundary) — identical semantics to the fslot line, so the entrance
    depths are directly comparable across the two readouts."""
    qtag = _qtag(qset)
    for src in PT_ID_TAGS:
        V, T = _runs(PTID_RUN_DIR, src, qtag)
        rec = tunnel_record_multi(src, V, T, RUN_SEEDS, run_type=RUN_TYPE,
                                  val_split_kind="explicit_temporal_val",
                                  extra={"quantile_set": qset, "readout": READOUT,
                                         "pooling_or_token_type": "content_slot_last",
                                         "slot_tokens": SLOT_TOKENS,
                                         "probe_protocol_version": PROBE_PROTOCOL_VERSION,
                                         "layer_labels": LAYER_LABELS})
        json.dump(rec, open(_tunnel_path(src, qset), "w"), indent=2)
        print(f"  [tunnel] {SHORT[src]:<12} l_start={rec['l_start']:>2} "
              f"({LAYER_LABELS[rec['l_start']]:>6})  D_ID={rec['D_ID']:+.4f}")


# --------------------------------------------------------------------------- #
# Stage 3 — figures (CPU, post-hoc)
# --------------------------------------------------------------------------- #
def _band(ax, curves, color, label, style="-", marker="o"):
    m, sd = curves.mean(axis=0), curves.std(axis=0)
    x = np.arange(curves.shape[1])
    ax.plot(x, m, style, marker=marker, ms=3.5, color=color, label=label, lw=1.6)
    ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.16, lw=0)


def make_figures(qset):
    """Two figures. (1) per-source val vs test content-slot curves (seed bands). (2) the headline
    head-to-head: content-slot vs the committed forecast-slot TEST curves on identical windows,
    same head, same protocol — the controlled 'which tokens' comparison."""
    qtag = _qtag(qset)
    srcs = [s for s in PT_ID_TAGS if (PTID_RUN_DIR / f"{s}__{qtag}__seed{RUN_SEEDS[0]}.json").exists()]
    if not srcs:
        raise FileNotFoundError(f"no cslot run artifacts for {qset} in {PTID_RUN_DIR} — "
                                "run --fit-ptid first")
    x = np.arange(NUM_LAYERS + 1)

    fig, axes = plt.subplots(1, len(srcs), figsize=(4.1 * len(srcs), 3.6), squeeze=False)
    for ax, src in zip(axes[0], srcs):
        V, T = _runs(PTID_RUN_DIR, src, qtag)
        _band(ax, V, "tab:blue", "validation")
        _band(ax, T, "tab:red", "test", style="--", marker="s")
        ax.set_title(f"{SHORT[src]} — content slots", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS, rotation=90, fontsize=6)
        ax.set_xlabel("readout point"); ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel(f"Chronos-2 quantile loss ({qset})")
    axes[0][0].legend(fontsize=7)
    fig.suptitle(f"Shared Linear(768, Q*{OUTPUT_PATCH_SIZE}) head on the K={K} content patches "
                 f"before REG (context steps {C - K * OUTPUT_PATCH_SIZE}..{C})", fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / f"cslot_by_layer__{qset}__{RUNS_TAG}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {p}")

    # --- head-to-head vs the committed forecast-slot line ------------------- #
    fig, axes = plt.subplots(1, len(srcs), figsize=(4.1 * len(srcs), 3.6), squeeze=False)
    rows, missing = [], []
    for ax, src in zip(axes[0], srcs):
        _, Tc = _runs(PTID_RUN_DIR, src, qtag)
        _, Tf = _runs(FSLOT_RUN_DIR, src, qtag, required=False)
        _band(ax, Tc, "tab:purple", "content slots")
        if Tf is None:
            missing.append(src)
        else:
            _band(ax, Tf, "tab:green", "forecast slots", style="--", marker="D")
            for i in range(Tc.shape[1]):
                rows.append({"dataset": src, "short": SHORT[src], "quantile_set": qset,
                             "layer": LAYER_LABELS[i],
                             "cslot_test": float(Tc[:, i].mean()),
                             "fslot_test": float(Tf[:, i].mean()),
                             "delta_cslot_minus_fslot": float(Tc[:, i].mean() - Tf[:, i].mean())})
        ax.set_title(SHORT[src], fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS, rotation=90, fontsize=6)
        ax.set_xlabel("readout point"); ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel(f"test quantile loss ({qset})")
    axes[0][0].legend(fontsize=7)
    fig.suptitle("Same shared head, same windows, same protocol — only the tokens differ "
                 f"(K={K}); lower is better", fontsize=9)
    fig.tight_layout()
    p = FIG_DIR / f"cslot_vs_fslot__{qset}__{RUNS_TAG}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"  [fig] {p}")
    if missing:
        print(f"  [warn] no committed fslot runs for {missing} at {qtag} — "
              f"panels show content slots only")
    if rows:
        t = TAB_DIR / f"cslot_vs_fslot__{qset}__{RUNS_TAG}.csv"
        with open(t, "w", newline="") as fh:
            wtr = csv.DictWriter(fh, list(rows[0])); wtr.writeheader(); wtr.writerows(rows)
        print(f"  [tab] {t}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--quantile-set", default="q9", choices=sorted(QUANTILE_SETS))
    p.add_argument("--fit-ptid", action="store_true",
                   help="GPU: extract content slots + fit the 4 PT-ID sources x 3 seeds")
    p.add_argument("--extract-ood", action="store_true",
                   help="GPU: content-slot TEST features for the 3 PT-OOD targets "
                        "(needed for the 7-dataset CKA; extraction only, no probe)")
    p.add_argument("--tunnels-only", action="store_true", help="CPU: tunnels from saved runs")
    p.add_argument("--figures", action="store_true", help="CPU: curves + cslot-vs-fslot")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    config.set_dataset_set(PTID_SET)          # roster + rolling windows + cache namespace
    _mkdirs()
    qset = a.quantile_set
    quantiles = validate_quantiles(QUANTILE_SETS[qset])
    print(f"[run_content_slot_probing] readout={READOUT}  slot_tokens={SLOT_TOKENS}  "
          f"{qset}  C={C} H={H} K={K}  seeds={list(RUN_SEEDS)}")
    if not (a.fit_ptid or a.extract_ood or a.tunnels_only or a.figures):
        raise SystemExit("choose a stage: --fit-ptid (GPU) / --extract-ood (GPU) / "
                         "--tunnels-only / --figures")
    if a.fit_ptid:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  device={device}")
        fit_ptid(qset, quantiles, device)
    if a.extract_ood:
        extract_ood()
    if a.tunnels_only:
        compute_tunnels(qset)
    if a.figures:
        make_figures(qset)
    print("=== DONE ===")


if __name__ == "__main__":
    main()
