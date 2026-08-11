"""Pretraining-OOD layerwise probing under PT-ID validation-defined tunnels (2023 protocol).

Framing: domain status is relative to the BACKBONE (probing.tunnel). The 4 extended_v3_rolling
datasets are PT-ID (in Chronos-2's pretraining corpus); sg_carpark / coastal_ts / boom_hourly
are PT-OOD. This driver replaces single "best layer" selection with a per-dataset TUNNEL RANGE
and — unlike the legacy zero-shot readout transfer in run_ood_pretrain_transfer.py, which is
kept separately as a diagnostic — trains a FRESH probe per PT-OOD dataset per layer.

Every reported condition uses 3 INDEPENDENT PROBE RUNS (RUN_SEEDS = 0, 1, 2; the backbone is
frozen, so the run seed controls the probe's Linear init — the only randomness in the
deterministic full-batch fit). Windows and features are run-seed-independent, so all runs share
identical splits/caches, curves average layerwise, and per-window losses average window-wise.
Tunnels are defined from the MEAN validation curve (never by averaging per-seed tunnel
indices); individual run curves are retained in every record for variability plots.

  Stage 1a (--fit-ptid-seeds; warm caches, GPU/CPU): fit the PT-ID rolling probes for run
    seeds 1 and 2 (seed 0 = the committed extended_v3_rolling checkpoints, reused as run 1)
    and score their own test split — per-seed val/test curves + per-window losses.
  Stage 1b (--tunnels-only; CPU): per PT-ID source, average the 3 temporal-validation curves;
    first-crossing (95%) tunnel l_start = min{l : mean_val(l) <= 1.05*mean_val(last)}
    (first layer within 5% of final-layer quality), excursion
    M = max_{j>=l_start}(mean_loss(j)/mean_loss(last) - 1) (<= tol on val; informative on test).
    Each dataset defines its OWN tunnel; the boundary is frozen, checked on the mean test
    curve, and D_ID(s) = (mean_test(last) - mean_test(l_s))/mean_test(l_s) gets a
    series-cluster-bootstrap CI on the seed-averaged window losses.
  Stage 2 (GPU): per PT-OOD target x run seed, build the rolling-origin train/val/test split
    ON the target (id_data.build_ood_rolling_windows), train fresh per-layer probes on
    target-train with wd selected on target-VAL only (fit_quantile_probe_explicit_val; the
    target test set never tunes anything), and score target-test at every layer.
  Stage 3 (--figure-only; CPU): overlay each PT-ID tunnel on each PT-OOD mean test curve and
    compute D_OOD(s,t) at source s's l_start plus Delta(s,t) = D_OOD(s,t) - D_ID(s)
    (independent-replicate bootstrap difference — disjoint test sets, so Delta cannot be
    paired). PT-OOD datasets NEVER define their own tunnel here.

Run:
    python -m experiments.run_ptood_probing --fit-ptid-seeds      # PT-ID runs 2-3 (warm cache)
    python -m experiments.run_ptood_probing --tunnels-only        # PT-ID tunnels (CPU)
    python -m experiments.run_ptood_probing --targets sg_carpark  # one target, all 3 runs (GPU)
    python -m experiments.run_ptood_probing --figure-only         # aggregate (CPU)
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
from probing.config import NUM_LAYERS, LAST_LAYER, SEED
from probing.extraction import extract_window_features
from probing.id_data import build_ood_rolling_windows, build_windows
from probing.probes import (QUANTILE_SETS, fit_quantile_probe_explicit_val,
                            predict_quantile_probe, validate_quantiles)
from probing.tunnel import (PT_ID_TAGS, PT_OOD_TAGS, TUNNEL_TOL, d_stat_boot, delta_stat,
                            domain_status, max_excursion, tunnel_record_multi,
                            val_curve_from_selection)

# fixed frame, inherited unchanged from the rolling 4x4 (run_ood_transfer)
PTID_SET = "extended_v3_rolling"   # the PT-ID experiment whose checkpoints define the tunnels
POOLING = "content"
C, H = 512, 64
QUANTILE_EPOCHS = 300
WD_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
RUN_SEEDS = (0, 1, 2)              # 3 independent probe-init runs (backbone frozen); seed 0 is
RUN_TYPE = "probe_seed"            # the committed single-seed run, reused verbatim as run 1
RUNS_TAG = "runs" + "-".join(str(s) for s in RUN_SEEDS)

SHORT = {"monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber",
         "m4_hourly": "M4", "wind_farms_hourly": "WindFarms",
         "sg_carpark": "SG Carpark", "coastal_ts": "Coastal T-S", "boom_hourly": "BOOM"}


def _derive_dirs():
    global OUT_DIR, TUNNEL_DIR, PER_TARGET_DIR, BOOT_IN_DIR, CKPT_DIR, FIG_DIR, PTID_RUN_DIR, \
        PTID_BOOT_IN, PTID_CKPT
    OUT_DIR = config.ID_OUT_DIR / "ptood_probing"
    TUNNEL_DIR = config.ID_OUT_DIR / "tunnels"          # PT-ID artifact, portable by source tag
    PER_TARGET_DIR = OUT_DIR / "per_target"
    BOOT_IN_DIR = OUT_DIR / "bootstrap_inputs"
    CKPT_DIR = OUT_DIR / "checkpoints"
    FIG_DIR = OUT_DIR / "figures"
    PTID_RUN_DIR = OUT_DIR / "ptid_runs"                # seed-1/2 PT-ID refits (seed 0 = legacy)
    # the PT-ID rolling run's seed-0 artifacts (val curves + diagonal per-window test losses)
    PTID_BOOT_IN = config.ID_OUT_DIR / "ood_transfer" / "bootstrap_inputs"
    PTID_CKPT = config.ID_OUT_DIR / "ood_transfer" / "checkpoints"
    for d in (OUT_DIR, TUNNEL_DIR, PER_TARGET_DIR, BOOT_IN_DIR, CKPT_DIR, FIG_DIR, PTID_RUN_DIR):
        d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# PT-ID per-run curves: seed 0 from the committed rolling artifacts, 1/2 from Stage 1a
# --------------------------------------------------------------------------- #
def _ptid_seed0_val_curve(src, qset):
    """Per-layer temporal-VALIDATION loss (min over the wd grid) from the rolling checkpoints."""
    d = PTID_CKPT / f"{src}__{POOLING}__C{C}_H{H}__{qset}__seed0"
    sel = {}
    for i in range(NUM_LAYERS):
        p = d / f"L{i:02d}.pt"
        if not p.exists():
            raise FileNotFoundError(f"missing PT-ID checkpoint {p} — run the {PTID_SET} 4x4 first")
        sel[i] = torch.load(p, map_location="cpu", weights_only=False)["selection"]
    return val_curve_from_selection(sel)


def _ptid_run_curves(src, qset, seed):
    """(val_curve list, window_loss (NUM_LAYERS, n), series_test) for ONE PT-ID probe run."""
    if seed == 0:
        p = PTID_BOOT_IN / f"{src}__to__{src}__{qset}__seed0.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing PT-ID diagonal bootstrap inputs {p}")
        z = np.load(p)
        return _ptid_seed0_val_curve(src, qset), z["window_loss"], z["series_test"]
    pj = PTID_RUN_DIR / f"{src}__{qset}__seed{seed}.json"
    pz = PTID_RUN_DIR / f"{src}__{qset}__seed{seed}.npz"
    if not (pj.exists() and pz.exists()):
        raise FileNotFoundError(f"missing PT-ID run artifacts for seed {seed} ({pj.name}) — "
                                "run --fit-ptid-seeds first")
    z = np.load(pz)
    return json.load(open(pj))["val_loss_by_layer"], z["window_loss"], z["series_test"]


def fit_ptid_seeds(qset, quantiles, device):
    """Stage 1a: PT-ID probe runs for the non-legacy seeds (identical windows/features/wd grid
    as the committed seed-0 rolling run; only the probe init differs)."""
    for src in PT_ID_TAGS:
        w = build_windows(src)                 # rolling split (window seed fixed at SEED)
        f_tr, _ = extract_window_features(src, "train", w["X_train"], w["y_train"], pooling=POOLING)
        f_va, _ = extract_window_features(src, "val", w["X_val"], w["y_val"], pooling=POOLING)
        f_te, _ = extract_window_features(src, "test", w["X_test"], w["y_test"], pooling=POOLING)
        for seed in RUN_SEEDS:
            if seed == 0:
                continue                       # legacy checkpoints ARE run 1
            pj = PTID_RUN_DIR / f"{src}__{qset}__seed{seed}.json"
            if pj.exists():
                print(f"  [skip] {SHORT[src]} seed {seed}: already fit ({pj.name})")
                continue
            print(f"\n[fit PT-ID] {SHORT[src]} run seed {seed} ({qset})")
            fitted = fit_quantile_probe_explicit_val(
                f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"], quantiles=quantiles,
                epochs=QUANTILE_EPOCHS, wd_grid=WD_GRID, device=device, init_seed=seed)
            out, diag = predict_quantile_probe(fitted, f_te, w["Y_test_traj"],
                                               quantiles=quantiles, device=device,
                                               collect_test_window_loss=True)
            wl = np.stack([diag["test_window_loss"][i] for i in range(NUM_LAYERS)]).astype(np.float64)
            np.savez(PTID_RUN_DIR / f"{src}__{qset}__seed{seed}.npz", window_loss=wl,
                     series_test=np.asarray(w["series_test"], np.int64))
            json.dump({"dataset": src, "quantile_set": qset, "run_seed": int(seed),
                       "run_type": RUN_TYPE, "pooling": POOLING,
                       "val_loss_by_layer": val_curve_from_selection(
                           {i: fitted[i]["selection"] for i in range(NUM_LAYERS)}),
                       "test_loss_by_layer": [float(out[i]) for i in range(NUM_LAYERS)]},
                      open(pj, "w"), indent=2)
            print(f"  [saved] {pj.name}")
        del w, f_tr, f_va, f_te
        gc.collect()


def _tunnel_path(src, qset):
    return TUNNEL_DIR / f"{src}__{POOLING}__{qset}__{RUNS_TAG}.json"


def _seed_mean_windows(curves_by_seed):
    """Seed-averaged per-window losses (same windows across runs — asserted)."""
    wls = [wl for _, wl, _ in curves_by_seed]
    sids = [sid for _, _, sid in curves_by_seed]
    assert all(w.shape == wls[0].shape for w in wls), "runs must share identical test windows"
    assert all(np.array_equal(s, sids[0]) for s in sids), "runs must share identical series ids"
    return np.mean(wls, axis=0), sids[0]


def compute_ptid_tunnels(qset):
    """Stage 1b: one tunnel per PT-ID dataset from the MEAN validation curve over RUN_SEEDS
    (independently per dataset; never pooled across datasets, never per-seed indices averaged),
    frozen, then checked + D_ID-quantified on the mean test curve. The D_ID CI is the cluster
    bootstrap of the seed-averaged window losses (sampling variance; seed variance is reported
    separately via the retained per-run curves)."""
    for src in PT_ID_TAGS:
        runs = [_ptid_run_curves(src, qset, s) for s in RUN_SEEDS]
        val_by_run = [v for v, _, _ in runs]
        test_by_run = [wl.mean(axis=1) for _, wl, _ in runs]
        wl_mean, sid = _seed_mean_windows(runs)
        rec = tunnel_record_multi(src, val_by_run, test_by_run, RUN_SEEDS, run_type=RUN_TYPE,
                                  val_split_kind="temporal_rolling",
                                  extra={"quantile_set": qset, "pooling": POOLING,
                                         "dataset_set": PTID_SET,
                                         "provenance": {"seed0": str(PTID_CKPT),
                                                        "seed1plus": str(PTID_RUN_DIR)}})
        d_id = d_stat_boot(wl_mean, sid, rec["l_start"], B=config.BOOT_B, seed=SEED)
        rec["d_id_ci"] = list(d_id["ci"])
        rec["d_id_by_run"] = [float((t[-1] - t[rec["l_start"]]) / t[rec["l_start"]])
                              for t in test_by_run]
        rec["n_clusters"] = d_id["n_clusters"]; rec["n_windows"] = d_id["n_windows"]
        out = _tunnel_path(src, qset)
        json.dump(rec, open(out, "w"), indent=2)
        print(f"  [{SHORT[src]:>12}] tunnel [L{rec['l_start']}, L{LAST_LAYER}]  "
              f"(first crossing 95%)  M_test={rec['M_test']:+.3f}  "
              f"D_ID={rec['D_ID']:+.3f} [{d_id['ci'][0]:+.3f}, {d_id['ci'][1]:+.3f}] "
              f"per-run {['%+.3f' % d for d in rec['d_id_by_run']]} -> {out.name}")
        _tunnel_figure(rec)


def _plot_runs(ax, curves_by_run, mean, std, color, label):
    """Mean curve + shaded +-1 std band + faint individual run curves."""
    x = np.arange(NUM_LAYERS)
    for c in curves_by_run:
        ax.plot(x, c, "-", color=color, alpha=0.25, lw=0.9)
    ax.fill_between(x, np.array(mean) - np.array(std), np.array(mean) + np.array(std),
                    color=color, alpha=0.15)
    ax.plot(x, mean, "o-", color=color, label=label)


def _tunnel_figure(rec):
    """PT-ID mean val + test curves (3-run mean +- std, faint per-run); first-crossing (95%) tunnel
    shaded, excursion M in the title."""
    src, ls = rec["dataset"], rec["l_start"]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(NUM_LAYERS)
    ax.axvspan(ls - 0.25, LAST_LAYER + 0.25, color="tab:green", alpha=0.12,
               label=f"tunnel (first crossing 95%) [L{ls}, L{LAST_LAYER}]")
    _plot_runs(ax, rec["val_loss_by_run"], rec["mean_val_loss_by_layer"],
               rec["std_val_loss_by_layer"], "tab:blue",
               f"validation, mean of {len(rec['run_seeds'])} runs (defines tunnel)")
    _plot_runs(ax, rec["test_loss_by_run"], rec["mean_test_loss_by_layer"],
               rec["std_test_loss_by_layer"], "tab:orange",
               f"test, mean of {len(rec['run_seeds'])} runs (tunnel frozen)")
    ax.axhline((1 + rec["tolerance"]) * rec["mean_val_loss_by_layer"][-1], color="tab:blue",
               ls=":", lw=1, label=f"{1 + rec['tolerance']:.2f} x final-layer mean val loss")
    ax.set_xticks(x); ax.set_xticklabels(["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)])
    ax.set_xlabel("layer"); ax.set_ylabel("Chronos-2 quantile loss")
    ax.set_title(f"{SHORT[src]} (PT-ID): first-crossing (95%) tunnel from MEAN validation curve  "
                 f"(M_test={rec['M_test']:+.2f}, D_ID={rec['D_ID']:+.3f})", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / f"ptid_tunnel__{src}__{rec['quantile_set']}__{RUNS_TAG}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"    [saved] {out.name}")


# --------------------------------------------------------------------------- #
# Stage 2 — fresh per-layer probes ON each PT-OOD target, one fit per run seed (GPU)
# --------------------------------------------------------------------------- #
def eval_target(tag, qset, quantiles, seed, device):
    """Rolling split on the PT-OOD target; probes trained on target-train, wd on target-VAL,
    scored on target-test. The target's test set never tunes anything. `seed` = probe init."""
    print(f"\n[eval PT-OOD] {SHORT.get(tag, tag)} ({qset}, run seed {seed})")
    w = build_ood_rolling_windows(tag, C=C, H=H, seed=SEED)     # window seed FIXED across runs
    m = w["meta"]
    if m["n_test"] == 0 or m["n_train"] == 0:
        raise RuntimeError(f"{tag}: empty split (train {m['n_train']} / test {m['n_test']}) — "
                           "check the loader / rolling eligibility")
    print(f"  windows: train {m['n_train']} / val {m['n_val']} / test {m['n_test']}  "
          f"({m['n_test_clusters']} {m['cluster_unit']} clusters)")
    # "*_rolling" split names keep these caches disjoint from the legacy zero-shot readout
    # transfer's eval-only test cache (same tag, different windowing -> would fail loud).
    f_tr, _ = extract_window_features(tag, "train_rolling", w["X_train"], w["y_train"], pooling=POOLING)
    f_va, _ = extract_window_features(tag, "val_rolling", w["X_val"], w["y_val"], pooling=POOLING)
    f_te, _ = extract_window_features(tag, "test_rolling", w["X_test"], w["y_test"], pooling=POOLING)
    fitted = fit_quantile_probe_explicit_val(f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"],
                                             quantiles=quantiles, epochs=QUANTILE_EPOCHS,
                                             wd_grid=WD_GRID, device=device, init_seed=seed)
    ckdir = CKPT_DIR / f"{tag}__{POOLING}__C{C}_H{H}__{qset}__seed{seed}"
    ckdir.mkdir(parents=True, exist_ok=True)
    for i, f in fitted.items():
        torch.save({"state_dict": f["linear"].state_dict(), "scaler_mean": f["scaler"].mean_,
                    "scaler_scale": f["scaler"].scale_, "wd": f["wd"],
                    "selection": f["selection"], "in_features": f["in_features"],
                    "out_features": f["out_features"]}, ckdir / f"L{i:02d}.pt")

    out, diag = predict_quantile_probe(fitted, f_te, w["Y_test_traj"], quantiles=quantiles,
                                       device=device, collect_test_window_loss=True)
    wl = np.stack([diag["test_window_loss"][i] for i in range(NUM_LAYERS)]).astype(np.float64)
    sid = np.asarray(w["series_test"], np.int64)
    np.savez(BOOT_IN_DIR / f"{tag}__{qset}__seed{seed}.npz", window_loss=wl, series_test=sid)

    val_curve = val_curve_from_selection({i: fitted[i]["selection"] for i in range(NUM_LAYERS)})
    payload = {
        "dataset": tag, "domain_status": domain_status(tag), "quantile_set": qset,
        "run_seed": int(seed), "run_type": RUN_TYPE, "pooling": POOLING,
        "protocol": "fresh_probe_2023", "val_split_kind": "temporal_rolling",
        "test_loss_by_layer": [float(out[i]) for i in range(NUM_LAYERS)],
        "val_loss_by_layer": val_curve,
        "chosen_wd_by_layer": {str(i): float(fitted[i]["wd"]) for i in range(NUM_LAYERS)},
        "meta": {k: v for k, v in m.items() if k != "notes"}, "notes": m["notes"],
    }
    outp = PER_TARGET_DIR / f"{tag}__{qset}__seed{seed}.json"
    json.dump(payload, open(outp, "w"), indent=2)
    ql = np.array(payload["test_loss_by_layer"])
    print(f"  test loss: last L{LAST_LAYER}={ql[-1]:.3f}  min L{int(ql.argmin())}={ql.min():.3f}"
          f"  -> {outp.name}")
    del w, f_tr, f_va, f_te
    gc.collect()


# --------------------------------------------------------------------------- #
# Stage 3 — apply PT-ID tunnels to PT-OOD mean curves; D_OOD + Delta (CPU)
# --------------------------------------------------------------------------- #
def _load_tunnels(qset):
    recs = {}
    for src in PT_ID_TAGS:
        p = _tunnel_path(src, qset)
        if not p.exists():
            raise FileNotFoundError(f"missing tunnel record {p} — run --tunnels-only first")
        recs[src] = json.load(open(p))
    return recs


def _load_ptood_runs(tgt, qset):
    """All available run seeds for one PT-OOD target -> (payloads, (val, wl, sid) per run).
    Returns None when no run is complete; warns if fewer than len(RUN_SEEDS)."""
    payloads, runs = [], []
    for seed in RUN_SEEDS:
        pj = PER_TARGET_DIR / f"{tgt}__{qset}__seed{seed}.json"
        pz = BOOT_IN_DIR / f"{tgt}__{qset}__seed{seed}.npz"
        if not (pj.exists() and pz.exists()):
            continue
        z = np.load(pz)
        payloads.append(json.load(open(pj)))
        runs.append((payloads[-1]["val_loss_by_layer"], z["window_loss"], z["series_test"]))
    if not runs:
        return None
    if len(runs) < len(RUN_SEEDS):
        print(f"  [warn] {tgt}: only {len(runs)}/{len(RUN_SEEDS)} runs present — "
              "aggregating what exists")
    return payloads, runs


def aggregate(qset):
    tunnels = _load_tunnels(qset)
    # PT-ID seed-averaged window losses (replicate vectors needed for Delta; deterministic)
    d_id = {}
    for src in PT_ID_TAGS:
        wl_mean, sid = _seed_mean_windows([_ptid_run_curves(src, qset, s) for s in RUN_SEEDS])
        d_id[src] = d_stat_boot(wl_mean, sid, tunnels[src]["l_start"], B=config.BOOT_B, seed=SEED)

    rows, targets = [], []
    for tgt in PT_OOD_TAGS:
        loaded = _load_ptood_runs(tgt, qset)
        if loaded is None:
            print(f"  [skip] {tgt}: no per-target results yet")
            continue
        payloads, runs = loaded
        test_by_run = [wl.mean(axis=1) for _, wl, _ in runs]
        wl_mean, sid = _seed_mean_windows(runs)
        mean_test = wl_mean.mean(axis=1)
        targets.append((tgt, payloads, runs))
        for src in PT_ID_TAGS:
            ls = tunnels[src]["l_start"]
            d_ood = d_stat_boot(wl_mean, sid, ls, B=config.BOOT_B, seed=SEED)
            dl = delta_stat(d_ood, d_id[src])
            rows.append({
                "source_dataset": src, "target_dataset": tgt, "quantile_set": qset,
                "run_type": RUN_TYPE, "run_seeds": " ".join(str(s) for s in RUN_SEEDS),
                "n_runs_present": len(runs),
                "tolerance": TUNNEL_TOL, "l_start": ls,
                "tunnel": f"[L{ls}, L{LAST_LAYER}]",
                "ptid_m_test": round(tunnels[src]["M_test"], 6),
                "ptood_m_test": round(max_excursion(mean_test, ls), 6),
                "ptid_test_criterion_holds": tunnels[src]["test_criterion_holds"],
                "d_id": round(d_id[src]["point"], 6),
                "d_id_ci_lo": round(d_id[src]["ci"][0], 6),
                "d_id_ci_hi": round(d_id[src]["ci"][1], 6),
                "d_id_by_run": " ".join(f"{d:+.4f}" for d in tunnels[src]["d_id_by_run"]),
                "d_ood": round(d_ood["point"], 6),
                "d_ood_ci_lo": round(d_ood["ci"][0], 6),
                "d_ood_ci_hi": round(d_ood["ci"][1], 6),
                "d_ood_by_run": " ".join(f"{(t[-1] - t[ls]) / t[ls]:+.4f}" for t in test_by_run),
                "delta": round(dl["point"], 6),
                "delta_ci_lo": round(dl["ci"][0], 6),
                "delta_ci_hi": round(dl["ci"][1], 6),
                "delta_excludes_zero": bool(dl["ci"][0] > 0 or dl["ci"][1] < 0),
                "n_ood_clusters": d_ood["n_clusters"], "n_ood_windows": d_ood["n_windows"],
                "delta_ci_note": "independent-replicate bootstrap difference (disjoint test "
                                 "sets); point stats on 3-run mean curves",
            })
    if not rows:
        print("  [aggregate] no PT-OOD results found — run the GPU eval first")
        return
    base = OUT_DIR / f"tunnel_effect_stats__{qset}__{RUNS_TAG}"
    with open(f"{base}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader(); wr.writerows(rows)
    json.dump(rows, open(f"{base}.json", "w"), indent=2)
    print(f"  [saved] {base}.csv/.json ({len(rows)} source x target cells)")

    for tgt, payloads, runs in targets:
        _target_overlay_figure(tgt, runs, tunnels, qset)
    _delta_heatmap(rows, qset)
    _print_summary(rows)


def _target_overlay_figure(tgt, runs, tunnels, qset):
    """PT-OOD mean layerwise test curve (fresh probes, 3-run mean +- std + faint runs) with
    all four PT-ID tunnel starts."""
    T = np.stack([wl.mean(axis=1) for _, wl, _ in runs])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _plot_runs(ax, T, T.mean(axis=0), T.std(axis=0), "tab:orange",
               f"PT-OOD test loss, mean of {len(runs)} fresh-probe runs")
    colors = plt.cm.tab10(np.linspace(0, 0.4, len(PT_ID_TAGS)))
    for c, src in zip(colors, PT_ID_TAGS):
        ls = tunnels[src]["l_start"]
        ax.axvline(ls, color=c, ls="--", lw=1.2, label=f"{SHORT[src]} l_start = L{ls}")
    x = np.arange(NUM_LAYERS)
    ax.set_xticks(x); ax.set_xticklabels(["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)])
    ax.set_xlabel("layer"); ax.set_ylabel("Chronos-2 quantile loss")
    ax.set_title(f"{SHORT.get(tgt, tgt)} (PT-OOD): fresh layerwise probes under the four "
                 "PT-ID mean-validation tunnels", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out = FIG_DIR / f"ptood_curve_with_tunnels__{tgt}__{qset}__{RUNS_TAG}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [saved] {out.name}")


def _delta_heatmap(rows, qset):
    """Delta(s,t) = D_OOD - D_ID heatmap; * = independent-bootstrap CI excludes zero."""
    tgts = sorted({r["target_dataset"] for r in rows}, key=list(PT_OOD_TAGS).index)
    M = np.full((len(PT_ID_TAGS), len(tgts)), np.nan)
    by = {(r["source_dataset"], r["target_dataset"]): r for r in rows}
    fig, ax = plt.subplots(figsize=(1.9 * len(tgts) + 2.6, 4.2))
    for i, s in enumerate(PT_ID_TAGS):
        for j, t in enumerate(tgts):
            r = by[(s, t)]
            M[i, j] = 100 * r["delta"]
            star = "*" if r["delta_excludes_zero"] else ""
            ax.text(j, i, f"{100*r['delta']:+.1f}%{star}\n(L{r['l_start']})",
                    ha="center", va="center", fontsize=8)
    vmax = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(tgts))); ax.set_xticklabels([SHORT.get(t, t) for t in tgts])
    ax.set_yticks(range(len(PT_ID_TAGS)))
    ax.set_yticklabels([SHORT[s] for s in PT_ID_TAGS])
    ax.set_xlabel("PT-OOD target (fresh probes)"); ax.set_ylabel("PT-ID source (tunnel)")
    ax.set_title(f"Delta(s,t) = D_OOD - D_ID at the source tunnel start  [{qset}, {RUNS_TAG}]\n"
                 "positive = late-layer degradation stronger PT-OOD; * = 95% CI excludes 0",
                 fontsize=9)
    fig.colorbar(im, ax=ax, label="Delta (pct points)")
    fig.tight_layout()
    out = FIG_DIR / f"delta_heatmap__{qset}__{RUNS_TAG}.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"  [saved] {out.name}")


def _print_summary(rows):
    print("\n  == tunnel-effect summary (3-run means; D > 0: final layer worse than tunnel "
          "entrance) ==")
    for r in rows:
        print(f"    {SHORT[r['source_dataset']]:>12} (L{r['l_start']}) -> "
              f"{SHORT.get(r['target_dataset'], r['target_dataset']):<12} "
              f"D_ID={r['d_id']:+.3f}  D_OOD={r['d_ood']:+.3f}  "
              f"Delta={r['delta']:+.3f} [{r['delta_ci_lo']:+.3f}, {r['delta_ci_hi']:+.3f}]"
              f"{' *' if r['delta_excludes_zero'] else ''}")


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--quantile-set", default="q9", choices=sorted(QUANTILE_SETS))
    p.add_argument("--targets", nargs="*", default=list(PT_OOD_TAGS),
                   choices=list(PT_OOD_TAGS))
    p.add_argument("--fit-ptid-seeds", action="store_true",
                   help="fit the PT-ID rolling probes for run seeds 1/2 (warm caches) and exit")
    p.add_argument("--tunnels-only", action="store_true",
                   help="compute the PT-ID tunnel records + figures from all runs (CPU) and exit")
    p.add_argument("--figure-only", action="store_true",
                   help="aggregate saved per-target results into stats + figures (CPU)")
    return p.parse_args(argv)


def main():
    args = _parse_args()
    config.set_dataset_set(PTID_SET)     # tunnels + outputs live in the rolling namespace
    _derive_dirs()
    quantiles = validate_quantiles(QUANTILE_SETS[args.quantile_set])
    print(f"[run_ptood_probing] set={PTID_SET}  qset={args.quantile_set}  {RUNS_TAG}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.fit_ptid_seeds:
        fit_ptid_seeds(args.quantile_set, quantiles, device)
        return
    compute_ptid_tunnels(args.quantile_set)
    if args.tunnels_only:
        return
    if not args.figure_only:
        for tgt in args.targets:
            for seed in RUN_SEEDS:
                eval_target(tgt, args.quantile_set, quantiles, seed, device)
    aggregate(args.quantile_set)


if __name__ == "__main__":
    main()
