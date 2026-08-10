"""Main paper figure package for the PT-ID / PT-OOD tunnel analysis.

Each figure answers ONE question; the visual progression is:
  Fig 1  Where does PT-ID performance saturate?           (4 panels, test loss + tunnel)
  Fig 2  What happens geometrically after saturation?     (4 panels, effective rank, same tunnel)
  Fig 3  What happens to PT-OOD usefulness across depth?  (3 panels, fresh-probe test loss)
  Fig 4  Is late-layer degradation stronger PT-OOD?       (4x3 Delta heatmap, bootstrap CIs)

This script only READS existing outputs (tunnel records, ptid_runs, PT-OOD per-target results,
spectral records) and renders/aggregates — no probe fits, no extraction, no SVDs. Missing
inputs skip that figure with a clear message naming the producing command. Conventions shared
across all figures: dataset order = probing.tunnel.PT_ID_TAGS / PT_OOD_TAGS, layer labels
Emb, L1..L12, tunnel = sustained-plateau boundary from the MEAN PT-ID validation curve
(never test, never PT-OOD), 3-run means with +-1 std variability, B=5000 paired cluster
bootstrap for test inference, effective rank in actual units (not normalized).

Run (login node, CPU, after the compute stages exist):
    python -m experiments.make_paper_figures                 # everything that has inputs
"""

from __future__ import annotations

import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probing import config
from probing.config import NUM_LAYERS, SEED   # last index is data-driven per readout (post-LN=13 for fslot)
from probing.tunnel import (PT_ID_TAGS, PT_OOD_TAGS, d_stat_boot, delta_stat, m_stat_boot,
                            tunnel_start)
from experiments import run_ptood_probing as ptood
from experiments.run_ptood_probing import (RUN_SEEDS, RUNS_TAG, SHORT, _load_ptood_runs,
                                           _ptid_run_curves, _seed_mean_windows, _tunnel_path)

# readout-mutable module state (set in main from --readout); content = byte-identical default.
# For --readout fslot, main rebinds `ptood` + its dir/loader helpers to run_ptood_probing_ftok,
# so every figure reads the v4 (ext_v4_future_tokens) outputs with ZERO call-site changes (the
# ftok driver exposes identically-named helpers). _results_root() then routes paper_figures /
# spectral reads to that module's output root.
READOUT = "content"
READOUT_TAG = "content"                  # spectral-record filename tag: "content" | "fslot"
READOUT_LABEL = "content-pooled"         # figure wording

TOL_GRID = (0.02, 0.05, 0.08, 0.10)      # threshold sensitivity; PRIMARY stays 0.05
# 14th label = the post-final-LN native-head input (fslot only); content records are 13 -> sliced away.
FULL_LABELS = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)] + ["L12+LN"]
TUNNEL_KW = dict(color="tab:green", alpha=0.12)


def _labels(n):
    """x-axis labels for n readout points (data-driven): 13 = content, 14 = fslot (+post-LN)."""
    return FULL_LABELS[:n]


def _results_root():
    """v3 (config.ID_OUT_DIR) for content; the ftok module's v4 OUT_ROOT for fslot."""
    return getattr(ptood, "OUT_ROOT", config.ID_OUT_DIR)


def _derive_dirs():
    global PAPER_DIR
    ptood._derive_dirs()
    PAPER_DIR = _results_root() / "paper_figures"
    PAPER_DIR.mkdir(parents=True, exist_ok=True)


def _spectral_record(tag, split="train"):
    p = _results_root() / "spectral" / f"spectral__{tag}__{READOUT_TAG}__probe_input__{split}.json"
    return json.load(open(p)) if p.exists() else None


def _tunnels(qset):
    recs = {}
    for src in PT_ID_TAGS:
        p = _tunnel_path(src, qset)
        if not p.exists():
            return None
        recs[src] = json.load(open(p))
    return recs


def _save(fig, name):
    out = PAPER_DIR / name
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"  [saved] {out.name}")


def _mark_tunnel(ax, ls, last_idx, label=True):
    ax.axvspan(ls - 0.25, last_idx + 0.25,
               label="validation-defined tunnel" if label else None, **TUNNEL_KW)
    ax.axvline(ls, color="tab:green", lw=1.4)


def _layer_axis(ax, n):
    ax.set_xticks(np.arange(n))
    ax.set_xticklabels(_labels(n), fontsize=7, rotation=45)


# --------------------------------------------------------------------------- #
# panel painters (reused by the single figures AND the combined 2x4 figure)
# --------------------------------------------------------------------------- #
def _panel_ptid_loss(ax, rec, show_runs=False):
    mt = np.array(rec["mean_test_loss_by_layer"])
    st = np.array(rec["std_test_loss_by_layer"])
    n = len(mt); last = n - 1                     # data-driven; last = post-LN reference for fslot
    x = np.arange(n)
    _mark_tunnel(ax, rec["l_start"], last)
    if show_runs:
        for run in rec["test_loss_by_run"]:
            ax.plot(x, run, "-", color="tab:orange", alpha=0.3, lw=0.9)
    ax.fill_between(x, mt - st, mt + st, color="tab:orange", alpha=0.2,
                    label=f"±1 std over {len(rec['run_seeds'])} probe runs")
    ax.plot(x, mt, "o-", color="tab:orange", ms=3.5, label="mean test loss")
    ax.plot(last, mt[-1], "s", color="k", ms=5, label="final layer")
    lbl = _labels(n)
    ax.set_title(f"{SHORT[rec['dataset']]}   tunnel [{lbl[rec['l_start']]}, {lbl[last]}]", fontsize=9)
    _layer_axis(ax, n)


def _panel_erank(ax, spec, ls):
    v = np.array([L["effective_rank"] for L in spec["layers"]])
    lo = np.array([L["subsample_ci"][0] for L in spec["layers"]])
    hi = np.array([L["subsample_ci"][1] for L in spec["layers"]])
    n = len(v); x = np.arange(n)
    if ls is not None:
        _mark_tunnel(ax, ls, n - 1)
    ax.fill_between(x, lo, hi, color="tab:purple", alpha=0.2, label="subsampling 95% interval")
    ax.plot(x, v, "o-", color="tab:purple", ms=3.5, label="effective rank")
    ax.set_title(f"{SHORT[spec['dataset']]}", fontsize=9)
    _layer_axis(ax, n)


# --------------------------------------------------------------------------- #
# Figures 1, 2, combined
# --------------------------------------------------------------------------- #
def fig1_ptid_loss(tunnels):
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
    for ax, src in zip(axes, PT_ID_TAGS):
        _panel_ptid_loss(ax, tunnels[src])
    axes[0].set_ylabel("Chronos-2 quantile loss (test)")
    axes[0].legend(fontsize=6, loc="upper right")
    fig.suptitle("PT-ID layerwise test loss — tunnel defined on the MEAN validation curve "
                 "(first crossing of 1.05 x final-layer val loss); 3 probe runs", fontsize=10)
    _save(fig, "fig1_ptid_test_loss.png")


def fig2_ptid_erank(tunnels, specs):
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
    for ax, src in zip(axes, PT_ID_TAGS):
        _panel_erank(ax, specs[src], tunnels[src]["l_start"])
    axes[0].set_ylabel("effective rank  exp(H)")
    axes[0].legend(fontsize=6, loc="upper right")
    fig.suptitle(f"PT-ID effective rank of the probe input ({READOUT_LABEL}) — SAME "
                 "validation-defined tunnel as Fig. 1 (rank defines nothing)", fontsize=10)
    _save(fig, "fig2_ptid_effective_rank.png")


def fig_combined(tunnels, specs):
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4), sharex="col")
    for j, src in enumerate(PT_ID_TAGS):
        _panel_ptid_loss(axes[0, j], tunnels[src])
        _panel_erank(axes[1, j], specs[src], tunnels[src]["l_start"])
        axes[1, j].set_title("")                       # dataset named once, in the top row
    axes[0, 0].set_ylabel("test loss")
    axes[1, 0].set_ylabel("effective rank")
    axes[0, 0].legend(fontsize=6, loc="upper right")
    fig.suptitle("PT-ID: forecast decodability saturates (top) while representation geometry "
                 "keeps changing (bottom) — tunnel shading identical per column", fontsize=10)
    _save(fig, "fig1x2_combined_loss_rank.png")


# --------------------------------------------------------------------------- #
# Figure 3 — PT-OOD fresh-probe curves + PT-ID tunnel entrances
# --------------------------------------------------------------------------- #
def fig3_ptood_loss(tunnels, qset):
    loaded = {t: _load_ptood_runs(t, qset) for t in PT_OOD_TAGS}
    if not any(loaded.values()):
        print("  [skip fig3] no PT-OOD runs yet -> python -m experiments.run_ptood_probing")
        return None
    src_colors = dict(zip(PT_ID_TAGS, plt.cm.tab10(np.linspace(0, 0.4, len(PT_ID_TAGS)))))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, tgt in zip(axes, PT_OOD_TAGS):
        if loaded[tgt] is None:
            ax.set_title(f"{SHORT[tgt]} (pending)", fontsize=9)
            continue
        _, runs = loaded[tgt]
        T = np.stack([wl.mean(axis=1) for _, wl, _ in runs])
        x = np.arange(T.shape[1])                     # data-driven (14 for fslot)
        ax.fill_between(x, T.mean(0) - T.std(0), T.mean(0) + T.std(0),
                        color="tab:orange", alpha=0.2)
        ax.plot(x, T.mean(0), "o-", color="tab:orange", ms=3.5,
                label=f"mean of {len(runs)} fresh-probe runs")
        for src in PT_ID_TAGS:                        # markers, not 4 overlapping shades
            ax.axvline(tunnels[src]["l_start"], color=src_colors[src], ls="--", lw=1.2,
                       label=f"{SHORT[src]} l_start")
        ax.set_title(SHORT[tgt], fontsize=9)
        _layer_axis(ax, T.shape[1])
    axes[0].set_ylabel("Chronos-2 quantile loss (test)")
    axes[0].legend(fontsize=6)
    fig.suptitle("PT-OOD fresh-probe layerwise test loss (probes trained/tuned/evaluated ON "
                 "the target) — PT-ID validation-defined tunnel entrances marked", fontsize=10)
    _save(fig, "fig3_ptood_test_loss.png")
    return loaded


# --------------------------------------------------------------------------- #
# Figure 4 — Delta heatmap (the quantitative hypothesis test)
# --------------------------------------------------------------------------- #
def fig4_delta_heatmap(tunnels, loaded, qset):
    d_id = {}
    for src in PT_ID_TAGS:
        wl, sid = _seed_mean_windows([_ptid_run_curves(src, qset, s) for s in RUN_SEEDS])
        d_id[src] = d_stat_boot(wl, sid, tunnels[src]["l_start"], last=wl.shape[0] - 1,
                                B=config.BOOT_B, seed=SEED)          # last=13 (post-LN) for fslot
    tgts = [t for t in PT_OOD_TAGS if loaded.get(t)]
    cells, M = {}, np.full((len(PT_ID_TAGS), len(tgts)), np.nan)
    for j, tgt in enumerate(tgts):
        wl, sid = _seed_mean_windows(loaded[tgt][1])
        for i, src in enumerate(PT_ID_TAGS):
            d_ood = d_stat_boot(wl, sid, tunnels[src]["l_start"], last=wl.shape[0] - 1,
                                B=config.BOOT_B, seed=SEED)
            dl = delta_stat(d_ood, d_id[src])         # Delta^(b) = D_OOD^(b) - D_ID^(b)
            cells[(src, tgt)] = {"delta": dl["point"], "ci": dl["ci"],
                                 "d_ood": d_ood["point"], "d_ood_ci": d_ood["ci"],
                                 "d_id": d_id[src]["point"], "d_id_ci": d_id[src]["ci"]}
            M[i, j] = 100 * dl["point"]
    fig, ax = plt.subplots(figsize=(2.0 * len(tgts) + 3.0, 4.4))
    vmax = np.nanmax(np.abs(M)) or 1.0
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    for i, src in enumerate(PT_ID_TAGS):
        for j, tgt in enumerate(tgts):
            c = cells[(src, tgt)]
            sig = c["ci"][0] > 0 or c["ci"][1] < 0
            ax.text(j, i, f"{100*c['delta']:+.1f}%" + (" •" if sig else ""),
                    ha="center", va="center", fontsize=9,
                    fontweight="bold" if sig else "normal")
    ax.set_xticks(range(len(tgts))); ax.set_xticklabels([SHORT[t] for t in tgts])
    ax.set_yticks(range(len(PT_ID_TAGS)))
    ax.set_yticklabels([f"{SHORT[s]}  (L{tunnels[s]['l_start']})" for s in PT_ID_TAGS])
    ax.set_xlabel("PT-OOD target (fresh probes)")
    ax.set_ylabel("PT-ID tunnel definition")
    ax.set_title("Delta(s,t) = D_OOD(s,t) - D_ID(s) at the source tunnel entrance\n"
                 "positive = late-layer degradation stronger PT-OOD;  • = 95% CI excludes 0  "
                 f"[{qset}, {RUNS_TAG}, B={config.BOOT_B}]", fontsize=9)
    fig.colorbar(im, ax=ax, label="Delta (pct points)")
    _save(fig, "fig4_delta_heatmap.png")
    json.dump([{"source_dataset": s, "target_dataset": t, "l_start": tunnels[s]["l_start"],
                **{k: (list(v) if isinstance(v, tuple) else v) for k, v in c.items()}}
               for (s, t), c in cells.items()],
              open(PAPER_DIR / f"fig4_delta_cells__{qset}.json", "w"), indent=2)


# --------------------------------------------------------------------------- #
# PT-ID summary table + sensitivity + geometry-vs-degradation prep
# --------------------------------------------------------------------------- #
def ptid_summary(tunnels, qset):
    rows = []
    for src in PT_ID_TAGS:
        rec = tunnels[src]
        wl, sid = _seed_mean_windows([_ptid_run_curves(src, qset, s) for s in RUN_SEEDS])
        ls = rec["l_start"]
        last = wl.shape[0] - 1                         # data-driven; last=13 (post-LN) for fslot
        d = d_stat_boot(wl, sid, ls, last=last, B=config.BOOT_B, seed=SEED)
        m = m_stat_boot(wl, sid, ls, last=last, B=config.BOOT_B, seed=SEED)
        mt = rec["mean_test_loss_by_layer"]
        rows.append({
            "dataset": src, "l_start": ls, "tunnel": f"[{_labels(last + 1)[ls]}, {_labels(last + 1)[last]}]",
            "tunnel_width": last - ls + 1,
            "d_id": round(d["point"], 6),
            "d_id_ci_lo": round(d["ci"][0], 6), "d_id_ci_hi": round(d["ci"][1], 6),
            "m_test": round(m["point"], 6),
            "m_test_ci_lo": round(m["ci"][0], 6), "m_test_ci_hi": round(m["ci"][1], 6),
            "test_criterion_holds": rec["test_criterion_holds"],
            "mean_final_layer_test_loss": round(mt[-1], 6),
            "mean_tunnel_entrance_test_loss": round(mt[ls], 6),
        })
    base = PAPER_DIR / f"ptid_tunnel_summary__{qset}"
    with open(f"{base}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    json.dump(rows, open(f"{base}.json", "w"), indent=2)
    print(f"  [saved] {base}.csv/.json")


def threshold_sensitivity(tunnels, qset):
    """How the sustained-plateau l_start moves with the tolerance. PRIMARY stays tol=0.05."""
    rows = []
    for src in PT_ID_TAGS:
        mv = tunnels[src]["mean_val_loss_by_layer"]
        for tol in TOL_GRID:
            rows.append({"dataset": src, "tolerance": tol,
                         "l_start": tunnel_start(mv, tol),
                         "is_primary": tol == 0.05})
    out = PAPER_DIR / f"tunnel_threshold_sensitivity__{qset}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"  [saved] {out.name}")


def geometry_delta_r_eff(tunnels, qset):
    """Delta_r_eff(s,t) prep for the later geometry-vs-degradation relation (no figure yet)."""
    rows = []
    for tgt in PT_ID_TAGS + PT_OOD_TAGS:
        spec = _spectral_record(tgt)
        if spec is None:
            continue
        er = [L["effective_rank"] for L in spec["layers"]]
        for src in PT_ID_TAGS:
            ls = tunnels[src]["l_start"]
            rows.append({"source_dataset": src, "target_dataset": tgt,
                         "target_domain": spec["domain_status"]["pretraining"],
                         "l_start": ls,
                         "r_eff_l_start": round(er[ls], 4), "r_eff_last": round(er[-1], 4),
                         "delta_r_eff_abs": round(er[-1] - er[ls], 4),
                         "delta_r_eff_rel": round((er[-1] - er[ls]) / er[ls], 6)})
    if not rows:
        print("  [skip geometry prep] no spectral records -> python -m experiments.run_spectral")
        return
    out = PAPER_DIR / f"geometry_delta_r_eff__{qset}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"  [saved] {out.name}")


# --------------------------------------------------------------------------- #
# supplementary figures
# --------------------------------------------------------------------------- #
def supp_val_curves(tunnels):
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.4))
    for ax, src in zip(axes, PT_ID_TAGS):
        rec = tunnels[src]
        mv = np.array(rec["mean_val_loss_by_layer"])
        x = np.arange(len(mv))                        # data-driven (14 for fslot)
        for run in rec["val_loss_by_run"]:
            ax.plot(x, run, "-", color="tab:blue", alpha=0.3, lw=0.9)
        ax.plot(x, mv, "o-", color="tab:blue", ms=3.5, label="mean validation")
        ax.axhline((1 + rec["tolerance"]) * mv[-1], color="k", ls=":", lw=1,
                   label="1.05 x final-layer")
        ax.axvline(rec["l_start"], color="tab:green", lw=1.4, label="l_start")
        ax.set_title(SHORT[src], fontsize=9)
        _layer_axis(ax, len(mv))
    axes[0].set_ylabel("validation quantile loss")
    axes[0].legend(fontsize=6)
    fig.suptitle("Supplementary: PT-ID mean validation curves defining the tunnels "
                 "(faint = individual runs)", fontsize=10)
    _save(fig, "supp_ptid_val_curves.png")


def supp_individual_runs(tunnels, loaded, qset):
    for tag in PT_ID_TAGS + PT_OOD_TAGS:
        if tag in PT_ID_TAGS:
            curves = np.array(tunnels[tag]["test_loss_by_run"])
        else:
            if not loaded.get(tag):
                continue
            curves = np.stack([wl.mean(axis=1) for _, wl, _ in loaded[tag][1]])
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        x = np.arange(curves.shape[1])                # data-driven (14 for fslot)
        for k, c in enumerate(curves):
            ax.plot(x, c, "-", alpha=0.5, lw=1.1, label=f"run seed {RUN_SEEDS[k]}")
        ax.plot(x, curves.mean(0), "ko-", ms=3.5, label="mean")
        ax.set_title(f"{SHORT[tag]}: individual probe runs (test loss)", fontsize=10)
        _layer_axis(ax, curves.shape[1])
        ax.set_ylabel("test quantile loss"); ax.legend(fontsize=7)
        _save(fig, f"supp_individual_runs__{tag}.png")


def supp_ptood_erank(tunnels):
    specs = {t: _spectral_record(t) for t in PT_OOD_TAGS}
    if not any(specs.values()):
        print("  [skip supp ptood erank] no PT-OOD spectral records yet")
        return
    src_colors = dict(zip(PT_ID_TAGS, plt.cm.tab10(np.linspace(0, 0.4, len(PT_ID_TAGS)))))
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    for ax, tgt in zip(axes, PT_OOD_TAGS):
        if specs[tgt] is None:
            ax.set_title(f"{SHORT[tgt]} (pending)", fontsize=9)
            continue
        _panel_erank(ax, specs[tgt], None)            # NO tunnel: PT-OOD defines none
        for src in PT_ID_TAGS:
            ax.axvline(tunnels[src]["l_start"], color=src_colors[src], ls="--", lw=1.0,
                       label=f"{SHORT[src]} l_start")
    axes[0].set_ylabel("effective rank")
    axes[0].legend(fontsize=6)
    fig.suptitle("Supplementary: PT-OOD effective rank (probe input) — PT-ID tunnel "
                 "entrances marked; PT-OOD defines no tunnel", fontsize=10)
    _save(fig, "supp_ptood_effective_rank.png")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--quantile-set", default="q9")
    ap.add_argument("--readout", default="content", choices=("content", "fslot"),
                    help="content = pooled (results/extended_v3_rolling); fslot = shared "
                         "forecast-token (results/ext_v4_future_tokens)")
    args = ap.parse_args()
    global READOUT, READOUT_TAG, READOUT_LABEL, ptood, RUN_SEEDS, RUNS_TAG, \
        _load_ptood_runs, _ptid_run_curves, _seed_mean_windows, _tunnel_path
    READOUT = args.readout
    if READOUT == "fslot":
        from experiments import run_ptood_probing_ftok as ptood      # rebind the producer module
        READOUT_TAG, READOUT_LABEL = "fslot", "shared forecast-token (stacked slots)"
        _load_ptood_runs, _ptid_run_curves = ptood._load_ptood_runs, ptood._ptid_run_curves
        _seed_mean_windows, _tunnel_path = ptood._seed_mean_windows, ptood._tunnel_path
        RUN_SEEDS, RUNS_TAG = ptood.RUN_SEEDS, ptood.RUNS_TAG        # identical values; rebound for clarity
    config.set_dataset_set(ptood.PTID_SET)
    _derive_dirs()
    qset = args.quantile_set

    tunnels = _tunnels(qset)
    if tunnels is None:
        fit_cmd = ("run_ptood_probing_ftok --fit-ptid" if READOUT == "fslot"
                   else "run_ptood_probing --fit-ptid-seeds")
        raise SystemExit(f"missing 3-run tunnel records -> run `python -m experiments.{fit_cmd}` "
                         "(compute node) then `--tunnels-only`")
    fig1_ptid_loss(tunnels)
    ptid_summary(tunnels, qset)
    threshold_sensitivity(tunnels, qset)
    supp_val_curves(tunnels)

    specs = {s: _spectral_record(s) for s in PT_ID_TAGS}
    if all(specs.values()):
        fig2_ptid_erank(tunnels, specs)
        fig_combined(tunnels, specs)
    else:
        print(f"  [skip fig2/combined] missing spectral records for "
              f"{[s for s in PT_ID_TAGS if not specs[s]]} -> python -m experiments.run_spectral")

    loaded = fig3_ptood_loss(tunnels, qset) or {}
    if loaded:
        fig4_delta_heatmap(tunnels, loaded, qset)
    supp_individual_runs(tunnels, loaded, qset)
    supp_ptood_erank(tunnels)
    geometry_delta_r_eff(tunnels, qset)


if __name__ == "__main__":
    main()
