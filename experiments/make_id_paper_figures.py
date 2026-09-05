"""Paper figures: PT-ID layerwise TEST loss (with paired series-cluster bootstrap CIs) over
effective rank, as two compact 2x2 panels (main + appendix).

POST-HOC PLOTTING ONLY. Reads committed per-window losses, tunnel records and spectral records;
it never loads Chronos-2, never touches a feature cache, never refits a probe and never
recomputes an effective rank. numpy + matplotlib only (no torch in its import chain), a few
seconds of CPU -> safe on a login node.

WHAT CHANGED VS THE OLD `loss_and_erank_2x4.png`
    old top row : train (recomputed) + validation loss, per-panel legends, 4 datasets
    new top row : TEST loss only, mean over the 3 probe-init runs, with a 95% bootstrap CI
    The saturation entrance is UNCHANGED and is still read from the committed record (the
    repo calls it the tunnel record; `l_start` there), i.e. selected
    from the mean VALIDATION curve under the committed 5% rule. Nothing here re-derives it,
    and no tunnel boundary is ever computed from a test curve.

INPUTS
    results/ext_v4_future_tokens/ptood_probing/ptid_runs/<tag>__q1__v2__seed{0,1,2}.npz
        window_loss (14, n) per-window TEST quantile loss, series_test (n,) cluster ids
        (written by experiments/run_ptood_probing_ftok.py --fit-ptid)
    results/ext_v4_future_tokens/q1/tunnels/<tag>__fslot__q1__v2__runs0-1-2.json
        l_start = validation-selected saturation entrance (written by ... --tunnels-only)
    results/ext_v4_future_tokens/spectral/spectral__<tag>__fslot__probe_input__train.json
        per-point effective rank (written by experiments/run_spectral.py --readout fslot)
    results/cka/ext_v4_future_tokens_fslot/matrices/<tag>__fslot__layerxlayer.npy
        14x14 linear-CKA matrix over the same fslot representation points
        CAVEAT: these four matrices were committed in 1bf1b56 by an ad-hoc script that was NOT
        committed (see the framework audit). Shape/symmetry/diagonal/value range are verified
        here, and the geometry is certain (14 fslot points, (n*K, 768) rows), but the exact
        split / subsample size / seed are UNVERIFIED -- so the caption states the geometry and
        makes no claim about n.

BOOTSTRAP. Per-window losses are averaged over the 3 probe-init runs FIRST (identical windows
and series ids across runs -- asserted, mirroring run_ptood_probing_ftok._seed_mean_windows),
then handed to `probing.tunnel._layer_mean_boot` -- the exact routine behind every committed
D_ID confidence interval. It draws ONE multinomial count matrix over series and reuses it at
every representation point, so the resamples are shared across layers and the curve stays
paired across depth. B is set here (default 5000, seed 0); note the committed D_ID CIs used
config.BOOT_B (2000) -- same estimator, more replicates.

Usage (repo root):
    python -m experiments.make_id_paper_figures            # main + appendix (PDF + PNG + table)
    python -m experiments.make_id_paper_figures --which main
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing.config import REPO_ROOT, SEED
# The estimator behind every committed D_ID CI: one shared series-count matrix, all layers paired.
from probing.tunnel import _layer_mean_boot
from probing.stats import ci_bounds, cluster_bootstrap_counts

QSET, PROTO, RUN_SEEDS = "q1", "v2", (0, 1, 2)
RUNS_TAG = "runs" + "-".join(str(s) for s in RUN_SEEDS)

V4 = REPO_ROOT / "results" / "ext_v4_future_tokens"
RUN_DIR = V4 / "ptood_probing" / "ptid_runs"
TUNNEL_DIR = V4 / QSET / "tunnels"
SPEC_DIR = V4 / "spectral"
TRANSFER_SUBS = {"cross_dataset": "transfer_summary__4x4__q1.csv",
                 "unseen": "transfer_summary__pt_ood__q1.csv"}
TRANSFER_OUT = V4 / QSET / "transfer_summary"
# BOOM-as-source frozen transfer (5th source; produced by experiments.run_boom_source_transfer). Its
# same-estimand block REPLACES the old "BOOM pretrained — fresh per-target probe" block in the combined
# appendix figure, so BOOM is now a genuine transfer source alongside Electricity / Uber / Wind Farms.
BOOM_SOURCE = "boom_hourly"
BOOM_SRC_DIR = V4 / QSET / "boom_source"
# --- BOOM domain-FT (Stage B): same layerwise question, backbone stage instead of probe source
# --- ext_v5 native-head adapter: native / zero-shot / linear adapter / q1 linear probe ------
NHA_ROOT = REPO_ROOT / "results" / "ext_v5_native_head_adapter"
NHA_BOOT = NHA_ROOT / "bootstrap_inputs"
NHA_TAGS = [("monash_electricity_hourly", "PT-ID"), ("uber_tlc_hourly", "PT-ID"),
            ("m4_hourly", "PT-ID"), ("wind_farms_hourly", "PT-ID"),
            ("sg_carpark", "PT-OOD"), ("coastal_ts", "PT-OOD"), ("boom_hourly", "PT-OOD")]
NHA_CONDS = [("native", "native Chronos-2 (frozen head)", "0.35", None, "--"),
             ("zero_shot", "zero-shot:  $h_\\ell\\rightarrow$ frozen head", "#E08214", "o", "-"),
             ("linear_adapter", "linear adapter:  $h_\\ell\\rightarrow A_\\ell\\rightarrow$ frozen head",
              "#1F5FA8", "s", "-"),
             ("linear_q1", "linear probe ($Q=1$):  $h_\\ell\\rightarrow$ fresh Linear(768, 16)",
              "#1B9E77", "^", "-")]
FT_PROBE_DIR = REPO_ROOT / "results" / "ft_specialization" / "stageB" / "probes"
FT_OUT = REPO_ROOT / "results" / "ft_specialization" / "domain_shift" / QSET
FT_STAGES = [("stage0_pretrained", "(a) Pretrained"),
             ("stage1_ft_early", "(b) BOOM FT, early"),
             ("stage2_ft_late", "(c) BOOM FT, late")]
# BOOM is the FT source (FT-ID); the rest are FT-OOD, split by pretraining status
FT_GROUPS = [(["boom_hourly"], "FT-ID"),
             (["monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly",
               "wind_farms_hourly"], "PT-ID / FT-OOD"),
             (["sg_carpark", "coastal_ts"], "PT-OOD / FT-OOD")]
CKA_MAT_DIR = REPO_ROOT / "results" / "cka" / "ext_v4_future_tokens_fslot" / "matrices"
CKA_FIG_DIR = REPO_ROOT / "results" / "cka" / "ext_v4_future_tokens_fslot" / "figures"
FIG_DIR = V4 / QSET / "id" / "figures"
TAB_DIR = V4 / QSET / "id" / "tables"

# Paper labels. The final norm is T5-style RMSNorm (encoder.final_layer_norm), so the last point
# is "L12+RMS" here even though the ext_v4 filenames/records spell it "L12+LN".
LABELS = ["Emb"] + [f"L{i}" for i in range(1, 13)] + ["L12+RMS"]

TITLES = {"monash_electricity_hourly": "Electricity", "m4_hourly": "M4",
          "uber_tlc_hourly": "Uber TLC", "wind_farms_hourly": "Wind Farms"}
GROUPS = {"main": (("monash_electricity_hourly", "m4_hourly"),
                   "PT-ID forecasting: shared forecast-slot linear probe ($Q = 1$)",
                   "main_id_testloss_erank_2x2"),
          "appendix": (("uber_tlc_hourly", "wind_farms_hourly"),
                       "Additional PT-ID forecasting: shared forecast-slot linear probe ($Q = 1$)",
                       "appendix_id_testloss_erank_2x2")}
CKA_GROUPS = {"main": ("PT-ID representation similarity: forecast-slot states",
                       "main_id_cka_1x2"),
              "appendix": ("Additional PT-ID representation similarity: forecast-slot states",
                           "appendix_id_cka_1x2")}

LOSS, LOSS_BAND = "#1F5FA8", "#AFC9E8"           # test-loss curve / bootstrap band
ERANK = "#5E3C99"                                 # matches make_erank_stability_figure.py
TUNNEL_LINE, TUNNEL_FILL = "#2E7D32", "#E4F0E4"   # validation-selected entrance

CKA_RC_BUMP = {"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
               "xtick.labelsize": 10, "ytick.labelsize": 10}

PAPER_RC = {"font.size": 9, "axes.labelsize": 10, "axes.titlesize": 11, "legend.fontsize": 9,
            "xtick.labelsize": 8, "ytick.labelsize": 8.5, "axes.linewidth": 0.8,
            "xtick.major.width": 0.7, "ytick.major.width": 0.7, "lines.linewidth": 1.4,
            "pdf.fonttype": 42, "ps.fonttype": 42}          # TrueType: camera-ready safe


def _need(path, how):
    if not path.exists():
        raise FileNotFoundError(f"missing {path}\n  produce it with: {how}")
    return path


def seed_mean_windows(tag):
    """Per-window TEST losses averaged over the 3 probe-init runs + the series ids.

    Mirrors run_ptood_probing_ftok._seed_mean_windows verbatim (same asserts): the runs differ
    only in probe init, so the windows and their series ids must be identical."""
    wls, sids = [], []
    for s in RUN_SEEDS:
        p = _need(RUN_DIR / f"{tag}__{QSET}__{PROTO}__seed{s}.npz",
                  "python -m experiments.run_ptood_probing_ftok --quantile-set q1 --fit-ptid")
        z = np.load(p)
        wls.append(np.asarray(z["window_loss"], np.float64))
        sids.append(np.asarray(z["series_test"], np.int64))
    assert all(w.shape == wls[0].shape for w in wls), "runs must share identical test windows"
    assert all(np.array_equal(s, sids[0]) for s in sids), "runs must share identical series ids"
    return np.mean(wls, axis=0), sids[0]


def load_dataset(tag, boot_b, seed):
    """Everything one panel column needs: test curve + CI, validation-selected entrance, erank."""
    wl_mean, sid = seed_mean_windows(tag)

    rec = json.load(open(_need(
        TUNNEL_DIR / f"{tag}__fslot__{QSET}__{PROTO}__{RUNS_TAG}.json",
        "python -m experiments.run_ptood_probing_ftok --quantile-set q1 --tunnels-only")))
    # Gate: our seed-averaged point estimate must reproduce the committed curve exactly.
    ref = np.asarray(rec["mean_test_loss_by_layer"], np.float64)
    point, boot = _layer_mean_boot(wl_mean, sid, B=boot_b, seed=seed)
    if not np.allclose(point, ref, rtol=0, atol=1e-12):
        raise ValueError(f"{tag}: recomputed test curve disagrees with the committed tunnel record "
                         f"(max |diff| = {np.abs(point - ref).max():.3e})")
    lo, hi = ci_bounds(boot)

    spec = json.load(open(_need(
        SPEC_DIR / f"spectral__{tag}__fslot__probe_input__train.json",
        "python -m experiments.run_spectral --readout fslot")))
    erank = np.array([p["effective_rank"] for p in spec["layers"]], dtype=np.float64)
    for name, arr in (("test curve", point), ("effective rank", erank)):
        if len(arr) != len(LABELS):
            raise ValueError(f"{tag}: {name} has {len(arr)} points, expected {len(LABELS)}")

    return {"tag": tag, "point": point, "lo": lo, "hi": hi, "erank": erank,
            "l_start": int(rec["l_start"]), "peak": int(erank.argmax()),
            "n_windows": int(wl_mean.shape[1]), "n_clusters": int(np.unique(sid).size),
            "tunnel_definition": rec["tunnel_definition"], "tolerance": rec["tolerance"],
            "erank_split": spec["split"], "erank_N": spec["sample_size"]}


def make_figure(rows, title, stem, boot_b, dpi=400, show_title=True):
    """2x2: columns = datasets, row 0 = test loss + CI, row 1 = effective rank."""
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4), layout="constrained", squeeze=False,
                                 sharex="col")
        fig.get_layout_engine().set(h_pad=0.05, w_pad=0.06, hspace=0.08, wspace=0.10)
        x = np.arange(len(LABELS))
        last = len(LABELS) - 1

        for col, d in enumerate(rows):
            ls = d["l_start"]
            # ---- row 0: TEST loss, CI, validation-selected saturation entrance ------------------
            ax = axes[0, col]
            ax.axvspan(ls, last, color=TUNNEL_FILL, lw=0, zorder=0)
            ax.axvline(ls, color=TUNNEL_LINE, lw=1.1, zorder=1,
                       label="Saturation entrance (validation)")
            ax.fill_between(x, d["lo"], d["hi"], color=LOSS_BAND, alpha=0.95, lw=0, zorder=2,
                            label=f"95% bootstrap CI ($B={boot_b}$)")
            ax.plot(x, d["point"], "-o", ms=3.0, color=LOSS, mfc=LOSS, mec=LOSS, zorder=3,
                    label=f"Test loss (mean of {len(RUN_SEEDS)} seeds)")
            ax.set_title(TITLES[d["tag"]], fontweight="bold")
            span = d["hi"].max() - d["lo"].min()      # headroom: never clip the CI band
            ax.set_ylim(d["lo"].min() - 0.06 * span, d["hi"].max() + 0.10 * span)
            ax.annotate(LABELS[ls], xy=(ls, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(2, -9), textcoords="offset points", fontsize=7.5,
                        color=TUNNEL_LINE, ha="left", va="top")

            # ---- row 1: effective rank ------------------------------------------------------
            ax = axes[1, col]
            ax.axvspan(ls, last, color=TUNNEL_FILL, alpha=0.55, lw=0, zorder=0)
            ax.axvline(ls, color=TUNNEL_LINE, lw=0.9, ls=(0, (4, 2)), alpha=0.75, zorder=1)
            ax.plot(x, d["erank"], "-o", ms=3.0, color=ERANK, mfc=ERANK, mec=ERANK, zorder=3,
                    label="Effective rank")
            pk = d["peak"]
            ax.plot([pk], [d["erank"][pk]], "*", ms=9.0, color=ERANK, mec="white", mew=0.6,
                    zorder=5, label=f"Peak effective rank")
            ax.set_ylim(0, d["erank"].max() * 1.08)      # no in-panel text -> less dead space

        for ax in axes.ravel():
            ax.set_xlim(-0.55, last + 0.55)
            ax.set_xticks(x)
            ax.tick_params(length=2.5, pad=1.5)
            ax.grid(axis="y", alpha=0.18, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xticklabels(LABELS, rotation=45, ha="right")
            ax.set_xlabel("Representation point")
        axes[0, 0].set_ylabel("Test quantile loss")
        axes[1, 0].set_ylabel("Effective rank")

        # one shared legend for the whole figure (loss row first, then the erank row)
        h0, l0 = axes[0, 0].get_legend_handles_labels()
        h1, l1 = axes[1, 0].get_legend_handles_labels()
        order = [l0.index(t) for t in sorted(l0, key=lambda s: ("Test" not in s, "95%" not in s))]
        fig.legend([h0[i] for i in order] + h1, [l0[i] for i in order] + l1,
                   loc="outside lower center", ncol=3, frameon=False, handlelength=1.6,
                   handletextpad=0.45, columnspacing=1.2, borderpad=0.15,
                   borderaxespad=0.25, fontsize=9.5)
        if show_title:
            fig.suptitle(title, fontsize=10, fontweight="bold")

        FIG_DIR.mkdir(parents=True, exist_ok=True)
        pdf, png = FIG_DIR / f"{stem}.pdf", FIG_DIR / f"{stem}.png"
        fig.savefig(pdf)                                   # vector, never rasterized
        fig.savefig(png, dpi=dpi)
        plt.close(fig)
    return pdf, png


GROUP_ORDER = ["PT-ID / Probe-ID", "PT-ID / Probe-OOD", "PT-OOD / Probe-OOD"]
SRC_COLOR = {"monash_electricity_hourly": "#1F5FA8", "uber_tlc_hourly": "#D95F02",
             "m4_hourly": "#7570B3", "wind_farms_hourly": "#1B9E77"}
SHORT = {"monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber TLC",
         "m4_hourly": "M4", "wind_farms_hourly": "Wind Farms", "sg_carpark": "SG Carpark",
         "coastal_ts": "Coastal T-S", "boom_hourly": "BOOM"}


def load_transfer_cells(boot_b, seed):
    """All 28 source->target cells with G and its PAIRED cluster-bootstrap CI.

        G = 100 * (L(L12+RMS) - L(l_s)) / L(L12+RMS)

    l_s is the SOURCE-validation-selected layer, read from the committed transfer summary
    (`val_selected_layer`); it never touches target data. Both losses come from the SAME
    seed-averaged per-window array and the SAME multinomial resamples, and the ratio is formed
    INSIDE each replicate -- identical to how tunnel.d_stat_boot builds its CI."""
    ref = len(LABELS) - 1
    out = []
    for sub, table in TRANSFER_SUBS.items():
        rows = csv.DictReader(open(_need(
            V4 / QSET / sub / "tables" / table,
            "python -m experiments.run_fslot_transfer (see the ext_v4 run recipe)")))
        for r in rows:
            src, tgt = r["source_dataset"], r["target_dataset"]
            ls = int(r["val_selected_layer"])
            wls, sids = [], []
            for sd in RUN_SEEDS:
                z = np.load(_need(V4 / QSET / sub / "bootstrap_inputs" /
                                  f"{src}__to__{tgt}__{QSET}__seed{sd}.npz", "as above"))
                wls.append(np.asarray(z["window_loss"], np.float64))
                sids.append(np.asarray(z["series_test"], np.int64))
            assert all(w.shape == wls[0].shape for w in wls), f"{src}->{tgt}: window mismatch"
            assert all(np.array_equal(x, sids[0]) for x in sids), f"{src}->{tgt}: series mismatch"
            point, boot = _layer_mean_boot(np.mean(wls, axis=0), sids[0], B=boot_b, seed=seed)
            g = 100.0 * (point[ref] - point[ls]) / point[ref]
            gb = 100.0 * (boot[:, ref] - boot[:, ls]) / boot[:, ref]
            lo, hi = ci_bounds(gb)
            clo, chi = ci_bounds(boot)
            # Delta_RMS = 100*(L(L12+RMS) - L(L12))/L(L12): does the final RMSNorm help transfer?
            drms = 100.0 * (point[ref] - point[ref - 1]) / point[ref - 1]
            drms_b = 100.0 * (boot[:, ref] - boot[:, ref - 1]) / boot[:, ref - 1]
            rlo, rhi = ci_bounds(drms_b)
            out.append({"_sub": sub, "_curve": point, "_lo": clo, "_hi": chi,
                        "group": r["quadrant"], "source": src, "target": tgt, "l_s": ls,
                        "l_s_label": LABELS[ls], "G": float(g), "ci_lo": float(lo),
                        "ci_hi": float(hi), "excludes_zero": bool(lo > 0 or hi < 0),
                        "delta_rms": float(drms), "delta_rms_ci_lo": float(rlo),
                        "delta_rms_ci_hi": float(rhi),
                        "delta_rms_excludes_zero": bool(rlo > 0 or rhi < 0),
                        "degenerate": ls == ref, "n_windows": int(wls[0].shape[1]),
                        "n_clusters": int(np.unique(sids[0]).size)})
    if len(out) != 28:
        raise ValueError(f"expected 28 transfer cells, got {len(out)}")
    return out


def load_boom_source_cells(boot_b, seed):
    """The 7 BOOM->target frozen-transfer cells: one probe fit on BOOM, applied unchanged to every
    target. SAME estimand and SAME point estimator as load_transfer_cells (per-window seed-mean loss
    via _layer_mean_boot); produced by experiments.run_boom_source_transfer. Keyed (BOOM_SOURCE, target)
    so make_multi_source_delta_figure can index it exactly like the `by` dict of the other sources."""
    rows = {r["target_dataset"]: r for r in csv.DictReader(open(_need(
        BOOM_SRC_DIR / "tables" / f"transfer_summary__boom_src__{QSET}.csv",
        "python -m experiments.run_boom_source_transfer --quantile-set q1")))}
    out = {}
    for tgt in COMBINED_TARGETS:
        if tgt not in rows:
            raise ValueError(f"BOOM-source summary is missing target '{tgt}'")
        ls = int(rows[tgt]["val_selected_layer"])
        wls, sids = [], []
        for sd in RUN_SEEDS:
            z = np.load(_need(BOOM_SRC_DIR / "bootstrap_inputs" /
                              f"{BOOM_SOURCE}__to__{tgt}__{QSET}__seed{sd}.npz", "as above"))
            wls.append(np.asarray(z["window_loss"], np.float64))
            sids.append(np.asarray(z["series_test"], np.int64))
        assert all(w.shape == wls[0].shape for w in wls), f"BOOM->{tgt}: window mismatch across seeds"
        assert all(np.array_equal(x, sids[0]) for x in sids), f"BOOM->{tgt}: series mismatch across seeds"
        point, _ = _layer_mean_boot(np.mean(wls, axis=0), sids[0], B=boot_b, seed=seed)
        out[(BOOM_SOURCE, tgt)] = {"_curve": point, "source": BOOM_SOURCE, "target": tgt,
                                   "l_s": ls, "l_s_label": LABELS[ls]}
    return out


def make_transfer_figure(cells, boot_b, stem="main_transfer_advantage", dpi=400,
                         show_title=True):
    """One compact 4x7 heatmap: rows = probe source, columns = evaluation target.

    The finding is a SIGN PATTERN over the complete grid, so the grid itself is the figure:
    diverging scale centred at 0, the value printed in every cell, a separator between PT-ID
    and PT-OOD target blocks, and a heavy border on the four Probe-ID diagonal cells. Per-cell
    confidence intervals live in the companion table, not here."""
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.patches import Rectangle

    src_order = list(SRC_COLOR)                                  # Elec, Uber, M4, Wind
    tgt_order = src_order + ["sg_carpark", "coastal_ts", "boom_hourly"]
    n_ptid = len(src_order)
    by = {(c["source"], c["target"]): c for c in cells}
    G = np.full((len(src_order), len(tgt_order)), np.nan)
    for i, sr in enumerate(src_order):
        for j, tg in enumerate(tgt_order):
            G[i, j] = by[(sr, tg)]["G"]
    if np.isnan(G).any():
        raise ValueError("transfer grid has holes; expected all 4x7 source-target cells")

    with plt.rc_context({**PAPER_RC, "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, ax = plt.subplots(figsize=(7.2, 2.9), layout="constrained")
        norm = TwoSlopeNorm(vmin=min(G.min(), -1.0), vcenter=0.0, vmax=max(G.max(), 1.0))
        im = ax.imshow(G, cmap="RdBu", norm=norm, aspect="auto", interpolation="nearest")

        for i in range(G.shape[0]):
            for j in range(G.shape[1]):
                degenerate = by[(src_order[i], tgt_order[j])]["degenerate"]
                txt = "0" if degenerate else f"{G[i, j]:+.1f}"
                rgba = im.cmap(im.norm(G[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, txt, ha="center", va="center", fontsize=8.5,
                        color=("white" if lum < 0.5 else "black"))
        # PT-ID | PT-OOD target separator
        ax.axvline(n_ptid - 0.5, color="black", lw=2.0)
        # Probe-ID diagonal
        for i in range(n_ptid):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="black", lw=2.0,
                                   zorder=5))
        ax.set_xticks(range(len(tgt_order)))
        ax.set_xticklabels([SHORT[t] for t in tgt_order])
        ax.set_yticks(range(len(src_order)))
        # dagger: validation selects L12+RMS itself, so G = 0 by construction for that row
        ax.set_yticklabels([SHORT[t] + (" $\\dagger$" if all(
            by[(t, x)]["degenerate"] for x in tgt_order) else "") for t in src_order])
        ax.set_xlabel("Evaluation target")
        ax.set_ylabel("Probe source")
        ax.tick_params(length=0, pad=3)
        for side in ax.spines.values():
            side.set_visible(False)
        # target-block headers
        for lo, hi, lab in [(0, n_ptid - 1, "PT-ID targets"),
                            (n_ptid, len(tgt_order) - 1, "PT-OOD targets")]:
            ax.text((lo + hi) / 2.0, -0.72, lab, ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
        ax.set_ylim(len(src_order) - 0.5, -0.75)
        cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
        cb.set_label("$G$ (%)")
        cb.ax.tick_params(labelsize=8)
        if show_title:
            fig.suptitle("Transfer of the source-selected representation point "
                         "$\\ell_s^{\\star}$", fontsize=10, fontweight="bold")
        d = TRANSFER_OUT / "figures"
        d.mkdir(parents=True, exist_ok=True)
        pdf, png = d / f"{stem}.pdf", d / f"{stem}.png"
        fig.savefig(pdf); fig.savefig(png, dpi=dpi); plt.close(fig)
    return pdf, png


DELTA_TOL = 5.0          # "close to final": within 5% of L12+RMS, the transfer analogue of TUNNEL_TOL


REF_SPEC = {"final": (-1, "L12+RMS", "from L12+RMS"),
            "l12":   (-2, "L12",     "from L12")}


def delta_curve(cell, ref="final"):
    """Delta(l) = 100 * (L(l) - L(ref)) / L(ref). >0 = worse than the reference point.

    ref='final' (L12+RMS, the model's actual final representation) is the scientific
    comparison; ref='l12' is the diagnostic that separates what the encoder blocks do from
    what the final RMSNorm does."""
    c = cell["_curve"]
    r = c[REF_SPEC[ref][0]]
    return 100.0 * (c - r) / r


def plateau_entrance(d, tol=DELTA_TOL):
    """Earliest l with Delta(j) <= tol for ALL j >= l -- the sustained form of the 5% rule,
    applied to the transferred curve. None when no such layer exists."""
    ok = [l for l in range(len(d)) if np.all(d[l:] <= tol)]
    return ok[0] if ok else None


def make_delta_figure(cells, src, boot_b, dpi=400, show_title=True, main=False,
                      ref="final"):
    """Two panels for ONE probe source: (a) PT-ID targets, (b) PT-OOD targets.

    Rows = target dataset, columns = representation point, colour AND printed integer =
    Delta vs the final representation. Negative = lower loss than L12+RMS. The panels are
    stacked rather than side by side so 14 columns of numbers stay legible after the usual
    two-column downscale; the colour scale is shared by both panels and by every source."""
    from matplotlib.colors import TwoSlopeNorm

    ptid = [src] + [t for t in SRC_COLOR if t != src]          # source first, then the rest
    ptood = ["sg_carpark", "coastal_ts", "boom_hourly"]
    by = {(c["source"], c["target"]): c for c in cells}
    norm = TwoSlopeNorm(vmin=-12.0, vcenter=0.0, vmax=30.0)

    def cell_text(v):
        r = int(round(v))
        return "0" if r == 0 else f"{r:+d}"

    with plt.rc_context({**PAPER_RC, "xtick.labelsize": 10, "ytick.labelsize": 10.5,
                         "axes.labelsize": 12}):
        fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.2), layout="constrained",
                                 gridspec_kw={"height_ratios": [len(ptid), len(ptood)]},
                                 sharex=True)
        fig.get_layout_engine().set(h_pad=0.04, hspace=0.16)
        for ax, tags, sub in ((axes[0], ptid, "(a) PT-ID targets"),
                              (axes[1], ptood, "(b) PT-OOD targets")):
            M = np.vstack([delta_curve(by[(src, t)], ref) for t in tags])
            im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    rgba = im.cmap(im.norm(M[i, j]))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    ax.text(j, i, cell_text(M[i, j]), ha="center", va="center", fontsize=8.5,
                            color=("white" if lum < 0.5 else "black"))
            ax.set_yticks(range(len(tags)))
            ax.set_yticklabels([SHORT[t] for t in tags])
            ax.set_xticks(np.arange(len(LABELS)))
            ax.tick_params(length=0, pad=3)
            ax.set_title(sub, fontsize=11, fontweight="bold", loc="left", pad=4)
            for side in ax.spines.values():
                side.set_visible(False)
        # only the reference column is marked; l_s* belongs to the G figure, not here
        ref_idx = len(LABELS) + REF_SPEC[ref][0]
        axes[0].annotate("reference", xy=(ref_idx, -0.62), xycoords="data", ha="center", va="center",
                         fontsize=9.5, style="italic", annotation_clip=False)
        axes[-1].set_xticklabels(LABELS, rotation=45, ha="right")
        axes[-1].set_xlabel("Representation point")
        # sits in the row-label column, level with the panel subtitle -- not floating above it
        cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.020, pad=0.012, extend="max")
        cb.set_label(f"Relative test-loss difference\n{REF_SPEC[ref][2]} (%)", fontsize=10.5)
        cb.ax.tick_params(labelsize=9)
        if show_title:
            fig.suptitle(f"Layerwise transfer from {SHORT[src]}"
                         + ("" if ref == "final" else "  (diagnostic: reference L12)"),
                         fontsize=10, fontweight="bold")
        d_ = TRANSFER_OUT / "figures"
        d_.mkdir(parents=True, exist_ok=True)
        base = ("main_delta" if (main and ref == "final") else "appendix_delta")
        stem = (f"{base}_vs_{'final' if ref == 'final' else 'l12'}"
                + f"__{SHORT[src].lower().replace(' ', '_').replace('-', '')}")
        pdf, png = d_ / f"{stem}.pdf", d_ / f"{stem}.png"
        fig.savefig(pdf); fig.savefig(png, dpi=dpi); plt.close(fig)
    return pdf, png


def make_rms_figure(cells, dpi=400, show_title=True):
    """4x7 summary of Delta_RMS: does Chronos-2's final RMSNorm help or hurt transferred
    readouts? Negative = the post-norm state gives LOWER loss than raw L12."""
    from matplotlib.colors import TwoSlopeNorm

    src_order = list(SRC_COLOR)
    tgt_order = src_order + ["sg_carpark", "coastal_ts", "boom_hourly"]
    by = {(c["source"], c["target"]): c for c in cells}
    M = np.array([[by[(sr, tg)]["delta_rms"] for tg in tgt_order] for sr in src_order])
    sig = np.array([[by[(sr, tg)]["delta_rms_excludes_zero"] for tg in tgt_order]
                    for sr in src_order])
    with plt.rc_context({**PAPER_RC, "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, ax = plt.subplots(figsize=(7.2, 2.7), layout="constrained")
        # SYMMETRIC limits: the data are overwhelmingly negative, and a TwoSlopeNorm clamped at
        # max(M)=+1.6 would paint a +1.6 cell as saturated as a -22.5 one.
        lim = float(np.abs(M).max())
        norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
        im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                rgba = im.cmap(im.norm(M[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f"{M[i, j]:+.1f}" + ("*" if sig[i, j] else ""),
                        ha="center", va="center", fontsize=8.5,
                        color=("white" if lum < 0.5 else "black"))
        ax.axvline(len(src_order) - 0.5, color="black", lw=2.0)
        ax.set_xticks(range(len(tgt_order)))
        ax.set_xticklabels([SHORT[t] for t in tgt_order])
        ax.set_yticks(range(len(src_order)))
        ax.set_yticklabels([SHORT[t] for t in src_order])
        ax.set_xlabel("Evaluation target")
        ax.set_ylabel("Probe source")
        ax.tick_params(length=0, pad=3)
        for side in ax.spines.values():
            side.set_visible(False)
        for lo, hi, lab in [(0, len(src_order) - 1, "PT-ID targets"),
                            (len(src_order), len(tgt_order) - 1, "PT-OOD targets")]:
            ax.text((lo + hi) / 2.0, -0.72, lab, ha="center", va="bottom", fontsize=9,
                    fontweight="bold")
        ax.set_ylim(len(src_order) - 0.5, -0.75)
        cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
        cb.set_label("$\\Delta_{\\mathrm{RMS}}$ (%)")
        cb.ax.tick_params(labelsize=8)
        if show_title:
            fig.suptitle("Effect of the final RMSNorm on transferred readouts",
                         fontsize=10, fontweight="bold")
        d_ = TRANSFER_OUT / "figures"
        d_.mkdir(parents=True, exist_ok=True)
        pdf, png = d_ / "main_rms_effect.pdf", d_ / "main_rms_effect.png"
        fig.savefig(pdf); fig.savefig(png, dpi=dpi); plt.close(fig)
    return pdf, png


# every block shows the SAME 7 evaluation targets, PT-ID group then PT-OOD group
COMBINED_TARGETS = ["monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly",
                    "wind_farms_hourly", "sg_carpark", "coastal_ts", "boom_hourly"]
COMBINED_RULE = 4                       # horizontal rule after the 4 PT-ID target rows


def make_multi_source_delta_figure(cells, boom_cells, srcs, boot_b, dpi=400, show_title=True,
                                   ref="final"):
    """One figure, one 7-row panel per block, all sharing the colour scale and reference.

    Every block is the SAME estimand: a frozen transferred probe (fit on one source, applied
    unchanged to all 7 targets).
    Blocks 1..n  : one per PT-ID source in `srcs`.
    Final block  : BOOM as a transfer source (probe fit on BOOM's train split), from
                   load_boom_source_cells / experiments.run_boom_source_transfer. This replaces the
                   old "BOOM pretrained -- fresh per-target probe" block, so the panel is now
                   directly comparable cell-for-cell across all blocks.
    """
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-12.0, vcenter=0.0, vmax=30.0)
    by = {(c["source"], c["target"]): c for c in cells}

    blocks = [(f"{SHORT[src]}  \u2014  frozen transferred probe",
               [by[(src, t)] for t in COMBINED_TARGETS]) for src in srcs]
    blocks.append(("BOOM  \u2014  frozen transferred probe",
                   [boom_cells[(BOOM_SOURCE, t)] for t in COMBINED_TARGETS]))

    with plt.rc_context({**PAPER_RC, "xtick.labelsize": 10, "ytick.labelsize": 10.5,
                         "axes.labelsize": 12}):
        fig, axes = plt.subplots(len(blocks), 1, figsize=(8.4, 11.8), layout="constrained",
                                 sharex=True)
        fig.get_layout_engine().set(h_pad=0.05, hspace=0.16)
        for ax, (title, cs) in zip(axes, blocks):
            M = np.vstack([delta_curve(c, ref) for c in cs])
            im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    rgba = im.cmap(im.norm(M[i, j]))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    r = int(round(M[i, j]))
                    ax.text(j, i, "0" if r == 0 else f"{r:+d}", ha="center", va="center",
                            fontsize=8.5, color=("white" if lum < 0.5 else "black"))
            ax.axhline(COMBINED_RULE - 0.5, color="black", lw=1.6)   # PT-ID | PT-OOD targets
            ax.set_yticks(range(len(COMBINED_TARGETS)))
            ax.set_yticklabels([SHORT[t] for t in COMBINED_TARGETS])
            ax.set_xticks(np.arange(len(LABELS)))
            ax.tick_params(length=0, pad=3)
            ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=4)
            for side in ax.spines.values():
                side.set_visible(False)
        ref_idx = len(LABELS) + REF_SPEC[ref][0]
        axes[0].annotate("reference", xy=(ref_idx, -0.62), xycoords="data", ha="center",
                         va="center", fontsize=9.5, style="italic", annotation_clip=False)
        axes[-1].set_xticklabels(LABELS, rotation=45, ha="right")
        axes[-1].set_xlabel("Representation point")
        cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.016, pad=0.012, extend="max")
        cb.set_label(f"Relative test-loss difference\n{REF_SPEC[ref][2]} (%)", fontsize=10.5)
        cb.ax.tick_params(labelsize=9)
        if show_title:
            fig.suptitle("Layerwise profiles: remaining transfer sources and the pretrained "
                         "backbone", fontsize=11, fontweight="bold")
        d_ = TRANSFER_OUT / "figures"
        d_.mkdir(parents=True, exist_ok=True)
        stem = f"appendix_delta_vs_{'final' if ref == 'final' else 'l12'}__combined"
        pdf, png = d_ / f"{stem}.pdf", d_ / f"{stem}.png"
        fig.savefig(pdf); fig.savefig(png, dpi=dpi); plt.close(fig)
    return pdf, png


def make_transfer_grid(cells, sub, boot_b, dpi=400, show_title=True):
    """Appendix grid of the full layerwise transfer curves (rows = source, cols = target).

    Same seed-averaged losses and same paired series-cluster bootstrap as the summary heatmap;
    y-limits are shared DOWN each column so a column is comparable across sources."""
    src_order = list(SRC_COLOR)
    tgt_order = (src_order if sub == "cross_dataset"
                 else ["sg_carpark", "coastal_ts", "boom_hourly"])
    by = {(c["source"], c["target"]): c for c in cells if c["_sub"] == sub}
    nr, nc = len(src_order), len(tgt_order)
    x = np.arange(len(LABELS))
    last = len(LABELS) - 1

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(nr, nc, figsize=(7.2, 1.55 * nr + 0.9), layout="constrained",
                                 squeeze=False, sharex="col")
        fig.get_layout_engine().set(h_pad=0.03, w_pad=0.05, hspace=0.06, wspace=0.07)
        for j, tg in enumerate(tgt_order):                       # shared y down each column
            lo = min(by[(sr, tg)]["_lo"].min() for sr in src_order)
            hi = max(by[(sr, tg)]["_hi"].max() for sr in src_order)
            pad = 0.06 * (hi - lo)
            for i, sr in enumerate(src_order):
                c, ax = by[(sr, tg)], axes[i, j]
                diag = sr == tg
                if diag:
                    ax.set_facecolor("#F4F1FA")                  # Probe-ID diagonal
                ax.fill_between(x, c["_lo"], c["_hi"], color=LOSS_BAND, alpha=0.95, lw=0, zorder=2)
                ax.plot(x, c["_curve"], "-", lw=1.2, color=LOSS, zorder=3)
                ax.plot([c["l_s"]], [c["_curve"][c["l_s"]]], "o", ms=4.0, color="#B02418",
                        mec="white", mew=0.6, zorder=4,
                        label="$\\ell_s^{\\star}$ (source validation)")
                ax.set_ylim(lo - pad, hi + pad)
                ax.set_xlim(-0.5, last + 0.5)
                ax.set_xticks(x)
                ax.grid(axis="y", alpha=0.15, lw=0.5)
                ax.set_axisbelow(True)
                ax.tick_params(length=2.0, pad=1.2, labelsize=7)
                for side in ("top", "right"):
                    ax.spines[side].set_visible(False)
                if i == 0:
                    ax.set_title(SHORT[tg], fontweight="bold", fontsize=9.5)
                if j == 0:
                    ax.set_ylabel(SHORT[sr], fontsize=9, fontweight="bold")
        for ax in axes[-1, :]:
            ax.set_xticklabels([t if ((i % 3 == 0 and i != last - 1) or i == last) else ""
                                for i, t in enumerate(LABELS)], rotation=45, ha="right",
                               fontsize=7)
        fig.supylabel("Test quantile loss", fontsize=10)
        h, l = axes[0, 0].get_legend_handles_labels()
        h += [plt.Line2D([], [], color=LOSS, lw=1.2, label="Test loss (mean of 3 seeds)"),
              plt.Line2D([], [], color=LOSS_BAND, lw=5,
                         label=f"95% bootstrap CI ($B={boot_b}$)")]
        l = [x_.get_label() for x_ in h]
        fig.legend(handles=h, labels=l, loc="outside lower center", ncol=3, frameon=False,
                   handlelength=1.6, handletextpad=0.5, columnspacing=1.4, borderaxespad=0.25,
                   fontsize=8.5)
        if show_title:
            fig.suptitle("Frozen-probe transfer, layerwise "
                         + ("(PT-ID targets)" if sub == "cross_dataset" else "(PT-OOD targets)"),
                         fontsize=10, fontweight="bold")
        d = TRANSFER_OUT / "figures"
        d.mkdir(parents=True, exist_ok=True)
        stem = f"appendix_transfer_grid__{'4x4' if sub == 'cross_dataset' else 'pt_ood'}"
        pdf, png = d / f"{stem}.pdf", d / f"{stem}.png"
        fig.savefig(pdf); fig.savefig(png, dpi=dpi); plt.close(fig)
    return pdf, png


def write_transfer_table(cells, boot_b, seed):
    d = TRANSFER_OUT / "tables"
    d.mkdir(parents=True, exist_ok=True)
    recs = [{**{k: v for k, v in c.items() if not k.startswith("_")},
             "source_title": SHORT[c["source"]], "target_title": SHORT[c["target"]],
             "G": round(c["G"], 4), "ci_lo": round(c["ci_lo"], 4), "ci_hi": round(c["ci_hi"], 4)}
            for c in cells]
    stem = f"transfer_advantage__{QSET}__{PROTO}__{RUNS_TAG}"
    with open(d / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    json.dump({"definition": "G = 100*(L(L12+RMS) - L(l_s))/L(L12+RMS); l_s = source-validation "
                             "selected layer, frozen before any target contact",
               "bootstrap": {"B": boot_b, "seed": seed, "unit": "target test series (cluster)",
                             "paired": "same resamples for l_s and L12+RMS; ratio formed inside "
                                       "each replicate"},
               "rows": recs}, open(d / f"{stem}.json", "w"), indent=1)
    return d / f"{stem}.csv"


def load_nha(tag, boot_b, seed, metric="mase"):
    """Per-condition MASE curve + 95% CI for one ext_v5 dataset.

    All conditions share ONE multinomial matrix over the test series (as run_native_head_adapter
    does), so every comparison inside a panel is paired. The q1 linear probe lives in a separate
    sidecar npz written by --linear-baseline; its series ids are asserted identical."""
    z = np.load(_need(NHA_BOOT / f"native_head_adapter__{tag}.npz",
                      "python -m experiments.run_native_head_adapter --adapt"))
    sid = np.asarray(z["series_test"], np.int64)
    pw = {}
    for k in z.files:
        if k == "series_test":
            continue
        cond, lab, met = k.split("__")
        if met == metric:
            pw[(cond, int(lab[1:]))] = np.asarray(z[k], np.float64)
    zq = _need(NHA_BOOT / f"native_head_adapter__linear_q1__{tag}.npz",
               "python -m experiments.run_native_head_adapter --linear-baseline")
    zq = np.load(zq)
    if not np.array_equal(np.asarray(zq["series_test"], np.int64), sid):
        raise ValueError(f"{tag}: linear_q1 sidecar has different test series than the adapter run")
    for k in zq.files:
        if k != "series_test":
            pw[("linear_q1", int(k.split("__")[1][1:]))] = np.asarray(zq[k], np.float64)

    uniq, inv = np.unique(sid, return_inverse=True)
    S = uniq.size
    M = cluster_bootstrap_counts(S, boot_b, seed)          # ONE matrix -> conditions paired
    cnt = np.bincount(inv, minlength=S).astype(np.float64)
    out = {}
    for key, vec in pw.items():
        ssum = np.bincount(inv, weights=vec, minlength=S)[:, None]
        b = ((M @ ssum) / (M @ cnt)[:, None])[:, 0]
        lo, hi = ci_bounds(b)
        out[key] = (float(vec.mean()), float(lo), float(hi))
    return out, S, sid.size


def make_nha_figure(boot_b, seed, dpi=400, show_title=True, metric="mase"):
    """2x4 panel: the three native-head conditions plus the Q=1 linear probe, on all 7 datasets."""
    x = np.arange(len(LABELS))
    with plt.rc_context({**PAPER_RC, "xtick.labelsize": 8, "ytick.labelsize": 9}):
        fig, axes = plt.subplots(2, 4, figsize=(13.0, 6.2), layout="constrained")
        fig.get_layout_engine().set(h_pad=0.05, w_pad=0.06, hspace=0.12, wspace=0.10)
        for ax, (tag, kind) in zip(axes.ravel(), NHA_TAGS):
            curves, S, nw = load_nha(tag, boot_b, seed, metric)
            nat = curves.get(("native", len(LABELS) - 1))
            if nat is not None:
                ax.axhline(nat[0], color="0.35", ls="--", lw=1.1, zorder=2)
                ax.axhspan(nat[1], nat[2], color="0.35", alpha=0.13, lw=0, zorder=1)
            # y-limits from the MEAN curves only: per-window MASE has heavy outliers on some
            # targets (WindFarms, M4), so a band-driven autoscale hides all the structure
            means = [v[0] for (c, _l), v in curves.items() if c != "native"]
            span = max(means) - min(means)
            ax.set_ylim(min(means) - 0.12 * span, max(means) + 0.10 * span)
            for cond, _lab, col, mk, ls in NHA_CONDS:
                if cond == "native":
                    continue
                xs = sorted(i for i in x if (cond, i) in curves)
                if not xs:
                    continue
                m = np.array([curves[(cond, i)][0] for i in xs])
                lo = np.array([curves[(cond, i)][1] for i in xs])
                hi = np.array([curves[(cond, i)][2] for i in xs])
                ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=3)
                ax.plot(xs, m, ls, marker=mk, ms=3.2, lw=1.3, color=col, zorder=4)
            ax.set_title(f"{SHORT[tag]}  [{kind}]", fontsize=10, fontweight="bold",
                         color=("#8B2E12" if kind == "PT-OOD" else "black"))
            ax.set_xticks(x)
            ax.set_xticklabels([t if (i % 2 == 0 or i == len(LABELS) - 1) else ""
                                for i, t in enumerate(LABELS)], rotation=45, ha="right")
            ax.tick_params(length=2.5, pad=1.5)
            ax.grid(axis="y", alpha=0.18, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for ax in axes[:, 0]:
            ax.set_ylabel("MASE (lower = better)")
        for ax in axes[-1, :]:
            ax.set_xlabel("Representation point")
        axes[1, 3].axis("off")
        handles = [plt.Line2D([], [], color=c, ls=ls, marker=mk, ms=3.2, lw=1.3, label=lab)
                   for _cond, lab, c, mk, ls in NHA_CONDS]
        axes[1, 3].legend(handles=handles, loc="center", frameon=False, fontsize=9.5)
        if show_title:
            fig.suptitle("Frozen native head vs. linear adapter vs. fresh linear probe "
                         "($Q=1$) — 4 PT-ID + 3 PT-OOD", fontsize=11, fontweight="bold")
        d_ = NHA_ROOT / "plots"
        d_.mkdir(parents=True, exist_ok=True)
        stem = f"native_head_adapter__with_linear_q1__2x4_all__{metric}"
        pdf, png = d_ / f"{stem}.pdf", d_ / f"{stem}.png"
        fig.savefig(pdf); fig.savefig(png, dpi=dpi); plt.close(fig)
    return pdf, png


def load_ft_cells(boot_b, seed):
    """Per-(stage, target) seed-averaged curve + paired cluster-bootstrap CI, for the BOOM
    domain-FT Stage B probes. Identical protocol to the ext_v4 cells: 3 probe-init runs averaged
    at the window level, then ONE shared multinomial matrix over target test series."""
    ref = len(LABELS) - 1
    out = {}
    for stage, _lab in FT_STAGES:
        for tags, _g in FT_GROUPS:
            for tgt in tags:
                wls, sids = [], []
                for sd in RUN_SEEDS:
                    z = np.load(_need(
                        FT_PROBE_DIR / f"{stage}__{tgt}__{QSET}__{PROTO}__seed{sd}.npz",
                        "sbatch job_ft_stageB.sh --probe  (Stage B B2)"))
                    wls.append(np.asarray(z["window_loss"], np.float64))
                    sids.append(np.asarray(z["series_test"], np.int64))
                assert all(np.array_equal(x, sids[0]) for x in sids), f"{stage}/{tgt}: series differ"
                point, boot = _layer_mean_boot(np.mean(wls, axis=0), sids[0], B=boot_b, seed=seed)
                drms = 100.0 * (point[ref] - point[ref - 1]) / point[ref - 1]
                rlo, rhi = ci_bounds(100.0 * (boot[:, ref] - boot[:, ref - 1]) / boot[:, ref - 1])
                out[(stage, tgt)] = {"_curve": point, "delta_rms": float(drms),
                                     "delta_rms_ci_lo": float(rlo), "delta_rms_ci_hi": float(rhi),
                                     "delta_rms_excludes_zero": bool(rlo > 0 or rhi < 0),
                                     "n_clusters": int(np.unique(sids[0]).size)}
    return out


def make_ft_delta_figure(ft, boot_b, dpi=400, show_title=True, ref="final"):
    """Three panels, one per backbone stage; rows = evaluation target. Reading DOWN a column
    across panels shows what BOOM fine-tuning did to that representation point."""
    from matplotlib.colors import TwoSlopeNorm
    norm = TwoSlopeNorm(vmin=-12.0, vcenter=0.0, vmax=30.0)
    tags = [t for grp, _ in FT_GROUPS for t in grp]
    rule_rows = np.cumsum([len(grp) for grp, _ in FT_GROUPS])[:-1]

    with plt.rc_context({**PAPER_RC, "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.labelsize": 12}):
        fig, axes = plt.subplots(len(FT_STAGES), 1, figsize=(8.4, 9.0), layout="constrained",
                                 sharex=True)
        fig.get_layout_engine().set(h_pad=0.04, hspace=0.13)
        for ax, (stage, lab) in zip(axes, FT_STAGES):
            M = np.vstack([delta_curve(ft[(stage, t)], ref) for t in tags])
            im = ax.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto", interpolation="nearest")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    rgba = im.cmap(im.norm(M[i, j]))
                    lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                    r = int(round(M[i, j]))
                    ax.text(j, i, "0" if r == 0 else f"{r:+d}", ha="center", va="center",
                            fontsize=8, color=("white" if lum < 0.5 else "black"))
            for rr in rule_rows:
                ax.axhline(rr - 0.5, color="black", lw=1.6)
            ax.set_yticks(range(len(tags)))
            ax.set_yticklabels([SHORT[t] for t in tags])
            ax.set_xticks(np.arange(len(LABELS)))
            ax.tick_params(length=0, pad=3)
            ax.set_title(lab, fontsize=11, fontweight="bold", loc="left", pad=4)
            for side in ax.spines.values():
                side.set_visible(False)
        ref_idx = len(LABELS) + REF_SPEC[ref][0]
        axes[0].annotate("reference", xy=(ref_idx, -0.62), xycoords="data", ha="center",
                         va="center", fontsize=9.5, style="italic", annotation_clip=False)
        axes[-1].set_xticklabels(LABELS, rotation=45, ha="right")
        axes[-1].set_xlabel("Representation point")
        cb = fig.colorbar(im, ax=axes.tolist(), fraction=0.018, pad=0.012, extend="max")
        cb.set_label(f"Relative test-loss difference\n{REF_SPEC[ref][2]} (%)", fontsize=10.5)
        cb.ax.tick_params(labelsize=9)
        if show_title:
            fig.suptitle("Layerwise probing across BOOM fine-tuning stages"
                         + ("" if ref == "final" else "  (diagnostic: reference L12)"),
                         fontsize=10, fontweight="bold")
        d_ = FT_OUT / "figures"
        d_.mkdir(parents=True, exist_ok=True)
        stem = f"ft_boom_delta_vs_{'final' if ref == 'final' else 'l12'}__{QSET}__{PROTO}"
        pdf, png = d_ / f"{stem}.pdf", d_ / f"{stem}.png"
        fig.savefig(pdf); fig.savefig(png, dpi=dpi); plt.close(fig)
    return pdf, png


def make_ft_rms_figure(ft, dpi=400, show_title=True):
    """3 x 7 Delta_RMS: does the final RMSNorm keep its role after the backbone is fine-tuned?"""
    from matplotlib.colors import TwoSlopeNorm
    tags = [t for grp, _ in FT_GROUPS for t in grp]
    M = np.array([[ft[(st, t)]["delta_rms"] for t in tags] for st, _ in FT_STAGES])
    sig = np.array([[ft[(st, t)]["delta_rms_excludes_zero"] for t in tags] for st, _ in FT_STAGES])
    with plt.rc_context({**PAPER_RC, "xtick.labelsize": 9, "ytick.labelsize": 9}):
        fig, ax = plt.subplots(figsize=(7.2, 2.4), layout="constrained")
        lim = float(np.abs(M).max())
        im = ax.imshow(M, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim),
                       aspect="auto", interpolation="nearest")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                rgba = im.cmap(im.norm(M[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                ax.text(j, i, f"{M[i, j]:+.1f}" + ("*" if sig[i, j] else ""), ha="center",
                        va="center", fontsize=8.5, color=("white" if lum < 0.5 else "black"))
        for rr in np.cumsum([len(g) for g, _ in FT_GROUPS])[:-1]:
            ax.axvline(rr - 0.5, color="black", lw=2.0)
        ax.set_xticks(range(len(tags)))
        ax.set_xticklabels([SHORT[t] for t in tags])
        ax.set_yticks(range(len(FT_STAGES)))
        ax.set_yticklabels([lab.split(") ")[-1] for _, lab in FT_STAGES])
        ax.set_xlabel("Evaluation target")
        ax.set_ylabel("Backbone stage")
        ax.tick_params(length=0, pad=3)
        for side in ax.spines.values():
            side.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.015)
        cb.set_label("$\\Delta_{\\mathrm{RMS}}$ (%)")
        cb.ax.tick_params(labelsize=8)
        if show_title:
            fig.suptitle("Effect of the final RMSNorm across BOOM fine-tuning stages",
                         fontsize=10, fontweight="bold")
        d_ = FT_OUT / "figures"
        d_.mkdir(parents=True, exist_ok=True)
        pdf, png = d_ / f"ft_boom_rms_effect__{QSET}__{PROTO}.pdf", d_ / f"ft_boom_rms_effect__{QSET}__{PROTO}.png"
        fig.savefig(pdf); fig.savefig(png, dpi=dpi); plt.close(fig)
    return pdf, png


def load_cka(tag):
    """14x14 linear-CKA matrix + the validation-selected entrance for one dataset."""
    m = _need(CKA_MAT_DIR / f"{tag}__fslot__layerxlayer.npy",
              "see the framework audit: these matrices have no committed producer")
    M = np.load(m).astype(np.float64)
    n = len(LABELS)
    if M.shape != (n, n):
        raise ValueError(f"{tag}: CKA matrix is {M.shape}, expected ({n}, {n})")
    if not np.allclose(np.diag(M), 1.0) or not np.allclose(M, M.T):
        raise ValueError(f"{tag}: CKA matrix is not symmetric with unit diagonal")
    rec = json.load(open(_need(
        TUNNEL_DIR / f"{tag}__fslot__{QSET}__{PROTO}__{RUNS_TAG}.json",
        "python -m experiments.run_ptood_probing_ftok --quantile-set q1 --tunnels-only")))
    return {"tag": tag, "M": M, "l_start": int(rec["l_start"])}


def make_cka_figure(rows, title, stem, dpi=400, show_title=True):
    """1x2 layer-by-layer linear-CKA heatmaps with one shared colour bar."""
    with plt.rc_context({**PAPER_RC, **CKA_RC_BUMP}):
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.8), layout="constrained", squeeze=False)
        fig.get_layout_engine().set(h_pad=0.04, w_pad=0.06, wspace=0.06)
        n = len(LABELS)
        for ax, d in zip(axes[0], rows):
            im = ax.imshow(d["M"], cmap="viridis", vmin=0.0, vmax=1.0,
                           origin="upper", interpolation="nearest")
            b = d["l_start"] - 0.5                      # boundary sits between the two cells
            for line in (ax.axvline, ax.axhline):
                line(b, color="white", lw=0.9, ls=(0, (3, 2)), alpha=0.85)
            ax.set_title(TITLES[d["tag"]], fontweight="bold")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(LABELS, rotation=90, fontsize=10)
            ax.set_yticklabels(LABELS, fontsize=10)
            ax.tick_params(length=2.0, pad=1.2)
        axes[0, 1].set_yticklabels([])                  # shared row labels
        fig.supxlabel("Representation point", fontsize=13)
        fig.supylabel("Representation point", fontsize=13)
        cb = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.02, shrink=0.92)
        cb.set_label("Linear CKA")
        cb.ax.tick_params(labelsize=10)
        if show_title:
            fig.suptitle(title, fontsize=12, fontweight="bold")
        CKA_FIG_DIR.mkdir(parents=True, exist_ok=True)
        pdf, png = CKA_FIG_DIR / f"{stem}.pdf", CKA_FIG_DIR / f"{stem}.png"
        fig.savefig(pdf)
        fig.savefig(png, dpi=dpi)
        plt.close(fig)
    return pdf, png


def write_table(rows, boot_b, seed):
    """The plotted numbers, so the figure is reproducible/citable without a recompute."""
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"id_testloss_bootstrap_ci__{QSET}__{PROTO}__{RUNS_TAG}"
    recs = []
    for d in rows:
        for i, lab in enumerate(LABELS):
            recs.append({"dataset": d["tag"], "dataset_title": TITLES[d["tag"]],
                         "layer_index": i, "layer_label": lab,
                         "test_loss": round(float(d["point"][i]), 8),
                         "ci_lo": round(float(d["lo"][i]), 8), "ci_hi": round(float(d["hi"][i]), 8),
                         "effective_rank": round(float(d["erank"][i]), 6),
                         "is_saturation_entrance": int(i == d["l_start"]),
                         "in_saturated_region": int(i >= d["l_start"])})
    with open(TAB_DIR / f"{stem}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]))
        w.writeheader()
        w.writerows(recs)
    meta = {"quantile_set": QSET, "readout": "fslot", "probe_family": "shared_linear",
            "probe_protocol_version": PROTO, "run_seeds": list(RUN_SEEDS),
            "point_estimate": "mean over probe-init runs of the per-window test quantile loss",
            "bootstrap": {"B": boot_b, "seed": seed, "unit": "test series (cluster)",
                          "estimator": "probing.tunnel._layer_mean_boot",
                          "paired_across_layers": True,
                          "note": "one shared multinomial count matrix reused at every "
                                  "representation point; seed-averaged per-window losses"},
            "saturation_entrance": {"selected_on": "mean validation curve", "shown_on": "test curve",
                                "definition": rows[0]["tunnel_definition"],
                                "tolerance": rows[0]["tolerance"]},
            "effective_rank": {"split": rows[0]["erank_split"], "N": rows[0]["erank_N"]},
            "per_dataset": {d["tag"]: {"l_start": d["l_start"], "l_start_label": LABELS[d["l_start"]],
                                       "erank_peak_label": LABELS[d["peak"]],
                                       "n_windows": d["n_windows"], "n_clusters": d["n_clusters"]}
                            for d in rows},
            "rows": recs}
    json.dump(meta, open(TAB_DIR / f"{stem}.json", "w"), indent=1)
    return TAB_DIR / f"{stem}.csv"


# --------------------------------------------------------------------------- #
# Saturation-entrance sensitivity: the same 2-row panel, with one shaded band per tolerance
# --------------------------------------------------------------------------- #
# The committed entrance uses TUNNEL_TOL = 5%. A stricter tolerance can only push the entrance
# later (the criterion is a first crossing of (1+eps) x final), so the bands nest: the strictest
# region sits inside the loosest and is drawn darkest. PT-OOD targets have no committed tunnel
# record -- compute_ptid_tunnels loops PT_ID_TAGS only -- so their entrance is derived here from
# the committed per_target validation curves with the SAME frozen criterion.
EPS_BANDS = (0.05, 0.02)
EPS_FILL = {0.05: "#E4F0E4", 0.02: "#BFDCBF"}     # looser = lighter, drawn first
PTOOD_FIG_DIR = V4 / "ptood_probing"
PT_OOD_FIG_TAGS = ("sg_carpark", "coastal_ts", "boom_hourly")
EPS_TITLES = {"monash_electricity_hourly": "Electricity", "m4_hourly": "M4",
              "uber_tlc_hourly": "Uber TLC", "wind_farms_hourly": "Wind Farms",
              "sg_carpark": "SG Carpark", "coastal_ts": "Coastal T-S", "boom_hourly": "BOOM"}


def _ptood_panel_curves(tag):
    """(seed-mean per-window test losses, series ids, mean validation curve) for a PT-OOD target.

    Different producer from the PT-ID sources: run_ptood_probing_ftok's default mode fits a FRESH
    probe on the target itself, writing per_target JSONs and bootstrap_inputs without the '__v2'
    path tag. The estimand is the same -- a fresh fslot probe with wd chosen on that dataset's own
    validation split -- so the curves are comparable to the PT-ID ones.
    """
    wls, sids, vals = [], [], []
    for sd in RUN_SEEDS:
        z = np.load(_need(PTOOD_FIG_DIR / "bootstrap_inputs" / f"{tag}__{QSET}__seed{sd}.npz",
                          f"python -m experiments.run_ptood_probing_ftok --quantile-set {QSET}"))
        wls.append(np.asarray(z["window_loss"], np.float64))
        sids.append(np.asarray(z["series_test"], np.int64))
        rec = json.load(open(_need(
            PTOOD_FIG_DIR / "per_target" / f"{tag}__{QSET}__seed{sd}.json", "as above")))
        vals.append(np.asarray(rec["val_loss_by_layer"], np.float64))
    assert all(np.array_equal(x, sids[0]) for x in sids), f"{tag}: runs differ in test series"
    return np.mean(wls, axis=0), sids[0], np.mean(vals, axis=0)


def load_eps_dataset(tag, boot_b, seed, epsilons=EPS_BANDS):
    """One panel column, with the entrance recomputed at every tolerance in ``epsilons``."""
    from probing.tunnel import TUNNEL_TOL, tunnel_start

    if tag in PT_OOD_FIG_TAGS:
        wl_mean, sid, val = _ptood_panel_curves(tag)
        point, boot = _layer_mean_boot(wl_mean, sid, B=boot_b, seed=seed)
        # gate against the producer's own per-seed test curves. The PT-OOD reference is the probe's
        # scalar loss (a float32 torch reduction), a different op chain from our float64 mean over
        # the saved per-window array -> compare at float32 precision.
        ref = np.mean([json.load(open(PTOOD_FIG_DIR / "per_target" / f"{tag}__{QSET}__seed{sd}.json"))
                       ["test_loss_by_layer"] for sd in RUN_SEEDS], axis=0)
        gate = dict(rtol=1e-6, atol=1e-9)
        committed = None
    else:
        wl_mean, sid = seed_mean_windows(tag)
        rec = json.load(open(_need(
            TUNNEL_DIR / f"{tag}__fslot__{QSET}__{PROTO}__{RUNS_TAG}.json",
            f"python -m experiments.run_ptood_probing_ftok --quantile-set {QSET} --tunnels-only")))
        val = np.asarray(rec["mean_val_loss_by_layer"], np.float64)
        ref = np.asarray(rec["mean_test_loss_by_layer"], np.float64)
        point, boot = _layer_mean_boot(wl_mean, sid, B=boot_b, seed=seed)
        gate = dict(rtol=0, atol=1e-12)
        committed = int(rec["l_start"])
    if not np.allclose(point, np.asarray(ref, np.float64), **gate):
        raise ValueError(f"{tag}: recomputed test curve disagrees with the committed record "
                         f"(max |diff| = {np.abs(point - np.asarray(ref, np.float64)).max():.3e})")

    starts = {float(e): int(tunnel_start(val, tol=float(e))) for e in epsilons}
    if committed is not None and abs(TUNNEL_TOL - 0.05) < 1e-12 and 0.05 in starts:
        if starts[0.05] != committed:                    # the 5% band must be the published one
            raise ValueError(f"{tag}: entrance recomputed at 5% is L{starts[0.05]} but the "
                             f"committed record says L{committed}")
    lo, hi = ci_bounds(boot)

    spec = json.load(open(_need(SPEC_DIR / f"spectral__{tag}__fslot__probe_input__train.json",
                                "python -m experiments.run_spectral --readout fslot")))
    erank = np.array([p["effective_rank"] for p in spec["layers"]], dtype=np.float64)
    for name, arr in (("test curve", point), ("effective rank", erank)):
        if len(arr) != len(LABELS):
            raise ValueError(f"{tag}: {name} has {len(arr)} points, expected {len(LABELS)}")
    return {"tag": tag, "point": point, "lo": lo, "hi": hi, "erank": erank,
            "starts": starts, "l_start": starts[max(starts)], "peak": int(erank.argmax()),
            "kind": "PT-OOD" if tag in PT_OOD_FIG_TAGS else "PT-ID",
            "n_windows": int(wl_mean.shape[1]), "n_clusters": int(np.unique(sid).size)}


def make_eps_figure(rows, title, stem, boot_b, epsilons=EPS_BANDS, dpi=400, show_title=True):
    """2 x N: columns = datasets, row 0 = test loss + CI, row 1 = effective rank.

    One shaded band per tolerance, nested: looser tolerances open earlier and are drawn lighter,
    so the overlap region is where every tolerance agrees the representation has saturated.
    """
    order = sorted((float(e) for e in epsilons), reverse=True)      # loosest first, so it is behind
    nc = len(rows)
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, nc, figsize=(3.6 * nc, 5.4), layout="constrained",
                                 squeeze=False, sharex="col")
        fig.get_layout_engine().set(h_pad=0.05, w_pad=0.06, hspace=0.08, wspace=0.10)
        x = np.arange(len(LABELS))
        last = len(LABELS) - 1

        for col, d in enumerate(rows):
            for row in (0, 1):
                ax = axes[row, col]
                for e in order:                                     # nested bands
                    ax.axvspan(d["starts"][e], last, color=EPS_FILL.get(e, "#E4F0E4"), lw=0,
                               zorder=0)
                for e in order:
                    ax.axvline(d["starts"][e], color=TUNNEL_LINE, lw=1.1, zorder=1,
                               ls="-" if e == max(order) else (0, (3, 2)),
                               label=(f"Saturation entrance, $\\varepsilon={int(round(e * 100))}\\%$"
                                      if col == 0 and row == 0 else None))

            ax = axes[0, col]
            ax.fill_between(x, d["lo"], d["hi"], color=LOSS_BAND, alpha=0.95, lw=0, zorder=2,
                            label=f"95% bootstrap CI ($B={boot_b}$)" if col == 0 else None)
            ax.plot(x, d["point"], "-o", ms=3.0, color=LOSS, mfc=LOSS, mec=LOSS, zorder=3,
                    label=f"Test loss (mean of {len(RUN_SEEDS)} seeds)" if col == 0 else None)
            ax.set_title(f"{EPS_TITLES[d['tag']]}  [{d['kind']}]", fontweight="bold")
            span = d["hi"].max() - d["lo"].min()
            ax.set_ylim(d["lo"].min() - 0.06 * span, d["hi"].max() + 0.16 * span)
            for e in order:                                          # label each entrance
                pos = d["starts"][e]
                near_edge = pos >= last - 2          # keep late labels inside the axes
                ax.annotate(LABELS[pos], xy=(pos, 1.0), xycoords=("data", "axes fraction"),
                            xytext=(-2 if near_edge else 2, -9 if e == max(order) else -19),
                            textcoords="offset points", fontsize=7.0, color=TUNNEL_LINE,
                            ha="right" if near_edge else "left", va="top")

            ax = axes[1, col]
            ax.plot(x, d["erank"], "-o", ms=3.0, color=ERANK, mfc=ERANK, mec=ERANK, zorder=3,
                    label="Effective rank" if col == 0 else None)
            ax.plot(d["peak"], d["erank"][d["peak"]], "*", ms=9, color=ERANK, mec="white",
                    mew=0.6, zorder=4, label="Peak effective rank" if col == 0 else None)
            ax.set_xticks(x)
            ax.set_xticklabels(LABELS, rotation=45, ha="right")
            ax.set_xlabel("Representation point")
            if col == 0:
                axes[0, 0].set_ylabel("Test quantile loss")
                axes[1, 0].set_ylabel("Effective rank")

        h, l = [], []
        for a in (axes[0, 0], axes[1, 0]):
            hh, ll = a.get_legend_handles_labels()
            h += hh
            l += ll
        fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
        if show_title:
            fig.suptitle(title, fontweight="bold")
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=dpi)
        plt.close(fig)
    print(f"[write] {FIG_DIR / (stem + '.png')}")
    for d in rows:
        print(f"    {EPS_TITLES[d['tag']]:<13} " +
              "  ".join(f"eps={e:.0%} -> {LABELS[d['starts'][e]]}" for e in order))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--which", default="both", choices=("main", "appendix", "both"))
    ap.add_argument("--eps-datasets", nargs="+",
                    default=["monash_electricity_hourly", "m4_hourly", "sg_carpark"],
                    help="columns of the saturation-sensitivity figure (PT-ID or PT-OOD)")
    ap.add_argument("--epsilons", type=float, nargs="+", default=list(EPS_BANDS),
                    help="tolerances to shade, loosest drawn first")
    ap.add_argument("--figure", default="all", choices=("loss_erank", "cka", "transfer", "ft_boom", "nha", "eps", "all"),
                    help="which figure family to build (default: all)")
    ap.add_argument("--boot-b", type=int, default=5000, help="bootstrap resamples (default 5000)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--main-source", default="m4_hourly",
                    help="probe source promoted to the main-paper delta figure")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--no-title", action="store_true",
                    help="omit the figure-level title (let the LaTeX caption carry it)")
    a = ap.parse_args()

    groups = ("main", "appendix") if a.which == "both" else (a.which,)

    if a.figure in ("eps", "all"):
        rows = [load_eps_dataset(t, a.boot_b, a.seed, a.epsilons) for t in a.eps_datasets]
        make_eps_figure(rows, "Saturation entrance under a stricter tolerance",
                        "appendix_id_saturation_sensitivity", a.boot_b, epsilons=a.epsilons,
                        dpi=a.dpi, show_title=not a.no_title)
        if a.figure == "eps":
            return

    if a.figure in ("nha", "all"):
        pdf, png = make_nha_figure(a.boot_b, a.seed, dpi=a.dpi, show_title=not a.no_title)
        for tag, kind in NHA_TAGS:
            c, S, nw = load_nha(tag, a.boot_b, a.seed)
            ref = len(LABELS) - 1
            nat = c[("native", ref)][0]
            q1b = min((c[("linear_q1", i)][0], i) for i in range(len(LABELS))
                      if ("linear_q1", i) in c)
            adb = min((c[("linear_adapter", i)][0], i) for i in range(len(LABELS))
                      if ("linear_adapter", i) in c)
            print(f"[nha] {SHORT[tag]:<12} {kind:<7} native {nat:6.3f} | best q1 probe "
                  f"{q1b[0]:6.3f} @{LABELS[q1b[1]]:<8} | best adapter {adb[0]:6.3f} @{LABELS[adb[1]]:<8}"
                  f" | {S} clusters")
        print(f"    -> {png.relative_to(REPO_ROOT)}")
    if a.figure == "nha":
        return

    if a.figure in ("ft_boom", "all"):
        ft = load_ft_cells(a.boot_b, a.seed)
        for ref in ("final", "l12"):
            pdf, png = make_ft_delta_figure(ft, a.boot_b, dpi=a.dpi,
                                            show_title=not a.no_title, ref=ref)
            print(f"[ft_boom:{ref}] -> {png.relative_to(REPO_ROOT)}")
        rp, rg = make_ft_rms_figure(ft, dpi=a.dpi, show_title=not a.no_title)
        print(f"[ft_boom:rms] -> {rg.relative_to(REPO_ROOT)}")
        for stage, lab in FT_STAGES:
            ent = []
            for t in [x for g, _ in FT_GROUPS for x in g]:
                pe = plateau_entrance(delta_curve(ft[(stage, t)]))
                ent.append(f"{SHORT[t]}:{LABELS[pe] if pe is not None else 'never'}")
            print(f"  {lab:<20} " + "  ".join(ent))
    if a.figure == "ft_boom":
        return

    if a.figure in ("transfer", "all"):
        cells = load_transfer_cells(a.boot_b, a.seed)
        pdf, png = make_transfer_figure(cells, a.boot_b, dpi=a.dpi, show_title=not a.no_title)
        csv_p = write_transfer_table(cells, a.boot_b, a.seed)
        for sub in TRANSFER_SUBS:
            gp, gg = make_transfer_grid(cells, sub, a.boot_b, dpi=a.dpi,
                                        show_title=not a.no_title)
            print(f"    -> {gp.relative_to(REPO_ROOT)}\n    -> {gg.relative_to(REPO_ROOT)}")
        by = {(c["source"], c["target"]): c for c in cells}
        print("\n[delta] plateau entrance = earliest layer with Delta <= 5% from there on")
        for src in SRC_COLOR:
            for ref in ("final", "l12"):
                pdf, png = make_delta_figure(cells, src, a.boot_b, dpi=a.dpi,
                                             show_title=not a.no_title,
                                             main=(src == a.main_source), ref=ref)
            ent = []
            for t in list(SRC_COLOR) + ["sg_carpark", "coastal_ts", "boom_hourly"]:
                pe = plateau_entrance(delta_curve(by[(src, t)]))
                ent.append(f"{SHORT[t]}:{LABELS[pe] if pe is not None else 'never'}")
            print(f"  {SHORT[src]:<12} " + "  ".join(ent))
            print(f"    -> {png.relative_to(REPO_ROOT)}")
        others = [t for t in SRC_COLOR if t != a.main_source]
        mp, mg = make_multi_source_delta_figure(cells, load_boom_source_cells(a.boot_b, a.seed), others,
                                                a.boot_b, dpi=a.dpi, show_title=not a.no_title)
        print(f"  [combined] {' + '.join(SHORT[t] for t in others)} + BOOM\n"
              f"    -> {mg.relative_to(REPO_ROOT)}")
        rp, rg = make_rms_figure(cells, dpi=a.dpi, show_title=not a.no_title)
        dr = np.array([c["delta_rms"] for c in cells])
        nsig = sum(c["delta_rms_excludes_zero"] for c in cells)
        print(f"\n[rms] Delta_RMS over 28 cells: median {np.median(dr):+.2f}%  "
              f"negative (RMSNorm helps) {int((dr < 0).sum())}/28  CI excludes 0: {nsig}/28  "
              f"range [{dr.min():+.1f}, {dr.max():+.1f}]")
        print(f"    -> {rg.relative_to(REPO_ROOT)}")
        for g in GROUP_ORDER:
            grp = [c for c in cells if c["group"] == g]
            pos = sum(c["G"] > 0 for c in grp)
            sig = sum(c["excludes_zero"] for c in grp)
            print(f"[transfer] {g:<20} n={len(grp):<3} median={np.median([c['G'] for c in grp]):+6.2f}%"
                  f"  G>0: {pos}/{len(grp)}  CI excludes 0: {sig}/{len(grp)}")
        print(f"    -> {pdf.relative_to(REPO_ROOT)}\n    -> {png.relative_to(REPO_ROOT)}"
              f"\n    -> {csv_p.relative_to(REPO_ROOT)}")
    if a.figure == "transfer":
        return

    if a.figure in ("cka", "all"):
        for g in groups:
            tags = GROUPS[g][0]
            title, stem = CKA_GROUPS[g]
            rows = [load_cka(t) for t in tags]
            pdf, png = make_cka_figure(rows, title, stem, dpi=a.dpi, show_title=not a.no_title)
            print(f"[cka:{g}] {' + '.join(TITLES[t] for t in tags)}")
            for d in rows:
                M = d["M"]
                print(f"    {TITLES[d['tag']]:<12} CKA(Emb, L12+RMS)={M[0, -1]:.3f}  "
                      f"CKA(L6, L12+RMS)={M[6, -1]:.3f}  entrance={LABELS[d['l_start']]}")
            print(f"    -> {pdf.relative_to(REPO_ROOT)}\n    -> {png.relative_to(REPO_ROOT)}")
    if a.figure == "cka":
        return

    all_rows = []
    for g in groups:
        tags, title, stem = GROUPS[g]
        rows = [load_dataset(t, a.boot_b, a.seed) for t in tags]
        pdf, png = make_figure(rows, title, stem, a.boot_b, dpi=a.dpi,
                               show_title=not a.no_title)
        all_rows += rows
        print(f"[{g}] {' + '.join(TITLES[t] for t in tags)}")
        for d in rows:
            print(f"    {TITLES[d['tag']]:<12} entrance={LABELS[d['l_start']]:<7} (validation) "
                  f"erank peak={LABELS[d['peak']]:<7} n_windows={d['n_windows']} "
                  f"n_clusters={d['n_clusters']}")
        print(f"    -> {pdf.relative_to(REPO_ROOT)}\n    -> {png.relative_to(REPO_ROOT)}")
    csv_path = write_table(all_rows, a.boot_b, a.seed)
    print(f"[table] -> {csv_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
