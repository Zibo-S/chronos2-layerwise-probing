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
# content-pooled readout: same cache/rows, no probe -> no committed forecasting tunnel to overlay,
# so the panels carry no validation-entrance dashed line (CKA is probe-independent regardless).
CKA_CONTENT_MAT_DIR = REPO_ROOT / "results" / "cka" / "ext_v4_future_tokens_content" / "matrices"
CKA_CONTENT_FIG_DIR = REPO_ROOT / "results" / "cka" / "ext_v4_future_tokens_content" / "figures"
# content-slot readout: the K content patches before REG through the SAME shared head as fslot.
# Same stacking as fslot, so these heatmaps are the controlled "which tokens" twin of the fslot
# ones; no committed forecasting tunnel to overlay either, so no dashed entrance line.
CKA_CSLOT_MAT_DIR = REPO_ROOT / "results" / "cka" / "ext_v4_future_tokens_cslot" / "matrices"
CKA_CSLOT_FIG_DIR = REPO_ROOT / "results" / "cka" / "ext_v4_future_tokens_cslot" / "figures"
FIG_DIR = V4 / QSET / "id" / "figures"
TAB_DIR = V4 / QSET / "id" / "tables"

# Paper labels. The final norm is T5-style RMSNorm (encoder.final_layer_norm), so the last point
# is "L12+RMS" here even though the ext_v4 filenames/records spell it "L12+LN".
LABELS = ["Emb"] + [f"L{i}" for i in range(1, 13)] + ["L12+RMS"]

TITLES = {"monash_electricity_hourly": "Electricity", "m4_hourly": "M4",
          "uber_tlc_hourly": "Uber TLC", "wind_farms_hourly": "Wind Farms",
          "sg_carpark": "SG Carpark", "coastal_ts": "Coastal T-S", "boom_hourly": "BOOM"}
# group -> (datasets, title, stem, drop_emb). drop_emb omits the input-embedding point from the
# x axis only; the loaded curves and the emitted table stay full length.
GROUPS = {"main": (("monash_electricity_hourly", "m4_hourly"),
                   "PT-ID forecasting: shared forecast-slot linear probe ($Q = 1$)",
                   "main_id_testloss_erank_2x2", False),
          "appendix": (("uber_tlc_hourly", "wind_farms_hourly"),
                       "Additional PT-ID forecasting: shared forecast-slot linear probe ($Q = 1$)",
                       "appendix_id_testloss_erank_2x2", False),
          # copy of "main" with the PT-OOD SG Carpark column added and Emb dropped, so the
          # post-embedding descent is not squashed by the Emb -> L2 collapse
          "main_sg": (("monash_electricity_hourly", "m4_hourly", "sg_carpark"),
                      "Forecasting: shared forecast-slot linear probe ($Q = 1$)",
                      "main_id_testloss_erank_2x3_no_emb", True)}
# group -> (title, stem, ncol). Datasets come from GROUPS[group] unless CKA_TAGS overrides them.
CKA_GROUPS = {"main": ("PT-ID representation similarity: forecast-slot states",
                       "main_id_cka_1x2", 2),
              "appendix": ("Additional PT-ID representation similarity: forecast-slot states",
                           "appendix_id_cka_1x2", 2),
              "main_sg": ("Representation similarity: forecast-slot states",
                          "main_id_cka", 3),
              # all seven datasets in one appendix panel, wrapping into rows of three
              "all7": ("Representation similarity across datasets: forecast-slot states",
                       "appendix_id_cka", 3)}
CKA_TAGS = {"all7": ("monash_electricity_hourly", "m4_hourly", "uber_tlc_hourly",
                     "wind_farms_hourly", "sg_carpark", "coastal_ts", "boom_hourly")}

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
    """Everything one panel column needs: test curve + CI, validation-selected entrance, erank.

    PT-OOD targets come from a different producer (a fresh probe fit on the target itself, no
    committed tunnel record), so their entrance is recomputed from the saved validation curve at
    the same tolerance -- the identical estimand and criterion, as in ``load_eps_dataset``."""
    from probing.tunnel import TUNNEL_TOL, tunnel_start

    if tag in PT_OOD_FIG_TAGS:
        wl_mean, sid, val = _ptood_panel_curves(tag)
        point, boot = _layer_mean_boot(wl_mean, sid, B=boot_b, seed=seed)
        # the PT-OOD reference is the probe's own float32 scalar reduction -> compare at float32
        ref = np.mean([json.load(open(PTOOD_FIG_DIR / "per_target" / f"{tag}__{QSET}__seed{sd}.json"))
                       ["test_loss_by_layer"] for sd in RUN_SEEDS], axis=0)
        gate, rec = dict(rtol=1e-6, atol=1e-9), {"l_start": int(tunnel_start(val, tol=TUNNEL_TOL)),
                                                 "tunnel_definition": "sustained_plateau",
                                                 "tolerance": TUNNEL_TOL}
    else:
        wl_mean, sid = seed_mean_windows(tag)
        rec = json.load(open(_need(
            TUNNEL_DIR / f"{tag}__fslot__{QSET}__{PROTO}__{RUNS_TAG}.json",
            "python -m experiments.run_ptood_probing_ftok --quantile-set q1 --tunnels-only")))
        # Gate: our seed-averaged point estimate must reproduce the committed curve exactly.
        ref = np.asarray(rec["mean_test_loss_by_layer"], np.float64)
        point, boot = _layer_mean_boot(wl_mean, sid, B=boot_b, seed=seed)
        gate = dict(rtol=0, atol=1e-12)
    if not np.allclose(point, np.asarray(ref, np.float64), **gate):
        raise ValueError(f"{tag}: recomputed test curve disagrees with the committed record "
                         f"(max |diff| = {np.abs(point - np.asarray(ref, np.float64)).max():.3e})")
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


def make_figure(rows, title, stem, boot_b, dpi=400, show_title=True, drop_emb=False):
    """2 x len(rows): columns = datasets, row 0 = test loss + CI, row 1 = effective rank.

    ``drop_emb`` omits the input-embedding point from the plot. Emb is where every curve starts
    its steepest fall, so keeping it compresses everything after L2 into a few pixels; dropping it
    changes nothing that is computed, only what is drawn."""
    off = 1 if drop_emb else 0
    labels = LABELS[off:]
    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, len(rows), figsize=(3.6 * len(rows), 5.4),
                                 layout="constrained", squeeze=False, sharex="col")
        fig.get_layout_engine().set(h_pad=0.05, w_pad=0.06, hspace=0.08, wspace=0.10)
        x = np.arange(len(labels))
        last = len(labels) - 1

        for col, d in enumerate(rows):
            if d["l_start"] < off or d["peak"] < off:
                raise ValueError(f"{d['tag']}: entrance/peak falls on a dropped point — "
                                 "drop_emb would put the marker off the axis")
            ls = d["l_start"] - off
            point, lo_, hi_, erank = (d["point"][off:], d["lo"][off:], d["hi"][off:],
                                      d["erank"][off:])
            # ---- row 0: TEST loss, CI, validation-selected saturation entrance ------------------
            ax = axes[0, col]
            ax.axvspan(ls, last, color=TUNNEL_FILL, lw=0, zorder=0)
            ax.axvline(ls, color=TUNNEL_LINE, lw=1.1, zorder=1,
                       label="Saturation entrance (validation)")
            ax.fill_between(x, lo_, hi_, color=LOSS_BAND, alpha=0.95, lw=0, zorder=2,
                            label=f"95% bootstrap CI ($B={boot_b}$)")
            ax.plot(x, point, "-o", ms=3.0, color=LOSS, mfc=LOSS, mec=LOSS, zorder=3,
                    label=f"Test loss (mean of {len(RUN_SEEDS)} seeds)")
            ax.set_title(TITLES[d["tag"]], fontweight="bold")
            span = hi_.max() - lo_.min()              # headroom: never clip the CI band
            ax.set_ylim(lo_.min() - 0.06 * span, hi_.max() + 0.10 * span)
            ax.annotate(labels[ls], xy=(ls, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(2, -9), textcoords="offset points", fontsize=7.5,
                        color=TUNNEL_LINE, ha="left", va="top")

            # ---- row 1: effective rank ------------------------------------------------------
            ax = axes[1, col]
            ax.axvspan(ls, last, color=TUNNEL_FILL, alpha=0.55, lw=0, zorder=0)
            ax.axvline(ls, color=TUNNEL_LINE, lw=0.9, ls=(0, (4, 2)), alpha=0.75, zorder=1)
            ax.plot(x, erank, "-o", ms=3.0, color=ERANK, mfc=ERANK, mec=ERANK, zorder=3,
                    label="Effective rank")
            pk = d["peak"] - off
            ax.plot([pk], [erank[pk]], "*", ms=9.0, color=ERANK, mec="white", mew=0.6,
                    zorder=5, label=f"Peak effective rank")
            ax.set_ylim(0, erank.max() * 1.08)      # no in-panel text -> less dead space

        for ax in axes.ravel():
            ax.set_xlim(-0.55, last + 0.55)
            ax.set_xticks(x)
            ax.tick_params(length=2.5, pad=1.5)
            ax.grid(axis="y", alpha=0.18, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for ax in axes[-1, :]:
            ax.set_xticklabels(labels, rotation=45, ha="right")
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
    if tag in PT_OOD_FIG_TAGS:                    # no committed tunnel record for these targets:
        from probing.tunnel import TUNNEL_TOL, tunnel_start   # recompute at the same tolerance
        l_start = int(tunnel_start(_ptood_panel_curves(tag)[2], tol=TUNNEL_TOL))
    else:
        rec = json.load(open(_need(
            TUNNEL_DIR / f"{tag}__fslot__{QSET}__{PROTO}__{RUNS_TAG}.json",
            "python -m experiments.run_ptood_probing_ftok --quantile-set q1 --tunnels-only")))
        l_start = int(rec["l_start"])
    return {"tag": tag, "M": M, "l_start": l_start}


def load_cka_content(tag):
    """14x14 linear-CKA matrix for the content-pooled readout of one dataset.

    Mirrors ``load_cka`` but reads the ext_v4_future_tokens_content matrices and carries no tunnel
    entrance (``l_start=None`` -> no dashed line): the content readout has no committed forecasting
    tunnel for the PT-OOD targets, and CKA is probe-independent, so the heatmaps stand on their own.
    """
    m = _need(CKA_CONTENT_MAT_DIR / f"{tag}__content__layerxlayer.npy",
              "python -m experiments.run_cka --readout content")
    M = np.load(m).astype(np.float64)
    n = len(LABELS)
    if M.shape != (n, n):
        raise ValueError(f"{tag}: content CKA matrix is {M.shape}, expected ({n}, {n})")
    if not np.allclose(np.diag(M), 1.0) or not np.allclose(M, M.T):
        raise ValueError(f"{tag}: content CKA matrix is not symmetric with unit diagonal")
    return {"tag": tag, "M": M, "l_start": None}


def load_cka_cslot(tag):
    """14x14 linear-CKA matrix for the CONTENT-SLOT readout of one dataset.

    Same contract as load_cka_content (l_start=None -> no dashed entrance line; CKA is
    probe-independent). Produced by run_cka_analysis --extv4-cslot off the cslotL_K4_H64 caches."""
    m = _need(CKA_CSLOT_MAT_DIR / f"{tag}__cslot__layerxlayer.npy",
              "python -m experiments.run_cka_analysis --extv4-cslot")
    M = np.load(m).astype(np.float64)
    n = len(LABELS)
    if M.shape != (n, n):
        raise ValueError(f"{tag}: content-slot CKA matrix is {M.shape}, expected ({n}, {n})")
    if not np.allclose(np.diag(M), 1.0) or not np.allclose(M, M.T):
        raise ValueError(f"{tag}: content-slot CKA matrix is not symmetric with unit diagonal")
    return {"tag": tag, "M": M, "l_start": None}


def make_cka_figure(rows, title, stem, dpi=400, show_title=True, ncol=2, font_scale=1.0,
                    out_dir=None):
    """Layer-by-layer linear-CKA heatmaps with one shared colour bar.

    Datasets wrap into rows of ``ncol`` panels; the panel box is square (the matrices are), so the
    figure grows with the number of datasets instead of squeezing them. ``font_scale`` scales every
    piece of type; the panel box grows only 0.55x as fast, so the text gets bigger relative to the
    heatmaps rather than the heatmaps getting smaller."""
    nr = -(-len(rows) // ncol)
    mixed = (any(d["tag"] in PT_OOD_FIG_TAGS for d in rows)
             and any(d["tag"] not in PT_OOD_FIG_TAGS for d in rows))
    base = {**PAPER_RC, **CKA_RC_BUMP}
    rc = {**base, **{k: base[k] * font_scale for k in
                     ("font.size", "axes.labelsize", "axes.titlesize", "legend.fontsize",
                      "xtick.labelsize", "ytick.labelsize")}}
    grow = 1 + 0.55 * (font_scale - 1)
    tick_pt, sup_pt, ttl_pt = 10 * font_scale, 13 * font_scale, 12 * font_scale
    with plt.rc_context(rc):
        fig, axes = plt.subplots(nr, ncol, figsize=(3.6 * grow * ncol, 3.8 * grow * nr),
                                 layout="constrained", squeeze=False)
        fig.get_layout_engine().set(h_pad=0.04, w_pad=0.06, wspace=0.06)
        n = len(LABELS)
        for ax, d in zip(axes.ravel(), rows):
            im = ax.imshow(d["M"], cmap="viridis", vmin=0.0, vmax=1.0,
                           origin="upper", interpolation="nearest")
            if d["l_start"] is not None:               # validation-selected forecasting entrance
                b = d["l_start"] - 0.5                  # boundary sits between the two cells
                for line in (ax.axvline, ax.axhline):
                    line(b, color="white", lw=0.9, ls=(0, (3, 2)), alpha=0.85)
            kind = "PT-OOD" if d["tag"] in PT_OOD_FIG_TAGS else "PT-ID"
            ax.set_title(f"{TITLES[d['tag']]}  [{kind}]" if mixed else TITLES[d["tag"]],
                         fontweight="bold")
            ax.set_xticks(range(n)); ax.set_yticks(range(n))
            ax.set_xticklabels(LABELS, rotation=90, fontsize=tick_pt)
            ax.set_yticklabels(LABELS, fontsize=tick_pt)
            ax.tick_params(length=2.0, pad=1.2)
        for j, ax in enumerate(axes.ravel()):
            if j >= len(rows):
                ax.axis("off")                          # blank the unused slots
            elif j % ncol:
                ax.set_yticklabels([])                  # row labels only on the first column
        fig.supxlabel("Representation point", fontsize=sup_pt)
        fig.supylabel("Representation point", fontsize=sup_pt)
        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.02, shrink=0.92)
        cb.set_label("Linear CKA")
        cb.ax.tick_params(labelsize=tick_pt)
        if show_title:
            fig.suptitle(title, fontsize=ttl_pt, fontweight="bold")
        dst = out_dir if out_dir is not None else CKA_FIG_DIR
        dst.mkdir(parents=True, exist_ok=True)
        pdf, png = dst / f"{stem}.pdf", dst / f"{stem}.png"
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
LEGEND_SCALE_REF = 1.25          # the legend scale the panel geometry is calibrated at

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


def make_cka_probe_figure(tags, boot_b, seed, stem="main_id_cka_probe_2x2", dpi=400,
                          show_title=False, drop_emb=True, show_panel_titles=False, gap=0.12,
                          out_dir=None):
    """2 x len(tags): row 0 = the shared forecast-slot probe's TEST loss, row 1 = the forecast-slot
    layer x layer CKA of the representations that probe reads.

    A column is two views of ONE depth sweep on ONE dataset -- how linearly decodable the forecast
    is (top) and how similar the representations are to each other (bottom) -- so both rows sit on
    the SAME representation-point axis, including the gap before the post-final-norm point.

    ``L12 (post-LN)`` is the encoder's post-final-norm state (the native head's actual input). It is
    a different KIND of readout point from a block output, so it is set apart by a gap and drawn as
    a detached open marker rather than joined to the curve. ``gap`` is that separation in cell
    widths: it is also the width of the blank cell in the heatmap, so keep it small enough that the
    white stripe reads as a seam rather than a band. The open marker and the axis label carry the
    distinction; the gap only has to hint at it.

    ``drop_emb`` removes the input embedding from BOTH rows: it is where the loss curve starts its
    steepest fall and, being near rank-1, the point that dominates the CKA colour scale.

    The dotted line is the 5% tolerance above the FINAL point's test loss -- the loss-side reading
    of the same "95% rule" that defines the tunnel (lower is better here, so the band sits above).
    The shaded region starts at the VALIDATION-selected entrance, so the test curve need not cross
    the dotted line exactly where the shading begins; that gap is a real train/test difference.

    No bootstrap CI is drawn. ``load_dataset`` still computes it, because the point estimate it
    returns alongside is what the committed-record gate checks."""
    from probing.tunnel import TUNNEL_TOL

    off = 1 if drop_emb else 0
    # the paper module spells the last point "L12+RMS"; this figure follows the screenshot style
    labels = (["Embed"] if not off else []) + [f"L{i}" for i in range(1, 13)] + ["L12 (post-LN)"]
    rows = [load_dataset(t, boot_b, seed) for t in tags]
    ckas = [load_cka(t) for t in tags]
    for d, c in zip(rows, ckas):
        if d["l_start"] != c["l_start"]:            # same record feeds both; a mismatch = drift
            raise ValueError(f"{d['tag']}: probe and CKA disagree on the entrance "
                             f"({d['l_start']} vs {c['l_start']})")
        if d["l_start"] < off:
            raise ValueError(f"{d['tag']}: entrance falls on the dropped Emb point")

    # x geometry: block outputs on the integer grid, then a GAP, then the post-final-norm point.
    # The mesh gets an EMPTY (NaN) cell spanning the gap, so the heatmap's post-LN row/column is
    # detached exactly like the curve's marker instead of the L12 cell stretching to fill it.
    n = len(labels)
    nb = n - 1                                     # number of block-output points
    GAP = float(gap)
    pos = np.concatenate([np.arange(nb), [nb + GAP]])
    edges = np.concatenate([np.arange(nb + 1) - 0.5, [nb + GAP - 0.5, nb + GAP + 0.5]])
    xlo, xhi = edges[0] - 0.05, edges[-1] + 0.05

    def _with_gap(M):
        """(n, n) -> (n+1, n+1) with a NaN row/column inserted before the post-final-norm point."""
        k = M.shape[0] - 1
        A = np.full((M.shape[0] + 1, M.shape[1] + 1), np.nan)
        A[:k, :k], A[:k, k + 1:] = M[:k, :k], M[:k, k:]
        A[k + 1:, :k], A[k + 1:, k + 1:] = M[k:, :k], M[k:, k:]
        return A

    gap_cmap = plt.get_cmap("viridis").copy()
    gap_cmap.set_bad(alpha=0.0)                    # the gap cell draws as nothing
    pct_lo, pct_hi = int(round((1 - TUNNEL_TOL) * 100)), int(round((1 + TUNNEL_TOL) * 100))

    with plt.rc_context(PAPER_RC):
        fig, axes = plt.subplots(2, len(tags), figsize=(3.7 * len(tags) + 0.7, 6.2),
                                 layout="constrained", squeeze=False, sharex="col")
        fig.get_layout_engine().set(h_pad=0.05, w_pad=0.06, hspace=0.06, wspace=0.10)
        im = None

        for col, (d, c) in enumerate(zip(rows, ckas)):
            ls = d["l_start"] - off
            point = d["point"][off:]

            # ---- row 0: shared forecast-slot probe, TEST loss ------------------------------
            ax = axes[0, col]
            thr = float(point[-1] * (1.0 + TUNNEL_TOL))
            # the tunnel is a statement about the BLOCK stack, so the shading stops at L12 and
            # does not run under the detached post-final-norm point
            ax.axvspan(pos[ls] - 0.5, pos[-2] + 0.5, color=TUNNEL_FILL, lw=0, zorder=0,
                       label=f"nominal tunnel ({pct_lo}% rule)")
            ax.axhline(thr, color=LOSS, lw=0.9, ls=(0, (2, 2)), alpha=0.8, zorder=2,
                       label=f"{pct_hi}% of final-point loss")
            ax.plot(pos[:-1], point[:-1], "-o", ms=3.6, color=LOSS, mfc=LOSS, mec=LOSS, zorder=3,
                    label="shared forecast-slot probe")
            ax.plot(pos[-1], point[-1], "s", ms=6.0, mfc="none", mec=LOSS, mew=1.4, zorder=4,
                    label="probe, post-LN")
            top, bot = max(point.max(), thr), min(point.min(), thr)
            span = top - bot
            ax.set_ylim(bot - 0.10 * span, top + 0.10 * span)
            ax.set_ylabel("test quantile loss", color=LOSS)
            ax.tick_params(axis="y", colors=LOSS)
            ax.spines["left"].set_color(LOSS)
            if not col:
                pass
            else:
                ax.set_ylabel("")
            if show_panel_titles:
                ax.set_title(TITLES[d["tag"]], fontweight="bold")
            ax.grid(axis="y", alpha=0.18, lw=0.5)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)

            # ---- row 1: layer x layer forecast-slot CKA -----------------------------------
            ax = axes[1, col]
            im = ax.pcolormesh(edges, edges, _with_gap(c["M"][off:, off:]), cmap=gap_cmap,
                               vmin=0.0, vmax=1.0, shading="flat")
            ax.invert_yaxis()                      # L1 at the top, like every other CKA panel
            ax.set_yticks(pos)
            ax.set_yticklabels(labels if col == 0 else [])
            ax.set_xticks(pos)
            ax.set_xticklabels(labels, rotation=45, ha="right")
            ax.set_xlim(xlo, xhi)

        for ax in axes.ravel():
            ax.tick_params(length=2.5, pad=1.5)
        for ax in axes[0, :]:
            ax.tick_params(labelbottom=False)
        axes[1, 0].set_ylabel("representation point")

        cb = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.030, pad=0.015, shrink=0.45,
                          anchor=(0.0, 0.0))
        cb.set_label("linear CKA")
        cb.ax.tick_params(length=2.0)

        h, l = axes[0, 0].get_legend_handles_labels()
        order = sorted(range(len(l)), key=lambda i: ("probe" not in l[i], "post-LN" in l[i]))
        fig.legend([h[i] for i in order], [l[i] for i in order], loc="outside lower center",
                   ncol=2, frameon=True, fancybox=False, edgecolor="0.7", framealpha=1.0,
                   handlelength=1.8, handletextpad=0.5, columnspacing=1.4, borderpad=0.5,
                   borderaxespad=0.3, fontsize=9.0)
        if show_title:
            fig.suptitle("Linear decodability and representation similarity by depth",
                         fontsize=10, fontweight="bold")

        dst = out_dir if out_dir is not None else FIG_DIR
        dst.mkdir(parents=True, exist_ok=True)
        pdf, png = dst / f"{stem}.pdf", dst / f"{stem}.png"
        fig.savefig(pdf)
        fig.savefig(png, dpi=dpi)
        plt.close(fig)
    return pdf, png


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


def make_eps_figure(rows, title, stem, boot_b, epsilons=EPS_BANDS, ncol=None, dpi=400,
                    show_title=True, font_scale=1.0, legend_scale=LEGEND_SCALE_REF,
                    panel_w=3.45, panel_h=4.95):
    """Test loss + effective rank per dataset, with one shaded band per tolerance.

    The bands nest: looser tolerances open earlier and are drawn lighter, so the darkest region is
    where every tolerance agrees the representation has saturated. Datasets wrap into blocks of
    ``ncol`` columns, each block being a loss row above an effective-rank row, so seven datasets
    fit a page instead of one very wide strip.
    """
    order = sorted((float(e) for e in epsilons), reverse=True)      # loosest first, drawn behind
    n = len(rows)
    ncol = ncol or n
    nblk = -(-n // ncol)
    # Scale type without shrinking the axes: the panel box grows with the fonts, so a larger
    # font_scale makes the text bigger rather than squeezing the curves.
    rc = {**PAPER_RC,
          **{k: PAPER_RC[k] * font_scale for k in
             ("font.size", "axes.labelsize", "axes.titlesize",
              "xtick.labelsize", "ytick.labelsize")},
          "legend.fontsize": PAPER_RC["legend.fontsize"] * font_scale * legend_scale,
          "lines.linewidth": PAPER_RC["lines.linewidth"] * (1 + 0.4 * (font_scale - 1)),
          "axes.linewidth": PAPER_RC["axes.linewidth"] * (1 + 0.4 * (font_scale - 1))}
    grow = 1 + 0.55 * (font_scale - 1)                    # give the type room, keep the aspect
    ms_pt, ms_star, ann_pt = 3.0 * grow, 9.0 * grow, 7.0 * font_scale
    with plt.rc_context(rc):
        fig, axes = plt.subplots(2 * nblk, ncol,
                                 figsize=(panel_w * grow * ncol, panel_h * grow * nblk),
                                 layout="constrained", squeeze=False)
        fig.get_layout_engine().set(h_pad=0.05, w_pad=0.06, hspace=0.10, wspace=0.12)
        x = np.arange(len(LABELS))
        last = len(LABELS) - 1

        for i, d in enumerate(rows):
            blk, col = divmod(i, ncol)
            ax_loss, ax_er = axes[2 * blk, col], axes[2 * blk + 1, col]
            for ax in (ax_loss, ax_er):
                for e in order:                                     # nested bands
                    ax.axvspan(d["starts"][e], last, color=EPS_FILL.get(e, "#E4F0E4"), lw=0,
                               zorder=0)
                for e in order:
                    ax.axvline(d["starts"][e], color=TUNNEL_LINE, lw=1.1, zorder=1,
                               ls="-" if e == max(order) else (0, (3, 2)),
                               label=(f"Saturation entrance, $\\varepsilon={int(round(e * 100))}\\%$"
                                      if i == 0 and ax is ax_loss else None))
                ax.set_xlim(-0.5, last + 0.5)

            ax_loss.fill_between(x, d["lo"], d["hi"], color=LOSS_BAND, alpha=0.95, lw=0, zorder=2,
                                 label=f"95% bootstrap CI ($B={boot_b}$)" if i == 0 else None)
            ax_loss.plot(x, d["point"], "-o", ms=3.0, color=LOSS, mfc=LOSS, mec=LOSS, zorder=3,
                         label=f"Test loss (mean of {len(RUN_SEEDS)} seeds)" if i == 0 else None)
            ax_loss.set_title(f"{EPS_TITLES[d['tag']]}  [{d['kind']}]", fontweight="bold")
            span = d["hi"].max() - d["lo"].min()
            ax_loss.set_ylim(d["lo"].min() - 0.06 * span, d["hi"].max() + 0.18 * span)
            ax_loss.set_xticks(x)
            ax_loss.set_xticklabels([])
            seen = set()                                  # tolerances often coincide -> label once
            for k, e in enumerate(order):
                pos = d["starts"][e]
                if pos in seen:
                    continue
                seen.add(pos)
                edge = pos >= last - 2                    # keep late labels inside the axes
                ax_loss.annotate(LABELS[pos], xy=(pos, 1.0), xycoords=("data", "axes fraction"),
                                 xytext=((-2 if edge else 2) * font_scale,
                                         (-9 - 10 * len(seen - {pos})) * font_scale),
                                 textcoords="offset points", fontsize=ann_pt, color=TUNNEL_LINE,
                                 ha="right" if edge else "left", va="top")

            ax_er.plot(x, d["erank"], "-o", ms=3.0, color=ERANK, mfc=ERANK, mec=ERANK, zorder=3,
                       label="Effective rank" if i == 0 else None)
            ax_er.plot(d["peak"], d["erank"][d["peak"]], "*", ms=ms_star, color=ERANK, mec="white",
                       mew=0.6, zorder=4, label="Peak effective rank" if i == 0 else None)
            ax_er.set_xticks(x)
            ax_er.set_xticklabels(LABELS, rotation=45, ha="right")
            ax_er.set_xlabel("Representation point")
            if col == 0:
                ax_loss.set_ylabel("Test quantile loss")
                ax_er.set_ylabel("Effective rank")

        for j in range(n, nblk * ncol):                   # blank the unused slots
            blk, col = divmod(j, ncol)
            for r in (2 * blk, 2 * blk + 1):
                axes[r, col].axis("off")

        h, l = [], []
        for a in (axes[0, 0], axes[1, 0]):
            hh, ll = a.get_legend_handles_labels()
            h += hh
            l += ll
        leg_pt = rc["legend.fontsize"]
        leg = fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
        fig.canvas.draw()
        w_px = fig.get_size_inches()[0] * fig.dpi
        if leg.get_window_extent().width > 0.98 * w_px:
            # A scaled-up legend can outgrow the canvas and get clipped; back it off to the
            # largest size that still fits, and say so rather than writing a cropped figure.
            leg_pt *= 0.98 * w_px / leg.get_window_extent().width
            leg.remove()
            leg = fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False,
                             fontsize=leg_pt)
            fig.canvas.draw()
            print(f"[warn] legend_scale clipped to "
                  f"{leg_pt / (PAPER_RC['legend.fontsize'] * font_scale):.2f} to fit the width")
        eff_leg = leg_pt / (PAPER_RC["legend.fontsize"] * font_scale)
        if abs(eff_leg - LEGEND_SCALE_REF) > 1e-9:
            # An outside legend takes its height out of the axes, so a bigger legend would squash
            # the panels. Buy the extra height from the canvas instead: the legend height is
            # ~linear in its font size, so at LEGEND_SCALE_REF it would have been
            # leg_h * LEGEND_SCALE_REF / eff_leg. The panels then keep the size they have at the
            # default, where this branch is skipped and the figure is unchanged.
            leg_h = leg.get_window_extent().height / fig.dpi
            w_in, h_in = fig.get_size_inches()
            fig.set_size_inches(w_in, h_in + leg_h * (1 - LEGEND_SCALE_REF / eff_leg))
        if show_title:
            fig.suptitle(title, fontweight="bold")
        for ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{stem}.{ext}", dpi=dpi)
        plt.close(fig)
    print(f"[write] {FIG_DIR / (stem + '.png')}")
    for d in rows:
        print(f"    {EPS_TITLES[d['tag']]:<13} " +
              "  ".join(f"eps={e:.0%} -> {LABELS[d['starts'][e]]}" for e in order))


# ---------------------------------------------------------------------------------------
# Transfer vs the target's OWN probe (the "how much did we lose by not training here?" view)
# ---------------------------------------------------------------------------------------
# make_delta_figure references every point to the SAME curve's final point, which answers the
# tunnel question ("is an intermediate layer better than L12+RMS *within* this transfer?") but
# says nothing about whether the transferred probe is any good in absolute terms. The reference
# here is instead the target's OWN probe on the SAME test windows, so the vertical distance
# between the two lines IS the transfer penalty at that depth.
#
# On top of the full layerwise curves this reports ONE selection-based scalar per cell, the
# transfer gap of the v4 design:
#
#     gap(s,t) = L_{s->t}(l_s) / L_{t->t}(l_t) - 1
#
# with l_s and l_t chosen INDEPENDENTLY by the 5% first-crossing rule on each dataset's own
# VALIDATION curve -- no test data touches either choice. Because l_s != l_t in general, that
# scalar mixes two effects, so it is reported together with its exact multiplicative split:
#
#     1 + gap = [ L_{s->t}(l_s) / L_{t->t}(l_s) ] * [ L_{t->t}(l_s) / L_{t->t}(l_t) ]
#                 probe penalty (depth FIXED)        depth mismatch (target's own curve)
#
# This matters: Electricity->Uber reads as gap -1.4% (better than Uber's own probe) while the
# probe penalty is +5.5% -- the whole apparent win is a -6.5% depth-mismatch term.
#
# Where each target's own-probe curve comes from: the 4 PT-ID targets have a diagonal cell in
# the 4x4 run (probe fit on the target, scored on its own test windows); the 3 PT-OOD targets
# have no diagonal there, so their fresh per-target probe comes from run_ptood_probing_ftok's
# default mode -- same estimand (fslot probe, wd chosen on that dataset's own validation split),
# different producer, exactly as _ptood_panel_curves already documents.
ID_REF_SUB = {"monash_electricity_hourly": "cross_dataset", "uber_tlc_hourly": "cross_dataset",
              "m4_hourly": "cross_dataset", "wind_farms_hourly": "cross_dataset",
              "sg_carpark": "ptood", "coastal_ts": "ptood", "boom_hourly": "ptood"}
TVI_TRANSFER, TVI_OWN = "#7570B3", "#111111"       # M4's source colour vs a neutral reference
TVI_CRITERION = "first_crossing_95"                # the rule the committed tunnel records carry


def _seed_mean_npz(paths, how):
    """Per-window test losses averaged over the 3 probe-init runs + the shared series ids."""
    wls, sids = [], []
    for p in paths:
        z = np.load(_need(p, how))
        wls.append(np.asarray(z["window_loss"], np.float64))
        sids.append(np.asarray(z["series_test"], np.int64))
    assert all(w.shape == wls[0].shape for w in wls), f"{paths[0].name}: windows differ across runs"
    assert all(np.array_equal(s, sids[0]) for s in sids), f"{paths[0].name}: series differ across runs"
    return np.mean(wls, axis=0), sids[0]


def load_own_probe(tag):
    """(seed-mean per-window TEST loss, series ids) for the probe fit on `tag` itself."""
    if ID_REF_SUB[tag] == "cross_dataset":
        return _seed_mean_npz([V4 / QSET / "cross_dataset" / "bootstrap_inputs" /
                               f"{tag}__to__{tag}__{QSET}__seed{s}.npz" for s in RUN_SEEDS],
                              "python -m experiments.run_fslot_transfer  (the 4x4 diagonal)")
    return _seed_mean_npz([PTOOD_FIG_DIR / "bootstrap_inputs" / f"{tag}__{QSET}__seed{s}.npz"
                           for s in RUN_SEEDS],
                          f"python -m experiments.run_ptood_probing_ftok --quantile-set {QSET}")


def val_curve(tag):
    """Mean-over-runs VALIDATION curve (14 points) for `tag`. PT-ID reads the committed tunnel
    record; PT-OOD has no such record and reads the per-target JSONs the fresh-probe run wrote."""
    if tag in PT_OOD_FIG_TAGS:
        return np.mean([json.load(open(_need(
            PTOOD_FIG_DIR / "per_target" / f"{tag}__{QSET}__seed{s}.json",
            f"python -m experiments.run_ptood_probing_ftok --quantile-set {QSET}")))
            ["val_loss_by_layer"] for s in RUN_SEEDS], axis=0)
    rec = json.load(open(_need(
        TUNNEL_DIR / f"{tag}__fslot__{QSET}__{PROTO}__{RUNS_TAG}.json",
        f"python -m experiments.run_ptood_probing_ftok --quantile-set {QSET} --tunnels-only")))
    return np.asarray(rec["mean_val_loss_by_layer"], np.float64)


def first_crossing_layer(tag, tol=None):
    """Earliest representation point within `tol` of the FINAL point's VALIDATION loss.

    Reuses probing.tunnel.tunnel_start (the repo's authoritative criterion) rather than
    re-implementing the scan. That function's definition has changed once before, so for every
    tag that HAS a committed tunnel record this gates the recomputation against it: the record's
    own `tunnel_definition`, `tolerance` and `l_start` must all agree. A criterion change then
    fails loud here instead of silently re-selecting every layer. The 3 PT-OOD targets carry no
    record, so they rest on the gate the PT-ID tags just passed."""
    from probing.tunnel import TUNNEL_TOL, tunnel_start
    tol = TUNNEL_TOL if tol is None else float(tol)
    l = int(tunnel_start(val_curve(tag), tol=tol))
    if tag not in PT_OOD_FIG_TAGS:
        rec = json.load(open(TUNNEL_DIR / f"{tag}__fslot__{QSET}__{PROTO}__{RUNS_TAG}.json"))
        if rec["tunnel_definition"] != TVI_CRITERION:
            raise ValueError(
                f"{tag}: this figure selects layers with the {TVI_CRITERION} rule, but the "
                f"committed tunnel record was built under '{rec['tunnel_definition']}' -- "
                "re-run --tunnels-only, or state which criterion the paper uses")
        if abs(float(rec["tolerance"]) - tol) > 1e-12 or int(rec["l_start"]) != l:
            raise ValueError(
                f"{tag}: recomputed first-crossing layer L{l} at tol={tol} disagrees with the "
                f"committed record (l_start=L{rec['l_start']}, tol={rec['tolerance']}) -- "
                "probing.tunnel.tunnel_start no longer reproduces the committed selection")
    return l


def load_transfer_vs_own(src, targets, boot_b, seed, tol=None):
    """One row per target: both layerwise curves, their CIs, the selected layers and the gap."""
    if src not in SRC_COLOR:
        raise ValueError(f"source '{src}' has no 4x4/pt_ood transfer cells; this figure reads "
                         f"{sorted(SRC_COLOR)} (BOOM-as-source lives in run_boom_source_transfer)")
    ls = first_crossing_layer(src, tol)
    rows = []
    for tgt in targets:
        sub = "cross_dataset" if ID_REF_SUB[tgt] == "cross_dataset" else "unseen"
        tw, ts = _seed_mean_npz([V4 / QSET / sub / "bootstrap_inputs" /
                                 f"{src}__to__{tgt}__{QSET}__seed{s}.npz" for s in RUN_SEEDS],
                                "python -m experiments.run_fslot_transfer (see the ext_v4 recipe)")
        ow, os_ = load_own_probe(tgt)
        # The overlay is only honest if both probes were scored on the SAME windows in the SAME
        # order. Fail loud rather than silently plotting two different test sets on one axis.
        if tw.shape != ow.shape or not np.array_equal(ts, os_):
            raise ValueError(f"{src}->{tgt}: the transferred and own-probe evaluations do not "
                             f"share test windows ({tw.shape} vs {ow.shape}) -- refusing to overlay")
        # Identical cluster ids => identical S => _layer_mean_boot draws the SAME multinomial
        # count matrix for both curves at this (B, seed), so every ratio below is formed INSIDE
        # paired replicates -- the same construction as tunnel.d_stat_boot.
        tp, tb = _layer_mean_boot(tw, ts, B=boot_b, seed=seed)
        op, ob = _layer_mean_boot(ow, os_, B=boot_b, seed=seed)
        lt = first_crossing_layer(tgt, tol)
        gap, gap_b = 100.0 * (tp / op - 1.0), 100.0 * (tb / ob - 1.0)
        glo, ghi = ci_bounds(gap_b)
        tlo, thi = ci_bounds(tb)
        olo, ohi = ci_bounds(ob)

        def _stat(num_p, num_b, den_p, den_b):
            pt = 100.0 * (num_p / den_p - 1.0)
            bt = 100.0 * (num_b / den_b - 1.0)
            lo, hi = ci_bounds(bt)
            return {"pct": float(pt), "lo": float(lo), "hi": float(hi),
                    "excludes_zero": bool(lo > 0 or hi < 0)}

        total = _stat(tp[ls], tb[:, ls], op[lt], ob[:, lt])       # the v4 transfer gap
        penalty = _stat(tp[ls], tb[:, ls], op[ls], ob[:, ls])     # depth FIXED at l_s
        depth = _stat(op[ls], ob[:, ls], op[lt], ob[:, lt])       # target's own curve only
        rows.append({"source": src, "target": tgt,
                     "kind": "PT-OOD" if tgt in PT_OOD_FIG_TAGS else "PT-ID",
                     "transfer": tp, "t_lo": tlo, "t_hi": thi,
                     "own": op, "o_lo": olo, "o_hi": ohi,
                     "gap": gap, "gap_lo": glo, "gap_hi": ghi,
                     "l_s": ls, "l_t": lt, "total": total, "penalty": penalty, "depth": depth,
                     "n_windows": int(tw.shape[1]), "n_clusters": int(np.unique(ts).size)})
    return rows


def write_transfer_vs_own_table(rows, src, boot_b, seed):
    """Tidy CSV: the selected layers, the gap, its two factors, and the final-point gap."""
    d_ = TRANSFER_OUT / "tables"
    d_.mkdir(parents=True, exist_ok=True)
    p = d_ / f"transfer_vs_own__{SHORT[src].lower().replace(' ', '_').replace('-', '')}__{QSET}.csv"
    cols = ["source", "target", "target_kind", "quantile_set", "criterion", "tolerance",
            "l_s", "l_s_label", "l_t", "l_t_label",
            "loss_transfer_at_l_s", "loss_own_at_l_t", "loss_own_at_l_s",
            "gap_pct", "gap_ci_lo", "gap_ci_hi", "gap_excludes_zero",
            "probe_penalty_pct", "probe_penalty_ci_lo", "probe_penalty_ci_hi",
            "probe_penalty_excludes_zero",
            "depth_mismatch_pct", "depth_mismatch_ci_lo", "depth_mismatch_ci_hi",
            "gap_at_final_pct", "gap_at_final_ci_lo", "gap_at_final_ci_hi",
            "n_windows", "n_clusters", "boot_b", "boot_seed"]
    from probing.tunnel import TUNNEL_TOL
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            ls, lt = r["l_s"], r["l_t"]
            w.writerow({
                "source": r["source"], "target": r["target"], "target_kind": r["kind"],
                "quantile_set": QSET, "criterion": TVI_CRITERION, "tolerance": TUNNEL_TOL,
                "l_s": ls, "l_s_label": LABELS[ls], "l_t": lt, "l_t_label": LABELS[lt],
                "loss_transfer_at_l_s": r["transfer"][ls], "loss_own_at_l_t": r["own"][lt],
                "loss_own_at_l_s": r["own"][ls],
                "gap_pct": r["total"]["pct"], "gap_ci_lo": r["total"]["lo"],
                "gap_ci_hi": r["total"]["hi"], "gap_excludes_zero": r["total"]["excludes_zero"],
                "probe_penalty_pct": r["penalty"]["pct"], "probe_penalty_ci_lo": r["penalty"]["lo"],
                "probe_penalty_ci_hi": r["penalty"]["hi"],
                "probe_penalty_excludes_zero": r["penalty"]["excludes_zero"],
                "depth_mismatch_pct": r["depth"]["pct"], "depth_mismatch_ci_lo": r["depth"]["lo"],
                "depth_mismatch_ci_hi": r["depth"]["hi"],
                "gap_at_final_pct": r["gap"][-1], "gap_at_final_ci_lo": r["gap_lo"][-1],
                "gap_at_final_ci_hi": r["gap_hi"][-1],
                "n_windows": r["n_windows"], "n_clusters": r["n_clusters"],
                "boot_b": boot_b, "boot_seed": seed})
    return p


def make_transfer_vs_own_figure(rows, src, boot_b, dpi=400, show_title=True, ncol=3,
                                stem=None):
    """2 x 3 panels: the frozen `src` probe and the target's own probe on the same windows.

    Absolute test loss, so the vertical distance between the lines is the transfer penalty in
    the units the probe is actually scored in. Stars mark each probe's INDEPENDENTLY validation-
    selected layer (5% first crossing): the source's l_s on the transferred curve, the target's
    l_t on its own curve -- the two points the reported gap compares. y-limits are PER PANEL;
    these are different datasets and their losses are not comparable to each other."""
    x = np.arange(len(LABELS))
    nr = int(np.ceil(len(rows) / ncol))
    with plt.rc_context({**PAPER_RC, "xtick.labelsize": 8.5, "ytick.labelsize": 9}):
        fig, axes = plt.subplots(nr, ncol, figsize=(3.45 * ncol, 3.0 * nr), layout="constrained",
                                 squeeze=False)
        for ax, r in zip(axes.ravel(), rows):
            ls, lt = r["l_s"], r["l_t"]
            ax.fill_between(x, r["own"], r["transfer"], color=TVI_TRANSFER, alpha=0.13, lw=0,
                            zorder=1)
            for key, lo, hi, c, m in (("own", "o_lo", "o_hi", TVI_OWN, "o"),
                                      ("transfer", "t_lo", "t_hi", TVI_TRANSFER, "s")):
                ax.fill_between(x, r[lo], r[hi], color=c, alpha=0.18, lw=0, zorder=2)
                ax.plot(x, r[key], color=c, marker=m, ms=3.2, lw=1.5, zorder=3)
            for l, key, c in ((lt, "own", TVI_OWN), (ls, "transfer", TVI_TRANSFER)):
                ax.axvline(l, color=c, ls=":", lw=0.9, alpha=0.6, zorder=0)
                ax.plot([l], [r[key][l]], marker="*", ms=12, color=c, mec="white", mew=0.8,
                        ls="none", zorder=5)
            t_, p_, d_ = r["total"], r["penalty"], r["depth"]
            ax.set_title(f"{SHORT[r['target']]}   ({r['kind']})", fontsize=10.5,
                         fontweight="bold", loc="left", pad=3)
            ax.text(0.97, 0.95,
                    f"$\\ell_s$={LABELS[ls]}  vs  $\\ell_t$={LABELS[lt]}\n"
                    f"gap {t_['pct']:+.1f}%  [{t_['lo']:+.1f}, {t_['hi']:+.1f}]\n"
                    f"probe {p_['pct']:+.1f}%  $\\times$  depth {d_['pct']:+.1f}%",
                    transform=ax.transAxes, ha="right", va="top", fontsize=7.6,
                    linespacing=1.35,
                    bbox=dict(fc="white", ec="0.8", lw=0.6, alpha=0.92, pad=2.4))
            ax.set_xticks(x)
            ax.set_xticklabels(LABELS, rotation=45, ha="right")
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_axisbelow(True)
        for ax in axes[:, 0]:
            ax.set_ylabel("Test quantile loss ($Q = 1$)")
        for ax in axes[-1]:
            ax.set_xlabel("Representation point")
        for ax in axes.ravel()[len(rows):]:
            ax.set_visible(False)
        h = [plt.Line2D([], [], color=TVI_OWN, marker="o", ms=3.6, lw=1.5),
             plt.Line2D([], [], color=TVI_TRANSFER, marker="s", ms=3.6, lw=1.5),
             plt.Rectangle((0, 0), 1, 1, fc=TVI_TRANSFER, alpha=0.13, ec="none"),
             plt.Line2D([], [], color="0.35", marker="*", ms=10, ls="none")]
        fig.legend(h, ["target's own probe (fit on the target)",
                       f"{SHORT[src]} probe, transferred frozen",
                       "transfer gap",
                       "validation-selected layer (5% first crossing)"],
                   loc="outside lower center", ncol=4, frameon=False)
        if show_title:
            fig.suptitle(f"Transferred {SHORT[src]} probe vs each target's own probe "
                         f"(same test windows)", fontsize=11, fontweight="bold")
        d_out = TRANSFER_OUT / "figures"
        d_out.mkdir(parents=True, exist_ok=True)
        stem = stem or ("transfer_vs_own__"
                        + SHORT[src].lower().replace(" ", "_").replace("-", ""))
        pdf, png = d_out / f"{stem}.pdf", d_out / f"{stem}.png"
        fig.savefig(pdf); fig.savefig(png, dpi=dpi); plt.close(fig)
    return pdf, png

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--which", default="both",
                    choices=("main", "appendix", "both", "main_sg", "all7"),
                    help="which figure group to build; main_sg is the three-column copy of "
                         "main that adds SG Carpark and drops the Emb point; all7 is CKA-only "
                         "(every dataset in one panel)")
    ap.add_argument("--cka-font-scale", type=float, default=1.0,
                    help="scale all type in the CKA figure; the panels grow more slowly than the "
                         "type, so text gets bigger instead of heatmaps getting smaller")
    ap.add_argument("--eps-datasets", nargs="+",
                    default=["monash_electricity_hourly", "m4_hourly", "sg_carpark"],
                    help="columns of the saturation-sensitivity figure (PT-ID or PT-OOD)")
    ap.add_argument("--epsilons", type=float, nargs="+", default=list(EPS_BANDS),
                    help="tolerances to shade, loosest drawn first")
    ap.add_argument("--eps-ncol", type=int, default=None,
                    help="columns per block in the saturation-sensitivity figure "
                         "(datasets wrap into blocks of two rows; default: one block)")
    ap.add_argument("--eps-font-scale", type=float, default=1.0,
                    help="scale all type in the saturation-sensitivity figure; the panels "
                         "grow with it, so text gets bigger instead of curves getting squeezed")
    ap.add_argument("--eps-legend-scale", type=float, default=LEGEND_SCALE_REF,
                    help="scale the bottom legend only; the canvas grows to pay for it, so "
                         "the panels keep their size")
    ap.add_argument("--eps-stem", default="appendix_id_saturation_sensitivity",
                    help="output filename stem for the saturation-sensitivity figure")
    ap.add_argument("--figure", default="all", choices=("loss_erank", "cka", "cka_content", "cka_cslot", "cka_probe", "transfer", "transfer_vs_own", "ft_boom", "nha", "eps", "all"),
                    help="which figure family to build (default: all)")
    ap.add_argument("--boot-b", type=int, default=5000, help="bootstrap resamples (default 5000)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--main-source", default="m4_hourly",
                    help="probe source promoted to the main-paper delta figure")
    ap.add_argument("--dpi", type=int, default=400)
    ap.add_argument("--no-title", action="store_true",
                    help="omit the figure-level title (let the LaTeX caption carry it)")
    ap.add_argument("--cka-probe-tags", nargs="+",
                    default=["monash_electricity_hourly", "m4_hourly"],
                    help="datasets (columns) for --figure cka_probe")
    ap.add_argument("--cka-probe-keep-emb", action="store_true",
                    help="keep the input-embedding point in the cka_probe figure (both rows)")
    ap.add_argument("--cka-probe-panel-titles", action="store_true",
                    help="add per-column dataset titles to the cka_probe figure (off by default)")
    ap.add_argument("--cka-probe-title", action="store_true",
                    help="add the figure-level title to the cka_probe figure (off by default)")
    ap.add_argument("--cka-probe-gap", type=float, default=0.12,
                    help="separation before the post-LN point, in cell widths; also the width of "
                         "the blank seam in the heatmap (default 0.12)")
    a = ap.parse_args()

    if a.figure == "cka_probe":
        pdf, png = make_cka_probe_figure(
            a.cka_probe_tags, a.boot_b, a.seed, dpi=a.dpi,
            show_title=a.cka_probe_title and not a.no_title,
            drop_emb=not a.cka_probe_keep_emb,
            show_panel_titles=a.cka_probe_panel_titles, gap=a.cka_probe_gap)
        print(f"[cka_probe] {' | '.join(TITLES[t] for t in a.cka_probe_tags)}\n  {png}\n  {pdf}")
        return

    groups = ("main", "appendix") if a.which == "both" else (a.which,)
    # "all7" is a CKA-only group (it has no loss/erank counterpart), so route it separately
    cka_groups = ("all7",) if a.which == "all7" else groups

    if a.figure in ("eps", "all"):
        rows = [load_eps_dataset(t, a.boot_b, a.seed, a.epsilons) for t in a.eps_datasets]
        make_eps_figure(rows, "Saturation entrance under a stricter tolerance", a.eps_stem,
                        a.boot_b, epsilons=a.epsilons, ncol=a.eps_ncol, dpi=a.dpi,
                        show_title=not a.no_title, font_scale=a.eps_font_scale,
                        legend_scale=a.eps_legend_scale)
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

    if a.figure in ("transfer_vs_own", "all"):
        tgts = [t for t in COMBINED_TARGETS if t != a.main_source]
        rows = load_transfer_vs_own(a.main_source, tgts, a.boot_b, a.seed)
        pdf, png = make_transfer_vs_own_figure(rows, a.main_source, a.boot_b, dpi=a.dpi,
                                               show_title=not a.no_title)
        csv_p = write_transfer_vs_own_table(rows, a.main_source, a.boot_b, a.seed)
        print(f"[tvo] source {SHORT[a.main_source]}, 5% first-crossing entrance "
              f"l_s = {LABELS[rows[0]['l_s']]}   (gap = probe penalty x depth mismatch)")
        for r in rows:
            t_, p_, d_ = r["total"], r["penalty"], r["depth"]
            print(f"      {SHORT[r['target']]:<12} {r['kind']:<7} l_t={LABELS[r['l_t']]:<8} "
                  f"gap {t_['pct']:+7.1f}% [{t_['lo']:+6.1f}, {t_['hi']:+6.1f}]"
                  f"{'*' if t_['excludes_zero'] else ' '}  = probe {p_['pct']:+7.1f}% "
                  f"x depth {d_['pct']:+6.1f}%   (at L12+RMS {r['gap'][-1]:+6.1f}%)")
        print(f"    -> {png.relative_to(REPO_ROOT)}\n    -> {csv_p.relative_to(REPO_ROOT)}")
        if a.figure == "transfer_vs_own":
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
        for g in cka_groups:
            tags = CKA_TAGS.get(g) or GROUPS[g][0]
            title, stem, ncol = CKA_GROUPS[g]
            rows = [load_cka(t) for t in tags]
            pdf, png = make_cka_figure(rows, title, stem, dpi=a.dpi, show_title=not a.no_title,
                                       ncol=ncol, font_scale=a.cka_font_scale)
            print(f"[cka:{g}] {' + '.join(TITLES[t] for t in tags)}")
            for d in rows:
                M = d["M"]
                print(f"    {TITLES[d['tag']]:<12} CKA(Emb, L12+RMS)={M[0, -1]:.3f}  "
                      f"CKA(L6, L12+RMS)={M[6, -1]:.3f}  entrance={LABELS[d['l_start']]}")
            print(f"    -> {pdf.relative_to(REPO_ROOT)}\n    -> {png.relative_to(REPO_ROOT)}")
    if a.figure == "cka":
        return

    if a.figure in ("cka_content", "all"):
        # content-pooled twin of the fslot "all7" appendix: all 7 datasets, three per row
        tags = CKA_TAGS["all7"]
        title = "Representation similarity across datasets: content-pooled states"
        rows = [load_cka_content(t) for t in tags]
        pdf, png = make_cka_figure(rows, title, "appendix_id_cka", dpi=a.dpi,
                                   show_title=not a.no_title, ncol=3,
                                   font_scale=a.cka_font_scale, out_dir=CKA_CONTENT_FIG_DIR)
        print(f"[cka_content:all7] {' + '.join(TITLES[t] for t in tags)}")
        for d in rows:
            M = d["M"]
            print(f"    {TITLES[d['tag']]:<12} CKA(Emb, L12+RMS)={M[0, -1]:.3f}  "
                  f"CKA(L6, L12+RMS)={M[6, -1]:.3f}")
        print(f"    -> {pdf.relative_to(REPO_ROOT)}\n    -> {png.relative_to(REPO_ROOT)}")
    if a.figure == "cka_content":
        return

    if a.figure in ("cka_cslot", "all"):
        # content-SLOT twin of the fslot "all7" appendix: all 7 datasets, three per row
        tags = CKA_TAGS["all7"]
        title = "Representation similarity across datasets: content-slot states"
        rows = [load_cka_cslot(t) for t in tags]
        pdf, png = make_cka_figure(rows, title, "appendix_id_cka", dpi=a.dpi,
                                   show_title=not a.no_title, ncol=3,
                                   font_scale=a.cka_font_scale, out_dir=CKA_CSLOT_FIG_DIR)
        print(f"[cka_cslot:all7] {' + '.join(TITLES[t] for t in tags)}")
        for d in rows:
            M = d["M"]
            print(f"    {TITLES[d['tag']]:<12} CKA(Emb, L12+RMS)={M[0, -1]:.3f}  "
                  f"CKA(L6, L12+RMS)={M[6, -1]:.3f}")
        print(f"    -> {pdf.relative_to(REPO_ROOT)}\n    -> {png.relative_to(REPO_ROOT)}")
    if a.figure == "cka_cslot":
        return

    all_rows = []
    for g in groups:
        tags, title, stem, drop_emb = GROUPS[g]
        rows = [load_dataset(t, a.boot_b, a.seed) for t in tags]
        pdf, png = make_figure(rows, title, stem, a.boot_b, dpi=a.dpi,
                               show_title=not a.no_title, drop_emb=drop_emb)
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
