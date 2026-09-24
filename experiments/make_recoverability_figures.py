"""Results §"Forecasting Before the Final Block": figures, tables and statistics.

    python -m experiments.make_recoverability_figures

Reads ONLY the per-cell Phase-1 artifacts already on disk (``layer_metrics.csv``,
``tunnel.json``, ``bootstrap_inputs.npz``, ``cell_config.json``). No model, no GPU, no refit.

WHERE TO RUN: login node is fine -- a few MB of CSV/NPZ, one small cluster bootstrap per cell
(B=5000 x <=354 clusters), plots. Finishes in well under a minute.

WHAT IT PRODUCES (``--out-dir``, default ``three_models_paper/figures/recoverability/``)
    main_representative_q9.{pdf,png}      3 panels (one per model), 3 datasets each, test loss
    appendix_{chronos2,timesfm3,tirex}_q9.{pdf,png}   14 dataset panels per model
    tables/tunnel_q9_05.tex               14 x 3 sustained 5% entrances, raw + normalized
    tables/tunnel_sensitivity.tex         entrances at 2% / 5% / 10%
    tables/q1_vs_q9.tex                   Delta l = l(Q=1) - l(Q=9), with the Q=1 entrance
    tables/heldout_q9.tex                 test degradation at the validation-selected entrance
    recoverability_stats.json             every number quoted in the prose

PROTOCOL (the Method definition, not the older first-crossing wording):
    l_tunnel(eps) = min { l : max_{j in l..L} R_val(j) / R_val(L) <= 1 + eps }
over BLOCK DEPTHS ONLY (Emb, L1..L_final). The reference L is the final model BLOCK (Chronos-2
L12, TimesFM-3 L20, TiRex L12); the post-block normalization diagnostics (L12+LN, L12+RMS) are
dropped before anything is computed. VALIDATION loss selects; test loss is only ever read AT the
already-selected entrance.

Every entrance is recomputed here from the saved validation curve and must equal the
``is_tunnel_entrance_<tol>`` flag the run itself wrote (via ``probing.tunnel``) -- a mismatch
aborts. The cluster bootstrap must reproduce the run's own stored delta-vs-final CI -- a mismatch
aborts. So this script cannot silently disagree with the experiment.

PENDING CELLS. A cell is pending when its artifacts are absent (e.g. TimesFM-3 x M4 Q=9), or when
it is listed in ``SUPERSEDED_Q9`` and its replacement run has not completed. Pending cells keep
their table row / panel slot and are marked, never dropped or guessed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402
import numpy as np                                               # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402
from matplotlib.patches import Patch                             # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
Q9_ROOT = REPO_ROOT / "results" / "three_model_phase1_q9_expanded"
Q1_ROOT = REPO_ROOT / "results" / "three_model_phase1_q1"
OUT_DIR = REPO_ROOT / "three_models_paper" / "figures" / "recoverability"

MODELS = ("chronos2", "timesfm3", "tirex")
MODEL_NAME = {"chronos2": "Chronos-2", "timesfm3": "TimesFM-3", "tirex": "TiRex"}
FINAL_BLOCK = {"chronos2": "L12", "timesfm3": "L20", "tirex": "L12"}
TOLS = (0.02, 0.05, 0.10)
HEADLINE_TOL = 0.05
FLAG = {0.02: "is_tunnel_entrance_0.02", 0.05: "is_tunnel_entrance_0.05",
        0.10: "is_tunnel_entrance_0.1"}

# Q=9 cells whose committed result is being REPLACED by a rerun. The LOOP Seattle cells clipped
# the weight-decay grid at lr=1e-2 (TimesFM-3 15/21 depths, TiRex 10/13 at wd=90), so they are
# being rerun at lr=1e-3 with a wider grid. Until that run's cell exists, the dataset is PENDING
# for Q=9 -- the clipped numbers are reported only in the ``legacy`` block of the stats JSON.
SUPERSEDED_Q9 = {
    "LOOP_SEATTLE_5T": (REPO_ROOT / "results" / "three_model_phase1_q9_loop_lowlr",
                        "Q=9 rerun at lr=1e-3 with widened weight-decay grid not yet complete"),
}
# The Q=1 LOOP Seattle cells were fit with the same clipped lr=1e-2 setting; they are held back
# the same way until the matched Q=1 rerun (same lr=1e-3 + widened grid as the Q=9 rerun) exists.
# Both the Q=1 figures and the Q1-vs-Q9 comparison resolve through this table.
SUPERSEDED_Q1 = {
    "LOOP_SEATTLE_5T": (REPO_ROOT / "results" / "three_model_phase1_q1_loop_lowlr",
                        "matched Q=1 rerun at lr=1e-3 not yet run"),
}

# ---- style: sized for a 5.5in ICLR \textwidth, so point sizes print 1:1 ----------------------
TEXTWIDTH_IN = 5.5
LOSS_C, LOSS_BAND = "#1F5FA8", "#AFC9E8"                 # appendix: one test curve per panel
VAL_C = "#D9730D"                                        # appendix: the selecting validation curve
TUNNEL_LINE, TUNNEL_FILL = "#2E7D32", "#E4F0E4"          # appendix: validation-selected tunnel
TRIO_C = ("#0072B2", "#D55E00", "#009E73")               # main: Okabe-Ito, colour-blind safe
# Main-figure datasets: complete for all three models (TimesFM-3 x M4 is still pending), tight
# bootstrap bands, separated loss levels, early/mid/late entrances. BOOM was tried and dropped:
# its heavy-tailed series give a CI band that swamps the other two curves.
MAIN_DATASETS = ("monash_electricity_hourly", "uber_tlc_hourly", "monash_london_smart_meters")
PENDING_C = "#8A8A8A"
RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 7.5, "xtick.labelsize": 6.5,
      "ytick.labelsize": 6.5, "legend.fontsize": 7.5, "axes.linewidth": 0.7,
      "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.minor.width": 0.5,
      "lines.linewidth": 1.3, "pdf.fonttype": 42, "ps.fonttype": 42}


# --------------------------------------------------------------------------- #
# the two statistics shared with the experiment
# --------------------------------------------------------------------------- #
try:                                    # the canonical implementations (Narval venv)
    from probing.stats import cluster_bootstrap_counts
    from probing.tunnel import sustained_tunnel_start
except Exception:                       # e.g. a machine whose features_cache symlink is dangling
    def sustained_tunnel_start(val_losses, tol):
        """Verbatim rule of ``probing.tunnel.sustained_tunnel_start`` (same inclusive
        ``v[j] <= (1+tol) v[last]`` comparison). Checked against the run's own flags below."""
        v = np.asarray(val_losses, np.float64)
        out = np.flatnonzero(v > (1.0 + float(tol)) * v[-1])
        return int(out[-1] + 1) if out.size else 0

    def cluster_bootstrap_counts(n_series, B, seed):
        """Verbatim ``probing.stats.cluster_bootstrap_counts``. Checked against stored CIs below."""
        rng = np.random.default_rng(seed)
        return rng.multinomial(n_series, np.full(n_series, 1.0 / n_series),
                               size=B).astype(np.float64)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def _roster(root: Path):
    rows = list(csv.DictReader(open(root / "combined" / "model_dataset_manifest.csv")))
    order, names = [], {}
    for r in rows:
        if r["dataset"] not in names:
            order.append(r["dataset"])
            names[r["dataset"]] = r["display_name"]
    if len(order) != 14:
        raise ValueError(f"expected the 14-dataset roster, got {len(order)}: {order}")
    return order, names


def _f(x):
    return float(x) if x not in ("", None) else float("nan")


def load_cell(root: Path, model: str, dataset: str):
    """Depth-axis curves of one COMPLETE cell, or None. Head-input diagnostics are dropped."""
    d = root / model / dataset
    if not (d / "COMPLETE").exists():
        return None
    rows = list(csv.DictReader(open(d / "layer_metrics.csv")))
    depth = [r for r in rows if r["include_in_main_depth_axis"] == "True"]
    dropped = sorted(r["layer"] for r in rows if r["include_in_main_depth_axis"] != "True")
    labels = [r["layer"] for r in depth]
    if [int(r["block_index"]) for r in depth] != list(range(len(depth))):
        raise ValueError(f"{model}/{dataset}: depth axis is not Emb, L1..L_N in order")
    if labels[-1] != FINAL_BLOCK[model]:
        raise ValueError(f"{model}/{dataset}: last depth is {labels[-1]}, "
                         f"expected the final block {FINAL_BLOCK[model]}")
    if any(r["point_type"] != "block_depth" for r in depth):
        raise ValueError(f"{model}/{dataset}: a non-block point leaked onto the depth axis")
    widths = {r["n_probe_params"] for r in depth}
    if len(widths) != 1:                 # probe size fixed across depth <=> d fixed across depth
        raise ValueError(f"{model}/{dataset}: probe size varies across depth: {widths}")
    tj = json.load(open(d / "tunnel.json"))
    cfg = json.load(open(d / "cell_config.json"))
    if tj["reference_point"] != FINAL_BLOCK[model] or not tj["test_never_used_for_selection"]:
        raise ValueError(f"{model}/{dataset}: tunnel.json reference/selection contract broken")
    if tj["definition"] != "sustained_suffix_v1" or tj["split_used"] != "validation":
        raise ValueError(f"{model}/{dataset}: unexpected tunnel definition {tj['definition']}")
    return {
        "model": model, "dataset": dataset, "labels": labels, "dropped": dropped,
        "L": len(labels) - 1, "dir": d, "cfg": cfg,
        "val": np.array([_f(r["val_loss"]) for r in depth]),
        "test": np.array([_f(r["test_loss"]) for r in depth]),
        # float64 mean of the per-window losses; ``test_loss`` is the line's own scalar
        # reduction (float32 for TimesFM-3, ~1e-7 relative apart)
        "test_point": np.array([_f(r["test_loss_point"]) for r in depth]),
        "test_ci": np.array([[_f(r["test_loss_ci_lo"]), _f(r["test_loss_ci_hi"])]
                             for r in depth]),
        "test_mase": np.array([_f(r["test_mase"]) for r in depth]),
        "delta_ci": np.array([[_f(r["test_loss_delta_ci_lo"]), _f(r["test_loss_delta_ci_hi"])]
                              for r in depth]),
        "flags": {t: [r[FLAG[t]] == "True" for r in depth] for t in TOLS},
    }


def resolve(model, dataset, root=Q9_ROOT, superseded=SUPERSEDED_Q9):
    """(cell or None, source root or None, pending reason or None)."""
    if dataset in superseded:
        sroot, why = superseded[dataset]
        c = load_cell(sroot, model, dataset)
        return (c, sroot, None) if c else (None, None, why)
    c = load_cell(root, model, dataset)
    return (c, root, None) if c else (None, None, "cell not yet complete")


def resolve_q9(model, dataset):
    return resolve(model, dataset, Q9_ROOT, SUPERSEDED_Q9)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def entrances(cell):
    """{tol: depth index} from VALIDATION only, cross-checked against the run's own flags."""
    out = {}
    for t in TOLS:
        l = sustained_tunnel_start(cell["val"], t)
        flagged = [i for i, f in enumerate(cell["flags"][t]) if f]
        if flagged != [l]:
            raise RuntimeError(f"{cell['model']}/{cell['dataset']} tol={t}: recomputed entrance "
                               f"{cell['labels'][l]} != run's flag {flagged}")
        out[t] = l
    if not out[0.10] <= out[0.05] <= out[0.02]:
        raise RuntimeError(f"{cell['model']}/{cell['dataset']}: entrance not monotone in tol")
    return out


def heldout(cell, l):
    """Test behaviour AT the validation-selected entrance l, with a paired cluster-bootstrap CI
    of the relative degradation R_test(l)/R_test(L) - 1 (same B, seed, clusters as the run)."""
    z = np.load(cell["dir"] / "bootstrap_inputs.npz", allow_pickle=True)
    lab = [str(s) for s in z["labels"]]
    rows = [lab.index(s) for s in cell["labels"]]
    W = np.asarray(z["test_loss_window"], np.float64)[rows]
    if not (np.allclose(W.mean(axis=1), cell["test_point"], rtol=1e-12, atol=0)
            and np.allclose(cell["test_point"], cell["test"], rtol=1e-6, atol=0)):
        raise RuntimeError(f"{cell['model']}/{cell['dataset']}: per-window test loss does not "
                           "reproduce layer_metrics")
    uniq, inv = np.unique(z["cluster_ids_test"], return_inverse=True)
    S = len(uniq)
    per_sum = np.zeros((S, W.shape[0]))
    np.add.at(per_sum, inv, W.T)
    cnt = np.bincount(inv, minlength=S).astype(np.float64)
    M = cluster_bootstrap_counts(S, int(cell["cfg"]["boot_b"]), int(cell["cfg"]["seed"]))
    boot = (M @ per_sum) / (M @ cnt)[:, None]
    L = cell["L"]
    # reproduce the run's stored paired delta CI at l (delta = final - l) -> same resampling
    d_lo, d_hi = np.percentile(boot[:, L] - boot[:, l], [2.5, 97.5])
    if not np.allclose([d_lo, d_hi], cell["delta_ci"][l], rtol=1e-7, atol=1e-10):
        raise RuntimeError(f"{cell['model']}/{cell['dataset']}: bootstrap does not reproduce the "
                           f"stored delta CI ({d_lo},{d_hi}) vs {cell['delta_ci'][l]}")
    ci = np.percentile(boot, [2.5, 97.5], axis=0).T
    if not np.allclose(ci, cell["test_ci"], rtol=1e-7, atol=1e-10):
        raise RuntimeError(f"{cell['model']}/{cell['dataset']}: bootstrap does not reproduce the "
                           "stored per-depth test-loss CI")
    rel_lo, rel_hi = np.percentile(boot[:, l] / boot[:, L] - 1.0, [2.5, 97.5])
    t = cell["test_point"]
    return {
        "test_rel_degradation": float(t[l] / t[L] - 1.0),
        "test_rel_ci": [float(rel_lo), float(rel_hi)],
        "test_within_5pct_at_entrance": bool(t[l] <= 1.05 * t[L]),
        "test_sustained_within_5pct": bool(np.all(t[l:] <= 1.05 * t[L])),
        "test_mase_rel_change": float(cell["test_mase"][l] / cell["test_mase"][L] - 1.0),
        "test_at_entrance": float(t[l]), "test_final": float(t[L]),
        "mase_at_entrance": float(cell["test_mase"][l]),
        "mase_final": float(cell["test_mase"][L]),
        "n_test_clusters": int(S), "B": int(cell["cfg"]["boot_b"]),
    }


def _q(a, p):
    return float(np.percentile(np.asarray(a, np.float64), p)) if len(a) else None


def summarize_entrances(recs):
    """recs: list of dicts with model, idx, L. Before-final counts and normalized-depth spread."""
    out = {}
    for m in MODELS + ("all",):
        rs = [r for r in recs if m == "all" or r["model"] == m]
        nd = [r["idx"] / r["L"] for r in rs]
        out[m] = {"n": len(rs), "n_before_final": sum(r["idx"] < r["L"] for r in rs),
                  "median_norm": _q(nd, 50), "q25_norm": _q(nd, 25), "q75_norm": _q(nd, 75),
                  "min_norm": min(nd) if nd else None, "max_norm": max(nd) if nd else None}
    return out


def agreement(pairs):
    out = {}
    for m in MODELS + ("all",):
        ps = [p for p in pairs if m == "all" or p["model"] == m]
        d = np.array([p["delta"] for p in ps], np.float64)
        out[m] = {"n": len(ps), "n_exact": int((d == 0).sum()),
                  "n_within_1": int((np.abs(d) <= 1).sum()),
                  "n_within_2": int((np.abs(d) <= 2).sum()),
                  "median_abs_delta": float(np.median(np.abs(d))) if len(d) else None,
                  "mean_signed_delta": float(d.mean()) if len(d) else None}
    return out


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def _disp(label):
    """The input embedding is depth 0 of the block axis; the paper calls it L0."""
    return "L0" if label == "Emb" else label


def _entrance_tag(ax, cell, l, col, row=0, fs=6.5, curve="test"):
    """Mark the tunnel entrance: a hollow ring ON the curve at block l, and a filled tag
    (white text on the curve's colour) where the entrance line meets the top of the panel.
    ``row`` stacks tags downward so adjacent entrances never overlap."""
    ax.plot([l], [cell[curve][l]], "o", ms=5.0, mfc="white", mec=col, mew=1.3, zorder=6)
    L = cell["L"]
    ha = "left" if l <= 0.1 * L else "right" if l >= 0.9 * L else "center"
    dx = {"left": -3, "right": 3, "center": 0}[ha]
    ax.annotate(_disp(cell["labels"][l]), xy=(l, 1.0), xycoords=("data", "axes fraction"),
                xytext=(dx, -2 - 10 * row), textcoords="offset points", fontsize=fs,
                fontweight="bold", color="white", ha=ha, va="top", zorder=7,
                bbox=dict(boxstyle="round,pad=0.18", fc=col, ec="none"))


def _xticks(ax, labels, show=True, compact=False):
    """Every 2nd block (12-block models) / 4th (TimesFM-3); ``compact`` panels use 3rd / 5th."""
    L = len(labels) - 1
    step = (5 if compact else 4) if L > 12 else (3 if compact else 2)
    major = list(range(0, L + 1, step))
    ax.set_xticks(major)
    ax.set_xticklabels([_disp(labels[i]) for i in major] if show else [])
    ax.set_xticks(range(L + 1), minor=True)
    ax.tick_params(which="major", length=2.5, pad=1.5)
    ax.tick_params(which="minor", length=1.5)
    ax.set_xlim(-0.5, L + 0.5)


def _clean(ax):
    ax.grid(axis="y", alpha=0.18, lw=0.5)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _ylim(ax, lo, hi, top=0.12):
    span = hi - lo
    ax.set_ylim(lo - 0.05 * span, hi + top * span)


def draw_single(ax, cell, l, show_xticklabels=True):
    """Appendix panel: TEST loss + 95% cluster-bootstrap band, and the VALIDATION curve that
    selects the entrance, with its 1.05 x final-block threshold. The tunnel (shaded) starts at
    the validation-selected entrance l."""
    x, L = np.arange(cell["L"] + 1), cell["L"]
    t, ci, v = cell["test"], cell["test_ci"], cell["val"]
    thr = (1.0 + HEADLINE_TOL) * v[L]
    # above the test band so neither is hidden by it; below the entrance line (zorder 5)
    ax.axhline(thr, color=VAL_C, lw=0.9, ls=(0, (1, 1.5)), zorder=4.2)
    ax.plot(x, v, ls=(0, (3.5, 1.6)), marker="s", ms=1.7, lw=1.0, color=VAL_C, zorder=4.4)
    ax.axvspan(l, L + 0.5, color=TUNNEL_FILL, lw=0, zorder=0)
    ax.axvline(l, color=TUNNEL_LINE, lw=1.1, zorder=5)   # above band + curve, below ring + tag
    ax.fill_between(x, ci[:, 0], ci[:, 1], color=LOSS_BAND, lw=0, zorder=2)
    ax.plot(x, t, "-o", ms=1.9, lw=1.2, color=LOSS_C, zorder=3)
    _ylim(ax, min(ci[:, 0].min(), v.min()), max(ci[:, 1].max(), v.max(), thr), top=0.22)
    _entrance_tag(ax, cell, l, TUNNEL_LINE, curve="val")    # the curve that selects it
    _xticks(ax, cell["labels"], show_xticklabels, compact=True)
    _clean(ax)


def draw_trio(ax, cells):
    """Main panel: one model, three datasets. Each dataset gets its own colour for the test curve,
    its CI band, its entrance line and its tunnel shading (from its OWN validation-selected
    entrance to the final block)."""
    lo = min(c["test_ci"][:, 0].min() for c, _ in cells)
    hi = max(c["test_ci"][:, 1].max() for c, _ in cells)
    for k, ((cell, l), col) in enumerate(zip(cells, TRIO_C)):
        x, L, t, ci = np.arange(cell["L"] + 1), cell["L"], cell["test"], cell["test_ci"]
        pre, tun = slice(0, l + 1), slice(l, L + 1)
        # outside the tunnel: faint band, thin translucent line
        ax.fill_between(x[pre], ci[pre, 0], ci[pre, 1], color=col, alpha=0.12, lw=0, zorder=2)
        ax.plot(x[pre], t[pre], "-o", ms=2.0, lw=1.0, color=col, alpha=0.55, zorder=3)
        # inside this curve's own tunnel: saturated band, solid line
        ax.fill_between(x[tun], ci[tun, 0], ci[tun, 1], color=col, alpha=0.38, lw=0, zorder=2)
        ax.plot(x[tun], t[tun], "-o", ms=2.6, lw=1.6, color=col, zorder=4)
        ax.axvline(l, color=col, lw=1.1, ls=(0, (3, 2)), zorder=5)
        _entrance_tag(ax, cell, l, col, row=k)
    _ylim(ax, lo, hi, top=0.36)                  # headroom for the stacked entrance labels
    _xticks(ax, cells[0][0]["labels"])
    _clean(ax)


def draw_pending(ax, reason_short, labels, show_xticklabels=True):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#C8C8C8")
    ax.set_yticks([])
    _xticks(ax, labels, show_xticklabels, compact=True)
    ax.tick_params(colors="#A0A0A0")
    ax.text(0.5, 0.58, "PENDING", transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=PENDING_C)
    ax.text(0.5, 0.36, reason_short, transform=ax.transAxes, ha="center", va="center",
            fontsize=6.5, color=PENDING_C)


def _save(fig, out_dir, stem, dpi=300):
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf, png = out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    return [str(pdf.relative_to(REPO_ROOT)), str(png.relative_to(REPO_ROOT))]


def main_figure(q9, datasets, names, out_dir):
    for m in MODELS:
        for ds in datasets:
            if q9[m][ds][0] is None:
                raise ValueError(f"main figure: {MODEL_NAME[m]} x {ds} is pending -- choose "
                                 "datasets complete for all three models (--main-datasets)")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(1, 3, figsize=(TEXTWIDTH_IN, 2.2), layout="constrained")
        fig.get_layout_engine().set(w_pad=0.03, h_pad=0.02, wspace=0.07)
        for ax, m in zip(axes, MODELS):
            draw_trio(ax, [(q9[m][ds][0], q9[m][ds][1][HEADLINE_TOL]) for ds in datasets])
            ax.set_title(MODEL_NAME[m], fontweight="bold", pad=2)
        axes[0].set_ylabel("Test quantile loss (Q=9)")
        axes[1].set_xlabel("Block depth", labelpad=4)
        handles = [Line2D([], [], color=c, marker="o", ms=2.8, lw=1.2, label=names[ds])
                   for c, ds in zip(TRIO_C, datasets)]
        handles += [Patch(facecolor="#9A9A9A", alpha=0.25, lw=0, label="95% bootstrap CI"),
                    Patch(facecolor="#555555", alpha=0.55, lw=0, label="Inside 5% tunnel"),
                    Line2D([], [], color="#555555", lw=1.0, ls=(0, (3, 2)), marker="o", ms=4.5,
                           mfc="white", mec="#555555", mew=1.2, label="Entrance (validation)")]
        fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False,
                   handlelength=2.0, columnspacing=1.3, borderpad=0.1)
        return _save(fig, out_dir, "main_representative_q9")


def appendix_figure(model, cells, order, names, out_dir, q="9"):
    """4 x 4 grid in roster order, row-major; the last two slots hold one shared legend.
    Four columns keep each panel near-square without a full-page figure. A pending cell keeps
    its slot (no reflow) and says so."""
    ref_labels = next(c["labels"] for c, _, _ in cells.values() if c is not None)
    ncol, nrow = 4, 4
    with plt.rc_context({**RC, "axes.titlesize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, axes = plt.subplots(nrow, ncol, figsize=(TEXTWIDTH_IN, 5.6), layout="constrained")
        # titles are kept OUT of the layout (below) so a long name cannot shrink its column;
        # hspace reserves their row gap instead
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.13)
        flat = axes.ravel()
        for i, ds in enumerate(order):
            ax = flat[i]
            cell, ent, why = cells[ds]
            show = i + ncol >= len(order)        # x labels only where no panel sits below
            if cell is None:
                draw_pending(ax, "rerun in progress" if "not yet complete" in why and "rerun" in why
                             else "matched rerun needed" if "rerun" in why
                             else "not yet complete", ref_labels, show)
            else:
                draw_single(ax, cell, ent[HEADLINE_TOL], show)
            ax.set_title(names[ds], fontweight="bold", pad=2).set_in_layout(False)
            if show:
                ax.set_xlabel("Block depth", fontsize=7, labelpad=3)
        for ax in flat[len(order):]:
            ax.remove()
        lax = fig.add_subplot(axes[0, 0].get_gridspec()[nrow - 1, len(order) % ncol:])
        lax.axis("off")
        fig.supylabel(f"Q={q} quantile loss", fontsize=7.5)
        lax.legend(handles=[
            Line2D([], [], color=LOSS_C, marker="o", ms=2.6, lw=1.2, label=f"Test loss (Q={q})"),
            Patch(facecolor=LOSS_BAND, lw=0, label="95% bootstrap CI (test)"),
            Line2D([], [], color=VAL_C, marker="s", ms=2.3, lw=1.0, ls=(0, (3.5, 1.6)),
                   label=f"Validation loss (Q={q})"),
            Line2D([], [], color=VAL_C, lw=0.9, ls=(0, (1, 1.5)),
                   label="1.05 x final-block val."),
            Line2D([], [], color=TUNNEL_LINE, lw=1.2, marker="o", ms=4.5, mfc="white",
                   mec=TUNNEL_LINE, mew=1.2, label="Tunnel entrance (val., 5%)"),
            Patch(facecolor=TUNNEL_FILL, lw=0, label="Sustained 5% tunnel")],
            loc="center", ncol=1, frameon=False, handlelength=1.8, fontsize=7,
            borderaxespad=0.0)
        return _save(fig, out_dir, f"appendix_{model}_q{q}")


# --------------------------------------------------------------------------- #
# LaTeX tables
# --------------------------------------------------------------------------- #
PENDING_TEX = r"\textit{pending}"


def _lab(labels, i):
    return "L0" if i == 0 else labels[i]


def _entry(cell, l):
    return f"{_lab(cell['labels'], l)} ({l / cell['L']:.2f})"


def tunnel_table(order, names, q9, summ):
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Dataset & Chronos-2 ($L{=}12$) & TimesFM-3 ($L{=}20$) & TiRex ($L{=}12$) \\",
             r"\midrule"]
    for ds in order:
        cells = []
        for m in MODELS:
            c, ent, _ = q9[m][ds]
            cells.append(_entry(c, ent[HEADLINE_TOL]) if c else PENDING_TEX)
        lines.append(f"{names[ds]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"Median $\ell/L$ & " + " & ".join(
        f"{summ[m]['median_norm']:.2f}" for m in MODELS) + r" \\")
    lines.append(r"Entrance $<L$ & " + " & ".join(
        f"{summ[m]['n_before_final']}/{summ[m]['n']}" for m in MODELS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def entrance_at(cell, ent, tol):
    """Sustained entrance at any tolerance, from the VALIDATION curve. Saved tolerances are
    checked against the run's flags in ``entrances``; any other one (e.g. 8%) is computed with
    the same rule and must sit between its saved neighbours (monotonicity)."""
    if tol in ent:
        return ent[tol]
    l = sustained_tunnel_start(cell["val"], tol)
    lo = [ent[t] for t in ent if t > tol]
    hi = [ent[t] for t in ent if t < tol]
    if (lo and l < max(lo)) or (hi and l > min(hi)):
        raise RuntimeError(f"{cell['model']}/{cell['dataset']}: tol={tol} entrance {l} breaks "
                           f"monotonicity against {ent}")
    return l


def tunnel_table_tol(order, names, q9, tol):
    """The 14 x 3 entrance table at one tolerance, with its own summary rows."""
    recs = []
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Dataset & Chronos-2 ($L{=}12$) & TimesFM-3 ($L{=}20$) & TiRex ($L{=}12$) \\",
             r"\midrule"]
    for ds in order:
        cells = []
        for m in MODELS:
            c, ent, _ = q9[m][ds]
            if c is None:
                cells.append(PENDING_TEX)
                continue
            l = entrance_at(c, ent, tol)
            recs.append({"model": m, "dataset": ds, "idx": l, "L": c["L"]})
            cells.append(_entry(c, l))
        lines.append(f"{names[ds]} & " + " & ".join(cells) + r" \\")
    summ = summarize_entrances(recs)
    lines.append(r"\midrule")
    lines.append(r"Median $\ell/L$ & " + " & ".join(
        f"{summ[m]['median_norm']:.2f}" for m in MODELS) + r" \\")
    lines.append(r"Entrance $<L$ & " + " & ".join(
        f"{summ[m]['n_before_final']}/{summ[m]['n']}" for m in MODELS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", summ


def sensitivity_table(order, names, q9, sens):
    head = " & ".join(["2\\%", "5\\%", "10\\%"] * 3)
    lines = [r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Chronos-2} & \multicolumn{3}{c}{TimesFM-3} & "
             r"\multicolumn{3}{c}{TiRex} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             f"Dataset & {head} \\\\", r"\midrule"]
    for ds in order:
        cells = []
        for m in MODELS:
            c, ent, _ = q9[m][ds]
            cells += ([_lab(c["labels"], ent[t]) for t in TOLS] if c
                      else [r"\multicolumn{3}{c}{\textit{pending}}"])
        lines.append(f"{names[ds]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"Median $\ell/L$ & " + " & ".join(
        f"{sens[t][m]['median_norm']:.2f}" for m in MODELS for t in TOLS) + r" \\")
    lines.append(r"Entrance $<L$ & " + " & ".join(
        f"{sens[t][m]['n_before_final']}/{sens[t][m]['n']}" for m in MODELS for t in TOLS)
        + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def sensitivity_table_eps(order, names, q9, tols=(0.02, 0.05, 0.08)):
    """Entrance at each tolerance in ``tols`` (wide: one column group per model), with the
    number of pairs entering before the final block and the median normalized depth."""
    pct = [f"{round(100 * t)}\\%" for t in tols]
    k = len(tols)
    lines = [r"\begin{tabular}{l " + " ".join(["c" * k] * 3) + "}", r"\toprule",
             " & " + " & ".join(f"\\multicolumn{{{k}}}{{c}}{{{MODEL_NAME[m]}}}"
                                for m in MODELS) + r" \\",
             "".join(f"\\cmidrule(lr){{{2 + i * k}-{1 + (i + 1) * k}}}" for i in range(3)),
             r"Dataset & $\epsilon$ = " + " & ".join(pct * 3) + r" \\", r"\midrule"]
    recs = {t: [] for t in tols}
    for ds in order:
        cells = []
        for m in MODELS:
            c, ent, _ = q9[m][ds]
            if c is None:
                cells.append(f"\\multicolumn{{{k}}}{{c}}{{\\textit{{pending}}}}")
                continue
            for t in tols:
                l = entrance_at(c, ent, t)
                recs[t].append({"model": m, "dataset": ds, "idx": l, "L": c["L"]})
                cells.append(_lab(c["labels"], l))
        lines.append(f"{names[ds]} & " + " & ".join(cells) + r" \\")
    summ = {t: summarize_entrances(recs[t]) for t in tols}
    lines.append(r"\midrule")
    lines.append(r"Before final & " + " & ".join(
        f"{summ[t][m]['n_before_final']}/{summ[t][m]['n']}" for m in MODELS for t in tols)
        + r" \\")
    lines.append(r"Median $\ell/L$ & " + " & ".join(
        f"{summ[t][m]['median_norm']:.2f}" for m in MODELS for t in tols) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n", summ


def heldout_table_long(order, names, q9, ho, hsum):
    """One row per model x dataset, grouped by model: validation-selected entrance, Q=9 test
    loss at the entrance and at the final block, relative increase with its paired cluster-
    bootstrap 95% CI, and median-forecast MASE at the entrance and at the final block."""
    def pct(v):
        return f"{100 * v:+.1f}".replace("+", "$+$").replace("-", "$-$")
    lines = [r"\begin{tabular}{l c cc c cc}", r"\toprule",
             r"Dataset & Entr. & \multicolumn{2}{c}{Q=9 test loss} & Rel.\ increase (\%) & "
             r"\multicolumn{2}{c}{MASE (median)} \\",
             r"\cmidrule(lr){3-4}\cmidrule(lr){6-7}",
             r" & & entrance & final & [95\% CI] & entrance & final \\"]
    for m in MODELS:
        lines += [r"\midrule", f"\\multicolumn{{7}}{{l}}{{\\textbf{{{MODEL_NAME[m]}}}"
                  f" (final block {FINAL_BLOCK[m]})}} \\\\"]
        for ds in order:
            c, ent, _ = q9[m][ds]
            if c is None:
                lines.append(f"{names[ds]} & \\multicolumn{{6}}{{c}}{{\\textit{{pending}}}} \\\\")
                continue
            h = ho[m][ds]
            lo, hi = h["test_rel_ci"]
            lines.append(
                f"{names[ds]} & {_lab(c['labels'], ent[HEADLINE_TOL])} & "
                f"{h['test_at_entrance']:.3f} & {h['test_final']:.3f} & "
                f"{pct(h['test_rel_degradation'])} [{pct(lo)}, {pct(hi)}] & "
                f"{h['mase_at_entrance']:.2f} & {h['mase_final']:.2f} \\\\")
    lines.append(r"\midrule")
    a = hsum["all"]
    lines.append(f"\\multicolumn{{7}}{{l}}{{All {a['n']} completed pairs: median relative "
                 f"increase {pct(a['median_rel_degradation'])}\\%, "
                 f"within 5\\% in {a['n_within_5pct_at_entrance']}/{a['n']}}} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def q1_table(order, names, cmp_cells, agr):
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Dataset & Chronos-2 & TimesFM-3 & TiRex \\", r"\midrule"]
    for ds in order:
        cells = []
        for m in MODELS:
            p = cmp_cells[m][ds]
            if p is None:
                cells.append(PENDING_TEX)
            else:
                d = p["delta"]
                s = "0" if d == 0 else f"${d:+d}$"
                cells.append(f"{s} ({p['q1_label']})")
        lines.append(f"{names[ds]} & " + " & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"Exact match & " + " & ".join(
        f"{agr[m]['n_exact']}/{agr[m]['n']}" for m in MODELS) + r" \\")
    lines.append(r"$|\Delta\ell|\le 1$ & " + " & ".join(
        f"{agr[m]['n_within_1']}/{agr[m]['n']}" for m in MODELS) + r" \\")
    lines.append(r"Median $|\Delta\ell|$ & " + " & ".join(
        f"{agr[m]['median_abs_delta']:g}" for m in MODELS) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def heldout_table(order, names, q9, ho):
    def pct(v):
        return f"{100 * v:+.1f}".replace("+", "$+$").replace("-", "$-$")
    lines = [r"\begin{tabular}{lccc}", r"\toprule",
             r"Dataset & Chronos-2 & TimesFM-3 & TiRex \\", r"\midrule"]
    for ds in order:
        cells = []
        for m in MODELS:
            c, ent, _ = q9[m][ds]
            if c is None:
                cells.append(PENDING_TEX)
                continue
            h = ho[m][ds]
            lo, hi = h["test_rel_ci"]
            cells.append(f"{pct(h['test_rel_degradation'])} [{pct(lo)}, {pct(hi)}]")
        lines.append(f"{names[ds]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--main-datasets", nargs=3, default=list(MAIN_DATASETS), metavar="TAG",
                    help="the three datasets drawn in every main-figure panel")
    args = ap.parse_args(argv)
    order, names = _roster(Q9_ROOT)
    stats = {"protocol": {"definition": "sustained_suffix_v1", "selection_split": "validation",
                          "headline_tol": HEADLINE_TOL, "tols": list(TOLS),
                          "final_block": FINAL_BLOCK,
                          "loss": "mean pinball over (windows, Q=9 quantiles, H=64)"},
             "sources": {"q9": str(Q9_ROOT.relative_to(REPO_ROOT)),
                         "q1": str(Q1_ROOT.relative_to(REPO_ROOT)),
                         "superseded_q9": {k: [str(v[0].relative_to(REPO_ROOT)), v[1]]
                                           for k, v in SUPERSEDED_Q9.items()},
                         "superseded_q1": {k: [str(v[0].relative_to(REPO_ROOT)), v[1]]
                                           for k, v in SUPERSEDED_Q1.items()}}}

    # ---- Q=9 headline cells -------------------------------------------------------------------
    q9, pending, ho, recs = {m: {} for m in MODELS}, [], {m: {} for m in MODELS}, []
    sens_recs = {t: [] for t in TOLS}
    for m in MODELS:
        for ds in order:
            c, root, why = resolve_q9(m, ds)
            if c is None:
                q9[m][ds] = (None, None, why)
                pending.append({"model": m, "dataset": ds, "reason": why})
                continue
            ent = entrances(c)
            q9[m][ds] = (c, ent, None)
            ho[m][ds] = heldout(c, ent[HEADLINE_TOL])
            recs.append({"model": m, "dataset": ds, "idx": ent[HEADLINE_TOL], "L": c["L"]})
            for t in TOLS:
                sens_recs[t].append({"model": m, "dataset": ds, "idx": ent[t], "L": c["L"]})
    summ = summarize_entrances(recs)
    sens = {t: summarize_entrances(sens_recs[t]) for t in TOLS}
    stats["pending_q9"] = pending
    stats["q9_entrances_05"] = {
        m: {ds: (None if q9[m][ds][0] is None else
                 {"layer": _lab(q9[m][ds][0]["labels"], q9[m][ds][1][HEADLINE_TOL]),
                  "index": q9[m][ds][1][HEADLINE_TOL], "L": q9[m][ds][0]["L"],
                  "normalized": q9[m][ds][1][HEADLINE_TOL] / q9[m][ds][0]["L"],
                  "dropped_diagnostics": q9[m][ds][0]["dropped"]})
            for ds in order} for m in MODELS}
    stats["q9_summary_05"] = summ
    stats["q9_sensitivity"] = {str(t): sens[t] for t in TOLS}
    n_moved = {"stricter_later": 0, "looser_earlier": 0, "n": 0}
    for m in MODELS:
        for ds in order:
            c, ent, _ = q9[m][ds]
            if c is None:
                continue
            n_moved["n"] += 1
            n_moved["stricter_later"] += ent[0.02] > ent[0.05]
            n_moved["looser_earlier"] += ent[0.10] < ent[0.05]
    stats["q9_sensitivity_movement"] = n_moved

    # ---- held-out behaviour at the validation-selected entrance ---------------------------------
    stats["heldout_q9_05"] = {m: ho[m] for m in MODELS}
    hsum = {}
    for m in MODELS + ("all",):
        hs = [ho[mm][ds] for mm in MODELS for ds in ho[mm] if m in ("all", mm)]
        deg = [h["test_rel_degradation"] for h in hs]
        hsum[m] = {"n": len(hs), "median_rel_degradation": _q(deg, 50),
                   "q25": _q(deg, 25), "q75": _q(deg, 75),
                   "mean_rel_degradation": float(np.mean(deg)) if deg else None,
                   "max_rel_degradation": max(deg) if deg else None,
                   "n_within_5pct_at_entrance": sum(h["test_within_5pct_at_entrance"] for h in hs),
                   "n_sustained_within_5pct": sum(h["test_sustained_within_5pct"] for h in hs),
                   "n_ci_upper_le_5pct": sum(h["test_rel_ci"][1] <= 0.05 for h in hs),
                   "median_mase_rel_change": _q([h["test_mase_rel_change"] for h in hs], 50)}
    stats["heldout_q9_05_summary"] = hsum

    # ---- Q=1 vs Q=9 -----------------------------------------------------------------------------
    cmp_cells, pairs, notcmp = {m: {} for m in MODELS}, [], []
    for m in MODELS:
        for ds in order:
            c9 = q9[m][ds][0]
            # same resolution as the Q=1 figures: a superseded Q=1 cell is read from its matched
            # rerun, so the config check below compares like with like (lr=1e-3 vs lr=1e-3)
            c1 = resolve(m, ds, Q1_ROOT, SUPERSEDED_Q1)[0]
            if c9 is None or c1 is None:
                cmp_cells[m][ds] = None
                notcmp.append({"model": m, "dataset": ds, "reason":
                               "Q=9 pending" if c9 is None else "Q=1 cell not complete"})
                continue
            keys = ("probe_lr", "probe_epochs", "wd_grid", "window_digest", "seed")
            diff = [k for k in keys if c9["cfg"].get(k) != c1["cfg"].get(k)]
            if diff:
                cmp_cells[m][ds] = None
                notcmp.append({"model": m, "dataset": ds,
                               "reason": f"Q=1/Q=9 fit configs differ in {diff}"})
                continue
            l9 = q9[m][ds][1][HEADLINE_TOL]
            l1 = entrances(c1)[HEADLINE_TOL]
            p = {"model": m, "dataset": ds, "q9": l9, "q1": l1, "delta": l1 - l9,
                 "q1_label": _lab(c1["labels"], l1), "q9_label": _lab(c9["labels"], l9)}
            cmp_cells[m][ds] = p
            pairs.append(p)
    agr = agreement(pairs)
    stats["q1_vs_q9"] = {"pairs": pairs, "not_comparable": notcmp, "agreement": agr}

    # ---- legacy: the committed (grid-clipped) LOOP Q=9 cells included, for the record ----------
    leg_pairs, leg_recs = [], []
    for m in MODELS:
        for ds in order:
            c9 = load_cell(Q9_ROOT, m, ds)
            if c9 is None:
                continue
            l9 = entrances(c9)[HEADLINE_TOL]
            leg_recs.append({"model": m, "dataset": ds, "idx": l9, "L": c9["L"]})
            c1 = load_cell(Q1_ROOT, m, ds)
            if c1 is not None:
                leg_pairs.append({"model": m, "dataset": ds,
                                  "delta": entrances(c1)[HEADLINE_TOL] - l9})
    stats["legacy_including_committed_loop_q9"] = {
        "note": "uses the committed lr=1e-2 LOOP Seattle Q=9 cells that SUPERSEDED_Q9 marks "
                "pending; reported only to reconcile with earlier counts",
        "q9_summary_05": summarize_entrances(leg_recs),
        "q1_vs_q9_agreement": agreement(leg_pairs)}

    stats["main_figure_datasets"] = {
        "datasets": list(args.main_datasets),
        "rule": "fixed illustrative trio, required complete for all three models; every dataset "
                "is shown in the appendix figures",
        "entrances_05": {m: {ds: _lab(q9[m][ds][0]["labels"], q9[m][ds][1][HEADLINE_TOL])
                             for ds in args.main_datasets if q9[m][ds][0] is not None}
                         for m in MODELS}}

    # ---- outputs --------------------------------------------------------------------------------
    out = args.out_dir
    figs = main_figure(q9, args.main_datasets, names, out)
    for m in MODELS:
        figs += appendix_figure(m, q9[m], order, names, out)
    # Q=1: the same appendix grids, same rule (sustained 5%, validation only), same checks
    q1_cells = {m: {} for m in MODELS}
    for m in MODELS:
        for ds in order:
            c, _, why = resolve(m, ds, Q1_ROOT, SUPERSEDED_Q1)
            if c is None:
                q1_cells[m][ds] = (None, None, why)
                continue
            ent = entrances(c)
            heldout(c, ent[HEADLINE_TOL])          # verifies the plotted CI against the bootstrap
            q1_cells[m][ds] = (c, ent, None)
        figs += appendix_figure(m, q1_cells[m], order, names, out, q="1")
    stats["pending_q1_figures"] = [{"model": m, "dataset": ds, "reason": q1_cells[m][ds][2]}
                                   for m in MODELS for ds in order if q1_cells[m][ds][0] is None]
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    t02, s02 = tunnel_table_tol(order, names, q9, 0.02)
    t08, s08 = tunnel_table_tol(order, names, q9, 0.08)
    stats["q9_summary_02"], stats["q9_summary_08"] = s02, s08
    tables = {"tunnel_q9_05.tex": tunnel_table(order, names, q9, summ),
              "tunnel_q9_02.tex": t02,
              "tunnel_q9_08.tex": t08,
              "tunnel_sensitivity.tex": sensitivity_table(order, names, q9, sens),
              "q1_vs_q9.tex": q1_table(order, names, cmp_cells, agr),
              "heldout_q9.tex": heldout_table(order, names, q9, ho),
              "heldout_q9_full.tex": heldout_table_long(order, names, q9, ho, hsum)}
    t258, s258 = sensitivity_table_eps(order, names, q9, (0.02, 0.05, 0.08))
    tables["tunnel_sensitivity_2_5_8.tex"] = t258
    stats["q9_sensitivity_2_5_8"] = {str(t): v for t, v in s258.items()}
    for fn, body in tables.items():
        (tdir / fn).write_text(body)
    stats["outputs"] = {"figures": figs,
                        "tables": [str((tdir / fn).relative_to(REPO_ROOT)) for fn in tables]}
    (out / "recoverability_stats.json").write_text(json.dumps(stats, indent=2))

    # ---- console summary --------------------------------------------------------------------------
    print(f"pending Q=9 cells: {[(p['model'], p['dataset']) for p in pending]}")
    for m in MODELS + ("all",):
        s, h, a = summ[m], hsum[m], agr[m]
        print(f"{m:9s} entrance<L {s['n_before_final']}/{s['n']}  median l/L "
              f"{s['median_norm']:.3f} IQR [{s['q25_norm']:.3f},{s['q75_norm']:.3f}]  "
              f"test deg median {100 * h['median_rel_degradation']:+.2f}%  "
              f"<=5% {h['n_within_5pct_at_entrance']}/{h['n']}  "
              f"Q1 exact {a['n_exact']}/{a['n']} within1 {a['n_within_1']}/{a['n']} "
              f"med|d| {a['median_abs_delta']}")
    print("legacy (committed LOOP included):",
          stats["legacy_including_committed_loop_q9"]["q1_vs_q9_agreement"]["all"])
    print("main-figure entrances:", stats["main_figure_datasets"]["entrances_05"])
    print("wrote", out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
