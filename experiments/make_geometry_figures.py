"""Results §"Geometry and forecast recoverability": appendix figures, tables and descriptors.

    python -m experiments.make_geometry_figures

Reads ONLY saved Phase-1 cell artifacts (``effective_rank.csv``, ``cka/*.npy``, and the Q=9
tunnel via ``make_recoverability_figures``, which re-verifies every entrance against the run's
own flags). No model, no GPU. Login node is fine (seconds).

CONVENTIONS (the committed Phase-1 ones, not re-chosen here)
  * CKA      : unbiased linear CKA, TEST split, raw hidden states (the Phase-1 headline).
  * erank    : entropy effective rank, TRAIN split (the committed convention); the test split is
               drawn beside it as a robustness curve.
  * variant  : Chronos-2 / TimesFM-3 "headline" (forecast slots / last real context token),
               TiRex "pos0". TiRex "all_positions_from_L1" is reported only as a diagnostic.
  * depth    : block depths only (L0..L_final). Chronos-2 L12+LN and TiRex L12+RMS are
               head-input diagnostics and never appear on a depth axis.
Geometry never sees a probe, so it exists even where the Q=9 tunnel is pending (TimesFM-3 x M4
geometry is read from the Q=1 run, whose geometry is verified identical to Q=9's; Loop Seattle
geometry is valid, only its tunnel is pending). Those panels are drawn without an entrance.

POST-HOC DESCRIPTORS (exploratory; NOT part of the registered Phase-1 analysis)
  erank peak          argmax of the train effective-rank curve.
  CKA boundary        the single split of the depth axis L1..L_final into two contiguous blocks
                      [L1, b-1] | [b, L] maximizing (mean within-block CKA - mean between-block
                      CKA). L0 is excluded because Chronos-2's L0 forecast-slot state is
                      near-degenerate (effective rank ~1), which makes any split isolating L0
                      trivially optimal.
  "near"              |depth - entrance| <= max(1, round(0.1 L)) blocks (1 for L=12, 2 for L=20).
  entry-step rank     rank of the step (entrance-1 -> entrance) among all L adjacent steps, by
                      CKA dissimilarity 1-CKA(l-1,l) and by |delta erank|; 1 = the largest change.
  TV in tunnel        share of the erank total variation sum_j |r_j - r_{j-1}| from steps
                      entering depths > entrance.
  CKA(entrance, L)    similarity of the entrance representation to the final block.
  sharpest CKA step   the adjacent step (k-1 -> k), k = 2..L, with the lowest CKA(k-1, k);
                      L0 -> L1 is excluded (the embedding step, and degenerate for Chronos-2).
                      "Inside the tunnel" = k > entrance.
  erank final/peak    r_L / max_l r_l; with the peak at or after the entrance, the whole decline
                      from peak to final block lies inside the tunnel.
  CKA boundary (excl. final)  as "CKA boundary" but over L1..L_{final-1}, so TiRex's
                      always-distinct final block cannot dominate the split.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.lines import Line2D                               # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

from experiments import make_recoverability_figures as rec       # noqa: E402
from experiments.make_recoverability_figures import (             # noqa: E402
    MODELS, MODEL_NAME, RC, TEXTWIDTH_IN, TUNNEL_FILL, TUNNEL_LINE, HEADLINE_TOL,
    _clean, _disp, _entrance_tag, _save, _xticks, draw_pending)

REPO_ROOT = rec.REPO_ROOT
OUT_DIR = REPO_ROOT / "three_models_paper" / "figures" / "geometry"
VARIANT = {"chronos2": "headline", "timesfm3": "headline", "tirex": "pos0"}
ERANK_SPLIT, CKA_SPLIT = "train", "test"
ERANK_C, ERANK_C2 = "#5E3C99", "#B2A4CF"
CKA_C = "#E08A00"               # orange: distinct from the green tunnel and the purple erank
# Main-figure rows (notes/geometry_results.txt, Sec. 4): SG Carpark = entrance without a clear
# geometric transition; London Smart Meters = geometric change continuing inside the tunnel.
MAIN_ROWS = ("sg_carpark", "monash_london_smart_meters")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def geom_dir(model, ds):
    """Geometry is Q-independent: prefer the Q=9 cell, fall back to the Q=1 cell."""
    for root in (rec.Q9_ROOT, rec.Q1_ROOT):
        if (root / model / ds / "COMPLETE").exists():
            return root / model / ds
    return None


def erank_curve(d, model, split, variant=None, depth_only=True):
    rows = [r for r in csv.DictReader(open(d / "effective_rank.csv"))
            if r["geometry_variant"] == (variant or VARIANT[model]) and r["split"] == split]
    if depth_only:
        rows = [r for r in rows if r["include_in_main_depth_axis"] == "True"]
    return [r["layer"] for r in rows], np.array([float(r["effective_rank"]) for r in rows])


def cka_matrix(d, model, split=CKA_SPLIT, variant=None, with_head=False):
    suf = "__with_head_input" if with_head else ""
    return np.load(d / "cka" / f"cka_unbiased_{variant or VARIANT[model]}_{split}{suf}.npy")


# --------------------------------------------------------------------------- #
# descriptors
# --------------------------------------------------------------------------- #
def interior_boundary(C, hi=None):
    n = len(C) if hi is None else hi + 1
    best = None
    for b in range(2, n):
        A, B = range(1, b), range(b, n)
        w = [C[i, j] for blk in (A, B) for i in blk for j in blk if i < j] or [1.0]
        sc = float(np.mean(w) - np.mean([C[i, j] for i in A for j in B]))
        if best is None or sc > best[1]:
            best = (b, sc)
    return best


def describe(model, ds, l):
    d = geom_dir(model, ds)
    labels, e = erank_curve(d, model, ERANK_SPLIT)
    _, et = erank_curve(d, model, "test")
    C, Ct = cka_matrix(d, model), cka_matrix(d, model, "train")
    L = len(e) - 1
    out = {"model": model, "dataset": ds, "L": L, "erank_peak": int(e.argmax()),
           "erank_peak_test_split": int(et.argmax()),
           "cka_boundary": interior_boundary(C)[0],
           "cka_boundary_score": round(interior_boundary(C)[1], 3),
           "cka_boundary_train_split": interior_boundary(Ct)[0], "entrance": l}
    if l is None:
        return out
    near = max(1, round(0.1 * L))
    rel = lambda x: "before" if x < l - near else "after" if x > l + near else "near"
    dC = np.array([1 - C[i - 1, i] for i in range(1, L + 1)])
    dE = np.abs(np.diff(e))
    out.update({
        "near_window": near, "erank_peak_rel": rel(out["erank_peak"]),
        "cka_boundary_rel": rel(out["cka_boundary"]),
        "entry_step_rank_cka": None if l == 0 else int((dC > dC[l - 1]).sum() + 1),
        "entry_step_rank_erank": None if l == 0 else int((dE > dE[l - 1]).sum() + 1),
        "erank_tv_in_tunnel": float(dE[l:].sum() / dE.sum()),
        "cka_entrance_to_final": float(C[l, L]),
        "erank_at_entrance": float(e[l]), "erank_peak_value": float(e.max()),
        "erank_final": float(e[L]),
        "erank_peak_at_or_after_entrance": bool(out["erank_peak"] >= l),
        "erank_final_over_peak": float(e[L] / e.max())})
    adj = np.array([C[k - 1, k] for k in range(2, L + 1)])
    k = int(adj.argmin()) + 2
    b2 = interior_boundary(C, hi=L - 1)[0]
    out.update({"sharpest_cka_step_to": k, "sharpest_cka_step_value": float(adj.min()),
                "sharpest_cka_step_in_tunnel": bool(k > l),
                "cka_boundary_excl_final": b2, "cka_boundary_excl_final_rel": rel(b2)})
    return out


def aggregate(descs):
    res = {}
    for m in MODELS + ("all",):
        allp = [x for x in descs if x["entrance"] is not None and m in ("all", x["model"])]
        nz = [x for x in allp if x["entrance"] > 0]
        cnt = lambda k, v: sum(x[k] == v for x in nz)
        res[m] = {
            "n_pairs": len(allp), "n_entrance_after_L0": len(nz),
            "erank_peak_before_near_after": [cnt("erank_peak_rel", v)
                                             for v in ("before", "near", "after")],
            "cka_boundary_before_near_after": [cnt("cka_boundary_rel", v)
                                               for v in ("before", "near", "after")],
            "entry_step_largest_cka": sum(x["entry_step_rank_cka"] == 1 for x in nz),
            "entry_step_top2_cka": sum(x["entry_step_rank_cka"] <= 2 for x in nz),
            "entry_step_largest_erank": sum(x["entry_step_rank_erank"] == 1 for x in nz),
            "entry_step_top2_erank": sum(x["entry_step_rank_erank"] <= 2 for x in nz),
            "tv_in_tunnel_ge_half": [sum(x["erank_tv_in_tunnel"] >= 0.5 for x in nz),
                                     sum(x["erank_tv_in_tunnel"] >= 0.5 for x in allp)],
            "cka_entrance_final_lt_half": [sum(x["cka_entrance_to_final"] < 0.5 for x in nz),
                                           sum(x["cka_entrance_to_final"] < 0.5 for x in allp)],
            "sharpest_cka_step_in_tunnel": sum(x["sharpest_cka_step_in_tunnel"] for x in allp),
            "sharpest_cka_step_is_final": sum(x["sharpest_cka_step_to"] == x["L"] for x in allp),
            "erank_peak_at_or_after_entrance": sum(x["erank_peak_at_or_after_entrance"]
                                                   for x in allp),
            "median_erank_final_over_peak": float(np.median(
                [x["erank_final_over_peak"] for x in allp])),
            "cka_boundary_excl_final_before_near_after": [
                cnt("cka_boundary_excl_final_rel", v) for v in ("before", "near", "after")],
            "erank_peak_split_agreement": sum(x["erank_peak"] == x["erank_peak_test_split"]
                                              for x in allp),
            "cka_boundary_split_agreement_within1": sum(
                abs(x["cka_boundary"] - x["cka_boundary_train_split"]) <= 1 for x in allp)}
    return res


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _grid(nrow=4, ncol=4, h=5.6, hspace=0.13):
    fig, axes = plt.subplots(nrow, ncol, figsize=(TEXTWIDTH_IN, h), layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=hspace)
    return fig, axes


def erank_figure(model, tun, order, names, out):
    with plt.rc_context({**RC, "axes.titlesize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, axes = _grid()
        flat = axes.ravel()
        for i, ds in enumerate(order):
            ax, show = flat[i], i + 4 >= len(order)
            d = geom_dir(model, ds)
            labels, e = erank_curve(d, model, ERANK_SPLIT)
            _, et = erank_curve(d, model, "test")
            x, L, l = np.arange(len(e)), len(e) - 1, tun[ds]
            if l is not None:
                ax.axvspan(l, L + 0.5, color=TUNNEL_FILL, lw=0, zorder=0)
                ax.axvline(l, color=TUNNEL_LINE, lw=1.1, zorder=5)
            ax.plot(x, et, ls=(0, (3, 1.5)), lw=1.0, color=ERANK_C2, zorder=3)
            ax.plot(x, e, "-o", ms=1.9, lw=1.2, color=ERANK_C, zorder=4)
            pk = int(e.argmax())
            ax.plot([pk], [e[pk]], "*", ms=7, color=ERANK_C, mec="white", mew=0.5, zorder=6)
            ax.set_ylim(0, max(e.max(), et.max()) * 1.25)
            if l is not None:
                _entrance_tag(ax, {"test": e, "labels": labels, "L": L}, l, TUNNEL_LINE)
            else:
                ax.text(0.97, 0.95, "tunnel pending", transform=ax.transAxes, ha="right",
                        va="top", fontsize=6, color="#8A8A8A")
            _xticks(ax, labels, show, compact=True)
            _clean(ax)
            ax.set_title(names[ds], fontweight="bold", pad=2).set_in_layout(False)
            if show:
                ax.set_xlabel("Block depth", fontsize=7, labelpad=3)
        for ax in flat[len(order):]:
            ax.remove()
        lax = fig.add_subplot(axes[0, 0].get_gridspec()[3, len(order) % 4:])
        lax.axis("off")
        fig.supylabel("Effective rank", fontsize=7.5)
        lax.legend(handles=[
            Line2D([], [], color=ERANK_C, marker="o", ms=2.6, lw=1.2,
                   label="Effective rank (train, headline)"),
            Line2D([], [], color=ERANK_C2, lw=1.0, ls=(0, (3, 1.5)), label="Effective rank (test)"),
            Line2D([], [], color=ERANK_C, marker="*", ms=6, lw=0, label="Peak (train)"),
            Line2D([], [], color=TUNNEL_LINE, lw=1.2, marker="o", ms=4.5, mfc="white",
                   mec=TUNNEL_LINE, mew=1.2, label="Q=9 tunnel entrance (val., 5%)"),
            Patch(facecolor=TUNNEL_FILL, lw=0, label="Sustained 5% tunnel")],
            loc="center", frameon=False, handlelength=1.8, fontsize=7, borderaxespad=0.0)
        return _save(fig, out, f"appendix_erank_{model}")


def cka_figure(model, tun, order, names, out):
    with plt.rc_context({**RC, "axes.titlesize": 8, "xtick.labelsize": 5.5,
                         "ytick.labelsize": 5.5}):
        fig, axes = _grid(h=5.9, hspace=0.2)
        flat = axes.ravel()
        im = None
        for i, ds in enumerate(order):
            ax, d = flat[i], geom_dir(model, ds)
            C = cka_matrix(d, model)
            labels, _ = erank_curve(d, model, ERANK_SPLIT)
            L = len(C) - 1
            im = ax.imshow(np.clip(C, 0, 1), vmin=0, vmax=1, cmap="viridis", origin="upper",
                           interpolation="nearest")
            step = 5 if L > 12 else 3
            ticks = list(range(0, L + 1, step))
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([_disp(labels[t]) for t in ticks])
            ax.set_yticklabels([_disp(labels[t]) for t in ticks])
            ax.tick_params(length=1.5, pad=1)
            l = tun[ds]
            if l is not None:
                for f in (ax.axvline, ax.axhline):
                    f(l - 0.5, color="white", lw=1.0, ls=(0, (3, 1.5)))
                ax.text(l - 0.5, -0.9, _disp(labels[l]), ha="center", va="bottom",
                        fontsize=6, fontweight="bold", color="white",
                        bbox=dict(boxstyle="round,pad=0.15", fc=TUNNEL_LINE, ec="none"),
                        clip_on=False)
            else:
                ax.text(0.97, 0.03, "tunnel pending", transform=ax.transAxes, ha="right",
                        va="bottom", fontsize=5.5, color="white")
            ax.set_title(names[ds], fontweight="bold", pad=12).set_in_layout(False)
        for ax in flat[len(order):]:
            ax.remove()
        cax = fig.add_subplot(axes[0, 0].get_gridspec()[3, len(order) % 4:])
        cax.axis("off")
        cb = fig.colorbar(im, ax=cax, orientation="horizontal", fraction=0.35, aspect=18,
                          location="bottom")
        cb.set_label("Unbiased linear CKA (test)", fontsize=7)
        cb.ax.tick_params(labelsize=6)
        cax.text(0.5, 0.75, "dashed lines: Q=9 tunnel entrance\n(validation, sustained 5%)",
                 ha="center", va="center", fontsize=7, transform=cax.transAxes)
        return _save(fig, out, f"appendix_cka_{model}")


def main_figure(tun, names, out, rows=MAIN_ROWS):
    """2 x 3: rows = datasets, columns = models. Left axis: effective rank (train split);
    right axis: unbiased CKA(l, L) with the final block (test split), fixed to [0, 1]. The
    validation-selected sustained 5% tunnel is shaded with its entrance line + tag."""
    for m in MODELS:
        for ds in rows:
            if tun[m][ds] is None:
                raise ValueError(f"main figure: {m} x {ds} tunnel is pending")
    with plt.rc_context(RC):
        fig, axes = plt.subplots(len(rows), 3, figsize=(TEXTWIDTH_IN, 3.35),
                                 layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.03, wspace=0.08, hspace=0.12)
        for r, ds in enumerate(rows):
            for c, m in enumerate(MODELS):
                ax = axes[r, c]
                d = geom_dir(m, ds)
                labels, e = erank_curve(d, m, ERANK_SPLIT)
                C = cka_matrix(d, m)
                x, L, l = np.arange(len(e)), len(e) - 1, tun[m][ds]
                ax.axvspan(l, L + 0.5, color=TUNNEL_FILL, lw=0, zorder=0)
                ax.axvline(l, color=TUNNEL_LINE, lw=1.1, zorder=5)
                ax.plot(x, e, "-o", ms=2.1, lw=1.3, color=ERANK_C, zorder=4)
                ax.set_ylim(0, e.max() * 1.3)
                _entrance_tag(ax, {"test": e, "labels": labels, "L": L}, l, TUNNEL_LINE)
                ax2 = ax.twinx()
                ax2.plot(x, np.clip(C[:, L], 0, 1), ls=(0, (3.5, 1.6)), marker="s", ms=1.8,
                         lw=1.1, color=CKA_C, zorder=3)
                ax2.set_ylim(0, 1.05)
                ax2.set_yticks([0, 0.5, 1])
                ax2.tick_params(labelsize=6.5, length=2.5, pad=1.5, colors=CKA_C)
                ax2.spines["top"].set_visible(False)
                ax2.spines["right"].set_color(CKA_C)
                if c < 2:
                    ax2.set_yticklabels([])
                else:
                    ax2.set_ylabel("CKA with final block", color=CKA_C, fontsize=7.5)
                _xticks(ax, labels, show=(r == len(rows) - 1), compact=True)
                _clean(ax)
                ax.tick_params(axis="y", colors=ERANK_C)
                ax.spines["left"].set_color(ERANK_C)
                if r == 0:
                    ax.set_title(MODEL_NAME[m], fontweight="bold", pad=2)
                if c == 0:
                    ax.set_ylabel("Effective rank", color=ERANK_C, fontsize=7.5)
                    # row label: dataset name, bold black, left of the axis label
                    # wrap long names onto two lines so the rotated label fits its row height
                    words = names[ds].split()
                    label = (names[ds] if len(names[ds]) <= 12 else
                             " ".join(words[:len(words) // 2]) + "\n"
                             + " ".join(words[len(words) // 2:]))
                    ax.annotate(label, xy=(0, 0.5), xycoords="axes fraction",
                                xytext=(-34, 0), textcoords="offset points", rotation=90,
                                ha="right", va="center", multialignment="center",
                                fontsize=9, fontweight="bold", color="black",
                                annotation_clip=False)
                if r == len(rows) - 1 and c == 1:
                    ax.set_xlabel("Block depth", labelpad=4)
        handles = [Line2D([], [], color=ERANK_C, marker="o", ms=2.6, lw=1.3,
                          label="Effective rank (train)"),
                   Line2D([], [], color=CKA_C, marker="s", ms=2.4, lw=1.1, ls=(0, (3.5, 1.6)),
                          label="CKA with final block (test)"),
                   Line2D([], [], color=TUNNEL_LINE, lw=1.1, marker="o", ms=4.5, mfc="white",
                          mec=TUNNEL_LINE, mew=1.2, label="Tunnel entrance (val., 5%)"),
                   Patch(facecolor=TUNNEL_FILL, lw=0, label="Sustained 5% tunnel")]
        fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False,
                   handlelength=1.8, columnspacing=1.6, fontsize=7, borderpad=0.1)
        return _save(fig, out, "geometry_main")


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def summary_table(descs, order, names):
    """Per pair: validation-selected entrance l*, erank peak, the sharpest adjacent CKA step
    (-> Lk; excludes L0->L1) and CKA(l*, L). A dagger marks a sharpest step inside the tunnel."""
    by = {(x["model"], x["dataset"]): x for x in descs}
    lines = [r"\begin{tabular}{l cccc cccc cccc}", r"\toprule",
             r" & \multicolumn{4}{c}{Chronos-2} & \multicolumn{4}{c}{TimesFM-3} & "
             r"\multicolumn{4}{c}{TiRex} \\",
             r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
             "Dataset" + r" & $\ell^\ast$ & peak & step & CKA" * 3 + r" \\", r"\midrule"]
    for ds in order:
        cells = []
        for m in MODELS:
            x = by[(m, ds)]
            if x["entrance"] is None:
                cells += [r"\textit{pend.}", f"L{x['erank_peak']}", "--", "--"]
            else:
                dag = r"$^\dagger$" if x["sharpest_cka_step_in_tunnel"] else ""
                cells += [f"L{x['entrance']}", f"L{x['erank_peak']}",
                          f"L{x['sharpest_cka_step_to']}{dag}",
                          f"{x['cka_entrance_to_final']:.2f}"]
        lines.append(f"{names[ds]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def diagnostics_table(order, names):
    """Final-normalization states (Chronos-2 L12+LN, TiRex L12+RMS) as separate diagnostics:
    erank at L12 vs the normalized head input, and their CKA. TiRex pos0 vs all_positions_from_L1:
    Spearman correlation of the two erank curves over L1..L12 and their argmax."""
    from scipy.stats import spearmanr
    lines = [r"\begin{tabular}{l ccc ccc ccc}", r"\toprule",
             r" & \multicolumn{3}{c}{Chronos-2: L12 vs L12+LN} & "
             r"\multicolumn{3}{c}{TiRex: L12 vs L12+RMS} & "
             r"\multicolumn{3}{c}{TiRex: pos0 vs all-pos.\ (L1--L12)} \\",
             r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(lr){8-10}",
             r"Dataset & $r_{\mathrm{eff}}$ L12 & $r_{\mathrm{eff}}$ +LN & CKA & "
             r"$r_{\mathrm{eff}}$ L12 & $r_{\mathrm{eff}}$ +RMS & CKA & $\rho_s$ & "
             r"peak pos0 & peak all \\", r"\midrule"]
    for ds in order:
        cells = []
        for m in ("chronos2", "tirex"):
            d = geom_dir(m, ds)
            lab, e = erank_curve(d, m, ERANK_SPLIT, depth_only=False)
            Cf = cka_matrix(d, m, with_head=True)
            cells += [f"{e[-2]:.1f}", f"{e[-1]:.1f}", f"{Cf[-2, -1]:.2f}"]
        d = geom_dir("tirex", ds)
        l0, e0 = erank_curve(d, "tirex", ERANK_SPLIT, "pos0")
        la, ea = erank_curve(d, "tirex", ERANK_SPLIT, "all_positions_from_L1")
        e0 = e0[[l0.index(x) for x in la]]
        rho = spearmanr(e0, ea).statistic
        cells += [f"{rho:.2f}", la[int(e0.argmax())], la[int(ea.argmax())]]
        lines.append(f"{names[ds]} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)
    order, names = rec._roster(rec.Q9_ROOT)
    tunnels = {m: {} for m in MODELS}
    for m in MODELS:
        for ds in order:
            c, _, _ = rec.resolve_q9(m, ds)
            tunnels[m][ds] = None if c is None else rec.entrances(c)[HEADLINE_TOL]
    descs = [describe(m, ds, tunnels[m][ds]) for m in MODELS for ds in order]
    agg = aggregate(descs)
    out = args.out_dir
    figs = []
    figs += main_figure(tunnels, names, out)
    for m in MODELS:
        figs += erank_figure(m, tunnels[m], order, names, out)
        figs += cka_figure(m, tunnels[m], order, names, out)
    tdir = out / "tables"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "geometry_summary.tex").write_text(summary_table(descs, order, names))
    (tdir / "geometry_diagnostics.tex").write_text(diagnostics_table(order, names))
    (out / "geometry_descriptors.json").write_text(json.dumps(
        {"conventions": {"cka": "unbiased, test split", "erank": "train split",
                         "variant": VARIANT, "status": "post-hoc exploratory descriptors"},
         "pairs": descs, "aggregate": agg, "figures": figs}, indent=1))
    for m in MODELS + ("all",):
        print(m, agg[m])
    print("wrote", out.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
