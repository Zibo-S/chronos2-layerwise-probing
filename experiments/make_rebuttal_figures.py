"""Revised Phase-2 figures with the zero-anchored TimesFM-3 adapter (rebuttal, Appendix B.3.1).

    python experiments/make_rebuttal_figures.py

Reuses the plotting code of ``experiments/make_phase2_paper_tables.py`` unchanged and swaps ONE
input: the TimesFM-3 aligned pathway ("noa") at depths L0..L19 is replaced by the zero-anchored
closed form of Phase 2g (``results/three_model_phase2g_timesfm3_zero_ridge/combined/
depth_metrics.csv``, arm "zero"). Chronos-2, TiRex, the hard cut, the readout, the speedups and
Chronos-2-small are read from the committed Phase-2 tables as before.

Writes NEW files next to the originals, which are left untouched:
    three_models_paper/figures/truncation/h3_main_electricity_rev.{pdf,png}   (Figure 4)
    three_models_paper/figures/truncation/h4_frontier_rev.{pdf,png}           (Figure 5)
    three_models_paper/figures/truncation/h3_depth_timesfm3_rev.{pdf,png}     (Appendix B.3)
The appendix TimesFM-3 figure also draws the identity-anchored fit of the submission.
"""
from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

# On the Mac checkout ``features_cache`` is a link to Narval scratch; probing.config's
# ``mkdir(exist_ok=True)`` raises on a dangling link. Tolerate exactly that case.
_mkdir = pathlib.Path.mkdir


def _tolerant_mkdir(self, *a, **k):
    try:
        return _mkdir(self, *a, **k)
    except FileExistsError:
        if self.is_symlink():
            return None
        raise


pathlib.Path.mkdir = _tolerant_mkdir

import experiments.make_phase2_paper_tables as M                                  # noqa: E402

COMB = REPO / "results" / "three_model_phase2" / "combined"
G2 = REPO / "results" / "three_model_phase2g_timesfm3_zero_ridge" / "combined"
OUT = REPO / "three_models_paper" / "figures" / "truncation"
FIELDS = ("mase_ratio_vs_native", "mase_ratio_ci_lo", "mase_ratio_ci_hi", "wql_ratio_vs_native")


def zero_rows():
    return {(r["dataset"], int(r["depth_index"])): r for r in M._rows(G2 / "depth_metrics.csv")
            if r["arm"] == "zero"}


def swapped_depth_rows(depth_rows, zr, keep_identity=False):
    """TimesFM-3 'noa' rows at L0..L19 -> the zero-anchored values; optionally keep the
    identity-anchored rows under the family 'noa_id'."""
    out = []
    for r in depth_rows:
        if r["model"] == "timesfm3" and r["family"] == "noa" and int(r["depth_index"]) < 20:
            z = zr[(r["dataset"], int(r["depth_index"]))]
            out.append({**r, **{k: z[k] for k in FIELDS}})
            if keep_identity:
                out.append({**r, "family": "noa_id"})
        else:
            out.append(r)
    return out


def swapped_frontier(fr_summ, zr):
    """Recompute the TimesFM-3 aligned frontier rows (median / IQR over the same datasets)."""
    out = []
    for r in fr_summ:
        if r["model"] == "timesfm3" and r["arm"] == "noa" and M._true(r["includes_low_skill"]):
            d = int(r["depth_index"])
            v = np.array([float(zr[(ds, d)]["mase_ratio_vs_native"]) - 1
                          for (ds, dd) in zr if dd == d])
            assert len(v) == int(r["n_datasets"]), (d, len(v), r["n_datasets"])
            q = np.percentile(v, [25, 50, 75])
            out.append({**r, "degradation_q25": q[0], "degradation_median": q[1],
                        "degradation_q75": q[2]})
        else:
            out.append(r)
    return out


def chronos2_small(joined):
    srows = [r for r in joined if r.get("arm") == "chronos2_small"]
    deg = [M._f(r["mase_ratio"]) - 1 for r in srows]
    return {"degradation": float(np.median(deg)),
            "speedup_e2e_b1": float(np.median([M._f(r["api_e2e_B1_speedup"]) for r in srows])),
            "speedup_e2e_b256": float(np.median([M._f(r["api_e2e_B256_speedup"]) for r in srows]))}


def depth_figure_timesfm3(depth_rows, ladder, out_dir, stem):
    """M.depth_figure_model for TimesFM-3 with the identity-anchored arm drawn as well."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model = "timesfm3"
    rm = [r for r in depth_rows if r["model"] == model]
    ent = M._entrances(ladder)
    order = M._order({r["dataset"] for r in rm})
    ncol, L = 4, M._final_index(model, rm)
    nrow = max(1, -(-(len(order) + 2) // ncol))
    arms = ("hard", "probe", "noa_id", "noa")
    with plt.rc_context({**M.RC, "axes.titlesize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6}):
        fig, axes = plt.subplots(nrow, ncol, figsize=(M.TEXTWIDTH_IN, 1.4 * nrow),
                                 layout="constrained", squeeze=False)
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.05, hspace=0.13)
        flat = axes.ravel()
        for i, tag in enumerate(order):
            ax = flat[i]
            show = i + ncol >= len(order)
            M._depth_panel(ax, [r for r in rm if r["dataset"] == tag], ent.get((model, tag)), L,
                           show_x=show, arms=arms)
            ax.set_title(M._dname(tag), fontweight="bold", pad=2).set_in_layout(False)
            if show:
                ax.set_xlabel("Block depth", fontsize=7, labelpad=3)
        for ax in flat[len(order):]:
            ax.remove()
        lax = fig.add_subplot(axes[0, 0].get_gridspec()[nrow - 1, len(order) % ncol:])
        lax.axis("off")
        lax.legend(handles=M._depth_legend_handles((model,), arms=("noa", "noa_id", "probe", "hard")),
                   loc="center", ncol=1, frameon=False, handlelength=1.8, fontsize=6.5,
                   borderaxespad=0.0)
        fig.supylabel("Test MASE increase over the full model", fontsize=7.5)
        return M._save(fig, out_dir, stem)


def main() -> int:
    depth_rows = M._rows(COMB / "h3_depth_metrics.csv")
    ladder = M._rows(COMB / "h3_ladder.csv")
    fr_summ = M._rows(COMB / "h4_frontier_summary.csv")
    small = chronos2_small(M._rows(COMB / "h4_with_latency.csv"))
    zr = zero_rows()
    assert len(zr) == 14 * 21, len(zr)
    rows = swapped_depth_rows(depth_rows, zr)
    # labels: the aligned arm keeps its name; the submission's fit is the extra arm
    M.ARMS["noa_id"] = ("Aligned, identity anchor (submission)",
                        "Aligned, identity anchor (submission)", "#7FB8DA", (0, (1.2, 1.2)), 0.8)
    M.depth_curves(rows, ladder, OUT, stem="h3_main_electricity_rev",
                   datasets=("monash_electricity_hourly",), height=1.95, legend_ncol=3,
                   arms=("hard", "probe", "noa"), floor_at_zero=True)
    M.frontier_figure(swapped_frontier(fr_summ, zr), OUT, small=small, stem="h4_frontier_rev")
    depth_figure_timesfm3(swapped_depth_rows(depth_rows, zr, keep_identity=True), ladder, OUT,
                          stem="h3_depth_timesfm3_rev")
    for s in ("h3_main_electricity_rev", "h4_frontier_rev", "h3_depth_timesfm3_rev"):
        print(OUT / f"{s}.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
