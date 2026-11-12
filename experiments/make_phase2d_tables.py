"""Phase 2d tables, figures and LaTeX -- rebuilt from the cell artifacts ONLY.

Aggregation and plotting on numbers the GPU jobs already computed (every bootstrap CI is in the
cells), so this is login-node safe:

    python -m experiments.make_phase2d_tables
    python -m experiments.make_phase2d_tables --output-root results/three_model_phase2d_pooled_smoke

Outputs
    <output-root>/combined/   entrances.csv, final_block.csv, decomposition.csv,
                              layer_comparison.csv, aggregate_by_model.csv, param_counts.csv,
                              stats.json
    <paper-dir>/              pooled_curves_main.pdf, pooled_curves_<model>.pdf,
                              pooled_final_delta.pdf, pooled_entrances.pdf (+ .png)
    <paper-dir>/tables/       tab_pooled_entrances.tex, tab_pooled_final.tex,
                              tab_pooled_aggregate.tex, tab_pooled_params.tex, pooled_macros.tex

A missing cell keeps its row / panel and is marked pending; it is never dropped or guessed.
Both the losses AND the entrances are reported: an entrance is relative to each arm's OWN final
block, so an earlier entrance can come from a weaker final block.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from probing import phase1, registry                                              # noqa: E402
from probing import phase2d_pooled as p2d                                         # noqa: E402

DEFAULT_OUT = REPO_ROOT / "results" / "three_model_phase2d_pooled"
PAPER_DIR = REPO_ROOT / "three_models_paper" / "figures" / "pooled"
MODEL_NAME = {"chronos2": "Chronos-2", "timesfm3": "TimesFM-3", "tirex": "TiRex"}
MACRO_MODEL = {"chronos2": "ChronosTwo", "timesfm3": "TimesFMThree", "tirex": "TiRex"}
ARM_ORDER = (p2d.ARM_NATIVE, p2d.ARM_CONCAT, p2d.ARM_POOLED, p2d.ARM_NATIVE_MEAN)
ARM_LABEL = {"native": "Architecture-aware (native)", "concat": "Native states, concatenated",
             "pooled": "Content mean (generic)", "native_mean": "Native states, averaged"}
ARM_SHORT = {"native": "Native", "concat": "Concat", "pooled": "Pooled",
             "native_mean": "Nat.\\,mean"}
MACRO_ARM = {"native": "Native", "concat": "Concat", "pooled": "Pooled",
             "native_mean": "NativeMean"}
# Okabe-Ito, in this legend order: validated (dataviz validate_palette.js, light surface):
# worst adjacent CVD dE 11.0, normal-vision 16.4. The pink is below 3:1 contrast, so every arm
# also carries its own line style + marker (secondary encoding) and the CSVs are the table view.
ARM_STYLE = {"native": ("#0072B2", "-", "o"), "concat": ("#009E73", "-.", "s"),
             "pooled": ("#D55E00", "--", "^"), "native_mean": ("#CC79A7", ":", "D")}
STEPS = ("readout_structure", "pooling", "token_selection", "representation")
STEP_LABEL = {"readout_structure": "readout structure (native -> concat)",
              "pooling": "pooling + capacity (concat -> native mean; params drop K-fold)",
              "token_selection": "token selection (native mean -> pooled; same head + params)",
              "representation": "representation (native -> pooled, TimesFM-3)"}


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _b(x) -> bool:
    return str(x).strip().lower() in ("true", "1")


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def load_cells(root: Path, models, tags) -> dict:
    return {(m, t): p2d.read_cell(root, m, t) for m in models for t in tags}


def curves_of(cell, arm: str) -> dict | None:
    rows = [r for r in cell["layer_rows"] if r["arm"] == arm and
            _b(r["include_in_main_depth_axis"])]
    if not rows:
        return None
    rows.sort(key=lambda r: int(r["point_index"]))
    return {"labels": [r["layer"] for r in rows],
            "val": np.array([_f(r["val_loss"]) for r in rows]),
            "test": np.array([_f(r["test_loss"]) for r in rows]),
            "lo": np.array([_f(r["test_loss_ci_lo"]) for r in rows]),
            "hi": np.array([_f(r["test_loss_ci_hi"]) for r in rows]),
            "entrance": next((i for i, r in enumerate(rows)
                              if _b(r.get("is_tunnel_entrance_0.05"))), None)}


def _cmp(cell, a, b, layer=None, metric="loss", split="test", reference=False):
    for r in cell["comparisons"]:
        if (r["from_arm"], r["to_arm"], r["metric"], r["split"]) != (a, b, metric, split):
            continue
        if (reference and _b(r["is_reference_point"])) or (layer is not None and
                                                            r["layer"] == layer):
            return r
    return None


def _spearman(x, y) -> float:
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #
def build_tables(cells: dict, models, tags) -> dict:
    ent_rows, fin_rows, dec_rows, cmp_rows, pending = [], [], [], [], []
    for m in models:
        spec = phase1.model_spec(m)
        n_depth = spec.n_depth_points
        for t in tags:
            c = cells[(m, t)]
            if c is None:
                pending.append(f"{m}/{t}")
                ent_rows.append({"model": m, "dataset": t,
                                 "display_name": registry.display_name(t), "status": "pending"})
                continue
            s = c["summary"]
            nat = s["per_arm"][p2d.ARM_NATIVE]
            row = {"model": m, "dataset": t, "display_name": registry.display_name(t),
                   "status": "complete", "n_depth_points": n_depth,
                   "final_block": s["reference_label"],
                   "native_forecast_mase": s.get("native_forecast", {}).get("mase", "")}
            for a in ARM_ORDER:
                if a not in s["per_arm"]:
                    continue
                r = s["per_arm"][a]
                fb = r["final_block"]
                row.update({
                    f"{a}_entrance": r["headline_entrance"],
                    f"{a}_entrance_index": r["headline_depth_axis_index"],
                    f"{a}_entrance_relative_depth": r["headline_relative_depth"],
                    f"{a}_pre_final": r["pre_final_entrance"],
                    f"{a}_entrance_2pct": r["entrance"]["tol_0.02"]["label"],
                    f"{a}_entrance_10pct": r["entrance"]["tol_0.1"]["label"],
                    f"{a}_final_val_loss": fb["val_loss"], f"{a}_final_test_loss": fb["test_loss"],
                    f"{a}_final_test_mase": fb["test_mase"],
                    f"{a}_params": r["params"]["n_params"]})
                if a != p2d.ARM_NATIVE:
                    row[f"{a}_entrance_minus_native_blocks"] = (
                        r["headline_depth_axis_index"] - nat["headline_depth_axis_index"])
                fin = {"model": m, "dataset": t, "display_name": registry.display_name(t),
                       "arm": a, "params": r["params"]["n_params"],
                       "final_block": s["reference_label"], "final_val_loss": fb["val_loss"],
                       "final_test_loss": fb["test_loss"],
                       "final_test_loss_ci_lo": fb["test_loss_ci"][0],
                       "final_test_loss_ci_hi": fb["test_loss_ci"][1],
                       "final_test_mase": fb["test_mase"],
                       "final_test_mase_ci_lo": fb["test_mase_ci"][0],
                       "final_test_mase_ci_hi": fb["test_mase_ci"][1],
                       "entrance": r["headline_entrance"],
                       "test_loss_at_entrance": r["at_entrance"]["test_loss"],
                       "test_mase_at_entrance": r["at_entrance"]["test_mase"]}
                if a != p2d.ARM_NATIVE:
                    for metric in ("loss", "mase"):
                        cr = _cmp(c, p2d.ARM_NATIVE, a, metric=metric, reference=True)
                        fin[f"{metric}_delta_rel_vs_native"] = _f(cr["delta_rel"])
                        fin[f"{metric}_delta_rel_ci_lo"] = _f(cr["delta_rel_ci_lo"])
                        fin[f"{metric}_delta_rel_ci_hi"] = _f(cr["delta_rel_ci_hi"])
                        fin[f"{metric}_direction"] = cr["direction"]
                    cn = _cmp(c, p2d.ARM_NATIVE, a, layer=nat["headline_entrance"])
                    fin["loss_delta_rel_at_native_entrance"] = _f(cn["delta_rel"])
                    fin["direction_at_native_entrance"] = cn["direction"]
                    ca, cv = curves_of(c, a), curves_of(c, p2d.ARM_NATIVE)
                    fin["spearman_val_curve_vs_native"] = _spearman(ca["val"], cv["val"])
                fin_rows.append(fin)
            ent_rows.append(row)
            for step in STEPS:
                pair = [(r["from_arm"], r["to_arm"]) for r in c["comparisons"]
                        if r["step"] == step]
                if not pair:
                    continue
                a, b = pair[0]
                for metric in ("loss", "mase"):
                    cr = _cmp(c, a, b, metric=metric, reference=True)
                    dec_rows.append({"model": m, "dataset": t, "step": step, "from_arm": a,
                                     "to_arm": b, "metric": metric, "point": cr["layer"],
                                     "delta_rel": _f(cr["delta_rel"]),
                                     "delta_rel_ci_lo": _f(cr["delta_rel_ci_lo"]),
                                     "delta_rel_ci_hi": _f(cr["delta_rel_ci_hi"]),
                                     "direction": cr["direction"],
                                     "params_from": s["readout_params"][a]["n_params"],
                                     "params_to": s["readout_params"][b]["n_params"]})
            cmp_rows += c["comparisons"]
    agg = aggregate(ent_rows, fin_rows, dec_rows, models, len(tags))
    return {"entrances": ent_rows, "final_block": fin_rows, "decomposition": dec_rows,
            "layer_comparison": cmp_rows, "aggregate": agg, "pending": pending}


def aggregate(ent_rows, fin_rows, dec_rows, models, n_tags) -> list[dict]:
    out = []
    for m in models:
        done = [r for r in ent_rows if r["model"] == m and r["status"] == "complete"]
        for a in ARM_ORDER:
            have = [r for r in done if f"{a}_entrance" in r]
            if not have:
                continue
            fr = [r for r in fin_rows if r["model"] == m and r["arm"] == a]
            rec = {"model": m, "arm": a, "n_cells": len(have), "n_planned": n_tags,
                   "params": have[0][f"{a}_params"],
                   "n_pre_final_entrance": int(sum(bool(r[f"{a}_pre_final"]) for r in have)),
                   "median_entrance_relative_depth": float(np.median(
                       [r[f"{a}_entrance_relative_depth"] for r in have])),
                   "median_final_test_loss": float(np.median(
                       [r["final_test_loss"] for r in fr]))}
            if a != p2d.ARM_NATIVE:
                d = np.array([r[f"{a}_entrance_minus_native_blocks"] for r in have])
                dl = np.array([r["loss_delta_rel_vs_native"] for r in fr])
                dirs = [r["loss_direction"] for r in fr]
                rec.update({
                    "entrance_same_as_native": int((d == 0).sum()),
                    "entrance_within_1_block": int((np.abs(d) <= 1).sum()),
                    "entrance_within_2_blocks": int((np.abs(d) <= 2).sum()),
                    "entrance_earlier_than_native": int((d < 0).sum()),
                    "entrance_later_than_native": int((d > 0).sum()),
                    "median_entrance_minus_native_blocks": float(np.median(d)),
                    "median_final_loss_delta_rel_vs_native": float(np.median(dl)),
                    "n_final_worse_than_native_ci": dirs.count("to_worse"),
                    "n_final_better_than_native_ci": dirs.count("to_better"),
                    "n_final_inconclusive": dirs.count("inconclusive"),
                    "median_loss_delta_rel_at_native_entrance": float(np.median(
                        [r["loss_delta_rel_at_native_entrance"] for r in fr])),
                    "median_spearman_val_curve_vs_native": float(np.nanmedian(
                        [r["spearman_val_curve_vs_native"] for r in fr]))})
            out.append(rec)
        for step in STEPS:
            dr = [r for r in dec_rows if r["model"] == m and r["step"] == step and
                  r["metric"] == "loss"]
            if dr:
                dirs = [r["direction"] for r in dr]
                out.append({"model": m, "arm": f"step:{step}", "n_cells": len(dr),
                            "n_planned": n_tags,
                            "median_final_loss_delta_rel_step": float(np.median(
                                [r["delta_rel"] for r in dr])),
                            "n_step_worse_ci": dirs.count("to_worse"),
                            "n_step_better_ci": dirs.count("to_better"),
                            "n_step_inconclusive": dirs.count("inconclusive")})
    return out


def write_csvs(T: dict, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name in ("entrances", "final_block", "decomposition", "layer_comparison", "aggregate"):
        rows = T[name]
        if not rows:
            continue
        fields = []
        for r in rows:
            fields += [k for k in r if k not in fields]
        fn = "aggregate_by_model.csv" if name == "aggregate" else f"{name}.csv"
        phase1.write_csv(out_dir / fn, rows, fields)
        written.append(str(out_dir / fn))
    phase1.write_csv(out_dir / "param_counts.csv", p2d.param_table())
    written.append(str(out_dir / "param_counts.csv"))
    return written


# --------------------------------------------------------------------------- #
# LaTeX
# --------------------------------------------------------------------------- #
def _pct(x, signed=True) -> str:
    if x is None or not np.isfinite(x):
        return "--"
    return f"{x * 100:+.1f}" if signed else f"{x * 100:.1f}"


def _arms(m):
    return (p2d.ARM_NATIVE,) + tuple(a for a in (p2d.ARM_CONCAT, p2d.ARM_NATIVE_MEAN,
                                                 p2d.ARM_POOLED) if a in p2d.ARMS_BY_MODEL[m])


def latex_tables(T: dict, models, tags, tex_dir: Path) -> list[str]:
    tex_dir.mkdir(parents=True, exist_ok=True)
    ent = {(r["model"], r["dataset"]): r for r in T["entrances"]}
    fin = {(r["model"], r["dataset"], r["arm"]): r for r in T["final_block"]}
    written = []

    # 1. entrances, 14 x (4 + 1 + 4)
    cols = [(m, a) for m in models for a in _arms(m)]
    head1 = " & ".join(f"\\multicolumn{{{len(_arms(m))}}}{{c}}{{{MODEL_NAME[m]}}}"
                       for m in models)
    head2 = " & ".join(ARM_SHORT[a] for _m, a in cols)
    lines = ["% Phase 2d: sustained 5% validation entrance per readout (generated; do not edit)",
             f"\\begin{{tabular}}{{l{'c' * len(cols)}}}", "\\toprule",
             f"Dataset & {head1} \\\\", f" & {head2} \\\\", "\\midrule"]
    for t in tags:
        cells = []
        for m, a in cols:
            r = ent.get((m, t))
            cells.append("--" if r is None or r.get("status") != "complete"
                         else r.get(f"{a}_entrance", "--"))
        lines.append(f"{registry.display_name(t)} & " + " & ".join(cells) + " \\\\")
    lines.append("\\midrule")
    pre = []
    for m, a in cols:
        done = [r for r in T["entrances"] if r["model"] == m and r["status"] == "complete"]
        pre.append(f"{sum(bool(r.get(f'{a}_pre_final')) for r in done)}/{len(done)}")
    lines += ["Pre-final & " + " & ".join(pre) + " \\\\", "\\bottomrule", "\\end{tabular}"]
    written.append(_write(tex_dir / "tab_pooled_entrances.tex", lines))

    # 2. final-block test loss: native absolute, other arms as % vs native (* = CI excludes 0)
    lines = ["% Phase 2d: final-block test loss. Native: absolute Q=9 mean pinball (model's own "
             "normalized space). Other readouts: % change vs native; * = 95% cluster CI excludes 0"
             " (generated; do not edit)",
             f"\\begin{{tabular}}{{l{'c' * len(cols)}}}", "\\toprule",
             f"Dataset & {head1} \\\\", f" & {head2} \\\\", "\\midrule"]
    for t in tags:
        cells = []
        for m, a in cols:
            r = fin.get((m, t, a))
            if r is None:
                cells.append("--")
            elif a == p2d.ARM_NATIVE:
                cells.append(f"{r['final_test_loss']:.3f}")
            else:
                star = "$^*$" if r["loss_direction"] != "inconclusive" else ""
                cells.append(f"{_pct(r['loss_delta_rel_vs_native'])}{star}")
        lines.append(f"{registry.display_name(t)} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    written.append(_write(tex_dir / "tab_pooled_final.tex", lines))

    # 3. aggregate per model x arm
    agg = [r for r in T["aggregate"] if not str(r["arm"]).startswith("step:")]
    lines = ["% Phase 2d: per-architecture aggregate (generated; do not edit)",
             "\\begin{tabular}{llrcccc}", "\\toprule",
             "Model & Readout & Params & Pre-final & Same entr. & $\\le$1 block & "
             "Final loss vs native \\\\", "\\midrule"]
    for r in agg:
        if r["arm"] == p2d.ARM_NATIVE:
            same, w1, dl = "--", "--", "--"
        else:
            same = f"{r['entrance_same_as_native']}/{r['n_cells']}"
            w1 = f"{r['entrance_within_1_block']}/{r['n_cells']}"
            dl = (f"{_pct(r['median_final_loss_delta_rel_vs_native'])}\\% "
                  f"({r['n_final_worse_than_native_ci']}$\\uparrow$/"
                  f"{r['n_final_better_than_native_ci']}$\\downarrow$)")
        lines.append(f"{MODEL_NAME[r['model']]} & {ARM_SHORT[r['arm']]} & "
                     f"{r['params']:,} & {r['n_pre_final_entrance']}/{r['n_cells']} & {same} & "
                     f"{w1} & {dl} \\\\".replace(",", "{,}"))
    lines += ["\\bottomrule", "\\end{tabular}"]
    written.append(_write(tex_dir / "tab_pooled_aggregate.tex", lines))

    # 4. parameter counts for every configuration
    lines = ["% Phase 2d: readout configurations at Q=9, H=64 (generated; do not edit)",
             "\\begin{tabular}{llrrrr}", "\\toprule",
             "Model & Readout & In & Out & Params & $\\times$ native \\\\", "\\midrule"]
    for r in p2d.param_table():
        if r["model"] in models:
            lines.append(f"{MODEL_NAME[r['model']]} & {ARM_SHORT[r['arm']]} & "
                         f"{r['in_features']} & {r['out_features']} & "
                         f"{r['n_params']:,}".replace(",", "{,}") +
                         f" & {r['ratio_to_native']:.2f} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    written.append(_write(tex_dir / "tab_pooled_params.tex", lines))

    # 5. macros
    mac = ["% Phase 2d macros (generated; do not edit)"]
    for r in agg:
        base = f"\\newcommand{{\\pd{MACRO_MODEL[r['model']]}{MACRO_ARM[r['arm']]}"
        mac.append(f"{base}PreFinal}}{{{r['n_pre_final_entrance']}}}")
        mac.append(f"{base}Cells}}{{{r['n_cells']}}}")
        if r["arm"] != p2d.ARM_NATIVE:
            mac.append(f"{base}SameEntrance}}{{{r['entrance_same_as_native']}}}")
            mac.append(f"{base}WithinOne}}{{{r['entrance_within_1_block']}}}")
            mac.append(f"{base}FinalDelta}}{{{_pct(r['median_final_loss_delta_rel_vs_native'])}"
                       "\\%}")
            mac.append(f"{base}FinalWorse}}{{{r['n_final_worse_than_native_ci']}}}")
            mac.append(f"{base}FinalBetter}}{{{r['n_final_better_than_native_ci']}}}")
    written.append(_write(tex_dir / "pooled_macros.tex", mac))
    return written


def _write(path: Path, lines) -> str:
    path.write_text("\n".join(lines) + "\n")
    return str(path)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _style():
    try:
        from experiments.make_recoverability_figures import RC, TEXTWIDTH_IN
        return RC, TEXTWIDTH_IN
    except Exception:                                              # pragma: no cover
        return {"font.size": 8, "pdf.fonttype": 42}, 5.5


def _save(fig, out_dir: Path, stem: str) -> list[str]:
    import matplotlib.pyplot as plt
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{stem}.pdf", out_dir / f"{stem}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=300)
    plt.close(fig)
    return [str(p) for p in paths]


def _panel(ax, cell, model, title, show_x=True, compact=False):
    spec = phase1.model_spec(model)
    if cell is None:
        ax.text(0.5, 0.5, "PENDING", transform=ax.transAxes, ha="center", va="center",
                fontsize=8, color="#8A8A8A", fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, pad=2)
        return
    x = np.arange(spec.n_depth_points)
    for a in ARM_ORDER:
        cv = curves_of(cell, a)
        if cv is None:
            continue
        col, ls, mk = ARM_STYLE[a]
        ax.fill_between(x, cv["lo"], cv["hi"], color=col, alpha=0.12, lw=0)
        ax.plot(x, cv["test"], color=col, ls=ls, lw=1.1, marker=mk, ms=2.0, mew=0)
        if cv["entrance"] is not None:
            e = cv["entrance"]
            ax.plot([e], [cv["test"][e]], marker="o", ms=5.0, mfc="white", mec=col, mew=1.1,
                    ls="none", zorder=5)
    step = 4 if spec.n_depth_points > 14 else 3 if compact else 2
    ax.set_xticks(x[::step])
    ax.set_xticklabels(spec.depth_labels[::step] if show_x else [], fontsize=6)
    ax.grid(axis="y", color="#E6E6E6", lw=0.5)
    ax.set_axisbelow(True)
    for sp_ in ("top", "right"):
        ax.spines[sp_].set_visible(False)
    ax.set_title(title, pad=2)


def _legend_handles(models):
    from matplotlib.lines import Line2D
    arms = [a for a in ARM_ORDER if any(a in _arms(m) for m in models)]
    h = [Line2D([], [], color=ARM_STYLE[a][0], ls=ARM_STYLE[a][1], marker=ARM_STYLE[a][2],
                ms=3, lw=1.1, label=ARM_LABEL[a]) for a in arms]
    h.append(Line2D([], [], color="#555555", marker="o", ms=5, mfc="white", ls="none",
                    label="5% entrance (validation)"))
    return h


def figures(cells, models, tags, main_datasets, out_dir: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RC, W = _style()
    written = []
    main = [t for t in main_datasets if t in tags]
    with plt.rc_context(RC):
        if main:
            fig, axes = plt.subplots(len(models), len(main), figsize=(W, 1.75 * len(models) + 0.5),
                                     layout="constrained", squeeze=False)
            for i, m in enumerate(models):
                for j, t in enumerate(main):
                    _panel(axes[i, j], cells[(m, t)], m,
                           registry.display_name(t) if i == 0 else "", show_x=True)
                axes[i, 0].set_ylabel(f"{MODEL_NAME[m]}\ntest loss (Q=9)")
            axes[-1, len(main) // 2].set_xlabel("Block depth", fontsize=7.5)
            fig.legend(handles=_legend_handles(models), loc="outside lower center", ncol=3,
                       frameon=False, fontsize=7)
            written += _save(fig, out_dir, "pooled_curves_main")
        for m in models:
            ncol, nrow = 4, 4
            fig, axes = plt.subplots(nrow, ncol, figsize=(W, 5.8), layout="constrained")
            flat = axes.ravel()
            for i, t in enumerate(tags):
                _panel(flat[i], cells[(m, t)], m, registry.display_name(t),
                       show_x=i + ncol >= len(tags), compact=True)
            for ax in flat[len(tags):]:
                ax.axis("off")
            flat[len(tags)].legend(handles=_legend_handles([m]), loc="center", frameon=False,
                                   fontsize=6.5)
            fig.suptitle(f"{MODEL_NAME[m]}: test loss by depth, architecture-aware vs generic "
                         "readouts", fontsize=8)
            written += _save(fig, out_dir, f"pooled_curves_{m}")

        # final-block delta vs native, per arm, with CI
        fig, axes = plt.subplots(1, len(models), figsize=(W, 3.3), layout="constrained",
                                 sharey=True, squeeze=False)
        y = np.arange(len(tags))
        for ax, m in zip(axes[0], models):
            arms = [a for a in _arms(m) if a != p2d.ARM_NATIVE]
            off = np.linspace(-0.25, 0.25, len(arms)) if len(arms) > 1 else [0.0]
            for k, a in enumerate(arms):
                col, _ls, mk = ARM_STYLE[a]
                for i, t in enumerate(tags):
                    c = cells[(m, t)]
                    if c is None:
                        continue
                    r = _cmp(c, p2d.ARM_NATIVE, a, reference=True)
                    d, lo, hi = (_f(r["delta_rel"]) * 100, _f(r["delta_rel_ci_lo"]) * 100,
                                 _f(r["delta_rel_ci_hi"]) * 100)
                    ax.plot([lo, hi], [y[i] + off[k]] * 2, color=col, lw=0.9)
                    ax.plot([d], [y[i] + off[k]], marker=mk, color=col, ms=3.2, ls="none")
            ax.axvline(0, color="#555555", lw=0.7)
            ax.set_title(MODEL_NAME[m], pad=2)
            ax.grid(axis="x", color="#E6E6E6", lw=0.5)
            for sp_ in ("top", "right"):
                ax.spines[sp_].set_visible(False)
        axes[0, 0].set_yticks(y)
        axes[0, 0].set_yticklabels([registry.display_name(t) for t in tags], fontsize=6)
        axes[0, 0].invert_yaxis()
        axes[0, len(models) // 2].set_xlabel(
            "Final-block test loss vs architecture-aware readout (%)", fontsize=7.5)
        h = [hh for hh in _legend_handles(models)[1:-1]]
        fig.legend(handles=h, loc="outside lower center", ncol=3, frameon=False, fontsize=7)
        written += _save(fig, out_dir, "pooled_final_delta")

        # entrance relative depth: native (x) vs each arm (y)
        fig, axes = plt.subplots(1, len(models), figsize=(W, 2.2), layout="constrained",
                                 squeeze=False)
        for ax, m in zip(axes[0], models):
            others = [a for a in _arms(m) if a != p2d.ARM_NATIVE]
            for k, a in enumerate(others):
                # entrances coincide often; a fixed per-arm x offset keeps coincident points
                # visible (it is a display offset only, never part of a number)
                dx = (k - (len(others) - 1) / 2) * 0.02
                col, _ls, mk = ARM_STYLE[a]
                xs, ys = [], []
                for t in tags:
                    c = cells[(m, t)]
                    if c is None:
                        continue
                    pa = c["summary"]["per_arm"]
                    xs.append(pa[p2d.ARM_NATIVE]["headline_relative_depth"] + dx)
                    ys.append(pa[a]["headline_relative_depth"])
                ax.scatter(xs, ys, s=12, marker=mk, facecolor="none", edgecolor=col, lw=0.9)
            ax.plot([0, 1], [0, 1], color="#999999", lw=0.7, ls=(0, (3, 2)))
            ax.set_xlim(-0.04, 1.04)
            ax.set_ylim(-0.04, 1.04)
            ax.set_aspect("equal")
            ax.set_title(MODEL_NAME[m], pad=2)
            ax.set_xlabel("Native entrance (rel. depth)", fontsize=6.5)
        axes[0, 0].set_ylabel("Other readout entrance", fontsize=6.5)
        fig.legend(handles=_legend_handles(models)[1:-1], loc="outside lower center", ncol=3,
                   frameon=False, fontsize=7)
        written += _save(fig, out_dir, "pooled_entrances")
    return written


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--output-root", default=str(DEFAULT_OUT))
    ap.add_argument("--paper-dir", default=str(PAPER_DIR))
    ap.add_argument("--models", nargs="*", default=list(phase1.MODELS))
    ap.add_argument("--main-datasets", nargs=3, default=None,
                    help="default: the main recoverability figure's three datasets")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)
    root = Path(args.output_root)
    tags = list(registry.roster("paper14"))
    cells = load_cells(root, args.models, tags)
    T = build_tables(cells, args.models, tags)
    written = write_csvs(T, root / "combined")
    paper = Path(args.paper_dir)
    written += latex_tables(T, args.models, tags, paper / "tables")
    if not args.no_figures:
        try:
            from experiments.make_recoverability_figures import MAIN_DATASETS
        except Exception:                                          # pragma: no cover
            MAIN_DATASETS = ("monash_electricity_hourly", "uber_tlc_hourly",
                             "monash_london_smart_meters")
        written += figures(cells, args.models, tags, args.main_datasets or MAIN_DATASETS, paper)
    stats = {"schema": "phase2d_stats/v1", "protocol": p2d.PROTOCOL,
             "n_complete": sum(c is not None for c in cells.values()),
             "n_planned": len(cells), "pending": T["pending"], "aggregate": T["aggregate"],
             "param_table": p2d.param_table(), "decomposition_steps": STEP_LABEL,
             "caveat": ("Entrances are relative to each readout's OWN final block, so an earlier "
                        "entrance can reflect a weaker final block: read entrances together with "
                        "the final-block losses. Losses are comparable across readouts within a "
                        "model (same windows, same normalized targets) but NOT across models."),
             "written": written}
    phase1.atomic_write_json(root / "combined" / "stats.json", stats)
    print(f"Phase 2d tables: {stats['n_complete']}/{stats['n_planned']} cells complete; "
          f"{len(T['pending'])} pending")
    for r in T["aggregate"]:
        if str(r["arm"]).startswith("step:"):
            print(f"  {MODEL_NAME[r['model']]:<10s} {r['arm']:<26s} median final dLoss "
                  f"{_pct(r['median_final_loss_delta_rel_step'])}%  worse/better/inconcl "
                  f"{r['n_step_worse_ci']}/{r['n_step_better_ci']}/{r['n_step_inconclusive']}")
        else:
            extra = ("" if r["arm"] == p2d.ARM_NATIVE else
                     f"  same entrance {r['entrance_same_as_native']}/{r['n_cells']}  "
                     f"final dLoss {_pct(r['median_final_loss_delta_rel_vs_native'])}%")
            print(f"  {MODEL_NAME[r['model']]:<10s} {r['arm']:<26s} pre-final "
                  f"{r['n_pre_final_entrance']}/{r['n_cells']}{extra}")
    print(f"  wrote {len(written)} files under {root / 'combined'} and {paper}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
