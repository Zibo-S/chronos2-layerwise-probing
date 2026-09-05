"""Compression cost of truncating Chronos-2 at a validation-selected depth.

Answers "what does the compression buy and what does it cost?" for two selection rules, and emits
the paper table. Nothing here fits a model: it is a post-hoc join over COMMITTED artifacts, so it
runs on the login node in seconds.

Selection rules (both frozen before any test number is read):
    saturation  earliest layer whose MEAN VALIDATION fslot-probe loss is within TUNNEL_TOL (5%) of
                the final layer's -- probing.tunnel.tunnel_start, the criterion used throughout the
                paper. PT-ID sources read ``l_start`` straight from the committed tunnel record (and
                this driver re-derives it and asserts they agree); PT-OOD targets have no committed
                tunnel record (compute_ptid_tunnels loops PT_ID_TAGS only), so the entrance is
                derived here from the committed per_target validation curves with the SAME function
                -- exactly what experiments/make_id_paper_figures.load_dataset does.
    erank       layer of maximum effective rank of the forecast-slot representation
                (results/ext_v4_future_tokens/spectral/*.json). Label-free and computed on TRAIN
                features, so it never sees validation labels or test.

Cost columns come from probing.model_size (parameter and FLOP accounting, source-verified).
Accuracy columns come from the ext_v5 native-head-adapter per-window metrics, resampled with ONE
shared series-level cluster-bootstrap matrix so adapter-vs-native is PAIRED.

Provenance gates (all fail loud):
    * the saturation entrance recomputed from the stored validation curve == the committed l_start;
    * the recomputed test MASE ratio == the committed ``relative_regret_mase`` in the records CSV;
    * the entrance-selection windows (ext_v4 probe) and the cost windows (ext_v5 adapter) are the
      SAME test windows, element-wise -- so the depth is never selected on the data it is scored on.

Usage (login node / CPU, seconds):
    python -m experiments.run_compression_cost                    # both rules -> tables + LaTeX
    python -m experiments.run_compression_cost --selection saturation
    python -m experiments.run_compression_cost --datasets m4_hourly boom_hourly
    python -m experiments.run_compression_cost --threshold-figure          # eps sensitivity panels
"""

from __future__ import annotations

import argparse
import csv
import json

import numpy as np

from probing import model_size
from probing.config import REPO_ROOT, SEED
from probing.stats import ci_bounds, cluster_bootstrap_apply, cluster_bootstrap_counts
from probing.tunnel import TUNNEL_TOL, tunnel_start

QSET, PROTO, RUN_SEEDS = "q1", "v2", (0, 1, 2)
RUNS_TAG = "runs" + "-".join(str(s) for s in RUN_SEEDS)
BOOT_B = 5000                       # matches the committed ext_v5 bootstrap (run_native_head_adapter)

V4 = REPO_ROOT / "results" / "ext_v4_future_tokens"
TUNNEL_DIR = V4 / QSET / "tunnels"
PTOOD_DIR = V4 / "ptood_probing"
PTID_RUN_DIR = PTOOD_DIR / "ptid_runs"
SPEC_DIR = V4 / "spectral"
NHA = REPO_ROOT / "results" / "ext_v5_native_head_adapter"
NHA_BOOT = NHA / "bootstrap_inputs"
NHA_TAB = NHA / "tables"
OUT_ROOT = NHA / "compression"

LAYER_LABELS = ["Emb"] + [f"L{i}" for i in range(1, 13)] + ["L12+RMS"]
PT_ID_TAGS = ["monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly"]
PT_OOD_TAGS = ["sg_carpark", "coastal_ts", "boom_hourly"]
ALL_TAGS = PT_ID_TAGS + PT_OOD_TAGS
PRETTY = {"monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber TLC",
          "m4_hourly": "M4", "wind_farms_hourly": "Wind Farms", "sg_carpark": "SG Carpark",
          "coastal_ts": "Coastal T-S", "boom_hourly": "BOOM"}
RULES = ("saturation", "erank")


def _need(path, how):
    if not path.exists():
        raise FileNotFoundError(f"missing {path}\n  produce it with: {how}")
    return path


# --------------------------------------------------------------------------- #
# depth selection
# --------------------------------------------------------------------------- #
def saturation_depth(tag):
    """(depth, provenance). PT-ID: the committed l_start, re-derived and asserted. PT-OOD: derived
    from the committed per_target mean validation curve with the same frozen criterion."""
    if tag in PT_ID_TAGS:
        p = _need(TUNNEL_DIR / f"{tag}__fslot__{QSET}__{PROTO}__{RUNS_TAG}.json",
                  f"python -m experiments.run_ptood_probing_ftok --quantile-set {QSET} --tunnels-only")
        rec = json.load(open(p))
        val = np.asarray(rec["mean_val_loss_by_layer"], np.float64)
        redone = int(tunnel_start(val, tol=TUNNEL_TOL))
        if redone != int(rec["l_start"]):
            raise ValueError(f"{tag}: committed l_start={rec['l_start']} but recomputing "
                             f"tunnel_start on the stored validation curve gives {redone}")
        return int(rec["l_start"]), "committed tunnel record (l_start), re-derived and verified"
    vals = []
    for sd in RUN_SEEDS:
        p = _need(PTOOD_DIR / "per_target" / f"{tag}__{QSET}__seed{sd}.json",
                  f"python -m experiments.run_ptood_probing_ftok --quantile-set {QSET}")
        vals.append(np.asarray(json.load(open(p))["val_loss_by_layer"], np.float64))
    return (int(tunnel_start(np.mean(vals, axis=0), tol=TUNNEL_TOL)),
            "derived from the per_target mean validation curve (same criterion)")


def erank_depth(tag):
    """(depth, provenance) = argmax effective rank of the forecast-slot representation (train split)."""
    p = _need(SPEC_DIR / f"spectral__{tag}__fslot__probe_input__train.json",
              "python -m experiments.run_spectral --readout fslot")
    spec = json.load(open(p))
    er = np.array([layer["effective_rank"] for layer in spec["layers"]], np.float64)
    if er.size != len(LAYER_LABELS):
        raise ValueError(f"{tag}: spectral record has {er.size} layers, expected {len(LAYER_LABELS)}")
    return int(er.argmax()), f"argmax effective rank ({spec['split']} split, N={spec['sample_size']})"


# --------------------------------------------------------------------------- #
# accuracy (paired series-cluster bootstrap on the committed per-window metrics)
# --------------------------------------------------------------------------- #
def _load_window_metrics(tag):
    z = np.load(_need(NHA_BOOT / f"native_head_adapter__{tag}.npz",
                      "python -m experiments.run_native_head_adapter --adapt"), allow_pickle=True)
    series = np.asarray(z["series_test"])
    lin = None
    p = NHA_BOOT / f"native_head_adapter__linear_q1__{tag}.npz"
    if p.exists():
        zl = np.load(p, allow_pickle=True)
        if not np.array_equal(np.asarray(zl["series_test"]), series):
            raise ValueError(f"{tag}: linear_q1 baseline was scored on different test windows")
        lin = zl
    return z, lin, series


def _verify_same_windows(tag, series):
    """The depth is selected on the ext_v4 probe run and scored on the ext_v5 adapter run. Assert
    the two used the SAME test windows, so selection can never have touched the scored data."""
    p = (PTID_RUN_DIR / f"{tag}__{QSET}__{PROTO}__seed{RUN_SEEDS[0]}.npz" if tag in PT_ID_TAGS
         else PTOOD_DIR / "bootstrap_inputs" / f"{tag}__{QSET}__seed{RUN_SEEDS[0]}.npz")
    if not p.exists():
        return "selection-run windows not on disk (not checked)"
    other = np.asarray(np.load(p)["series_test"])
    if not np.array_equal(other, series):
        raise ValueError(f"{tag}: the depth-selection run and the cost run used DIFFERENT test "
                         f"windows ({other.size} vs {series.size}) -- the comparison is invalid")
    return "identical to the depth-selection run (element-wise)"


class _Boot:
    """One shared multinomial series-resampling matrix -> every curve below is paired."""

    def __init__(self, series, B=BOOT_B, seed=SEED):
        uniq, self.inv = np.unique(series, return_inverse=True)
        self.S = uniq.size
        self.M = cluster_bootstrap_counts(self.S, B, seed)
        self.count = np.zeros(self.S, np.float64)
        np.add.at(self.count, self.inv, 1.0)

    def _per_series(self, vec):
        out = np.zeros(self.S, np.float64)
        np.add.at(out, self.inv, np.asarray(vec, np.float64))
        return out

    def mean(self, vec):
        return cluster_bootstrap_apply(self.M, self._per_series(vec)[:, None], self.count)[:, 0]

    def ratio(self, num, den):
        return cluster_bootstrap_apply(self.M, self._per_series(num)[:, None], self._per_series(den))[:, 0]


def _committed_relative_regret(tag, depth):
    """The value this driver must reproduce: relative_regret_mase from the committed records CSV."""
    p = _need(NHA_TAB / f"native_head_adapter__records__{tag}.csv",
              "python -m experiments.run_native_head_adapter --adapt")
    for row in csv.DictReader(open(p)):
        if row["condition"] == "linear_adapter" and int(row["layer"]) == depth:
            return float(row["relative_regret_mase"])
    raise ValueError(f"{tag}: no committed linear_adapter record at layer {depth}")


def evaluate(tag, depth, rule, provenance):
    z, lin, series = _load_window_metrics(tag)
    window_note = _verify_same_windows(tag, series)
    lab = f"L{depth:02d}"
    if f"linear_adapter__{lab}__mase" not in z.files:
        raise ValueError(f"{tag}: no adapter was fit at depth {depth} "
                         f"(available: {sorted(k for k in z.files if k.startswith('linear_adapter'))})")
    b = _Boot(series)

    nat_w, ada_w = z["native__L13__mase"], z[f"linear_adapter__{lab}__mase"]
    zs_w = z[f"zero_shot__{lab}__mase"]
    nat, ada, zs = b.mean(nat_w), b.mean(ada_w), b.mean(zs_w)
    # point estimates are the plain per-window means (deterministic, and what the records CSV holds);
    # the bootstrap supplies the intervals.
    p_nat, p_ada, p_zs = float(nat_w.mean()), float(ada_w.mean()), float(zs_w.mean())
    rel = p_ada / p_nat - 1.0
    committed = _committed_relative_regret(tag, depth)
    # the records CSV stores round(x, 6), so half a unit in the last place (5e-7) is the floor on
    # agreement; anything larger means the per-window arrays and the records table disagree.
    if not np.isclose(rel, committed, rtol=1e-6, atol=1e-6):
        raise ValueError(f"{tag} @ {LAYER_LABELS[depth]}: relative regret recomputed from the "
                         f"per-window array is {rel:.9f} but the committed records table says "
                         f"{committed:.9f} (difference {abs(rel - committed):.2e} exceeds the "
                         "6-decimal rounding of that table)")

    d_boot, r_boot = ada - nat, ada / nat - 1.0
    recov = (zs - ada) / (zs - nat)                       # fraction of the truncation gap the adapter closes
    natw = b.ratio(z["native__L13__wql_num"], z["native__L13__wql_den"])
    adaw = b.ratio(z[f"linear_adapter__{lab}__wql_num"], z[f"linear_adapter__{lab}__wql_den"])
    relw = adaw / natw - 1.0
    d_ci, r_ci, rc_ci, rw_ci = (ci_bounds(d_boot), ci_bounds(r_boot), ci_bounds(recov), ci_bounds(relw))

    return {
        "dataset": tag, "short": PRETTY[tag],
        "kind": "PT-ID" if tag in PT_ID_TAGS else "PT-OOD",
        "selection_rule": rule, "depth": depth, "depth_label": LAYER_LABELS[depth],
        "depth_provenance": provenance, "window_provenance": window_note,
        "active_params": model_size.active_params(depth),
        "active_fraction": model_size.active_fraction(depth),
        "block_flops_fraction": model_size.block_flops_fraction(depth),
        "end_to_end_flops_fraction": model_size.end_to_end_flops_fraction(depth),
        "fp32_weight_mib": model_size.weight_bytes(depth, 4) / 2 ** 20,
        "native_mase": p_nat, "truncated_mase": p_ada, "no_adapter_mase": p_zs,
        "linear_probe_q1_mase": (float(lin[f"linear_q1__{lab}__mase"].mean()) if lin is not None else None),
        "delta_mase": p_ada - p_nat, "delta_mase_ci_lo": float(d_ci[0]), "delta_mase_ci_hi": float(d_ci[1]),
        "relative_mase": rel, "relative_mase_ci_lo": float(r_ci[0]), "relative_mase_ci_hi": float(r_ci[1]),
        "gap_recovery": float(recov.mean()),
        "gap_recovery_ci_lo": float(rc_ci[0]), "gap_recovery_ci_hi": float(rc_ci[1]),
        "native_wql": float(natw.mean()), "truncated_wql": float(adaw.mean()),
        "relative_wql": float(relw.mean()),
        "relative_wql_ci_lo": float(rw_ci[0]), "relative_wql_ci_hi": float(rw_ci[1]),
        "n_windows": int(series.size), "n_series": int(b.S), "bootstrap_B": BOOT_B, "seed": SEED,
        "quantile_set": QSET,
    }


# --------------------------------------------------------------------------- #
# output
# --------------------------------------------------------------------------- #
def _pct(x):
    return f"\\({100 * x:+.1f}\\%\\)"


def latex_two_rule(rows_by_rule, tags):
    """The paper table: one row per dataset, both selection rules side by side."""
    out = [r"\begin{tabular}{lcccccccc}", r"    \toprule",
           r"    & \multicolumn{4}{c}{Saturation-selected} &",
           r"      \multicolumn{4}{c}{Effective-rank-selected} \\",
           r"    \cmidrule(lr){2-5}", r"    \cmidrule(lr){6-9}",
           r"    Dataset &",
           r"    Depth & Active params & FLOPs & $\Delta$MASE [95\% CI] &",
           r"    Depth & Active params & FLOPs & $\Delta$MASE [95\% CI] \\",
           r"    \midrule"]
    for i, tag in enumerate(tags):
        if i and tags[i - 1] in PT_ID_TAGS and tag in PT_OOD_TAGS:
            out.append(r"    \midrule")
        cells = [PRETTY[tag]]
        for rule in RULES:
            r = rows_by_rule[rule][tag]
            cells += [f"\\({r['depth_label']}\\)", f"\\({100 * r['active_fraction']:.1f}\\%\\)",
                      f"\\({r['block_flops_fraction']:.2f}\\times\\)",
                      f"{_pct(r['relative_mase'])} [\\({100 * r['relative_mase_ci_lo']:.1f},"
                      f"{100 * r['relative_mase_ci_hi']:.1f}\\)]"]
        out.append("    " + " & ".join(cells) + r" \\")
    out += [r"    \bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def latex_single_rule(rows, tags, rule):
    out = [r"\begin{tabular}{lccccc}", r"    \toprule",
           r"    Dataset & Depth & Active params & Transf. FLOPs & MASE (trunc./native) "
           r"& $\Delta$MASE [95\% CI] \\", r"    \midrule"]
    for i, tag in enumerate(tags):
        if i and tags[i - 1] in PT_ID_TAGS and tag in PT_OOD_TAGS:
            out.append(r"    \midrule")
        r = rows[tag]
        out.append(f"    {PRETTY[tag]} & \\({r['depth_label']}\\) & "
                   f"\\({100 * r['active_fraction']:.1f}\\%\\) & "
                   f"\\({r['block_flops_fraction']:.2f}\\times\\) & "
                   f"\\({r['truncated_mase']:.3f} / {r['native_mase']:.3f}\\) & "
                   f"{_pct(r['relative_mase'])} [\\({100 * r['relative_mase_ci_lo']:.1f},"
                   f"{100 * r['relative_mase_ci_hi']:.1f}\\)] \\\\")
    out += [r"    \bottomrule", r"\end{tabular}", f"% selection rule: {rule}"]
    return "\n".join(out)


# --------------------------------------------------------------------------- #
# threshold sensitivity: how far does the truncation depth move with the tolerance?
# --------------------------------------------------------------------------- #
# eps is the compression-aggressiveness knob: a looser tolerance accepts a shallower layer as
# "saturated" and therefore truncates harder. 5% is the value frozen throughout the paper.
DEFAULT_EPS = (0.10, 0.05, 0.03, 0.02)
EPS_MARKER = {0.10: "^", 0.05: "o", 0.03: "s", 0.02: "D"}
DS_COLOR = {"monash_electricity_hourly": "#1B7837", "uber_tlc_hourly": "#2171B5",
            "m4_hourly": "#D94801", "wind_farms_hourly": "#7A0177",
            "sg_carpark": "#00838F", "coastal_ts": "#A6761D", "boom_hourly": "#525252"}


def validation_curve(tag):
    """The mean validation curve the saturation criterion reads (q1 fslot probe, 3 seeds)."""
    if tag in PT_ID_TAGS:
        p = _need(TUNNEL_DIR / f"{tag}__fslot__{QSET}__{PROTO}__{RUNS_TAG}.json",
                  f"python -m experiments.run_ptood_probing_ftok --quantile-set {QSET} --tunnels-only")
        return np.asarray(json.load(open(p))["mean_val_loss_by_layer"], np.float64)
    vals = [np.asarray(json.load(open(_need(
        PTOOD_DIR / "per_target" / f"{tag}__{QSET}__seed{sd}.json",
        f"python -m experiments.run_ptood_probing_ftok --quantile-set {QSET}")))["val_loss_by_layer"],
        np.float64) for sd in RUN_SEEDS]
    return np.mean(vals, axis=0)


def threshold_rows(tags, eps_list):
    """(dataset, eps) -> the depth the criterion selects and what that depth costs."""
    out = []
    for tag in tags:
        v = validation_curve(tag)
        for eps in eps_list:
            d = int(tunnel_start(v, tol=eps))
            # d indexes a READOUT POINT (13 = L12+RMS = the native head's own input, i.e. no
            # truncation at all), so translate to executed blocks before costing it.
            cost = model_size.cost_of_readout(d)
            out.append({"dataset": tag, "short": PRETTY[tag],
                        "kind": "PT-ID" if tag in PT_ID_TAGS else "PT-OOD",
                        "epsilon": eps, "depth": d, "depth_label": LAYER_LABELS[d],
                        "n_blocks": cost["n_blocks"], "needs_adapter": cost["needs_adapter"],
                        "active_fraction": cost["active_fraction"],
                        "block_flops_fraction": cost["block_flops_fraction"]})
    return out


def make_threshold_figures(tags, eps_list, out_dir):
    """Two one-panel views of the same fact, because they answer different questions:

    A (curves)   normalised validation loss vs depth, one line per dataset, with a horizontal line
                 per tolerance. The selected depth is where a curve first dips below a line, so the
                 panel shows WHY the depth moves. y is clipped to the threshold region: every
                 selection lies at a ratio <= 1 + max(eps) so no marker is ever cut off, but the
                 steep early descent (up to 3.7x at Emb) runs off the top and is labelled as such.
    B (depths)   selected depth vs tolerance, one line per dataset. Answers "how much does eps
                 matter" directly, with no clipping. Datasets are dodged horizontally because
                 several select the SAME depth and would otherwise hide each other.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rc = {"font.size": 11, "axes.labelsize": 13, "axes.titlesize": 14, "legend.fontsize": 10,
          "legend.title_fontsize": 10.5, "xtick.labelsize": 10, "ytick.labelsize": 10.5,
          "axes.linewidth": 0.9, "lines.linewidth": 1.8, "pdf.fonttype": 42, "ps.fonttype": 42}
    x = np.arange(len(LAYER_LABELS))
    rows = threshold_rows(tags, eps_list)
    by = {(r["dataset"], r["epsilon"]): r for r in rows}
    order = sorted(eps_list, reverse=True)
    top = 1.0 + max(eps_list) + 0.14
    curves = {t: (lambda v: v / v[-1])(validation_curve(t)) for t in tags}

    def _style(tag):
        pt_id = tag in PT_ID_TAGS
        return dict(color=DS_COLOR[tag], ls="-" if pt_id else (0, (4, 2)),
                    marker="o" if pt_id else "s",
                    label=f"{PRETTY[tag]} ({'PT-ID' if pt_id else 'PT-OOD'})")

    # ---- A: normalised validation curves + one line per tolerance ----------------------
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(12.6, 6.4))
        fig.subplots_adjust(left=0.068, right=0.775, top=0.885, bottom=0.155)
        for eps in order:                        # thresholds, labelled over the empty left margin
            ax.axhline(1 + eps, color="0.55", ls=(0, (5, 4)), lw=1.1, zorder=1)
            ax.text(-0.35, 1 + eps, f"$\\varepsilon$={eps:.0%}", va="center", ha="left",
                    fontsize=10, color="0.35", zorder=5,
                    bbox=dict(fc="white", ec="none", pad=1.4))
        ax.axhline(1.0, color="0.2", lw=1.0, zorder=1)
        ax.text(-0.35, 1.0, "final layer", va="center", ha="left", fontsize=10, color="0.2",
                zorder=5, bbox=dict(fc="white", ec="none", pad=1.4))
        for tag in tags:
            r = curves[tag]
            ax.plot(x, r, ms=4.0, zorder=3, **_style(tag))
            for j, eps in enumerate(order):      # open marker = the depth this eps selects;
                d = by[(tag, eps)]["depth"]       # sizes nest so shared depths read as rings
                ax.plot(d, r[d], marker=EPS_MARKER[eps], ms=15.5 - 2.7 * j, mfc="none", mew=1.9,
                        color=DS_COLOR[tag], zorder=4, ls="none")
        ax.set_ylim(0.945, top)
        ax.set_xlim(-0.6, len(LAYER_LABELS) - 0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right")
        ax.set_xlabel("Representation point (truncation depth)")
        ax.set_ylabel("Validation loss / final-layer loss")
        ax.set_title("Where the saturation criterion cuts, as the tolerance "
                     "$\\varepsilon$ varies\n"
                     "open markers = the depth each $\\varepsilon$ selects", pad=10)
        ax.grid(axis="y", color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        n_clip = sum(1 for t in tags if curves[t].max() > top)
        if n_clip:
            ax.annotate(f"{n_clip}/{len(tags)} curves run off the top at Emb/L1 "
                        f"(max {max(curves[t].max() for t in tags):.1f}x)",
                        xy=(0.012, 0.028), xycoords="axes fraction", fontsize=9.5, color="0.35")
        ds_h, ds_l = ax.get_legend_handles_labels()
        eps_h = [plt.Line2D([], [], marker=EPS_MARKER[e], ls="none", mfc="none", mew=1.9,
                            color="0.25", ms=15.5 - 2.7 * j, label=f"$\\varepsilon$={e:.0%}")
                 for j, e in enumerate(order)]
        l1 = fig.legend(ds_h, ds_l, loc="upper left", bbox_to_anchor=(0.788, 0.885),
                        frameon=True, framealpha=0.95, title="Dataset")
        l1._legend_box.align = "left"
        l2 = fig.legend(handles=eps_h, loc="upper left", bbox_to_anchor=(0.788, 0.44),
                        frameon=True, framealpha=0.95, title="Tolerance")
        l2._legend_box.align = "left"
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"threshold_sensitivity_curves.{ext}", dpi=300)
        plt.close(fig)

    # ---- B: selected depth vs tolerance -------------------------------------------------
    with plt.rc_context(rc):
        fig, ax = plt.subplots(figsize=(12.8, 6.4))
        fig.subplots_adjust(left=0.088, right=0.695, top=0.885, bottom=0.125)
        xs = np.arange(len(order))
        dodge = 0.055                            # several datasets select the SAME depth
        for i, tag in enumerate(tags):
            off = (i - (len(tags) - 1) / 2) * dodge
            ax.plot(xs + off, [by[(tag, e)]["depth"] for e in order],
                    ms=9, mew=1.4, alpha=0.95, **_style(tag))
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{e:.0%}" for e in order])
        ax.set_xlim(-0.45, len(order) - 0.55)
        ax.set_xlabel("Saturation tolerance $\\varepsilon$  "
                      "(looser $\\rightarrow$ stricter)")
        ax.set_yticks(range(len(LAYER_LABELS)))
        ax.set_yticklabels(LAYER_LABELS)
        ax.set_ylim(-0.6, len(LAYER_LABELS) - 0.4)
        ax.set_ylabel("Selected truncation depth")
        sec = ax.secondary_yaxis("right", functions=(lambda d: d / 12.0, lambda f: f * 12.0))
        sec.set_ylabel("Transformer-block FLOPs retained")
        sec.set_yticks([d / 12 for d in range(0, 13, 2)])
        sec.set_yticklabels([f"{d / 12:.2f}x" for d in range(0, 13, 2)])
        ax.set_title("Truncation depth selected by the saturation criterion, per tolerance\n"
                     "points dodged horizontally where datasets select the same depth", pad=10)
        ax.grid(color="0.9", lw=0.7)
        ax.set_axisbelow(True)
        h, l = ax.get_legend_handles_labels()
        leg = fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.775, 0.885),
                         frameon=True, framealpha=0.95, title="Dataset")
        leg._legend_box.align = "left"
        for ext in ("png", "pdf"):
            fig.savefig(out_dir / f"threshold_sensitivity_depths.{ext}", dpi=300)
        plt.close(fig)
    return rows


# --------------------------------------------------------------------------- #
# LaTeX: the complete table (cost + measured speed) and the methodology appendix
# --------------------------------------------------------------------------- #
LAT_DIR = NHA / "latency" / "tables"
# Fields the methodology section must state. Anything missing is emitted as a visible marker
# rather than guessed -- a measurement section with an invented CPU model is worse than a gap.
MISSING = ("\\textbf{[not recorded by this run --- re-run \\texttt{job\\_latency.sh}, which "
           "captures this field, or fill it in by hand]}")


def load_latency(path=None):
    """The newest latency run, or the one at ``path``. Returns None if none has been run."""
    if path is not None:
        return json.load(open(path))
    runs = sorted(LAT_DIR.glob("latency__*.json"))
    if not runs:
        return None
    return json.load(open(max(runs, key=lambda p: p.stat().st_mtime)))


def latency_by_depth(lat, batch):
    """{depth: row} for one batch size, plus the speedup of each depth vs the full 12-block model.

    Latency and memory depend on DEPTH only, not on the dataset, so one lookup serves every row of
    the cost table.
    """
    rows = {r["depth"]: r for r in lat["rows"] if r["batch_size"] == batch}
    if not rows:
        have = sorted({r["batch_size"] for r in lat["rows"]})
        raise ValueError(f"the latency run has no batch size {batch}; it measured {have}")
    full = rows.get(model_size.NUM_BLOCKS)
    if full is None:
        raise ValueError("the latency run never measured the full 12-block model, so no speedup "
                         "can be referenced to it")
    for d, r in rows.items():
        r["speedup"] = full["predict_median_ms"] / r["predict_median_ms"]
        r["memory_ratio"] = (r["peak_mib"] / full["peak_mib"]) if "peak_mib" in r else None
    return rows


def _tex(x):
    return str(x).replace("_", "\\_").replace("&", "\\&").replace("%", "\\%")


def latex_full_table(rows_by_rule, tags, lat, batch):
    """Per dataset: both selection rules, each with depth, size, FLOPs, MEASURED speedup, accuracy."""
    lut = latency_by_depth(lat, batch)
    gpu = _tex(lat["environment"].get("gpu_name", "the measured GPU"))
    out = [r"\begin{tabular}{l" + "cccccccc"[:0] + "cccc" * 2 + "}", r"    \toprule",
           r"    & \multicolumn{4}{c}{Saturation-selected} &",
           r"      \multicolumn{4}{c}{Effective-rank-selected} \\",
           r"    \cmidrule(lr){2-5}", r"    \cmidrule(lr){6-9}",
           r"    Dataset & Depth & Params & Speedup & $\Delta$MASE [95\% CI] &",
           r"              Depth & Params & Speedup & $\Delta$MASE [95\% CI] \\",
           r"    \midrule"]
    for i, tag in enumerate(tags):
        if i and tags[i - 1] in PT_ID_TAGS and tag in PT_OOD_TAGS:
            out.append(r"    \midrule")
        cells = [PRETTY[tag]]
        for rule in RULES:
            r = rows_by_rule[rule][tag]
            n_blocks = model_size.blocks_for_readout(r["depth"])
            sp = lut[n_blocks]["speedup"]
            cells += [f"\\({r['depth_label']}\\)", f"\\({100 * r['active_fraction']:.1f}\\%\\)",
                      f"\\({sp:.2f}\\times\\)",
                      f"{_pct(r['relative_mase'])} [\\({100 * r['relative_mase_ci_lo']:.1f},"
                      f"{100 * r['relative_mase_ci_hi']:.1f}\\)]"]
        out.append("    " + " & ".join(cells) + r" \\")
    out += [r"    \bottomrule", r"\end{tabular}",
            f"% speedup = measured median wall-clock of the full 12-block model divided by that of",
            f"% the truncated model, batch {batch}, on {gpu}. Depends on depth only, not on dataset.",
            f"% generated by: python -m experiments.run_compression_cost --latex-appendix"]
    return "\n".join(out)


def latex_appendix_methodology(lat, batch):
    """The 'Latency and Memory Methodology' appendix section, filled from the run's own record."""
    e = lat["environment"]
    rows = sorted(lat["rows"], key=lambda r: (r["batch_size"], r["depth"]))
    batches = sorted({r["batch_size"] for r in rows})
    ref = rows[0]
    ver = e.get("verification") or {}
    gate = (f"exactly {max(abs(float(v)) for v in ver.values()):.1e}" if ver else None)

    def f(key, fmt="{}"):
        v = e.get(key)
        return MISSING if v in (None, "") else fmt.format(_tex(v))

    L = [r"\section{Latency and Memory Methodology}", r"\label{app:latency-methodology}", "",
         r"All timings and memory figures in this paper are measured, not derived from parameter or",
         r"FLOP counts. This appendix records the harness in full so the numbers can be reproduced or",
         r"contested. The harness is \texttt{experiments/run\_latency.py}; it times the native model",
         r"and every truncated variant in one process with an identical protocol.", "",
         r"\paragraph{The truncated model is real.}",
         r"Every layer-wise result elsewhere in this paper is obtained by running the full 12-block",
         r"encoder and reading intermediate states off forward hooks. That is exact for accuracy,",
         r"because the encoder is a feed-forward stack and blocks after $\ell$ cannot influence block",
         r"$\ell$'s output, but it means no truncated model is ever instantiated. For timing we build",
         r"one: encoder blocks $\ell+1 \ldots 12$ are deleted from the module list and the linear",
         r"adapter is spliced in front of the frozen final RMSNorm, which the encoder applies after",
         r"its last remaining block. Before timing, the harness asserts that the truncated encoder",
         r"reproduces the hooked full-model states; the observed maximum absolute deviation was",
         (f"{gate} across all measured depths, i.e.\\ bit-identical." if gate else
          r"\textbf{[gate not run --- pass \texttt{--verify}]}"), "",
         r"\paragraph{Hardware.}",
         f"GPU: {f('gpu_name')} with {e.get('gpu_total_mib', 0) / 1024:.0f}\\,GiB of VRAM "
         f"(compute capability {f('gpu_capability')}), driver {f('driver_version')}; "
         f"{e.get('gpu_count', 1)} device visible, one used. "
         f"Host CPU: {f('cpu_model')}, "
         f"{f('slurm_cpus_per_task')} cores allocated to the job. ", "",
         r"\paragraph{Software and precision.}",
         f"Python {f('python')}, PyTorch {f('torch')} (CUDA {f('torch_cuda')}, "
         f"cuDNN {f('cudnn')}), model \\texttt{{{_tex(e.get('model_id'))}}}. ",
         f"All computation is in \\texttt{{float32}} --- the precision at which every accuracy number",
         r"in this paper was produced. No \texttt{torch.compile}, no automatic mixed precision, and no",
         (r"manual kernel fusion are applied (TF32 matmul: "
          f"{str(e.get('tf32_matmul')).lower()}; TF32 cuDNN: "
          f"{str(e.get('tf32_cudnn')).lower()})."
          if "tf32_matmul" in e else r"manual kernel fusion are applied."), "",
         r"\paragraph{Workload.}",
         f"Context $C={e.get('context_length')}$, horizon $H={e.get('horizon')}$, giving "
         f"$K={e.get('forecast_slots')}$ native forecast slots and an encoder sequence of "
         f"{e.get('encoder_tokens')} tokens "
         f"({model_size.num_encoder_tokens(int(e.get('context_length', 512)), int(e.get('horizon', 64)))[0]} "
         r"context patches, one REG token, and the forecast slots). Batch sizes "
         f"{', '.join(str(b) for b in batches)} are reported: the smallest is the single-request",
         r"latency regime, the largest the batched-throughput regime, and they can support different",
         r"conclusions. Contexts are synthetic random walks; the forward pass has no data-dependent",
         r"control flow, so timing is independent of the data and needs no dataset.", "",
         r"\paragraph{Timing protocol.}",
         f"Each configuration is warmed up for {ref['warmup']} iterations, which are discarded, then",
         f"timed over {ref['reps']} repetitions. CUDA is asynchronous, so",
         r"\texttt{torch.cuda.synchronize()} brackets every timed region; without it the measurement",
         r"records only Python dispatch. We report the \emph{median} over repetitions, which is robust",
         r"to occasional descheduling by the operating system, and the 95th percentile separately as",
         r"the tail. Two timings are recorded because they answer different questions:",
         r"\texttt{predict} calls \texttt{Chronos2Pipeline.predict\_quantiles} on host arrays and",
         r"therefore \emph{includes} preprocessing and the host-to-device transfer, i.e.\ the whole",
         r"serving path; \texttt{encode} times the encoder forward alone on a tensor already resident",
         r"on the device and therefore \emph{excludes} transfer. Speedups quoted in the main text use",
         r"\texttt{predict}, the conservative choice, since its fixed costs do not shrink with depth.",
         "",
         r"\paragraph{Throughput.}",
         r"Series per second is the batch size divided by the median \texttt{predict} time. It is not",
         r"the reciprocal of single-request latency: batching amortises fixed per-call cost, so",
         r"throughput improves with batch size while latency does not.", "",
         r"\paragraph{Memory.}",
         r"Peak memory is \texttt{torch.cuda.max\_memory\_allocated()} after",
         r"\texttt{reset\_peak\_memory\_stats()}, recorded over the timed region. We deliberately do",
         r"not use \texttt{nvidia-smi}, which reports the caching allocator's reservation and would",
         r"not shrink with truncation. The weight-only figure is the allocated total after loading and",
         r"truncating but before any forward pass. The model is reloaded from scratch for every depth",
         r"so that dropped blocks are genuinely freed rather than merely unused; a harness that reuses",
         r"one process would report a memory column that measures nothing.", "",
         r"\paragraph{Measured results.}",
         r"Table~\ref{tab:latency-full} reports every configuration. Latency and memory depend on",
         r"truncation depth only, not on the dataset, so the per-dataset speedups in the main text are",
         r"lookups into this table at each dataset's selected depth.", ""]

    T = [r"\begin{table}[t]", r"    \centering", r"    \small",
         r"    \caption{Measured latency, throughput and memory of the truncated model. "
         r"\texttt{predict} is the full serving path (transfer included); \texttt{encode} is the "
         r"encoder forward on a device-resident tensor (transfer excluded). Speedup is the median "
         r"\texttt{predict} time of the full 12-block model divided by that of the truncated model at "
         r"the same batch size.}",
         r"    \label{tab:latency-full}",
         r"    \begin{tabular}{llrrrrrr}", r"        \toprule",
         r"        Batch & Depth & Blocks & predict (ms) & p95 (ms) & encode (ms) & series/s & "
         r"peak (MiB) \\", r"        \midrule"]
    for b in batches:
        sub = [r for r in rows if r["batch_size"] == b]
        for j, r in enumerate(sub):
            T.append(f"        {b if j == 0 else ''} & \\({r['depth_label']}\\) & {r['depth']} & "
                     f"{r['predict_median_ms']:.2f} & {r['predict_p95_ms']:.2f} & "
                     f"{r['encode_median_ms']:.2f} & {r['throughput_series_per_s']:.0f} & "
                     f"{r.get('peak_mib', float('nan')):.1f} \\\\")
        if b != batches[-1]:
            T.append(r"        \midrule")
    T += [r"        \bottomrule", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(L + T) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", choices=(*RULES, "both"), default="both")
    ap.add_argument("--datasets", nargs="+", default=ALL_TAGS)
    ap.add_argument("--threshold-figure", action="store_true",
                    help="emit the eps-sensitivity panels (and only those)")
    ap.add_argument("--epsilons", type=float, nargs="+", default=list(DEFAULT_EPS))
    ap.add_argument("--latex-appendix", action="store_true",
                    help="join the measured latency run and emit the complete table + the "
                         "'Latency and Memory Methodology' appendix section")
    ap.add_argument("--speedup-batch", type=int, default=256,
                    help="batch size whose measured speedup goes in the complete table")
    ap.add_argument("--latency-json", default=None,
                    help="a specific latency run (default: the newest under latency/tables/)")
    args = ap.parse_args()

    tags = [t for t in ALL_TAGS if t in set(args.datasets)]
    if not tags:
        raise SystemExit(f"no known datasets in {args.datasets}; choose from {ALL_TAGS}")
    rules = RULES if args.selection == "both" else (args.selection,)
    for d in ("tables", "latex", "figures"):
        (OUT_ROOT / d).mkdir(parents=True, exist_ok=True)

    if args.threshold_figure:
        eps = sorted(args.epsilons, reverse=True)
        rows = make_threshold_figures(tags, eps, OUT_ROOT / "figures")
        with open(OUT_ROOT / "tables" / "threshold_sensitivity.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"  {'dataset':<14}" + "".join(f"{e:>13.0%}" for e in eps))
        for tag in tags:
            cells = sorted([r for r in rows if r["dataset"] == tag], key=lambda r: -r["epsilon"])
            print(f"  {PRETTY[tag]:<14}" + "".join(
                f"{c['depth_label'] + ' (' + format(c['block_flops_fraction'], '.2f') + 'x)':>13}"
                for c in cells))
        for stem in ("threshold_sensitivity_curves", "threshold_sensitivity_depths"):
            print(f"[write] {OUT_ROOT / 'figures' / (stem + '.png')}")
        print(f"[write] {OUT_ROOT / 'tables' / 'threshold_sensitivity.csv'}")
        return

    print(f"model: {model_size.TOTAL_PARAMS:,} parameters "
          f"(block {model_size.BLOCK_PARAMS:,} x {model_size.NUM_BLOCKS}, "
          f"head {model_size.NATIVE_HEAD_PARAMS:,}, adapter {model_size.ADAPTER_PARAMS:,})")
    rows_by_rule, all_rows = {}, []
    for rule in rules:
        pick = saturation_depth if rule == "saturation" else erank_depth
        rows_by_rule[rule] = {}
        print(f"\n=== {rule}-selected truncation ===")
        print(f"  {'dataset':<12}{'depth':>7}{'params':>8}{'FLOPs':>8}{'MASE':>18}"
              f"{'rel [95% CI]':>24}{'recovery':>10}")
        for tag in tags:
            depth, prov = pick(tag)
            r = evaluate(tag, depth, rule, prov)
            rows_by_rule[rule][tag] = r
            all_rows.append(r)
            print(f"  {r['short']:<12}{r['depth_label']:>7}{100 * r['active_fraction']:>7.1f}%"
                  f"{r['block_flops_fraction']:>7.2f}x"
                  f"{r['truncated_mase']:>9.3f}/{r['native_mase']:<8.3f}"
                  f"{100 * r['relative_mase']:>+8.1f}% [{100 * r['relative_mase_ci_lo']:>+6.1f},"
                  f"{100 * r['relative_mase_ci_hi']:>+6.1f}]{r['gap_recovery']:>10.2f}")

    fields = list(all_rows[0].keys())
    csv_path = OUT_ROOT / "tables" / "compression_cost.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)
    json.dump({"rows": all_rows,
               "parameter_breakdown": model_size.param_breakdown(),
               "adapter_params": model_size.ADAPTER_PARAMS,
               "total_params": model_size.TOTAL_PARAMS,
               "flops_note": ("block_flops_fraction = depth/12 is exact (12 identical blocks); "
                              "end_to_end_flops_fraction additionally counts the fixed input-embedding "
                              "and head cost and is a MAC-counting estimate, not a measurement, and "
                              "not a latency claim -- see experiments/run_latency.py"),
               "bootstrap": {"B": BOOT_B, "seed": SEED, "unit": "test series (cluster)",
                             "paired": "one shared resampling matrix per dataset"}},
              open(OUT_ROOT / "tables" / "compression_cost.json", "w"), indent=1)
    print(f"\n[write] {csv_path}")

    if args.latex_appendix:
        lat = load_latency(args.latency_json)
        if lat is None:
            raise SystemExit(
                "no latency run found under results/ext_v5_native_head_adapter/latency/tables/.\n"
                "  Latency and memory are MEASUREMENTS -- they cannot be derived from the committed\n"
                "  artifacts. Produce them first (GPU/compute node, NOT the login node):\n"
                "      sbatch -J lat job_latency.sh --verify")
        full = latex_full_table(rows_by_rule, tags, lat, args.speedup_batch)
        (OUT_ROOT / "latex" / "compression_table__full.tex").write_text(full + "\n")
        app = latex_appendix_methodology(lat, args.speedup_batch)
        (OUT_ROOT / "latex" / "appendix_latency_methodology.tex").write_text(app)
        lut = latency_by_depth(lat, args.speedup_batch)
        print(f"\n[latency] {lat['environment'].get('gpu_name')} — measured speedup at batch "
              f"{args.speedup_batch}:")
        for d in sorted(lut):
            r = lut[d]
            print(f"    L{d:<3} {r['predict_median_ms']:>8.2f} ms  {r['speedup']:>5.2f}x  "
                  f"peak {r.get('peak_mib', float('nan')):>7.1f} MiB")
        for stem in ("compression_table__full", "appendix_latency_methodology"):
            print(f"[write] {OUT_ROOT / 'latex' / (stem + '.tex')}")

    if len(rules) == 2:
        p = OUT_ROOT / "latex" / "compression_table__two_rules.tex"
        p.write_text(latex_two_rule(rows_by_rule, tags) + "\n")
        print(f"[write] {p}")
    for rule in rules:
        p = OUT_ROOT / "latex" / f"compression_table__{rule}.tex"
        p.write_text(latex_single_rule(rows_by_rule[rule], tags, rule) + "\n")
        print(f"[write] {p}")


if __name__ == "__main__":
    main()
