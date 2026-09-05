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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", choices=(*RULES, "both"), default="both")
    ap.add_argument("--datasets", nargs="+", default=ALL_TAGS)
    args = ap.parse_args()

    tags = [t for t in ALL_TAGS if t in set(args.datasets)]
    if not tags:
        raise SystemExit(f"no known datasets in {args.datasets}; choose from {ALL_TAGS}")
    rules = RULES if args.selection == "both" else (args.selection,)
    for d in ("tables", "latex"):
        (OUT_ROOT / d).mkdir(parents=True, exist_ok=True)

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
