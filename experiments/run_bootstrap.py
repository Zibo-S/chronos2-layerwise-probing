"""Post-hoc series-level CLUSTER bootstrap CIs for the ID forecasting probes.

Adjacent test windows from the same series overlap heavily (stride 64 << span 576), so an
IID bootstrap over windows would treat correlated windows as independent and produce CIs
that are too narrow. The resampling unit is therefore the SERIES: each replicate draws S
series with replacement and keeps ALL of a sampled series' windows (a series drawn twice
contributes its windows twice). Implemented with multinomial cluster counts
(probing.stats.cluster_bootstrap_apply = (M @ per_series_sum) / (M @ per_series_count) —
the ratio of weighted sums, i.e. the window-weighted mean under duplication); an
explicit-expansion self-check verifies the equivalence on real data every run.

ONE counts matrix per dataset is shared across every layer, readout, metric, and the native
Chronos-2 benchmark, so all paired differences (Delta_l = metric[L11] - metric[l], positive =
layer l better) use the same replicates on both sides — a paired bootstrap, never two
independent bootstraps subtracted.

Layer comparison protocol: L* is chosen on VALIDATION loss only (recorded during wd selection
in the GPU run; test never touched) and frozen; the PRIMARY comparison is Delta at that fixed
L*. The per-layer Delta_l intervals are reported too, but flagged as exploratory — the
test-set argmin is never called a validation-selected layer.

Metrics: ``quantile_loss`` (Chronos-2 pinball loss) and ``mase_context`` — the in-context
seasonal-naive MASE behind the currently reported numbers (NOT the canonical train-series
MASE; if that variant is ever added it needs its own input keys and, because its denominator
can be NaN, masked aggregation — all metrics consumed here must be finite, enforced on load).

Outputs are horizon-safe: everything lands under an ``H{H}_K{K}`` experiment id, so a run at
a different horizon coexists with (never overwrites) the current one.

Reads only results/bootstrap/inputs/<tag>__H*_K*.npz (written by run_id_forecasting's
save_bootstrap_inputs). No probe training, no model inference, no feature-cache access —
CPU-only, numpy-sized arrays, runs on the login node.

Run:  python -m experiments.run_bootstrap
"""

from __future__ import annotations

import argparse
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probing import config
from probing.config import NUM_LAYERS, LAST_LAYER, SEED, BOOT_DIR
from probing.stats import cluster_bootstrap_counts, cluster_bootstrap_apply

BOOT_B_CLUSTER = 5000          # replicates (task spec; the legacy IID helpers keep BOOT_B=2000)
BOOT_SEED = SEED               # one seed -> one shared counts matrix per dataset
CI_LO, CI_HI = 2.5, 97.5       # 95% percentile interval
MIN_SERIES_WARN = 20           # warn when a dataset retains fewer unique test series
METRICS = ("quantile_loss", "mase_context")
_ARRAY_PREFIX = {"quantile_loss": "window_loss", "mase_context": "window_mase_context"}
_YLABEL = {"quantile_loss": "Chronos-2 quantile loss (test)",
           "mase_context": "test MASE (in-context seasonal-naive scale)"}

def _derive_dirs() -> None:
    """Derive the output dirs from the ACTIVE BOOT_DIR — re-run after a --dataset-set
    override so this script targets the same results/<set>/ namespace as the driver run."""
    global RAW_DIR, TAB_DIR, FIG_DIRS
    RAW_DIR = BOOT_DIR / "raw"
    TAB_DIR = BOOT_DIR / "tables"
    FIG_DIRS = {"primary": BOOT_DIR / "figures" / "primary",
                "controlled": BOOT_DIR / "figures" / "controlled_k_slots"}
    for _d in (RAW_DIR, TAB_DIR, *FIG_DIRS.values()):
        _d.mkdir(parents=True, exist_ok=True)


_derive_dirs()

# same per-dataset colors as the driver figures (entity-stable; fallback for unknown tags)
try:
    from experiments.run_id_forecasting import ID_STYLE
except Exception:                                          # keep the script usable standalone
    ID_STYLE = {}
_FALLBACK = ("#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2")


def _style(tag, j):
    st = ID_STYLE.get(tag, {})
    return st.get("color", _FALLBACK[j % len(_FALLBACK)]), st.get("label", tag)


# --------------------------------------------------------------------------- #
# inputs + validation
# --------------------------------------------------------------------------- #

def load_inputs(path):
    """Load one dataset's per-window metrics and validate them against the reported numbers."""
    with np.load(path, allow_pickle=False) as d:
        meta = json.loads(str(d["meta"]))
        tag = meta["tag"]
        n = meta["n_test_windows"]
        sid = np.asarray(d["series_test"], np.int64)
        assert sid.shape == (n,), f"{tag}: series ids {sid.shape} not aligned with {n} test windows"

        readouts = list(meta["primary_readouts"]) + list(meta["controlled_readouts"])
        data = {}
        for r in readouts:
            data[r] = {}
            for metric in METRICS:
                W = np.asarray(d[f"{_ARRAY_PREFIX[metric]}__{r}"], np.float64)
                assert W.shape == (NUM_LAYERS, n), (
                    f"{tag}/{r}/{metric}: per-window array {W.shape} != ({NUM_LAYERS}, {n})")
                assert np.isfinite(W).all(), (
                    f"{tag}/{r}/{metric}: non-finite per-window values — this script's "
                    "aggregation is unmasked by policy; a NaN-bearing metric variant needs "
                    "its own keys + masked aggregation")
                data[r][metric] = W
            # per-window means must reproduce the currently reported window-mean aggregates
            np.testing.assert_allclose(
                data[r]["quantile_loss"].mean(axis=1), meta["reported"]["quantile_loss"][r],
                rtol=1e-5, atol=1e-7, err_msg=f"{tag}/{r}: per-window loss means != reported")
            np.testing.assert_allclose(
                data[r]["mase_context"].mean(axis=1), meta["reported"]["mase_context"][r],
                rtol=1e-5, atol=1e-7, err_msg=f"{tag}/{r}: per-window MASE means != reported")
        native = np.asarray(d["window_mase_context__native"], np.float64)
    assert native.shape == (n,), f"{tag}: native per-window MASE {native.shape} != ({n},)"
    assert np.isfinite(native).all(), f"{tag}: non-finite native per-window MASE"
    np.testing.assert_allclose(native.mean(), meta["reported"]["native_mase_context"],
                               rtol=1e-5, atol=1e-7,
                               err_msg=f"{tag}: native per-window MASE mean != reported")
    print(f"  [verified] per-window means reproduce the reported loss/MASE "
          f"({len(readouts)} readouts + native)")
    return meta, sid, data, native


def check_readout_consistency(exp_id, metas):
    """Every dataset in an experiment must carry the SAME readout sets — a missing readout
    means an incomplete GPU run and must fail loudly, never be silently intersected away."""
    first = next(iter(metas.values()))
    expected_primary = list(first["primary_readouts"])
    expected_controlled = list(first["controlled_readouts"])
    for tag, meta in metas.items():
        if list(meta["primary_readouts"]) != expected_primary:
            raise RuntimeError(f"{exp_id}/{tag}: primary readouts {meta['primary_readouts']} "
                               f"!= expected {expected_primary} — incomplete run?")
        if list(meta["controlled_readouts"]) != expected_controlled:
            raise RuntimeError(f"{exp_id}/{tag}: controlled readouts "
                               f"{meta['controlled_readouts']} != expected "
                               f"{expected_controlled} — incomplete run?")
    return expected_primary, expected_controlled


# --------------------------------------------------------------------------- #
# cluster bootstrap
# --------------------------------------------------------------------------- #

def _per_series_sums(W, inv, S):
    """W (L, n) or (n,) -> per-series sums (S, L)."""
    W2 = np.atleast_2d(np.asarray(W, np.float64))          # (L, n)
    sums = np.zeros((S, W2.shape[0]))
    np.add.at(sums, inv, W2.T)
    return sums


def _explicit_replicate_check(W, counts_row, inv, boot_row):
    """Expand one multinomial counts row into an explicit duplicated-window index list and
    verify the matmul shortcut equals the explicitly resampled window mean."""
    idx = np.concatenate([np.tile(np.flatnonzero(inv == j), int(c))
                          for j, c in enumerate(counts_row) if c > 0])
    explicit = np.atleast_2d(np.asarray(W, np.float64))[:, idx].mean(axis=1)
    np.testing.assert_allclose(explicit, np.atleast_1d(boot_row), rtol=1e-10,
                               err_msg="multinomial shortcut != explicit series resampling")


def bootstrap_dataset(exp_id, meta, sid, data, native):
    """All bootstrap distributions for one dataset from ONE shared counts matrix."""
    tag = meta["tag"]
    uniq, inv = np.unique(sid, return_inverse=True)
    S = len(uniq)
    cnt = np.bincount(inv).astype(np.float64)              # windows per series (S,)
    if S < MIN_SERIES_WARN:
        print(f"  [WARN] only {S} unique test series — cluster-bootstrap CIs may be unstable")
    counts = cluster_bootstrap_counts(S, BOOT_B_CLUSTER, BOOT_SEED)   # (B, S), shared by ALL

    raw_dir = RAW_DIR / exp_id
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {"n_test_series": S, "n_test_windows": int(len(sid)), "readouts": {}}
    checked = False
    for r, mats in data.items():
        ent = {}
        for metric in METRICS:
            boot = cluster_bootstrap_apply(counts, _per_series_sums(mats[metric], inv, S), cnt)
            assert boot.shape == (BOOT_B_CLUSTER, NUM_LAYERS)
            if not checked:                                # one explicit-expansion self-check
                _explicit_replicate_check(mats[metric], counts[0], inv, boot[0])
                print(f"  [verified] explicit series resampling == multinomial shortcut "
                      f"(replicate 0, {r}/{metric})")
                checked = True
            ent[metric] = boot                             # (B, NUM_LAYERS)
        out["readouts"][r] = ent
        np.savez(raw_dir / f"{tag}__{r}.npz",
                 loss_bootstrap=ent["quantile_loss"], mase_bootstrap=ent["mase_context"])
    nat_boot = cluster_bootstrap_apply(counts, _per_series_sums(native, inv, S), cnt)
    assert nat_boot.shape == (BOOT_B_CLUSTER, 1)
    out["native_mase_boot"] = nat_boot[:, 0]               # (B,)
    np.savez(raw_dir / f"{tag}__native.npz", mase_bootstrap=out["native_mase_boot"])
    print(f"  [saved] raw bootstrap arrays -> {raw_dir}/{tag}__*.npz "
          f"(B={BOOT_B_CLUSTER}, S={S})")
    return out


def _ci(boot):
    return np.percentile(boot, CI_LO, axis=0), np.percentile(boot, CI_HI, axis=0)


def summarize_dataset(meta, boot):
    """Point estimates (the reported numbers), CIs, and PAIRED Delta-vs-L11 CIs per readout."""
    summ = {"H": meta["H"], "K": meta["K"], "output_patch_size": meta["output_patch_size"],
            "n_test_series": boot["n_test_series"], "n_test_windows": boot["n_test_windows"],
            "native": {}, "readouts": {}}
    lo, hi = _ci(boot["native_mase_boot"])
    summ["native"] = {"mase_context_point": float(meta["reported"]["native_mase_context"]),
                      "mase_context_ci": [float(lo), float(hi)]}
    for r, ent in boot["readouts"].items():
        rs = {"val_selected_layer": meta["val_selected_layer"].get(r),
              "val_loss_by_layer": meta["val_loss_by_layer"].get(r)}
        for metric in METRICS:
            point = np.asarray(meta["reported"][metric][r], np.float64)
            b = ent[metric]
            lo, hi = _ci(b)
            delta_b = b[:, [LAST_LAYER]] - b               # paired: same replicates both sides
            dlo, dhi = _ci(delta_b)
            rs[metric] = {
                "point": point.tolist(),
                "ci_lo": lo.tolist(), "ci_hi": hi.tolist(),
                "delta_vs_L11": (point[LAST_LAYER] - point).tolist(),
                "delta_ci_lo": dlo.tolist(), "delta_ci_hi": dhi.tolist(),
                "delta_above_zero": (dlo > 0).tolist(),
            }
        # PRIMARY comparison: the frozen validation-selected layer vs L11 (both metrics).
        # Everything else in delta_vs_L11 is exploratory (per-layer, not selection-corrected).
        Ls = rs["val_selected_layer"]
        if Ls is not None:
            rs["primary_comparison"] = {
                "layer": Ls,
                **{metric: {"delta": rs[metric]["delta_vs_L11"][Ls],
                            "ci": [rs[metric]["delta_ci_lo"][Ls], rs[metric]["delta_ci_hi"][Ls]],
                            "supported": bool(rs[metric]["delta_ci_lo"][Ls] > 0)}
                   for metric in METRICS}}
        summ["readouts"][r] = rs
    return summ


# --------------------------------------------------------------------------- #
# tables
# --------------------------------------------------------------------------- #

TABLE_FIELDS = ["dataset", "H", "K", "output_patch_size", "readout", "metric", "layer",
                "point", "ci_lo", "ci_hi",
                "delta_vs_L11", "delta_ci_lo", "delta_ci_hi", "delta_above_zero",
                "is_primary_comparison", "val_selected_layer",
                "n_test_series", "n_test_windows", "bootstrap_seed", "n_replicates"]


def build_rows(all_summ):
    """Flat rows across all experiments/datasets (native = its own readout, mase only)."""
    rows = []
    for exp_id, datasets in all_summ.items():
        for tag, ds in datasets.items():
            base = {"dataset": tag, "H": ds["H"], "K": ds["K"],
                    "output_patch_size": ds["output_patch_size"],
                    "n_test_series": ds["n_test_series"],
                    "n_test_windows": ds["n_test_windows"],
                    "bootstrap_seed": BOOT_SEED, "n_replicates": BOOT_B_CLUSTER}
            for r, rs in ds["readouts"].items():
                Ls = rs["val_selected_layer"]
                for metric in METRICS:
                    m = rs[metric]
                    for layer in range(NUM_LAYERS):
                        rows.append({**base, "readout": r, "metric": metric, "layer": layer,
                                     "point": m["point"][layer],
                                     "ci_lo": m["ci_lo"][layer], "ci_hi": m["ci_hi"][layer],
                                     "delta_vs_L11": m["delta_vs_L11"][layer],
                                     "delta_ci_lo": m["delta_ci_lo"][layer],
                                     "delta_ci_hi": m["delta_ci_hi"][layer],
                                     "delta_above_zero": m["delta_above_zero"][layer],
                                     "is_primary_comparison": layer == Ls,
                                     "val_selected_layer": Ls})
            rows.append({**base, "readout": "native", "metric": "mase_context", "layer": "",
                         "point": ds["native"]["mase_context_point"],
                         "ci_lo": ds["native"]["mase_context_ci"][0],
                         "ci_hi": ds["native"]["mase_context_ci"][1],
                         "delta_vs_L11": "", "delta_ci_lo": "", "delta_ci_hi": "",
                         "delta_above_zero": "", "is_primary_comparison": "",
                         "val_selected_layer": ""})
    return rows


def write_tables(rows):
    csv_path = TAB_DIR / "bootstrap_table.csv"
    with open(csv_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=TABLE_FIELDS)
        wr.writeheader()
        wr.writerows(rows)
    json_path = TAB_DIR / "bootstrap_table.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  [saved] {csv_path}\n  [saved] {json_path}")


# --------------------------------------------------------------------------- #
# figures — repo idiom: per-dataset entity colors, ★/vline markers, recessive grid
# --------------------------------------------------------------------------- #

def _panels(n):
    rows = math.ceil(n / 2)
    fig, axes = plt.subplots(rows, min(2, n), figsize=(13, 4.5 * rows),
                             sharex=True, squeeze=False)
    axes = axes.ravel()
    for ax in axes[n:]:                                    # hide unused panels
        ax.set_visible(False)
    return fig, axes


def _mark_val_layer(ax, Ls):
    if Ls is not None:
        ax.axvline(Ls, color="0.35", ls=":", lw=1.3)
        ax.annotate("L* (val-selected)", xy=(Ls, 0.02), xycoords=("data", "axes fraction"),
                    fontsize=7, color="0.35", rotation=90, va="bottom", ha="right")


def make_by_layer_figures(exp_id, datasets, group, readouts):
    """Metric by layer with a 95% cluster-bootstrap band; one figure per readout x metric,
    one panel per dataset. MASE figures include the native Chronos-2 line + its CI band."""
    xs = np.arange(NUM_LAYERS)
    fig_dir = FIG_DIRS[group] / exp_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    for r in readouts:
        for metric in METRICS:
            fig, axes = _panels(len(datasets))
            for j, (ax, (tag, ds)) in enumerate(zip(axes, datasets.items())):
                color, label = _style(tag, j)
                rs = ds["readouts"][r]
                m = rs[metric]
                ax.plot(xs, m["point"], color=color, lw=2.2, marker="o", ms=4,
                        label="point estimate")
                ax.fill_between(xs, m["ci_lo"], m["ci_hi"], color=color, alpha=0.18, lw=0,
                                label="95% cluster-bootstrap CI")
                if metric == "mase_context":
                    nat = ds["native"]
                    ax.axhline(nat["mase_context_point"], color="k", ls="--", lw=1.3,
                               label=f"native Chronos-2 = {nat['mase_context_point']:.3f}")
                    ax.axhspan(nat["mase_context_ci"][0], nat["mase_context_ci"][1],
                               color="k", alpha=0.08, lw=0)
                _mark_val_layer(ax, rs["val_selected_layer"])
                ax.set_title(f"{label}  (S={ds['n_test_series']} series, "
                             f"n={ds['n_test_windows']} windows)", fontsize=10)
                ax.set_xticks(xs)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7, loc="best")
                ax.set_ylabel(_YLABEL[metric])
                ax.set_xlabel("encoder layer")
            fig.suptitle(f"{metric} by layer — {r}  [{group} experiment, {exp_id}]\n"
                         f"series-level cluster bootstrap, B={BOOT_B_CLUSTER}; "
                         "dotted vline = validation-selected L*", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            out = fig_dir / f"{metric}_by_layer__{r}.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"  [saved] {out}")


def make_delta_figures(exp_id, datasets, group, readouts):
    """Paired Delta_l = metric[L11] - metric[l] with 95% paired-bootstrap CIs. Positive =
    layer l better than L11. Filled markers = CI entirely above zero; ★ = the frozen
    validation-selected primary layer (all other layers are exploratory)."""
    xs = np.arange(NUM_LAYERS)
    fig_dir = FIG_DIRS[group] / exp_id
    fig_dir.mkdir(parents=True, exist_ok=True)
    for r in readouts:
        for metric in METRICS:
            fig, axes = _panels(len(datasets))
            for j, (ax, (tag, ds)) in enumerate(zip(axes, datasets.items())):
                color, label = _style(tag, j)
                rs = ds["readouts"][r]
                m = rs[metric]
                d = np.asarray(m["delta_vs_L11"])
                dlo, dhi = np.asarray(m["delta_ci_lo"]), np.asarray(m["delta_ci_hi"])
                ax.axhline(0.0, color="gray", ls=":", lw=1)
                ax.errorbar(xs, d, yerr=[d - dlo, dhi - d], fmt="none", ecolor=color,
                            elinewidth=1.6, capsize=3, alpha=0.9)
                sup = dlo > 0
                if sup.any():
                    ax.plot(xs[sup], d[sup], "o", color=color, ms=6, mec="k", mew=0.5,
                            ls="none", label="CI entirely above zero")
                if (~sup).any():
                    ax.plot(xs[~sup], d[~sup], "o", mfc="white", mec=color, ms=6,
                            ls="none", label="CI crosses zero")
                Ls = rs["val_selected_layer"]
                if Ls is not None:
                    ax.plot(Ls, d[Ls], marker="*", ms=15, color=color, mec="k", mew=0.6,
                            ls="none", zorder=5, label="primary (val-selected L*)")
                ax.set_title(f"{label}", fontsize=10)
                ax.set_xticks(xs)
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7, loc="best")
                ax.set_ylabel(f"Δ {metric}  (L11 − layer)")
                ax.set_xlabel("encoder layer")
            fig.suptitle(f"paired Δ vs L11 — {r}  [{group} experiment, {exp_id}]\n"
                         "positive = layer beats L11; same resampled series on both sides "
                         f"(paired, B={BOOT_B_CLUSTER}); non-★ layers are exploratory",
                         fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            out = fig_dir / f"{metric}_delta_vs_L11__{r}.png"
            fig.savefig(out, dpi=140, bbox_inches="tight")
            plt.close(fig)
            print(f"  [saved] {out}")


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def _parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Series-level cluster bootstrap over one dataset set's driver outputs "
                    "(reads/writes results/<set>/bootstrap/).")
    ap.add_argument("--dataset-set", default=None, metavar="NAME",
                    help="dataset set to process (see probing.id_data.ID_DATASET_SPECS); "
                         "precedence: CLI > env ID_DATASET_SET > default. Use the same "
                         "value the driver run used.")
    return ap.parse_args(argv)


def main():
    args = _parse_args()
    if args.dataset_set:
        config.set_dataset_set(args.dataset_set)
        global BOOT_DIR
        BOOT_DIR = config.BOOT_DIR         # import-time snapshot -> re-derived value
        _derive_dirs()

    inputs = sorted((BOOT_DIR / "inputs").glob("*.npz"))
    if not inputs:
        raise SystemExit(
            f"no bootstrap inputs in {BOOT_DIR / 'inputs'} — run "
            "`python -m experiments.run_id_forecasting` (GPU) first; it writes the "
            "per-window metrics this post-hoc script consumes.")

    # load everything, keyed by experiment id; duplicates are an error, never an overwrite
    loaded, seen = {}, set()
    for path in inputs:
        print(f"\n{'='*70}\n[{path.stem}] loading + validating inputs\n{'='*70}")
        meta, sid, data, native = load_inputs(path)
        key = (meta["tag"], meta["H"], meta["K"])
        if key in seen:
            raise RuntimeError(f"duplicate bootstrap input for {key} ({path.name})")
        seen.add(key)
        exp_id = f"H{meta['H']}_K{meta['K']}"
        loaded.setdefault(exp_id, {})[meta["tag"]] = (meta, sid, data, native)

    all_summ, config_datasets = {}, {}
    for exp_id, entries in loaded.items():
        metas = {tag: e[0] for tag, e in entries.items()}
        # readouts are DISCOVERED from the saved metadata (never hard-coded K/horizon names)
        # and must be identical across a run's datasets — a mismatch fails loudly.
        primary, controlled = check_readout_consistency(exp_id, metas)
        datasets = {}
        for tag, (meta, sid, data, native) in entries.items():
            print(f"\n{'='*70}\n[{exp_id}/{tag}] cluster bootstrap\n{'='*70}")
            boot = bootstrap_dataset(exp_id, meta, sid, data, native)
            datasets[tag] = summarize_dataset(meta, boot)
        all_summ[exp_id] = datasets

        # figure groups are kept separate: primary pooled readouts vs the controlled K-slot pass
        make_by_layer_figures(exp_id, datasets, "primary", primary)
        make_delta_figures(exp_id, datasets, "primary", primary)
        make_by_layer_figures(exp_id, datasets, "controlled", controlled)
        make_delta_figures(exp_id, datasets, "controlled", controlled)

        config_datasets[exp_id] = {
            "readouts": {"primary": primary, "controlled": controlled},
            "datasets": {tag: {"n_test_series": ds["n_test_series"],
                               "n_test_windows": ds["n_test_windows"],
                               "H": ds["H"], "K": ds["K"]}
                         for tag, ds in datasets.items()}}

    rows = build_rows(all_summ)
    write_tables(rows)

    config = {"n_replicates": BOOT_B_CLUSTER, "seed": BOOT_SEED,
              "ci_percentiles": [CI_LO, CI_HI],
              "resampling_unit": "series (cluster bootstrap)",
              "method": "S series drawn with replacement per replicate; all windows of a "
                        "sampled series included (twice if drawn twice); implemented as "
                        "multinomial counts, verified equivalent to explicit resampling; "
                        "ONE counts matrix per dataset shared across all layers, readouts, "
                        "metrics, and native (paired differences use identical replicates)",
              "aggregation": "window-weighted mean (matches the reported metrics; series "
                             "with more test windows contribute more windows)",
              "metrics": {"quantile_loss": "Chronos-2 pinball loss, mean(H)->sum(Q)->mean(n)",
                          "mase_context": "in-context seasonal-naive MASE (the reported "
                                          "definition; not canonical train-series MASE)"},
              "layer_selection": "L* frozen from validation loss (wd-selection carve); "
                                 "per-layer deltas are exploratory",
              "experiments": config_datasets}
    with open(BOOT_DIR / "bootstrap_config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(BOOT_DIR / "bootstrap_summary.json", "w") as f:
        json.dump(all_summ, f, indent=2)
    print(f"\n  [saved] {BOOT_DIR / 'bootstrap_config.json'}")
    print(f"  [saved] {BOOT_DIR / 'bootstrap_summary.json'}")

    # concise report: the PRIMARY (validation-selected) comparisons only
    print(f"\n{'='*70}\nPRIMARY comparisons (frozen val-selected L* vs L11; positive Δ = L* "
          f"better)\n{'='*70}")
    for exp_id, datasets in all_summ.items():
        for tag, ds in datasets.items():
            print(f"  [{exp_id}] {tag}  (S={ds['n_test_series']} series, "
                  f"n={ds['n_test_windows']} windows)")
            for r, rs in ds["readouts"].items():
                pc = rs.get("primary_comparison")
                if pc is None:
                    print(f"    {r:>10}: no validation-selected layer (wd grid was off)")
                    continue
                for metric in METRICS:
                    e = pc[metric]
                    flag = "SUPPORTED" if e["supported"] else "crosses 0"
                    print(f"    {r:>10} L{pc['layer']:<2} vs L11  Δ={e['delta']:+.4f}  "
                          f"CI[{e['ci'][0]:+.4f}, {e['ci'][1]:+.4f}]  {flag}  ({metric})")


if __name__ == "__main__":
    main()
