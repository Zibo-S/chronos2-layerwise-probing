"""q1-vs-q9 comparison + train-vs-val recompute + CKA collection for the ext-v4 fslot rerun.

The per-quantile experiments (run_ptood_probing_ftok / run_fslot_transfer) write their own q9 and q1
figures side by side under results/ext_v4_future_tokens/<qset>/. This driver adds the SMALL set of
DIRECT q1-vs-q9 comparison figures the project wants (not dozens), plus the training-vs-validation
recompute figures, and collects the (q-independent) CKA figures into one browsable place. It is
post-hoc: it reads the versioned v2 ID run artifacts + checkpoints + tunnels; nothing is re-extracted.

Scientific framing (do NOT optimise toward a U-shape): q9 measures probabilistic forecasting
accessibility, q1 the median-only readout with a much smaller head. The comparison asks whether the
layerwise transfer/specialization pattern is ROBUST to the readout's capacity/objective.

Modes (run from the repo root, venv active):
  --train-recompute   [COMPUTE NODE — reads the ~344 MB train fslot caches + builds windows; CPU, warm
                       caches, no model] For each qset in {q9,q1} x PT-ID dataset x seed: reload the
                       frozen v2 probe and score the TRAIN and VAL splits, writing per-run train/val/
                       selected-wd JSONs + the per-qset 2x2 train-vs-val figure (Q stated in the title).
  --figures           [LOGIN/CPU] Read those recompute JSONs + the per-seed run JSONs + both quantile
                       sets' tunnels and emit results/comparisons/q1_vs_q9/{relative_regret_shape,
                       generalization_gap,selected_wd}.png + tunnel_entrances.{csv,json} + the combined
                       train_vs_val table.
  --cka               [LOGIN/CPU] Collect the committed results/cka/ figures into results/comparisons/cka/.

    salloc --account=def-irina --cpus-per-task=4 --mem=32G --time=0:45:00     # for --train-recompute
    python -m experiments.run_q1q9_compare --train-recompute
    python -m experiments.run_q1q9_compare --figures --cka                    # login/CPU
"""
import argparse
import csv
import json
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from probing import config
from probing.probes import QUANTILE_SETS, PROBE_PROTOCOL_VERSION
from experiments.run_ptood_probing_ftok import (OUT_ROOT, PT_ID_TAGS, LAYER_LABELS, RUN_SEEDS, SHORT,
                                                 load_ptid_ckpt, _fslot_feats, _run_qtag,
                                                 _linear_tunnel_path, FAMILY, PROBE_FAMILIES)

QSETS = ("q9", "q1")
QLABEL = {"q9": "Q=9 (deciles)", "q1": "Q=1 (median, q=0.5)"}
QCOLOR = {"q9": "#0072B2", "q1": "#D55E00"}     # Okabe-Ito blue / vermillion
CMP_ROOT = config.REPO_ROOT / "results" / "comparisons"
CKA_SRC = config.REPO_ROOT / "results" / "cka"


# --------------------------------------------------------------------------- #
def _id_dir(qset):
    return OUT_ROOT / qset / "id"


def _run_json(qset, src, seed):
    """Per-seed ID run JSON written by run_ptood_probing_ftok --fit-ptid (versioned filename)."""
    return OUT_ROOT / "ptood_probing" / "ptid_runs" / f"{src}__{_run_qtag(qset)}__seed{seed}.json"


def _recompute_json(qset, src, seed):
    return _id_dir(qset) / "tables" / f"train_vs_val__{src}__{_run_qtag(qset)}__seed{seed}.json"


def _relative_regret(curve):
    """(loss(layer) - min_layer loss) / min_layer loss — the project's normalized layerwise shape,
    comparable ACROSS quantile sets whose raw losses live on different scales."""
    c = np.asarray(curve, float)
    m = float(c.min())
    return (c - m) / m if m > 0 else c * 0.0


# --------------------------------------------------------------------------- #
# --train-recompute : reload frozen v2 probes, score train+val, render per-qset train-vs-val
# --------------------------------------------------------------------------- #
def train_recompute():
    from probing.id_data import build_windows
    from probing.probes import predict_shared_forecast_probe
    global FAMILY
    import experiments.run_ptood_probing_ftok as ftok
    ftok.FAMILY = PROBE_FAMILIES["shared_linear"]        # linear only — no MLP anywhere in this rerun
    config.set_dataset_set("extended_v3_rolling")        # roster + rolling windows + cache namespace

    for qset in QSETS:
        quantiles = QUANTILE_SETS[qset]
        (_id_dir(qset) / "tables").mkdir(parents=True, exist_ok=True)
        (_id_dir(qset) / "figures").mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=False)
        drawn = 0
        for ax, src in zip(axes.ravel(), PT_ID_TAGS):
            w = None
            tr_seeds, va_seeds = [], []
            for seed in RUN_SEEDS:
                try:
                    fitted = load_ptid_ckpt(src, qset, seed, device="cpu")
                except FileNotFoundError:
                    print(f"  [skip] {SHORT[src]} {qset} seed{seed}: no v2 checkpoint — run --fit-ptid first")
                    fitted = None
                if fitted is None:
                    continue
                if w is None:
                    w = build_windows(src)
                    f_tr = _fslot_feats(src, "train", w["X_train"], w["y_train"])
                    f_va = _fslot_feats(src, "val", w["X_val"], w["y_val"])
                out_tr = predict_shared_forecast_probe(fitted, f_tr, w["Y_train_traj"],
                                                       quantiles=quantiles, device="cpu")
                out_va = predict_shared_forecast_probe(fitted, f_va, w["Y_val_traj"],
                                                       quantiles=quantiles, device="cpu")
                tr = [float(out_tr[i]) for i in sorted(out_tr)]
                va = [float(out_va[i]) for i in sorted(out_va)]
                wd = [float(fitted[i]["wd"]) for i in sorted(fitted)]
                json.dump({"dataset": src, "quantile_set": qset,
                           "probe_protocol_version": PROBE_PROTOCOL_VERSION, "run_seed": int(seed),
                           "train_loss_by_layer": tr, "val_loss_by_layer_recomputed": va,
                           "selected_wd_by_layer": wd},
                          open(_recompute_json(qset, src, seed), "w"), indent=2)
                tr_seeds.append(tr); va_seeds.append(va)
            if not tr_seeds:
                ax.set_title(f"{SHORT[src]} — no v2 probes yet"); continue
            tr_a, va_a = np.array(tr_seeds), np.array(va_seeds)
            x = np.arange(tr_a.shape[1]); trm, vam = tr_a.mean(0), va_a.mean(0)
            ax.fill_between(x, tr_a.min(0), tr_a.max(0), color=QCOLOR["q1"], alpha=.15, lw=0)
            ax.fill_between(x, va_a.min(0), va_a.max(0), color=QCOLOR["q9"], alpha=.15, lw=0)
            ax.plot(x, trm, "-o", color=QCOLOR["q1"], ms=4, lw=1.8, label="train (recomputed)")
            ax.plot(x, vam, "--s", color=QCOLOR["q9"], ms=4, lw=1.8, label="validation")
            gap = vam - trm
            ax.set_title(f"{SHORT[src]}   (train–val gap: {gap.min():+.2f} … {gap.max():+.2f})",
                         fontsize=11, weight="bold")
            ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right", fontsize=8)
            ax.grid(alpha=.25); ax.set_ylabel(f"{qset} quantile loss"); ax.legend(fontsize=8)
            drawn += 1
        fig.suptitle(f"Shared-LINEAR fslot readout — training vs validation loss by layer  "
                     f"[{QLABEL[qset]}; wide WD grid {PROBE_PROTOCOL_VERSION}]",
                     fontsize=12.5, weight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = _id_dir(qset) / "figures" / "train_vs_val.png"
        fig.savefig(out, dpi=140); plt.close(fig)
        print(f"[train-recompute {qset}] {drawn}/{len(PT_ID_TAGS)} datasets -> {out}")


# --------------------------------------------------------------------------- #
# --figures : the small set of DIRECT q1-vs-q9 comparison figures + tables
# --------------------------------------------------------------------------- #
def _load_run_curves(qset):
    """{src: {'val': (S,L), 'test': (S,L)}} seed-stacked from the versioned per-seed run JSONs."""
    out = {}
    for src in PT_ID_TAGS:
        val, test = [], []
        for seed in RUN_SEEDS:
            p = _run_json(qset, src, seed)
            if not p.exists():
                break
            d = json.load(open(p))
            val.append(d["val_loss_by_layer"]); test.append(d["test_loss_by_layer"])
        if val:
            out[src] = {"val": np.array(val), "test": np.array(test)}
    return out


def _load_recompute(qset):
    """{src: {'train': (S,L), 'val': (S,L), 'wd': (S,L)}} from the --train-recompute JSONs."""
    out = {}
    for src in PT_ID_TAGS:
        tr, va, wd = [], [], []
        for seed in RUN_SEEDS:
            p = _recompute_json(qset, src, seed)
            if not p.exists():
                break
            d = json.load(open(p))
            tr.append(d["train_loss_by_layer"]); va.append(d["val_loss_by_layer_recomputed"])
            wd.append(d["selected_wd_by_layer"])
        if tr:
            out[src] = {"train": np.array(tr), "val": np.array(va), "wd": np.array(wd)}
    return out


def _figure_relative_regret_shape(runs):
    """A. Layerwise shape (relative regret) q1 vs q9 per dataset — is the tunnel structure readout-robust?"""
    present = [s for s in PT_ID_TAGS if all(s in runs[q] for q in QSETS)]
    if not present:
        print("[compare] relative-regret shape skipped (need both q run JSONs)"); return
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 4.2), squeeze=False)
    for ax, src in zip(axes[0], present):
        for q in QSETS:
            rr = _relative_regret(runs[q][src]["test"].mean(0))
            ax.plot(np.arange(len(rr)), rr, "-o", ms=3, color=QCOLOR[q], label=QLABEL[q])
        ax.axhline(0, ls=":", c="gray", lw=1)
        ax.set_xticks(np.arange(len(LAYER_LABELS)))
        ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_title(SHORT[src]); ax.set_ylabel("relative regret vs best layer"); ax.grid(alpha=.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("q1 vs q9 — layerwise accessibility shape (test relative regret)", weight="bold")
    fig.tight_layout()
    out = CMP_ROOT / "q1_vs_q9" / "relative_regret_shape.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[compare] A relative-regret shape -> {out}")


def _figure_generalization_gap(recomp):
    """B. Train-vs-val gap q1 vs q9 per dataset — does Q=1 reduce deep-layer overfitting?"""
    present = [s for s in PT_ID_TAGS if all(s in recomp[q] for q in QSETS)]
    if not present:
        print("[compare] generalization-gap skipped (need --train-recompute for both q)"); return
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 4.2), squeeze=False)
    for ax, src in zip(axes[0], present):
        for q in QSETS:
            gap = (recomp[q][src]["val"] - recomp[q][src]["train"]).mean(0)
            ax.plot(np.arange(len(gap)), gap, "-o", ms=3, color=QCOLOR[q], label=QLABEL[q])
        ax.axhline(0, ls=":", c="gray", lw=1)
        ax.set_xticks(np.arange(len(LAYER_LABELS)))
        ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_title(SHORT[src]); ax.set_ylabel("val − train (raw gap)"); ax.grid(alpha=.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("q1 vs q9 — generalization gap by layer (val − train)", weight="bold")
    fig.tight_layout()
    out = CMP_ROOT / "q1_vs_q9" / "generalization_gap.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[compare] B generalization-gap -> {out}")


def _figure_selected_wd(recomp):
    """C. Selected weight decay by layer q1 vs q9 — does q9 systematically need stronger regularization?"""
    present = [s for s in PT_ID_TAGS if all(s in recomp[q] for q in QSETS)]
    if not present:
        print("[compare] selected-wd skipped (need --train-recompute for both q)"); return
    fig, axes = plt.subplots(1, len(present), figsize=(4.6 * len(present), 4.2), squeeze=False)
    for ax, src in zip(axes[0], present):
        for q in QSETS:
            wd = recomp[q][src]["wd"]
            gm = np.exp(np.log(np.clip(wd, 1e-12, None)).mean(0))     # geometric mean over seeds
            ax.plot(np.arange(len(gm)), gm, "-o", ms=3, color=QCOLOR[q], label=QLABEL[q])
        ax.set_yscale("log")
        ax.set_xticks(np.arange(len(LAYER_LABELS)))
        ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right", fontsize=7)
        ax.set_title(SHORT[src]); ax.set_ylabel("selected weight decay (log)"); ax.grid(alpha=.25)
    axes[0][0].legend(fontsize=8)
    fig.suptitle("q1 vs q9 — validation-selected weight decay by layer", weight="bold")
    fig.tight_layout()
    out = CMP_ROOT / "q1_vs_q9" / "selected_wd.png"
    fig.savefig(out, dpi=150); plt.close(fig)
    print(f"[compare] C selected-wd -> {out}")


def _tunnel_entrances():
    """q9 entrance / q1 entrance / difference per PT-ID dataset (tunnels recomputed independently per q)."""
    rows = []
    for src in PT_ID_TAGS:
        ent = {}
        for q in QSETS:
            p = _linear_tunnel_path(src, q)
            ent[q] = int(json.load(open(p))["l_start"]) if p.exists() else None
        row = {"dataset": src, "short": SHORT[src],
               "q9_entrance": ent["q9"], "q1_entrance": ent["q1"],
               "q9_entrance_label": LAYER_LABELS[ent["q9"]] if ent["q9"] is not None else None,
               "q1_entrance_label": LAYER_LABELS[ent["q1"]] if ent["q1"] is not None else None,
               "difference": (None if None in (ent["q9"], ent["q1"]) else ent["q1"] - ent["q9"])}
        rows.append(row)
    return rows


def make_comparison_figures():
    (CMP_ROOT / "q1_vs_q9").mkdir(parents=True, exist_ok=True)
    runs = {q: _load_run_curves(q) for q in QSETS}
    recomp = {q: _load_recompute(q) for q in QSETS}
    _figure_relative_regret_shape(runs)
    _figure_generalization_gap(recomp)
    _figure_selected_wd(recomp)

    # tunnel-entrance table
    rows = _tunnel_entrances()
    (CMP_ROOT / "q1_vs_q9" / "tunnel_entrances.json").write_text(json.dumps(rows, indent=2))
    with open(CMP_ROOT / "q1_vs_q9" / "tunnel_entrances.csv", "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0])); wcsv.writeheader(); wcsv.writerows(rows)
    print(f"[compare] tunnel entrances ({len(rows)} datasets) -> {CMP_ROOT/'q1_vs_q9'/'tunnel_entrances.csv'}")

    # combined machine-readable train/val table (Section 18 schema)
    tab = []
    for q in QSETS:
        for src in PT_ID_TAGS:
            if src not in recomp[q]:
                continue
            tr, va, wd = recomp[q][src]["train"], recomp[q][src]["val"], recomp[q][src]["wd"]
            for si, seed in enumerate(RUN_SEEDS[:tr.shape[0]]):
                for L in range(tr.shape[1]):
                    g = float(va[si, L] - tr[si, L])
                    tab.append({"dataset": src, "layer": L, "layer_label": LAYER_LABELS[L],
                                "Q": len(QUANTILE_SETS[q]), "quantile_set": q,
                                "train_loss": round(float(tr[si, L]), 6),
                                "val_loss": round(float(va[si, L]), 6), "gap": round(g, 6),
                                "relative_gap": round(g / float(tr[si, L]), 6) if tr[si, L] > 0 else None,
                                "selected_wd": float(wd[si, L]), "seed": int(seed)})
    if tab:
        with open(CMP_ROOT / "q1_vs_q9" / "train_vs_val_table.csv", "w", newline="") as f:
            wcsv = csv.DictWriter(f, fieldnames=list(tab[0])); wcsv.writeheader(); wcsv.writerows(tab)
        print(f"[compare] train/val table ({len(tab)} rows) -> {CMP_ROOT/'q1_vs_q9'/'train_vs_val_table.csv'}")


# --------------------------------------------------------------------------- #
# --cka : collect the committed (q-independent) CKA figures into one browsable place
# --------------------------------------------------------------------------- #
def collect_cka():
    dst = CMP_ROOT / "cka"
    dst.mkdir(parents=True, exist_ok=True)
    if not CKA_SRC.exists():
        print(f"[cka] no {CKA_SRC} — run run_cka_analysis first (CKA is Q-independent)"); return
    n = 0
    for p in sorted(CKA_SRC.rglob("*.png")):
        rel = p.relative_to(CKA_SRC)
        tgt = dst / str(rel).replace("/", "__")
        shutil.copy2(p, tgt); n += 1
    (dst / "README.txt").write_text(
        "CKA is independent of the probe quantile set (backbone representations are identical for "
        "q1/q9), so it lives OUTSIDE the q1/q9 namespaces. These figures are copied from results/cka/ "
        "(source of truth). Blocks: pretrained-transfer (within-dataset layer x layer), domain "
        "specialization (pretrained vs BOOM early/late), task specialization (pretrained vs cls early/late).\n")
    print(f"[cka] collected {n} figures -> {dst}")


# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-recompute", action="store_true",
                   help="COMPUTE NODE: reload v2 probes, score train+val, render per-qset train-vs-val")
    p.add_argument("--figures", action="store_true",
                   help="LOGIN/CPU: emit the q1-vs-q9 comparison figures + tunnel-entrance + train/val tables")
    p.add_argument("--cka", action="store_true", help="LOGIN/CPU: collect results/cka into comparisons/cka")
    return p.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    if not (a.train_recompute or a.figures or a.cka):
        print("nothing to do — pass --train-recompute / --figures / --cka", file=sys.stderr)
        return
    if a.train_recompute:
        train_recompute()
    if a.figures:
        make_comparison_figures()
    if a.cka:
        collect_cka()


if __name__ == "__main__":
    main()
