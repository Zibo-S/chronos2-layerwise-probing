"""Stage B of the FT-specialization experiment — catastrophic forgetting on the BOOM backbones.

Design (notes/PLAN.md, STAGE B — DESIGN LOCKED 2026-08-11). The FT source is BOOM (PT-OOD): full-FT
genuinely specializes Chronos-2 on it (ft_val -5.3%), whereas the PT-ID sources are already at their
optimum (flat). We compare 3 backbone stages — stage0_pretrained / stage1_ft_early@300 /
stage2_ft_late@1000 — and ask whether specializing on unseen BOOM DEGRADES the domains the model
already forecast well, growing pretrained -> early -> late.

7 eval targets (pt_status / ft_status / probe_status recorded on every row):
    boom_hourly                                  PT-OOD / FT-ID  / probe-ID   (the source)
    electricity, uber, m4, windfarms             PT-ID  / FT-OOD / probe-ID   (known domains)
    sg_carpark, coastal_ts                       PT-OOD / FT-OOD / probe-ID

Metric hierarchy:
  1. PRIMARY forgetting  = NATIVE forecasting (each stage's native head, identical target-test
                           windows, ORIGINAL-scale MASE + WQL).                       [B3]
  2. PRIMARY layerwise   = FRESH shared-forecast-slot probes (probe-ID): per stage x target, fit on
                           the TARGET's own train, wd on target-val, score target-test.  [B2]
  3. SECONDARY transfer  = FROZEN BOOM probe applied to the 6 non-BOOM targets (probe-OOD). [B4]
Tunnel: each stage's tunnel is defined ONLY from its BOOM FT-ID/probe-ID validation curve, then used
as the layer-region lens for the FT-OOD curves (tunnel.py unchanged). Never from FT-OOD data.

Modes (each a build sub-stage; stage0 reuses the committed pretrained caches, stage1/stage2 load the
BOOM FT checkpoints via the extract_kout_features ``pipeline=`` / ``cache_prefix=`` injection into
collision-proof ``IDF_<tag>__ft__boom__<stage>__<hash8>`` caches):
  --extract/--smoke [B0/B1]  fslot feature caches, 3 stages x 7 targets x {train,val,test}.
  --probe/--tunnels/--figures [B2]  FRESH per-target shared-forecast-slot linear probes (probe-ID),
                             per-stage BOOM tunnels, layerwise curves — "does forecasting info stay
                             linearly recoverable after BOOM FT?".
  --native  [B3, PRIMARY]    each stage's OWN native head on IDENTICAL target-test windows, ORIGINAL
                             -scale MASE + WQL. The model's own forecast, no probe. Delta = FT -
                             pretrained (POSITIVE = worse = forgetting); BOOM (FT-ID) is the in-domain
                             control (expected to IMPROVE). GPU on a cold native cache.
  --transfer [B4, SECONDARY] the FROZEN BOOM readout (re-derived per stage/seed = B2's BOOM probe)
                             applied predict-only to the 6 non-BOOM targets (probe-OOD) — "does a
                             BOOM readout TRANSFER without retraining?". CPU / warm fslot caches.
  --forgetting [B5]          paired cluster-bootstrap forgetting stats (native, per target: early-vs-
                             pretrained + late-vs-pretrained CIs) + the B2-vs-B4 comparison + the
                             headline figures/tables. CPU aggregation over B2/B3/B4 outputs.

The three stages score IDENTICAL target-test windows (built once per target, seed 0), so every
pretrained->FT comparison is paired series-for-series. B3/B4/B5 use the BOOM-VALIDATION tunnel
entrance (from B2) as the layer lens — never a target-test-selected layer.

Run (GPU only for --extract and --native; OOD_TARGET_ROOT + HF offline set, e.g. via job_ft_stageB.sh):
    python -m experiments.run_ft_specialization --extract --smoke        # B0: BOOM/test, all 3 stages
    python -m experiments.run_ft_specialization --extract                # B1: full 3 x 7 x 3
    python -m experiments.run_ft_specialization --probe --tunnels --figures  # B2 (CPU, warm caches)
    python -m experiments.run_ft_specialization --native                 # B3 native MASE/WQL (GPU cold)
    python -m experiments.run_ft_specialization --transfer               # B4 frozen-BOOM (CPU)
    python -m experiments.run_ft_specialization --forgetting             # B5 stats + figures (CPU)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import math
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from probing import config
from probing.config import NUM_LAYERS, SEED, OUTPUT_PATCH_SIZE, CACHE_DIR
from probing.extraction import extract_kout_features, get_pipeline, _cache_path, _idf_prefix
from probing.id_data import build_ood_rolling_windows, build_windows
from probing.finetune import ft_cache_prefix, checkpoint_hash, _select_device
from probing.probes import (QUANTILE_SETS, PROBE_PROTOCOL_VERSION, WD_GRID_V2, validate_quantiles,
                            median_index, fit_shared_forecast_probe_explicit_val,
                            predict_shared_forecast_probe)
from probing.stats import cluster_bootstrap_counts, ci_bounds
from probing.tunnel import d_stat_boot, tunnel_record_multi, val_curve_from_selection
# B3/B5 reuse the native-forecasting + paired-bootstrap primitives verbatim (NO parallel evaluation
# logic): the in-context MASE denom + arcsinh inverse from run_id_forecasting / the forecasting
# comparison, and the per-window MASE/WQL + series-cluster-bootstrap adapters from that same driver.
from experiments.run_id_forecasting import M_SEASON, _ctx_stats, _mase_denominator
from experiments.run_fslot_forecasting_comparison import (
    _raw_future, _mase_pw, _mae_pw, _wql_pw_parts, _series_group, _boot_mean, _boot_ratio)

C, H = 512, 64
K = math.ceil(H / OUTPUT_PATCH_SIZE)              # native forecast-slot count (=4 at H=64)

FT_SOURCE = "boom"                               # the fine-tuning source label (SOURCE_TAGS['boom'])
FT_ID_TAG = "boom_hourly"                        # the source's HF/OOD tag = the FT-ID target
STAGE0 = "stage0_pretrained"

# roster: 4 PT-ID (build_windows / extended_v3_rolling) + 3 PT-OOD (build_ood_rolling_windows)
PT_ID_TARGETS = ("monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly")
PT_OOD_TARGETS = ("boom_hourly", "sg_carpark", "coastal_ts")
ALL_TARGETS = PT_ID_TARGETS + PT_OOD_TARGETS

SHORT = {"monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber",
         "m4_hourly": "M4", "wind_farms_hourly": "WindFarms",
         "sg_carpark": "SG-Carpark", "coastal_ts": "Coastal-TS", "boom_hourly": "BOOM"}

OUT_ROOT = config.REPO_ROOT / "results" / "ft_specialization" / "stageB"
FT_MANIFEST = config.REPO_ROOT / "results" / "ft_specialization" / FT_SOURCE / "manifest.json"

# --- B2 config (fresh shared-forecast-slot LINEAR probes; tunnel.py consumed unchanged) ---------- #
# QSET defaults to q9 but is set from --quantile-set in main() so the SAME driver reruns q9 and q1.
# QVER stamps the wide-grid protocol into every B2 filename so a legacy narrow-grid q9 record can
# never satisfy the new q9 skip (§10/§25). WD_GRID is the shared wide grid (WD_GRID_V2).
QSET = "q9"                                # probe-head quantile vector (features are qset-independent)
QVER = f"{QSET}__{PROBE_PROTOCOL_VERSION}"  # versioned tag for B2 filenames; recomputed in main()
QUANTILE_EPOCHS = 300                      # matches the v4 fslot probe fit
WD_GRID = WD_GRID_V2                        # wide weight-decay grid (shared source of truth)
PROBE_SEEDS = (0, 1, 2)                    # 3 independent probe-init runs; backbone is fixed per stage
RUN_TYPE = "probe_seed"                    # only the Linear init varies across the 3 runs (tunnel.py field)
# The fslot line carries NUM_LAYERS+1 = 14 points: Emb, L1..L12, then the POST-final-LN native-head
# input (extract_kout_features's final["fslot"]) as the tunnel's "last" reference (curve[-1]).
POST_LN_LABEL = "L12+LN"
LAYER_LABELS = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)] + [POST_LN_LABEL]
STAGE_COLOR = {STAGE0: "tab:blue", "stage1_ft_early": "tab:orange", "stage2_ft_late": "tab:red"}

# Compute artifacts (per-run probe records) keep the stageB/ location but carry QVER in the filename;
# B2 presentation (tunnels/figures/tables) routes into the browsable per-quantile domain-shift tree.
DOMAIN_SHIFT_ROOT = config.REPO_ROOT / "results" / "ft_specialization" / "domain_shift"
PROBE_DIR = OUT_ROOT / "probes"            # per (stage,target,seed): val/test curves + per-window loss
TUNNEL_DIR = DOMAIN_SHIFT_ROOT / QSET / "tunnels"   # per-stage BOOM (FT-ID/probe-ID) tunnel record
FIG_DIR = DOMAIN_SHIFT_ROOT / QSET / "figures"      # per-stage tunnel + per-target 3-stage overlays
TABLE_DIR = DOMAIN_SHIFT_ROOT / QSET / "tables"     # stage x target D + loss table


def target_status(tag: str) -> tuple[str, str]:
    """(pt_status, ft_status) for the TARGET. pt_status = in Chronos-2 pretraining; ft_status = this
    backbone's fine-tuning source (BOOM) or not."""
    pt = "PT-ID" if tag in PT_ID_TARGETS else "PT-OOD"
    ft = "FT-ID" if tag == FT_ID_TAG else "FT-OOD"
    return pt, ft


def target_windows(tag: str):
    """Rolling train/val/test windows for a target + its split-name map (matching the committed
    pretrained cache keys: PT-ID -> train/val/test; PT-OOD -> *_rolling)."""
    if tag in PT_ID_TARGETS:
        if config.DATASET_SET != "extended_v3_rolling":
            config.set_dataset_set("extended_v3_rolling")
        w = build_windows(tag)
        splits = {"train": ("X_train", "y_train"), "val": ("X_val", "y_val"),
                  "test": ("X_test", "y_test")}
    else:
        w = build_ood_rolling_windows(tag, C=C, H=H, seed=SEED)
        splits = {"train_rolling": ("X_train", "y_train"), "val_rolling": ("X_val", "y_val"),
                  "test_rolling": ("X_test", "y_test")}
    return w, splits


# --------------------------------------------------------------------------- #
# backbone stages
# --------------------------------------------------------------------------- #
@dataclass
class Stage:
    label: str
    ckpt_dir: str | None            # None -> pretrained (default get_pipeline)
    hash8: str | None               # None -> pretrained (default cache namespace)

    def cache_prefix(self, tag: str):
        # stage0 -> None (default _idf_prefix = committed pretrained caches); FT stages -> collision-proof
        return None if self.hash8 is None else ft_cache_prefix(tag, FT_SOURCE, self.label, self.hash8)


def load_stages(which) -> list[Stage]:
    """stage0_pretrained (no checkpoint) + the BOOM FT checkpoints read from the FT manifest.
    Fail loud if a requested FT stage is missing or its safetensors hash disagrees with the manifest."""
    if not FT_MANIFEST.exists():
        raise FileNotFoundError(f"missing BOOM FT manifest {FT_MANIFEST} — run the BOOM pilot first "
                                f"(`python -m probing.finetune --source boom`)")
    manifest = json.load(open(FT_MANIFEST))
    if manifest.get("source") != FT_SOURCE:
        raise RuntimeError(f"{FT_MANIFEST}: source={manifest.get('source')!r} != {FT_SOURCE!r}")
    stages: list[Stage] = []
    if STAGE0 in which:
        stages.append(Stage(STAGE0, None, None))
    for lbl, ck in manifest["checkpoints"].items():
        if lbl not in which:
            continue
        from pathlib import Path
        ckdir = Path(ck["checkpoint_dir"])
        if not (ckdir / "model.safetensors").exists():
            raise FileNotFoundError(f"{lbl}: checkpoint missing at {ckdir} (on $SCRATCH?) — "
                                    f"re-run the BOOM pilot or set FT_CKPT_ROOT")
        got = checkpoint_hash(ckdir)
        if got != ck["checkpoint_hash"]:
            raise RuntimeError(f"{lbl}: safetensors hash {got} != manifest {ck['checkpoint_hash']} "
                               f"— checkpoint changed since the pilot; caches would be mislabeled")
        stages.append(Stage(lbl, str(ckdir), ck["checkpoint_hash"]))
    missing = set(which) - {s.label for s in stages}
    if missing:
        raise RuntimeError(f"requested stages not found: {sorted(missing)}")
    return stages


def load_ft_pipeline(ckpt_dir):
    """Load a FROZEN Chronos-2 pipeline from a local FT checkpoint (eval, no grad) for extraction —
    separate from the pretrained get_pipeline singleton (which stage0 uses)."""
    from chronos import Chronos2Pipeline
    dev = _select_device()
    pipe = Chronos2Pipeline.from_pretrained(ckpt_dir, torch_dtype=torch.float32)
    pipe.model.to(dev).eval()
    for p in pipe.model.parameters():
        p.requires_grad_(False)
    return pipe


def _fslot_feats_stage(tag, split, X, y, pipeline, cache_prefix):
    """14-point fslot features {0..NUM_LAYERS-1 = pre-final-LN block slots, NUM_LAYERS = post-final-LN
    native-head input} for one (stage, target, split), via the extract_kout_features injection."""
    fk, final, _ = extract_kout_features(tag, split, X, y, horizon=H,
                                         pipeline=pipeline, cache_prefix=cache_prefix)
    feats = dict(fk["fslot"])
    feats[NUM_LAYERS] = final["fslot"]
    for i, arr in feats.items():
        assert np.ndim(arr) == 3 and arr.shape[1] == K, \
            f"{tag}/{split} L{i}: expected (n, {K}, 768) forecast slots, got {np.shape(arr)}"
    return feats


# --------------------------------------------------------------------------- #
# B1 extraction
# --------------------------------------------------------------------------- #
def run_extract(stages, targets, splits_filter=None):
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    report = []
    for stage in stages:
        pipe = None
        if stage.hash8 is not None:
            print(f"[B1] loading {stage.label} checkpoint ({stage.hash8}) -> {stage.ckpt_dir}")
            pipe = load_ft_pipeline(stage.ckpt_dir)
        else:
            print(f"[B1] {stage.label}: pretrained (default get_pipeline + committed cache namespace)")
        try:
            for tag in targets:
                pt, ft = target_status(tag)
                w, splits = target_windows(tag)
                for split_name, (xk, yk) in splits.items():
                    if splits_filter is not None and split_name not in splits_filter:
                        continue
                    X, y = w[xk], w[yk]
                    if len(X) == 0:
                        print(f"  [skip] {stage.label}/{SHORT[tag]}/{split_name}: 0 windows")
                        continue
                    feats = _fslot_feats_stage(tag, split_name, X, y, pipe, stage.cache_prefix(tag))
                    n = feats[0].shape[0]
                    report.append({"stage": stage.label, "target": tag, "short": SHORT[tag],
                                   "pt_status": pt, "ft_status": ft, "split": split_name,
                                   "n_windows": int(n), "n_points": len(feats),
                                   "cache_prefix": stage.cache_prefix(tag) or f"<default:{tag}>"})
                    print(f"  [ok] {stage.label}/{SHORT[tag]}/{split_name}: "
                          f"{pt}/{ft}  n={n}  fslot points={len(feats)} (expect {NUM_LAYERS + 1})")
        finally:
            if pipe is not None:
                del pipe
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    out = OUT_ROOT / ("extract_report_smoke.json" if splits_filter else "extract_report.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\n[B1] extracted {len(report)} (stage,target,split) cells -> {out}")
    return report


# --------------------------------------------------------------------------- #
# B2 — fresh probes (PRIMARY layerwise, probe-ID)
# --------------------------------------------------------------------------- #
def _role_split(tag):
    """role -> the split NAME the B1 cache was written under (PT-ID: train/val/test; PT-OOD:
    *_rolling), matching target_windows()."""
    return ({"train": "train", "val": "val", "test": "test"} if tag in PT_ID_TARGETS
            else {"train": "train_rolling", "val": "val_rolling", "test": "test_rolling"})


def _fslot_cache_path(stage, tag, split):
    """On-disk fslot cache path for (stage, target, split): FT-namespaced for the FT stages, default
    committed namespace (via _idf_prefix) for stage0. Needs config.DATASET_SET=extended_v3_rolling for
    a PT-ID stage0 target (target_windows sets it); PT-OOD tags are set-independent (__ood)."""
    prefix = stage.cache_prefix(tag) or _idf_prefix(tag)
    return _cache_path(prefix, split, None, f"K{K}_H{H}")


def _load_fslot(stage, tag, split, X, y):
    """14-point fslot features for (stage, target, split) from its B1 cache. CACHE-HIT ONLY — probe
    time is CPU/no-model, so a missing cache fails loud (it must never silently extract off the
    pretrained singleton for an FT stage)."""
    p = _fslot_cache_path(stage, tag, split)
    if not p.exists():
        raise FileNotFoundError(
            f"missing fslot cache {p.name} for {stage.label}/{SHORT[tag]}/{split} — run B1 first: "
            "`sbatch job_ft_stageB.sh --extract`")
    return _fslot_feats_stage(tag, split, X, y, pipeline=None, cache_prefix=stage.cache_prefix(tag))


def _run_stem(stage_label, tag, seed):
    return f"{stage_label}__{tag}__{QVER}__seed{seed}"


def _b2_run_compatible(path):
    """Skip predicate for a B2 per-run JSON: False if absent; True if present AND its recorded
    (quantile_set, probe_protocol_version, wd_grid) match this run; RAISE if present-but-incompatible
    so a stale/foreign result can never silently satisfy the new run (§10/§25)."""
    if not path.exists():
        return False
    meta = json.load(open(path))
    want = (QSET, PROBE_PROTOCOL_VERSION, [float(w) for w in WD_GRID])
    got = (meta.get("quantile_set"), meta.get("probe_protocol_version"), meta.get("wd_grid"))
    if got != want:
        raise RuntimeError(f"incompatible B2 result {path.name}: got (qset,protocol,wd_grid)={got} "
                           f"but this run wants {want}. Delete it or bump the protocol version.")
    return True


def _fit_one(stage_label, tag, f_tr, Ytr, f_va, Yva, f_te, Yte, series_test, seed, quantiles, device):
    """Fit ONE fresh shared-forecast-slot linear probe (wd on target-val, seed = Linear init) and
    score the target's own test. Writes the per-run curve JSON + per-window-loss NPZ. Takes feature
    dicts directly (no windowing/caches) so it is CPU-testable on synthetic features."""
    fitted = fit_shared_forecast_probe_explicit_val(
        f_tr, Ytr, f_va, Yva, quantiles=quantiles, epochs=QUANTILE_EPOCHS, wd_grid=WD_GRID,
        device=device, init_seed=seed)
    out, diag = predict_shared_forecast_probe(
        fitted, f_te, Yte, quantiles=quantiles, device=device, collect_test_window_loss=True)
    wl = np.stack([diag["test_window_loss"][i]                       # 14 rows (fslot: L0..L12 + post-LN)
                   for i in sorted(diag["test_window_loss"])]).astype(np.float64)
    pt, ft = target_status(tag)
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    stem = _run_stem(stage_label, tag, seed)
    np.savez(PROBE_DIR / f"{stem}.npz", window_loss=wl, series_test=np.asarray(series_test, np.int64))
    rec = {"experiment": "ft_specialization_stageB", "stage": stage_label, "target": tag,
           "short": SHORT[tag], "pt_status": pt, "ft_status": ft, "probe_status": "probe-ID",
           "ft_source": FT_SOURCE, "quantile_set": QSET, "readout": "fslot",
           "pooling_or_token_type": "forecast_slot", "run_type": RUN_TYPE, "run_seed": int(seed),
           "probe_protocol_version": PROBE_PROTOCOL_VERSION, "wd_grid": [float(w) for w in WD_GRID],
           "quantiles": [float(q) for q in quantiles], "Q": len(quantiles),
           "C": C, "H": H, "P": OUTPUT_PATCH_SIZE, "K": K,
           "val_loss_by_layer": val_curve_from_selection(
               {i: fitted[i]["selection"] for i in sorted(fitted)}, num_layers=len(fitted)),
           "test_loss_by_layer": [float(out[i]) for i in sorted(out)]}
    json.dump(rec, open(PROBE_DIR / f"{stem}.json", "w"), indent=2)
    return rec, wl


def run_probe(stages, targets, seeds, device):
    """B2 fresh probes: a shared-forecast-slot linear probe per (stage, target, seed), fit on the
    TARGET's own train (wd on target-val), scored on target-test. Windows built once per target; each
    stage reads its own B1 fslot cache. Idempotent (skips runs already on disk)."""
    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    quantiles = validate_quantiles(QUANTILE_SETS[QSET])
    for tag in targets:
        roles, w = _role_split(tag), None
        for stage in stages:
            pending = [s for s in seeds
                       if not _b2_run_compatible(PROBE_DIR / f"{_run_stem(stage.label, tag, s)}.json")]
            if not pending:
                print(f"  [skip] {stage.label}/{SHORT[tag]}: all seeds fit ({QVER})")
                continue
            if w is None:
                w, _ = target_windows(tag)          # sets config.DATASET_SET for PT-ID cache namespacing
            f_tr = _load_fslot(stage, tag, roles["train"], w["X_train"], w["y_train"])
            f_va = _load_fslot(stage, tag, roles["val"], w["X_val"], w["y_val"])
            f_te = _load_fslot(stage, tag, roles["test"], w["X_test"], w["y_test"])
            for seed in pending:
                print(f"\n[B2 probe] {stage.label} / {SHORT[tag]} ({'/'.join(target_status(tag))}) "
                      f"/ seed {seed}")
                _fit_one(stage.label, tag, f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"],
                         f_te, w["Y_test_traj"], w["series_test"], seed, quantiles, device)
                print(f"  [saved] {_run_stem(stage.label, tag, seed)}")
            del f_tr, f_va, f_te
            gc.collect()
        del w
        gc.collect()


# --------------------------------------------------------------------------- #
# B2 — per-stage BOOM tunnels (defined ONLY from the FT-ID/probe-ID validation curve)
# --------------------------------------------------------------------------- #
def _probe_run_curves(stage_label, tag, seed):
    """(val_curve list[14], window_loss (14, n), series_test) for one probe run — reads --probe output."""
    stem = _run_stem(stage_label, tag, seed)
    pj, pz = PROBE_DIR / f"{stem}.json", PROBE_DIR / f"{stem}.npz"
    if not (pj.exists() and pz.exists()):
        raise FileNotFoundError(f"missing probe run {stem} — run --probe first")
    z = np.load(pz)
    return json.load(open(pj))["val_loss_by_layer"], z["window_loss"], z["series_test"]


def _seed_mean_windows(runs):
    """Seed-averaged per-window losses (identical windows across runs — asserted)."""
    wls = [wl for _, wl, _ in runs]
    sids = [s for _, _, s in runs]
    assert all(w.shape == wls[0].shape for w in wls), "runs must share identical test windows"
    assert all(np.array_equal(s, sids[0]) for s in sids), "runs must share identical series ids"
    return np.mean(wls, axis=0), sids[0]


def _plot_runs(ax, curves_by_run, mean, std, color, label):
    x = np.arange(len(mean))
    for c in curves_by_run:
        ax.plot(x, c, "-", color=color, alpha=0.25, lw=0.9)
    mean, std = np.asarray(mean), np.asarray(std)
    ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
    ax.plot(x, mean, "o-", color=color, label=label)


def _tunnel_figure(rec, stage_label):
    ls = rec["l_start"]
    mv = np.asarray(rec["mean_val_loss_by_layer"])
    x = np.arange(len(mv)); last = len(x) - 1
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.axvspan(ls - 0.25, last + 0.25, color="tab:green", alpha=0.12,
               label=f"BOOM tunnel (first crossing 95%) [{LAYER_LABELS[ls]}, {LAYER_LABELS[last]}]")
    _plot_runs(ax, rec["val_loss_by_run"], rec["mean_val_loss_by_layer"],
               rec["std_val_loss_by_layer"], "tab:blue",
               f"validation, mean of {len(rec['run_seeds'])} runs (defines tunnel)")
    _plot_runs(ax, rec["test_loss_by_run"], rec["mean_test_loss_by_layer"],
               rec["std_test_loss_by_layer"], "tab:orange", "test, mean of runs (tunnel frozen)")
    ax.axhline((1 + rec["tolerance"]) * mv[-1], color="tab:blue", ls=":", lw=1,
               label=f"{1 + rec['tolerance']:.2f} x final-layer mean val loss")
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS[:len(x)])
    ax.set_xlabel("layer"); ax.set_ylabel("Chronos-2 quantile loss")
    ax.set_title(f"{stage_label}: BOOM FT-ID / probe-ID tunnel")
    ax.legend(fontsize=7)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"tunnel__{stage_label}__{QVER}.png", dpi=140)
    plt.close(fig)


def _stage_boom_tunnel(stage_label):
    """One tunnel per STAGE, defined ONLY from that stage's BOOM (FT-ID / probe-ID) MEAN validation
    curve (tunnel.py unchanged = first-crossing 95%); D_ID on BOOM test windows. This layer region is
    the lens later applied to the FT-OOD targets — never define a tunnel from FT-OOD data."""
    runs = [_probe_run_curves(stage_label, FT_ID_TAG, s) for s in PROBE_SEEDS]
    val_by_run = [v for v, _, _ in runs]
    test_by_run = [wl.mean(axis=1) for _, wl, _ in runs]
    wl_mean, sid = _seed_mean_windows(runs)
    rec = tunnel_record_multi(
        FT_ID_TAG, val_by_run, test_by_run, PROBE_SEEDS, run_type=RUN_TYPE,
        val_split_kind="temporal_rolling",
        extra={"experiment": "ft_specialization_stageB", "stage": stage_label, "ft_source": FT_SOURCE,
               "quantile_set": QSET, "readout": "fslot", "pooling_or_token_type": "forecast_slot",
               "probe_protocol_version": PROBE_PROTOCOL_VERSION, "wd_grid": [float(w) for w in WD_GRID],
               "defined_on": "BOOM (FT-ID / probe-ID) validation",
               "pt_status": "PT-OOD", "ft_status": "FT-ID", "probe_status": "probe-ID"})
    last = wl_mean.shape[0] - 1
    d = d_stat_boot(wl_mean, sid, rec["l_start"], last=last, B=config.BOOT_B, seed=SEED)
    rec["d_id_ci"] = list(d["ci"]); rec["n_clusters"] = d["n_clusters"]; rec["n_windows"] = d["n_windows"]
    rec["d_id_by_run"] = [float((t[-1] - t[rec["l_start"]]) / t[rec["l_start"]]) for t in test_by_run]
    TUNNEL_DIR.mkdir(parents=True, exist_ok=True)
    out = TUNNEL_DIR / f"tunnel__{stage_label}__{QVER}.json"
    json.dump(rec, open(out, "w"), indent=2)
    _tunnel_figure(rec, stage_label)
    print(f"  [{stage_label:>18}] BOOM tunnel [{LAYER_LABELS[rec['l_start']]}, {LAYER_LABELS[last]}]"
          f"  D_ID={rec['D_ID']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]  -> {out.name}")
    return rec


def run_tunnels(stages):
    print("[B2 tunnels] per-stage BOOM (FT-ID / probe-ID) first-crossing tunnels:")
    return {s.label: _stage_boom_tunnel(s.label) for s in stages}


# --------------------------------------------------------------------------- #
# B2 — layerwise curves + stage x target table (BOOM tunnel used as the lens)
# --------------------------------------------------------------------------- #
def _load_stage_tunnel(stage_label):
    p = TUNNEL_DIR / f"tunnel__{stage_label}__{QVER}.json"
    if not p.exists():
        raise FileNotFoundError(f"missing stage tunnel {p.name} — run --tunnels first")
    return json.load(open(p))


def _target_cell(stage_label, tag, l_start):
    """Seed-mean val/test curves + D at the STAGE's BOOM-tunnel entrance on this target's own test."""
    runs = [_probe_run_curves(stage_label, tag, s) for s in PROBE_SEEDS]
    val = np.array([v for v, _, _ in runs], float)
    test = np.array([wl.mean(axis=1) for _, wl, _ in runs], float)
    wl_mean, sid = _seed_mean_windows(runs)
    last = wl_mean.shape[0] - 1
    d = d_stat_boot(wl_mean, sid, l_start, last=last, B=config.BOOT_B, seed=SEED)
    return {"mean_val": val.mean(0), "mean_test": test.mean(0), "std_test": test.std(0),
            "l_start": l_start, "last": last, "d": d}


def _cell_row(stage_label, tag, cell):
    pt, ft = target_status(tag)
    ls, last, mt = cell["l_start"], cell["last"], np.asarray(cell["mean_test"])
    return {"experiment": "ft_specialization_stageB", "stage": stage_label, "target": tag,
            "short": SHORT[tag], "pt_status": pt, "ft_status": ft, "probe_status": "probe-ID",
            "ft_source": FT_SOURCE, "quantile_set": QSET, "readout": "fslot",
            "l_start": ls, "l_start_label": LAYER_LABELS[ls], "last_label": LAYER_LABELS[last],
            "loss_at_l_start": float(mt[ls]), "loss_at_last": float(mt[last]),
            "D_last_vs_lstart": cell["d"]["point"],
            "D_ci_lo": cell["d"]["ci"][0], "D_ci_hi": cell["d"]["ci"][1],
            "n_clusters": cell["d"]["n_clusters"], "n_windows": cell["d"]["n_windows"],
            "C": C, "H": H, "P": OUTPUT_PATCH_SIZE, "K": K}


def _overlay_figure(tag, stages, cells):
    """Per target: the 3 stages' seed-mean test curves overlaid (main forgetting view), each stage's
    BOOM tunnel entrance marked in the stage's colour."""
    x = np.arange(len(LAYER_LABELS)); pt, ft = target_status(tag)
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for stage in stages:
        c = cells[(stage.label, tag)]
        col = STAGE_COLOR.get(stage.label, "tab:gray")
        mt, st = np.asarray(c["mean_test"]), np.asarray(c["std_test"])
        ax.fill_between(x[:len(mt)], mt - st, mt + st, color=col, alpha=0.12)
        ax.plot(x[:len(mt)], mt, "o-", color=col, label=stage.label)
        ax.axvline(c["l_start"], color=col, ls=":", lw=1, alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(LAYER_LABELS)
    ax.set_xlabel("layer"); ax.set_ylabel("Chronos-2 quantile loss (test)")
    ax.set_title(f"{SHORT[tag]}  ({pt} / {ft} / probe-ID)  —  dotted = each stage's BOOM tunnel entrance")
    ax.legend(fontsize=8, title="backbone stage")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FIG_DIR / f"layerwise__{tag}__{QVER}.png", dpi=140)
    plt.close(fig)


def _write_table(rows):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / f"stageB_layerwise__{QVER}.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    json.dump(rows, open(TABLE_DIR / f"stageB_layerwise__{QVER}.json", "w"), indent=2)


def run_figures(stages, targets):
    """B2 layerwise curves: per (stage, target) seed-mean fslot test curve, D at the STAGE's BOOM
    tunnel entrance, a per-target 3-stage overlay, and the stage x target table. The lens (l_start)
    ALWAYS comes from the stage's BOOM FT-ID tunnel — never from FT-OOD data."""
    tunnels = {s.label: _load_stage_tunnel(s.label) for s in stages}
    cells, rows = {}, []
    for tag in targets:
        for stage in stages:
            cell = _target_cell(stage.label, tag, tunnels[stage.label]["l_start"])
            cells[(stage.label, tag)] = cell
            rows.append(_cell_row(stage.label, tag, cell))
        _overlay_figure(tag, stages, cells)
    _write_table(rows)
    print(f"[B2 figures] {len(rows)} (stage,target) cells -> "
          f"{TABLE_DIR}/stageB_layerwise__{QVER}.csv")
    return rows


def _dump_rows(stem, rows):
    """Write rows to <stem>.csv and <stem>.json (fail loud on nothing computed)."""
    if not rows:
        raise RuntimeError(f"no rows to write for {stem} — nothing was computed")
    with open(f"{stem}.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    json.dump(rows, open(f"{stem}.json", "w"), indent=2)


# --------------------------------------------------------------------------- #
# B3 — native catastrophic-forgetting evaluation (PRIMARY forgetting metric)
# --------------------------------------------------------------------------- #
# Each backbone stage's OWN native forecasting head is scored on IDENTICAL target-test windows (built
# once per target, seed 0, shared across stages), in ORIGINAL units — the model's own forecast, no
# probe trained. Primary quantity = the change vs pretrained (Delta = FT - pretrained; POSITIVE =
# worse = forgetting). BOOM (FT-ID) is the in-domain control (expected to IMPROVE, Delta < 0). The
# native quantile pass reuses run_fslot_forecasting_comparison's predict_quantiles -> (n, Q, H)
# transform, made STAGE-AWARE (stage0 = pretrained singleton; FT stages = the loaded checkpoint) and
# cached per checkpoint hash so two stages can never alias one native cache.
NATIVE_DIR = OUT_ROOT / "native"
NATIVE_IN_DIR = NATIVE_DIR / "inputs"          # per (stage,target) per-window MASE/WQL parts (feed B5)


def _native_cache_path(stage, tag, n_levels):
    """Per-STAGE native multi-quantile cache path. stage0 -> the default _idf_prefix namespace (so a
    PT-ID target reuses the committed pretrained native q-cache from run_fslot_forecasting_comparison
    --native-wql, identical key); FT stages -> the checkpoint-hash namespace. The split token matches
    the target's fslot test split (test / test_rolling), so nothing collides across stages."""
    prefix = stage.cache_prefix(tag) or _idf_prefix(tag)
    return CACHE_DIR / f"{prefix}__{_role_split(tag)['test']}__native_q{n_levels}_H{H}.npz"


def _native_quantiles_stage(stage, tag, X_test, quantiles, pipe_holder):
    """Multi-quantile native Chronos-2 forecast (n, Q, H) in RAW units for THIS stage's head. Cold ->
    load the stage pipeline once (lazily, reused across the stage's targets via pipe_holder) and run
    predict_quantiles; warm -> read the per-stage cache (context-tail guard fails loud on a
    re-window). Same call + reshape as run_fslot_forecasting_comparison._native_quantiles_raw,
    parameterized by the stage pipeline and cache namespace."""
    levels = [float(q) for q in quantiles]
    cache = _native_cache_path(stage, tag, len(levels))
    X = np.asarray(X_test, np.float32)
    if cache.exists():
        d = np.load(cache)
        if d["ctx_tail"].shape[0] == len(X) and np.allclose(d["ctx_tail"], X[:, -8:]):
            print(f"  [cache HIT]  {cache.name}")
            return d["quant"].astype(np.float64)
        raise RuntimeError(f"stale native cache {cache.name}: contexts changed since it was written "
                           "— delete it and re-run")
    if pipe_holder["pipe"] is None:            # load the stage's backbone once, on the first cold cell
        pipe_holder["pipe"] = (get_pipeline()[0] if stage.hash8 is None
                               else load_ft_pipeline(stage.ckpt_dir))
    print(f"  [native] {stage.label}/{SHORT[tag]}: {len(X)} windows x {len(levels)} quantiles (H={H})")
    qt, _mean = pipe_holder["pipe"].predict_quantiles(list(X), prediction_length=H,
                                                      quantile_levels=levels)
    quant = np.stack([q.reshape(H, len(levels)).cpu().numpy() for q in qt]).transpose(0, 2, 1)  # (n,Q,H)
    np.savez(cache, quant=quant.astype(np.float32), ctx_tail=X[:, -8:])
    print(f"  [saved]      {cache.name}  shape={quant.shape}")
    return quant.astype(np.float64)


def _native_cell(stage_label, hash8, tag, w, qr, quantiles):
    """ORIGINAL-scale native metrics for one (stage, target): MASE (median row vs the in-context m=24
    seasonal-naive denom), median MAE, and WQL (from the full quantile grid). PURE over (w, qr) so it
    is CPU/data-free testable. Returns (row, parts); parts feed the B5 paired bootstrap."""
    X_test = np.asarray(w["X_test"], np.float64)
    mu, s = _ctx_stats(X_test, w["meta"]["sigma_eps"])
    y_raw = _raw_future(w, mu, s)                                   # arcsinh inverse mu + s*sinh(z)
    denom = np.maximum(_mase_denominator(X_test), 1e-8)[:, None]
    qmid = median_index(quantiles)
    med = qr[:, qmid, :]
    mase_pw, mae_pw = _mase_pw(y_raw, med, denom), _mae_pw(y_raw, med)
    num, den = _wql_pw_parts(y_raw, qr, quantiles)                  # WQL = sum(num)/sum(den)
    sid = np.asarray(w["series_test"], np.int64)
    pt, ft = target_status(tag)
    row = {"experiment": "ft_specialization_stageB", "analysis": "B3_native_forgetting",
           "stage": stage_label, "target": tag, "short": SHORT[tag],
           "pt_status": pt, "ft_status": ft, "probe_status": "native_head",
           "method": "native_chronos2", "ft_source": FT_SOURCE, "quantile_set": QSET,
           "checkpoint_hash": (hash8 or "pretrained"), "seasonal_m": M_SEASON,
           "mase_denominator": "in_context_seasonal_naive_m24",
           "mase": round(float(mase_pw.mean()), 6), "median_mae": round(float(mae_pw.mean()), 6),
           "wql": round(float(num.sum() / max(den.sum(), 1e-12)), 6),
           "n_windows": int(sid.size), "n_series": int(np.unique(sid).size),
           "C": C, "H": H, "P": OUTPUT_PATCH_SIZE, "K": K}
    parts = {"mase_pw": mase_pw, "mae_pw": mae_pw, "wql_num": num, "wql_den": den, "series_test": sid}
    return row, parts


def run_native(stages, targets, device=None):
    """B3: score EACH stage's native head on identical target-test windows. GPU on a cold native
    cache; warm re-runs are CPU (cache-hit). Idempotent via the per-stage native cache. Writes the
    metrics table + per-window parts (native/inputs) for the B5 bootstrap. `device` is accepted for
    CLI symmetry but the pipeline's own device governs the forecast."""
    NATIVE_DIR.mkdir(parents=True, exist_ok=True); NATIVE_IN_DIR.mkdir(parents=True, exist_ok=True)
    config.set_dataset_set("extended_v3_rolling")     # FT geometry set (PT-OOD tags are set-independent)
    quantiles = validate_quantiles(QUANTILE_SETS[QSET])
    win = {tag: target_windows(tag)[0] for tag in targets}   # ONCE per target -> identical across stages
    rows = []
    for stage in stages:
        holder = {"pipe": None}
        try:
            for tag in targets:
                qr = _native_quantiles_stage(stage, tag, win[tag]["X_test"], quantiles, holder)
                row, parts = _native_cell(stage.label, stage.hash8, tag, win[tag], qr, quantiles)
                np.savez(NATIVE_IN_DIR / f"{stage.label}__{tag}__{QSET}.npz", **parts)
                rows.append(row)
                print(f"  [{stage.label:>18}/{SHORT[tag]:>11}]  MASE {row['mase']:.3f}  "
                      f"WQL {row['wql']:.3f}  MAE {row['median_mae']:.3f}  "
                      f"({row['pt_status']}/{row['ft_status']})")
        finally:
            if stage.hash8 is not None and holder["pipe"] is not None:     # free the FT checkpoint
                del holder["pipe"]; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    _dump_rows(str(NATIVE_DIR / f"native_metrics__{QSET}"), rows)
    print(f"[B3 native] {len(rows)} (stage,target) native cells -> "
          f"{NATIVE_DIR}/native_metrics__{QSET}.csv")
    return rows


# --------------------------------------------------------------------------- #
# B4 — frozen BOOM readout transfer (SECONDARY diagnostic, probe-OOD)
# --------------------------------------------------------------------------- #
# Where B2 fits a FRESH probe on each target (is the info recoverable?), B4 FREEZES the BOOM-trained
# readout and applies it unchanged to the 6 non-BOOM targets (does a BOOM readout TRANSFER without
# retraining?). The frozen probe is re-derived per (stage, seed) from the stage's BOOM train/val
# fslot caches — deterministic, so byte-identical to B2's BOOM probe — then predict-only on each
# target's test cache. NEVER fit on a target. Compared at the stage's BOOM tunnel entrance (the
# validation-defined lens), never at a target-test-selected layer.
TRANSFER_DIR = OUT_ROOT / "transfer"
TRANSFER_IN_DIR = TRANSFER_DIR / "inputs"
TRANSFER_TARGETS = tuple(t for t in ALL_TARGETS if t != FT_ID_TAG)   # the 6 non-BOOM (probe-OOD)


def _fit_boom_probe(stage, seed, quantiles, device):
    """Fit the shared-forecast-slot linear probe on the STAGE's BOOM train (wd on BOOM val, seed =
    Linear init). Reads BOOM's B1 fslot caches (cache-hit only). Deterministic => the same frozen
    probe B2 fit for this (stage, seed). Fits ONLY on BOOM — never touches a transfer target."""
    w, _ = target_windows(FT_ID_TAG)
    roles = _role_split(FT_ID_TAG)
    f_tr = _load_fslot(stage, FT_ID_TAG, roles["train"], w["X_train"], w["y_train"])
    f_va = _load_fslot(stage, FT_ID_TAG, roles["val"], w["X_val"], w["y_val"])
    return fit_shared_forecast_probe_explicit_val(
        f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"], quantiles=quantiles,
        epochs=QUANTILE_EPOCHS, wd_grid=WD_GRID, device=device, init_seed=seed)


def _transfer_stem(stage_label, tag, seed):
    return f"{stage_label}__{tag}__{QSET}__seed{seed}"


def _transfer_cell(stage, tag, fitted, seed, quantiles, device):
    """Apply the FROZEN BOOM probe to one target's test split — PREDICT-ONLY, never trains on the
    target. Saves the per-window loss (14, n) + series ids for the B5 bootstrap; returns the 14-point
    seed curve, the per-window losses, and the series ids."""
    w, _ = target_windows(tag)
    f_te = _load_fslot(stage, tag, _role_split(tag)["test"], w["X_test"], w["y_test"])
    out, diag = predict_shared_forecast_probe(fitted, f_te, w["Y_test_traj"], quantiles=quantiles,
                                             device=device, collect_test_window_loss=True)
    wl = np.stack([diag["test_window_loss"][i]
                   for i in sorted(diag["test_window_loss"])]).astype(np.float64)
    sid = np.asarray(w["series_test"], np.int64)
    TRANSFER_IN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(TRANSFER_IN_DIR / f"{_transfer_stem(stage.label, tag, seed)}.npz",
             window_loss=wl, series_test=sid)
    return [float(out[i]) for i in sorted(out)], wl, sid


def _transfer_runs(stage_label, tag, seeds):
    runs = []
    for s in seeds:
        p = TRANSFER_IN_DIR / f"{_transfer_stem(stage_label, tag, s)}.npz"
        if not p.exists():
            raise FileNotFoundError(f"missing transfer input {p.name} — run --transfer (B4) first")
        z = np.load(p)
        runs.append((z["window_loss"], z["series_test"]))
    return runs


def _aggregate_transfer(stages):
    """Seed-mean frozen-BOOM curve per (stage, target); D at the stage's BOOM tunnel entrance (the
    validation-defined lens). probe_status = probe-OOD (a BOOM readout on a non-BOOM target)."""
    tunnels = {s.label: _load_stage_tunnel(s.label) for s in stages}
    rows = []
    for stage in stages:
        ls = tunnels[stage.label]["l_start"]
        for tag in TRANSFER_TARGETS:
            runs = _transfer_runs(stage.label, tag, PROBE_SEEDS)
            wls = [wl for wl, _ in runs]
            sids = [sid for _, sid in runs]
            assert all(np.array_equal(sid, sids[0]) for sid in sids), \
                "transfer seeds must share identical test windows"
            wl_mean = np.mean(wls, axis=0)
            last = wl_mean.shape[0] - 1
            d = d_stat_boot(wl_mean, sids[0], ls, last=last, B=config.BOOT_B, seed=SEED)
            mt = wl_mean.mean(axis=1)
            pt, ft = target_status(tag)
            rows.append({"experiment": "ft_specialization_stageB",
                         "analysis": "B4_frozen_boom_transfer", "stage": stage.label, "target": tag,
                         "short": SHORT[tag], "pt_status": pt, "ft_status": ft,
                         "probe_status": "probe-OOD", "readout": "fslot_frozen_boom",
                         "ft_source": FT_SOURCE, "quantile_set": QSET,
                         "l_start": ls, "l_start_label": LAYER_LABELS[ls],
                         "last_label": LAYER_LABELS[last],
                         "loss_at_entrance": float(mt[ls]), "loss_at_ref": float(mt[last]),
                         "D_last_vs_lstart": d["point"], "D_ci_lo": d["ci"][0], "D_ci_hi": d["ci"][1],
                         "n_clusters": d["n_clusters"], "n_windows": d["n_windows"],
                         "C": C, "H": H, "P": OUTPUT_PATCH_SIZE, "K": K})
    return rows


def run_transfer(stages, seeds, device):
    """B4: per stage, FREEZE the BOOM probe (each seed) and score the 6 non-BOOM targets predict-only.
    Idempotent: a (stage, seed) whose 6 target inputs all exist skips the BOOM fit entirely."""
    TRANSFER_DIR.mkdir(parents=True, exist_ok=True); TRANSFER_IN_DIR.mkdir(parents=True, exist_ok=True)
    config.set_dataset_set("extended_v3_rolling")
    quantiles = validate_quantiles(QUANTILE_SETS[QSET])
    for stage in stages:
        for seed in seeds:
            pending = [t for t in TRANSFER_TARGETS
                       if not (TRANSFER_IN_DIR / f"{_transfer_stem(stage.label, t, seed)}.npz").exists()]
            if not pending:
                print(f"  [skip] {stage.label}/seed{seed}: all {len(TRANSFER_TARGETS)} transfers done")
                continue
            print(f"\n[B4 transfer] {stage.label} / frozen BOOM probe / seed {seed} "
                  f"-> {len(pending)} target(s)")
            fitted = _fit_boom_probe(stage, seed, quantiles, device)
            for tag in pending:
                _transfer_cell(stage, tag, fitted, seed, quantiles, device)
                print(f"  [saved] {_transfer_stem(stage.label, tag, seed)}  "
                      f"({'/'.join(target_status(tag))} / probe-OOD)")
            del fitted
            gc.collect()
    rows = _aggregate_transfer(stages)
    _dump_rows(str(TRANSFER_DIR / f"transfer_metrics__{QSET}"), rows)
    print(f"[B4 transfer] {len(rows)} (stage,target) frozen-BOOM cells -> "
          f"{TRANSFER_DIR}/transfer_metrics__{QSET}.csv")
    return rows


# --------------------------------------------------------------------------- #
# B5 — paired statistics + catastrophic-forgetting figures/tables
# --------------------------------------------------------------------------- #
FORGET_DIR = OUT_ROOT / "forgetting"
FT_STAGES = ("stage1_ft_early", "stage2_ft_late")   # compared against stage0_pretrained


def _load_native_parts(stage_label, tag):
    p = NATIVE_IN_DIR / f"{stage_label}__{tag}__{QSET}.npz"
    if not p.exists():
        raise FileNotFoundError(f"missing native parts {p.name} — run --native (B3) first")
    z = np.load(p)
    return {"mase_pw": z["mase_pw"], "mae_pw": z["mae_pw"], "wql_num": z["wql_num"],
            "wql_den": z["wql_den"], "series_test": z["series_test"]}


def _paired_native_stats(tag, stages):
    """Per target: absolute native MASE/WQL per stage + Delta vs pretrained (FT - pretrained; >0 =
    worse) with PAIRED cluster-bootstrap CIs. All stages MUST score identical windows (asserted on
    the series ids) so the ONE shared count matrix pairs the stages window-for-window."""
    parts = {s.label: _load_native_parts(s.label, tag) for s in stages}
    ref = parts[STAGE0]["series_test"]
    for lbl, p in parts.items():
        if not np.array_equal(p["series_test"], ref):
            raise RuntimeError(f"{tag}/{lbl}: native series ids differ from {STAGE0} — the three "
                               "stages must be scored on identical windows for a paired comparison")
    S, inv = _series_group(ref)
    M = cluster_bootstrap_counts(S, config.BOOT_B, SEED)               # ONE resample -> paired stages
    boot_mase = {lbl: _boot_mean(M, p["mase_pw"], inv, S) for lbl, p in parts.items()}
    boot_wql = {lbl: _boot_ratio(M, p["wql_num"], p["wql_den"], inv, S) for lbl, p in parts.items()}
    pt, ft = target_status(tag)
    row = {"analysis": "B5_native_forgetting", "target": tag, "short": SHORT[tag],
           "pt_status": pt, "ft_status": ft, "ft_source": FT_SOURCE, "quantile_set": QSET,
           "n_windows": int(ref.size), "n_series": int(S),
           "direction": "delta = FT - pretrained; POSITIVE = worse (forgetting)"}
    for lbl in (STAGE0, *FT_STAGES):
        mlo, mhi = ci_bounds(boot_mase[lbl])
        wlo, whi = ci_bounds(boot_wql[lbl])
        row[f"{lbl}__mase"] = round(float(parts[lbl]["mase_pw"].mean()), 6)
        row[f"{lbl}__mase_ci_lo"] = round(float(mlo), 6)
        row[f"{lbl}__mase_ci_hi"] = round(float(mhi), 6)
        row[f"{lbl}__wql"] = round(float(parts[lbl]["wql_num"].sum()
                                         / max(parts[lbl]["wql_den"].sum(), 1e-12)), 6)
        row[f"{lbl}__wql_ci_lo"] = round(float(wlo), 6)
        row[f"{lbl}__wql_ci_hi"] = round(float(whi), 6)
    for lbl in FT_STAGES:
        dm = boot_mase[lbl] - boot_mase[STAGE0]
        dw = boot_wql[lbl] - boot_wql[STAGE0]
        mlo, mhi = ci_bounds(dm)
        wlo, whi = ci_bounds(dw)
        row[f"{lbl}__dmase"] = round(float(dm.mean()), 6)
        row[f"{lbl}__dmase_ci_lo"] = round(float(mlo), 6)
        row[f"{lbl}__dmase_ci_hi"] = round(float(mhi), 6)
        row[f"{lbl}__dmase_sig"] = bool(mlo > 0 or mhi < 0)
        row[f"{lbl}__dwql"] = round(float(dw.mean()), 6)
        row[f"{lbl}__dwql_ci_lo"] = round(float(wlo), 6)
        row[f"{lbl}__dwql_ci_hi"] = round(float(whi), 6)
        row[f"{lbl}__dwql_sig"] = bool(wlo > 0 or whi < 0)
    return row


def _entrance_pw_b2(stage_label, tag, l_start):
    """B2 fresh-probe per-window loss AT the entrance layer (seed-mean), + series ids."""
    runs = [_probe_run_curves(stage_label, tag, s) for s in PROBE_SEEDS]
    wl_mean = np.mean([wl for _, wl, _ in runs], axis=0)
    return wl_mean[l_start], runs[0][2]


def _entrance_pw_b4(stage_label, tag, l_start):
    """B4 frozen-BOOM per-window loss AT the entrance layer (seed-mean), + series ids."""
    runs = _transfer_runs(stage_label, tag, PROBE_SEEDS)
    wl_mean = np.mean([wl for wl, _ in runs], axis=0)
    return wl_mean[l_start], runs[0][1]


def _b2_vs_b4_row(stage_label, tag, l_start):
    """Fresh target probe (B2) vs frozen BOOM readout (B4) at the stage's BOOM tunnel entrance, both
    on the target's OWN test windows (paired). b4_minus_b2 > 0 & significant = the info is recoverable
    by a fresh probe but the BOOM readout does not transfer."""
    b2_pw, sid2 = _entrance_pw_b2(stage_label, tag, l_start)
    b4_pw, sid4 = _entrance_pw_b4(stage_label, tag, l_start)
    if not np.array_equal(sid2, sid4):
        raise RuntimeError(f"{stage_label}/{tag}: B2 and B4 windows differ — cannot pair")
    S, inv = _series_group(sid2)
    M = cluster_bootstrap_counts(S, config.BOOT_B, SEED)
    boot2 = _boot_mean(M, b2_pw, inv, S)
    boot4 = _boot_mean(M, b4_pw, inv, S)
    diff = boot4 - boot2
    lo, hi = ci_bounds(diff)
    pt, ft = target_status(tag)
    return {"analysis": "B5_b2_vs_b4", "stage": stage_label, "target": tag, "short": SHORT[tag],
            "pt_status": pt, "ft_status": ft, "quantile_set": QSET,
            "l_start": l_start, "l_start_label": LAYER_LABELS[l_start],
            "b2_fresh_loss": round(float(b2_pw.mean()), 6),
            "b4_frozen_boom_loss": round(float(b4_pw.mean()), 6),
            "b4_minus_b2": round(float(diff.mean()), 6),
            "b4_minus_b2_ci_lo": round(float(lo), 6), "b4_minus_b2_ci_hi": round(float(hi), 6),
            "b4_minus_b2_sig": bool(lo > 0 or hi < 0), "n_series": int(S),
            "interpretation": "b4_minus_b2 > 0 & sig: info recoverable by a fresh probe but the BOOM "
                              "readout does not transfer"}


def _forgetting_heatmap(native_rows):
    """Rows = targets, cols = {ft_early, ft_late}, cell = ΔMASE vs pretrained (>0 = worse). Diverging
    colormap centered at 0; * = 95% paired CI excludes 0."""
    targets = [r["target"] for r in native_rows]
    Mv = np.array([[r[f"{st}__dmase"] for st in FT_STAGES] for r in native_rows], float)
    sig = np.array([[r[f"{st}__dmase_sig"] for st in FT_STAGES] for r in native_rows])
    vmax = float(np.abs(Mv).max()) or 1.0
    fig, ax = plt.subplots(figsize=(5.4, 0.62 * len(targets) + 1.8))
    im = ax.imshow(Mv, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(FT_STAGES))); ax.set_xticklabels(["ft_early\n(300 steps)", "ft_late\n(1000 steps)"])
    ax.set_yticks(range(len(targets))); ax.set_yticklabels([SHORT[t] for t in targets])
    for i in range(len(targets)):
        for j in range(len(FT_STAGES)):
            ax.text(j, i, f"{Mv[i, j]:+.3f}" + ("*" if sig[i, j] else ""), ha="center", va="center",
                    color=("white" if abs(Mv[i, j]) > 0.6 * vmax else "black"), fontsize=8)
    cb = fig.colorbar(im, ax=ax); cb.set_label("ΔMASE (FT − pretrained)\n>0 = worse = forgetting")
    ax.set_title("Native catastrophic forgetting: ΔMASE vs pretrained\n"
                 "(* = 95% paired cluster bootstrap CI excludes 0)", fontsize=10)
    FORGET_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FORGET_DIR / f"native_forgetting_heatmap__{QSET}.png", dpi=140)
    plt.close(fig)


def _native_bars(native_rows):
    """Per target: native MASE for pretrained / ft_early / ft_late with 95% CI whiskers."""
    stages = (STAGE0, *FT_STAGES)
    targets = [r["target"] for r in native_rows]
    x = np.arange(len(targets)); w = 0.26
    fig, ax = plt.subplots(figsize=(1.5 * len(targets) + 3, 5))
    for k, st in enumerate(stages):
        vals = [r[f"{st}__mase"] for r in native_rows]
        lo = [r[f"{st}__mase"] - r[f"{st}__mase_ci_lo"] for r in native_rows]
        hi = [r[f"{st}__mase_ci_hi"] - r[f"{st}__mase"] for r in native_rows]
        ax.bar(x + (k - 1) * w, vals, w, yerr=[lo, hi], capsize=2, label=st,
               color=STAGE_COLOR.get(st))
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in targets], rotation=20, ha="right")
    ax.set_ylabel("native MASE (original scale; lower = better)")
    ax.set_title(f"Native forecasting MASE per stage x target [{QSET}]  "
                 "(error bars = 95% paired cluster bootstrap)", fontsize=10)
    ax.legend(fontsize=8, title="backbone stage"); ax.grid(alpha=0.25, axis="y")
    FORGET_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FORGET_DIR / f"native_mase_bars__{QSET}.png", dpi=140)
    plt.close(fig)


def _b2_vs_b4_figure(b2b4_rows, stage_label="stage2_ft_late"):
    """Fresh probe (B2) vs frozen-BOOM readout (B4) at the BOOM tunnel entrance, for one stage."""
    sub = [r for r in b2b4_rows if r["stage"] == stage_label]
    targets = [r["target"] for r in sub]
    x = np.arange(len(targets)); w = 0.38
    fig, ax = plt.subplots(figsize=(1.3 * len(targets) + 3, 4.6))
    ax.bar(x - w / 2, [r["b2_fresh_loss"] for r in sub], w, label="B2 fresh target probe (probe-ID)",
           color="tab:green")
    ax.bar(x + w / 2, [r["b4_frozen_boom_loss"] for r in sub], w,
           label="B4 frozen BOOM probe (probe-OOD)", color="tab:red")
    ax.set_xticks(x); ax.set_xticklabels([SHORT[t] for t in targets], rotation=20, ha="right")
    ax.set_ylabel("Chronos-2 quantile loss at BOOM tunnel entrance (lower = better)")
    ax.set_title(f"{stage_label}: fresh probe (B2) vs frozen-BOOM readout (B4)\n"
                 "large B4−B2 gap = info recoverable but the BOOM readout does not transfer",
                 fontsize=10)
    ax.legend(fontsize=8); ax.grid(alpha=0.25, axis="y")
    FORGET_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(FORGET_DIR / f"b2_vs_b4_entrance__{QSET}.png", dpi=140)
    plt.close(fig)


def run_forgetting(stages, targets):
    """B5: paired cluster-bootstrap forgetting stats (native, B3) + the B2-vs-B4 readout comparison +
    the headline figures/tables. PURE aggregation over B2/B3/B4 outputs on disk (CPU)."""
    if {s.label for s in stages} != {STAGE0, *FT_STAGES}:
        raise RuntimeError("B5 forgetting needs all 3 stages (pretrained / ft_early / ft_late)")
    FORGET_DIR.mkdir(parents=True, exist_ok=True)
    native_rows = [_paired_native_stats(tag, stages) for tag in targets]
    _dump_rows(str(FORGET_DIR / f"native_forgetting__{QSET}"), native_rows)
    _forgetting_heatmap(native_rows)
    _native_bars(native_rows)
    tunnels = {s.label: _load_stage_tunnel(s.label) for s in stages}   # BOOM entrance = the lens
    b2b4_rows = [_b2_vs_b4_row(stage.label, tag, tunnels[stage.label]["l_start"])
                 for stage in stages for tag in TRANSFER_TARGETS]
    _dump_rows(str(FORGET_DIR / f"b2_vs_b4__{QSET}"), b2b4_rows)
    _b2_vs_b4_figure(b2b4_rows, "stage2_ft_late")
    print(f"[B5 forgetting] native_forgetting ({len(native_rows)} targets) + b2_vs_b4 "
          f"({len(b2b4_rows)} cells) + 3 figures -> {FORGET_DIR}")
    return native_rows, b2b4_rows


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extract", action="store_true", help="B1: extract fslot features (3 x 7 x 3)")
    ap.add_argument("--smoke", action="store_true",
                    help="B0: restrict to BOOM + the test split (fast 3-backbone verification)")
    ap.add_argument("--probe", action="store_true",
                    help="B2: fit fresh fslot probes (stage x target x seed) on each target's own train")
    ap.add_argument("--tunnels", action="store_true",
                    help="B2: per-stage BOOM (FT-ID/probe-ID) tunnels (needs --probe output)")
    ap.add_argument("--figures", action="store_true",
                    help="B2: layerwise curves + stage x target table (needs --tunnels output)")
    ap.add_argument("--native", action="store_true",
                    help="B3: each stage's NATIVE head on identical target-test windows (MASE/WQL); "
                         "GPU on a cold native cache")
    ap.add_argument("--transfer", action="store_true",
                    help="B4: frozen BOOM readout applied to the 6 non-BOOM targets (probe-OOD); "
                         "CPU / warm fslot caches")
    ap.add_argument("--forgetting", action="store_true",
                    help="B5: paired-bootstrap forgetting stats + B2-vs-B4 + figures (needs B2/B3/B4)")
    ap.add_argument("--stages", nargs="+", default=[STAGE0, "stage1_ft_early", "stage2_ft_late"],
                    help="backbone stages to run (default all 3)")
    ap.add_argument("--targets", nargs="+", default=list(ALL_TARGETS),
                    help="eval targets (default all 7)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PROBE_SEEDS),
                    help="probe-init seeds for --probe (default 0 1 2)")
    ap.add_argument("--device", default=None, help="probe-fit device (default: auto — GPU if present)")
    ap.add_argument("--quantile-set", default="q9", choices=sorted(QUANTILE_SETS),
                    help="B2 probe-head quantile vector: q9 (deciles) or q1 (median only). Reruns the "
                         "SAME fslot analysis; features are qset-independent. B3/B4/B5 keep q9.")
    return ap.parse_args(argv)


def _configure_qset(qset):
    """Rebind the B2 quantile-config globals from --quantile-set: QSET drives the probe head, QVER
    stamps the wide-grid protocol into B2 filenames, and the tunnel/figure/table dirs route into the
    browsable per-quantile domain-shift tree. Compute artifacts (PROBE_DIR) keep stageB/ + QVER names."""
    global QSET, QVER, TUNNEL_DIR, FIG_DIR, TABLE_DIR
    QSET = qset
    QVER = f"{QSET}__{PROBE_PROTOCOL_VERSION}"
    TUNNEL_DIR = DOMAIN_SHIFT_ROOT / QSET / "tunnels"
    FIG_DIR = DOMAIN_SHIFT_ROOT / QSET / "figures"
    TABLE_DIR = DOMAIN_SHIFT_ROOT / QSET / "tables"


def main(argv=None):
    args = _parse_args(argv)
    _configure_qset(args.quantile_set)
    stages = load_stages(args.stages)
    did = False
    if args.extract or args.smoke:
        targets = ["boom_hourly"] if args.smoke else args.targets
        splits_filter = {"test_rolling"} if args.smoke else None
        run_extract(stages, targets, splits_filter=splits_filter); did = True
    if args.probe:
        run_probe(stages, args.targets, args.seeds, _select_device(args.device)); did = True
    if args.tunnels:
        run_tunnels(stages); did = True
    if args.figures:
        run_figures(stages, args.targets); did = True
    if args.native:
        run_native(stages, args.targets, _select_device(args.device)); did = True
    if args.transfer:
        run_transfer(stages, args.seeds, _select_device(args.device)); did = True
    if args.forgetting:
        run_forgetting(stages, args.targets); did = True
    if not did:
        raise SystemExit("nothing to do — pass --extract/--smoke (B0/B1), --probe/--tunnels/--figures "
                         "(B2), --native (B3), --transfer (B4), or --forgetting (B5)")


if __name__ == "__main__":
    main()
