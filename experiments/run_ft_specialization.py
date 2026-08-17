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

THIS FILE (B0/B1): the ``--extract`` mode populates the shared-forecast-slot (fslot) feature caches
for 3 stages x 7 targets x {train,val,test}. stage0 reuses the committed pretrained caches (default
get_pipeline + default cache namespace); stage1/stage2 load the BOOM FT checkpoints and extract into
collision-proof ``IDF_<tag>__ft__boom__<stage>__<hash8>`` caches via the extract_kout_features
``pipeline=`` / ``cache_prefix=`` injection. ``--smoke`` restricts to BOOM/test for a fast 3-backbone
check. B2-B5 (fresh probes + tunnels, native MASE/WQL, frozen-BOOM transfer, bootstrap+figures) are
added after B1 is verified.

Run (GPU; OOD_TARGET_ROOT + HF offline set, e.g. via job_ft_pilot.sh's env):
    python -m experiments.run_ft_specialization --extract --smoke      # B0: BOOM/test, all 3 stages
    python -m experiments.run_ft_specialization --extract               # B1: full 3 x 7 x 3
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
from probing.config import NUM_LAYERS, SEED, OUTPUT_PATCH_SIZE
from probing.extraction import extract_kout_features, _cache_path, _idf_prefix
from probing.id_data import build_ood_rolling_windows, build_windows
from probing.finetune import ft_cache_prefix, checkpoint_hash, _select_device
from probing.probes import (QUANTILE_SETS, validate_quantiles,
                            fit_shared_forecast_probe_explicit_val, predict_shared_forecast_probe)
from probing.tunnel import d_stat_boot, tunnel_record_multi, val_curve_from_selection

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

# --- B2 config (fresh shared-forecast-slot LINEAR probes, q9; tunnel.py consumed unchanged) ------ #
QSET = "q9"                                # probe-head quantile vector (features are qset-independent)
QUANTILE_EPOCHS = 300                      # matches the v4 fslot probe fit
WD_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)   # weight decay selected on the TARGET's own validation split
PROBE_SEEDS = (0, 1, 2)                    # 3 independent probe-init runs; backbone is fixed per stage
RUN_TYPE = "probe_seed"                    # only the Linear init varies across the 3 runs (tunnel.py field)
# The fslot line carries NUM_LAYERS+1 = 14 points: Emb, L1..L12, then the POST-final-LN native-head
# input (extract_kout_features's final["fslot"]) as the tunnel's "last" reference (curve[-1]).
POST_LN_LABEL = "L12+LN"
LAYER_LABELS = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)] + [POST_LN_LABEL]
STAGE_COLOR = {STAGE0: "tab:blue", "stage1_ft_early": "tab:orange", "stage2_ft_late": "tab:red"}

PROBE_DIR = OUT_ROOT / "probes"            # per (stage,target,seed): val/test curves + per-window loss
TUNNEL_DIR = OUT_ROOT / "tunnels"          # per-stage BOOM (FT-ID/probe-ID) tunnel record + figure
FIG_DIR = OUT_ROOT / "figures"             # per-stage tunnel + per-target 3-stage overlays
TABLE_DIR = OUT_ROOT / "tables"            # stage x target D + loss table


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
    return f"{stage_label}__{tag}__{QSET}__seed{seed}"


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
                       if not (PROBE_DIR / f"{_run_stem(stage.label, tag, s)}.json").exists()]
            if not pending:
                print(f"  [skip] {stage.label}/{SHORT[tag]}: all seeds fit")
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
    fig.tight_layout(); fig.savefig(FIG_DIR / f"tunnel__{stage_label}__{QSET}.png", dpi=140)
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
               "defined_on": "BOOM (FT-ID / probe-ID) validation",
               "pt_status": "PT-OOD", "ft_status": "FT-ID", "probe_status": "probe-ID"})
    last = wl_mean.shape[0] - 1
    d = d_stat_boot(wl_mean, sid, rec["l_start"], last=last, B=config.BOOT_B, seed=SEED)
    rec["d_id_ci"] = list(d["ci"]); rec["n_clusters"] = d["n_clusters"]; rec["n_windows"] = d["n_windows"]
    rec["d_id_by_run"] = [float((t[-1] - t[rec["l_start"]]) / t[rec["l_start"]]) for t in test_by_run]
    TUNNEL_DIR.mkdir(parents=True, exist_ok=True)
    out = TUNNEL_DIR / f"tunnel__{stage_label}__{QSET}.json"
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
    p = TUNNEL_DIR / f"tunnel__{stage_label}__{QSET}.json"
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
    fig.tight_layout(); fig.savefig(FIG_DIR / f"layerwise__{tag}__{QSET}.png", dpi=140)
    plt.close(fig)


def _write_table(rows):
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLE_DIR / f"stageB_layerwise__{QSET}.csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader(); wtr.writerows(rows)
    json.dump(rows, open(TABLE_DIR / f"stageB_layerwise__{QSET}.json", "w"), indent=2)


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
          f"{TABLE_DIR}/stageB_layerwise__{QSET}.csv")
    return rows


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
    ap.add_argument("--stages", nargs="+", default=[STAGE0, "stage1_ft_early", "stage2_ft_late"],
                    help="backbone stages to run (default all 3)")
    ap.add_argument("--targets", nargs="+", default=list(ALL_TARGETS),
                    help="eval targets (default all 7)")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PROBE_SEEDS),
                    help="probe-init seeds for --probe (default 0 1 2)")
    ap.add_argument("--device", default=None, help="probe-fit device (default: auto — GPU if present)")
    return ap.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
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
    if not did:
        raise SystemExit("nothing to do — pass --extract/--smoke (B0/B1) or "
                         "--probe/--tunnels/--figures (B2). B3-B5 not built yet.")


if __name__ == "__main__":
    main()
