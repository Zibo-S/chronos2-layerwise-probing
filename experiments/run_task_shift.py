"""TASK-SHIFT driver — layerwise probing of Chronos-2 after CLASSIFICATION fine-tuning.

notes/PLAN.md, TASK-SHIFT. The companion of the DOMAIN-shift experiment (run_ft_specialization, BOOM
forecasting FT). Same frozen-then-fine-tuned Chronos-2, same layerwise-linear-probe lens, three backbone
stages (stage0_pretrained / stage1_cls_early / stage2_cls_late from probing.finetune_cls). THREE
classification sources are supported via ``--cls-source`` (probing.cls_data.CLS_SPECS):
  * forda       — easy/control (univariate, near-saturated).
  * uwave       — UWaveGestureLibrary (3-channel, 8-class; pretrained probe peaks ~L4 then declines).
  * handwriting — Handwriting (3-channel, 26-class; pretrained probe peaks ~L6 then declines).
Multivariate is handled per the old-UEA convention: each channel is encoded UNIVARIATELY and the
per-channel 768-d content-pooled vectors are CONCATENATED (c*768) — identical FT<->probe.

  Exp A (Plot A) — layerwise CLASSIFICATION probes. Fresh LINEAR Linear(c*768, C)+CE probes (seed bands)
                   on the source's features per stage x 14 layers (Emb..L12+LN) x seeds; wd on VAL, score
                   TEST accuracy. Does classification accessibility rise toward late layers after task-FT?
                   stage0 is the mandatory within-experiment control (the old UEA JSON is motivation, not
                   a numeric anchor — different estimator/point-count).
  Exp B (Plot B) — layerwise FORECASTING probes AFTER classification FT. The EXISTING fslot linear
                   forecasting probe (fit/predict_shared_forecast_probe), reused verbatim, on the
                   forecasting targets. Does forecasting accessibility FALL late once the task changed?
  Plot C         — DOMAIN vs TASK (per source): normalized Delta(fslot loss) vs stage0, BOOM-FT (read
                   stageB) beside this source's FordA/UWave/Handwriting-FT, on the SAME targets + probe.
  --compare      — CROSS-SOURCE DOMAIN-vs-TASK: BOOM domain-FT vs forda/uwave/handwriting task-FT on the
                   SAME targets (grid of Δ-vs-stage0 curves + a late-layer scalar heatmap; reads all
                   sources present on disk). Answers whether task difficulty / pretrained task
                   accessibility affects how much classification FT changes forecasting representations.
  --cka          — optional linear-CKA(stage0, FT) per layer on fixed examples (WHERE FT changed).

Do NOT engineer a U-shape — flat / no-specialization is a valid answer (PLAN stopping rule). Checkpoint
selection + FT validity use classification-side evidence ONLY (never a forecasting curve).

Modes (Exp B reuses run_ft_specialization's source-agnostic fslot machinery by IMPORT; this driver's
own Stage duck-types run_ft_specialization.Stage so _load_fslot works unchanged):
    python -m experiments.run_task_shift --cls-source uwave --extract           # C2 Exp-A cls features (GPU)
    python -m experiments.run_task_shift --cls-source uwave --forecast-extract  # C2 Exp-B fslot features (GPU)
    python -m experiments.run_task_shift --cls-source uwave --probe             # C3 cls probes (14 layers x seeds)
    python -m experiments.run_task_shift --cls-source uwave --forecast-probe    # C4 fslot probes (warm caches)
    python -m experiments.run_task_shift --cls-source uwave --figures [--cka]   # C5 Plots A/B/C (+CKA); CPU/login
    python -m experiments.run_task_shift --compare                              # C5 cross-source comparison; CPU/login

Everything is namespaced disjoint per source and from BOOM (results/task_shift_classification/<source>/;
caches carry source=<source>_cls). CPU/synthetic contracts: tests/test_task_shift.py (no GPU/model/download).
"""

from __future__ import annotations

import argparse
import gc
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from probing import config
from probing.config import NUM_LAYERS, OUTPUT_PATCH_SIZE
from probing.cls_data import CLS_SPECS, load_cls
from probing.extraction import extract_kout_features, _cache_path, _idf_prefix
from probing.finetune import ft_cache_prefix, checkpoint_hash
from probing.finetune_cls import cls_source_label, STAGE1, STAGE2
from probing.probes import (QUANTILE_SETS, validate_quantiles,
                            fit_linear_cls_probe_explicit_val, predict_linear_cls_probe,
                            fit_shared_forecast_probe_explicit_val, predict_shared_forecast_probe)
from probing.tunnel import val_curve_from_selection
# Exp-B reuse (source-agnostic helpers; NEVER read a hardcoded FT source): windowing + fslot cache I/O.
from experiments.run_ft_specialization import (target_windows, _role_split, _load_fslot,
                                               _fslot_feats_stage, load_ft_pipeline, SHORT,
                                               target_status)
from experiments.run_ft_specialization import H as FCAST_H, K as FCAST_K  # 64 / 4

STAGE0 = "stage0_pretrained"
CLS_STAGES = (STAGE0, STAGE1, STAGE2)
ALL_CLS_SOURCES = ("forda", "uwave", "handwriting")     # cross-source comparison roster (--compare)

CLS_HORIZON = 16                       # -> K=1 (num_output_patches=1); matches the FT forward
CLS_KTAG = f"K{math.ceil(CLS_HORIZON / OUTPUT_PATCH_SIZE)}_H{CLS_HORIZON}"     # "K1_H16"

FORECAST_TARGETS = ("boom_hourly", "m4_hourly", "coastal_ts")   # PT-OOD / PT-ID / PT-OOD
QSET = "q9"
QUANTILES = validate_quantiles(QUANTILE_SETS[QSET])
WD_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)
PROBE_SEEDS = (0, 1, 2)
CLS_EPOCHS = 300                       # probe-fit epochs (NOT the FT epochs)
CLS_LR = 1e-2                          # probe-fit lr
LATE_LAYERS = tuple(range(9, NUM_LAYERS + 1))   # L9..L12+LN — the "late" band for the comparison scalar

OUT_ROOT = config.REPO_ROOT / "results" / "task_shift_classification"
STAGEB_PROBE_DIR = config.REPO_ROOT / "results" / "ft_specialization" / "stageB" / "probes"  # BOOM Δ (Plot C)
COMPARE_DIR = OUT_ROOT / "comparison"

LAYER_LABELS = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)] + ["L12+LN"]   # 14 points
STAGE_COLOR = {STAGE0: "tab:blue", STAGE1: "tab:orange", STAGE2: "tab:red"}
STAGE_SHORT = {STAGE0: "pretrained", STAGE1: "cls-early", STAGE2: "cls-late"}

# --- source-dependent state (rebound by configure(); default = forda so the module imports/tests work) #
CLS_SOURCE = CLS_TAG = FT_SOURCE = CLS_STAGE0_PREFIX = None
SPEC = None
N_CLASSES = CHANNELS = None
SRC_OUT = CLS_PROBE_DIR = FCAST_PROBE_DIR = FIG_DIR = TABLE_DIR = FT_MANIFEST = None


def configure(src: str) -> None:
    """Bind every source-dependent global for classification source ``src`` (forda/uwave/handwriting).
    Output dirs are per-source subdirs (results/task_shift_classification/<src>/...), the FT manifest is
    results/ft_specialization/<src>_cls/manifest.json, and the stage0 cls cache prefix + FT_SOURCE carry
    the source so caches/checkpoints can never collide across sources or with BOOM."""
    global CLS_SOURCE, CLS_TAG, FT_SOURCE, SPEC, N_CLASSES, CHANNELS, CLS_STAGE0_PREFIX
    global SRC_OUT, CLS_PROBE_DIR, FCAST_PROBE_DIR, FIG_DIR, TABLE_DIR, FT_MANIFEST
    if src not in CLS_SPECS:
        raise ValueError(f"unknown cls source {src!r}; known: {sorted(CLS_SPECS)}")
    CLS_SOURCE = src
    CLS_TAG = src
    FT_SOURCE = cls_source_label(src)
    SPEC = CLS_SPECS[src]
    N_CLASSES = SPEC["n_classes"]
    CHANNELS = SPEC["channels"]
    CLS_STAGE0_PREFIX = f"IDF_{CLS_TAG}__task_shift_cls"     # stage0 cls cache prefix (set-independent)
    SRC_OUT = OUT_ROOT / src
    CLS_PROBE_DIR = SRC_OUT / "cls_probes"
    FCAST_PROBE_DIR = SRC_OUT / "forecast_probes"
    FIG_DIR = SRC_OUT / "figures"
    TABLE_DIR = SRC_OUT / "tables"
    FT_MANIFEST = config.REPO_ROOT / "results" / "ft_specialization" / FT_SOURCE / "manifest.json"


configure("forda")     # default binding at import time


# --------------------------------------------------------------------------- #
# backbone stages (own; duck-types run_ft_specialization.Stage via .cache_prefix)
# --------------------------------------------------------------------------- #
@dataclass
class Stage:
    label: str
    ckpt_dir: str | None            # None -> pretrained singleton
    hash8: str | None               # None -> pretrained (default cache namespace)

    def cache_prefix(self, tag: str):
        """FT fslot/cls cache prefix for `tag` (None for stage0). The forecasting fslot loaders in
        run_ft_specialization call exactly this signature, so passing a Stage of THIS class into
        _load_fslot works unchanged — with FT_SOURCE=<src>_cls, disjoint from BOOM and other sources."""
        return None if self.hash8 is None else ft_cache_prefix(tag, FT_SOURCE, self.label, self.hash8)


def load_stages(which) -> list[Stage]:
    """stage0 (no checkpoint) + the classification-FT checkpoints read from the finetune_cls manifest.
    Fail loud if a requested FT stage is missing (e.g. the validity gate refused to emit stage2) or its
    safetensors hash disagrees with the manifest."""
    order = [s for s in CLS_STAGES if s in which]
    stages: list[Stage] = []
    ft_needed = [s for s in order if s != STAGE0]
    manifest = None
    if ft_needed:
        if not FT_MANIFEST.exists():
            raise FileNotFoundError(f"missing {CLS_SOURCE}-cls manifest {FT_MANIFEST} — run the FT first "
                                    f"(`python -m probing.finetune_cls --cls-source {CLS_SOURCE}`)")
        manifest = json.load(open(FT_MANIFEST))
        if manifest.get("source") != FT_SOURCE:
            raise RuntimeError(f"{FT_MANIFEST}: source={manifest.get('source')!r} != {FT_SOURCE!r}")
    for lbl in order:
        if lbl == STAGE0:
            stages.append(Stage(STAGE0, None, None))
            continue
        if lbl not in manifest["checkpoints"]:
            raise RuntimeError(f"stage {lbl!r} not in manifest checkpoints {list(manifest['checkpoints'])} "
                               f"— the validity gate may have refused a late stage (see manifest['validity'])")
        ck = manifest["checkpoints"][lbl]
        ckdir = Path(ck["checkpoint_dir"])
        if not (ckdir / "model.safetensors").exists():
            raise FileNotFoundError(f"{lbl}: checkpoint missing at {ckdir} (on $SCRATCH? set FT_CKPT_ROOT)")
        got = checkpoint_hash(ckdir)
        if got != ck["checkpoint_hash"]:
            raise RuntimeError(f"{lbl}: safetensors hash {got} != manifest {ck['checkpoint_hash']} "
                               f"— checkpoint changed since FT; caches would be mislabeled")
        stages.append(Stage(lbl, str(ckdir), ck["checkpoint_hash"]))
    return stages


# --------------------------------------------------------------------------- #
# Exp A — classification features (14-pt content, per-channel concat) + probes
# --------------------------------------------------------------------------- #
def _cls_prefix(stage: Stage) -> str:
    return CLS_STAGE0_PREFIX if stage.hash8 is None else stage.cache_prefix(CLS_TAG)


def _ch_suffix(ch: int) -> str:
    """Per-channel cache suffix. Univariate (CHANNELS==1, e.g. FordA) keeps the BARE prefix so the
    single-channel cache key is byte-identical to the original FordA pipeline; multivariate sources get
    ``__ch{ch}`` so the c per-channel caches are disjoint."""
    return "" if CHANNELS == 1 else f"__ch{ch}"


def _cls_cache_path(stage: Stage, split: str, ch: int = 0) -> Path:
    return _cache_path(_cls_prefix(stage) + _ch_suffix(ch), split, None, CLS_KTAG)


def cls_feats(stage: Stage, split: str, X, y, pipe=None, allow_extract: bool = False) -> dict:
    """14-point CLASSIFICATION feature dict {0..12 pre-final-LN block content-pooled, 13 = L12+LN} for
    (stage, split), per-channel concatenated to (n, c*768). X is (n, c, L) (2-D (n, L) is promoted to a
    single channel). Fail loud if an FT-stage cache is missing (never extract an FT stage off the
    pretrained singleton). On cache HIT the pipeline is unused; on a stage0 MISS with allow_extract the
    frozen singleton extracts (correct)."""
    X = np.asarray(X)
    if X.ndim == 2:                      # (n, L) -> (n, 1, L) convenience for the univariate case
        X = X[:, None, :]
    channels = CHANNELS
    # guard: all per-channel caches must exist at probe time (they are extracted together)
    missing = [ch for ch in range(channels) if not _cls_cache_path(stage, split, ch).exists()]
    if missing:
        if not allow_extract:
            raise FileNotFoundError(
                f"{stage.label}: missing cls cache {_cls_cache_path(stage, split, missing[0]).name} — run "
                "--extract first (FT stages must never be extracted off the pretrained singleton)")
        if stage.ckpt_dir is not None and pipe is None:
            raise FileNotFoundError(
                f"{stage.label}: FT cls cache {_cls_cache_path(stage, split, missing[0]).name} missing and "
                "no FT pipeline supplied — extraction must use the FT checkpoint, not the singleton")
    per_layer = {i: [] for i in range(NUM_LAYERS + 1)}
    for ch in range(channels):
        feats, final, _ = extract_kout_features(CLS_TAG, split, X[:, ch, :], y, horizon=CLS_HORIZON,
                                                pipeline=pipe, cache_prefix=_cls_prefix(stage) + _ch_suffix(ch))
        for i in range(NUM_LAYERS):
            per_layer[i].append(feats["content"][i])      # keys 0..12 = Emb, L1..L12 (content-pooled)
        per_layer[NUM_LAYERS].append(final["content"])     # key 13 = L12+LN (post-final-LN content)
    return {i: np.concatenate(per_layer[i], axis=1) for i in per_layer}   # (n, c*768)


def run_cls_extract(stages):
    """C2 Exp-A: one extraction per (stage, split, channel) into the source-namespaced cls cache."""
    data = load_cls(CLS_TAG)
    splits = {"train": ("X_train", "y_train"), "val": ("X_val", "y_val"), "test": ("X_test", "y_test")}
    for stage in stages:
        pipe = None if stage.ckpt_dir is None else load_ft_pipeline(stage.ckpt_dir)
        for split, (xk, yk) in splits.items():
            cls_feats(stage, split, data[xk], data[yk], pipe=pipe, allow_extract=True)
            print(f"[A-extract] {stage.label}/{split}: cls features ({CHANNELS} ch) "
                  f"-> {_cls_cache_path(stage, split, 0).name}")
        del pipe
        gc.collect()


def _cls_probe_json(stage_label, seed):
    return CLS_PROBE_DIR / f"{stage_label}__seed{seed}.json"


def run_cls_probe(stages, seeds, device):
    """C3 Exp-A: fresh LINEAR Linear(c*768, C)+CE probe per (stage, seed) over 14 layers; wd on VAL,
    score TEST accuracy + CE. Backbone FROZEN (probes see only cached features). Idempotent."""
    CLS_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    data = load_cls(CLS_TAG)
    y_tr, y_va, y_te = data["y_train"], data["y_val"], data["y_test"]
    n_classes = N_CLASSES
    for stage in stages:
        pending = [s for s in seeds if not _cls_probe_json(stage.label, s).exists()]
        if not pending:
            print(f"  [skip] {stage.label}: all cls seeds fit")
            continue
        tr = cls_feats(stage, "train", data["X_train"], y_tr)
        va = cls_feats(stage, "val", data["X_val"], y_va)
        te = cls_feats(stage, "test", data["X_test"], y_te)
        for seed in pending:
            print(f"\n[A-probe] {stage.label} / seed {seed} (14 layers, wd-grid on val)")
            fitted = fit_linear_cls_probe_explicit_val(tr, y_tr, va, y_va, n_classes=n_classes,
                                                       epochs=CLS_EPOCHS, lr=CLS_LR, wd_grid=WD_GRID,
                                                       device=device, init_seed=seed)
            acc, diag = predict_linear_cls_probe(fitted, te, y_te, device=device,
                                                 collect_test_correct=False, collect_test_ce=True)
            rec = {"experiment": "task_shift_cls", "cls_source": CLS_SOURCE, "aeon_name": SPEC["aeon_name"],
                   "stage": stage.label, "seed": int(seed), "checkpoint_hash": stage.hash8,
                   "n_classes": n_classes, "channels": CHANNELS, "layer_labels": LAYER_LABELS,
                   "test_acc_by_layer": [float(acc[i]) for i in sorted(acc)],
                   "test_ce_by_layer": [float(diag["test_ce"][i]) for i in sorted(diag["test_ce"])],
                   "chosen_wd_by_layer": [fitted[i]["selection"]["chosen_wd"] for i in sorted(fitted)]}
            _cls_probe_json(stage.label, seed).write_text(json.dumps(rec, indent=2))
            print(f"  [saved] {_cls_probe_json(stage.label, seed).name}  "
                  f"acc@Emb={acc[0]:.3f} acc@L12+LN={acc[NUM_LAYERS]:.3f}")
        del tr, va, te
        gc.collect()


# --------------------------------------------------------------------------- #
# Exp B — forecasting features (fslot) + probes (reuse the fslot machinery)
# --------------------------------------------------------------------------- #
def run_fcast_extract(stages, targets):
    """C2 Exp-B: fslot features (14-pt, K=4, H=64) on the forecasting targets. stage0 reuses committed
    pretrained caches (cache_prefix None -> _idf_prefix); FT stages write <src>_cls-namespaced caches."""
    for stage in stages:
        pipe = None if stage.ckpt_dir is None else load_ft_pipeline(stage.ckpt_dir)
        for tag in targets:
            roles = _role_split(tag)
            w, _ = target_windows(tag)              # sets config.DATASET_SET for PT-ID cache namespacing
            for role, split in roles.items():
                X, y = w[f"X_{role}"], w[f"y_{role}"]
                if len(X) == 0:
                    continue
                _fslot_feats_stage(tag, split, X, y, pipe, stage.cache_prefix(tag))
                print(f"[B-extract] {stage.label}/{SHORT[tag]}/{split}: fslot ({len(X)} windows)")
            del w
            gc.collect()
        del pipe
        gc.collect()


def _fcast_probe_stem(stage_label, tag, seed):
    return f"{stage_label}__{tag}__{QSET}__seed{seed}"


def run_fcast_probe(stages, targets, seeds, device):
    """C4 Exp-B: fresh fslot linear forecasting probe per (stage, target, seed) — fit on the target's
    own train (wd on target-val), score target-test quantile loss. Reuses fit/predict_shared_forecast_probe
    VERBATIM (no new head). Idempotent."""
    FCAST_PROBE_DIR.mkdir(parents=True, exist_ok=True)
    for tag in targets:
        roles, w = _role_split(tag), None
        for stage in stages:
            pending = [s for s in seeds
                       if not (FCAST_PROBE_DIR / f"{_fcast_probe_stem(stage.label, tag, s)}.json").exists()]
            if not pending:
                print(f"  [skip] {stage.label}/{SHORT[tag]}: all fslot seeds fit")
                continue
            if w is None:
                w, _ = target_windows(tag)
            f_tr = _load_fslot(stage, tag, roles["train"], w["X_train"], w["y_train"])
            f_va = _load_fslot(stage, tag, roles["val"], w["X_val"], w["y_val"])
            f_te = _load_fslot(stage, tag, roles["test"], w["X_test"], w["y_test"])
            for seed in pending:
                print(f"\n[B-probe] {stage.label} / {SHORT[tag]} / seed {seed}")
                fitted = fit_shared_forecast_probe_explicit_val(
                    f_tr, w["Y_train_traj"], f_va, w["Y_val_traj"], quantiles=QUANTILES,
                    epochs=CLS_EPOCHS, wd_grid=WD_GRID, device=device, init_seed=seed)
                out = predict_shared_forecast_probe(fitted, f_te, w["Y_test_traj"],
                                                    quantiles=QUANTILES, device=device)
                pt, ft = target_status(tag)
                rec = {"experiment": "task_shift_forecast", "cls_source": CLS_SOURCE,
                       "stage": stage.label, "target": tag, "short": SHORT[tag], "pt_status": pt,
                       "ft_status": ft, "ft_source": FT_SOURCE, "quantile_set": QSET, "readout": "fslot",
                       "run_seed": int(seed), "H": FCAST_H, "K": FCAST_K, "layer_labels": LAYER_LABELS,
                       "val_loss_by_layer": val_curve_from_selection(
                           {i: fitted[i]["selection"] for i in sorted(fitted)}, num_layers=len(fitted)),
                       "test_loss_by_layer": [float(out[i]) for i in sorted(out)]}
                (FCAST_PROBE_DIR / f"{_fcast_probe_stem(stage.label, tag, seed)}.json").write_text(
                    json.dumps(rec, indent=2))
                print(f"  [saved] {_fcast_probe_stem(stage.label, tag, seed)}")
            del f_tr, f_va, f_te
            gc.collect()
        del w
        gc.collect()


# --------------------------------------------------------------------------- #
# aggregation helpers
# --------------------------------------------------------------------------- #
def _stack_seed_curves(json_glob, key):
    """{stage: (n_seed, 14) array} of `key` curves from probe JSONs matching a stage->paths map."""
    out = {}
    for stage, paths in json_glob.items():
        curves = [np.asarray(json.load(open(p))[key], float) for p in paths]
        if curves:
            out[stage] = np.stack(curves)
    return out


def _cls_curves(stages):
    g = {s.label: sorted(CLS_PROBE_DIR.glob(f"{s.label}__seed*.json")) for s in stages}
    return _stack_seed_curves(g, "test_acc_by_layer")


def _fcast_curves(stages, tag):
    g = {s.label: sorted(FCAST_PROBE_DIR.glob(f"{s.label}__{tag}__{QSET}__seed*.json")) for s in stages}
    return _stack_seed_curves(g, "test_loss_by_layer")


def _fcast_curves_dir(fcast_dir, tag):
    """{stage_label: (n_seed, 14)} test-loss curves read from an ARBITRARY forecast_probes dir (used by
    the cross-source comparison, which reads every source's dir regardless of the active configure())."""
    g = {lbl: sorted(Path(fcast_dir).glob(f"{lbl}__{tag}__{QSET}__seed*.json")) for lbl in CLS_STAGES}
    return _stack_seed_curves(g, "test_loss_by_layer")


def _delta_vs_stage0(curves):
    """mean-over-seed Delta(loss) vs stage0 per layer: {stage: (14,)}. None if stage0 absent."""
    if STAGE0 not in curves:
        return None
    base = curves[STAGE0].mean(0)
    return {st: c.mean(0) - base for st, c in curves.items() if st != STAGE0}


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
def _mean_band(ax, curves, color, label):
    m, sd = curves.mean(0), curves.std(0)
    x = np.arange(curves.shape[1])
    ax.plot(x, m, "-o", ms=3, color=color, label=label)
    if curves.shape[0] > 1:
        ax.fill_between(x, m - sd, m + sd, color=color, alpha=0.18)


def _xaxis(ax):
    ax.set_xticks(np.arange(len(LAYER_LABELS)))
    ax.set_xticklabels(LAYER_LABELS, rotation=45, ha="right", fontsize=7)


def make_plot_a(stages):
    curves = _cls_curves(stages)
    if not curves:
        print("[figures] Plot A skipped (no cls probe JSONs — run --probe)"); return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s in stages:
        if s.label in curves:
            _mean_band(ax, curves[s.label], STAGE_COLOR[s.label],
                       f"{STAGE_SHORT[s.label]} (n={curves[s.label].shape[0]})")
    ax.axhline(1.0 / N_CLASSES, ls=":", c="gray", lw=1, label=f"chance (1/{N_CLASSES})")
    _xaxis(ax)
    ax.set_ylabel(f"{SPEC['aeon_name']} test accuracy"); ax.set_xlabel("probed representation")
    ax.set_title(f"Exp A — {SPEC['aeon_name']} layerwise CLASSIFICATION accessibility (seed bands)")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG_DIR / "plotA_classification_accessibility.png", dpi=150); plt.close(fig)
    print(f"[figures] Plot A -> {FIG_DIR/'plotA_classification_accessibility.png'}")


def make_plot_b(stages, targets):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    present = [t for t in targets if _fcast_curves(stages, t)]
    if not present:
        print("[figures] Plot B skipped (no forecast probe JSONs — run --forecast-probe)"); return
    fig, axes = plt.subplots(1, len(present), figsize=(5.2 * len(present), 4.5), squeeze=False)
    for ax, tag in zip(axes[0], present):
        curves = _fcast_curves(stages, tag)
        for s in stages:
            if s.label in curves:
                _mean_band(ax, curves[s.label], STAGE_COLOR[s.label], STAGE_SHORT[s.label])
        _xaxis(ax)
        ax.set_title(f"{SHORT[tag]} ({'/'.join(target_status(tag))})")
        ax.set_ylabel(f"fslot quantile loss ({QSET}, lower=better)")
    fig.suptitle(f"Exp B — forecasting accessibility after {SPEC['aeon_name']} classification FT")
    axes[0][0].legend(fontsize=8); fig.tight_layout()
    fig.savefig(FIG_DIR / "plotB_forecast_accessibility.png", dpi=150); plt.close(fig)
    print(f"[figures] Plot B -> {FIG_DIR/'plotB_forecast_accessibility.png'}")


# BOOM (domain-shift) stage-B probes use ft_early/ft_late labels; alias them to THIS run's cls
# early/late so Plot C can place the two FT conditions side by side with the same stage colors.
_BOOM_STAGES = (STAGE0, "stage1_ft_early", "stage2_ft_late")
_BOOM_ALIAS = {"stage1_ft_early": STAGE1, "stage2_ft_late": STAGE2}


def _boom_delta(tag):
    """BOOM-FT (DOMAIN shift) normalized Delta(fslot loss) vs stage0 for `tag`, read from the stage-B B2
    probes (labelled ft_early/ft_late). Keys are ALIASED to this run's cls early/late so the comparison
    shares colors. None if the stage-B probes for `tag` are absent."""
    g = {}
    for st in _BOOM_STAGES:
        paths = sorted(STAGEB_PROBE_DIR.glob(f"{st}__{tag}__{QSET}__seed*.json"))
        if paths:
            g[st] = paths
    curves = _stack_seed_curves(g, "test_loss_by_layer")
    if STAGE0 not in curves:
        return None
    base = curves[STAGE0].mean(0)
    return {_BOOM_ALIAS[st]: c.mean(0) - base for st, c in curves.items() if st != STAGE0}


def make_plot_c(stages, targets):
    """Per-source Domain (BOOM-FT, from stageB) vs Task (this source's FT): normalized Delta vs stage0.
    Only targets present under BOTH conditions are drawn; if none overlap, the panel is honestly skipped."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for tag in targets:
        task = _delta_vs_stage0(_fcast_curves(stages, tag))
        boom = _boom_delta(tag)
        if task and boom:
            rows.append((tag, boom, task))
    if not rows:
        print("[figures] Plot C skipped (need BOTH BOOM-FT stageB and this-source deltas on the same target)")
        return
    fig, axes = plt.subplots(len(rows), 2, figsize=(11, 4.0 * len(rows)), squeeze=False)
    for r, (tag, boom, task) in enumerate(rows):
        for c, (title, delta) in enumerate([("DOMAIN shift (BOOM-FT)", boom),
                                            (f"TASK shift ({SPEC['aeon_name']}-cls-FT)", task)]):
            ax = axes[r][c]
            for st, d in delta.items():
                ax.plot(np.arange(len(d)), d, "-o", ms=3, color=STAGE_COLOR[st], label=STAGE_SHORT[st])
            ax.axhline(0, ls=":", c="gray", lw=1)
            _xaxis(ax)
            ax.set_title(f"{SHORT[tag]} — {title}")
            ax.set_ylabel("Δ fslot loss vs stage0")
            if r == 0 and c == 0:
                ax.legend(fontsize=8)
    fig.suptitle(f"Plot C — DOMAIN vs TASK ({SPEC['aeon_name']}) specialization (normalized Δ vs pretrained)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "plotC_domain_vs_task_delta.png", dpi=150); plt.close(fig)
    print(f"[figures] Plot C -> {FIG_DIR/'plotC_domain_vs_task_delta.png'}")


# --------------------------------------------------------------------------- #
# CROSS-SOURCE comparison (§10): BOOM domain-FT vs forda/uwave/handwriting task-FT
# --------------------------------------------------------------------------- #
def _condition_delta(condition, tag):
    """Normalized Δ(fslot loss) vs stage0 for a comparison CONDITION on forecasting target `tag`:
    {early:.., late:..} or None. condition == 'boom' -> stageB domain-FT; else a cls source name whose
    forecast_probes dir is read directly (independent of the active configure())."""
    if condition == "boom":
        return _boom_delta(tag)
    return _delta_vs_stage0(_fcast_curves_dir(OUT_ROOT / condition / "forecast_probes", tag))


def _late_scalar(delta, stage):
    """Mean Δ over the LATE layer band (L9..L12+LN) for `stage` in a delta dict — the headline scalar
    'did FT damage late-layer forecasting accessibility'. NaN if the stage is absent."""
    if not delta or stage not in delta:
        return float("nan")
    d = np.asarray(delta[stage], float)
    return float(d[list(LATE_LAYERS)].mean())


def make_comparison(targets):
    """Cross-source DOMAIN-vs-TASK comparison. Grid of Δ-vs-stage0 curves (rows = forecasting target,
    cols = FT condition = BOOM domain + each cls source present) + a late-layer scalar heatmap +
    a machine-readable table. Reads every source's forecast_probes on disk; conditions/targets without
    data are drawn as an annotated empty cell (grid) / left NaN (heatmap)."""
    conditions = ["boom"] + list(ALL_CLS_SOURCES)
    cond_label = {"boom": "BOOM (domain)", **{s: f"{CLS_SPECS[s]['aeon_name']} (task)" for s in ALL_CLS_SOURCES}}
    # which conditions have ANY data (across the requested targets)?
    deltas = {(c, t): _condition_delta(c, t) for c in conditions for t in targets}
    live_conds = [c for c in conditions if any(deltas[(c, t)] for t in targets)]
    if not live_conds:
        print("[compare] skipped (no BOOM stageB or task forecast probes found on disk)"); return
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)
    (COMPARE_DIR / "figures").mkdir(exist_ok=True)
    (COMPARE_DIR / "tables").mkdir(exist_ok=True)

    # --- grid of Δ curves (rows = target, cols = live condition) ---
    nrow, ncol = len(targets), len(live_conds)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.6 * nrow), squeeze=False)
    for r, tag in enumerate(targets):
        for c, cond in enumerate(live_conds):
            ax = axes[r][c]
            delta = deltas[(cond, tag)]
            if delta:
                for st, d in delta.items():
                    ax.plot(np.arange(len(d)), d, "-o", ms=3, color=STAGE_COLOR[st], label=STAGE_SHORT[st])
                ax.axhline(0, ls=":", c="gray", lw=1)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes,
                        fontsize=9, color="gray")
            _xaxis(ax)
            if r == 0:
                ax.set_title(cond_label[cond], fontsize=10)
            if c == 0:
                ax.set_ylabel(f"{SHORT.get(tag, tag)}\nΔ fslot loss vs stage0", fontsize=9)
            if r == 0 and c == 0 and delta:
                ax.legend(fontsize=8)
    fig.suptitle("DOMAIN vs TASK — normalized Δ(fslot loss) vs pretrained, per forecasting target",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(COMPARE_DIR / "figures" / "domain_vs_task_grid.png", dpi=150); plt.close(fig)
    print(f"[compare] grid -> {COMPARE_DIR/'figures'/'domain_vs_task_grid.png'}")

    # --- late-layer scalar heatmap (rows = condition x {early,late}, cols = target) ---
    row_keys = [(c, st) for c in live_conds for st in (STAGE1, STAGE2)]
    M = np.full((len(row_keys), len(targets)), np.nan)
    for i, (cond, st) in enumerate(row_keys):
        for j, tag in enumerate(targets):
            M[i, j] = _late_scalar(deltas[(cond, tag)], st)
    vmax = np.nanmax(np.abs(M)) if np.isfinite(M).any() else 1.0
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(targets), 0.6 + 0.5 * len(row_keys)))
    im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(targets))); ax.set_xticklabels([SHORT.get(t, t) for t in targets])
    ax.set_yticks(range(len(row_keys)))
    ax.set_yticklabels([f"{cond_label[c]} · {STAGE_SHORT[st]}" for c, st in row_keys], fontsize=8)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                ax.text(j, i, f"{M[i, j]:+.3f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j]) > 0.5 * vmax else "black")
    ax.set_title(f"Late-layer (L9..L12+LN) mean Δ fslot loss vs stage0\n(+ = FT worsened late forecasting)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(COMPARE_DIR / "figures" / "late_layer_delta_heatmap.png", dpi=150); plt.close(fig)
    print(f"[compare] heatmap -> {COMPARE_DIR/'figures'/'late_layer_delta_heatmap.png'}")

    # --- machine-readable table ---
    table = []
    for cond in live_conds:
        for tag in targets:
            delta = deltas[(cond, tag)]
            if not delta:
                continue
            for st in (STAGE1, STAGE2):
                if st in delta:
                    table.append({"condition": cond, "condition_label": cond_label[cond],
                                  "target": tag, "short": SHORT.get(tag, tag), "stage": st,
                                  "late_layers": list(LATE_LAYERS),
                                  "late_mean_delta_vs_stage0": _late_scalar(delta, st),
                                  "delta_by_layer": [float(v) for v in delta[st]]})
    (COMPARE_DIR / "tables" / "domain_vs_task__q9.json").write_text(json.dumps(table, indent=2))
    print(f"[compare] table -> {COMPARE_DIR/'tables'/'domain_vs_task__q9.json'} ({len(table)} rows)")


def _linear_cka(X, Y):
    """Linear CKA between two (n, d) feature matrices (centered). 1 = identical representation."""
    X = np.asarray(X, float) - np.asarray(X, float).mean(0, keepdims=True)
    Y = np.asarray(Y, float) - np.asarray(Y, float).mean(0, keepdims=True)
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    denom = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / denom) if denom > 0 else float("nan")


def make_cka(stages):
    """Optional: linear CKA(stage0, FT) per layer on the fixed TEST features (WHERE FT changed)."""
    labels = [s.label for s in stages]
    if STAGE0 not in labels:
        print("[CKA] skipped (needs stage0 in --stages)"); return
    data = load_cls(CLS_TAG)
    ref = cls_feats([s for s in stages if s.label == STAGE0][0], "test", data["X_test"], data["y_test"])
    fig, ax = plt.subplots(figsize=(8, 4.5))
    rows = []
    for s in stages:
        if s.label == STAGE0:
            continue
        ft = cls_feats(s, "test", data["X_test"], data["y_test"])
        cka = [_linear_cka(ref[i], ft[i]) for i in sorted(ref)]
        ax.plot(np.arange(len(cka)), cka, "-o", ms=3, color=STAGE_COLOR[s.label], label=STAGE_SHORT[s.label])
        rows.append({"stage": s.label, "cka_by_layer": cka})
    _xaxis(ax)
    ax.set_ylabel("linear CKA vs stage0")
    ax.set_title(f"{SPEC['aeon_name']} — representation drift under classification FT")
    ax.legend(fontsize=8); fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "cka_vs_stage0.png", dpi=150); plt.close(fig)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    (TABLE_DIR / "cka_vs_stage0.json").write_text(json.dumps(rows, indent=2))
    print(f"[CKA] -> {FIG_DIR/'cka_vs_stage0.png'}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--cls-source", default="forda", choices=sorted(CLS_SPECS),
                    help="classification source (forda | uwave | handwriting)")
    ap.add_argument("--extract", action="store_true", help="C2 Exp-A: cls features (GPU)")
    ap.add_argument("--forecast-extract", action="store_true", help="C2 Exp-B: fslot features (GPU)")
    ap.add_argument("--probe", action="store_true", help="C3 Exp-A: cls probes (14 layers x seeds)")
    ap.add_argument("--forecast-probe", action="store_true", help="C4 Exp-B: fslot probes (warm caches)")
    ap.add_argument("--figures", action="store_true", help="C5: per-source Plots A/B/C (CPU)")
    ap.add_argument("--compare", action="store_true", help="C5: cross-source DOMAIN-vs-TASK comparison (CPU)")
    ap.add_argument("--cka", action="store_true", help="C5: optional linear CKA vs stage0")
    ap.add_argument("--stages", nargs="+", default=list(CLS_STAGES))
    ap.add_argument("--targets", nargs="+", default=list(FORECAST_TARGETS))
    ap.add_argument("--seeds", nargs="+", type=int, default=list(PROBE_SEEDS))
    ap.add_argument("--device", default=None, help="probe-fit device (default: auto)")
    return ap.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    configure(a.cls_source)
    from probing.finetune import _select_device
    device = str(_select_device(a.device))
    if not any([a.extract, a.forecast_extract, a.probe, a.forecast_probe, a.figures, a.compare, a.cka]):
        print("nothing to do; pass one of --extract / --forecast-extract / --probe / "
              "--forecast-probe / --figures / --compare [--cka]"); return

    if a.extract:
        run_cls_extract(load_stages(a.stages))
    if a.forecast_extract:
        run_fcast_extract(load_stages(a.stages), a.targets)
    if a.probe:
        run_cls_probe(load_stages(a.stages), a.seeds, device)
    if a.forecast_probe:
        run_fcast_probe(load_stages(a.stages), a.targets, a.seeds, device)
    if a.figures:
        stages = load_stages(a.stages)
        make_plot_a(stages)
        make_plot_b(stages, a.targets)
        make_plot_c(stages, a.targets)
    if a.compare:
        make_comparison(a.targets)
    if a.cka:
        make_cka(load_stages(a.stages))


if __name__ == "__main__":
    main()
