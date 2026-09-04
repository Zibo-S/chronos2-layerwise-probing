"""CKA representation-similarity analysis across the three Chronos-2 probing lines.

Complements the linear probes (how linearly ACCESSIBLE task info is) with linear CKA (how
similar/different the representations THEMSELVES are). Reuses each experiment's OWN cached
representation — no re-extraction, no model load, CPU-only:

  extended_v3_rolling          content-pooled, 13 points (Emb, L1..L12)   [pretrained backbone]
  ft_specialization (BOOM FT)  fslot (n,K,768)->(n*K,768), 14 pts (+L12+LN) across 3 stages
  task_shift (FordA cls FT)    A. FordA content 14 pts ; B. forecasting fslot 14 pts, 3 stages

SCIENTIFIC RULE: CKA rows must be the SAME examples. Valid comparisons only:
  * within a dataset: layer x layer (same windows through every layer);
  * across checkpoints on the SAME dataset: pretrained-layer x FT-layer (same windows through
    two backbones — FT caches were extracted from the same target_windows()/FordA arrays).
There is NO cross-dataset row pairing anywhere (extended_v3 gets ONE within-dataset matrix per
dataset; we compare the PATTERNS, never CKA(Electricity, Uber)).

Analyses (select with flags; default = --all):
  --extended-v3     within-dataset 13x13 CKA per dataset + CKA-to-final summary curves
  --extv4-fslot     within-dataset 14x14 FORECAST-SLOT CKA for all 7 pretrained datasets (the
                    representation the ext_v4 fslot probes read); writes provenance.json
  --domain-ft       BOOM-FT: within-stage 14x14, cross-stage alignment, same-layer drift (per target)
  --task-ft         FordA-cls-FT: same three, for FordA content AND forecasting fslot targets
  --domain-vs-task  same-layer CKA-to-pretrained drift, DOMAIN (BOOM) vs TASK (FordA), fslot vs fslot
  --probe-relation  (best-effort) 1-CKA drift vs probe-performance change, merged table + scatter

Outputs under results/cka/{extended_v3_rolling, ft_specialization, task_shift_classification,
domain_vs_task}/ as .npy matrices, labelled CSVs, and publication-style heatmaps/curves.

Compute discipline: cache-only + CKA matmuls loop datasets x stages x layers, so run under
salloc (CPU is fine; no GPU) with OMP_NUM_THREADS set — NOT the login node. The contracts test
(tests/test_cka.py) is synthetic and login-node-safe.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from probing import config
from probing import cka
from probing.config import NUM_LAYERS, OUTPUT_PATCH_SIZE
from probing.extraction import _cache_path, _idf_prefix
from probing.finetune import ft_cache_prefix

# extended_v3 uses the same rolling windows as ft_specialization's PT-ID targets -> namespace them.
config.set_dataset_set("extended_v3_rolling")

H, K = 64, math.ceil(64 / OUTPUT_PATCH_SIZE)          # forecasting fslot: K=4, H=64
FSLOT_POOL = f"K{K}_H{H}"                              # "K4_H64"
CLS_POOL = f"K{math.ceil(16 / OUTPUT_PATCH_SIZE)}_H16"  # FordA cls extraction horizon 16 -> K1 -> "K1_H16"

PT_ID_TAGS = {"monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly"}
SHORT = {"monash_electricity_hourly": "Electricity", "uber_tlc_hourly": "Uber", "m4_hourly": "M4",
         "wind_farms_hourly": "WindFarms", "sg_carpark": "SG-Carpark", "coastal_ts": "Coastal-TS",
         "boom_hourly": "BOOM"}

# extended_v3 roster: Group A (in-domain PT-ID) == Group B (cross-dataset transfer sources) -> one
# matrix each (no duplication); Group C = completely unseen / exploratory OOD.
EXTV3_GROUPS = {"in_domain_PT-ID": ["monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly",
                                    "wind_farms_hourly"],
                "unseen_PT-OOD": ["sg_carpark", "coastal_ts", "boom_hourly"]}

DOMAIN_TARGETS = ("monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly",
                  "boom_hourly", "sg_carpark", "coastal_ts")
TASK_FCAST_TARGETS = ("boom_hourly", "m4_hourly", "coastal_ts")

DOMAIN_STAGES = ("stage0_pretrained", "stage1_ft_early", "stage2_ft_late")
TASK_STAGES = ("stage0_pretrained", "stage1_cls_early", "stage2_cls_late")
STAGE_SHORT = {"stage0_pretrained": "pretrained", "stage1_ft_early": "early BOOM-FT",
               "stage2_ft_late": "late BOOM-FT", "stage1_cls_early": "early cls-FT",
               "stage2_cls_late": "late cls-FT"}
STAGE_COLOR = {"stage0_pretrained": "tab:blue", "stage1_ft_early": "tab:orange",
               "stage2_ft_late": "tab:red", "stage1_cls_early": "tab:orange",
               "stage2_cls_late": "tab:red"}

LABELS_13 = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)]                     # 13
LABELS_14 = ["Emb"] + [f"L{i}" for i in range(1, NUM_LAYERS)] + ["L12+LN"]        # 14
CONTENT13_KEYS = [f"layer_{i}" for i in range(NUM_LAYERS)]                        # layer_0..layer_12
CONTENT14_KEYS = [f"content_L{i}" for i in range(NUM_LAYERS)] + ["content_final"]
FSLOT14_KEYS = [f"fslot_L{i}" for i in range(NUM_LAYERS)] + ["fslot_final"]

OUT = config.REPO_ROOT / "results" / "cka"
BOOM_MANIFEST = config.REPO_ROOT / "results" / "ft_specialization" / "boom" / "manifest.json"
FORDA_MANIFEST = config.REPO_ROOT / "results" / "ft_specialization" / "forda_cls" / "manifest.json"
STAGEB_PROBE_DIR = config.REPO_ROOT / "results" / "ft_specialization" / "stageB" / "probes"
TASK_ROOT = config.REPO_ROOT / "results" / "task_shift_classification"


# --------------------------------------------------------------------------- #
# cache-path resolution (pure string; the actual load fails loud in cka.load_npz_reps)
# --------------------------------------------------------------------------- #
def _load_manifest(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing FT manifest {path} — run the fine-tuning first")
    return json.load(open(path))


def _fcast_split(tag: str) -> str:
    return "test" if tag in PT_ID_TAGS else "test_rolling"


def _fcast_prefix(tag: str, source: str, stage: str, manifest: dict | None) -> str:
    if stage.startswith("stage0"):
        return _idf_prefix(tag)                                     # committed pretrained namespace
    return ft_cache_prefix(tag, source, stage, cka.stage_hash_from_manifest(manifest, stage))


def read_fslot_reps(tag: str, source: str, stage: str, manifest: dict | None) -> list[np.ndarray]:
    """14 forecast-slot layer matrices for (tag, stage), each folded (n,K,768)->(n*K,768). stage0
    reads the committed pretrained cache; FT stages read the source/stage/hash-namespaced cache."""
    prefix = _fcast_prefix(tag, source, stage, manifest)
    path = _cache_path(prefix, _fcast_split(tag), None, FSLOT_POOL)
    return [cka.stack_slots(a) for a in cka.load_npz_reps(path, FSLOT14_KEYS)]


def read_forda_reps(stage: str, manifest: dict | None) -> list[np.ndarray]:
    """14 content-pooled FordA classification layer matrices (n_examples, 768) for a cls stage."""
    prefix = ("IDF_forda__task_shift_cls" if stage.startswith("stage0")
              else ft_cache_prefix("forda", "forda_cls", stage,
                                   cka.stage_hash_from_manifest(manifest, stage)))
    path = _cache_path(prefix, "test", None, CLS_POOL)
    return cka.load_npz_reps(path, CONTENT14_KEYS)


# ext_v4 forecast-slot line: the SAME 14 representation points the fslot probes read, for all 7
# pretrained-backbone datasets. Replaces the four ad-hoc matrices committed in 1bf1b56, which had
# no producer and therefore no recorded split/subsample/seed (see the framework audit).
EXTV4_TAGS = ["monash_electricity_hourly", "uber_tlc_hourly", "m4_hourly", "wind_farms_hourly",
              "sg_carpark", "coastal_ts", "boom_hourly"]


def _fslot_split(tag: str, split: str) -> str:
    """Concrete cache split name. PT-OOD targets were windowed by build_ood_rolling_windows and
    their caches carry the '_rolling' suffix that keeps them disjoint from the legacy eval-only
    caches (see run_ptood_probing_ftok._fslot_feats)."""
    if split not in ("train", "test"):
        raise ValueError(f"unknown split {split!r}; choose train or test")
    if tag in PT_ID_TAGS:
        return split
    return f"{split}_rolling"


def read_extv4_fslot_reps(tag: str, split: str) -> list[np.ndarray]:
    """14 forecast-slot layer matrices for the PRETRAINED backbone, each folded (n,K,768)->(n*K,768).

    Identical representation to read_fslot_reps(stage0) — same committed cache, same keys, same
    stacking — just addressed by dataset+split instead of by FT stage."""
    path = _cache_path(_idf_prefix(tag), _fslot_split(tag, split), None, FSLOT_POOL)
    return [cka.stack_slots(a) for a in cka.load_npz_reps(path, FSLOT14_KEYS)]


def run_extv4_fslot(max_rows, seed, split="test", tags=None):
    """One reproducible 14x14 forecast-slot CKA per dataset + a recorded provenance sidecar."""
    root = OUT / "ext_v4_future_tokens_fslot"
    for sub in ("matrices", "figures", "tables"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    tofinal_rows, curves, prov = [], [], {}
    for tag in (tags or EXTV4_TAGS):
        reps = read_extv4_fslot_reps(tag, split)
        n_avail = reps[0].shape[0]
        idx = cka.subsample_indices(n_avail, max_rows, seed)
        reps = [r[idx] for r in reps]
        M = cka.cka_matrix(reps)
        short = SHORT.get(tag, tag)
        kind = "PT-ID" if tag in PT_ID_TAGS else "PT-OOD"
        np.save(root / "matrices" / f"{tag}__fslot__layerxlayer.npy", M)
        cka.save_matrix_csv(M, LABELS_14, LABELS_14,
                            root / "tables" / f"{tag}__fslot__layerxlayer.csv")
        cka.heatmap(M, LABELS_14, LABELS_14,
                    root / "figures" / f"{tag}__fslot__layerxlayer.png",
                    title=f"{short} — forecast-slot layer x layer CKA ({kind})",
                    xaxis_label="representation point", yaxis_label="representation point")
        tf = cka.cka_to_reference(reps, ref_index=-1)          # to L12+LN (the native head's input)
        curves.append((short, tf))
        for lab, v in zip(LABELS_14, tf):
            tofinal_rows.append({"dataset": tag, "short": short, "kind": kind,
                                 "layer": lab, "cka_to_final": float(v)})
        prov[tag] = {"kind": kind, "cache_split": _fslot_split(tag, split),
                     "rows_available": int(n_avail), "rows_used": int(len(idx)),
                     "windows": int(n_avail // K), "K": K}
        print(f"[extv4-fslot] {short:<12} ({kind:<6}) 14x14  rows {len(idx)}/{n_avail}  "
              f"split={_fslot_split(tag, split)}")
    cka.drift_curve([(lab, v, None) for lab, v in curves], LABELS_14,
                    root / "figures" / "cka_to_final__fslot_all.png",
                    title="Forecast-slot CKA to the native head's input (L12+LN)",
                    ylabel="Linear CKA to L12+LN")
    with open(root / "tables" / "cka_to_final__fslot.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["dataset", "short", "kind", "layer", "cka_to_final"])
        w.writeheader(); w.writerows(tofinal_rows)
    json.dump({"analysis": "ext_v4_future_tokens_fslot",
               "representation": "forecast slots (n,K,768) -> (n*K,768), 14 points Emb..L12+LN",
               "backbone": "pretrained amazon/chronos-2 (frozen)", "C": 512, "H": H, "K": K,
               "requested_split": split, "max_rows": max_rows, "seed": seed,
               "probe_independent": "CKA uses no probe: quantile set / weight decay do not enter",
               "per_dataset": prov},
              open(root / "provenance.json", "w"), indent=2)
    print(f"[extv4-fslot] -> {root}  (provenance.json records split/rows/seed)")


def read_extv3_reps(tag: str) -> list[np.ndarray]:
    """13 content-pooled layer matrices (n, 768) for a pretrained-backbone dataset (extended_v3)."""
    split = "test" if tag in PT_ID_TAGS else "test_rolling"
    path = _cache_path(_idf_prefix(tag), split, None, "content")
    return cka.load_npz_reps(path, CONTENT13_KEYS)


def _apply_subsample(reps_by_stage: dict[str, list[np.ndarray]], max_rows, seed):
    """Deterministically subsample rows with ONE index set shared across all stages/layers (rows must
    correspond for cross-stage CKA). No-op if max_rows is None or >= n."""
    n = reps_by_stage[next(iter(reps_by_stage))][0].shape[0]
    idx = cka.subsample_indices(n, max_rows, seed)
    if len(idx) == n:
        return reps_by_stage, n, n
    return ({s: [r[idx] for r in layers] for s, layers in reps_by_stage.items()}, n, len(idx))


# --------------------------------------------------------------------------- #
# generic per-family stage analysis (within-stage + cross-stage + same-layer drift)
# --------------------------------------------------------------------------- #
def _stage_analysis(root: Path, name: str, reps_by_stage: dict, stages, labels) -> dict:
    """Write within-stage layer x layer matrices, cross-stage (pretrained x FT) alignment matrices,
    and same-layer drift curves for ONE dataset. Returns {ft_stage_label: drift_array} for reuse."""
    (root / "matrices").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    stage0 = stages[0]

    # --- within-stage layer x layer (shared [0,1] colour scale so panels are comparable) ---
    for st in stages:
        if st not in reps_by_stage:
            continue
        M = cka.cka_matrix(reps_by_stage[st])
        np.save(root / "matrices" / f"{name}__within__{st}.npy", M)
        cka.save_matrix_csv(M, labels, labels, root / "tables" / f"{name}__within__{st}.csv")
        cka.heatmap(M, labels, labels, root / "figures" / f"{name}__within__{st}.png",
                    title=f"{name} — {STAGE_SHORT.get(st, st)} (layer x layer)",
                    xaxis_label="layer", yaxis_label="layer")

    # --- cross-stage alignment (rows = pretrained layers, cols = FT layers) + same-layer drift ---
    drift = {}
    drift_curves = []
    drift_rows = []
    for st in stages[1:]:
        if st not in reps_by_stage or stage0 not in reps_by_stage:
            continue
        Mc = cka.cka_matrix(rows=reps_by_stage[stage0], cols=reps_by_stage[st])
        np.save(root / "matrices" / f"{name}__cross__pretrained_x_{st}.npy", Mc)
        cka.save_matrix_csv(Mc, labels, labels, root / "tables" / f"{name}__cross__pretrained_x_{st}.csv")
        cka.heatmap(Mc, labels, labels, root / "figures" / f"{name}__cross__pretrained_x_{st}.png",
                    title=f"{name} — pretrained vs {STAGE_SHORT.get(st, st)}",
                    xaxis_label=f"{STAGE_SHORT.get(st, st)} layer", yaxis_label="pretrained layer")
        d = np.diagonal(Mc).copy()                       # = same-layer CKA(pretrained_l, FT_l)
        drift[st] = d
        drift_curves.append((STAGE_SHORT.get(st, st), d, STAGE_COLOR.get(st)))
        for lab, v in zip(labels, d):
            drift_rows.append({"dataset": name, "stage": st, "layer": lab,
                               "cka_to_pretrained_same_layer": float(v)})
    if drift_curves:
        cka.drift_curve(drift_curves, labels, root / "figures" / f"{name}__same_layer_drift.png",
                        title=f"{name} — same-layer CKA to pretrained")
        with open(root / "tables" / f"{name}__same_layer_drift.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, ["dataset", "stage", "layer", "cka_to_pretrained_same_layer"])
            w.writeheader(); w.writerows(drift_rows)
    return drift


# --------------------------------------------------------------------------- #
# Analysis I — extended_v3_rolling (pretrained backbone, content 13 pts)
# --------------------------------------------------------------------------- #
def run_extended_v3(max_rows, seed):
    root = OUT / "extended_v3_rolling"
    (root / "matrices").mkdir(parents=True, exist_ok=True)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    (root / "tables").mkdir(parents=True, exist_ok=True)
    tofinal_rows = []
    per_group_curves = {g: [] for g in EXTV3_GROUPS}
    for group, tags in EXTV3_GROUPS.items():
        for tag in tags:
            reps = read_extv3_reps(tag)
            idx = cka.subsample_indices(reps[0].shape[0], max_rows, seed)
            reps = [r[idx] for r in reps]
            M = cka.cka_matrix(reps)
            short = SHORT.get(tag, tag)
            np.save(root / "matrices" / f"{tag}__layerxlayer.npy", M)
            cka.save_matrix_csv(M, LABELS_13, LABELS_13, root / "tables" / f"{tag}__layerxlayer.csv")
            cka.heatmap(M, LABELS_13, LABELS_13, root / "figures" / f"{tag}__layerxlayer.png",
                        title=f"{short} — layer x layer CKA ({group})",
                        xaxis_label="layer", yaxis_label="layer")
            tf = cka.cka_to_reference(reps, ref_index=-1)     # to L12 (extended_v3 has no L12+LN)
            per_group_curves[group].append((short, tf))
            for lab, v in zip(LABELS_13, tf):
                tofinal_rows.append({"dataset": tag, "short": short, "group": group,
                                     "layer": lab, "cka_to_final": float(v)})
            print(f"[extv3] {short} ({group}): 13x13 CKA, n={len(idx)}")
    # summary: one CKA-to-final curve panel per group
    for group, curves in per_group_curves.items():
        if curves:
            cka.drift_curve([(lab, v, None) for lab, v in curves], LABELS_13,
                            root / "figures" / f"cka_to_final__{group}.png",
                            title=f"CKA to final representation — {group}",
                            ylabel="Linear CKA to L12")
    with open(root / "tables" / "cka_to_final.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["dataset", "short", "group", "layer", "cka_to_final"])
        w.writeheader(); w.writerows(tofinal_rows)
    print(f"[extv3] -> {root}")


# --------------------------------------------------------------------------- #
# Analysis II / III — domain-FT (BOOM) and task-FT (FordA) stage analyses
# --------------------------------------------------------------------------- #
def _available_fslot(tags, source, manifest, stages):
    """Keep only targets whose ALL requested stages have an fslot cache on disk (auto-roster so a
    default run never hard-fails); print what is skipped. An explicitly requested but missing cache
    still fails loud later in read_fslot_reps."""
    ok = []
    for tag in tags:
        paths = [_cache_path(_fcast_prefix(tag, source, st, manifest), _fcast_split(tag), None,
                             FSLOT_POOL) for st in stages]
        if all(p.exists() for p in paths):
            ok.append(tag)
        else:
            print(f"  [skip] {SHORT.get(tag, tag)}: missing fslot cache for some {source} stage")
    return ok


def run_domain_ft(targets, max_rows, seed) -> dict:
    manifest = _load_manifest(BOOM_MANIFEST)
    root = OUT / "ft_specialization"
    present = _available_fslot(targets, "boom", manifest, DOMAIN_STAGES)
    drift_by_target = {}
    for tag in present:
        reps = {st: read_fslot_reps(tag, "boom", st, manifest) for st in DOMAIN_STAGES}
        reps, n, m = _apply_subsample(reps, max_rows, seed)
        drift = _stage_analysis(root / SHORT.get(tag, tag), SHORT.get(tag, tag), reps,
                                DOMAIN_STAGES, LABELS_14)
        drift_by_target[tag] = {"early": drift.get("stage1_ft_early"), "late": drift.get("stage2_ft_late")}
        print(f"[domain-FT] {SHORT.get(tag, tag)}: within+cross+drift, rows={m}/{n*K}")
    print(f"[domain-FT] -> {root}")
    return drift_by_target


def run_task_ft(max_rows, seed) -> tuple[dict, dict]:
    manifest = _load_manifest(FORDA_MANIFEST)
    # A. FordA classification content (one 'dataset')
    forda_reps = {st: read_forda_reps(st, manifest) for st in TASK_STAGES}
    forda_reps, n, m = _apply_subsample(forda_reps, max_rows, seed)
    cls_drift = _stage_analysis(OUT / "task_shift_classification" / "forda_content", "FordA",
                                forda_reps, TASK_STAGES, LABELS_14)
    print(f"[task-FT/cls] FordA content: within+cross+drift, rows={m}/{n}")
    cls_drift = {"early": cls_drift.get("stage1_cls_early"), "late": cls_drift.get("stage2_cls_late")}
    # B. forecasting fslot after cls-FT
    fcast_root = OUT / "task_shift_classification" / "forecasting"
    present = _available_fslot(TASK_FCAST_TARGETS, "forda_cls", manifest, TASK_STAGES)
    fcast_drift = {}
    for tag in present:
        reps = {st: read_fslot_reps(tag, "forda_cls", st, manifest) for st in TASK_STAGES}
        reps, n, m = _apply_subsample(reps, max_rows, seed)
        d = _stage_analysis(fcast_root / SHORT.get(tag, tag), SHORT.get(tag, tag), reps,
                            TASK_STAGES, LABELS_14)
        fcast_drift[tag] = {"early": d.get("stage1_cls_early"), "late": d.get("stage2_cls_late")}
        print(f"[task-FT/fcast] {SHORT.get(tag, tag)}: within+cross+drift, rows={m}/{n*K}")
    print(f"[task-FT] -> {OUT / 'task_shift_classification'}")
    return cls_drift, fcast_drift


# --------------------------------------------------------------------------- #
# Analysis — DOMAIN vs TASK same-layer drift (fslot vs fslot only)
# --------------------------------------------------------------------------- #
def run_domain_vs_task(domain_drift, task_fcast_drift, max_rows, seed):
    """For targets common to BOTH FT conditions, overlay same-layer CKA-to-pretrained (fslot). Both
    conditions share the SAME pretrained stage0 cache + windows, so the drift curves are comparable."""
    root = OUT / "domain_vs_task"
    root.mkdir(parents=True, exist_ok=True)
    common = [t for t in task_fcast_drift if t in domain_drift]
    if not common:
        print("[domain-vs-task] skipped (no target present under both BOOM-FT and FordA-cls-FT)")
        return
    rows = []
    for tag in common:
        curves = []
        for cond, dd, colmap in [("BOOM-FT (domain)", domain_drift[tag],
                                  {"early": "tab:green", "late": "tab:olive"}),
                                 ("FordA-cls-FT (task)", task_fcast_drift[tag],
                                  {"early": "tab:orange", "late": "tab:red"})]:
            for stage in ("early", "late"):
                v = dd.get(stage)
                if v is None:
                    continue
                curves.append((f"{cond} — {stage}", v, colmap[stage]))
                for lab, val in zip(LABELS_14, v):
                    rows.append({"target": SHORT.get(tag, tag), "condition": cond, "stage": stage,
                                 "layer": lab, "cka_to_pretrained_same_layer": float(val)})
        cka.drift_curve(curves, LABELS_14, root / f"{SHORT.get(tag, tag)}__domain_vs_task.png",
                        title=f"{SHORT.get(tag, tag)} — representation drift: DOMAIN vs TASK FT")
    with open(root / "domain_vs_task_drift.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["target", "condition", "stage", "layer",
                                "cka_to_pretrained_same_layer"])
        w.writeheader(); w.writerows(rows)
    print(f"[domain-vs-task] {[SHORT.get(t, t) for t in common]} -> {root}")


# --------------------------------------------------------------------------- #
# Analysis — probe-change vs representation-drift (best-effort scatter + merged table)
# --------------------------------------------------------------------------- #
def _seed_mean_curve(dirpath: Path, pattern: str, key: str):
    paths = sorted(Path(dirpath).glob(pattern))
    if not paths:
        return None
    return np.mean([np.asarray(json.load(open(p))[key], float) for p in paths], axis=0)


def run_probe_relation(domain_drift, cls_drift, task_fcast_drift):
    """x = 1 - CKA(pretrained_l, FT_l) (drift); y = probe-performance change vs pretrained. Forecasting
    (Δ fslot quantile loss) and classification (Δ accuracy) are tagged separately. Always writes the
    merged CSV; renders a scatter only if there are points. Skips any source whose probe JSONs are
    absent (this analysis is secondary and must never block the CKA matrices)."""
    root = OUT / "domain_vs_task"
    root.mkdir(parents=True, exist_ok=True)
    rows = []

    def _forecast(cond, drift, probe_dir, tag):
        base = _seed_mean_curve(probe_dir, f"stage0_pretrained__{tag}__q9__seed*.json", "test_loss_by_layer")
        for stage_key, stage_lbl in [("early", None), ("late", None)]:
            d = drift.get(tag, {}).get(stage_key) if isinstance(drift.get(tag), dict) else None
            if d is None or base is None:
                continue
            st = ({"early": "stage1_ft_early", "late": "stage2_ft_late"} if cond == "domain"
                  else {"early": "stage1_cls_early", "late": "stage2_cls_late"})[stage_key]
            ft = _seed_mean_curve(probe_dir, f"{st}__{tag}__q9__seed*.json", "test_loss_by_layer")
            if ft is None:
                continue
            dy = ft - base
            for lab, xv, yv in zip(LABELS_14, 1.0 - d, dy):
                rows.append({"condition": cond, "metric": "d_fslot_quantile_loss", "target": SHORT.get(tag, tag),
                             "stage": stage_key, "layer": lab, "drift_1_minus_cka": float(xv),
                             "probe_change": float(yv)})

    for tag in domain_drift:
        _forecast("domain", domain_drift, STAGEB_PROBE_DIR, tag)
    for tag in task_fcast_drift:
        _forecast("task", task_fcast_drift, TASK_ROOT / "forecast_probes", tag)

    # classification accuracy change vs FordA content drift
    base_acc = _seed_mean_curve(TASK_ROOT / "cls_probes", "stage0_pretrained__seed*.json", "test_acc_by_layer")
    for stage_key, st in [("early", "stage1_cls_early"), ("late", "stage2_cls_late")]:
        d = cls_drift.get(stage_key) if isinstance(cls_drift, dict) else None
        acc = _seed_mean_curve(TASK_ROOT / "cls_probes", f"{st}__seed*.json", "test_acc_by_layer")
        if d is None or base_acc is None or acc is None:
            continue
        for lab, xv, yv in zip(LABELS_14, 1.0 - d, acc - base_acc):
            rows.append({"condition": "task", "metric": "d_forda_accuracy", "target": "FordA",
                         "stage": stage_key, "layer": lab, "drift_1_minus_cka": float(xv),
                         "probe_change": float(yv)})

    if not rows:
        print("[probe-relation] no probe JSONs found — skipped (merged table empty)")
        return
    with open(root / "drift_vs_probe_change.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["condition", "metric", "target", "stage", "layer",
                                "drift_1_minus_cka", "probe_change"])
        w.writeheader(); w.writerows(rows)
    _probe_scatter(rows, root / "drift_vs_probe_change.png")
    print(f"[probe-relation] {len(rows)} points -> {root}")


def _probe_scatter(rows, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = sorted({r["metric"] for r in rows})
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4 * len(metrics), 4.4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        pts = [r for r in rows if r["metric"] == metric]
        for cond, color in [("domain", "tab:green"), ("task", "tab:red")]:
            xs = [r["drift_1_minus_cka"] for r in pts if r["condition"] == cond]
            ys = [r["probe_change"] for r in pts if r["condition"] == cond]
            if xs:
                ax.scatter(xs, ys, s=14, alpha=0.6, color=color, label=cond)
        ax.axhline(0, ls=":", c="gray", lw=1)
        ax.set_xlabel("representation drift  (1 - CKA)"); ax.set_ylabel(f"Δ {metric}")
        ax.set_title(metric, fontsize=9); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=200); fig.savefig(Path(path).with_suffix(".pdf"))
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--extended-v3", action="store_true")
    ap.add_argument("--extv4-fslot", action="store_true",
                    help="14-pt forecast-slot CKA, all 7 pretrained-backbone datasets")
    ap.add_argument("--fslot-split", default="test", choices=("test", "train"),
                    help="which cached split the fslot CKA reads (recorded in provenance.json)")
    ap.add_argument("--domain-ft", action="store_true")
    ap.add_argument("--task-ft", action="store_true")
    ap.add_argument("--domain-vs-task", action="store_true")
    ap.add_argument("--probe-relation", action="store_true")
    ap.add_argument("--all", action="store_true", help="run every analysis (default if no flag given)")
    ap.add_argument("--targets", nargs="+", default=list(DOMAIN_TARGETS), help="domain-FT targets")
    ap.add_argument("--max-rows", type=int, default=None, help="deterministic row subsample (default: all)")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv=None):
    a = _parse_args(argv)
    run_all = a.all or not any([a.extended_v3, a.extv4_fslot, a.domain_ft, a.task_ft,
                                a.domain_vs_task, a.probe_relation])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "run_config.json").write_text(json.dumps(
        {"split": "test", "max_rows": a.max_rows, "seed": a.seed,
         "representations": {"extended_v3_rolling": "content-pooled, 13 pts (Emb,L1..L12)",
                             "ft_specialization": "fslot (n*K,768), 14 pts (+L12+LN)",
                             "task_cls": "FordA content, 14 pts", "task_fcast": "fslot, 14 pts",
                             "ext_v4_future_tokens_fslot":
                                 f"fslot (n*K,768), 14 pts (+L12+LN), split={a.fslot_split}"}}, indent=2))

    domain_drift, cls_drift, task_fcast_drift = {}, {}, {}
    if run_all or a.extended_v3:
        run_extended_v3(a.max_rows, a.seed)
    if run_all or a.extv4_fslot:
        run_extv4_fslot(a.max_rows, a.seed, a.fslot_split)
    if run_all or a.domain_ft or a.domain_vs_task or a.probe_relation:
        domain_drift = run_domain_ft(a.targets, a.max_rows, a.seed)
    if run_all or a.task_ft or a.domain_vs_task or a.probe_relation:
        cls_drift, task_fcast_drift = run_task_ft(a.max_rows, a.seed)
    if run_all or a.domain_vs_task:
        run_domain_vs_task(domain_drift, task_fcast_drift, a.max_rows, a.seed)
    if run_all or a.probe_relation:
        run_probe_relation(domain_drift, cls_drift, task_fcast_drift)


if __name__ == "__main__":
    main()
