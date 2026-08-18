"""UEA/UCR classification data for the TASK-SHIFT experiment (notes/PLAN.md, TASK-SHIFT).

The TASK-SHIFT experiment fine-tunes frozen Chronos-2 on a *classification* task and then probes BOTH
classification accessibility (Exp A) and forecasting accessibility (Exp B) layer by layer. This module
is the data layer for Exp A: it loads a classification dataset and carves a deterministic, stratified,
leakage-free train/val split of the official TRAIN partition. The official TEST partition is held out
entirely and never touched by fine-tuning or probe/wd selection.

Sources (``CLS_SPECS`` registry — one row per classification task):
  * ``forda``       — FordA (UCR): univariate, length 500, 2 classes {-1,+1}. The EASY/control task
                      (near-saturated, already linearly decodable pre-FT).
  * ``uwave``       — UWaveGestureLibrary (UEA): 3-channel, length 315, 8 classes. Non-saturated;
                      pretrained probe curve peaks intermediate (~L4) then declines.
  * ``handwriting`` — Handwriting (UEA): 3-channel, length 152, 26 classes. Hard/multiclass;
                      pretrained probe curve peaks intermediate (~L6) then declines.
UWave/Handwriting were identified as non-saturated tasks with a pretrained intermediate-layer maximum in
the earlier UEA probing (results/uea/perdataset_summary.json), NOT chosen for a desired FT outcome.

MULTIVARIATE CONVENTION (decision 2026-08-18): each channel is fed through the (shared) encoder
UNIVARIATELY and the per-channel 768-d representations are CONCATENATED -> c*768. This is exactly the
old UEA ``extract_features`` interpretation (per-channel-through-encoder then concatenate); Chronos-2's
native group-attention multivariate path is NOT used (it is unprobed in this project). So sequences are
returned RAW as ``(n, c, L)`` and the head/probe read a c*768 pooled vector. For FordA (c=1) this is the
single-channel case and is numerically identical to the original univariate FordA pipeline.

PREPROCESSING (the ENTIRE preprocessing — documented on purpose):
  * sequences are fed RAW (length L) as ``context``. Chronos-2 applies its OWN per-instance
    InstanceNorm + arcsinh internally, so there is NO external normalization and NO cross-split label
    leakage (normalization is per-series, computed inside the model).
  * labels -> contiguous {0..C-1} via a deterministic sorted-unique map (stored in meta). np.unique
    sorts strings lexicographically, which is still a deterministic, contiguous, class-agnostic encoding.
  * no padding / truncation: L < 8192, and Chronos-2 handles any length.

SPLIT (source-aware index spaces — this matters for the overlap check):
  UEA/UCR ship SEPARATE TRAIN and TEST partitions, each with its OWN 0-based index space. We
  stratified-split the TRAIN partition into train/val (seed 0, VAL_FRAC). So train_idx / val_idx live in
  the TRAIN index space; the test rows live in the TEST index space. The overlap invariant is therefore
  ``train_idx ∩ val_idx = ∅`` WITHIN TRAIN — TEST is disjoint by construction (different partition),
  never compared by raw index. The exact indices + seed + val fraction are stored in the returned meta.

C0 smoke (login node OK after ``module load arrow`` — seconds, no model):
    python -m probing.cls_data --smoke --cls-source uwave
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from probing.config import OUTPUT_PATCH_SIZE, SEED

# --- dataset registry (one row per classification task) --------------------------------------- #
# Each row carries the dataset geometry (n_classes / channels / length) AND source-appropriate FT
# defaults (epochs / batch_size / lrs). The FT defaults are DEFAULTS ONLY (CLI overrides them); they
# differ per source because the training-set size differs by ~25x (FordA 3601 vs UWave 120 / HW 150):
# the tiny multivariate sets need more epochs + a smaller batch to reach a comparable optimizer-step
# budget and a valid early/late specialization gradient. Deciding these uses classification-side
# evidence only (val CE/acc + backbone drift), never any forecasting curve.
CLS_SPECS = {
    "forda": {"aeon_name": "FordA", "n_classes": 2, "channels": 1, "length": 500,
              "epochs": 10, "batch_size": 64, "backbone_lr": 1e-5, "head_lr": 1e-3},
    "uwave": {"aeon_name": "UWaveGestureLibrary", "n_classes": 8, "channels": 3, "length": 315,
              "epochs": 60, "batch_size": 16, "backbone_lr": 1e-5, "head_lr": 1e-3},
    "handwriting": {"aeon_name": "Handwriting", "n_classes": 26, "channels": 3, "length": 152,
                    "epochs": 60, "batch_size": 16, "backbone_lr": 1e-5, "head_lr": 1e-3},
}

VAL_FRAC = 0.2                       # stratified fraction of the TRAIN partition carved to validation


def map_labels_to_int(y_raw) -> tuple[np.ndarray, dict]:
    """Map arbitrary class labels to contiguous ints {0..C-1} by SORTED unique order and return
    (y_int, label_map). np.unique sorts both numbers and strings (lexicographically for strings), so the
    encoding is deterministic and contiguous whatever the raw dtype: FordA {-1,+1}->{0,1}; UWave/HW
    string labels '1.0'..'26.0' -> a lexicographic-but-contiguous [0,C-1]. The map is stored in meta so
    the encoding is auditable (label ORDER is irrelevant for classification; contiguity is what matters).
    """
    classes = np.unique(np.asarray(y_raw))
    label_map = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_map[v] for v in np.asarray(y_raw)], dtype=np.int64)
    # meta-friendly (JSON-serializable) view of the map
    label_map_str = {str(c): int(i) for c, i in label_map.items()}
    return y_int, label_map_str


def stratified_train_val_split(y: np.ndarray, val_frac: float = VAL_FRAC,
                               seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic STRATIFIED train/val index split of a 1-D label array. Pure (no I/O, no model)
    so the determinism is unit-testable without the dataset download. Within each class we take the
    LAST ``round(val_frac * n_class)`` of a seed-shuffled per-class index list to validation; the
    rest to train. Returns (train_idx, val_idx), both sorted ascending, disjoint, covering all rows.
    Every class keeps >=1 row in EACH of train/val (important for the small multiclass sets).
    """
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    tr, va = [], []
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        idx = rng.permutation(idx)                 # seed-deterministic per-class shuffle
        n_val = int(round(val_frac * len(idx)))
        n_val = min(max(n_val, 1), len(idx) - 1)   # keep >=1 in each of train/val per class
        va.append(idx[:n_val])
        tr.append(idx[n_val:])
    train_idx = np.sort(np.concatenate(tr))
    val_idx = np.sort(np.concatenate(va))
    assert len(np.intersect1d(train_idx, val_idx)) == 0, "stratified split produced overlap"
    return train_idx, val_idx


def ncp_for_length(length: int) -> int:
    """Content-patch count = ceil(length / OUTPUT_PATCH_SIZE) — the number of content tokens pooled
    PER CHANNEL (FordA 500->32, UWave 315->20, Handwriting 152->10). Per-channel univariate encoding
    means the multivariate case does NOT change this: each channel contributes ceil(L/16) content
    tokens, pooled to one 768-d vector, then concatenated across channels."""
    return math.ceil(length / OUTPUT_PATCH_SIZE)


# back-compat alias (was ``_ncp``); kept so any older caller/import still resolves
_ncp = ncp_for_length


def load_cls(tag: str = "forda") -> dict:
    """Load a classification source and return the RAW-context splits + reproducibility meta.

    Returns dict with X_train/X_val/X_test as ``(n, c, L)`` float32 raw sequences (c=1 for FordA), y_*
    as (n,) int {0..C-1}, and meta carrying: the sorted-unique label_map, the exact train/val indices
    (TRAIN index space) + seed + val_frac, per-split class balance, channels, ncp, and the split-source
    note. Requires ``aeon`` + internet on first use (the login node); downloads to the aeon cache. NO
    external normalization is applied (the model normalizes internally)."""
    spec = CLS_SPECS[tag]
    from aeon.datasets import load_classification

    def _load(split):
        X, y = load_classification(spec["aeon_name"], split=split)
        X = np.asarray(X, dtype=np.float32)                   # (n, c, L)
        assert X.ndim == 3, f"{spec['aeon_name']} expected (n, c, L), got {X.shape}"
        assert X.shape[1] == spec["channels"], \
            f"{spec['aeon_name']} channels {X.shape[1]} != spec {spec['channels']}"
        assert X.shape[2] == spec["length"], \
            f"{spec['aeon_name']} length {X.shape[2]} != spec {spec['length']}"
        return X, y

    Xtr_full, ytr_raw = _load("TRAIN")
    Xte, yte_raw = _load("TEST")

    ytr_full, label_map = map_labels_to_int(ytr_raw)
    yte, label_map_te = map_labels_to_int(yte_raw)
    assert label_map == label_map_te, f"TRAIN/TEST label maps differ: {label_map} vs {label_map_te}"
    n_classes = len(label_map)
    assert n_classes == spec["n_classes"], f"got {n_classes} classes, spec says {spec['n_classes']}"
    assert sorted(int(v) for v in label_map.values()) == list(range(n_classes)), \
        f"label map is not contiguous [0,{n_classes - 1}]: {label_map}"

    tr_idx, va_idx = stratified_train_val_split(ytr_full, VAL_FRAC, SEED)
    Xtr, ytr = Xtr_full[tr_idx], ytr_full[tr_idx]
    Xva, yva = Xtr_full[va_idx], ytr_full[va_idx]

    def _balance(y):
        vals, counts = np.unique(y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    meta = {
        "dataset": tag, "aeon_name": spec["aeon_name"], "channels": spec["channels"],
        "length": spec["length"], "n_classes": n_classes, "ncp": ncp_for_length(spec["length"]),
        "label_map": label_map, "seed": SEED, "val_frac": VAL_FRAC,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "class_balance": {"train": _balance(ytr), "val": _balance(yva), "test": _balance(yte)},
        "split_source": (f"{spec['aeon_name']} TRAIN({len(ytr_full)}) stratified-carved into train/val "
                         f"(seed {SEED}); TEST({len(yte)}) held out. train_idx/val_idx are in the TRAIN "
                         "index space; TEST is a SEPARATE index space (disjoint by construction, never "
                         "index-compared)."),
        "train_idx": tr_idx.tolist(), "val_idx": va_idx.tolist(),
        "multivariate_convention": ("per-channel univariate encode -> concat 768 per channel -> c*768 "
                                    "(matches old UEA extract_features; group-attention path unused)"),
        "preprocessing": (f"RAW length-{spec['length']} per-channel contexts; model applies its own "
                          "InstanceNorm+arcsinh; no external norm"),
    }
    return {"X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva,
            "X_test": Xte, "y_test": yte, "meta": meta}


# back-compat alias (the original FordA-only entry point)
def load_forda(tag: str = "forda") -> dict:
    """Deprecated alias for :func:`load_cls` (kept so older imports still resolve)."""
    return load_cls(tag)


def _smoke(tag: str = "forda") -> None:
    """C0 data smoke: load the source, print shapes + class balance, assert the source-aware split
    invariants. Login node OK (after ``module load arrow``) — seconds, no model."""
    d = load_cls(tag)
    m = d["meta"]
    print(f"[cls_data smoke] {m['aeon_name']}  channels={m['channels']}  n_classes={m['n_classes']}  "
          f"label_map={m['label_map']}  ncp={m['ncp']}")
    for s in ("train", "val", "test"):
        X, y = d[f"X_{s}"], d[f"y_{s}"]
        print(f"  {s:>5}: X={X.shape} {X.dtype}  y={y.shape}  balance={m['class_balance'][s]}")
    tr, va = np.asarray(m["train_idx"]), np.asarray(m["val_idx"])
    # source-aware overlap check: train∩val = ∅ WITHIN the TRAIN index space; TEST is a disjoint partition
    assert len(np.intersect1d(tr, va)) == 0, "train/val overlap in TRAIN index space!"
    assert len(tr) + len(va) == m["n_train"] + m["n_val"], "train+val != TRAIN partition size"
    assert d["X_train"].shape[1] == m["channels"] and d["X_train"].shape[2] == m["length"]
    assert d["X_test"].shape[1] == m["channels"] and d["X_test"].shape[2] == m["length"]
    assert set(np.unique(d["y_train"]).tolist()) <= set(range(m["n_classes"]))
    print(f"  OK: train∩val=∅ (TRAIN space), train+val={len(tr) + len(va)}, TEST is a separate disjoint "
          f"partition, raw ({m['channels']}, {m['length']}) contexts, labels in [0,{m['n_classes'] - 1}].")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Classification data for the TASK-SHIFT experiment (Exp A).")
    ap.add_argument("--smoke", action="store_true", help="C0: load + print + assert split invariants")
    ap.add_argument("--cls-source", "--tag", dest="cls_source", default="forda",
                    choices=sorted(CLS_SPECS), help="which classification source to load")
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.smoke:
        _smoke(args.cls_source)
    else:
        print("nothing to do; pass --smoke [--cls-source uwave|handwriting|forda]")
