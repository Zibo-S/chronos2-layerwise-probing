"""FordA classification data for the TASK-SHIFT experiment (notes/PLAN.md, TASK-SHIFT).

The TASK-SHIFT experiment fine-tunes frozen Chronos-2 on a *classification* task (FordA) and then
probes BOTH classification accessibility (Exp A) and forecasting accessibility (Exp B) layer by
layer. This module is the data layer for Exp A: it loads FordA and carves a deterministic,
stratified, leakage-free train/val split of the UCR TRAIN partition. The UCR TEST partition is held
out entirely and never touched by fine-tuning or probe/wd selection.

PREPROCESSING (the ENTIRE preprocessing — documented on purpose):
  * sequences are fed RAW (length 500) as ``context``. Chronos-2 applies its OWN per-instance
    InstanceNorm + arcsinh internally, so there is NO external normalization and NO cross-split
    label leakage (normalization is per-series, computed inside the model).
  * labels {-1, +1} -> {0, 1} via a deterministic sorted-unique map (stored in meta).
  * no padding / truncation: 500 < 8192, and Chronos-2 handles any length.

SPLIT (source-aware index spaces — this matters for the overlap check):
  UCR ships FordA as two SEPARATE partitions, TRAIN (3601) and TEST (1320), each with its OWN
  0-based index space. We stratified-split the TRAIN partition into train/val (seed 0, VAL_FRAC).
  So train_idx / val_idx live in the TRAIN index space; the test rows live in the TEST index space.
  The overlap invariant is therefore ``train_idx ∩ val_idx = ∅`` WITHIN TRAIN — TEST is disjoint by
  construction (different partition), never compared by raw index. The exact indices + seed + val
  fraction are stored in the returned meta so the carve is fully reproducible.

Generic by design: ``CLS_SPECS`` is a tag table so Wafer / ECG5000 / UWave can drop in later; only
FordA is registered now (do NOT add the others until asked).

C0 smoke (login node OK after ``module load arrow`` — seconds, no model):
    python -m probing.cls_data --smoke
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from probing.config import OUTPUT_PATCH_SIZE, SEED

# --- dataset registry (one row per classification task; only FordA now) ----------------------- #
CLS_SPECS = {
    "forda": {"aeon_name": "FordA", "n_classes": 2, "length": 500},
}

VAL_FRAC = 0.2                       # stratified fraction of UCR TRAIN carved to validation


def map_labels_to_int(y_raw) -> tuple[np.ndarray, dict]:
    """Map arbitrary class labels to contiguous ints {0, 1, ...} by SORTED unique order and return
    (y_int, label_map). For FordA this sends -1 -> 0 and +1 -> 1 whether the raw labels arrive as
    strings ('-1', '1') or numbers (np.unique sorts both so '-1'/-1 is the 0 class). Deterministic
    and dataset-agnostic; the map is stored in meta so the encoding is auditable."""
    classes = np.unique(np.asarray(y_raw))
    label_map = {c: i for i, c in enumerate(classes)}
    y_int = np.array([label_map[v] for v in np.asarray(y_raw)], dtype=np.int64)
    # meta-friendly (JSON-serializable) view of the map
    label_map_str = {str(c): int(i) for c, i in label_map.items()}
    return y_int, label_map_str


def stratified_train_val_split(y: np.ndarray, val_frac: float = VAL_FRAC,
                               seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic STRATIFIED train/val index split of a 1-D label array. Pure (no I/O, no model)
    so the determinism is unit-testable without the FordA download. Within each class we take the
    LAST ``round(val_frac * n_class)`` of a seed-shuffled per-class index list to validation; the
    rest to train. Returns (train_idx, val_idx), both sorted ascending, disjoint, covering all rows.
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


def _ncp(length: int) -> int:
    """Content-patch count = ceil(length / OUTPUT_PATCH_SIZE) — the number of content tokens the
    classification head pools over (= 32 for FordA length 500)."""
    return math.ceil(length / OUTPUT_PATCH_SIZE)


def load_forda(tag: str = "forda") -> dict:
    """Load FordA and return the RAW-context classification splits + reproducibility meta.

    Returns dict with X_train/X_val/X_test as (n, length) float32 raw sequences, y_* as (n,) int
    {0,1}, and meta carrying: the sorted-unique label_map, the exact train/val indices (TRAIN index
    space) + seed + val_frac, per-split class balance, ncp, and the split-source note. Requires
    ``aeon`` + internet on first use (the login node); downloads to the aeon cache. NO external
    normalization is applied (the model normalizes internally)."""
    spec = CLS_SPECS[tag]
    from aeon.datasets import load_classification

    def _load(split):
        X, y = load_classification(spec["aeon_name"], split=split)
        X = np.asarray(X, dtype=np.float32)
        assert X.ndim == 3 and X.shape[1] == 1, \
            f"{spec['aeon_name']} expected univariate (n,1,L), got {X.shape}"
        assert X.shape[2] == spec["length"], f"length {X.shape[2]} != spec {spec['length']}"
        return X[:, 0, :], y                      # (n, length) raw, channel squeezed

    Xtr_full, ytr_raw = _load("TRAIN")
    Xte, yte_raw = _load("TEST")

    ytr_full, label_map = map_labels_to_int(ytr_raw)
    yte, label_map_te = map_labels_to_int(yte_raw)
    assert label_map == label_map_te, f"TRAIN/TEST label maps differ: {label_map} vs {label_map_te}"
    n_classes = len(label_map)
    assert n_classes == spec["n_classes"], f"got {n_classes} classes, spec says {spec['n_classes']}"

    tr_idx, va_idx = stratified_train_val_split(ytr_full, VAL_FRAC, SEED)
    Xtr, ytr = Xtr_full[tr_idx], ytr_full[tr_idx]
    Xva, yva = Xtr_full[va_idx], ytr_full[va_idx]

    def _balance(y):
        vals, counts = np.unique(y, return_counts=True)
        return {int(v): int(c) for v, c in zip(vals, counts)}

    meta = {
        "dataset": tag, "aeon_name": spec["aeon_name"], "length": spec["length"],
        "n_classes": n_classes, "ncp": _ncp(spec["length"]), "label_map": label_map,
        "seed": SEED, "val_frac": VAL_FRAC,
        "n_train": int(len(ytr)), "n_val": int(len(yva)), "n_test": int(len(yte)),
        "class_balance": {"train": _balance(ytr), "val": _balance(yva), "test": _balance(yte)},
        "split_source": ("UCR TRAIN(3601) stratified-carved into train/val (seed 0); UCR TEST(1320) "
                         "held out. train_idx/val_idx are in the TRAIN index space; TEST is a SEPARATE "
                         "index space (disjoint by construction, never index-compared)."),
        "train_idx": tr_idx.tolist(), "val_idx": va_idx.tolist(),
        "preprocessing": "RAW length-500 context; model applies its own InstanceNorm+arcsinh; no external norm",
    }
    return {"X_train": Xtr, "y_train": ytr, "X_val": Xva, "y_val": yva,
            "X_test": Xte, "y_test": yte, "meta": meta}


def _smoke(tag: str = "forda") -> None:
    """C0 data smoke: load FordA, print shapes + class balance, assert the source-aware split
    invariants. Login node OK (after ``module load arrow``) — seconds, no model."""
    d = load_forda(tag)
    m = d["meta"]
    print(f"[cls_data smoke] {m['aeon_name']}  label_map={m['label_map']}  ncp={m['ncp']}")
    for s in ("train", "val", "test"):
        X, y = d[f"X_{s}"], d[f"y_{s}"]
        print(f"  {s:>5}: X={X.shape} {X.dtype}  y={y.shape}  balance={m['class_balance'][s]}")
    tr, va = np.asarray(m["train_idx"]), np.asarray(m["val_idx"])
    # source-aware overlap check: train∩val = ∅ WITHIN the TRAIN index space; TEST disjoint partition
    assert len(np.intersect1d(tr, va)) == 0, "train/val overlap in TRAIN index space!"
    assert len(tr) + len(va) == 3601, f"train+val ({len(tr)+len(va)}) != UCR TRAIN 3601"
    assert d["X_train"].shape[1] == m["length"] and d["X_test"].shape[1] == m["length"]
    assert set(np.unique(d["y_train"]).tolist()) <= set(range(m["n_classes"]))
    print("  OK: train∩val=∅ (TRAIN space), train+val=3601, TEST is a separate disjoint partition, "
          "raw length-500 contexts, labels in {0,1}.")


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="FordA classification data (TASK-SHIFT Exp A).")
    ap.add_argument("--smoke", action="store_true", help="C0: load + print + assert split invariants")
    ap.add_argument("--tag", default="forda", choices=sorted(CLS_SPECS))
    return ap.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    if args.smoke:
        _smoke(args.tag)
    else:
        print("nothing to do; pass --smoke")
