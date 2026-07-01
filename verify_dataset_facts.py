"""
Read-only verification of UEA dataset shape facts used in the probing experiment.

Loads each dataset's canonical train/test splits via aeon and prints the REAL shapes
computed from the arrays (never from memory or external specs). Does NOT load Chronos-2,
run any probe, or extract features.

Run: python verify_dataset_facts.py
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")  # silence aeon's load_classification FutureWarning noise

import numpy as np
from aeon.datasets import load_classification


DATASETS = [
    "Handwriting", "SelfRegulationSCP1", "UWaveGestureLibrary", "EthanolConcentration",
    "LSST", "SelfRegulationSCP2", "Epilepsy", "Cricket",
]

EXPECTED = {
    "Handwriting":          {"n_channels": 3, "series_length": 152,  "n_classes": 26, "n_train": 150,  "n_test": 850},
    "SelfRegulationSCP1":   {"n_channels": 6, "series_length": 896,  "n_classes": 2,  "n_train": 268,  "n_test": 293},
    "UWaveGestureLibrary":  {"n_channels": 3, "series_length": 315,  "n_classes": 8,  "n_train": 120,  "n_test": 320},
    "EthanolConcentration": {"n_channels": 3, "series_length": 1751, "n_classes": 4,  "n_train": 261,  "n_test": 263},
    "LSST":                 {"n_channels": 6, "series_length": 36,   "n_classes": 14, "n_train": 2459, "n_test": 2466},
    "SelfRegulationSCP2":   {"n_channels": 7, "series_length": 1152, "n_classes": 2,  "n_train": 200,  "n_test": 180},
    "Epilepsy":             {"n_channels": 3, "series_length": 206,  "n_classes": 4,  "n_train": 137,  "n_test": 138},
    "Cricket":              {"n_channels": 6, "series_length": 1197, "n_classes": 12, "n_train": 108,  "n_test": 72},
}


def load_split(name, split):
    """Try the canonical split; if this aeon version rejects `split`, signal the caller."""
    try:
        X, y = load_classification(name, split=split)
        return X, y, True
    except TypeError:
        # `split` arg not supported -> caller falls back to the full set
        X, y = load_classification(name)
        return X, y, False


def describe(X):
    """Return (n_cases, n_channels, length_repr, is_variable, min_len, max_len, raw_shape_repr).

    Handles both equal-length (3D ndarray) and unequal-length (list / object array) returns.
    """
    if isinstance(X, np.ndarray) and X.ndim == 3:
        n_cases, n_channels, length = X.shape
        return n_cases, n_channels, length, False, length, length, str(X.shape)

    # Unequal-length or list-like: each element is (n_channels, length_i)
    elems = list(X)
    n_cases = len(elems)
    first = np.asarray(elems[0])
    n_channels = first.shape[0]
    lengths = [np.asarray(e).shape[1] for e in elems]
    min_len, max_len = min(lengths), max(lengths)
    consistent_ch = all(np.asarray(e).shape[0] == n_channels for e in elems)
    if not consistent_ch:
        n_channels = "VARIES"
    length_repr = "variable" if min_len != max_len else min_len
    raw = f"list[{n_cases}] of ({n_channels} x [{min_len}..{max_len}])"
    return n_cases, n_channels, length_repr, (min_len != max_len), min_len, max_len, raw


def main():
    print("=" * 100)
    print("READ-ONLY UEA dataset shape verification (computed from arrays via aeon; no model loaded)")
    print(f"aeon: {__import__('aeon').__version__}")
    print("=" * 100)

    results = {}   # name -> dict of facts (or None if failed)

    for name in DATASETS:
        print(f"\n[{name}]")
        try:
            X_tr, y_tr, tr_split_ok = load_split(name, "train")
            X_te, y_te, te_split_ok = load_split(name, "test")

            if not (tr_split_ok and te_split_ok):
                print("  WARNING: this aeon version did not accept split='train'/'test'; "
                      "loaded FULL set for both -- splits could NOT be separated.")

            n_tr, n_ch_tr, len_tr, var_tr, min_tr, max_tr, raw_tr = describe(X_tr)
            n_te, n_ch_te, len_te, var_te, min_te, max_te, raw_te = describe(X_te)

            labels_tr = sorted(np.unique(np.asarray(y_tr)).tolist(), key=lambda v: str(v))
            labels_te = sorted(np.unique(np.asarray(y_te)).tolist(), key=lambda v: str(v))
            n_classes = len(labels_tr)
            same_labels = labels_tr == labels_te
            chance = 1.0 / n_classes if n_classes else float("nan")

            print(f"  X_tr.shape (raw): {raw_tr}")
            print(f"  X_te.shape (raw): {raw_te}")
            print(f"  n_train       = {n_tr}")
            print(f"  n_test        = {n_te}")
            print(f"  n_channels    = {n_ch_tr}" + ("" if n_ch_tr == n_ch_te else f"  (TEST has {n_ch_te}!)"))
            if var_tr:
                print(f"  series_length = variable  (train min={min_tr}, max={max_tr}; test min={min_te}, max={max_te})")
            else:
                print(f"  series_length = {len_tr}" + ("" if len_tr == len_te else f"  (TEST has {len_te}!)"))
            print(f"  n_classes     = {n_classes}  labels(train)={labels_tr}")
            print(f"  y_te same label set as y_tr? {same_labels}" +
                  ("" if same_labels else f"  test labels={labels_te}"))
            print(f"  chance        = {chance:.3f}")

            results[name] = {
                "n_train": n_tr, "n_test": n_te,
                "n_channels": n_ch_tr,
                "series_length": ("variable" if var_tr else len_tr),
                "n_classes": n_classes, "chance": chance,
                "same_labels": same_labels,
            }
        except Exception as e:
            print(f"  FAILED to load: {type(e).__name__}: {e}")
            results[name] = None

    # ---------------- fixed-width table ----------------
    print("\n" + "=" * 100)
    print("SUMMARY TABLE (values computed from the loaded arrays)")
    print("=" * 100)
    header = f"{'dataset':<22} | {'n_train':>7} | {'n_test':>6} | {'n_channels':>10} | {'series_length':>13} | {'n_classes':>9} | {'chance':>6}"
    print(header)
    print("-" * len(header))
    for name in DATASETS:
        r = results.get(name)
        if r is None:
            print(f"{name:<22} | {'(failed to load)':>7}")
            continue
        print(f"{name:<22} | {r['n_train']:>7} | {r['n_test']:>6} | {str(r['n_channels']):>10} | "
              f"{str(r['series_length']):>13} | {r['n_classes']:>9} | {r['chance']:>6.3f}")

    # ---------------- mismatch block ----------------
    print("\n" + "=" * 100)
    print("MISMATCH CHECK vs the values you believed (only differing fields are flagged)")
    print("=" * 100)
    any_mismatch = False
    for name in DATASETS:
        r = results.get(name)
        if r is None:
            print(f"\n[{name}]  could not verify (failed to load)")
            any_mismatch = True
            continue
        exp = EXPECTED.get(name, {})
        flags = []
        for field in ("n_channels", "series_length", "n_classes", "n_train", "n_test"):
            had = exp.get(field)
            actual = r.get(field)
            if had is not None and had != actual:
                flags.append(f"    {field}: <-- MISMATCH (had: {had}, actual: {actual})")
        if flags:
            any_mismatch = True
            print(f"\n[{name}]")
            for f in flags:
                print(f)
        else:
            print(f"\n[{name}]  all fields match")
        if r.get("same_labels") is False:
            print("    NOTE: train and test label sets differ!")

    print("\n" + "=" * 100)
    print("RESULT: " + ("MISMATCHES FOUND (see above)" if any_mismatch else
                        "ALL DATASETS MATCH THE EXPECTED VALUES"))
    print("=" * 100)


if __name__ == "__main__":
    main()
