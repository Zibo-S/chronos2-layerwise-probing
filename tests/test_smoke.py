"""Behavior-preservation smoke test (fast; reads cache + committed JSON, no model load).

Confirms the refactored `probing` package reproduces the committed per-layer results
EXACTLY. If anything here fails, the tidy-up changed the numbers — stop and investigate.

Run:  python tests/test_smoke.py   (or `make smoke`)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from probing import extract_features, linear_probe, NUM_LAYERS, MIDDLE_BAND, LAST_LAYER

SUMMARY = json.load(open(REPO / "results" / "perdataset_summary.json"))["datasets"]

# Small/cheap datasets whose clean features are already cached.
CHECK = ["Handwriting", "Epilepsy", "SelfRegulationSCP1"]
TOL = 1e-9


def check_dataset(name: str) -> None:
    gold = SUMMARY[name]
    f_tr, y_tr = extract_features(name, "train", pooling="content")
    f_te, y_te = extract_features(name, "test", pooling="content")

    # (1) per-layer ID accuracy reproduces the committed array exactly
    scores = linear_probe(f_tr, y_tr, f_te, y_te)
    got = np.array([scores[i] for i in range(NUM_LAYERS)])
    want = np.array(gold["per_layer_accuracy"]["ID"])
    acc_err = float(np.abs(got - want).max())
    assert acc_err < TOL, f"{name}: per-layer ID accuracy drift {acc_err:.2e}"

    # (2) late_drop_band point (mean L3-8 minus L11) reproduces the committed value
    late_drop = float(np.mean([got[i] for i in MIDDLE_BAND]) - got[LAST_LAYER])
    want_ld = gold["id_late_drop_band"]["point"]
    ld_err = abs(late_drop - want_ld)
    assert ld_err < TOL, f"{name}: late_drop_band drift {ld_err:.2e}"

    print(f"  PASS  {name:<20}  max|Δacc|={acc_err:.1e}  |Δlate_drop|={ld_err:.1e}  "
          f"(argmax L{int(np.argmax(got))})")


def main() -> int:
    print("Smoke test: refactored probing package vs committed results/perdataset_summary.json")
    for name in CHECK:
        check_dataset(name)
    print("ALL SMOKE CHECKS PASSED — numbers are behavior-preserving.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
