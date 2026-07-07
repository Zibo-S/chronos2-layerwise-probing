"""Labeled distribution-shift registry (Surgical Fine-Tuning taxonomy).

Lee et al., "Surgical Fine-Tuning Improves Adaptation to Distribution Shifts"
(arXiv:2210.11466), distinguish three shift *levels*, each best matched by tuning a
different part of a network:

    input-level    shift in the raw input distribution      -> earliest layers most affected
    feature-level  shift in intermediate feature statistics -> middle layers most affected
    output-level   shift in the label / target relationship -> latest layers most affected

This module TAGS our shift constructions by level so the layer-wise probing curves can be
read in that frame. It is **structure only** — it introduces no new shift experiments and
duplicates no perturbation code; it re-exports the existing, validated implementations from
``probing.extraction``.

Why this matters for Phase 0: the earlier Gaussian-noise / drift "OOD" perturbations are
**input-level** shifts. Under this taxonomy an input-level shift is predicted to disrupt the
EARLY layers, not to amplify a middle-vs-last gap — so the earlier amplification null is
consistent with the taxonomy rather than a failed tunnel-effect replication. (See
``paper/phase0_fixes.md``.)
"""

from __future__ import annotations

# Re-export the existing perturbation implementations (no duplication).
from probing.extraction import _apply_gaussian_noise, _apply_timewarp, _apply_drift


# --------------------------------------------------------------------------- #
# input-level: perturb the raw input signal. Predicted to disrupt EARLY layers.
# These are exactly the perturbations already used in run_perdataset / run_harden,
# with the parameter values from those runs, now explicitly labeled by level.
# --------------------------------------------------------------------------- #
INPUT_LEVEL = {
    "gauss": {
        "apply": _apply_gaussian_noise,
        "params": {"alpha": 0.25, "seed": 0},
        "description": "Additive per-series/per-channel Gaussian noise (std = alpha * context std).",
    },
    "timewarp": {
        "apply": _apply_timewarp,
        "params": {"factor": 1.2},
        "description": "Linear-interpolation time warp, center-cropped/edge-padded back to length.",
    },
    "drift": {
        "apply": _apply_drift,
        "params": {"amplitude": 0.3},
        "description": "Additive half-cycle sinusoid baseline drift (scaled by per-series std).",
    },
}


# --------------------------------------------------------------------------- #
# feature-level: shift in intermediate feature statistics — operationalized as
# genuine cross-domain transfer (probe trained on one TS domain, evaluated on a
# documented-UNSEEN TS domain). Predicted to affect MIDDLE layers most.
# STUB — to be implemented in Track B; see data/chronos2_seen_manifest.md for the
# documented-unseen reservoir (fev-bench / Chronos Benchmark II / BOOM).
# --------------------------------------------------------------------------- #
FEATURE_LEVEL: dict = {
    # "cross_domain": {"apply": None, "params": {}, "description": "seen-domain probe -> unseen-domain eval"},
}


# --------------------------------------------------------------------------- #
# output-level: shift in the target/label relationship — operationalized as a
# change in the forecasting horizon H (the label the probe must read out).
# Predicted to affect the LATEST layers most (closest to the forecasting head).
# STUB — to be implemented in Track B.
# --------------------------------------------------------------------------- #
OUTPUT_LEVEL: dict = {
    # "horizon_change": {"apply": None, "params": {}, "description": "train H != eval H"},
}


# level name -> registry
SHIFTS = {
    "input_level": INPUT_LEVEL,
    "feature_level": FEATURE_LEVEL,
    "output_level": OUTPUT_LEVEL,
}


def shift_level(name: str) -> str | None:
    """Return the taxonomy level ('input_level'/'feature_level'/'output_level') of a named shift."""
    for level, registry in SHIFTS.items():
        if name in registry:
            return level
    return None
