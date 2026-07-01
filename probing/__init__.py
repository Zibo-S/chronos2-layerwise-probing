"""Layer-wise linear probing of a frozen Chronos-2 encoder.

Reusable core package. The experiment drivers in ``experiments/`` import from here.

Typical use:
    from probing.extraction import extract_features, fit_layerwise_probes
    from probing.probes import PROBES, linear_probe, score_layerwise_correctness
    from probing.stats import bootstrap_ci, paired_diff_ci
    from probing.config import SEED, NUM_LAYERS, MIDDLE_BAND, LAST_LAYER, BOOT_B, CACHE_DIR, OUT_DIR
"""

from probing.config import (
    SEED,
    NUM_LAYERS,
    MIDDLE_BAND,
    LAST_LAYER,
    BOOT_B,
    CACHE_DIR,
    OUT_DIR,
    REPO_ROOT,
)
from probing.extraction import extract_features, fit_layerwise_probes, get_pipeline
from probing.probes import PROBES, linear_probe, score_layerwise_correctness
from probing.stats import bootstrap_ci, paired_diff_ci

__all__ = [
    "SEED", "NUM_LAYERS", "MIDDLE_BAND", "LAST_LAYER", "BOOT_B",
    "CACHE_DIR", "OUT_DIR", "REPO_ROOT",
    "extract_features", "fit_layerwise_probes", "get_pipeline",
    "PROBES", "linear_probe", "score_layerwise_correctness",
    "bootstrap_ci", "paired_diff_ci",
]
