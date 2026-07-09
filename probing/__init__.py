"""Layer-wise linear probing of a frozen Chronos-2 encoder.

Reusable core package. The experiment drivers in ``experiments/`` import from here.

Typical use:
    from probing.extraction import get_pipeline, extract_window_features
    from probing.probes import PROBES
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
from probing.extraction import get_pipeline, extract_window_features
from probing.probes import PROBES
from probing.stats import bootstrap_ci, paired_diff_ci

__all__ = [
    "SEED", "NUM_LAYERS", "MIDDLE_BAND", "LAST_LAYER", "BOOT_B",
    "CACHE_DIR", "OUT_DIR", "REPO_ROOT",
    "get_pipeline", "extract_window_features",
    "PROBES",
    "bootstrap_ci", "paired_diff_ci",
]
