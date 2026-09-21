"""Layer-wise linear probing of a frozen Chronos-2 encoder.

Reusable core package. The experiment drivers in ``experiments/`` import from here.

Typical use:
    from probing.extraction import get_pipeline, extract_window_features
    from probing.probes import PROBES
    from probing.stats import bootstrap_ci, paired_diff_ci
    from probing.config import SEED, NUM_LAYERS, MIDDLE_BAND, LAST_LAYER, BOOT_B, CACHE_DIR, OUT_DIR

WHY THE HEAVY NAMES ARE LAZY. ``get_pipeline`` / ``extract_window_features`` / ``PROBES`` live
in modules that import torch at module scope. Importing them eagerly here meant that ANY
``from probing.x import y`` — including the torch-free ones such as ``probing.registry``,
``probing.mase`` and ``probing.id_data`` — dragged in the whole deep-learning stack. That made
genuinely model-free work (building windows, auditing the roster, resolving a seasonal period)
impossible without a GPU-sized environment, and it is why the window-only smoke and the yield
screen had to reach around the package.

PEP 562 module ``__getattr__`` keeps every existing spelling working — ``from probing import
get_pipeline``, ``probing.PROBES`` — and simply defers the import to first ACCESS. Nothing
about the resolved objects changes; only when torch is loaded does.
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
from probing.stats import bootstrap_ci, paired_diff_ci

# name -> (module, attribute). Resolved on first access, never at import time.
_LAZY = {
    "get_pipeline": ("probing.extraction", "get_pipeline"),
    "extract_window_features": ("probing.extraction", "extract_window_features"),
    "PROBES": ("probing.probes", "PROBES"),
}


def __getattr__(name):                       # PEP 562
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(target[0]), target[1])
    globals()[name] = value                  # cache, so this costs one import at most
    return value


def __dir__():
    return sorted(list(globals()) + list(_LAZY))


__all__ = [
    "SEED", "NUM_LAYERS", "MIDDLE_BAND", "LAST_LAYER", "BOOT_B",
    "CACHE_DIR", "OUT_DIR", "REPO_ROOT",
    "get_pipeline", "extract_window_features",
    "PROBES",
    "bootstrap_ci", "paired_diff_ci",
]
