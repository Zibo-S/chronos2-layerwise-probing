"""Central configuration for the layer-wise probing pipeline.

All shared constants and filesystem paths live here so the experiment scripts and the
`probing` package agree on one source of truth. Paths are anchored to the repository
root (not the current working directory), so cache keys are identical no matter where a
script is launched from — this is what keeps results reproducible after the reorg.

Layer indexing (L0 included): L0 is the embedded token sequence entering the first transformer
block (content patches + special tokens, pre-attention); L1..L12 are the outputs of encoder
blocks 1..12. Hence NUM_LAYERS = 13 and LAST_LAYER = 12 (the final block, whose
final_layer_norm output is what the native head consumes). Adding L0 shifts every layer label
by +1 relative to the original committed numbers, which are regenerated under this scheme.
"""

from __future__ import annotations

import os
from pathlib import Path

# ---- reproducibility ----
SEED = 0                       # global seed (numpy + torch + sklearn random_state)

# ---- model / probe geometry ----
NUM_LAYERS = 13                # L0 embedding (pre-block-1) + 12 encoder-block outputs (L1..L12)
MIDDLE_BAND = list(range(4, 10))  # a-priori "middle" layers (same blocks as before, +1 shifted)
LAST_LAYER = NUM_LAYERS - 1    # L12 = last encoder block; native head consumes its final-norm
# Chronos-2 output patch size (== input_patch_size; verified from amazon/chronos-2 config.json).
# The number of native forecast slots for a horizon H is K = ceil(H / OUTPUT_PATCH_SIZE) —
# Chronos-2's own rule (pipeline.get_num_output_patches); predictions for the last partial
# patch are trimmed to H. This is what makes the shared forecast-token extractor/probe
# horizon-aware instead of hardcoding K=4. extract_kout_features asserts the loaded model's
# chronos_config.output_patch_size matches this constant, so the two can never drift apart.
OUTPUT_PATCH_SIZE = 16
BOOT_B = 2000                  # bootstrap resamples for confidence intervals

# ---- ID dataset-set selector ----
# Which named set of ID forecasting datasets a run uses (see probing.id_data.ID_DATASET_SPECS):
#   "phase0_trio"  — the original Phase 0 run (m4_hourly / electricity / solar_1h)
#   "extended_v1"  — four long-series hourly datasets, all within_series splits
# Selected here (env ID_DATASET_SET) rather than in id_data so the results namespacing below
# can never disagree with the dataset list a run actually used.
DATASET_SET = os.environ.get("ID_DATASET_SET", "extended_v1")

# ---- filesystem (anchored to repo root = parent of this package) ----
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "features_cache"   # extracted hidden-state features (~13 GB, gitignored)
OUT_DIR = REPO_ROOT / "results"            # shared inputs (e.g. the UEA classification summary)

CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

# UEA classification baseline outputs (maintained baseline; set-independent, so NOT under
# the ID dataset-set namespacing below).
UEA_OUT_DIR = OUT_DIR / "uea"
UEA_OUT_DIR.mkdir(parents=True, exist_ok=True)

# All ID-forecasting outputs are namespaced per dataset set so phase0_trio and extended_v1
# results can never overwrite each other. Derivation lives in one function so the CLI
# override (set_dataset_set) re-derives exactly what import time derived.
def _derive_id_dirs() -> None:
    global ID_OUT_DIR, QUANT_DIR, BOOT_DIR
    ID_OUT_DIR = OUT_DIR / DATASET_SET
    ID_OUT_DIR.mkdir(parents=True, exist_ok=True)

    QUANT_DIR = ID_OUT_DIR / "quantile_loss"   # quantile-probe figures + focused JSON
    for _sub in ("content", "reg", "pooling_comparison", "training_curves"):
        (QUANT_DIR / _sub).mkdir(parents=True, exist_ok=True)

    # Series-level cluster bootstrap: the GPU run writes per-window test metrics to inputs/;
    # experiments.run_bootstrap (CPU-only, post-hoc) reads them and fills raw/tables/figures.
    BOOT_DIR = ID_OUT_DIR / "bootstrap"
    (BOOT_DIR / "inputs").mkdir(parents=True, exist_ok=True)


_derive_id_dirs()


def set_dataset_set(name: str) -> None:
    """CLI override of the active ID dataset set (precedence: CLI > env > default).

    Validates against probing.id_data.ID_DATASET_SPECS, updates DATASET_SET, and re-derives
    the namespaced output dirs (ID_OUT_DIR / QUANT_DIR / BOOT_DIR). NOTE: `from`-imports of
    these names taken at import time are snapshots — a caller honoring the override must
    re-read them from this module afterwards (experiments.run_id_forecasting does)."""
    from probing.id_data import ID_DATASET_SPECS   # deferred: id_data imports config
    if name not in ID_DATASET_SPECS:
        raise ValueError(f"unknown dataset set {name!r}; known sets: {sorted(ID_DATASET_SPECS)}")
    global DATASET_SET
    DATASET_SET = name
    _derive_id_dirs()
