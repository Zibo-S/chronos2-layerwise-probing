"""Central configuration for the layer-wise probing pipeline.

All shared constants and filesystem paths live here so the experiment scripts and the
`probing` package agree on one source of truth. Paths are anchored to the repository
root (not the current working directory), so cache keys are identical no matter where a
script is launched from — this is what keeps results reproducible after the reorg.

Nothing here changes the numerical behaviour of the original scripts: SEED, NUM_LAYERS,
the middle band (L3-8), the last layer (L11) and the bootstrap resample count (2000) are
exactly the values used to produce the committed results.
"""

from __future__ import annotations

from pathlib import Path

# ---- reproducibility ----
SEED = 0                       # global seed (numpy + torch + sklearn random_state)

# ---- model / probe geometry ----
NUM_LAYERS = 12                # Chronos-2 encoder blocks
MIDDLE_BAND = list(range(3, 9))  # a-priori "middle" layers L3..L8 (inclusive)
LAST_LAYER = NUM_LAYERS - 1    # L11
# Chronos-2 output patch size (== input_patch_size; verified from amazon/chronos-2 config.json).
# The number of native forecast slots for a horizon H is K = ceil(H / OUTPUT_PATCH_SIZE) —
# Chronos-2's own rule (pipeline.get_num_output_patches); predictions for the last partial
# patch are trimmed to H. This is what makes the shared forecast-token extractor/probe
# horizon-aware instead of hardcoding K=4. extract_kout_features asserts the loaded model's
# chronos_config.output_patch_size matches this constant, so the two can never drift apart.
OUTPUT_PATCH_SIZE = 16
BOOT_B = 2000                  # bootstrap resamples for confidence intervals

# ---- filesystem (anchored to repo root = parent of this package) ----
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "features_cache"   # extracted hidden-state features (~13 GB, gitignored)
OUT_DIR = REPO_ROOT / "results"            # figures + summary JSON land here

CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)

QUANT_DIR = OUT_DIR / "quantile_loss"          # quantile-probe figures + focused JSON
for _sub in ("content", "reg", "pooling_comparison", "training_curves"):
    (QUANT_DIR / _sub).mkdir(parents=True, exist_ok=True)

# Series-level cluster bootstrap: the GPU run writes per-window test metrics to inputs/;
# experiments.run_bootstrap (CPU-only, post-hoc) reads them and fills raw/tables/figures.
BOOT_DIR = OUT_DIR / "bootstrap"
(BOOT_DIR / "inputs").mkdir(parents=True, exist_ok=True)
