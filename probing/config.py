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
BOOT_B = 2000                  # bootstrap resamples for confidence intervals

# ---- filesystem (anchored to repo root = parent of this package) ----
REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "features_cache"   # extracted hidden-state features (~13 GB, gitignored)
OUT_DIR = REPO_ROOT / "results"            # figures + summary JSON land here

CACHE_DIR.mkdir(exist_ok=True)
OUT_DIR.mkdir(exist_ok=True)
