#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # ONE full-context pass per batch + 21 probe fits
#SBATCH --cpus-per-task=4            # StandardScaler + the numpy target build + data loading
#SBATCH --mem=32G                    # dominated by loading the raw series in build_windows
#SBATCH --time=2:00:00               # cold caches ~20-30 min measured-estimate; warm ~10-15
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

set -euo pipefail

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1               # compute nodes are offline; model + datasets pre-cached
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # never grab every core on a shared node
export PYTHONHASHSEED=0               # deterministic alongside --seed

# =======================================================================================
# HEADLINE TimesFM-3 experiment: LAST-CONTEXT-TOKEN layer-wise probing, native Q=9.
#
#   question   at each layer, how linearly decodable is the native C=512 -> H=64 forecast
#              from the EXACT token the native TimesFM-3 head reads?
#   geometry   ONE full-context decode() pass; 16 real context patches + 2 horizon
#              placeholders = 18 tokens; readout = token 15, the LAST REAL context patch
#              (== decode()'s own forecast index, asserted every batch)
#   probe      one INDEPENDENT Linear(1280, 64*9) per point in {Emb, L1..L20}
#   objective  mean pinball loss over all 64 steps x all 9 NATIVE quantiles (the full native
#              probabilistic forecasting objective). At L20 the probe and the native output
#              head are the same hypothesis class R^1280 -> R^(64x9).
#   tunnel     l_tun = min { l : L_val(l) <= 1.05 * L_val(L20) }, VALIDATION only
#              (1%/2%/5%/10% all saved, so a 2% criterion needs no re-extraction)
#
# NOT this job: the 16-prefix / shared-origin ABLATION (job_timesfm3_probing.sh,
# experiments/run_timesfm3_probing.py, cache tag tfm3-prefix-v1). Separate entry point,
# separate cache tag (tfm3-last-token-q9-fp32-v1) and separate results dir -- they cannot collide,
# and the loader REFUSES the other line's cache instead of reusing it. No Chronos-2 file is
# touched by either.
#
# Cost, from the implementation (4 datasets x 21 points x Q=9, C=512, H=64):
#   feature cache ~500 MB per dataset (21 layers x ~4500 windows x 1280 x float32, lossless)
#                 ~2 GB for all four         ->  $SCRATCH, NEVER $HOME (~50 GB quota)
#   GPU  < 4 GB   (1.3 GB frozen model + a 2400x1280 feature block + a 1280x576 probe)
#   CPU  ~32 GB requested for raw-series loading; the probe stage itself needs < 2 GB
#   time  extraction is ONE pass per batch (16x cheaper than the prefix ablation): a few
#         seconds per dataset. Probe fitting dominates: 21 layers x (8 wd candidates + refit)
#         x 300 full-batch epochs, ~1-3 min per dataset on an A100.
#
# ---------------------------------------------------------------------------------------
# ONE-TIME SETUP ON THE LOGIN NODE (it has internet; compute nodes do not):
#
#   module load gcc python/3.11 arrow/24.0.0 && source .venv/bin/activate
#   pip install "timesfm[torch]==3.0.2"        # not in the cluster wheelhouse -> no --no-index
#   export HF_HOME=$SCRATCH/chronos2/hf_cache
#   python -c "from timesfm3.torch import TimesFM3Forecaster as F; \
#              F.from_pretrained('google/timesfm-3.0-pytorch', device='cpu')"    # ~1.2 GB
#   python -m tests.test_timesfm3_last_token_probe     # model-free contracts: seconds, 2 threads
#
#   The datasets are the SAME autogluon/chronos_datasets the Chronos-2 line uses; if that
#   cache is warm there is nothing else to download.
#
# Resubmitting is cheap: features are cached per (dataset, split, layer) and re-HIT, and the
# cache is REJECTED (never silently reused) if checkpoint, timesfm version, geometry, token
# index, layer set, dtype, detrending, dataset/split or seed disagree. On a cache HIT the L20
# all-quantile native reconstruction is re-proved on one fresh batch (--no-recheck-native
# disables that).
# =======================================================================================

CACHE=${TFM3_LT_CACHE_DIR:-$SCRATCH/timesfm3_last_token/features_cache}
OUT=${TFM3_LT_OUT_ROOT:-$SCRATCH/timesfm3_last_token/results}
CKPT=${TIMESFM3_CHECKPOINT:-google/timesfm-3.0-pytorch}
mkdir -p "$CACHE" "$OUT" logs

if [ "$CACHE" = "${TFM3_CACHE_DIR:-}" ] || [ "$OUT" = "${TFM3_OUT_ROOT:-}" ]; then
    echo "REFUSING TO RUN: this experiment's cache/results must NOT share a directory with"
    echo "the shared-origin ablation (TFM3_CACHE_DIR / TFM3_OUT_ROOT)."
    exit 1
fi

echo "=== TimesFM-3 last-context-token probing (token 15, native Q=9) ==="
echo "    checkpoint=$CKPT"
echo "    cache=$CACHE"
echo "    out=$OUT   HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m experiments.run_timesfm3_last_token_probing \
    --checkpoint "$CKPT" \
    --cache-dir "$CACHE" \
    --out-root "$OUT" \
    --device cuda \
    --seed 0 \
    "$@"
echo "=== DONE ==="

# Useful variants (extra args are forwarded, and a later --flag overrides the one above):
#
#   GPU SMOKE RUN (do this FIRST -- one dataset, three representation points):
#     sbatch --time=0:30:00 -J tfm3lt-smoke job_timesfm3_last_token_q9.sh \
#         --cache-dir $SCRATCH/timesfm3_last_token/smoke_cache \
#         --out-root  $SCRATCH/timesfm3_last_token/smoke_results \
#         --datasets monash_electricity_hourly --layers 0 10 20
#   It prints num_real_context_patches, selected_token_index, feature shapes, the target
#   round-trip error, the L20 all-quantile native reconstruction error, the frozen-backbone
#   status, and then the Emb/L10/L20 probe results.
#
#   sbatch job_timesfm3_last_token_q9.sh --no-detrend
#       ablation: TimesFM-3's linear detrending off in the BACKBONE and in the targets.
#
#   sbatch job_timesfm3_last_token_q9.sh --feature-dtype float16
#       HALVES the cache (~1 GB) via a lossy float16 storage cast (~1.2e-2 of the layer std,
#       measured and reported in feature_dtype_check). float32 (lossless) is the default.
#
#   sbatch job_timesfm3_last_token_q9.sh --allow-sorted-reference
#       ONLY if the L20 check reports that the installed timesfm3 sorts quantiles inside
#       decode() and exposes no bypass knob. Sorting is postprocessing; this records it
#       explicitly instead of silently comparing against a sorted path.
#
#   sbatch job_timesfm3_last_token_q9.sh --collect-history --wd-grid 1e-3 1e-2 1e-1 1 3 10
#       per-epoch train/val curves, and a wider decay grid if a layer selects the maximum
#       (the driver warns when it does).
