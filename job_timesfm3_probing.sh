#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): 16 prefix passes/batch + 21 probe fits
#SBATCH --cpus-per-task=4            # StandardScaler + the numpy target build
#SBATCH --mem=32G                    # measured peak ~0.5 GB/layer of probe matrices
#SBATCH --time=4:00:00               # cold caches; a resubmit re-HITs them and is much shorter
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

set -euo pipefail

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1               # compute nodes are offline; model + datasets pre-cached
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # never grab every core on a shared node
export PYTHONHASHSEED=0               # deterministic alongside --seed

# TimesFM-3 layer-wise shared-origin probing, INDEPENDENT CAUSAL PREFIXES, Q=1 / tau=0.5.
# The Chronos-2 line (code and results/ namespaces) is untouched.
#
#   origin j in 1..16 : its OWN decode() pass on x[1:32j]; readout = token j-1, the last REAL
#                       context patch. Detrend coefficients and RevIN stats therefore depend
#                       on x[1:32j] alone -- a single 512-point pass would NOT be causal,
#                       because the detrend line is fit on the whole supplied context.
#   probe             : one Linear(1280, 64) per representation point {Emb, L1..L20}, shared
#                       across all 16 origins, trained on every valid origin.
#   headline          : origin 16 only (C=512 -> H=64), the real forecasting task.
#
# Feature cache ~15.5 GB (21 layers x 16 origins x 1280 x 4500 windows x 4 datasets, float16,
# one .npy per layer) -> $SCRATCH, NEVER $HOME (~50 GB quota).
#
# ---------------------------------------------------------------------------------------
# ONE-TIME SETUP ON THE LOGIN NODE (it has internet; compute nodes do not):
#
#   module load gcc python/3.11 arrow/24.0.0 && source .venv/bin/activate
#   pip install "timesfm[torch]==3.0.2"        # not in the cluster wheelhouse -> no --no-index
#   export HF_HOME=$SCRATCH/chronos2/hf_cache
#   python -c "from timesfm3.torch import TimesFM3Forecaster as F; \
#              F.from_pretrained('google/timesfm-3.0-pytorch', device='cpu')"    # ~1.2 GB
#   python -m tests.test_timesfm3_probe        # model-free contracts: seconds, one core
#
#   The probing datasets are the SAME autogluon/chronos_datasets the Chronos-2 line uses; if
#   that cache is warm there is nothing else to download.
#
# Resubmitting is cheap: extraction is cached per (dataset, split, layer) and re-HIT, and the
# cache is REJECTED (never silently reused) if checkpoint, timesfm version, geometry, layer
# set, detrending, dataset/split or seed disagree.
# ---------------------------------------------------------------------------------------

CACHE=${TFM3_CACHE_DIR:-$SCRATCH/timesfm3/features_cache}
OUT=${TFM3_OUT_ROOT:-$SCRATCH/timesfm3/results}
CKPT=${TIMESFM3_CHECKPOINT:-google/timesfm-3.0-pytorch}
mkdir -p "$CACHE" "$OUT" logs

echo "=== TimesFM-3 causal-prefix probing (Q=1) ==="
echo "    checkpoint=$CKPT  cache=$CACHE  out=$OUT  HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m experiments.run_timesfm3_probing \
    --checkpoint "$CKPT" \
    --cache-dir "$CACHE" \
    --out-root "$OUT" \
    --device cuda \
    --seed 0 \
    "$@"
echo "=== DONE ==="

# Useful variants (extra args are forwarded):
#
#   sbatch job_timesfm3_probing.sh --origins 8 9 10 11 12 13 14 15 16
#       sensitivity to the context-length shift: origin j has context 32j, so training on all
#       16 origins mixes C in {32..512} while the headline is C=512 only. Short prefixes are
#       far outside TimesFM-3's operating regime.
#
#   sbatch job_timesfm3_probing.sh --no-detrend
#       turns TimesFM-3's linear detrending off in the BACKBONE and in the targets. Then the
#       prefix and single-pass constructions coincide exactly, so this also measures how much
#       the causal-prefix fix is worth on this data.
#
#   sbatch job_timesfm3_probing.sh --datasets monash_electricity_hourly --layers 0 10 20
#       fast smoke run: one dataset, three representation points.
#
#   sbatch job_timesfm3_probing.sh --diagnostic-prefix-vs-full
#       records per-origin |h_prefix - h_full| (expected to be non-zero where detrending fires).
