#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # REQUIRED: this runs the backbone (see below)
#SBATCH --cpus-per-task=4            # window building + the numpy target/metric work
#SBATCH --mem=32G                    # dominated by build_windows loading the raw series
#SBATCH --time=1:30:00               # 7 test splits = 1804 windows; window building dominates
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONHASHSEED=0

# =======================================================================================
# FROZEN NATIVE-HEAD TRANSFER across depth -- TimesFM-3, paper7 seven, TEST splits only.
#
#   h_{l,15}  --model.output_head (pretrained, frozen)-->  H=64 x Q=9 forecast,  l = Emb..L20
#
# No probe, no optimizer, no weight decay, no adapter, no parameter update. The only variable
# is the depth l.
#
# WHY THIS NEEDS A GPU -- and why it does NOT use the feature cache.
# The experiment's validity rests on one identity: at L20 this path IS the native forecasting
# pathway, so it must reproduce decode()'s own nine quantiles. That identity is EXACT
# (max_abs == 0.0) when the head is applied to the states decode() just produced, in memory,
# on the same device. It is NOT exact through the feature cache: reloading the token-15 state
# and re-applying the head changes the matmul shape (and can change the device), and on
# TF32-enabled GPUs that is not bit-identical -- the cached implementation this replaces
# measured ~1e-4 relative divergence and correctly refused to proceed.
# So this job runs the backbone itself: ONE decode() pass per batch, the frozen head applied to
# all 21 representation points in memory, no serialization in between.
#
# The tolerance was NOT loosened to make the cached version pass; the cache was removed.
#
# NOT this job: representation geometry (CKA + effective rank) is genuinely cache-based and
# model-free -- run job_timesfm3_geometry.sh for that, on CPU.
#
# Prerequisite: none beyond the checkpoint and the datasets. --probe-results is optional and
# only adds the learned-probe comparison and the alignment gap A_l.
#
# Outputs: lightweight summary + figures inside the repo at
#   results/timesfm3_representation_geometry/{native_head_transfer_*.json,*.csv,figures/native_head/}
# heavy per-dataset JSON under $SCRATCH.
#
# Usage:
#   sbatch job_timesfm3_native_head.sh
#   sbatch job_timesfm3_native_head.sh --datasets coastal_ts      # one dataset, quick check
# =======================================================================================

PROBE_SUMMARY=${TFM3_PROBE_SUMMARY:-$SCRATCH/timesfm3_last_token/results/timesfm3_last_token_paper7_q9/timesfm3_last_token_summary.json}
WORK=${TFM3_GEOM_OUT_ROOT:-$SCRATCH/timesfm3_geometry}
CKPT=${TIMESFM3_CHECKPOINT:-google/timesfm-3.0-pytorch}

echo "checkpoint   : $CKPT"
echo "probe summary: $PROBE_SUMMARY"
echo "work         : $WORK"
echo "repo outputs : $(git rev-parse --show-toplevel)/results/timesfm3_representation_geometry"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || {
  echo "ERROR: no GPU visible. This job runs the backbone; submit it with --gres=gpu:1."; exit 1; }

PROBE_ARG=()
if [ -f "$PROBE_SUMMARY" ]; then
  PROBE_ARG=(--probe-results "$PROBE_SUMMARY")
else
  echo "[warn] no probe summary at $PROBE_SUMMARY: the frozen-head curve will be saved alone, "
  echo "       with no alignment gap and no probe comparison figure."
fi

python -m experiments.run_timesfm3_native_head_transfer \
    --checkpoint "$CKPT" \
    --out-root "$WORK" \
    "${PROBE_ARG[@]}" \
    "$@"

echo; echo "done. Final paper outputs are inside the repo:"
echo "  $(git rev-parse --show-toplevel)/results/timesfm3_representation_geometry/figures/native_head/"
