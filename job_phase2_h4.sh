#!/bin/bash
#SBATCH --account=def-irina
#SBATCH --gres=gpu:1                 # physical truncated models run on the GPU
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=03:00:00              # <=3 h tier; evaluate resumes (COMPLETE cells skipped)
#SBATCH --output=results/three_model_phase2/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2/logs/%x-%j.err

set -euo pipefail

# =======================================================================================
# PHASE 2 / H4 -- physically truncate at the frozen Phase-1 entrance, verify, score.
#
#   stage "verify"   V1 truncated states == hooked full model (every depth), V2 identity splice ==
#                    native through the public API, V3 physical == offline with a non-trivial
#                    adapter, V4 removed blocks never run + are freed, V5 identity == hard cut,
#                    V6 TiRex passes, PH folded probe head == scaler + probe.
#                    -> results/three_model_phase2/h4/verification/<model>.json
#                    The latency harness REFUSES to time anything this stage did not pass.
#   stage "evaluate" per dataset (needs the COMPLETE H3 cell): native | hard-cut truncation |
#                    label-free aligned truncation | supervised aligned truncation (C2, TiRex) |
#                    probe-head truncation | Chronos-2-small; V3 vs the offline H3 prediction on
#                    every test window; test MASE/WQL, paired CI vs native, measured parameters.
#
# USAGE
#   sbatch --time=01:00:00 -J p2h4-verify job_phase2_h4.sh verify \
#          --cache-dataset monash_electricity_hourly        # also V1 vs the Phase-1 cache
#   sbatch --time=01:00:00 -J p2h4-eval-chronos2 job_phase2_h4.sh evaluate --models chronos2
#   sbatch --time=01:00:00 -J p2h4-bud-chronos2  job_phase2_h4.sh evaluate --operating-points budget \
#          --models chronos2       # the MAIN H4 result: the validation-chosen budget cuts, physical
#   (one job per model and mode; resubmitting the same line skips COMPLETE cells)
#   (Chronos-2-small must be pre-downloaded on the LOGIN node, which has internet -- a download
#    only, the model is not loaded, so it is seconds of network I/O at ~0% CPU:
#      python -c "from huggingface_hub import snapshot_download; snapshot_download('autogluon/chronos-2-small')"
#    with HF_HOME=$SCRATCH/chronos2/hf_cache exported first.)
# =======================================================================================

mkdir -p results/three_model_phase2/logs
module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}
export PHASE1_CACHE=${PHASE1_CACHE:-$SCRATCH/chronos2/phase1_shared_cache}
export PHASE1_WINDOWS=${PHASE1_WINDOWS:-$SCRATCH/chronos2/phase1_shared_windows}

echo "PHASE 2 / H4  host=$(hostname)  job=${SLURM_JOB_ID:-none}  $(date -Is)  args: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
python -m experiments.run_phase2_truncation "$@" --device cuda
