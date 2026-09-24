#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # ONE A100: the iterative adapter fits (full batch, 1500 epochs)
#SBATCH --cpus-per-task=4            # data loading + the closed-form TimesFM-3 eigendecompositions
#SBATCH --mem=48G                    # one cell's features + windows, as job_phase2_h3.sh
#SBATCH --time=01:30:00              # estimate ~45 min: 3 models x 3 depths x 2 iterative fits
#SBATCH --output=results/three_model_phase2/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2/logs/%x-%j.err
#
# PHASE 2b SMOKE (exploratory) -- nonlinear branch on the output-matched adapter, Electricity.
# Pre-registered in notes/PLAN.md. Reads Phase 1 + the committed H3 tree; writes ONLY
# results/three_model_phase2_nonlinear_smoke/summary.json (no adapters saved).
#
#   sbatch -J p2b-nonlin job_phase2_nonlinear_smoke.sh
#   sbatch -J p2b-nonlin-tfm3 job_phase2_nonlinear_smoke.sh --models timesfm3   # one model

set -euo pipefail
mkdir -p results/three_model_phase2/logs

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1                    # compute nodes are offline; checkpoints pre-cached
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # deterministic cuBLAS (full-batch fits)
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}
export PHASE1_CACHE=${PHASE1_CACHE:-$SCRATCH/chronos2/phase1_shared_cache}
export PHASE1_WINDOWS=${PHASE1_WINDOWS:-$SCRATCH/chronos2/phase1_shared_windows}

echo "PHASE 2b NONLINEAR SMOKE  host=$(hostname)  job=${SLURM_JOB_ID:-none}  $(date -Is)"
echo "  extra args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m experiments.run_phase2_nonlinear_smoke --device cuda "$@"
