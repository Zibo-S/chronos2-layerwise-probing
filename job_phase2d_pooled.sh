#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # ONE A100: TimesFM-3 / TiRex pooled extraction + probe fits
#SBATCH --cpus-per-task=4            # StandardScaler, targets, the cluster bootstrap
#SBATCH --mem=32G                    # windows come from the Phase-1 window cache (no rebuild);
                                     # TimesFM-3 states ~0.3 GB per dataset, TiRex ~0.2 GB
#SBATCH --time=03:00:00              # <=3 h = Narval's shortest time tier (most nodes, backfill);
                                     # a timeout costs little: resubmit the SAME line to resume
#SBATCH --output=results/three_model_phase2d_pooled/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2d_pooled/logs/%x-%j.err

set -euo pipefail

# =======================================================================================
# PHASE 2d -- content-mean-pooled vs architecture-aware layerwise readouts (3 models x 14)
#
# Per (model, dataset): the native arm is READ from the committed Phase-1 cell (never refit)
# and reproduced from its artifacts; the generic arms are fit on the SAME windows, the SAME
# forward pass's states, the SAME targets, with the Phase-1 cell's own lr / epochs / wd grid:
#     pooled       mean of valid content tokens     -> Linear(d, 64*9)       all three models
#     concat       native readout states, concat'd  -> Linear(K*d, 64*9)     Chronos-2, TiRex
#     native_mean  native readout states, averaged  -> Linear(d, 64*9)       Chronos-2, TiRex
#
# INDEPENDENT OF PHASE 2c. Writes ONLY:
#     results/three_model_phase2d_pooled[_smoke]/      (compact cell artifacts, in the repo)
#     $SCRATCH/chronos2/phase2d_pooled_cache/          (pooled TimesFM-3 / TiRex states)
#     $SCRATCH/chronos2/phase2d_pooled_artifacts/      (readout checkpoints + predictions)
# READS (never writes): the Phase-1 cells, $SCRATCH/chronos2/phase1_shared_windows,
#     $SCRATCH/chronos2/phase1_shared_cache, and the Chronos-2 K4 caches in features_cache/.
# The driver refuses any --output-root inside a Phase-1 / Phase-2 / LODO tree.
#
# ENVIRONMENT: mirrors the Phase-1 job exactly (no CUBLAS_WORKSPACE_CONFIG, same batch sizes),
# so the pass that produces the pooled states can reproduce the Phase-1 cached readout states --
# that identity is a GATE (max|d|/std <= 1e-3), not an assumption.
#
# ---------------------------------------------------------------------------------------
# HOW TO USE
# ---------------------------------------------------------------------------------------
#  0. LOGIN NODE, once (seconds; model-free; 2 threads):
#       mkdir -p results/three_model_phase2d_pooled/logs     # SLURM opens the log file before
#                                                            # the job starts: the dir must exist
#       export OMP_NUM_THREADS=2
#       python -m tests.test_phase2d_pooled                  # 20 contract groups, ~15 s
#       python -m experiments.run_phase2d_pooled --plan      # 42 cells, sources, param table
#
#  1. SMOKE (TiRex x Electricity, the riskiest path; plus one-batch checks for all three
#     models on Electricity and a native-refit check). Output: results/three_model_phase2d_pooled_smoke/
#       sbatch --time=00:45:00 -J p2d-smoke job_phase2d_pooled.sh --smoke
#
#  2. STOP AND REVIEW the smoke log + results/three_model_phase2d_pooled_smoke/runs/<jobid>/:
#       one_batch_checks.json     all three PASS; token ranges; identity vs Phase-1 caches
#       native_refit__tirex__*    status pass (Phase-1 fitter reproduces the committed curve)
#       the cell table printed at the end of the log; timings in cells.json; seff <jobid>
#
#  3. FULL RUN -- one job per model (each resumes: resubmit the same line after a timeout):
#       sbatch -J p2d-chronos2 job_phase2d_pooled.sh --models chronos2
#       sbatch -J p2d-timesfm3 job_phase2d_pooled.sh --models timesfm3
#       sbatch -J p2d-tirex    job_phase2d_pooled.sh --models tirex
#     If the smoke says TiRex needs > ~2.5 h for 14 datasets, split it in two:
#       sbatch -J p2d-tirex-a job_phase2d_pooled.sh --models tirex --datasets <first 7 tags>
#
#  4. LOGIN NODE (aggregation + plotting of computed numbers only):
#       python -m experiments.make_phase2d_tables
# =======================================================================================

mkdir -p results/three_model_phase2d_pooled/logs

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1                    # compute nodes are offline; checkpoints pre-cached
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONHASHSEED=0
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}
export PHASE1_WINDOWS=${PHASE1_WINDOWS:-$SCRATCH/chronos2/phase1_shared_windows}

echo "=================================================================================="
echo "PHASE 2d / pooled readouts  host=$(hostname)  job=${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "  phase-1 windows (read-only)  : $PHASE1_WINDOWS"
echo "  phase-1 caches (read-only)   : $SCRATCH/chronos2/phase1_shared_cache, features_cache/"
echo "  pooled cache (written)       : $SCRATCH/chronos2/phase2d_pooled_cache"
echo "  readout checkpoints (written): $SCRATCH/chronos2/phase2d_pooled_artifacts"
echo "  extra args                   : $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=================================================================================="

python -m experiments.run_phase2d_pooled --device cuda "$@"
