#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # ONE A100: the full-batch pooled adapter fits
#SBATCH --cpus-per-task=4            # data loading + the bootstrap
#SBATCH --mem=48G                    # forecast-slot states of all 14 datasets (~1.5 GB kept) + loading peaks
#SBATCH --time=03:00:00              # <=3 h = Narval's shortest time tier (most nodes, backfill);
                                     # a timeout costs little: resubmit the SAME line to resume
#SBATCH --output=results/three_model_phase2_lodo/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2_lodo/logs/%x-%j.err

set -euo pipefail

# =======================================================================================
# PHASE 2c -- pooled leave-one-dataset-out (LODO) adapter transfer, Chronos-2, L4 / L6 / L9
#
# Per (variant, target, depth): the H3 label-free adapter fit, UNCHANGED, on the pooled sources
# (each source weighs 1/|S|), then ONE read of the target's test split. Main protocol
# "sibling_excluded": target AND its series sibling (Electricity hourly <-> 15-min) excluded.
# Sensitivity "sibling_included": the 13-source fit, only for the two Electricity targets.
#
# ---------------------------------------------------------------------------------------
# HOW TO USE
# ---------------------------------------------------------------------------------------
#   0. LOGIN NODE (seconds, 2 threads -- model-free only):
#        export OMP_NUM_THREADS=2
#        python -m tests.test_phase2_lodo              # 10 contracts, ~10 s
#        python -m experiments.run_phase2_lodo --plan  # 48 cells, folds, status (JSON reads only)
#
#   1. SMOKE (Electricity hourly x L9, main protocol = the 12-source sibling-excluded path, the
#      REAL 1500 epochs; writes to results/three_model_phase2_lodo_smoke/):
#        sbatch --time=00:45:00 -J p2c-smoke job_phase2_lodo.sh --smoke
#      CHECK in the log: every [data] line; native gate + hard/specific reproduction gates passed
#      (the cell is only written if they pass); the per-cell timings (fit_s, cell_total_s,
#      cuda_max_memory_allocated_gb); the flags (late_checkpoint / grid_max_wd).
#
#   2. FULL. Main protocol: 3 depths x 2 target halves = 6 jobs of 7 cells; then the 6
#      sensitivity cells. They may all run AT THE SAME TIME (each job cleans/logs only its cells):
#        for d in 4 6 9; do for c in 1/2 2/2; do
#          sbatch -J p2c-L$d-${c%/*} job_phase2_lodo.sh --variants sibling_excluded --depths $d --chunk $c
#        done; done
#        sbatch -J p2c-sens job_phase2_lodo.sh --variants sibling_included
#      If the smoke's cell_total_s x 7 + ~10 min of loading exceeds ~2.5 h, use --chunk i/3.
#
#   3. RESUME: resubmit the identical line (COMPLETE cells are skipped).
#   4. TABLES (login node, seconds):  python -m experiments.make_phase2_lodo_tables
#
# WHAT THE SBATCH FLAGS MEAN: --gres=gpu:1 asks for one GPU; --cpus-per-task caps the CPU cores
# (OMP_NUM_THREADS below matches it so numpy never grabs the whole node); --mem is host RAM;
# --time is the wall-clock limit after which SLURM kills the job (a killed cell leaves only a
# .building-* staging dir, which the next run of the same cells deletes).
# =======================================================================================

mkdir -p results/three_model_phase2_lodo/logs

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1                    # compute nodes are offline; checkpoints pre-cached
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # deterministic cuBLAS (full-batch fits)

# READ-ONLY inputs: Phase-1 windows, Chronos-2 forecast-slot caches (features_cache/ -> $SCRATCH),
# the committed H3 cells (results/three_model_phase2) and H3 adapters.
export PHASE1_WINDOWS=${PHASE1_WINDOWS:-$SCRATCH/chronos2/phase1_shared_windows}

echo "=================================================================================="
echo "PHASE 2c / LODO  host=$(hostname)  job=${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "  phase-1 windows (read-only) : $PHASE1_WINDOWS"
echo "  H3 adapters (read-only)     : $SCRATCH/chronos2_phase2/adapters"
echo "  pooled adapters (written)   : $SCRATCH/chronos2_phase2/adapters_lodo"
echo "  extra args                  : $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=================================================================================="

python -m experiments.run_phase2_lodo --device cuda "$@"
