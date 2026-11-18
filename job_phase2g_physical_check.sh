#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --cpus-per-task=16           # TimesFM-3 forward passes on CPU (no GPU queue)
#SBATCH --mem=48G                    # model + deep copies + one dataset's cached token-15 states
#SBATCH --time=03:00:00              # 14 datasets x 7 depths; resumable per dataset
#SBATCH --output=results/three_model_phase2g_timesfm3_zero_ridge/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2g_timesfm3_zero_ridge/logs/%x-%j.err

# =======================================================================================
# PHASE 2g physical check: the zero-anchored TimesFM-3 adapters in physically truncated models.
#
# CPU only. Reads the Phase-1 caches, the Phase-1 windows and the committed 2g tree; writes ONLY
# results/three_model_phase2g_timesfm3_zero_ridge/physical/{<dataset>,summary}.json.
#
#   sbatch -J p2g-phys job_phase2g_physical_check.sh
#   sbatch -J p2g-phys job_phase2g_physical_check.sh --datasets monash_electricity_hourly --depths 5 19
# Resume = resubmit the identical line (finished datasets are skipped).
# =======================================================================================

set -uo pipefail
source ./job_rebuttal_common.sh

rb_setup "PHASE 2g / physical check of the zero-anchored TimesFM-3 adapters" \
    results/three_model_phase2g_timesfm3_zero_ridge job_phase2g_physical_check.sh "$@"
export PHASE1_CACHE=${PHASE1_CACHE:-$SCRATCH/chronos2/phase1_shared_cache}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

rb_main_begin
rb_stage main python -m experiments.run_phase2g_physical_check --device cpu "$@"
RB_MAIN_RC=$?
rb_main_end
rb_finish
