#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --cpus-per-task=8            # closed-form fits: eigendecompositions + matrix products
#SBATCH --mem=32G                    # one dataset's 21-depth token-15 states (< 1 GB) + the model
#SBATCH --time=03:00:00              # ~1 h expected (280 closed-form fits + bootstraps); NO GPU
#SBATCH --output=results/three_model_phase2g_timesfm3_zero_ridge/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2g_timesfm3_zero_ridge/logs/%x-%j.err

# =======================================================================================
# PHASE 2g -- TimesFM-3: identity-anchored vs zero-anchored closed-form alignment.
#
# CPU only (no --gres): every fit is a closed form on cached token-15 states. The committed H3
# tree (results/three_model_phase2) and the Phase-1 caches are READ-ONLY; everything is written
# under results/three_model_phase2g_timesfm3_zero_ridge/ (or the --output-root you pass).
#
#   mkdir -p results/three_model_phase2g_timesfm3_zero_ridge/logs
#   sbatch -J p2g-zero job_phase2g_timesfm3_zero_ridge.sh                 # all 14 datasets
#   sbatch --time=00:20:00 -J p2g-smoke job_phase2g_timesfm3_zero_ridge.sh \
#       --datasets monash_electricity_hourly --depths 0 5 19 \
#       --output-root results/three_model_phase2g_timesfm3_zero_ridge_smoke
# Resume = resubmit the identical line (COMPLETE cells are skipped).
# =======================================================================================

set -uo pipefail
source ./job_rebuttal_common.sh       # modules + scratch venv + caches off /home + provenance

rb_setup "PHASE 2g / TimesFM-3 zero-anchored ridge" \
    results/three_model_phase2g_timesfm3_zero_ridge job_phase2g_timesfm3_zero_ridge.sh "$@"
export PHASE1_CACHE=${PHASE1_CACHE:-$SCRATCH/chronos2/phase1_shared_cache}

rb_stage selfcheck python -m experiments.run_phase2g_timesfm3_zero_ridge --selfcheck || {
    RB_MAIN_RC=2; echo "synthetic self-check FAILED: nothing else runs"; rb_finish; exit $?; }
rb_stage plan python -m experiments.run_phase2g_timesfm3_zero_ridge --plan "$@" || {
    RB_MAIN_RC=2; echo "plan FAILED (an input is missing): nothing else runs"; rb_finish; exit $?; }

rb_main_begin
rb_stage main python -m experiments.run_phase2g_timesfm3_zero_ridge --device cpu "$@"
RB_MAIN_RC=$?
rb_main_end

rb_stage tables python -m experiments.run_phase2g_timesfm3_zero_ridge --tables-only "$@"
rb_finish
