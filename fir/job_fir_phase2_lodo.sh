#!/bin/bash
#SBATCH --account=def-irina          # Fir maps it to def-irina_gpu for a GPU request
#SBATCH --gpus-per-node=h100:1       # ONE FULL H100 80GB (never a MIG slice)
#SBATCH --cpus-per-task=4            # = the Narval job: data loading + the bootstrap
#SBATCH --mem=48G                    # = the Narval job (a chunk loads 14 datasets at ONE depth)
#SBATCH --time=03:00:00              # <= 3 h = Fir's shortest GPU tier (b1): starts far sooner than
                                     # 12 h / 24 h requests. One chunk = 6-7 cells (~1-2 h on an H100).
#SBATCH --signal=B:USR1@900          # 15 min before the limit -> automatic continuation
#SBATCH --output=results/three_model_phase2_lodo/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2_lodo/logs/%x-%j.err

# NOTE: no `set -e`: every stage's exit code is recorded (see job_rebuttal_common.sh).
set -uo pipefail

# =======================================================================================
# FIR VARIANT of job_phase2_lodo.sh (Phase 2c, pooled LODO adapter transfer, Chronos-2).
#
# The stages, commands and arguments are those of job_phase2_lodo.sh, unchanged:
#   tests -> plan -> main (preflight gates, then the fits; every cell re-checks its gates).
# Differences (cluster only, no protocol change):
#   * resources: one full H100 instead of one full A100; <= 3 h per job;
#   * the 48 cells are split into disjoint chunks (fir/submit_fir.sh), one job each, e.g.
#       --variants sibling_excluded --depths 4 --chunk 1/2
#     (the driver's own selectors; per-job run dirs and job-scoped staging cleanup make
#     concurrent chunks safe);
#   * tables / report / archive run ONCE after all chunks, in fir/job_fir_aggregate.sh
#     (set RB_POST=1 to run them at the end of this job instead).
# Inputs on Fir mirror the Narval layout under /scratch/eserebri (copied from Narval).
# =======================================================================================

export RB_CLUSTER=fir
source ./job_rebuttal_common.sh       # explicit path: never resolved through $PATH
rb_setup "PHASE 2c / LODO [fir]" results/three_model_phase2_lodo fir/job_fir_phase2_lodo.sh "$@"
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # deterministic cuBLAS (full-batch fits), as in H3
echo "  cluster                     : $RB_CLUSTER"
echo "  phase-1 windows (read-only) : $PHASE1_WINDOWS"
echo "  H3 adapters (read-only)     : $SCRATCH/chronos2_phase2/adapters"
echo "  pooled adapters (written)   : $SCRATCH/chronos2_phase2/adapters_lodo"

rb_stage tests python -m tests.test_phase2_lodo || {
    RB_MAIN_RC=2; echo "model-free contracts FAILED on $(hostname): nothing else runs"; rb_finish; exit $?; }
rb_stage plan python -m experiments.run_phase2_lodo --plan "$@" || {
    RB_MAIN_RC=2; echo "plan FAILED (inputs not readable?): nothing else runs"; rb_finish; exit $?; }

rb_main_begin
rb_stage main python -m experiments.run_phase2_lodo --device cuda --fail-on-error "$@"
RB_MAIN_RC=$?
rb_main_end

if [ "${RB_POST:-0}" = 1 ]; then
    rb_stage tables python -m experiments.make_phase2_lodo_tables
    rb_stage report python -m experiments.make_phase2_lodo_report
    if [ "$RB_MAIN_RC" = 0 ]; then
        rb_stage archive rb_archive fir_phase2c_lodo results/three_model_phase2_lodo \
            "$SCRATCH/chronos2_phase2/adapters_lodo" three_models_paper/figures/lodo
    fi
else
    echo "chunk job: tables / report / archive run once in fir/job_fir_aggregate.sh 2c"
fi
rb_finish
