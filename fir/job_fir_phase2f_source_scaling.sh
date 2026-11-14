#!/bin/bash
#SBATCH --account=def-irina          # Fir maps it to def-irina_gpu for a GPU request
#SBATCH --gpus-per-node=h100:1       # ONE FULL H100 80GB (never a MIG slice)
#SBATCH --cpus-per-task=4            # = the Narval job
#SBATCH --mem=48G                    # = the Narval job
#SBATCH --time=03:00:00              # <= 3 h = Fir's shortest GPU tier; one job = one target
                                     # (10 fits = 52 one-source fit units, < 1 h on an H100)
#SBATCH --signal=B:USR1@900          # 15 min before the limit -> automatic continuation
#SBATCH --output=results/three_model_phase2f_source_scaling/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2f_source_scaling/logs/%x-%j.err

set -uo pipefail

# =======================================================================================
# FIR VARIANT of job_phase2f_source_scaling.sh (Phase 2f, source-dataset scaling, L6).
#
# Stages and commands are those of job_phase2f_source_scaling.sh. Differences (cluster only):
#   * resources: one full H100; <= 3 h per job;
#   * ONE JOB PER TARGET (fir/submit_fir.sh passes --targets <t>): each job runs that target's
#     preflight, then its N = 1 cells FIRST (each must reproduce the committed H3 fit of its
#     single source; a failure stops the job, exit 4), then N = 4, 8, 13 -- the Narval order,
#     restricted to one target;
#   * tables / report / archive run ONCE after all targets, in fir/job_fir_aggregate.sh 2f.
# =======================================================================================

export RB_CLUSTER=fir
source ./job_rebuttal_common.sh       # explicit path: never resolved through $PATH
rb_setup "PHASE 2f / source scaling [fir]" results/three_model_phase2f_source_scaling fir/job_fir_phase2f_source_scaling.sh "$@"
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # deterministic cuBLAS (full-batch fits), as in H3 / 2c
echo "  cluster                     : $RB_CLUSTER"
echo "  phase-1 windows (read-only) : $PHASE1_WINDOWS"
echo "  H3 adapters (read-only)     : $SCRATCH/chronos2_phase2/adapters"
echo "  2f adapters (written)       : $SCRATCH/chronos2/phase2f_source_scaling/adapters"

rb_stage tests python -m tests.test_phase2f_source_scaling || {
    RB_MAIN_RC=2; echo "model-free contracts FAILED on $(hostname): nothing else runs"; rb_finish; exit $?; }
rb_stage plan python -m experiments.run_phase2f_source_scaling --plan --check-inputs "$@" || {
    RB_MAIN_RC=2; echo "plan / input check FAILED: nothing else runs"; rb_finish; exit $?; }

rb_main_begin
rb_stage main python -m experiments.run_phase2f_source_scaling --device cuda --fail-on-error "$@"
RB_MAIN_RC=$?
rb_main_end

if [ "${RB_POST:-0}" = 1 ]; then
    rb_stage tables python -m experiments.make_phase2f_tables
    rb_stage report python -m experiments.make_phase2f_report
    if [ "$RB_MAIN_RC" = 0 ]; then
        rb_stage archive rb_archive fir_phase2f_source_scaling results/three_model_phase2f_source_scaling \
            "$SCRATCH/chronos2/phase2f_source_scaling/adapters" three_models_paper/figures/source_scaling
    fi
else
    echo "per-target job: tables / report / archive run once in fir/job_fir_aggregate.sh 2f"
fi
rb_finish
