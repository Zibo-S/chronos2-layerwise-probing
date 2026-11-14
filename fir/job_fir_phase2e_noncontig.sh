#!/bin/bash
#SBATCH --account=def-irina          # Fir maps it to def-irina_gpu for a GPU request
#SBATCH --gpus-per-node=h100:1       # ONE FULL H100 80GB (the latency stage refuses a MIG slice)
#SBATCH --cpus-per-task=12           # = the Narval job (latency api_e2e is host-side Python);
                                     # 12 = one H100's share of a Fir GPU node (48 cores / 4)
#SBATCH --mem=64G                    # = the Narval job
#SBATCH --time=03:00:00              # <= 3 h = Fir's shortest GPU tier; one job = 4-5 datasets
#SBATCH --signal=B:USR1@900          # 15 min before the limit -> automatic continuation
#SBATCH --output=results/three_model_phase2e_noncontig/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2e_noncontig/logs/%x-%j.err

set -uo pipefail

# =======================================================================================
# FIR VARIANT of job_phase2e_noncontig.sh (Phase 2e, non-contiguous block selection).
#
# Stages and commands are those of job_phase2e_noncontig.sh. Differences (cluster only):
#   * resources: one full H100; <= 3 h per job;
#   * the 14 datasets are split into disjoint --datasets groups (fir/submit_fir.sh); every job
#     runs the model-level mask verification before its datasets, as on Narval;
#   * the latency stage runs in ONE of the jobs (PHASE2E_SKIP_LATENCY=1 in the others). It
#     times the masks on THIS GPU: on Fir that is an H100, not the paper's A100, and every
#     latency record carries the GPU name;
#   * tables + archive run ONCE after all groups, in fir/job_fir_aggregate.sh 2e.
# =======================================================================================

export RB_CLUSTER=fir
source ./job_rebuttal_common.sh       # explicit path: never resolved through $PATH
rb_setup "PHASE 2e / non-contiguous [fir]" results/three_model_phase2e_noncontig fir/job_fir_phase2e_noncontig.sh "$@"
echo "  cluster                 : $RB_CLUSTER"
echo "  H3 adapters (read-only) : $SCRATCH/chronos2_phase2/adapters"
echo "  2e adapters + states    : $SCRATCH/chronos2/phase2e_noncontig/"
echo "  latency stage           : $([ "${PHASE2E_SKIP_LATENCY:-0}" = 1 ] && echo skipped in this job || echo runs in this job)"

rb_stage tests python -m tests.test_phase2e_noncontig || {
    RB_MAIN_RC=2; echo "model-free contracts FAILED on $(hostname): nothing else runs"; rb_finish; exit $?; }
rb_stage plan python -m experiments.run_phase2e_noncontig --plan --check-inputs "$@" || {
    RB_MAIN_RC=2; echo "plan / input check FAILED: nothing else runs"; rb_finish; exit $?; }

rb_main_begin
# deterministic cuBLAS for the full-batch adapter fits, exactly as the H3 job
rb_stage accuracy env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    python -m experiments.run_phase2e_noncontig --device cuda --fail-on-error "$@"
rc1=$?
rc2=0
if [ "${PHASE2E_SKIP_LATENCY:-0}" != "1" ]; then
    # timing runs WITHOUT the deterministic-cuBLAS setting, exactly as job_phase2_latency.sh
    rb_stage latency env -u CUBLAS_WORKSPACE_CONFIG \
        python -m experiments.run_phase2e_latency --repeats 3 --device cuda
    rc2=$?
fi
rb_main_end
RB_MAIN_RC=$(( rc1 != 0 || rc2 != 0 ))

if [ "${RB_POST:-0}" = 1 ]; then
    rb_stage tables python -m experiments.make_phase2e_tables
    if [ "$RB_MAIN_RC" = 0 ]; then
        rb_stage archive rb_archive fir_phase2e_noncontig results/three_model_phase2e_noncontig \
            "$SCRATCH/chronos2/phase2e_noncontig/adapters" three_models_paper/figures/noncontig
    fi
else
    echo "group job: tables / archive run once in fir/job_fir_aggregate.sh 2e"
fi
rb_finish
