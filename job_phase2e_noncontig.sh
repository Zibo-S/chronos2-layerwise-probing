#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:a100:1            # ONE FULL A100 (typed request: never a MIG slice, which the
                                     # latency stage would refuse and the fits do not need)
#SBATCH --cpus-per-task=12           # the Phase-2 latency geometry (api_e2e is host-side Python:
                                     # fewer cores = noisier timings); also numpy / bootstrap
#SBATCH --mem=64G                    # one dataset's K=4 states (<0.5 GB) + windows + loading peaks
#SBATCH --time=08:00:00              # estimate ~4 h (2.6 h fits + 1.4 h latency) -> 2x margin. Any
                                     # request in (3 h, 12 h] sits in the SAME scheduler tier (b2);
                                     # a shorter request backfills more easily. Max GPU wall is 7 d.
#SBATCH --signal=B:USR1@900          # 15 min before the limit -> automatic continuation
#SBATCH --output=results/three_model_phase2e_noncontig/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2e_noncontig/logs/%x-%j.err

# NOTE: no `set -e`. Accuracy and latency are independent; a failure in one must not skip the other
# (e.g. one dataset failing must not cancel the latency measurement). Each stage's exit code is
# reported at the end and the job exits non-zero if the tests, accuracy or latency failed.
set -uo pipefail

# =======================================================================================
# PHASE 2e -- non-contiguous block selection (Chronos-2, 14 datasets), ONE production job
#
#   stage tests     python -m tests.test_phase2e_noncontig      (model-free contracts)
#   stage plan      run_phase2e_noncontig --plan --check-inputs (JSON + stat only)
#   stage accuracy  run_phase2e_noncontig: model-level mask checks (synthetic contexts) BEFORE any
#                   dataset; then per dataset: dataset gates (the all-12-block backbone == the full
#                   model: states vs the Phase-1 L12 cache, physical == unmodified API bitwise,
#                   physical vs offline), 9 new NOA adapter fits (the H3 fit, unchanged), 3 prefix
#                   cells reusing H3 only after their reproduction gates, the single-block-skipping
#                   diagnostic, the paired comparison vs prefix. A failed gate stops that dataset /
#                   mask and is recorded; a physical != offline cell is written as invalid_v3.
#   stage latency   run_phase2e_latency: 12 masks x {hard, adapter} + natives, the Phase-2 timing
#                   protocol, 3 in-job repeats (fresh process per configuration).
#   stage tables    make_phase2e_tables: tables, figures, macros, report + rebuttal draft.
#   stage archive   results + fitted adapters ($SCRATCH) after clean accuracy AND latency stages.
#                   The skipped-block state cache (~5 GB, regenerable) is NOT archived.
#
# INDEPENDENT of Phase 2c (LODO) and 2d (pooled): different modules, output tree, scratch dirs and
# job name; their trees are refused as output roots. Reads READ-ONLY: Phase-1 windows, K=4 caches
# and cells; the committed H3/H4 cells; the H3 adapters.
#
# SUBMIT:  mkdir -p results/three_model_phase2e_noncontig/logs && sbatch -J p2e-noncontig job_phase2e_noncontig.sh
# FOLLOW:  every cell prints "[progress] ... ETA"; runs/<jobid>/progress.json has the same.
# RESUME:  resubmit the IDENTICAL line; COMPLETE cells and latency repeats are skipped. A TIMEOUT
#          resubmits itself (at most RB_MAX_CONT=2 times).
# Optional env: PHASE2E_SKIP_LATENCY=1 (accuracy + tables only).
# =======================================================================================

source ./job_rebuttal_common.sh       # explicit path: never resolved through $PATH
rb_setup "PHASE 2e / non-contiguous" results/three_model_phase2e_noncontig job_phase2e_noncontig.sh "$@"
echo "  H3 adapters (read-only) : $SCRATCH/chronos2_phase2/adapters"
echo "  2e adapters + states    : $SCRATCH/chronos2/phase2e_noncontig/"

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

rb_stage tables python -m experiments.make_phase2e_tables
if [ "$RB_MAIN_RC" = 0 ]; then
    rb_stage archive rb_archive phase2e_noncontig results/three_model_phase2e_noncontig \
        "$SCRATCH/chronos2/phase2e_noncontig/adapters" three_models_paper/figures/noncontig
else
    echo "accuracy rc=$rc1 latency rc=$rc2: archive NOT staged (results stay in place)"
fi
rb_finish
