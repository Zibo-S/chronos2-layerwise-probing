#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:a100:1            # ONE FULL A100 (typed: never a MIG slice); = the H3 / 2c hardware
#SBATCH --cpus-per-task=4            # data loading + the bootstrap (= the Phase 2c job)
#SBATCH --mem=48G                    # forecast-slot states of all 14 datasets at L6 + loading peaks
#SBATCH --time=08:00:00              # estimate ~3.5 h: 40 fits = 208 one-source fit units (4 targets x
                                     # (3 x (1+4+8) + 13)); the H3 NOA fit is 62 s per 1,394 windows and
                                     # compute-bound, so ~45-62 s per unit = 2.6-3.6 h, + ~20 min load,
                                     # gates, bootstrap. 8 h = >2x margin, in the (3 h, 12 h] tier, which
                                     # backfills far more easily than 24 h.
#SBATCH --signal=B:USR1@900          # 15 min before the limit -> automatic continuation
#SBATCH --output=results/three_model_phase2f_source_scaling/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2f_source_scaling/logs/%x-%j.err

# NOTE: no `set -e`: every stage's exit code is recorded and the job exits non-zero at the end
# if the tests or the main stage failed. See job_rebuttal_common.sh for the shared machinery.
set -uo pipefail

# =======================================================================================
# PHASE 2f -- source-dataset scaling of zero-shot adapter transfer (Chronos-2, L6), ONE job
#
# Per (target, N, order seed): the Phase 2c pooled label-free adapter fit, UNCHANGED, on the first
# N of a fixed order of the target's 13 other datasets (N = 1, 4, 8, nested; N = 13 = all, one cell
# shared by the three orders), then ONE read of the target's test split. Targets (frozen from
# metadata): Loop Seattle, London Smart Meters, BOOM, M5. 40 adapter fits.
#
#   stage tests     python -m tests.test_phase2f_source_scaling     (model-free contracts)
#   stage plan      run_phase2f_source_scaling --plan --check-inputs (JSON + checksums)
#   stage main      run_phase2f_source_scaling: loads 14 datasets, then the PREFLIGHT (Phase 2c
#                   reproduction gates per target at L6; failure -> exit 3, no fit), then the cells
#                   in order of N: every N = 1 fit must reproduce the committed H3 fit of its single
#                   source (failure -> exit 4, no further fit); every cell re-runs the 2c gates.
#   stage tables    make_phase2f_tables (artifacts only): CSV, figures, LaTeX, macros
#   stage report    make_phase2f_report: report.md + rebuttal_draft.md (INCOMPLETE if partial)
#   stage archive   tar + SHA256SUMS of results + adapters on $SCRATCH/archive (clean main only);
#                   the nearline copy is a LOGIN-node step: bash tools/rebuttal_nearline_copy.sh <stage>
#
# INDEPENDENT of Phase 2c / 2d / 2e: own module, output tree, adapter dir and job name; the driver
# refuses any root that overlaps theirs. Reads READ-ONLY: Phase-1 windows/caches/cells, the
# committed H3 cells and the H3 adapters.
#
# SUBMIT (repo root = the $SCRATCH clone; the log dir must exist before sbatch):
#   mkdir -p results/three_model_phase2f_source_scaling/logs && sbatch -J p2f-source-scaling job_phase2f_source_scaling.sh
# RESUME after a failure: fix the cause, resubmit the identical line (COMPLETE cells are skipped).
# A TIMEOUT resubmits itself (at most RB_MAX_CONT=2 times); a gate failure never does.
# =======================================================================================

source ./job_rebuttal_common.sh       # explicit path: never resolved through $PATH
rb_setup "PHASE 2f / source scaling" results/three_model_phase2f_source_scaling job_phase2f_source_scaling.sh "$@"
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # deterministic cuBLAS (full-batch fits), as in H3 / 2c
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

rb_stage tables python -m experiments.make_phase2f_tables
rb_stage report python -m experiments.make_phase2f_report
if [ "$RB_MAIN_RC" = 0 ]; then
    rb_stage archive rb_archive phase2f_source_scaling results/three_model_phase2f_source_scaling \
        "$SCRATCH/chronos2/phase2f_source_scaling/adapters" three_models_paper/figures/source_scaling
else
    echo "main stage rc=$RB_MAIN_RC: archive NOT staged (results stay in place; nothing deleted)"
fi
rb_finish
