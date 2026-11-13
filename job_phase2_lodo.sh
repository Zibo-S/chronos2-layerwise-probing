#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:a100:1            # ONE FULL A100 (typed: never a MIG slice); = the H3 hardware
#SBATCH --cpus-per-task=4            # data loading + the bootstrap
#SBATCH --mem=48G                    # forecast-slot states of all 14 datasets (~1.5 GB kept) + peaks
#SBATCH --time=24:00:00              # estimate ~11 h: 48 pooled fits x ~13 min (H3 NOA fit = 62 s on
                                     # 1394 windows, compute-bound; LODO pools 12-13x the rows) + load.
                                     # Full-A100 nodes are the SAME 118 for every tier up to 3 d.
#SBATCH --signal=B:USR1@900          # 15 min before the limit -> automatic continuation (see below)
#SBATCH --output=results/three_model_phase2_lodo/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2_lodo/logs/%x-%j.err

# NOTE: no `set -e`: every stage's exit code is recorded and the job exits non-zero at the end
# if the tests or the main stage failed. See job_rebuttal_common.sh for the shared machinery.
set -uo pipefail

# =======================================================================================
# PHASE 2c -- pooled leave-one-dataset-out (LODO) adapter transfer, Chronos-2, L4 / L6 / L9
# ONE PRODUCTION JOB for all 48 cells (42 main "sibling_excluded" + 6 "sibling_included").
#
# Per (variant, target, depth): the H3 label-free adapter fit, UNCHANGED, on the pooled sources
# (each source weighs 1/|S|), then ONE read of the target's test split.
#
#   stage tests     python -m tests.test_phase2_lodo           (model-free contracts, ~15 s)
#   stage plan      run_phase2_lodo --plan                     (JSON reads: 48 cells + status)
#   stage main      run_phase2_lodo: loads 14 datasets, then the PREFLIGHT -- the cell's own
#                   reproduction gates (full model vs Phase 1 + H3, hard cut and dataset-specific
#                   adapter vs H3, test rows, identity adapter, adapter sha256) for every pending
#                   (target, depth) BEFORE any fit; any failure exits 3 with runs/<job>/preflight.json
#                   and no fit. Then the 48 fits; every cell re-checks its gates before it is written.
#   stage tables    make_phase2_lodo_tables + make_phase2_lodo_report (artifacts only)
#   stage archive   tar + SHA256SUMS of results + pooled adapters on $SCRATCH/archive (only after a
#                   clean main stage); the nearline copy is a LOGIN-node step:
#                   bash tools/rebuttal_nearline_copy.sh <stage printed in the log>
#
# SUBMIT (repo root = the $SCRATCH clone; the log dir must exist before sbatch):
#   mkdir -p results/three_model_phase2_lodo/logs && sbatch -J p2c-lodo job_phase2_lodo.sh
# RESUME after a failure: fix the cause, resubmit the identical line (COMPLETE cells are skipped).
# A TIMEOUT resubmits itself (at most RB_MAX_CONT=2 times). `--smoke` still works (separate tree).
# =======================================================================================

source job_rebuttal_common.sh
rb_setup "PHASE 2c / LODO" results/three_model_phase2_lodo job_phase2_lodo.sh "$@"
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # deterministic cuBLAS (full-batch fits), as in H3
echo "  phase-1 windows (read-only) : $PHASE1_WINDOWS"
echo "  H3 adapters (read-only)     : $SCRATCH/chronos2_phase2/adapters"
echo "  pooled adapters (written)   : $SCRATCH/chronos2_phase2/adapters_lodo"

SMOKE=0
case " $* " in *" --smoke "*) SMOKE=1 ;; esac

rb_stage tests python -m tests.test_phase2_lodo || {
    RB_MAIN_RC=2; echo "model-free contracts FAILED on $(hostname): nothing else runs"; rb_finish; exit $?; }
rb_stage plan python -m experiments.run_phase2_lodo --plan "$@" || {
    RB_MAIN_RC=2; echo "plan FAILED (inputs not readable?): nothing else runs"; rb_finish; exit $?; }

rb_main_begin
rb_stage main python -m experiments.run_phase2_lodo --device cuda --fail-on-error "$@"
RB_MAIN_RC=$?
rb_main_end

if [ "$SMOKE" = 0 ]; then
    rb_stage tables python -m experiments.make_phase2_lodo_tables
    rb_stage report python -m experiments.make_phase2_lodo_report
    if [ "$RB_MAIN_RC" = 0 ]; then
        rb_stage archive rb_archive phase2c_lodo results/three_model_phase2_lodo \
            "$SCRATCH/chronos2_phase2/adapters_lodo" three_models_paper/figures/lodo
    else
        echo "main stage rc=$RB_MAIN_RC: archive NOT staged (results stay in place; nothing deleted)"
    fi
fi
rb_finish
