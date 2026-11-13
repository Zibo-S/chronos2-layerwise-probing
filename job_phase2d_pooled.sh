#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:a100:1            # ONE FULL A100 (typed: never a MIG slice); = the Phase-1 GPU
#SBATCH --cpus-per-task=4            # = the Phase-1 job (the environment is mirrored ON PURPOSE)
#SBATCH --mem=48G                    # all three backbones in one job; states < 0.5 GB per dataset
#SBATCH --time=08:00:00              # estimate ~3 h for 42 cells (Phase-1 probe fits 45-75 s per
                                     # model x dataset; 2d fits up to 3 arms + TimesFM-3 / TiRex
                                     # extraction) + gates. (3 h, 12 h] is one scheduler tier.
#SBATCH --signal=B:USR1@900          # 15 min before the limit -> automatic continuation
#SBATCH --output=results/three_model_phase2d_pooled/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2d_pooled/logs/%x-%j.err

set -uo pipefail

# =======================================================================================
# PHASE 2d -- content-mean-pooled vs architecture-aware layerwise readouts (3 models x 14)
# ONE PRODUCTION JOB for all 42 (model, dataset) cells.
#
# Per cell: the native arm is READ from the committed Phase-1 cell (never refit) and reproduced
# from its artifacts; the generic arms are fit on the SAME windows, the SAME forward pass's states,
# the SAME targets, with the Phase-1 cell's own lr / epochs / wd grid:
#     pooled       mean of valid content tokens     -> Linear(d, 64*9)       all three models
#     concat       native readout states, concat'd  -> Linear(K*d, 64*9)     Chronos-2, TiRex
#     native_mean  native readout states, averaged  -> Linear(d, 64*9)       Chronos-2, TiRex
#
#   stage tests     python -m tests.test_phase2d_pooled        (model-free contracts)
#   stage plan      run_phase2d_pooled --plan                  (all 42 Phase-1 sources COMPLETE?)
#   stage main      run_phase2d_pooled with the two PRODUCTION GATES before the workload:
#                   --one-batch-checks chronos2 timesfm3 tirex  real-model extraction == the Phase-1
#                       caches on one Electricity batch (token ranges, content mean, token 15, TiRex
#                       readouts); ANY failure -> no cell runs at all
#                   --refit-native-preflight  per model, BEFORE its cells: the Phase-1 fitter in this
#                       job's environment must reproduce the committed native Electricity curve
#                       (1e-4) and every entrance; FAIL -> that model's cells are not run
#                   then 42 cells; each re-checks window digest, valid tokens, state identity, the
#                   native reproduction from artifacts, before it is written.
#   stage tables    make_phase2d_tables + make_phase2d_report (artifacts only)
#   stage archive   results + readout checkpoints/predictions ($SCRATCH) after a clean main stage
#
# ENVIRONMENT mirrors the Phase-1 job exactly (no CUBLAS_WORKSPACE_CONFIG, same batch sizes, 4
# CPUs), so the pass that produces the pooled states reproduces the Phase-1 cached readout states.
# INDEPENDENT of Phase 2c / 2e; the driver refuses any --output-root inside another phase's tree.
#
# SUBMIT:  mkdir -p results/three_model_phase2d_pooled/logs && sbatch -J p2d-pooled job_phase2d_pooled.sh
# RESUME:  resubmit the identical line (COMPLETE cells skipped). A TIMEOUT resubmits itself.
# =======================================================================================

source job_rebuttal_common.sh
rb_setup "PHASE 2d / pooled readouts" results/three_model_phase2d_pooled job_phase2d_pooled.sh "$@"
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}
echo "  phase-1 caches (read-only)   : $SCRATCH/chronos2/phase1_shared_cache, features_cache/"
echo "  pooled cache (written)       : $SCRATCH/chronos2/phase2d_pooled_cache"
echo "  readout checkpoints (written): $SCRATCH/chronos2/phase2d_pooled_artifacts"

SMOKE=0
case " $* " in *" --smoke "*) SMOKE=1 ;; esac

rb_stage tests python -m tests.test_phase2d_pooled || {
    RB_MAIN_RC=2; echo "model-free contracts FAILED on $(hostname): nothing else runs"; rb_finish; exit $?; }
rb_stage plan python -m experiments.run_phase2d_pooled --plan "$@" || {
    RB_MAIN_RC=2; echo "plan FAILED (a Phase-1 source is not ready): nothing else runs"; rb_finish; exit $?; }

GATES=(--one-batch-checks chronos2 timesfm3 tirex --refit-native-preflight)
[ "$SMOKE" = 1 ] && GATES=()                    # --smoke sets its own checks
rb_main_begin
rb_stage main python -m experiments.run_phase2d_pooled --device cuda "${GATES[@]}" "$@"
RB_MAIN_RC=$?
rb_main_end

if [ "$SMOKE" = 0 ]; then
    rb_stage tables python -m experiments.make_phase2d_tables
    rb_stage report python -m experiments.make_phase2d_report
    if [ "$RB_MAIN_RC" = 0 ]; then
        rb_stage archive rb_archive phase2d_pooled results/three_model_phase2d_pooled \
            "$SCRATCH/chronos2/phase2d_pooled_artifacts" three_models_paper/figures/pooled
    else
        echo "main stage rc=$RB_MAIN_RC: archive NOT staged (results stay in place; nothing deleted)"
    fi
fi
rb_finish
