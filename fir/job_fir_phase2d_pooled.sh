#!/bin/bash
#SBATCH --account=def-irina          # Fir maps it to def-irina_gpu for a GPU request
#SBATCH --gpus-per-node=h100:1       # ONE FULL H100 80GB (never a MIG slice)
#SBATCH --cpus-per-task=4            # = the Narval / Phase-1 job (the environment is mirrored)
#SBATCH --mem=48G                    # = the Narval job
#SBATCH --time=03:00:00              # <= 3 h = Fir's shortest GPU tier; one job per model
#SBATCH --signal=B:USR1@900          # 15 min before the limit -> automatic continuation
#SBATCH --output=results/three_model_phase2d_pooled/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2d_pooled/logs/%x-%j.err

set -uo pipefail

# =======================================================================================
# FIR VARIANT of job_phase2d_pooled.sh (Phase 2d, pooled vs architecture-aware readouts).
#
# Stages and commands are those of job_phase2d_pooled.sh. Differences (cluster only):
#   * resources: one full H100; <= 3 h per job;
#   * ONE JOB PER MODEL (fir/submit_fir.sh passes --models <m>). The two production gates run
#     for that job's model, BEFORE any of its cells, exactly as on Narval:
#         --one-batch-checks <m>      real extraction == the Phase-1 cache on one batch
#         --refit-native-preflight    the Phase-1 fitter reproduces the committed native curve
#     (Narval's single job checked all three models up front; here a failed check for one
#     model stops that model's job only. Every cell still re-checks its own gates.);
#   * tables / report / archive run ONCE after all models, in fir/job_fir_aggregate.sh 2d.
# =======================================================================================

export RB_CLUSTER=fir
source ./job_rebuttal_common.sh       # explicit path: never resolved through $PATH
rb_setup "PHASE 2d / pooled readouts [fir]" results/three_model_phase2d_pooled fir/job_fir_phase2d_pooled.sh "$@"
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}
echo "  cluster                      : $RB_CLUSTER"
echo "  phase-1 caches (read-only)   : $SCRATCH/chronos2/phase1_shared_cache, features_cache/"
echo "  pooled cache (written)       : $SCRATCH/chronos2/phase2d_pooled_cache"
echo "  readout checkpoints (written): $SCRATCH/chronos2/phase2d_pooled_artifacts"

# the models this job runs (from --models), for the one-batch identity checks
CHECK_MODELS=()
grab=0
for a in "$@"; do
    case "$a" in
        --models) grab=1 ;;
        --*) grab=0 ;;
        *) [ "$grab" = 1 ] && CHECK_MODELS+=("$a") ;;
    esac
done
[ ${#CHECK_MODELS[@]} -eq 0 ] && CHECK_MODELS=(chronos2 timesfm3 tirex)
echo "  one-batch checks for        : ${CHECK_MODELS[*]}"

rb_stage tests python -m tests.test_phase2d_pooled || {
    RB_MAIN_RC=2; echo "model-free contracts FAILED on $(hostname): nothing else runs"; rb_finish; exit $?; }
rb_stage plan python -m experiments.run_phase2d_pooled --plan "$@" || {
    RB_MAIN_RC=2; echo "plan FAILED (a Phase-1 source is not ready): nothing else runs"; rb_finish; exit $?; }

GATES=(--one-batch-checks "${CHECK_MODELS[@]}" --refit-native-preflight)
rb_main_begin
rb_stage main python -m experiments.run_phase2d_pooled --device cuda "${GATES[@]}" "$@"
RB_MAIN_RC=$?
rb_main_end

if [ "${RB_POST:-0}" = 1 ]; then
    rb_stage tables python -m experiments.make_phase2d_tables
    rb_stage report python -m experiments.make_phase2d_report
    if [ "$RB_MAIN_RC" = 0 ]; then
        rb_stage archive rb_archive fir_phase2d_pooled results/three_model_phase2d_pooled \
            "$SCRATCH/chronos2/phase2d_pooled_artifacts" three_models_paper/figures/pooled
    fi
else
    echo "per-model job: tables / report / archive run once in fir/job_fir_aggregate.sh 2d"
fi
rb_finish
