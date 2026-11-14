#!/bin/bash
#SBATCH --account=def-irina          # Fir maps it to def-irina_cpu (no GPU requested)
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/fir_logs/%x-%j.out
#SBATCH --error=results/fir_logs/%x-%j.err

set -uo pipefail

# =======================================================================================
# FIR: tables / report / archive for ONE phase, once, after all its chunk jobs (CPU only).
#
#   sbatch --dependency=afterany:<j1>:<j2>... fir/job_fir_aggregate.sh <2c|2d|2e|2f> <j1> <j2> ...
#
# 1. FOLLOW CONTINUATIONS. A chunk that hit its wall time resubmitted itself (job_rebuttal_common.sh
#    writes results/<tree>/slurm/<job>/continuation_job_id.txt). If any continuation has not
#    finished yet, this job resubmits itself to start after it, and exits.
# 2. The phase's own table / report scripts (artifacts only; a partial run is stamped INCOMPLETE).
# 3. The checksummed archive is staged only if EVERY final chunk job is COMPLETED with exit 0,
#    i.e. the same condition as the single Narval job ("clean main stage").
# =======================================================================================

PHASE=${1:?phase 2c|2d|2e|2f}; shift
CHUNKS=("$@")
case "$PHASE" in
    2c) RES=results/three_model_phase2_lodo
        TABLES=(python -m experiments.make_phase2_lodo_tables)
        REPORT=(python -m experiments.make_phase2_lodo_report)
        ARCH=(fir_phase2c_lodo results/three_model_phase2_lodo
              "$SCRATCH/chronos2_phase2/adapters_lodo" three_models_paper/figures/lodo) ;;
    2d) RES=results/three_model_phase2d_pooled
        TABLES=(python -m experiments.make_phase2d_tables)
        REPORT=(python -m experiments.make_phase2d_report)
        ARCH=(fir_phase2d_pooled results/three_model_phase2d_pooled
              "$SCRATCH/chronos2/phase2d_pooled_artifacts" three_models_paper/figures/pooled) ;;
    2e) RES=results/three_model_phase2e_noncontig
        TABLES=(python -m experiments.make_phase2e_tables)     # also writes report + draft
        REPORT=()
        ARCH=(fir_phase2e_noncontig results/three_model_phase2e_noncontig
              "$SCRATCH/chronos2/phase2e_noncontig/adapters" three_models_paper/figures/noncontig) ;;
    2f) RES=results/three_model_phase2f_source_scaling
        TABLES=(python -m experiments.make_phase2f_tables)
        REPORT=(python -m experiments.make_phase2f_report)
        ARCH=(fir_phase2f_source_scaling results/three_model_phase2f_source_scaling
              "$SCRATCH/chronos2/phase2f_source_scaling/adapters" three_models_paper/figures/source_scaling) ;;
    *) echo "unknown phase $PHASE"; exit 2 ;;
esac

# ---- 1. resolve every chunk to the last job of its continuation chain
FINAL=(); WAIT=()
for j in "${CHUNKS[@]}"; do
    cur=$j
    while [ -s "$RES/slurm/$cur/continuation_job_id.txt" ]; do
        # the file holds `sbatch --parsable` output (2>&1): keep only a "<id>[;cluster]" line
        nxt=$(grep -E '^[0-9]+(;[A-Za-z0-9_-]+)?$' "$RES/slurm/$cur/continuation_job_id.txt" \
              | tail -1 | cut -d';' -f1)
        [ -n "$nxt" ] || break
        cur=$nxt
    done
    FINAL+=("$cur")
    st=$(sacct -j "$cur" -X -n -P -o State 2>/dev/null | head -1)
    case "$st" in
        PENDING*|RUNNING*|REQUEUED*|SUSPENDED*|CONFIGURING*|COMPLETING*|RESIZING*) WAIT+=("$cur") ;;
    esac
done
if [ ${#WAIT[@]} -gt 0 ]; then
    dep=$(IFS=:; echo "${WAIT[*]}")
    nxt=$(sbatch --parsable --dependency="afterany:$dep" --job-name="${SLURM_JOB_NAME:-fir-agg-$PHASE}" \
          --output="results/fir_logs/%x-%j.out" --error="results/fir_logs/%x-%j.err" \
          fir/job_fir_aggregate.sh "$PHASE" "${FINAL[@]}" 2>&1)
    echo "continuations still running (${WAIT[*]}): aggregation resubmitted as $nxt"
    mkdir -p "$RES/slurm"
    echo "$(date -Is) aggregate ${SLURM_JOB_ID:-local} deferred -> $nxt (waits for ${WAIT[*]})" \
        >> "$RES/slurm/fir_aggregate.txt"
    exit 0
fi

export RB_CLUSTER=fir
source ./job_rebuttal_common.sh       # explicit path: never resolved through $PATH
rb_setup "AGGREGATE $PHASE [fir]" "$RES" fir/job_fir_aggregate.sh "$PHASE" "${CHUNKS[@]}"

# ---- 2. the clean-run condition, over the FINAL job of every chain
ALL_OK=1
{
    echo "phase=$PHASE chunks=${CHUNKS[*]} final=${FINAL[*]}"
    for j in "${FINAL[@]}"; do
        line=$(sacct -j "$j" -X -n -P -o JobID,JobName,State,ExitCode,Elapsed 2>/dev/null | head -1)
        echo "  $line"
        st=$(echo "$line" | cut -d'|' -f3); ec=$(echo "$line" | cut -d'|' -f4)
        [ "$st" = COMPLETED ] && [ "$ec" = "0:0" ] || ALL_OK=0
    done
    echo "all_chunks_completed_cleanly=$ALL_OK"
} | tee "$RB_REC/chunks.txt"
ALL_OK=$(grep -c "all_chunks_completed_cleanly=1" "$RB_REC/chunks.txt")

# ---- 3. tables / report (artifacts only), then the archive on a clean run
rb_stage tables "${TABLES[@]}"
[ ${#REPORT[@]} -gt 0 ] && rb_stage report "${REPORT[@]}"
if [ "$ALL_OK" = 1 ]; then
    rb_stage archive rb_archive "${ARCH[@]}"
else
    echo "not every chunk completed cleanly: archive NOT staged (results stay in place)"
fi
echo "$(date -Is) aggregate ${SLURM_JOB_ID:-local} done (clean=$ALL_OK, final=${FINAL[*]})" \
    >> "$RES/slurm/fir_aggregate.txt"
rb_finish
