#!/bin/bash
# =======================================================================================
# FIR submission of the rebuttal phases 2c / 2f / 2d / 2e as DISJOINT <= 3 h chunks.
# LOGIN node only (sbatch calls, no compute). Run from the repo root of the Fir clone.
#
#   bash fir/submit_fir.sh validate                   # CPU validation (tests, plans, input sizes)
#   bash fir/submit_fir.sh smoke                      # the GPU smoke job only
#   bash fir/submit_fir.sh full [--after <jobid>]     # all chunks + one aggregate job per phase
#   bash fir/submit_fir.sh full --phases 2c 2f        # a subset of the phases
#   bash fir/submit_fir.sh full --dry-run             # print the sbatch lines, submit nothing
#
# --after <jobid>: every chunk waits for that job to end with exit 0 (afterok), e.g. the smoke.
# Every submission is appended to results/fir_submissions.tsv (cluster, commit, job, args).
#
# GPU SHAPE. The job scripts' headers ask for one full H100. The chunks are FP32 full-batch fits
# and probe fits that need < 10 GB of GPU memory, so they are submitted on an H100 3g.40gb MIG slice
# (COMPUTE_GPU): 60 of the 132 SMs (~A100-class FP32 throughput), 40 GB = the memory of Narval's
# A100-40GB, and on 2026-11-14 the scheduler started 3g slices ~2 h sooner than full H100s. The
# 2e job that carries the latency stage keeps the FULL H100 (the latency stage refuses MIG). A
# timeout continuation resubmits the script with its header, i.e. on a full H100 (harmless).
#
# The chunks partition each phase's cells (no cell is in two chunks):
#   2c  48 cells: sibling_excluded x {L4, L6, L9} x 2 target halves (7 cells each) + the 6
#       sibling_included cells                                                 -> 7 jobs
#   2f  40 cells: one job per target (10 cells; N = 1 first, as on Narval)       -> 4 jobs
#   2d  42 cells: one job per backbone (14 datasets)                           -> 3 jobs
#   2e  14 datasets: 4 + 5 + 5; the latency stage runs in the first job only   -> 3 jobs
# =======================================================================================
set -euo pipefail

MODE=${1:?usage: submit_fir.sh validate|smoke|full [--after JOBID] [--phases ...] [--dry-run]}; shift
AFTER=""; DRY=0; PHASES=(2c 2f 2d 2e)
COMPUTE_GPU=${COMPUTE_GPU:-nvidia_h100_80gb_hbm3_3g.40gb:1}
FULL_GPU=h100:1
CG="--gpus-per-node=$COMPUTE_GPU"; FG="--gpus-per-node=$FULL_GPU"
while [ $# -gt 0 ]; do
    case "$1" in
        --after) AFTER=$2; shift 2 ;;
        --dry-run) DRY=1; shift ;;
        --phases) shift; PHASES=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do PHASES+=("$1"); shift; done ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
done

[ "${CC_CLUSTER:-}" = fir ] || { echo "refusing: CC_CLUSTER=${CC_CLUSTER:-unset}, this script is for Fir"; exit 2; }
[ -f job_rebuttal_common.sh ] && [ -d fir ] || { echo "run from the repo root"; exit 2; }
COMMIT=$(git rev-parse HEAD)
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "refusing: tracked files differ from $COMMIT (provenance). git status:"; git status --short; exit 2
fi
mkdir -p results/fir_logs results/fir_smoke/logs results/three_model_phase2_lodo/logs \
         results/three_model_phase2d_pooled/logs results/three_model_phase2e_noncontig/logs \
         results/three_model_phase2f_source_scaling/logs
LEDGER=results/fir_submissions.tsv
# This script may run inside an srun step (Fir login1 was unusable on 2026-11-14): never let the
# step's own SLURM_* / SBATCH_* variables leak into the jobs it submits.
CLEAN_ENV=()
for v in $(compgen -e | grep -E '^(SLURM|SBATCH)_' || true); do CLEAN_ENV+=(-u "$v"); done
sb() { env "${CLEAN_ENV[@]}" sbatch "$@"; }
[ -f "$LEDGER" ] || printf 'submitted\tcluster\tcommit\tjob_id\tjob_name\tphase\tscript\targs\n' > "$LEDGER"

submit() {           # <phase> <name> <extra sbatch opts (one string)> <script> [args...] -> $LAST_ID
    local phase=$1 name=$2 opts=$3 script=$4; shift 4
    local dep=() out
    [ -n "$AFTER" ] && dep=(--dependency="afterok:$AFTER")
    if [ "$DRY" = 1 ]; then
        echo "sbatch -J $name ${dep[*]} $opts $script $*"; LAST_ID="DRY$RANDOM"; return
    fi
    # shellcheck disable=SC2086
    out=$(sb --parsable -J "$name" "${dep[@]}" $opts "$script" "$@" 2>&1) || {
        echo "sbatch FAILED for $name: $out"; exit 1; }
    LAST_ID=$(echo "$out" | grep -E '^[0-9]+(;[A-Za-z0-9_-]+)?$' | tail -1 | cut -d';' -f1)
    [[ "$LAST_ID" =~ ^[0-9]+$ ]] || { echo "sbatch gave no job id for $name: $out"; exit 1; }
    printf '%s\tfir\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$COMMIT" "$LAST_ID" "$name" "$phase" \
        "$script" "$opts $*" >> "$LEDGER"
    echo "  submitted $name -> $LAST_ID"
}

aggregate() {        # <phase> <job ids...>
    local phase=$1; shift
    local dep out id
    dep=$(IFS=:; echo "$*")
    if [ "$DRY" = 1 ]; then
        echo "sbatch -J fir-agg-$phase --dependency=afterany:$dep fir/job_fir_aggregate.sh $phase $*"
        return
    fi
    out=$(sb --parsable -J "fir-agg-$phase" --dependency="afterany:$dep" \
          fir/job_fir_aggregate.sh "$phase" "$@" 2>&1) || { echo "sbatch FAILED for fir-agg-$phase: $out"; exit 1; }
    id=$(echo "$out" | grep -E '^[0-9]+(;[A-Za-z0-9_-]+)?$' | tail -1 | cut -d';' -f1)
    [[ "$id" =~ ^[0-9]+$ ]] || { echo "sbatch gave no job id for fir-agg-$phase: $out"; exit 1; }
    printf '%s\tfir\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$COMMIT" "$id" "fir-agg-$phase" "$phase" \
        fir/job_fir_aggregate.sh "$phase $*" >> "$LEDGER"
    echo "  submitted fir-agg-$phase -> $id (after $*)"
}

if [ "$MODE" = validate ]; then
    submit validate fir-validate-cpu "" fir/job_fir_validate_cpu.sh
    echo "cpu validation: $LAST_ID"
    exit 0
fi
if [ "$MODE" = smoke ]; then
    submit smoke fir-smoke "$CG" fir/job_fir_smoke.sh
    echo "smoke: $LAST_ID"
    exit 0
fi
[ "$MODE" = full ] || { echo "unknown mode $MODE"; exit 2; }

for ph in "${PHASES[@]}"; do
    ids=()
    case "$ph" in
    2c)
        for d in 4 6 9; do
            for c in 1 2; do
                ab=$([ $c = 1 ] && echo a || echo b)
                submit 2c "fir-p2c-L$d-$ab" "$CG" fir/job_fir_phase2_lodo.sh \
                    --variants sibling_excluded --depths "$d" --chunk "$c/2"; ids+=("$LAST_ID")
            done
        done
        submit 2c fir-p2c-sib "$CG" fir/job_fir_phase2_lodo.sh --variants sibling_included; ids+=("$LAST_ID") ;;
    2f)
        for t in LOOP_SEATTLE_5T monash_london_smart_meters boom_hourly m5; do
            submit 2f "fir-p2f-$t" "$CG" fir/job_fir_phase2f_source_scaling.sh --targets "$t"; ids+=("$LAST_ID")
        done ;;
    2d)
        for m in chronos2 timesfm3 tirex; do
            submit 2d "fir-p2d-$m" "$CG" fir/job_fir_phase2d_pooled.sh --models "$m"; ids+=("$LAST_ID")
        done ;;
    2e)
        submit 2e fir-p2e-A "$FG" fir/job_fir_phase2e_noncontig.sh \
            --datasets m4_hourly monash_electricity_hourly uber_tlc_hourly wind_farms_hourly; ids+=("$LAST_ID")
        submit 2e fir-p2e-B "$CG --export=ALL,PHASE2E_SKIP_LATENCY=1" fir/job_fir_phase2e_noncontig.sh \
            --datasets sg_carpark coastal_ts boom_hourly LOOP_SEATTLE_5T electricity_15min; ids+=("$LAST_ID")
        submit 2e fir-p2e-C "$CG --export=ALL,PHASE2E_SKIP_LATENCY=1" fir/job_fir_phase2e_noncontig.sh \
            --datasets SZ_TAXI_15T monash_london_smart_meters m5 wiki_daily_100k monash_traffic; ids+=("$LAST_ID") ;;
    *) echo "unknown phase $ph"; exit 2 ;;
    esac
    echo "$ph chunks: ${ids[*]}"
    aggregate "$ph" "${ids[@]}"
done
echo "ledger: $LEDGER"
