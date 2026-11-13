# shellcheck shell=bash
# =======================================================================================
# Shared helpers for the three rebuttal production jobs (Phase 2c / 2d / 2e). SOURCED, never run.
#
#   rb_setup <phase> <results-dir> <script>   modules + venv + caches off /home + provenance record
#   rb_stage <name> <cmd...>                  run one stage; survives the timeout signal; records rc
#   rb_main_begin / rb_main_end               mark the GPU work: only THERE a timeout resubmits
#   rb_archive <name> <src>...                tar + contents + SHA256SUMS on $SCRATCH/archive
#   rb_finish                                 stage summary; exit code
#
# TIMEOUT CONTINUATION. Each job script carries `#SBATCH --signal=B:USR1@900`: SLURM signals the
# batch shell (only the shell, never python) 15 min before the wall-time limit. If that happens while
# the GPU work is still running, the trap submits the SAME script with the SAME arguments as a new
# job that starts after this one ends (--dependency=afterany), at most RB_MAX_CONT times (default 2).
# Every driver resumes (COMPLETE cells are skipped; a killed cell leaves only a staging dir that the
# next run deletes), so the continuation picks up exactly where this job was killed. A job that
# ends for any other reason (gate failure, error) is NOT resubmitted: that needs a human.
#
# WHY CACHES ARE REDIRECTED. /home is Lustre lustre05, whose target OST000d was unreachable on
# 2026-11-12 (`lfs check osts`: "Cannot send after transport endpoint shutdown"). Nothing in these
# jobs may depend on a file under /home: the repo clone, the venv, HF / pip / matplotlib caches and
# every output live on /scratch.
# =======================================================================================

RB_MAX_CONT=${RB_MAX_CONT:-2}
RB_CONT=${RB_CONT:-0}
RB_IN_MAIN=0
RB_STAGES=()
RB_MAIN_RC=0

rb_log() { echo "[rb $(date +%H:%M:%S)] $*"; }

rb_setup() {          # <phase> <results-dir> <script-path-relative-to-repo> [args...]
    RB_PHASE=$1; RB_RESULTS=$2; RB_SCRIPT=$3; shift 3
    RB_ARGS=("$@")
    RB_JOB=${SLURM_JOB_ID:-local-$$}
    RB_REPO=$(pwd -P)
    mkdir -p "$RB_RESULTS/logs" "$RB_RESULTS/slurm/$RB_JOB"
    RB_REC=$RB_RESULTS/slurm/$RB_JOB

    module load gcc python/3.11 arrow/24.0.0
    # shellcheck disable=SC1091
    source .venv/bin/activate
    export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
    export HF_HUB_OFFLINE=1                       # compute nodes are offline; checkpoints cached
    export PYTHONHASHSEED=0
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
    export XDG_CACHE_HOME=$SCRATCH/.cache         # never ~/.cache (on /home)
    export PIP_CACHE_DIR=$SCRATCH/.cache/pip
    export MPLCONFIGDIR=$SCRATCH/.cache/matplotlib-$RB_JOB
    export PHASE1_WINDOWS=${PHASE1_WINDOWS:-$SCRATCH/chronos2/phase1_shared_windows}
    export MPLBACKEND=Agg
    mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR"

    {
        echo "phase=$RB_PHASE job=$RB_JOB continuation=$RB_CONT/$RB_MAX_CONT host=$(hostname)"
        echo "started=$(date -Is) repo=$RB_REPO script=$RB_SCRIPT args=${RB_ARGS[*]}"
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null) branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)"
        echo "git_dirty_files:"; git status --porcelain 2>/dev/null | grep -v '^??' | head -50
        echo "python=$(command -v python) $(python -V 2>&1)"
    } > "$RB_REC/job.txt"
    pip freeze > "$RB_REC/pip_freeze.txt" 2>&1
    module -t list > "$RB_REC/modules.txt" 2>&1
    env | grep -E '^(SLURM|CUDA|OMP|HF_|PYTHON|CUBLAS|TORCH|PHASE|OOD|RB_|MPL|XDG)' | sort > "$RB_REC/env.txt"
    [ -n "${SLURM_JOB_ID:-}" ] && scontrol show job "$SLURM_JOB_ID" > "$RB_REC/scontrol.txt" 2>&1
    nvidia-smi --query-gpu=name,uuid,memory.total,driver_version,mig.mode.current \
        --format=csv > "$RB_REC/nvidia_smi.csv" 2>&1 || true

    echo "=================================================================================="
    echo "$RB_PHASE  host=$(hostname)  job=$RB_JOB  continuation $RB_CONT/$RB_MAX_CONT  $(date -Is)"
    echo "  repo       : $RB_REPO   (git $(git rev-parse --short HEAD 2>/dev/null))"
    echo "  results    : $RB_RESULTS   provenance: $RB_REC"
    echo "  extra args : ${RB_ARGS[*]}"
    cat "$RB_REC/nvidia_smi.csv" 2>/dev/null
    echo "=================================================================================="
    trap rb_on_usr1 USR1
}

rb_on_usr1() {
    rb_log "USR1: the wall-time limit is ~15 min away"
    if [ "$RB_IN_MAIN" != 1 ]; then
        rb_log "  not inside the GPU work (tables / archive): no continuation submitted"
        return
    fi
    if [ "$RB_CONT" -ge "$RB_MAX_CONT" ]; then
        rb_log "  continuation budget used ($RB_CONT/$RB_MAX_CONT): NOT resubmitting. Resume by hand:"
        rb_log "    cd $RB_REPO && sbatch -J ${SLURM_JOB_NAME:-job} $RB_SCRIPT ${RB_ARGS[*]}"
        return
    fi
    local unset_args=() v nxt
    for v in $(compgen -e | grep -E '^(SLURM|SBATCH)_'); do unset_args+=(-u "$v"); done
    nxt=$(cd "$RB_REPO" && env "${unset_args[@]}" sbatch --parsable \
          --dependency="afterany:${SLURM_JOB_ID}" --job-name="${SLURM_JOB_NAME:-job}" \
          --export="ALL,RB_CONT=$((RB_CONT + 1)),RB_MAX_CONT=$RB_MAX_CONT" \
          "$RB_SCRIPT" "${RB_ARGS[@]}" 2>&1)
    rb_log "  continuation submitted: $nxt  (starts after $SLURM_JOB_ID ends; resumes the cells)"
    echo "$(date -Is) continuation of $SLURM_JOB_ID -> $nxt" >> "$RB_RESULTS/slurm/continuations.txt"
    echo "$nxt" > "$RB_REC/continuation_job_id.txt"
}

rb_main_begin() { RB_IN_MAIN=1; }
rb_main_end() { RB_IN_MAIN=0; }

rb_stage() {          # <name> <cmd...>: run in the background and wait (so the USR1 trap can run)
    local name=$1; shift
    local t0 rc pid
    t0=$(date +%s)
    echo; echo "######## STAGE $name -- $(date -Is) ########"
    "$@" &
    pid=$!
    while true; do
        wait "$pid"; rc=$?
        if [ "$rc" -gt 128 ] && kill -0 "$pid" 2>/dev/null; then continue; fi   # trap fired
        break
    done
    RB_STAGES+=("$name rc=$rc $(( $(date +%s) - t0 ))s")
    echo "######## STAGE $name finished rc=$rc in $(( $(date +%s) - t0 ))s ########"
    printf '%s\n' "${RB_STAGES[@]}" > "$RB_REC/stages.txt"
    return "$rc"
}

rb_archive() {        # <name> <path>... : absolute or repo-relative paths, missing ones skipped
    local name=$1; shift
    local stage=$SCRATCH/archive/rebuttal_${name}_${RB_JOB}
    local dest=/nearline/def-irina/$USER/chronos2_archive/rebuttal_${name}_${RB_JOB}
    mkdir -p "$stage"
    local p i=0
    for p in "$@"; do
        [ -e "$p" ] || { rb_log "archive: skip missing $p"; continue; }
        i=$((i + 1))
        local abs base parent tf trc
        abs=$(cd "$(dirname "$p")" && pwd -P)/$(basename "$p")
        base=$(basename "$abs"); parent=$(dirname "$abs")
        tf="$stage/$(printf '%02d' "$i")_${base}.tar"
        # rc 1 = "file changed as we read it" (this job's own open log): content is still valid
        tar --warning=no-file-changed -cf "$tf" -C "$parent" "$base"; trc=$?
        [ "$trc" -le 1 ] || { rb_log "archive: tar of $abs failed rc=$trc"; return 1; }
        tar -tf "$tf" > "${tf%.tar}.contents.txt"
        echo "$(basename "$tf") <- $abs" >> "$stage/SOURCES.txt"
    done
    {
        echo "archive=rebuttal_${name}_${RB_JOB} created=$(date -Is) host=$(hostname)"
        echo "git_commit=$(git rev-parse HEAD 2>/dev/null) job=$RB_JOB continuation=$RB_CONT"
        echo "results_dir=$RB_REPO/$RB_RESULTS"
        echo "nearline_destination=$dest"
        echo "status=STAGED_ON_SCRATCH_ONLY (not archived until tools/rebuttal_nearline_copy.sh verifies the nearline copy)"
    } > "$stage/MANIFEST.txt"
    (cd "$stage" && sha256sum ./*.tar > SHA256SUMS) || return 1
    rb_log "archive staged + checksummed: $stage"
    cat "$stage/SHA256SUMS"
    echo "$stage" > "$RB_REC/archive_stage.txt"
    echo "$(date -Is) $stage" >> "$RB_RESULTS/slurm/archives_staged.txt"
    if [ -d /nearline/def-irina ] && [ -w /nearline/def-irina ]; then
        rb_log "nearline IS visible from $(hostname): run tools/rebuttal_nearline_copy.sh $stage"
    else
        rb_log "nearline is NOT visible from compute nodes. On a LOGIN node run:"
        rb_log "    bash tools/rebuttal_nearline_copy.sh $stage"
    fi
}

rb_finish() {
    echo; echo "=================================================================================="
    echo "$RB_PHASE finished $(date -Is)  job=$RB_JOB  continuation $RB_CONT/$RB_MAX_CONT"
    printf '  %s\n' "${RB_STAGES[@]}"
    [ -s "$RB_REC/continuation_job_id.txt" ] && echo "  continuation job: $(cat "$RB_REC/continuation_job_id.txt")"
    echo "=================================================================================="
    return "$RB_MAIN_RC"
}
