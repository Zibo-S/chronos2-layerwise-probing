#!/bin/bash
# Contract test for job_rebuttal_common.sh (the shared machinery of the three rebuttal jobs).
# No SLURM, no GPU, no Python package: `sbatch`, `module`, `pip` and `scontrol` are mocked.
# Needs GNU tar + bash >= 4.4 (Narval login node):   bash tests/test_job_rebuttal_common.sh
#
#   C1  a stage survives the USR1 timeout signal and returns the INNER command's exit code
#   C2  USR1 during the GPU work submits ONE continuation: same script + args, afterany:<job>,
#       RB_CONT+1, SLURM_* variables stripped from the nested sbatch call
#   C3  USR1 outside the GPU work (tables / archive) submits nothing
#   C4  USR1 with the continuation budget used up submits nothing
#   C5  rb_archive: one tar per source, contents lists, MANIFEST, SHA256SUMS that verify
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd -P)
T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT
fail() { echo "FAIL: $*"; exit 1; }
cd "$T" || exit 1
mkdir -p .venv/bin bin scratch results/x/cells
: > .venv/bin/activate
cat > bin/sbatch <<'EOF'
#!/bin/bash
{ echo "ARGS: $*"; env | grep -c '^SLURM_' | sed 's/^/SLURM_VARS: /'; } >> "$MOCK_SB_LOG"
echo 4242
EOF
cat > bin/scontrol <<'EOF'
#!/bin/bash
echo "JobId=$3"
EOF
chmod +x bin/sbatch bin/scontrol
export PATH="$T/bin:$PATH" MOCK_SB_LOG="$T/sbatch.log" SCRATCH="$T/scratch"
export SLURM_JOB_ID=1001 SLURM_JOB_NAME=testjob SLURM_CPUS_PER_TASK=2
module() { :; }
pip() { echo "mock-pip==1"; }
# shellcheck disable=SC1091
source "$REPO/job_rebuttal_common.sh"
rb_setup "TEST" results/x job_x.sh --alpha "two words" >/dev/null

# C1 + C2
rb_main_begin
( sleep 1; kill -USR1 $$ ) &
rb_stage main bash -c 'sleep 3; exit 7' >/dev/null; rc=$?
rb_main_end
[ "$rc" = 7 ] || fail "C1 stage rc=$rc, expected the inner 7"
[ "$(grep -c '^ARGS:' "$MOCK_SB_LOG")" = 1 ] || fail "C2 expected one sbatch call"
grep -q -- "--dependency=afterany:1001" "$MOCK_SB_LOG" || fail "C2 dependency"
grep -q -- "RB_CONT=1" "$MOCK_SB_LOG" || fail "C2 RB_CONT"
grep -q -- "--job-name=testjob" "$MOCK_SB_LOG" || fail "C2 job name"
grep -q -- "job_x.sh --alpha two words" "$MOCK_SB_LOG" || fail "C2 script + args"
grep -q "^SLURM_VARS: 0" "$MOCK_SB_LOG" || fail "C2 SLURM_* leaked into the nested sbatch"
[ "$(cat results/x/slurm/1001/continuation_job_id.txt)" = 4242 ] || fail "C2 record"
echo " C1  a stage survives USR1 and returns the inner exit code                     OK"
echo " C2  USR1 in the GPU work -> one continuation (script, args, afterany, RB_CONT) OK"

# C3
: > "$MOCK_SB_LOG"
( sleep 1; kill -USR1 $$ ) &
rb_stage tables bash -c 'sleep 2' >/dev/null
[ ! -s "$MOCK_SB_LOG" ] || fail "C3 resubmitted outside the GPU work"
echo " C3  USR1 during tables / archive submits nothing                            OK"

# C4
RB_CONT=$RB_MAX_CONT
rb_main_begin
( sleep 1; kill -USR1 $$ ) &
rb_stage main2 bash -c 'sleep 2' >/dev/null
rb_main_end
[ ! -s "$MOCK_SB_LOG" ] || fail "C4 resubmitted beyond the budget"
echo " C4  USR1 with the continuation budget used up submits nothing               OK"

# C5
echo hello > results/x/cells/a.txt
mkdir -p "$SCRATCH/adapters" && echo w > "$SCRATCH/adapters/w.npz"
rb_archive testphase results/x "$SCRATCH/adapters" "$SCRATCH/missing" >/dev/null || fail "C5 rc"
A="$SCRATCH/archive/rebuttal_testphase_1001"
[ -f "$A/01_x.tar" ] && [ -f "$A/02_adapters.tar" ] || fail "C5 tars"
grep -q "cells/a.txt" "$A/01_x.contents.txt" || fail "C5 contents"
grep -q "^results_dir=" "$A/MANIFEST.txt" || fail "C5 manifest"
(cd "$A" && sha256sum -c SHA256SUMS >/dev/null) || fail "C5 checksums"
echo " C5  archive: tars, contents, manifest, SHA256SUMS verify                     OK"
echo "all 5 job_rebuttal_common contracts hold"
