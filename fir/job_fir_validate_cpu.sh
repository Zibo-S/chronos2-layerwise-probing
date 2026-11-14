#!/bin/bash
#SBATCH --account=def-irina          # Fir maps it to def-irina_cpu (no GPU)
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --time=01:00:00
#SBATCH --output=results/fir_smoke/logs/%x-%j.out
#SBATCH --error=results/fir_smoke/logs/%x-%j.err

set -uo pipefail
# =======================================================================================
# FIR CPU VALIDATION (no GPU, no model): the venv, the copied inputs and the job machinery.
#   1. the model-free contracts of Phases 2c / 2d / 2e / 2f and of job_rebuttal_common.sh
#   2. every driver's --plan (2e / 2f also --check-inputs: adapter sha256 vs the H3 records)
#   3. the copied input files: sizes against the Narval listings (.fir_transfer_<group>.sizes)
# EXIT 0 iff everything passes.
# =======================================================================================
export RB_CLUSTER=fir
source ./job_rebuttal_common.sh
rb_setup "FIR CPU VALIDATION" results/fir_smoke fir/job_fir_validate_cpu.sh "$@"
FAIL=0
rb_stage tests_jobs   bash tests/test_job_rebuttal_common.sh          || FAIL=1
rb_stage tests_2c     python -m tests.test_phase2_lodo               || FAIL=1
rb_stage tests_2d     python -m tests.test_phase2d_pooled            || FAIL=1
rb_stage tests_2e     python -m tests.test_phase2e_noncontig         || FAIL=1
rb_stage tests_2f     python -m tests.test_phase2f_source_scaling    || FAIL=1
rb_stage plan_2c      python -m experiments.run_phase2_lodo --plan                          || FAIL=1
rb_stage plan_2d      python -m experiments.run_phase2d_pooled --plan                       || FAIL=1
rb_stage plan_2e      python -m experiments.run_phase2e_noncontig --plan --check-inputs     || FAIL=1
rb_stage plan_2f      python -m experiments.run_phase2f_source_scaling --plan --check-inputs || FAIL=1
for g in core p2d; do
    L=/scratch/$USER/.fir_transfer_$g.sizes
    if [ -f "$L" ]; then
        (cd /scratch/$USER && cut -d' ' -f2- "$L" | xargs -d '\n' stat -c '%s %n') > "$RB_REC/sizes_$g.fir"
        if diff -q <(sort "$L") <(sort "$RB_REC/sizes_$g.fir") >/dev/null; then
            echo "inputs $g: $(wc -l < "$L") files, sizes identical to Narval"
        else
            echo "inputs $g: SIZE MISMATCH"; diff <(sort "$L") <(sort "$RB_REC/sizes_$g.fir") | head; FAIL=1
        fi
    else
        echo "inputs $g: no Narval size listing at $L (group not transferred yet)"
    fi
done
echo "cpu_validation_passed=$([ "$FAIL" = 0 ] && echo 1 || echo 0)" | tee "$RB_REC/verdict.txt"
RB_MAIN_RC=$FAIL
rb_finish
