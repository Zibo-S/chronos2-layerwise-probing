#!/bin/bash
#SBATCH --account=def-irina          # Fir maps it to def-irina_gpu for a GPU request
#SBATCH --gpus-per-node=h100:1       # fir/submit_fir.sh requests the chunks' GPU shape (3g.40gb)
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=results/fir_smoke/logs/%x-%j.out
#SBATCH --error=results/fir_smoke/logs/%x-%j.err

set -uo pipefail

# =======================================================================================
# FIR SMOKE: is this Fir setup (venv, copied inputs, H100) able to run Phases 2c / 2d / 2e / 2f?
# Writes ONLY to smoke trees (never to a production results tree or adapter directory).
#
#   A. (CPU, before this job: fir/job_fir_validate_cpu.sh) model-free contracts of all four
#      phases + the job machinery, every driver's --plan / --check-inputs, the input sizes
#   B. every driver's --plan (and --check-inputs) again, on the GPU node
#   C. GPU, real checkpoints, the PRODUCTION code paths (short: the chunks wait for this job):
#      C1 2f N = 1 fits for M5 at L6 (3 cells): the Phase 2c preflight gates on the target (full
#         model vs Phase 1 + H3, hard cut + dataset-specific adapter vs H3, adapter sha256), then
#         three real adapter fits, each of which must reproduce the committed (A100) H3 fit of its
#         single source (same wd and epoch, val 1e-4, hard-cut val 1e-6).
#      C2 2d one-batch identity checks, all three backbones (real extraction vs the A100 Phase-1
#         caches, tolerance 1e-3 of the layer std).
# EXIT 0 iff B passes and C1 exits 0, or 4 (a fit-reproduction mismatch: 2f-specific, the 2f chunks
# stop on that gate themselves); a preflight failure (3) or any other error blocks every chunk.
# C2 is reported only (each 2d job gates its own model).
# =======================================================================================

export RB_CLUSTER=fir
source ./job_rebuttal_common.sh       # explicit path: never resolved through $PATH
rb_setup "FIR SMOKE" results/fir_smoke fir/job_fir_smoke.sh "$@"
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}
python - <<'EOF' > "$RB_REC/torch_cuda.txt" 2>&1
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "capability", torch.cuda.get_device_capability(0))
    x = torch.randn(4096, 4096, device="cuda")
    print("matmul ok", float((x @ x).abs().mean()))
EOF
cat "$RB_REC/torch_cuda.txt"

FAIL=0
rb_stage plan_2c      python -m experiments.run_phase2_lodo --plan                       || FAIL=1
rb_stage plan_2d      python -m experiments.run_phase2d_pooled --plan                    || FAIL=1
rb_stage plan_2e      python -m experiments.run_phase2e_noncontig --plan --check-inputs  || FAIL=1
rb_stage plan_2f      python -m experiments.run_phase2f_source_scaling --plan --check-inputs || FAIL=1

rb_stage gpu_2f_n1 env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    python -m experiments.run_phase2f_source_scaling --device cuda --fail-on-error \
        --targets m5 --n 1 \
        --output-root results/three_model_phase2f_source_scaling_smoke \
        --adapter-root "$SCRATCH/chronos2/phase2f_source_scaling_smoke/adapters"
RC_2F=$?
case "$RC_2F" in
    0) ;;
    4) echo "2f fit reproduction did NOT hold on this GPU (exit 4): reported; 2f chunks gate it" ;;
    *) echo "2f smoke failed (exit $RC_2F): blocking"; FAIL=1 ;;
esac

# 2d mirrors the Phase-1 environment: NO deterministic-cuBLAS setting
rb_stage gpu_2d_checks env -u CUBLAS_WORKSPACE_CONFIG \
    python -m experiments.run_phase2d_pooled --device cuda \
        --one-batch-checks chronos2 timesfm3 tirex --checks-only \
        --output-root results/three_model_phase2d_pooled_smoke
RC_2D=$?

{
    echo "smoke_required_passed=$([ "$FAIL" = 0 ] && echo 1 || echo 0)"
    echo "gpu_2f_n1_rc=$RC_2F"
    echo "gpu_2d_checks_rc=$RC_2D"
} | tee "$RB_REC/smoke_verdict.txt"
RB_MAIN_RC=$FAIL
rb_finish
