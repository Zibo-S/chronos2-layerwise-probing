#!/bin/bash
#SBATCH --account=def-irina
#SBATCH --gres=gpu:1                 # a FULL A100 (the harness refuses MIG slices)
#SBATCH --cpus-per-task=12           # the api_e2e level is host-side Python: fewer cores = noisier
#SBATCH --mem=64G
#SBATCH --time=03:00:00              # <=3 h tier: run ONE model per job (below)
#SBATCH --output=results/three_model_phase2/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2/logs/%x-%j.err

set -euo pipefail

# =======================================================================================
# PHASE 2 / H4 -- latency / throughput / memory. MEASURED, one fresh process per configuration.
#
# THREE INDEPENDENT REPEATS, sized from the smoke's MEASURED per-call times (B = 1 and 256):
#   Chronos-2 ~20 min | TimesFM-3 ~1.4 h | TiRex ~2.3 h -> TiRex split into two depth halves
#   balanced by cost (a TiRex call scales with the blocks kept: ~41 ms per block at B=1).
#   for r in 1 2 3; do
#     sbatch --time=01:00:00 -J p2lat-c2-$r     job_phase2_latency.sh --models chronos2 --job-tag job${r}_chronos2
#     sbatch --time=02:30:00 -J p2lat-tfm3-$r   job_phase2_latency.sh --models timesfm3 --job-tag job${r}_timesfm3
#     sbatch --time=02:00:00 -J p2lat-tirA-$r   job_phase2_latency.sh --models tirex --depths 0 1 2 3 4 5 6 7 8 --job-tag job${r}_tirex_a
#     sbatch --time=02:00:00 -J p2lat-tirB-$r   job_phase2_latency.sh --models tirex --depths 9 10 11 12 --job-tag job${r}_tirex_b
#   done
#   (every job times its own native model at start / middle / end; speedups are formed within
#   the job, and each depth lives in exactly one job per repeat)
# Speedups are formed WITHIN each job (same node, the model's native timed at start/middle/end),
# then aggregated across a model's jobs by experiments.make_phase2_tables (median of per-job
# medians + range). 100 in-run repetitions are NOT independent runs -- the repeats are.
# SIZE --time FROM THE SMOKE: per configuration ~= load time + (120/7) x the smoke's timed part.
#
# SMOKE FIRST (all 3 models, 2 depths, 5 reps; allowed before verification; its own output root):
#   sbatch --time=01:00:00 -J p2lat-smoke job_phase2_latency.sh --depths 3 12 --reps 5 --warmup 2 \
#          --job-tag smoke --allow-unverified --output-root results/three_model_phase2_smoke
#
# The full grid refuses any configuration without a PASSED record from
#   job_phase2_h4.sh verify
# A job tag is ONE measurement: reusing one is refused. Only jobs run with the headline protocol
# (20 warmup / 100 reps, CUDA, verified, no MIG) enter the combined tables; index.json records why
# any other job was left out.
# =======================================================================================

mkdir -p results/three_model_phase2/logs
module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-12}
export PYTHONHASHSEED=0
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}

echo "PHASE 2 / latency  host=$(hostname)  job=${SLURM_JOB_ID:-none}  $(date -Is)  args: $*"
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv,noheader || true
python -m experiments.run_phase2_latency --device cuda "$@"
