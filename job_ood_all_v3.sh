#!/bin/bash
# One-shot: full extended_v3_rolling pipeline end-to-end in a single GPU allocation.
#   1) 4×4 rolling transfer   — train each source once, eval all four targets
#   2) 4×4 figures            — matrix + relative-gain heatmaps + tables
#   3) 4×4 baseline figure    — probe vs native / seasonal / last-value
#   4) unseen-target screen   — data + native-Chronos-2 gate (fails fast if a target is broken)
#   5) 4×3 pretraining-OOD    — the 4 ROLLING source probes -> BOOM / SG Carpark / Coastal T-S
# The 4×3 (step 5) reuses the frozen source checkpoints step 1 writes, hence one sequential job.
# Steps run under `set -e`: a failure stops the job but every earlier step's outputs are already
# saved. Submit:  sbatch job_ood_all_v3.sh
#SBATCH --account=def-irina          # only account on Narval
#SBATCH --gres=gpu:1                 # one A100
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G                    # BOOM windowing is the memory driver
#SBATCH --time=3:00:00               # ~1.5–2 h expected; headroom for cold caches
#SBATCH --job-name=ood_v3_all
#SBATCH --output=logs/%x-%j.out      # %x=job name, %j=job id

set -euo pipefail
module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1               # compute nodes are offline; model+datasets pre-cached
export OMP_NUM_THREADS=2
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets   # pre-staged OOD-target arrow shards

SET=extended_v3_rolling

echo "########## 1/5  4x4: train each source once, eval all four targets ##########"
for src in monash_electricity_hourly uber_tlc_hourly m4_hourly wind_farms_hourly; do
  echo "----- source: $src -----"
  python -m experiments.run_ood_transfer --dataset-set "$SET" --source-dataset "$src"
done

echo "########## 2/5  4x4 figures (matrix + relative-gain) ##########"
python -m experiments.run_ood_transfer --dataset-set "$SET" --figure-only

echo "########## 3/5  4x4 baseline-comparison figure ##########"
python -m experiments.run_ood_baselines --dataset-set "$SET"

echo "########## 4/5  unseen-target screen (native-Chronos-2 gate) ##########"
python -m experiments.run_ood_screen --dataset-set "$SET" \
    --ood-targets sg_carpark coastal_ts boom_hourly --native

echo "########## 5/5  4x3 pretraining-OOD transfer (ROLLING source probes) ##########"
python -m experiments.run_ood_pretrain_transfer --source-set "$SET"

echo "########## DONE — all outputs under results/$SET/ ##########"
