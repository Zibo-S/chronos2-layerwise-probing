#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): target feature extraction + native forecast
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G                    # BOOM = 356 hourly queries + windowing; be generous
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1               # compute nodes are offline; model+datasets pre-cached
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets   # pre-staged OOD-target arrow shards

# Pretraining-OOD probe transfer: 4 frozen extended_v2 source probes -> 3 OOD targets (4x3).
# Extracts each target's 650-window content features on the GPU (cached after), computes the
# native Chronos-2 gate, scores every frozen source checkpoint, then aggregates. Nothing is
# trained. Extra args forwarded (e.g. --target sg_carpark, or --figure-only for CPU aggregate).
python -m experiments.run_ood_pretrain_transfer "$@"
