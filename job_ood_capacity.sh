#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100); the nonlinear heads are ~3-5M params — real compute
#SBATCH --cpus-per-task=2            # data loading / sklearn scalers need little CPU
#SBATCH --mem=32G                    # the K4_H64 forecast-slot caches are ~0.8 GB each on disk
#SBATCH --time=1:00:00               # generous; one source (both fits + aggregate) is a few min warm-cache
#SBATCH --output=logs/%x-%j.out      # %x = job name (set with sbatch -J), %j = job id

module load gcc python/3.11 arrow/24.0.0 cuda/13.2   # cuda/13.2 -> torch.cuda sees the A100
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1              # compute nodes have no internet; model+datasets are pre-cached
export PYTHONUNBUFFERED=1            # stream prints live so a timeout shows WHERE it was

# Higher-capacity forecasting probe (capacity control) — one (family, source) per job.
# Pass e.g.  --probe-family content_mlp_head --source-dataset monash_electricity_hourly
# on the sbatch line; it is forwarded here. Trains the head once on the source and scores it,
# frozen, on all three target test splits. Outputs land under
# results/<set>/ood_transfer/capacity/<family>/ and never touch the committed linear pilot.
python -u -m experiments.run_ood_capacity "$@"
