#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100 on Narval)
#SBATCH --cpus-per-task=2            # data loading / sklearn scalers need little CPU
#SBATCH --mem=16G
#SBATCH --time=1:00:00               # short walltime -> fast backfill; run itself is ~10-20 min
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job id

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1              # compute nodes have no internet; everything is pre-cached

python -m experiments.run_id_forecasting "$@"
