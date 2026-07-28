#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100); probe fitting = real compute, off the login node
#SBATCH --cpus-per-task=2            # data loading / sklearn scalers need little CPU
#SBATCH --mem=16G
#SBATCH --time=1:00:00               # generous; with warm caches one source is ~1-3 min (no extraction)
#SBATCH --output=logs/%x-%j.out      # %x = job name (set with sbatch -J), %j = job id

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1              # compute nodes have no internet; model+datasets are pre-cached

# One source per job (there are only 3 unique source-training configs). Pass --source-dataset
# <tag> on the sbatch command line; it is forwarded here. Trains the frozen probe once on the
# source and evaluates it on all three target test splits (diagonal = in-dataset, off-diagonal
# = strict cross-dataset transfer). Defaults: --dataset-set extended_v1, --quantile-set q9.
python -m experiments.run_ood_transfer "$@"
