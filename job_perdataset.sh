#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100 on Narval)
#SBATCH --cpus-per-task=2            # data loading / sklearn scalers need little CPU
#SBATCH --mem=16G                    # bump if a large UEA dataset (LSST/Handwriting) OOMs
#SBATCH --time=3:00:00               # UEA re-extraction is cold here (caches invalidated):
                                     # 8 datasets x {clean + 3 shifts} through the encoder
#SBATCH --output=logs/%x-%j.out      # %x = job name, %j = job id

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1              # model loads from the offline HF cache; UEA data comes from
                                     # aeon's cache (pre-fetch on the login node before submitting)

python -m experiments.run_perdataset
