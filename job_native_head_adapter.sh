#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): model load + adapter fits through the frozen head
#SBATCH --cpus-per-task=4            # PT-OOD (SG/BOOM) window building is CPU-bound
#SBATCH --mem=16G                    # 16G schedules MUCH faster than 32G on Narval; one dataset in memory
#SBATCH --time=2:00:00              # covers --adapt over all 7 datasets; override tighter for --sanity
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# ext_v5 NATIVE-HEAD ADAPTER experiment (results/ext_v5_native_head_adapter/, disjoint from ext_v4).
# For each dataset x layer: native baseline / zero-shot native head / shared Linear(768,768) adapter +
# frozen native head. All forecast slots are already cached (K4_H64) -> NO re-extraction; the GPU is
# used only to load amazon/chronos-2 (for the frozen Quantile Head + final RMSNorm) and to fit adapters.
#
# Recipe (submit ONE stage at a time; --sanity and --adapt are GPU/compute-node, --figures is login/CPU):
#   sbatch -J nha_sanity --time=0:30:00 job_native_head_adapter.sh --sanity          # 1 dataset, 5 layers, GATES
#     -> inspect the log: native-reconstruction gate OK? zero-shot@L12+RMS==native? frozen-check 0 model params?
#   sbatch -J nha_adapt job_native_head_adapter.sh --adapt                           # all 7 datasets [GPU]
#   sbatch -J nha_adapt job_native_head_adapter.sh --adapt --datasets m4_hourly      # a subset [GPU]
#   python -m experiments.run_native_head_adapter --figures                          # aggregate + plots (LOGIN/CPU)
#
# Do NOT run --sanity / --adapt on the login node — they load Chronos-2 and fit adapters (compute work).
# Mem default is 16G (schedules far faster than 32G on Narval). If BOOM alone OOMs, run it separately with
# a per-command bump: sbatch --mem=32G -J nha_boom job_native_head_adapter.sh --adapt --datasets boom_hourly

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets    # PT-OOD targets (SG_Carpark / Coastal / BOOM)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}        # cap BLAS threads to the allocation

python -m experiments.run_native_head_adapter "$@"
