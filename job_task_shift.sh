#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): cls FT + feature extraction
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G                    # FordA load + forecasting-target windowing
#SBATCH --time=0:35:00              # C1 FT is minutes; C2 extraction dominates
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# TASK-SHIFT experiment — driver env. Classification fine-tuning of Chronos-2 on FordA, then layerwise
# CLASSIFICATION probes (Exp A) and FORECASTING probes (Exp B) across stage0/stage1/stage2. Everything
# is namespaced disjoint from the BOOM domain-shift run (source=forda_cls, results/task_shift_classification/).
#
# Compute stages (submit ONE at a time; C0 is login-node OK after `module load arrow`):
#   python -m probing.cls_data --smoke                            # C0: data smoke (LOGIN node OK, seconds)
#   sbatch job_task_shift.sh --finetune                          # C1: classification FT  [GPU] -> validity gate
#   sbatch job_task_shift.sh --extract --forecast-extract        # C2: cls + fslot features [GPU]
#   sbatch job_task_shift.sh --probe                             # C3: cls probes (14 layers x seeds)
#   sbatch job_task_shift.sh --forecast-probe                    # C4: fslot probes (warm caches)
#   python -m experiments.run_task_shift --figures --cka         # C5: Plots A/B/C (+CKA)  LOGIN/CPU
#
# Do NOT run C1-C4 (FT / extraction / probe fits) on the login node.

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export FT_CKPT_ROOT=$SCRATCH/chronos2/ft_specialization # FordA-cls checkpoints -> $SCRATCH/.../forda_cls/
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets    # PT-OOD forecasting targets (BOOM/Coastal)

if [ "$1" == "--finetune" ]; then
    shift
    python -m probing.finetune_cls "$@"                 # C1: writes 2 checkpoints + manifest + validity gate
else
    python -m experiments.run_task_shift "$@"           # C2-C5 modes
fi
