#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): cls FT + feature extraction
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                    # dataset load + forecasting-target windowing (16G schedules faster than 32G)
#SBATCH --time=1:00:00              # covers C1 (FT) or C2 (extraction dominates); override per stage if wanted
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# TASK-SHIFT experiment — driver env. Classification fine-tuning of Chronos-2 on a UEA/UCR source
# (forda | uwave | handwriting), then layerwise CLASSIFICATION probes (Exp A) and FORECASTING probes
# (Exp B) across stage0/stage1/stage2. Everything is namespaced disjoint per source and from the BOOM
# domain-shift run (source=<src>_cls, results/task_shift_classification/<src>/).
#
# --finetune (anywhere in the args) routes to probing.finetune_cls; otherwise to run_task_shift.
# Compute stages (submit ONE at a time; C0 is login-node OK after `module load arrow`):
#   python -m probing.cls_data --smoke --cls-source uwave                 # C0: data smoke (LOGIN node OK, seconds)
#   sbatch job_task_shift.sh --cls-source uwave --finetune               # C1: classification FT [GPU] -> validity gate
#   sbatch job_task_shift.sh --cls-source uwave --extract --forecast-extract   # C2: cls + fslot features [GPU]
#   sbatch job_task_shift.sh --cls-source uwave --probe                  # C3: cls probes (14 layers x seeds)
#   sbatch job_task_shift.sh --cls-source uwave --forecast-probe         # C4: fslot probes (warm caches)
#   python -m experiments.run_task_shift --cls-source uwave --figures --cka    # C5: per-source Plots A/B/C (LOGIN/CPU)
#   python -m experiments.run_task_shift --compare                       # C5: cross-source comparison (LOGIN/CPU)
#
# Do NOT run C1-C4 (FT / extraction / probe fits) on the login node.

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export FT_CKPT_ROOT=$SCRATCH/chronos2/ft_specialization # cls checkpoints -> $SCRATCH/.../<src>_cls/
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets    # PT-OOD forecasting targets (BOOM/Coastal)

# Route --finetune (order-independent) to the FT entry point; strip it before forwarding the rest.
mode="driver"; fwd=()
for a in "$@"; do
    if [ "$a" == "--finetune" ]; then mode="finetune"; else fwd+=("$a"); fi
done

if [ "$mode" == "finetune" ]; then
    python -m probing.finetune_cls "${fwd[@]}"          # C1: writes 2 checkpoints + manifest + validity gate
else
    python -m experiments.run_task_shift "${fwd[@]}"    # C2-C5 modes
fi
