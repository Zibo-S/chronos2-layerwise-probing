#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): full fine-tuning + bf16/tf32 on sm80
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G                    # BOOM = 356 hourly query series loaded + windowing; be generous
#SBATCH --time=1:30:00               # Stage-A pilot: 1 FT run (300+1000 steps); FT is cheap
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# FT-SPECIALIZATION Stage-A pilot. One full fine-tuning run of frozen Chronos-2, checkpointing at 300
# (stage1_ft_early) and 1000 (stage2_ft_late) optimizer steps, OFFICIAL full-FT defaults (LR 1e-6,
# linear decay, warmup 0, adamw_torch_fused, grad clip 1.0, seed 0) EXCEPT batch 64. Training windows
# come from Chronos-2's OWN random-cut-point sampler over the COMPLETE, leakage-truncated source
# histories. DEFAULT SOURCE = boom (PT-OOD: the model has NOT seen it, so full-FT can actually
# specialize — the PT-ID electricity pilot showed official full-FT cannot reduce loss on in-pretraining
# data). Stage B (transfer) is NOT run here — STOP for review of whether BOOM actually specializes.

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export FT_CKPT_ROOT=$SCRATCH/chronos2/ft_specialization # ~478 MB safetensors x2 -> $SCRATCH (gitignored)
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets    # pre-staged PT-OOD source arrow shards (BOOM)

# Manifest + train/ft_val histories + parameter-drift land under results/ft_specialization/<source>/.
# Forwarded args override the default, e.g. `--source electricity` (PT-ID baseline), `--learning-rate
# 3e-6`, or `--split-only` to report the corpus counts.
python -m probing.finetune --source boom "$@"
