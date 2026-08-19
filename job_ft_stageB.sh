#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): fslot extraction off the 3 backbones
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                    # BOOM/SG/Coastal series load + windowing (16G schedules faster than 32G)
#SBATCH --time=2:30:00               # B1 full extract (3 backbones x 7 targets x 3 splits); B0 is minutes
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# FT-SPECIALIZATION Stage B — driver env. Extracts shared-forecast-slot (fslot) features for the 3
# BOOM backbone stages (pretrained / ft_early@300 / ft_late@1000) x 7 eval targets. stage0 reuses the
# committed pretrained caches; the FT stages load the $SCRATCH BOOM checkpoints and extract into
# collision-proof IDF_<tag>__ft__boom__<stage>__<hash8> caches.
#
# Sub-stages (GPU needed only for --extract and --native; --transfer/--forgetting are CPU/warm-cache
# and can run on the login node — but keep multi-second probe fits off the login node, use this job):
#   sbatch job_ft_stageB.sh --extract --smoke               # B0: BOOM/test, all 3 stages (fast check)
#   sbatch job_ft_stageB.sh --extract                       # B1: full 3 x 7 x 3 extraction  [GPU]
#   sbatch job_ft_stageB.sh --probe --tunnels --figures     # B2: fresh probes + tunnels + curves
#   sbatch job_ft_stageB.sh --native                        # B3: native MASE/WQL, 3 x 7      [GPU cold]
#   sbatch job_ft_stageB.sh --transfer                      # B4: frozen-BOOM transfer (3 x 6)
#   sbatch job_ft_stageB.sh --forgetting                    # B5: paired-bootstrap stats + figures

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export FT_CKPT_ROOT=$SCRATCH/chronos2/ft_specialization # BOOM FT checkpoints live here
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets    # PT-OOD target arrow shards (BOOM/SG/Coastal)

python -m experiments.run_ft_specialization "$@"
