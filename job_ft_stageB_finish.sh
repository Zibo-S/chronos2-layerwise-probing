#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): B3 native forecasting passes need it
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                    # BOOM/SG/Coastal windowing + probe refits + bootstrap
#SBATCH --time=1:30:00              # B3~=30-45m (GPU) + B4~=15-30m (CPU) + B5~=5m; tight, and RESUMABLE
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# FT-SPECIALIZATION Stage B — FINISH (B3 -> B4 -> B5) in ONE job. B0/B1/B2 already ran (extraction +
# fresh probes + tunnels + curves on disk); this completes the catastrophic-forgetting evaluation on the
# 3 BOOM backbones (stage0_pretrained / stage1_ft_early@300 / stage2_ft_late@1000) x 7 targets:
#   B3 --native      PRIMARY forgetting: each stage's OWN native head on identical target-test windows
#                    (stage0=pretrained singleton, FT stages=load_ft_pipeline), original-scale MASE+WQL. GPU.
#   B4 --transfer    SECONDARY, probe-OOD: frozen BOOM probe re-fit per (stage,seed) from BOOM train/val
#                    fslot caches, predict-only on the 6 non-BOOM targets. CPU / warm caches.
#   B5 --forgetting  paired series-cluster bootstrap ACROSS stages + forgetting heatmap/bars. CPU.
#
# The three modes are chained with && so B4/B5 only run once the prior step succeeds. Every mode is
# IDEMPOTENT: native cells cache per checkpoint hash, transfer skips finished (stage,seed) runs, and B5
# is pure stats over B3/B4 outputs. So if this job TIMES OUT, just resubmit -- it resumes where it left
# off (finished work is skipped). The GPU sits idle during B4/B5 (a few min) -- accepted for a single
# submit-and-wait job. Do NOT run these on the login node.
#
# Submit:  sbatch job_ft_stageB_finish.sh

set -o pipefail

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export OMP_NUM_THREADS=2                                # B4/B5 do CPU probe fits + bootstrap; cap threads
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export FT_CKPT_ROOT=$SCRATCH/chronos2/ft_specialization # BOOM FT checkpoints live here (load_stages verifies hashes)
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets    # PT-OOD target arrow shards (BOOM/SG/Coastal)

echo "================ B3 --native  (GPU: native forecasting MASE/WQL, 3 stages x 7 targets) ================"
python -m experiments.run_ft_specialization --native      || { echo "B3 FAILED"; exit 1; }

echo "================ B4 --transfer  (CPU/warm: frozen-BOOM probe transfer, 3 stages x 6 targets) =========="
python -m experiments.run_ft_specialization --transfer    || { echo "B4 FAILED"; exit 1; }

echo "================ B5 --forgetting  (CPU: paired bootstrap + forgetting figures/tables) ================="
python -m experiments.run_ft_specialization --forgetting  || { echo "B5 FAILED"; exit 1; }

echo "================ DONE — Stage B B3/B4/B5 complete. Outputs: ================"
find results/ft_specialization/stageB/native results/ft_specialization/stageB/transfer \
     results/ft_specialization/stageB/forgetting -maxdepth 2 -type f 2>/dev/null | sort || true
