#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): C2 extraction needs it (C3/C4 read caches)
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                    # FordA + BOOM/M4/Coastal windowing (16G schedules faster than 32G)
#SBATCH --time=1:30:00              # C2 extract dominates + C3/C4 probe fits; tight, and RESUMABLE
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# TASK-SHIFT — FINISH (C2 -> C3 -> C4) in ONE job, AFTER C1's validity gate PASSED. Fine-tuning already
# produced stage1_cls_early + stage2_cls_late (results/ft_specialization/forda_cls/manifest.json). This
# completes the layerwise probing across stage0/stage1/stage2:
#   C2 --extract --forecast-extract   Exp-A cls features (14-pt content, FordA) + Exp-B fslot features
#                                     (BOOM/M4/Coastal). stage0 reuses committed pretrained caches; the
#                                     FordA-cls FT stages extract fresh into __ft__forda_cls__ caches. GPU.
#   C3 --probe                        Exp-A linear Linear(768,2)+CE probes (3 stages x 3 seeds x 14 layers).
#   C4 --forecast-probe               Exp-B fslot forecasting probes (3 stages x 3 targets x 3 seeds).
#
# Chained with && (later steps run only if the prior one succeeds). Every mode is IDEMPOTENT: extraction
# skips cache HITs, probes skip finished (stage[,target],seed) runs. So a TIMEOUT/OOM just needs a
# resubmit -- it resumes. An FT-stage cache is NEVER extracted off the pretrained singleton (fail-loud).
# GPU idles during C3/C4 (a few min) -- accepted for a single submit-and-wait job. Do NOT run on the login node.
#
# Then C5 (figures) runs on the LOGIN node (CPU, seconds):
#   python -m experiments.run_task_shift --figures --cka
#
# Submit:  sbatch job_task_shift_finish.sh

set -o pipefail

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export OMP_NUM_THREADS=2
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export FT_CKPT_ROOT=$SCRATCH/chronos2/ft_specialization # FordA-cls checkpoints -> $SCRATCH/.../forda_cls/
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets    # PT-OOD forecasting targets (BOOM/Coastal)

echo "================ C2 --extract --forecast-extract  (GPU: cls + fslot features, 3 stages) ============"
python -m experiments.run_task_shift --extract --forecast-extract || { echo "C2 FAILED"; exit 1; }

echo "================ C3 --probe  (Exp-A cls probes: 3 stages x 3 seeds x 14 layers) ===================="
python -m experiments.run_task_shift --probe                      || { echo "C3 FAILED"; exit 1; }

echo "================ C4 --forecast-probe  (Exp-B fslot probes: 3 stages x 3 targets x 3 seeds) ========="
python -m experiments.run_task_shift --forecast-probe             || { echo "C4 FAILED"; exit 1; }

echo "================ DONE — C2/C3/C4 complete. Next: run C5 figures on the LOGIN node:"
echo "  python -m experiments.run_task_shift --figures --cka"
ls results/task_shift_classification/cls_probes/*.json results/task_shift_classification/forecast_probes/*.json 2>/dev/null | wc -l
