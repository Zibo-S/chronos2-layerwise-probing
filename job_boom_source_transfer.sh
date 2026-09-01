#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # A100: only speeds the linear-probe AdamW fits (features are warm)
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                    # BOOM/SG/Coastal series load + rolling windowing (16G schedules faster than 32G)
#SBATCH --time=0:30:00               # one source fit (3 seeds) + 7 warm-cache predict passes
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# BOOM-AS-SOURCE frozen transfer — the missing 5th source of the q1 appendix combined figure.
# Fits ONE shared-forecast linear probe on BOOM's train split (pretrained backbone, wd on BOOM-val,
# 3 probe seeds), freezes it, and scores it on all 7 eval targets. No feature re-extraction: BOOM's
# pretrained fslot caches (IDF_boom_hourly__ood__{train,val,test}_rolling__clean__K4_H64.npz) and the
# 7 targets' test caches are already on disk. OOD_TARGET_ROOT is still needed because the rolling
# windows for BOOM / SG Carpark / Coastal are rebuilt from the raw $SCRATCH arrow shards.
#
#   sbatch -J boom_src job_boom_source_transfer.sh                       # q1 (default)
#   sbatch -J boom_src job_boom_source_transfer.sh --quantile-set q9     # extra args forwarded
# then regenerate the figure on the LOGIN node (CPU, post-hoc):
#   python -m experiments.make_id_paper_figures --figure transfer

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets    # PT-OOD target arrow shards (BOOM/SG/Coastal)

python -m experiments.run_boom_source_transfer "$@"   # driver defaults to --quantile-set q1
