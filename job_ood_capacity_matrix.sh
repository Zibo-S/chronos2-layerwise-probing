#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100)
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G                    # forecast-slot caches are ~0.8 GB each
#SBATCH --time=2:30:00               # full matrix: 2 families x 3 sources of head-fitting + aggregate
#SBATCH --output=logs/%x-%j.out

# Compute-node runs need THREE env layers, not two — miss any and you get a silent hang or a
# ~hours CPU fallback instead of a clean error:
#   1) module load ... cuda/13.2  -> torch.cuda finds the A100 (CC torch links module-provided CUDA)
#   2) source .venv/bin/activate  -> the venv's packages (torch/matplotlib/...) are separate from modules
#   3) HF_HOME + HF_HUB_OFFLINE=1 -> load_dataset uses the pre-cached files (nodes have no internet)
module load gcc python/3.11 arrow/24.0.0 cuda/13.2
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1             # stream prints live so a timeout shows WHERE it was, not a frozen log

# Overnight full 3x3 x 2-family run. NO `set -e`: each command is independent, so a plotting
# hiccup in one aggregate never aborts later fits, and everything is checkpointed — a re-submit
# resumes (get_source_probe loads existing checkpoints instead of re-fitting).
#
# ELECTRICITY FIRST (both families) so the never-yet-run real-data path fails fast if it is going
# to fail at all, before spending compute on kdd/uber. The cross-family `--compare` figures and
# the per-family `--figure-only` re-aggregation are CPU/seconds — do those on the LOGIN NODE after
# inspecting, not here.
set -x
for SOURCE in monash_electricity_hourly monash_kdd_cup_2018 uber_tlc_hourly; do
  for FAMILY in content_mlp_head forecast_slot_native_head; do
    echo "=================== ${FAMILY}  <-  ${SOURCE} ==================="
    python -u -m experiments.run_ood_capacity --probe-family "${FAMILY}" --source-dataset "${SOURCE}"
  done
done
echo "ALL SOURCES DONE — inspect results, then run --figure-only per family and --compare on the login node."
