#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU: the whole point is to MEASURE on the deployment device
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                    # schedules much faster than 32G; one model at a time (~460MB fp32)
#SBATCH --time=0:45:00               # 7 depths x 3 batch sizes, model reloaded per depth
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# Latency / throughput / memory of a DEPTH-TRUNCATED Chronos-2 (results/ext_v5_native_head_adapter/latency/).
#
# Why this needs a GPU node and cannot be derived: every other layerwise number in this project comes
# from the FULL model plus forward hooks, so a truncated model has never actually been built. This job
# builds it (drops encoder blocks, splices the adapter in front of the final RMSNorm) and times it.
# Latency, p95, throughput and peak memory are measurements — there is nothing on disk to compute them
# from. Do NOT run this on the login node: it loads amazon/chronos-2 once per depth.
#
# Recipe:
#   sbatch -J lat job_latency.sh --verify                      # gate + full sweep (recommended first run)
#   sbatch -J lat job_latency.sh --depths 3 6 12 --batch-sizes 1 256 --reps 200
#
# --verify runs the gate that the whole paper rests on: a model truncated after block l must reproduce
# the states this project reads off block l with a hook. It loads the model twice per depth, so it adds
# a few minutes; run it at least once and keep the log.
#
# Afterwards (login node, CPU, seconds) the cost table that joins these depths to accuracy:
#   python -m experiments.run_compression_cost

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-2}        # cap BLAS threads to the allocation

python -m experiments.run_latency "$@"
