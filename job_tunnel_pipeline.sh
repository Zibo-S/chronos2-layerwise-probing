#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): PT-OOD feature extraction + probe fits
#SBATCH --cpus-per-task=4            # spectral SVDs run threaded on CPU after the GPU stages
#SBATCH --mem=32G                    # BOOM/SG rolling window build needs headroom
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

set -e                               # stop at the first failing stage (later stages depend on it)

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1               # compute nodes are offline; model+datasets pre-cached
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets   # pre-staged OOD-target arrow shards

# Full compute side of the PT-ID/PT-OOD tunnel + spectral-geometry pipeline, in dependency
# order. Every stage is resumable/idempotent (existing per-seed outputs and warm feature
# caches are skipped), so re-submitting after a timeout just continues where it stopped.

echo "=== stage 1/3: PT-ID probe runs for seeds 1-2 (warm caches; seed 0 = committed) ==="
python -m experiments.run_ptood_probing --fit-ptid-seeds

echo "=== stage 2/3: tunnels + PT-OOD fresh probes (3 targets x 3 run seeds) + D/Delta stats ==="
python -m experiments.run_ptood_probing      # also writes the 3-run tunnel records + aggregate

echo "=== stage 3/3: spectral geometry (effective rank), all datasets with warm caches ==="
python -m experiments.run_spectral

echo "=== compute done. On the LOGIN node run: ==="
echo "  python -m experiments.make_paper_figures"
