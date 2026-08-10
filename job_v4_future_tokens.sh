#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): K-slot feature extraction + probe fits
#SBATCH --cpus-per-task=4            # spectral SVDs run threaded on CPU after the GPU stages
#SBATCH --mem=32G                    # rolling window build (BOOM/SG) + 4096x768 SVDs need headroom
#SBATCH --time=3:00:00               # cold K-slot caches this run; resubmit is cheap (see below)
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

set -e                               # stop at the first failing stage (later stages depend on it)

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1               # compute nodes are offline; model+datasets pre-cached
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets   # pre-staged OOD-target arrow shards

# v4 "future tokens" pipeline: the SHARED forecast-token readout (Chronos-2's native-head layout),
# with the extra post-final-LN readout point (L12+LN) and the sustained-plateau tunnel criterion.
# All outputs land under results/ext_v4_future_tokens/ (the ftok driver's OUT_ROOT); the content
# line in results/extended_v3_rolling/ is untouched.
#
# Both stages are resumable: fit_ptid skips run seeds already on disk, and the K-slot feature
# caches (the only expensive step) are written once and re-HIT on any resubmit — so if this job
# hits the time limit, just `sbatch` it again and it continues cheaply where it stopped.

echo "=== stage 1/2: PT-ID probes (3 seeds) + sustained tunnels + PT-OOD fresh probes + D/Delta ==="
# no flags = fit PT-ID K-slot probes (all 3 run seeds, cold caches extracted here) -> tunnels ->
# eval the 3 PT-OOD targets x 3 seeds -> aggregate (stats CSV/JSON + delta heatmap). qset default q9.
python -m experiments.run_ptood_probing_ftok

echo "=== stage 2/2: spectral geometry on the fslot K-slot caches (stacked slots, 14 points) ==="
# reads the K-slot caches stage 1 just wrote; --readout fslot stacks the K slots -> (N*K, 768) and
# adds the post-LN point. Heavy (200 subsamples x 14 layers x 7 datasets SVDs) -> compute node.
python -m experiments.run_spectral --readout fslot

echo "=== compute done. On the LOGIN node (fast, CPU) render the paper figures: ==="
echo "  python -m experiments.make_paper_figures --readout fslot"
