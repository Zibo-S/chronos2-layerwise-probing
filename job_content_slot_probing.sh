#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): cold content-slot extraction + probe fits
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                    # rolling window build + 4 x (1394+262+262) window forward pass
#SBATCH --time=3:00:00               # cold caches this run; a resubmit resumes cheaply (see below)
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

set -e

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1               # compute nodes are offline; model + datasets pre-cached

# Content-slot shared-head probing: the SAME Linear(768, Q*P) head the committed fslot line uses,
# reading the K=4 CONTENT patches immediately before the REG token instead of the K forecast slots.
# Completes the readout 2x2 (pooled/shared x content/forecast) that the committed comparison
# confounds. Writes ONLY under results/ext_v4_future_tokens/cslot/; the fslot artifacts are read
# for the head-to-head figure and never written.
#
# This stage is GPU-bound because the content-slot states are NOT in any existing cache: the
# committed K4_H64 caches store the content MEAN (extraction.py pool_content), which is not
# invertible. So this is a genuine forward pass per (dataset, split) into the new cslotL_K4_H64
# cache. It is resumable twice over: the caches are written once and re-HIT on resubmit, and
# fit_ptid skips any run seed already on disk. If the job times out, just sbatch it again.

echo "=== stage 1/2: extract content slots + fit 4 PT-ID sources x 3 seeds x 14 layers ==="
python -m experiments.run_content_slot_probing --fit-ptid "$@"

echo "=== stage 2/2: sustained-plateau tunnels from the mean validation curves (CPU) ==="
python -m experiments.run_content_slot_probing --tunnels-only "$@"

echo "=== compute done. On the LOGIN node (CPU, seconds) render the figures: ==="
echo "  python -m experiments.run_content_slot_probing --figures"
echo "=== DONE ==="
