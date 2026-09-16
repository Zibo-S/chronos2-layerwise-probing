#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --cpus-per-task=4            # SVDs (effective rank) + the CKA matmuls
#SBATCH --mem=32G                    # dominated by build_windows loading the raw series
#SBATCH --time=2:00:00               # geometry maths is ~4x the single-estimator cost (2x2 grid)
                                     # but still minutes; window building dominates the wall time
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

set -euo pipefail

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1               # compute nodes are offline
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}   # PT-OOD arrow shards
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # never grab every core on a shared node
export PYTHONHASHSEED=0

# =======================================================================================
# TimesFM-3 REPRESENTATION GEOMETRY + FROZEN NATIVE-HEAD TRANSFER, paper7 seven datasets.
#
# Two analyses, both reading the last-context-token cache the Q=9 probe run already wrote.
# Neither fits anything; neither touches a Chronos-2 file, the probe experiment, the TimesFM
# backbone code or the 16-prefix shared-origin ablation.
#
#   A. representation geometry   run_timesfm3_representation_geometry.py    NO model at all
#        CKA            probing.cka.cka_matrix over the FULL 2x2 grid, all seven datasets,
#                       identical layer ordering and plotting style throughout:
#                         biased/test    HEADLINE -- exact parity with the committed Chronos-2
#                                        CKA (run_cka_analysis.py: biased estimator, test split)
#                         unbiased/test  REQUIRED COMPANION -- the finite-sample-bias-corrected
#                                        analysis, and the informative one for ABSOLUTE
#                                        similarity at this N and d
#                         biased/train   robustness (N=1394)
#                         unbiased/train robustness -- the only reading with real power for
#                                        Coastal T-S, whose test split has 48 windows
#        effective rank probing.spectral_metrics.spectral_metrics -- exp(-sum p log p) with
#                       p = s^2 / sum(s^2) on the example-centred matrix. Split: TRAIN,
#                       mirroring run_spectral.py's committed protocol. UNCHANGED.
#
#   B. frozen native-head transfer  run_timesfm3_native_head_transfer.py   loads the checkpoint
#        for its output_head ONLY -- with a warm cache there is NO backbone forward pass. The
#        pretrained Linear(1280, 576) is applied to h_{l,15} at every point Emb..L20 and scored
#        on the paper7 TEST windows. No probe, no optimizer, no weight decay, no adapter.
#        ABORTS unless L20 reproduces decode()'s own nine quantiles (< 1e-4 relative) AND its
#        Q=9 loss equals the native baseline (< 1e-6 relative), and unless the head's parameter
#        checksum is identical before and after.
#
#   representation  h_{l,15}, the LAST REAL context token -- the one the native head reads at
#                   C=512 / H=64 / P=32. (N, 1280) per point, 21 points, NO reshape: TimesFM has
#                   K=1 native readout token per window where Chronos-2 stacks K=4 slots.
#   windows         the committed Chronos-2 ones; both drivers ABORT on a parity mismatch.
#   tunnel          NOT redefined here. The 5% entrance is read from the probe run's summary
#                   purely as a figure overlay and a table index.
#
# WHY BOTH ESTIMATORS: the biased estimator has an O(1/n) UPWARD bias, and ONE readout token per
# window means n = the window count. Measured on INDEPENDENT Gaussian representations at d=1280,
# where the true CKA is 0, it returns ~0.83 at N=262, ~0.78 at N=354 and ~0.96 at Coastal T-S's
# N=48; Chronos-2's committed CKA sat at ~0.42 because K=4 slot stacking gave it 1048 rows at
# d=768. So absolute biased values are NOT readable as similarity here, and
#   *** absolute biased CKA must NOT be compared across Chronos-2 and TimesFM-3 ***
# as though the two were on one scale. Biased is kept for strict methodological parity;
# unbiased is the estimator to interpret. The run measures each estimator's own null floor per
# dataset (--cka-null-floor-reps, on by default), prints it on every heatmap, stores it in the
# summary, and draws figures/cka/cka_vs_null_floor.png comparing values against their floors.
#
# OUTPUTS are split by weight (the project rule):
#   $SCRATCH  matrices/<estimator>/<split>/cka__<tag>.{npy,csv} for all 4 x 7, full spectra,
#             per-dataset JSON
#   repo      results/timesfm3_representation_geometry/{summary.json, *.csv,
#             figures/cka/<estimator>/<split>/, figures/effective_rank/, figures/native_head/}
#             -- a few MB, versionable, ready for LaTeX next to the Chronos-2 panels.
#
# Usage:
#   sbatch job_timesfm3_geometry.sh                  # both analyses
#   sbatch job_timesfm3_geometry.sh --geometry-only
#   sbatch job_timesfm3_geometry.sh --head-only
#   extra args after those are forwarded to BOTH drivers.
# =======================================================================================

CACHE_DIR=${TFM3_LT_CACHE_DIR:-$SCRATCH/timesfm3_last_token/features_cache}
PROBE_SUMMARY=${TFM3_PROBE_SUMMARY:-$SCRATCH/timesfm3_last_token/results/timesfm3_last_token_paper7_q9/timesfm3_last_token_summary.json}
WORK=${TFM3_GEOM_OUT_ROOT:-$SCRATCH/timesfm3_geometry}
CKPT=${TIMESFM3_CHECKPOINT:-google/timesfm-3.0-pytorch}

RUN_GEOM=1; RUN_HEAD=1; EXTRA=()
for a in "$@"; do
  case "$a" in
    --geometry-only) RUN_HEAD=0 ;;
    --head-only)     RUN_GEOM=0 ;;
    *)               EXTRA+=("$a") ;;
  esac
done

echo "cache        : $CACHE_DIR"
echo "probe summary: $PROBE_SUMMARY"
echo "work ($SCRATCH): $WORK"
echo "repo outputs : $(git rev-parse --show-toplevel)/results/timesfm3_representation_geometry"
test -d "$CACHE_DIR" || { echo "MISSING feature cache $CACHE_DIR -- run job_timesfm3_last_token_q9.sh first"; exit 1; }
test -f "$PROBE_SUMMARY" || echo "[warn] no probe summary at $PROBE_SUMMARY: the geometry run will omit the tunnel overlay and the native-head run will save no alignment gap"

if [ "$RUN_GEOM" = 1 ]; then
  echo; echo "=== A. representation geometry (CKA on test, effective rank on train) ==="
  python -m experiments.run_timesfm3_representation_geometry \
      --cache-dir "$CACHE_DIR" \
      --probe-results "$PROBE_SUMMARY" \
      --out-root "$WORK" \
      "${EXTRA[@]}"
fi

if [ "$RUN_HEAD" = 1 ]; then
  echo; echo "=== B. frozen native-head transfer (test split, no fitting) ==="
  python -m experiments.run_timesfm3_native_head_transfer \
      --checkpoint "$CKPT" \
      --cache-dir "$CACHE_DIR" \
      --probe-results "$PROBE_SUMMARY" \
      --out-root "$WORK" \
      --device cpu \
      "${EXTRA[@]}"
fi

echo; echo "done. Final paper outputs are inside the repo:"
echo "  $(git rev-parse --show-toplevel)/results/timesfm3_representation_geometry/"
