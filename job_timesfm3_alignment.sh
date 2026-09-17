#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --cpus-per-task=4            # 140 economy SVDs of (1394, 1280) float64 + the grid scans
#SBATCH --mem=32G                    # dominated by build_windows loading the raw series
#SBATCH --time=3:00:00               # window building ~1-1.5 h; the linear algebra is ~15 min
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
# TimesFM-3 LINEAR REPRESENTATION ALIGNMENT, paper7 seven datasets.   *** NO GPU ***
#
#     h_{l,15}  --A_l-->  ~ h_{20,15}  --W_native (frozen)-->  y_hat
#
# The question: the learned probe shows the forecast is linearly DECODABLE from intermediate
# layers, while the frozen native head applied DIRECTLY to those layers is catastrophic (e.g.
# Electricity at its 5% tunnel entrance L15: probe 0.167 vs frozen head 85.79, native 0.145).
# Is the information simply in the wrong BASIS? Fit an affine map into the L20 representation
# and push the result through the pretrained head.
#
#   *** THE ADAPTER IS NEVER TRAINED ON FORECAST TARGETS ***
#   W_native is itself linear, so an adapter fit on forecast loss would collapse to another
#   linear forecasting map W_native A_l -- the learned probe in disguise. The objective here is
#         min_{A,b} ||X_l A + 1 b^T - H_20||_F^2 + lambda ||A||_F^2      (bias unpenalized)
#   and lambda is chosen on VALIDATION REPRESENTATION MSE. No future value, forecast, quantile
#   or loss enters the fit or the selection. This is enforced structurally: the fitting path
#   loads representations through load_last_token_reps, which verifies the cache's mu/sd/native
#   and then DISCARDS them.
#
#   NOT the Chronos-2 adapter (probing/native_head_adapter.py), which trains Linear(768,768) on
#   FORECAST loss into a NONLINEAR ResidualBlock head. Different objective, different question.
#
# WHY NO GPU: the only model contact is model.output_head, a single frozen Linear(1280, 576),
# applied to (N, 1280) matrices. Representations come from the validated float32 last-token
# cache the Q=9 probe run wrote. ZERO backbone forward passes -- decode() is never called.
#
# SOLVER: one economy SVD per layer in float64; A_lambda = V diag(s/(s^2+lambda)) U^T Yc makes
# the whole lambda grid a re-weighting of ONE decomposition. N=1394 train rows against d=1280
# (1.64M coefficients), so ridge is mandatory and the run reports the spectrum, the condition
# number and df(lambda) per layer, and WARNS when lambda lands on a grid edge.
#
# ENDPOINTS -- both reported, and they are NOT required to be bit-identical:
#   cached_L20_native_head   the frozen head on the CACHED L20 representation, the same
#                            (N, 1280) path every aligned curve takes. A_20 = I, b_20 = 0
#                            produces it by construction. This is the alignment endpoint.
#   official_native_decode   decode()'s own forecast, captured at extraction time. The true
#                            model baseline.
# Their ~1e-6 difference is the (N,1280)-vs-(b,1,18,1280) accumulation effect the native-head
# run measured with its own slice-order control. Numerical provenance, not an adapter result.
#
# CONTROLS: L20 identity (must be EXACT), mean baseline, permuted correspondence, the
# no-forecast-target shape guard, and the head's parameter checksum before/after.
#
# OUTPUTS are split by weight (the project rule):
#   $SCRATCH  adapters/<dataset>/<alignment>/L{00..19}.npz (float32, ~6.6 MB each),
#             numerical_results/representation_alignment__<tag>.json
#   repo      results/timesfm3_representation_alignment/{representation_alignment_summary.json,
#             representation_alignment_table.csv, figures/{forecast,representation,recovery}/}
#
# Usage:
#   sbatch job_timesfm3_alignment.sh                                   # ridge, all seven
#   sbatch job_timesfm3_alignment.sh --alignment ridge procrustes      # + orthogonal control
#   sbatch job_timesfm3_alignment.sh --datasets monash_electricity_hourly --layers 0 10 19 20
#   extra args are forwarded to the driver.
# =======================================================================================

CACHE_DIR=${TFM3_LT_CACHE_DIR:-$SCRATCH/timesfm3_last_token/features_cache}
WORK=${TFM3_ALIGN_OUT_ROOT:-$SCRATCH/timesfm3_alignment}
REF=${TFM3_ALIGN_REFERENCE:-$(git rev-parse --show-toplevel)/results/timesfm3_representation_geometry/native_head_transfer_summary.json}
CKPT=${TIMESFM3_CHECKPOINT:-google/timesfm-3.0-pytorch}

echo "cache         : $CACHE_DIR"
echo "reference     : $REF"
echo "work (SCRATCH): $WORK"
echo "repo outputs  : $(git rev-parse --show-toplevel)/results/timesfm3_representation_alignment"
echo "checkpoint    : $CKPT   (output_head only -- no backbone forward pass)"
echo "threads       : OMP_NUM_THREADS=$OMP_NUM_THREADS"

test -d "$CACHE_DIR" || { echo "MISSING feature cache $CACHE_DIR -- run job_timesfm3_last_token_q9.sh first"; exit 1; }
test -f "$REF"       || echo "[warn] no reference results at $REF: the run will save the aligned curve alone, without the direct-head / probe / tunnel overlays"

python -m experiments.run_timesfm3_representation_alignment \
    --cache-dir "$CACHE_DIR" \
    --out-root "$WORK" \
    --reference-results "$REF" \
    --checkpoint "$CKPT" \
    --device cpu \
    "$@"

echo
echo "done. Paper outputs are inside the repo:"
echo "  $(git rev-parse --show-toplevel)/results/timesfm3_representation_alignment/"
echo "Heavy adapter matrices stayed on \$SCRATCH:"
echo "  $WORK/adapters/"
