#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # ONE full-context pass per batch + 21 probe fits
#SBATCH --cpus-per-task=4            # StandardScaler + the numpy target build + data loading
#SBATCH --mem=32G                    # dominated by loading the raw series in build_windows
#SBATCH --time=3:00:00               # 7 datasets: window building (PT-OOD arrow shards) dominates
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

set -euo pipefail

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1               # compute nodes are offline; model + datasets pre-cached
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}   # PT-OOD arrow shards
                                      # (SG Carpark / Coastal T-S / BOOM) -- the same path the
                                      # Chronos-2 jobs use; the loaders need it to build windows
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # never grab every core on a shared node
export PYTHONHASHSEED=0               # deterministic alongside --seed

# =======================================================================================
# HEADLINE TimesFM-3 experiment: LAST-CONTEXT-TOKEN layer-wise probing, native Q=9,
# on the SEVEN Chronos-2 paper datasets.
#
#   suite paper7   PT-ID  (in Chronos-2 pretraining): m4_hourly (M4), monash_electricity_hourly
#                         (Electricity), uber_tlc_hourly (Uber TLC), wind_farms_hourly (Wind Farms)
#                  PT-OOD (documented outside it)   : sg_carpark (SG Carpark), coastal_ts
#                         (Coastal T-S), boom_hourly (BOOM)
#   windows        EXACTLY the Chronos-2 ones -- PT-ID via build_windows under the
#                  "extended_v3_rolling" set (rolling origins, 1394/262/262, seed 0), PT-OOD via
#                  build_ood_rolling_windows(tag, C=512, H=64, seed=0). The run ABORTS unless the
#                  window counts AND the per-window test series/cluster ids match the committed
#                  results/ext_v5_native_head_adapter artifacts element-wise.
#   validation     the datasets' own dedicated rolling val split (never a carve): weight decay and
#                  the tunnel entrance are chosen on it, exactly as the Chronos-2 line does.
#   KDD Cup 2018 / Pedestrian Counts are NOT in the headline suite; they stay reachable with
#   `--suite extended_v1` for implementation checks only.
#
#   question   at each layer, how linearly decodable is the native C=512 -> H=64 forecast
#              from the EXACT token the native TimesFM-3 head reads?
#   geometry   ONE full-context decode() pass; 16 real context patches + 2 horizon
#              placeholders = 18 tokens; readout = token 15, the LAST REAL context patch
#              (== decode()'s own forecast index, asserted every batch)
#   probe      one INDEPENDENT Linear(1280, 64*9) per point in {Emb, L1..L20}
#   objective  mean pinball loss over all 64 steps x all 9 NATIVE quantiles (the full native
#              probabilistic forecasting objective). At L20 the probe and the native output
#              head are the same hypothesis class R^1280 -> R^(64x9).
#   tunnel     l_tun = min { l : L_val(l) <= 1.05 * L_val(L20) }, VALIDATION only
#              (1%/2%/5%/10% all saved, so a 2% criterion needs no re-extraction)
#
# NOT this job: the 16-prefix / shared-origin ABLATION (job_timesfm3_probing.sh,
# experiments/run_timesfm3_probing.py, cache tag tfm3-prefix-v1). Separate entry point,
# separate cache tag (tfm3-last-token-q9-fp32-v1) and separate results dir -- they cannot collide,
# and the loader REFUSES the other line's cache instead of reusing it. No Chronos-2 file is
# touched by either.
#
# Cost, from the implementation (7 datasets x 21 points x Q=9, C=512, H=64; 13.4k windows):
#   feature cache 21 layers x (n_train + n_val + n_test) x 1280 x float32 (lossless):
#                 ~206 MB per PT-ID dataset (1394+262+262), ~226 MB for SG Carpark / BOOM
#                 (1394+354+354), ~160 MB for Coastal T-S (1394+48+48) -> ~1.5 GB for the seven
#                 ->  $SCRATCH, NEVER $HOME (~50 GB quota). --feature-dtype float16 halves it.
#   GPU  < 4 GB   (1.3 GB frozen model + a 1394x1280 feature block + a 1280x576 probe)
#   CPU  32 GB requested for raw-series / arrow-shard loading (the Chronos-2 seven-dataset job
#        ran on 16 GB, noting BOOM alone may want 32 GB); the probe stage needs < 2 GB
#   time  extraction is ONE pass per batch (16x cheaper than the prefix ablation): seconds per
#         dataset. Probe fitting: 21 layers x (10 wd candidates + 1 null baseline) x 300
#         full-batch epochs on 1394 train rows, and the explicit-val protocol needs NO refit ->
#         ~2-3 min per dataset on an A100. Building the PT-OOD windows (arrow shards) is the slow
#         part of a cold run.
#
# ---------------------------------------------------------------------------------------
# ONE-TIME SETUP ON THE LOGIN NODE (it has internet; compute nodes do not):
#
#   module load gcc python/3.11 arrow/24.0.0 && source .venv/bin/activate
#   pip install "timesfm[torch]==3.0.2"        # not in the cluster wheelhouse -> no --no-index
#   export HF_HOME=$SCRATCH/chronos2/hf_cache
#   python -c "from timesfm3.torch import TimesFM3Forecaster as F; \
#              F.from_pretrained('google/timesfm-3.0-pytorch', device='cpu')"    # ~1.2 GB
#   python -m tests.test_timesfm3_last_token_probe     # model-free contracts: seconds, 2 threads
#
#   The PT-ID datasets are the SAME autogluon/chronos_datasets the Chronos-2 line uses, and the
#   PT-OOD arrow shards are already staged at $OOD_TARGET_ROOT for the Chronos-2 runs; if those
#   are warm there is nothing to download.
#
# Resubmitting is cheap: features are cached per (dataset, split, layer) and re-HIT, and the
# cache is REJECTED (never silently reused) if checkpoint, timesfm version, geometry, token
# index, layer set, dtype, detrending, dataset/split or seed disagree. On a cache HIT the L20
# all-quantile native reconstruction is re-proved on one fresh batch (--no-recheck-native
# disables that).
# =======================================================================================

CACHE=${TFM3_LT_CACHE_DIR:-$SCRATCH/timesfm3_last_token/features_cache}
OUT=${TFM3_LT_OUT_ROOT:-$SCRATCH/timesfm3_last_token/results}
CKPT=${TIMESFM3_CHECKPOINT:-google/timesfm-3.0-pytorch}
mkdir -p "$CACHE" "$OUT" logs

if [ "$CACHE" = "${TFM3_CACHE_DIR:-}" ] || [ "$OUT" = "${TFM3_OUT_ROOT:-}" ]; then
    echo "REFUSING TO RUN: this experiment's cache/results must NOT share a directory with"
    echo "the shared-origin ablation (TFM3_CACHE_DIR / TFM3_OUT_ROOT)."
    exit 1
fi

echo "=== TimesFM-3 last-context-token probing (token 15, native Q=9, suite paper7) ==="
echo "    checkpoint=$CKPT"
echo "    cache=$CACHE"
echo "    out=$OUT   HF_HOME=$HF_HOME"
echo "    OOD_TARGET_ROOT=$OOD_TARGET_ROOT  (PT-OOD window building)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

python -m experiments.run_timesfm3_last_token_probing \
    --checkpoint "$CKPT" \
    --cache-dir "$CACHE" \
    --out-root "$OUT" \
    --device cuda \
    --seed 0 \
    "$@"
echo "=== DONE ==="

# Useful variants (extra args are forwarded, and a later --flag overrides the one above):
#
#   STEP 1 - WINDOW AUDIT, no GPU, no model (do this FIRST). Builds all seven datasets' windows
#   and checks them against the committed Chronos-2 run, then exits. CPU+I/O heavy (arrow
#   shards), so it is compute-node work, NOT a login-node command:
#     sbatch --gres=gpu:0 --time=0:40:00 -J tfm3lt-audit \
#         job_timesfm3_last_token_q9.sh --audit-only --device cpu
#   Read the WINDOW AUDIT table in the log: every row must say parity=match, with
#   1394/262/262 windows for the four PT-ID datasets, 1394/354/354 for SG Carpark and BOOM,
#   and 1394/48/48 (24 stations) for Coastal T-S.
#
#   STEP 2 - GPU SMOKE RUN (one PT-ID + one PT-OOD dataset, three representation points):
#     sbatch --time=0:40:00 -J tfm3lt-smoke job_timesfm3_last_token_q9.sh \
#         --cache-dir $SCRATCH/timesfm3_last_token/smoke_cache \
#         --out-root  $SCRATCH/timesfm3_last_token/smoke_results \
#         --datasets monash_electricity_hourly sg_carpark --layers 0 10 20
#   It prints num_real_context_patches, selected_token_index, feature shapes, the target
#   round-trip error, the L20 all-quantile native reconstruction error, the frozen-backbone
#   status, and then the Emb/L10/L20 probe results.
#
#   sbatch job_timesfm3_last_token_q9.sh --suite extended_v1
#       the 4-dataset implementation-validation set (KDD Cup 2018 + Pedestrian Counts live
#       here); auto-split windows and the 80/20 carve. NOT the headline result.
#
#   sbatch job_timesfm3_last_token_q9.sh --no-detrend
#       ablation: TimesFM-3's linear detrending off in the BACKBONE and in the targets.
#
#   sbatch job_timesfm3_last_token_q9.sh --feature-dtype float16
#       HALVES the cache (~1 GB) via a lossy float16 storage cast (~1.2e-2 of the layer std,
#       measured and reported in feature_dtype_check). float32 (lossless) is the default.
#
#   sbatch job_timesfm3_last_token_q9.sh --allow-sorted-reference
#       ONLY if the L20 check reports that the installed timesfm3 sorts quantiles inside
#       decode() and exposes no bypass knob. Sorting is postprocessing; this records it
#       explicitly instead of silently comparing against a sorted path.
#
#   sbatch job_timesfm3_last_token_q9.sh --collect-history
#       per-epoch train/val curves. The SELECTION grid is
#       1e-5 1e-4 1e-3 1e-2 1e-1 0.3 1 3 10 30 (probes.WD_GRID_V2 extended by 10/30); the driver
#       warns if a layer selects the maximum, which would mean the grid is clipping. It stops at
#       30 on purpose: at lr=1e-2, wd=100 (lr*wd=1) zeroes the weight every step and wd=300
#       (lr*wd=3) makes AdamW's decoupled decay unstable, so either would report an optimizer
#       artifact as a probe. Passing such a value in --wd-grid is REFUSED.
#
#   --null-wd 100 (the default) fits ONE extra probe per layer at that extreme decay and reports
#       it as the no-information floor (~bias-only / marginal-quantile fit, max|W| ~ lr). It is
#       never a selection candidate; --null-wd 0 turns it off.
