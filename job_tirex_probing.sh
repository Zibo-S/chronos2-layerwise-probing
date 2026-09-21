#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # K=2 native passes per batch + 14 probe fits
#SBATCH --cpus-per-task=4            # StandardScaler + the numpy target build + data loading
#SBATCH --mem=32G                    # dominated by loading the raw series in build_windows
#SBATCH --time=6:00:00               # 7 datasets x two_pass (K=2 passes/window); window
                                     # building (PT-OOD arrow shards) also dominates a cold run
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

set -euo pipefail

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1               # compute nodes are offline; model + datasets pre-cached
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}   # PT-OOD arrow shards
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # never grab every core on a shared node
export PYTHONHASHSEED=0               # deterministic alongside --seed
# TiRex may build custom xLSTM CUDA kernels; give ninja a WRITABLE dir inside our own space
# (the default ~/.cache/torch_extensions is fine on Narval but $SCRATCH keeps $HOME quota free).
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}

# =======================================================================================
# TiRex layer-wise probing: Q=1 median shared-patch probe + tunnel + CKA + effective rank.
#
#   model      NX-AI/TiRex via tirex-ts, FROZEN (load_model returns requires_grad=True --
#              probing.tirex_model.get_model freezes it and asserts the result)
#   geometry   RE-DERIVED from the checkpoint every run, never hardcoded:
#                input/output patch 32 | d=512 | 12 sLSTM blocks | 4 heads | ff 2048
#                quantiles 0.1..0.9 (median index 4) | train_ctx_len 2048
#              _forecast_single_step forces EVERY context to exactly train_ctx_len by NaN
#              LEFT-padding, so C=512 becomes 48 all-masked pad patches + 16 real patches =
#              64 context tokens. The last REAL context patch is token 63, NOT token 15.
#   rollout    PRIMARY = two_pass (the DEFAULT of --rollout-mode), because it is the released
#              tirex-ts package's own default inference pathway. The mode is part of the CACHE
#              PATH, so the two can never be mixed.
#                two_pass [PRIMARY]  max_accelerated_rollout_steps=1 -> 2 passes of 64 tokens,
#                             each read at ITS OWN token 63, each with ITS OWN (loc, scale):
#                               pass 0: ctx 512  -> pad 1536, real 48..63, readout 63 = LAST REAL
#                               pass 1: ctx 544  -> pad 1504, real 47..62, readout 63 = APPENDED
#                                                   MISSING patch
#                             Pass 1 is handed the context extended by 32 MISSING (NaN) values --
#                             NEVER the previous forecast:
#                               _forecast_tensor: torch.cat([context,
#                                   torch.full_like(prediction[:, 0, :], fill_value=torch.nan)])
#                             `full_like` takes the forecast's SHAPE and fills it with NaN, so the
#                             prediction is a shape template only. assert_rollout_appends_missing
#                             proves this against the running model every run.
#                single_pass [robustness]  max_accelerated_rollout_steps=2 -> ONE 65-token pass,
#                             readouts at tokens [63, 64], ONE (loc, scale).
#              MEASURED on Electricity: the two modes agree EXACTLY on the first readout state
#              (the sLSTM is causal) and differ on the second; the tunnel entrance, the depth
#              ordering and the headline CKA are IDENTICAL, the curves move ~3% of their range.
#              --compare-rollout-modes reproduces that comparison.
#   depths     14 points: Emb (pre-block-1), L1..L12 (COMPLETE blocks: post-sLSTM AND post-FFN),
#              L12+RMS (what output_patch_embedding reads)
#   probe      ONE SHARED Linear(512, 32) per depth applied to BOTH readout states, concatenated
#              to H=64. Q=1, tau=0.5. 16,416 parameters on 2*1394 = 2788 train rows.
#   windows    EXACTLY the Chronos-2 / TimesFM-3 ones. The run ABORTS unless the counts AND the
#              per-window test series ids match results/ext_v5_native_head_adapter element-wise.
#   tunnel     probing.tunnel.tunnel_start, UNCHANGED: min { l : L_val(l) <= 1.05*L_val(final) }.
#              VALIDATION only (1%/2%/5%/10% all saved).
#   geometry   probing.cka (biased AND unbiased) + probing.spectral_metrics, UNCHANGED.
#                HEADLINE  `pos0`: readout position 0 (the last-real-context state) at EVERY
#                          depth -> (N, 512), N CONSTANT across depth, for BOTH estimators. A
#                          curve whose observation count changes with depth is NOT produced: both
#                          min(N-1, d) and the O(1/N) biased-CKA floor move with N, so the jump
#                          would be indistinguishable from a representational change.
#                COMPANION `all_positions_from_L1`: both forecast states -> (2N, 512), over
#                          L1..L12+RMS ONLY. Emb is omitted because its second readout state is
#                          constant before recurrence (measured every run).
#
# THE GATE: every dataset re-proves that the forecast-producing states, pushed through the frozen
# output_patch_embedding and TiRex's own per-pass de-normalization, reproduce the native H=64
# forecast, with wrong-index controls that must differ by orders of magnitude. A failure ABORTS
# the run -- do not loosen the tolerance.
#   two_pass    state for patch k from the pass that produced patch k; gated elementwise
#               (|d| <= atol + rtol*|ref|). Measured 2.4e-4 abs / 0.25 scaled on Electricity; NOT
#               bit-exact because the native path runs the head on the full (n, 64, 512) sequence
#               and slices after, while we slice first -- a slice-order control (1.9e-6) attributes
#               the residual to exactly that. assert_two_pass_matches_package separately proves
#               our replicated rollout equals the package's own output BITWISE.
#   single_pass bit-exact (max|d| = 0.0).
#
# BACKEND. --backend torch is the pure-PyTorch sLSTM: its recurrence runs in bfloat16 and it
# needs NO custom kernels, so it always works. --backend cuda uses xLSTM's compiled kernels and
# requires `pip install xlstm ninja` on the LOGIN node (internet) plus a writable
# TORCH_EXTENSIONS_DIR. The two are NOT bit-identical -- pick one, keep it for the whole paper,
# and note that the cache records which was used and REFUSES features extracted under the other.
# =======================================================================================

CACHE=${TIREX_CACHE_DIR:-$SCRATCH/tirex/features_cache}
OUT=${TIREX_OUT_ROOT:-$SCRATCH/tirex/results}
CKPT=${TIREX_CHECKPOINT:-NX-AI/TiRex}
BACKEND=${TIREX_BACKEND:-torch}
mkdir -p "$CACHE" "$OUT" "$TORCH_EXTENSIONS_DIR" logs

echo "=== TiRex layer-wise probing (Q=1 tau=0.5, shared patch head, suite paper7) ==="
echo "    PRIMARY rollout mode: two_pass (the released package's default inference pathway)"
echo "    checkpoint=$CKPT  backend=$BACKEND"
echo "    cache=$CACHE"
echo "    out=$OUT   HF_HOME=$HF_HOME"
echo "    OOD_TARGET_ROOT=$OOD_TARGET_ROOT   TORCH_EXTENSIONS_DIR=$TORCH_EXTENSIONS_DIR"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
python -c "import tirex, torch; print('tirex-ts', __import__('importlib.metadata', fromlist=['version']).version('tirex-ts'), '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"

python -m experiments.run_tirex_probing \
    --checkpoint "$CKPT" \
    --backend "$BACKEND" \
    --cache-dir "$CACHE" \
    --out-root "$OUT" \
    --device cuda \
    --batch-size 256 \
    --seed 0 \
    "$@"
echo "=== DONE ==="

# ---------------------------------------------------------------------------------------
# ONE-TIME SETUP ON THE LOGIN NODE (it has internet; compute nodes do not):
#
#   module load gcc python/3.11 arrow/24.0.0 && source .venv/bin/activate
#   pip install "tirex-ts==1.4.2"                  # not in the cluster wheelhouse -> no --no-index
#   export HF_HOME=$SCRATCH/chronos2/hf_cache
#   python -c "from tirex import load_model; load_model('NX-AI/TiRex', device='cpu', backend='torch')"
#                                                  # ~141 MB checkpoint, pre-cache it while online
#   python -m tests.test_tirex_probing             # model-free contracts: ~5 s, 2 threads
#
#   OPTIONAL, only for --backend cuda:
#   pip install xlstm ninja
#
# RUN ORDER
#
#   STEP 1 - GEOMETRY REPORT (loads the model, touches no data; ~30 s, so it is compute-node
#   work under salloc, NOT a login-node command):
#     salloc --account=def-irina --gres=gpu:0 --cpus-per-task=2 --mem=8G --time=0:20:00
#     export OMP_NUM_THREADS=2
#     python -m experiments.run_tirex_probing --discover-only --device cpu \
#         --out-root $SCRATCH/tirex/results
#   Confirm: train_ctx_len 2048, 64 context tokens, READOUT TOKENS [63, 64], 14 points.
#
#   STEP 2 - MODEL-BACKED CONTRACTS (the native-head identity, inside the same salloc):
#     python -m tests.test_tirex_probing --with-model
#   Every line must say OK, and contract 5 must report max scaled err <= 1 with the three
#   wrong-index controls orders of magnitude larger.
#
#   STEP 3 - GPU SMOKE RUN (one dataset, four depths, 64 windows per split):
#     sbatch --time=0:40:00 -J tirex-smoke job_tirex_probing.sh \
#         --datasets monash_electricity_hourly --points Emb L6 L12 "L12+RMS" \
#         --limit 64 --epochs 60 --wd-grid 1e-3 1e-1 --boot-b 200 \
#         --cache-dir $SCRATCH/tirex/smoke_cache --out-root $SCRATCH/tirex/smoke_results
#
#   STEP 4 - ONE FULL DATASET (all 14 depths, full splits) before committing to the seven:
#     sbatch --time=1:00:00 -J tirex-elec job_tirex_probing.sh \
#         --datasets monash_electricity_hourly \
#         --cache-dir $SCRATCH/tirex/elec_cache --out-root $SCRATCH/tirex/elec_results
#
#   STEP 4b - THE ROLLOUT-MODE DECISION (run BEFORE committing to the seven). Probes the same
#   dataset under BOTH native inference modes and writes rollout_mode_comparison.json:
#     sbatch --time=1:30:00 -J tirex-modes job_tirex_probing.sh \
#         --datasets monash_electricity_hourly --compare-rollout-modes \
#         --cache-dir $SCRATCH/tirex/cmp_cache --out-root $SCRATCH/tirex/cmp_results
#   Read the ROLLOUT-MODE COMPARISON table: it reports whether the tunnel entrance moves, whether
#   the test-loss argmin and depth ORDERING survive, and how far the curves and the effective
#   rank shift. Pick the paper's primary mode from that, then pass --rollout-mode explicitly.
#
#   STEP 5 - THE SEVEN-DATASET RUN (only after steps 1-4b are green). two_pass is the DEFAULT,
#   so no flag is needed:
#     sbatch job_tirex_probing.sh
#
# Useful variants (extra args are forwarded; a later --flag overrides the one above):
#
#   sbatch job_tirex_probing.sh --rollout-mode single_pass
#       the one-pass accelerated path, as a robustness check. Separate cache path, separate
#       numbers; extraction costs HALF (one forward pass per window instead of K=2).
#
#   sbatch job_tirex_probing.sh --backend cuda
#       the xLSTM custom kernels. Requires `pip install xlstm ninja`. Features extracted under
#       the other backend are REFUSED by the cache, not silently reused.
#
#   sbatch job_tirex_probing.sh --suite extended_v1
#       the 4-dataset implementation-validation set. NOT the headline result, and note it has no
#       dedicated validation split, so the driver will refuse it -- use paper7 for real numbers.
#
#   sbatch job_tirex_probing.sh --erank-subsamples 200
#       adds the Chronos-2 subsampling uncertainty protocol to the effective rank (200 SVDs per
#       depth per split -- minutes, not seconds).
#
#   sbatch job_tirex_probing.sh --wd-grid 1e-5 1e-4 1e-3 1e-2 1e-1 0.3 1 3 10 30
#       widen the selection grid if the run WARNS that a depth selected the grid maximum.
#
# COST (7 datasets x 14 depths, C=512, H=64, ~1918 windows each):
#   feature cache 14 depths x n_windows x 2 positions x 512 x float32
#                 ~110 MB per PT-ID dataset -> ~0.8 GB for the seven  -> $SCRATCH, NEVER $HOME
#   GPU  < 3 GB   (141 MB frozen model + a 2788x512 feature block + a 512x32 probe)
#   NOTE  two_pass (the PRIMARY mode) runs K=2 forward passes per window. Measured on CPU:
#         472 s/dataset vs 279 s for single_pass (probe/geometry stages unchanged).
#   time  the sLSTM torch backend is a PYTHON LOOP over 65 timesteps x 12 blocks, so it is
#         launch-bound on a GPU: use a LARGE --batch-size (256) to amortize it. Measured on CPU
#         (Apple M-series, 4 threads): ~117 ms/window. Probe fitting: 14 depths x 8 wd
#         candidates x 300 full-batch epochs on 2788x512 -- a few minutes per dataset.
#         Building the PT-OOD windows (arrow shards) is the slow part of a cold run.
