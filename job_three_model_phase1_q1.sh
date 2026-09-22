#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # ONE GPU, Narval A100-40GB. NOT an H100/Fir request.
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G                    # build_windows dominates (wiki_daily_100k ~2.2 GB raw)
#SBATCH --time=12:00:00              # resubmit the SAME line to resume
#SBATCH --output=results/three_model_phase1_q1/logs/%x-%j.out
#SBATCH --error=results/three_model_phase1_q1/logs/%x-%j.err

set -euo pipefail

# =======================================================================================
# PHASE 1 -- EXPERIMENT B: the Q=1 (tau=0.5) ROBUSTNESS RERUN.
#
# NOT the headline. Q=9 remains the intended headline probabilistic forecasting experiment
# (job_three_model_phase1_q9_expanded.sh). This run asks ONE question:
#
#     does the layerwise forecast-recoverability / tunnel conclusion persist when the readout
#     is much lower capacity and predicts only the median?
#
# It is a PROBE-CAPACITY and OBJECTIVE robustness check. A Q1/Q9 difference is NOT by itself
# evidence that Q=9 overfits, and no artifact of this run says that it is.
#
# IDENTICAL TO EXPERIMENT A IN EVERYTHING EXCEPT Q:
#   same 14 datasets, same canonical windows, same train/val/test splits, C=512, H=64;
#   same representation points and final-block references; same EXPANDED wd grid; same
#   optimizer/epochs/lr/seeds; same validation-only wd selection; same SUSTAINED tunnel at
#   2/5/10%; same bootstrap; same MASE/MAE from the median prediction; same checkpoints.
#
# THE Q=1 PROBES ARE THE VALIDATED MODEL-SPECIFIC FORMS, reached by passing Q=1 to the SAME
# fitting code the Q=9 run uses -- nothing is reimplemented:
#   chronos2  Linear(768, 1*16)  ONE shared head over the K=4 forecast slots -> H=64 median
#   timesfm3  Linear(1280, 64*1) per representation point
#   tirex     Linear(512, 1*32)  ONE shared head over both native forecast-producing states
# (tests D and E of tests/test_phase1_wd_and_sustained_tunnel.py pin these shapes and the
# median-target construction against the live implementations.)
#
# SEPARATE TREE, SEPARATE HASHES. Q1 and Q9 cells can never collide: the quantile set, the
# quantile vector and the probe output width are all in the cell config hash, and the two runs
# write to different --output-root trees.
#
# THE FEATURE CACHE IS SHARED WITH EXPERIMENT A, ON PURPOSE. Extraction never sees a quantile
# vector, so the cached representations are byte-identical for Q1 and Q9. Pointing both jobs at
# the same --cache-root means the three backbones run ONCE for both experiments; this job is
# then probe-fitting only and much faster than A's cold run. Do NOT give it its own cache root.
# Geometry (CKA / effective rank) is likewise Q-independent; it is cheaply RECOMPUTED from
# those same cached representations rather than copied, and
# experiments.make_phase1_q1_q9_comparison VERIFIES the two runs' matrices are identical
# element-wise instead of assuming it.
# =======================================================================================
#
# ---------------------------------------------------------------------------------------
# HOW TO USE THIS FILE
# ---------------------------------------------------------------------------------------
#   0. ONE-TIME, LOGIN NODE:
#        mkdir -p results/three_model_phase1_q1/logs
#        python -m tests.test_three_model_phase1
#        python -m tests.test_phase1_wd_and_sustained_tunnel
#
#   1. SMOKE (GPU, ~30 min each). Run these BEFORE the two full jobs.
#
#      1a. THE ONE THAT MATTERS -- TimesFM-3 x M5 at Q=9. The old run selected wd=30, the old
#          grid maximum, at ALL 21 depths, so this is where the expanded grid must show
#          interior selections:
#            sbatch --time=1:00:00 -J p1-smoke-m5 job_three_model_phase1_q9_expanded.sh \
#                --output-root $SCRATCH/phase1_smoke/q9 \
#                --cache-root  $SCRATCH/phase1_smoke/cache \
#                --datasets m5 --models timesfm3 \
#                --boot-b 200 --cka-null-floor-reps 1
#          Then, on the login node:
#            column -s, -t $SCRATCH/phase1_smoke/q9/combined/wd_selection_summary.csv
#            python -c "import json;d=json.load(open('$SCRATCH/phase1_smoke/q9/timesfm3/m5/probe_hparams.json'));print(d['grid_clipping'])"
#          READ: n_layers_at_grid_max should now be small. If it is still 21, read
#          at_grid_max_val_over_floor -- a ratio near 1.0 means those depths' optima ARE the
#          no-information floor (a finding), not a cut-off search.
#          NOTE: --probe-epochs is deliberately NOT reduced here. Weight-decay selection is the
#          thing under test and it depends on the number of steps, so the smoke must use the
#          real 300 epochs or it measures a different optimization problem.
#
#      1b. Shape / device smoke for the other two models, BOTH quantile sets, one cheap dataset:
#            sbatch --time=1:00:00 -J p1-smoke-q9 job_three_model_phase1_q9_expanded.sh \
#                --output-root $SCRATCH/phase1_smoke/q9 \
#                --cache-root  $SCRATCH/phase1_smoke/cache \
#                --datasets monash_electricity_hourly --models chronos2 tirex \
#                --probe-epochs 30 --boot-b 200 --cka-null-floor-reps 1
#            sbatch --time=1:00:00 -J p1-smoke-q1 job_three_model_phase1_q1.sh \
#                --output-root $SCRATCH/phase1_smoke/q1 \
#                --cache-root  $SCRATCH/phase1_smoke/cache \
#                --datasets monash_electricity_hourly --models chronos2 tirex timesfm3 \
#                --probe-epochs 30 --boot-b 200 --cka-null-floor-reps 1
#          (--probe-epochs 30 IS fine here: 1b only checks shapes, devices and that Q=1 runs
#          end to end. Do not read wd selections off it.)
#
#   2. THE FULL RUNS, after the smokes pass:
#        sbatch job_three_model_phase1_q9_expanded.sh     # headline first: it fills the cache
#        sbatch job_three_model_phase1_q1.sh              # then this one, much faster
#      Resume either by submitting the IDENTICAL line again.
#
#   3. THE CROSS-RUN COMPARISON (login node, seconds, after BOTH finish):
#        python -m experiments.make_phase1_q1_q9_comparison
#      -> results/phase1_q1_q9_comparison/combined_q1_q9/
# =======================================================================================

mkdir -p results/three_model_phase1_q1/logs

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONHASHSEED=0
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}

PHASE1_OUT=${PHASE1_OUT:-results/three_model_phase1_q1}
# THE SAME caches as Experiment A. Extraction and window building are Q-independent.
PHASE1_CACHE=${PHASE1_CACHE:-$SCRATCH/chronos2/phase1_shared_cache}
PHASE1_WINDOWS=${PHASE1_WINDOWS:-$SCRATCH/chronos2/phase1_shared_windows}

echo "=================================================================================="
echo "PHASE 1 / EXPERIMENT B -- Q=1 (tau=0.5) ROBUSTNESS  [NOT the headline]"
echo "  host=$(hostname)   job=${SLURM_JOB_ID:-none}   started $(date -Is)"
echo "  output (DURABLE, in the repo)   : $PHASE1_OUT"
echo "  feature cache (SHARED with Q9)  : $PHASE1_CACHE"
echo "  window cache  (SHARED with Q9)  : $PHASE1_WINDOWS"
echo "  extra args                      : $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "  (no GPU visible -- fine for --audit-only)"
echo "=================================================================================="

python -m experiments.run_three_model_phase1 \
    --output-root  "$PHASE1_OUT" \
    --cache-root   "$PHASE1_CACHE" \
    --window-cache "$PHASE1_WINDOWS" \
    --quantile-set q1 \
    --suite paper14 \
    --tunnel-tol 0.05 \
    --tunnel-tols 0.01 0.02 0.05 0.10 \
    --device cuda \
    --resume \
    "$@"

echo "=================================================================================="
echo "PHASE 1 / EXPERIMENT B finished $(date -Is)"
echo "  combined tables : $PHASE1_OUT/combined/"
echo "  cell statuses   : $PHASE1_OUT/cells.json"
echo "  NEXT (login node): python -m experiments.make_phase1_q1_q9_comparison"
echo "  resume with the IDENTICAL command: sbatch job_three_model_phase1_q1.sh $*"
echo "=================================================================================="
