#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # ONE GPU. Narval's are A100-40GB; nothing here is
                                     # multi-GPU aware, so asking for more wastes an allocation.
                                     # Deliberately NOT --gres=gpu:h100 / --constraint=... :
                                     # this must schedule on Narval's A100 partition.
#SBATCH --cpus-per-task=4            # window building + StandardScaler + numpy target builds
#SBATCH --mem=48G                    # dominated by build_windows on the big rosters:
                                     # wiki_daily_100k loads ~2.2 GB of float64 raw series
                                     # BEFORE the deterministic cap can reduce it
#SBATCH --time=12:00:00              # resubmit the SAME line to resume; see WALL TIME below
#SBATCH --output=results/three_model_phase1_q9_expanded/logs/%x-%j.out
#SBATCH --error=results/three_model_phase1_q9_expanded/logs/%x-%j.err

set -euo pipefail

# =======================================================================================
# PHASE 1 -- EXPERIMENT A: the Q=9 HEADLINE RERUN (expanded wd grid + sustained tunnel).
#
# This is the canonical Phase-1 headline run. It differs from the earlier
# results/three_model_final/ run in EXACTLY THREE ways, and in nothing else:
#
#   1. WEIGHT-DECAY GRID -- expanded, and now SHARED by all three models:
#        old  chronos2   1e-5 1e-4 1e-3 1e-2 1e-1 0.3 1 3                 (max 3)
#        old  timesfm3   ... + 10 30                                      (max 30)
#        old  tirex      ... + 10 30                                      (max 30)
#        NEW  all three  1e-5 1e-4 1e-3 1e-2 1e-1 0.3 1 3 10 30 45 65 90  (max 90)
#      Why: measured clipping. timesfm3 x Electricity selected the old max 30 at 16 of 21
#      depths with validation still falling; chronos2 x m4_hourly selected ITS max 3 at 9 of
#      14. Why it stops at 90 and not at 100/300/1000: AdamW's decay is DECOUPLED, so each
#      step multiplies the weight by (1 - lr*wd). At lr=1e-2, wd=100 gives lr*wd=1 and zeroes
#      the weight every step -- that is the no-information NULL, not a stronger regularizer --
#      and wd>=300 diverges. The grid runs UP TO the wall, not through it.
#
#   2. TUNNEL DEFINITION -- SUSTAINED ENTRY, not first crossing:
#        l_tunnel(tol) = min { l : max_{j >= l} (L_val(j)/L_val(final block) - 1) <= tol }
#      An isolated early crossing no longer opens a tunnel. The old statistic is still
#      computed and saved, under the name first_crossing_<tol>, as a diagnostic.
#
#   3. TOLERANCE SENSITIVITY -- 2% / 5% / 10% all saved (5% is the headline), plus 1% for free.
#
# EVERYTHING ELSE IS UNCHANGED: 14 datasets, the same windows and splits, C=512, H=64, the
# same Q = [0.1 .. 0.9], the same representation points and final-block references, the same
# probe architectures, optimizer, epochs (300), lr (1e-2), seeds, bootstrap, MASE/MAE/WQL,
# CKA, effective rank, provenance and model checkpoints.
#
# THE OLD RUN IS NOT TOUCHED. results/three_model_final/ is the audit trail. This job writes
# to its own tree and the driver refuses to mix two protocols in one directory.
# =======================================================================================
#
# ---------------------------------------------------------------------------------------
# HOW TO USE THIS FILE
# ---------------------------------------------------------------------------------------
#   0. ONE-TIME, LOGIN NODE (it has internet; compute nodes do not):
#        mkdir -p results/three_model_phase1_q9_expanded/logs
#        python -m tests.test_three_model_phase1              # 57 contracts, ~15 s, 2 threads
#        python -m tests.test_phase1_wd_and_sustained_tunnel  # 14 contracts, ~10 s, 2 threads
#
#   1. SMOKE FIRST (see the smoke section of the Q1 script; both scripts share it).
#
#   2. THE FULL RUN:            sbatch job_three_model_phase1_q9_expanded.sh
#      RESUME (identical line): sbatch job_three_model_phase1_q9_expanded.sh
#
#   3. Inspect WHILE it runs (login node, seconds):
#        python -m experiments.make_phase1_tables \
#            --output-root results/three_model_phase1_q9_expanded
#        column -s, -t results/three_model_phase1_q9_expanded/combined/wd_selection_summary.csv
#
# WALL TIME. Window building dominates a COLD run; the three models then share those windows.
# The measured per-cell times from the first Phase-1 attempt were 48-123 s per cell with a WARM
# feature cache, so the 12 h request is dominated by cold window building and extraction, not by
# probe fitting. The expanded grid adds 3 of 13 candidates = ~30% more probe fits per depth.
# Read the real numbers out of cells.json after the smoke and resubmit if 12 h is short --
# resuming is the whole point.
#
# DISK. Chronos-2's feature cache is anchored to the repo root (probing.config.CACHE_DIR), so
# redirect it once if $HOME is tight:
#     mv features_cache $SCRATCH/chronos2/features_cache && ln -s $SCRATCH/chronos2/features_cache features_cache
# Durable results in the repo: ~1.8 GB (~0.8 GB with --no-probe-artifacts).
# FEATURE CACHES ARE SHARED WITH THE Q1 RUN -- they are keyed by dataset/split/checkpoint/
# extraction parameters and never by Q -- so pointing both jobs at the same --cache-root means
# the backbones run ONCE for both experiments. That is deliberate; do not give them separate
# cache roots.
# =======================================================================================

mkdir -p results/three_model_phase1_q9_expanded/logs

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1               # compute nodes are offline; models + datasets pre-cached
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}   # staged arrow shards
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # never grab every core on a shared node
export PYTHONHASHSEED=0               # deterministic alongside --seed
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}

PHASE1_OUT=${PHASE1_OUT:-results/three_model_phase1_q9_expanded}
# SHARED with the Q1 job on purpose: feature caches are Q-independent.
PHASE1_CACHE=${PHASE1_CACHE:-$SCRATCH/chronos2/phase1_shared_cache}
PHASE1_WINDOWS=${PHASE1_WINDOWS:-$SCRATCH/chronos2/phase1_shared_windows}

echo "=================================================================================="
echo "PHASE 1 / EXPERIMENT A -- Q=9 HEADLINE (expanded wd grid, sustained tunnel)"
echo "  host=$(hostname)   job=${SLURM_JOB_ID:-none}   started $(date -Is)"
echo "  output (DURABLE, in the repo)   : $PHASE1_OUT"
echo "  feature cache (SHARED with Q1)  : $PHASE1_CACHE"
echo "  window cache  (SHARED with Q1)  : $PHASE1_WINDOWS"
echo "  extra args                      : $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "  (no GPU visible -- fine for --audit-only)"
echo "=================================================================================="

python -m experiments.run_three_model_phase1 \
    --output-root  "$PHASE1_OUT" \
    --cache-root   "$PHASE1_CACHE" \
    --window-cache "$PHASE1_WINDOWS" \
    --quantile-set q9 \
    --suite paper14 \
    --tunnel-tol 0.05 \
    --tunnel-tols 0.01 0.02 0.05 0.10 \
    --device cuda \
    --resume \
    "$@"
# The wd grid is NOT passed on the command line on purpose: it is probing.phase1.PHASE1_WD_GRID,
# one grid in one place, recorded in every cell_config.json and in the config hash. Overriding it
# with --wd-grid is possible but then the run is no longer the canonical protocol.

echo "=================================================================================="
echo "PHASE 1 / EXPERIMENT A finished $(date -Is)"
echo "  combined tables : $PHASE1_OUT/combined/"
echo "     tunnel_matrix.csv            sustained 5% headline"
echo "     tunnel_matrix_{02,05,10}.csv tolerance sensitivity"
echo "     tunnel_sensitivity.csv       per cell x tolerance, + first-crossing diagnostic"
echo "     wd_selection_summary.csv     IS THE EXPANDED GRID STILL CLIPPED?"
echo "  cell statuses   : $PHASE1_OUT/cells.json"
echo "  resume with the IDENTICAL command: sbatch job_three_model_phase1_q9_expanded.sh $*"
echo "=================================================================================="
