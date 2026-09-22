#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # ONE GPU. Three backbones fit easily; nothing here is
                                     # multi-GPU aware, so asking for more would waste an
                                     # allocation without being used (spec V).
#SBATCH --cpus-per-task=4            # window building + StandardScaler + numpy target builds
#SBATCH --mem=48G                    # dominated by build_windows on the big rosters:
                                     # wiki_daily_100k loads ~2.2 GB of float64 raw series
                                     # BEFORE the deterministic cap can reduce it
#SBATCH --time=12:00:00              # see "WALL TIME" below -- 5 h is NOT enough for 42 cells
#SBATCH --output=results/three_model_final/logs/%x-%j.out
#SBATCH --error=results/three_model_final/logs/%x-%j.err

set -euo pipefail

# The log directory must exist BEFORE sbatch redirects into it, which is why the submit
# instructions below say to create it once. Re-created here too for a bare `bash` invocation.
mkdir -p results/three_model_final/logs

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1               # compute nodes are offline; models + datasets pre-cached
export OOD_TARGET_ROOT=${OOD_TARGET_ROOT:-$SCRATCH/chronos2/ood_targets}   # staged arrow shards
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}   # never grab every core on a shared node
export PYTHONHASHSEED=0               # deterministic alongside --seed
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}

PHASE1_OUT=${PHASE1_OUT:-results/three_model_final}
PHASE1_CACHE=${PHASE1_CACHE:-$SCRATCH/chronos2/three_model_final_cache}

# =======================================================================================
# PHASE 1 -- 14 datasets x 3 models = 42 cells, one resumable job.
#
#   Q1  At what depth does forecasting become RECOVERABLE?
#   Q2  How does that functional transition compare with representation GEOMETRY?
#
#   measured: layerwise Q=9 forecasting probes -> 5% VALIDATION tunnel entrance
#             unbiased linear CKA (headline) + biased (diagnostic)
#             entropy effective rank, on the SAME raw representation matrices
#   NOT in Phase 1: alignment, adapters, truncation.
#
# ONE PROTOCOL, THREE MODELS
#   C=512, H=64, Q=9 with the SAME quantile vector 0.1..0.9 everywhere (verified against each
#   model's own module at startup -- the run ABORTS if they ever disagree).
#   The reported loss is the mean pinball over (batch, quantiles, horizon) for all three.
#   The tunnel is probing.tunnel.tunnel_start, VALIDATION ONLY, tol 0.05.
#
#   chronos2  K=4 native forecast slots       -> ONE shared Linear(768, 9*16)   14 depths
#   timesfm3  last REAL context token (15)    -> Linear(1280, 64*9) per depth   21 depths
#   tirex     two_pass, token 63 of each pass -> ONE shared Linear(512, 9*32)   14 depths
#
# EACH MODEL IS PROBED IN ITS OWN TARGET SPACE, so the ABSOLUTE loss is NOT comparable across
# models. Cross-model comparison is valid for the tunnel entrance (a within-model ratio) and
# for MASE (raw units, one shared in-context seasonal-naive denominator with the dataset's own
# seasonal period from probing/registry.py). Every summary.json repeats this verbatim.
#
# RESUMABILITY -- the reason this is ONE job and not 42.
#   Every cell is built in `<model>/<dataset>.building-<pid>/` and moved into place with an
#   atomic rename only after all required artifacts validate. A preemption, a timeout or one
#   dataset failing therefore leaves NO partial cell. Submitting the EXACT SAME command again
#   skips every complete cell whose config hash matches and continues with the rest.
#   Feature caches live in $SCRATCH and are keyed by their own metadata, so a recomputed cell
#   still reuses valid extractions. Built windows are cached too -- on a resume they are NOT
#   rebuilt, which is what makes a second submission fast.
#
# WALL TIME -- read this before trusting 12:00:00.
#   Window building dominates a COLD run (the committed TimesFM-3 geometry job measured ~1.5 h
#   for SEVEN datasets, almost all of it windows). Phase 1 builds windows ONCE per dataset and
#   shares them across the three models, then extracts + fits ~700 linear probes.
#   The estimate has NOT been measured end to end on Narval yet: run the smokes in the
#   "SMOKE FIRST" section and read the per-cell timings out of
#   results/three_model_final/cells.json before deciding. 12 h is a deliberately conservative
#   first request; if it is not enough, resubmit the SAME command -- that is the whole point.
#
# DISK -- ONE THING TO SET UP BEFORE THE FIRST RUN
#   TimesFM-3 and TiRex feature caches honour --cache-root ($SCRATCH): ~2.7 GB + ~1.5 GB.
#   Chronos-2's DOES NOT. `probing.config.CACHE_DIR` is anchored to the repo root, so its
#   ~6.4 GB lands in <repo>/features_cache, i.e. in $HOME. That is deliberate -- it is the same
#   cache key the committed Chronos-2 runs use, so Phase 1 REUSES anything already extracted --
#   but against a ~50 GB $HOME quota it is worth redirecting once:
#       mv features_cache $SCRATCH/chronos2/features_cache   # if it already has contents
#       ln -s $SCRATCH/chronos2/features_cache features_cache
#   Durable results in the repo: ~1.8 GB (~0.8 GB with --no-probe-artifacts).
#
# WHERE TO RUN
#   This script: a COMPUTE NODE, via sbatch. Never a login node -- it loads three foundation
#   models and sustains a GPU for hours.
#   Login-node safe: `--plan` (cell list, no data) and `python -m tests.test_three_model_phase1`.
#   `--audit-only` builds real windows and needs a COMPUTE NODE (multi-GB raw series), but no GPU.
# =======================================================================================
#
# ---------------------------------------------------------------------------------------
# ONE-TIME, ON THE LOGIN NODE (it has internet; compute nodes do not)
# ---------------------------------------------------------------------------------------
#   mkdir -p results/three_model_final/logs
#   pip install --no-index -r requirements.txt   # + "tirex-ts==1.4.2" and timesfm (need internet)
#   export HF_HOME=$SCRATCH/chronos2/hf_cache
#   python -c "from probing.extraction import get_pipeline; get_pipeline()"        # chronos-2
#   python -c "from probing.timesfm3_last_token import get_model; get_model()"     # timesfm-3
#   python -c "from probing.tirex_model import get_model; get_model()"             # tirex
#   # stage the three cluster-rostered datasets' arrow shards into $OOD_TARGET_ROOT
#   python -m tests.test_three_model_phase1        # 49 model-free contracts, ~15 s, 2 threads
#   python -m experiments.run_three_model_phase1 --plan        # the 42-cell list, no data read
#
# ---------------------------------------------------------------------------------------
# STEP 1 -- WINDOW AUDIT (compute node, NO GPU, NO model)
# ---------------------------------------------------------------------------------------
# Builds all 14 datasets' real windows, checks parity against the committed Chronos-2 artifacts
# (the original seven) and writes the audits. It ABORTS on any dataset that has no window
# reference, which is the seven new ones on a first run -- see STEP 2.
#
#   sbatch --gres=gpu:0 --time=3:00:00 -J p1-audit job_three_model_phase1.sh --audit-only
#
# ---------------------------------------------------------------------------------------
# STEP 2 -- CREATE THE SEVEN NEW WINDOW REFERENCES (compute node, NO GPU) -- DELIBERATE
# ---------------------------------------------------------------------------------------
# A reference pins the counts, geometry, seasonal period, the deterministic series cap, the
# per-window test ids ELEMENT-WISE and a sha256 of the test contexts. Once written, every later
# run of ANY model line is checked against it. Creating one is a separate, explicit act
# precisely because it defines what "the same windows" means from then on.
#
#   sbatch --gres=gpu:0 --time=3:00:00 -J p1-ref job_three_model_phase1.sh \
#       --audit-only --reference-mode create
#
#   Then INSPECT results/three_model_final/window_audits/*.json and
#   results/window_references/*.json before going further, and COMMIT the references.
#
# ---------------------------------------------------------------------------------------
# STEP 3 -- GPU SMOKE: all three models, one dataset, real Q=9 shapes, real model paths
# ---------------------------------------------------------------------------------------
#   sbatch --time=1:30:00 -J p1-smoke job_three_model_phase1.sh \
#       --output-root $SCRATCH/phase1_smoke/results \
#       --cache-root  $SCRATCH/phase1_smoke/cache \
#       --datasets monash_electricity_hourly \
#       --probe-epochs 30 --boot-b 200 --cka-null-floor-reps 1
#
#   Then a NON-HOURLY smoke, to exercise the new seasonal period and the new window logic:
#   sbatch --time=1:30:00 -J p1-smoke5m job_three_model_phase1.sh \
#       --output-root $SCRATCH/phase1_smoke/results \
#       --cache-root  $SCRATCH/phase1_smoke/cache \
#       --datasets LOOP_SEATTLE_5T \
#       --probe-epochs 30 --boot-b 200 --cka-null-floor-reps 1
#   (LOOP_SEATTLE_5T is m=288; a silent fallback to 24 would be a 2-hour "season" on 5-minute
#   data. The banner prints the period it used for every cell.)
#
#   Read the per-cell wall times out of $SCRATCH/phase1_smoke/results/cells.json and multiply
#   by 14 before trusting the --time above.
#
# ---------------------------------------------------------------------------------------
# STEP 4 -- THE FULL RUN, and the IDENTICAL resume command
# ---------------------------------------------------------------------------------------
#   sbatch job_three_model_phase1.sh
#
#   If it is preempted, times out, or a dataset fails, submit the SAME line again:
#   sbatch job_three_model_phase1.sh
#
#   Inspect it WHILE it runs (login node, seconds):
#   python -m experiments.make_phase1_tables
#   cat results/three_model_final/cells.json | python -m json.tool | head -40
# =======================================================================================

echo "=================================================================================="
echo "PHASE 1   host=$(hostname)   job=${SLURM_JOB_ID:-none}   started $(date -Is)"
echo "  output (DURABLE, in the repo)   : $PHASE1_OUT"
echo "  cache  (HEAVY, on scratch)      : $PHASE1_CACHE"
echo "  extra args                      : $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
  || echo "  (no GPU visible -- fine for --audit-only)"
echo "=================================================================================="

python -m experiments.run_three_model_phase1 \
    --output-root "$PHASE1_OUT" \
    --cache-root  "$PHASE1_CACHE" \
    --quantile-set q9 \
    --suite paper14 \
    --device cuda \
    --resume \
    "$@"
# --device cuda is explicit (not left to auto-detect) so every model's probe fit AND predict run
# on the same concrete device; extra args in "$@" (e.g. --audit-only --device cpu) still override.

echo "=================================================================================="
echo "PHASE 1 finished $(date -Is)"
echo "  combined tables : $PHASE1_OUT/combined/"
echo "  cell statuses   : $PHASE1_OUT/cells.json"
echo "  resume with the IDENTICAL command: sbatch job_three_model_phase1.sh $*"
echo "=================================================================================="
