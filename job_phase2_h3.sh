#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # ONE A100: the iterative Chronos-2 / TiRex adapter fits
#SBATCH --cpus-per-task=4            # data loading + the closed-form TimesFM-3 eigendecompositions
#SBATCH --mem=48G                    # all-depth features of one cell stay in RAM (<1 GB) + windows
#SBATCH --time=03:00:00              # <=3 h = Narval's shortest time tier (most nodes, backfill);
                                     # a timeout costs little: resubmit the SAME line to resume
#SBATCH --output=results/three_model_phase2/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2/logs/%x-%j.err

set -euo pipefail

# =======================================================================================
# PHASE 2 / H3 -- can an intermediate representation FUNCTIONALLY REPLACE the final one?
#
# Per model x dataset x block depth, offline from the Phase-1 caches (READ-ONLY):
#   native | hard cut | label-free native-output alignment (NOA, primary) |
#   supervised forecast-loss alignment (FL; Chronos-2, TiRex) | hidden ridge (RA, diagnostic) |
#   the Phase-1 probe (reused, never refit).
# Train fits, validation selects, test is read once. Main operating point = the frozen Phase-1
# sustained 5% tunnel entrance.
#
# ---------------------------------------------------------------------------------------
# HOW TO USE (read top to bottom the first time)
# ---------------------------------------------------------------------------------------
#   0. LOGIN NODE (seconds, 2 threads -- model-free contracts only):
#        export OMP_NUM_THREADS=2
#        python -m tests.test_phase2_h3        # 25 contracts, ~20 s
#        python -m tests.test_phase2_h4        # 28 contracts on tiny random models, ~15 s
#        python -m experiments.run_phase2_h3 --plan    # which Phase-1 cells are ready
#
#   1. SMOKE (one dataset, 4 depths incl. the Phase-1 entrance, the REAL 300 epochs; writes to
#      results/three_model_phase2_smoke/ so it can never mix with the real run):
#        sbatch --time=00:45:00 -J p2h3-smoke job_phase2_h3.sh --smoke
#      Read the log: every cell prints the native gate (must pass), the ladder at the entrance,
#      and per-depth timings. Inspect verification.json of each smoke cell.
#      SIZE THE FULL RUN FROM IT: full-run minutes per model ~= the smoke cell's
#      summary.json timings.cell_total_s / 60 x 50 (Chronos-2, TiRex: 150 fits x 14 datasets vs
#      42 fits x 1) or x 70 (TimesFM-3: 20 depths vs 4). `seff <jobid>` shows the real wall time.
#
#   2. FULL, one model per job (independent; they may run AT THE SAME TIME -- each job cleans and
#      logs only its own cells). If the smoke predicts > ~2.5 h for a model, split it in halves:
#        sbatch -J p2h3-c2-a job_phase2_h3.sh --models chronos2 --datasets <first 7 tags>
#        sbatch -J p2h3-c2-b job_phase2_h3.sh --models chronos2 --datasets <last 7 tags>
#        sbatch -J p2h3-chronos2 job_phase2_h3.sh --models chronos2
#        sbatch -J p2h3-timesfm3 job_phase2_h3.sh --models timesfm3
#        sbatch -J p2h3-tirex    job_phase2_h3.sh --models tirex
#      LOOP Seattle's Phase-1 cells are read from the lr=1e-3 rerun tree
#      (results/three_model_phase1_q9_loop_lowlr) AUTOMATICALLY (probing.phase2.PHASE1_REROUTES);
#      the superseded lr=1e-2 LOOP cells are never read. --plan prints the root used per cell.
#      --phase1-override TAG=DIR still wins over the reroute if you ever need another tree.
#
#   3. RESUME: resubmit the identical line. 4. TABLES (login node, seconds):
#        python -m experiments.make_phase2_tables && python -m experiments.make_phase2_paper_tables
#
# WHAT THE SBATCH FLAGS MEAN: --gres=gpu:1 asks for one GPU; --cpus-per-task caps the CPU cores
# (OMP_NUM_THREADS below is set to match so numpy never grabs the whole node); --mem is host RAM;
# --time is the wall-clock limit after which SLURM kills the job (a killed cell leaves only a
# .building-* staging dir, which the next run deletes -- never a half-written cell).
# =======================================================================================

mkdir -p results/three_model_phase2/logs

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=${HF_HOME:-$SCRATCH/chronos2/hf_cache}
export HF_HUB_OFFLINE=1                    # compute nodes are offline; checkpoints pre-cached
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8     # deterministic cuBLAS (the adapter fits are full-batch)
export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$SCRATCH/tirex/torch_extensions}

# The Phase-1 caches (READ-ONLY here) and where the adapters go (persistent, not purged).
export PHASE1_CACHE=${PHASE1_CACHE:-$SCRATCH/chronos2/phase1_shared_cache}
export PHASE1_WINDOWS=${PHASE1_WINDOWS:-$SCRATCH/chronos2/phase1_shared_windows}

echo "=================================================================================="
echo "PHASE 2 / H3  host=$(hostname)  job=${SLURM_JOB_ID:-none}  started $(date -Is)"
echo "  phase-1 caches (read-only) : $PHASE1_CACHE"
echo "  phase-1 windows            : $PHASE1_WINDOWS"
echo "  adapters                   : ${PROJECT:-$SCRATCH}/chronos2_phase2/"
echo "  extra args                 : $*"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "=================================================================================="

python -m experiments.run_phase2_h3 --device cuda "$@"
