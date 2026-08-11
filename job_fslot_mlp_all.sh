#!/bin/bash
#SBATCH --account=def-irina          # the only account on Narval
#SBATCH --gres=gpu:1                 # one GPU (A100): only stage 3 (MLP head training) is GPU-heavy
#SBATCH --cpus-per-task=4            # rolling-window builds (SG/BOOM) + predict-only transfers
#SBATCH --mem=32G                    # OOD rolling-window builds need headroom; head training is modest
#SBATCH --time=6:00:00               # warm feature caches -> ~2-3h real; margin + resumable (see below)
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

set -e                               # stop at the first failing stage (later stages depend on earlier)

module load gcc python/3.11 arrow/24.0.0     # arrow supplies pyarrow (OOD-target loaders); load BEFORE venv
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache          # model+datasets pre-cached (compute nodes are offline)
export HF_HUB_OFFLINE=1
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets   # pre-staged SG/Coastal/BOOM arrow shards
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}       # keep CPU BLAS from oversubscribing

# ==========================================================================================
# Full v4 fslot pipeline for BOTH readout families, in dependency order.
#
# What is already on disk (so this job does NOT redo it):
#   * ALL fslot K4_H64 feature caches (4 PT-ID + 3 PT-OOD, every split) -> NO re-extraction; the MLP
#     fit is warm-cache and never loads Chronos-2 (cache HITs skip the model).
#   * LINEAR PT-ID checkpoints (168) + sustained tunnels (4)  -> stages 1-2 just transfer them.
#   * Native MEDIAN forecast caches (4 PT-ID) -> stage 6 MASE/MAE need no GPU native pass.
#
# Resumability: stage 3 (`--fit-ptid`) skips run seeds already written, so a timeout is safe —
# just `sbatch` this script again and it continues where it stopped (earlier predict-only stages
# re-run cheaply and overwrite identically). The only GPU-heavy work is stage 3 (2.92M-param heads).
# NOTE: native-Chronos-2 WQL is intentionally NOT computed here (it would need a fresh multi-quantile
# model pass); MASE/median-MAE/probe+naive WQL all come from warm caches. Add `--native-wql` later.
# ==========================================================================================

echo "=== 1/6  LINEAR 4x4 transfer  (frozen linear probes -> 4 PT-ID targets; predict-only, CPU) ==="
python -m experiments.run_fslot_transfer --probe-family shared_linear

echo "=== 2/6  LINEAR PT-OOD transfer  (Electricity -> SG/Coastal/BOOM; builds OOD rolling windows) ==="
python -m experiments.run_fslot_transfer --probe-family shared_linear --experiment pt_ood

echo "=== 3/6  MLP PT-ID probes  (native-structure head; 4 datasets x 3 seeds x 14 layers; GPU) ==="
# the only GPU-heavy stage. Writes fslot_mlp/ptid_checkpoints + per-seed training-history JSONs
# (inspect fslot_mlp/ptid_runs/*__history.json afterwards for convergence / WindFarms overfitting).
python -m experiments.run_ptood_probing_ftok --probe-family native_mlp --fit-ptid

echo "=== 4/6  MLP sustained-plateau tunnels  (from the 3-run mean validation curve; CPU) ==="
python -m experiments.run_ptood_probing_ftok --probe-family native_mlp --tunnels-only

echo "=== 5/6  MLP transfers  (4x4 + Electricity->PT-OOD; frozen MLP probes, predict-only) ==="
python -m experiments.run_fslot_transfer --probe-family native_mlp
python -m experiments.run_fslot_transfer --probe-family native_mlp --experiment pt_ood

echo "=== 6/6  ORIGINAL-scale forecasting comparison  (7 methods x 4 PT-ID datasets) ==="
# needs BOTH families' tunnels (linear on disk, MLP from stage 4) + native median (warm)
python -m experiments.run_fslot_forecasting_comparison

echo "=== DONE. Results under results/ext_v4_future_tokens/{fslot_transfer,fslot_pt_ood,fslot_mlp,forecasting_comparison}/ ==="
