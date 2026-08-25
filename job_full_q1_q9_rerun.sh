#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one A100: warm-cache linear probe fits are seconds each on GPU
#SBATCH --cpus-per-task=4            # transfer/bootstrap/figures are CPU; 4 keeps warm-cache reads snappy
#SBATCH --mem=16G                    # rolling windows + ~344 MB train fslot caches (train-recompute)
#SBATCH --time=9:00:00              # ~4-8 h of probe fits (q9+q1) with margin; RESUMABLE -> just resubmit
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# ================================================================================================= #
# FINAL q1/q9 rerun — the WHOLE forecasting story, LINEAR FSLOT PROBES ONLY, wide WD grid (protocol
# v2). ONE overnight job runs every warm-cache probe fit + stats + tunnels + figures for BOTH quantile
# configs across the three experimental blocks, then the q1-vs-q9 comparisons + CKA collection.
#
#   Block A  ext_v4_future_tokens : ID (diag) + cross-dataset (4x4) + unseen (4x3 frozen transfer)
#   Block B  ft_specialization    : DOMAIN change  (BOOM forecasting FT: pretrained/early/late)
#   Block C  task_shift_classif.  : TASK change    (FordA/UWave/Handwriting classification FT)
#
# NO MLP anywhere (every driver defaults --probe-family shared_linear; this script never passes
# native_mlp). Representation caches + FT checkpoints are REUSED — no backbone extraction, no FT rerun.
# Every command is IDEMPOTENT + protocol-version-gated: a legacy narrow-grid q9 result can never satisfy
# a v2 skip, and a completed v2 cell is skipped. So a TIMEOUT/OOM just needs `sbatch job_full_q1_q9_rerun.sh`
# again — it resumes. FAIL-LOUD: set -euo pipefail aborts on the first real error.
#
# Submit from the repo root:   sbatch job_full_q1_q9_rerun.sh
# ================================================================================================= #
set -euo pipefail

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                  # compute nodes are offline; model pre-cached
export FT_CKPT_ROOT=$SCRATCH/chronos2/ft_specialization  # BOOM + <src>_cls checkpoints
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets     # PT-OOD forecasting targets (BOOM/SG/Coastal)
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MPLBACKEND=Agg

QSETS=(q9 q1)
CLS_SOURCES=(forda uwave handwriting)

hdr() { printf '\n==================================================\n%s\n==================================================\n' "$1"; }
run() { echo "+ $*"; "$@"; }

# ------------------------------------------------------------------------------------------------- #
hdr "0. PREFLIGHT — provenance + caches + checkpoints + no-MLP + protocol"
echo "git commit : $(git rev-parse HEAD)"
echo "hostname   : $(hostname)"
echo "date       : $(date)"
echo "result root: results/{ext_v4_future_tokens,ft_specialization/domain_shift,task_shift_classification,comparisons}"
python - <<'PY'
from probing.probes import WD_GRID_V2, PROBE_PROTOCOL_VERSION, QUANTILE_SETS
print("protocol   :", PROBE_PROTOCOL_VERSION)
print("q configs  : q9 =", QUANTILE_SETS['q9'].tolist(), "| q1 =", QUANTILE_SETS['q1'].tolist())
print("WD grid    :", list(WD_GRID_V2))
import os, sys, pathlib
cache = pathlib.Path("features_cache")
req = []
# ext-v4 ID (4 PT-ID) + unseen (3 PT-OOD) fslot caches
for tag in ("monash_electricity_hourly","uber_tlc_hourly","m4_hourly","wind_farms_hourly"):
    for sp in ("train","val","test"):
        req.append(f"IDF_{tag}__extended_v3_rolling__{sp}__clean__K4_H64.npz")
for tag in ("sg_carpark","coastal_ts","boom_hourly"):
    for sp in ("train_rolling","val_rolling","test_rolling"):
        req.append(f"IDF_{tag}__ood__{sp}__clean__K4_H64.npz")
missing = [f for f in req if not (cache/f).exists()]
if missing:
    print("MISSING required fslot caches (extract prerequisite):", *missing[:8], "...", file=sys.stderr)
    sys.exit(1)
# FT-stage caches must exist (do NOT extract off the pretrained singleton for an FT stage)
def count(glob): return len(list(cache.glob(glob)))
for label, g, need in [("BOOM domain-FT","IDF_*__ft__boom__*K4_H64.npz",42),
                       ("forda_cls","IDF_*__ft__forda_cls__*K4_H64.npz",18),
                       ("uwave_cls","IDF_*__ft__uwave_cls__*K4_H64.npz",18),
                       ("handwriting_cls","IDF_*__ft__handwriting_cls__*K4_H64.npz",18)]:
    n = count(g); print(f"FT caches {label:16s}: {n}"); assert n>=need, f"{label}: {n} < {need}"
# FT checkpoints present
ck = pathlib.Path(os.environ["FT_CKPT_ROOT"])
for d in ("boom/stage1_ft_early","boom/stage2_ft_late","forda_cls","uwave_cls","handwriting_cls"):
    assert (ck/d).exists(), f"missing FT checkpoint {ck/d}"
print("preflight OK — all caches + checkpoints present; extraction NOT required")
PY

# ------------------------------------------------------------------------------------------------- #
for Q in "${QSETS[@]}"; do
  hdr "BLOCK A  ext_v4_future_tokens — ${Q}  (ID + cross-dataset + unseen)"
  run python -m experiments.run_ptood_probing_ftok --fit-ptid     --quantile-set "$Q"   # ID probes (GPU)
  run python -m experiments.run_ptood_probing_ftok --tunnels-only --quantile-set "$Q"   # ID tunnels (CPU)
  run python -m experiments.run_fslot_transfer --experiment transfer_4x4 --quantile-set "$Q"  # 4x4 (CPU)
  run python -m experiments.run_fslot_transfer --experiment pt_ood       --quantile-set "$Q"  # 4x3 (CPU)
done

for Q in "${QSETS[@]}"; do
  hdr "BLOCK B  ft_specialization / domain_shift — ${Q}  (BOOM forecasting FT)"
  run python -m experiments.run_ft_specialization --probe    --quantile-set "$Q"   # B2 fslot probes (GPU)
  run python -m experiments.run_ft_specialization --tunnels  --quantile-set "$Q"   # per-stage BOOM tunnels
  run python -m experiments.run_ft_specialization --figures  --quantile-set "$Q"   # layerwise curves + table
done

for SRC in "${CLS_SOURCES[@]}"; do
  for Q in "${QSETS[@]}"; do
    hdr "BLOCK C  task_shift / ${SRC} — ${Q}  (classification FT -> forecasting probes)"
    run python -m experiments.run_task_shift --cls-source "$SRC" --forecast-probe        --quantile-set "$Q"
    run python -m experiments.run_task_shift --cls-source "$SRC" --forecast-frozen-probe --quantile-set "$Q"
    run python -m experiments.run_task_shift --cls-source "$SRC" --figures               --quantile-set "$Q"
    run python -m experiments.run_task_shift --cls-source "$SRC" --frozen-figures        --quantile-set "$Q"
  done
done

hdr "15/16. FROZEN READOUT + TASK cross-source comparison (Q-flavoured tables already written above)"
run python -m experiments.run_task_shift --compare                         # cross-source DOMAIN-vs-TASK

hdr "17a. CKA (Q-independent) — regenerate + collect"
run python -m experiments.run_cka_analysis --all                           # results/cka/ (backbone reps)
run python -m experiments.run_q1q9_compare --cka                           # -> comparisons/cka/

hdr "17b. TRAIN-vs-VAL recompute (both q) + q1-vs-q9 comparison figures"
run python -m experiments.run_q1q9_compare --train-recompute               # <qset>/id/{figures,tables}
run python -m experiments.run_q1q9_compare --figures                       # comparisons/q1_vs_q9/

# ------------------------------------------------------------------------------------------------- #
hdr "18. FINAL VALIDATION — expected files + missing cells"
python - <<'PY'
import pathlib
R = pathlib.Path("results")
ev = R/"ext_v4_future_tokens"
miss = []
def want(p):
    p = pathlib.Path(p)
    (miss.append(str(p)) if not p.exists() else None)
    print(("  OK " if p.exists() else "  MISS ")+str(p))
for q in ("q9","q1"):
    want(ev/q/"tunnels")
    want(ev/q/"cross_dataset"/"figures")
    want(ev/q/"unseen"/"figures")
    want(ev/q/"id"/"figures"/"train_vs_val.png")
    want(R/"ft_specialization"/"domain_shift"/q/"figures")
    for src in ("forda","uwave","handwriting"):
        want(R/"task_shift_classification"/src/"forecasting"/q/"figures")
for f in ("relative_regret_shape.png","generalization_gap.png","selected_wd.png","tunnel_entrances.csv"):
    want(R/"comparisons"/"q1_vs_q9"/f)
want(R/"comparisons"/"cka")
print(f"\nMISSING {len(miss)} expected paths" if miss else "\nALL expected paths present")
print("=== q1/q9 rerun DONE ===")
PY
