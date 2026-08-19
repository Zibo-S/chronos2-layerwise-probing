#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --gres=gpu:1                 # one A100: cls FT + all feature extraction + (GPU) probe fits
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G                    # proven sufficient for the FordA-cls run (forecast-extract dominates)
#SBATCH --time=4:00:00               # full C1..C5 for BOTH sources; RESUMABLE -> just resubmit on TIMEOUT
#SBATCH --output=logs/%x-%j.out      # %x = job name (sbatch -J), %j = job id

# TASK-SHIFT — FULL pipeline for uwave + handwriting in ONE job (C1 finetune -> C5 figures), then the
# cross-source DOMAIN-vs-TASK comparison. Everything is namespaced per source (source=<src>_cls,
# results/task_shift_classification/<src>/) and disjoint from FordA + the BOOM domain-shift run.
#
# RESUMABLE / IDEMPOTENT: extraction skips existing caches, probes skip existing JSONs, and C1 is skipped
# if a valid late stage already exists. If this TIMES OUT, just `sbatch job_task_shift_full.sh` again — it
# continues where it stopped. Submit from the repo root: `sbatch job_task_shift_full.sh`.
#
# VALIDITY-AWARE: after each C1, only the stages the FT actually produced (stage0 + stage1 [+ stage2 iff
# the validity gate emitted it]) are passed to C2..C5 via --stages, so a handwriting run whose gate
# refuses a late stage degrades to a stage0-vs-stage1 result instead of crashing load_stages.

set -o pipefail
cd "${SLURM_SUBMIT_DIR:-.}" || exit 1

module load gcc python/3.11 arrow/24.0.0
source .venv/bin/activate
export HF_HOME=$SCRATCH/chronos2/hf_cache
export HF_HUB_OFFLINE=1                                 # compute nodes are offline; model pre-cached
export FT_CKPT_ROOT=$SCRATCH/chronos2/ft_specialization # cls checkpoints -> $SCRATCH/.../<src>_cls/
export OOD_TARGET_ROOT=$SCRATCH/chronos2/ood_targets    # PT-OOD forecasting targets (BOOM/Coastal)

DRIVER="python -m experiments.run_task_shift"

# --- helpers that read the FT manifest (results/ft_specialization/<src>_cls/manifest.json) ------------ #
has_late () {     # "yes" iff a stage2_cls_late checkpoint is recorded (C1 already done + gate passed)
    python - "$1" <<'PY'
import json, os, sys
mf = f"results/ft_specialization/{sys.argv[1]}/manifest.json"
ok = os.path.exists(mf) and "stage2_cls_late" in json.load(open(mf)).get("checkpoints", {})
print("yes" if ok else "no")
PY
}

avail_stages () { # space-separated stages that actually exist: stage0 always, stage1/stage2 iff in manifest
    python - "$1" <<'PY'
import json, os, sys
mf = f"results/ft_specialization/{sys.argv[1]}/manifest.json"
stages = ["stage0_pretrained"]
if os.path.exists(mf):
    ck = json.load(open(mf)).get("checkpoints", {})
    stages += [s for s in ("stage1_cls_early", "stage2_cls_late") if s in ck]
print(" ".join(stages))
PY
}

print_validity () {
    python - "$1" <<'PY'
import json, os, sys
mf = f"results/ft_specialization/{sys.argv[1]}/manifest.json"
if os.path.exists(mf):
    v = json.load(open(mf))["validity"]
    print(f"[validity] {v['verdict']}")
    print(f"[validity] stage0_val_acc={v['stage0_val_acc']:.4f} stage1_val_acc={v['stage1_val_acc']:.4f} "
          f"stage2_val_acc={v['stage2_val_acc']}")
PY
}

# --- one source: C1 (finetune) -> C2 (extract) -> C3 (probe) -> C4 (forecast-probe) -> C5 (figures) --- #
run_source () {
    local src="$1"; local label="${src}_cls"
    echo; echo "##################### SOURCE = ${src} #####################"

    # C1 — classification fine-tuning (skip if a valid late stage already exists; deterministic seed 0)
    if [ "$(has_late "$label")" == "yes" ]; then
        echo "[C1/${src}] manifest already has stage2_cls_late -> skipping finetune"
    else
        echo "[C1/${src}] fine-tuning (per-source CLS_SPECS defaults)"
        python -m probing.finetune_cls --cls-source "$src" || { echo "!! [C1/${src}] FAILED"; return 1; }
    fi
    print_validity "$label"

    local STAGES; STAGES="$(avail_stages "$label")"
    echo "[stages/${src}] using: ${STAGES}"

    # C2 — Exp-A cls features (per-channel) + Exp-B fslot features (GPU)
    echo "[C2/${src}] extract cls + fslot features"
    $DRIVER --cls-source "$src" --stages $STAGES --extract --forecast-extract \
        || { echo "!! [C2/${src}] FAILED"; return 1; }

    # C3 — Exp-A layerwise classification probes (14 layers x seeds)
    echo "[C3/${src}] classification probes"
    $DRIVER --cls-source "$src" --stages $STAGES --probe \
        || { echo "!! [C3/${src}] FAILED"; return 1; }

    # C4 — Exp-B layerwise forecasting probes (fslot; warm caches)
    echo "[C4/${src}] forecasting probes"
    $DRIVER --cls-source "$src" --stages $STAGES --forecast-probe \
        || { echo "!! [C4/${src}] FAILED"; return 1; }

    # C5 — per-source Plots A/B/C + CKA (CPU; non-fatal so a plotting hiccup can't lose the probe JSONs)
    echo "[C5/${src}] figures + CKA"
    $DRIVER --cls-source "$src" --stages $STAGES --figures --cka \
        || echo "!! [C5/${src}] figures hit an error (probe JSONs are safe on disk)"

    echo "[done/${src}] stages=${STAGES}"
}

# ---------------------------------------------------------------------------------------------------- #
run_source uwave       || echo "!! uwave pipeline stopped early (see log above); continuing to handwriting"
run_source handwriting || echo "!! handwriting pipeline stopped early (see log above)"

# Cross-source DOMAIN-vs-TASK comparison (reads BOOM stageB + forda/uwave/handwriting forecast probes on
# disk; conditions/targets without data render as empty cells). Non-fatal.
echo; echo "##################### CROSS-SOURCE COMPARISON #####################"
$DRIVER --compare || echo "!! --compare hit an error"

echo; echo "===================== ALL DONE ====================="
