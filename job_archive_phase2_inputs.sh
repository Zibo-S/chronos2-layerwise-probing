#!/bin/bash
#SBATCH --account=def-irina          # the only account we have on Narval
#SBATCH --cpus-per-task=1            # tar + sha256 are I/O-bound; one core is enough
#SBATCH --mem=4G
#SBATCH --time=03:00:00              # ~15 GB of reads + writes on Lustre; usually well under 1 h
#SBATCH --output=results/three_model_phase2_lodo/logs/%x-%j.out
#SBATCH --error=results/three_model_phase2_lodo/logs/%x-%j.err

set -euo pipefail

# =======================================================================================
# ARCHIVE the Phase-1 / Phase-2 / Phase-2c inputs that cost GPU time to regenerate, before the
# $SCRATCH purge (files untouched for 60 days are deleted; ours become eligible 11-14 .. 11-24).
# /project/def-irina is OVER QUOTA (2026-11-11), so the destination is NEARLINE (tape-backed,
# 1000 GB quota, 5000-file limit -> a few large tar files, never loose files).
#
# Groups (ARCHIVE_GROUPS, default "fslot phase1 adapters"):
#   fslot     the exact 42 Chronos-2 forecast-slot caches the committed H3 cells read
#             (listed from their summary.json provenance), ~4.6 GB
#   phase1    $SCRATCH/chronos2/phase1_shared_cache (TimesFM-3 + TiRex) + phase1_shared_windows
#   adapters  $SCRATCH/chronos2_phase2/adapters (the committed H3 adapters, all three models)
#   lodo      $SCRATCH/chronos2_phase2/adapters_lodo (run AFTER Phase 2c: ARCHIVE_GROUPS=lodo)
# NOT archived (re-downloadable on a login node): hf_cache (models + raw datasets). NOT archived
# (superseded lines): timesfm3/, timesfm3_*, chronos2/features_cache beyond the 42 files above.
#
# HOW TO USE (from the repo root on Narval):
#   sbatch -J archive-p2 job_archive_phase2_inputs.sh
#   ... after the full Phase-2c run:
#   ARCHIVE_GROUPS=lodo sbatch -J archive-p2c job_archive_phase2_inputs.sh
#
# The tars are built in $SCRATCH/archive/ (fresh files, so not purge-eligible for 60 days), then
# copied to nearline and VERIFIED there with sha256sum -c. If this compute node cannot see
# /nearline, the job stops after building + checksumming and prints the copy command for a login
# node. Reading the originals updates their access time; that is a side effect of the backup,
# not a way around the purge policy -- the archive on nearline is the copy to rely on.
# =======================================================================================

STAMP=${ARCHIVE_STAMP:-$(date +%Y-%m-%d)}
GROUPS_=${ARCHIVE_GROUPS:-fslot phase1 adapters}
STAGE=${ARCHIVE_STAGE:-$SCRATCH/archive/phase2_inputs_$STAMP}
DEST=${ARCHIVE_DEST:-/nearline/def-irina/$USER/chronos2_archive/phase2_inputs_$STAMP}
mkdir -p "$STAGE" results/three_model_phase2_lodo/logs

echo "=================================================================================="
echo "ARCHIVE  host=$(hostname)  job=${SLURM_JOB_ID:-none}  $(date -Is)"
echo "  groups : $GROUPS_"
echo "  stage  : $STAGE"
echo "  dest   : $DEST"
echo "=================================================================================="

tar_group() {                      # name base [tar args...]
    local name=$1 base=$2
    shift 2
    echo "[tar] $name.tar  <-  $base  $*"
    tar -cf "$STAGE/$name.tar" -C "$base" "$@"
    tar -tf "$STAGE/$name.tar" > "$STAGE/$name.contents.txt"
    echo "      $(wc -l < "$STAGE/$name.contents.txt") entries, $(du -h "$STAGE/$name.tar" | cut -f1)"
}

for g in $GROUPS_; do
    case $g in
    fslot)
        # the EXACT files the committed H3 cells read (stdlib python, no venv needed)
        python3 - "$STAGE/chronos2_fslot_h3.list" <<'EOF'
import glob, json, os, sys
names = []
for f in sorted(glob.glob("results/three_model_phase2/h3/chronos2/*/summary.json")):
    for rec in json.load(open(f))["provenance"]["feature_files"].values():
        names.append(os.path.basename(rec["path"]))
if len(names) != 42 or len(set(names)) != 42:
    raise SystemExit(f"expected 42 distinct forecast-slot files (14 datasets x 3 splits), "
                     f"got {len(names)} ({len(set(names))} distinct)")
open(sys.argv[1], "w").write("\n".join(names) + "\n")
EOF
        tar_group chronos2_fslot_h3 "$SCRATCH/chronos2/features_cache" -T "$STAGE/chronos2_fslot_h3.list"
        [ "$(wc -l < "$STAGE/chronos2_fslot_h3.contents.txt")" -eq 42 ] \
            || { echo "FAIL: chronos2_fslot_h3.tar does not hold 42 files"; exit 1; } ;;
    phase1)
        tar_group phase1_shared "$SCRATCH/chronos2" phase1_shared_cache phase1_shared_windows ;;
    adapters)
        tar_group phase2_h3_adapters "$SCRATCH/chronos2_phase2" adapters ;;
    lodo)
        tar_group phase2c_lodo_adapters "$SCRATCH/chronos2_phase2" adapters_lodo ;;
    *)
        echo "unknown group $g"; exit 2 ;;
    esac
done

(cd "$STAGE" && sha256sum ./*.tar > SHA256SUMS && cat SHA256SUMS)

if mkdir -p "$DEST" 2>/dev/null && [ -w "$DEST" ]; then
    cp -v "$STAGE"/*.tar "$STAGE"/*.txt "$STAGE"/SHA256SUMS "$DEST"/
    [ -e "$STAGE/chronos2_fslot_h3.list" ] && cp "$STAGE/chronos2_fslot_h3.list" "$DEST"/
    (cd "$DEST" && sha256sum -c SHA256SUMS)
    echo "ARCHIVED + VERIFIED on nearline: $DEST"
else
    echo "nearline is NOT visible from $(hostname). The tars are built and checksummed in"
    echo "  $STAGE"
    echo "Copy them from a LOGIN node (a handful of large files):"
    echo "  mkdir -p $DEST && cp $STAGE/* $DEST/ && (cd $DEST && sha256sum -c SHA256SUMS)"
fi
