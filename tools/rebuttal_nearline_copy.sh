#!/bin/bash
# =======================================================================================
# Copy ONE staged rebuttal archive ($SCRATCH/archive/rebuttal_<phase>_<job>/) to NEARLINE and
# VERIFY it there. Run on a LOGIN node: compute nodes cannot see /nearline. I/O-bound, low CPU
# (nice 19); a few GB per phase, a handful of large files (nearline's 5000-file limit).
#
#   bash tools/rebuttal_nearline_copy.sh $SCRATCH/archive/rebuttal_phase2c_lodo_<jobid>
#
# Steps: (1) re-verify the staged tars against SHA256SUMS (catches scratch corruption before the
# copy); (2) tar the FINAL logs + slurm provenance (the job's own log was still open when it built
# the archive) with their own checksum file; (3) copy everything to
# /nearline/def-irina/$USER/chronos2_archive/<name>/; (4) re-run sha256sum -c ON NEARLINE.
# Only after (4) passes is ARCHIVED_ON_NEARLINE.txt written into the stage directory. Nothing is
# deleted anywhere: the scratch stage and the working results stay as they are.
# =======================================================================================
set -euo pipefail
STAGE=${1:?usage: bash tools/rebuttal_nearline_copy.sh <stage-dir> [nearline-dest]}
STAGE=$(cd "$STAGE" && pwd -P)
NAME=$(basename "$STAGE")
DEST=${2:-/nearline/def-irina/$USER/chronos2_archive/$NAME}
[ -f "$STAGE/SHA256SUMS" ] || { echo "no SHA256SUMS in $STAGE"; exit 2; }
[ -d /nearline/def-irina ] || { echo "/nearline is not visible here: run this on a LOGIN node"; exit 2; }

echo "[1/4] verify the staged tars"
(cd "$STAGE" && nice -n 19 sha256sum -c SHA256SUMS)

echo "[2/4] final logs + provenance"
RES=$(grep '^results_dir=' "$STAGE/MANIFEST.txt" | cut -d= -f2- || true)
if [ -n "$RES" ] && [ -d "$RES" ]; then
    parts=()
    for d in logs slurm runs; do [ -d "$RES/$d" ] && parts+=("$d"); done
    if [ ${#parts[@]} -gt 0 ]; then
        nice -n 19 tar -cf "$STAGE/zz_logs_final.tar" -C "$RES" "${parts[@]}"
        (cd "$STAGE" && sha256sum ./zz_logs_final.tar > SHA256SUMS.logs)
    fi
fi

echo "[3/4] copy -> $DEST"
mkdir -p "$DEST"
nice -n 19 cp -v "$STAGE"/*.tar "$STAGE"/*.txt "$STAGE"/SHA256SUMS* "$DEST"/

echo "[4/4] verify ON NEARLINE"
(cd "$DEST" && nice -n 19 sha256sum -c SHA256SUMS && { [ ! -f SHA256SUMS.logs ] || nice -n 19 sha256sum -c SHA256SUMS.logs; })
echo "VERIFIED $(date -Is) $DEST" | tee "$STAGE/ARCHIVED_ON_NEARLINE.txt"
cp "$STAGE/ARCHIVED_ON_NEARLINE.txt" "$DEST/"
