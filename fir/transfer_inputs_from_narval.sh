#!/bin/bash
# =======================================================================================
# Copy the READ-ONLY inputs of Phases 2c / 2d / 2e / 2f from Narval $SCRATCH to Fir $SCRATCH,
# keeping the SAME paths under /scratch/$USER, so no job argument or default changes.
# Runs on the user's MAC: both clusters need MFA, so the stream goes Narval -> Mac -> Fir through
# existing SSH master connections (nothing is stored on the Mac).
#
#   bash fir/transfer_inputs_from_narval.sh core     # 2c / 2e / 2f inputs: 229 files, ~7.8 GB
#   bash fir/transfer_inputs_from_narval.sh p2d      # extra 2d inputs:   1382 files, ~6.1 GB
#
# WHAT (exactly the files the jobs read, nothing else):
#   core  chronos2/phase1_shared_windows/*                  (window caches, digest-checked)
#         chronos2_phase2/adapters/chronos2/*/noa__*.npz     (H3 adapters, sha256 vs H3 records)
#         chronos2/hf_cache/hub/models--amazon--chronos-2     (checkpoint, offline in jobs)
#         chronos2/features_cache/IDF_<tag>__{paper14_rolling|ood}__<split>__clean__K4_H64.npz
#             = the 42 files the committed H3 cells record as their inputs (summary.json
#               "feature_files"); the un-suffixed duplicates are not copied
#   p2d   chronos2/phase1_shared_cache/*  (TimesFM-3 / TiRex Phase-1 caches), chronos2/ood_targets,
#         the TimesFM-3 and TiRex checkpoints
# HOW: Narval `tar -T list | zstd -1` -> Mac (sha256 of the stream) -> Fir receiver
#      (sha256 of the stream | zstd -d | tar -x). On 2026-11-14 Fir login1 could not execute
#      binaries (SIGBUS), so the receiver ran as an `srun` step on a Fir compute node, through the
#      login1 SSH connection (~2 MB/s end to end). Set FIR_RECV="" to receive on a healthy login
#      node instead (FIR_SSH pointing at it).
# CHECKS: the stream sha256 must match on both ends; fir/job_fir_validate_cpu.sh then compares
#      every file's size with the Narval listing (.fir_transfer_<group>.sizes), and the drivers
#      validate content (window digests, cache labels, adapter sha256).
# =======================================================================================
set -euo pipefail
G=${1:?core|p2d}
U=eserebri
read -r -a NSSH <<< "${NARVAL_SSH:-ssh -o BatchMode=yes narval}"
read -r -a FSSH <<< "${FIR_SSH:-ssh -o BatchMode=yes fir}"
FIR_RECV=${FIR_RECV-srun --account=def-irina --time=03:00:00 --cpus-per-task=2 --mem=4000M --job-name=fir-xfer-$G}
LOG=${LOG_DIR:-.}/fir_transfer_$G

list_cmd() {
    case "$1" in
    core) cat <<'EOF'
cd /scratch/eserebri || exit 1
{
find chronos2/phase1_shared_windows -type f
find chronos2_phase2/adapters/chronos2 -type f -name 'noa__*.npz'
find chronos2/hf_cache/hub/models--amazon--chronos-2 \( -type f -o -type l \)
for t in m4_hourly monash_electricity_hourly uber_tlc_hourly wind_farms_hourly LOOP_SEATTLE_5T \
         electricity_15min SZ_TAXI_15T monash_london_smart_meters m5 wiki_daily_100k monash_traffic; do
  for s in train val test; do echo chronos2/features_cache/IDF_${t}__paper14_rolling__${s}__clean__K4_H64.npz; done; done
for t in sg_carpark coastal_ts boom_hourly; do
  for s in train val test; do echo chronos2/features_cache/IDF_${t}__ood__${s}__clean__K4_H64.npz; done; done
} | sort
EOF
    ;;
    p2d) cat <<'EOF'
cd /scratch/eserebri || exit 1
{
find chronos2/phase1_shared_cache \( -type f -o -type l \)
find chronos2/ood_targets \( -type f -o -type l \)
find chronos2/hf_cache/hub/models--google--timesfm-3.0-pytorch chronos2/hf_cache/hub/models--NX-AI--TiRex \( -type f -o -type l \)
} | sort
EOF
    ;;
    *) echo "unknown group $1" >&2; exit 2 ;;
    esac
}

"${NSSH[@]}" "bash -s" < <(list_cmd "$G") > "$LOG.list"
echo "[$(date +%T)] $G: $(wc -l < "$LOG.list") paths"
"${NSSH[@]}" "cat > /scratch/$U/.fir_transfer_$G.list && cd /scratch/$U && xargs -d '\n' stat -c '%s %n' < .fir_transfer_$G.list > .fir_transfer_$G.sizes" < "$LOG.list"
RECV="cd /scratch/$U && tee >(sha256sum | cut -d' ' -f1 > /scratch/$U/.fir_transfer_$G.sha256) | zstd -d -c | tar -xf -; echo RC=\${PIPESTATUS[*]}"
if [ -n "$FIR_RECV" ]; then RCMD="$FIR_RECV bash -c $(printf %q "$RECV")"; else RCMD="bash -c $(printf %q "$RECV")"; fi
"${NSSH[@]}" "cd /scratch/$U && nice -n 10 tar -cf - -T .fir_transfer_$G.list | nice -n 10 zstd -1 -T2 -c" \
  | tee >(shasum -a 256 | cut -d' ' -f1 > "$LOG.mac.sha256") \
  | "${FSSH[@]}" "$RCMD"
sleep 3
echo "[$(date +%T)] stream sha256 mac=$(cat "$LOG.mac.sha256")  (compare: /scratch/$U/.fir_transfer_$G.sha256 on Fir)"
