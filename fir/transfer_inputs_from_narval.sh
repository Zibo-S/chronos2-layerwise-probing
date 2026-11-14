#!/bin/bash
# =======================================================================================
# Copy the READ-ONLY inputs of Phases 2c / 2d / 2e / 2f from Narval $SCRATCH to Fir $SCRATCH,
# keeping the SAME paths under /scratch/$USER, so no job argument or default changes.
# Runs on the user's MAC: both clusters need MFA, so the tar stream goes Narval -> Mac -> Fir
# through two existing SSH master connections (nothing is stored on the Mac).
#
#   bash fir/transfer_inputs_from_narval.sh core     # 2c / 2e / 2f inputs (~12.5 GB)
#   bash fir/transfer_inputs_from_narval.sh p2d      # extra 2d inputs     (~5.8 GB)
#   bash fir/transfer_inputs_from_narval.sh verify   # sizes of every file, both sides
#
# Integrity: the tar stream's sha256 is computed on the Mac AND on Fir (must match), and the
# per-file size listing of both sides is compared. The drivers then validate their inputs again
# (window digests, cache labels, adapter sha256 against the committed H3 records).
# Env: NARVAL_SSH / FIR_SSH = the ssh command for each side (defaults below).
# =======================================================================================
set -euo pipefail
GROUP=${1:?core|p2d|datasets|verify}
read -r -a NSSH <<< "${NARVAL_SSH:-ssh -o BatchMode=yes narval}"
read -r -a FSSH <<< "${FIR_SSH:-ssh -o BatchMode=yes -o ControlPath=~/.ssh/cm-%r@%h:%p eserebri@login2.fir.alliancecan.ca}"
U=eserebri
LOG=${LOG_DIR:-.}/fir_transfer_${GROUP}_$(date +%Y%m%d_%H%M%S)
TAGS="m4_hourly monash_electricity_hourly uber_tlc_hourly wind_farms_hourly sg_carpark coastal_ts
      boom_hourly LOOP_SEATTLE_5T electricity_15min SZ_TAXI_15T monash_london_smart_meters m5
      wiki_daily_100k monash_traffic"

# the file list, built ON NARVAL (relative to /scratch/$U); only existing paths are listed
list_cmd() {
    case "$1" in
    core) cat <<EOF
cd /scratch/$U
for d in chronos2/phase1_shared_windows chronos2_phase2/adapters/chronos2 \
         chronos2/hf_cache/hub/models--amazon--chronos-2; do find "\$d" \( -type f -o -type l \); done
for t in $(echo $TAGS); do for s in train val test; do
  case \$t in sg_carpark|coastal_ts|boom_hourly) pre="IDF_\${t}__ood";; *) pre="IDF_\${t}__paper14_rolling IDF_\${t}";; esac
  for p in \$pre; do f=chronos2/features_cache/\${p}__\${s}__clean__K4_H64.npz; [ -f "\$f" ] && echo "\$f"; done
done; done
EOF
    ;;
    p2d) cat <<EOF
cd /scratch/$U
for d in chronos2/phase1_shared_cache chronos2/ood_targets \
         chronos2/hf_cache/hub/models--google--timesfm-3.0-pytorch \
         chronos2/hf_cache/hub/models--NX-AI--TiRex \
         chronos2/hf_cache/hub/models--autogluon--chronos-2-small; do find "\$d" \( -type f -o -type l \); done
EOF
    ;;
    datasets) cat <<EOF
cd /scratch/$U
for d in chronos2/hf_cache/datasets chronos2/hf_cache/hub/datasets--Datadog--BOOM \
         chronos2/hf_cache/hub/datasets--autogluon--chronos_datasets \
         chronos2/hf_cache/hub/datasets--autogluon--fev_datasets; do find "\$d" \( -type f -o -type l \); done
EOF
    ;;
    esac
}

if [ "$GROUP" = verify ]; then
    for g in core p2d; do
        "${NSSH[@]}" "bash -s" < <(list_cmd $g) | sort > "$LOG.$g.list"
        "${NSSH[@]}" "cd /scratch/$U && xargs -d '\n' stat -c '%s %n'" < "$LOG.$g.list" | sort > "$LOG.$g.narval"
        "${FSSH[@]}" "cd /scratch/$U && xargs -d '\n' stat -c '%s %n'" < "$LOG.$g.list" | sort > "$LOG.$g.fir"
        if diff -q "$LOG.$g.narval" "$LOG.$g.fir" >/dev/null; then
            echo "$g: $(wc -l < "$LOG.$g.list") files, sizes IDENTICAL on Narval and Fir"
        else
            echo "$g: DIFFERENCES"; diff "$LOG.$g.narval" "$LOG.$g.fir" | head -20
        fi
    done
    exit 0
fi

"${NSSH[@]}" "bash -s" < <(list_cmd "$GROUP") | sort > "$LOG.list"
N=$(wc -l < "$LOG.list")
echo "[$(date +%T)] $GROUP: $N paths"
[ "$N" -gt 0 ] || { echo "empty list"; exit 1; }
"${NSSH[@]}" "cat > /scratch/$U/.fir_transfer_$GROUP.list" < "$LOG.list"
# Narval: tar from the list (niced, I/O only). Mac: hash the stream. Fir: hash + extract.
"${NSSH[@]}" "cd /scratch/$U && nice -n 10 tar -cf - -T /scratch/$U/.fir_transfer_$GROUP.list" \
  | tee >(shasum -a 256 | cut -d' ' -f1 > "$LOG.mac.sha256") \
  | "${FSSH[@]}" "mkdir -p /scratch/$U && cd /scratch/$U && tee >(sha256sum | cut -d' ' -f1 > /scratch/$U/.fir_transfer_$GROUP.sha256) | tar -xf -"
sleep 2
MAC=$(cat "$LOG.mac.sha256"); FIRSHA=$("${FSSH[@]}" "cat /scratch/$U/.fir_transfer_$GROUP.sha256")
echo "[$(date +%T)] stream sha256 mac=$MAC fir=$FIRSHA"
[ -n "$MAC" ] && [ "$MAC" = "$FIRSHA" ] && echo "STREAM VERIFIED" || { echo "STREAM MISMATCH"; exit 1; }
