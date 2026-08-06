#!/usr/bin/env bash
# Watch fresh-open -> Load Batch -> batch-running for the NEXT phase.
# Freshness-gated: only trusts AHK/PI-1 logs recreated AFTER this monitor starts,
# so a stale log from the killed run cannot trigger false positives.
EMC=/mnt/c/Users/muthusamy/Desktop/design/H-shape.emc
PILOG="$EMC/PI-1/log.txt"
find_ahk(){ ls -t /mnt/c/Users/muthusamy/AppData/Local/Temp/*/ecadstar_piemi_batch.log \
                  /mnt/c/Users/muthusamy/AppData/Local/Temp/ecadstar_piemi_batch.log 2>/dev/null | head -1; }
mt(){ stat -c %Y "$1" 2>/dev/null || echo 0; }
sa=0 so=0 sl=0 sr=0 srun=0
start=$(date +%s); maxsec=${1:-5400}
echo "MONITOR v2 armed $(date '+%H:%M:%S') start_epoch=$start — launch the pipeline now (RDP visible)…"
while :; do
  now=$(date +%s); el=$((now-start))
  AHK=$(find_ahk)
  if [ -n "$AHK" ] && [ "$(mt "$AHK")" -ge "$start" ]; then
    [ $sa -eq 0 ] && { echo "STAGE 1/5: AHK automation started (fresh log $AHK)"; sa=1; }
    if [ $so -eq 0 ] && grep -qiE "window appeared|already-open|open .erf via" "$AHK"; then echo "STAGE 2/5: analysis window opening (fresh)"; so=1; fi
    if [ $sl -eq 0 ] && grep -qi "Load Batch submitted" "$AHK"; then echo "STAGE 3/5: Load Batch submitted"; sl=1; fi
  fi
  if [ -f "$PILOG" ] && [ "$(mt "$PILOG")" -ge "$start" ] && [ $sr -eq 0 ] && grep -qi "read successfully" "$PILOG"; then
    echo "STAGE 4/5: PEB read successfully — batch accepted"; sr=1; fi
  cnt=$(ls -d $EMC/PI-* 2>/dev/null | wc -l)
  if [ "$cnt" -gt 0 ]; then
    [ $srun -eq 0 ] && { echo "STAGE 5/5: BATCH RUNNING — PI folders appearing"; srun=1; }
    echo "PROGRESS: PI-* = $cnt  (elapsed ${el}s)"
    [ "$cnt" -ge 50 ] && { echo "DONE: end-to-end CONFIRMED (>=50 PI outputs). kill->open->LoadBatch->run all fired."; exit 0; }
  fi
  [ $el -ge $maxsec ] && { echo "TIMEOUT ${el}s — batch not confirmed. last PI count=$cnt (GUI open/LoadBatch may have failed; check the window / AHK log)"; exit 1; }
  sleep 8
done
