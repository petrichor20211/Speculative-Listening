#!/usr/bin/env bash
# A queue lane: scripts/evaluate/queue.sh <jobs file> <gpu> <lane name>
# Job line: <target> <run> <nshards> <sessions.jsonl> <methods> [args...]   (# comments allowed)
# Takes the first (job, shard) neither done nor claimed (mkdir lock in the run's log directory), runs it with
# run.sh, marks it done when it exits 0 without a traceback (else moves its log aside as failed.*), and repeats
# until nothing is left. One lane per GPU. A lane killed mid-shard leaves its lock.* directory: remove it by hand.
set -u
jobs=$1; gpu=$2; lane=$3
root=$(cd "$(dirname "$0")/../.." && pwd)
outputs=$(python3 -c 'from speclisten.paths import path; print(path("outputs"))') || exit 1
while true; do
  claimed=0
  while read -r target run n sessions methods extra; do
    [ -z "${target:-}" ] && continue; [ "${target:0:1}" = "#" ] && continue
    log=${SPECLISTEN_LOGS:+$SPECLISTEN_LOGS/$run}; log=${log:-$outputs/$run/logs}; mkdir -p "$log"
    for ((g = 0; g < n; g++)); do
      [ -f "$log/done.${g}of$n" ] && continue
      [ "$(ls "$log"/failed.${g}of$n.* 2>/dev/null | wc -l)" -ge 2 ] && continue  # failed twice: leave it
      mkdir "$log/lock.${g}of$n" 2>/dev/null || continue
      claimed=1
      echo "$(date -u +%FT%TZ) gpu$gpu/$lane start $run $g/$n"
      rm -f "$log/status.${g}of$n"
      pid=$("$root/scripts/evaluate/run.sh" "$target" "$gpu" "$g/$n" "$run" "$root/$sessions" "$methods" $extra < /dev/null)
      while [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; do sleep 20; done
      if [ "$(cat "$log/status.${g}of$n" 2>/dev/null)" != 0 ] || grep -q Traceback "$log/run.${g}of$n.log"; then
        mv "$log/run.${g}of$n.log" "$log/failed.${g}of$n.$(date +%s).log" 2>/dev/null; rmdir "$log/lock.${g}of$n"
        echo "$(date -u +%FT%TZ) gpu$gpu/$lane FAILED $run $g/$n"
      else
        touch "$log/done.${g}of$n"; echo "$(date -u +%FT%TZ) gpu$gpu/$lane done $run $g/$n"
      fi
      break 2  # re-read the jobs file after every shard (it may have been edited)
    done
  done < "$jobs"
  [ "$claimed" = 0 ] && { echo "$(date -u +%FT%TZ) gpu$gpu/$lane idle"; break; }
done
