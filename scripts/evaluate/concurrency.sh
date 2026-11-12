#!/usr/bin/env bash
# Multi-session serving: scripts/evaluate/concurrency.sh <gpu>
# For N in 1 2 4, N replay processes of the Streaming SLM share the GPU (time-sliced, no cross-session batching),
# each on its own shard of FDB v3; the runs are conc_n<N>_slm_fdbv3. The processes are not CPU-pinned: they would
# share one core set.
set -eu
gpu=$1
export PIN=0
root=$(cd "$(dirname "$0")/../.." && pwd)
for n in 1 2 4; do
  pids=()
  for ((i = 0; i < n; i++)); do
    pids+=("$("$root/scripts/evaluate/run.sh" slm "$gpu" "$i/$n" "conc_n${n}_slm_fdbv3" "$root/config/eval/sessions/fdbv3.jsonl" ar,self_silence_h_b)")
  done
  echo "$(date -u +%FT%TZ) N=$n started: ${pids[*]}"
  for pid in "${pids[@]}"; do while kill -0 "$pid" 2>/dev/null; do sleep 30; done; done
  echo "$(date -u +%FT%TZ) N=$n done"
done
