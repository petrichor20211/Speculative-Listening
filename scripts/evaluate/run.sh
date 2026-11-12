#!/usr/bin/env bash
# One replay job in the background: scripts/evaluate/run.sh <target> <gpu> <shard i/n> <run> <sessions.jsonl> <methods> [args...]
#   target: slm | gander | venus | lws;  methods: comma-separated (docs/BENCH.md)
# Timing is end-to-end wall clock on the GPU: run one job per GPU. Writes <outputs>/<run>/<method>.<i>.jsonl and
# the log <outputs>/<run>/logs/run.<i>of<n>.log and its exit code to status.<i>of<n> (SPECLISTEN_LOGS=<dir> keeps logs
# elsewhere); prints the pid.
set -euo pipefail
target=$1; gpu=$2; shard=$3; run=$4; sessions=$5; methods=$6; shift 6
root=$(cd "$(dirname "$0")/../.." && pwd)
case $target in  # each runtime runs in its own interpreter with its repository on PYTHONPATH
  slm)    interpreter=python_duplex; repos=(duplex_repo) ;;
  gander) interpreter=python_venus;  repos=(gander_repo) ;;
  venus)  interpreter=python_venus;  repos=(venus_repo venus_model) ;;
  lws)    interpreter=python_lws;    repos=(lws_repo) ;;
  *) echo "unknown target $target" >&2; exit 2 ;;
esac
{ read -r outputs; read -r py; read -r extra; } < <(python3 -c 'import sys; from speclisten.paths import path
print(path("outputs"), path(sys.argv[1]), ":".join(str(path(k)) for k in sys.argv[2:]), sep="\n")' "$interpreter" "${repos[@]}")
[ -n "$outputs" ] && [ -n "$py" ] || exit 1
out=$outputs/$run; log=${SPECLISTEN_LOGS:+$SPECLISTEN_LOGS/$run}; log=${log:-$out/logs}
mkdir -p "$out" "$log"
# PYTHONSAFEPATH keeps the working directory off sys.path: the Streaming SLM's model code is the ``src`` package of
# duplex_repo, which this repository's src/ would shadow. The package itself is installed in every interpreter.
# Pin the process to its own CPU cores on the GPU's NUMA node (GPUs that share a node split its cores), so
# kernel launches of the eager runtimes are not slowed by jobs on other GPUs. PIN=0 turns it off.
pin=()
if [ "${PIN:-1}" = 1 ] && command -v numactl > /dev/null; then
  topo=$(nvidia-smi topo -m | sed 's/\x1b\[[0-9;]*m//g')
  aff=$(awk -v g="GPU$gpu" '$1 == g && $(NF-2) ~ /^[0-9]/ {print $(NF-2)}' <<< "$topo")
  node=$(awk -v g="GPU$gpu" '$1 == g && $(NF-2) ~ /^[0-9]/ {print $(NF-1)}' <<< "$topo")
  peers=($(awk -v a="$aff" '$1 ~ /^GPU[0-9]+$/ && $(NF-2) ~ /^[0-9]/ && $(NF-2) == a {print $1}' <<< "$topo"))
  for i in "${!peers[@]}"; do [ "${peers[$i]}" = "GPU$gpu" ] && idx=$i; done
  cpus=""
  for range in ${aff//,/ }; do
    lo=${range%-*}; hi=${range#*-}; part=$(( (hi - lo + 1) / ${#peers[@]} ))
    a=$(( lo + idx * part )); b=$(( idx == ${#peers[@]} - 1 ? hi : a + part - 1 ))
    cpus+="${cpus:+,}$a-$b"
  done
  [[ "$node" =~ ^[0-9]+$ ]] && pin=(numactl --physcpubind="$cpus" --membind="$node") || pin=(numactl --physcpubind="$cpus")
fi
echo "$(date -u +%FT%TZ) $target gpu=$gpu shard=$shard sessions=$sessions methods=$methods pin=${pin[*]:-none} $*" >> "$log/launcher.log"
cd "$root"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=$gpu DUPLEX_LOG_DIR=$log PYTHONPATH=$extra PYTHONSAFEPATH=1
setsid nohup bash -c '"$@"; echo $? > "$0"' "$log/status.${shard/\//of}" ${pin[@]+"${pin[@]}"} "$py" -m speclisten.bench.replay \
  --target "$target" --sessions "$sessions" --out "$out" --shard "$shard" --methods "$methods" "$@" \
  > "$log/run.${shard/\//of}.log" 2>&1 < /dev/null &
echo "$! $target $run $shard gpu$gpu" >> "$log/pids"
echo $!
