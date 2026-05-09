#!/usr/bin/env bash
set -euo pipefail

# Launch all 60 OpenAI trials in parallel.
# Concurrency capped at $MAX_PARALLEL (default 10) to stay within
# Kernel browser limits and OpenAI rate limits.
#
# Each trial is a fully independent process: own Kernel browser, own
# output dir, own API calls. No shared state.
#
# Usage:
#   cd harness
#   bash launch_parallel.sh          # 60 trials, 10 concurrent
#   MAX_PARALLEL=15 bash launch_parallel.sh   # 60 trials, 15 concurrent
#   bash launch_parallel.sh 5        # override concurrency

cd "$(dirname "$0")"

MAX_PARALLEL="${1:-${MAX_PARALLEL:-10}}"
TOTAL=0

# Build the list of (task, trial_num) pairs to run.
# Skip any trial that already has a summary.json (completed).
WORK_FILE=$(mktemp)
trap "rm -f $WORK_FILE" EXIT

for task in T2 T3 T4; do
  task_lower=$(echo "$task" | tr A-Z a-z)
  job_dir="../outputs/jobs/${task_lower}_openai"
  for i in $(seq 1 20); do
    trial_dir="${job_dir}/trial_$(printf '%03d' $i)"
    if [[ -f "${trial_dir}/summary.json" ]]; then
      echo "SKIP $task trial $i (already done)" >&2
      continue
    fi
    echo "${task} ${i}"
    TOTAL=$((TOTAL + 1))
  done
done > "$WORK_FILE"

echo "============================================================"
echo " Launching $TOTAL trials, $MAX_PARALLEL concurrent"
echo "============================================================"
echo ""

run_one() {
  local task="$1"
  local trial="$2"
  local task_lower=$(echo "$task" | tr A-Z a-z)
  local logfile="../outputs/jobs/${task_lower}_openai/trial_$(printf '%03d' $trial).log"
  mkdir -p "$(dirname "$logfile")"
  echo "[$(date +%H:%M:%S)] START  ${task} trial ${trial}"
  PYTHONUNBUFFERED=1 uv run python -u run_eval.py \
    --task "$task" \
    --model openai \
    --runs 1 \
    --start-trial "$trial" \
    > "$logfile" 2>&1
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "[$(date +%H:%M:%S)] DONE   ${task} trial ${trial} (exit 0)"
  else
    echo "[$(date +%H:%M:%S)] FAIL   ${task} trial ${trial} (exit $rc)"
  fi
}
export -f run_one

cat "$WORK_FILE" | xargs -P "$MAX_PARALLEL" -L1 bash -c 'run_one $0 $1'

echo ""
echo "============================================================"
echo " ALL $TOTAL TRIALS LAUNCHED AND COMPLETED"
echo "============================================================"

# Quick summary
for task in T2 T3 T4; do
  task_lower=$(echo "$task" | tr A-Z a-z)
  done_count=$(find "../outputs/jobs/${task_lower}_openai" -name summary.json 2>/dev/null | wc -l | tr -d ' ')
  echo "$task: $done_count / 20 completed"
done
