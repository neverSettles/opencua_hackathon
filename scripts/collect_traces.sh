#!/usr/bin/env bash
# Batch-launch trajectory collection across {T2,T3,T4} x {opus,gpt55} for SFT.
#
# Usage:
#   ./scripts/collect_traces.sh                        # 1 rollout each (6 total)
#   ROLLOUTS=5 ./scripts/collect_traces.sh             # 5 rollouts each (30 total)
#   CONCURRENCY=4 ROLLOUTS=10 ./scripts/collect_traces.sh
#
# Each rollout writes to outputs/<task>_<model>_<timestamp>/. Logs in /tmp/.
# Concurrency cap defaults to 3 to respect Kernel + provider rate limits.

set -uo pipefail

cd "$(dirname "$0")/.."

ROLLOUTS=${ROLLOUTS:-1}
CONCURRENCY=${CONCURRENCY:-3}
LOG_DIR=${LOG_DIR:-/tmp/cua_traces}
mkdir -p "$LOG_DIR"

if [[ ! -f .env ]]; then
  echo "ERROR: .env missing"; exit 1
fi
set -a; source .env; set +a

TASKS=("t2:run_t2_spike.py:80" "t3:run_t3_spike.py:100" "t4:run_t4_spike.py:160")
MODELS=("anthropic:claude-opus-4-7" "openai:gpt-5.5")

wait_for_slot() {
  while [[ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$CONCURRENCY" ]]; do
    sleep 5
  done
}

LAUNCHED=0
for ((i=1; i<=ROLLOUTS; i++)); do
  for task_spec in "${TASKS[@]}"; do
    IFS=':' read -r task script max_steps <<< "$task_spec"
    for model_spec in "${MODELS[@]}"; do
      IFS=':' read -r model_kind model_id <<< "$model_spec"
      tag="${task}_${model_kind}_$(date -u +%Y%m%dT%H%M%SZ)_r${i}"
      log="$LOG_DIR/$tag.log"
      echo "==> [$tag] launching"
      wait_for_slot
      .venv/bin/python -u "harness/$script" \
        --model "$model_kind" \
        $( [[ "$model_kind" == "anthropic" ]] && echo "--anthropic-model" || echo "--openai-model" ) "$model_id" \
        --max-steps "$max_steps" \
        > "$log" 2>&1 &
      LAUNCHED=$((LAUNCHED+1))
      sleep 3   # stagger so Kernel browsers don't all spin up at once
    done
  done
done

echo "==> $LAUNCHED rollouts launched. Waiting for all to finish..."
wait
echo "==> All done. Logs in $LOG_DIR/, outputs in outputs/."
