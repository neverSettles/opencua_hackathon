#!/usr/bin/env bash
set -euo pipefail

# Launch 60 OpenAI trajectories: 20 per task (T2, T3, T4).
# Each trial creates a fresh Kernel browser, runs to completion, scores, and
# saves to outputs/jobs/{task}_openai/trial_NNN/.
#
# Runs sequentially (one trial at a time) to stay within Kernel concurrency.
# Expect ~2-10 min per trial, ~5-8 hours total.
#
# Resume: if interrupted, check outputs/jobs/{task}_openai/ for existing
# trial dirs and pass --start-trial N+1 to skip already-done trials.
#
# Usage:
#   cd harness
#   bash launch_60_openai.sh           # full 60
#   bash launch_60_openai.sh T2        # just T2 (20 trials)
#   bash launch_60_openai.sh T3 11     # just T3, starting at trial 11

cd "$(dirname "$0")"

TASKS="${1:-T2 T3 T4}"
START="${2:-1}"
RUNS=20

for task in $TASKS; do
  echo ""
  echo "============================================================"
  echo " Starting $task — $RUNS trials (OpenAI gpt-5.5)"
  echo "============================================================"

  PYTHONUNBUFFERED=1 uv run python -u run_eval.py \
    --task "$task" \
    --model openai \
    --runs "$RUNS" \
    --start-trial "$START"

  echo ""
  echo "$task done. Results in outputs/jobs/${task,,}_openai/"
  echo ""
done

echo ""
echo "============================================================"
echo " ALL DONE — 60 trajectories produced"
echo "============================================================"
echo ""
echo "Next: convert to Harbor format and upload:"
echo "  uv run python -m harness.to_harbor build ..."
echo "  npx --yes trajectories-sh upload trajectory ..."
