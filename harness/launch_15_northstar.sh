#!/usr/bin/env bash
set -euo pipefail

# Launch 15 Northstar trajectories: 5 per task (T2, T3, T4).
# Companion to launch_60_openai.sh for the cross-model pass-rate matrix.
#
# Each trial creates a fresh Kernel browser, runs to completion, scores, and
# saves to outputs/jobs/{task}_northstar/trial_NNN/.
#
# Resume: pass start trial as second arg.
#
# Usage:
#   cd harness
#   bash launch_15_northstar.sh           # full 15
#   bash launch_15_northstar.sh T2        # just T2 (5 trials)
#   bash launch_15_northstar.sh T3 3      # just T3, starting at trial 3

cd "$(dirname "$0")"

TASKS="${1:-T2 T3 T4}"
START="${2:-1}"
RUNS=5

for task in $TASKS; do
  echo ""
  echo "============================================================"
  echo " Starting $task — $RUNS trials (Northstar tzafon.northstar-cua-fast)"
  echo "============================================================"

  PYTHONUNBUFFERED=1 uv run python -u run_eval.py \
    --task "$task" \
    --model northstar \
    --runs "$RUNS" \
    --start-trial "$START"

  task_lower=$(echo "$task" | tr '[:upper:]' '[:lower:]')
  echo ""
  echo "$task done. Results in outputs/jobs/${task_lower}_northstar/"
  echo ""
done

echo ""
echo "============================================================"
echo " ALL DONE — 15 Northstar trajectories produced"
echo "============================================================"
echo ""
echo "Next: convert to Harbor format and upload."
echo "See HANDOFF_2.md → 'Step 4 — convert each job to Harbor format and upload'."
