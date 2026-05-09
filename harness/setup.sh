#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not installed. Install with: brew install uv" >&2
  exit 1
fi

if [[ ! -d .venv ]]; then
  uv venv --python 3.12
fi

uv pip install -r requirements.txt
uv run playwright install chromium

echo "Setup complete. Activate with: source harness/.venv/bin/activate"
