#!/usr/bin/env bash
# Bootstrap an H100 (Brev / NVIDIA AI Workbench / any Ubuntu+CUDA box) for
# Northstar 4B SFT.
#
# Usage (paste this whole line inside your `brev shell above-blue-gibbon`):
#
#   curl -fsSL https://raw.githubusercontent.com/neverSettles/opencua_hackathon/opus-trace-collection/scripts/setup_h100.sh | bash
#
# Or, if you cloned manually:
#   bash scripts/setup_h100.sh
#
# What it does:
#   1. nvidia-smi sanity check
#   2. apt deps (python3.12-venv, git, tmux)
#   3. clone the repo (idempotent)
#   4. python venv + pip install (torch, transformers, peft, trl, datasets,
#      accelerate, bitsandbytes, pillow)
#   5. echo next steps for data + training

set -euo pipefail

REPO_URL="https://github.com/neverSettles/opencua_hackathon.git"
BRANCH="opus-trace-collection"
REPO_DIR="$HOME/opencua_hackathon"

echo "==> [1/5] nvidia-smi sanity check"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "    ERROR: nvidia-smi not found. Are you on a GPU instance?" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits

echo
echo "==> [2/5] apt deps"
sudo apt-get update -qq
sudo apt-get install -y python3.12-venv python3.12-dev python3-pip git tmux htop curl

echo
echo "==> [3/5] clone repo (branch: $BRANCH)"
if [[ -d "$REPO_DIR/.git" ]]; then
  cd "$REPO_DIR"
  git fetch origin "$BRANCH"
  git checkout "$BRANCH"
  git pull --ff-only origin "$BRANCH"
else
  git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
  cd "$REPO_DIR"
fi

echo
echo "==> [4/5] python venv + deps (this takes ~3-5 min)"
if [[ ! -d .venv ]]; then
  python3.12 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip wheel --quiet
# CUDA-enabled torch first (auto-detects local CUDA), then the rest
pip install --quiet \
  "torch>=2.4" \
  "transformers>=4.46" \
  "peft>=0.13" \
  "trl>=0.12" \
  "datasets>=3.0" \
  "accelerate>=1.0" \
  "bitsandbytes>=0.44" \
  pillow \
  python-dotenv

echo "    torch: $(python -c 'import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))')"

echo
echo "==> [5/5] done. next steps:"
cat <<'NEXT'
  # 1. From your laptop, push the trajectory tarball:
  #      tar czf /tmp/sft_data.tar.gz outputs/
  #      brev cp /tmp/sft_data.tar.gz above-blue-gibbon:~/opencua_hackathon/

  # 2. On the H100 (here):
  cd ~/opencua_hackathon
  source .venv/bin/activate
  tar xzf sft_data.tar.gz
  python scripts/prepare_sft_data.py outputs/ --out sft_data/sft.jsonl

  # 3. Kick off the LoRA fine-tune:
  python scripts/train_lora.py \
    --base-model Tzafon/Northstar-CUA-Fast \
    --data sft_data/sft.jsonl \
    --output checkpoints/northstar-cua-fast-sft \
    --lora-rank 64 --batch-size 4 --epochs 3 --bf16

  # 4. Stop the Brev instance after training finishes (it still bills $2.28/hr).
NEXT
