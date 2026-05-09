# H100 fine-tune setup — Northstar 4B SFT on collected trajectories

Brev instance: **above-blue-gibbon** (1× H100 80GB, Hyperstack Montreal, $2.28/hr)

---

## 1. SSH in (from your laptop)

```bash
brev shell above-blue-gibbon
```

If `brev` isn't installed yet:
```bash
brew install brevdev/homebrew-brev/brev
brev login
brev shell above-blue-gibbon
```

## 2. Share with teammate

On brev.nvidia.com → click instance → **Share SSH Access** section → **Add Users** → enter teammate's email.
They sign up at brev.nvidia.com (free), then run the same `brev shell above-blue-gibbon` from their machine.

## 3. First-time setup on the H100 (run once you're SSH'd in)

```bash
# Verify GPU
nvidia-smi
# Expect: 1× NVIDIA H100 80GB

# System deps
sudo apt update
sudo apt install -y python3.12-venv git tmux htop

# Clone the repo
git clone https://github.com/neverSettles/opencua_hackathon.git
cd opencua_hackathon
git checkout opus-trace-collection   # the branch with the new harness + adapters

# Python env
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip wheel
pip install torch transformers peft trl datasets accelerate bitsandbytes pillow
```

## 4. Get the trajectory data onto the box

From your **laptop** (one-time, after the local Opus/GPT runs finish):
```bash
# from the repo root on your laptop:
tar czf /tmp/sft_data.tar.gz outputs/
# upload to the H100. Brev makes this easy:
brev cp /tmp/sft_data.tar.gz above-blue-gibbon:~/opencua_hackathon/
```

On the **H100**:
```bash
cd ~/opencua_hackathon
tar xzf sft_data.tar.gz
.venv/bin/python scripts/prepare_sft_data.py outputs/ --out sft_data/sft.jsonl
# Confirms: "Wrote NNN (image, action) examples"
```

## 5. Fine-tune Northstar-CUA-Fast 4B with LoRA

`scripts/prepare_sft_data.py` writes per-step `(instruction, current_image_path, action)` tuples in JSONL. Adapt these to whatever loader Tzafon's training pipeline expects:

- Model on HF: `Tzafon/Northstar-CUA-Fast`
- Recommended config (per teammate's screenshots):
  - LoRA rank 64
  - fp16 base + AdamW
  - batch size 4-8
  - 3 epochs
  - ~25-30 GB VRAM, ~15-30 min per teammate's table

A skeleton training command (you'll need to fit `model_id` + chat template):

```bash
# After loading the data into a HF dataset object:
accelerate launch --config_file accelerate_h100.yaml \
  scripts/train_lora.py \
  --base-model Tzafon/Northstar-CUA-Fast \
  --data sft_data/sft.jsonl \
  --output ./checkpoints/northstar-cua-fast-sft \
  --lora-rank 64 \
  --batch-size 4 \
  --epochs 3 \
  --learning-rate 2e-4 \
  --bf16
```

(Teammate has the exact training script; ask them or adapt their reference.)

## 6. Quick monitoring while training

```bash
nvidia-smi -l 5         # GPU util every 5s
tmux                    # stays alive if SSH drops
htop                    # CPU/RAM
```

---

## Reference: what's in this repo for the SFT pipeline

| Path | Purpose |
|------|---------|
| `tasks/T2_*.md`, `T3_*.md`, `T4_*.md` | Task specs (agent prompts in fenced blocks) |
| `ground_truth/T<N>_ground_truth.json` | What the scorer compares against |
| `scripts/scoring.py` | `score_t2 / score_t3 / score_t4` |
| `harness/run_t<N>_spike.py` | Multi-model trace collectors (`--model anthropic|openai|northstar|gemini`) |
| `harness/adapters/anthropic_direct.py` | Opus 4.7 / Sonnet 4.6 with `computer_20251124` |
| `harness/adapters/openai_cua.py` | GPT-5.5 with the GA `computer` tool |
| `scripts/prepare_sft_data.py` | Trajectory.jsonl → SFT (image, action) pairs |
| `scripts/collect_traces.sh` | Batch driver for many rollouts |
| `outputs/<run_id>/<world>/<task>/` | Per-rollout artifacts: trajectory, screenshots, agent_response, score |

## Stop the instance when done

Don't forget — at $2.28/hr, the instance keeps billing while it's "Running". Stop it from the Brev dashboard the moment training finishes.
