"""LoRA fine-tune for Northstar-CUA-Fast (4B) on collected (image, action) pairs.

This is a generic vision-language LoRA training scaffold. It assumes the model
is loadable via `transformers.AutoModelForCausalLM` (or `AutoModelForVision2Seq`)
and uses TRL's `SFTTrainer`. Northstar-specific details (chat template,
image-token wrapping, action-output schema) MAY need small adjustments — see
the TODO markers below. On an H100 80GB this fits comfortably with rank-64
LoRA at batch size 4.

Input data: JSONL produced by `scripts/prepare_sft_data.py`. Each line:
  {
    "instruction": "<task agent prompt>",
    "current_image_path": "/abs/path/to/step_NNN.png",
    "action": {"type": "click", "x": 607, "y": 157, ...},
    "model": "anthropic"|"openai",
    "task": "T2"|"T3"|"T4",
    "step": <int>
  }

Usage:
  python scripts/train_lora.py \
    --base-model Tzafon/Northstar-CUA-Fast \
    --data sft_data/sft.jsonl \
    --output checkpoints/northstar-cua-fast-sft \
    --lora-rank 64 --batch-size 4 --epochs 3 --bf16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from PIL import Image


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #

def _action_to_text(action: dict[str, Any]) -> str:
    """Render an action as a deterministic JSON string. The model learns to
    emit this exact format on the assistant turn. If Northstar uses a
    different action schema (e.g. xdotool-style strings), adjust here."""
    return json.dumps(action, sort_keys=True)


def _format_example(ex: dict[str, Any]) -> dict[str, Any]:
    """Build a (prompt, completion, image) triple. The chat template is what
    most VLM trainers expect; SFTTrainer can also take pre-tokenized inputs."""
    instruction = ex["instruction"]
    action_text = _action_to_text(ex["action"])
    img_path = ex["current_image_path"]

    # Conversation-style format. Adjust the system role / image tag if the
    # tokenizer expects something specific (e.g. <|image|>, <image>).
    messages = [
        {"role": "system", "content": "You are a computer-use agent. Given the current screenshot and task, emit the next browser action as a JSON object."},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction},
        ]},
        {"role": "assistant", "content": action_text},
    ]
    return {
        "messages": messages,
        "image_path": img_path,
        "completion": action_text,
    }


def _load_dataset(jsonl_path: Path) -> Dataset:
    rows = []
    with jsonl_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if not Path(ex.get("current_image_path", "")).exists():
                continue
            rows.append(_format_example(ex))
    if not rows:
        raise RuntimeError(f"No usable examples in {jsonl_path}")
    return Dataset.from_list(rows)


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #

@dataclass
class TrainConfig:
    base_model: str
    data: Path
    output: Path
    lora_rank: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    batch_size: int = 4
    grad_accum: int = 4
    epochs: int = 3
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    bf16: bool = True
    fp16: bool = False
    save_steps: int = 100
    logging_steps: int = 10
    max_seq_len: int = 4096
    seed: int = 42
    use_4bit: bool = False
    hf_token: str | None = None


def _load_model_and_tokenizer(cfg: TrainConfig):
    from transformers import (  # noqa: PLC0415
        AutoModelForCausalLM,
        AutoTokenizer,
        AutoProcessor,
        BitsAndBytesConfig,
    )

    quant_cfg = None
    if cfg.use_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if cfg.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    common = dict(
        token=cfg.hf_token or os.environ.get("HF_TOKEN"),
        trust_remote_code=True,
    )

    # Try a vision-aware processor first; fall back to text-only tokenizer.
    processor = None
    try:
        processor = AutoProcessor.from_pretrained(cfg.base_model, **common)
        tokenizer = processor.tokenizer
        print("==> loaded AutoProcessor (vision-aware)")
    except Exception as exc:
        print(f"==> AutoProcessor unavailable ({exc}); falling back to AutoTokenizer")
        tokenizer = AutoTokenizer.from_pretrained(cfg.base_model, **common)

    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        torch_dtype=torch.bfloat16 if cfg.bf16 else (torch.float16 if cfg.fp16 else "auto"),
        device_map="auto",
        quantization_config=quant_cfg,
        **common,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, processor


def _attach_lora(model, cfg: TrainConfig):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: PLC0415

    if cfg.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # Target the projection layers most VLMs share. Adjust if the model uses
    # different module names (`gate_proj`, `up_proj`, `down_proj`, etc.).
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    try:
        # Inspect modules; add MLP projections if present.
        names = {n for n, _ in model.named_modules()}
        for extra in ("gate_proj", "up_proj", "down_proj"):
            if any(n.endswith(extra) for n in names):
                target_modules.append(extra)
    except Exception:
        pass

    lora_cfg = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def _collator(processor, tokenizer, cfg: TrainConfig):
    """Build batches of (image, prompt, completion) → input_ids/labels.

    If `processor` is available (vision-aware), use it. Otherwise we fall back
    to a text-only path that ignores images (useful for sanity-checking the
    pipeline before you've debugged the multimodal tokenizer).
    """
    def _collate(examples: list[dict[str, Any]]):
        images = [Image.open(ex["image_path"]).convert("RGB") for ex in examples]
        # Use the chat template if available
        try:
            prompts = [
                tokenizer.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=False)
                for ex in examples
            ]
        except Exception:
            prompts = [
                f"Instruction: {ex['messages'][1]['content'][1]['text']}\nAction: {ex['completion']}"
                for ex in examples
            ]
        if processor is not None:
            batch = processor(images=images, text=prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len)
        else:
            batch = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=cfg.max_seq_len)
        batch["labels"] = batch["input_ids"].clone()
        return batch

    return _collate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", default="Tzafon/Northstar-CUA-Fast")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-seq-len", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    args = parser.parse_args()

    cfg = TrainConfig(**{k.replace("-", "_"): v for k, v in vars(args).items()})
    cfg.output.mkdir(parents=True, exist_ok=True)

    print(f"==> base model: {cfg.base_model}")
    print(f"==> data:       {cfg.data}")
    print(f"==> output:     {cfg.output}")
    print(f"==> lora rank:  {cfg.lora_rank} (alpha={cfg.lora_alpha})")
    print(f"==> batch size: {cfg.batch_size} x grad_accum={cfg.grad_accum} (effective {cfg.batch_size * cfg.grad_accum})")
    print(f"==> epochs:     {cfg.epochs}")
    print(f"==> dtype:      {'bf16' if cfg.bf16 else 'fp16' if cfg.fp16 else 'fp32'}")

    print("==> loading data...")
    ds = _load_dataset(cfg.data)
    print(f"    {len(ds)} examples")

    print("==> loading base model + tokenizer...")
    model, tokenizer, processor = _load_model_and_tokenizer(cfg)

    print("==> attaching LoRA...")
    model = _attach_lora(model, cfg)

    print("==> setting up trainer...")
    from transformers import TrainingArguments, Trainer  # noqa: PLC0415

    targs = TrainingArguments(
        output_dir=str(cfg.output),
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        num_train_epochs=cfg.epochs,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        bf16=cfg.bf16 and not cfg.fp16,
        fp16=cfg.fp16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        remove_unused_columns=False,
        report_to="none",
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds,
        data_collator=_collator(processor, tokenizer, cfg),
        tokenizer=tokenizer,
    )

    print("==> training...")
    trainer.train()

    print(f"==> saving LoRA adapter to {cfg.output}/adapter")
    model.save_pretrained(str(cfg.output / "adapter"))
    tokenizer.save_pretrained(str(cfg.output / "adapter"))
    print("==> done")


if __name__ == "__main__":
    main()
