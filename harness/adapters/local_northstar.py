"""Local-inference adapter for the fine-tuned Northstar-CUA-Fast 4B.

Loads the base Tzafon/Northstar-CUA-Fast model + a LoRA adapter via
transformers + PEFT, runs inference inline on a CUDA device, and returns
normalized Actions to the runner. No Tzafon API call.

Use this to evaluate a locally fine-tuned checkpoint on the same Kernel-driven
T2/T3/T4 flow that the API-based adapters use.

The model was SFT'd to emit a single JSON object describing the next browser
action, e.g. {"type": "click", "x": 607, "y": 157} — see
scripts/train_lora.py / scripts/prepare_sft_data.py.
"""
from __future__ import annotations

import io
import json
import os
import re
from typing import Any

import torch
from PIL import Image

from .base import Action, ActionType, AdapterStep, ModelAdapter


_DEFAULT_SYSTEM = (
    "You are a computer-use agent. Given the current screenshot and task, "
    "emit the next browser action as a JSON object."
)


def _png_bytes_to_pil(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png)).convert("RGB")


def _action_from_dict(d: dict[str, Any]) -> Action | None:
    """Convert the model's JSON output dict into a normalized Action."""
    if not isinstance(d, dict):
        return None
    t = (d.get("type") or "").lower()
    x = d.get("x")
    y = d.get("y")
    text = d.get("text")
    if x is not None:
        try:
            x = int(x)
        except Exception:
            x = None
    if y is not None:
        try:
            y = int(y)
        except Exception:
            y = None

    if t == "click":
        return Action(type=ActionType.CLICK, x=x, y=y)
    if t == "double_click":
        return Action(type=ActionType.DOUBLE_CLICK, x=x, y=y)
    if t == "right_click":
        return Action(type=ActionType.RIGHT_CLICK, x=x, y=y)
    if t == "type":
        return Action(type=ActionType.TYPE, text=text or "")
    if t in ("key_press", "keypress", "key"):
        keys = d.get("keys") or []
        if isinstance(keys, str):
            keys = [keys]
        return Action(type=ActionType.KEY_PRESS, keys=[str(k).lower() for k in keys])
    if t == "scroll":
        return Action(
            type=ActionType.SCROLL,
            x=x if x is not None else 640,
            y=y if y is not None else 400,
            delta_x=int(d.get("delta_x", 0) or 0),
            delta_y=int(d.get("delta_y", 0) or 0),
        )
    if t == "navigate":
        return Action(type=ActionType.NAVIGATE, url=d.get("url"))
    if t == "wait":
        return Action(type=ActionType.WAIT, seconds=float(d.get("seconds", 1.0)))
    if t in ("done", "terminate", "answer"):
        return Action(type=ActionType.DONE, final_text=d.get("final_text") or text)
    return None


def _try_parse_action_json(raw: str) -> dict[str, Any] | None:
    """Pull a JSON object out of the model's raw text output."""
    raw = (raw or "").strip()
    # Direct
    try:
        v = json.loads(raw)
        return v if isinstance(v, dict) else None
    except Exception:
        pass
    # Fenced
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.S)
    if m:
        try:
            v = json.loads(m.group(1))
            return v if isinstance(v, dict) else None
        except Exception:
            pass
    # First {...last}
    s = raw.find("{")
    e = raw.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            v = json.loads(raw[s : e + 1])
            return v if isinstance(v, dict) else None
        except Exception:
            return None
    return None


class LocalNorthstarAdapter(ModelAdapter):
    """Local-inference adapter for the fine-tuned Northstar 4B."""

    name = "local_northstar"

    def __init__(
        self,
        viewport_w: int = 1280,
        viewport_h: int = 800,
        base_model: str = "Tzafon/Northstar-CUA-Fast",
        adapter_path: str = "checkpoints/northstar-cua-fast-sft/adapter",
        device: str = "cuda",
        dtype: str = "bfloat16",
        max_new_tokens: int = 128,
        system_prompt: str | None = None,
        hf_token: str | None = None,
    ) -> None:
        from transformers import AutoModelForImageTextToText, AutoProcessor  # noqa: PLC0415
        from peft import PeftModel  # noqa: PLC0415

        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.model_id = f"local:{base_model}+{adapter_path}"
        self.max_new_tokens = max_new_tokens
        self.device = device
        torch_dtype = getattr(torch, dtype) if hasattr(torch, dtype) else torch.bfloat16
        token = hf_token or os.environ.get("HF_TOKEN")

        print(f"==> [local_northstar] loading base {base_model}")
        base = AutoModelForImageTextToText.from_pretrained(
            base_model,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
            token=token,
        )
        if adapter_path and os.path.exists(adapter_path):
            print(f"==> [local_northstar] attaching LoRA from {adapter_path}")
            self.model = PeftModel.from_pretrained(base, adapter_path)
        else:
            print(f"==> [local_northstar] no adapter at {adapter_path}; running base only")
            self.model = base
        self.model.eval()

        print(f"==> [local_northstar] loading processor")
        self.processor = AutoProcessor.from_pretrained(
            base_model, trust_remote_code=True, token=token
        )
        self._system = system_prompt or _DEFAULT_SYSTEM
        self._history: list[dict[str, Any]] = []

    # ---- public API ---------------------------------------------------------

    def first_step(self, instruction: str, screenshot_png: bytes, page_url: str) -> AdapterStep:
        self._history = []
        return self._step(instruction, screenshot_png)

    def next_step(
        self,
        executed: list[tuple[Action, str]],
        screenshot_png: bytes,
        page_url: str,
        nudge_text: str | None = None,
    ) -> AdapterStep:
        # We intentionally re-issue the original instruction with the latest
        # screenshot; the model was SFT'd as a Markov-style action policy and
        # we don't try to chain assistant tokens here.
        nudge = nudge_text or "Continue. Output the next action as a JSON object."
        return self._step(nudge, screenshot_png)

    # ---- internal -----------------------------------------------------------

    def _step(self, instruction: str, screenshot_png: bytes) -> AdapterStep:
        img = _png_bytes_to_pil(screenshot_png)
        messages = [
            {"role": "system", "content": self._system},
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": instruction},
            ]},
        ]
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        gen = out[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(gen, skip_special_tokens=True)[0]

        parsed = _try_parse_action_json(text)
        actions: list[Action] = []
        if parsed is not None:
            a = _action_from_dict(parsed)
            if a is not None:
                actions.append(a)

        return AdapterStep(actions=actions, message_text=text, raw_response=text)
