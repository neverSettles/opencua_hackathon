"""Anthropic Claude on AWS Bedrock with the computer-use tool.

Coordinates returned by Anthropic's computer tool are in absolute pixels of the
declared `display_width_px` x `display_height_px` viewport (matching Kernel's
viewport). No normalization required.

Action vocabulary (Anthropic computer_20250124):
  key, type, mouse_move, left_click, right_click, middle_click, double_click,
  triple_click, left_click_drag, screenshot, cursor_position, wait, scroll,
  hold_key, left_mouse_down, left_mouse_up
"""

from __future__ import annotations

import base64
import os
from typing import Any

from anthropic import AnthropicBedrock

from .base import AdapterStep, CanonicalAction, ModelAdapter


# Mapping of Anthropic computer-use key spellings to Kernel/xdotool spellings.
# Bedrock Claude uses xdotool-style keys (e.g. 'Return', 'Tab', 'Page_Down').
# Kernel's press_key passes through whatever we give it. We normalize a few
# common aliases here for safety.
_KEY_ALIASES = {
    "Return": "enter",
    "Enter": "enter",
    "Tab": "tab",
    "Escape": "escape",
    "Esc": "escape",
    "BackSpace": "backspace",
    "Backspace": "backspace",
    "Delete": "delete",
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    "Home": "home",
    "End": "end",
    "Page_Up": "pageup",
    "PageUp": "pageup",
    "Page_Down": "pagedown",
    "PageDown": "pagedown",
    "ctrl": "ctrl",
    "Control": "ctrl",
    "shift": "shift",
    "Shift": "shift",
    "alt": "alt",
    "Alt": "alt",
    "super": "meta",
    "Super": "meta",
    "Meta": "meta",
    "cmd": "meta",
    "Command": "meta",
    "space": "space",
    "Space": " ",
}


def _split_keys(spec: str) -> list[str]:
    """Convert e.g. 'ctrl+l' or 'Return' to a list of canonical keys."""
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    return [_KEY_ALIASES.get(p, p.lower()) for p in parts]


class BedrockClaudeAdapter(ModelAdapter):
    """Claude on Bedrock with computer-use beta."""

    name = "bedrock_claude"

    def __init__(
        self,
        viewport_w: int,
        viewport_h: int,
        model_id: str = "us.anthropic.claude-opus-4-6-v1",
        tool_type: str = "computer_20250124",
        beta: str = "computer-use-2025-01-24",
        max_tokens: int = 4096,
        aws_access_key_id: str | None = None,
        aws_secret_access_key: str | None = None,
        aws_region: str | None = None,
        aws_session_token: str | None = None,
        system_instructions: str | None = None,
    ) -> None:
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.model_id = model_id
        self._tool_type = tool_type
        self._beta = beta
        self._max_tokens = max_tokens
        self._client = AnthropicBedrock(
            aws_access_key=aws_access_key_id or os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_key=aws_secret_access_key or os.environ.get("AWS_SECRET_ACCESS_KEY"),
            aws_region=aws_region or os.environ.get("AWS_REGION", "us-east-1"),
            aws_session_token=aws_session_token or os.environ.get("AWS_SESSION_TOKEN"),
        )
        self._system = system_instructions or (
            "You are operating a real browser tab via the `computer` tool. "
            "Always emit a tool call (click/type/key/scroll/etc.) on every turn "
            "until the entire task is complete. When the entire task is complete, "
            "emit your final answer as a single JSON object in your text reply "
            "(no prose, exactly the schema requested)."
        )
        self._tool = {
            "type": tool_type,
            "name": "computer",
            "display_width_px": viewport_w,
            "display_height_px": viewport_h,
        }
        self._messages: list[dict] = []
        self._last_tool_use_id: str | None = None

    def _img_block(self, png: bytes) -> dict:
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": base64.b64encode(png).decode("ascii"),
            },
        }

    def _canonical(self, tu_input: dict) -> CanonicalAction | None:
        a = tu_input.get("action")
        coord = tu_input.get("coordinate") or [None, None]
        x = int(coord[0]) if coord and coord[0] is not None else None
        y = int(coord[1]) if coord and coord[1] is not None else None

        if a == "screenshot":
            return CanonicalAction(type="screenshot")
        if a == "left_click":
            return CanonicalAction(type="click", x=x, y=y, button="left")
        if a == "right_click":
            return CanonicalAction(type="right_click", x=x, y=y, button="right")
        if a == "middle_click":
            return CanonicalAction(type="click", x=x, y=y, button="middle")
        if a == "double_click":
            return CanonicalAction(type="double_click", x=x, y=y)
        if a == "triple_click":
            return CanonicalAction(type="triple_click", x=x, y=y)
        if a == "left_click_drag":
            start = tu_input.get("start_coordinate") or [None, None]
            sx = int(start[0]) if start and start[0] is not None else None
            sy = int(start[1]) if start and start[1] is not None else None
            return CanonicalAction(type="drag", x=sx, y=sy, end_x=x, end_y=y)
        if a == "mouse_move":
            return CanonicalAction(type="move", x=x, y=y)
        if a == "type":
            return CanonicalAction(type="type", text=tu_input.get("text", "") or "")
        if a in ("key", "hold_key"):
            return CanonicalAction(type="key", keys=_split_keys(tu_input.get("text", "") or ""))
        if a == "scroll":
            direction = tu_input.get("scroll_direction", "down")
            amount = int(tu_input.get("scroll_amount", 3) or 3)
            # Translate to delta_x/delta_y. Each "amount" unit ~= 100 px.
            unit = 100
            dx, dy = 0, 0
            if direction == "down":
                dy = -amount * unit
            elif direction == "up":
                dy = amount * unit
            elif direction == "left":
                dx = amount * unit
            elif direction == "right":
                dx = -amount * unit
            return CanonicalAction(type="scroll", x=x, y=y, delta_x=dx, delta_y=dy)
        if a == "wait":
            dur = float(tu_input.get("duration", 1) or 1)
            return CanonicalAction(type="wait", duration_seconds=dur)
        if a == "cursor_position":
            return CanonicalAction(type="screenshot")  # treat as no-op observation
        return None

    def _create(self, messages: list[dict]) -> Any:
        return self._client.beta.messages.create(
            model=self.model_id,
            max_tokens=self._max_tokens,
            system=self._system,
            tools=[self._tool],
            betas=[self._beta],
            messages=messages,
        )

    def _parse(self, response: Any) -> AdapterStep:
        # Append assistant message to history for next turn.
        assistant_content = [
            (b.model_dump() if hasattr(b, "model_dump") else b) for b in response.content
        ]
        self._messages.append({"role": "assistant", "content": assistant_content})

        action: CanonicalAction | None = None
        message_text: str | None = None
        final_text: str | None = None
        last_tool_use_id: str | None = None
        for block in response.content:
            if block.type == "text":
                message_text = (message_text or "") + (block.text or "")
            elif block.type == "tool_use" and block.name == "computer":
                last_tool_use_id = block.id
                ct = self._canonical(block.input or {})
                if ct is not None:
                    action = ct

        if action is not None:
            self._last_tool_use_id = last_tool_use_id
        else:
            self._last_tool_use_id = None

        # If model stopped without a tool call, that's the final answer turn.
        stop_reason = getattr(response, "stop_reason", None)
        if stop_reason == "end_turn" and action is None:
            final_text = message_text

        return AdapterStep(
            action=action,
            message_text=message_text,
            final_text=final_text,
            raw_response=response,
            metadata={"stop_reason": stop_reason, "tool_use_id": last_tool_use_id},
        )

    def start(self, instruction: str, screenshot_png: bytes) -> AdapterStep:
        self._messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction},
                    self._img_block(screenshot_png),
                ],
            }
        ]
        response = self._create(self._messages)
        return self._parse(response)

    def step(self, screenshot_png: bytes) -> AdapterStep:
        if self._last_tool_use_id is None:
            # No prior tool_use -> send a user message with the new screenshot
            # as a nudge to keep going.
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Continue. Emit the next computer-tool call to make progress. "
                                "Do not reply with text only until the task is complete."
                            ),
                        },
                        self._img_block(screenshot_png),
                    ],
                }
            )
        else:
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": self._last_tool_use_id,
                            "content": [self._img_block(screenshot_png)],
                        }
                    ],
                }
            )
        response = self._create(self._messages)
        return self._parse(response)
