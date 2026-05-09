"""
Anthropic Claude (direct API, not Bedrock) with the computer-use tool.

Coordinates returned by Anthropic's computer tool are in absolute pixels of the
declared `display_width_px` x `display_height_px` viewport — Kernel's viewport
is configured to match (1280x800), so no normalization required.

Action vocabulary (Anthropic computer_20250124):
  key, type, mouse_move, left_click, right_click, middle_click, double_click,
  triple_click, left_click_drag, screenshot, cursor_position, wait, scroll,
  hold_key, left_mouse_down, left_mouse_up

Loop:
  - first_step: messages=[{role: user, content: [text + image]}]
  - next_step: append assistant message, append user message with tool_result(image)
  - terminal: stop_reason == "end_turn" with no tool_use
"""

from __future__ import annotations

import base64
import os
from typing import Any

from anthropic import Anthropic

from .base import Action, ActionType, AdapterStep, ModelAdapter


# Anthropic computer-use uses xdotool key spellings. Map to canonical lowercase
# names that the runner's execute_action(KEY_PRESS) passes through to Kernel.
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
}


def _split_keys(spec: str) -> list[str]:
    parts = [p.strip() for p in spec.split("+") if p.strip()]
    return [_KEY_ALIASES.get(p, p.lower()) for p in parts]


def _img_block(png: bytes) -> dict:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/png",
            "data": base64.b64encode(png).decode("ascii"),
        },
    }


class AnthropicDirectAdapter(ModelAdapter):
    """Direct Anthropic API with computer-use beta. Use this when the user has
    an Anthropic API key (vs. AWS Bedrock credentials)."""

    name = "anthropic"

    def __init__(
        self,
        viewport_w: int = 1280,
        viewport_h: int = 800,
        model_id: str = "claude-opus-4-7",
        tool_type: str | None = None,
        beta: str | None = None,
        max_tokens: int = 4096,
        api_key: str | None = None,
        system_instructions: str | None = None,
    ) -> None:
        # Per Anthropic docs (https://platform.claude.com/docs/en/agents-and-tools/tool-use/computer-use-tool):
        #   Opus 4.7, Opus 4.6, Sonnet 4.6, Opus 4.5  -> computer_20251124 + computer-use-2025-11-24
        #   Sonnet 4.5, Haiku 4.5, Opus 4.1, etc.     -> computer_20250124 + computer-use-2025-01-24
        if tool_type is None or beta is None:
            new_models = ("opus-4-7", "opus-4-6", "sonnet-4-6", "opus-4-5")
            if any(m in model_id for m in new_models):
                tool_type = tool_type or "computer_20251124"
                beta = beta or "computer-use-2025-11-24"
            else:
                tool_type = tool_type or "computer_20250124"
                beta = beta or "computer-use-2025-01-24"
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self.model_id = model_id
        self._tool_type = tool_type
        self._beta = beta
        self._max_tokens = max_tokens
        self._client = Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
        self._system = system_instructions or (
            "You are operating a real Chromium browser tab via the `computer` tool. "
            "Always emit a tool call (click/type/key/scroll/etc.) on every turn until "
            "the entire task is complete. Dismiss any cookie banners or popups. "
            "When the entire task is complete, emit your final answer as a single JSON "
            "object in your text reply (no prose, exactly the schema requested)."
        )
        self._tool = {
            "type": tool_type,
            "name": "computer",
            "display_width_px": viewport_w,
            "display_height_px": viewport_h,
        }
        self._messages: list[dict[str, Any]] = []
        self._last_tool_use_id: str | None = None

    # ---------- public API ----------

    def first_step(self, instruction: str, screenshot_png: bytes, page_url: str) -> AdapterStep:
        self._messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": instruction, "cache_control": {"type": "ephemeral"}},
                    _img_block(screenshot_png),
                ],
            }
        ]
        response = self._create()
        return self._parse(response)

    def next_step(
        self,
        executed: list[tuple[Action, str]],
        screenshot_png: bytes,
        page_url: str,
        nudge_text: str | None = None,
    ) -> AdapterStep:
        if nudge_text is not None or self._last_tool_use_id is None:
            # Nudge path: send a fresh user message with the new screenshot
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": nudge_text
                            or "Continue. Emit the next computer-tool call to make progress. "
                            "Do not reply with text only until the task is complete.",
                        },
                        _img_block(screenshot_png),
                    ],
                }
            )
        else:
            # Standard path: send tool_result image keyed off the prior tool_use_id
            self._messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": self._last_tool_use_id,
                            "content": [_img_block(screenshot_png)],
                        }
                    ],
                }
            )
        response = self._create()
        return self._parse(response)

    # ---------- internals ----------

    def _prune_old_screenshots(self, keep_recent: int = 3) -> None:
        """Replace base64 image blocks in older tool_result messages with a
        small placeholder text. Anthropic computer-use racks up MBs of base64
        per screenshot; without pruning we hit RequestTooLargeError around
        step 70-80. We keep the latest `keep_recent` screenshots intact (so the
        model can still see what's on screen) and replace older ones with a
        text marker so the call_id chain remains valid.
        """
        # Collect (msg_idx, block_idx) of every tool_result in user messages
        idxs: list[tuple[int, int]] = []
        for i, m in enumerate(self._messages):
            content = m.get("content")
            if m.get("role") == "user" and isinstance(content, list):
                for j, block in enumerate(content):
                    if isinstance(block, dict) and block.get("type") == "tool_result":
                        idxs.append((i, j))
        if len(idxs) <= keep_recent:
            return
        for i, j in idxs[:-keep_recent]:
            block = self._messages[i]["content"][j]
            inner = block.get("content")
            if isinstance(inner, list) and any(
                isinstance(x, dict) and x.get("type") == "image" for x in inner
            ):
                block["content"] = [
                    {"type": "text", "text": "[older screenshot elided to save tokens]"}
                ]
        # Also drop the initial user message's image if it's no longer the
        # only screenshot context the model has — the system prompt + task
        # instruction stay; just the early screenshot goes.
        if self._messages and self._messages[0].get("role") == "user":
            content = self._messages[0].get("content")
            if isinstance(content, list) and len(content) > 1:
                # Replace any image blocks with a placeholder; keep text.
                new_content = []
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "image":
                        new_content.append(
                            {"type": "text", "text": "[initial screenshot elided]"}
                        )
                    else:
                        new_content.append(block)
                self._messages[0]["content"] = new_content

    def _create(self) -> Any:
        # First: prune older screenshots to keep payload under the 32MB cap.
        self._prune_old_screenshots(keep_recent=3)

        # Anthropic caps cache_control at 4 blocks per request. Strip every
        # existing cache_control from the message history first, then re-stamp
        # on (a) the very first user text block (the task instruction — stays
        # cached across the whole run) and (b) the most recent user content
        # block (the latest screenshot — keeps the prefix warm). That keeps us
        # at <= 2 cache breakpoints regardless of trajectory length.
        try:
            for m in self._messages:
                content = m.get("content")
                if not isinstance(content, list):
                    continue
                for block in content:
                    if isinstance(block, dict) and "cache_control" in block:
                        block.pop("cache_control", None)
                    if isinstance(block, dict) and "content" in block and isinstance(block["content"], list):
                        for inner in block["content"]:
                            if isinstance(inner, dict) and "cache_control" in inner:
                                inner.pop("cache_control", None)

            # Re-stamp: first user text (instruction) + latest user content's last block.
            for m in self._messages:
                if m.get("role") == "user" and isinstance(m.get("content"), list):
                    for block in m["content"]:
                        if isinstance(block, dict) and block.get("type") == "text":
                            block["cache_control"] = {"type": "ephemeral"}
                            break
                    break

            for m in reversed(self._messages):
                if m.get("role") == "user" and isinstance(m.get("content"), list) and m["content"]:
                    last = m["content"][-1]
                    if isinstance(last, dict):
                        if last.get("type") == "tool_result" and isinstance(last.get("content"), list) and last["content"]:
                            inner_last = last["content"][-1]
                            if isinstance(inner_last, dict):
                                inner_last["cache_control"] = {"type": "ephemeral"}
                        else:
                            last["cache_control"] = {"type": "ephemeral"}
                    break
        except Exception:
            pass

        return self._client.beta.messages.create(
            model=self.model_id,
            max_tokens=self._max_tokens,
            system=self._system,
            tools=[self._tool],
            betas=[self._beta],
            messages=self._messages,
        )

    def _parse(self, response: Any) -> AdapterStep:
        # Append the assistant turn to history.
        assistant_content: list[Any] = []
        message_text_parts: list[str] = []
        actions: list[Action] = []
        last_tool_use_id: str | None = None

        for block in response.content or []:
            btype = getattr(block, "type", None)
            # Build the dict form we need for the next-turn payload
            if btype == "text":
                txt = getattr(block, "text", "") or ""
                if txt:
                    message_text_parts.append(txt)
                assistant_content.append({"type": "text", "text": txt})
            elif btype == "tool_use":
                tu_name = getattr(block, "name", "")
                tu_id = getattr(block, "id", "")
                tu_input = getattr(block, "input", {}) or {}
                assistant_content.append(
                    {
                        "type": "tool_use",
                        "id": tu_id,
                        "name": tu_name,
                        "input": tu_input,
                    }
                )
                if tu_name == "computer":
                    last_tool_use_id = tu_id
                    canonical = self._action_from_input(tu_input, tu_id)
                    if canonical is not None:
                        actions.append(canonical)

        self._messages.append({"role": "assistant", "content": assistant_content})
        self._last_tool_use_id = last_tool_use_id

        return AdapterStep(
            actions=actions,
            message_text="\n".join(message_text_parts) if message_text_parts else None,
            raw_response=response,
        )

    def _action_from_input(self, tu_input: dict[str, Any], call_id: str) -> Action | None:
        a = tu_input.get("action")
        coord = tu_input.get("coordinate") or [None, None]
        x = int(coord[0]) if coord and coord[0] is not None else None
        y = int(coord[1]) if coord and coord[1] is not None else None

        corr = {"call_id": call_id}

        if a == "screenshot":
            # Treat as a no-op wait so the runner sends a fresh screenshot next turn.
            return Action(type=ActionType.WAIT, seconds=0.1, raw=tu_input, correlation=corr)
        if a == "left_click":
            return Action(type=ActionType.CLICK, x=x, y=y, button="left", raw=tu_input, correlation=corr)
        if a == "right_click":
            return Action(type=ActionType.RIGHT_CLICK, x=x, y=y, button="right", raw=tu_input, correlation=corr)
        if a == "middle_click":
            return Action(type=ActionType.CLICK, x=x, y=y, button="middle", raw=tu_input, correlation=corr)
        if a == "double_click":
            return Action(type=ActionType.DOUBLE_CLICK, x=x, y=y, raw=tu_input, correlation=corr)
        if a == "triple_click":
            # No dedicated triple_click in our base; emit double then single (single click_count=3 would be ideal but not in API).
            return Action(type=ActionType.DOUBLE_CLICK, x=x, y=y, raw=tu_input, correlation=corr)
        if a == "left_click_drag":
            start = tu_input.get("start_coordinate") or [None, None]
            sx = int(start[0]) if start and start[0] is not None else None
            sy = int(start[1]) if start and start[1] is not None else None
            return Action(type=ActionType.DRAG, x=sx, y=sy, end_x=x, end_y=y, raw=tu_input, correlation=corr)
        if a == "mouse_move":
            return Action(type=ActionType.HOVER, x=x, y=y, raw=tu_input, correlation=corr)
        if a == "type":
            return Action(type=ActionType.TYPE, text=tu_input.get("text", "") or "", raw=tu_input, correlation=corr)
        if a in ("key", "hold_key"):
            keys = _split_keys(tu_input.get("text", "") or "")
            return Action(type=ActionType.KEY_PRESS, keys=keys, raw=tu_input, correlation=corr)
        if a == "scroll":
            direction = tu_input.get("scroll_direction", "down")
            amount = int(tu_input.get("scroll_amount", 3) or 3)
            unit = 100
            dx, dy = 0, 0
            if direction == "down":
                dy = amount * unit
            elif direction == "up":
                dy = -amount * unit
            elif direction == "right":
                dx = amount * unit
            elif direction == "left":
                dx = -amount * unit
            return Action(
                type=ActionType.SCROLL,
                x=x if x is not None else self.viewport_w // 2,
                y=y if y is not None else self.viewport_h // 2,
                delta_x=dx,
                delta_y=dy,
                raw=tu_input,
                correlation=corr,
            )
        if a == "wait":
            dur = float(tu_input.get("duration", 1.0) or 1.0)
            return Action(type=ActionType.WAIT, seconds=dur, raw=tu_input, correlation=corr)
        if a == "cursor_position":
            return Action(type=ActionType.WAIT, seconds=0.1, raw=tu_input, correlation=corr)
        if a in ("left_mouse_down", "left_mouse_up"):
            # Not supported in our base ActionType; treat as no-op.
            return Action(type=ActionType.WAIT, seconds=0.1, raw=tu_input, correlation=corr)
        return None
