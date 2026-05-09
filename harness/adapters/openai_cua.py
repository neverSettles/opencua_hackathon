"""
OpenAI computer-use adapter (GA `computer` tool, gpt-5.5+).

Uses Responses API. Two key differences from Northstar:

  1. Coordinates are in **pixels** (the model's image size), not normalized 0-999.
     We pass the screenshot at our target viewport (1280x800) and consume the
     pixel coords directly.

  2. The GA `computer_call` carries an **`actions` array** (batched) instead of
     a single `action`. We iterate them in order each turn.

If the model is set to legacy `computer-use-preview`, we instead use
`tools=[{"type": "computer_use_preview"}]` with a single `action`.

Loop:
  - first turn: input=[user message + screenshot], tools=[computer]
  - subsequent: previous_response_id + input=[{computer_call_output, ...}]
  - on text-only turn: re-prompt with new user message + nudge
"""

from __future__ import annotations

import base64
import os
from typing import Any

from openai import OpenAI

from .base import Action, ActionType, AdapterStep, ModelAdapter


def _png_to_data_url(png: bytes) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


class OpenAIComputerUseAdapter(ModelAdapter):
    name = "openai"

    def __init__(
        self,
        model_id: str = "gpt-5.5",
        viewport_w: int = 1280,
        viewport_h: int = 800,
        api_key: str | None = None,
        system_instructions: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self._client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        # Detect legacy preview model.
        self._is_preview = "preview" in model_id
        if self._is_preview:
            self._tools = [
                {
                    "type": "computer_use_preview",
                    "display_width": viewport_w,
                    "display_height": viewport_h,
                    "environment": "browser",
                }
            ]
        else:
            # GA `computer` tool: dimensions inferred from screenshot pixels.
            self._tools = [{"type": "computer"}]
        self._sys = system_instructions or (
            "You are operating a real browser tab. Use clicks, typing, and scrolling. "
            "Dismiss any popups or cookie banners. Always emit at least one computer action "
            "until the entire task is complete. When complete, reply in plain text with the "
            "requested final answer (the user has specified the schema)."
        )
        self._last_response_id: str | None = None
        self._last_call_id: str | None = None

    # ---- public API ---------------------------------------------------------

    def first_step(self, instruction: str, screenshot_png: bytes, page_url: str) -> AdapterStep:
        url = _png_to_data_url(screenshot_png)
        kwargs: dict[str, Any] = dict(
            model=self.model_id,
            instructions=self._sys,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": url, "detail": "auto"},
                    ],
                }
            ],
            tools=self._tools,
        )
        if self._is_preview:
            kwargs["truncation"] = "auto"
        resp = self._client.responses.create(**kwargs)
        return self._parse(resp)

    def next_step(
        self,
        executed: list[tuple[Action, str]],
        screenshot_png: bytes,
        page_url: str,
        nudge_text: str | None = None,
    ) -> AdapterStep:
        url = _png_to_data_url(screenshot_png)
        if nudge_text is not None or self._last_call_id is None:
            kwargs: dict[str, Any] = dict(
                model=self.model_id,
                previous_response_id=self._last_response_id,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": nudge_text
                                or "Continue. Emit the next computer action.",
                            },
                            {"type": "input_image", "image_url": url, "detail": "auto"},
                        ],
                    }
                ],
                tools=self._tools,
            )
            if self._is_preview:
                kwargs["truncation"] = "auto"
            resp = self._client.responses.create(**kwargs)
            return self._parse(resp)

        kwargs = dict(
            model=self.model_id,
            previous_response_id=self._last_response_id,
            input=[
                {
                    "type": "computer_call_output",
                    "call_id": self._last_call_id,
                    "output": {"type": "input_image", "image_url": url, "detail": "auto"},
                }
            ],
            tools=self._tools,
        )
        if self._is_preview:
            kwargs["truncation"] = "auto"
        resp = self._client.responses.create(**kwargs)
        return self._parse(resp)

    # ---- internals ----------------------------------------------------------

    def _parse(self, resp: Any) -> AdapterStep:
        self._last_response_id = getattr(resp, "id", None)
        actions: list[Action] = []
        message_text: str | None = None
        new_call_id: str | None = None

        for item in resp.output or []:
            t = getattr(item, "type", None)
            if t == "computer_call":
                new_call_id = item.call_id
                # GA: item has `actions` (list); preview: item has `action` (single).
                ga_actions = getattr(item, "actions", None)
                if ga_actions:
                    for a in ga_actions:
                        n = self._action_from_raw(a, item.call_id)
                        if n is not None:
                            actions.append(n)
                else:
                    a = getattr(item, "action", None)
                    if a is not None:
                        n = self._action_from_raw(a, item.call_id)
                        if n is not None:
                            actions.append(n)
            elif t == "message":
                content = getattr(item, "content", None) or []
                for block in content:
                    txt = getattr(block, "text", None)
                    if txt:
                        message_text = (message_text or "") + txt
            elif t == "reasoning":
                # Skip silent reasoning blocks.
                pass

        if new_call_id is not None:
            self._last_call_id = new_call_id

        return AdapterStep(actions=actions, message_text=message_text, raw_response=resp)

    def _action_from_raw(self, a: Any, call_id: str) -> Action | None:
        atype = getattr(a, "type", None)
        x = getattr(a, "x", None)
        y = getattr(a, "y", None)
        keys = list(getattr(a, "keys", None) or [])

        if atype == "screenshot":
            # Model wants a fresh screenshot; we always send one next turn.
            # Treat as a no-op wait so the loop continues without breaking.
            return Action(type=ActionType.WAIT, seconds=0.1, raw=a,
                          correlation={"call_id": call_id})
        if atype == "click":
            button = getattr(a, "button", "left") or "left"
            # Map non-standard buttons to left.
            if button not in ("left", "right", "middle", "back", "forward"):
                button = "left"
            return Action(
                type=ActionType.CLICK if button != "right" else ActionType.RIGHT_CLICK,
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
                button=button,
                keys=keys,
                raw=a,
                correlation={"call_id": call_id},
            )
        if atype == "double_click":
            return Action(
                type=ActionType.DOUBLE_CLICK,
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
                keys=keys,
                raw=a,
                correlation={"call_id": call_id},
            )
        if atype == "type":
            return Action(
                type=ActionType.TYPE,
                text=getattr(a, "text", "") or "",
                raw=a,
                correlation={"call_id": call_id},
            )
        if atype == "keypress":
            return Action(
                type=ActionType.KEY_PRESS,
                keys=[k.lower() for k in keys],
                raw=a,
                correlation={"call_id": call_id},
            )
        if atype == "scroll":
            sx = int(getattr(a, "scroll_x", 0) or 0)
            sy = int(getattr(a, "scroll_y", 0) or 0)
            return Action(
                type=ActionType.SCROLL,
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
                delta_x=sx,
                delta_y=sy,
                raw=a,
                correlation={"call_id": call_id},
            )
        if atype == "move":
            return Action(
                type=ActionType.HOVER,
                x=int(x) if x is not None else None,
                y=int(y) if y is not None else None,
                raw=a,
                correlation={"call_id": call_id},
            )
        if atype == "drag":
            path = getattr(a, "path", None) or []
            if len(path) >= 2:
                p0, pn = path[0], path[-1]
                return Action(
                    type=ActionType.DRAG,
                    x=int(p0.x) if hasattr(p0, "x") else None,
                    y=int(p0.y) if hasattr(p0, "y") else None,
                    end_x=int(pn.x) if hasattr(pn, "x") else None,
                    end_y=int(pn.y) if hasattr(pn, "y") else None,
                    raw=a,
                    correlation={"call_id": call_id},
                )
            return None
        if atype == "wait":
            return Action(type=ActionType.WAIT, seconds=2.0, raw=a,
                          correlation={"call_id": call_id})
        return Action(type=ActionType.UNKNOWN, raw=a, correlation={"call_id": call_id})
