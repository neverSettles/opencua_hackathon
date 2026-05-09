"""
Northstar (Tzafon Lightcone) adapter.

Uses the Lightcone Responses API:
  - first call: input=[user message {text + image}], tools=[computer_use]
  - subsequent calls: previous_response_id + input=[{computer_call_output, ...}]
  - on text-only turns, we do NOT chain via computer_call_output (we just
    re-issue the previous_response_id with a user nudge + new screenshot).
"""

from __future__ import annotations

import base64
import os
from typing import Any

from tzafon import Lightcone

from .base import Action, ActionType, AdapterStep, ModelAdapter


def _denorm(v, full: int) -> int:
    if v is None:
        return full // 2
    return int(round(float(v) / 1000.0 * full))


def _png_to_data_url(png: bytes) -> str:
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


class NorthstarAdapter(ModelAdapter):
    name = "northstar"

    def __init__(
        self,
        model_id: str = "tzafon.northstar-cua-fast",
        viewport_w: int = 1280,
        viewport_h: int = 800,
        api_key: str | None = None,
        system_instructions: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        self._client = Lightcone(api_key=api_key or os.environ.get("TZAFON_API_KEY"))
        self._tools = [{
            "type": "computer_use",
            "display_width": viewport_w,
            "display_height": viewport_h,
            "environment": "browser",
        }]
        self._sys = system_instructions or (
            "You are operating a real browser tab. Use clicks, typing, and scrolling. "
            "Dismiss any popups or cookie banners. Always emit a computer action on every "
            "turn until the task is complete. When complete, emit an `answer` or `done` "
            "action whose text is the requested final output."
        )
        self._last_response_id: str | None = None
        self._last_call_id: str | None = None

    # --- public API ----------------------------------------------------------

    def first_step(self, instruction: str, screenshot_png: bytes, page_url: str) -> AdapterStep:
        url = _png_to_data_url(screenshot_png)
        resp = self._client.responses.create(
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
            # Either the previous turn was text-only (no call_id to chain), or
            # we explicitly want to nudge. Send a regular user message.
            resp = self._client.responses.create(
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
            return self._parse(resp)

        # Chain via computer_call_output for the most recent call_id.
        resp = self._client.responses.create(
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
        return self._parse(resp)

    # --- internals -----------------------------------------------------------

    def _parse(self, resp: Any) -> AdapterStep:
        self._last_response_id = getattr(resp, "id", None)
        actions: list[Action] = []
        message_text: str | None = None

        # Northstar emits at most one computer_call per turn, but we iterate to
        # catch message text + the call together.
        new_call_id: str | None = None
        for item in resp.output or []:
            if item.type == "computer_call":
                new_call_id = item.call_id
                act = self._action_from_call(item)
                if act is not None:
                    actions.append(act)
            elif item.type == "message":
                for block in item.content or []:
                    txt = getattr(block, "text", None)
                    if txt:
                        message_text = (message_text or "") + txt

        # only update last_call_id if this turn produced one, so retries-after-
        # text-only turns keep the right call_id around.
        if new_call_id is not None:
            self._last_call_id = new_call_id

        return AdapterStep(actions=actions, message_text=message_text, raw_response=resp)

    def _action_from_call(self, call: Any) -> Action | None:
        a = call.action
        atype = a.type
        x = _denorm(getattr(a, "x", None), self.viewport_w)
        y = _denorm(getattr(a, "y", None), self.viewport_h)
        ex = _denorm(getattr(a, "end_x", None), self.viewport_w)
        ey = _denorm(getattr(a, "end_y", None), self.viewport_h)
        button = getattr(a, "button", "left") or "left"

        if atype == "click":
            return Action(type=ActionType.CLICK, x=x, y=y, button=button, raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "double_click":
            return Action(type=ActionType.DOUBLE_CLICK, x=x, y=y, raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "triple_click":
            # not in normalized vocab; emulate as double_click + click to land near triple
            return Action(type=ActionType.DOUBLE_CLICK, x=x, y=y, raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "right_click":
            return Action(type=ActionType.RIGHT_CLICK, x=x, y=y, raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "type":
            return Action(type=ActionType.TYPE, text=getattr(a, "text", "") or "", raw=a,
                          correlation={"call_id": call.call_id})
        if atype in ("key", "keypress"):
            keys = list(getattr(a, "keys", []) or [])
            return Action(type=ActionType.KEY_PRESS, keys=keys, raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "scroll":
            dy = int(getattr(a, "scroll_y", 0) or 0)
            return Action(type=ActionType.SCROLL, x=x, y=y, delta_y=dy, raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "hscroll":
            dx = int(getattr(a, "scroll_x", 0) or 0)
            return Action(type=ActionType.SCROLL, x=x, y=y, delta_x=dx, raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "drag":
            return Action(type=ActionType.DRAG, x=x, y=y, end_x=ex, end_y=ey, raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "move":
            return Action(type=ActionType.HOVER, x=x, y=y, raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "navigate":
            return Action(type=ActionType.NAVIGATE, url=getattr(a, "url", None), raw=a,
                          correlation={"call_id": call.call_id})
        if atype == "wait":
            return Action(type=ActionType.WAIT, seconds=2.0, raw=a,
                          correlation={"call_id": call.call_id})
        if atype in ("terminate", "answer", "done"):
            return Action(
                type=ActionType.DONE,
                final_text=(getattr(a, "result", None) or getattr(a, "text", None)),
                raw=a,
                correlation={"call_id": call.call_id, "status": getattr(a, "status", None)},
            )
        return Action(type=ActionType.UNKNOWN, raw=a, correlation={"call_id": call.call_id})
