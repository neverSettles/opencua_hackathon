"""
Gemini computer-use adapter (built-in computer use in Gemini 3.x).

Uses google-genai SDK. The agent loop maintains a `contents` list of
Content objects: alternating user (initial prompt + screenshots) and
model (text + function_call) parts. After each round we append the
model response and a user FunctionResponse with the new screenshot.

Predefined function vocabulary (per Gemini 3 docs):
  click_at, type_text_at, scroll_document, scroll_at, hover_at,
  drag_and_drop, key_combination, navigate, search, open_web_browser,
  wait_5_seconds, go_back, go_forward
Coordinates are in 0-999 normalized space (1000x1000 grid).
"""

from __future__ import annotations

import os
from typing import Any

from google import genai
from google.genai import types

from .base import Action, ActionType, AdapterStep, ModelAdapter


def _denorm(v, full: int) -> int:
    if v is None:
        return full // 2
    return int(round(float(v) / 1000.0 * full))


# Gemini's key_combination uses "Control+A" / "Enter" / "control+c" style.
def _parse_key_combo(s: str) -> list[str]:
    parts = [p.strip() for p in s.replace("-", "+").split("+") if p.strip()]
    out = []
    for p in parts:
        pl = p.lower()
        # normalize a few common ones; Kernel's press_key accepts standard names
        if pl in ("ctrl", "control"):
            out.append("ctrl")
        elif pl in ("opt", "option", "alt"):
            out.append("alt")
        elif pl == "cmd" or pl == "command" or pl == "meta":
            out.append("meta")
        elif pl == "shift":
            out.append("shift")
        else:
            out.append(pl)
    return out


class GeminiAdapter(ModelAdapter):
    name = "gemini"

    def __init__(
        self,
        model_id: str = "gemini-3-pro-preview",
        viewport_w: int = 1280,
        viewport_h: int = 800,
        api_key: str | None = None,
        system_instructions: str | None = None,
        excluded_predefined_functions: list[str] | None = None,
    ) -> None:
        self.model_id = model_id
        self.viewport_w = viewport_w
        self.viewport_h = viewport_h
        api_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        self._client = genai.Client(api_key=api_key)
        self._sys = system_instructions or (
            "You are operating a real browser tab to complete the user's task. "
            "On every step, propose at most a handful of UI actions using the predefined "
            "computer use functions. Dismiss popups and cookie banners as you encounter them. "
            "When the task is fully complete, output your final answer as a plain text reply "
            "(no function call) following the schema the user requested."
        )
        cu = types.ComputerUse(
            environment=types.Environment.ENVIRONMENT_BROWSER,
            excluded_predefined_functions=excluded_predefined_functions or [],
        )
        self._config = types.GenerateContentConfig(
            tools=[types.Tool(computer_use=cu)],
            system_instruction=self._sys,
        )
        self._contents: list[types.Content] = []
        self._pending_calls: list[Any] = []   # function_calls from the previous model turn

    # --- public API ----------------------------------------------------------

    def first_step(self, instruction: str, screenshot_png: bytes, page_url: str) -> AdapterStep:
        self._contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part(text=instruction),
                    types.Part.from_bytes(data=screenshot_png, mime_type="image/png"),
                ],
            )
        ]
        return self._send()

    def next_step(
        self,
        executed: list[tuple[Action, str]],
        screenshot_png: bytes,
        page_url: str,
        nudge_text: str | None = None,
    ) -> AdapterStep:
        # Build FunctionResponse parts mirroring the order of pending_calls.
        # Even if the user passed a nudge, we need to acknowledge each pending
        # function_call (Gemini will error otherwise).
        if self._pending_calls:
            response_parts: list[types.Part] = []
            for i, fc in enumerate(self._pending_calls):
                # If we have an executed entry at this index, use its result;
                # otherwise mark as not_executed.
                result_str = executed[i][1] if i < len(executed) else "not_executed"
                # Attach the screenshot to the LAST function response so the
                # model has visual feedback for the latest state.
                fr_parts = []
                if i == len(self._pending_calls) - 1:
                    fr_parts.append(
                        types.FunctionResponsePart(
                            inline_data=types.FunctionResponseBlob(
                                mime_type="image/png", data=screenshot_png
                            )
                        )
                    )
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"url": page_url, "result": result_str},
                            parts=fr_parts or None,
                        )
                    )
                )
            self._contents.append(types.Content(role="user", parts=response_parts))
        else:
            # No pending calls (model returned text only). Send a fresh user
            # message with the latest screenshot + optional nudge.
            self._contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            text=nudge_text
                            or "Continue. Emit the next computer use function call to make progress."
                        ),
                        types.Part.from_bytes(data=screenshot_png, mime_type="image/png"),
                    ],
                )
            )

        return self._send()

    # --- internals -----------------------------------------------------------

    def _send(self) -> AdapterStep:
        resp = self._client.models.generate_content(
            model=self.model_id,
            contents=self._contents,
            config=self._config,
        )
        # Append the model response to history for the next turn.
        cand = (resp.candidates or [None])[0]
        if cand and cand.content:
            self._contents.append(cand.content)

        actions: list[Action] = []
        message_text: str | None = None
        self._pending_calls = []

        if cand and cand.content and cand.content.parts:
            for part in cand.content.parts:
                fc = getattr(part, "function_call", None)
                if fc is not None and fc.name:
                    self._pending_calls.append(fc)
                    a = self._action_from_call(fc)
                    if a is not None:
                        actions.append(a)
                elif getattr(part, "text", None):
                    message_text = (message_text or "") + part.text

        # If model returned text only with no function_call, treat as done IF
        # the runner can parse JSON from message_text; otherwise the runner
        # will nudge.
        return AdapterStep(actions=actions, message_text=message_text, raw_response=resp)

    def _action_from_call(self, fc: Any) -> Action | None:
        name = fc.name
        # google-genai exposes args as a dict-like
        args = dict(fc.args or {})
        x_norm = args.get("x")
        y_norm = args.get("y")
        ex_norm = args.get("destination_x")
        ey_norm = args.get("destination_y")
        x = _denorm(x_norm, self.viewport_w) if x_norm is not None else None
        y = _denorm(y_norm, self.viewport_h) if y_norm is not None else None
        ex = _denorm(ex_norm, self.viewport_w) if ex_norm is not None else None
        ey = _denorm(ey_norm, self.viewport_h) if ey_norm is not None else None

        if name == "open_web_browser":
            # Browser is already open via Kernel; treat as no-op
            return Action(type=ActionType.WAIT, seconds=0.1, raw=fc, correlation={"name": name})
        if name == "wait_5_seconds":
            return Action(type=ActionType.WAIT, seconds=5.0, raw=fc, correlation={"name": name})
        if name == "go_back":
            return Action(type=ActionType.GO_BACK, raw=fc, correlation={"name": name})
        if name == "go_forward":
            return Action(type=ActionType.GO_FORWARD, raw=fc, correlation={"name": name})
        if name == "search":
            return Action(type=ActionType.NAVIGATE, url="https://www.google.com/", raw=fc, correlation={"name": name})
        if name == "navigate":
            return Action(type=ActionType.NAVIGATE, url=args.get("url"), raw=fc, correlation={"name": name})
        if name == "click_at":
            return Action(type=ActionType.CLICK, x=x, y=y, raw=fc, correlation={"name": name})
        if name == "hover_at":
            return Action(type=ActionType.HOVER, x=x, y=y, raw=fc, correlation={"name": name})
        if name == "type_text_at":
            text = args.get("text", "")
            press_enter = bool(args.get("press_enter", True))
            clear_before = bool(args.get("clear_before_typing", True))
            return Action(
                type=ActionType.TYPE_AT,
                x=x, y=y,
                text=text,
                press_enter=press_enter,
                clear_before_typing=clear_before,
                raw=fc,
                correlation={"name": name},
            )
        if name == "key_combination":
            keys = _parse_key_combo(args.get("keys", "") or "")
            return Action(type=ActionType.KEY_PRESS, keys=keys, raw=fc, correlation={"name": name})
        if name == "scroll_document":
            d = (args.get("direction") or "down").lower()
            return Action(type=ActionType.SCROLL_DIR, direction=d, magnitude_px=600, raw=fc,
                          correlation={"name": name})
        if name == "scroll_at":
            d = (args.get("direction") or "down").lower()
            mag_norm = int(args.get("magnitude", 800))
            mag_px = int(round(mag_norm / 1000.0 * (self.viewport_h if d in ("up", "down") else self.viewport_w)))
            return Action(
                type=ActionType.SCROLL_DIR, x=x, y=y, direction=d, magnitude_px=mag_px,
                raw=fc, correlation={"name": name},
            )
        if name == "drag_and_drop":
            return Action(type=ActionType.DRAG, x=x, y=y, end_x=ex, end_y=ey, raw=fc,
                          correlation={"name": name})
        return Action(type=ActionType.UNKNOWN, raw=fc, correlation={"name": name})
