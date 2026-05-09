"""
Model-agnostic adapter interface for CUA loops.

Each adapter owns its own conversation/history bookkeeping and exposes
two methods (`first_step`, `next_step`) that take the current screenshot
and return a list of normalized `Action`s plus optional narration.

The runner is then identical across models: capture screenshot -> call
adapter -> execute returned actions on Kernel -> repeat.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    TYPE = "type"               # type at current focus
    TYPE_AT = "type_at"         # click(x,y) [+ optional clear] + type [+ optional Enter]
    KEY_PRESS = "key_press"
    SCROLL = "scroll"           # x,y + delta_x/delta_y in px
    SCROLL_DIR = "scroll_dir"   # direction string + magnitude px
    HOVER = "hover"
    DRAG = "drag"
    NAVIGATE = "navigate"
    GO_BACK = "go_back"
    GO_FORWARD = "go_forward"
    WAIT = "wait"
    DONE = "done"               # final answer
    TERMINATE = "terminate"
    UNKNOWN = "unknown"


@dataclass
class Action:
    type: ActionType
    x: int | None = None              # viewport pixels
    y: int | None = None
    end_x: int | None = None
    end_y: int | None = None
    button: str = "left"
    text: str | None = None
    keys: list[str] = field(default_factory=list)
    delta_x: int = 0
    delta_y: int = 0
    direction: str | None = None
    magnitude_px: int = 400
    url: str | None = None
    seconds: float = 1.0
    press_enter: bool = False
    clear_before_typing: bool = False
    raw: Any = None
    # Echo back when adapter needs it for the next-turn payload (e.g. Northstar
    # uses call_id to chain computer_call_output, Gemini uses the function_call
    # name to build a FunctionResponse).
    correlation: dict[str, Any] = field(default_factory=dict)
    # Final-answer text for DONE/TERMINATE.
    final_text: str | None = None


@dataclass
class AdapterStep:
    """One model turn worth of output."""
    actions: list[Action]
    message_text: str | None = None    # narration this turn (if any)
    is_terminal: bool = False          # set by runner after executing terminal action
    raw_response: Any = None


class ModelAdapter:
    """Subclass and implement first_step / next_step."""
    name: str = "base"
    model_id: str = ""

    def first_step(self, instruction: str, screenshot_png: bytes, page_url: str) -> AdapterStep:
        raise NotImplementedError

    def next_step(
        self,
        executed: list[tuple[Action, str]],   # (action, result_str) for each action this turn
        screenshot_png: bytes,
        page_url: str,
        nudge_text: str | None = None,
    ) -> AdapterStep:
        raise NotImplementedError


# --- Kernel executor: takes a normalized Action, executes on Kernel/Playwright -----

def execute_action(kernel, session_id: str, page, action: Action) -> str:
    """Execute one normalized Action on the Kernel browser. Returns a short
    string status (used as feedback to some adapters)."""
    t = action.type

    if t == ActionType.CLICK:
        kernel.browsers.computer.click_mouse(
            session_id, x=action.x or 0, y=action.y or 0, button=action.button or "left"
        )
        return "ok"
    if t == ActionType.DOUBLE_CLICK:
        kernel.browsers.computer.click_mouse(
            session_id, x=action.x or 0, y=action.y or 0, num_clicks=2
        )
        return "ok"
    if t == ActionType.RIGHT_CLICK:
        kernel.browsers.computer.click_mouse(
            session_id, x=action.x or 0, y=action.y or 0, button="right"
        )
        return "ok"
    if t == ActionType.TYPE:
        kernel.browsers.computer.type_text(session_id, text=action.text or "")
        return "ok"
    if t == ActionType.TYPE_AT:
        kernel.browsers.computer.click_mouse(
            session_id, x=action.x or 0, y=action.y or 0, button="left"
        )
        time.sleep(0.15)
        if action.clear_before_typing:
            kernel.browsers.computer.press_key(session_id, keys=["ctrl", "a"])
            time.sleep(0.05)
            kernel.browsers.computer.press_key(session_id, keys=["delete"])
            time.sleep(0.05)
        if action.text:
            kernel.browsers.computer.type_text(session_id, text=action.text)
        if action.press_enter:
            time.sleep(0.05)
            kernel.browsers.computer.press_key(session_id, keys=["enter"])
        return "ok"
    if t == ActionType.KEY_PRESS:
        if action.keys:
            kernel.browsers.computer.press_key(session_id, keys=action.keys)
        return "ok"
    if t == ActionType.SCROLL:
        kernel.browsers.computer.scroll(
            session_id,
            x=action.x or 640,
            y=action.y or 400,
            delta_x=action.delta_x,
            delta_y=action.delta_y,
        )
        return "ok"
    if t == ActionType.SCROLL_DIR:
        d = (action.direction or "down").lower()
        mag = action.magnitude_px or 400
        dx = dy = 0
        if d == "down":
            dy = mag
        elif d == "up":
            dy = -mag
        elif d == "right":
            dx = mag
        elif d == "left":
            dx = -mag
        x = action.x if action.x is not None else 640
        y = action.y if action.y is not None else 400
        kernel.browsers.computer.scroll(session_id, x=x, y=y, delta_x=dx, delta_y=dy)
        return "ok"
    if t == ActionType.HOVER:
        kernel.browsers.computer.move_mouse(session_id, x=action.x or 0, y=action.y or 0)
        return "ok"
    if t == ActionType.DRAG:
        kernel.browsers.computer.drag_mouse(
            session_id,
            path=[
                {"x": action.x or 0, "y": action.y or 0},
                {"x": action.end_x or 0, "y": action.end_y or 0},
            ],
        )
        return "ok"
    if t == ActionType.NAVIGATE:
        if page is not None and action.url:
            page.goto(action.url, wait_until="domcontentloaded", timeout=45_000)
        return "ok"
    if t == ActionType.GO_BACK:
        if page is not None:
            page.go_back(wait_until="domcontentloaded", timeout=45_000)
        return "ok"
    if t == ActionType.GO_FORWARD:
        if page is not None:
            page.go_forward(wait_until="domcontentloaded", timeout=45_000)
        return "ok"
    if t == ActionType.WAIT:
        time.sleep(max(0.1, action.seconds))
        return "ok"
    if t in (ActionType.DONE, ActionType.TERMINATE):
        return "terminal"
    return f"unhandled:{t.value}"
