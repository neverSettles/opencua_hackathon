"""Run Anthropic Computer Use against real websites for T2 / T3 / T4.

Drives a real Playwright Chromium. Loops Claude messages.create with the
computer-use tool until the model returns a final text answer (parseable JSON)
or hits max_steps. Saves trajectory.jsonl + agent_output.json compatible with
scripts/scoring.py and bench.harness.viewer.

Usage:
    set -a; source .env; set +a
    python scripts/run_anthropic_cua.py --task T2 --model claude-sonnet-4-6
    python scripts/run_anthropic_cua.py --task T2 --model claude-opus-4-7 --max-steps 80
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ---------- task spec extraction ----------

def _extract_agent_prompt(spec_md: str) -> str:
    """Pull the first fenced code block under '## Agent prompt'."""
    m = re.search(r"## Agent prompt\s*\n+```\w*\n(.*?)\n```", spec_md, re.S)
    if not m:
        raise RuntimeError("Could not find agent prompt in spec markdown")
    return m.group(1).strip()


TASK_CONFIG = {
    "T2": {
        "spec_path": ROOT / "tasks" / "T2_restaurant_restock.md",
        "start_url": "https://www.webstaurantstore.com",
        "output_filename": "agent_output.json",
    },
    "T3": {
        "spec_path": ROOT / "tasks" / "T3_rei_hiking_boots.md",
        "start_url": "https://www.rei.com/c/mens-hiking-boots",
        "output_filename": "agent_output.json",
    },
    "T4": {
        "spec_path": ROOT / "tasks" / "T4_used_books_basket.md",
        "start_url": "https://www.abebooks.com",
        "output_filename": "agent_output.json",
    },
}


# ---------- Playwright helpers ----------

class BrowserSession:
    """Synchronous Playwright wrapper sized for the computer tool's viewport."""

    def __init__(self, width: int = 1280, height: int = 800, headless: bool = False):
        self.width = width
        self.height = height
        self.headless = headless
        self._pw = None
        self.browser = None
        self.context = None
        self.page = None

    def __enter__(self):
        from playwright.sync_api import sync_playwright
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            viewport={"width": self.width, "height": self.height},
            device_scale_factor=1,
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
            ),
        )
        self.page = self.context.new_page()
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.context:
                self.context.close()
        except Exception:
            pass
        try:
            if self.browser:
                self.browser.close()
        except Exception:
            pass
        try:
            if self._pw:
                self._pw.stop()
        except Exception:
            pass

    def screenshot_b64(self) -> str:
        png = self.page.screenshot(type="png", full_page=False)
        return base64.standard_b64encode(png).decode("ascii")

    def screenshot_save(self, path: Path) -> None:
        png = self.page.screenshot(type="png", full_page=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(png)

    def navigate(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded", timeout=30000)

    def execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Map an Anthropic computer-use action onto Playwright. Returns a small log dict."""
        kind = action.get("action")
        coord = action.get("coordinate")
        text = action.get("text")
        try:
            if kind == "screenshot":
                # caller will take a screenshot afterwards; nothing to do
                return {"ok": True, "note": "screenshot requested"}
            if kind == "left_click":
                x, y = coord or [0, 0]
                self.page.mouse.click(x, y)
            elif kind == "double_click":
                x, y = coord or [0, 0]
                self.page.mouse.dblclick(x, y)
            elif kind == "right_click":
                x, y = coord or [0, 0]
                self.page.mouse.click(x, y, button="right")
            elif kind == "triple_click":
                x, y = coord or [0, 0]
                self.page.mouse.click(x, y, click_count=3)
            elif kind == "middle_click":
                x, y = coord or [0, 0]
                self.page.mouse.click(x, y, button="middle")
            elif kind == "mouse_move":
                x, y = coord or [0, 0]
                self.page.mouse.move(x, y)
            elif kind == "left_mouse_down":
                self.page.mouse.down()
            elif kind == "left_mouse_up":
                self.page.mouse.up()
            elif kind == "left_click_drag":
                start = action.get("start_coordinate") or coord
                end = action.get("end_coordinate") or action.get("coordinate")
                if start and end:
                    self.page.mouse.move(start[0], start[1])
                    self.page.mouse.down()
                    self.page.mouse.move(end[0], end[1], steps=10)
                    self.page.mouse.up()
            elif kind == "type":
                self.page.keyboard.type(text or "", delay=15)
            elif kind == "key":
                # Anthropic uses xdotool key strings: "Return", "ctrl+a", "Tab", "Escape", "BackSpace"
                self.page.keyboard.press(_xdotool_key_to_playwright(text or ""))
            elif kind == "hold_key":
                key = _xdotool_key_to_playwright(text or "")
                duration = float(action.get("duration", 0.1))
                self.page.keyboard.down(key)
                time.sleep(duration)
                self.page.keyboard.up(key)
            elif kind == "scroll":
                x, y = coord or [self.width // 2, self.height // 2]
                self.page.mouse.move(x, y)
                direction = action.get("scroll_direction", "down")
                amount = int(action.get("scroll_amount", 3))
                dy = 100 * amount if direction == "down" else -100 * amount
                dx = 100 * amount if direction == "right" else -100 * amount if direction == "left" else 0
                if direction in ("up", "down"):
                    self.page.mouse.wheel(0, dy)
                else:
                    self.page.mouse.wheel(dx, 0)
            elif kind == "wait":
                time.sleep(float(action.get("duration", 1.0)))
            elif kind == "cursor_position":
                pass  # informational only
            else:
                return {"ok": False, "note": f"unsupported action: {kind}"}
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}


_KEY_MAP = {
    "Return": "Enter",
    "BackSpace": "Backspace",
    "Page_Up": "PageUp",
    "Page_Down": "PageDown",
    "space": "Space",
    "Escape": "Escape",
    "Tab": "Tab",
    "Up": "ArrowUp",
    "Down": "ArrowDown",
    "Left": "ArrowLeft",
    "Right": "ArrowRight",
    "ctrl": "Control",
    "shift": "Shift",
    "alt": "Alt",
    "super": "Meta",
    "cmd": "Meta",
}


def _xdotool_key_to_playwright(s: str) -> str:
    parts = [p.strip() for p in s.replace(" ", "+").split("+") if p.strip()]
    mapped = [_KEY_MAP.get(p, _KEY_MAP.get(p.lower(), p)) for p in parts]
    return "+".join(mapped) if len(mapped) > 1 else (mapped[0] if mapped else "")


# ---------- Anthropic loop ----------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_text(content_blocks: list[Any]) -> str:
    parts: list[str] = []
    for b in content_blocks or []:
        # SDK returns objects; both .type and dict access work
        t = getattr(b, "type", None) or (b.get("type") if isinstance(b, dict) else None)
        if t == "text":
            txt = getattr(b, "text", None) or (b.get("text") if isinstance(b, dict) else None)
            if txt:
                parts.append(txt)
    return "\n".join(parts)


def _try_parse_json(text: str) -> Any | None:
    """Parse a JSON object out of a string, tolerating wrapping prose / fences."""
    text = text.strip()
    # Direct
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fenced
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    # First { ... last }
    s = text.find("{")
    e = text.rfind("}")
    if s != -1 and e != -1 and e > s:
        try:
            return json.loads(text[s : e + 1])
        except Exception:
            pass
    return None


def run_loop(
    task: str,
    model: str,
    max_steps: int,
    output_dir: Path,
    headless: bool = False,
    extra_instruction: str | None = None,
    save_screenshots: bool = True,
) -> dict[str, Any]:
    cfg = TASK_CONFIG[task]
    spec_md = cfg["spec_path"].read_text()
    agent_prompt = _extract_agent_prompt(spec_md)
    if extra_instruction:
        agent_prompt = agent_prompt + "\n\n---\n" + extra_instruction
    start_url = cfg["start_url"]

    output_dir.mkdir(parents=True, exist_ok=True)
    traj_path = output_dir / "trajectory.jsonl"
    if traj_path.exists():
        traj_path.unlink()
    screenshots_dir = output_dir / "screenshots"

    def log(rec: dict[str, Any]) -> None:
        with traj_path.open("a") as f:
            f.write(json.dumps(rec, default=str) + "\n")

    log({
        "event": "task_start",
        "task": task,
        "model": model,
        "start_url": start_url,
        "max_steps": max_steps,
        "instruction": agent_prompt,
        "timestamp": _now(),
    })

    import anthropic
    client = anthropic.Anthropic()

    # System prompt: keep short; the agent prompt carries the task. Computer use
    # requires the model to know it's operating a real browser.
    system_blocks = [
        {
            "type": "text",
            "text": (
                "You are an AI agent operating a real Chromium browser at 1280x800 via the computer tool. "
                "The user will give you a task. Use the computer tool to navigate and interact, then return a single JSON object as your final answer. "
                "Be efficient: scroll the page to see content, click links to follow, type into focused inputs, and do not repeat the same action. "
                "When you finish, output the JSON answer in a final text-only message (no further tool calls)."
            ),
            "cache_control": {"type": "ephemeral"},
        }
    ]

    tool_def = {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1280,
        "display_height_px": 800,
    }
    betas = ["computer-use-2025-01-24"]

    final_answer_text: str | None = None
    final_answer_obj: Any = None
    final_step = 0

    with BrowserSession(headless=headless) as bs:
        bs.navigate(start_url)
        time.sleep(2.0)

        # Initial screenshot in the first user turn
        screenshot_b64 = bs.screenshot_b64()
        if save_screenshots:
            bs.screenshot_save(screenshots_dir / "step-000.png")

        log({"event": "navigate", "url": start_url, "timestamp": _now()})
        log({"event": "screenshot", "step": 0, "path": "screenshots/step-000.png", "timestamp": _now()})

        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": agent_prompt, "cache_control": {"type": "ephemeral"}},
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64},
                    },
                ],
            }
        ]

        for step in range(1, max_steps + 1):
            final_step = step
            try:
                response = client.beta.messages.create(
                    model=model,
                    max_tokens=4096,
                    system=system_blocks,
                    tools=[tool_def],
                    betas=betas,
                    messages=messages,
                )
            except Exception as exc:
                log({"event": "anthropic_error", "step": step, "error": str(exc), "timestamp": _now()})
                raise

            usage = getattr(response, "usage", None)
            log({
                "event": "model_response",
                "step": step,
                "stop_reason": getattr(response, "stop_reason", None),
                "input_tokens": getattr(usage, "input_tokens", None) if usage else None,
                "output_tokens": getattr(usage, "output_tokens", None) if usage else None,
                "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None) if usage else None,
                "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None) if usage else None,
                "timestamp": _now(),
            })

            # Build the assistant message to append (raw content)
            content_blocks_for_history = []
            tool_uses: list[Any] = []
            assistant_text_parts: list[str] = []

            for block in response.content or []:
                btype = getattr(block, "type", None)
                if btype == "text":
                    txt = getattr(block, "text", "") or ""
                    if txt:
                        assistant_text_parts.append(txt)
                        log({"event": "thinking", "step": step, "content": txt, "timestamp": _now()})
                    content_blocks_for_history.append({"type": "text", "text": txt})
                elif btype == "tool_use":
                    tool_uses.append(block)
                    inp = getattr(block, "input", None) or {}
                    log({
                        "event": "tool_use",
                        "step": step,
                        "tool": getattr(block, "name", "computer"),
                        "tool_use_id": getattr(block, "id", ""),
                        "input": inp,
                        "timestamp": _now(),
                    })
                    content_blocks_for_history.append(
                        {
                            "type": "tool_use",
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", "computer"),
                            "input": inp,
                        }
                    )

            messages.append({"role": "assistant", "content": content_blocks_for_history})

            # Stop conditions
            if getattr(response, "stop_reason", None) == "end_turn" and not tool_uses:
                final_answer_text = "\n".join(assistant_text_parts)
                final_answer_obj = _try_parse_json(final_answer_text)
                log({"event": "end_turn", "step": step, "final_text": final_answer_text, "timestamp": _now()})
                break

            if not tool_uses:
                # Stuck (no actions and not end_turn)
                log({"event": "no_tool_uses", "step": step, "timestamp": _now()})
                break

            # Execute every tool_use, build tool_result blocks
            tool_results = []
            for tu in tool_uses:
                inp = getattr(tu, "input", None) or {}
                exec_log = bs.execute_action(inp)
                # Wait briefly for layout to settle (esp. clicks/navigation)
                time.sleep(0.6)
                screenshot_b64 = bs.screenshot_b64()
                if save_screenshots:
                    bs.screenshot_save(screenshots_dir / f"step-{step:03d}.png")
                log({
                    "event": "executed",
                    "step": step,
                    "input": inp,
                    "result": exec_log,
                    "screenshot_path": f"screenshots/step-{step:03d}.png",
                    "timestamp": _now(),
                })
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": getattr(tu, "id", ""),
                        "content": [
                            {
                                "type": "image",
                                "source": {"type": "base64", "media_type": "image/png", "data": screenshot_b64},
                            }
                        ],
                    }
                )

            # Mark the last tool_result as a cache breakpoint to pin the prefix
            if tool_results:
                tool_results[-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
            messages.append({"role": "user", "content": tool_results})

    # Final write-out
    log({"event": "task_end", "step_count": final_step, "final_text": final_answer_text, "timestamp": _now()})

    if final_answer_obj is not None:
        (output_dir / "agent_output.json").write_text(json.dumps(final_answer_obj, indent=2) + "\n")
        print(f"\n==> Saved {output_dir / 'agent_output.json'}")
    else:
        print("\n==> No JSON answer parsed.")
        if final_answer_text:
            (output_dir / "agent_output_raw.txt").write_text(final_answer_text)
            print(f"   raw text saved to {output_dir / 'agent_output_raw.txt'}")

    return {
        "step_count": final_step,
        "final_answer": final_answer_obj,
        "final_text": final_answer_text,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["T2", "T3", "T4"], required=True)
    parser.add_argument("--model", default="claude-sonnet-4-6")
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--rollout-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--output-root", type=Path, default=ROOT / "bench" / "outputs" / "anthropic")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--no-save-screenshots", action="store_true")
    args = parser.parse_args()

    out_dir = args.output_root / args.task / args.model.replace("/", "_") / args.rollout_id
    print(f"==> Task {args.task} on {args.model} (max_steps={args.max_steps})")
    print(f"==> Output dir: {out_dir}")
    result = run_loop(
        task=args.task,
        model=args.model,
        max_steps=args.max_steps,
        output_dir=out_dir,
        headless=args.headless,
        save_screenshots=not args.no_save_screenshots,
    )
    print(f"==> Done in {result['step_count']} steps.")


if __name__ == "__main__":
    main()
