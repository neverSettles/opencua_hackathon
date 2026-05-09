"""
Run T4 (used books basket) end-to-end on Kernel + a pluggable model backend.

Reads:
  - tasks/T4_used_books_basket.md      (extracts the fenced "Agent prompt" block)
  - ground_truth/T4_ground_truth.json

Usage:
  uv run python -u run_t4_spike.py --model anthropic --anthropic-model claude-opus-4-7
  uv run python -u run_t4_spike.py --model openai --openai-model gpt-5.5
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from kernel import Kernel
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapters import build_adapter, execute_action, ActionType, Action  # noqa: E402
from bench.evaluators.t2_restaurant_restock import extract_json_object  # noqa: E402
from scripts.scoring import score_t4  # noqa: E402

VIEWPORT_W = 1280
VIEWPORT_H = 800
START_URL = "https://www.abebooks.com/"
TASK_SPEC_PATH = REPO_ROOT / "tasks" / "T4_used_books_basket.md"
GT_PATH = REPO_ROOT / "ground_truth" / "T4_ground_truth.json"
WORLD_ID = "used_books_v1"
TASK_ID = "T4-001"


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _extract_agent_prompt(spec_md: str) -> str:
    m = re.search(r"## Agent prompt\s*\n+```\w*\n(.*?)\n```", spec_md, re.S)
    if not m:
        raise RuntimeError("Could not find fenced agent prompt block in T4 spec")
    return m.group(1).strip()


def _capture_screenshot(kernel: Kernel, session_id: str) -> bytes:
    resp = kernel.browsers.computer.capture_screenshot(session_id)
    raw = resp.read() if hasattr(resp, "read") else bytes(resp)
    img = Image.open(io.BytesIO(raw))
    if img.size != (VIEWPORT_W, VIEWPORT_H):
        img = img.resize((VIEWPORT_W, VIEWPORT_H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _ensure_keys(model: str) -> None:
    if not os.environ.get("KERNEL_API_KEY"):
        sys.exit("ERROR: KERNEL_API_KEY missing in .env")
    if model in ("openai", "gpt") and not os.environ.get("OPENAI_API_KEY"):
        sys.exit("ERROR: OPENAI_API_KEY missing in .env")
    if model in ("anthropic", "claude") and not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ERROR: ANTHROPIC_API_KEY missing in .env")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="anthropic", choices=["anthropic", "openai"])
    parser.add_argument("--anthropic-model", default="claude-opus-4-7")
    parser.add_argument("--openai-model", default="gpt-5.5")
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--start-url", default=START_URL)
    args = parser.parse_args()

    _ensure_keys(args.model)
    if args.model == "anthropic":
        adapter = build_adapter("anthropic", model_id=args.anthropic_model,
                                viewport_w=VIEWPORT_W, viewport_h=VIEWPORT_H)
    else:
        adapter = build_adapter("openai", model_id=args.openai_model,
                                viewport_w=VIEWPORT_W, viewport_h=VIEWPORT_H)

    instruction = _extract_agent_prompt(TASK_SPEC_PATH.read_text())
    gt = json.loads(GT_PATH.read_text()) if GT_PATH.exists() else None

    run_id = f"t4_{adapter.name}_{_now_id()}"
    out_dir = REPO_ROOT / "outputs" / run_id / WORLD_ID / TASK_ID
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectory.jsonl"
    summary_path = out_dir / "summary.json"
    print(f"Run dir: {out_dir}")
    print(f"Adapter: {adapter.name} model_id={adapter.model_id}")
    print(f"Mode: T4 used books | max_steps={args.max_steps}")

    kernel = Kernel(api_key=os.environ["KERNEL_API_KEY"])
    print(f"Creating Kernel browser ...")
    browser = kernel.browsers.create(
        stealth=True, headless=False, timeout_seconds=2400,
        viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
    )
    session_id = browser.session_id
    cdp_ws = getattr(browser, "cdp_ws_url", None)
    live_view = getattr(browser, "browser_live_view_url", None)
    print(f"  session_id={session_id}")
    if live_view:
        print(f"  live_view={live_view}")

    final_status = "incomplete"
    final_message: str | None = None
    last_message: str | None = None
    pw = None
    chromium = None
    page = None
    try:
        from playwright.sync_api import sync_playwright
        pw = sync_playwright().start()
        chromium = pw.chromium.connect_over_cdp(cdp_ws)
        ctx = chromium.contexts[0] if chromium.contexts else chromium.new_context()
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        try:
            page.set_viewport_size({"width": VIEWPORT_W, "height": VIEWPORT_H})
        except Exception:
            pass

        print(f"Navigating to {args.start_url} ...")
        page.goto(args.start_url, wait_until="domcontentloaded", timeout=60_000)
        time.sleep(2)

        png = _capture_screenshot(kernel, session_id)
        (out_dir / "step_000_initial.png").write_bytes(png)
        page_url = page.url

        with traj_path.open("w") as traj:
            print("Sending initial request ...")
            step_resp = adapter.first_step(instruction, png, page_url)

            consecutive_no_action = 0
            MAX_NO_ACTION = 3
            step = 0
            while step < args.max_steps:
                if step_resp.message_text:
                    last_message = step_resp.message_text
                    short = step_resp.message_text[:300].replace("\n", " ")
                    print(f"[step {step:02d}] msg: {short}")

                if not step_resp.actions:
                    parsed_now = extract_json_object(step_resp.message_text or "")
                    if parsed_now is not None:
                        print(f"[step {step:02d}] message contains JSON -> done.")
                        final_status = "json_in_message"
                        final_message = step_resp.message_text
                        traj.write(json.dumps({"step": step, "message": step_resp.message_text, "terminal": True}) + "\n")
                        break

                    consecutive_no_action += 1
                    if consecutive_no_action > MAX_NO_ACTION:
                        print(f"[step {step:02d}] {consecutive_no_action} no-action turns; stopping.")
                        final_status = "stopped_no_action"
                        final_message = step_resp.message_text
                        traj.write(json.dumps({"step": step, "message": step_resp.message_text, "terminal": True}) + "\n")
                        break

                    print(f"[step {step:02d}] no action; nudging ({consecutive_no_action}).")
                    traj.write(json.dumps({"step": step, "no_action": True, "message": step_resp.message_text}) + "\n")
                    traj.flush()
                    time.sleep(0.6)
                    png = _capture_screenshot(kernel, session_id)
                    (out_dir / f"step_{step+1:03d}_nudge.png").write_bytes(png)
                    page_url = page.url
                    step_resp = adapter.next_step(
                        executed=[], screenshot_png=png, page_url=page_url,
                        nudge_text="Continue. Emit the next computer-use call to make progress.",
                    )
                    step += 1
                    continue

                consecutive_no_action = 0
                executed: list[tuple[Action, str]] = []
                terminal = False
                for a in step_resp.actions:
                    log = {
                        "step": step, "ts": datetime.now(timezone.utc).isoformat(),
                        "model": adapter.name, "action_type": a.type.value,
                        "x": a.x, "y": a.y, "text": a.text, "keys": a.keys,
                        "url": a.url, "direction": a.direction,
                        "delta_y": a.delta_y, "delta_x": a.delta_x,
                        "press_enter": a.press_enter,
                        "clear_before": a.clear_before_typing,
                        "final_text": a.final_text,
                    }
                    try:
                        result = execute_action(kernel, session_id, page, a)
                    except Exception as e:
                        result = f"error:{e!r}"
                    log["result"] = result
                    print(f"[step {step:02d}] {log}")
                    traj.write(json.dumps(log) + "\n")
                    traj.flush()
                    executed.append((a, result))
                    if a.type in (ActionType.DONE, ActionType.TERMINATE):
                        terminal = True
                        final_status = "terminal"
                        final_message = a.final_text or last_message
                        break

                if terminal:
                    break

                time.sleep(1.0)
                png = _capture_screenshot(kernel, session_id)
                (out_dir / f"step_{step+1:03d}.png").write_bytes(png)
                page_url = page.url
                step_resp = adapter.next_step(executed=executed,
                                              screenshot_png=png, page_url=page_url)
                step += 1
            else:
                final_status = "max_steps"
                print(f"Hit max_steps={args.max_steps}.")

    except Exception:
        traceback.print_exc()
        final_status = "exception"
    finally:
        try:
            if chromium is not None:
                chromium.close()
        except Exception:
            pass
        try:
            if pw is not None:
                pw.stop()
        except Exception:
            pass
        try:
            kernel.browsers.delete_by_id(session_id)
        except Exception as e:
            print(f"(cleanup) browser delete error: {e}")

    candidate_text = final_message or last_message or ""
    parsed = extract_json_object(candidate_text) if candidate_text else None
    (out_dir / "agent_response.raw.txt").write_text(candidate_text or "")
    if parsed is not None:
        (out_dir / "agent_response.json").write_text(json.dumps(parsed, indent=2))

    score_dict = None
    if gt is not None and parsed is not None:
        s = score_t4(parsed, gt)
        score_dict = s
        (out_dir / "score.json").write_text(json.dumps(s, indent=2))

    summary = {
        "run_id": run_id, "world_id": WORLD_ID, "task_id": TASK_ID,
        "adapter": adapter.name, "model_id": adapter.model_id,
        "session_id": session_id, "live_view": live_view,
        "status": final_status, "max_steps": args.max_steps,
        "final_message_excerpt": (candidate_text[:500] if candidate_text else None),
        "parsed_json_present": parsed is not None,
        "headline": (score_dict["headline"] if score_dict else None),
        "out_dir": str(out_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
