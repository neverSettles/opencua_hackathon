"""Run T2 (restaurant restock) on live WebstaurantStore using the vendored
tzafon CuaRunner over a Kernel-backed browser.

Flow:
  1. Read intent.md + ground_truth.json from bench/worlds/webstaurant_v1/tasks/T2-001/
  2. Spin up Kernel browser via KernelComputerAdapter (stealth, viewport-matched)
  3. Drive screenshot-action loop via CuaRunner (circuit breaker, click-before-type, hooks)
  4. Parse final message as JSON and run the T2 scorer
  5. Save trajectory, screenshots, agent_response.json, score.json

Usage:
  uv run python -u run_t2.py
  uv run python -u run_t2.py --max-steps 200
  uv run python -u run_t2.py --runs 3        # reliability check
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from kernel import Kernel
from tzafon import Lightcone

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT))
from bench.evaluators.t2_restaurant_restock import score_t2, extract_json_object  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cua_runner import (  # noqa: E402
    ActionEvent,
    CuaRunner,
    MessageEvent,
    RunConfig,
    RunResult,
)
from kernel_computer import KernelComputerAdapter  # noqa: E402

WORLD_ID = "webstaurant_v1"
TASK_ID = "T2-001"
TASK_DIR = REPO_ROOT / "bench" / "worlds" / WORLD_ID / "tasks" / TASK_ID

VIEWPORT_W = 1280
VIEWPORT_H = 720  # match tzafon's harness default to keep their tips applicable
START_URL = "https://www.webstaurantstore.com/"


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_action_printer():
    def cb(event: ActionEvent) -> None:
        a = event.action
        bits: list[str] = [a.type]
        for k in ("x", "y", "end_x", "end_y", "scroll_x", "scroll_y", "url", "text", "keys"):
            v = getattr(a, k, None)
            if v not in (None, "", []):
                if k == "text" and isinstance(v, str):
                    v = v[:60]
                bits.append(f"{k}={v}")
        print(f"  [step {event.step:03d}] {' '.join(bits)}", flush=True)

    return cb


# Coordinates of WebstaurantStore search box at 1280x720 viewport, in 0-999 model
# space (used to focus the search box when Northstar emits a coord-less `type`).
WSS_SEARCH_BOX_NORM = (469, 173)  # ~ pixel (600, 125) at 1280x720


def make_focus_search_box_hook(adapter, viewport_w: int, viewport_h: int):
    """Click the search box first if `type` arrives without coords. Otherwise
    Kernel routes the keystrokes to whatever is focused (often the URL bar)."""
    from cua_runner import ActionDecision  # noqa: PLC0415

    def hook(event: ActionEvent):
        a = event.action
        if a.type == "type" and (getattr(a, "x", None) is None or getattr(a, "y", None) is None):
            sx = int(WSS_SEARCH_BOX_NORM[0] / 1000 * viewport_w)
            sy = int(WSS_SEARCH_BOX_NORM[1] / 1000 * viewport_h)
            print(f"  [hook] focusing WSS search box at ({sx}, {sy}) before type", flush=True)
            try:
                adapter.click(sx, sy)
                time.sleep(0.3)
            except Exception as e:
                print(f"  [hook] focus click failed: {e}")
        return ActionDecision()

    return hook


def make_message_printer():
    def cb(event: MessageEvent) -> None:
        text = event.text.replace("\n", " ").strip()
        print(f"  [step {event.step:03d}] msg: {text[:240]}", flush=True)

    return cb


def run_once(
    *,
    kernel: Kernel,
    lc: Lightcone,
    instruction: str,
    out_dir: Path,
    max_steps: int,
) -> tuple[RunResult, str | None]:
    out_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = out_dir / "screenshots"
    screenshots_dir.mkdir(exist_ok=True)
    trace_path = out_dir / "trace.jsonl"

    config = RunConfig(
        model="tzafon.northstar-cua-fast",
        kind="browser",
        display_width=VIEWPORT_W,
        display_height=VIEWPORT_H,
        max_steps=max_steps,
        step_delay_seconds=0.8,
        wait_action_seconds=2.0,
        initial_navigation_delay_seconds=2.5,
        circuit_breaker_threshold=3,
        budget_warning_pct=0.7,
        trace_path=str(trace_path),
        print_progress=True,
        system_instructions=(
            "You are operating a Chromium tab on www.webstaurantstore.com to complete "
            "a procurement workflow. "
            "Tips: "
            "Press Enter (key action with keys=['enter']) to submit a search after typing — "
            "do NOT try to click a magnifying-glass icon. "
            "If a popup or cookie banner appears, dismiss it first. "
            "To scroll the main page, scroll at the far left edge (x=20). "
            "When you have completed all 6 items and confirmed the cart, emit an `answer` "
            "action whose `result` field contains ONLY the JSON object specified in the "
            "task instructions (no prose, no code fences). "
            "If you get stuck repeating an action, try a completely different approach — "
            "different search terms, different category navigation, or skip the item and "
            "record it as a failure."
        ),
    )

    adapter = KernelComputerAdapter(
        kernel=kernel,
        viewport_width=VIEWPORT_W,
        viewport_height=VIEWPORT_H,
        stealth=True,
        screenshot_dir=screenshots_dir,
        verbose=True,
    )

    with adapter as computer:
        runner = CuaRunner(
            lc,
            config=config,
            on_action=make_action_printer(),
            on_message=make_message_printer(),
            before_action=make_focus_search_box_hook(adapter, VIEWPORT_W, VIEWPORT_H),
        )
        result = runner.run(instruction, start_url=START_URL, computer=computer)
        live_view = adapter.live_view_url
    return result, live_view


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated runs")
    args = parser.parse_args()

    kernel_key = os.environ.get("KERNEL_API_KEY")
    tzafon_key = os.environ.get("TZAFON_API_KEY") or os.environ.get("LIGHTCONE_API_KEY")
    if not kernel_key:
        print("ERROR: KERNEL_API_KEY missing in .env", file=sys.stderr)
        return 1
    if not tzafon_key:
        print("ERROR: TZAFON_API_KEY missing in .env", file=sys.stderr)
        return 1
    os.environ["TZAFON_API_KEY"] = tzafon_key

    instruction = (TASK_DIR / "intent.md").read_text()
    gt = json.loads((TASK_DIR / "ground_truth.json").read_text())

    kernel = Kernel(api_key=kernel_key)
    lc = Lightcone(api_key=tzafon_key)

    base_run_id = f"t2_{_now_id()}"
    summaries: list[dict] = []

    for run_num in range(1, args.runs + 1):
        run_id = base_run_id if args.runs == 1 else f"{base_run_id}_run{run_num}"
        out_dir = REPO_ROOT / "outputs" / run_id / WORLD_ID / TASK_ID
        print()
        print(f"=== Run {run_num}/{args.runs}  ({run_id}) ===")
        print(f"Out dir: {out_dir}")
        print(f"Max steps: {args.max_steps}")

        start = time.time()
        result, live_view = run_once(
            kernel=kernel,
            lc=lc,
            instruction=instruction,
            out_dir=out_dir,
            max_steps=args.max_steps,
        )
        elapsed = time.time() - start

        print()
        print(
            f"  status={result.status} steps={result.steps} elapsed={elapsed:.1f}s"
        )
        if result.final_message:
            print(f"  final_message[:240]: {result.final_message[:240]}")

        # Parse + score
        candidate = result.final_message or ""
        (out_dir / "agent_response.raw.txt").write_text(candidate)
        parsed = extract_json_object(candidate) if candidate else None
        score_dict: dict | None = None
        if parsed is not None:
            (out_dir / "agent_response.json").write_text(json.dumps(parsed, indent=2))
            s = score_t2(parsed, gt)
            score_dict = s.to_dict()
            (out_dir / "score.json").write_text(json.dumps(score_dict, indent=2))
            print(f"  headline score: {score_dict['headline']:.3f}")
        else:
            print("  WARNING: agent did not emit parseable JSON; cannot score.")

        summary = {
            "run_id": run_id,
            "world_id": WORLD_ID,
            "task_id": TASK_ID,
            "session_id": result.computer_id,
            "live_view": live_view,
            "status": result.status,
            "steps": result.steps,
            "elapsed_seconds": elapsed,
            "final_message_excerpt": candidate[:500] if candidate else None,
            "parsed_json_present": parsed is not None,
            "score": (
                {
                    "headline": score_dict["headline"],
                    "valid_json": score_dict["valid_json"],
                    "item_match_mean": score_dict["item_match_mean"],
                    "pack_extraction_mean": score_dict["pack_extraction_mean"],
                    "quantity_correctness_mean": score_dict["quantity_correctness_mean"],
                    "cart_added_mean": score_dict["cart_added_mean"],
                    "completeness": score_dict["completeness"],
                    "notes": score_dict["notes"],
                }
                if score_dict
                else None
            ),
            "out_dir": str(out_dir),
        }
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)

    print()
    print("=== Aggregate ===")
    print(json.dumps(summaries, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
