"""
Generalized eval runner: T2/T3/T4 × any model adapter × N trials.

Usage:
  uv run python -u run_eval.py --task T2 --model openai --runs 1
  uv run python -u run_eval.py --task T3 --model openai --runs 20
  uv run python -u run_eval.py --task T4 --model openai --openai-model gpt-5.5 --runs 20

Outputs land in:
  outputs/jobs/{task}_{model}/trial_{NNN}/
    summary.json, trajectory.jsonl, agent_response.json, score.json,
    step_NNN.png
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
from typing import Any

from dotenv import load_dotenv
from kernel import Kernel
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import build_adapter, execute_action, ActionType, Action  # noqa: E402

VIEWPORT_W = 1280
VIEWPORT_H = 800

TASK_CONFIG = {
    "T2": {
        "md": "T2_restaurant_restock.md",
        "gt": "T2_ground_truth.json",
        "start_url": "https://www.webstaurantstore.com/",
        "max_steps": 120,
        "timeout_seconds": 1500,
    },
    "T3": {
        "md": "T3_rei_hiking_boots.md",
        "gt": "T3_ground_truth.json",
        "start_url": "https://www.rei.com/",
        "max_steps": 120,
        "timeout_seconds": 1500,
    },
    "T4": {
        "md": "T4_used_books_basket.md",
        "gt": "T4_ground_truth.json",
        "start_url": "https://www.abebooks.com/",
        "max_steps": 200,
        "timeout_seconds": 1800,
    },
}


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def extract_agent_prompt(md_text: str) -> str:
    m = re.search(r"^## Agent prompt\s*\n+```(?:\w*)?\n(.*?)```", md_text, re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else md_text


def extract_json_object(raw: str) -> dict | None:
    if not raw:
        return None
    s = raw.strip()
    try:
        return json.loads(s)
    except Exception:
        pass
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except Exception:
                    return None
    return None


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
    if model in ("northstar",):
        if not os.environ.get("TZAFON_API_KEY") and not os.environ.get("LIGHTCONE_API_KEY"):
            sys.exit("ERROR: TZAFON_API_KEY missing in .env")
    if model in ("openai",):
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("ERROR: OPENAI_API_KEY missing in .env")


def run_one_trial(
    task: str,
    adapter,
    instruction: str,
    gt: dict,
    start_url: str,
    max_steps: int,
    timeout_seconds: int,
    out_dir: Path,
    scorer,
    trial_num: int,
) -> dict:
    """Execute one end-to-end trial. Returns the summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "trajectory.jsonl"
    summary_path = out_dir / "summary.json"

    kernel = Kernel(api_key=os.environ["KERNEL_API_KEY"])

    print(f"\n{'='*60}")
    print(f"[Trial {trial_num}] {task} / {adapter.name} ({adapter.model_id})")
    print(f"  out: {out_dir}")
    browser = kernel.browsers.create(
        stealth=True,
        headless=False,
        timeout_seconds=timeout_seconds,
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
    total_actions = 0
    start_time = time.monotonic()

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

        print(f"  navigating to {start_url} ...")
        page.goto(start_url, wait_until="domcontentloaded", timeout=60_000)
        time.sleep(2)

        png = _capture_screenshot(kernel, session_id)
        (out_dir / "step_000_initial.png").write_bytes(png)
        page_url = page.url

        with traj_path.open("w") as traj:
            step_resp = adapter.first_step(instruction, png, page_url)
            consecutive_no_action = 0
            MAX_NO_ACTION = 3
            step = 0

            while step < max_steps:
                if step_resp.message_text:
                    last_message = step_resp.message_text
                    if step % 10 == 0 or not step_resp.actions:
                        short = step_resp.message_text[:200].replace("\n", " ")
                        print(f"  [step {step:03d}] msg: {short}")

                if not step_resp.actions:
                    parsed_now = extract_json_object(step_resp.message_text or "")
                    if parsed_now is not None:
                        final_status = "json_in_message"
                        final_message = step_resp.message_text
                        traj.write(json.dumps({"step": step, "message": step_resp.message_text, "terminal": True}) + "\n")
                        break
                    consecutive_no_action += 1
                    if consecutive_no_action > MAX_NO_ACTION:
                        final_status = "stopped_no_action"
                        final_message = step_resp.message_text
                        traj.write(json.dumps({"step": step, "message": step_resp.message_text, "terminal": True}) + "\n")
                        break
                    traj.write(json.dumps({"step": step, "no_action": True}) + "\n")
                    traj.flush()
                    time.sleep(0.6)
                    png = _capture_screenshot(kernel, session_id)
                    (out_dir / f"step_{step+1:03d}_nudge.png").write_bytes(png)
                    page_url = page.url
                    step_resp = adapter.next_step(
                        executed=[], screenshot_png=png, page_url=page_url,
                        nudge_text="Continue. Emit the next computer-use action to make progress.",
                    )
                    step += 1
                    continue

                consecutive_no_action = 0
                executed: list[tuple[Action, str]] = []
                terminal = False
                for a in step_resp.actions:
                    total_actions += 1
                    log = {
                        "step": step,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "action_type": a.type.value,
                        "x": a.x, "y": a.y, "text": a.text,
                        "keys": a.keys, "url": a.url,
                        "direction": a.direction,
                        "delta_y": a.delta_y, "delta_x": a.delta_x,
                        "press_enter": a.press_enter,
                        "final_text": a.final_text,
                    }
                    try:
                        result = execute_action(kernel, session_id, page, a)
                    except Exception as e:
                        result = f"error:{e!r}"
                    log["result"] = result
                    traj.write(json.dumps(log) + "\n")
                    traj.flush()
                    executed.append((a, result))
                    if a.type in (ActionType.DONE, ActionType.TERMINATE):
                        terminal = True
                        final_status = (a.correlation or {}).get("status") or "terminal"
                        final_message = a.final_text or last_message
                        break

                if terminal:
                    break
                time.sleep(1.0)
                png = _capture_screenshot(kernel, session_id)
                (out_dir / f"step_{step+1:03d}.png").write_bytes(png)
                page_url = page.url
                step_resp = adapter.next_step(
                    executed=executed, screenshot_png=png, page_url=page_url,
                )
                step += 1
            else:
                final_status = "max_steps"

    except Exception:
        traceback.print_exc()
        final_status = "exception"
    finally:
        for cleanup in [
            lambda: chromium and chromium.close(),
            lambda: pw and pw.stop(),
            lambda: kernel.browsers.delete_by_id(session_id),
        ]:
            try:
                cleanup()
            except Exception:
                pass

    elapsed = time.monotonic() - start_time
    candidate_text = final_message or last_message or ""
    parsed = extract_json_object(candidate_text) if candidate_text else None
    (out_dir / "agent_response.raw.txt").write_text(candidate_text or "")
    if parsed is not None:
        (out_dir / "agent_response.json").write_text(json.dumps(parsed, indent=2))

    score_dict: dict | None = None
    if parsed is not None and scorer is not None:
        try:
            if task == "T3":
                traj_actions = []
                if traj_path.exists():
                    for line in traj_path.read_text().strip().split("\n"):
                        try:
                            traj_actions.append(json.loads(line))
                        except Exception:
                            pass
                score_dict = scorer(parsed, gt, traj_actions)
            else:
                score_dict = scorer(parsed, gt)
        except Exception as e:
            print(f"  ! scoring error: {e}")
    if score_dict is not None:
        (out_dir / "score.json").write_text(json.dumps(score_dict, indent=2))

    summary = {
        "task": task,
        "adapter": adapter.name,
        "model_id": adapter.model_id,
        "trial": trial_num,
        "session_id": session_id,
        "live_view": live_view,
        "status": final_status,
        "total_steps": step if 'step' in dir() else 0,
        "total_actions": total_actions,
        "elapsed_seconds": round(elapsed, 1),
        "parsed_json_present": parsed is not None,
        "headline": score_dict.get("headline") if score_dict else None,
        "score": score_dict,
        "out_dir": str(out_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    hl = score_dict.get("headline", "n/a") if score_dict else "n/a"
    print(f"  [Trial {trial_num}] DONE  status={final_status}  steps={summary['total_steps']}  actions={total_actions}  headline={hl}  elapsed={elapsed:.0f}s")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run eval trials: T2/T3/T4 × model × N runs")
    parser.add_argument("--task", required=True, choices=["T2", "T3", "T4"])
    parser.add_argument("--model", default="openai", choices=["northstar", "openai", "gemini"])
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--northstar-model", default="tzafon.northstar-cua-fast")
    parser.add_argument("--openai-model", default="gpt-5.5")
    parser.add_argument("--gemini-model", default="gemini-3-pro-preview")
    parser.add_argument("--max-steps", type=int, default=None, help="Override default max_steps")
    parser.add_argument("--start-trial", type=int, default=1, help="Start numbering at this trial (for resumption)")
    args = parser.parse_args()

    _ensure_keys(args.model)

    tc = TASK_CONFIG[args.task]
    max_steps = args.max_steps or tc["max_steps"]
    start_url = tc["start_url"]
    timeout_seconds = tc["timeout_seconds"]

    md_path = REPO_ROOT / "tasks" / tc["md"]
    gt_path = REPO_ROOT / "ground_truth" / tc["gt"]
    instruction = extract_agent_prompt(md_path.read_text())
    gt = json.loads(gt_path.read_text())

    # Import scorer
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from scoring import score_t2, score_t3, score_t4
    scorer = {"T2": score_t2, "T3": score_t3, "T4": score_t4}[args.task]

    job_name = f"{args.task.lower()}_{args.model}"
    job_dir = REPO_ROOT / "outputs" / "jobs" / job_name

    print(f"Job: {job_name}")
    print(f"Task: {args.task} | Model: {args.model} | Runs: {args.runs}")
    print(f"Prompt: {instruction[:120]}...")
    print(f"GT: {gt_path}")
    print(f"Output: {job_dir}")

    summaries = []
    for i in range(args.runs):
        trial_num = args.start_trial + i
        trial_dir = job_dir / f"trial_{trial_num:03d}"

        model_kwargs = {"viewport_w": VIEWPORT_W, "viewport_h": VIEWPORT_H}
        if args.model == "northstar":
            model_kwargs["model_id"] = args.northstar_model
        elif args.model == "openai":
            model_kwargs["model_id"] = args.openai_model
        elif args.model == "gemini":
            model_kwargs["model_id"] = args.gemini_model
        adapter = build_adapter(args.model, **model_kwargs)

        s = run_one_trial(
            task=args.task,
            adapter=adapter,
            instruction=instruction,
            gt=gt,
            start_url=start_url,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
            out_dir=trial_dir,
            scorer=scorer,
            trial_num=trial_num,
        )
        summaries.append(s)

    # Write job-level summary
    headlines = [s["headline"] for s in summaries if s["headline"] is not None]
    passed = [s for s in summaries if s["status"] in ("json_in_message", "terminal") and s["parsed_json_present"]]
    job_summary = {
        "job_name": job_name,
        "task": args.task,
        "model": args.model,
        "model_id": summaries[0]["model_id"] if summaries else None,
        "total_trials": len(summaries),
        "completed_trials": len(passed),
        "pass_rate": len(passed) / len(summaries) if summaries else 0,
        "mean_headline": sum(headlines) / len(headlines) if headlines else None,
        "min_headline": min(headlines) if headlines else None,
        "max_headline": max(headlines) if headlines else None,
        "per_trial": [
            {"trial": s["trial"], "status": s["status"], "headline": s["headline"], "elapsed": s["elapsed_seconds"]}
            for s in summaries
        ],
    }
    (job_dir / "job_summary.json").write_text(json.dumps(job_summary, indent=2))
    print(f"\n{'='*60}")
    print(f"JOB COMPLETE: {job_name}")
    print(json.dumps(job_summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
