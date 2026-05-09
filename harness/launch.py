"""Buy-side bench matrix launcher.

Runs N trials of (task, model) on Kernel + the chosen CUA backend, scores each
trial against the calibrated ground truth in ground_truth/, and writes outputs
to outputs/jobs/<task>_<model>_<ts>/trial_NNN/.

Tasks are loaded from tasks/T<N>_*.md (the calibrated specs from
task-spec-calibration). Scoring uses scripts/scoring.py (Dimitry's official
scorer). Models are dispatched through harness/adapters/.

Usage:
  uv run python -u launch.py --task T2 --model openai --runs 20
  uv run python -u launch.py --task T3 --model northstar --runs 5
  uv run python -u launch.py --task T4 --model openai --runs 20 --max-steps 200

  # smoke
  uv run python -u launch.py --task T2 --model northstar --runs 1 --max-steps 60
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
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters import build_adapter, execute_action, ActionType, Action  # noqa: E402
from scoring import score_t2, score_t3, score_t4  # noqa: E402  (Dimitry's scorer)


# ----------------------------------------------------------- task config

VIEWPORT_W = 1280
VIEWPORT_H = 800

TASK_CONFIG: dict[str, dict[str, Any]] = {
    "T2": {
        "spec_md": "T2_restaurant_restock.md",
        "ground_truth": "T2_ground_truth.json",
        "start_url": "https://www.webstaurantstore.com/",
        "max_steps": 120,
        "scorer": "T2",
        "needs_trajectory": False,
    },
    "T3": {
        "spec_md": "T3_rei_hiking_boots.md",
        "ground_truth": "T3_ground_truth.json",
        "start_url": "https://www.rei.com/c/hiking-boots",
        "max_steps": 120,
        "scorer": "T3",
        "needs_trajectory": True,  # filter_engagement sub-score wants action list
    },
    "T4": {
        "spec_md": "T4_used_books_basket.md",
        "ground_truth": "T4_ground_truth.json",
        "start_url": "https://www.abebooks.com/",
        "max_steps": 200,
        "scorer": "T4",
        "needs_trajectory": False,
    },
}


def extract_agent_prompt(md_path: Path) -> str:
    """Pull the fenced code block under '## Agent prompt' from a task spec."""
    text = md_path.read_text()
    m = re.search(
        r"^##\s+Agent prompt\s*\n+```(?:\w*)?\n(.*?)```",
        text,
        re.DOTALL | re.MULTILINE,
    )
    return (m.group(1).strip() if m else text)


def extract_json_object(raw: str) -> dict | None:
    """Tolerant JSON extraction (matches scripts/scoring.py contract)."""
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
        ch = s[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[start : i + 1])
                except Exception:
                    return None
    return None


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _capture_screenshot(kernel: Kernel, session_id: str) -> bytes:
    resp = kernel.browsers.computer.capture_screenshot(session_id)
    raw = resp.read() if hasattr(resp, "read") else bytes(resp)
    img = Image.open(io.BytesIO(raw))
    if img.size != (VIEWPORT_W, VIEWPORT_H):
        img = img.resize((VIEWPORT_W, VIEWPORT_H), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ----------------------------------------------------- per-trial runner


def _ensure_keys(model: str) -> None:
    if not os.environ.get("KERNEL_API_KEY"):
        sys.exit("ERROR: KERNEL_API_KEY missing in .env")
    if model in ("northstar", "tzafon", "lightcone"):
        if not (os.environ.get("TZAFON_API_KEY") or os.environ.get("LIGHTCONE_API_KEY")):
            sys.exit("ERROR: TZAFON_API_KEY missing in .env")
    if model in ("gemini", "google"):
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            sys.exit("ERROR: GEMINI_API_KEY missing in .env")
    if model in ("openai", "gpt"):
        if not os.environ.get("OPENAI_API_KEY"):
            sys.exit("ERROR: OPENAI_API_KEY missing in .env")


def run_one_trial(
    *,
    task: str,
    model: str,
    model_id: str | None,
    trial_dir: Path,
    max_steps: int,
    start_url: str,
    instruction: str,
) -> dict:
    """Run a single trial and write all artifacts under `trial_dir`."""
    trial_dir.mkdir(parents=True, exist_ok=True)
    traj_path = trial_dir / "trajectory.jsonl"
    summary_path = trial_dir / "summary.json"

    kwargs: dict[str, Any] = {"viewport_w": VIEWPORT_W, "viewport_h": VIEWPORT_H}
    if model_id:
        kwargs["model_id"] = model_id
    adapter = build_adapter(model, **kwargs)

    kernel = Kernel(api_key=os.environ["KERNEL_API_KEY"])
    print(f"  [trial] adapter={adapter.name} model_id={adapter.model_id}")
    print(f"  [trial] creating Kernel browser (stealth, {VIEWPORT_W}x{VIEWPORT_H})...")
    browser = kernel.browsers.create(
        stealth=True,
        headless=False,
        timeout_seconds=1800,
        viewport={"width": VIEWPORT_W, "height": VIEWPORT_H},
    )
    session_id = browser.session_id
    cdp_ws = getattr(browser, "cdp_ws_url", None)
    live_view = getattr(browser, "browser_live_view_url", None)
    print(f"  [trial] session_id={session_id}")
    if live_view:
        print(f"  [trial] live_view={live_view}")

    final_status = "incomplete"
    final_message: str | None = None
    last_message: str | None = None
    page_url_final: str | None = None

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

        print(f"  [trial] navigating to {start_url}")
        page.goto(start_url, wait_until="domcontentloaded", timeout=60_000)
        time.sleep(2)

        png = _capture_screenshot(kernel, session_id)
        (trial_dir / "step_000_initial.png").write_bytes(png)
        page_url = page.url

        with traj_path.open("w") as traj:
            print("  [trial] sending initial request to model ...")
            step_resp = adapter.first_step(instruction, png, page_url)

            consecutive_no_action = 0
            MAX_NO_ACTION = 3
            step = 0
            while step < max_steps:
                if step_resp.message_text:
                    last_message = step_resp.message_text

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
                    traj.write(json.dumps({"step": step, "no_action": True, "message": step_resp.message_text}) + "\n")
                    traj.flush()
                    time.sleep(0.6)
                    png = _capture_screenshot(kernel, session_id)
                    (trial_dir / f"step_{step+1:03d}_nudge.png").write_bytes(png)
                    page_url = page.url
                    step_resp = adapter.next_step(
                        executed=[],
                        screenshot_png=png,
                        page_url=page_url,
                        nudge_text="Continue. Emit the next computer-use function call to make progress on the task.",
                    )
                    step += 1
                    continue

                consecutive_no_action = 0
                executed: list[tuple[Action, str]] = []
                terminal = False
                for a in step_resp.actions:
                    log = {
                        "step": step,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "model": adapter.name,
                        "action_type": a.type.value,
                        "x": a.x, "y": a.y, "text": a.text,
                        "keys": a.keys, "url": a.url,
                        "direction": a.direction, "delta_y": a.delta_y, "delta_x": a.delta_x,
                        "press_enter": a.press_enter, "clear_before": a.clear_before_typing,
                        "final_text": a.final_text,
                    }
                    try:
                        result = execute_action(kernel, session_id, page, a)
                    except Exception as e:
                        result = f"error:{e!r}"
                    log["result"] = result
                    log["page_url"] = page.url if page else None
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
                (trial_dir / f"step_{step+1:03d}.png").write_bytes(png)
                page_url = page.url
                step_resp = adapter.next_step(
                    executed=executed,
                    screenshot_png=png,
                    page_url=page_url,
                )
                step += 1
            else:
                final_status = "max_steps"
        page_url_final = page.url if page else None
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
            print(f"  [trial] cleanup error: {e}")

    candidate_text = final_message or last_message or ""
    parsed = extract_json_object(candidate_text) if candidate_text else None
    (trial_dir / "agent_response.raw.txt").write_text(candidate_text or "")
    if parsed is not None:
        (trial_dir / "agent_response.json").write_text(json.dumps(parsed, indent=2))

    # Score using Dimitry's calibrated scorer.
    cfg = TASK_CONFIG[task]
    gt_path = REPO_ROOT / "ground_truth" / cfg["ground_truth"]
    gt = json.loads(gt_path.read_text())
    score_dict: dict | None = None
    if parsed is not None:
        try:
            if cfg["scorer"] == "T2":
                score_dict = score_t2(parsed, gt)
            elif cfg["scorer"] == "T3":
                trajectory_actions: list[dict] = []
                if cfg.get("needs_trajectory") and traj_path.exists():
                    with traj_path.open() as f:
                        for line in f:
                            try:
                                row = json.loads(line)
                            except Exception:
                                continue
                            if "action_type" not in row:
                                continue
                            trajectory_actions.append(
                                {
                                    "target": str(row.get("action_type", "")),
                                    "url": row.get("page_url") or row.get("url") or "",
                                }
                            )
                score_dict = score_t3(parsed, gt, trajectory_actions)
            elif cfg["scorer"] == "T4":
                score_dict = score_t4(parsed, gt)
        except Exception as e:
            print(f"  [trial] scorer error: {e}")
            score_dict = None
    if score_dict is not None:
        (trial_dir / "score.json").write_text(json.dumps(score_dict, indent=2))
        print(f"  [trial] headline={score_dict.get('headline'):.3f}")
    elif parsed is None:
        print("  [trial] WARNING: agent did not emit parseable JSON; no score.")

    summary = {
        "task": task,
        "adapter": adapter.name,
        "model_id": adapter.model_id,
        "session_id": session_id,
        "live_view": live_view,
        "status": final_status,
        "max_steps": max_steps,
        "final_message_excerpt": (candidate_text[:500] if candidate_text else None),
        "parsed_json_present": parsed is not None,
        "page_url_final": page_url_final,
        "score": score_dict,
        "trial_dir": str(trial_dir),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return summary


# ------------------------------------------------------------- CLI


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["T2", "T3", "T4"], required=True)
    parser.add_argument(
        "--model",
        choices=["northstar", "openai", "gemini", "bedrock_claude"],
        required=True,
    )
    parser.add_argument("--model-id", default=None,
                        help="Override default model id for the chosen backend "
                             "(e.g. tzafon.northstar-cua-fast, gpt-5.5, "
                             "computer-use-preview, gemini-3-flash-preview).")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Override task default.")
    parser.add_argument("--out", type=Path, default=None,
                        help="Job dir. Default: outputs/jobs/<task>_<model>_<ts>")
    parser.add_argument("--start-url", default=None,
                        help="Override task default start URL.")
    args = parser.parse_args()

    _ensure_keys(args.model)

    cfg = TASK_CONFIG[args.task]
    spec_md = REPO_ROOT / "tasks" / cfg["spec_md"]
    if not spec_md.exists():
        sys.exit(f"ERROR: spec md not found: {spec_md}")
    instruction = extract_agent_prompt(spec_md)
    start_url = args.start_url or cfg["start_url"]
    max_steps = args.max_steps or cfg["max_steps"]

    job_dir = args.out or REPO_ROOT / "outputs" / "jobs" / f"{args.task.lower()}_{args.model}_{_now_id()}"
    job_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Launching {args.task} x {args.model} x {args.runs} trials ===")
    print(f"Job dir: {job_dir}")
    print(f"Spec: {spec_md.name}")
    print(f"Start URL: {start_url}")
    print(f"Max steps: {max_steps}")
    print()

    summaries: list[dict] = []
    for i in range(1, args.runs + 1):
        trial_dir = job_dir / f"trial_{i:03d}"
        print(f"--- Trial {i}/{args.runs} ---")
        t0 = time.time()
        summary = run_one_trial(
            task=args.task,
            model=args.model,
            model_id=args.model_id,
            trial_dir=trial_dir,
            max_steps=max_steps,
            start_url=start_url,
            instruction=instruction,
        )
        elapsed = time.time() - t0
        summary["elapsed_seconds"] = elapsed
        summaries.append(summary)
        headline = (summary.get("score") or {}).get("headline")
        hl = f"{headline:.3f}" if isinstance(headline, (int, float)) else "—"
        print(f"--- Trial {i}/{args.runs} done | status={summary['status']} headline={hl} elapsed={elapsed:.1f}s ---")
        print()

    # Job-level aggregate.
    headlines = [s.get("score", {}).get("headline") for s in summaries if s.get("score")]
    headlines = [h for h in headlines if isinstance(h, (int, float))]
    aggregate = {
        "task": args.task,
        "model": args.model,
        "model_id": args.model_id,
        "n_trials": args.runs,
        "n_with_score": len(headlines),
        "n_with_json": sum(1 for s in summaries if s.get("parsed_json_present")),
        "headline_mean": sum(headlines) / len(headlines) if headlines else None,
        "headlines": headlines,
        "started_at": summaries[0].get("session_id"),  # not great but available
        "trial_dirs": [s["trial_dir"] for s in summaries],
    }
    (job_dir / "job_summary.json").write_text(json.dumps(aggregate, indent=2, default=str))

    print("=== Job complete ===")
    print(json.dumps(aggregate, indent=2, default=str))
    print()
    print(f"Build harbor job (run from harness/):")
    print(
        f"  uv run python to_harbor.py build "
        f"--job-name buy_side_{args.task.lower()}_{args.model} "
        f"--task-name {args.task} --out ../outputs/harbor/{args.task.lower()}_{args.model} "
        + " ".join(f"--run {args.model}={t}" for t in [s['trial_dir'] for s in summaries])
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
