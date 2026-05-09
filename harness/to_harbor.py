"""Convert our run outputs into Harbor / ATIF-v1.6 trajectory job dirs for
upload to trajectories.sh.

Two source formats supported:

  (a) "vendored CuaRunner" runs (e.g. outputs/t2_<ts>/...): produced by
      harness/run_t2.py. Has trace.jsonl with structured events
      {run_started, screenshot_captured, message_received, action_selected,
       circuit_breaker, no_action_retry, run_completed}, plus
      screenshots/step_NNN.png and summary.json.

  (b) "adapter" runs (e.g. outputs/t2_openai_<ts>/...): produced by the
      teammate's harness/run_t2_spike.py with the adapters/ system. Has
      trajectory.jsonl with one row per granular action (action_type, x, y,
      text, keys, url, ...), plus step_NNN.png in the trial dir.

Output:

  <job_dir>/
    config.json                 # {job_name, agents, datasets, tasks}
    result.json                 # aggregate {n_total_trials, stats: {...}}
    <agent>__<short>/
      config.json
      result.json               # per-trial metadata + verifier_result
      agent/
        trajectory.json         # ATIF-v1.6
        screenshots/
          step-00.jpg ... step-NN.jpg

Usage:

  uv run python -m harness.to_harbor build \\
      --job-name t2_webstaurant_restock \\
      --task-name T2-001 \\
      --out outputs/harbor/<job_name> \\
      --run northstar=outputs/t2_20260509T221703Z/webstaurant_v1/T2-001 \\
      --run openai=outputs/t2_openai_20260509T222050Z/webstaurant_v1/T2-001

  npx --yes trajectories-sh auth login        # one-time
  npx --yes trajectories-sh upload trajectory outputs/harbor/<job_name>
"""

from __future__ import annotations

import argparse
import io
import json
import re
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent

ISO_FALLBACK = "1970-01-01T00:00:00Z"


# ----------------------------------------------------------- helpers

def _png_to_jpg_bytes(png_path: Path) -> bytes:
    """Read a PNG and return JPEG bytes (matches harbor convention)."""
    img = Image.open(png_path).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85, optimize=True)
    return buf.getvalue()


def _ensure_iso(ts: str | None) -> str:
    if not ts:
        return ISO_FALLBACK
    if ts.endswith("Z"):
        return ts
    if "+" not in ts and "-" not in ts.split("T")[-1]:
        return ts + "Z"
    return ts.replace("+00:00", "Z")


def _short_id() -> str:
    return uuid.uuid4().hex[:7]


def _trial_id_for(agent: str) -> str:
    return f"{agent}__{_short_id()}"


# --------------------------------------------------------- vendored runner

def _detect_format(run_dir: Path) -> str:
    if (run_dir / "trace.jsonl").exists():
        return "trace_events"
    if (run_dir / "trajectory.jsonl").exists():
        return "flat_actions"
    raise ValueError(f"unrecognized run dir layout: {run_dir}")


def _read_jsonl(p: Path) -> list[dict]:
    out: list[dict] = []
    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _gather_screenshots_trace(run_dir: Path) -> list[Path]:
    """For trace_events runs, screenshots live in screenshots/step_NNN.png."""
    sd = run_dir / "screenshots"
    if not sd.exists():
        return []
    files = sorted(sd.glob("step_*.png"))
    return files


def _gather_screenshots_flat(run_dir: Path) -> list[Path]:
    """For flat_actions runs, screenshots live in run_dir/step_NNN.png."""
    files = sorted(run_dir.glob("step_*.png"))
    return files


def _convert_trace_events(
    run_dir: Path,
    *,
    agent_name: str,
    model_name: str,
    task_text: str,
) -> tuple[dict, list[tuple[str, bytes]]]:
    """Convert a trace.jsonl + screenshots/ run into ATIF steps + screenshots.

    Returns (trajectory_dict, [(jpg_filename, jpg_bytes), ...]).
    """
    events = _read_jsonl(run_dir / "trace.jsonl")
    raw_screens = _gather_screenshots_trace(run_dir)

    # Map step numbers to screenshots. For trace_events we record one screenshot
    # per step in CuaRunner's flow (post-action). The first one is the initial.
    # CuaRunner saves them as step_001.png, step_002.png, ... in our adapter.
    # We renumber to step-00.jpg (initial), step-01.jpg (after step 1 action), ...
    screenshots: list[tuple[str, bytes]] = []
    if raw_screens:
        # Treat the first numeric screenshot as step-00 (initial).
        for idx, src in enumerate(raw_screens):
            jpg_name = f"step-{idx:02d}.jpg"
            screenshots.append((jpg_name, _png_to_jpg_bytes(src)))

    # Build steps. Step 0 is the user message + initial screenshot.
    steps: list[dict] = []
    started_at: str | None = None
    finished_at: str | None = None

    for ev in events:
        ts = _ensure_iso(ev.get("timestamp"))
        if ev.get("event") == "run_started":
            started_at = ts
        if ev.get("event") == "run_completed":
            finished_at = ts

    initial_screenshot = "step-00.jpg" if screenshots else None
    steps.append(
        {
            "step_id": 1,
            "timestamp": started_at or ISO_FALLBACK,
            "source": "user",
            "message": (
                [
                    {"type": "text", "text": task_text},
                    {
                        "type": "image",
                        "source": {
                            "media_type": "image/jpeg",
                            "path": f"screenshots/{initial_screenshot}",
                        },
                    },
                ]
                if initial_screenshot
                else [{"type": "text", "text": task_text}]
            ),
        }
    )

    # An "agent" step in our trace is approximated by a (message_received,
    # action_selected) pair sharing the same step number (1-indexed). Walk the
    # events sequentially, accumulating until each action_selected, then emit.
    pending_message: str | None = None
    step_id = 2  # next step_id (after the user step)
    action_index = 0  # 0-based for screenshot mapping (post-action screenshot)

    for ev in events:
        et = ev.get("event")
        ts = _ensure_iso(ev.get("timestamp"))
        if et == "message_received":
            txt = ev.get("text", "") or ""
            pending_message = (pending_message or "") + ("\n" if pending_message else "") + txt
        elif et == "action_selected":
            action = ev.get("action", {}) or {}
            atype = action.get("type", "unknown")
            # Build harbor tool_call from action.
            args: dict[str, Any] = {}
            for k in ("x", "y", "end_x", "end_y", "scroll_x", "scroll_y", "url", "text", "keys", "button"):
                v = action.get(k)
                if v not in (None, "", []):
                    args[k] = v
            tool_call = {
                "tool_call_id": ev.get("response_id") or f"tc_{step_id}",
                "function_name": atype,
                "arguments": args,
            }
            # Observation = post-action screenshot (next index).
            action_index += 1
            obs_path = (
                f"screenshots/step-{action_index:02d}.jpg"
                if action_index < len(screenshots)
                else None
            )
            obs_content: list[dict] = [
                {"type": "text", "text": f"Executed {atype}"},
            ]
            if obs_path:
                obs_content.append(
                    {
                        "type": "image",
                        "source": {"media_type": "image/jpeg", "path": obs_path},
                    }
                )
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": ts,
                    "source": "agent",
                    "model_name": model_name,
                    "message": pending_message or "",
                    "tool_calls": [tool_call],
                    "observation": {
                        "results": [
                            {
                                "source_call_id": tool_call["tool_call_id"],
                                "content": obs_content,
                            }
                        ]
                    },
                }
            )
            step_id += 1
            pending_message = None
        elif et == "circuit_breaker":
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": ts,
                    "source": "system",
                    "message": (
                        f"[harness] circuit breaker: {ev.get('kind', 'identical')} cycle detected"
                    ),
                    "extra": {"recent_actions": ev.get("recent", [])},
                }
            )
            step_id += 1
        elif et == "no_action_retry":
            steps.append(
                {
                    "step_id": step_id,
                    "timestamp": ts,
                    "source": "system",
                    "message": f"[harness] no_action_retry streak={ev.get('streak')}",
                }
            )
            step_id += 1

    # Trailing assistant message (if model spoke without acting at end).
    if pending_message:
        steps.append(
            {
                "step_id": step_id,
                "timestamp": finished_at or ISO_FALLBACK,
                "source": "agent",
                "model_name": model_name,
                "message": pending_message,
                "tool_calls": [],
            }
        )
        step_id += 1

    trajectory: dict = {
        "schema_version": "ATIF-v1.6",
        "session_id": str(uuid.uuid4()),
        "agent": {
            "name": agent_name,
            "version": "0.1.0",
            "model_name": model_name,
            "extra": {"backend": "kernel+lightcone+cuarunner"},
        },
        "steps": steps,
        "final_metrics": {
            "total_steps": len(steps),
        },
    }
    return trajectory, screenshots


# --------------------------------------------------------- flat actions

def _convert_flat_actions(
    run_dir: Path,
    *,
    agent_name: str,
    model_name: str,
    task_text: str,
) -> tuple[dict, list[tuple[str, bytes]]]:
    """Convert a teammate adapter run (trajectory.jsonl + step_NNN.png)."""
    all_rows = _read_jsonl(run_dir / "trajectory.jsonl")
    # launch.py writes mixed rows (action / no_action / terminal). Keep only
    # action rows for the action-grouping pass; surface terminal/message rows
    # via the agent's final message instead (already in agent_response.raw.txt).
    rows = [r for r in all_rows if "action_type" in r and "ts" in r]
    raw_screens = _gather_screenshots_flat(run_dir)

    # Screenshots from teammate are step_000_initial.png plus step_NNN.png.
    # Sort numerically by trailing index from filename.
    def _idx(p: Path) -> int:
        m = re.search(r"step_(\d+)", p.name)
        return int(m.group(1)) if m else -1

    raw_screens = sorted(raw_screens, key=_idx)

    screenshots: list[tuple[str, bytes]] = []
    for idx, src in enumerate(raw_screens):
        jpg_name = f"step-{idx:02d}.jpg"
        screenshots.append((jpg_name, _png_to_jpg_bytes(src)))

    if rows:
        started_at = _ensure_iso(rows[0].get("ts"))
        finished_at = _ensure_iso(rows[-1].get("ts"))
    else:
        started_at = ISO_FALLBACK
        finished_at = ISO_FALLBACK

    initial_screenshot = "step-00.jpg" if screenshots else None
    steps: list[dict] = [
        {
            "step_id": 1,
            "timestamp": started_at,
            "source": "user",
            "message": (
                [
                    {"type": "text", "text": task_text},
                    {
                        "type": "image",
                        "source": {
                            "media_type": "image/jpeg",
                            "path": f"screenshots/{initial_screenshot}",
                        },
                    },
                ]
                if initial_screenshot
                else [{"type": "text", "text": task_text}]
            ),
        }
    ]

    # Group rows by their `step` field (one model turn = many granular actions).
    groups: dict[int, list[dict]] = {}
    for row in rows:
        step_num = int(row.get("step", 0))
        groups.setdefault(step_num, []).append(row)

    step_id = 2
    for step_num in sorted(groups.keys()):
        actions = groups[step_num]
        first_ts = _ensure_iso(actions[0].get("ts"))
        tool_calls: list[dict] = []
        for j, a in enumerate(actions):
            atype = a.get("action_type", "unknown")
            args: dict[str, Any] = {}
            for k in ("x", "y", "text", "keys", "url", "direction", "delta_x", "delta_y", "press_enter", "clear_before", "final_text"):
                v = a.get(k)
                if v not in (None, "", [], False):
                    args[k] = v
            tool_calls.append(
                {
                    "tool_call_id": f"tc_{step_num}_{j}",
                    "function_name": atype,
                    "arguments": args,
                }
            )
        # Heuristic post-action screenshot: use the (step_num+1)-th screenshot
        # if it exists. Step 0 is initial, step 1's observation is step-01.jpg, etc.
        obs_idx = step_num + 1
        obs_path = (
            f"screenshots/step-{obs_idx:02d}.jpg"
            if 0 <= obs_idx < len(screenshots)
            else None
        )
        obs_content: list[dict] = [
            {"type": "text", "text": f"Executed {len(tool_calls)} action(s)"}
        ]
        if obs_path:
            obs_content.append(
                {
                    "type": "image",
                    "source": {"media_type": "image/jpeg", "path": obs_path},
                }
            )
        steps.append(
            {
                "step_id": step_id,
                "timestamp": first_ts,
                "source": "agent",
                "model_name": model_name,
                "message": "",
                "tool_calls": tool_calls,
                "observation": {
                    "results": [
                        {
                            "source_call_id": tool_calls[0]["tool_call_id"] if tool_calls else "tc_unknown",
                            "content": obs_content,
                        }
                    ]
                },
            }
        )
        step_id += 1

    trajectory = {
        "schema_version": "ATIF-v1.6",
        "session_id": str(uuid.uuid4()),
        "agent": {
            "name": agent_name,
            "version": "0.1.0",
            "model_name": model_name,
            "extra": {"backend": "kernel+adapter"},
        },
        "steps": steps,
        "final_metrics": {"total_steps": len(steps)},
    }
    return trajectory, screenshots


# ---------------------------------------------------- model name registry

DEFAULT_MODEL_NAMES = {
    "northstar": "tzafon.northstar-cua-fast",
    "tzafon": "tzafon.northstar-cua-fast",
    "openai": "openai.computer-use-preview",
    "claude": "anthropic.claude-sonnet-4-6",
    "bedrock_claude": "global.anthropic.claude-sonnet-4-6",
    "gemini": "google.gemini-3-flash-preview",
}


def convert_run_to_trial(
    run_dir: Path,
    out_trial_dir: Path,
    *,
    agent_name: str,
    model_name: str,
    task_name: str,
    task_text: str,
    score_summary: dict | None = None,
) -> dict:
    """Build a single Harbor trial directory in `out_trial_dir`."""
    fmt = _detect_format(run_dir)
    if fmt == "trace_events":
        trajectory, screenshots = _convert_trace_events(
            run_dir,
            agent_name=agent_name,
            model_name=model_name,
            task_text=task_text,
        )
    else:
        trajectory, screenshots = _convert_flat_actions(
            run_dir,
            agent_name=agent_name,
            model_name=model_name,
            task_text=task_text,
        )

    # Write screenshots.
    screens_dir = out_trial_dir / "agent" / "screenshots"
    screens_dir.mkdir(parents=True, exist_ok=True)
    for name, jpg in screenshots:
        (screens_dir / name).write_bytes(jpg)

    # Write trajectory.json.
    (out_trial_dir / "agent" / "trajectory.json").write_text(json.dumps(trajectory, indent=2))

    # Compute timestamps from the trajectory steps.
    step_ts = [s.get("timestamp") for s in trajectory["steps"] if s.get("timestamp")]
    started_at = step_ts[0] if step_ts else ISO_FALLBACK
    finished_at = step_ts[-1] if step_ts else ISO_FALLBACK

    trial_name = out_trial_dir.name

    # Per-spec: rewards keyed by metric name, values 0.0..1.0.
    reward = float(score_summary.get("score", {}).get("headline")) if (
        score_summary and score_summary.get("score") and score_summary["score"].get("headline") is not None
    ) else 0.0
    valid_json = bool(score_summary.get("parsed_json_present")) if score_summary else False

    config = {
        "trial_name": trial_name,
        "agent": {"name": agent_name, "model_name": model_name},
    }
    (out_trial_dir / "config.json").write_text(json.dumps(config, indent=2))

    result = {
        "id": trial_name,
        "task_name": task_name,
        "trial_name": trial_name,
        "trial_uri": f"opencua://{trial_name}",
        "agent_info": {
            "name": agent_name,
            "version": "0.1.0",
            "model_info": {"name": model_name},
        },
        "agent_result": {"n_input_tokens": None, "n_output_tokens": None, "cost_usd": None},
        "verifier_result": {"rewards": {"reward": reward, "valid_json": float(valid_json)}},
        "exception_info": None,
        "started_at": started_at,
        "finished_at": finished_at,
    }
    (out_trial_dir / "result.json").write_text(json.dumps(result, indent=2))

    return {
        "trial_name": trial_name,
        "agent": agent_name,
        "model_name": model_name,
        "reward": reward,
        "started_at": started_at,
        "finished_at": finished_at,
    }


TASK_SPEC_MD = {
    "T2": "T2_restaurant_restock.md",
    "T2-001": "T2_restaurant_restock.md",
    "T3": "T3_rei_hiking_boots.md",
    "T4": "T4_used_books_basket.md",
}


def _task_prompt_for(task_name: str) -> str:
    """Read the agent prompt (fenced block under ## Agent prompt) from the
    canonical task spec. Falls back to bench/worlds/<world>/tasks/<id>/intent.md
    if the new path isn't present (older runs)."""
    md_name = TASK_SPEC_MD.get(task_name)
    if md_name:
        md_path = REPO_ROOT / "tasks" / md_name
        if md_path.exists():
            text = md_path.read_text()
            m = re.search(
                r"^##\s+Agent prompt\s*\n+```(?:\w*)?\n(.*?)```",
                text,
                re.DOTALL | re.MULTILINE,
            )
            if m:
                return m.group(1).strip()
            return text
    legacy = REPO_ROOT / "bench" / "worlds" / "webstaurant_v1" / "tasks" / task_name / "intent.md"
    if legacy.exists():
        return legacy.read_text()
    return f"(task prompt for {task_name} not found)"


def build_job(
    *,
    job_name: str,
    task_name: str,
    out_dir: Path,
    runs: list[tuple[str, Path]],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    task_text = _task_prompt_for(task_name)

    trials_built: list[dict] = []
    by_agent: dict[str, list[float]] = {}
    started_overall: list[str] = []
    finished_overall: list[str] = []

    for agent_label, run_dir in runs:
        agent_name = agent_label
        model_name = DEFAULT_MODEL_NAMES.get(agent_label, agent_label)
        trial_dir_name = _trial_id_for(agent_label)
        trial_dir = out_dir / trial_dir_name
        score_path = run_dir / "summary.json"
        score_summary = json.loads(score_path.read_text()) if score_path.exists() else None
        info = convert_run_to_trial(
            run_dir,
            trial_dir,
            agent_name=agent_name,
            model_name=model_name,
            task_name=task_name,
            task_text=task_text,
            score_summary=score_summary,
        )
        trials_built.append(info)
        by_agent.setdefault(f"{agent_name}__{model_name}", []).append(info["reward"])
        started_overall.append(info["started_at"])
        finished_overall.append(info["finished_at"])

    # Job config.
    agents_seen = []
    seen = set()
    for t in trials_built:
        key = (t["agent"], t["model_name"])
        if key in seen:
            continue
        seen.add(key)
        agents_seen.append({"name": t["agent"], "model_name": t["model_name"]})

    job_config = {
        "job_name": job_name,
        "agents": agents_seen,
        "datasets": [],
        "tasks": [{"name": task_name}],
    }
    (out_dir / "config.json").write_text(json.dumps(job_config, indent=2))

    # Job result aggregate.
    evals = {}
    for k, rewards in by_agent.items():
        evals[k] = {
            "n_trials": len(rewards),
            "n_errors": 0,
            "metrics": [{"mean": sum(rewards) / len(rewards) if rewards else 0.0}],
            "reward_stats": {
                "reward": {
                    f"{r}": [t["trial_name"] for t in trials_built if t["reward"] == r]
                    for r in sorted(set(rewards), reverse=True)
                }
            },
            "exception_stats": {},
        }

    job_result = {
        "id": str(uuid.uuid4()),
        "started_at": min(started_overall) if started_overall else ISO_FALLBACK,
        "finished_at": max(finished_overall) if finished_overall else ISO_FALLBACK,
        "n_total_trials": len(trials_built),
        "stats": {
            "n_trials": len(trials_built),
            "n_errors": 0,
            "evals": evals,
        },
    }
    (out_dir / "result.json").write_text(json.dumps(job_result, indent=2))

    return out_dir


# --------------------------------------------------------- CLI


def _parse_run(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError(f"--run expects agent=path, got: {spec}")
    label, path = spec.split("=", 1)
    p = Path(path)
    if not p.exists():
        raise argparse.ArgumentTypeError(f"run path does not exist: {p}")
    return label, p


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build")
    p_build.add_argument("--job-name", required=True)
    p_build.add_argument("--task-name", default="T2-001")
    p_build.add_argument("--out", required=True, type=Path)
    p_build.add_argument("--run", action="append", type=_parse_run, required=True,
                         help="agent=path/to/run_dir   (can pass multiple)")
    args = parser.parse_args()

    if args.cmd == "build":
        out = build_job(
            job_name=args.job_name,
            task_name=args.task_name,
            out_dir=args.out,
            runs=args.run,
        )
        print(f"Built harbor job at: {out}")
        print(f"Upload with:  npx --yes trajectories-sh upload trajectory {out}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
