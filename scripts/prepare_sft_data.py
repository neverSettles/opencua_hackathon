"""Convert collected trajectories into SFT training pairs.

Input: outputs/<run_id>/<world_id>/<task_id>/{trajectory.jsonl, step_NNN.png, ...}
Output: a single JSONL where each line is one training example:

    {
      "instruction": "<task agent_prompt>",
      "history": [
        {"step": 0, "image_path": "<abs path>"},
        ...
      ],
      "current_image_path": "<abs path of latest screenshot before this action>",
      "action": {
        "type": "click|type|key_press|scroll|...",
        "x": 607, "y": 157,
        "text": "...",
        "keys": ["enter"],
        ...
      },
      "model": "anthropic|openai",
      "task": "T2|T3|T4",
      "world_id": "...",
      "task_id": "...",
      "rollout_dir": "..."
    }

The training script can then chunk history as needed (e.g. drop oldest to fit
context, or keep only the latest screenshot). We just emit raw pairs here.

Usage:
    python scripts/prepare_sft_data.py outputs/  --out sft_data/sft.jsonl
    python scripts/prepare_sft_data.py outputs/  --out sft_data/sft.jsonl --tasks T2 T3
    python scripts/prepare_sft_data.py outputs/  --out sft_data/sft.jsonl --models anthropic
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]


SPEC_FILES = {
    "T2": REPO_ROOT / "tasks" / "T2_restaurant_restock.md",
    "T3": REPO_ROOT / "tasks" / "T3_rei_hiking_boots.md",
    "T4": REPO_ROOT / "tasks" / "T4_used_books_basket.md",
}


def _extract_agent_prompt(spec_md: str) -> str:
    m = re.search(r"## Agent prompt\s*\n+```\w*\n(.*?)\n```", spec_md, re.S)
    return m.group(1).strip() if m else spec_md.strip()


def _load_instruction(task: str) -> str:
    p = SPEC_FILES.get(task)
    if p is None or not p.exists():
        return ""
    return _extract_agent_prompt(p.read_text())


def _detect_task(rollout_dir: Path) -> str | None:
    # Run dirs are named like t2_anthropic_..., t3_openai_..., t4_anthropic_...
    name = rollout_dir.name.lower()
    for t in ("t2", "t3", "t4"):
        if name.startswith(t + "_"):
            return t.upper()
    return None


def _detect_model(rollout_dir: Path) -> str | None:
    name = rollout_dir.name.lower()
    for m in ("anthropic", "openai", "northstar", "gemini"):
        if f"_{m}_" in name:
            return m
    return None


def _iter_rollouts(outputs_root: Path) -> Iterable[Path]:
    """Find every directory containing both trajectory.jsonl and step_*.png."""
    for traj in outputs_root.glob("**/trajectory.jsonl"):
        yield traj.parent


def _action_records(traj_path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with traj_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            # Action records have action_type + step. Skip metadata/no-action lines.
            if "action_type" in rec and rec.get("action_type"):
                out.append(rec)
    return out


def _step_image_path(rollout_dir: Path, step: int) -> Path | None:
    # Initial screenshot is step_000_initial.png; subsequent are step_NNN.png
    candidates = [
        rollout_dir / f"step_{step:03d}.png",
        rollout_dir / f"step_{step:03d}_initial.png",
        rollout_dir / f"step_{step:03d}_nudge.png",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def build_examples(rollout_dir: Path, *, max_history: int = 0) -> list[dict[str, Any]]:
    """Build (image, action) pairs for one rollout.

    `max_history` controls how many prior screenshot paths are attached per
    example (0 = none, just the current image). Training scripts that use a
    rolling window of screenshots can use this; others can ignore it.
    """
    task = _detect_task(rollout_dir.parents[1] if rollout_dir.parents[1] != rollout_dir else rollout_dir)
    if task is None:
        # rollout_dir is .../<run_id>/<world_id>/<task_id>; the run_id is parents[2]
        for p in (rollout_dir.parent, rollout_dir.parents[1] if len(rollout_dir.parents) > 1 else None,
                  rollout_dir.parents[2] if len(rollout_dir.parents) > 2 else None):
            if p is None:
                continue
            t = _detect_task(p)
            if t:
                task = t
                break
    model = _detect_model(rollout_dir)
    if model is None:
        for p in (rollout_dir.parent, rollout_dir.parents[1] if len(rollout_dir.parents) > 1 else None,
                  rollout_dir.parents[2] if len(rollout_dir.parents) > 2 else None):
            if p is None:
                continue
            m = _detect_model(p)
            if m:
                model = m
                break

    instruction = _load_instruction(task) if task else ""
    actions = _action_records(rollout_dir / "trajectory.jsonl")
    if not actions:
        return []

    examples: list[dict[str, Any]] = []
    initial = rollout_dir / "step_000_initial.png"
    history: list[dict[str, Any]] = []
    if initial.exists():
        history.append({"step": 0, "image_path": str(initial.resolve())})

    for rec in actions:
        step = int(rec.get("step", -1))
        # The current screenshot the model saw BEFORE deciding this action is
        # the screenshot from step (step) — except step 0 uses the initial.
        # In our log: step N's action follows screenshot step_NNN.png
        # (taken after step N-1's action), so the current image is step_NNN-1.png.
        cur_step = max(0, step)  # use initial for step 0; later steps use prior
        img = _step_image_path(rollout_dir, cur_step) or initial
        if img is None or not img.exists():
            continue

        action: dict[str, Any] = {"type": rec.get("action_type")}
        for k in ("x", "y", "text", "keys", "url", "direction",
                  "delta_x", "delta_y", "press_enter", "clear_before"):
            v = rec.get(k)
            if v not in (None, "", []):
                action[k] = v

        ex: dict[str, Any] = {
            "instruction": instruction,
            "current_image_path": str(img.resolve()),
            "action": action,
            "model": model or "unknown",
            "task": task or "unknown",
            "rollout_dir": str(rollout_dir.resolve()),
            "step": step,
        }
        if max_history > 0:
            ex["history"] = list(history[-max_history:])
        examples.append(ex)

        history.append({"step": step + 1, "image_path": str(img.resolve())})

    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_root", type=Path, default=REPO_ROOT / "outputs", nargs="?")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "sft_data" / "sft.jsonl")
    parser.add_argument("--tasks", nargs="*", help="Filter by task IDs (e.g. T2 T3)")
    parser.add_argument("--models", nargs="*", help="Filter by model adapter names (e.g. anthropic openai)")
    parser.add_argument("--max-history", type=int, default=0,
                        help="Include up to N prior screenshot paths in `history` per example (default 0)")
    args = parser.parse_args()

    if not args.outputs_root.exists():
        print(f"outputs_root not found: {args.outputs_root}", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    rollouts_seen = 0
    rollouts_used = 0
    with args.out.open("w") as f:
        for rollout_dir in _iter_rollouts(args.outputs_root):
            rollouts_seen += 1
            examples = build_examples(rollout_dir, max_history=args.max_history)
            if not examples:
                continue
            ex0 = examples[0]
            if args.tasks and ex0["task"] not in args.tasks:
                continue
            if args.models and ex0["model"] not in args.models:
                continue
            rollouts_used += 1
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
                total += 1

    print(f"Scanned {rollouts_seen} rollouts, used {rollouts_used}.")
    print(f"Wrote {total} (image, action) examples -> {args.out}")


if __name__ == "__main__":
    main()
