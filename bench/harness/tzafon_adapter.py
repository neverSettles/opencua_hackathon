from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from bench.harness.io import append_jsonl


LOCAL_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


def _is_local_url(url: str) -> bool:
    return any(host in url for host in LOCAL_HOSTS)


def build_instruction(base_url: str, task_intent: str, response_type: str) -> str:
    return f"""
You are the buyer in a procurement benchmark.

Open this browser URL and complete the task using only the visible web UI:
{base_url}

Task:
{task_intent}

Rules:
- Use only allowed sellers visible in the benchmark UI.
- For quote-based tasks, use the Email page to send RFQs and accept useful quotes.
- For storefront tasks, add valid items to cart.
- Finish by opening the Final Answer page and submitting JSON with status, summary, chosen_sellers, and estimated_total_usd.
- Expected response type: {response_type}.
"""


def run_tzafon_task(
    *,
    base_url: str,
    task_intent: str,
    response_type: str,
    model_id: str,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run Northstar through Lightcone's high-level browser task API.

    The remote Tzafon browser must be able to reach base_url. If the local app is
    running on localhost, expose it with a tunnel and pass --public-base-url.
    """
    if _is_local_url(base_url):
        raise RuntimeError(
            "Tzafon's remote browser usually cannot access localhost. "
            "Run the app with --host 0.0.0.0 and pass a public tunnel URL via --public-base-url."
        )

    try:
        from tzafon import Lightcone  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Install the Tzafon SDK first: pip install tzafon") from exc

    output_dir = Path(output_dir)
    trajectory_path = output_dir / "trajectory.jsonl"
    instruction = build_instruction(base_url, task_intent, response_type)
    append_jsonl(
        trajectory_path,
        {"event": "task_start", "model_id": model_id, "base_url": base_url, "instruction": instruction},
    )
    client = Lightcone()
    stream = client.agent.tasks.start_stream(
        instruction=instruction,
        kind="browser",
    )
    event_count = 0
    last_event: Any = None
    for event in stream:
        event_count += 1
        last_event = event
        append_jsonl(trajectory_path, {"event": "tzafon_stream", "payload": _jsonable(event)})
    append_jsonl(trajectory_path, {"event": "task_end", "event_count": event_count})
    return {"event_count": event_count, "last_event": _jsonable(last_event)}


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return value.__dict__
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--intent-file", required=True)
    parser.add_argument("--response-type", default="cart_assembly_v1")
    parser.add_argument("--model-id", default=os.getenv("TZAFON_MODEL", "northstar-browser-task"))
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = run_tzafon_task(
        base_url=args.base_url,
        task_intent=Path(args.intent_file).read_text(),
        response_type=args.response_type,
        model_id=args.model_id,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
