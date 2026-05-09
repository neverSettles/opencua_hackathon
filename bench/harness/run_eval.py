from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from bench.harness.compile_ground_truth import compile_task
from bench.harness.evaluator import evaluate
from bench.harness.io import (
    BENCH_DIR,
    append_jsonl,
    load_ground_truth,
    load_task,
    task_dir,
    write_json,
)
from bench.harness.tzafon_adapter import run_tzafon_task


def _fetch_json(url: str) -> dict[str, Any]:
    with urlopen(url, timeout=5) as response:
        return json.loads(response.read().decode())


def _wait_for_health(base_url: str, timeout_seconds: int = 10) -> None:
    deadline = time.time() + timeout_seconds
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            _fetch_json(f"{base_url}/api/health")
            return
        except (URLError, TimeoutError, ConnectionError) as exc:
            last_error = exc
            time.sleep(0.2)
    raise RuntimeError(f"Server did not become healthy at {base_url}: {last_error}")


def _start_server(world_id: str, task_id: str, output_dir: Path, host: str, port: int) -> subprocess.Popen[str]:
    cmd = [
        sys.executable,
        "-m",
        "bench.env.app",
        "--world",
        world_id,
        "--task",
        task_id,
        "--state-dir",
        str(output_dir),
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(BENCH_DIR.parent),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def _stop_server(proc: subprocess.Popen[str]) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def _mock_final_state(world_id: str, task_id: str, output_dir: Path) -> dict[str, Any]:
    task = load_task(world_id, task_id)
    ground_truth = load_ground_truth(world_id, task_id)
    allocation = [dict(row) for row in ground_truth["optimal_allocation"]]
    state = {
        "world_id": world_id,
        "task_id": task_id,
        "started_at": time.time(),
        "carts": {},
        "emails": [],
        "quotes": [],
        "accepted_quotes": [],
        "final_response": {
            "status": "SUCCESS",
            "summary": "Mock baseline applied the generated optimal allocation.",
            "chosen_sellers": sorted({line["seller_id"] for line in allocation}),
            "estimated_total_usd": ground_truth["optimal_total_usd"],
        },
        "events": [{"ts": time.time(), "kind": "mock_optimal_allocation", "payload": allocation}],
    }
    if task["task_type"] == "negotiation_fixed_list":
        quote_id = "mock-optimal"
        state["emails"] = [
            {
                "direction": "outgoing",
                "seller_id": "multiple",
                "body": "Please send your best bulk quote for the requested benchmark task.",
                "ts": time.time(),
            },
            {
                "direction": "incoming",
                "seller_id": "multiple",
                "body": "Mock quote accepted at generated optimal prices.",
                "quote_id": quote_id,
                "ts": time.time(),
            },
        ]
        state["quotes"] = [
            {
                "quote_id": quote_id,
                "seller_id": "multiple",
                "seller_name": "Mock optimal quote",
                "body": "Mock quote accepted at generated optimal prices.",
                "lines": allocation,
                "total_usd": ground_truth["optimal_total_usd"],
                "status": "accepted",
            }
        ]
        by_seller: dict[str, list[dict[str, Any]]] = {}
        for line in allocation:
            by_seller.setdefault(line["seller_id"], []).append(line)
        state["accepted_quotes"] = [
            {
                "quote_id": f"mock-{seller_id}",
                "seller_id": seller_id,
                "seller_name": seller_id,
                "body": "Mock accepted quote.",
                "lines": lines,
                "total_usd": round(sum(row["line_total_usd"] for row in lines), 2),
                "status": "accepted",
            }
            for seller_id, lines in by_seller.items()
        ]
    else:
        for line in allocation:
            enriched = dict(line)
            enriched["source"] = "cart"
            state["carts"].setdefault(line["seller_id"], []).append(enriched)

    write_json(output_dir / "final_state.json", state)
    for event in state["events"]:
        append_jsonl(output_dir / "trajectory.jsonl", event)
    for email in state["emails"]:
        append_jsonl(output_dir / "emails.jsonl", email)
    write_json(output_dir / "agent_response.json", state["final_response"])
    return state


def _write_emails_jsonl(final_state: dict[str, Any], output_dir: Path) -> None:
    emails_path = output_dir / "emails.jsonl"
    if emails_path.exists():
        emails_path.unlink()
    for email in final_state.get("emails", []):
        append_jsonl(emails_path, email)


def run_one(
    *,
    world_id: str,
    task_id: str,
    model_id: str,
    mode: str,
    run_id: str,
    rollout_n: int,
    host: str,
    port: int,
    public_base_url: str | None,
) -> dict[str, Any]:
    output_dir = BENCH_DIR / "outputs" / run_id / model_id / world_id / task_id / f"rollout_{rollout_n}"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        output_dir / "run_metadata.json",
        {
            "world_id": world_id,
            "task_id": task_id,
            "model_id": model_id,
            "mode": mode,
            "rollout_n": rollout_n,
            "started_at": time.time(),
        },
    )
    compile_task(world_id, task_id, write=True)
    task = load_task(world_id, task_id)
    intent = (task_dir(world_id, task_id) / "intent.md").read_text()
    local_base_url = f"http://127.0.0.1:{port}"
    browser_base_url = public_base_url or local_base_url
    proc: subprocess.Popen[str] | None = None
    started = time.time()
    try:
        proc = _start_server(world_id, task_id, output_dir, host, port)
        _wait_for_health(local_base_url)
        if mode == "mock":
            final_state = _mock_final_state(world_id, task_id, output_dir)
        elif mode == "manual":
            print(f"Manual run ready: {browser_base_url}")
            print("Complete the task in a browser, submit Final Answer, then press Enter here.")
            input()
            final_state = _fetch_json(f"{local_base_url}/api/state")
            write_json(output_dir / "final_state.json", final_state)
        elif mode == "tzafon":
            run_tzafon_task(
                base_url=browser_base_url,
                task_intent=intent,
                response_type=task["expected_response"]["type"],
                model_id=model_id,
                output_dir=output_dir,
            )
            final_state = _fetch_json(f"{local_base_url}/api/state")
            write_json(output_dir / "final_state.json", final_state)
            if final_state.get("final_response") is not None:
                write_json(output_dir / "agent_response.json", final_state["final_response"])
        else:
            raise ValueError(f"Unknown mode: {mode}")
        _write_emails_jsonl(final_state, output_dir)
        score = evaluate(world_id, task_id, final_state, output_dir / "score.json")
        timing = {
            "wall_clock_seconds": round(time.time() - started, 3),
            "mode": mode,
            "port": port,
            "browser_base_url": browser_base_url,
        }
        write_json(output_dir / "timing.json", timing)
        return score
    finally:
        if proc is not None:
            _stop_server(proc)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", action="append", help="World id. May be repeated.")
    parser.add_argument("--task", action="append", help="Task id. May be repeated; pairs with --world by index.")
    parser.add_argument("--all-demo", action="store_true", help="Run all three demo tasks.")
    parser.add_argument("--model", default="tzafon-northstar")
    parser.add_argument("--mode", choices=["mock", "manual", "tzafon"], default="mock")
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--run-id", default=time.strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=8765)
    parser.add_argument("--public-base-url", help="Public tunnel/deploy URL reachable by Tzafon remote browser.")
    args = parser.parse_args()

    pairs: list[tuple[str, str]]
    if args.all_demo:
        pairs = [
            ("wine_restock_v1", "wine-001"),
            ("office_bulk_v1", "office-001"),
            ("pc_build_v1", "pc-001"),
        ]
    else:
        if not args.world or not args.task or len(args.world) != len(args.task):
            raise SystemExit("Pass --all-demo or matching --world/--task pairs.")
        pairs = list(zip(args.world, args.task))

    scores = []
    for pair_index, (world_id, task_id) in enumerate(pairs):
        for rollout_n in range(args.rollouts):
            port = args.base_port + pair_index * 10 + rollout_n
            print(f"Running {world_id}/{task_id} rollout {rollout_n} in {args.mode} mode")
            score = run_one(
                world_id=world_id,
                task_id=task_id,
                model_id=args.model,
                mode=args.mode,
                run_id=args.run_id,
                rollout_n=rollout_n,
                host=args.host,
                port=port,
                public_base_url=args.public_base_url,
            )
            scores.append(score)
            print(json.dumps({"task": task_id, "score": score["final_score"], "metrics": score["metrics"]}, indent=2))
    print(f"Done. Outputs: {BENCH_DIR / 'outputs' / args.run_id}")


if __name__ == "__main__":
    main()
