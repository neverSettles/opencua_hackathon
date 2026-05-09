from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from bench.harness.io import BENCH_DIR, load_structured


def aggregate(run_id: str) -> tuple[Path, Path]:
    run_dir = BENCH_DIR / "outputs" / run_id
    rows = []
    for score_path in run_dir.glob("*/*/*/*/score.json"):
        score = load_structured(score_path)
        rows.append(
            {
                "model_id": score_path.relative_to(run_dir).parts[0],
                "world_id": score["world_id"],
                "task_id": score["task_id"],
                "rollout": score_path.parent.name,
                "final_score": score["final_score"],
                "agent_total_usd": score["metrics"]["agent_total_usd"],
                "optimal_total_usd": score["metrics"]["optimal_total_usd"],
                "cost_ratio": score["metrics"]["cost_ratio"],
                "surplus_capture": score["metrics"]["surplus_capture"],
                "net_value_usd": score["metrics"]["net_value_usd"],
                "passed_overall": score["passed_overall"],
            }
        )
    csv_path = run_dir / "leaderboard.csv"
    html_path = run_dir / "leaderboard.html"
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    summary = defaultdict(list)
    for row in rows:
        summary[(row["model_id"], row["world_id"], row["task_id"])].append(float(row["final_score"]))
    summary_rows = []
    for (model_id, world_id, task_id), values in summary.items():
        summary_rows.append(
            f"<tr><td>{model_id}</td><td>{world_id}</td><td>{task_id}</td><td>{sum(values)/len(values):.3f}</td><td>{len(values)}</td></tr>"
        )
    html_path.write_text(
        """<!doctype html><html><head><meta charset='utf-8'><title>Buy-Side Bench Leaderboard</title>
<style>body{font-family:sans-serif;margin:32px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}th{background:#f2f2f2}</style>
</head><body><h1>Buy-Side Optimization Bench</h1><table><thead><tr><th>Model</th><th>World</th><th>Task</th><th>Mean Score</th><th>Rollouts</th></tr></thead><tbody>"""
        + "".join(summary_rows)
        + "</tbody></table></body></html>\n"
    )
    return csv_path, html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    csv_path, html_path = aggregate(args.run_id)
    print(f"Wrote {csv_path}")
    print(f"Wrote {html_path}")


if __name__ == "__main__":
    main()
