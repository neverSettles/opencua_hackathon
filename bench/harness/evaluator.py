from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from bench.harness.io import (
    inventory_by_sku,
    item_meets_specs,
    load_ground_truth,
    load_rubric,
    load_sellers,
    load_task,
    write_json,
)


def _purchased_lines(final_state: dict[str, Any]) -> list[dict[str, Any]]:
    lines: list[dict[str, Any]] = []
    for seller_lines in final_state.get("carts", {}).values():
        lines.extend(seller_lines)
    for quote in final_state.get("accepted_quotes", []):
        for line in quote.get("lines", []):
            enriched = dict(line)
            enriched.setdefault("seller_id", quote["seller_id"])
            enriched["source"] = "accepted_quote"
            lines.append(enriched)
    return lines


def _total(lines: list[dict[str, Any]]) -> float:
    return round(sum(float(line["line_total_usd"]) for line in lines), 2)


def _seller_count(lines: list[dict[str, Any]]) -> int:
    return len({line["seller_id"] for line in lines})


def _fixed_completion(task: dict[str, Any], lines: list[dict[str, Any]]) -> tuple[bool, dict[str, Any]]:
    coverage: dict[str, int] = {}
    for line in lines:
        item_id = line.get("item_id")
        if item_id:
            coverage[item_id] = coverage.get(item_id, 0) + int(line.get("qty", 0))
    missing = []
    for item in task["config"].get("items", []):
        item_id = item["item_id"]
        required = int(item["qty"])
        bought = coverage.get(item_id, 0)
        if bought < required:
            missing.append({"item_id": item_id, "required": required, "bought": bought})
    return not missing, {"coverage": coverage, "missing": missing}


def _pc_completion(
    task: dict[str, Any],
    lines: list[dict[str, Any]],
    sellers: dict[str, dict[str, Any]],
) -> tuple[bool, dict[str, Any]]:
    sku_index = inventory_by_sku(sellers)
    # A valid prebuilt solves the whole task.
    for line in lines:
        sku = line.get("sku")
        if not sku or sku not in sku_index:
            continue
        _, item = sku_index[sku]
        if item.get("kind") == "prebuilt" and item_meets_specs(item, task["config"]["prebuilt_requirements"]):
            return True, {"strategy": "prebuilt", "sku": sku}

    satisfied: dict[str, bool] = {}
    for requirement in task["config"]["required_components"]:
        category = requirement["category"]
        satisfied[category] = False
        for line in lines:
            sku = line.get("sku")
            if not sku or sku not in sku_index:
                continue
            _, item = sku_index[sku]
            if item.get("kind") != "component":
                continue
            if item.get("category") == category and item_meets_specs(item, requirement):
                satisfied[category] = True
                break
    missing = [category for category, ok in satisfied.items() if not ok]
    return not missing, {"strategy": "parts", "satisfied": satisfied, "missing": missing}


def evaluate(
    world_id: str,
    task_id: str,
    final_state: dict[str, Any],
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    task = load_task(world_id, task_id)
    sellers = load_sellers(world_id)
    rubric = load_rubric(world_id, task_id)
    ground_truth = load_ground_truth(world_id, task_id)
    lines = _purchased_lines(final_state)
    total = _total(lines)
    allowed = set(task["config"]["allowed_sellers"])
    budget = task["config"].get("constraints", {}).get("budget_usd")
    max_sellers = task["config"].get("constraints", {}).get("max_sellers")

    if task["task_type"] == "pc_build":
        completed, completion_detail = _pc_completion(task, lines, sellers)
    else:
        completed, completion_detail = _fixed_completion(task, lines)

    gates = {
        "all_required_items_satisfied": completed,
        "pc_requirements_satisfied": completed,
        "allowed_sellers_only": all(line.get("seller_id") in allowed for line in lines) and bool(lines),
        "within_budget": True if budget is None else total <= float(budget),
    }

    optimal_total = float(ground_truth.get("optimal_total_usd", math.inf))
    list_total = float(ground_truth.get("list_price_best_total_usd", optimal_total))
    naive_total = float(ground_truth.get("naive_baseline_usd", list_total))
    cost_ratio = math.inf if optimal_total <= 0 else total / optimal_total
    denom = list_total - optimal_total
    if denom <= 0:
        surplus_capture = 1.0 if total <= optimal_total else 0.0
    else:
        surplus_capture = max(0.0, min(1.0, (list_total - total) / denom))

    outgoing_emails = [
        email
        for email in final_state.get("emails", [])
        if email.get("direction") == "outgoing"
    ]
    final_response = final_state.get("final_response")
    criteria_values = {
        "cost_within_5pct_of_optimal": cost_ratio <= 1.05,
        "cost_within_15pct_of_optimal": cost_ratio <= 1.15,
        "surplus_capture_at_least_50pct": surplus_capture >= 0.50,
        "used_at_most_max_sellers": True if max_sellers is None else _seller_count(lines) <= int(max_sellers),
        "submitted_final_response": bool(final_response) and final_response.get("status") != "UNPARSEABLE",
        "sent_at_least_one_rfq": bool(outgoing_emails),
    }

    gate_results = {gate: bool(gates.get(gate, False)) for gate in rubric.get("gates", [])}
    all_gates_pass = all(gate_results.values())
    weighted_total = sum(float(row["weight"]) for row in rubric.get("criteria", [])) or 1.0
    raw_score = 0.0
    scored_criteria = {}
    for criterion in rubric.get("criteria", []):
        criterion_id = criterion["id"]
        passed = bool(criteria_values.get(criterion_id, False))
        weight = float(criterion["weight"])
        if passed:
            raw_score += weight
        scored_criteria[criterion_id] = {"passed": passed, "weight": weight}
    final_score = round(raw_score / weighted_total, 4) if all_gates_pass else 0.0

    score = {
        "world_id": world_id,
        "task_id": task_id,
        "gates": gate_results,
        "all_gates_pass": all_gates_pass,
        "scored_criteria": scored_criteria,
        "final_score": final_score,
        "passed_overall": all_gates_pass and final_score > 0,
        "metrics": {
            "agent_total_usd": total,
            "optimal_total_usd": optimal_total,
            "naive_baseline_usd": naive_total,
            "cost_ratio": None if math.isinf(cost_ratio) else round(cost_ratio, 4),
            "surplus_capture": round(surplus_capture, 4),
            "net_value_usd": round(naive_total - total, 2),
            "seller_count": _seller_count(lines),
            "line_count": len(lines),
            "outgoing_email_count": len(outgoing_emails),
        },
        "completion_detail": completion_detail,
    }
    if output_path is not None:
        write_json(output_path, score)
    return score


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--final-state", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    import json

    final_state = json.loads(Path(args.final_state).read_text())
    score = evaluate(args.world, args.task, final_state, args.output)
    print(json.dumps(score, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
