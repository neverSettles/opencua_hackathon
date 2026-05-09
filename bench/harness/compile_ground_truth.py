from __future__ import annotations

import argparse
import itertools
from pathlib import Path
from typing import Any

from bench.harness.io import (
    effective_unit_price,
    item_matches_request,
    item_meets_specs,
    load_sellers,
    load_task,
    load_world,
    task_dir,
    write_json,
)


def _seller_candidates(
    request: dict[str, Any],
    sellers: dict[str, dict[str, Any]],
    allowed_sellers: list[str],
    qty: int,
    basis: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for seller_id in allowed_sellers:
        seller = sellers[seller_id]
        for inventory_item in seller.get("inventory", []):
            if item_matches_request(request, inventory_item) and int(inventory_item.get("stock", 0)) >= qty:
                unit_price = effective_unit_price(inventory_item, qty, basis=basis)
                candidates.append(
                    {
                        "seller_id": seller_id,
                        "sku": inventory_item["sku"],
                        "item_id": request["item_id"],
                        "name": inventory_item["name"],
                        "qty": qty,
                        "unit_price_usd": unit_price,
                        "line_total_usd": round(unit_price * qty, 2),
                        "basis": basis,
                    }
                )
    return sorted(candidates, key=lambda row: row["line_total_usd"])


def _solve_fixed_list(
    task: dict[str, Any],
    sellers: dict[str, dict[str, Any]],
    basis: str,
) -> tuple[float, list[dict[str, Any]]]:
    allowed = task["config"]["allowed_sellers"]
    max_sellers = task["config"].get("constraints", {}).get("max_sellers")
    per_item: list[list[dict[str, Any]]] = []
    for request in task["config"]["items"]:
        qty = int(request["qty"])
        candidates = _seller_candidates(request, sellers, allowed, qty, basis)
        if not candidates:
            raise ValueError(f"No seller can satisfy {request['item_id']} in {task['id']}")
        per_item.append(candidates)

    best_total = float("inf")
    best_allocation: list[dict[str, Any]] = []
    for allocation in itertools.product(*per_item):
        used_sellers = {row["seller_id"] for row in allocation}
        if max_sellers is not None and len(used_sellers) > int(max_sellers):
            continue
        total = round(sum(row["line_total_usd"] for row in allocation), 2)
        if total < best_total:
            best_total = total
            best_allocation = [dict(row) for row in allocation]
    if not best_allocation:
        raise ValueError(f"No valid allocation for {task['id']} with max_sellers={max_sellers}")
    return best_total, best_allocation


def _solve_pc_build(
    task: dict[str, Any],
    sellers: dict[str, dict[str, Any]],
    basis: str,
) -> tuple[float, list[dict[str, Any]], str]:
    allowed = task["config"]["allowed_sellers"]
    budget = task["config"].get("constraints", {}).get("budget_usd")
    required_components = task["config"]["required_components"]

    component_allocation: list[dict[str, Any]] = []
    for requirement in required_components:
        candidates: list[dict[str, Any]] = []
        for seller_id in allowed:
            seller = sellers[seller_id]
            for inventory_item in seller.get("inventory", []):
                if inventory_item.get("kind") != "component":
                    continue
                if inventory_item.get("category") != requirement["category"]:
                    continue
                if int(inventory_item.get("stock", 0)) < 1:
                    continue
                if not item_meets_specs(inventory_item, requirement):
                    continue
                unit_price = effective_unit_price(inventory_item, 1, basis=basis)
                candidates.append(
                    {
                        "seller_id": seller_id,
                        "sku": inventory_item["sku"],
                        "item_id": requirement["category"],
                        "category": requirement["category"],
                        "name": inventory_item["name"],
                        "qty": 1,
                        "unit_price_usd": unit_price,
                        "line_total_usd": unit_price,
                        "basis": basis,
                    }
                )
        if not candidates:
            raise ValueError(f"No PC component can satisfy {requirement['category']}")
        component_allocation.append(sorted(candidates, key=lambda row: row["line_total_usd"])[0])
    component_total = round(sum(row["line_total_usd"] for row in component_allocation), 2)

    prebuilts: list[dict[str, Any]] = []
    required_summary = task["config"]["prebuilt_requirements"]
    for seller_id in allowed:
        seller = sellers[seller_id]
        for inventory_item in seller.get("inventory", []):
            if inventory_item.get("kind") != "prebuilt":
                continue
            if not item_meets_specs(inventory_item, required_summary):
                continue
            unit_price = effective_unit_price(inventory_item, 1, basis=basis)
            prebuilts.append(
                {
                    "seller_id": seller_id,
                    "sku": inventory_item["sku"],
                    "item_id": "prebuilt",
                    "category": "prebuilt",
                    "name": inventory_item["name"],
                    "qty": 1,
                    "unit_price_usd": unit_price,
                    "line_total_usd": unit_price,
                    "basis": basis,
                }
            )
    best_prebuilt = min(prebuilts, key=lambda row: row["line_total_usd"], default=None)
    if best_prebuilt and best_prebuilt["line_total_usd"] < component_total:
        total = best_prebuilt["line_total_usd"]
        allocation = [best_prebuilt]
        strategy = "prebuilt"
    else:
        total = component_total
        allocation = component_allocation
        strategy = "parts"

    if budget is not None and total > float(budget):
        raise ValueError(f"Optimal PC build exceeds budget: {total} > {budget}")
    return total, allocation, strategy


def compile_task(world_id: str, task_id: str, write: bool = True) -> dict[str, Any]:
    world = load_world(world_id)
    task = load_task(world_id, task_id)
    sellers = load_sellers(world_id)
    basis = task["config"].get("ground_truth_price_basis", "list")

    if task["task_type"] == "pc_build":
        optimal_total, optimal_allocation, strategy = _solve_pc_build(task, sellers, basis)
        list_total, list_allocation, _ = _solve_pc_build(task, sellers, "list")
    else:
        optimal_total, optimal_allocation = _solve_fixed_list(task, sellers, basis)
        list_total, list_allocation = _solve_fixed_list(task, sellers, "list")
        strategy = "fixed_list"

    first_allowed = task["config"]["allowed_sellers"][0]
    naive_lines = []
    for line in list_allocation:
        seller_id = first_allowed
        request = {"item_id": line["item_id"], "sku": line.get("sku")}
        if task["task_type"] != "pc_build":
            matching_request = next(
                item for item in task["config"]["items"] if item["item_id"] == line["item_id"]
            )
            request = matching_request
        qty = int(line["qty"])
        candidates = _seller_candidates(request, sellers, [seller_id], qty, "list")
        naive_lines.append(candidates[0] if candidates else line)
    naive_total = round(sum(row["line_total_usd"] for row in naive_lines), 2)

    result = {
        "_generated_by": "bench/harness/compile_ground_truth.py",
        "world_id": world["id"],
        "task_id": task["id"],
        "price_basis": basis,
        "optimal_strategy": strategy,
        "optimal_total_usd": round(optimal_total, 2),
        "optimal_allocation": optimal_allocation,
        "list_price_best_total_usd": round(list_total, 2),
        "list_price_best_allocation": list_allocation,
        "naive_baseline_usd": naive_total,
    }
    if write:
        write_json(task_dir(world_id, task_id) / "ground_truth.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = compile_task(args.world, args.task, write=not args.no_write)
    print(f"{args.world}/{args.task}: optimal=${result['optimal_total_usd']:.2f}")


if __name__ == "__main__":
    main()
