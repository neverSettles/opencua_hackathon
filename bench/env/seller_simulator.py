from __future__ import annotations

import hashlib
from typing import Any

from bench.harness.io import effective_unit_price, item_matches_request, item_meets_specs


PERSONALITY_MARGIN = {
    "flexible_bulk": 0.20,
    "eager_discount": 0.25,
    "deal_hunter": 0.30,
    "firm_b2b": 0.55,
    "component_retailer": 0.60,
    "prebuilt_upseller": 0.65,
    "retail_baseline": 0.75,
    "professional_firm": 0.65,
    "premium_no_discount": 0.85,
}


def _quote_price(item: dict[str, Any], qty: int, personality: str) -> float:
    list_price = effective_unit_price(item, qty, "list")
    reservation = effective_unit_price(item, qty, "reservation")
    margin = PERSONALITY_MARGIN.get(personality, 0.50)
    return round(reservation + (list_price - reservation) * margin, 2)


def _quote_id(seller_id: str, lines: list[dict[str, Any]], email_count: int) -> str:
    raw = seller_id + str(email_count) + "".join(line["sku"] for line in lines)
    return hashlib.sha1(raw.encode()).hexdigest()[:10]


def generate_quote(
    seller: dict[str, Any],
    task: dict[str, Any],
    buyer_message: str,
    email_count: int,
) -> dict[str, Any]:
    """Generate a deterministic quote from hidden seller config.

    This is intentionally not judge-LLM based: the benchmark can explain exactly
    where every quote came from, and ground truth remains reproducible.
    """
    seller_id = seller["id"]
    personality = seller.get("personality", "firm_b2b")
    lines: list[dict[str, Any]] = []
    task_type = task["task_type"]

    if task_type == "pc_build":
        # Quote either valid prebuilts or one good component per required category.
        required = task["config"]["prebuilt_requirements"]
        for inventory_item in seller.get("inventory", []):
            if inventory_item.get("kind") == "prebuilt" and item_meets_specs(inventory_item, required):
                qty = 1
                unit_price = _quote_price(inventory_item, qty, personality)
                lines.append(
                    {
                        "seller_id": seller_id,
                        "item_id": inventory_item["item_id"],
                        "sku": inventory_item["sku"],
                        "name": inventory_item["name"],
                        "qty": qty,
                        "unit_price_usd": unit_price,
                        "line_total_usd": unit_price,
                    }
                )
        for requirement in task["config"].get("required_components", []):
            candidates = [
                item
                for item in seller.get("inventory", [])
                if item.get("kind") == "component"
                and item.get("category") == requirement["category"]
                and item_meets_specs(item, requirement)
            ]
            if candidates:
                best = sorted(candidates, key=lambda item: _quote_price(item, 1, personality))[0]
                unit_price = _quote_price(best, 1, personality)
                lines.append(
                    {
                        "seller_id": seller_id,
                        "item_id": best["item_id"],
                        "sku": best["sku"],
                        "name": best["name"],
                        "qty": 1,
                        "unit_price_usd": unit_price,
                        "line_total_usd": unit_price,
                    }
                )
    else:
        for request in task["config"].get("items", []):
            qty = int(request["qty"])
            for inventory_item in seller.get("inventory", []):
                if item_matches_request(request, inventory_item) and int(inventory_item.get("stock", 0)) >= qty:
                    unit_price = _quote_price(inventory_item, qty, personality)
                    lines.append(
                        {
                            "seller_id": seller_id,
                            "item_id": request["item_id"],
                            "sku": inventory_item["sku"],
                            "name": inventory_item["name"],
                            "qty": qty,
                            "unit_price_usd": unit_price,
                            "line_total_usd": round(unit_price * qty, 2),
                        }
                    )
                    break

    total = round(sum(line["line_total_usd"] for line in lines), 2)
    if not lines:
        body = "Thanks for reaching out. We do not have a valid quote for this task."
    elif "budget" in buyer_message.lower() or "best" in buyer_message.lower() or "bulk" in buyer_message.lower():
        body = (
            f"Thanks for the RFQ. Because you asked for our best bulk pricing, "
            f"we can offer the attached lines for ${total:.2f} total."
        )
    else:
        body = f"Thanks for the RFQ. We can offer the attached lines for ${total:.2f} total."

    return {
        "quote_id": _quote_id(seller_id, lines, email_count),
        "seller_id": seller_id,
        "seller_name": seller.get("name", seller_id),
        "body": body,
        "lines": lines,
        "total_usd": total,
        "status": "open",
    }
