"""Scoring functions for T2, T3, T4.

Each `score_t<N>(agent_output, ground_truth, ...)` returns a dict with
sub-scores plus a `headline` aggregate. These implement the spec scoring
outlines from tasks/T<N>_*.md, with the post-collection calibrations:

- T3: rating_optimal compares against the optimum chosen from the
  >=100-review pool (already pre-computed in the ground-truth file).
- T4: headline weights default to the post-calibration mix
  (0.40 validity + 0.30 completeness + 0.20 cost_correctness + 0.10 optimality).
  If the ground-truth file carries `headline_weights`, those are used instead,
  so the eval can be rewound to the original weights without code changes.

This module is self-contained — no dependency on t4_optimizer. compute_total
logic for T4 is duplicated here on purpose: scoring must work against
arbitrary agent output, not just optimal sourcing.
"""

from __future__ import annotations

import math
from typing import Any


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


# ---- T2 ---------------------------------------------------------------------


def score_t2(agent_output: dict, ground_truth: dict) -> dict:
    items = ground_truth["items_needed"]
    sub = {
        "item_match": [],
        "pack_extraction": [],
        "quantity_correctness": [],
        "cart_added": [],
    }
    cart_lookup = {entry["item_id"]: entry for entry in agent_output.get("cart", [])}

    for gt_item in items:
        item_id = gt_item["item_id"]
        gt = gt_item["ground_truth"]
        agent_entry = cart_lookup.get(item_id)
        if agent_entry is None:
            for k in sub:
                sub[k].append(0)
            continue

        acceptable = {gt["product_url"]} | set(gt.get("alternative_acceptable_products", []) or [])
        sub["item_match"].append(1 if agent_entry.get("product_url") in acceptable else 0)
        sub["pack_extraction"].append(
            1 if agent_entry.get("extracted_case_pack_size") == gt["case_pack_size"] else 0
        )
        try:
            expected = math.ceil(gt_item["weekly_usage"] / agent_entry["extracted_case_pack_size"])
            sub["quantity_correctness"].append(
                1 if agent_entry["cases_added_to_cart"] == expected else 0
            )
        except (KeyError, ZeroDivisionError, TypeError):
            sub["quantity_correctness"].append(0)
        sub["cart_added"].append(1)

    n = len(items)
    completeness = sum(sub["cart_added"]) / n
    headline = (
        0.30 * _mean(sub["item_match"])
        + 0.30 * _mean(sub["pack_extraction"])
        + 0.25 * _mean(sub["quantity_correctness"])
        + 0.10 * _mean(sub["cart_added"])
        + 0.05 * completeness
    )
    return {
        **{k: _mean(v) for k, v in sub.items()},
        "completeness": completeness,
        "headline": headline,
        "per_item": [{k: v[i] for k, v in sub.items()} for i in range(n)],
    }


# ---- T3 ---------------------------------------------------------------------

_T3_FILTER_INDICATORS = (
    "filter", "facet", "checkbox", "waterproof",
    "height", "price-range", "size-filter",
)


def _t3_check_filter_engagement(trajectory):
    for action in (trajectory or []):
        target = str(action.get("target", "")).lower()
        url = str(action.get("url", "")).lower()
        if any(ind in target or ind in url for ind in _T3_FILTER_INDICATORS):
            return 1
    return 0


def _t3_finalize(scores):
    hard_keys = [
        "category_match", "waterproof_match", "height_match",
        "price_match", "size_match", "weight_match",
    ]
    hard = [scores[k] for k in hard_keys]
    hard_score = sum(hard) / 6
    all_hard = 1 if all(hard) else 0
    scores["hard_constraint_score"] = hard_score
    scores["all_hard_satisfied"] = all_hard
    scores["headline"] = (
        0.40 * hard_score
        + 0.25 * scores["rating_optimal"]
        + 0.15 * scores["cart_state"]
        + 0.10 * scores["filter_engagement"]
        + 0.10 * all_hard
    )
    return scores


def score_t3(agent_output: dict, ground_truth: dict, trajectory: list | None = None) -> dict:
    scores = {k: 0 for k in [
        "filter_engagement", "category_match", "waterproof_match", "height_match",
        "price_match", "size_match", "weight_match", "rating_optimal", "cart_state",
    ]}
    try:
        chosen = agent_output["chosen_product"]
    except (KeyError, TypeError):
        return _t3_finalize(scores)

    cart = agent_output.get("cart_added") or {}
    scores["filter_engagement"] = _t3_check_filter_engagement(trajectory)

    archived = {p["product_url"]: p for p in ground_truth.get("archived_products", [])}
    p = archived.get(chosen.get("product_url"))
    if p is None:
        return _t3_finalize(scores)

    c = ground_truth["constraints"]
    scores["category_match"] = 1 if p.get("category") == c["category"] else 0
    scores["waterproof_match"] = 1 if p.get("is_waterproof") else 0
    scores["height_match"] = 1 if p.get("height_category") in c["height_in"] else 0
    scores["price_match"] = 1 if (p.get("price") is not None and p["price"] < c["price_max"]) else 0
    scores["size_match"] = 1 if p.get("sizes_in_stock", {}).get(c["size_required"], False) else 0
    scores["weight_match"] = 1 if (p.get("weight_oz") is not None and p["weight_oz"] < c["weight_oz_max"]) else 0
    scores["rating_optimal"] = 1 if chosen.get("product_url") == ground_truth.get("optimal_product_url") else 0
    scores["cart_state"] = 1 if (
        cart.get("size") == "10.5"
        and cart.get("quantity") == 1
        and cart.get("in_stock_at_add") is True
    ) else 0
    return _t3_finalize(scores)


# ---- T4 ---------------------------------------------------------------------


def _t4_order_total(seller, listings):
    if not listings:
        return 0.0
    subtotal = sum(l["price"] for l in listings)
    if seller == "AbeBooks":
        return subtotal + sum(l.get("shipping", 0.0) for l in listings)
    if seller == "ThriftBooks":
        return subtotal + (0.0 if subtotal >= 15.0 else 4.99)
    if seller == "BetterWorldBooks":
        return subtotal
    if seller == "Powells":
        return subtotal + 4.99
    raise ValueError(f"Unknown seller: {seller}")


def _t4_listings_index(ground_truth):
    return {
        (b["isbn"], l.get("listing_url")): l
        for b in ground_truth["books"]
        for seller_listings in b["listings"].values()
        for l in seller_listings
    }


def _t4_recompute_total(sourcing, ground_truth):
    """Recompute total from agent's sourcing using each seller's shipping rule.

    Per-listing AbeBooks shipping is taken from the ground-truth listing when
    the agent's listing_url resolves; otherwise from the agent's claimed
    `shipping` field; otherwise treated as 0.
    """
    idx = _t4_listings_index(ground_truth)
    by_seller = {}
    for isbn, choice in sourcing.items():
        seller = choice.get("seller")
        if seller is None:
            continue
        gt_listing = idx.get((isbn, choice.get("listing_url"))) or {}
        merged = {"price": float(choice.get("price", gt_listing.get("price", 0)))}
        if seller == "AbeBooks":
            merged["shipping"] = float(gt_listing.get("shipping", choice.get("shipping", 0)))
        by_seller.setdefault(seller, []).append(merged)
    return round(sum(_t4_order_total(s, lst) for s, lst in by_seller.items()), 2)


_T4_DEFAULT_WEIGHTS = {
    "validity": 0.40,
    "completeness": 0.30,
    "cost_correctness": 0.20,
    "optimality": 0.10,
}


def score_t4(agent_output: dict, ground_truth: dict) -> dict:
    sourcing = agent_output.get("sourcing", {}) or {}
    idx = _t4_listings_index(ground_truth)

    valid = 0
    complete = 0
    for book in ground_truth["books"]:
        isbn = book["isbn"]
        choice = sourcing.get(isbn)
        if choice is None:
            continue
        complete += 1
        gt_listing = idx.get((isbn, choice.get("listing_url")))
        if gt_listing is not None and abs(gt_listing.get("price", -1) - choice.get("price", -2)) < 0.01:
            valid += 1

    n = len(ground_truth["books"])
    validity = valid / n
    completeness = complete / n

    recomputed = _t4_recompute_total(sourcing, ground_truth)
    claimed = float(agent_output.get("total_cost", 0))
    cost_correctness = 1 if abs(recomputed - claimed) < 0.50 else 0

    if validity < 1.0 or completeness < 1.0:
        optimality = 0.0
    else:
        opt = ground_truth.get("optimal_total_cost") or 0
        optimality = max(0.0, 1 - (recomputed - opt) / opt) if opt else 0.0

    weights = ground_truth.get("headline_weights") or _T4_DEFAULT_WEIGHTS
    headline = (
        weights["validity"] * validity
        + weights["completeness"] * completeness
        + weights["cost_correctness"] * cost_correctness
        + weights["optimality"] * optimality
    )
    return {
        "validity": validity,
        "completeness": completeness,
        "cost_correctness": cost_correctness,
        "optimality": optimality,
        "recomputed_total": recomputed,
        "claimed_total": claimed,
        "optimum": ground_truth.get("optimal_total_cost"),
        "headline": headline,
        "weights_used": weights,
    }


# ---- CLI --------------------------------------------------------------------


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Score an agent's output against ground truth.")
    parser.add_argument("task", choices=["t2", "t3", "t4"])
    parser.add_argument("--agent", required=True, type=Path,
                        help="Path to agent output JSON")
    parser.add_argument("--gt", required=True, type=Path,
                        help="Path to ground-truth JSON")
    parser.add_argument("--trajectory", type=Path, default=None,
                        help="Path to trajectory JSON (T3 only)")
    args = parser.parse_args()

    agent = json.loads(args.agent.read_text())
    gt = json.loads(args.gt.read_text())

    if args.task == "t2":
        result = score_t2(agent, gt)
    elif args.task == "t3":
        traj = json.loads(args.trajectory.read_text()) if args.trajectory else []
        result = score_t3(agent, gt, traj)
    else:
        result = score_t4(agent, gt)

    json.dump(result, sys.stdout, indent=2)
    sys.stdout.write("\n")
