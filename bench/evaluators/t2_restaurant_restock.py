"""
T2 scorer per `tasks/T2_restaurant_restock.md` spec.

Sub-scores per item (each 0 or 1):
  - item_match           : agent's chosen product_url matches GT primary or alternatives
  - pack_extraction      : agent's extracted_case_pack_size matches GT case_pack_size
  - quantity_correctness : cases_added_to_cart == ceil(weekly_usage / agent_extracted_pack_size)
                           (scored against agent's own extracted pack to isolate arithmetic
                           from extraction)
  - cart_added           : item_id present in agent's cart

Aggregate:
  - completeness         : items_in_cart / 6
  - headline             : 0.30*item_match + 0.30*pack_extraction
                          + 0.25*quantity_correctness + 0.10*cart_added + 0.05*completeness

Ground-truth handling: if `ground_truth` is null for an item (or the whole file is
unpopulated), item_match and pack_extraction return None for that item and are
EXCLUDED from the headline (and from their respective means). This lets us score
agent runs before human-verified GT is populated.

Validity gate: `valid_json` is true iff agent_output parses as JSON and contains a
`cart` list. If false, headline is 0 and every sub-score is 0.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# --- low-level helpers --------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _mean_optional(xs: list[float | None]) -> float | None:
    """Mean over only the non-None entries; None if no entries are present."""
    vals = [x for x in xs if x is not None]
    return (sum(vals) / len(vals)) if vals else None


def extract_json_object(raw: str) -> dict | None:
    """Tolerantly extract a JSON object from agent text (model may surround
    with prose or code fences despite being told not to)."""
    if not raw:
        return None
    s = raw.strip()
    # Try direct parse first.
    try:
        return json.loads(s)
    except Exception:
        pass
    # Strip code fences.
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    # Find the first balanced {...} block.
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
                candidate = s[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


# --- main scorer --------------------------------------------------------------

@dataclass
class T2Score:
    valid_json: bool
    item_match_per: list[int | None] = field(default_factory=list)
    pack_extraction_per: list[int | None] = field(default_factory=list)
    quantity_correctness_per: list[int | None] = field(default_factory=list)
    cart_added_per: list[int] = field(default_factory=list)
    item_match_mean: float | None = None
    pack_extraction_mean: float | None = None
    quantity_correctness_mean: float = 0.0
    cart_added_mean: float = 0.0
    completeness: float = 0.0
    headline: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = self.__dict__.copy()
        return d


def score_t2(agent_output: dict | str, ground_truth: dict, n_items: int = 6) -> T2Score:
    if isinstance(agent_output, str):
        parsed = extract_json_object(agent_output)
    else:
        parsed = agent_output

    score = T2Score(valid_json=False)

    if not isinstance(parsed, dict) or not isinstance(parsed.get("cart"), list):
        score.notes.append("agent_output not parseable as {cart: [...]}.")
        score.headline = 0.0
        return score

    score.valid_json = True

    cart_lookup: dict[str, dict] = {
        e.get("item_id"): e for e in parsed["cart"] if isinstance(e, dict) and e.get("item_id")
    }

    items = ground_truth.get("items_needed", [])
    if len(items) != n_items:
        score.notes.append(
            f"ground_truth.items_needed has {len(items)} entries; expected {n_items}."
        )

    for gt_item in items:
        item_id = gt_item.get("item_id")
        gt = gt_item.get("ground_truth")
        agent_entry = cart_lookup.get(item_id)

        if agent_entry is None:
            score.item_match_per.append(0 if gt else None)
            score.pack_extraction_per.append(0 if gt else None)
            score.quantity_correctness_per.append(0)
            score.cart_added_per.append(0)
            continue

        # cart_added: present in agent's cart
        score.cart_added_per.append(1)

        # item_match (requires GT URL)
        if gt:
            acceptable = {gt.get("product_url")} | set(gt.get("alternative_acceptable_products", []))
            acceptable.discard(None)
            score.item_match_per.append(1 if agent_entry.get("product_url") in acceptable else 0)
        else:
            score.item_match_per.append(None)

        # pack_extraction (requires GT pack size)
        if gt and gt.get("case_pack_size") is not None:
            score.pack_extraction_per.append(
                1 if agent_entry.get("extracted_case_pack_size") == gt["case_pack_size"] else 0
            )
        else:
            score.pack_extraction_per.append(None)

        # quantity_correctness: agent's math against agent's own extracted pack
        try:
            extracted = int(agent_entry["extracted_case_pack_size"])
            agent_cases = int(agent_entry["cases_added_to_cart"])
            expected = math.ceil(gt_item["weekly_usage"] / extracted)
            score.quantity_correctness_per.append(1 if agent_cases == expected else 0)
        except (KeyError, TypeError, ValueError, ZeroDivisionError):
            score.quantity_correctness_per.append(0)

    score.item_match_mean = _mean_optional(score.item_match_per)
    score.pack_extraction_mean = _mean_optional(score.pack_extraction_per)
    score.quantity_correctness_mean = _mean(score.quantity_correctness_per)
    score.cart_added_mean = _mean(score.cart_added_per)
    score.completeness = sum(score.cart_added_per) / max(n_items, 1)

    # Headline: redistribute weight of any skipped (None-mean) sub-score
    # proportionally across the others.
    weights = {
        "item_match": (0.30, score.item_match_mean),
        "pack_extraction": (0.30, score.pack_extraction_mean),
        "quantity_correctness": (0.25, score.quantity_correctness_mean),
        "cart_added": (0.10, score.cart_added_mean),
        "completeness": (0.05, score.completeness),
    }
    available = {k: (w, v) for k, (w, v) in weights.items() if v is not None}
    if not available:
        score.headline = 0.0
    else:
        total_w = sum(w for w, _ in available.values())
        score.headline = sum((w / total_w) * v for w, v in available.values())

    if score.item_match_mean is None:
        score.notes.append("item_match skipped (no ground-truth URLs populated).")
    if score.pack_extraction_mean is None:
        score.notes.append("pack_extraction skipped (no ground-truth pack sizes populated).")

    return score


def load_and_score(
    agent_output_path: str | Path,
    ground_truth_path: str | Path,
) -> T2Score:
    agent_text = Path(agent_output_path).read_text()
    try:
        agent_obj: Any = json.loads(agent_text)
    except Exception:
        agent_obj = agent_text
    gt = json.loads(Path(ground_truth_path).read_text())
    return score_t2(agent_obj, gt)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--agent", required=True, help="Path to agent_response.json (or text)")
    p.add_argument("--ground-truth", required=True, help="Path to ground_truth.json")
    args = p.parse_args()

    s = load_and_score(args.agent, args.ground_truth)
    print(json.dumps(s.to_dict(), indent=2))
