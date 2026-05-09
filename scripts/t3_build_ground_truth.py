"""Build T3 ground truth from the REI listing-page extract.

Input: ground_truth/_rei_raw.json (raw scrape of all 146 products under
filters waterproof + over-the-ankle + size 10.5).

Output: ground_truth/T3_ground_truth.json with full archived_products list,
qualifying_product_urls, optimal_product_url.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "ground_truth" / "_rei_raw.json"
OUT = ROOT / "ground_truth" / "T3_ground_truth.json"

# Hard constraints (mirror the spec)
PRICE_MAX = 250.0
WEIGHT_OZ_MAX = 40.0
SIZE_REQUIRED = "10.5"
MIN_REVIEW_THRESHOLD = 100  # added to make optimum unique; see calibration_note


def main():
    raw = json.loads(RAW.read_text())
    archived = []
    for p in raw:
        archived.append({
            "product_url": p["href"],
            "product_name": (p.get("brand", "") + " " + p.get("model", "")).strip(),
            "category": "mens_hiking_boots",
            "is_waterproof": True,           # filter applied
            "height_category": "mid",        # over-the-ankle filter ~= mid-cut
            "price": p.get("price"),
            "sizes_in_stock": {"10.5": True},  # size-10.5 filter applied
            "weight_oz": p.get("weight_oz"),
            "weight_text": p.get("weight_text"),
            "rating": p.get("rating"),
            "review_count": p.get("reviews"),
        })

    def passes_hard(p):
        return (
            p["price"] is not None and p["price"] < PRICE_MAX
            and p["weight_oz"] is not None and p["weight_oz"] < WEIGHT_OZ_MAX
            and p["is_waterproof"]
            and p["height_category"] in ("mid", "high")
            and p["sizes_in_stock"].get(SIZE_REQUIRED) is True
            and p["rating"] is not None
        )

    qualifying = [p for p in archived if passes_hard(p)]
    optimum_pool = [p for p in qualifying if (p["review_count"] or 0) >= MIN_REVIEW_THRESHOLD]
    # tiebreak: highest rating, then most reviews
    optimum_pool.sort(key=lambda p: (-p["rating"], -(p["review_count"] or 0)))
    optimal = optimum_pool[0] if optimum_pool else None

    out = {
        "task_id": "T3_rei_hiking_boots",
        "source": "https://www.rei.com",
        "collected_via": "live_browse_listing_extraction",
        "collection_date": "2026-05-09",
        "constraints": {
            "category": "mens_hiking_boots",
            "waterproof": True,
            "height_in": ["mid", "high"],
            "price_max": PRICE_MAX,
            "size_required": SIZE_REQUIRED,
            "weight_oz_max": WEIGHT_OZ_MAX,
            "min_reviews_for_optimum_pool": MIN_REVIEW_THRESHOLD,
        },
        "calibration_note": (
            "Without a minimum-review threshold, three boots tied at 5.0 stars "
            "with 1 review each would all be optimal, breaking the eval. We "
            "added a >=100-reviews threshold so the rating signal is "
            "statistically meaningful. The agent prompt should be updated to "
            "match: 'among boots with at least 100 reviews, prefer highest "
            "rating; ties broken by review count.'"
        ),
        "archived_products": archived,
        "qualifying_product_urls": [p["product_url"] for p in qualifying],
        "qualifying_count": len(qualifying),
        "qualifying_with_min_reviews_count": len(optimum_pool),
        "optimal_product_url": optimal["product_url"] if optimal else None,
        "optimal_product_name": optimal["product_name"] if optimal else None,
        "optimal_attributes": optimal,
        "rating_spread": (
            optimum_pool[0]["rating"] - optimum_pool[1]["rating"]
            if len(optimum_pool) >= 2 else None
        ),
    }

    OUT.write_text(json.dumps(out, indent=2) + "\n")

    print(f"Total scraped: {len(archived)}")
    print(f"Qualifying (hard constraints): {len(qualifying)}")
    print(f"Qualifying with >= {MIN_REVIEW_THRESHOLD} reviews: {len(optimum_pool)}")
    print(f"Optimal: {optimal['product_name']} | {optimal['rating']}* / {optimal['review_count']} reviews / ${optimal['price']} / {optimal['weight_oz']} oz")
    print(f"Runner-up: {optimum_pool[1]['product_name']} | {optimum_pool[1]['rating']}* / {optimum_pool[1]['review_count']} reviews")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
