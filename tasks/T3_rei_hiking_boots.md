# T3: REI Hiking Boot Constraint Search

**Task type:** Buy-side, single-source constraint satisfaction with optimization
**Domain:** Outdoor gear
**Source:** REI.com
**Capability tested:** Faceted filter use, multi-constraint reasoning, detail-page extraction for non-filterable attributes, optimization over a qualifying set

---

## Persona

A backpacker preparing for a multi-day trip in the Sierra Nevada in early autumn. Needs reliable, waterproof boots for mixed terrain. Has a budget cap, a preferred ankle-support level, and a known shoe size.

## Task description

The agent must find a pair of men's hiking boots on REI.com satisfying six hard constraints, choose the highest-rated option among qualifiers, and add size 10.5 to the cart.

Five of the six hard constraints are exposed in REI's faceted filter UI (category, waterproof, height, price, size availability). The sixth — pair weight — is **only** available on the product detail page. This forces the agent to use filters to narrow the qualifying set, then read each remaining product page to verify weight.

## Constraints

### Hard constraints (all must be satisfied)

| Constraint | Specification |
|---|---|
| Category | Men's hiking boots (not trail runners, not low-cut hikers) |
| Waterproof | Must have a waterproof membrane (Gore-Tex, eVent, or proprietary equivalent) |
| Height | Mid-cut or high-cut ankle support; low-cut excluded |
| Price | Listed price strictly under $250 USD |
| Size availability | Size 10.5 (US Men's) must be in stock and addable to cart |
| Pair weight | Total pair weight under 2.5 lbs (40 oz) |

### Soft preference

Among boots meeting all hard constraints, prefer the one with the highest average customer rating. If multiple boots tie on rating, prefer the one with the most reviews.

## Agent prompt

```
You are helping a backpacker buy hiking boots on REI.com.

TASK
Find a pair of men's hiking boots on REI.com that satisfies all of the following hard constraints, and add size 10.5 to the cart.

HARD CONSTRAINTS (all must be met)
1. Category: Men's hiking boots — not trail runners, not low-cut hiking shoes
2. Waterproof: Must have a waterproof membrane (Gore-Tex, eVent, or proprietary equivalent)
3. Height: Mid-cut or high-cut ankle support — low-cut shoes are not eligible
4. Price: Listed price strictly under $250 USD
5. Size availability: Size 10.5 (US Men's) must be in stock and able to be added to cart
6. Pair weight: Total pair weight strictly under 2.5 lbs (40 oz). Weight is on the product specifications section of each product page; you may need to scroll or expand a section to find it.

SOFT PREFERENCE
Of all boots meeting the hard constraints, pick the one with the highest average customer rating. If multiple boots are tied on rating, prefer the one with the most reviews.

INSTRUCTIONS
- Use REI's filtering tools to narrow results before evaluating individual products. Filters available on the men's hiking boots page include waterproof, height, price, and size.
- Weight is NOT in the filter sidebar. You will need to open product pages to verify weight for each candidate.
- Verify size 10.5 is in stock before committing to a product.
- Once you have selected the boot meeting all hard constraints with the highest rating, add size 10.5 (one pair) to the cart.

OUTPUT
After completing the task, return a single JSON object with this structure:

{
  "chosen_product": {
    "product_url": "<full URL of the product page>",
    "product_name": "<exact product name>"
  },
  "extracted_attributes": {
    "price": <number, USD>,
    "weight_oz": <number, total pair weight in ounces>,
    "waterproof": <true|false>,
    "height_category": "<low|mid|high>",
    "rating": <number, average customer rating>,
    "review_count": <integer, number of reviews>
  },
  "cart_added": {
    "size": "10.5",
    "quantity": 1,
    "in_stock_at_add": <true|false>
  },
  "reasoning": "<brief explanation: which constraints narrowed the choice, and why this boot was selected>"
}

Output only the JSON object. No other commentary.
```

## Ground-truth schema

The ground truth is built after archive capture by walking every captured product, extracting attributes, applying the constraint filter offline, and identifying the unique optimal answer.

```json
{
  "task_id": "T3_rei_hiking_boots",
  "constraints": {
    "category": "mens_hiking_boots",
    "waterproof": true,
    "height_in": ["mid", "high"],
    "price_max": 250,
    "size_required": "10.5",
    "weight_oz_max": 40
  },
  "archived_products": [
    {
      "product_url": "https://www.rei.com/product/...",
      "product_name": "<exact product name>",
      "category": "mens_hiking_boots",
      "is_waterproof": true,
      "height_category": "mid",
      "price": 219.95,
      "sizes_in_stock": {
        "10.5": true,
        "11.0": true
      },
      "weight_oz": 36.4,
      "rating": 4.6,
      "review_count": 287
    }
  ],
  "qualifying_product_urls": [
    "https://www.rei.com/product/...",
    "https://www.rei.com/product/..."
  ],
  "optimal_product_url": "https://www.rei.com/product/..."
}
```

## Success criteria

| Score | Description |
|---|---|
| `filter_engagement` | Did the agent's action trajectory show interaction with REI's filter UI (rather than just scrolling)? |
| `category_match` | Final product is in men's hiking boots category |
| `waterproof_match` | Final product is waterproof per its archived product page |
| `height_match` | Final product is mid- or high-cut |
| `price_match` | Final product price is under $250 |
| `size_match` | Size 10.5 is in stock per archived state |
| `weight_match` | Final product's listed weight is under 40 oz per pair |
| `rating_optimal` | Of all archived products meeting hard constraints, did the agent pick the highest-rated (ties broken by review count)? |
| `cart_state` | Boot in size 10.5 was successfully added to cart |

### Headline aggregate

```
hard_constraint_score = mean([category_match, waterproof_match, height_match,
                              price_match, size_match, weight_match])
all_hard_satisfied = 1 if all six are satisfied else 0

headline = 0.40 × hard_constraint_score
         + 0.25 × rating_optimal
         + 0.15 × cart_state
         + 0.10 × filter_engagement
         + 0.10 × all_hard_satisfied
```

The `all_hard_satisfied` bonus rewards finding *any* qualifying boot over partially-qualifying ones. The `rating_optimal` weight is significant because it tests the optimization step distinct from constraint satisfaction.

## Scoring code outline

```python
def score_t3(agent_output, ground_truth, trajectory):
    scores = {
        "filter_engagement": 0,
        "category_match": 0,
        "waterproof_match": 0,
        "height_match": 0,
        "price_match": 0,
        "size_match": 0,
        "weight_match": 0,
        "rating_optimal": 0,
        "cart_state": 0
    }

    try:
        chosen = agent_output["chosen_product"]
        attrs = agent_output["extracted_attributes"]
        cart = agent_output["cart_added"]
    except (KeyError, TypeError):
        return finalize(scores)

    # Filter engagement is checked from trajectory, not output JSON
    scores["filter_engagement"] = check_filter_engagement(trajectory)

    # Look up chosen product in archive
    archived_lookup = {p["product_url"]: p for p in ground_truth["archived_products"]}
    archived_product = archived_lookup.get(chosen.get("product_url"))

    if archived_product is None:
        # Hallucinated product URL — every product-related score stays at 0
        return finalize(scores)

    constraints = ground_truth["constraints"]

    scores["category_match"] = 1 if archived_product["category"] == constraints["category"] else 0
    scores["waterproof_match"] = 1 if archived_product["is_waterproof"] else 0
    scores["height_match"] = 1 if archived_product["height_category"] in constraints["height_in"] else 0
    scores["price_match"] = 1 if archived_product["price"] < constraints["price_max"] else 0
    scores["size_match"] = 1 if archived_product["sizes_in_stock"].get(constraints["size_required"], False) else 0
    scores["weight_match"] = 1 if archived_product["weight_oz"] < constraints["weight_oz_max"] else 0

    scores["rating_optimal"] = 1 if chosen.get("product_url") == ground_truth["optimal_product_url"] else 0

    scores["cart_state"] = 1 if (
        cart.get("size") == "10.5"
        and cart.get("quantity") == 1
        and cart.get("in_stock_at_add") is True
    ) else 0

    return finalize(scores)


def finalize(scores):
    hard = [
        scores["category_match"], scores["waterproof_match"], scores["height_match"],
        scores["price_match"], scores["size_match"], scores["weight_match"]
    ]
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


def check_filter_engagement(trajectory):
    """Return 1 if the trajectory shows interaction with filter UI elements."""
    filter_indicators = [
        "filter", "facet", "checkbox", "waterproof",
        "height", "price-range", "size-filter"
    ]
    for action in trajectory:
        target = str(action.get("target", "")).lower()
        url = str(action.get("url", "")).lower()
        if any(ind in target or ind in url for ind in filter_indicators):
            return 1
    return 0
```

## Archive capture requirements

The pywb archive must support filter use, product page reading, and add-to-cart actions. Capture:

- REI men's hiking boots category landing page (unfiltered)
- Filter-applied result pages for plausible filter combinations the agent might construct (waterproof on, mid-height on, price-under-$250 on)
- Product detail pages for every product appearing in any filtered result set — realistically 20-40 detail pages
- Each detail page must include the specifications section showing weight (this often requires expanding an accordion; capture the expanded state)
- Size selector state showing 10.5 availability per product
- Add-to-cart flow on at least the optimal product

### Risks

The weight specification often lives in a "Specs" or "Details" accordion that's collapsed by default. Verify that captured pages show the weight in a state the agent can read. If the weight is only revealed by JavaScript expansion, capture the expanded version.

REI's filter URLs use query parameters, so filter-clicking should replay correctly. Verify this in the spike.

The add-to-cart flow is the highest-risk piece for replay. If it doesn't work deterministically, fall back to scoring on the product page reached rather than cart state.

## Calibration of the qualifying set

Before locking the task, verify the qualifying set has the right shape:

- **Too small** (1-2 products): the rating-optimization score becomes trivial. Loosen one constraint or capture more boots.
- **Just right** (5-15 products): the optimization is non-trivial but the unique correct answer is well-defined. Verify rating spread is meaningful — at least 0.2 stars between the optimum and the runner-up.
- **Too large** (40+ products): the eval doesn't differentiate models well. Tighten a constraint.

The optimum must be unique. If the top two products have identical rating and review count, modify constraints or pick a different size requirement that breaks the tie.
