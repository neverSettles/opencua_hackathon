# T2: Restaurant Supply Restock

**Task type:** Buy-side, single-source detail extraction
**Domain:** Restaurant supply
**Source:** WebstaurantStore
**Capability tested:** Within-source navigation, structured fact extraction from detail pages, arithmetic application, repetition reliability

---

## Persona

A small-restaurant owner doing a weekly restock of disposables and consumables. The owner knows roughly how many of each item they go through per week and needs to order enough cases to meet next week's demand. They are not optimizing across vendors — WebstaurantStore is their established supplier.

## Task description

The agent is given six items the restaurant needs to restock, with a weekly usage estimate for each. For each item, the agent must:

1. Search WebstaurantStore for a product matching the description
2. Navigate to the product detail page
3. Extract the case-pack size from the product page (e.g. "Case of 1000," "Pack of 200")
4. Calculate the number of cases needed: `ceil(weekly_usage / case_pack_size)`
5. Add that quantity of cases to the cart
6. Repeat for all six items

The agent then returns a structured JSON output describing what was added.

## Input items

```json
{
  "items_needed": [
    {
      "item_id": "item_001",
      "description": "16oz disposable coffee cups, white",
      "weekly_usage": 2400
    },
    {
      "item_id": "item_002",
      "description": "9-inch white paper plates, heavy duty",
      "weekly_usage": 1500
    },
    {
      "item_id": "item_003",
      "description": "Black nitrile gloves, large, powder-free",
      "weekly_usage": 800
    },
    {
      "item_id": "item_004",
      "description": "32oz translucent deli containers with lids",
      "weekly_usage": 600
    },
    {
      "item_id": "item_005",
      "description": "13-gallon trash can liners, clear",
      "weekly_usage": 200
    },
    {
      "item_id": "item_006",
      "description": "Stainless steel scouring pads, heavy duty",
      "weekly_usage": 50
    }
  ]
}
```

## Agent prompt

```
You are helping a small-restaurant owner do a weekly restock of disposables on WebstaurantStore.

For each item below, you need to:
1. Search WebstaurantStore for a product matching the description.
2. Open the product page for the best match.
3. Read the product page to determine the case-pack size (e.g., "Case of 1000," "Pack of 200").
4. Calculate the number of cases needed to cover the weekly usage. Round up — you cannot order partial cases.
5. Add that number of cases to the cart.

Repeat for all six items. Then return your final cart contents as a JSON object.

ITEMS NEEDED
1. 16oz disposable coffee cups, white — weekly usage: 2400
2. 9-inch white paper plates, heavy duty — weekly usage: 1500
3. Black nitrile gloves, large, powder-free — weekly usage: 800
4. 32oz translucent deli containers with lids — weekly usage: 600
5. 13-gallon trash can liners, clear — weekly usage: 200
6. Stainless steel scouring pads, heavy duty — weekly usage: 50

RULES
- Pick a product whose description reasonably matches the item needed. Do not substitute meaningfully different products (e.g., do not pick foam cups for paper cups).
- The case-pack size is on the product page, usually in the title, the specifications section, or the unit-of-measure dropdown.
- If a product page lists multiple pack-size options (e.g., "Pack" vs "Case"), use the largest reasonable unit and verify the count.
- If you cannot find a matching product after a reasonable search, record the failure rather than substituting wildly.

OUTPUT FORMAT
After completing all six items, return a single JSON object matching this schema:

{
  "cart": [
    {
      "item_id": "item_001",
      "product_name_chosen": "<exact product name from WebstaurantStore>",
      "product_url": "<URL of the product page>",
      "extracted_case_pack_size": <integer, units per case>,
      "cases_added_to_cart": <integer, number of cases added>,
      "calculated_total_units": <integer, cases × case_pack_size>
    }
  ],
  "total_items_added": <integer, count of items successfully added>,
  "failures": [
    {
      "item_id": "<id>",
      "reason": "<brief explanation>"
    }
  ]
}

Output only the JSON object. No other commentary.
```

## Ground-truth schema

The ground truth is built once during archive capture. For each item, the team records the correct product, its case-pack size, and the expected number of cases.

```json
{
  "task_id": "T2_restaurant_restock",
  "items_needed": [
    {
      "item_id": "item_001",
      "description": "16oz disposable coffee cups, white",
      "weekly_usage": 2400,
      "ground_truth": {
        "product_url": "https://www.webstaurantstore.com/[specific-product-path]",
        "product_name": "<exact product name>",
        "case_pack_size": 1000,
        "expected_cases": 3,
        "alternative_acceptable_products": [
          "<URL of equally-acceptable alternate product>"
        ]
      }
    }
  ]
}
```

`alternative_acceptable_products` accommodates the reality that more than one product may legitimately match a description. The agent's chosen product passes the item-match check if it is the primary ground-truth product OR any of the listed alternatives.

## Catalog reality notes

When ground truth was collected against the live WebstaurantStore catalog (2026-05-09), three of the six items had no clean canonical SKU. The eval relies on the `alternative_acceptable_products` mechanism rather than tightening the spec, so any reasonable agent-pick passes `item_match`:

- **item_002** (heavy-duty 9" white paper plates): no SKU is labeled "heavy duty". The closest match is `Choice 9" White Coated Paper Plate - 1,000/Case`. The uncoated `Choice 9" White Uncoated Paper Plate - 1,200/Case` is included as an alternative.
- **item_005** (13-gallon clear can liners): WebstaurantStore catalogs liners by the standard 12–16 gallon tall-kitchen range; there is no 13-gallon-only clear SKU. The Lavex Li'l Herc 12–16 Gallon clear liner is treated as canonical.
- **item_006** (heavy-duty stainless steel scouring pads): the catalog sells these as `12/Pack` scrubbers, not "cases". The pack is treated as the case for arithmetic purposes (`expected_cases = ceil(weekly_usage / 12)`).

## Success criteria

Per item, four binary sub-scores:

| Score | Description |
|---|---|
| `item_match` | Did the agent find a product whose URL matches the ground-truth primary or alternative? |
| `pack_extraction` | Did the agent's `extracted_case_pack_size` match the ground-truth case pack size? |
| `quantity_correctness` | Did the agent's `cases_added_to_cart` equal `ceil(weekly_usage / case_pack_size)`? |
| `cart_added` | Was the item actually added to cart (per cart-state observation)? |

Plus aggregate:

| Score | Description |
|---|---|
| `completeness` | Number of items in cart / 6 |

### Headline aggregate

```
headline = 0.30 × mean(item_match)
         + 0.30 × mean(pack_extraction)
         + 0.25 × mean(quantity_correctness)
         + 0.10 × mean(cart_added)
         + 0.05 × completeness
```

The weighting puts most of the score on the things only this task tests well — product matching from descriptions, structured fact extraction (pack sizes), and arithmetic application. The cart-added and completeness checks act as sanity gates rather than primary signals.

## Scoring code outline

```python
import math

def score_t2(agent_output, ground_truth):
    items = ground_truth["items_needed"]
    sub_scores = {
        "item_match": [],
        "pack_extraction": [],
        "quantity_correctness": [],
        "cart_added": []
    }

    cart_lookup = {entry["item_id"]: entry for entry in agent_output.get("cart", [])}

    for gt_item in items:
        item_id = gt_item["item_id"]
        gt = gt_item["ground_truth"]
        agent_entry = cart_lookup.get(item_id)

        if agent_entry is None:
            for k in sub_scores:
                sub_scores[k].append(0)
            continue

        # Item match: does the chosen product URL match primary or alternative?
        acceptable_urls = {gt["product_url"]} | set(gt.get("alternative_acceptable_products", []))
        item_match = 1 if agent_entry.get("product_url") in acceptable_urls else 0
        sub_scores["item_match"].append(item_match)

        # Pack extraction
        pack_match = 1 if agent_entry.get("extracted_case_pack_size") == gt["case_pack_size"] else 0
        sub_scores["pack_extraction"].append(pack_match)

        # Quantity correctness: math is correct given the case-pack size the agent extracted
        # Note: scored against agent's own extracted pack size, not ground truth — this isolates
        # arithmetic from extraction so we see them as separate failure modes.
        try:
            expected_from_extracted = math.ceil(
                gt_item["weekly_usage"] / agent_entry["extracted_case_pack_size"]
            )
            qty_match = 1 if agent_entry["cases_added_to_cart"] == expected_from_extracted else 0
        except (ZeroDivisionError, KeyError, TypeError):
            qty_match = 0
        sub_scores["quantity_correctness"].append(qty_match)

        # Cart added
        sub_scores["cart_added"].append(1)  # presence in cart_lookup means it was added

    completeness = sum(sub_scores["cart_added"]) / 6

    headline = (
        0.30 * mean(sub_scores["item_match"])
        + 0.30 * mean(sub_scores["pack_extraction"])
        + 0.25 * mean(sub_scores["quantity_correctness"])
        + 0.10 * mean(sub_scores["cart_added"])
        + 0.05 * completeness
    )

    return {
        **{k: mean(v) for k, v in sub_scores.items()},
        "completeness": completeness,
        "headline": headline,
        "per_item": [
            {k: v[i] for k, v in sub_scores.items()} for i in range(6)
        ]
    }


def mean(xs):
    return sum(xs) / len(xs) if xs else 0
```

## Archive capture requirements

The pywb archive must support the agent's full trajectory. Capture:

- WebstaurantStore homepage
- Search results pages for each of the six item descriptions (search the agent's likely query strings, capture the top page of results)
- Product detail pages for the ground-truth product AND at least one or two plausible alternates per item
- The cart page in its empty state and after each add-to-cart action
- Any intermediate "added to cart" confirmation overlays

Risk areas: cart-state mutation may rely on JavaScript and AJAX calls. Test the cart-add replay specifically before locking the task in. If cart state doesn't replay deterministically, fall back to scoring on the last product page reached rather than the cart state.
