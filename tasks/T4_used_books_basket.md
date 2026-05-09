# T4: Used Book Basket Across Multiple Sellers

**Task type:** Buy-side, multi-source basket optimization
**Domain:** Used books
**Sources:** AbeBooks, ThriftBooks, Better World Books, Powell's
**Capability tested:** Cross-source navigation, structured listing extraction, constrained optimization with shipping-threshold non-linearities, basket consolidation reasoning

---

## Persona

A reader assembling a small reading list. Has six specific books in mind, identified by ISBN. Wants the cheapest valid total cost across the four major used-book sellers, accounting for each seller's shipping policy. Cares about getting the right edition and a copy in good-or-better condition; doesn't care which specific seller fulfills which book.

## Task description

The agent is given six specific ISBNs and a fixed set of four sellers. For each book, the agent searches each seller for the ISBN, identifies eligible copies (Good condition or better, USD-priced), and records pricing. The agent then computes the optimal sourcing — which seller to buy each book from — to minimize total cost including shipping, applying each seller's stated shipping rules. The output is a structured sourcing recommendation with the computed total.

The optimization is non-trivial because:
- ThriftBooks has a free-shipping threshold that creates consolidation incentives.
- AbeBooks shipping is per-listing-per-seller, which can favor or disfavor consolidation depending on prices.
- Some books may not be available at all four sellers, forcing routing constraints.
- Powell's flat per-order shipping means each Powell's-routed book amortizes against the others routed there.

A naive strategy ("buy each book from its individually cheapest seller") will typically be meaningfully suboptimal once shipping rules are applied.

## Books

| # | Title | Author | ISBN | Role |
|---|---|---|---|---|
| 1 | The Kite Runner | Khaled Hosseini | 9781594631931 | Cheap common (consolidation candidate) |
| 2 | The Curious Incident of the Dog in the Night-Time | Mark Haddon | 9781400032716 | Cheap common (consolidation candidate) |
| 3 | The Sixth Extinction | Elizabeth Kolbert | 9781250062185 | Mid-priced anchor |
| 4 | The Power Broker | Robert Caro | 9780394720241 | Expensive (price-driven sourcing) |
| 5 | Speedboat | Renata Adler | 9781590176160 | Limited availability (routing constraint) |
| 6 | Slaughterhouse-Five | Kurt Vonnegut | 9780385333849 | Edition variant (ISBN-precise match required) |

## Sellers and shipping rules

| Seller | Shipping rule |
|---|---|
| AbeBooks | Per-seller-listing as quoted on the listing page. Different AbeBooks sellers ship separately. |
| ThriftBooks | Free over $15 order total; otherwise $4.99 flat per order |
| Better World Books | Always free shipping |
| Powell's | $4.99 flat per order |

> **Note:** These shipping rules must be verified against current seller policies before the hackathon and frozen into the task spec. The shipping rules are part of the task — the agent applies them; it does not discover them.

## Constraints

| Constraint | Specification |
|---|---|
| Condition | Good condition or better. Acceptable, Reading Copy, or Poor are not eligible. |
| Currency | USD listings only. International or non-USD listings are not eligible. |
| Edition | The agent must select copies matching the specified ISBN exactly. Other editions of the same title are not eligible. |
| Quantity | One copy of each book. |

## Training set JSON

```json
{
  "task_id": "T4_used_books_basket_training",
  "books": [
    {
      "isbn": "9781594631931",
      "title": "The Kite Runner",
      "author": "Khaled Hosseini",
      "edition_note": "Riverhead Books 10th Anniversary paperback (2013)",
      "role": "cheap_common"
    },
    {
      "isbn": "9781400032716",
      "title": "The Curious Incident of the Dog in the Night-Time",
      "author": "Mark Haddon",
      "edition_note": "Vintage Contemporaries paperback (2004)",
      "role": "cheap_common"
    },
    {
      "isbn": "9781250062185",
      "title": "The Sixth Extinction",
      "author": "Elizabeth Kolbert",
      "edition_note": "Picador paperback (2015)",
      "role": "mid_priced_anchor"
    },
    {
      "isbn": "9780394720241",
      "title": "The Power Broker",
      "author": "Robert Caro",
      "edition_note": "Vintage paperback",
      "role": "expensive"
    },
    {
      "isbn": "9781590176160",
      "title": "Speedboat",
      "author": "Renata Adler",
      "edition_note": "NYRB Classics paperback",
      "role": "limited_availability"
    },
    {
      "isbn": "9780385333849",
      "title": "Slaughterhouse-Five",
      "author": "Kurt Vonnegut",
      "edition_note": "Dial Press paperback - ISBN-precise match required",
      "role": "edition_variant"
    }
  ],
  "sellers": ["AbeBooks", "ThriftBooks", "BetterWorldBooks", "Powells"],
  "constraints": {
    "minimum_condition": "Good",
    "ineligible_conditions": ["Acceptable", "Reading Copy", "Poor"],
    "currency": "USD",
    "quantity_per_book": 1
  },
  "shipping_rules": {
    "AbeBooks": {
      "type": "per_seller_listing",
      "note": "Use the per-listing shipping cost as quoted on each AbeBooks listing page. Different AbeBooks sellers ship separately."
    },
    "ThriftBooks": {
      "type": "tiered",
      "free_shipping_threshold_usd": 15.00,
      "flat_rate_below_threshold_usd": 4.99
    },
    "BetterWorldBooks": {
      "type": "always_free",
      "flat_rate_usd": 0.00
    },
    "Powells": {
      "type": "flat_rate_per_order",
      "flat_rate_usd": 4.99
    }
  },
  "verification_status": {
    "isbns_verified": true,
    "edition_verified": true,
    "shipping_rules_verified": false,
    "availability_verified": false,
    "listings_captured": false,
    "notes": "ISBNs and editions confirmed by team. Shipping rules are placeholders pending verification against current seller policies. Per-seller availability and per-listing pricing must be captured into a separate listings.json file during archive capture."
  }
}
```

## Holdout set JSON

```json
{
  "task_id": "T4_used_books_basket_holdout",
  "books": [
    {
      "isbn": "9780156027328",
      "title": "Life of Pi",
      "author": "Yann Martel",
      "edition_note": "Mariner Books paperback",
      "role": "cheap_common"
    },
    {
      "isbn": "9781594483295",
      "title": "The Brief Wondrous Life of Oscar Wao",
      "author": "Junot Díaz",
      "edition_note": "Riverhead Books paperback",
      "role": "cheap_common"
    },
    {
      "isbn": "9780143110910",
      "title": "Behave: The Biology of Humans at Our Best and Worst",
      "author": "Robert Sapolsky",
      "edition_note": "Penguin Books paperback",
      "role": "mid_priced_anchor"
    },
    {
      "isbn": "9780143037750",
      "title": "Postwar: A History of Europe Since 1945",
      "author": "Tony Judt",
      "edition_note": "Penguin Books paperback",
      "role": "expensive"
    },
    {
      "isbn": "9780312429935",
      "title": "Train Dreams",
      "author": "Denis Johnson",
      "edition_note": "Picador paperback (FSG-published novella reissued by Picador)",
      "role": "limited_availability"
    },
    {
      "isbn": "9780743273565",
      "title": "The Great Gatsby",
      "author": "F. Scott Fitzgerald",
      "edition_note": "Scribner trade paperback - ISBN-precise match required (many editions exist)",
      "role": "edition_variant"
    }
  ],
  "sellers": ["AbeBooks", "ThriftBooks", "BetterWorldBooks", "Powells"],
  "constraints": {
    "minimum_condition": "Good",
    "ineligible_conditions": ["Acceptable", "Reading Copy", "Poor"],
    "currency": "USD",
    "quantity_per_book": 1
  },
  "shipping_rules": {
    "AbeBooks": {
      "type": "per_seller_listing",
      "note": "Use the per-listing shipping cost as quoted on each AbeBooks listing page. Different AbeBooks sellers ship separately."
    },
    "ThriftBooks": {
      "type": "tiered",
      "free_shipping_threshold_usd": 15.00,
      "flat_rate_below_threshold_usd": 4.99
    },
    "BetterWorldBooks": {
      "type": "always_free",
      "flat_rate_usd": 0.00
    },
    "Powells": {
      "type": "flat_rate_per_order",
      "flat_rate_usd": 4.99
    }
  },
  "verification_status": {
    "isbns_verified": false,
    "edition_verified": false,
    "shipping_rules_verified": false,
    "availability_verified": false,
    "listings_captured": false,
    "notes": "ISBNs in this held-out set are unverified candidates. Each must be checked against ISBNdb/WorldCat to confirm it resolves to the intended edition before archive capture. Shipping rules match the training set and should be kept consistent."
  }
}
```

## Agent prompt

```
You are helping a reader assemble a basket of six specific used books at the lowest possible total cost across four sellers.

BOOKS NEEDED (by ISBN)
1. 9781594631931 — The Kite Runner by Khaled Hosseini
2. 9781400032716 — The Curious Incident of the Dog in the Night-Time by Mark Haddon
3. 9781250062185 — The Sixth Extinction by Elizabeth Kolbert
4. 9780394720241 — The Power Broker by Robert Caro
5. 9781590176160 — Speedboat by Renata Adler
6. 9780385333849 — Slaughterhouse-Five by Kurt Vonnegut

ELIGIBLE SELLERS
- AbeBooks
- ThriftBooks
- Better World Books
- Powell's

SHIPPING RULES (apply these to compute totals; do not try to discover them)
- AbeBooks: per-listing as quoted on the listing page. Different AbeBooks sellers ship separately.
- ThriftBooks: free over $15 order total, otherwise $4.99 flat per order
- Better World Books: always free shipping
- Powell's: $4.99 flat per order

CONSTRAINTS
- Condition: Good or better. Do not select copies in Acceptable, Reading Copy, or Poor condition.
- Currency: USD listings only. Skip international or non-USD listings.
- Edition: select copies whose ISBN exactly matches the one above. Other editions of the same title do not count.

YOUR TASK
For each book, search each seller (or as many as needed), and identify the cheapest eligible copy. Then determine the sourcing assignment — which seller to buy each book from — that minimizes the total cost including shipping.

A naive "cheapest-per-book" strategy is often suboptimal. Consider:
- Consolidating multiple books at ThriftBooks to clear the $15 free-shipping threshold
- Routing books to Powell's so they share the $4.99 flat shipping
- The fact that AbeBooks shipping is per-seller, so multiple AbeBooks listings from different sellers each ship separately

Some books may not be available at all four sellers. Route accordingly.

OUTPUT
Return a single JSON object matching this structure:

{
  "sourcing": {
    "9781594631931": {
      "seller": "<one of: AbeBooks | ThriftBooks | BetterWorldBooks | Powells>",
      "condition": "<Good | Very Good | Like New | New>",
      "price": <number, listing price in USD, excluding shipping>,
      "shipping": <number, shipping attributed to this book in USD>,
      "listing_url": "<URL of the specific listing>"
    },
    "9781400032716": { ... },
    "9781250062185": { ... },
    "9780394720241": { ... },
    "9781590176160": { ... },
    "9780385333849": { ... }
  },
  "total_cost": <number, sum of all prices and shipping in USD>,
  "currency": "USD",
  "reasoning": "<brief explanation of the routing strategy and any consolidation decisions>"
}

For shipping attribution within a multi-book seller order, divide that seller's total order shipping equally across the books from that seller.

Output only the JSON object. No other commentary.
```

## Ground-truth schema

The ground truth is built offline after archive capture. For each book, the team records every eligible listing across the four sellers. The optimal sourcing is computed by enumerating all valid assignments and selecting the minimum-cost one given the shipping rules.

```json
{
  "task_id": "T4_used_book_basket_training",
  "books": [
    {
      "isbn": "9781594631931",
      "title": "The Kite Runner",
      "listings": {
        "AbeBooks": [
          {
            "seller_id": "abe_seller_xyz",
            "listing_url": "...",
            "condition": "Good",
            "price": 4.19,
            "shipping": 3.99
          }
        ],
        "ThriftBooks": [
          {
            "listing_url": "...",
            "condition": "Good",
            "price": 4.49
          }
        ],
        "BetterWorldBooks": [
          {
            "listing_url": "...",
            "condition": "Good",
            "price": 5.49
          }
        ],
        "Powells": [
          {
            "listing_url": "...",
            "condition": "Good",
            "price": 5.95
          }
        ]
      }
    }
  ],
  "shipping_rules": {
    "AbeBooks": "per_listing",
    "ThriftBooks": {"threshold": 15.00, "below_threshold_rate": 4.99},
    "BetterWorldBooks": {"flat_rate": 0.00},
    "Powells": {"flat_rate": 4.99}
  },
  "optimal_sourcing": {
    "9781594631931": {"seller": "ThriftBooks", "listing_url": "...", "price": 4.49},
    "...": "..."
  },
  "optimal_total_cost": 0.00,
  "naive_total_cost": 0.00
}
```

The `optimal_total_cost` and `naive_total_cost` are both computed offline. The naive cost is the per-book cheapest assignment with shipping applied; the optimal cost beats it. The gap between them measures how much the optimization actually does.

## Success criteria

| Score | Description |
|---|---|
| `validity` | Fraction of (isbn, seller, listing_url, price) tuples in agent's output that exist in archived data |
| `completeness` | Fraction of the 6 books with a valid sourcing recommendation |
| `cost_correctness` | 1 if agent's claimed `total_cost` matches the recomputed total (within $0.50) given its sourcing choices |
| `optimality` | `max(0, 1 - (agent_total - optimum_total) / optimum_total)` — 1.0 if matching optimum, declines as agent's total exceeds optimum |

### Headline aggregate

```
headline = 0.40 × optimality
         + 0.30 × validity
         + 0.20 × completeness
         + 0.10 × cost_correctness
```

The optimality weight dominates because the optimization is what this task uniquely tests. Validity (no hallucinated listings) and completeness (all six books sourced) act as gates — a model with high optimality but lots of hallucinations is worse than the optimality score alone suggests.

If `validity < 1.0` or `completeness < 1.0`, optimality is set to 0. You cannot be "optimal" if your sourcing includes hallucinated entries or skips books.

## Scoring code outline

```python
def score_t4(agent_output, ground_truth):
    sourcing = agent_output.get("sourcing", {})
    listings_index = build_listings_index(ground_truth)  # (isbn, listing_url) -> listing

    valid_count = 0
    complete_count = 0

    for book in ground_truth["books"]:
        isbn = book["isbn"]
        agent_choice = sourcing.get(isbn)
        if agent_choice is None:
            continue
        complete_count += 1

        listing = listings_index.get((isbn, agent_choice.get("listing_url")))
        if listing is not None:
            # Verify price matches (within rounding)
            if abs(listing["price"] - agent_choice.get("price", -1)) < 0.01:
                valid_count += 1

    validity = valid_count / 6
    completeness = complete_count / 6

    # Recompute total from agent's sourcing using shipping rules
    recomputed_total = compute_total(sourcing, ground_truth["shipping_rules"], ground_truth)
    claimed_total = agent_output.get("total_cost", 0)
    cost_correctness = 1 if abs(recomputed_total - claimed_total) < 0.50 else 0

    # Optimality
    if validity < 1.0 or completeness < 1.0:
        optimality = 0
    else:
        optimum = ground_truth["optimal_total_cost"]
        optimality = max(0, 1 - (recomputed_total - optimum) / optimum)

    headline = (
        0.40 * optimality
        + 0.30 * validity
        + 0.20 * completeness
        + 0.10 * cost_correctness
    )

    return {
        "validity": validity,
        "completeness": completeness,
        "cost_correctness": cost_correctness,
        "optimality": optimality,
        "recomputed_total": recomputed_total,
        "claimed_total": claimed_total,
        "optimum": ground_truth["optimal_total_cost"],
        "headline": headline
    }


def compute_total(sourcing, shipping_rules, ground_truth):
    """Apply each seller's shipping rule to compute the true total."""
    by_seller = {}
    for isbn, choice in sourcing.items():
        seller = choice.get("seller")
        by_seller.setdefault(seller, []).append((isbn, choice))

    total = 0.0
    for seller, items in by_seller.items():
        item_subtotal = sum(c["price"] for _, c in items)
        total += item_subtotal

        if seller == "BetterWorldBooks":
            shipping = 0
        elif seller == "ThriftBooks":
            shipping = 0 if item_subtotal >= 15 else 4.99
        elif seller == "Powells":
            shipping = 4.99
        elif seller == "AbeBooks":
            # Per-listing shipping; sum each listing's quoted shipping
            shipping = 0
            for isbn, choice in items:
                shipping += lookup_listing_shipping(isbn, choice["listing_url"], ground_truth)
        else:
            shipping = 0
        total += shipping

    return round(total, 2)
```

## Archive capture requirements

For each of the six ISBNs on each of the four sellers, capture:

- The ISBN search results page (or title-fallback if the seller's ISBN search is unreliable)
- The listing detail page for at least the cheapest eligible copy
- Pagination or alternate listings if the cheapest eligible copy isn't on the first results page
- Each seller's shipping policy page (so the agent could verify shipping rules if it tries — though the rules are also given in the prompt)

Roughly 50-80 archived pages total for the six books across four sellers. Write a Playwright capture script rather than capturing manually so the process is reproducible.

### Risks

AbeBooks aggregates many independent sellers. The cheapest listing on AbeBooks may be from an international seller in non-USD; capture must include enough listings that an eligible USD copy exists for each book on AbeBooks (or accept that AbeBooks may legitimately have no eligible listing for some books, which is its own routing constraint).

Speedboat (the limited-availability book) must be captured carefully. The hypothesis is that ThriftBooks and Better World Books don't stock it — verify this during capture. If it turns out to be available everywhere, swap to a different limited-availability candidate (NYRB Classics editions, New Directions Pearls, or similar).

## Calibration of the optimization

Before locking the task, verify the optimization is non-trivial:

1. Compute `naive_total_cost` (each book from its individually cheapest seller, shipping applied).
2. Compute `optimal_total_cost` (best assignment found by enumeration).
3. Verify `(naive - optimal) / naive >= 0.10` — at least a 10% gap.

If the gap is smaller than 10%, the optimization isn't doing meaningful work and the eval can't distinguish good optimizers from naive ones. Adjust by:
- Picking books with stronger price variation across sellers
- Tightening shipping rules (lower free-shipping thresholds, higher flat rates)
- Adding more limited-availability constraints

The optimum must also be unique up to ties. If two different assignments produce the same total, the eval can't measure which the agent picked. Adjust until there is a clear best.

## Pre-hackathon prep

1. Verify the six ISBNs resolve to the intended editions (use ISBNdb or WorldCat).
2. Verify each ISBN is currently stocked at the expected sellers in eligible condition.
3. Verify Speedboat (or your chosen limited-availability candidate) is genuinely unavailable at ThriftBooks and Better World Books.
4. Capture the archive across all four sellers for all six books (~50-80 pages).
5. Build the structured listings file from archived data.
6. Compute optimal and naive costs offline. Verify the gap is meaningful (≥10%).
7. Verify the optimum is unique.
8. Test the prompt on Claude or GPT-5 with computer use as a sanity check.
