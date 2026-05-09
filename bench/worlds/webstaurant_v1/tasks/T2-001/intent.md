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
