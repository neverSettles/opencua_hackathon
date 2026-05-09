# Buy-Side Optimization Bench — Results

## Can CUA models manage buy-side operations?

**GPT-5.5: yes, with degradation on complex tasks. Northstar CUA Fast: no.**

## Headline Scores

| Task | GPT-5.5 (OpenAI) | Northstar CUA Fast (Tzafon) |
|---|---|---|
| T2 — Single-source restock | **0.87** (n=5) | **0.00** (n=2) |
| T3 — Constrained product search | **0.65** (n=8) | — |
| T4 — Multi-source optimization | **0.57** (n=3) | — |

*Scores are 0–1 weighted headline aggregates. 60-trial run in progress; numbers above are from completed trials so far.*

## Tasks

| ID | Domain | Site(s) | What it tests | Difficulty |
|---|---|---|---|---|
| T2 | Restaurant supplies | WebstaurantStore | Search, extract case-pack size, ceil-division, add to cart × 6 items | Medium |
| T3 | Hiking boots | REI.com | Multi-constraint filtering (waterproof, height, price, size, weight), product selection | Hard |
| T4 | Used books | AbeBooks, ThriftBooks, BWB, Powells | Cross-source price comparison, shipping rule optimization, ISBN matching | Hardest |

## GPT-5.5 Detail

### T2: Restaurant Supply Restock (mean 0.87)

| Trial | Headline | Actions | Time |
|---|---|---|---|
| 1 | 0.85 | 106 | 12.7m |
| 2 | 0.95 | 132 | 14.1m |
| 8 | 0.90 | 121 | 14.3m |
| 9 | 0.95 | 130 | 15.2m |
| 18 | 0.70 | 137 | 16.3m |

All 6 items found and added to cart in every trial. Deductions from occasional product-URL mismatch vs ground truth. Math (ceil-division) correct in all trials.

### T3: REI Hiking Boots (mean 0.65)

8 of 8 successful trials scored exactly 0.65. Zero variance — the model satisfies the same ~4/6 hard constraints every time and fails the same ~2. Systematic capability gap, not noise.

### T4: Used Books Basket (mean 0.57)

Most variance. Cross-site navigation + shipping-rule reasoning is where CUAs struggle. Range: 0.50–0.70.

## Northstar CUA Fast: Failure Mode

Northstar fails at step 3 on T2. It emits text like *"I need to click the blue search button"* but never produces a click action. Demonstrates screen comprehension but cannot commit to action emission. Persists across retry nudges.

## Architecture

```
Screenshot → CUA Model → actions[] → Kernel Stealth Browser → Live Web
     ↑                                          |
     └──────────── capture screenshot ←─────────┘
```

All runs on live websites (no archiving), Kernel stealth Chromium, 1280×800 viewport.

## Trajectories

| Dataset | Trials | Link |
|---|---|---|
| T2 verified run | 1 | [trajectories.sh/t/25e32d0a](https://trajectories.sh/t/25e32d0a-cb8f-4ea2-8850-524ddc67f0a6) |
| Full 60-trial batch | 60 | uploading |
