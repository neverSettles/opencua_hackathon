# Buy-Side Optimization Bench — Results

## Can CUA models manage buy-side operations?

**GPT-5.5: yes, with degradation on complex tasks. Northstar CUA Fast: no — 0% pass rate across all tasks.**

## Headline Scores (Side-by-Side)

| Task | GPT-5.5 (OpenAI) | Northstar CUA Fast (Tzafon) |
|---|---|---|
| T2 — Single-source restock | **0.90** (n=14) | **0.00** (n=4, all failed) |
| T3 — Constrained product search | **0.65** (n=10) | **0.00** (n=2, all failed) |
| T4 — Multi-source optimization | **0.63** (n=9) | **0.00** (n=3, all failed) |

*75-trial run in progress. Scores above are from completed trials so far. GPT-5.5: 33/60 done. Northstar: 9/15 done.*

## GPT-5.5 vs Northstar: What Happened

| | GPT-5.5 | Northstar CUA Fast |
|---|---|---|
| **Pass rate** | 100% (all trials produce valid JSON) | 0% (all trials stall) |
| **Failure mode** | Occasional product mismatch vs GT | Narrates actions without executing them |
| **Avg actions/trial** | 100–190 depending on task | 2–4 before stall |
| **Avg time/trial** | 12–16 min | <2 min (fails fast) |
| **Search refinement** | Yes — retries 3–5 query variants | Never reaches search |
| **Cart management** | Adds items, sets quantities, reviews | Never reaches cart |

### Northstar failure pattern

Every Northstar trial across all 3 tasks follows the same pattern:
1. Receives screenshot of the homepage
2. Emits text: *"I need to click the search bar"*
3. Emits text: *"I need to click the blue search button"*
4. Stalls — never produces a click action despite correctly identifying the target

## Per-Task Breakdown

### T2: Restaurant Supply Restock (WebstaurantStore)

| Metric | GPT-5.5 | Northstar |
|---|---|---|
| Mean headline | **0.90** | 0.00 |
| Range | 0.70 – 0.95 | — |
| Trials completed | 14 | 4 |
| Quantity math correct | ~95% | 0% |

GPT-5.5 searches for each item, extracts pack size from product pages (e.g., "1,000/Case"), computes `ceil(weekly_usage / pack_size)`, and adds the right number of cases to cart.

### T3: REI Hiking Boots

| Metric | GPT-5.5 | Northstar |
|---|---|---|
| Mean headline | **0.65** | 0.00 |
| Range | 0.65 – 0.65 | — |
| Trials completed | 10 | 2 |

Zero variance in GPT-5.5 scores — the model consistently satisfies ~4/6 hard constraints (waterproof, height, price, etc.) and misses the same ~2 every time. This systematic ceiling is a fine-tuning opportunity.

### T4: Used Books Basket (Multi-Source)

| Metric | GPT-5.5 | Northstar |
|---|---|---|
| Mean headline | **0.63** | 0.00 |
| Range | 0.50 – 0.77 | — |
| Trials completed | 9 | 3 |

Hardest task. Requires navigating AbeBooks, ThriftBooks, BetterWorldBooks, and Powells, comparing prices, and accounting for different shipping rules per seller. Most variance — some trials achieve 0.77, others only 0.50.

## Trajectories (All Uploaded)

### GPT-5.5 (OpenAI)
| Task | Trials | Link |
|---|---|---|
| T2 | 20 | [trajectories.sh/t/b2ff60bb](https://trajectories.sh/t/b2ff60bb-5f4a-45e8-be14-5bf5ea14dd09) |
| T3 | 20 | [trajectories.sh/t/73a35321](https://trajectories.sh/t/73a35321-4eac-411b-9639-a85048fbf0d1) |
| T4 | 11 | [trajectories.sh/t/fba109a6](https://trajectories.sh/t/fba109a6-4b4c-4a04-acac-49fec8b68a80) |

### Northstar CUA Fast (Tzafon)
| Task | Trials | Link |
|---|---|---|
| T2 | 5 | [trajectories.sh/t/0d67b6c8](https://trajectories.sh/t/0d67b6c8-d0bf-47af-bc0e-3b8381da2f57) |
| T3 | 5 | [trajectories.sh/t/796b12fa](https://trajectories.sh/t/796b12fa-8dd8-4238-8c8b-fb36c653805c) |
| T4 | 5 | [trajectories.sh/t/b790ed30](https://trajectories.sh/t/b790ed30-1400-4a20-b759-b7ea5f10e0df) |

## Architecture

```
Screenshot → CUA Model → actions[] → Kernel Stealth Browser → Live Web
     ↑                                          |
     └──────────── capture screenshot ←─────────┘
```

- **Kernel:** Cloud stealth Chromium, 1280×800, anti-bot
- **Models:** GPT-5.5 (batched actions, pixel coords) and Northstar CUA Fast (single action, 0–999 normalized coords)
- **Live web:** No archiving. WebstaurantStore, REI.com, AbeBooks, ThriftBooks, BetterWorldBooks, Powells
- **Scoring:** Calibrated per-task (T2: 5 sub-scores, T3: 9 sub-scores, T4: 4 sub-scores)
