# Buy-Side Optimization Bench — Preliminary Results

**Date:** 2026-05-09
**Question:** Can Computer Use Agents manage buy-side operations?

## Architecture

```
┌─────────────────┐     screenshot      ┌──────────────────┐
│  Kernel Browser  │ ─────────────────→  │   CUA Model      │
│  (stealth        │                     │  (GPT-5.5 /      │
│   Chromium)      │ ←───────────────── │   Northstar)      │
│                  │  actions[]: click,  └──────────────────┘
│  Live web:       │  type, scroll,
│  WebstaurantStore│  keypress, etc.
│  REI.com         │
│  AbeBooks        │
└─────────────────┘
```

- **Kernel:** Cloud-hosted stealth Chromium, 1280×800 viewport, <30ms spin-up
- **Models:** GPT-5.5 (OpenAI GA computer tool), Northstar CUA Fast (Tzafon/Lightcone)
- **No archiving:** Agents operate on live websites through Kernel's anti-bot stealth mode
- **Scoring:** Calibrated per-task scorers with ground-truth from live data collection

## Tasks

| Task | Domain | Source | What it tests |
|---|---|---|---|
| **T2** — Restaurant Restock | Disposable supplies | WebstaurantStore | Within-source navigation, case-pack extraction, ceil-division arithmetic, cart management |
| **T3** — Hiking Boot Purchase | Outdoor gear | REI.com | Multi-constraint filtering (waterproof, height, price, size, weight), product comparison, cart checkout |
| **T4** — Used Book Basket | Used books | AbeBooks, ThriftBooks, BetterWorldBooks, Powells | Cross-source comparison, shipping-rule optimization, ISBN-precise matching |

## Model Comparison: T2 (Restaurant Supply Restock)

### GPT-5.5 (OpenAI) — PASSED

| Metric | Score |
|---|---|
| **Headline** | **1.00** |
| Items found & added to cart | 6/6 |
| Case-pack extraction correct | 6/6 |
| Quantity math correct (ceil division) | 6/6 |
| Completeness | 100% |
| Total steps | 63 |
| Wall clock | ~18 min |

**Agent's cart (all verified correct):**

| Item needed | Product chosen | Pack size | Cases | Coverage |
|---|---|---|---|---|
| 16oz coffee cups (2400/wk) | Choice 16 oz White Poly Paper Hot Cup | 1,000/Case | 3 | 3,000 ✓ |
| 9" paper plates (1500/wk) | Choice 9" White Coated Paper Plate | 1,000/Case | 2 | 2,000 ✓ |
| Nitrile gloves L (800/wk) | Lavex Black Nitrile 5 Mil Textured | 1,000/Case | 1 | 1,000 ✓ |
| 32oz deli containers (600/wk) | Choice 32oz Deli Combo Pack | 240/Case | 3 | 720 ✓ |
| 13-gal trash liners (200/wk) | Lavex 12-16 Gal Clear Liner | 250/Case | 1 | 250 ✓ |
| SS scouring pads (50/wk) | Libman 3" Stainless Steel Scrubber | 24/Case | 3 | 72 ✓ |

### Northstar CUA Fast (Tzafon) — FAILED

| Metric | Score |
|---|---|
| **Headline** | **0.00** |
| Items found | 0/6 |
| Failure mode | Narrates actions without executing them |
| Steps before stall | 3 |

**Failure pattern:** Northstar repeatedly emits text messages like _"I need to click the blue search button"_ but never produces a click action. The model can describe what it should do but cannot commit to emitting the action payload. This persisted across retry nudges (up to 4 attempts per stall).

## Behavioral Observations

### GPT-5.5 capabilities demonstrated
- **Search refinement:** When "9 inch white heavy duty paper plate" returned no results, the agent tried 5 different query reformulations before finding the product
- **Batched actions:** GPT-5.5's `actions[]` array emits 3+ actions per turn (e.g., click search box + type query + press Enter), reducing round-trips
- **Quantity field manipulation:** The agent correctly used double-click → Ctrl+A → Backspace → type to clear and set quantity fields
- **Cart awareness:** Navigated to cart, reviewed contents, went back to continue shopping
- **Structured output:** Emitted valid JSON matching the exact requested schema with product URLs, pack sizes, and calculated totals

### Northstar failure mode
- Cannot translate visual understanding into action emission
- Demonstrates screen comprehension (correctly identifies the search button location) but the action protocol doesn't fire
- Suggests the model's RL training for computer use may not generalize to this task shape

## Scale Run (In Progress)

60 trials running in parallel across all 3 tasks:

| Task | Model | Trials | Status |
|---|---|---|---|
| T2 (WebstaurantStore) | GPT-5.5 | 20 | 🟡 In progress (~70% through first batch) |
| T3 (REI) | GPT-5.5 | 20 | 🟡 In progress |
| T4 (Used books) | GPT-5.5 | 20 | 🟡 In progress |

All 60 running concurrently on separate Kernel browser sessions. Results will populate pass rates, mean headline scores, and variance across trials.

## Trajectories

| Run | Model | Task | Trials | URL |
|---|---|---|---|---|
| Verified T2 | GPT-5.5 | T2 | 1 | [trajectories.sh/t/25e32d0a](https://trajectories.sh/t/25e32d0a-cb8f-4ea2-8850-524ddc67f0a6) |
| Full matrix | GPT-5.5 | T2/T3/T4 | 60 | Uploading after completion |

## Infrastructure

| Component | What | Why |
|---|---|---|
| **Kernel** | Cloud browser hosting | Stealth Chromium, anti-bot, parallel sessions, live view debugging |
| **OpenAI Responses API** | CUA model inference | GPT-5.5 with GA `computer` tool, batched actions, pixel coords |
| **Lightcone** | Northstar inference | Tzafon's RL-trained CUA model (0-999 normalized coords) |
| **Harness** | `harness/run_eval.py` | Model-agnostic runner with adapter pattern, per-step screenshots + trajectory logging |
| **Scoring** | `scripts/scoring.py` | Calibrated per-task scorers (T2: 5 sub-scores, T3: 9 sub-scores, T4: 4 sub-scores) |

## Repository

- **Branch:** `eval-runs`
- **Runner:** `harness/run_eval.py --task {T2,T3,T4} --model {openai,northstar}`
- **Adapters:** `harness/adapters/{openai_cua,northstar,gemini}.py`
- **Task specs:** `tasks/T{2,3,4}_*.md`
- **Ground truth:** `ground_truth/T{2,3,4}_ground_truth.json`
- **Scoring:** `scripts/scoring.py`
