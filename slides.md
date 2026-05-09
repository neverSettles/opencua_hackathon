# Buy-Side Optimization Bench

> *Can Computer-Use Agents manage buy-side procurement operations?*

A hackathon benchmark for evaluating CUA models on live e-commerce procurement, with reusable infrastructure across Kernel, Lightcone, and trajectories.sh.

---

## The question

Procurement is the boring 80% of business that nobody has automated yet. SKU lookup, case-pack arithmetic, basket optimization across vendors, judgment calls under under-specified intents. **Can CUA models actually do this work, today, in production-shaped conditions?**

We built a benchmark to measure it across three task shapes:

- **T2 — single-source restock** (well-structured): WebstaurantStore, 6 disposables, weekly-usage → ceiling-divide → cases-to-cart.
- **T3 — judgment with constraints**: REI hiking boots within budget, with soft trade-offs.
- **T4 — multi-source basket optimization**: 6 used books across 4 sellers, shipping thresholds, ISBN-precise editions.

---

## Stack

```
┌────────────────────────────────────────────────────────────────────┐
│  Eval runner (our harness)                                         │
│  ─ task spec from bench/worlds/<world>/tasks/<task_id>/            │
│  ─ rubric scoring per bench/evaluators/<task>.py                   │
│  ─ trajectory logging in Harbor / ATIF-v1.6                        │
│                                                                    │
│  ┌──────────────┐         ┌──────────────────────────────┐         │
│  │ Adapters/    │         │ Vendored CuaRunner           │         │
│  │ - Northstar  │         │   (tzafon/lightcone fork)    │         │
│  │ - OpenAI     │         │   + cycle detection          │         │
│  │ - Claude     │         │   + no-action retry ladder   │         │
│  │ - Gemini     │         │                              │         │
│  └──────┬───────┘         └──────┬───────────────────────┘         │
│         │ unified Action vocab   │                                 │
│         ▼                        ▼                                 │
│  ┌──────────────────────────────────────────────────────┐          │
│  │ KernelComputerAdapter                                │          │
│  │  ↳ Kernel browser (stealth, viewport-matched)        │          │
│  │  ↳ Playwright over CDP for navigate()                │          │
│  └──────────────────────────────────────────────────────┘          │
└──────────────────────────┬─────────────────────────────────────────┘
                           ▼
                  ┌──────────────────┐
                  │  Live e-commerce │
                  │  (WebstaurantStore, etc.)  │
                  └──────────────────┘
```

---

## What we built

### 1. Task structure with calibrated specs
- `bench/worlds/<world>/tasks/<id>/` with `task.yaml`, `intent.md`, `ground_truth.json`, `rubric.yaml`
- T2 fully materialized; T3/T4 specs done, world materialization in flight on team branches

### 2. Per-model rubric scorers
- `bench/evaluators/t2_restaurant_restock.py`: 4 sub-scores per item + completeness, weighted headline
- Quantity correctness scored against agent's *own* extracted pack — isolates arithmetic from extraction failure
- Graceful degradation when ground-truth URLs aren't populated yet

### 3. Live-Kernel harness
- Vendored upstream CuaRunner + Kernel adapter so we get Tzafon's tested loop logic over Kernel's anti-bot stealth
- Added cycle detection (period 2–4 patterns), not just identical-3-in-a-row, after observing real failure mode
- No-action retry ladder for when the model narrates without acting

### 4. Multi-model uniformity
- One `Action` vocabulary, four adapters: Northstar, OpenAI computer-use, Bedrock Claude, Gemini
- Same task → same scorer → comparable headline across models

### 5. Harbor exporter
- `harness/to_harbor.py` converts our run outputs to ATIF-v1.6 job dirs
- One job = many trials = many models on the same task
- Uploaded via `trajectories-sh upload trajectory`

🔗 [**Live job: T2 Northstar vs. OpenAI on WebstaurantStore**](https://trajectories.sh/t/e490fb79-1224-48a8-88ec-f86b3faec9a0)

---

## What worked

- **Kernel's stealth handled live anti-bot.** No 403s, no CAPTCHA fights — the agent talks to the real WebstaurantStore.
- **The vendored CuaRunner + Kernel adapter pattern.** Two models, two backends, one harness. Adding the third or fourth model is now ~50 lines of glue.
- **The mini single-item spike.** Northstar correctly extracted *"1,000/Case"* from a Choice 16oz cup product page in 4 actions / 53 seconds. End-to-end pipeline proven.
- **Trajectory upload.** Anyone with the link can step through every action, screenshot, and tool call in the harbor viewer.

---

## What we found

> Northstar CUA Fast and OpenAI's computer-use both struggle on the full 6-item T2.

- Both models successfully execute search-and-extract on item 1, then **lose spatial state after the first navigation**.
- Northstar's failure mode: *narrate without acting* (model emits text-only message describing what it wants to do, but no action). Our retry ladder catches it but the model often re-narrates the same intent.
- Cycle detection fired correctly on real loops (e.g. `back→click→type→back→click→type` for 10 iterations).
- Neither model reached the cart-add phase or emitted the structured JSON answer in our timed runs.

This is **honest CUA-eval signal**, not infrastructure failure. Tzafon themselves report Northstar at ~37% pass@1 on OSWorld; multi-step procurement on a complex B2B catalog is at the edge of current-generation capability.

---

## Why we skipped archiving

- Goal was deterministic replay via WACZ + pywb. Spent a few hours on infra:
  - `browsertrix-crawler` (3 versions tried) hits a Redis Lua bug on this Mac
  - `wget --warc` fallback gets 403'd by WebstaurantStore's anti-bot
- Pivoted to **live web through Kernel's stealth** because anti-bot bypass is precisely what Kernel sells.
- Determinism for post-training comparison handled by running baseline + post-trained in tight back-to-back windows.

---

## Why this is a defensible result

1. **The harness is reusable.** Any task = bench/worlds/<id>/{task,intent,ground_truth,rubric}. Any model = adapter. Any browser backend = swap KernelComputerAdapter for another that exposes the same surface.
2. **The trajectories are inspectable.** Harbor viewer makes failure modes legible: where exactly did Northstar lose track? Click step 14 in the viewer.
3. **The scoring is honest.** Sub-scores expose *which* capability dimension failed (extraction vs arithmetic vs cart-add), not just a single pass/fail bit.
4. **The cross-model comparison is set up.** Same task, same rubric, two models in one job. Adding Claude or Gemini is one more `--run` flag.

---

## Next steps

- Populate ground-truth URLs for T2 → fill in the missing 60% of headline weight (item_match + pack_extraction)
- Materialize T3 + T4 worlds from existing specs and ground-truth tooling on team branches
- Multi-run reliability: `--runs 5` for variance bars
- Wrap the eval runner with OpenShell so the security narrative is concrete
- Post-training step on harvested trajectories — start with successful trajectories from the *mini* tasks where we have positive signal

---

## Links

- **GitHub:** [neverSettles/opencua_hackathon](https://github.com/neverSettles/opencua_hackathon)
- **Live trajectory job:** [trajectories.sh/t/e490fb79-...](https://trajectories.sh/t/e490fb79-1224-48a8-88ec-f86b3faec9a0)
- **Sponsor stacks used:** [Kernel](https://kernel.sh) · [Lightcone (Tzafon)](https://docs.lightcone.ai) · [trajectories.sh](https://www.trajectories.sh)
