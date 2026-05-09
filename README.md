# Buy-Side Optimization Bench

> **Can Computer-Use Agents manage buy-side procurement operations?**

A benchmark for evaluating CUA (Computer Use Agent) models on real procurement tasks across live e-commerce sites: single-source restock, multi-source basket optimization, and judgment-under-constraint shopping. Trajectories are exported in [Harbor / ATIF-v1.6](https://www.trajectories.sh/) format and uploaded to trajectories.sh for inspection.

Built during a hackathon sponsored by [Tzafon (Lightcone)](https://docs.lightcone.ai), [Kernel](https://kernel.sh), and NVIDIA OpenShell.

---

## Trajectories (Browse Online)

All trajectories with step-by-step screenshots and actions:

### GPT-5.5 (OpenAI)
| Task | Trials | Headline | Link |
|---|---|---|---|
| **T2** — WebstaurantStore restock | 20 | **0.90** | [trajectories.sh/t/b2ff60bb](https://trajectories.sh/t/b2ff60bb-5f4a-45e8-be14-5bf5ea14dd09) |
| **T3** — REI hiking boots | 20 | **0.65** | [trajectories.sh/t/73a35321](https://trajectories.sh/t/73a35321-4eac-411b-9639-a85048fbf0d1) |
| **T4** — Used books basket | 11 | **0.63** | [trajectories.sh/t/fba109a6](https://trajectories.sh/t/fba109a6-4b4c-4a04-acac-49fec8b68a80) |

### Northstar CUA Fast (Tzafon)
| Task | Trials | Headline | Link |
|---|---|---|---|
| **T2** — WebstaurantStore restock | 5 | **0.00** | [trajectories.sh/t/5746d053](https://trajectories.sh/t/5746d053-187b-493e-a5d9-955a7f816ce0) |
| **T3** — REI hiking boots | 5 | **0.00** | [trajectories.sh/t/796b12fa](https://trajectories.sh/t/796b12fa-8dd8-4238-8c8b-fb36c653805c) |
| **T4** — Used books basket | 5 | **0.00** | [trajectories.sh/t/f567760e](https://trajectories.sh/t/f567760e-a96e-4ad3-8c54-642e1019f0a7) |

Raw trajectory data (JSON) also in this repo: [`outputs/jobs/`](outputs/jobs/)

### Results Summary

| Task | GPT-5.5 | Northstar CUA Fast |
|---|---|---|
| T2 — Single-source restock | **0.90** (n=14) | **0.00** (all stall at step 3) |
| T3 — Constrained product search | **0.65** (n=10, zero variance) | **0.00** (all stall at step 2) |
| T4 — Multi-source optimization | **0.63** (n=9, range 0.50–0.77) | **0.00** (all stall at step 2) |

GPT-5.5 completes all three task types. Northstar cannot commit to action emission — it narrates *"I need to click the search button"* but never produces a click. See [RESULTS.md](RESULTS.md) for detailed analysis.

---

## At a glance

| Component | Used for |
|---|---|
| **[Kernel](https://kernel.sh)** | Hosts the Chromium browser the agent operates (stealth mode, viewport control). |
| **[Lightcone (Tzafon)](https://docs.lightcone.ai)** | Northstar CUA Fast via the Responses API. |
| **OpenAI computer-use** | gpt-5.5 / `computer-use-preview`, behind a uniform `Action` adapter. |
| **Anthropic Claude (Bedrock), Google Gemini** | Adapters built; Gemini key was expired during testing. |
| **[trajectories.sh](https://www.trajectories.sh)** | Hosts trajectories in Harbor format with step-by-step replay. |

Three tasks span complexity, source-fragmentation, and decision-structure:

| ID | Source | What it tests | Capabilities |
|---|---|---|---|
| **T2** | WebstaurantStore (live) | Single-source restock | Search → product page → case-pack extraction → ceiling-divide → cart-add, repeated 6×. |
| **T3** | REI (live) | Judgment with hard + soft constraints | Filter UI engagement, review-volume gating, soft-constraint trade-offs. |
| **T4** | AbeBooks / ThriftBooks / Powell's / Better World Books (live) | Multi-source basket optimization | Cross-source price comparison, shipping-threshold reasoning, edition matching. |

Each task ships with a calibrated agent prompt, a populated ground-truth file (live-collected on 2026-05-09), and a deterministic Python scorer that returns sub-scores plus a weighted headline.

---

## Repository layout

```
.
├── tasks/                            # Source-of-truth task specs (markdown)
│   ├── T2_restaurant_restock.md
│   ├── T3_rei_hiking_boots.md
│   └── T4_used_books_basket.md
├── ground_truth/                     # Live-collected ground truth per task
│   ├── T2_ground_truth.json
│   ├── T3_ground_truth.json
│   ├── T4_ground_truth.json
│   └── _rei_raw.json                 # raw scrape backing T3
├── scripts/
│   ├── scoring.py                    # score_t2 / score_t3 / score_t4
│   ├── t3_build_ground_truth.py      # rebuild T3 GT from a fresh REI scrape
│   └── t4_optimizer.py               # recompute T4 totals after editing listings
├── harness/
│   ├── adapters/                     # Per-model CUA adapters (uniform Action vocab)
│   │   ├── base.py                   #   ActionType, Action, ModelAdapter
│   │   ├── northstar.py              #   Tzafon Lightcone Responses API
│   │   ├── openai_cua.py             #   OpenAI computer-use
│   │   ├── bedrock_claude.py         #   Anthropic via Bedrock
│   │   └── gemini.py                 #   Google Gemini
│   ├── run_eval.py                   # MAIN entry — runs N trials of one (task, model)
│   ├── launch_15_northstar.sh        # 5 Northstar trials × 3 tasks
│   ├── launch_60_openai.sh           # 20 OpenAI trials × 3 tasks (sequential)
│   ├── launch_parallel.sh            # 60 OpenAI trials, configurable concurrency
│   ├── to_harbor.py                  # Convert run outputs → Harbor / ATIF-v1.6 job dirs
│   ├── requirements.txt
│   └── setup.sh
├── docs/
│   ├── handoff_v1.md                 # Original session handoff (Dimitry)
│   └── handoff_v2.md                 # Post-merge session handoff (full matrix recipe)
├── slides.md                         # Presentable deck
├── README.md                         # this file
├── LICENSE                           # Apache 2.0
├── .env.example
├── archive/, docker/pywb/            # Deferred WACZ-replay infra (see Status below)
└── outputs/                          # gitignored — one dir per run
```

---

## Quick start

```bash
# 1. Set up env
cp .env.example .env
# Fill in:
#   KERNEL_API_KEY              # Kernel browser sessions
#   TZAFON_API_KEY              # Lightcone / Northstar inference
#   OPENAI_API_KEY              # gpt-5.5 / computer-use-preview
#   GEMINI_API_KEY              # (optional) Google AI Studio
#   AWS_*                       # (optional) Bedrock Claude
#   TRAJECTORIES_SH_API_KEY     # for upload to trajectories.sh

# 2. Install Python deps + Playwright Chromium
cd harness && bash setup.sh && cd ..

# 3. Single trial smoke test
cd harness
uv run python -u run_eval.py --task T2 --model northstar --runs 1
uv run python -u run_eval.py --task T3 --model openai   --runs 1
uv run python -u run_eval.py --task T4 --model openai   --runs 1

# 4. Full matrix (5 Northstar + 20 OpenAI per task = 75 trials)
bash launch_15_northstar.sh
bash launch_parallel.sh                    # 60 OpenAI in 10-way parallel

# 5. Bundle into Harbor jobs and upload to trajectories.sh
for job_dir in ../outputs/jobs/*/; do
  job=$(basename "$job_dir")
  task=$(echo "$job" | cut -d_ -f1 | tr a-z A-Z)
  model=$(echo "$job" | cut -d_ -f2)
  runs=()
  for trial in "$job_dir"trial_*; do
    runs+=(--run "${model}=${trial}")
  done
  uv run python to_harbor.py build \
    --job-name "buy_side_${job}" \
    --task-name "$task" \
    --out "../outputs/harbor/${job}" \
    "${runs[@]}"
  npx --yes trajectories-sh upload trajectory "../outputs/harbor/${job}" \
    --api-key "$TRAJECTORIES_SH_API_KEY"
done
```

---

## Scoring

Each task returns a dict of sub-scores and a weighted headline. From `scripts/scoring.py`:

### T2 — single-source restock (`score_t2`)

| Sub-score | Weight | Definition |
|---|---|---|
| `item_match` | 0.30 | Agent's `product_url` matches GT primary or alternative URL |
| `pack_extraction` | 0.30 | Agent's extracted `case_pack_size` matches GT |
| `quantity_correctness` | 0.25 | `cases_added_to_cart == ceil(weekly_usage / agent_extracted_pack)` (scored against agent's *own* extracted pack to isolate arithmetic from extraction) |
| `cart_added` | 0.10 | Item present in agent's returned cart |
| `completeness` | 0.05 | items_in_cart / 6 |

### T3 — judgment with constraints (`score_t3`)

Hard constraints (`category_match`, `waterproof_match`, `height_match`, `price_match`, `size_match`, `weight_match`) plus `rating_optimal` (matches the GT's >=100-review optimum), `cart_state`, `filter_engagement` (rewards using REI's filter UI rather than direct-URL navigation).

```
headline = 0.40·hard_constraint_score + 0.25·rating_optimal
        + 0.15·cart_state + 0.10·filter_engagement + 0.10·all_hard_satisfied
```

### T4 — multi-source basket (`score_t4`)

`validity` (schema), `completeness` (6/6 books), `cost_correctness` (claimed total matches recomputed total), `optimality` (ratio to optimal allocation).

```
headline = 0.40·validity + 0.30·completeness + 0.20·cost_correctness + 0.10·optimality
```

The original T4 weighting put `optimality` at 0.40, but live-collection found AbeBooks free shipping had eaten the consolidation incentive (optimum tied with naive at $41.41). T4 now primarily tests cross-source navigation. The ground-truth JSON carries `headline_weights` so flipping back is a JSON edit.

---

## Why no archiving (yet)?

The original plan was to capture each task's site into WACZ archives served via pywb so evals were deterministic across runs. We hit a wall:

- `webrecorder/browsertrix-crawler` workers crashed on every version we tried (`:latest`, `:1.5.0`, `:1.0.4`) with a Redis Lua-script error (`ERR value is not an integer or out of range`) — a known Redis 7.2 + `tonumber()` compatibility bug bundled into the crawler image.
- A `wget --warc` fallback got 403'd by WebstaurantStore's anti-bot.

We pivoted to **live web through Kernel's stealth mode** for the spike — because anti-bot bypass is exactly what Kernel's stealth + residential proxies are for. Determinism for post-training comparison is controlled by running baseline + post-trained back-to-back in tight time windows. The `archive/` and `docker/pywb/` dirs hold the partial work for later.

---

## Status

- Working harness: 4 model adapters (Northstar, OpenAI, Bedrock Claude, Gemini), Kernel-native browsers, screenshot-action loop with cycle detection and no-action retry.
- T2/T3/T4 specs calibrated, ground truth populated and committed.
- Scoring module covers all three with sub-scores + weighted headline.
- Cross-model trajectory upload to trajectories.sh works (one early job public at https://trajectories.sh/t/e490fb79-1224-48a8-88ec-f86b3faec9a0).
- Full 75-trial matrix runs are queued via the `launch_*.sh` scripts.

---

## Acknowledgements

- The `harness/adapters/*.py` adapters were adapted from the public examples in [tzafon/lightcone](https://github.com/tzafon/lightcone). Earlier development used a vendored `cua_runner.py` from the same repository (Apache 2.0); that file was removed during cleanup once the adapter system superseded it.
- T3 and T4 ground-truth tooling (`scripts/t3_build_ground_truth.py`, `scripts/t4_optimizer.py`) and the calibrated `scripts/scoring.py` are by Dimitry; documentation of his calibration decisions is preserved in [`docs/handoff_v1.md`](docs/handoff_v1.md).

## License

Apache 2.0 — see [LICENSE](LICENSE).
