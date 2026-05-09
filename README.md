# Buy-Side Optimization Bench

Can Computer-Use Agents manage buy-side procurement operations?

A hackathon benchmark that drives multiple CUA models against live e-commerce sites through a deterministic harness, scores their structured outputs against a ground-truth rubric, and exports trajectories in [Harbor / ATIF-v1.6](https://www.trajectories.sh/) format for inspection and comparison.

## Stack

| Component | Role |
|---|---|
| **[Kernel](https://kernel.sh)** | Hosts the Chromium browser the agent operates (stealth mode + viewport control). |
| **[Lightcone (Tzafon)](https://docs.lightcone.ai)** | Serves Northstar CUA Fast via the Responses API. |
| **OpenAI computer-use, Anthropic Bedrock Claude, Gemini** | Alternate CUA backends behind a uniform `Action` adapter (`harness/adapters/`). |
| **[trajectories.sh](https://www.trajectories.sh)** | Hosts the resulting trajectories in Harbor format with step-by-step replay. |
| **[OpenShell](https://github.com/NVIDIA/OpenShell)** *(planned)* | Sandbox the eval runner so it can only reach the CUA APIs and the task definitions. |

## Project layout

```
.
├── tasks/                                 # Spec docs (T2/T3/T4) — single source of truth
│   ├── T2_restaurant_restock.md           #   single-source restock + arithmetic (live WebstaurantStore)
│   ├── T3_rei_hiking_boots.md             #   judgment task on REI
│   └── T4_used_books_basket.md            #   multi-source basket optimization (used books)
├── bench/
│   ├── worlds/<world>/
│   │   ├── world.yaml                     # world definition (sellers, persona, evaluator)
│   │   ├── sellers/<seller>.yaml          # per-seller config
│   │   └── tasks/<task_id>/
│   │       ├── task.yaml                  # task config + items
│   │       ├── intent.md                  # exact agent prompt
│   │       ├── ground_truth.json          # ground-truth picks (when populated)
│   │       └── rubric.yaml                # scoring weights & gates
│   ├── evaluators/                        # Python scorers (one per task)
│   │   └── t2_restaurant_restock.py       # scores agent JSON against rubric
│   └── schemas/                           # JSON schemas (planned)
├── harness/
│   ├── adapters/                          # Per-model CUA adapters w/ unified Action vocab
│   │   ├── base.py                        #   ActionType, Action, ModelAdapter
│   │   ├── northstar.py                   #   Tzafon Lightcone Responses API
│   │   ├── openai_cua.py                  #   OpenAI computer-use Responses API
│   │   ├── bedrock_claude.py              #   Anthropic Claude on Bedrock
│   │   └── gemini.py                      #   Google Gemini computer-use
│   ├── cua_runner.py                      # Vendored from tzafon/lightcone (Apache 2.0)
│   │                                      #   + cycle detection (period 2-4)
│   │                                      #   + no-action retry ladder
│   ├── kernel_computer.py                 # Kernel-backed adapter for the upstream
│   │                                      #   `computer` interface CuaRunner expects
│   ├── run_t2_spike.py                    # Multi-model dispatcher (--model {northstar,openai,gemini,bedrock_claude})
│   ├── run_t2.py                          # Tzafon-only driver using vendored CuaRunner
│   ├── to_harbor.py                       # Convert run outputs → Harbor / ATIF-v1.6 job dir
│   ├── requirements.txt
│   └── setup.sh
├── archive/                               # WACZ-replay path (deferred — see "Why no archiving?" below)
├── docker/pywb/                           # pywb container (deferred)
├── outputs/                               # one dir per run, gitignored
└── slides.md                              # what we built + findings (presentable)
```

## Quick start

```bash
# 1. Set up env
cp .env.example .env
# Fill in: KERNEL_API_KEY, TZAFON_API_KEY, OPENAI_API_KEY (for cross-model),
#          TRAJECTORIES_SH_API_KEY (for upload)

# 2. Install Python deps and Playwright Chromium (used for navigate())
cd harness && bash setup.sh && cd ..

# 3. Run T2 (single-source WebstaurantStore restock) end-to-end
cd harness
uv run python -u run_t2.py                                  # Tzafon Northstar, full T2
uv run python -u run_t2_spike.py --model openai             # OpenAI computer-use
uv run python -u run_t2_spike.py --model northstar --mini   # single-item smoke test

# 4. Convert outputs to Harbor format and upload to trajectories.sh
uv run python -m harness.to_harbor build \
  --job-name t2_webstaurant_restock_kernel \
  --task-name T2-001 \
  --out outputs/harbor/t2_webstaurant_restock_kernel \
  --run northstar=outputs/<northstar_run_id>/webstaurant_v1/T2-001 \
  --run openai=outputs/<openai_run_id>/webstaurant_v1/T2-001
npx --yes trajectories-sh upload trajectory outputs/harbor/t2_webstaurant_restock_kernel \
  --api-key "$TRAJECTORIES_SH_API_KEY"
```

## Live runs already uploaded

🔗 **T2 cross-model (Northstar + OpenAI on live WebstaurantStore):**
[trajectories.sh/t/e490fb79-1224-48a8-88ec-f86b3faec9a0](https://trajectories.sh/t/e490fb79-1224-48a8-88ec-f86b3faec9a0)

## Tasks

| ID | World | Type | Capabilities tested |
|---|---|---|---|
| **T2** | webstaurant_v1 (live) | Single-source restock | Within-source navigation, structured fact extraction, ceiling-division arithmetic, repetition reliability across 6 items |
| **T3** | rei_v1 *(spec only)* | Judgment with constraints | Filtering, review-volume gating, soft-constraint trade-offs |
| **T4** | used_books_v1 *(spec only)* | Multi-source basket optimization | Cross-source price comparison, shipping-threshold reasoning, ISBN-precise edition matching |

T2 is the primary live task in this snapshot. T3 and T4 specs exist in `tasks/` and on the `ground-truth-data` / `task-spec-calibration` branches; their bench-world materialization is in flight.

## Scoring

Per the T2 spec (`tasks/T2_restaurant_restock.md`):

| Sub-score | Weight | Definition |
|---|---|---|
| `item_match` | 0.30 | Agent's `product_url` matches GT primary or alternative |
| `pack_extraction` | 0.30 | Agent's `extracted_case_pack_size` == GT `case_pack_size` |
| `quantity_correctness` | 0.25 | `cases_added_to_cart == ceil(weekly_usage / agent_extracted_pack_size)` (scored against agent's *own* extracted pack — isolates arithmetic from extraction) |
| `cart_added` | 0.10 | Item present in agent's returned cart |
| `completeness` | 0.05 | items_in_cart / 6 |

`headline = 0.30·item_match + 0.30·pack_extraction + 0.25·quantity_correctness + 0.10·cart_added + 0.05·completeness`

If ground-truth URLs aren't populated yet, the `item_match` and `pack_extraction` sub-scores return `None` and are excluded from the headline (their weight is redistributed proportionally).

## Why no archiving (yet)?

The original plan was to capture each task's site into WACZ archives served via pywb so evals were deterministic across runs. We hit a wall:

- `webrecorder/browsertrix-crawler` workers crashed on every version we tried (`:latest`, `:1.5.0`, `:1.0.4`) with a Redis Lua script error (`ERR value is not an integer or out of range`) — a known Redis 7.2 + `tonumber()` compatibility bug bundled into the crawler image.
- A `wget --warc` fallback got 403'd by WebstaurantStore's anti-bot.

We pivoted to live web through Kernel's stealth mode for the spike (because anti-bot bypass is exactly what Kernel's stealth + residential proxies are for), and accepted that determinism for post-training comparison would be controlled by running baseline + post-trained back-to-back in tight time windows. The `archive/` and `docker/pywb/` dirs hold the partial work for later.

## Status

- ✅ Project scaffolding (bench worlds, task specs, scoring rubric)
- ✅ T2 scorer (`bench/evaluators/t2_restaurant_restock.py`)
- ✅ Live Kernel + Lightcone harness with vendored CuaRunner + cycle detection + no-action retry
- ✅ Multi-model adapter system (Northstar, OpenAI, Bedrock Claude, Gemini) sharing a uniform `Action` vocab
- ✅ Harbor (ATIF-v1.6) trajectory exporter and upload
- ⏳ T3, T4 worlds materialized in `bench/worlds/` (specs exist; tooling on team branches)
- ⏳ Ground-truth URL population for live T2 (tooling on `ground-truth-tooling` branch)
- ⏳ OpenShell sandbox wrap of the eval runner
- ⏳ Post-training step on the harvested trajectories

## License

Vendored upstream code retains its original license (see `harness/cua_runner.py` header for the Apache 2.0 attribution to [tzafon/lightcone](https://github.com/tzafon/lightcone)).
