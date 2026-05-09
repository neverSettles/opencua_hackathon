# Buy-Side Bench — Session Handoff

You're picking up a hackathon project that builds a buy-side eval for Computer Use Agents (CUAs). One task is implemented end-to-end on live web. This file tells you exactly what's wired, what works, and how to run things.

## TL;DR — run the OpenAI job

```bash
cd harness
uv run python -u run_t2_spike.py --model openai            # full T2, 6 items, scored
uv run python -u run_t2_spike.py --model openai --mini     # ~30s smoke test (single item, no score)
```

Model defaults to `gpt-5.5` (GA `computer` tool). Override with `--openai-model gpt-5.5-pro` (or `gpt-5.4`, `gpt-5.4-mini`, `computer-use-preview`).

Each run writes `outputs/t2_openai_<ts>/webstaurant_v1/T2-001/` containing:
- `summary.json` — run metadata + headline score
- `trajectory.jsonl` — one line per executed action (model, action_type, coords, text, result)
- `agent_response.json` — parsed final JSON the agent emitted (if any)
- `score.json` — per-item sub-scores + headline (full-mode only)
- `step_NNN.png` — screenshot per step (incl. `_initial` and `_nudge`)

## What's in repo

```
bench/
  evaluators/
    t2_restaurant_restock.py   # T2 scorer per the spec markdown
  worlds/webstaurant_v1/
    world.yaml                 # world def
    sellers/webstaurantstore.yaml
    tasks/T2-001/
      task.yaml                # config + 6 items + weekly_usage
      intent.md                # exact agent prompt
      ground_truth.json        # SCAFFOLD ONLY (ground_truth: null per item)
      rubric.yaml              # weights + which sub-scores need GT
harness/
  adapters/
    base.py                    # normalized Action vocab + Kernel executor
    northstar.py               # Tzafon Lightcone Responses API
    gemini.py                  # Gemini 3.x computer use (NOT TESTED — key expired)
    openai_cua.py              # OpenAI gpt-5.5 / computer-use-preview
  cua_runner.py                # teammate: vendored from tzafon/lightcone + cycle break + retry ladder
  kernel_computer.py           # teammate: Kernel adapter for CuaRunner interface
  run_t2_spike.py              # MAIN multi-model runner (--model {northstar,gemini,openai})
  run_t2.py                    # teammate: alt runner using CuaRunner + Northstar
  to_harbor.py                 # teammate (uncommitted): converts our outputs → trajectories.sh format
  requirements.txt
  setup.sh                     # uv venv + playwright install
.env.example                   # which keys are needed
tasks/                         # human-readable task spec markdowns (T2 source of truth)
outputs/                       # all run artifacts (gitignored)
```

## Required env (.env at repo root, gitignored)

```bash
KERNEL_API_KEY=sk_...          # Kernel browser sessions (provided)
TZAFON_API_KEY=sk_...          # Lightcone / Northstar inference (provided)
OPENAI_API_KEY=sk-proj-...     # gpt-5.5 + computer-use models (provided)
GEMINI_API_KEY=                # Google AI Studio key — last one expired, NOT WORKING
```

Set up the venv if not already:
```bash
cd harness && bash setup.sh
```

## Architecture

The harness is **model-agnostic**. Same Kernel browser, same screenshot capture, same task definition — only the inference + action-vocabulary differ per model. That's the `ModelAdapter` interface (`harness/adapters/base.py`).

```
Kernel browser (stealth Chromium, viewport 1280x800)
        ↓                    ↑
  capture_screenshot()   click_mouse / type_text / press_key / scroll / drag_mouse / move_mouse
        ↓                    ↑
        └── ModelAdapter ────┘
            ├── NorthstarAdapter (Lightcone Responses API, 0-999 normalized coords)
            ├── GeminiAdapter    (google-genai computer_use, 0-999 normalized)
            └── OpenAIComputerUseAdapter (Responses API, pixel coords, batched actions[])
```

Each adapter normalizes its model's actions into a common `Action` dataclass (`base.py`); the runner then has one path that executes any normalized action on Kernel via `execute_action(...)`.

Initial navigation uses Playwright over Kernel's CDP URL (`page.goto(...)`); after that, all subsequent actions go through Kernel's `computer.*` controls API. Playwright is also used for `navigate` actions emitted mid-task.

## What works

**Mini smoke test on `gpt-5.5`** (single-item, "find case pack of 16oz Choice cold cup, answer with the integer"):
- `outputs/t2_openai_mini_20260509T221937Z/...`
- 4 turns, 7 actions, correct answer `1000`, ~50s wall time
- GA's batched `actions[]` worked: turn 0 emitted [click, type, enter] in one response

**Full 6-item T2 on `gpt-5.5`** (most recent run):
- `outputs/t2_openai_20260509T222050Z/...`
- Status: was running at write time, ~60+ trajectory entries through step 22+
- Watch live view URL printed in the run's stdout to monitor

## What didn't work (don't redo)

| Attempt | Outcome | Lesson |
|---|---|---|
| browsertrix-crawler v1.0.4 / 1.5.0 / latest, all platforms | All hit Redis 7.2 + Lua `tonumber()` script bug; 0 pages crawled | Don't waste time on browsertrix locally on this machine. |
| `wget --warc-file` against WebstaurantStore | DDG + homepage OK; **403 Forbidden** on every search/product page (anti-bot) | wget is fine for DDG/homepage but not for WebstaurantStore catalog. |
| Archived-replay path (pywb container + cloudflared tunnel + Kernel custom proxy) | Architecture is sound but capture step was the blocker | Skip archiving entirely for the spike. Live web through Kernel stealth works. |
| Northstar `tzafon.northstar-cua-fast` on T2 | Stuck at step 3 narrating "I need to click the search button" without ever emitting a click | Model can't commit to actions in this scenario. Try `tzafon.northstar-cua-fast-1.7-experiment` next. |
| Gemini 3 Pro (`gemini-3-pro-preview`) | API key `AIzaSy...0UU` returned `400 API_KEY_EXPIRED` | Need a fresh AI Studio key. Adapter is built and ready. |
| OpenAI GA tool with `display_width`/`display_height` | `400 unknown_parameter` | GA `computer` tool infers dims from screenshot; preview tool still needs them. Adapter handles both. |

## Key design decisions

1. **No archive layer.** The original plan called for pywb-replayed WACZ archives for determinism. We pivoted to live web through Kernel's stealth mode. This skips an entire category of infrastructure work and the stealth-protected live site loads cleanly. Determinism trade-off: run baseline + comparison back-to-back in the same time window.

2. **Coordinate spaces differ per model.**
   - Northstar/Lightcone: 0-999 normalized (denormalize against viewport)
   - Gemini: 0-999 normalized (same)
   - OpenAI GA: pixel coords directly (no denormalize)
   The adapters handle this; runner sees only pixel-coordinate `Action`s.

3. **OpenAI GA `actions[]` is batched.** A single model turn can emit 3+ actions (click + type + enter). The runner iterates them in order and only takes a new screenshot once per turn (after the batch). Northstar emits one action per turn.

4. **Text-only "I need to click X" responses are recoverable.** When a model emits a message but no action, the runner sends a fresh screenshot back with a "Continue" nudge, up to 3 retries before bailing. Northstar still gets stuck even with this; gpt-5.5 doesn't.

5. **Scoring is GT-tolerant.** `bench/worlds/webstaurant_v1/tasks/T2-001/ground_truth.json` is a scaffold (per-item `ground_truth: null`). The scorer skips `item_match` and `pack_extraction` for items with no GT and redistributes their weight across the available sub-scores. So you can score immediately on `quantity_correctness` (math is deterministic given the agent's own extracted pack), `cart_added`, and `completeness`. **To enable full scoring**: populate each item's `ground_truth: { product_url, product_name, case_pack_size, alternative_acceptable_products: [...] }` after a human verifies a canonical pick on WebstaurantStore.

## Recommended next steps for you

In order of value:

1. **Wait for / re-run the full T2 on gpt-5.5 to completion** and inspect:
   - `outputs/t2_openai_<ts>/.../score.json` — see headline + per-item sub-scores
   - `outputs/t2_openai_<ts>/.../agent_response.json` — what the agent put in the cart
   - `step_NNN.png` to debug any item that failed

2. **Populate ground_truth.json** for the 6 items by manually picking canonical WebstaurantStore products. Once done, `score.json` will include `item_match` + `pack_extraction` (60% of the headline weight is locked behind GT).

3. **Cross-model comparison.** Run the same T2 on:
   - `--model openai --openai-model gpt-5.5` (current)
   - `--model openai --openai-model gpt-5.5-pro`
   - `--model openai --openai-model gpt-5.4-mini` (cheap baseline)
   - `--model northstar --northstar-model tzafon.northstar-cua-fast-1.7-experiment` (newer Tzafon)
   - `--model northstar` via teammate's `run_t2.py` (uses CuaRunner with cycle break)

   Compare headline scores. The user's headline question is "Can CUAs manage buy-side ops?" — a single run isn't an answer, a model-comparison matrix is.

4. **Add T1 / T3 / T4** following the `webstaurant_v1` structure pattern. Specs are in `tasks/*.md`. Each new world gets a directory under `bench/worlds/`.

5. **Trajectory upload via Harbor.** `harness/to_harbor.py` (currently uncommitted) converts our run outputs to Harbor / ATIF-v1.6 format for upload to trajectories.sh. Run it after a successful full T2 run to upstream the trajectories.

## Branch state

- You are on `kernel-live-harness`. Tracks `origin/kernel-live-harness`.
- Other branches the team is using: `main`, `buy-side-harness-v0`, `ground-truth-data`, `ground-truth-tooling`, `handoff-readme`, `scoring-module`, `task-spec-calibration`. Coordinate before merging.
- Uncommitted: `harness/to_harbor.py` (teammate added).

## Things that might bite you

- The Kernel browser auto-deletes after `timeout_seconds` (default 60s, we override to 1500). Long T2 runs need this raised.
- Kernel's `live_view` URL is printed at start of each run — open it during a run to debug visually. The browser is real Chromium with cursor, so you can see exactly where the agent is clicking.
- WebstaurantStore is anti-bot-protected. Stealth mode (set in `browsers.create(stealth=True, ...)`) handles it for now. If you start seeing 403s, check stealth flag.
- Northstar's coordinate space is 0-999, **not 0-1280**. If you're modifying NorthstarAdapter, run `_denorm` against viewport.
- The Docker `archive/` path is broken (browsertrix Redis bug); `docker/pywb/` is also stubbed but not used. Don't try to make these work without good reason.
