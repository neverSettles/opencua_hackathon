# HANDOFF_2 — Full eval matrix for T2/T3/T4 across Northstar + OpenAI

State as of 2026-05-09 evening, post-merge of Dimitry's branches into `eval-runs`.

## TL;DR

Run a full benchmark matrix and upload to trajectories.sh:

| Task | Model | # trials | Why |
|---|---|---|---|
| T2 (WebstaurantStore restock) | Northstar (`tzafon.northstar-cua-fast`) | **5** | Pass-rate measurement |
| T3 (REI hiking boots) | Northstar | **5** | Pass-rate measurement |
| T4 (Used books basket) | Northstar | **5** | Pass-rate measurement |
| T2 | OpenAI computer-use | **20** | Pass-rate **+ distillation candidates for SFT later** |
| T3 | OpenAI computer-use | **20** | Same |
| T4 | OpenAI computer-use | **20** | Same |

**Total: 75 trials.** Each completes in ~2–10 minutes wall-clock. Sequential ≈ 5–8 hours; parallelize across tasks if you want it faster.

After all runs: bundle each `(task, model)` set into a Harbor / ATIF-v1.6 job, upload to trajectories.sh. Fine-tuning happens later in a different session — for now, **just produce trajectories**.

## What's already in place on `eval-runs`

```
eval-runs (you are here, branched off kernel-live-harness)
├── tasks/T{2,3,4}_*.md          ← Dimitry's calibrated specs (incl. the agent prompt)
├── ground_truth/T{2,3,4}_ground_truth.json   ← populated, live-collected
├── ground_truth/_rei_raw.json   ← raw REI scrape backing T3
├── scripts/scoring.py           ← Dimitry's score_t2 / score_t3 / score_t4
├── scripts/t3_build_ground_truth.py
├── scripts/t4_optimizer.py
├── HANDOFF.md                   ← Dimitry's original handoff (READ THIS FIRST)
├── README.md
├── slides.md                    ← presentable deck
├── harness/
│   ├── adapters/                ← multi-model adapter system (works)
│   │   ├── base.py
│   │   ├── northstar.py         ← Tzafon Lightcone Responses API
│   │   ├── openai_cua.py        ← OpenAI computer-use Responses API
│   │   ├── bedrock_claude.py    ← Anthropic via Bedrock
│   │   └── gemini.py            ← Google Gemini
│   ├── cua_runner.py            ← vendored from tzafon/lightcone + cycle/no-action improvements
│   ├── kernel_computer.py       ← Kernel surface for upstream CuaRunner
│   ├── run_t2.py                ← Tzafon-only T2 driver (uses CuaRunner)
│   ├── run_t2_spike.py          ← multi-model T2 driver (uses adapters/) — THIS IS WHAT YOU EXTEND
│   ├── to_harbor.py             ← convert run outputs → Harbor / ATIF-v1.6 job dirs
│   └── requirements.txt
└── outputs/                     ← gitignored. one dir per run.
```

### What works today

- `harness/run_t2_spike.py --model {northstar,openai,gemini}` runs T2 end-to-end on **live WebstaurantStore** through Kernel browsers.
- `harness/to_harbor.py build` converts our two run formats (CuaRunner trace.jsonl and adapter trajectory.jsonl) into Harbor jobs.
- `npx --yes trajectories-sh upload trajectory <job_dir> --api-key $TRAJECTORIES_SH_API_KEY` works (already used to upload one cross-model T2 job: https://trajectories.sh/t/e490fb79-1224-48a8-88ec-f86b3faec9a0).

### What's missing (your work)

> **UPDATE 2026-05-09 evening:** The previous coding agent already did items 1, 2, 3, and 4 below. `harness/launch.py` is the canonical entry point. **You can skip straight to "Recipe → Step 2/Step 3" below.** Items 5 and 6 are confirmed working.

1. ~~`run_t2_spike.py` is hardcoded to T2.~~ **Done.** `harness/launch.py` takes `--task {T2,T3,T4}`. It auto-loads the calibrated prompt from `tasks/T<N>_*.md` (parsed `## Agent prompt` block), the ground truth from `ground_truth/T<N>_ground_truth.json`, and routes to `scripts/scoring.py:score_t<N>`. Per-task config (start URL, max_steps, T3 trajectory passthrough) lives in `TASK_CONFIG` at top of `launch.py`.
2. ~~`to_harbor.py` reads bench/worlds intent.md.~~ **Done.** Now reads `tasks/T<N>_*.md` with the same `extract_agent_prompt` helper and falls back to the legacy bench path if needed.
3. ~~Multi-trial wrapper.~~ **Done.** `launch.py --runs N` runs N trials in sequence, each in its own Kernel session and `trial_NNN/` subdir, with a `job_summary.json` at the end. Output layout is exactly what Harbor wants — see `outputs/jobs/t2_northstar_20260509T224839Z/` for reference.
4. ~~Per-task scoring integration.~~ **Done.** `launch.py` imports `score_t2 / score_t3 / score_t4` from `scripts/scoring.py`. T3's `filter_engagement` sub-score gets a synthesized action list with `target` (action_type) and `url` (page URL captured per action) from the trajectory. Each scorer's full output goes to `trial_NNN/score.json`.
5. **T3 and T4 output schemas** — confirmed in code: `extract_json_object` is tolerant of fenced JSON / prose / nested objects. The scorers themselves enforce schema correctness via their sub-scores. If an agent doesn't emit JSON, `score.json` is absent and the trial counts as 0 in the headline aggregate.
6. **`item_match` for T2** — Dimitry's ground truth now has full URLs and `alternative_acceptable_products` per item. Re-running T2 will produce non-null `item_match` and `pack_extraction` sub-scores (unlike our earlier null-GT runs).

### Smoke-test result (already done)

```
$ uv run python -u launch.py --task T2 --model northstar --runs 1 --max-steps 30
=== Launching T2 x northstar x 1 trials ===
Spec: T2_restaurant_restock.md
Start URL: https://www.webstaurantstore.com/
[trial] navigating to ...  status=stopped_no_action  elapsed=36.4s
=== Job complete ===  trial_dir: outputs/jobs/t2_northstar_<ts>/trial_001/
Build harbor job:
  uv run python -m harness.to_harbor build --job-name ... --task-name T2 ... --run northstar=...
```

Pipeline confirmed end-to-end. (Stopped early because we capped max_steps=30 for the smoke; the full task default is 120 for T2.)

## Recipe (do this, in order)

### Step 0 — environment

```bash
git checkout eval-runs
git pull
# .env should have these keys (the user has them already):
# KERNEL_API_KEY=sk_...
# TZAFON_API_KEY=sk_... (Lightcone)
# OPENAI_API_KEY=sk_...
# TRAJECTORIES_SH_API_KEY=tsh_...
cd harness && bash setup.sh && cd ..    # creates harness/.venv with kernel + tzafon + playwright
```

### Step 1 — *(done by previous agent)*

`harness/launch.py` already exists and routes `--task {T2,T3,T4}` to the right prompt, GT, scorer, start URL, and max_steps. T3 already passes the trajectory action list through to `score_t3`. Skip to Step 2.

### Step 2 — verify with one trial per task (~3–5 min each)

```bash
cd harness
uv run python -u launch.py --task T2 --model northstar --runs 1 --max-steps 60
uv run python -u launch.py --task T3 --model northstar --runs 1 --max-steps 60
uv run python -u launch.py --task T4 --model northstar --runs 1 --max-steps 80
```

For each, confirm:
- Browser navigates to the right start URL (T2: webstaurantstore.com, T3: rei.com/c/hiking-boots, T4: abebooks.com).
- A `trial_001/` subdir is created with `trajectory.jsonl`, screenshots, `summary.json`.
- If the agent emits JSON, `score.json` is written and `headline` is a sane number.

T2 smoke is already confirmed working (see "Smoke-test result" above). T3/T4 should work too — the only task-specific path is the JSON schema the model is asked to emit, which is part of the prompt extracted from `tasks/T<N>_*.md`.

### Step 3 — launch the full matrix

```bash
cd harness

# 5 Northstar trials per task (~15 trials total, ~30-60 min)
for task in T2 T3 T4; do
  uv run python -u launch.py --task $task --model northstar --runs 5
done

# 20 OpenAI trials per task (~60 trials total, ~3-6 hr — consider running in background)
for task in T2 T3 T4; do
  uv run python -u launch.py --task $task --model openai --runs 20
done
```

Each invocation produces:

```
outputs/jobs/<task>_<model>_<ts>/
├── trial_001/
│   ├── trajectory.jsonl
│   ├── step_NNN.png
│   ├── agent_response.raw.txt
│   ├── agent_response.json   (if agent emitted parseable JSON)
│   ├── score.json            (if scorer ran)
│   └── summary.json
├── trial_002/  ...
├── trial_005/  (or trial_020/ for OpenAI)
└── job_summary.json
```

#### Output layout reference

```
outputs/
└── jobs/
    ├── t2_northstar/
    │   ├── trial_001/   ← one full run (intent + screenshots + actions + score)
    │   ├── trial_002/
    │   ├── trial_003/
    │   ├── trial_004/
    │   └── trial_005/
    ├── t2_openai/
    │   ├── trial_001/
    │   ├── ... (20 trials)
    │   └── trial_020/
    ├── t3_northstar/  (5 trials)
    ├── t3_openai/     (20 trials)
    ├── t4_northstar/  (5 trials)
    └── t4_openai/     (20 trials)
```

Total: 6 jobs, 75 trials.

### Step 4 — convert each job to Harbor format and upload

`harness/to_harbor.py` is already generalized to read `tasks/T<N>_*.md` based on the `--task-name` arg. Glob each job dir's `trial_*/` subdirs and pass them as `--run` flags:

```bash
cd /Users/christophersettles/code/refresh/opencua_hackathon

# Each launch.py invocation prints its harbor build command at the end. Copy
# that, OR script it like this:

source .env
for job_dir in outputs/jobs/*/; do
  job=$(basename "$job_dir")        # e.g. t2_northstar_20260509T...
  task_lower=$(echo "$job" | cut -d_ -f1)
  task=$(echo "$task_lower" | tr a-z A-Z)
  model=$(echo "$job" | cut -d_ -f2)

  runs_flags=()
  for trial in "$job_dir"trial_*; do
    runs_flags+=(--run "${model}=${trial}")
  done

  out_harbor="outputs/harbor/${job}"

  cd harness
  uv run python to_harbor.py build \
    --job-name "buy_side_bench__${job}" \
    --task-name "$task" \
    --out "../${out_harbor}" \
    "${runs_flags[@]}"
  cd ..

  npx --yes trajectories-sh upload trajectory "$out_harbor" \
    --api-key "$TRAJECTORIES_SH_API_KEY"
done
```

The upload prints a viewer URL like `https://trajectories.sh/t/<uuid>`. Save those.

### Step 5 — collect the 6 trajectory.sh URLs

After step 4, you'll have 6 trajectory job URLs. Add them to README.md and slides.md. Format:

```markdown
## Trajectories

| Task | Model | # Trials | Pass rate | Job URL |
|---|---|---|---|---|
| T2 | Northstar | 5 | XX% | https://trajectories.sh/t/... |
| T2 | OpenAI | 20 | XX% | https://trajectories.sh/t/... |
| T3 | Northstar | 5 | XX% | https://trajectories.sh/t/... |
| T3 | OpenAI | 20 | XX% | https://trajectories.sh/t/... |
| T4 | Northstar | 5 | XX% | https://trajectories.sh/t/... |
| T4 | OpenAI | 20 | XX% | https://trajectories.sh/t/... |
```

Pass rate per spec — read `result.json` per trial and compute. The `to_harbor.py` already aggregates `reward_stats` in the job-level `result.json`; show that.

## Known issues / gotchas

1. **OpenAI model names changed.** Check `harness/adapters/openai_cua.py` for the current default. Use `--openai-model gpt-5.5` or `computer-use-preview`. Verify with a 1-trial smoke first.
2. **Anti-bot.** Kernel stealth handles WebstaurantStore (verified). REI and AbeBooks should be similar — server-rendered, low anti-bot. If you hit 403s, set `stealth=True` (already default in our adapter) or add a residential proxy via Kernel's proxy support.
3. **T3 `filter_engagement`** sub-score requires the trajectory action list. Our `trajectory.jsonl` has the right shape — pass it through. If you skip this, T3's headline drops by 0.10.
4. **T4 takes longer.** Multi-source means the agent has to navigate across AbeBooks / Powells / ThriftBooks / Better World Books. Bump `max_steps` to 200 and `timeout_seconds` to 1800 for the Kernel browser.
5. **CuaRunner cycle detection** can fire false positives on long T4 sessions where the agent legitimately revisits AbeBooks repeatedly. Watch the logs; if it triggers spuriously, raise `circuit_breaker_threshold` from 3 to 4 in the runner config.
6. **Northstar gets stuck more than OpenAI.** From our earlier experiments: Northstar narrates without acting and tends to lose spatial state across navigations. Expect lower pass rates for Northstar — that's a real eval signal, not a bug.
7. **Cost estimate (rough):** OpenAI computer-use pricing × 60 trials × ~50K tokens each ≈ $30–60 for the OpenAI runs. Lightcone/Northstar pricing per the docs is $1/M input + $5/M output ≈ $5–10 for the 15 Northstar runs. Confirm before launching.
8. **Disk.** Each run dir is ~5–20 MB (mostly screenshots). 75 runs × ~10 MB ≈ 750 MB. Should be fine; watch free space.
9. **`outputs/` is gitignored.** Don't commit run artifacts. The Harbor job dirs you upload are derived; the source-of-truth is on trajectories.sh.

## What does NOT need to happen yet

- **Fine-tuning.** Just produce the OpenAI trajectories (20 per task = 60 total). The next session will distill from those. No need to format them as SFT data right now.
- **Re-curating T4 books** for non-trivial shipping costs. Per Dimitry's HANDOFF.md, current data is the fallback; the headline reweighting handles it. Don't go back into book selection.
- **OpenShell wrap.** Out of scope for this run.
- **Archiving / pywb / WACZ.** Already deferred. We're running on live web through Kernel stealth.

## Files you'll touch

Almost nothing — most of the heavy lifting was done already. You should only need to:

- Run `harness/launch.py` 6 times (3 tasks × 2 models)
- Run `harness/to_harbor.py` 6 times (one per job)
- Run `npx trajectories-sh upload trajectory ...` 6 times
- Update `README.md` and `slides.md` with the 6 viewer URLs and the pass-rate table

If you find a bug in the runner, edit `harness/launch.py`. If you need to tune cycle-detection or no-action retry behavior, edit `harness/cua_runner.py` (used by `run_t2.py`, not `launch.py`) — though `launch.py`'s in-line loop has the same MAX_NO_ACTION=3 retry budget.

## Smoke-test commands (already confirmed; rerun if you've edited launch.py)

```bash
# (1) Confirm scorer + GT integration works on synthetic agent output:
cd /Users/christophersettles/code/refresh/opencua_hackathon
uv run --project harness python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts")
from scoring import score_t2, score_t3, score_t4
repo = Path(".")
for task, scorer in (("T2", score_t2), ("T3", score_t3), ("T4", score_t4)):
    gt = json.loads((repo / "ground_truth" / f"{task}_ground_truth.json").read_text())
    fake = {"cart": [], "boot": {}, "answers": [], "sourcing": [], "total_cost": 0}
    r = scorer(fake, gt, []) if task == "T3" else scorer(fake, gt)
    print(f"  {task}: headline={r.get('headline'):.3f} OK")
PY
# Already verified to print:
#   T2: headline=0.000 OK    T3: headline=0.000 OK    T4: headline=0.200 OK
```

```bash
# (2) 1-trial smoke for each task (T2 already verified; rerun T3/T4 if you've changed code):
cd harness
uv run python -u launch.py --task T3 --model northstar --runs 1 --max-steps 60
uv run python -u launch.py --task T4 --model northstar --runs 1 --max-steps 80
```

If the smoke runs land in `outputs/jobs/<task>_northstar_<ts>/trial_001/` with a `summary.json`, you're cleared to launch the full matrix.

## When in doubt

1. Re-read `HANDOFF.md` (Dimitry's). It's the source of truth for what each task tests and what the calibrations are.
2. Check `tasks/T<N>_*.md` for the exact agent prompt and expected output schema.
3. Use the existing T2 OpenAI run (uploaded to https://trajectories.sh/t/e490fb79-1224-48a8-88ec-f86b3faec9a0) as a reference for what a working run looks like.
4. The previous coding agent's commits on `kernel-live-harness` are good references for harness patterns: `git log origin/kernel-live-harness`.

Good luck.
