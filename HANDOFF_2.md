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

1. **`run_t2_spike.py` is hardcoded to T2.** It loads `bench/worlds/webstaurant_v1/tasks/T2-001/intent.md` and `ground_truth.json` (path-based). It needs to generalize to T3 and T4. Two options:
   - (Recommended) Add `--task {T2,T3,T4}` flag that switches:
     - prompt source (`tasks/T<N>_*.md`, parsed `## Agent prompt` section)
     - ground truth path (`ground_truth/T<N>_ground_truth.json`)
     - start URL (T2: WebstaurantStore homepage; T3: REI homepage; T4: AbeBooks homepage or DDG)
     - scorer (`scripts/scoring.py:score_t<N>`)
     - max_steps (T2: 120, T3: 120, T4: 200 — multi-source needs more)
   - (Alternative) Write a new `harness/launch.py` and leave run_t2_spike.py alone. Cleaner break, more code to maintain.
2. **`to_harbor.py` reads `bench/worlds/webstaurant_v1/tasks/T2-001/intent.md`** for the user-step text. Generalize to read `tasks/T<N>_*.md` based on a `--task` arg.
3. **Multi-trial wrapper.** Currently each invocation runs *one* trial. Need a wrapper that runs N trials and aggregates them under a single Harbor job. See "Multi-trial pattern" below.
4. **Per-task scoring integration.** `run_t2_spike.py` already calls `bench/evaluators/t2_restaurant_restock.py`. Switch to **`scripts/scoring.py:score_t<N>`** (the official, calibrated scorer) and pass `trajectory` for T3 (it scores `filter_engagement`).
5. **T3 and T4 output schemas differ from T2.** Each task's `## Agent prompt` section in `tasks/T<N>_*.md` defines the schema. Verify the harness extracts the agent's final JSON correctly for T3 (boot pick + cart state + filter actions) and T4 (per-book sourcing + total cost).
6. **`item_match` for T2 is currently null** in our previous runs because GT URLs weren't populated. They're populated now (`ground_truth/T2_ground_truth.json` has full primary + alternative URLs). Re-running T2 should produce non-null `item_match` and `pack_extraction` sub-scores.

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

### Step 1 — generalize the runner to all 3 tasks (~30–45 min)

Edit `harness/run_t2_spike.py` (or write a new `harness/launch.py`). Concrete changes:

```python
# Add CLI:
parser.add_argument("--task", choices=["T2", "T3", "T4"], required=True)
parser.add_argument("--runs", type=int, default=1)

# Replace the T2-hardcoded block with:
TASK_MD = REPO_ROOT / "tasks" / {
    "T2": "T2_restaurant_restock.md",
    "T3": "T3_rei_hiking_boots.md",
    "T4": "T4_used_books_basket.md",
}[args.task]
GT_PATH = REPO_ROOT / "ground_truth" / f"{args.task}_ground_truth.json"
START_URL = {
    "T2": "https://www.webstaurantstore.com/",
    "T3": "https://www.rei.com/",
    "T4": "https://www.abebooks.com/",   # or DDG; the agent will navigate
}[args.task]
MAX_STEPS = {"T2": 120, "T3": 120, "T4": 200}[args.task]

# Extract agent prompt from the markdown's `## Agent prompt` code block:
def extract_agent_prompt(md_text: str) -> str:
    # Find "## Agent prompt" header, then the next ``` ... ``` block.
    import re
    m = re.search(r"^## Agent prompt\s*\n+```(?:\w*)?\n(.*?)```", md_text, re.DOTALL | re.MULTILINE)
    return m.group(1).strip() if m else md_text  # fallback: pass whole md
```

Replace the scoring call with Dimitry's scorer:

```python
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from scoring import score_t2, score_t3, score_t4

scorer = {"T2": score_t2, "T3": score_t3, "T4": score_t4}[args.task]
gt = json.loads(GT_PATH.read_text())
parsed = extract_json_object(final_message)
if args.task == "T3":
    # T3 also wants the action list for filter_engagement; build it from traj.jsonl
    trajectory_actions = [...]  # list of {"target": ..., "url": ...} per action
    score_dict = scorer(parsed, gt, trajectory_actions)
else:
    score_dict = scorer(parsed, gt)
```

### Step 2 — verify with one trial per task (~5 min each)

```bash
cd harness
uv run python -u run_t2_spike.py --task T2 --model northstar --runs 1
uv run python -u run_t2_spike.py --task T3 --model northstar --runs 1
uv run python -u run_t2_spike.py --task T4 --model northstar --runs 1
```

For each, confirm:
- Browser navigates to the right start URL.
- Agent JSON extracts (check `outputs/<run_id>/.../agent_response.json`).
- `score.json` is written and `headline` is a sane number.
- The screenshots make sense.

If a task fails this smoke test, fix it before launching the full matrix.

### Step 3 — launch the full matrix

The cleanest pattern is a tiny shell driver that loops `--runs N` once but loops outer `(task × model)` in bash, since `--runs` inside the harness should reuse the same browser-creation block per trial but **isolate trial outputs**.

```bash
# 5 Northstar trials per task
for task in T2 T3 T4; do
  uv run python -u run_t2_spike.py --task $task --model northstar --runs 5
done

# 20 OpenAI trials per task
for task in T2 T3 T4; do
  uv run python -u run_t2_spike.py --task $task --model openai --runs 20
done
```

Each invocation should produce a directory `outputs/<task>_<model>_<ts>/` containing N trial subdirs (one per trial). Either implement that as N invocations from inside Python OR the bash loop can do `for i in {1..20}; do uv run ...; done` and accept N separate run dirs.

#### Recommended structure

To match Harbor's job-of-trials shape, lay out outputs like:

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

`harness/to_harbor.py` already does single-job conversion. Generalize it to take many `--run` args (it already does!). Then upload:

```bash
# For each job, build the harbor dir and upload.
for job in t2_northstar t2_openai t3_northstar t3_openai t4_northstar t4_openai; do
  TASK=$(echo $job | cut -d_ -f1 | tr a-z A-Z)
  MODEL=$(echo $job | cut -d_ -f2)

  RUNS_FLAGS=""
  for trial in outputs/jobs/${job}/trial_*; do
    RUNS_FLAGS="$RUNS_FLAGS --run ${MODEL}=${trial}"
  done

  uv run python -m harness.to_harbor build \
    --job-name buy_side_bench__${job} \
    --task-name $TASK \
    --out outputs/harbor/${job} \
    $RUNS_FLAGS

  npx --yes trajectories-sh upload trajectory outputs/harbor/${job} \
    --api-key "$TRAJECTORIES_SH_API_KEY"
done
```

⚠️ **`to_harbor.py` currently hardcodes T2's intent.md path.** Patch it to read `tasks/T<task>_*.md` and parse the agent prompt section instead. Search for:

```python
task_text = (
    REPO_ROOT / "bench" / "worlds" / "webstaurant_v1" / "tasks" / task_name / "intent.md"
).read_text()
```

Replace with:

```python
TASK_MD_FOR = {
    "T2": "T2_restaurant_restock.md",
    "T3": "T3_rei_hiking_boots.md",
    "T4": "T4_used_books_basket.md",
}
task_text = extract_agent_prompt(
    (REPO_ROOT / "tasks" / TASK_MD_FOR[task_name]).read_text()
)
```

(Reuse the same `extract_agent_prompt` helper from step 1.)

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

- `harness/run_t2_spike.py` — generalize to `--task` (Step 1)
- `harness/to_harbor.py` — generalize task prompt loading (Step 4 patch)
- New: `harness/launch.sh` (or similar) — bash wrapper for the matrix in Step 3
- `README.md`, `slides.md` — update with the 6 trajectory URLs after upload (Step 5)

## Smoke-test commands (sanity check before launching the matrix)

```bash
# Confirm scorer + GT integration works on a synthetic agent output:
cd /Users/christophersettles/code/refresh/opencua_hackathon
python3 -c "
import json
from scripts.scoring import score_t2
gt = json.load(open('ground_truth/T2_ground_truth.json'))
fake_agent = {'cart': [], 'total_items_added': 0, 'failures': []}
print(score_t2(fake_agent, gt))
"
# Should print a dict with headline ~ 0.0, completeness=0.0, cart_added=0.0
```

```bash
# 1-trial smoke for each task (~5 min each):
cd harness
uv run python -u run_t2_spike.py --task T2 --model northstar --max-steps 60
uv run python -u run_t2_spike.py --task T3 --model northstar --max-steps 60
uv run python -u run_t2_spike.py --task T4 --model northstar --max-steps 60
```

If those three smoke runs each produce a valid `score.json` with a non-error `headline`, you're cleared to launch the full matrix.

## When in doubt

1. Re-read `HANDOFF.md` (Dimitry's). It's the source of truth for what each task tests and what the calibrations are.
2. Check `tasks/T<N>_*.md` for the exact agent prompt and expected output schema.
3. Use the existing T2 OpenAI run (uploaded to https://trajectories.sh/t/e490fb79-1224-48a8-88ec-f86b3faec9a0) as a reference for what a working run looks like.
4. The previous coding agent's commits on `kernel-live-harness` are good references for harness patterns: `git log origin/kernel-live-harness`.

Good luck.
