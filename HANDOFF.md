# Buy-side CUA bench — handoff for harness work

State as of 2026-05-09. Three branches not yet merged carry everything below; merge order doesn't matter except that `task-spec-calibration` is what the calibration narrative below describes.

## Branches

| Branch | What it adds |
|---|---|
| `task-spec-calibration` | Edits to `tasks/T2_*.md`, `tasks/T3_*.md`, `tasks/T4_*.md` reflecting two calibration changes plus a Speedboat ISBN fix |
| `ground-truth-data` | `ground_truth/T2_ground_truth.json`, `T3_ground_truth.json`, `T4_ground_truth.json`, plus the raw REI scrape `_rei_raw.json` |
| `ground-truth-tooling` | `scripts/t3_build_ground_truth.py`, `scripts/t4_optimizer.py` — the deterministic builders that produced the JSONs above |
| `scoring-module` | `scripts/scoring.py` — `score_t2 / score_t3 / score_t4(agent_output, ground_truth, ...)` |

## What the harness consumes

For each task:

1. **Task spec** (`tasks/T<N>_*.md`) — agent prompt, output schema, calibration notes.
2. **Ground truth** (`ground_truth/T<N>_ground_truth.json`) — what the scorer compares against.
3. **Scorer** (`scripts/scoring.py`) — `score_t<N>(agent_output, ground_truth)` returns a dict with sub-scores and a `headline` aggregate.

CLI shape for one-shot scoring:

```bash
python scripts/scoring.py t2 --agent run.json --gt ground_truth/T2_ground_truth.json
python scripts/scoring.py t3 --agent run.json --gt ground_truth/T3_ground_truth.json --trajectory traj.json
python scripts/scoring.py t4 --agent run.json --gt ground_truth/T4_ground_truth.json
```

## Two calibrations applied during ground-truth collection

These are in the spec markdowns as well, captured here for context.

### T3: minimum-review threshold for the optimum pool

Without it, three boots tied at 5.0 stars / 1 review each would all be "optimal," producing a non-unique answer and breaking `rating_optimal`. The T3 builder (`scripts/t3_build_ground_truth.py`) and the agent prompt now require ≥100 reviews to be in the optimum pool.

Resulting optimum: **adidas Terrex Free Hiker GORE-TEX 2.0** (4.7★, 1277 reviews, $220, 31.4 oz). Caveat — adidas brands this as "Hiking Shoes" though it sits in REI's `over-the-ankle` filter (which the spec accepts as mid-cut). Models that strictly read the prompt's "not low-cut hiking shoes" line may pick the runner-up Skychaser AX5 (4.7★, 166 reviews) instead.

### T4: headline reweighted, optimality demoted to sanity gate

When ground truth was collected, the gap between the per-book-cheapest (naive) and the consolidation-aware (optimal) total came out to **0%** at $41.41. Real-world AbeBooks free-shipping prevalence (World of Books, Gulf Coast, Your Online Bookstore) ate the consolidation incentive the original task was designed for.

T4 headline was reweighted from `0.40 optimality + 0.30 validity + 0.20 completeness + 0.10 cost_correctness` → `0.40 validity + 0.30 completeness + 0.20 cost_correctness + 0.10 optimality`. T4 now primarily tests cross-source navigation. The optimizer is still run; optimality stays as a sanity gate that catches agents producing sourcings that cost more than naive.

The ground-truth JSON carries `headline_weights` so the scorer reads them dynamically — flipping back to the original weights is a JSON edit, no code change.

If a teammate has bandwidth to re-curate the book list with niche/recent ISBNs that have non-trivial AbeBooks shipping, the headline can flip back. Current data ships as the fallback.

### T2: catalog-reality acceptance policy (not a calibration, just notes)

WebstaurantStore has no SKU explicitly labeled "heavy duty" 9" plates, no 13-gallon-only clear liner SKU, and sells stainless steel scour pads in 12/Pack rather than "cases." The eval relies on `alternative_acceptable_products` in the T2 ground truth so reasonable agent picks pass `item_match` regardless.

## Other notes carried forward

- **T4 Speedboat ISBN was wrong in the seed** (9781590176160 → resolves to *The Green Man* by Kingsley Amis). Corrected on the `task-spec-calibration` branch to **9781590176139** in both the spec markdown and the ground truth.
- **T4 matching policy** is `title_author_format` (relaxed from strict ISBN). ThriftBooks and Powell's roll up at the work level rather than per-ISBN, so strict ISBN matching wasn't enforceable across all four sellers. AbeBooks listings still come from ISBN search; matching is loose at the resulting edition level.
- **Reproducing collection:** drive the same filter URLs / search URLs against live sites; use `scripts/t3_build_ground_truth.py` to rebuild T3 from a fresh REI scrape; use `scripts/t4_optimizer.py --inplace` to recompute T4 totals after editing listings.

## What's not done yet

- Eval runner (per-(model, task) execution loop, screenshot/action logging for SFT data, multi-run variance estimation).
- Northstar / Lightcone integration (`pip install tzafon`, computer-loop scaffold).
- Trajectory format definition for the `filter_engagement` check in T3 — `score_t3(...)` accepts a list of action dicts with `target` / `url` keys; the harness needs to decide on the canonical action-log shape.
- Re-curated T4 book list to restore the optimality signal (deferred; current data ships as fallback).
