# Buy-Side Optimization Bench

V0 benchmark for the question: **Can computer-use agents manage buy-side procurement?**

The benchmark gives a CUA model a browser UI with simulated sellers, storefronts,
carts, email RFQs, and final-answer submission. The evaluator scores the actual
environment state, not just the agent's claimed answer.

## What Ships

- 3 scenario-versioned worlds:
  - `wine_restock_v1` / `wine-001`: compare storefront prices and fill carts.
  - `office_bulk_v1` / `office-001`: send RFQs, accept bulk quotes, minimize cost.
  - `pc_build_v1` / `pc-001`: choose parts or a prebuilt under budget.
- Deterministic seller configs with hidden reservation prices.
- Ground-truth compiler for optimal allocation and naive baseline.
- Local browser app for storefront, email, cart, and final answer.
- Evaluator with gates, weighted rubric criteria, and economic metrics.
- Tzafon/Lightcone adapter for Northstar browser tasks.

## Quickstart

Validate and generate ground truth:

```bash
python -m bench.harness.validate_bench
python -m bench.harness.compile_ground_truth --world wine_restock_v1 --task wine-001
```

Run all demo tasks with the deterministic mock agent:

```bash
python -m bench.harness.run_eval --all-demo --mode mock --model mock-optimal --run-id demo_mock
python -m bench.harness.aggregate --run-id demo_mock
```

Open the leaderboard:

```text
bench/outputs/demo_mock/leaderboard.html
```

Run a manual browser demo:

```bash
python -m bench.harness.run_eval --world office_bulk_v1 --task office-001 --mode manual --model human-demo --run-id manual_demo
```

## Tzafon / Northstar Run

Install and configure Tzafon:

```bash
pip install tzafon
export TZAFON_API_KEY=sk_...
```

Tzafon's remote browser must be able to reach the benchmark UI. If the app is
running locally, expose it with a tunnel and pass that public URL:

```bash
python -m bench.harness.run_eval \
  --world wine_restock_v1 \
  --task wine-001 \
  --mode tzafon \
  --model northstar-browser-task \
  --host 0.0.0.0 \
  --public-base-url https://YOUR-TUNNEL.example \
  --run-id tzafon_wine
```

The adapter uses `Lightcone().agent.tasks.start_stream(..., kind="browser")`,
so Northstar is doing the browser work. The harness logs streamed events to
`trajectory.jsonl` and scores the final app state.

## Scoring

Hard gates:

- all required items or PC requirements satisfied
- allowed sellers only
- within budget

Weighted criteria are task-specific and include:

- cost within 5% or 15% of generated optimum
- surplus capture for negotiation tasks
- RFQ usage for quote-based tasks
- max-seller compliance
- final structured response submitted

Headline metric:

```text
net_value_usd = naive_baseline_usd - agent_total_usd
```

## Design Notes

This follows the useful parts of APEX-style worlds/tasks and WebArena-style
task execution, but stays deterministic for a hackathon. Sellers are simulated
instead of archived websites so negotiation, hidden reservation prices, and
ground truth remain scoreable.
