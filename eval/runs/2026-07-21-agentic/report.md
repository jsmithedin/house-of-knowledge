# Eval Report — 2026-07-21-agentic

## Human scores

## Judge faithfulness only — no human scoring this run

No blind human scoring was done for this run. These are RAGAS/judge-model faithfulness scores only (claims supported by retrieved context — not correctness against the golden answer, not completeness). The judge agreed with human scoring only 31% of the time in Post 6 and tends to punish honest refusals — treat as a rough signal, not a verdict.

### haiku-4.5

| tier | n | mean_faithfulness | below_threshold |
|------|---|--------------------|------------------|
| lookup | 8 | 0.84 | 6 |
| synthesis | 8 | 0.83 | 5 |
| temporal | 8 | 0.91 | 5 |

### nova-lite

| tier | n | mean_faithfulness | below_threshold |
|------|---|--------------------|------------------|
| lookup | 8 | 0.94 | 3 |
| synthesis | 8 | 0.87 | 3 |
| temporal | 7 | 0.77 | 6 |

## Retrieval (model-independent)

| tier | n | avg_recall | hit_rate |
|------|---|-----------|----------|
| lookup | 15 | 1.00 | 1.00 |
| synthesis | 15 | 0.64 | 1.00 |
| temporal | 15 | 0.67 | 0.73 |

## Retrieval by model

Retrieval diverged by model (each drove its own searches) — the table above blends both; this breaks them out.

| tier | model | n | avg_recall | hit_rate |
|------|-------|---|-----------|----------|
| lookup | haiku-4.5 | 7 | 1.00 | 1.00 |
| lookup | nova-lite | 8 | 1.00 | 1.00 |
| synthesis | haiku-4.5 | 7 | 0.76 | 1.00 |
| synthesis | nova-lite | 8 | 0.54 | 1.00 |
| temporal | haiku-4.5 | 7 | 0.71 | 0.71 |
| temporal | nova-lite | 8 | 0.62 | 0.75 |

## Cost and latency

| model | n_queries | total_cost_usd | cost_per_query | latency_p50_ms | latency_p95_ms |
|-------|-----------|---------------|----------------|----------------|----------------|
| haiku-4.5 | 24 | $0.2626 | $0.01094 | 6101 | 11556 |
| nova-lite | 24 | $0.0066 | $0.00027 | 2375 | 6653 |
