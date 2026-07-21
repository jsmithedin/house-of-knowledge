# Eval Report — 2026-07-01-baseline

## Human scores

### haiku-4.5

| tier | n | mean_correctness | mean_completeness | hallucinations |
|------|---|-----------------|-------------------|----------------|
| lookup | 8 | 2.00 | 2.00 | 0 |
| synthesis | 8 | 1.88 | 1.88 | 0 |
| temporal | 8 | 1.62 | 1.62 | 0 |

### nova-lite

| tier | n | mean_correctness | mean_completeness | hallucinations |
|------|---|-----------------|-------------------|----------------|
| lookup | 8 | 2.00 | 2.00 | 0 |
| synthesis | 8 | 1.88 | 1.62 | 0 |
| temporal | 8 | 1.62 | 1.50 | 1 |

## Retrieval (model-independent)

| tier | n | avg_recall | hit_rate |
|------|---|-----------|----------|
| lookup | 8 | 1.00 | 1.00 |
| synthesis | 8 | 0.60 | 1.00 |
| temporal | 8 | 0.62 | 0.75 |

## Cost and latency

| model | n_queries | total_cost_usd | cost_per_query | latency_p50_ms | latency_p95_ms |
|-------|-----------|---------------|----------------|----------------|----------------|
| haiku-4.5 | 24 | $0.0621 | $0.00259 | 2902 | 4403 |
| nova-lite | 24 | $0.0026 | $0.00011 | 642 | 972 |

## Human–judge agreement

Agreement rate: **31.2%** (15/48)

### Disagreements

- `q14/nova-lite`: human_hallucination=False, judge_faithfulness=0.56
- `q18/nova-lite`: human_hallucination=False, judge_faithfulness=0.60
- `q19/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.94
- `q04/nova-lite`: human_hallucination=False, judge_faithfulness=0.89
- `q23/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.00
- `q09/nova-lite`: human_hallucination=False, judge_faithfulness=0.67
- `q14/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.92
- `q24/nova-lite`: human_hallucination=False, judge_faithfulness=0.60
- `q04/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.93
- `q11/nova-lite`: human_hallucination=False, judge_faithfulness=0.60
- `q20/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.93
- `q01/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.94
- `q02/nova-lite`: human_hallucination=False, judge_faithfulness=0.80
- `q22/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.87
- `q08/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.75
- `q15/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.78
- `q11/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.79
- `q15/nova-lite`: human_hallucination=False, judge_faithfulness=0.80
- `q22/nova-lite`: human_hallucination=False, judge_faithfulness=0.67
- `q24/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.67
- `q23/nova-lite`: human_hallucination=False, judge_faithfulness=0.75
- `q16/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.89
- `q10/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.87
- `q08/nova-lite`: human_hallucination=False, judge_faithfulness=0.50
- `q09/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.67
- `q07/nova-lite`: human_hallucination=False, judge_faithfulness=0.33
- `q02/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.67
- `q07/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.25
- `q12/nova-lite`: human_hallucination=False, judge_faithfulness=0.79
- `q17/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.75
- `q12/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.73
- `q18/haiku-4.5`: human_hallucination=False, judge_faithfulness=0.78
- `q16/nova-lite`: human_hallucination=False, judge_faithfulness=0.92

## Largest per-query score gaps

### `q24` — gap 0 (query: How has the plan for Nand's funeral shifted, and why does the timing matter?)

**haiku-4.5** (correctness=2):
> # Nand's Funeral: Timing and Rationale

Sister Maelin deferred Nand's funeral until after the party's planned mission to [[Neverdeath Graveyard]]. 

**Why the timing matters:** Tolling funeral bells prevent spirits from dispersing to other planes. This consideration affects both spiritual protocol a

**nova-lite** (correctness=2):
> The plan for Nand's funeral has been deferred until after the party's mission to the Neverdeath Graveyard. Sister Maelin explained that tolling funeral bells prevent spirits from dispersing to other planes, which is a consideration that affects both spiritual protocol and tactical timing. This decis

### `q23` — gap 0 (query: How did the portable ward device behave during testing, and what changed when Ul)

**haiku-4.5** (correctness=0):
> I don't have session notes containing information about a portable ward device or its behavior during testing. The retrieved notes don't include details about this specific item or experiment.

To answer your question accurately, I would need session notes that document:
- When the portable ward dev

**nova-lite** (correctness=0):
> The session notes do not provide information on the portable ward device or its behavior during testing, nor do they mention any changes when Ultrin approached it. Therefore, I cannot provide an answer based on the provided notes.

### `q22` — gap 0 (query: What did Lord Neverember reveal to the party in private before the proclamation,)

**haiku-4.5** (correctness=2):
> # Lord Neverember's Private Revelations (Session 36)

In Session 36, Lord Neverember privately confessed to the party:

1. **The city's desperate state** — He candidly admitted Neverwinter's dire condition, dropping his usual facade of authority.

2. **Edgar's death** — He revealed that [[Edgar]], t

**nova-lite** (correctness=2):
> In private, Lord Neverember revealed to the party that the city of Neverwinter was in a desperate state and confessed his own struggles with the crisis. He also disclosed that Edgar had died defending unmarked graves at the party's estate. Additionally, Neverember reconciled with Arlon and showed a 
