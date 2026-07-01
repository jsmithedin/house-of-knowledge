#!/usr/bin/env python3
"""Stage 6: LLM faithfulness judge via RAGAS.

Scores each answer's faithfulness against its retrieved context using RAGAS'
Faithfulness metric, driven by a neutral judge model on Bedrock (Mistral Large
by default — see eval/config.yaml). Faithfulness is claims-supported / total
claims, in [0, 1]; higher is better.

Scoring goes through ragas.evaluate() rather than per-sample single_turn_ascore.
evaluate() runs each sample inside a real asyncio Task via its own executor;
single_turn_ascore driven by a bare asyncio.run trips "Timeout should be used
inside a task" on Python 3.12+, because ragas patches the loop with nest_asyncio
and the timeout context in asyncio.wait_for then has no current task.

Usage:
    python -m eval.judge --run-id 2026-06-15-baseline
"""
import argparse
import asyncio
import json
import math
from pathlib import Path

# --- Python 3.12+ / nest_asyncio compatibility shim (must run before ragas import) ---
# Python 3.12 rewrote asyncio.wait_for to always open an asyncio.timeout() block,
# which must run inside a Task. ragas applies nest_asyncio at import, and on
# Python 3.14 that leaves current_task() as None, so every wait_for in ragas'
# scoring path raises "Timeout should be used inside a task" and each score
# silently becomes NaN. evaluate() passes run_config.timeout (default 180), so the
# timeout is never None — we must bypass the broken timeout context for every
# value. asyncio.wait enforces the timeout the older way (loop.call_later), which
# needs no current task, so it works under nest_asyncio on 3.14.
async def _wait_for_compat(fut, timeout=None):
    task = asyncio.ensure_future(fut)
    done, _ = await asyncio.wait({task}, timeout=timeout)
    if task in done:
        return task.result()
    task.cancel()
    raise asyncio.TimeoutError()


asyncio.wait_for = _wait_for_compat
# --- end shim ---

import yaml
from langchain_aws import ChatBedrockConverse
from ragas import EvaluationDataset, evaluate
from ragas.dataset_schema import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import Faithfulness

from app.config import Settings
from eval.runner import load_golden_set


def _contexts_from_retrieved(retrieved: list[dict]) -> list[str]:
    """One context string per retrieved chunk, matching RAGAS' retrieved_contexts."""
    contexts = []
    for r in retrieved:
        doc = r.get("document", "")
        if doc:
            contexts.append(doc)
    return contexts


def _build_judge_llm(judge_model_id: str, region: str) -> LangchainLLMWrapper:
    # temperature=0 for a deterministic judge. The judge sits outside both
    # evaluated families (Amazon, Anthropic) to avoid self-preference bias.
    llm = ChatBedrockConverse(
        model_id=judge_model_id,
        region_name=region,
        temperature=0,
    )
    return LangchainLLMWrapper(llm)


def _clean_score(value: object) -> float:
    """RAGAS returns NaN when it can't extract any claims. Treat as 0.0 so the
    record stays comparable and trips the unfaithful threshold."""
    if value is None:
        return 0.0
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(f):
        return 0.0
    return round(f, 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    eval_dir = Path("eval")
    run_id = args.run_id or _latest_run_id(eval_dir)
    run_dir = eval_dir / "runs" / run_id

    with open(eval_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    settings = Settings()
    judge_model_id: str = cfg["judge_model"]
    judge_llm = _build_judge_llm(judge_model_id, settings.aws_region)

    # results.jsonl stores query_id, not the query text — join back to the golden set.
    queries = {q["id"]: q["query"] for q in load_golden_set(eval_dir)}

    output_path = run_dir / "judge_scores.jsonl"
    done: set[tuple[str, str]] = set()
    if output_path.exists():
        with open(output_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done.add((rec["query_id"], rec["model"]))

    # Collect every (query, model) answer still needing a score, in a fixed order
    # so we can map RAGAS' row-ordered results back to their records.
    pending: list[tuple[str, str]] = []
    samples: list[SingleTurnSample] = []
    with open(run_dir / "results.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid, model = rec["query_id"], rec["model"]
            if (qid, model) in done:
                continue
            pending.append((qid, model))
            samples.append(
                SingleTurnSample(
                    user_input=queries[qid],
                    response=rec["answer"],
                    retrieved_contexts=_contexts_from_retrieved(rec["retrieved"]),
                )
            )

    if not samples:
        print("Nothing to score — all answers already judged.")
        return

    print(f"Scoring {len(samples)} answers with RAGAS Faithfulness ({judge_model_id})…")
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),
        metrics=[Faithfulness(llm=judge_llm)],
        llm=judge_llm,
        show_progress=True,
        raise_exceptions=False,
    )
    scores = result.to_pandas()["faithfulness"].tolist()

    threshold = cfg.get("faithfulness_threshold", 1.0)
    with open(output_path, "a") as out:
        for (qid, model), raw in zip(pending, scores):
            faithfulness = _clean_score(raw)
            out.write(
                json.dumps(
                    {
                        "query_id": qid,
                        "model": model,
                        "faithfulness": faithfulness,
                        "judge_model": judge_model_id,
                        "method": "ragas",
                    }
                )
                + "\n"
            )
            flag = "UNFAITHFUL" if faithfulness < threshold else "ok"
            print(f"  {qid}/{model}: faithfulness={faithfulness:.2f} [{flag}]")

    print(f"\nJudge scores written to {output_path}")


def _latest_run_id(eval_dir: Path) -> str:
    runs = sorted((eval_dir / "runs").iterdir())
    if not runs:
        raise SystemExit("No runs found. Pass --run-id explicitly.")
    return runs[-1].name


if __name__ == "__main__":
    main()
