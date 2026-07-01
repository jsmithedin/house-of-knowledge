#!/usr/bin/env python3
"""Stage 6: LLM faithfulness judge via RAGAS.

Scores each answer's faithfulness against its retrieved context using RAGAS'
Faithfulness metric, driven by a neutral judge model on Bedrock (Mistral Large
by default — see eval/config.yaml). Faithfulness is claims-supported / total
claims, in [0, 1]; higher is better.

Usage:
    python -m eval.judge --run-id 2026-06-15-baseline
"""
import argparse
import asyncio
import json
import math
from pathlib import Path

import yaml
from langchain_aws import ChatBedrockConverse
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


def _build_scorer(judge_model_id: str, region: str) -> Faithfulness:
    # temperature=0 for a deterministic judge. The judge sits outside both
    # evaluated families (Amazon, Anthropic) to avoid self-preference bias.
    llm = ChatBedrockConverse(
        model_id=judge_model_id,
        region_name=region,
        temperature=0,
    )
    return Faithfulness(llm=LangchainLLMWrapper(llm))


def judge_one(scorer: Faithfulness, query: str, answer: str, retrieved: list[dict]) -> float:
    sample = SingleTurnSample(
        user_input=query,
        response=answer,
        retrieved_contexts=_contexts_from_retrieved(retrieved),
    )
    score = asyncio.run(scorer.single_turn_ascore(sample))
    if score is None or (isinstance(score, float) and math.isnan(score)):
        # RAGAS returns NaN when it can't extract any claims. Treat as 0.0 so
        # the record stays comparable and trips the unfaithful threshold.
        return 0.0
    return round(float(score), 4)


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
    scorer = _build_scorer(judge_model_id, settings.aws_region)

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

    with open(run_dir / "results.jsonl") as f, open(output_path, "a") as out:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid, model = rec["query_id"], rec["model"]
            if (qid, model) in done:
                continue

            faithfulness = judge_one(scorer, queries[qid], rec["answer"], rec["retrieved"])
            score_rec = {
                "query_id": qid,
                "model": model,
                "faithfulness": faithfulness,
                "judge_model": judge_model_id,
                "method": "ragas",
            }
            out.write(json.dumps(score_rec) + "\n")
            out.flush()
            flag = "UNFAITHFUL" if faithfulness < cfg.get("faithfulness_threshold", 1.0) else "ok"
            print(f"  {qid}/{model}: faithfulness={faithfulness:.2f} [{flag}]")

    print(f"\nJudge scores written to {output_path}")


def _latest_run_id(eval_dir: Path) -> str:
    runs = sorted((eval_dir / "runs").iterdir())
    if not runs:
        raise SystemExit("No runs found. Pass --run-id explicitly.")
    return runs[-1].name


if __name__ == "__main__":
    main()
