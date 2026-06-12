#!/usr/bin/env python3
"""Stage 6: LLM faithfulness judge via BedrockClient (hand-rolled fallback).

Usage:
    python -m eval.judge --run-id 2026-06-15-baseline
"""
import argparse
import json
import re
from pathlib import Path

import yaml

from app.bedrock import BedrockClient
from app.config import Settings

_JUDGE_SYSTEM = (
    "You are an impartial faithfulness evaluator. "
    "Your task is to assess whether an answer is supported by the provided context. "
    "Do not use any knowledge outside the provided context."
)

_JUDGE_PROMPT_TMPL = """\
Context retrieved from the knowledge base:
<context>
{context}
</context>

Answer to evaluate:
<answer>
{answer}
</answer>

Instructions:
1. List each distinct factual claim made in the answer (ignore hedges like "the notes don't say").
2. For each claim, determine whether it is directly supported by the context above.
3. Return JSON only, no other text:
{{"claims": [{{"claim": "...", "supported": true}}, ...]}}
"""


def _build_context_from_retrieved(retrieved: list[dict]) -> str:
    parts = []
    for r in retrieved:
        doc = r.get("document", "")
        if doc:
            parts.append(f"[{r.get('chunk_id', '')}]\n{doc}")
    return "\n\n---\n\n".join(parts)


def _parse_claims(text: str) -> list[dict]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
        return data.get("claims", [])
    except json.JSONDecodeError:
        return []


def judge_one(client: BedrockClient, answer: str, retrieved: list[dict]) -> tuple[float, list[str]]:
    context = _build_context_from_retrieved(retrieved)
    prompt = _JUDGE_PROMPT_TMPL.format(context=context[:8000], answer=answer)
    result = client.invoke(_JUDGE_SYSTEM, prompt)
    claims = _parse_claims(result.text)
    if not claims:
        return 0.0, []
    unsupported = [c["claim"] for c in claims if not c.get("supported", True)]
    faithfulness = (len(claims) - len(unsupported)) / len(claims)
    return round(faithfulness, 4), unsupported


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
    client = BedrockClient(model_id=judge_model_id, region=settings.aws_region)

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

            faithfulness, unsupported = judge_one(client, rec["answer"], rec["retrieved"])
            score_rec = {
                "query_id": qid,
                "model": model,
                "faithfulness": faithfulness,
                "unsupported_claims": unsupported,
                "judge_model": judge_model_id,
                "method": "fallback",
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
