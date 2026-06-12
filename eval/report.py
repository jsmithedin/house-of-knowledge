#!/usr/bin/env python3
"""Stage 7: aggregate scores into a markdown report.

Usage:
    python -m eval.report --run-id 2026-06-15-baseline
"""
import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import yaml

from app.config import estimate_cost_usd


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _p(values: list[float], pct: int) -> float:
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def scored_answer(score: dict, results: list[dict]) -> str:
    for r in results:
        if r["query_id"] == score["query_id"] and r["model"] == score.get("model", "?"):
            return r["answer"][:300]
    return "(not found)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    eval_dir = Path("eval")
    run_id = args.run_id or _latest_run_id(eval_dir)
    run_dir = eval_dir / "runs" / run_id

    with open(eval_dir / "config.yaml") as f:
        cfg = yaml.safe_load(f)

    with open(eval_dir / "golden_set.jsonl") as f:
        golden = [json.loads(l) for l in f if l.strip()]
    golden_by_id = {r["id"]: r for r in golden}

    results = _load_jsonl(run_dir / "results.jsonl")
    retrieval = _load_jsonl(run_dir / "retrieval_scores.jsonl")
    scored = _load_jsonl(run_dir / "scored.jsonl")
    judge = _load_jsonl(run_dir / "judge_scores.jsonl")

    lines: list[str] = [f"# Eval Report — {run_id}\n"]

    # --- Human scores per model per tier ---
    lines.append("## Human scores\n")
    by_model_tier: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for s in scored:
        qid = s["query_id"]
        tier = golden_by_id.get(qid, {}).get("tier", "unknown")
        model = s.get("model", "?")
        by_model_tier[model][tier].append(s)

    for model, tiers in sorted(by_model_tier.items()):
        lines.append(f"### {model}\n")
        header = "| tier | n | mean_correctness | mean_completeness | hallucinations |"
        sep    = "|------|---|-----------------|-------------------|----------------|"
        lines.append(header)
        lines.append(sep)
        for tier, items in sorted(tiers.items()):
            mc = statistics.mean(i["correctness"] for i in items)
            mco = statistics.mean(i["completeness"] for i in items)
            nh = sum(1 for i in items if i["hallucination"])
            lines.append(f"| {tier} | {len(items)} | {mc:.2f} | {mco:.2f} | {nh} |")
        lines.append("")

    # --- Retrieval scores per tier ---
    lines.append("## Retrieval (model-independent)\n")
    tier_ret: dict[str, list[dict]] = defaultdict(list)
    for r in retrieval:
        tier = golden_by_id.get(r["query_id"], {}).get("tier", "unknown")
        tier_ret[tier].append(r)

    lines.append("| tier | n | avg_recall | hit_rate |")
    lines.append("|------|---|-----------|----------|")
    for tier, items in sorted(tier_ret.items()):
        ar = statistics.mean(i["recall"] for i in items)
        hr = sum(1 for i in items if i["hit"]) / len(items)
        lines.append(f"| {tier} | {len(items)} | {ar:.2f} | {hr:.2f} |")
    lines.append("")

    # --- Cost and latency ---
    lines.append("## Cost and latency\n")
    by_model_results: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_model_results[r["model"]].append(r)

    lines.append("| model | n_queries | total_cost_usd | cost_per_query | latency_p50_ms | latency_p95_ms |")
    lines.append("|-------|-----------|---------------|----------------|----------------|----------------|")
    for model, items in sorted(by_model_results.items()):
        total_cost = sum(
            estimate_cost_usd(
                r["config"]["model_id"],
                r["usage"]["input_tokens"],
                r["usage"]["output_tokens"],
            )
            for r in items
        )
        latencies = [r["latency_ms"] for r in items]
        p50 = _p(latencies, 50)
        p95 = _p(latencies, 95)
        cpq = total_cost / len(items) if items else 0.0
        lines.append(f"| {model} | {len(items)} | ${total_cost:.4f} | ${cpq:.5f} | {p50:.0f} | {p95:.0f} |")
    lines.append("")

    # --- Human–judge agreement ---
    if judge and scored:
        lines.append("## Human–judge agreement\n")
        threshold = cfg.get("faithfulness_threshold", 1.0)
        judge_by_key = {(r["query_id"], r["model"]): r for r in judge}

        agree = 0
        total = 0
        disagreements: list[str] = []

        for s in scored:
            qid = s["query_id"]
            model = s.get("model")
            key = (qid, model)
            j = judge_by_key.get(key)
            if j is None:
                continue
            human_flag = s["hallucination"]
            judge_flag = j["faithfulness"] < threshold
            total += 1
            if human_flag == judge_flag:
                agree += 1
            else:
                disagreements.append(
                    f"- `{qid}/{model}`: human_hallucination={human_flag}, "
                    f"judge_faithfulness={j['faithfulness']:.2f}"
                )

        rate = agree / total if total else 0.0
        lines.append(f"Agreement rate: **{rate:.1%}** ({agree}/{total})\n")
        if disagreements:
            lines.append("### Disagreements\n")
            lines.extend(disagreements)
            lines.append("")

    # --- Top 3 score gaps ---
    if scored:
        lines.append("## Largest per-query score gaps\n")
        score_by_qm: dict[tuple[str, str], dict] = {
            (s["query_id"], s.get("model", "?")): s for s in scored
        }
        models_list = sorted({s.get("model", "?") for s in scored})
        gaps: list[tuple[float, str, dict, dict]] = []
        if len(models_list) == 2:
            m0, m1 = models_list
            queries = {s["query_id"] for s in scored}
            for qid in queries:
                s0 = score_by_qm.get((qid, m0))
                s1 = score_by_qm.get((qid, m1))
                if s0 and s1:
                    gap = abs(s0["correctness"] - s1["correctness"])
                    gaps.append((gap, qid, s0, s1))
        gaps.sort(reverse=True)
        for gap, qid, s0, s1 in gaps[:3]:
            q_text = golden_by_id.get(qid, {}).get("query", qid)
            lines.append(f"### `{qid}` — gap {gap:.0f} (query: {q_text[:80]})\n")
            lines.append(f"**{models_list[0]}** (correctness={s0['correctness']}):\n> {scored_answer(s0, results)}\n")
            lines.append(f"**{models_list[1]}** (correctness={s1['correctness']}):\n> {scored_answer(s1, results)}\n")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines))
    print(f"Report written to {report_path}")


def _latest_run_id(eval_dir: Path) -> str:
    runs = sorted((eval_dir / "runs").iterdir())
    if not runs:
        raise SystemExit("No runs found. Pass --run-id explicitly.")
    return runs[-1].name


if __name__ == "__main__":
    main()
