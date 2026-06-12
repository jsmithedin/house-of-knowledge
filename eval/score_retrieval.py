#!/usr/bin/env python3
"""Stage 2: deterministic retrieval scoring (no LLM, no network).

Usage:
    python -m eval.score_retrieval --run-id 2026-06-15-baseline
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

try:
    from tabulate import tabulate
    _HAS_TABULATE = True
except ImportError:
    _HAS_TABULATE = False


def compute_retrieval_score(golden_rec: dict, retrieved: list[dict]) -> dict:
    expected = set(golden_rec["source_notes"])
    found: set[str] = set()
    first_relevant_rank: int | None = None

    for r in retrieved:
        sp = r["source_path"]
        if sp in expected:
            found.add(sp)
            if first_relevant_rank is None:
                first_relevant_rank = r["rank"]

    recall = len(found) / len(expected) if expected else 0.0
    return {
        "query_id": golden_rec["id"],
        "expected_notes": list(expected),
        "retrieved_notes": list(dict.fromkeys(r["source_path"] for r in retrieved)),
        "recall": round(recall, 4),
        "first_relevant_rank": first_relevant_rank,
        "hit": recall > 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    eval_dir = Path("eval")
    run_id = args.run_id or _latest_run_id(eval_dir)
    run_dir = eval_dir / "runs" / run_id

    golden_by_id: dict[str, dict] = {}
    with open(eval_dir / "golden_set.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                golden_by_id[rec["id"]] = rec

    # One retrieved list per query (identical across models — use first occurrence)
    seen_queries: set[str] = set()
    scores: list[dict] = []
    with open(run_dir / "results.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid = rec["query_id"]
            if qid in seen_queries:
                continue
            seen_queries.add(qid)
            if qid not in golden_by_id:
                continue
            score = compute_retrieval_score(golden_by_id[qid], rec["retrieved"])
            scores.append(score)

    scores_path = run_dir / "retrieval_scores.jsonl"
    with open(scores_path, "w") as f:
        for s in scores:
            f.write(json.dumps(s) + "\n")

    print(f"Wrote {len(scores)} retrieval scores to {scores_path}")

    # Per-tier summary
    tier_scores: dict[str, list[dict]] = defaultdict(list)
    for s in scores:
        tier = golden_by_id[s["query_id"]].get("tier", "unknown")
        tier_scores[tier].append(s)

    rows = []
    for tier, items in sorted(tier_scores.items()):
        avg_recall = sum(i["recall"] for i in items) / len(items)
        hit_rate = sum(1 for i in items if i["hit"]) / len(items)
        rows.append([tier, len(items), f"{avg_recall:.2f}", f"{hit_rate:.2f}"])

    headers = ["tier", "n", "avg_recall", "hit_rate"]
    if _HAS_TABULATE:
        print(tabulate(rows, headers=headers, tablefmt="simple"))
    else:
        print("\t".join(headers))
        for row in rows:
            print("\t".join(str(c) for c in row))


def _latest_run_id(eval_dir: Path) -> str:
    runs = sorted((eval_dir / "runs").iterdir())
    if not runs:
        raise SystemExit("No runs found. Pass --run-id explicitly.")
    return runs[-1].name


if __name__ == "__main__":
    main()
