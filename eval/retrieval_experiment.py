#!/usr/bin/env python3
"""Retrieval-only k experiment for Post 7 (no Bedrock calls, no cost).

Re-runs retrieval against the golden set at one or more values of k and
scores it the same way score_retrieval.py scores a full run, so the
numbers are directly comparable to the 2026-07-01-baseline (k=5) report.

Must be run against the exact same index the baseline used — do not
re-index the vault first, or the comparison isn't clean.

Usage:
    python -m eval.retrieval_experiment --k 5 10 15
    python -m eval.retrieval_experiment --k 10 15 --query-id q21 q23 q24
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

from app.config import Settings
from app.embedder import Embedder
from app.store import NoteStore
from eval.score_retrieval import compute_retrieval_score


def load_golden_set(eval_dir: Path) -> list[dict]:
    records: list[dict] = []
    with open(eval_dir / "golden_set.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 15])
    parser.add_argument("--query-id", nargs="+", default=None,
                        help="Only test these query IDs (e.g. q21 q23 q24). Default: all.")
    parser.add_argument("--out", default=None,
                        help="Optional path to write raw per-k, per-query results as JSON.")
    args = parser.parse_args()

    eval_dir = Path("eval")
    settings = Settings()

    golden = load_golden_set(eval_dir)
    if args.query_id:
        wanted = set(args.query_id)
        golden = [g for g in golden if g["id"] in wanted]
    golden_by_id = {g["id"]: g for g in golden}

    print("Loading embedder…")
    embedder = Embedder()
    store = NoteStore(chroma_dir=settings.chroma_dir, collection_name=settings.collection_name)
    print(f"Index size: {store.count} chunks. Confirm this matches the 247-note baseline "
          f"before trusting the comparison.\n")

    all_results: dict[int, list[dict]] = {}

    for k in args.k:
        print(f"=== k={k} ===")
        scores = []
        for rec in golden:
            embedding = embedder.embed_query(rec["query"])
            store_results = store.query(query_embedding=embedding, n_results=k, where=None)
            documents = store_results["documents"][0]
            metadatas = store_results["metadatas"][0]

            retrieved = [
                {"rank": i + 1, "source_path": meta.get("source_path", ""), "heading": meta.get("heading", "")}
                for i, meta in enumerate(metadatas)
            ]
            score = compute_retrieval_score(rec, retrieved)
            score["k"] = k
            scores.append(score)

        all_results[k] = scores

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
        print()

    # Per-query diff across k values, easiest way to see if specific misses (q21/q23/q24) resolve
    print("=== Per-query recall across k values ===")
    ks = args.k
    header_row = ["query_id", "tier"] + [f"k={k}" for k in ks]
    diff_rows = []
    for qid in golden_by_id:
        tier = golden_by_id[qid]["tier"]
        row = [qid, tier]
        for k in ks:
            match = next(s for s in all_results[k] if s["query_id"] == qid)
            row.append(f"{match['recall']:.2f}")
        diff_rows.append(row)

    if _HAS_TABULATE:
        print(tabulate(diff_rows, headers=header_row, tablefmt="simple"))
    else:
        print("\t".join(header_row))
        for row in diff_rows:
            print("\t".join(row))

    if args.out:
        out_path = Path(args.out)
        with open(out_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nWrote raw results to {out_path}")


if __name__ == "__main__":
    main()
