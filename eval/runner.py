#!/usr/bin/env python3
"""Stage 1: batch RAG runs against the golden set.

Usage:
    python -m eval.runner --run-id 2026-06-15-baseline [--limit N] [--query-id q01]
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.bedrock import BedrockClient
from app.config import NOVA_LITE_MODEL_ID, HAIKU_MODEL_ID, Settings, estimate_cost_usd
from app.embedder import Embedder
from app.rag import SYSTEM_PROMPT, build_context, build_user_message
from app.store import NoteStore

_MODEL_IDS: dict[str, str] = {
    "nova-lite": NOVA_LITE_MODEL_ID,
    "haiku-4.5": HAIKU_MODEL_ID,
}

_INFERENCE_CONFIG = {"maxTokens": 2048, "temperature": 0.0}


def load_config(eval_dir: Path) -> dict:
    with open(eval_dir / "config.yaml") as f:
        return yaml.safe_load(f)


def load_golden_set(eval_dir: Path) -> list[dict]:
    records: list[dict] = []
    with open(eval_dir / "golden_set.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def validate_golden_set(records: list[dict], obsidian_dir: str) -> None:
    ids = [r["id"] for r in records]
    if len(ids) != len(set(ids)):
        print("WARNING: duplicate query ids in golden_set.jsonl", file=sys.stderr)
    valid_tiers = {"lookup", "synthesis", "temporal"}
    vault = Path(obsidian_dir)
    for r in records:
        if r.get("tier") not in valid_tiers:
            print(f"WARNING: {r['id']} has invalid tier {r.get('tier')!r}", file=sys.stderr)
        if not r.get("source_notes"):
            print(f"WARNING: {r['id']} has empty source_notes", file=sys.stderr)
        for note in r.get("source_notes", []):
            if not (vault / note).exists():
                print(f"WARNING: {r['id']} source_note {note!r} not found under {vault}", file=sys.stderr)


def load_existing_results(results_path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not results_path.exists():
        return done
    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                done.add((rec["query_id"], rec["model"]))
    return done


def _heading_slug(heading: str) -> str:
    return heading.lower().replace(" ", "-")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-id", default=None)
    args = parser.parse_args()

    eval_dir = Path("eval")
    cfg = load_config(eval_dir)
    settings = Settings()

    run_dir = eval_dir / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    golden = load_golden_set(eval_dir)
    validate_golden_set(golden, settings.obsidian_dir)

    if args.query_id:
        golden = [q for q in golden if q["id"] == args.query_id]
    if args.limit:
        golden = golden[: args.limit]

    done = load_existing_results(results_path)

    print("Loading embedder…")
    embedder = Embedder()
    store = NoteStore(chroma_dir=settings.chroma_dir, collection_name=settings.collection_name)

    k: int = cfg["k"]
    model_labels: list[str] = cfg["models"]

    clients: dict[str, BedrockClient] = {}
    for label in model_labels:
        model_id = _MODEL_IDS[label]
        clients[label] = BedrockClient(model_id=model_id, region=settings.aws_region)

    total_costs: dict[str, float] = {label: 0.0 for label in model_labels}

    with open(results_path, "a") as out:
        for query_rec in golden:
            qid: str = query_rec["id"]
            print(f"\n[{qid}] {query_rec['query'][:60]}")

            t0 = time.perf_counter()
            embedding = embedder.embed_query(query_rec["query"])
            store_results = store.query(query_embedding=embedding, n_results=k, where=None)
            retrieval_ms = int((time.perf_counter() - t0) * 1000)

            documents: list[str] = store_results["documents"][0]
            metadatas: list[dict] = store_results["metadatas"][0]
            distances: list[float] = store_results.get("distances", [[]])[0]

            retrieved = []
            for rank, (doc, meta, dist) in enumerate(
                zip(documents, metadatas, distances), start=1
            ):
                sp = meta.get("source_path", "")
                heading = meta.get("heading", "")
                retrieved.append({
                    "rank": rank,
                    "chunk_id": f"{sp}#{_heading_slug(heading)}",
                    "source_path": sp,
                    "heading": heading,
                    "session": str(meta.get("session", "")),
                    "distance": round(float(dist), 4),
                    "document": doc,
                })

            context = build_context(documents, metadatas)
            user_message = build_user_message("", context, query_rec["query"])

            for label in model_labels:
                if (qid, label) in done:
                    print(f"  skip {qid}/{label} (already done)")
                    continue

                client = clients[label]
                t1 = time.perf_counter()
                result = client.invoke(SYSTEM_PROMPT, user_message, inference_config=_INFERENCE_CONFIG)
                latency_ms = int((time.perf_counter() - t1) * 1000)

                cost = estimate_cost_usd(client.model_id, result.input_tokens, result.output_tokens)
                total_costs[label] += cost

                record = {
                    "run_id": args.run_id,
                    "query_id": qid,
                    "model": label,
                    "answer": result.text,
                    "retrieved": retrieved,
                    "context_chars": len(context),
                    "usage": {
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    },
                    "latency_ms": latency_ms,
                    "retrieval_ms": retrieval_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "config": {
                        "k": k,
                        "temperature": _INFERENCE_CONFIG["temperature"],
                        "model_id": client.model_id,
                    },
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(f"  {label}: {latency_ms}ms, ${cost:.5f}")

    print("\n--- Total cost ---")
    for label, cost in total_costs.items():
        print(f"  {label}: ${cost:.4f}")


if __name__ == "__main__":
    main()
