#!/usr/bin/env python3
"""Stage 1 (agentic variant): batch agentic-RAG runs against the golden set.

Same golden set and models as eval/runner.py, but exercises the agentic path
(app.rag.AGENTIC_SYSTEM_PROMPT + SEARCH_TOOL via BedrockClient.invoke_with_tools)
instead of a single retrieve-then-generate call. The model decides what to
search and how many times.

Output schema matches eval/runner.py's results.jsonl (query_id, model, answer,
retrieved, usage, latency_ms, config) so score_retrieval.py, make_blind.py,
judge.py, and report.py all work unmodified against an agentic run-id — just
point them at this run's --run-id. Two agentic-only fields are added:
tool_calls (what the model searched for, in order) and trace (the full
per-iteration model output, for pulling a worked reasoning-trace example).

Usage:
    python -m eval.agentic_runner --run-id 2026-07-20-agentic [--limit N] [--query-id q01]
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

from app.bedrock import BedrockClient, BedrockError
from app.config import NOVA_LITE_MODEL_ID, HAIKU_MODEL_ID, Settings, estimate_cost_usd
from app.embedder import Embedder
from app.rag import AGENTIC_SYSTEM_PROMPT, SEARCH_TOOL
from app.store import NoteStore

_MODEL_IDS: dict[str, str] = {
    "nova-lite": NOVA_LITE_MODEL_ID,
    "haiku-4.5": HAIKU_MODEL_ID,
}

# temperature=0.0 to match eval/runner.py's standard-mode runs — keeps Post 8
# comparable to Post 6 rather than introducing a second variable.
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


def _make_tool_executor(embedder, store, default_k: int, tool_calls: list[dict], retrieved_flat: list[dict]):
    """Same retrieve-and-cap-k behaviour as RagPipeline._query_agentic's executor,
    but logs every call (query/k/latency) and flattens retrieved chunks with a
    running global rank, so score_retrieval.py can score recall the same way
    it does for the standard runner's single retrieval pass."""

    def execute(name: str, tool_input: dict) -> str:
        if name != "search_knowledge_base":
            return "Unknown tool."
        query = tool_input["query"]
        n = min(max(int(tool_input.get("k", default_k)), 1), 10)

        t0 = time.perf_counter()
        embedding = embedder.embed_query(query)
        results = store.query(query_embedding=embedding, n_results=n, where=None)
        call_latency_ms = int((time.perf_counter() - t0) * 1000)

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0]

        call_index = len(tool_calls) + 1
        tool_calls.append({
            "call_index": call_index,
            "query": query,
            "k_requested": tool_input.get("k"),
            "k_used": n,
            "doc_count": len(documents),
            "latency_ms": call_latency_ms,
        })

        for doc, meta, dist in zip(documents, metadatas, distances):
            sp = meta.get("source_path", "")
            heading = meta.get("heading", "")
            retrieved_flat.append({
                "rank": len(retrieved_flat) + 1,
                "tool_call": call_index,
                "chunk_id": f"{sp}#{_heading_slug(heading)}",
                "source_path": sp,
                "heading": heading,
                "session": str(meta.get("session", "")),
                "distance": round(float(dist), 4),
                "document": doc,
            })

        if not documents:
            return "No results found for that query."

        parts = []
        for doc, meta in zip(documents, metadatas):
            parts.append(
                f"[Session {meta.get('session', '?')} — {meta.get('heading', '?')} "
                f"({meta.get('date', '?')})]\n{doc}"
            )
        return "\n\n---\n\n".join(parts)

    return execute


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--query-id", default=None)
    parser.add_argument("--max-iterations", type=int, default=5)
    args = parser.parse_args()

    eval_dir = Path("eval")
    cfg = load_config(eval_dir)
    settings = Settings()

    run_dir = eval_dir / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    results_path = run_dir / "results.jsonl"

    golden = load_golden_set(eval_dir)

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

            for label in model_labels:
                if (qid, label) in done:
                    print(f"  skip {qid}/{label} (already done)")
                    continue

                client = clients[label]
                tool_calls: list[dict] = []
                retrieved_flat: list[dict] = []
                trace: list[dict] = []

                tool_executor = _make_tool_executor(embedder, store, k, tool_calls, retrieved_flat)

                def on_iteration(iteration: int, stop_reason: str, output_message: dict) -> None:
                    trace.append({
                        "iteration": iteration,
                        "stop_reason": stop_reason,
                        "content": output_message["content"],
                    })

                initial_messages = [
                    {"role": "user", "content": [{"text": f"Question: {query_rec['query']}"}]},
                ]

                base_config = {
                    "default_k": k,
                    "max_iterations": args.max_iterations,
                    "temperature": _INFERENCE_CONFIG["temperature"],
                    "model_id": client.model_id,
                }

                t1 = time.perf_counter()
                try:
                    result = client.invoke_with_tools(
                        system_prompt=AGENTIC_SYSTEM_PROMPT,
                        initial_messages=initial_messages,
                        tools=[SEARCH_TOOL],
                        tool_executor=tool_executor,
                        max_iterations=args.max_iterations,
                        on_iteration=on_iteration,
                        inference_config=_INFERENCE_CONFIG,
                    )
                except BedrockError as e:
                    # Most likely cause: max_iterations reached without end_turn — the
                    # model kept searching and never converged on an answer. That's a
                    # real result for this eval (a failure mode standard RAG can't even
                    # have), not a run-breaking bug — record it and move on, same as the
                    # live app's RagPipeline._query_agentic falling back on BedrockError
                    # instead of crashing.
                    latency_ms = int((time.perf_counter() - t1) * 1000)
                    record = {
                        "run_id": args.run_id,
                        "query_id": qid,
                        "model": label,
                        "answer": f"[NO ANSWER — {e}]",
                        "error": str(e),
                        "retrieved": retrieved_flat,
                        "tool_calls": tool_calls,
                        "iterations": len(trace),
                        "trace": trace,
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                        "latency_ms": latency_ms,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "config": base_config,
                    }
                    out.write(json.dumps(record) + "\n")
                    out.flush()
                    print(
                        f"  {label}: FAILED after {len(trace)} iteration(s), "
                        f"{len(tool_calls)} search call(s) — {e}"
                    )
                    continue
                latency_ms = int((time.perf_counter() - t1) * 1000)

                cost = estimate_cost_usd(client.model_id, result.input_tokens, result.output_tokens)
                total_costs[label] += cost

                record = {
                    "run_id": args.run_id,
                    "query_id": qid,
                    "model": label,
                    "answer": result.text,
                    "error": None,
                    "retrieved": retrieved_flat,
                    "tool_calls": tool_calls,
                    "iterations": len(trace),
                    "trace": trace,
                    "usage": {
                        "input_tokens": result.input_tokens,
                        "output_tokens": result.output_tokens,
                    },
                    "latency_ms": latency_ms,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "config": base_config,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()
                print(
                    f"  {label}: {latency_ms}ms, {len(tool_calls)} search call(s), "
                    f"{len(trace)} iteration(s), ${cost:.5f}"
                )

    print("\n--- Total cost ---")
    for label, cost in total_costs.items():
        print(f"  {label}: ${cost:.4f}")


if __name__ == "__main__":
    main()
