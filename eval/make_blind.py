#!/usr/bin/env python3
"""Stage 3: anonymise answers for blind scoring.

Usage:
    python -m eval.make_blind --run-id 2026-06-15-baseline
"""
import argparse
import hashlib
import json
import random
import re
from pathlib import Path

# Nova Lite's agentic-mode answers sometimes include a raw <thinking>...</thinking>
# block as literal text in the final response (Haiku 4.5 never does this, in either
# mode) — an artifact of Nova Lite's tool-use output, not something app/bedrock.py
# currently strips. Left in, it's an instant tell for blind scoring: the moment a
# scorer spots a thinking block, they know which answer is Nova Lite. Strip it before
# handing answers to the blind pack. No-op for every other answer.
_THINKING_BLOCK = re.compile(r"<thinking>.*?</thinking>\s*", re.DOTALL)


def _strip_thinking(answer: str) -> str:
    return _THINKING_BLOCK.sub("", answer).strip()


def label_assignments(run_id: str, query_ids: list[str], models: list[str]) -> dict[str, dict[str, str]]:
    """Return {query_id: {"A": model_label, "B": model_label}}, seeded from run_id."""
    result: dict[str, dict[str, str]] = {}
    for qid in query_ids:
        seed_str = f"{run_id}:{qid}"
        seed = int(hashlib.md5(seed_str.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        shuffled = list(models)
        rng.shuffle(shuffled)
        result[qid] = {"A": shuffled[0], "B": shuffled[1]}
    return result


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

    # Load results: one record per (query_id, model)
    results_by_query: dict[str, dict[str, dict]] = {}
    models_seen: list[str] = []
    with open(run_dir / "results.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            qid, model = rec["query_id"], rec["model"]
            results_by_query.setdefault(qid, {})[model] = rec
            if model not in models_seen:
                models_seen.append(model)

    query_ids = list(results_by_query.keys())
    assignments = label_assignments(run_id, query_ids, models_seen)

    blind_dir = run_dir / "blind"
    blind_dir.mkdir(exist_ok=True)
    private_dir = blind_dir / "private"
    private_dir.mkdir(exist_ok=True)

    pack: list[dict] = []
    for qid in query_ids:
        golden = golden_by_id.get(qid, {})
        for label in ("A", "B"):
            model = assignments[qid][label]
            answer = _strip_thinking(results_by_query[qid][model]["answer"])
            pack.append({
                "query_id": qid,
                "label": label,
                "tier": golden.get("tier", ""),
                "query": golden.get("query", ""),
                "expected_answer": golden.get("expected_answer", ""),
                "answer": answer,
            })

    pack_path = blind_dir / "pack.jsonl"
    with open(pack_path, "w") as f:
        for item in pack:
            f.write(json.dumps(item) + "\n")

    mapping_path = private_dir / "mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(assignments, f, indent=2)

    print(f"Wrote {len(pack)} pack entries to {pack_path}")
    print(f"Mapping written to {mapping_path} (keep this private until unmask)")


def _latest_run_id(eval_dir: Path) -> str:
    runs = sorted((eval_dir / "runs").iterdir())
    if not runs:
        raise SystemExit("No runs found. Pass --run-id explicitly.")
    return runs[-1].name


if __name__ == "__main__":
    main()
