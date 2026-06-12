#!/usr/bin/env python3
"""Stage 5: join blind scores back to model names.

Usage:
    python -m eval.unmask --run-id 2026-06-15-baseline [--partial]
"""
import argparse
import json
from pathlib import Path


def join_scores(scores: list[dict], mapping: dict[str, dict[str, str]]) -> list[dict]:
    result = []
    for score in scores:
        qid = score["query_id"]
        label = score["label"]
        model = mapping[qid][label]
        result.append({**score, "model": model})
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--partial", action="store_true", help="Allow incomplete scoring")
    args = parser.parse_args()

    eval_dir = Path("eval")
    run_id = args.run_id or _latest_run_id(eval_dir)
    run_dir = eval_dir / "runs" / run_id

    pack_path = run_dir / "blind" / "pack.jsonl"
    scores_path = run_dir / "blind" / "scores.jsonl"
    mapping_path = run_dir / "blind" / "private" / "mapping.json"
    output_path = run_dir / "scored.jsonl"

    pack_keys: set[tuple[str, str]] = set()
    with open(pack_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rec = json.loads(line)
                pack_keys.add((rec["query_id"], rec["label"]))

    scores: list[dict] = []
    with open(scores_path) as f:
        for line in f:
            line = line.strip()
            if line:
                scores.append(json.loads(line))

    scored_keys = {(s["query_id"], s["label"]) for s in scores}
    missing = pack_keys - scored_keys
    if missing and not args.partial:
        print(f"ERROR: {len(missing)} pack entries not yet scored: {sorted(missing)}")
        print("Finish scoring in the Streamlit app, or pass --partial to override.")
        raise SystemExit(1)

    with open(mapping_path) as f:
        mapping = json.load(f)

    joined = join_scores(scores, mapping)
    with open(output_path, "w") as f:
        for rec in joined:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(joined)} scored records to {output_path}")


def _latest_run_id(eval_dir: Path) -> str:
    runs = sorted((eval_dir / "runs").iterdir())
    if not runs:
        raise SystemExit("No runs found. Pass --run-id explicitly.")
    return runs[-1].name


if __name__ == "__main__":
    main()
