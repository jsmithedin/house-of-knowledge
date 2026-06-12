"""Stage 4: Streamlit blind-scoring UI.

Run:
    streamlit run eval/scoring_app.py -- --run-id 2026-06-15-baseline
"""
import argparse
import json
import random
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Eval Scoring", layout="wide")


def _parse_run_id() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args, _ = parser.parse_known_args()
    return args.run_id


def _load_pack(run_dir: Path) -> list[dict]:
    items = []
    with open(run_dir / "blind" / "pack.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_scores(scores_path: Path) -> dict[tuple[str, str], dict]:
    done: dict[tuple[str, str], dict] = {}
    if scores_path.exists():
        with open(scores_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[(rec["query_id"], rec["label"])] = rec
    return done


def _save_score(scores_path: Path, score: dict) -> None:
    with open(scores_path, "a") as f:
        f.write(json.dumps(score) + "\n")


def main() -> None:
    run_id = _parse_run_id()
    run_dir = Path("eval") / "runs" / run_id
    scores_path = run_dir / "blind" / "scores.jsonl"

    pack = _load_pack(run_dir)
    done = _load_scores(scores_path)

    # Stable random order seeded from run_id
    rng = random.Random(run_id)
    order = list(range(len(pack)))
    rng.shuffle(order)
    items = [pack[i] for i in order]

    todo = [it for it in items if (it["query_id"], it["label"]) not in done]

    total = len(pack)
    scored_count = total - len(todo)
    st.title("Blind Answer Scoring")
    st.progress(scored_count / total if total else 1.0, text=f"{scored_count} / {total} scored")

    if not todo:
        st.success("All answers scored. Run `python -m eval.unmask --run-id " + run_id + "` to restore model names.")
        return

    item = todo[0]

    st.markdown(f"**Run:** `{run_id}`  |  **Tier:** `{item['tier']}`  |  **Answer:** `{item['label']}`")
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Question")
        st.write(item["query"])
        st.subheader("Expected answer")
        st.write(item["expected_answer"])
    with col2:
        st.subheader("Answer under review")
        st.write(item["answer"])

    st.divider()
    with st.form("score_form"):
        correctness = st.select_slider(
            "Correctness (0=wrong, 1=partial, 2=correct)",
            options=[0, 1, 2], value=1,
        )
        completeness = st.select_slider(
            "Completeness (0=missing key details, 1=mostly complete, 2=complete)",
            options=[0, 1, 2], value=1,
        )
        hallucination = st.checkbox("Hallucination detected (answer asserts facts not in notes)")
        notes = st.text_area("Notes (optional)")
        submitted = st.form_submit_button("Submit and continue")

    if submitted:
        from datetime import datetime, timezone
        score = {
            "query_id": item["query_id"],
            "label": item["label"],
            "correctness": correctness,
            "completeness": completeness,
            "hallucination": hallucination,
            "notes": notes,
            "scored_at": datetime.now(timezone.utc).isoformat(),
        }
        _save_score(scores_path, score)
        st.rerun()


main()
