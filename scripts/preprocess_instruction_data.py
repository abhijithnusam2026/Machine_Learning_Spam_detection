"""Convert synthetic calls into multitask instruction-tuning records."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.schema import INTENT_LABELS, SYSTEM_PROMPT, TASK_INTENT, TASK_SUMMARIZATION


def summarization_prompt(raw_dialogue: str) -> str:
    return (
        "Summarize the customer support call. Include the customer issue, important context, "
        "the agent action, and the agreed next step.\n\n"
        f"Transcript:\n{raw_dialogue}"
    )


def intent_prompt(raw_dialogue: str) -> str:
    labels = ", ".join(INTENT_LABELS)
    return (
        "Classify the primary intent of the customer support call.\n"
        f"Return exactly one label from this list: {labels}.\n\n"
        f"Transcript:\n{raw_dialogue}"
    )


def build_instruction_records(call: dict[str, Any]) -> list[dict[str, Any]]:
    call_id = call["id"]
    raw_dialogue = call["raw_dialogue"].strip()
    summary = call["summary"].strip()
    intent = call["intent"].strip()
    return [
        {
            "id": f"{call_id}__summary",
            "source_call_id": call_id,
            "task": TASK_SUMMARIZATION,
            "intent": intent,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": summarization_prompt(raw_dialogue)},
                {"role": "assistant", "content": summary},
            ],
            "input": raw_dialogue,
            "output": summary,
        },
        {
            "id": f"{call_id}__intent",
            "source_call_id": call_id,
            "task": TASK_INTENT,
            "intent": intent,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": intent_prompt(raw_dialogue)},
                {"role": "assistant", "content": intent},
            ],
            "input": raw_dialogue,
            "output": intent,
        },
    ]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def split_by_call_id(
    calls: list[dict[str, Any]],
    train_ratio: float,
    eval_ratio: float,
    seed: int,
) -> dict[str, list[dict[str, Any]]]:
    rng = random.Random(seed)
    shuffled = calls[:]
    rng.shuffle(shuffled)

    train_end = int(len(shuffled) * train_ratio)
    eval_end = train_end + int(len(shuffled) * eval_ratio)
    return {
        "train": shuffled[:train_end],
        "validation": shuffled[train_end:eval_end],
        "test": shuffled[eval_end:],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/synthetic/calls.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/phase1"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=13)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    calls = load_jsonl(args.input)
    splits = split_by_call_id(
        calls=calls,
        train_ratio=args.train_ratio,
        eval_ratio=args.eval_ratio,
        seed=args.seed,
    )

    for split_name, split_calls in splits.items():
        records = [
            instruction_record
            for call in split_calls
            for instruction_record in build_instruction_records(call)
        ]
        write_jsonl(args.output_dir / f"{split_name}.jsonl", records)

    write_jsonl(args.output_dir / "calls_all.jsonl", calls)


if __name__ == "__main__":
    main()
