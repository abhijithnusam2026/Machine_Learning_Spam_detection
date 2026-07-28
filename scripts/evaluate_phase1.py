"""Evaluate base vs fine-tuned model on summarization and intent classification."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import evaluate
import torch
from peft import PeftModel
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.schema import INTENT_LABELS, TASK_INTENT, TASK_SUMMARIZATION


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_model(model_name: str, adapter_path: str | None) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return tokenizer, model


def build_prompt(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [message for message in messages if message["role"] != "assistant"]


@torch.inference_mode()
def generate(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    input_ids = tokenizer.apply_chat_template(
        build_prompt(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def normalize_intent(prediction: str) -> str:
    text = prediction.strip().lower()
    text = re.sub(r"[^a-z_ ]", " ", text)
    text = text.replace(" ", "_")
    for label in INTENT_LABELS:
        if label in text:
            return label
    return text.split()[0] if text.split() else "unknown"


def evaluate_records(
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    records: list[dict[str, Any]],
    limit: int | None,
) -> dict[str, Any]:
    rouge = evaluate.load("rouge")
    selected = records[:limit] if limit else records

    summary_predictions: list[str] = []
    summary_references: list[str] = []
    intent_predictions: list[str] = []
    intent_references: list[str] = []

    for record in tqdm(selected, desc="Evaluating"):
        max_new_tokens = 160 if record["task"] == TASK_SUMMARIZATION else 16
        prediction = generate(tokenizer, model, record["messages"], max_new_tokens=max_new_tokens)

        if record["task"] == TASK_SUMMARIZATION:
            summary_predictions.append(prediction)
            summary_references.append(record["output"])
        elif record["task"] == TASK_INTENT:
            intent_predictions.append(normalize_intent(prediction))
            intent_references.append(record["output"])

    results: dict[str, Any] = {}
    if summary_predictions:
        rouge_result = rouge.compute(
            predictions=summary_predictions,
            references=summary_references,
            use_stemmer=True,
        )
        results["summarization"] = {"rougeL": rouge_result["rougeL"]}

    if intent_predictions:
        results["intent_classification"] = {
            "accuracy": accuracy_score(intent_references, intent_predictions),
            "macro_f1": f1_score(
                intent_references,
                intent_predictions,
                labels=INTENT_LABELS,
                average="macro",
                zero_division=0,
            ),
            "classification_report": classification_report(
                intent_references,
                intent_predictions,
                labels=INTENT_LABELS,
                zero_division=0,
                output_dict=True,
            ),
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/phase1/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/eval/phase1_results.json"))
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.test_file)
    tokenizer, model = load_model(args.model_name, args.adapter_path)
    results = evaluate_records(tokenizer, model, records, args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
