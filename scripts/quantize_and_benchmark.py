"""Benchmark FP16 and INT8 bitsandbytes loading for a fine-tuned model.

This script compares quality and serving-facing metrics across precision modes:
- fp32/fp16 PyTorch loading, depending on device and --precisions.
- int8 bitsandbytes loading with LLM.int8.

It reports perplexity on assistant response tokens and intent accuracy/F1 using the
Phase 1 instruction dataset. Results are written as JSON so they can be copied into
docs/benchmarks.md after a real hardware run.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.schema import INTENT_LABELS, TASK_INTENT
from src.optimization.metrics import benchmark_generation, clear_memory, memory_footprint_mb


def import_runtime_dependencies() -> None:
    global AutoModelForCausalLM
    global AutoTokenizer
    global BitsAndBytesConfig
    global PeftModel
    global accuracy_score
    global f1_score
    global tqdm

    from peft import PeftModel
    from sklearn.metrics import accuracy_score, f1_score
    from tqdm import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [message for message in messages if message["role"] != "assistant"]


def load_tokenizer(model_name: str, adapter_path: str | None) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(adapter_path or model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(model_name: str, adapter_path: str | None, precision: str) -> AutoModelForCausalLM:
    quantization_config = None
    dtype = torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    if precision == "fp16":
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif precision == "bf16":
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    elif precision == "int8":
        if not torch.cuda.is_available():
            raise RuntimeError("bitsandbytes INT8 inference requires a CUDA GPU.")
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map=device_map,
        quantization_config=quantization_config,
        attn_implementation="sdpa",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


def tokenized_with_labels(
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_length: int,
) -> dict[str, torch.Tensor]:
    prompt_ids = tokenizer.apply_chat_template(
        prompt_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
    )
    full_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )[:max_length]
    prompt_length = min(len(prompt_ids), len(full_ids))
    labels = [-100] * prompt_length + full_ids[prompt_length:]
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones(1, len(full_ids), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


@torch.inference_mode()
def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, Any]],
    max_length: int,
) -> float:
    losses: list[float] = []
    for record in tqdm(records, desc="Perplexity"):
        batch = tokenized_with_labels(tokenizer, record["messages"], max_length)
        batch = {key: value.to(model.device) for key, value in batch.items()}
        output = model(**batch)
        losses.append(float(output.loss.detach().cpu()))
    mean_loss = sum(losses) / len(losses)
    return math.exp(mean_loss)


def normalize_intent(prediction: str) -> str:
    text = prediction.strip().lower()
    text = re.sub(r"[^a-z_ ]", " ", text).replace(" ", "_")
    for label in INTENT_LABELS:
        if label in text:
            return label
    return "unknown"


@torch.inference_mode()
def generate_text(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> tuple[str, int]:
    input_ids = tokenizer.apply_chat_template(
        prompt_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_ids = output_ids[0, input_ids.shape[-1] :]
    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip(), len(generated_ids)


def compute_intent_metrics(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: list[dict[str, Any]],
) -> dict[str, float]:
    intent_records = [record for record in records if record["task"] == TASK_INTENT]
    predictions: list[str] = []
    references: list[str] = []
    for record in tqdm(intent_records, desc="Intent accuracy"):
        prediction, _ = generate_text(model, tokenizer, record["messages"], max_new_tokens=16)
        predictions.append(normalize_intent(prediction))
        references.append(record["output"])
    return {
        "accuracy": accuracy_score(references, predictions),
        "macro_f1": f1_score(
            references,
            predictions,
            labels=INTENT_LABELS,
            average="macro",
            zero_division=0,
        ),
    }


def benchmark_latency(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: dict[str, Any],
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, float]:
    def generate_once() -> int:
        _, generated_tokens = generate_text(model, tokenizer, record["messages"], max_new_tokens=64)
        return generated_tokens

    latency = benchmark_generation(generate_once, warmup_runs, benchmark_runs)
    return {
        "total_seconds": latency.total_seconds,
        "generated_tokens": latency.generated_tokens,
        "milliseconds_per_token": latency.milliseconds_per_token,
        "tokens_per_second": latency.tokens_per_second,
    }


def benchmark_precision(args: argparse.Namespace, precision: str, records: list[dict[str, Any]]) -> dict[str, Any]:
    clear_memory()
    tokenizer = load_tokenizer(args.model_name, args.adapter_path)
    model = load_model(args.model_name, args.adapter_path, precision)

    selected = records[: args.limit] if args.limit else records
    perplexity = compute_perplexity(model, tokenizer, selected, args.max_length)
    intent_metrics = compute_intent_metrics(model, tokenizer, selected)
    latency = benchmark_latency(
        model=model,
        tokenizer=tokenizer,
        record=selected[0],
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    memory = memory_footprint_mb(model)

    del model
    clear_memory()
    return {
        "precision": precision,
        "perplexity": perplexity,
        "intent_accuracy": intent_metrics["accuracy"],
        "intent_macro_f1": intent_metrics["macro_f1"],
        "latency": latency,
        "memory": memory,
    }


def add_degradation(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not results:
        return results
    baseline = results[0]
    for result in results:
        result["degradation_vs_baseline"] = {
            "perplexity_delta": result["perplexity"] - baseline["perplexity"],
            "intent_accuracy_delta": result["intent_accuracy"] - baseline["intent_accuracy"],
            "intent_macro_f1_delta": result["intent_macro_f1"] - baseline["intent_macro_f1"],
        }
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/phase1/test.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/quantization.json"))
    parser.add_argument("--precisions", nargs="+", default=["fp16", "int8"], choices=["fp32", "fp16", "bf16", "int8"])
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_runtime_dependencies()
    records = load_jsonl(args.test_file)
    results = []
    for precision in args.precisions:
        results.append(benchmark_precision(args, precision, records))
    payload = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "test_file": str(args.test_file),
        "results": add_degradation(results),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
