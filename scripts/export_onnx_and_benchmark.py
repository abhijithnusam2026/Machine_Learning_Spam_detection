"""Export a fine-tuned model to ONNX and benchmark PyTorch vs ONNX Runtime.

For PEFT adapters, the script first merges the adapter into the base model in a
temporary directory, then uses Optimum to export an ONNX Runtime causal LM.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Protocol

import torch

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.optimization.metrics import benchmark_generation, clear_memory, memory_footprint_mb


class GenerativeModel(Protocol):
    def generate(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        ...


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def prompt_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [message for message in messages if message["role"] != "assistant"]


def load_tokenizer(model_source: str) -> AutoTokenizer:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def merge_adapter_to_temp_model(model_name: str, adapter_path: str, dtype: torch.dtype) -> Path:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    temp_dir = Path(tempfile.mkdtemp(prefix="merged-peft-"))
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    merged = model.merge_and_unload()
    merged.save_pretrained(temp_dir, safe_serialization=True)
    tokenizer = load_tokenizer(adapter_path)
    tokenizer.save_pretrained(temp_dir)
    del model
    del merged
    clear_memory()
    return temp_dir


def export_onnx(model_source: str, output_dir: Path, provider: str) -> ORTModelForCausalLM:
    from optimum.onnxruntime import ORTModelForCausalLM

    output_dir.mkdir(parents=True, exist_ok=True)
    model = ORTModelForCausalLM.from_pretrained(
        model_source,
        export=True,
        provider=provider,
    )
    model.save_pretrained(output_dir)
    tokenizer = load_tokenizer(model_source)
    tokenizer.save_pretrained(output_dir)
    return model


def load_pytorch_model(model_name: str, adapter_path: str | None, dtype: torch.dtype) -> AutoModelForCausalLM:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model


@torch.inference_mode()
def generated_token_count(
    model: GenerativeModel,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> int:
    input_ids = tokenizer.apply_chat_template(
        prompt_messages(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    )
    model_device = getattr(model, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    input_ids = input_ids.to(model_device)
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return int(output_ids.shape[-1] - input_ids.shape[-1])


def run_latency_benchmark(
    model: GenerativeModel,
    tokenizer: AutoTokenizer,
    record: dict[str, Any],
    max_new_tokens: int,
    warmup_runs: int,
    benchmark_runs: int,
) -> dict[str, float]:
    def generate_once() -> int:
        return generated_token_count(model, tokenizer, record["messages"], max_new_tokens)

    result = benchmark_generation(generate_once, warmup_runs, benchmark_runs)
    return {
        "total_seconds": result.total_seconds,
        "generated_tokens": result.generated_tokens,
        "milliseconds_per_token": result.milliseconds_per_token,
        "tokens_per_second": result.tokens_per_second,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--test-file", type=Path, default=Path("data/processed/phase1/test.jsonl"))
    parser.add_argument("--onnx-output-dir", type=Path, default=Path("artifacts/onnx/qwen25_phase1"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/onnx_runtime.json"))
    parser.add_argument("--provider", default=None, help="CPUExecutionProvider or CUDAExecutionProvider.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--benchmark-runs", type=int, default=5)
    parser.add_argument("--keep-merged-temp", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    provider = args.provider or ("CUDAExecutionProvider" if torch.cuda.is_available() else "CPUExecutionProvider")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    records = load_jsonl(args.test_file)
    benchmark_record = records[0]

    merged_temp_dir: Path | None = None
    model_source = args.model_name
    if args.adapter_path:
        merged_temp_dir = merge_adapter_to_temp_model(args.model_name, args.adapter_path, dtype)
        model_source = str(merged_temp_dir)

    tokenizer = load_tokenizer(model_source)

    clear_memory()
    pytorch_model = load_pytorch_model(args.model_name, args.adapter_path, dtype)
    pytorch_latency = run_latency_benchmark(
        model=pytorch_model,
        tokenizer=tokenizer,
        record=benchmark_record,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    pytorch_memory = memory_footprint_mb(pytorch_model)
    del pytorch_model
    clear_memory()

    onnx_model = export_onnx(model_source, args.onnx_output_dir, provider)
    onnx_latency = run_latency_benchmark(
        model=onnx_model,
        tokenizer=tokenizer,
        record=benchmark_record,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
    )
    onnx_memory = memory_footprint_mb(None)

    payload = {
        "model_name": args.model_name,
        "adapter_path": args.adapter_path,
        "onnx_output_dir": str(args.onnx_output_dir),
        "provider": provider,
        "pytorch": {
            "latency": pytorch_latency,
            "memory": pytorch_memory,
        },
        "onnx_runtime": {
            "latency": onnx_latency,
            "memory": onnx_memory,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    print(json.dumps(payload, indent=2))

    if merged_temp_dir and not args.keep_merged_temp:
        shutil.rmtree(merged_temp_dir)


if __name__ == "__main__":
    main()
