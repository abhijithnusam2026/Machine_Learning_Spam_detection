"""Shared benchmarking helpers."""

from __future__ import annotations

import gc
import os
import resource
import time
from dataclasses import dataclass
from typing import Callable

import torch


@dataclass(frozen=True)
class LatencyResult:
    total_seconds: float
    generated_tokens: int
    milliseconds_per_token: float
    tokens_per_second: float


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def memory_footprint_mb(model: torch.nn.Module | None = None) -> dict[str, float]:
    result = {
        "process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "cuda_allocated_mb": 0.0,
        "cuda_reserved_mb": 0.0,
        "cuda_peak_allocated_mb": 0.0,
        "model_parameters_mb": 0.0,
    }
    if os.uname().sysname == "Darwin":
        result["process_rss_mb"] = result["process_rss_mb"] / 1024
    if torch.cuda.is_available():
        result["cuda_allocated_mb"] = torch.cuda.memory_allocated() / 1024**2
        result["cuda_reserved_mb"] = torch.cuda.memory_reserved() / 1024**2
        result["cuda_peak_allocated_mb"] = torch.cuda.max_memory_allocated() / 1024**2
    if model is not None:
        parameter_bytes = 0
        for parameter in model.parameters():
            parameter_bytes += parameter.numel() * parameter.element_size()
        result["model_parameters_mb"] = parameter_bytes / 1024**2
    return result


def benchmark_generation(
    generate_once: Callable[[], int],
    warmup_runs: int,
    benchmark_runs: int,
) -> LatencyResult:
    for _ in range(warmup_runs):
        generate_once()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    start = time.perf_counter()
    generated_tokens = 0
    for _ in range(benchmark_runs):
        generated_tokens += generate_once()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_seconds = time.perf_counter() - start

    tokens_per_second = generated_tokens / total_seconds if total_seconds else 0.0
    milliseconds_per_token = (
        (total_seconds * 1000) / generated_tokens if generated_tokens else float("inf")
    )
    return LatencyResult(
        total_seconds=total_seconds,
        generated_tokens=generated_tokens,
        milliseconds_per_token=milliseconds_per_token,
        tokens_per_second=tokens_per_second,
    )
