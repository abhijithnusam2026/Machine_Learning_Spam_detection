"""Runtime configuration for the serving API."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class ServingConfig:
    vllm_base_url: str
    model_name: str
    request_timeout_seconds: float
    max_summary_tokens: int
    max_intent_tokens: int


def load_config() -> ServingConfig:
    return ServingConfig(
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8001/v1"),
        model_name=os.getenv("SERVED_MODEL_NAME", "call-center-qwen25"),
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60")),
        max_summary_tokens=int(os.getenv("MAX_SUMMARY_TOKENS", "180")),
        max_intent_tokens=int(os.getenv("MAX_INTENT_TOKENS", "16")),
    )
