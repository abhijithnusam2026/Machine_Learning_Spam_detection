"""Async client for vLLM's OpenAI-compatible API."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

import httpx

from src.data.schema import INTENT_LABELS
from src.serving.config import ServingConfig


@dataclass(frozen=True)
class GenerationResult:
    text: str
    latency_ms: float


class VLLMClient:
    def __init__(self, config: ServingConfig) -> None:
        self._config = config
        self._client = httpx.AsyncClient(
            base_url=config.vllm_base_url.rstrip("/"),
            timeout=config.request_timeout_seconds,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def list_models(self) -> dict:
        response = await self._client.get("/models")
        response.raise_for_status()
        return response.json()

    async def generate(
        self,
        messages: list[dict[str, str]],
        max_tokens: int,
        temperature: float = 0.0,
    ) -> GenerationResult:
        start = time.perf_counter()
        response = await self._client.post(
            "/chat/completions",
            json={
                "model": self._config.model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        response.raise_for_status()
        payload = response.json()
        latency_ms = (time.perf_counter() - start) * 1000
        text = payload["choices"][0]["message"]["content"].strip()
        return GenerationResult(text=text, latency_ms=latency_ms)


def normalize_intent(prediction: str) -> str:
    text = prediction.strip().lower()
    text = re.sub(r"[^a-z_ ]", " ", text).replace(" ", "_")
    for label in INTENT_LABELS:
        if label in text:
            return label
    return "unknown"
