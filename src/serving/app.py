"""FastAPI app for call summarization and intent classification."""

from __future__ import annotations

import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import httpx
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.monitoring.prometheus import instrument_app
from src.serving.config import load_config
from src.serving.prompts import classify_intent_messages, summarize_messages
from src.serving.schemas import (
    AnalyzeCallResponse,
    HealthResponse,
    IntentResponse,
    ReadinessResponse,
    SummarizeResponse,
    TranscriptRequest,
)
from src.serving.vllm_client import VLLMClient, normalize_intent


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = load_config()
    app.state.config = config
    app.state.vllm_client = VLLMClient(config)
    try:
        yield
    finally:
        await app.state.vllm_client.close()


app = FastAPI(
    title="Call Center Intelligence API",
    version="0.1.0",
    description="Async API gateway over vLLM for call summarization and intent classification.",
    lifespan=lifespan,
)
instrument_app(app)


def as_http_exception(error: httpx.HTTPStatusError) -> HTTPException:
    detail = {
        "message": "vLLM request failed",
        "status_code": error.response.status_code,
        "body": error.response.text,
    }
    return HTTPException(status_code=502, detail=detail)


def as_upstream_unavailable(error: httpx.RequestError) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={
            "message": "vLLM server is unavailable",
            "error": str(error),
        },
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    config = app.state.config
    return HealthResponse(
        status="ok",
        model=config.model_name,
        vllm_base_url=config.vllm_base_url,
    )


@app.get("/ready", response_model=ReadinessResponse)
async def ready() -> ReadinessResponse:
    config = app.state.config
    try:
        await app.state.vllm_client.list_models()
    except httpx.HTTPStatusError as error:
        raise as_http_exception(error) from error
    except httpx.RequestError as error:
        raise as_upstream_unavailable(error) from error
    return ReadinessResponse(status="ready", model=config.model_name)


@app.post("/summarize", response_model=SummarizeResponse)
async def summarize(request: TranscriptRequest) -> SummarizeResponse:
    config = app.state.config
    try:
        result = await app.state.vllm_client.generate(
            messages=summarize_messages(request.transcript),
            max_tokens=config.max_summary_tokens,
        )
    except httpx.HTTPStatusError as error:
        raise as_http_exception(error) from error
    except httpx.RequestError as error:
        raise as_upstream_unavailable(error) from error

    return SummarizeResponse(
        call_id=request.call_id,
        summary=result.text,
        model=config.model_name,
        latency_ms=result.latency_ms,
    )


@app.post("/classify-intent", response_model=IntentResponse)
async def classify_intent(request: TranscriptRequest) -> IntentResponse:
    config = app.state.config
    try:
        result = await app.state.vllm_client.generate(
            messages=classify_intent_messages(request.transcript),
            max_tokens=config.max_intent_tokens,
        )
    except httpx.HTTPStatusError as error:
        raise as_http_exception(error) from error
    except httpx.RequestError as error:
        raise as_upstream_unavailable(error) from error

    return IntentResponse(
        call_id=request.call_id,
        intent=normalize_intent(result.text),
        raw_prediction=result.text,
        model=config.model_name,
        latency_ms=result.latency_ms,
    )


@app.post("/analyze-call", response_model=AnalyzeCallResponse)
async def analyze_call(request: TranscriptRequest) -> AnalyzeCallResponse:
    config = app.state.config
    start = time.perf_counter()
    try:
        summary_result, intent_result = await asyncio.gather(
            app.state.vllm_client.generate(
                messages=summarize_messages(request.transcript),
                max_tokens=config.max_summary_tokens,
            ),
            app.state.vllm_client.generate(
                messages=classify_intent_messages(request.transcript),
                max_tokens=config.max_intent_tokens,
            ),
        )
    except httpx.HTTPStatusError as error:
        raise as_http_exception(error) from error
    except httpx.RequestError as error:
        raise as_upstream_unavailable(error) from error

    return AnalyzeCallResponse(
        call_id=request.call_id,
        summary=summary_result.text,
        intent=normalize_intent(intent_result.text),
        raw_intent_prediction=intent_result.text,
        model=config.model_name,
        latency_ms=(time.perf_counter() - start) * 1000,
    )
