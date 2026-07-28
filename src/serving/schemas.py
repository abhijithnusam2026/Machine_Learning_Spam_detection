"""Pydantic schemas for the public serving API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

IntentLabel = Literal[
    "billing_issue",
    "technical_support",
    "cancellation_request",
    "complaint",
    "general_inquiry",
    "unknown",
]


class TranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=20, description="Raw call transcript.")
    call_id: str | None = Field(default=None, description="Optional caller-provided call id.")


class SummarizeResponse(BaseModel):
    call_id: str | None
    summary: str
    model: str
    latency_ms: float

class ScamDetectionResponse(BaseModel):
    call_id: str | None
    is_scam: bool
    confidence: float
    model: str
    latency_ms: float



class IntentResponse(BaseModel):
    call_id: str | None
    intent: IntentLabel
    raw_prediction: str
    model: str
    latency_ms: float


class AnalyzeCallResponse(BaseModel):
    call_id: str | None
    summary: str
    intent: IntentLabel
    raw_intent_prediction: str
    model: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
    vllm_base_url: str


class ReadinessResponse(BaseModel):
    status: Literal["ready"]
    model: str
