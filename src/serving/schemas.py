"""Pydantic schemas for the DistilBERT Scam Detection API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

class TranscriptRequest(BaseModel):
    transcript: str = Field(..., min_length=20, description="Raw call transcript.")
    call_id: str | None = Field(default=None, description="Optional caller-provided call id.")

class ScamDetectionResponse(BaseModel):
    call_id: str | None
    is_scam: bool
    confidence: float
    model: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: Literal["ok"]
    model: str
