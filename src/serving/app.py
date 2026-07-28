"""FastAPI app for DistilBERT Scam Detection."""

from __future__ import annotations

import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parents[2]))

from src.serving.schemas import (
    HealthResponse,
    ScamDetectionResponse,
    TranscriptRequest,
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load DistilBERT model for scam detection if available
    app.state.scam_model = None
    app.state.scam_tokenizer = None
    app.state.device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model_dir = "scam-classifier-model"
        app.state.scam_tokenizer = AutoTokenizer.from_pretrained(model_dir)
        app.state.scam_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        app.state.scam_model.to(app.state.device)
        app.state.scam_model.eval()
        print("Scam detection model loaded successfully.")
    except Exception as e:
        print(f"CRITICAL ERROR: Failed to load DistilBERT scam model from '{model_dir}'.")
        print(f"Ensure you have trained the model using scripts/train_scam_classifier.py")
        # Raise the exception so the container crashes loudly instead of silently failing later.
        raise RuntimeError(f"Model load failure: {e}") from e

    yield


app = FastAPI(
    title="Scam Detection API",
    version="0.1.0",
    description="FastAPI gateway for DistilBERT Scam classification.",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model="distilbert-scam-classifier",
    )


@app.post("/detect-scam", response_model=ScamDetectionResponse)
async def detect_scam(request: TranscriptRequest) -> ScamDetectionResponse:
    if app.state.scam_model is None:
        raise HTTPException(status_code=503, detail="Scam detection model not loaded.")
    
    start_time = time.perf_counter()
    inputs = app.state.scam_tokenizer(
        request.transcript, return_tensors="pt", truncation=True, max_length=256
    ).to(app.state.device)
    
    with torch.no_grad():
        logits = app.state.scam_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
    
    is_scam = bool(probs[1] > probs[0])
    confidence = float(probs.max().item())
    latency_ms = (time.perf_counter() - start_time) * 1000
    
    return ScamDetectionResponse(
        call_id=request.call_id,
        is_scam=is_scam,
        confidence=confidence,
        model="distilbert-scam-classifier",
        latency_ms=latency_ms,
    )
