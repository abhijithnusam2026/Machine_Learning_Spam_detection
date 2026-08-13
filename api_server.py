import os
import json
import time
import asyncio
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi.staticfiles import StaticFiles

# Force backend to fp16 for the API server
config_path = "configs/inference_config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)
    if config.get("backend") != "fp16":
        config["backend"] = "fp16"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=4)

# MLflow DagsHub Authentication
from dotenv import load_dotenv
import mlflow
load_dotenv()
owner = os.getenv("DAGSHUB_REPO_OWNER")
name = os.getenv("DAGSHUB_REPO_NAME")
token = os.getenv("MLFLOW_TRACKING_PASSWORD")

if owner and name and token:
    import dagshub
    dagshub.auth.add_app_token(token)
    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(f"https://dagshub.com/{owner}/{name}.mlflow")

import importlib
infer_module = importlib.import_module("scripts.10_inference_pipeline")
pipeline = infer_module.InferencePipeline()

app = FastAPI(title="Scam Detection API", description="Live Voice Scam Detection", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/uploads", exist_ok=True)

@app.post("/detect")
async def detect_scam(file: UploadFile = File(...)):
    """Standard blocking endpoint. Returns everything at once."""
    file_path = f"data/uploads/{int(time.time())}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        res = pipeline.process_audio(file_path)
        return res
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/detect_stream")
async def detect_stream(file: UploadFile = File(...)):
    """Streaming endpoint (SSE) to update the UI step-by-step."""
    file_path = f"data/uploads/{int(time.time())}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())

    async def event_generator():
        try:
            # 1. Start Transcription
            yield {"event": "status", "data": json.dumps({"step": "transcribing", "message": "Converting Audio to Text..."})}
            await asyncio.sleep(0.1) # Yield control
            
            t0 = time.time()
            transcript = pipeline._transcribe(file_path)
            asr_latency = time.time() - t0
            
            # 2. Transcription Finished
            yield {"event": "transcript", "data": json.dumps({"text": transcript, "latency": f"{asr_latency:.3f}s"})}
            
            # 3. Start Classification
            yield {"event": "status", "data": json.dumps({"step": "classifying", "message": "Analyzing Intent via ModernBERT..."})}
            await asyncio.sleep(0.1)
            
            t1 = time.time()
            prediction = pipeline._classify(transcript)
            clf_latency = time.time() - t1
            
            # 4. Final Result
            yield {"event": "result", "data": json.dumps({
                "prediction": prediction,
                "asr_latency": asr_latency,
                "clf_latency": clf_latency,
                "total_latency": asr_latency + clf_latency
            })}
        except Exception as e:
            yield {"event": "error", "data": str(e)}
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)

    return EventSourceResponse(event_generator())

os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    print("Starting Scam Detection Server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
