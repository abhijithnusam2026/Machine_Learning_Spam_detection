import os
import json
import time
import asyncio
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi.staticfiles import StaticFiles

# We rely on configs/inference_config.json for the backend selection.
config_path = "configs/inference_config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

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
infer_module = importlib.import_module("src.evaluation.inference_pipeline")
pipeline = infer_module.InferencePipeline()

app = FastAPI(title="Scam Detection API", description="Live Voice Scam Detection", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("data/uploads", exist_ok=True)

LIVE_WINDOW_SEC = 3.0

def _write_file(path: str, contents: bytes):
    with open(path, "wb") as f:
        f.write(contents)

def _display_prediction(prediction):
    """pipeline._classify() returns raw 0/1 (or an already-worded string for
    the empty-audio case) -- normalize to the label string clients render."""
    if prediction == 1:
        return "Scam"
    if prediction == 0:
        return "Legitimate"
    return prediction

@app.websocket("/ws/live")
async def live_ws(websocket: WebSocket):
    """Live sliding-window monitor over a persistent connection.

    Repeated short-lived POSTs (one per audio window) pay a new connection
    cost per window and are vulnerable to proxy idle/timeout limits under
    load. A single long-lived WebSocket avoids both -- the browser mic sends
    each ~LIVE_WINDOW_SEC window as a binary frame and gets a JSON result
    frame back. The running transcript lives in this connection's own scope,
    so there's no session_id/dict bookkeeping to leak across restarts.
    """
    await websocket.accept()
    transcript = ""
    chunk_idx = 0
    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break

            raw_text = message.get("text")
            if raw_text is not None:
                try:
                    payload = json.loads(raw_text)
                except ValueError:
                    payload = {}
                if payload.get("type") == "reset":
                    transcript = ""
                    await websocket.send_json({"type": "reset_ack"})
                continue

            data = message.get("bytes")
            if not data:
                continue

            chunk_idx += 1
            file_path = f"data/uploads/live_ws_{id(websocket)}_{chunk_idx}_{int(time.time() * 1000)}.webm"
            await asyncio.to_thread(_write_file, file_path, data)

            try:
                t0 = time.time()
                # A silent window is a normal pause in the call, not an
                # empty upload -- don't let it blow up the whole stream.
                chunk_text = (await asyncio.to_thread(pipeline._transcribe, file_path, allow_empty=True)).strip()
                asr_latency = time.time() - t0

                if chunk_text:
                    transcript = (transcript + " " + chunk_text).strip()

                t1 = time.time()
                if transcript:
                    prediction = await asyncio.to_thread(pipeline._classify, transcript)
                else:
                    prediction = "Legitimate (No speech yet)"
                clf_latency = time.time() - t1

                await websocket.send_json({
                    "type": "result",
                    "transcript": transcript,
                    "chunk_text": chunk_text,
                    "prediction": _display_prediction(prediction),
                    "asr_latency": asr_latency,
                    "clf_latency": clf_latency,
                })
            except Exception as e:
                await websocket.send_json({"type": "error", "error": str(e)})
            finally:
                if os.path.exists(file_path):
                    os.remove(file_path)
    except WebSocketDisconnect:
        pass

@app.post("/detect")
async def detect_scam(file: UploadFile = File(...)):
    """Standard blocking endpoint. Returns everything at once."""
    file_path = f"data/uploads/{int(time.time())}_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        # Run blocking CPU/GPU task in a thread
        res = await asyncio.to_thread(pipeline.process_audio, file_path)
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
        
    print(f"[API] Received audio file: {file.filename}")

    async def event_generator():
        try:
            # 1. Start Transcription
            print("[API] Starting Transcription...")
            yield {"event": "status", "data": json.dumps({"step": "transcribing", "message": "Converting Audio to Text..."})}
            await asyncio.sleep(0.1) # Yield control to flush message
            
            t0 = time.time()
            # Run blocking Whisper inference in a threadpool so we don't freeze the async event loop!
            transcript = await asyncio.to_thread(pipeline._transcribe, file_path)
            asr_latency = time.time() - t0
            print(f"[API] Transcription finished in {asr_latency:.3f}s: {transcript}")
            
            # 2. Transcription Finished
            yield {"event": "transcript", "data": json.dumps({"text": transcript, "latency": f"{asr_latency:.3f}s"})}
            await asyncio.sleep(0.1)
            
            # 3. Start Classification
            print("[API] Starting ModernBERT Classification...")
            yield {"event": "status", "data": json.dumps({"step": "classifying", "message": "Analyzing Intent via ModernBERT..."})}
            await asyncio.sleep(0.1)
            
            t1 = time.time()
            # Run blocking PyTorch classification in threadpool
            prediction = await asyncio.to_thread(pipeline._classify, transcript)
            clf_latency = time.time() - t1
            print(f"[API] Classification finished in {clf_latency:.3f}s: {prediction}")
            
            # 4. Final Result
            yield {"event": "result", "data": json.dumps({
                "prediction": _display_prediction(prediction),
                "asr_latency": asr_latency,
                "clf_latency": clf_latency,
                "total_latency": asr_latency + clf_latency
            })}
        except Exception as e:
            print(f"[API] ERROR: {e}")
            yield {"event": "error", "data": str(e)}
        finally:
            if os.path.exists(file_path):
                os.remove(file_path)
            print("[API] Cleaned up temporary file.")

    return EventSourceResponse(event_generator())

os.makedirs("static", exist_ok=True)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    print(f"Starting Scam Detection Server on http://0.0.0.0:{port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        # /ws/live is a long-lived connection idle between windows -- keep it
        # alive through intermediate proxies instead of relying on uvicorn's
        # defaults, which some platforms' idle-connection reapers beat.
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
