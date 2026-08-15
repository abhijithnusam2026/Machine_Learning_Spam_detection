import modal
import os
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# 1. Define the Modal App
app = modal.App("scam-detector-gpu-api")

# 2. Define the Docker Image Environment for the GPU
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "git")
    .pip_install(
        "fastapi",
        "uvicorn",
        "python-multipart",
        "sse-starlette",
        "torch",
        "transformers",
        "mlflow",
        "dagshub",
        "python-dotenv",
        "librosa",
        "soundfile",
        "edge-tts",
        "joblib",
        "scikit-learn"
    )
)

# 3. Import the FastAPI app inside the Modal function so it mounts on boot
@app.function(
    image=image, 
    gpu="T4",  # Request an NVIDIA T4 GPU
    secrets=[modal.Secret.from_name("dagshub-secret")], # Must configure this in Modal UI
    mounts=[
        modal.Mount.from_local_dir("scripts", remote_path="/root/scripts"),
        modal.Mount.from_local_dir("configs", remote_path="/root/configs"),
        modal.Mount.from_local_dir("static", remote_path="/root/static"),
        modal.Mount.from_local_dir("models", remote_path="/root/models"),
    ],
    timeout=600 # 10 minutes timeout per request max
)
@modal.asgi_app()
def fastapi_app():
    # Force the backend to FP16 since we have a GPU
    os.environ["INFERENCE_BACKEND"] = "fp16"
    
    # We must import our api_server AFTER setting environment variables
    # We also have to do this locally inside the function so it executes inside the container
    import api_server
    return api_server.app
