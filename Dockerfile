# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for whisper.cpp and audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to the container
COPY . /app/

# Install uv for fast dependency installation
RUN pip install uv

# Install Python dependencies
# Note: Since this is for HF Spaces (CPU), we don't need the heavy CUDA PyTorch
RUN uv pip install --system fastapi uvicorn python-multipart sse-starlette \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu \
    transformers mlflow dagshub python-dotenv librosa soundfile llama-cpp-python scikit-learn joblib edge-tts pandas

# Ensure the whisper.cpp binary is built
RUN if [ ! -d "whisper.cpp" ]; then git clone https://github.com/ggerganov/whisper.cpp.git; fi
RUN cd whisper.cpp && make

# Set the environment variable to force GGUF backend on HF Spaces
ENV INFERENCE_BACKEND=gguf

# Pre-download models if needed, or allow pipeline to pull from DagsHub at boot
# Note: DagsHub secrets (MLFLOW_TRACKING_PASSWORD, etc.) must be set in HF Spaces Secret settings.

# HuggingFace Spaces expect the app to run on port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "7860"]
