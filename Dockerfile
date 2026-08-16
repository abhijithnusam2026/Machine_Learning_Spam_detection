# Use an official Python runtime as a parent image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for whisper.cpp and audio processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    cmake \
    make \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy local code to the container
COPY . /app/

# Install uv for fast dependency installation
RUN pip install uv

# Install pinned Python dependencies. Since this is for HF Spaces CPU, replace
# the default torch wheel with the CPU wheel after installing the project stack.
RUN uv pip install --system -r requirements.txt
RUN uv pip install --system --force-reinstall torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu

# Ensure the whisper.cpp binary is built
RUN if [ ! -d "whisper.cpp" ]; then git clone https://github.com/ggerganov/whisper.cpp.git; fi
RUN cd whisper.cpp && cmake -B build && cmake --build build --config Release -j

# Pre-download models if needed, or allow pipeline to pull from DagsHub at boot
# Note: DagsHub secrets (MLFLOW_TRACKING_PASSWORD, etc.) must be set in HF Spaces Secret settings.

# HuggingFace Spaces expect the app to run on port 7860
EXPOSE 7860

# Run the FastAPI server
CMD ["uvicorn", "src.deployment.api_server:app", "--host", "0.0.0.0", "--port", "7860"]
