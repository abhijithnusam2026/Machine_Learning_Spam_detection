FROM python:3.11-slim

# Note: This Dockerfile is for reproducible data orchestration and baseline evaluation.
# Training DistilBERT via `make all` in this container will be extremely slow (CPU-only).
# For DistilBERT training, we highly recommend using Kaggle/Colab with GPU acceleration.

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y make build-essential && rm -rf /var/lib/apt/lists/*

# Install python dependencies securely via frozen lockfile
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

# Copy source code
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .
COPY . .

CMD ["make", "all"]
