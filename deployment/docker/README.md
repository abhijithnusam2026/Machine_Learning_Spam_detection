# Serving Container

Build locally:

```bash
docker build \
  -f deployment/docker/Dockerfile.serving \
  -t call-center-intel-serving:latest .
```

Run with a merged fine-tuned or quantized model:

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 \
  -e MODEL_NAME=/models/qwen25-call-center-merged \
  -e SERVED_MODEL_NAME=call-center-qwen25 \
  -e DTYPE=float16 \
  -v "$PWD/artifacts:/models" \
  call-center-intel-serving:latest
```

Run with a LoRA adapter:

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 \
  -e MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
  -e LORA_ADAPTER_PATH=/models/adapters/qwen25-phase1 \
  -e SERVED_MODEL_NAME=call-center-qwen25 \
  -v "$PWD/checkpoints:/models/adapters" \
  call-center-intel-serving:latest
```

Optional quantization:

```bash
-e QUANTIZATION=bitsandbytes
```

The container starts vLLM on port `8001` and the FastAPI gateway on port `8000`.
