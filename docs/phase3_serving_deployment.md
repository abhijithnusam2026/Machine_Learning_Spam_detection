# Phase 3: Serving and Deployment

Phase 3 serves the fine-tuned call-center model through vLLM and exposes task-specific FastAPI endpoints.

## Serving Architecture

```text
Client
  |
  v
FastAPI gateway
  |  POST /v1/chat/completions
  v
vLLM OpenAI-compatible server
  |
  v
Fine-tuned or quantized Qwen2.5 model
```

The FastAPI layer owns product-facing schemas and endpoint names. vLLM owns efficient LLM execution, batching, KV-cache management, LoRA adapter loading, and optional quantized serving.

## Local vLLM + FastAPI

Run with a merged fine-tuned model:

```bash
MODEL_NAME=/path/to/qwen25-call-center-merged \
SERVED_MODEL_NAME=call-center-qwen25 \
deployment/docker/serve_vllm_and_api.sh
```

Run with a LoRA adapter:

```bash
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
LORA_ADAPTER_PATH=checkpoints/phase1/qwen25-1p5b-r16-a32-qlora/final_adapter \
SERVED_MODEL_NAME=call-center-qwen25 \
deployment/docker/serve_vllm_and_api.sh
```

Run with bitsandbytes quantization when supported by the local vLLM build:

```bash
MODEL_NAME=/path/to/qwen25-call-center-merged \
QUANTIZATION=bitsandbytes \
deployment/docker/serve_vllm_and_api.sh
```

## API Endpoints

Start the API, then open:

```text
http://127.0.0.1:8000/docs
```

Summarization:

```bash
curl -X POST http://127.0.0.1:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"call_id":"demo-1","transcript":"Agent: Thanks for calling. Customer: I was charged twice this month. Agent: I can review the invoice and submit a refund request."}'
```

Intent classification:

```bash
curl -X POST http://127.0.0.1:8000/classify-intent \
  -H "Content-Type: application/json" \
  -d '{"call_id":"demo-1","transcript":"Agent: Thanks for calling. Customer: I was charged twice this month. Agent: I can review the invoice and submit a refund request."}'
```

Combined analysis:

```bash
curl -X POST http://127.0.0.1:8000/analyze-call \
  -H "Content-Type: application/json" \
  -d '{"call_id":"demo-1","transcript":"Agent: Thanks for calling. Customer: I was charged twice this month. Agent: I can review the invoice and submit a refund request."}'
```

## Docker

```bash
docker build \
  -f deployment/docker/Dockerfile.serving \
  -t call-center-intel-serving:latest .
```

```bash
docker run --gpus all --rm -p 8000:8000 -p 8001:8001 \
  -e MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct \
  -e LORA_ADAPTER_PATH=/models/adapters/qwen25-phase1 \
  -v "$PWD/checkpoints/phase1/qwen25-1p5b-r16-a32-qlora/final_adapter:/models/adapters/qwen25-phase1:ro" \
  call-center-intel-serving:latest
```

## Kubernetes

Apply the local manifests:

```bash
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/hpa.yaml
```

Port-forward:

```bash
kubectl port-forward service/call-center-serving 8000:80
```

The default resource profile is sized for a 1.5B model on one local GPU:

- `4` CPU / `16Gi` memory requested.
- `8` CPU / `24Gi` memory limited.
- `1` NVIDIA GPU requested and limited.

## Production Notes

- Prefer a merged model artifact for production unless you specifically need adapter hot-swapping.
- Keep the FastAPI gateway stateless; scale replicas horizontally when GPUs are available.
- Use latency, queue depth, and GPU utilization for production autoscaling instead of CPU alone.
- For Phase 4 monitoring, add Prometheus metrics around request count, endpoint latency, vLLM error rate, and generated token counts.
