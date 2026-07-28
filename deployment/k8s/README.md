# Local Kubernetes Deployment

These manifests run one pod containing:

- vLLM OpenAI-compatible server on port `8001`.
- FastAPI gateway on port `8000`.

The default resources target Qwen2.5-1.5B-class serving on a single local GPU:

- Requests: `4 CPU`, `16Gi` memory, `1 GPU`.
- Limits: `8 CPU`, `24Gi` memory, `1 GPU`.

## Build and Load Image

kind:

```bash
docker build -f deployment/docker/Dockerfile.serving -t call-center-intel-serving:latest .
kind load docker-image call-center-intel-serving:latest
```

minikube:

```bash
eval "$(minikube docker-env)"
docker build -f deployment/docker/Dockerfile.serving -t call-center-intel-serving:latest .
```

## Deploy

```bash
kubectl apply -f deployment/k8s/configmap.yaml
kubectl apply -f deployment/k8s/deployment.yaml
kubectl apply -f deployment/k8s/service.yaml
kubectl apply -f deployment/k8s/hpa.yaml
```

Port-forward the API:

```bash
kubectl port-forward service/call-center-serving 8000:80
```

Smoke test:

```bash
curl -X POST http://127.0.0.1:8000/analyze-call \
  -H "Content-Type: application/json" \
  -d '{"call_id":"demo-1","transcript":"Agent: Thanks for calling support. Customer: I was charged twice this month and need help. Agent: I can review the invoice and open a refund request. Customer: Please do, thank you."}'
```

## Model Artifacts

For a merged or quantized local model, mount it under:

```text
/var/local/call-center-intelligence/artifacts
```

Then update `MODEL_NAME` in `configmap.yaml`, for example:

```yaml
MODEL_NAME: "/models/artifacts/qwen25-call-center-merged"
```

For a LoRA adapter, set:

```yaml
MODEL_NAME: "Qwen/Qwen2.5-1.5B-Instruct"
LORA_ADAPTER_PATH: "/models/artifacts/adapters/qwen25-phase1"
```

and add `LORA_ADAPTER_PATH` to the ConfigMap.

## Notes

- Local GPU Kubernetes requires the NVIDIA container runtime and device plugin.
- HPA on CPU is included for local completeness. For production, prefer autoscaling on request latency, queue depth, or GPU utilization.
- Scaling GPU LLM serving above one replica requires enough GPUs for each pod.
