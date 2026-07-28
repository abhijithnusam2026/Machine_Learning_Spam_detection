# Managed Cloud Deployment Path

This project can be deployed to a managed endpoint by packaging the same vLLM + FastAPI container used locally and pointing the endpoint at a merged or quantized model artifact.

## SageMaker Path

Recommended approach:

1. Merge the LoRA adapter into the base model and optionally quantize it.
2. Upload the model artifact to S3:

```text
s3://<bucket>/call-center-intelligence/models/qwen25-phase1/
```

3. Build and push the serving image to ECR:

```text
<account>.dkr.ecr.<region>.amazonaws.com/call-center-intel-serving:<tag>
```

4. Create a SageMaker model with:

- Container image: the ECR image.
- Model data: S3 model artifact path.
- Environment:
  - `MODEL_NAME=/opt/ml/model`
  - `SERVED_MODEL_NAME=call-center-qwen25`
  - `DTYPE=float16`
  - `GPU_MEMORY_UTILIZATION=0.85`
  - `MAX_MODEL_LEN=4096`

5. Deploy to a GPU endpoint.

Candidate instance types:

- `ml.g5.xlarge` for initial Qwen2.5-1.5B FP16/INT8 validation.
- `ml.g5.2xlarge` if concurrency or context length grows.

Key production settings:

- Use autoscaling on invocation latency and GPU utilization.
- Enable CloudWatch logs and metrics.
- Put the endpoint behind IAM-authenticated access or an internal service.
- Keep a separate canary endpoint for new adapters or quantized artifacts.

## Azure ML Path

Recommended approach:

1. Register the merged or quantized model in Azure ML:

```text
name: call-center-qwen25-phase1
path: azureml://datastores/workspaceblobstore/paths/models/qwen25-phase1/
```

2. Push the serving image to Azure Container Registry.
3. Create a Managed Online Endpoint with a GPU SKU.

Candidate VM SKUs:

- `Standard_NC4as_T4_v3` for initial validation.
- `Standard_NC8as_T4_v3` for more CPU/memory headroom.

Deployment environment variables:

- `MODEL_NAME=/var/azureml-app/model`
- `SERVED_MODEL_NAME=call-center-qwen25`
- `DTYPE=float16`
- `GPU_MEMORY_UTILIZATION=0.85`
- `MAX_MODEL_LEN=4096`

Key production settings:

- Route a small percentage of traffic to a new deployment for canary releases.
- Use Application Insights for request latency and error metrics.
- Store model artifacts and container versions together for rollback.
- Set request timeout high enough for long transcripts, then tighten after observing generated-token latency.

## Why This Container Transfers Cleanly

The local container has only two runtime assumptions:

- The model path is provided through `MODEL_NAME`.
- The API talks to vLLM through `VLLM_BASE_URL`.

That means the same image can run in Docker, Kubernetes, SageMaker, or Azure ML with environment-variable changes rather than code changes.
