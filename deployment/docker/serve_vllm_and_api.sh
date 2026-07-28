#!/usr/bin/env bash
set -euo pipefail

MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-1.5B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-call-center-qwen25}"
LORA_ADAPTER_PATH="${LORA_ADAPTER_PATH:-}"
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8001}"
API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
DTYPE="${DTYPE:-float16}"
QUANTIZATION="${QUANTIZATION:-}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"

vllm_args=(
  --model "${MODEL_NAME}"
  --host "${VLLM_HOST}"
  --port "${VLLM_PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --dtype "${DTYPE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
)

if [[ -n "${QUANTIZATION}" ]]; then
  vllm_args+=(--quantization "${QUANTIZATION}")
fi

if [[ -n "${LORA_ADAPTER_PATH}" ]]; then
  vllm_args+=(--enable-lora --lora-modules "${SERVED_MODEL_NAME}=${LORA_ADAPTER_PATH}")
fi

python -m vllm.entrypoints.openai.api_server "${vllm_args[@]}" &
vllm_pid=$!

export VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/v1}"
export SERVED_MODEL_NAME

cleanup() {
  kill "${vllm_pid}" 2>/dev/null || true
}
trap cleanup EXIT

python -m uvicorn src.serving.app:app --host "${API_HOST}" --port "${API_PORT}"
