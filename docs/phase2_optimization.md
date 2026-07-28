# Phase 2: Optimization

Phase 2 covers precision benchmarking and ONNX Runtime export for the fine-tuned call-center model.

## Quantization Strategy

The first optimization comparison uses:

- FP16 PyTorch loading as the GPU baseline.
- INT8 bitsandbytes loading with `BitsAndBytesConfig(load_in_8bit=True)`.

The benchmark script reports:

- Perplexity on assistant response tokens.
- Intent classification accuracy.
- Intent classification macro F1.
- Latency in milliseconds per generated token.
- Throughput in generated tokens per second.
- Memory footprint from model parameters, process RSS, and CUDA allocator stats.

Run:

```bash
python scripts/quantize_and_benchmark.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/phase1/qwen25-1p5b-r16-a32-qlora/final_adapter \
  --precisions fp16 int8 \
  --limit 100
```

## ONNX Runtime Export

For PEFT adapters, the ONNX export script merges the LoRA adapter into the base model in a temporary directory, exports that merged model through Optimum, and benchmarks generation latency against the PyTorch adapter stack.

Run:

```bash
python scripts/export_onnx_and_benchmark.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/phase1/qwen25-1p5b-r16-a32-qlora/final_adapter \
  --provider CPUExecutionProvider
```

For GPU:

```bash
python scripts/export_onnx_and_benchmark.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/phase1/qwen25-1p5b-r16-a32-qlora/final_adapter \
  --provider CUDAExecutionProvider
```

## Practical Caveats

- bitsandbytes INT8 requires CUDA. On CPU-only machines, run FP32/FP16 PyTorch and ONNX CPU comparisons instead.
- ONNX export for decoder-only LLMs can be slow and memory-intensive. Export after validating the adapter on a smaller benchmark limit.
- For GPU production serving, ONNX Runtime is a useful comparison point, but vLLM is still the expected serving runtime for Phase 3 because it provides continuous batching and efficient KV-cache management.
- If the fine-tuned adapter was trained with QLoRA, export should use the merged adapter path produced by this script rather than trying to export the 4-bit training graph.
