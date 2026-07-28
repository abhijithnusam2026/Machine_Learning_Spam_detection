# Optimization Benchmarks

This document summarizes Phase 2 optimization results for the fine-tuned call-center model. Fill in the measured values after running the benchmark scripts on the target hardware.

## Benchmark Commands

Quantization and quality degradation:

```bash
python scripts/quantize_and_benchmark.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/phase1/qwen25-1p5b-r16-a32-qlora/final_adapter \
  --test-file data/processed/phase1/test.jsonl \
  --precisions fp16 int8 \
  --limit 100 \
  --output outputs/benchmarks/quantization.json
```

ONNX export and latency benchmark:

```bash
python scripts/export_onnx_and_benchmark.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/phase1/qwen25-1p5b-r16-a32-qlora/final_adapter \
  --test-file data/processed/phase1/test.jsonl \
  --onnx-output-dir artifacts/onnx/qwen25_phase1 \
  --output outputs/benchmarks/onnx_runtime.json
```

CPU-only ONNX Runtime run:

```bash
python scripts/export_onnx_and_benchmark.py \
  --provider CPUExecutionProvider \
  --onnx-output-dir artifacts/onnx/qwen25_phase1_cpu \
  --output outputs/benchmarks/onnx_runtime_cpu.json
```

GPU ONNX Runtime run, when CUDA execution provider is available:

```bash
python scripts/export_onnx_and_benchmark.py \
  --provider CUDAExecutionProvider \
  --onnx-output-dir artifacts/onnx/qwen25_phase1_cuda \
  --output outputs/benchmarks/onnx_runtime_cuda.json
```

## Summary Table

| Variant | Runtime | Precision | Device | Latency (ms/token) | Throughput (tokens/sec) | Memory Footprint (MB) | Perplexity | Intent Accuracy | Intent Macro F1 | Notes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Base fine-tuned adapter | PyTorch | FP16 | GPU | TBD | TBD | TBD | TBD | TBD | TBD | Pre-optimization baseline |
| Quantized adapter | PyTorch + bitsandbytes | INT8 | GPU | TBD | TBD | TBD | TBD | TBD | TBD | LLM.int8 load path |
| Exported merged model | ONNX Runtime | FP16/FP32 | CPU | TBD | TBD | TBD | N/A | N/A | N/A | Latency-only CPU comparison |
| Exported merged model | ONNX Runtime | FP16 | GPU | TBD | TBD | TBD | N/A | N/A | N/A | Requires CUDAExecutionProvider |

## Quality Degradation

| Variant | Perplexity Delta | Intent Accuracy Delta | Intent Macro F1 Delta | Accept/Reject |
|---|---:|---:|---:|---|
| INT8 vs FP16 | TBD | TBD | TBD | TBD |

## Interpretation Notes

- Use FP16 as the main GPU serving baseline for this model size.
- INT8 should be accepted only if latency or memory improves enough to justify any drop in intent accuracy or macro F1.
- ONNX Runtime export is most useful as a controlled latency comparison and as a CPU fallback path. For production LLM serving on GPU, vLLM will usually be the stronger Phase 3 target.
- Perplexity is measured only on assistant response tokens, matching the supervised fine-tuning loss mask.
- Intent accuracy and macro F1 are generated-output metrics, so they reflect both model quality and decoding behavior.
