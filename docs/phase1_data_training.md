# Phase 1: Data Prep and Fine-Tuning

## Design Decisions

The initial fine-tune uses a multitask instruction dataset instead of separate adapters for summarization and intent classification.

Why this is the better Phase 1 default:

- The project has a small synthetic dataset target of roughly 2,000-3,000 calls, so sharing one adapter lets both tasks learn the same call-center language and issue patterns.
- One adapter is cheaper to train, deploy, benchmark, and compare against the base model.
- Task separation is still preserved in evaluation because summarization and classification use different prompts and metrics.

Separate fine-tunes become attractive later if one task needs a materially different dataset, decoding policy, latency target, or quality threshold.

## Data Generation

Local deterministic smoke test:

```bash
python scripts/generate_synthetic_calls.py \
  --provider template \
  --total 2500 \
  --output data/synthetic/calls.jsonl
```

OpenAI-compatible API mode:

```bash
export OPENAI_COMPATIBLE_BASE_URL="https://your-provider.example/v1"
export OPENAI_COMPATIBLE_API_KEY="..."
export OPENAI_COMPATIBLE_MODEL="your-open-model"

python scripts/generate_synthetic_calls.py \
  --provider openai_compatible \
  --total 2500 \
  --concurrency 8 \
  --output data/synthetic/calls.jsonl
```

Each raw record contains:

- `id`
- `intent`
- `raw_dialogue`
- `summary`
- `metadata`

Supported intent labels:

- `billing_issue`
- `technical_support`
- `cancellation_request`
- `complaint`
- `general_inquiry`

## Preprocessing

```bash
python scripts/preprocess_instruction_data.py \
  --input data/synthetic/calls.jsonl \
  --output-dir data/processed/phase1
```

This creates:

- `data/processed/phase1/train.jsonl`
- `data/processed/phase1/validation.jsonl`
- `data/processed/phase1/test.jsonl`
- `data/processed/phase1/calls_all.jsonl`

Each call becomes two instruction records:

- summarization
- intent classification

## Fine-Tuning

Single GPU QLoRA:

```bash
python scripts/train_lora_qwen.py \
  --use-qlora \
  --gradient-checkpointing \
  --bf16 \
  --wandb-project call-center-intelligence \
  --rank 16 \
  --alpha 32
```

Single GPU full precision LoRA:

```bash
python scripts/train_lora_qwen.py \
  --bf16 \
  --wandb-project call-center-intelligence \
  --rank 16 \
  --alpha 32
```

Multi-GPU DDP:

```bash
torchrun --nproc_per_node=2 scripts/train_lora_qwen.py \
  --use-qlora \
  --gradient-checkpointing \
  --bf16 \
  --wandb-project call-center-intelligence \
  --rank 16 \
  --alpha 32
```

FSDP launch:

```bash
torchrun --nproc_per_node=2 scripts/train_lora_qwen.py \
  --gradient-checkpointing \
  --bf16 \
  --fsdp "full_shard auto_wrap" \
  --wandb-project call-center-intelligence \
  --rank 16 \
  --alpha 32
```

Notes:

- QLoRA is usually the right first run on constrained GPUs.
- FSDP is included for distributed training practice, but LoRA/QLoRA often has enough memory savings that plain DDP is simpler and more stable for this model size.
- `configs/lora_sweep.json` defines the default rank/alpha sweep. Passing `--rank` and `--alpha` runs a single configuration.

## Evaluation

Base model:

```bash
python scripts/evaluate_phase1.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --output outputs/eval/base_phase1.json
```

Fine-tuned adapter:

```bash
python scripts/evaluate_phase1.py \
  --model-name Qwen/Qwen2.5-1.5B-Instruct \
  --adapter-path checkpoints/phase1/qwen25-1p5b-r16-a32-qlora/final_adapter \
  --output outputs/eval/finetuned_phase1.json
```

Metrics:

- Summarization: ROUGE-L
- Intent classification: accuracy and macro F1

## Inputs Needed

- Which open LLM API/provider you want to use for synthetic data generation.
- Available GPU memory and number of GPUs.
- Whether the first fine-tuning run should prioritize quality, cost, or speed.
