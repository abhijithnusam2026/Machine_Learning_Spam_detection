---
title: Scam Detection AI
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: "4.44.1"
python_version: 3.10.13
app_file: src/deployment/app.py
pinned: false
---

# Scam Detection AI - Unified Modular Pipeline

An end-to-end Machine Learning pipeline for automated scam transcript detection. This project utilizes DistilBERT as a baseline and scales up to ModernBERT with long-context awareness, specifically fine-tuned on diverse text and ASR transcripts.

## 🚀 The Unified Architecture

The entire codebase has been refactored into a clean, modular structure under the `src/` directory.

### 1. Data Ingestion (`src/data/`)
Downloads raw sources (Kaggle Phishing/Enron/SMS datasets, Teeconnie data, synthetic LLM JSONs, and raw ASR transcripts). A 20% **Global Hold-Out Set** is deterministically carved out and frozen *before* any modeling occurs. A separate `ptq_calibration.csv` is then sampled only from `global_train`, with assertions preventing overlap with validation or holdout data.

Data partition diagram: [docs/data_partition_flow.md](docs/data_partition_flow.md) or [SVG](docs/assets/data_partition_flow.svg).

### 2. Model Training (`src/models/`)
- **Baseline**: DistilBERT trained only on the original Kaggle composite corpus.
- **Universal**: ModernBERT trained on the expanded written corpus (`kaggle_composite`, synthetic LLM data, and Teeconnie non-scam call examples used as legitimate-call balancing augmentation).
- **Transcript Retraining**: ModernBERT fine-tuned exclusively on ASR spoken transcripts.

### 3. Optimization (`src/optimization/`)
Converts the final PyTorch ModernBERT model into highly optimized **GGUF** weight-only quantized formats (FP16, Q8_0, Q4_K_M) via `llama.cpp` and fetches corresponding Whisper variants for local edge deployment. It also includes an optional calibrated ONNX Runtime static INT8 PTQ path that uses `data/processed/ptq_calibration.csv` for calibration and the frozen `global_test.csv` for post-quantization evaluation.

### 4. Evaluation & Benchmarking (`src/evaluation/`)
Runs all models and optimized E2E pipelines (Whisper + Classifier) against the Global Hold-Out Set, logging inference latency, model sizing, and confusion matrices directly to isolated DagsHub MLflow experiments.

---

## 🛠️ Quickstart & Reproducibility

### Environment Setup
1. Clone the repo and checkout the `refactor/unified-pipeline` branch.
2. Install the locked dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   For the optional calibrated ONNX PTQ experiment:
   ```bash
   pip install -r requirements-onnx.txt
   ```
3. Set your `.env` variables (DagsHub credentials for MLflow logging & HuggingFace token):
   ```
   DAGSHUB_REPO_OWNER="your-owner"
   DAGSHUB_REPO_NAME="your-repo"
   MLFLOW_TRACKING_USERNAME="..."
   MLFLOW_TRACKING_PASSWORD="..."
   ```

### Run the Pipeline
To execute the canonical journey from raw data download all the way to final benchmarking:

```bash
make all
# OR
bash run_pipeline.sh
```

To validate processed train/validation/test/PTQ partitions before launching long GPU training:

```bash
make validate-data
```

To rebuild data in a fresh workspace without creating duplicate data-prep MLflow runs:

```bash
python src/data/00_download_raw_data.py --skip_mlflow
python src/data/02_build_datasets.py --skip_mlflow
python src/data/03_validate_partitions.py --skip_mlflow
```

### Fast LoRA Completion Path
To complete the end-to-end pipeline quickly, train ModernBERT with LoRA, merge the adapter into a normal Hugging Face model directory, then pass that explicit model directory into transcript retraining and PTQ:

```bash
python src/models/04_train_universal_modernbert.py \
  --finetune_method lora \
  --epochs 4 \
  --output_dir ./scam-classifier-model-universal-lora

python src/models/05_retrain_transcript_modernbert.py \
  --model_name ./scam-classifier-model-universal-lora \
  --finetune_method lora \
  --epochs 4 \
  --output_dir ./scam-classifier-model-transcript-lora

python src/optimization/06_ptq_modernbert.py \
  --model_name ./scam-classifier-model-transcript-lora
```

For the slower full-parameter comparison run, use `--finetune_method full` and pass that full model output directory into the same transcript retraining and PTQ commands.

### Live Deployment
The Gradio App and FastAPI inference servers are located in `src/deployment/`.
```bash
python src/deployment/api_server.py
```
