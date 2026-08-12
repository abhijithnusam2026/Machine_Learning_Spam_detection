# Scam Alert System (ModernBERT Phase 1.5)

An end-to-end Machine Learning pipeline for automated scam transcript detection. This branch extends the baseline DistilBERT work with a broader ModernBERT stage that pulls in more non-transcript call-center data and writes all runs and artifacts to the `model-modernbert-universal` bucket on DagsHub.

## Objective

Detect malicious or fraudulent intents in call transcripts and messages. Pre-trained models (like `facebook/bart-large-mnli`) lack domain-specific vocabulary and struggle with Domain Mismatch Leakage, making them computationally heavy and easily fooled by adversarial keyword stuffing.

By fine-tuning **ModernBERT** (`answerdotai/ModernBERT-base`) on the expanded phase 1.5 dataset, we aim to:
1. Improve recall on scam variants that were not present in the base branch.
2. Keep inference practical for later quantization and serving stages.
3. Preserve a clean DagsHub audit trail for data, metrics, and model registry entries.

*(See `notebooks/04_fine_tuning_justification.ipynb` for empirical and visual comparisons between baseline and fine-tuned models).*

## Data Pipeline

1. **Download**: Existing DagsHub datasets and external complements are pulled idempotently via `scripts/01_prep_data.py`.
2. **Preprocess**: Non-transcript data is cleaned, deduplicated, balanced, and split 70/10/20 before upload to the branch bucket.
3. **Baseline**: ModernBERT is trained from the stage-scoped config in `configs/training_config.json`.
4. **Evaluation**: Cross-dataset evaluation and confusion-matrix logging are tracked via MLflow on DagsHub.

## Branch Strategy

| Git branch | Logical stage bucket | Purpose |
| --- | --- | --- |
| `model-distilbert` | `model-distilbert` | Base branch with the original DistilBERT pass |
| `feature/phase-1.5-ultimate-dataset` | `model-modernbert-universal` | Broader non-transcript data on ModernBERT |
| `model-long-context` | `model-modernbert-universal` | Alias for the broader-data stage bucket |
| `feature/phase-2-audio-asr` | `feature/phase-2-audio-asr` | Transcript retraining and ASR/audio work |
| `feature/phase-3-serving-quantization` | `feature/phase-3-serving-quantization` | Quantization, serving, and dynamic inference |

The code in this branch writes its uploads and MLflow runs to `scam-detection/model-modernbert-universal/*`.

---

## Reproducibility & Orchestration (Makefile + Docker)

To guarantee exact reproducibility, this repository uses a `Makefile` and a frozen `requirements.lock.txt`.

### Method 1: The Docker Container (Best for Environment Portability)
We provide a `python:3.11-slim` Docker container. 
> ⚠️ **WARNING:** Docker on macOS does not have access to the Apple GPU. Running the training step inside Docker will be extremely slow (CPU-only).
```bash
docker build -t scam-alert .
docker run -it scam-alert
```

### Method 2: Native Execution (Best for Kaggle GPUs & Mac MPS)
If you run this natively in a Kaggle notebook or on a Mac, the scripts will automatically utilize **Nvidia T4 GPUs** (via DataParallel) or **Apple Silicon GPUs** (via MPS).

**1. Setup Environment**
Ensure your `.env` contains your Hugging Face token with Write permissions.
```bash
HF_TOKEN="hf_..."
```

**2. Run the Entire Pipeline**
Use the `Makefile` to automatically orchestrate the download, preprocessing, baselines, and ModernBERT training in the correct order:
```bash
make all
```

*(Alternatively, run individual steps: `make download`, `make preprocess`, `make baseline`, `make train`)*

The branch-scoped scripts already default to the `model-modernbert-universal` bucket in DagsHub. If you want to be explicit, you can set `DAGSHUB_BRANCH=feature/phase-1.5-ultimate-dataset` in your environment.

---

## Live Inference API

Once the model is saved to `./scam-classifier-model`, you can spin up the FastAPI server to test it in real-time.
```bash
uvicorn src.serving.app:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` to use the interactive Swagger UI and send custom text to the `/detect-scam` endpoint!
