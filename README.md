# Scam Alert System (DistilBERT Fine-Tuning)

An end-to-end Machine Learning pipeline for automated scam transcript detection. This project utilizes lightweight NLP models (DistilBERT) for rapid, domain-specific inference.

## Objective

Detect malicious or fraudulent intents in call transcripts and messages. Pre-trained models (like `facebook/bart-large-mnli`) lack domain-specific vocabulary and struggle with Domain Mismatch Leakage, making them computationally heavy and easily fooled by adversarial keyword stuffing.

By fine-tuning **DistilBERT** (`distilbert-base-uncased`), we achieve:
1. **99.8% Test Set Accuracy** (F1: 99.8%, PR-AUC: 99.9%).
2. Deep contextual understanding that defeats simple keyword stuffing.
3. Sub-50ms inference latency, making it ideal for edge deployment.

*(See `notebooks/04_fine_tuning_justification.ipynb` for empirical and visual comparisons between baseline and fine-tuned models).*

## Branch Strategy

This repo is organized as a progression of branch buckets, and the scripts will mirror datasets, artifacts, and experiment runs into the matching bucket on DagsHub.

| Git branch | Logical stage bucket | Purpose |
| --- | --- | --- |
| `model-distilbert` | `model-distilbert` | Base DistilBERT run on the initial data slice |
| `feature/phase-1.5-ultimate-dataset` | `model-modernbert-universal` | Broader data pass with ModernBERT-style experiments |
| `model-long-context` | `model-modernbert-universal` | Alias for the broader-data branch bucket |
| `feature/phase-2-audio-asr` | `feature/phase-2-audio-asr` | Transcript retraining, ASR, and audio evaluation |
| `feature/phase-3-serving-quantization` | `feature/phase-3-serving-quantization` | Quantization, serving, and inference plumbing |

Runs are logged to MLflow experiments named like `scam-detection/<stage>/<phase>`, so the DagsHub UI stays readable even when branches move in parallel.

## Data Pipeline

1. **Download**: Raw composite datasets are pulled idempotently via `scripts/download.py` and mirrored to the matching DagsHub branch bucket.
2. **Preprocess**: Text is strictly deduplicated, stripped of leaky quote artifacts, and heuristically audited for label noise via `scripts/preprocess.py`.
3. **Baseline**: Classical ML models (TF-IDF + Logistic Regression / LightGBM) are trained as a benchmark via `scripts/train_baseline.py`.
4. **Fine-Tuning**: DistilBERT is trained with FP16 mixed-precision and strict random seeding via `scripts/train_scam_classifier.py`.

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
Use the `Makefile` to automatically orchestrate the download, preprocessing, baselines, and DistilBERT training in the correct order:
```bash
make all
```

*(Alternatively, run individual steps: `make download`, `make preprocess`, `make baseline`, `make train`)*

When running from the branch buckets directly, the scripts accept a `--branch` flag so you can keep the DagsHub uploads aligned with the branch you are working on:

```bash
python scripts/download.py --branch model-distilbert
python scripts/preprocess.py --branch model-distilbert
python scripts/train_baseline.py --train_data data/model-distilbert/processed/composite_train.csv --test_data data/model-distilbert/processed/composite_test.csv --branch model-distilbert
python scripts/train_scam_classifier.py --train_data data/model-distilbert/processed/composite_train.csv --test_data data/model-distilbert/processed/composite_test.csv --branch model-distilbert
```

---

## Live Inference API

Once the model is saved to `./scam-classifier-model`, you can spin up the FastAPI server to test it in real-time.
```bash
uvicorn src.serving.app:app --reload
```
Navigate to `http://127.0.0.1:8000/docs` to use the interactive Swagger UI and send custom text to the `/detect-scam` endpoint!
