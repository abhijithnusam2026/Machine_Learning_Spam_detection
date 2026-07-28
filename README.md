# Scam Alert System

An end-to-end Machine Learning system for automated scam transcript detection. This project utilizes lightweight NLP models for rapid inference on constrained hardware.

## Objective

Detect malicious or fraudulent intents in call transcripts and messages to protect users from financial and identity-related scams.

We built a **Scam Alert System** utilizing a lightweight transformer architecture. The goal is to provide rapid, privacy-conscious alerts when suspicious patterns (urgency, credential requests, impersonation) are identified.

### Methodology
- **Exploratory Data Analysis (EDA)**: Analyzed the `composite-scam-transcript-dataset` for text lengths, class imbalances, and keyword frequencies (see `notebooks/03a_EDA_DistilBERT.ipynb`).
- **Baseline Modeling**: Initially benchmarked using TF-IDF + Logistic Regression. 
- **Fine-Tuning**: Fine-tuned a **DistilBERT** (`distilbert-base-uncased`) sequence classification model. By using DistilBERT, we retain 97% of BERT's language understanding while being 60% faster and 40% smaller—ideal for mobile or edge inference.

### Why Fine-Tune?
Pre-trained models (zero-shot) lack the domain-specific vocabulary to reliably flag novel scam structures. Fine-tuning our DistilBERT model on conversational scam transcripts improved our F1-score drastically over lexical baselines while maintaining sub-50ms inference latency on CPU, proving that lightweight contextual models outperform basic keyword matching without requiring massive GPU resources.

*(See `notebooks/04_fine_tuning_justification.ipynb` for empirical comparisons between baseline and fine-tuned models).*

## Data Pipeline

1. **Ingestion**: Audio is passed through a Speech-to-Text module (like Whisper, see `src/serving/asr.py`).
2. **Download**: Raw data is pulled via `scripts/download.py`.
3. **Preprocess**: Text is cleaned and saved to `data/processed/` using `scripts/preprocess.py`.

## Quick Start (Training & Inference)

1. Setup environment and add `KAGGLE_API_TOKEN` to `.env`.
2. Download and Preprocess:
   ```bash
   python scripts/download.py
   python scripts/preprocess.py
   ```
3. Train the model:
   ```bash
   python scripts/train_scam_classifier.py --data data/processed/composite_train.csv
   ```
4. Serve the API locally:
   ```bash
   uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
   ```
