# End-to-End Call Center Security & Intelligence System

An end-to-end Machine Learning system for automated scam transcript detection, call summarization, and intent classification. This project integrates lightweight NLP models for rapid inference alongside Large Language Models (LLMs) deployed on optimized infrastructure. 

---

## Part 1: Scam Alert System (College Project)

**Objective**: Detect malicious or fraudulent intents in call transcripts and messages to protect users from financial and identity-related scams.

We built a **Scam Alert System** utilizing a lightweight transformer architecture. The goal is to provide rapid, privacy-conscious alerts when suspicious patterns (urgency, credential requests, impersonation) are identified.

### Methodology
- **Exploratory Data Analysis (EDA)**: Analyzed the `composite-scam-transcript-dataset` for text lengths, class imbalances, and keyword frequencies.
- **Baseline Modeling**: Initially benchmarked using TF-IDF + Logistic Regression. 
- **Fine-Tuning**: Fine-tuned a **DistilBERT** (`distilbert-base-uncased`) sequence classification model. By using DistilBERT, we retain 97% of BERT's language understanding while being 60% faster and 40% smaller—ideal for mobile or edge inference.

### Why Fine-Tune?
Pre-trained models (zero-shot) lack the domain-specific vocabulary to reliably flag novel scam structures. Fine-tuning our DistilBERT model on conversational scam transcripts improved our F1-score drastically over lexical baselines while maintaining sub-50ms inference latency on CPU, proving that lightweight contextual models outperform basic keyword matching without requiring massive GPU resources.

*(See `notebooks/04_fine_tuning_justification.ipynb` for empirical comparisons between baseline and fine-tuned models).*

---

## Part 2: Advanced MLOps & LLM Serving (Resume & Production System)

To extend the security system into a full **Call Center Intelligence System**, we process legitimate calls for summarization, intent extraction, and analytics using massive LLMs.

### Architecture Flow

1. **Ingestion & ASR**: Raw audio calls are processed through a Speech-to-Text module (like Whisper, see `src/serving/asr.py` placeholder) to generate text transcripts.
2. **Training & Optimization Pipeline**: Synthetic dialogue data is preprocessed into instruction records, which are fed into a distributed QLoRA fine-tuning process for models like Qwen2.5-1.5B-Instruct or Llama-3.2-3B-Instruct. 
3. **Serving Layer (FastAPI)**: 
   - **`/detect-scam`**: Synchronously routes to the lightweight DistilBERT model.
   - **`/summarize` & `/classify-intent`**: Asynchronously routes to a high-performance **vLLM** backend, optimizing continuous batching and PagedAttention for the LLM on multiple GPUs.
4. **Monitoring & Maintenance**: Prometheus metrics are scraped from the FastAPI gateway and visualized in Grafana. A scheduled background job computes cosine distance on SentenceTransformer embeddings of incoming transcripts against a training baseline, triggering alerts for data drift.

### Infrastructure & Deployment
- **Lambda Labs GPU Compute**: Scripts provided to rapidly provision multi-GPU environments (see `docs/lambda_labs_setup.md`).
- **Containerization**: Fully containerized using Docker, with Kubernetes manifests ready for horizontal pod autoscaling.
- **Model Compression**: The adapter weights are merged and quantized (INT8/FP16), allowing large models to fit into constrained GPU memory without sacrificing classification accuracy.

---

## Setup & Execution

### 1. Local Environment
Environment configuration requires Python 3.10+ and CUDA-enabled hardware for model training and serving.

```bash
# Clone the repository and configure the virtual environment
python -m venv .venv
source .venv/bin/activate

# Install the package with development and GPU dependencies
pip install -e ".[dev,gpu]"

# Set up environment variables
cp .env.example .env
```

### 2. Run the Dual-Endpoint Server
```bash
# Start the FastAPI gateway 
uvicorn src.serving.app:app --host 0.0.0.0 --port 8000
```

## Repository Structure

- `configs/`: Hyperparameter sweeps and Grafana dashboard definitions.
- `data/`: Raw and processed transcripts, synthetic data.
- `deployment/`: Dockerfiles and Kubernetes manifests for the serving layer.
- `docs/`: Lambda Labs setup, training guides, and system design docs.
- `notebooks/`: EDA (`03_interim_project_report.ipynb`) and Fine-Tuning Justifications (`04_fine_tuning_justification.ipynb`).
- `scripts/`: Executable entrypoints for DistilBERT training, QLoRA tuning, quantization, and drift detection.
- `src/`: Core Python modules (`data`, `training`, `serving`, `optimization`, `monitoring`).
