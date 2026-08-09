# Phase 1: Universal Dataset Ablation Study

## Executive Summary
This document outlines the final results of Phase 1 of the Scam Classification project. To resolve the catastrophic domain shift observed in earlier experiments (where F1 scores plummeted to 65%), we engineered a **Universal Stratified Dataset** and conducted a head-to-head ablation study between two foundational models: **DistilBERT** (512 token limit) and **ModernBERT** (8192 token limit).

The results decisively prove that training on a highly diverse, rigorously balanced dataset completely neutralizes domain shift, catapulting both models to **>98.6% accuracy** on entirely unseen data.

---

## 1. Dataset Architecture (The Universal Dataset)
To prevent the models from overfitting to LLM-specific conversational quirks, we built an explicit data pipeline (`scripts/01_prep_final_data.py`) to aggregate and balance three distinct data sources.

### Training Set (2,550 rows)
Perfectly balanced (50% Scam / 50% Legitimate). Designed to retain 100% of our high-quality synthetic data while grounding the model in real-world patterns.
- **Scams (1,275):** 425 LLM Synthetic + 850 Legacy Kaggle
- **Legitimate (1,275):** 75 LLM Synthetic + 350 Teeconnie Call Center + 850 Legacy Kaggle

### The Holdout Evaluation Set (4,000 rows)
A massive, perfectly balanced (2,000 Scam / 2,000 Legitimate) dataset of **completely unseen** real-world audio transcripts.
- **Scams:** 2,000 Legacy Kaggle
- **Legitimate:** 1,000 Legacy Kaggle + 1,000 Teeconnie Call Center

**Data Quality:** Prior to training, every single row was passed through a regex formatting scrubber to strip structural data leakage (e.g., `[Greetings]`, `Suspect:` tags).

---

## 2. Engineering Challenges: The Context Limit Memory Wall
Training ModernBERT with a massive **8,192 token context window** introduced severe hardware limitations. During the initial training phase, the NVIDIA T4 GPU (15GB VRAM) experienced catastrophic PyTorch `OutOfMemoryError` (OOM) crashes during the backward pass due to the immense size of the attention matrices.

**The Solution:**
To successfully train the model on a 15GB GPU without sacrificing the mathematical stability of a standard batch size, we deployed two critical PyTorch optimizations in `scripts/03_train_modernbert.py`:
1. **Gradient Checkpointing:** We traded computation time for memory efficiency by configuring the model to discard intermediate forward-pass activations and recompute them dynamically during the backward pass.
2. **Gradient Accumulation:** We reduced the physical batch size to `1` (guaranteeing the sequence would fit in VRAM) while setting gradient accumulation steps to `16`. This perfectly simulated the convergence stability of a large batch size while completely eliminating OOM crashes.

---

## 3. Final Evaluation Metrics
*Evaluated on the 4,000-row diverse holdout set (`test.csv`).*

| Metric | DistilBERT | ModernBERT | Winner |
| :--- | :--- | :--- | :--- |
| **Run ID (MLflow)** | `420c8c7b958a4aaf86edbe2aad53ef42` | `b06071db90694957ab3572fd94e73f13` | N/A |
| **Model Registry Name** | `distilbert-base-uncased-Scam-Classifier` | `ModernBERT-base-Scam-Classifier` | N/A |
| **Accuracy** | 98.62% | **98.85%** | ModernBERT |
| **F1 Score** | 98.62% | **98.84%** | ModernBERT |
| **Precision** | 98.45% | **99.19%** | ModernBERT |
| **Recall** | **98.80%** | 98.50% | DistilBERT |
| **PR AUC** | 99.87% | **99.91%** | ModernBERT |

### Computational Performance

| Metric | DistilBERT | ModernBERT | Winner |
| :--- | :--- | :--- | :--- |
| **Training Time** | **27 Minutes** (`1621s`) | 73 Minutes (`4385s`) | DistilBERT |
| **Inference Time (Test Set)**| **1.3 Minutes** (`78s`) | 10.3 Minutes (`621s`) | DistilBERT |
| **Inference Speed** | **50.8 samples/sec** | 6.4 samples/sec | DistilBERT |

---

## 4. Conclusion & Phase 2 Recommendations

### Key Findings
1. **The Universal Dataset is a Success:** By simply balancing the class distribution and injecting real-world legacy data into the training loop, we pushed model generalization from 65% up to **98.85%**. 
2. **ModernBERT is the Accuracy Champion:** ModernBERT's massive 8,192-token context window allowed it to analyze full-length transcripts without truncation, granting it the highest overall Accuracy and Precision.
3. **DistilBERT is the Production Champion:** DistilBERT achieved remarkably similar accuracy (98.62%) but runs **~8x faster** during inference. 

### Phase 2: Moving to Speech-to-Text (ASR)
While 98.8% accuracy on clean text is phenomenal, real-world deployment requires processing raw audio. 
Our next milestone is **Noise-Robust Domain Adaptation**:
1. Integrate an ASR model (e.g., Whisper).
2. Process messy, real-world scam audio (e.g., Kitboga YouTube transcripts).
3. Store the messy ASR transcripts in a separate DagsHub bucket.
4. Inject these transcription-error-heavy transcripts back into the Universal Dataset to train the models to detect scams *despite* poor audio quality and ASR stutters.
