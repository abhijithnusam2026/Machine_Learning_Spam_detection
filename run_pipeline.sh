#!/bin/bash
set -e

echo "=========================================================="
echo "      Scam Detection AI - Unified Modular Pipeline"
echo "=========================================================="

echo "[1/8] Downloading Raw Data..."
python src/data/00_download_data.py

echo "[2/8] Building Modular Datasets and Global Hold-out Set..."
python src/data/02_build_datasets.py

echo "[3/8] Training Baseline (DistilBERT) on Written Text..."
python src/models/03_train_baseline_distilbert.py

echo "[4/8] Training Universal Text Classifier (ModernBERT)..."
python src/models/04_train_universal_modernbert.py

echo "[5/8] Retraining Universal Classifier on ASR Transcripts..."
python src/models/05_retrain_transcript_modernbert.py

echo "[6/8] Exporting and Quantizing to GGUF (PTQ)..."
python src/optimization/06_export_and_quantize_gguf.py

echo "[7/8] Preparing Whisper Quantized Models..."
python src/optimization/07_quantize_whisper.py

echo "[8/8] Benchmarking E2E against Global Hold-out Set..."
python src/evaluation/08_evaluate_all_models.py

echo "=========================================================="
echo " Pipeline Complete. Metrics logged to DagsHub MLflow."
echo "=========================================================="
