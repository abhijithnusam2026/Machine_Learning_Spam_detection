#!/bin/bash
set -e

echo "=========================================================="
echo "      Scam Detection AI - Unified Modular Pipeline"
echo "=========================================================="

echo "[1/9] Downloading Raw Data..."
python src/data/00_download_raw_data.py

echo "[2/9] Building Modular Datasets and Global Hold-out Set..."
python src/data/02_build_datasets.py

echo "[3/9] Training Baseline (DistilBERT) on Written Text..."
python src/models/03_train_baseline_distilbert.py

echo "[4/9] Training Universal Text Classifier (ModernBERT)..."
python src/models/04_train_universal_modernbert.py

echo "[5/9] Retraining Universal Classifier on ASR Transcripts..."
python src/models/05_retrain_transcript_modernbert.py

echo "[6/9] Exporting and Quantizing to GGUF (PTQ)..."
python src/optimization/06_ptq_modernbert.py

echo "[7/9] Benchmarking Whisper Quantized Models..."
python src/evaluation/07_whisper_quant_benchmark.py

echo "[8/9] Benchmarking E2E Combinations against Global Hold-out Set..."
python src/evaluation/08_combo_benchmark.py

echo "[9/9] Selecting Best Pipeline Configuration..."
python src/evaluation/09_best_pipeline_selection.py

echo "=========================================================="
echo " Pipeline Complete. Metrics logged to DagsHub MLflow."
echo "=========================================================="
