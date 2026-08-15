#!/bin/bash
set -e

echo "=========================================================="
echo "      Scam Detection AI - Unified Modular Pipeline"
echo "=========================================================="

echo "[1/10] Downloading Raw Data..."
python src/data/00_download_raw_data.py

echo "[2/10] Building Modular Datasets and Global Hold-out Set..."
python src/data/02_build_datasets.py

echo "[3/10] Validating Processed Data Partitions..."
python src/data/03_validate_partitions.py

echo "[4/10] Training Baseline (DistilBERT) on Written Text..."
python src/models/03_train_baseline_distilbert.py

echo "[5/10] Training Universal Text Classifier (ModernBERT)..."
python src/models/04_train_universal_modernbert.py

echo "[6/10] Retraining Universal Classifier on ASR Transcripts..."
python src/models/05_retrain_transcript_modernbert.py

echo "[7/10] Exporting and Quantizing to GGUF (PTQ)..."
python src/optimization/06_ptq_modernbert.py

echo "[8/10] Benchmarking Whisper Quantized Models..."
python src/evaluation/07_whisper_quant_benchmark.py

echo "[9/10] Benchmarking E2E Combinations against Global Hold-out Set..."
python src/evaluation/08_combo_benchmark.py

echo "[10/10] Selecting Best Pipeline Configuration..."
python src/evaluation/09_best_pipeline_selection.py

echo "=========================================================="
echo " Pipeline Complete. Metrics logged to DagsHub MLflow."
echo "=========================================================="
