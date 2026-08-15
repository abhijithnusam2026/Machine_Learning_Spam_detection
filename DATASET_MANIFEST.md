# Dataset Lineage Manifest

This document tracks the provenance, composition, and transformations of datasets across the iterative phases of the Scam Detection pipeline.

## Phase 1: Initial Baseline Dataset
- **Pipeline Stage:** `03_baseline_distilbert`
- **Source:** Kaggle SMS & Email Spam Datasets + Synthetic LLM generation.
  - *SMS Spam Collection Dataset* (https://www.kaggle.com/uciml/sms-spam-collection-dataset)
  - *Enron Email Dataset* (https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
- **DagsHub S3 Path:** `data/raw/` and `data/processed/`
- **Description:** Text-only messages, heavily focused on email and SMS structures.
- **Row Count:** ~10,000 samples.
- **Label Distribution:** ~8,500 Legitimate, ~1,500 Scam.
- **Preprocessing:** `python src/data/00_download_raw_data.py` followed by `python src/data/02_build_datasets.py`.

## Phase 1.5: The Universal Refinement
- **Pipeline Stage:** `04_universal_modernbert`
- **Source:** Merged multiple external Kaggle corpora.
  - *Phishing Email Dataset* (https://www.kaggle.com/datasets/subhajournal/phishingemails)
  - *Spam Email Data* (https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset)
- **DagsHub S3 Path:** `data/phase1.5/`
- **Description:** A massively expanded dataset to fix label noise and domain mismatch.
- **Row Count:** ~40,000 samples total.
  - **Train:** 32,154 rows (16,082 Scam, 16,072 Legitimate)
  - **Val:** 4,019 rows (2,010 Scam, 2,009 Legitimate)
  - **Test:** 4,020 rows (2,010 Scam, 2,010 Legitimate)
- **Preprocessing:** `python src/data/02_build_datasets.py` (Deep heuristic cleaning, quote-marker stripping, deduplication, and stratified splitting).

## Phase 2: ASR Transcript Retraining
- **Pipeline Stage:** `05_transcript_modernbert`
- **Source:** Derived from actual ASR Whisper transcripts and spoken conversational datasets.
- **DagsHub S3 Path:** `data/phase2_asr/`
- **Description:** Captures spoken disfluencies (uhs, ahs), conversational structure, and transcription artifacts that `phase1.5` text-models missed.
- **Row Count:** ~5,000 transcript samples.
- **Label Distribution:** 2,500 Legitimate, 2,500 Scam.
- **Preprocessing:** `python src/data/00_download_raw_data.py` downloads raw ASR transcripts, then `python src/data/02_build_datasets.py` deduplicates and splits them.
- **PTQ Calibration Split:** `data/processed/ptq_calibration.csv` is a stratified sample from `global_train` only. It is asserted to be disjoint from `global_val` and the frozen `global_test` holdout to avoid leakage during calibrated ONNX Runtime static INT8 PTQ.
- **Final Evaluation Split:** `data/processed/global_test.csv` remains frozen and is used for post-quantization classifier evaluation. It is not used for training, validation, calibration, or model selection.
- **Audio Benchmark Holdout:** `data/large_audio_test/manifest.csv` is used only for end-to-end ASR + classifier latency/quality benchmarking across deployment combinations.
- **Checksum / Git Hash:** Derived at commit `b19b8326d41442109fbdfd641a42799f`
