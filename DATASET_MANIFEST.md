# Dataset Lineage Manifest

This document tracks the provenance, composition, and transformations of datasets across the iterative phases of the Scam Detection pipeline.

## Phase 1: Initial Baseline Dataset
- **Git Branch:** `main`, `model-distilbert`
- **Source:** Kaggle SMS & Email Spam Datasets + Synthetic LLM generation.
  - *SMS Spam Collection Dataset* (https://www.kaggle.com/uciml/sms-spam-collection-dataset)
  - *Enron Email Dataset* (https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)
- **DagsHub S3 Path:** `data/raw/` and `data/processed/`
- **Description:** Text-only messages, heavily focused on email and SMS structures.
- **Row Count:** ~10,000 samples.
- **Label Distribution:** ~8,500 Legitimate, ~1,500 Scam.
- **Preprocessing:** `python scripts/preprocess.py` (Initial deduplication).

## Phase 1.5: The Universal Refinement
- **Git Branch:** `feature/phase-1.5-ultimate-dataset`, `model-modernbert-universal`
- **Source:** Merged multiple external Kaggle corpora.
  - *Phishing Email Dataset* (https://www.kaggle.com/datasets/subhajournal/phishingemails)
  - *Spam Email Data* (https://www.kaggle.com/datasets/nitishabharathi/email-spam-dataset)
- **DagsHub S3 Path:** `data/phase1.5/`
- **Description:** A massively expanded dataset to fix label noise and domain mismatch.
- **Row Count:** ~40,000 samples total.
  - **Train:** 32,154 rows (16,082 Scam, 16,072 Legitimate)
  - **Val:** 4,019 rows (2,010 Scam, 2,009 Legitimate)
  - **Test:** 4,020 rows (2,010 Scam, 2,010 Legitimate)
- **Preprocessing:** `python scripts/01_prep_data.py` (Deep heuristic cleaning, stripping quote markers, removing email artifacts, stratified splitting).

## Phase 2: ASR Transcript Retraining
- **Git Branch:** `feature/phase-2-audio-asr`
- **Source:** Derived from actual ASR Whisper transcripts and spoken conversational datasets.
- **DagsHub S3 Path:** `data/phase2_asr/`
- **Description:** Captures spoken disfluencies (uhs, ahs), conversational structure, and transcription artifacts that `phase1.5` text-models missed.
- **Row Count:** ~5,000 transcript samples.
- **Label Distribution:** 2,500 Legitimate, 2,500 Scam.
- **Preprocessing:** `python scripts/01_fetch_asr_data.py` (Audio converted to 16kHz WAV -> Whisper ASR transcription -> Deduplication -> Split into `ptq_calibration.csv`).
- **Checksum / Git Hash:** Derived at commit `b19b8326d41442109fbdfd641a42799f`
