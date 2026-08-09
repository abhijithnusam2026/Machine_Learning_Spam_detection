# Project TODOs

## Data Pipeline & EDA
- [ ] **Phase Data Characteristics:** For all future data processing phases (Phase 3, etc.), ensure we generate and save a basic EDA report (Size, Class Balance, Length Distribution constraints) similar to `Phase2_Dataset_Summary.md`. This is critical for configuring model hyperparameters (like `max_seq_length`) and keeping a historical record of the dataset state before each training run.

## Training
- [ ] Execute Kaggle GPU training for Phase 2 (Domain Adaptation on ASR data) using `data/phase2_asr/train.csv`.
- [ ] Extract and log final F1/Accuracy metrics for Phase 1.5 and Phase 2 training.

## Optimization
- [ ] Execute Post-Training Quantization (PTQ) to INT8 and benchmark against the reserved 10% calibration and 20% test sets.
