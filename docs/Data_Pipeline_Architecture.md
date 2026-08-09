# Data Pipeline & S3 Storage Architecture

This document serves as a checkpoint to clearly define the flow of data from its raw source, through the ETL pipeline, into DagsHub S3 storage, and ultimately into the ML training scripts for both Phase 1 and Phase 1.5 of the project.

---

## 1. DagsHub S3 Storage Structure

Because DagsHub provides a single, global S3 bucket per repository (`s3://.../data/`), the storage is strictly partitioned by directories to prevent branches from overwriting each other's data.

The global S3 `data/` bucket contains the following structure:
```text
data/
├── legacy_composite/   # Phase 1: Legacy Kaggle datasets
├── raw_jsons/          # Phase 1: Synthetically generated LLM scam transcripts
├── raw_teeconnie/      # Phase 1: Teeconnie dataset zip
├── raw_external/       # Phase 1.5: Switchboard SWDA, YouTube Kitboga, Thai, BetterUp
├── phase1/             # Phase 1 Output: train.csv, test.csv
└── phase1.5/           # Phase 1.5 Output: train.csv, val.csv, test.csv
```

---

## 2. Phase 1: `model-long-context` Branch

### Data Sources
1. **Synthetically Generated Data (`raw_jsons`)**:
   - `scam_call_hard_examples_250.json` and `scam_call_transcripts_250_combined.json`.
   - Used for highly complex, multi-turn scam transcripts.
2. **Teeconnie Dataset (`raw_teeconnie`)**:
   - Zip file containing legitimate conversational text. Used as non-scam comparison data.
3. **Legacy Kaggle Dataset (`legacy_composite`)**:
   - Original baseline dataset combining various Kaggle sources.

### The Pipeline Journey
1. **Raw Download**: `scripts/01_prep_data.py` connects to DagsHub S3 and pulls the `raw_jsons`, `raw_teeconnie`, and `legacy_composite` datasets into the local machine.
2. **Mathematical Balancing**: The script combines the sources and forces a strict mathematical balance. It guarantees a perfectly balanced Train dataset (2550 rows) and Test dataset (4000 rows).
3. **Output Storage**: The final, balanced datasets are written locally to `data/phase1/train.csv` and `data/phase1/test.csv`. 
4. **Cloud Synchronization**: The script uploads these final datasets straight to `s3://.../data/phase1/` so they are safely isolated.
5. **Training**: `scripts/02_train_models.py` uses `argparse` defaults targeting `data/phase1/train.csv` and `data/phase1/test.csv` to fine-tune the `ModernBERT` neural network.

---

## 3. Phase 1.5: `feature/phase-1.5-ultimate-dataset` Branch

### Additional Data Sources
In addition to all Phase 1 sources, Phase 1.5 introduces massive external datasets stored in `data/raw_external/`:
1. **Switchboard Dialog Act Corpus (SWDA)**:
   - 1,155 highly complex, legitimate telephone conversations. Provides robust linguistic variance for "Non-Scam" classification.
2. **YouTube Scam Calls (Kaggle & Hugging Face)**:
   - Real-world scam baiter transcripts from users `rivalcults` and `BothBosu`.
3. **Thai Call Center Dataset**:
   - Additional cross-domain call center conversation data.
4. **BetterUp Candor Corpus (Kaggle Subset)**:
   - File `BETTER30.csv` containing legitimate business calls.

### The Pipeline Journey
1. **Raw Download**: `scripts/01_prep_data.py` pulls all legacy sources *plus* the entire `raw_external/` directory from DagsHub S3.
2. **Mathematical Balancing & Splitting**:
   - The script concatenates over 39,000 total rows.
   - It downsamples the majority class (Scams) to perfectly match the minority class (Legitimate), ensuring a strict 1:1 ratio (preventing domain shift). Excess data is safely discarded from memory (but remains in the raw S3 bucket).
   - The balanced dataset is split into **70% Train, 10% Validation, 20% Test** (roughly 28,000 rows total).
3. **Output Storage**: The splits are saved locally to `data/phase1.5/` and simultaneously uploaded to `s3://.../data/phase1.5/` for cloud preservation.
4. **Training via JSON Config**:
   - `scripts/02_train_models.py` reads hyperparameters and dataset paths exclusively from `configs/training_config.json` (which points to `data/phase1.5/`).
   - The script trains the model, continuously evaluating against `data/phase1.5/val.csv`.
5. **Advanced Evaluation**: Upon training completion, the script automatically tests against the held-out `test.csv`, dynamically generates **ROC Curves, Precision-Recall Curves, and Confusion Matrix Heatmaps**, and ships them directly to the `scam-detection-ultimate` MLflow dashboard.
