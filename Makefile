.PHONY: download preprocess baseline train all

# Download raw data from Kaggle (idempotent if metadata.json exists)
data/raw/metadata.json:
	python scripts/download.py

# Preprocess the data (depends on download)
data/processed/composite_train.csv data/processed/composite_test.csv: data/raw/metadata.json scripts/preprocess.py
	python scripts/preprocess.py

# Run the classical ML baselines
baseline: data/processed/composite_train.csv data/processed/composite_test.csv
	python scripts/train_baseline.py --train_data data/processed/composite_train.csv --test_data data/processed/composite_test.csv

# Run the deep learning DistilBERT training
train: data/processed/composite_train.csv data/processed/composite_test.csv
	python scripts/train_scam_classifier.py --train_data data/processed/composite_train.csv --test_data data/processed/composite_test.csv

# Run the entire pipeline in order
all: baseline train
