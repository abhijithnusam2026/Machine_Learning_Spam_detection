.PHONY: data train-distilbert train-modernbert train-transcript quantize evaluate all

data:
	python src/data/00_download_data.py
	python src/data/02_build_datasets.py

train-distilbert: data
	python src/models/03_train_baseline_distilbert.py

train-modernbert: data
	python src/models/04_train_universal_modernbert.py

train-transcript: train-modernbert
	python src/models/05_retrain_transcript_modernbert.py

quantize: train-transcript
	python src/optimization/06_export_and_quantize_gguf.py
	python src/optimization/07_quantize_whisper.py

evaluate: quantize
	python src/evaluation/08_evaluate_all_models.py

all:
	bash run_pipeline.sh
