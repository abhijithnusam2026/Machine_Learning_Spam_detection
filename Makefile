.PHONY: download-data build-data validate-data data train-distilbert train-modernbert train-transcript quantize evaluate all

download-data:
	python src/data/00_download_raw_data.py

build-data: download-data
	python src/data/02_build_datasets.py

validate-data:
	python src/data/03_validate_partitions.py

data: build-data validate-data

train-distilbert: data
	python src/models/03_train_baseline_distilbert.py

train-modernbert: data
	python src/models/04_train_universal_modernbert.py

train-transcript: train-modernbert
	python src/models/05_retrain_transcript_modernbert.py

quantize: train-transcript
	python src/optimization/06_ptq_modernbert.py

evaluate: quantize
	python src/evaluation/07_whisper_quant_benchmark.py
	python src/evaluation/08_combo_benchmark.py
	python src/evaluation/09_best_pipeline_selection.py

all:
	bash run_pipeline.sh
