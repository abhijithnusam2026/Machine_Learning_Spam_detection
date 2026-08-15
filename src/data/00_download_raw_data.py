import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import dagshub
import mlflow
import pandas as pd
from dotenv import load_dotenv
from src.utils.mlflow_reporting import log_dataframe_artifact

def download_file(s3_client, repo_name, s3_key, local_path):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            print(f"Downloading {s3_key} to {local_path}...")
            s3_client.download_file(repo_name, s3_key, local_path)
            return True
        except Exception as e:
            print(f"Failed to download {s3_key}: {e}")
            return False
    else:
        print(f"File {local_path} already exists. Skipping download.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Download foundational raw datasets from DagsHub.")
    parser.add_argument(
        "--skip_mlflow",
        action="store_true",
        help="Download raw files without creating a DagsHub MLflow tracking run.",
    )
    args = parser.parse_args()

    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if not (repo_owner and repo_name):
        print("ERROR: DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set in .env")
        return

    s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
    
    # We download ONLY the absolute foundational raw sources into data/raw/
    raw_datasets = [
        # LLM Synthetic JSONs
        ("data/raw_jsons/scam_call_hard_examples_250.json", "data/raw/jsons/scam_call_hard_examples_250.json"),
        ("data/raw_jsons/scam_call_transcripts_250_combined.json", "data/raw/jsons/scam_call_transcripts_250_combined.json"),
        # Teeconnie Dataset Zip
        ("data/raw_teeconnie/teeconnie_dataset.zip", "data/raw/teeconnie/teeconnie_dataset.zip"),
        # Legacy Kaggle Composite (Enron, SMS, Phishing)
        ("data/legacy_composite/composite_train.csv", "data/raw/legacy_composite/composite_train.csv"),
        ("data/legacy_composite/composite_test.csv", "data/raw/legacy_composite/composite_test.csv"),
        # Labeled ASR transcript corpus (Phase 2 foundation), re-partitioned by 02_build_datasets.py
        ("data/phase2_asr/train.csv", "data/raw/asr/phase2_train.csv"),
        ("data/phase2_asr/val.csv", "data/raw/asr/phase2_val.csv"),
        ("data/phase2_asr/test.csv", "data/raw/asr/phase2_test.csv"),
    ]

    print("--- Downloading Foundational Raw Datasets from DagsHub ---")

    source_rows = []
    for s3_key, local_path in raw_datasets:
        download_file(s3_client, repo_name, s3_key, local_path)
        exists = os.path.exists(local_path)
        size_mb = os.path.getsize(local_path) / (1024 * 1024) if exists else 0.0
        source_rows.append(
            {
                "s3_key": s3_key,
                "local_path": local_path,
                "downloaded": exists,
                "size_mb": size_mb,
            }
        )

    missing_sources = [row for row in source_rows if not row["downloaded"]]
    if missing_sources:
        missing_list = "\n".join(f"  - {row['s3_key']} -> {row['local_path']}" for row in missing_sources)
        raise RuntimeError(
            "Required raw data sources are missing from DagsHub. "
            "Stopping before dataset processing or MLflow logging to avoid silent source-domain loss.\n"
            f"{missing_list}"
        )

    if args.skip_mlflow:
        print("Download complete. Skipped MLflow logging because --skip_mlflow was set.")
        return

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    mlflow.set_experiment("scam-detection/refactored_pipeline/00_download_raw_data")

    with mlflow.start_run(run_name="download_raw_data"):
        mlflow.set_tag("project_stage", "refactored_pipeline")
        mlflow.set_tag("pipeline_stage", "00_download_raw_data")
        mlflow.log_param("source_storage", "DagsHub repo bucket")
        for row in source_rows:
            mlflow.log_artifact(row["local_path"], artifact_path=os.path.dirname(row["local_path"]).replace("data/", ""))
        log_dataframe_artifact(pd.DataFrame(source_rows), "raw_source_manifest.csv", "raw_manifest")
        mlflow.log_metric("raw_sources_expected", len(raw_datasets))
        mlflow.log_metric("raw_sources_available", sum(1 for row in source_rows if row["downloaded"]))
        print("Download and MLflow logging complete.")

if __name__ == "__main__":
    main()
