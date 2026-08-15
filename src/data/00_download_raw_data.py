import os
import dagshub
import mlflow
from dotenv import load_dotenv

def download_file(s3_client, repo_name, s3_key, local_path):
    if not os.path.exists(local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        try:
            print(f"Downloading {s3_key} to {local_path}...")
            s3_client.download_file(repo_name, s3_key, local_path)
        except Exception as e:
            print(f"Failed to download {s3_key}: {e}")
    else:
        print(f"File {local_path} already exists. Skipping download.")

def main():
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
        # Raw ASR Transcripts (Phase 2 foundation)
        ("data/phase2_asr/raw_asr_transcripts.csv", "data/raw/asr/raw_asr_transcripts.csv")
    ]

    print("--- Downloading Foundational Raw Datasets from DagsHub ---")
    
    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    mlflow.set_experiment("scam-detection/refactored_pipeline/00_download_raw_data")
    
    with mlflow.start_run(run_name="download_raw_data"):
        for s3_key, local_path in raw_datasets:
            download_file(s3_client, repo_name, s3_key, local_path)
            if os.path.exists(local_path):
                mlflow.log_artifact(local_path, artifact_path=os.path.dirname(local_path).replace("data/", ""))
                
        print("Download and MLflow logging complete.")

if __name__ == "__main__":
    main()
