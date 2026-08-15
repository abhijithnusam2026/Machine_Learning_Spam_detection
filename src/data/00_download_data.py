import os
import dagshub
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
    
    datasets_to_download = [
        # Phase 1 & 1.5 Text Data
        ("data/phase1.5/train.csv", "data/raw/phase1.5/train.csv"),
        ("data/phase1.5/val.csv", "data/raw/phase1.5/val.csv"),
        ("data/phase1.5/test.csv", "data/raw/phase1.5/test.csv"),
        # Phase 2 ASR Data
        ("data/phase2_asr/raw_asr_transcripts.csv", "data/raw/phase2_asr/raw_asr_transcripts.csv"),
        ("data/phase2_asr/train.csv", "data/raw/phase2_asr/train.csv"),
        ("data/phase2_asr/val.csv", "data/raw/phase2_asr/val.csv"),
        ("data/phase2_asr/test.csv", "data/raw/phase2_asr/test.csv"),
        # Collated EDA Data
        ("data/collated_for_eda/collated_data.csv", "data/raw/collated/collated_data.csv")
    ]

    print("--- Downloading Raw Datasets from DagsHub ---")
    for s3_key, local_path in datasets_to_download:
        download_file(s3_client, repo_name, s3_key, local_path)
        
    print("Download complete.")

if __name__ == "__main__":
    main()
