import os
import requests
import kagglehub
import glob
import shutil
from dotenv import load_dotenv

def download_hf_json(url, output_path):
    print(f"Downloading {url}...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert github/hf blob URLs to raw download URLs
    url = url.replace("/blob/", "/resolve/")
    resp = requests.get(url)
    if resp.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print(f"Saved to {output_path}")
        return True
    else:
        print(f"Failed to download from HF (Status {resp.status_code})")
        return False

def main():
    load_dotenv()
    
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if not repo_owner or not repo_name:
        print("ERROR: DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME missing from .env")
        return

    print("--- 1. Downloading historical data from web ---")
    
    # 1. LLM JSONs (from Hugging Face)
    json1 = "data/raw_jsons/scam_call_hard_examples_250.json"
    json2 = "data/raw_jsons/scam_call_transcripts_250_combined.json"
    download_hf_json("https://huggingface.co/datasets/tanu011235/spam/resolve/main/scam_call_hard_examples_250.json", json1)
    download_hf_json("https://huggingface.co/datasets/tanu011235/spam/resolve/main/scam_call_transcripts_250_combined.json", json2)
    
    # 2. Legacy Composite (from Kaggle)
    print("\nDownloading Kaggle dataset...")
    kaggle_path = kagglehub.dataset_download("ibrahimbagwan12/composite-scam-transcript-dataset")
    print(f"Kaggle data downloaded to: {kaggle_path}")
    
    csv1 = "data/legacy_composite/composite_train.csv"
    csv2 = "data/legacy_composite/composite_test.csv"
    os.makedirs("data/legacy_composite", exist_ok=True)
    
    # Copy from kaggle cache to our local data folder
    for f in glob.glob(kaggle_path + "/*.csv"):
        if "train" in f.lower():
            shutil.copy(f, csv1)
        elif "test" in f.lower():
            shutil.copy(f, csv2)

    print("\n--- 2. Uploading to DagsHub ---")
    try:
        from dagshub.upload import Repo
        repo = Repo(repo_owner, repo_name)
        
        for file in [json1, json2, csv1, csv2]:
            if os.path.exists(file):
                print(f"Uploading {file}...")
                repo.upload(file=file, path=file, commit_message=f"Archive {file} from external source")
                
        print("Upload complete! All historical datasets are now centralized in DagsHub.")
    except Exception as e:
        print(f"Failed to upload to DagsHub: {e}")

if __name__ == "__main__":
    main()
