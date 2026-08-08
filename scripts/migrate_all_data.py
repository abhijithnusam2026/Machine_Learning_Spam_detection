import os
import requests
import kagglehub
import glob
import shutil
import zipfile
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
    print("\nDownloading Kaggle Composite dataset...")
    kaggle_path = kagglehub.dataset_download("ibrahimbagwan12/composite-scam-transcript-dataset")
    csv1 = "data/legacy_composite/composite_train.csv"
    csv2 = "data/legacy_composite/composite_test.csv"
    os.makedirs("data/legacy_composite", exist_ok=True)
    for f in glob.glob(kaggle_path + "/*.csv"):
        if "train" in f.lower():
            shutil.copy(f, csv1)
        elif "test" in f.lower():
            shutil.copy(f, csv2)

    # 3. Teeconnie Kaggle Dataset
    print("\nDownloading Kaggle teeconnie dataset...")
    teeconnie_path = kagglehub.dataset_download("teeconnie/scam-and-non-scam-call-conversation-dataset")
    teeconnie_zip = "data/raw_teeconnie/teeconnie_dataset.zip"
    os.makedirs("data/raw_teeconnie", exist_ok=True)
    
    print(f"Zipping teeconnie dataset to {teeconnie_zip}...")
    with zipfile.ZipFile(teeconnie_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(teeconnie_path):
            for file in files:
                file_path = os.path.join(root, file)
                # Keep the folder structure inside the zip
                arcname = os.path.relpath(file_path, teeconnie_path)
                zipf.write(file_path, arcname)

    print("\n--- 2. Uploading to DagsHub Data Storage (Bucket) ---")
    try:
        import dagshub
        repo_id = f"{repo_owner}/{repo_name}"
        
        for file in [json1, json2, csv1, csv2, teeconnie_zip]:
            if os.path.exists(file):
                print(f"Uploading {file} to DagsHub storage bucket...")
                dagshub.upload_files(
                    repo_id, 
                    local_path=file, 
                    remote_path=file, 
                    bucket=True
                )
                
        print("Upload complete! All historical datasets are now centralized in DagsHub Storage.")
    except Exception as e:
        print(f"Failed to upload to DagsHub: {e}")

if __name__ == "__main__":
    main()
