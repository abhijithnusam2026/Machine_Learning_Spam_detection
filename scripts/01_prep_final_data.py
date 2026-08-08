"""
Prepares the final dataset by combining 500 JSON examples (highly imbalanced)
with 350 legitimate transcripts sampled from the AIxBlock real-world call center dataset.
This yields a perfectly balanced 850-row dataset for training.
"""

import os
import json
import random
import re
import pandas as pd
import requests
from requests.auth import HTTPBasicAuth
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)

def main():
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    print("Loading synthesized JSON data from DagsHub...")
    json_dir = "data/raw_jsons"
    os.makedirs(json_dir, exist_ok=True)
    json_files = ["scam_call_hard_examples_250.json", "scam_call_transcripts_250_combined.json"]
    
    auth = HTTPBasicAuth(username, password) if username and password else None
    synth_data = []
    
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        # Download from DagsHub
        if repo_owner and repo_name:
            url = f"https://dagshub.com/{repo_owner}/{repo_name}/raw/main/data/raw_jsons/{fname}"
            resp = requests.get(url, auth=auth)
            if resp.status_code == 200:
                with open(path, "wb") as f:
                    f.write(resp.content)
            else:
                print(f"Warning: Failed to download {fname} from DagsHub (Status {resp.status_code})")
        
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                synth_data.extend(data)
        else:
            print(f"ERROR: {path} not found locally or in DagsHub.")
            return

    df_synth = pd.DataFrame(synth_data)
    print(f"Loaded {len(df_synth)} synthetic examples.")
    print("Synthetic label distribution:", df_synth["label"].value_counts().to_dict())

    # We have 425 scams and 75 legits in the synthetic data.
    # We need 350 more legits to balance it to 425/425.
    print("\nDownloading teeconnie dataset from DagsHub...")
    teeconnie_zip = "data/raw_teeconnie/teeconnie_dataset.zip"
    os.makedirs("data/raw_teeconnie", exist_ok=True)
    
    if repo_owner and repo_name:
        url = f"https://dagshub.com/{repo_owner}/{repo_name}/raw/main/{teeconnie_zip}"
        resp = requests.get(url, auth=auth)
        if resp.status_code == 200:
            with open(teeconnie_zip, "wb") as f:
                f.write(resp.content)
        else:
            print(f"ERROR: Failed to download {teeconnie_zip} from DagsHub (Status {resp.status_code})")
            return
    else:
        print("ERROR: DAGSHUB_REPO_OWNER missing in .env")
        return
        
    import zipfile
    print("Unzipping teeconnie dataset...")
    with zipfile.ZipFile(teeconnie_zip, 'r') as zipf:
        zipf.extractall("data/raw_teeconnie/")
        
    import glob
    teeconnie_files = glob.glob("data/raw_teeconnie/**/*", recursive=True)
    nonscam_txt = [f for f in teeconnie_files if f.lower().endswith(".txt") and "non" in f.lower() and "scam" in f.lower()]
    
    entries = []
    if nonscam_txt:
        with open(nonscam_txt[0], "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        entries = [e.strip() for e in content.split("\n\n") if e.strip()]
        if len(entries) < 5:
            entries = [e.strip() for e in content.split("\n") if e.strip()]
    
    random.shuffle(entries)
    aix_samples = entries[:350]
    
    aix_df = pd.DataFrame({
        "text": aix_samples,
        "label": [0] * len(aix_samples)
    })
    
    print(f"Sampled {len(aix_samples)} legitimate transcripts from Kaggle teeconnie dataset.")

    print("\nMerging datasets...")
    df_all = pd.concat([df_synth, aix_df], ignore_index=True)
    df_all = df_all.dropna(subset=["text", "label"]).reset_index(drop=True)
    df_all["label"] = df_all["label"].astype(int)
    
    print("Scrubbing structural formatting leakage...")
    def clean_text(text):
        text = str(text)
        # Remove Innocent/Suspect tags from JSONs
        text = re.sub(r'(?i)(innocent|suspect):\s*', '', text)
        # Remove template brackets [Greetings], [Name], etc from Kaggle
        text = re.sub(r'\[.*?\]', '', text)
        # Remove multiple spaces and lowercase everything to unify texture
        text = re.sub(r'\s+', ' ', text).strip()
        return text.lower()
        
    df_all["text"] = df_all["text"].apply(clean_text)
    
    # Simple deduplication just in case
    df_all = df_all.drop_duplicates(subset=["text"], keep="first")
    
    print(f"Final combined dataset shape: {df_all.shape}")
    print("Final label distribution:")
    print(df_all["label"].value_counts(normalize=True) * 100)
    
    # Stratified split 80/20
    print("\nPerforming 80/20 stratified split...")
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=SEED, stratify=df_all["label"])
    
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("\nSaved locally temporarily.")
    
    # Upload to DagsHub Data Storage
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if repo_owner and repo_name:
        print(f"\nUploading datasets directly to DagsHub ({repo_owner}/{repo_name})...")
        try:
            from dagshub.upload import Repo
            repo = Repo(repo_owner, repo_name)
            repo.upload(local_path="data/train.csv", remote_path="data/train.csv", commit_message="Update train dataset via pipeline", branch="model-long-context")
            repo.upload(local_path="data/test.csv", remote_path="data/test.csv", commit_message="Update test dataset via pipeline", branch="model-long-context")
            print("Successfully uploaded to DagsHub!")
        except Exception as e:
            print(f"Failed to upload to DagsHub: {e}")
            print("Make sure you are logged in using `dagshub login`")
    else:
        print("DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME not found in .env. Skipping cloud upload.")

if __name__ == "__main__":
    main()
