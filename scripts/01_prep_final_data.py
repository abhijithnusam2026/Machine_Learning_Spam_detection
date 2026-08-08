"""
Prepares the final dataset by explicitly constructing a balanced Train and Test set.
Train Set: 100% of LLM data + Teeconnie + Legacy (2550 rows, perfectly balanced)
Test Set: Completely unseen Teeconnie + Legacy (4000 rows, perfectly balanced)
"""

import os
import json
import random
import re
import glob
import zipfile
import pandas as pd
import requests
import dagshub
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)

def clean_text(text):
    text = str(text)
    text = re.sub(r'(?i)(innocent|suspect):\s*', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def main():
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not (repo_owner and repo_name):
        print("ERROR: DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set in .env")
        return

    # Map MLflow credentials to DagsHub authentication
    if username and password:
        os.environ["DAGSHUB_USER"] = username
        os.environ["DAGSHUB_TOKEN"] = password
        dagshub.auth.add_app_token(password)

    # Initialize DagsHub S3 client
    repo_id = f"{repo_owner}/{repo_name}"
    s3_client = dagshub.get_repo_bucket_client(repo_id)

    # 1. Load Synthesized JSON data (LLM Data)
    print("Loading synthesized JSON data from DagsHub...")
    json_dir = "data/raw_jsons"
    os.makedirs(json_dir, exist_ok=True)
    json_files = ["scam_call_hard_examples_250.json", "scam_call_transcripts_250_combined.json"]
        
    synth_data = []
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        s3_key = f"data/raw_jsons/{fname}"
        
        try:
            s3_client.download_file(repo_name, s3_key, path)
        except Exception as e:
            pass # ignore if exists locally
        
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                synth_data.extend(data)

    df_synth = pd.DataFrame(synth_data)
    df_synth["text"] = df_synth["text"].apply(clean_text)
    
    synth_scams = df_synth[df_synth["label"] == 1]
    synth_legits = df_synth[df_synth["label"] == 0]
    
    print(f"Loaded LLM Data: {len(synth_scams)} Scams, {len(synth_legits)} Legits.")

    # 2. Download and Load Teeconnie dataset
    print("\nDownloading teeconnie dataset from DagsHub S3...")
    teeconnie_zip = "data/raw_teeconnie/teeconnie_dataset.zip"
    os.makedirs("data/raw_teeconnie", exist_ok=True)
    s3_key = "data/raw_teeconnie/teeconnie_dataset.zip"
    
    try:
        s3_client.download_file(repo_name, s3_key, teeconnie_zip)
    except Exception as e:
        pass
        
    with zipfile.ZipFile(teeconnie_zip, 'r') as zipf:
        zipf.extractall("data/raw_teeconnie/")
        
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
    
    df_teeconnie_full = pd.DataFrame({
        "text": entries,
        "label": [0] * len(entries)
    })
    df_teeconnie_full["text"] = df_teeconnie_full["text"].apply(clean_text)
    
    # We need 350 for train, 1000 for test
    teeconnie_train = df_teeconnie_full.iloc[:350]
    teeconnie_test = df_teeconnie_full.iloc[350:1350]
    print(f"Sampled Teeconnie Data: {len(teeconnie_train)} Train Legits, {len(teeconnie_test)} Test Legits.")

    # 3. Download and Load Legacy Kaggle Dataset
    print("\nDownloading Legacy Kaggle composite dataset from DagsHub S3...")
    csv1 = "data/legacy_composite/composite_train.csv"
    csv2 = "data/legacy_composite/composite_test.csv"
    os.makedirs("data/legacy_composite", exist_ok=True)
    
    try:
        s3_client.download_file(repo_name, csv1, csv1)
        s3_client.download_file(repo_name, csv2, csv2)
    except Exception as e:
        pass
        
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df_legacy_full = pd.concat([df1, df2], ignore_index=True)
    df_legacy_full = df_legacy_full.dropna(subset=["text", "label"])
    df_legacy_full["text"] = df_legacy_full["text"].apply(clean_text)
    
    legacy_scams_full = df_legacy_full[df_legacy_full["label"] == 1].sample(frac=1, random_state=SEED)
    legacy_legits_full = df_legacy_full[df_legacy_full["label"] == 0].sample(frac=1, random_state=SEED)
    
    # We need 850 Scams / 850 Legits for Train
    legacy_train_scams = legacy_scams_full.iloc[:850]
    legacy_train_legits = legacy_legits_full.iloc[:850]
    
    # We need 2000 Scams / 1000 Legits for Test (to pair with 1000 Teeconnie Legits)
    legacy_test_scams = legacy_scams_full.iloc[850:2850]
    legacy_test_legits = legacy_legits_full.iloc[850:1850]
    
    print(f"Sampled Legacy Train Data: {len(legacy_train_scams)} Scams, {len(legacy_train_legits)} Legits.")
    print(f"Sampled Legacy Test Data: {len(legacy_test_scams)} Scams, {len(legacy_test_legits)} Legits.")

    # 4. Construct Explicit Train and Test Datasets
    print("\nConstructing Explicit Train and Test Datasets...")
    train_df = pd.concat([synth_scams, synth_legits, teeconnie_train, legacy_train_scams, legacy_train_legits], ignore_index=True)
    test_df = pd.concat([legacy_test_scams, legacy_test_legits, teeconnie_test], ignore_index=True)
    
    # Shuffle the datasets thoroughly
    train_df = train_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    print(f"\nFINAL TRAIN DATASET: {len(train_df)} rows")
    print(train_df["label"].value_counts().to_dict())
    
    print(f"FINAL TEST DATASET: {len(test_df)} rows")
    print(test_df["label"].value_counts().to_dict())

    # 5. Save and Upload
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("\nSaved locally.")
    
    print(f"\nUploading datasets directly to DagsHub ({repo_id})...")
    try:
        dagshub.upload_files(repo_id, local_path="data/train.csv", remote_path="data/train.csv", bucket=True)
        dagshub.upload_files(repo_id, local_path="data/test.csv", remote_path="data/test.csv", bucket=True)
        print("Successfully uploaded to DagsHub Storage Bucket!")
    except Exception as e:
        print(f"Failed to upload to DagsHub: {e}")

if __name__ == "__main__":
    main()