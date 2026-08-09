"""
Phase 2: Hybrid ASR Data Acquisition
- Scams (Label 1): Fetches pre-transcribed (Whisper) robocall metadata from NCSU github.
- Legit (Label 0): Fetches 'english_transcription' from Hugging Face PolyAI/minds14 dataset.
Aggregates, balances (1:1), and splits into Train/Val/Test/PTQ.
"""

import os
import io
import requests
import pandas as pd
import dagshub
from dotenv import load_dotenv
from datasets import load_dataset
from sklearn.model_selection import train_test_split

def fetch_ncsu_scam_data():
    """
    Downloads the metadata.csv from the NCSU Robocall dataset repository.
    Extracts the 'transcript' column as Label 1 (Scam).
    """
    print("Fetching NCSU Robocall Dataset (Scams)...")
    url = "https://raw.githubusercontent.com/wspr-ncsu/robocall-audio-dataset/main/metadata.csv"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Read CSV from string
        df = pd.read_csv(io.StringIO(response.text))
        
        # We only need the transcription
        transcripts = df['transcript'].dropna().tolist()
        
        results = []
        for text in transcripts:
            # Filter extremely short artifacts
            if len(str(text).split()) > 10:
                results.append({
                    "source": "ncsu_robocall",
                    "text": str(text),
                    "label": 1
                })
                
        print(f"  [SUCCESS] Extracted {len(results)} scam transcripts.")
        return results
    except Exception as e:
        print(f"  [FAILED] Could not fetch NCSU dataset: {e}")
        return []

def fetch_hf_legit_data():
    """
    Fetches the PolyAI/minds14 dataset from Hugging Face (en-US, en-GB, en-AU).
    Drops the audio column immediately to prevent decoding errors.
    Returns a list of dictionaries with text and label 0.
    """
    results = []
    print(f"\nFetching PolyAI/minds14 Dataset (Legit)...")
    
    locales = ["en-US", "en-GB", "en-AU"]
    
    for loc in locales:
        try:
            print(f"  Loading subset: {loc}...")
            # Load dataset and immediately drop the 'audio' column to prevent torchcodec decoding crashes
            ds = load_dataset("PolyAI/minds14", name=loc, split="train", trust_remote_code=True)
            ds = ds.remove_columns(["audio", "path", "intent_class", "lang_id"])
            
            for row in ds:
                text = row.get("english_transcription", "")
                if text and len(str(text).split()) > 5:
                    results.append({
                        "source": f"minds14_{loc}",
                        "text": str(text),
                        "label": 0
                    })
            print(f"  [SUCCESS] Extracted transcripts from {loc}.")
        except Exception as e:
            print(f"  [FAILED] {loc}: {e}")
            
    print(f"  [SUCCESS] Total legit transcripts extracted: {len(results)}")
    return results

def split_and_upload(df, output_dir="data/phase2_asr"):
    """
    Splits the data into Train(60), Val(10), Test(20), PTQ(10) and uploads to S3.
    """
    print("\n--- Splitting Dataset ---")
    
    if len(df) < 4:
        print("ERROR: Not enough data to split into 4 buckets. Skipping split.")
        return
        
    train_df, rest_df = train_test_split(df, test_size=0.4, stratify=df["label"], random_state=42)
    test_df, val_ptq_df = train_test_split(rest_df, test_size=0.5, stratify=rest_df["label"], random_state=42)
    val_df, ptq_df = train_test_split(val_ptq_df, test_size=0.5, stratify=val_ptq_df["label"], random_state=42)
    
    print("\n=========================================")
    print("      DATASET SPLIT SUMMARY STATS")
    print("=========================================")
    
    def print_stats(name, dataframe):
        total = len(dataframe)
        scams = len(dataframe[dataframe['label'] == 1])
        legits = len(dataframe[dataframe['label'] == 0])
        print(f"--- {name} ---")
        print(f"Total Rows: {total}")
        if total > 0:
            print(f"Class Balance: {scams} Scams ({scams/total:.1%}) | {legits} Legits ({legits/total:.1%})")
        print("")

    print_stats("Train (Fine-Tuning, 60%)", train_df)
    print_stats("Test (Evaluation, 20%)", test_df)
    print_stats("Validation (10%)", val_df)
    print_stats("PTQ Calibration (10%)", ptq_df)
    print("=========================================\n")
    
    # Save locally
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")
    ptq_path = os.path.join(output_dir, "ptq_calibration.csv")
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    ptq_df.to_csv(ptq_path, index=False)
    
    # Upload to DagsHub S3
    print("\n--- Uploading to DagsHub S3 ---")
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if repo_owner and repo_name:
        try:
            dagshub.auth.add_app_token(os.getenv("MLFLOW_TRACKING_PASSWORD"))
            s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
            
            for file_path in [train_path, val_path, test_path, ptq_path]:
                print(f"Uploading {file_path} to S3...")
                s3_client.upload_file(file_path, repo_name, file_path)
            print("Successfully backed up Phase 2 datasets to DagsHub S3!")
        except Exception as e:
            print(f"Failed to upload to S3: {e}")
    else:
        print("Skipping DagsHub S3 upload (missing env vars).")

def main():
    print("Initializing Phase 2 Hybrid ASR Data Acquisition...\n")
    
    # 1. Fetch 1400+ Scams from NCSU
    scam_data = fetch_ncsu_scam_data()
    
    # 2. Fetch Legit Transcripts from Hugging Face
    legit_data = fetch_hf_legit_data()
    
    # 3. Balance Dataset (Downsample majority class)
    min_size = min(len(scam_data), len(legit_data))
    print(f"\nBalancing Dataset: Downsampling to {min_size} files per class.")
    
    if min_size == 0:
        print("ERROR: One of the classes has 0 transcripts. Cannot proceed.")
        return
        
    balanced_data = scam_data[:min_size] + legit_data[:min_size]
    
    df = pd.DataFrame(balanced_data)
    
    # Save the raw aggregated dataset
    output_dir = "data/phase2_asr"
    os.makedirs(output_dir, exist_ok=True)
    raw_path = os.path.join(output_dir, "raw_asr_transcripts.csv")
    df.to_csv(raw_path, index=False)
    print(f"Saved {len(df)} raw transcripts to {raw_path}")
    
    # Generate splits and upload
    split_and_upload(df)

if __name__ == "__main__":
    main()
