"""
Phase 2: ASR Transcription Pipeline
Uses Hugging Face Whisper to transcribe the 16kHz WAV files from Phase 2,
splits the dataset (including a PTQ holdout), and uploads to DagsHub S3.
"""

import os
import argparse
import pandas as pd
import torch
import dagshub
from dotenv import load_dotenv
from transformers import pipeline
from sklearn.model_selection import train_test_split

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def transcribe_audio(metadata_path, output_path, model_id="openai/whisper-base"):
    """
    Transcribes audio files using Whisper and saves to CSV.
    """
    print(f"Loading Whisper model: {model_id}...")
    device = get_device()
    
    # Initialize Hugging Face pipeline for ASR
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=0 if device == "cuda" else -1, # Simplistic device mapping
        chunk_length_s=30, # Whisper processes in 30s chunks
    )
    
    df_meta = pd.read_csv(metadata_path)
    transcripts = []
    labels = []
    
    print(f"Starting transcription of {len(df_meta)} files...")
    
    for idx, row in df_meta.iterrows():
        audio_path = row["file_path"]
        label = row["label"]
        
        if not os.path.exists(audio_path):
            print(f"WARNING: Audio file missing: {audio_path}")
            continue
            
        print(f"[{idx+1}/{len(df_meta)}] Transcribing {os.path.basename(audio_path)}...")
        
        try:
            result = asr_pipeline(audio_path)
            text = result["text"].strip()
            
            transcripts.append(text)
            labels.append(label)
        except Exception as e:
            print(f"Failed to transcribe {audio_path}: {e}")
            
    df_results = pd.DataFrame({
        "text": transcripts,
        "label": labels
    })
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved {len(df_results)} raw transcripts to {output_path}")
    
    return df_results

def split_and_upload(df, output_dir="data/phase2_asr"):
    """
    Splits the data into Train(60), Val(10), Test(20), PTQ(10) and uploads to S3.
    """
    print("\n--- Splitting Dataset ---")
    # 1. Split off Train (60%) vs Rest (40%)
    train_df, rest_df = train_test_split(df, test_size=0.4, stratify=df["label"], random_state=42)
    
    # 2. Split Rest (40%) into Test (20%) vs Val+PTQ (20%)
    test_df, val_ptq_df = train_test_split(rest_df, test_size=0.5, stratify=rest_df["label"], random_state=42)
    
    # 3. Split Val+PTQ (20%) into Val (10%) and PTQ (10%)
    val_df, ptq_df = train_test_split(val_ptq_df, test_size=0.5, stratify=val_ptq_df["label"], random_state=42)
    
    print(f"Train (Fine-Tuning): {len(train_df)} rows")
    print(f"Validation:          {len(val_df)} rows")
    print(f"Test (Evaluation):   {len(test_df)} rows")
    print(f"PTQ Calibration:     {len(ptq_df)} rows")
    
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
    
    # Upload to DagsHub
    print("\n--- Uploading to DagsHub S3 ---")
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if repo_owner and repo_name:
        try:
            dagshub.auth.add_app_token(os.getenv("MLFLOW_TRACKING_PASSWORD"))
            s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
            
            for file_path in [train_path, val_path, test_path, ptq_path]:
                remote_path = file_path # Keep same directory structure in S3
                print(f"Uploading {file_path} to s3://{repo_name}/{remote_path}...")
                s3_client.upload_file(file_path, repo_name, remote_path)
            print("Successfully uploaded Phase 2 datasets to S3!")
        except Exception as e:
            print(f"Failed to upload to S3: {e}")
    else:
        print("Skipping DagsHub S3 upload (missing env vars).")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="openai/whisper-base", help="Hugging Face Whisper model ID")
    args = parser.parse_args()
    
    meta_path = "data/raw_audio/audio_metadata.csv"
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found. Please run 01a_download_audio.py first.")
        return
        
    raw_output = "data/phase2_asr/raw_asr_transcripts.csv"
    df = transcribe_audio(meta_path, raw_output, args.model)
    
    if len(df) > 3: # Only split if we have enough data (avoid crashing on test run)
        split_and_upload(df)
    else:
        print("Not enough data to perform Train/Val/Test/PTQ splits (Need >3 files). Skipping split.")

if __name__ == "__main__":
    main()
