"""
Phase 2: YouTube Transcript API Pipeline
Directly fetches auto-generated (or manual) captions from YouTube using 
youtube-transcript-api. This acts as our source of "messy ASR" text 
without requiring local Whisper execution.
"""

import os
import pandas as pd
import dagshub
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from sklearn.model_selection import train_test_split

def fetch_transcripts(video_ids, label):
    """
    Fetches the transcript for a list of video IDs.
    Returns a list of dictionaries with text and label.
    """
    results = []
    print(f"Fetching transcripts for {len(video_ids)} videos (Label: {label})...")
    
    for vid in video_ids:
        try:
            # This fetches the transcript as a list of dictionaries (text, start, duration)
            transcript_list = YouTubeTranscriptApi.get_transcript(vid)
            
            # Combine all pieces of text into a single document
            full_text = " ".join([item['text'] for item in transcript_list])
            
            if len(full_text.split()) > 50: # Filter out extremely short errors
                results.append({
                    "video_id": vid,
                    "text": full_text,
                    "label": label
                })
                print(f"  [SUCCESS] {vid}")
            else:
                print(f"  [SKIPPED] {vid} - Transcript too short")
                
        except Exception as e:
            print(f"  [FAILED] {vid}: {e}")
            
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
    print("Initializing Phase 2 YouTube Transcript API Pipeline...\n")
    
    # Provide the 11-character YouTube video IDs
    # E.g. https://www.youtube.com/watch?v=5Vj-b7-Tf1M -> 5Vj-b7-Tf1M
    
    scam_ids = [
        "5Vj-b7-Tf1M", # Kitboga: The Angriest Scammer
        "1F_47Z9e3L4"  # Scammer Payback: Destroying Scammer
    ]
    
    legit_ids = [
        "jNQXAC9IVRw", # Me at the zoo (Has captions)
        "QH2-TGUlwu4", # Nyan Cat (No captions, will test failure handling)
        "BaW_jenozKc"  # BBC News
    ]
    
    scam_data = fetch_transcripts(scam_ids, label=1)
    legit_data = fetch_transcripts(legit_ids, label=0)
    
    # Downsample to enforce strict 1:1 balance
    min_size = min(len(scam_data), len(legit_data))
    
    print(f"\nBalancing Dataset: Downsampling to {min_size} files per class.")
    balanced_data = scam_data[:min_size] + legit_data[:min_size]
    
    if not balanced_data:
        print("ERROR: No valid transcripts fetched. Cannot proceed.")
        return
        
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
