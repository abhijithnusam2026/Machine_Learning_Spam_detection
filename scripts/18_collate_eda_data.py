import os
import pandas as pd
from dotenv import load_dotenv
import dagshub

def main():
    print("--- Collating Data for EDA ---")
    
    # 1. Load Phase 1.5 Universal Data
    print("Loading Phase 1.5 data...")
    p1_train = pd.read_csv("data/phase1.5/train.csv")
    p1_val = pd.read_csv("data/phase1.5/val.csv")
    p1_test = pd.read_csv("data/phase1.5/test.csv")
    
    # Combine and tag
    p1_combined = pd.concat([p1_train, p1_val, p1_test])
    p1_combined['source_phase'] = 'Phase_1_and_1.5_Written_Text'
    
    # 2. Load Phase 2 ASR Transcript Data
    print("Loading Phase 2 ASR Transcript data...")
    p2_data = pd.read_csv("data/phase2_asr/raw_asr_transcripts.csv")
    p2_data['source_phase'] = 'Phase_2_Spoken_ASR_Transcripts'
    
    # Ensure columns align (text, label, source_phase)
    cols_to_keep = ['text', 'label', 'source_phase']
    
    final_df = pd.concat([
        p1_combined[cols_to_keep], 
        p2_data[cols_to_keep]
    ])
    
    print(f"Total collated rows for EDA: {len(final_df)}")
    
    # 3. Save locally
    out_path = "data/collated_for_eda.csv"
    final_df.to_csv(out_path, index=False)
    print(f"Saved locally to {out_path}")
    
    # 4. Upload to DagsHub S3
    print("Uploading to DagsHub S3: data/collated_for_eda/collated_data.csv")
    load_dotenv()
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    name = os.getenv("DAGSHUB_REPO_NAME")
    
    if owner and name:
        s3 = dagshub.get_repo_bucket_client(f"{owner}/{name}")
        s3.upload_file(out_path, name, "data/collated_for_eda/collated_data.csv")
        print("Upload complete!")
        print(f"The data is available on S3 at: s3://{name}/data/collated_for_eda/collated_data.csv")
    else:
        print("DagsHub credentials not found. File saved locally only.")

if __name__ == "__main__":
    main()
