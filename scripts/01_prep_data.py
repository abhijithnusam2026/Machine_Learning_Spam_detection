"""
Prepares the Ultimate Phase 1.5 Dataset.
Integrates existing DagsHub datasets (LLM Synthetic, Teeconnie, Legacy)
with massive new external datasets (Kaggle Scam Transcripts, SWDA, Thai Call Center).
Splits mathematically into: 70% Train, 10% Validation, 20% Held-Out Eval.
Enforces strict 1:1 Scam/Legit balancing.
"""

import os
import json
import random
import re
import glob
import zipfile
import pandas as pd
import dagshub
from dotenv import load_dotenv

load_dotenv()

SEED = 42
random.seed(SEED)

def clean_text(text):
    text = str(text)
    # Remove speaker tags like 'Caller A:', 'User:', 'Call:', 'innocent:', 'suspect:'
    text = re.sub(r'(?i)(innocent|suspect|caller a|caller b|user|call|speaker \d+):\s*', '', text)
    # Remove bracketed metadata like [laughter], [Step: 1], etc.
    text = re.sub(r'\[.*?\]', '', text)
    # Remove multi-spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def load_swda(swda_dir="data/raw_external/swda"):
    """Extracts and parses the Switchboard Dialog Act Corpus from SWDA zip."""
    swda_zip = os.path.join(swda_dir, "swda.zip")
    if not os.path.exists(swda_zip):
        print(f"SWDA zip not found at {swda_zip}. Skipping...")
        return pd.DataFrame(columns=["text", "label"])
        
    extract_dir = os.path.join(swda_dir, "extracted")
    with zipfile.ZipFile(swda_zip, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
        
    # Find all transcript CSVs in SWDA
    csv_files = glob.glob(os.path.join(extract_dir, "**", "*.csv"), recursive=True)
    
    entries = []
    for csv_file in csv_files:
        try:
            # SWDA CSVs have columns like: swda_filename,ptb_basename,conversation_no,transcript_index,act_tag,caller,pos,text,trees,ptb_treenumbers
            df = pd.read_csv(csv_file)
            if "text" in df.columns:
                # Concatenate all text in the conversation
                conv_text = " ".join(df["text"].astype(str).tolist())
                entries.append(conv_text)
        except Exception:
            continue
            
    df_swda = pd.DataFrame({"text": entries, "label": 0})
    df_swda["text"] = df_swda["text"].apply(clean_text)
    
    # Filter out empty or extremely short ones
    df_swda = df_swda[df_swda["text"].str.len() > 20]
    print(f"Loaded SWDA (Switchboard): {len(df_swda)} Legitimate Conversations.")
    return df_swda

def load_kaggle_datasets():
    """Loads external Kaggle datasets from data/raw_external"""
    scams, legits = [], []
    
    # 1. YouTube Scam Calls (rivalcults)
    yt_csv = "data/raw_external/FullTranscriptData.csv"
    if os.path.exists(yt_csv):
        df_yt = pd.read_csv(yt_csv)
        if "Content" in df_yt.columns:
            for text in df_yt["Content"].tolist():
                scams.append({"text": text, "label": 1})
                
    # 2. Thai Call Center Dataset (jxxn03x)
    thai_csv = "data/raw_external/Dataset.csv"
    if os.path.exists(thai_csv):
        df_thai = pd.read_csv(thai_csv)
        if "Text" in df_thai.columns and "Type" in df_thai.columns:
            for _, row in df_thai.iterrows():
                if str(row["Type"]).lower().strip() == "scam":
                    scams.append({"text": row["Text"], "label": 1})
                else:
                    legits.append({"text": row["Text"], "label": 0})
                    
    # 3. Call Transcripts / Scam Determinations (mealss)
    better_csv = "data/raw_external/BETTER30.csv"
    if os.path.exists(better_csv):
        df_better = pd.read_csv(better_csv)
        if "CONVERSATION_ID" in df_better.columns and "TEXT" in df_better.columns and "LABEL" in df_better.columns:
            grouped = df_better.groupby("CONVERSATION_ID")
            for _, group in grouped:
                conv_text = " ".join(group["TEXT"].astype(str).tolist())
                # If any step is labeled scam/suspicious, mark whole conversation as scam
                is_scam = any(str(lbl).lower() in ["scam", "suspicious"] for lbl in group["LABEL"].tolist())
                if is_scam:
                    scams.append({"text": conv_text, "label": 1})
                else:
                    legits.append({"text": conv_text, "label": 0})
                    
    df_external = pd.DataFrame(scams + legits)
    if not df_external.empty:
        df_external["text"] = df_external["text"].apply(clean_text)
        df_external = df_external[df_external["text"].str.len() > 20] # Filter out too short text
    
    print(f"Loaded Kaggle Datasets: {len([s for s in scams])} Scams, {len([l for l in legits])} Legits.")
    return df_external

def main():
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not (repo_owner and repo_name):
        print("ERROR: DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set in .env")
        return

    if username and password:
        os.environ["DAGSHUB_USER"] = username
        os.environ["DAGSHUB_TOKEN"] = password
        dagshub.auth.add_app_token(password)

    repo_id = f"{repo_owner}/{repo_name}"
    s3_client = dagshub.get_repo_bucket_client(repo_id)

    # --- 1. Load Existing DagsHub Data ---
    
    # LLM Data
    print("Loading existing DagsHub datasets...")
    json_dir = "data/raw_jsons"
    os.makedirs(json_dir, exist_ok=True)
    for fname in ["scam_call_hard_examples_250.json", "scam_call_transcripts_250_combined.json"]:
        try:
            s3_client.download_file(repo_name, f"data/raw_jsons/{fname}", os.path.join(json_dir, fname))
        except: pass
        
    synth_data = []
    for path in glob.glob(f"{json_dir}/*.json"):
        with open(path, "r", encoding="utf-8") as f:
            synth_data.extend(json.load(f))
    df_synth = pd.DataFrame(synth_data)
    df_synth["text"] = df_synth["text"].apply(clean_text)

    # Teeconnie Data
    teeconnie_zip = "data/raw_teeconnie/teeconnie_dataset.zip"
    os.makedirs("data/raw_teeconnie", exist_ok=True)
    try:
        s3_client.download_file(repo_name, "data/raw_teeconnie/teeconnie_dataset.zip", teeconnie_zip)
    except: pass
    
    if os.path.exists(teeconnie_zip):
        with zipfile.ZipFile(teeconnie_zip, 'r') as zipf:
            zipf.extractall("data/raw_teeconnie/")
            
    teeconnie_files = glob.glob("data/raw_teeconnie/**/*", recursive=True)
    nonscam_txt = [f for f in teeconnie_files if f.lower().endswith(".txt") and "non" in f.lower() and "scam" in f.lower()]
    teeconnie_entries = []
    if nonscam_txt:
        content = open(nonscam_txt[0], "r", encoding="utf-8", errors="ignore").read()
        teeconnie_entries = [e.strip() for e in content.split("\n\n") if e.strip()]
        if len(teeconnie_entries) < 5:
            teeconnie_entries = [e.strip() for e in content.split("\n") if e.strip()]
            
    df_teeconnie = pd.DataFrame({"text": teeconnie_entries, "label": 0})
    df_teeconnie["text"] = df_teeconnie["text"].apply(clean_text)

    # Legacy Data
    os.makedirs("data/legacy_composite", exist_ok=True)
    try:
        s3_client.download_file(repo_name, "data/legacy_composite/composite_train.csv", "data/legacy_composite/composite_train.csv")
        s3_client.download_file(repo_name, "data/legacy_composite/composite_test.csv", "data/legacy_composite/composite_test.csv")
    except: pass
    
    df1 = pd.read_csv("data/legacy_composite/composite_train.csv") if os.path.exists("data/legacy_composite/composite_train.csv") else pd.DataFrame()
    df2 = pd.read_csv("data/legacy_composite/composite_test.csv") if os.path.exists("data/legacy_composite/composite_test.csv") else pd.DataFrame()
    df_legacy = pd.concat([df1, df2], ignore_index=True).dropna(subset=["text", "label"])
    df_legacy["text"] = df_legacy["text"].apply(clean_text)
    
    # --- 2. Load New External Data ---
    print("\nLoading new external datasets (Phase 1.5)...")
    df_swda = load_swda()
    df_external_kaggle = load_kaggle_datasets()
    
    # --- 3. Consolidate and Balance ---
    print("\nConsolidating into the Ultimate Dataset...")
    df_all = pd.concat([df_synth, df_teeconnie, df_legacy, df_swda, df_external_kaggle], ignore_index=True)
    df_all = df_all.drop_duplicates(subset=["text"])
    
    all_scams = df_all[df_all["label"] == 1]
    all_legits = df_all[df_all["label"] == 0]
    
    # Mathematical Balancing (Strict 1:1 Ratio)
    min_len = min(len(all_scams), len(all_legits))
    print(f"Total Available Data -> Scams: {len(all_scams)}, Legits: {len(all_legits)}")
    print(f"Balancing strictly to {min_len} rows per class to prevent domain shift...")
    
    balanced_scams = all_scams.sample(n=min_len, random_state=SEED)
    balanced_legits = all_legits.sample(n=min_len, random_state=SEED)
    df_balanced = pd.concat([balanced_scams, balanced_legits], ignore_index=True).sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # --- 4. 70/10/20 Splits ---
    total_size = len(df_balanced)
    train_size = int(total_size * 0.70)
    val_size = int(total_size * 0.10)
    
    train_df = df_balanced.iloc[:train_size]
    val_df = df_balanced.iloc[train_size:train_size + val_size]
    test_df = df_balanced.iloc[train_size + val_size:]
    
    print(f"\n--- Final Dataset Splits ---")
    print(f"Train (70%): {len(train_df)} rows")
    print(f"Val (10%): {len(val_df)} rows")
    print(f"Test (20%): {len(test_df)} rows")
    
    # --- 5. Save & Upload ---
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    val_df.to_csv("data/val.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("\nSaved locally.")
    
    print(f"\nUploading Ultimate Datasets directly to DagsHub ({repo_id})...")
    try:
        dagshub.upload_files(repo_id, local_path="data/train.csv", remote_path="data/train.csv", bucket=True)
        dagshub.upload_files(repo_id, local_path="data/val.csv", remote_path="data/val.csv", bucket=True)
        dagshub.upload_files(repo_id, local_path="data/test.csv", remote_path="data/test.csv", bucket=True)
        print("Successfully uploaded to DagsHub Storage Bucket!")
    except Exception as e:
        print(f"Failed to upload to DagsHub: {e}")

if __name__ == "__main__":
    main()