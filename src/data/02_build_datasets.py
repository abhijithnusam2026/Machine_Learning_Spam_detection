import os
import json
import random
import re
import glob
import zipfile
import pandas as pd

SEED = 42
random.seed(SEED)

def clean_text(text):
    text = str(text)
    text = re.sub(r'(?i)(innocent|suspect):\s*', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def main():
    print("--- Building Unified Modular Datasets ---")
    
    # 1. Load Synthesized JSON data (LLM Data)
    print("Loading synthesized JSON data...")
    json_dir = "data/raw_jsons"
    json_files = ["scam_call_hard_examples_250.json", "scam_call_transcripts_250_combined.json"]
    synth_data = []
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                synth_data.extend(json.load(f))
                
    df_synth = pd.DataFrame(synth_data) if synth_data else pd.DataFrame(columns=["text", "label"])
    df_synth["source_domain"] = "written_text"
    
    # 2. Extract and Load Teeconnie dataset
    print("Processing teeconnie dataset...")
    teeconnie_zip = "data/raw_teeconnie/teeconnie_dataset.zip"
    if os.path.exists(teeconnie_zip):
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
    
    df_teeconnie = pd.DataFrame({"text": entries, "label": [0] * len(entries)})
    df_teeconnie["source_domain"] = "written_text"
    
    # 3. Load Legacy Kaggle Composite (Phishing, SMS, Enron)
    print("Processing Legacy Kaggle composite...")
    csv1 = "data/legacy_composite/composite_train.csv"
    csv2 = "data/legacy_composite/composite_test.csv"
    df_legacy1 = pd.read_csv(csv1) if os.path.exists(csv1) else pd.DataFrame()
    df_legacy2 = pd.read_csv(csv2) if os.path.exists(csv2) else pd.DataFrame()
    df_legacy = pd.concat([df_legacy1, df_legacy2], ignore_index=True)
    df_legacy["source_domain"] = "written_text"
    
    # 4. Load Raw ASR Transcripts
    print("Processing Raw ASR Transcripts...")
    asr_path = "data/raw_asr/raw_asr_transcripts.csv"
    df_asr = pd.read_csv(asr_path) if os.path.exists(asr_path) else pd.DataFrame()
    df_asr["source_domain"] = "spoken_asr"
    
    # Combine ALL sources
    print("\nAssembling massive canonical dataset...")
    all_data = pd.concat([df_synth, df_teeconnie, df_legacy, df_asr], ignore_index=True)
    all_data = all_data.dropna(subset=["text", "label"])
    all_data["text"] = all_data["text"].apply(clean_text)
    
    # Drop duplicates to prevent train/test leakage
    all_data = all_data.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    print(f"Total Unique Samples: {len(all_data)}")
    
    # 5. IMMEDIATELY split off the 20% Global Hold-Out Set
    print("\nCreating stratified Global Hold-Out Set...")
    global_test = all_data.groupby(["label", "source_domain"], group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=SEED))
    
    # The remainder goes to train/val
    train_val_data = all_data.drop(global_test.index)
    
    # Split train/val (80/20 of the remainder)
    global_val = train_val_data.groupby(["label", "source_domain"], group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=SEED))
    global_train = train_val_data.drop(global_val.index)
    
    # Save datasets
    os.makedirs("data/processed", exist_ok=True)
    
    global_test.to_csv("data/processed/global_test.csv", index=False)
    global_train.to_csv("data/processed/global_train.csv", index=False)
    global_val.to_csv("data/processed/global_val.csv", index=False)
    
    print(f"Global Train Set: {len(global_train)} rows")
    print(f"Global Val Set:   {len(global_val)} rows")
    print(f"Global Test Set:  {len(global_test)} rows (Frozen for final evaluation)")
    
    print("\nDataset building complete. Data ready for modeling in data/processed/.")

if __name__ == "__main__":
    main()
