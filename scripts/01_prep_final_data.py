"""
Prepares the final dataset by combining 500 JSON examples (highly imbalanced)
with 350 legitimate transcripts sampled from the AIxBlock real-world call center dataset.
This yields a perfectly balanced 850-row dataset for training.
"""

import os
import json
import random
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

def main():
    print("Loading synthesized JSON data...")
    json_dir = "data/synthesized_data"
    json_files = ["scam_call_hard_examples_250_fable.json", "scam_call_transcripts_250_combined_gpt5.6.json"]
    
    synth_data = []
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        if os.path.exists(path):
            with open(path, "r") as f:
                data = json.load(f)
                synth_data.extend(data)
        else:
            print(f"Warning: {path} not found.")

    df_synth = pd.DataFrame(synth_data)
    print(f"Loaded {len(df_synth)} synthetic examples.")
    print("Synthetic label distribution:", df_synth["label"].value_counts().to_dict())

    # We have 425 scams and 75 legits in the synthetic data.
    # We need 350 more legits to balance it to 425/425.
    print("\nLoading teeconnie non-scam real-world dataset to pad the legit class...")
    import kagglehub
    import glob
    teeconnie_path = kagglehub.dataset_download("teeconnie/scam-and-non-scam-call-conversation-dataset")
    teeconnie_files = glob.glob(teeconnie_path + "/**/*", recursive=True)
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
    print("\nSaved robust merged datasets to data/train.csv and data/test.csv")

if __name__ == "__main__":
    main()
