"""
Downloads and merges multiple scam detection datasets into a composite dataset.
Ensures rigorous stratified splits (by label AND source).
"""

import os
import glob
import random
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import kagglehub

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

INCLUDE_COMPOSITE_DATASET = True
INCLUDE_FRAUD_INDIA_DATASET = False

def to_df(ds_dict, source_name):
    frames = []
    for split in ds_dict.keys():
        d = ds_dict[split].to_pandas()
        d["source_dataset"] = source_name
        d["source_split"] = split
        frames.append(d)
    return pd.concat(frames, ignore_index=True)

def load_txt_samples(filepath, label):
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    entries = [e.strip() for e in content.split("\n\n") if e.strip()]
    if len(entries) < 5:
        entries = [e.strip() for e in content.split("\n") if e.strip()]
    return pd.DataFrame({"text": entries, "label": label})

def main():
    print("Loading BothBosu HF datasets...")
    ds1 = load_dataset("BothBosu/multi-agent-scam-conversation")
    ds2 = load_dataset("BothBosu/scam-dialogue")
    
    df1 = to_df(ds1, "multi-agent-scam-conversation")
    df2 = to_df(ds2, "scam-dialogue")
    
    print("Downloading teeconnie dataset from Kaggle...")
    teeconnie_path = kagglehub.dataset_download("teeconnie/scam-and-non-scam-call-conversation-dataset")
    teeconnie_files = glob.glob(teeconnie_path + "/**/*", recursive=True)
    
    scam_txt = [f for f in teeconnie_files if f.lower().endswith(".txt") and "non" not in f.lower() and "scam" in f.lower()]
    nonscam_txt = [f for f in teeconnie_files if f.lower().endswith(".txt") and "non" in f.lower() and "scam" in f.lower()]
    
    df3_parts = []
    if scam_txt:
        df3_parts.append(load_txt_samples(scam_txt[0], label=1))
    if nonscam_txt:
        df3_parts.append(load_txt_samples(nonscam_txt[0], label=0))

    if df3_parts:
        df3 = pd.concat(df3_parts, ignore_index=True)
        df3["source_dataset"] = "teeconnie-scam-nonscam"
        df3["source_split"] = "all"
    else:
        df3 = pd.DataFrame(columns=["text", "label", "source_dataset", "source_split"])
        
    df4 = pd.DataFrame(columns=["text", "label", "source_dataset", "source_split"])
    if INCLUDE_COMPOSITE_DATASET:
        print("Downloading ibrahimbagwan12 composite dataset from Kaggle...")
        composite_path = kagglehub.dataset_download("ibrahimbagwan12/composite-scam-transcript-dataset")
        csv_files = glob.glob(composite_path + "/**/*.csv", recursive=True)
        frames = []
        for f in csv_files:
            d = pd.read_csv(f)
            if "text" in d.columns and "label" in d.columns:
                frames.append(d[["text", "label"]])
        if frames:
            df4 = pd.concat(frames, ignore_index=True)
            df4["source_dataset"] = "composite-scam-transcript"
            df4["source_split"] = "all"
            
    df5 = pd.DataFrame(columns=["text", "label", "source_dataset", "source_split"])
    if INCLUDE_FRAUD_INDIA_DATASET:
        print("Downloading narayanyadav fraud-call-india dataset from Kaggle...")
        fraud_india_path = kagglehub.dataset_download("narayanyadav/fraud-call-india-dataset")
        fraud_india_files = glob.glob(fraud_india_path + "/**/*.csv", recursive=True)
        if fraud_india_files:
            d5 = pd.read_csv(fraud_india_files[0])
            d5 = d5.rename(columns={"clue": "text", "type": "label"})
            d5["label"] = d5["label"].astype(str).str.strip().str.lower().map(lambda v: 1 if v == "fraud" else 0)
            d5["source_dataset"] = "fraud-call-india"
            d5["source_split"] = "all"
            df5 = d5

    # Merge
    rename_map_1 = {"dialogue": "text", "labels": "label"}
    rename_map_2 = {"dialogue": "text", "labels": "label"}

    df1 = df1.rename(columns=rename_map_1)
    df2 = df2.rename(columns=rename_map_2)

    keep_cols = ["text", "label", "type", "source_dataset", "source_split"]
    df1 = df1[[c for c in keep_cols if c in df1.columns]]
    df2 = df2[[c for c in keep_cols if c in df2.columns]]
    df3_aligned = df3.reindex(columns=keep_cols)
    df4_aligned = df4.reindex(columns=keep_cols)
    df5_aligned = df5.reindex(columns=keep_cols)

    df_all = pd.concat([df1, df2, df3_aligned, df4_aligned, df5_aligned], ignore_index=True)
    df_all = df_all.dropna(subset=["text", "label"]).reset_index(drop=True)
    df_all["label"] = df_all["label"].astype(int)
    
    print(f"\nCombined shape: {df_all.shape}")
    print("Rows per source dataset:")
    print(df_all["source_dataset"].value_counts())
    
    # Stratified split
    df_all["strata"] = df_all["label"].astype(str) + "_" + df_all["source_dataset"]
    train_df, test_df = train_test_split(df_all, test_size=0.2, random_state=SEED, stratify=df_all["strata"])
    train_df = train_df.drop(columns=["strata"]).reset_index(drop=True)
    test_df = test_df.drop(columns=["strata"]).reset_index(drop=True)
    
    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)
    print("\nSaved robust merged datasets to data/train.csv and data/test.csv")

if __name__ == "__main__":
    main()
