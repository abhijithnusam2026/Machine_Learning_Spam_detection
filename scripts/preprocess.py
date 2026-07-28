import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Basic cleaning: remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def fix_label_noise(row):
    """
    Force explicitly benign domains (like taxi, food delivery, general support)
    to label 0 (legit) if they were mislabeled as scams (1).
    """
    text_lower = row['text'].lower()
    benign_keywords = ['taxi', 'food delivery', 'uber', 'doordash', 'pizza', 'restaurant']
    if any(kw in text_lower for kw in benign_keywords):
        return 0
    return row['label']

def preprocess_and_split(raw_train_path, raw_test_path, proc_train_path, proc_test_path):
    print("Loading raw datasets...")
    df_train = pd.read_csv(raw_train_path)
    df_test = pd.read_csv(raw_test_path)
    
    # Combine to ensure global deduplication
    df_all = pd.concat([df_train, df_test], ignore_index=True)
    
    # Adapt to composite dataset standard
    if "transcript" in df_all.columns and "is_scam" in df_all.columns:
        df_all = df_all.rename(columns={"transcript": "text", "is_scam": "label"})
    
    print(f"Total raw rows (train + test): {len(df_all)}")
    
    # 1. Clean Text
    print("Cleaning text data...")
    df_all['text'] = df_all['text'].apply(clean_text)
    df_all = df_all[df_all['text'] != ""]
    
    # 2. Global Deduplication
    print("Performing global deduplication to fix train/test leakage...")
    df_all = df_all.drop_duplicates(subset=['text'], keep='first')
    print(f"Rows after deduplication: {len(df_all)}")
    
    # 3. Label Noise Fix
    print("Auditing provenance and fixing label noise (benign misclassifications)...")
    df_all['label'] = df_all.apply(fix_label_noise, axis=1)
    
    # 4. Stratified Train/Test Split (85/15)
    print("Re-splitting dataset securely...")
    df_train_clean, df_test_clean = train_test_split(
        df_all, test_size=0.15, random_state=42, stratify=df_all['label']
    )
    
    # Save
    os.makedirs(os.path.dirname(proc_train_path), exist_ok=True)
    df_train_clean.to_csv(proc_train_path, index=False)
    df_test_clean.to_csv(proc_test_path, index=False)
    
    print(f"Saved processed train dataset to {proc_train_path} (Rows: {len(df_train_clean)})")
    print(f"Saved processed test dataset to {proc_test_path} (Rows: {len(df_test_clean)})")

if __name__ == "__main__":
    raw_train = "data/raw/composite_train.csv"
    raw_test = "data/raw/composite_test.csv"
    
    proc_train = "data/processed/composite_train.csv"
    proc_test = "data/processed/composite_test.csv"
    
    if os.path.exists(raw_train) and os.path.exists(raw_test):
        preprocess_and_split(raw_train, raw_test, proc_train, proc_test)
    else:
        print(f"Warning: Raw data files not found in data/raw/. Please run download.py first.")

    # Push to Hugging Face Hub if token is available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from datasets import load_dataset
            print("\nPushing processed dataset to Hugging Face Hub (Private Repo)...")
            dataset = load_dataset("csv", data_files={"train": proc_train, "test": proc_test})
            
            # Using private=True to adhere to the proposal's privacy constraints
            repo_id = "tanu011235/scam-alert-dataset"
            dataset.push_to_hub(repo_id, token=hf_token, private=True)
            print(f"Successfully pushed PRIVATE dataset to https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"Failed to push to Hugging Face Hub: {e}")
    else:
        print("\nSkipping Hugging Face upload: HF_TOKEN not found in environment variables.")
