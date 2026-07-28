import os
import pandas as pd
import re

def clean_text(text):
    if pd.isna(text):
        return ""
    text = str(text)
    # Basic cleaning: remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess_dataset(input_path, output_path):
    print(f"Reading {input_path}...")
    df = pd.read_csv(input_path)
    
    # Adapt to composite dataset standard
    if "transcript" in df.columns and "is_scam" in df.columns:
        df = df.rename(columns={"transcript": "text", "is_scam": "label"})
    
    print("Cleaning text data...")
    df['text'] = df['text'].apply(clean_text)
    
    # Remove empty rows
    df = df[df['text'] != ""]
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved processed dataset to {output_path} (Rows: {len(df)})")

if __name__ == "__main__":
    raw_train = "data/raw/composite_train.csv"
    raw_test = "data/raw/composite_test.csv"
    
    proc_train = "data/processed/composite_train.csv"
    proc_test = "data/processed/composite_test.csv"
    
    if os.path.exists(raw_train):
        preprocess_dataset(raw_train, proc_train)
    else:
        print(f"Warning: {raw_train} not found.")
        
    if os.path.exists(raw_test):
        preprocess_dataset(raw_test, proc_test)
    else:
        print(f"Warning: {raw_test} not found.")

    # Push to Hugging Face Hub if token is available
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        try:
            from datasets import load_dataset
            print("\nPushing processed dataset to Hugging Face Hub...")
            dataset = load_dataset("csv", data_files={"train": proc_train, "test": proc_test})
            
            # Using tanu320 as the default username as inferred.
            # Users can change this in the script if needed.
            repo_id = "tanu320/scam-alert-dataset"
            dataset.push_to_hub(repo_id, token=hf_token)
            print(f"Successfully pushed dataset to https://huggingface.co/datasets/{repo_id}")
        except Exception as e:
            print(f"Failed to push to Hugging Face Hub: {e}")
    else:
        print("\nSkipping Hugging Face upload: HF_TOKEN not found in environment variables.")
