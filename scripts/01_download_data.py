import os
import sys
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def main():
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)
    
    train_path = os.path.join(raw_dir, "v2_composite_train.csv")
    test_path = os.path.join(raw_dir, "v2_composite_test.csv")
    
    print("Downloading BothBosu/multi-agent-scam-conversation from Hugging Face...")
    try:
        # Load dataset
        ds = load_dataset("BothBosu/multi-agent-scam-conversation")
        
        # Format Train
        if 'train' in ds:
            df_train = ds['train'].to_pandas()
            df_train = df_train.rename(columns={"dialogue": "text", "labels": "label"})
            df_train = df_train[["text", "label", "type", "personality"]]
            df_train.to_csv(train_path, index=False)
            print(f"Saved V2 Train: {len(df_train)} rows.")
            
        # Format Test
        if 'test' in ds:
            df_test = ds['test'].to_pandas()
            df_test = df_test.rename(columns={"dialogue": "text", "labels": "label"})
            df_test = df_test[["text", "label", "type", "personality"]]
            df_test.to_csv(test_path, index=False)
            print(f"Saved V2 Test: {len(df_test)} rows.")
            
        print("Download and formatting complete.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
