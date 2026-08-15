import os
import pandas as pd

def main():
    print("--- Building Modular Datasets with Global Hold-out ---")
    
    # Define paths based on what 00_download_data.py pulled
    p1_train_path = "data/raw/phase1.5/train.csv"
    p1_val_path = "data/raw/phase1.5/val.csv"
    p1_test_path = "data/raw/phase1.5/test.csv"
    
    p2_train_path = "data/raw/phase2_asr/train.csv"
    p2_val_path = "data/raw/phase2_asr/val.csv"
    p2_test_path = "data/raw/phase2_asr/test.csv"
    
    # Load all files
    p1_train = pd.read_csv(p1_train_path)
    p1_val = pd.read_csv(p1_val_path)
    p1_test = pd.read_csv(p1_test_path)
    
    p2_train = pd.read_csv(p2_train_path)
    p2_val = pd.read_csv(p2_val_path)
    p2_test = pd.read_csv(p2_test_path)
    
    # Tag them so we can selectively train models
    p1_train['source_domain'] = 'written_text'
    p1_val['source_domain'] = 'written_text'
    p1_test['source_domain'] = 'written_text'
    
    p2_train['source_domain'] = 'spoken_asr'
    p2_val['source_domain'] = 'spoken_asr'
    p2_test['source_domain'] = 'spoken_asr'
    
    # 1. Create Global Hold-out Test Set
    # We combine the Test set from Phase 1.5 (Written) and Phase 2 (Spoken)
    # This guarantees that the final evaluation set has both text and audio domains,
    # and guarantees no model has ever seen this data during any training step.
    global_test = pd.concat([p1_test, p2_test]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 2. Create Train and Val sets
    global_train = pd.concat([p1_train, p2_train]).sample(frac=1, random_state=42).reset_index(drop=True)
    global_val = pd.concat([p1_val, p2_val]).sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save them to data/processed
    os.makedirs("data/processed", exist_ok=True)
    
    global_test.to_csv("data/processed/global_test.csv", index=False)
    global_train.to_csv("data/processed/global_train.csv", index=False)
    global_val.to_csv("data/processed/global_val.csv", index=False)
    
    print(f"Global Train Set: {len(global_train)} rows")
    print(f"Global Val Set: {len(global_val)} rows")
    print(f"Global Test Set (Hold-out): {len(global_test)} rows")
    
    print("\nDataset building complete. Data ready for modeling in data/processed/.")

if __name__ == "__main__":
    main()
