"""Tests to prove data leakage has been plugged in the processed dataset."""

import os
import pandas as pd
from src.eda_utils import get_overlap_count, get_quote_artifact_stats

def test_no_train_test_overlap():
    train_path = "data/processed/composite_train.csv"
    test_path = "data/processed/composite_test.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        # Skip if data hasn't been generated yet
        return
        
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    overlap = get_overlap_count(df_train, df_test)
    assert overlap == 0, f"Leakage detected! Found {overlap} overlapping rows between train and test."

def test_no_quote_artifacts_in_processed_data():
    train_path = "data/processed/composite_train.csv"
    if not os.path.exists(train_path):
        return
        
    df_train = pd.read_csv(train_path)
    stats = get_quote_artifact_stats(df_train)
    
    # We should only have the "No leading quote" artifact left
    # or if we have others, they shouldn't perfectly correlate to a class.
    # But theoretically, they should be fully stripped.
    artifacts_found = stats["artifact"].tolist()
    assert "Starts with '" not in artifacts_found, "Single quotes were not fully stripped!"
    assert 'Starts with "' not in artifacts_found, "Double quotes were not fully stripped!"
