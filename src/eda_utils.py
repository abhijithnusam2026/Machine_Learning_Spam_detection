"""Utility functions for Exploratory Data Analysis and Leakage Auditing."""

import pandas as pd

def get_overlap_count(df_train: pd.DataFrame, df_test: pd.DataFrame, text_col: str = "text") -> int:
    """Calculate the exact number of overlapping transcripts between train and test."""
    train_texts = set(df_train[text_col].dropna().tolist())
    test_texts = set(df_test[text_col].dropna().tolist())
    return len(train_texts.intersection(test_texts))

def get_quote_artifact_stats(df: pd.DataFrame, text_col: str = "text", label_col: str = "label") -> pd.DataFrame:
    """
    Check how many transcripts start with a specific quote artifact,
    and measure the % of those that are labeled as scams.
    """
    stats = []
    total = len(df)
    
    # Check single quotes
    sq = df[df[text_col].str.startswith("'")]
    if len(sq) > 0:
        stats.append({
            "artifact": "Starts with '",
            "n_rows": len(sq),
            "scam_percentage": sq[label_col].mean() * 100
        })
        
    # Check double quotes
    dq = df[df[text_col].str.startswith('"')]
    if len(dq) > 0:
        stats.append({
            "artifact": 'Starts with "',
            "n_rows": len(dq),
            "scam_percentage": dq[label_col].mean() * 100
        })
        
    # Check no quotes
    nq = df[~df[text_col].str.startswith("'") & ~df[text_col].str.startswith('"')]
    if len(nq) > 0:
        stats.append({
            "artifact": "No leading quote",
            "n_rows": len(nq),
            "scam_percentage": nq[label_col].mean() * 100
        })
        
    return pd.DataFrame(stats)
