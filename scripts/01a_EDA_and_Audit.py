"""
Runs rigorous exploratory data analysis and data quality auditing on the merged dataset.
Checks for source predictability (leakage), duplicates, and named-entity bias via signed log-odds.
"""

import os
import argparse
import numpy as np
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def signed_log_odds(df, text_col="text", label_col="label", top_n=20, min_count=5):
    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(df[text_col].astype(str))
    vocab = np.array(vec.get_feature_names_out())

    counts_pos = np.asarray(X[df[label_col].values == 1].sum(axis=0)).flatten()
    counts_neg = np.asarray(X[df[label_col].values == 0].sum(axis=0)).flatten()

    mask = (counts_pos + counts_neg) >= min_count
    vocab, counts_pos, counts_neg = vocab[mask], counts_pos[mask], counts_neg[mask]

    total_pos, total_neg = counts_pos.sum(), counts_neg.sum()
    log_odds = np.log((counts_pos + 1) / (total_pos - counts_pos + 1)) - \
               np.log((counts_neg + 1) / (total_neg - counts_neg + 1))

    order = np.argsort(log_odds)
    top_legit = vocab[order[:top_n]]
    top_scam = vocab[order[::-1][:top_n]]
    return top_scam, top_legit

def get_quote_artifact_stats(df):
    d = df.copy()
    d["starts_with_quote"] = d["text"].astype(str).str.strip().str.startswith(('"', "'"))
    d["num_quoted_segments"] = d["text"].astype(str).apply(
        lambda t: len(re.findall(r'["\'][^"\']{5,}["\']', t))
    )
    return d.groupby("label")[["starts_with_quote", "num_quoted_segments"]].mean()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/train.csv")
    parser.add_argument("--test_path", type=str, default="data/test.csv")
    args = parser.parse_args()

    if not os.path.exists(args.data_path) or not os.path.exists(args.test_path):
        print("ERROR: Run scripts/01_download_data.py first to generate the datasets.")
        return

    train_df = pd.read_csv(args.data_path)
    test_df = pd.read_csv(args.test_path)
    df_all = pd.concat([train_df, test_df], ignore_index=True)

    print("\n" + "="*50)
    print("1. SOURCE-ORIGIN PREDICTABILITY CHECK (LEAKAGE)")
    print("="*50)
    print(pd.crosstab(df_all["source_dataset"], df_all["label"], normalize="index"))
    
    X_src = TfidfVectorizer(max_features=5000, stop_words="english").fit_transform(df_all["text"].astype(str))
    y_src = df_all["source_dataset"].values
    
    # Needs at least 2 classes to stratify
    if len(np.unique(y_src)) > 1:
        src_train_idx, src_test_idx = train_test_split(
            np.arange(len(df_all)), test_size=0.2, random_state=42, stratify=y_src
        )
        src_clf = LogisticRegression(max_iter=1000, multi_class="auto").fit(X_src[src_train_idx], y_src[src_train_idx])
        src_acc = src_clf.score(X_src[src_test_idx], y_src[src_test_idx])
        print(f"\nAccuracy of predicting SOURCE DATASET from text alone: {src_acc:.2%}")
        naive_baseline = df_all['source_dataset'].value_counts(normalize=True).max()
        print(f"(A naive majority-class baseline would score: {naive_baseline:.2%})")
    
    print("\n" + "="*50)
    print("2. EXACT AND NEAR DUPLICATE CHECK")
    print("="*50)
    train_texts = set(train_df["text"].astype(str))
    test_texts = set(test_df["text"].astype(str))
    overlap = train_texts.intersection(test_texts)
    print(f"Exact train/test overlap: {len(overlap)} rows")
    
    vec = TfidfVectorizer(max_features=5000).fit(df_all["text"].astype(str))
    train_vecs = vec.transform(train_df["text"].astype(str))
    test_vecs = vec.transform(test_df["text"].astype(str))
    
    sim = cosine_similarity(test_vecs, train_vecs).max(axis=1)
    near_dup_90 = (sim > 0.9).sum()
    near_dup_95 = (sim > 0.95).sum()
    print(f"Test rows with >0.90 cosine similarity to some train row: {near_dup_90} / {len(test_df)}")
    print(f"Test rows with >0.95 cosine similarity to some train row: {near_dup_95} / {len(test_df)}")

    print("\n" + "="*50)
    print("3. STRUCTURAL ARTIFACT CHECK")
    print("="*50)
    print(get_quote_artifact_stats(train_df))

    print("\n" + "="*50)
    print("4. SIGNED LOG-ODDS (NAMED ENTITY LEAKAGE)")
    print("="*50)
    top_scam_terms, top_legit_terms = signed_log_odds(train_df)
    print("Top SCAM-indicative terms:", list(top_scam_terms))
    print("Top LEGIT-indicative terms:", list(top_legit_terms))

if __name__ == "__main__":
    main()
