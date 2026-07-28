"""
Train classical ML baselines (TF-IDF + Logistic Regression / LightGBM) 
for Scam Detection to justify DistilBERT.

Usage:
    python scripts/train_baseline.py --data data/processed/composite_train.csv
"""

import argparse
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb
from sklearn.model_selection import train_test_split

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV")
    args = parser.parse_args()

    print(f"Loading data from {args.data}...")
    df = pd.read_csv(args.data)
    
    # In case data isn't split yet or we want a clean eval split
    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )

    X_train_raw = train_df['text'].fillna("")
    y_train = train_df['label']
    
    X_val_raw = val_df['text'].fillna("")
    y_val = val_df['label']

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)

    print("\n--- Baseline 1: Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_val)
    
    print(f"Accuracy: {accuracy_score(y_val, lr_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_val, lr_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, lr_preds))

    print("\n--- Baseline 2: LightGBM ---")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1
    }

    print("Training LightGBM...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )

    lgb_probs = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    lgb_preds = (lgb_probs > 0.5).astype(int)
    
    print(f"Accuracy: {accuracy_score(y_val, lgb_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_val, lgb_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_val, lgb_preds))

    print("\nBaseline evaluation complete. Use these metrics in Deliverable 02 to justify the Deep Learning approach.")

if __name__ == "__main__":
    main()
