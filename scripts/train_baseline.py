"""
Train classical ML baselines (TF-IDF + Logistic Regression / LightGBM) 
for Scam Detection to justify DistilBERT.

Usage:
    python scripts/train_baseline.py --train_data data/processed/composite_train.csv --test_data data/processed/composite_test.csv
"""

import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import lightgbm as lgb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to processed train CSV")
    parser.add_argument("--test_data", type=str, required=True, help="Path to processed test CSV")
    args = parser.parse_args()

    # Enforce reproducibility
    np.random.seed(42)

    print(f"Loading train data from {args.train_data}...")
    train_df = pd.read_csv(args.train_data)
    
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)

    X_train_raw = train_df['text'].fillna("")
    y_train = train_df['label']
    
    X_test_raw = test_df['text'].fillna("")
    y_test = test_df['label']

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    print("\n--- Baseline 1: Logistic Regression ---")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_preds = lr.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, lr_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lr_preds))

    print("\n--- Baseline 2: LightGBM ---")
    lgb_train = lgb.Dataset(X_train, y_train)
    lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)

    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'verbose': -1,
        'seed': 42
    }

    print("Training LightGBM...")
    gbm = lgb.train(
        params,
        lgb_train,
        num_boost_round=100,
        valid_sets=[lgb_test],
        callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
    )

    lgb_probs = gbm.predict(X_test, num_iteration=gbm.best_iteration)
    lgb_preds = (lgb_probs > 0.5).astype(int)
    
    print(f"Accuracy: {accuracy_score(y_test, lgb_preds):.4f}")
    print("Classification Report:")
    print(classification_report(y_test, lgb_preds))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, lgb_preds))

    print("\nBaseline evaluation complete. Note: metrics evaluate against the true held-out test set.")

if __name__ == "__main__":
    main()
