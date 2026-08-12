"""Train classical ML baselines for scam detection.

Usage:
    python scripts/train_baseline.py --train_data data/processed/composite_train.csv --test_data data/processed/composite_test.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_STAGE_MAP = {
    "model-distilbert": "model-distilbert",
    "feature/phase-1.5-ultimate-dataset": "model-modernbert-universal",
    "model-long-context": "model-modernbert-universal",
    "feature/phase-2-audio-asr": "feature/phase-2-audio-asr",
    "feature/phase-3-serving-quantization": "feature/phase-3-serving-quantization",
    "main": "main",
}


def detect_branch(default: str = "main") -> str:
    env_branch = os.getenv("DAGSHUB_BRANCH") or os.getenv("GIT_BRANCH")
    if env_branch:
        return env_branch.strip()
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    branch = result.stdout.strip()
    return branch or default


def stage_for_branch(branch: str) -> str:
    return BRANCH_STAGE_MAP.get(branch, branch.replace("/", "-") or "main")


def parse_dagshub_repo() -> tuple[str, str] | None:
    repo = os.getenv("DAGSHUB_REPO")
    if not repo or "/" not in repo:
        return None
    return tuple(repo.split("/", 1))  # type: ignore[return-value]


def init_mlflow(stage: str) -> object | None:
    try:
        import dagshub
        import mlflow
    except ImportError:
        print("dagshub/mlflow are not installed; proceeding without remote experiment logging.")
        return None

    repo_parts = parse_dagshub_repo()
    if repo_parts is None:
        print("DAGSHUB_REPO is not configured; proceeding without remote experiment logging.")
        return mlflow

    repo_owner, repo_name = repo_parts
    dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True, root=str(REPO_ROOT))
    mlflow.set_experiment(f"scam-detection/{stage}/baseline")
    return mlflow


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to processed train CSV")
    parser.add_argument("--test_data", type=str, required=True, help="Path to processed test CSV")
    parser.add_argument(
        "--branch",
        default=detect_branch(),
        help="Logical branch bucket to use for experiment logging.",
    )
    args = parser.parse_args()
    stage = stage_for_branch(args.branch)

    mlflow = init_mlflow(stage)

    np.random.seed(42)

    print(f"Loading train data from {args.train_data}...")
    train_df = pd.read_csv(args.train_data)
    print(f"Loading test data from {args.test_data}...")
    test_df = pd.read_csv(args.test_data)

    X_train_raw = train_df["text"].fillna("")
    y_train = train_df["label"]
    X_test_raw = test_df["text"].fillna("")
    y_test = test_df["label"]

    print("Extracting TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_test = vectorizer.transform(X_test_raw)

    report_root = REPO_ROOT / "reports" / stage / "baseline"
    report_root.mkdir(parents=True, exist_ok=True)

    if mlflow is not None:
        with mlflow.start_run(run_name=f"{stage}-baseline"):
            mlflow.log_params(
                {
                    "stage": stage,
                    "branch": args.branch,
                    "model_family": "tfidf+logreg+lightgbm",
                    "tfidf_max_features": 10000,
                    "tfidf_ngram_range": "(1,2)",
                }
            )

            print("\n--- Baseline 1: Logistic Regression ---")
            lr = LogisticRegression(max_iter=1000, random_state=42)
            lr.fit(X_train, y_train)
            lr_preds = lr.predict(X_test)
            lr_accuracy = accuracy_score(y_test, lr_preds)
            lr_report = classification_report(y_test, lr_preds)
            lr_cm = confusion_matrix(y_test, lr_preds)

            print(f"Accuracy: {lr_accuracy:.4f}")
            print("Classification Report:")
            print(lr_report)
            print("Confusion Matrix:")
            print(lr_cm)

            mlflow.log_metric("logreg_accuracy", lr_accuracy)
            mlflow.log_text(lr_report, "logreg_classification_report.txt")
            np.savetxt(report_root / "logreg_confusion_matrix.csv", lr_cm, delimiter=",", fmt="%d")
            mlflow.log_artifact(str(report_root / "logreg_confusion_matrix.csv"))

            print("\n--- Baseline 2: LightGBM ---")
            lgb_train = lgb.Dataset(X_train, y_train)
            lgb_test = lgb.Dataset(X_test, y_test, reference=lgb_train)
            params = {
                "objective": "binary",
                "metric": "binary_logloss",
                "boosting_type": "gbdt",
                "num_leaves": 31,
                "learning_rate": 0.05,
                "feature_fraction": 0.9,
                "verbose": -1,
                "seed": 42,
            }

            print("Training LightGBM...")
            gbm = lgb.train(
                params,
                lgb_train,
                num_boost_round=100,
                valid_sets=[lgb_test],
                callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
            )

            lgb_probs = gbm.predict(X_test, num_iteration=gbm.best_iteration)
            lgb_preds = (lgb_probs > 0.5).astype(int)
            lgb_accuracy = accuracy_score(y_test, lgb_preds)
            lgb_report = classification_report(y_test, lgb_preds)
            lgb_cm = confusion_matrix(y_test, lgb_preds)

            print(f"Accuracy: {lgb_accuracy:.4f}")
            print("Classification Report:")
            print(lgb_report)
            print("Confusion Matrix:")
            print(lgb_cm)

            mlflow.log_metric("lightgbm_accuracy", lgb_accuracy)
            mlflow.log_text(lgb_report, "lightgbm_classification_report.txt")
            np.savetxt(report_root / "lightgbm_confusion_matrix.csv", lgb_cm, delimiter=",", fmt="%d")
            mlflow.log_artifact(str(report_root / "lightgbm_confusion_matrix.csv"))

            print("\nBaseline evaluation complete. Note: metrics evaluate against the true held-out test set.")
    else:
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
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "verbose": -1,
            "seed": 42,
        }

        print("Training LightGBM...")
        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=100,
            valid_sets=[lgb_test],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)],
        )
        lgb_probs = gbm.predict(X_test, num_iteration=gbm.best_iteration)
        lgb_preds = (lgb_probs > 0.5).astype(int)
        print(f"Accuracy: {accuracy_score(y_test, lgb_preds):.4f}")
        print("Classification Report:")
        print(classification_report(y_test, lgb_preds))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, lgb_preds))

        print("\nBaseline evaluation complete. Note: metrics evaluate against the true held-out test set.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
