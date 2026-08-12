"""Fine-tune DistilBERT to classify text as scam (1) or legit (0)."""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)


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
    mlflow.set_experiment(f"scam-detection/{stage}/distilbert-train")
    return mlflow


def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    pos_probs = probs[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    pr_auc = average_precision_score(labels, pos_probs)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--branch",
        default=detect_branch(),
        help="Logical branch bucket to use for experiment logging.",
    )
    args = parser.parse_args()

    stage = stage_for_branch(args.branch)
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "artifacts" / stage / "distilbert"
    output_dir.mkdir(parents=True, exist_ok=True)

    mlflow = init_mlflow(stage)

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    assert "text" in train_df.columns and "label" in train_df.columns
    assert "text" in test_df.columns and "label" in test_df.columns

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    push_to_hub = args.push_to_hub
    hf_token = os.environ.get("HF_TOKEN")
    if push_to_hub and not hf_token:
        print("ERROR: --push_to_hub requested but HF_TOKEN environment variable not set.")
        return 1

    hub_model_id = "tanu011235/distilbert-scam-classifier" if push_to_hub else None

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="none",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=hf_token,
        hub_private_repo=True,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final evaluation metrics on test set:", metrics)

    predictions = trainer.predict(val_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    cm = confusion_matrix(val_ds["label"], preds)
    print("\n--- Detailed Evaluation ---")
    print("Confusion Matrix:")
    print(cm)

    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Model saved locally to {output_dir}")

    if mlflow is not None:
        with mlflow.start_run(run_name=f"{stage}-distilbert-train"):
            mlflow.log_params(
                {
                    "stage": stage,
                    "branch": args.branch,
                    "model_name": args.model_name,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "learning_rate": args.lr,
                    "output_dir": str(output_dir),
                }
            )
            mlflow.log_metrics(
                {
                    key: float(value)
                    for key, value in metrics.items()
                    if isinstance(value, (int, float, np.floating))
                }
            )
            cm_path = output_dir / "confusion_matrix.csv"
            np.savetxt(cm_path, cm, delimiter=",", fmt="%d")
            mlflow.log_artifact(str(cm_path))
            mlflow.log_artifacts(str(output_dir), artifact_path=f"{stage}/distilbert")

    if push_to_hub:
        print(f"Pushing model to Hugging Face Hub (repo: {hub_model_id})...")
        trainer.push_to_hub()
        print("Model successfully pushed to Hugging Face Hub as a PRIVATE repository!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
