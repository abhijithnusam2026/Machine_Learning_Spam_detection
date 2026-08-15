"""
Recover/log an already-trained Hugging Face sequence classifier without retraining.

Use this when training completed and saved a local model directory, but MLflow/DagsHub
registry logging failed at the final upload step.
"""

import argparse
import os
import random
import sys
from pathlib import Path

import dagshub
import mlflow
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from dotenv import load_dotenv
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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.data_access import ensure_processed_data
from src.utils.mlflow_reporting import (
    log_classification_artifacts,
    log_split_profile,
    log_transformer_model_with_fallback,
)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def configure_dagshub():
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if username and password:
        os.environ["DAGSHUB_USER"] = username
        os.environ["DAGSHUB_TOKEN"] = password
        dagshub.auth.add_app_token(password)

    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    pos_probs = probs[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "pr_auc": average_precision_score(labels, pos_probs),
    }


def apply_filter(df, filter_name):
    if filter_name == "distilbert_baseline":
        return df[df["source_dataset"] == "kaggle_composite"].copy()
    if filter_name == "written_text":
        return df[df["source_domain"] == "written_text"].copy()
    if filter_name == "asr_transcript":
        return df[df["source_domain"] == "asr_transcript"].copy()
    if filter_name == "none":
        return df.copy()
    raise ValueError(f"Unsupported data filter: {filter_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="./scam-classifier-model-baseline")
    parser.add_argument("--eval_data", default="data/processed/global_val.csv")
    parser.add_argument("--experiment", default="scam-detection/refactored_pipeline/03_baseline_distilbert")
    parser.add_argument("--run_name", default="03-distilbert-recovery")
    parser.add_argument("--pipeline_stage", default="03_baseline_distilbert")
    parser.add_argument("--base_model_name", default="distilbert-base-uncased")
    parser.add_argument("--registered_model_name", default=None)
    parser.add_argument(
        "--data_filter",
        choices=["distilbert_baseline", "written_text", "asr_transcript", "none"],
        default="distilbert_baseline",
    )
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--failed_run_id", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Saved model directory not found: {model_dir}")

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    configure_dagshub()
    ensure_processed_data([args.eval_data])

    eval_df = pd.read_csv(args.eval_data)
    eval_df = apply_filter(eval_df, args.data_filter)
    if eval_df.empty:
        raise ValueError(f"No evaluation rows remain after filter: {args.data_filter}")

    device = get_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)

    max_len = args.max_length or tokenizer.model_max_length
    if max_len > 100000:
        max_len = 512
    print(f"Evaluating saved classifier from {model_dir} on {len(eval_df)} rows.")
    print(f"Tokenization max_length set to: {max_len}")

    eval_ds = Dataset.from_pandas(eval_df.reset_index(drop=True))

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    eval_ds = eval_ds.map(tokenize_fn, batched=True)
    training_args = TrainingArguments(
        output_dir=str(model_dir / "_recovery_eval"),
        per_device_eval_batch_size=8,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    metrics = trainer.evaluate()
    predictions = trainer.predict(eval_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    y_true = np.array(eval_ds["label"])
    cm = confusion_matrix(y_true, preds, labels=[0, 1])

    print("Recovered evaluation metrics:", metrics)
    print("Confusion Matrix:")
    print(cm)

    clean_model_name = args.base_model_name.split("/")[-1]
    registry_name = args.registered_model_name or (
        f"{clean_model_name}-Scam-Classifier-{args.pipeline_stage.replace('/', '-')}"
    )

    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("project_stage", "refactored_pipeline")
        mlflow.set_tag("pipeline_stage", args.pipeline_stage)
        mlflow.set_tag("recovery_run", True)
        mlflow.set_tag("model_source", "saved_local_hf_directory")
        if args.failed_run_id:
            mlflow.set_tag("recovered_from_failed_run_id", args.failed_run_id)

        mlflow.log_params(
            {
                "model_dir": str(model_dir),
                "base_model_name": args.base_model_name,
                "eval_data": args.eval_data,
                "data_filter": args.data_filter,
                "max_length": max_len,
                "seed": seed,
                "registered_model_name": registry_name,
                "training_reexecuted": False,
            }
        )
        mlflow.log_metrics(
            {
                f"recovered_{k}": float(v)
                for k, v in metrics.items()
                if isinstance(v, (int, float, np.floating))
            }
        )
        log_split_profile({"recovery_eval": eval_df}, artifact_path="dataset_profile")
        log_classification_artifacts(
            y_true,
            preds,
            artifact_path="evaluation",
            prefix=f"{args.pipeline_stage}_recovery",
        )
        log_transformer_model_with_fallback(
            components={"model": model, "tokenizer": tokenizer},
            output_dir=str(model_dir),
            artifact_path=f"{args.pipeline_stage}/{clean_model_name}",
            registered_model_name=registry_name,
            task="text-classification",
        )


if __name__ == "__main__":
    main()
