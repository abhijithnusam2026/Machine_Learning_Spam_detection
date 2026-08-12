"""
Fine-tune ModernBERT to classify text as scam (1) or legit (0).

Usage:
    python scripts/02_train_models.py --train_data data/phase1/train.csv --test_data data/phase1/test.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
import os
import random
import subprocess
from pathlib import Path

import mlflow
import requests
import dagshub
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_STAGE_MAP = {
    "feature/phase-2-audio-asr": "feature/phase-2-audio-asr",
    "feature/phase-3-serving-quantization": "feature/phase-3-serving-quantization",
    "feature/phase-1.5-ultimate-dataset": "model-modernbert-universal",
    "model-long-context": "model-modernbert-universal",
    "model-distilbert": "model-distilbert",
    "main": "main",
}


def detect_branch(default="main"):
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


def stage_for_branch(branch):
    return BRANCH_STAGE_MAP.get(branch, branch.replace("/", "-") or "main")

from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, average_precision_score, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    # Softmax to get probabilities for the positive class (1)
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    pos_probs = probs[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    pr_auc = average_precision_score(labels, pos_probs)
    
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, default="data/phase2_asr/train.csv", help="Path to train CSV")
    parser.add_argument("--test_data", type=str, default="data/phase2_asr/test.csv", help="Path to test CSV")
    parser.add_argument(
        "--model_name",
        type=str,
        default="models:/ModernBERT-Scam-Classifier-model-modernbert-universal/latest",
    )
    parser.add_argument("--output_dir", type=str, default="./scam-classifier-model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub privately")
    args = parser.parse_args()
    branch = detect_branch()
    stage = stage_for_branch(branch)
    stage_slug = stage.replace("/", "-")

    # Enforce strict reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)  # Transformers set_seed

    device = get_device()
    print(f"Using device: {device}")

    # 0. Download data from DagsHub if needed
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
        print(f"Fetching latest datasets from DagsHub S3 Bucket ({repo_owner}/{repo_name})...")
        try:
            s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
            for file in [args.train_data, args.test_data]:
                print(f"Downloading {file} from S3...")
                os.makedirs(os.path.dirname(file), exist_ok=True)
                remote_path = f"data/{stage}/phase2_asr/{os.path.basename(file)}"
                s3_client.download_file(repo_name, remote_path, file)
        except Exception as e:
            print(f"Warning: Failed to download datasets from S3 ({e}). Will try to use local copy if it exists.")

    # 1. Load data
    if not os.path.exists(args.train_data) or not os.path.exists(args.test_data):
        print("ERROR: Datasets not found locally and failed to download from DagsHub.")
        import sys
        sys.exit(1)
        
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    
    assert "text" in train_df.columns and "label" in train_df.columns, "Train CSV must have 'text' and 'label' columns"
    assert "text" in test_df.columns and "label" in test_df.columns, "Test CSV must have 'text' and 'label' columns"

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    # 2. Tokenizer and Model Loading (Stateless MLflow Support)
    # If the user passes an MLflow registry URI (e.g., models:/ModernBERT-base-Scam-Classifier/latest), download it.
    if args.model_name.startswith("models:/"):
        print(f"Stateless fetch: Downloading '{args.model_name}' from DagsHub MLflow Registry...")
        try:
            local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=args.model_name)
            print(f"  [SUCCESS] Model downloaded to {local_model_path}")
            print(f"  Loading components via MLflow...")
            components = mlflow.transformers.load_model(local_model_path, return_type="components")
            tokenizer = components["tokenizer"]
            model = components["model"]
            model.to(device)
            # We must set this so metrics tracking knows the original architecture name
            args.model_name = "ModernBERT" 
        except Exception as e:
            print(f"ERROR: Failed to load model from MLflow: {e}")
            import sys
            sys.exit(1)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name, num_labels=2
        )
        model.to(device)
    
    max_len = tokenizer.model_max_length
    if max_len > 100000:
        max_len = 8192  # Fallback for models without a defined max length
        
    print(f"Tokenization max_length set to: {max_len}")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    push_to_hub = args.push_to_hub
    hf_token = os.environ.get("HF_TOKEN")
    if push_to_hub and not hf_token:
        print("ERROR: --push_to_hub requested but HF_TOKEN environment variable not set.")
        import sys
        sys.exit(1)

    hub_model_id = "tanu011235/modernbert-scam-classifier" if push_to_hub else None

    # 4. Training config
    # We use a small physical batch size + gradient accumulation to prevent CUDA OOM on 8192 tokens
    physical_batch_size = 1
    gradient_accumulation_steps = args.batch_size // physical_batch_size
    if gradient_accumulation_steps < 1:
        gradient_accumulation_steps = 1
        
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=physical_batch_size,
        per_device_eval_batch_size=physical_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,  # Crucial for 8192 max_length on 15GB GPUs
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="mlflow",
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_token=hf_token,
        hub_private_repo=True,
        fp16=torch.cuda.is_available(), # Massively speeds up training on T4 GPUs
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

    # 5. Train
    print("Starting MLflow run to log datasets and metrics...")
    mlflow.set_experiment(f"scam-detection/{stage}/train")
    with mlflow.start_run(run_name=f"{stage}-train"):
        mlflow.log_artifact(args.train_data, "dataset")
        mlflow.log_artifact(args.test_data, "dataset")
        mlflow.set_tag("project_stage", stage)
        mlflow.set_tag("git_branch", branch)
        trainer.train()

        # 6. Evaluate
        metrics = trainer.evaluate()
        print("Final evaluation metrics on test set:", metrics)
    
        # Generate Confusion Matrix
        print("\n--- Detailed Evaluation ---")
        predictions = trainer.predict(val_ds)
        preds = np.argmax(predictions.predictions, axis=-1)
        y_true = np.array(val_ds["label"])
        cm = confusion_matrix(y_true, preds)
        print("Confusion Matrix:")
        print(cm)
        
        print("\n--- Stratified Context-Length Evaluation ---")
        # Calculate true token lengths
        def token_len(text):
            return len(tokenizer.encode(str(text), truncation=False))
            
        test_df["token_count"] = test_df["text"].apply(token_len)
        long_mask = test_df["token_count"] > max_len
        short_mask = ~long_mask
        
        if short_mask.sum() > 0:
            acc_short = accuracy_score(y_true[short_mask], preds[short_mask])
            print(f"{args.model_name} accuracy, short transcripts (≤ {max_len} tokens): {acc_short:.2%} (N={short_mask.sum()})")
        
        if long_mask.sum() > 0:
            acc_long = accuracy_score(y_true[long_mask], preds[long_mask])
            print(f"{args.model_name} accuracy, long transcripts (> {max_len} tokens): {acc_long:.2%} (N={long_mask.sum()})")


        # 7. Save final model + tokenizer
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved locally to {args.output_dir}")
        
        # 8. Log the model to DagsHub MLflow
        print("Uploading model weights to DagsHub MLflow registry...")
        components = {
            "model": trainer.model,
            "tokenizer": tokenizer,
        }
        
        # Format the model name for the registry (e.g. "distilbert-base-uncased" -> "distilbert-base-uncased-Scam-Classifier")
        clean_model_name = args.model_name.split("/")[-1]
        registry_name = f"{clean_model_name}-Scam-Classifier-{stage_slug}"
        
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path=f"{stage}/{clean_model_name}",
            registered_model_name=registry_name,
            task="text-classification"
        )
        print(f"Model successfully registered to DagsHub Model Registry as '{registry_name}'!")

    if push_to_hub:
        print(f"Pushing model to Hugging Face Hub (repo: {hub_model_id})...")
        # Ensure it is pushed privately to respect the data privacy proposal
        trainer.push_to_hub()
        print("Model successfully pushed to Hugging Face Hub as a PRIVATE repository!")


if __name__ == "__main__":
    main()
