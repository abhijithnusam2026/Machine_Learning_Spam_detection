"""
Fine-tune ModernBERT to classify text as scam (1) or legit (0).

Usage:
    python scripts/02_train_models.py --model_name answerdotai/ModernBERT-base
"""

import argparse
import numpy as np
import pandas as pd
import torch
import os
import random
import mlflow
import dagshub
from dotenv import load_dotenv

load_dotenv()

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
    
    exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
    pos_probs = probs[:, 1]
    
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
    acc = accuracy_score(labels, preds)
    pr_auc = average_precision_score(labels, pos_probs)
    
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1, "pr_auc": pr_auc}

def load_data():
    """Loads the train, val, and test datasets from DagsHub S3."""
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if not (repo_owner and repo_name):
        raise ValueError("Missing DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME in .env")
        
    s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
    os.makedirs("data", exist_ok=True)
    
    print("Downloading train, val, and test datasets from DagsHub S3...")
    s3_client.download_file(repo_name, "data/train.csv", "data/train.csv")
    s3_client.download_file(repo_name, "data/val.csv", "data/val.csv")
    s3_client.download_file(repo_name, "data/test.csv", "data/test.csv")
    
    train_df = pd.read_csv("data/train.csv").dropna(subset=["text", "label"])
    val_df = pd.read_csv("data/val.csv").dropna(subset=["text", "label"])
    test_df = pd.read_csv("data/test.csv").dropna(subset=["text", "label"])
    
    return train_df, val_df, test_df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--output_dir", type=str, default="./scam-classifier-model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    # Reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    set_seed(seed)

    device = get_device()
    print(f"Using device: {device}")

    # Auth
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

    # 1. Load data
    try:
        train_df, val_df, test_df = load_data()
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
        
    print(f"Train dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")
    
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    max_len = tokenizer.model_max_length
    if max_len > 100000:
        max_len = 8192
        
    print(f"Tokenization max_length set to: {max_len}")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    val_tokenized = val_ds.map(tokenize_fn, batched=True)

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Training config
    physical_batch_size = 1 if max_len > 512 else 16
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
        gradient_checkpointing=True,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="mlflow",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 5. Train
    experiment_name = "scam-detection-ultimate"
    print(f"Setting MLflow experiment to '{experiment_name}'...")
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run():
        trainer.train()
        metrics = trainer.evaluate()
        print("Final evaluation metrics on Validation set:", metrics)
    
        # 6. Save final model + tokenizer
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print(f"Model saved locally to {args.output_dir}")
        
        # 7. Log the model to DagsHub MLflow
        print("Uploading model weights to DagsHub MLflow registry...")
        components = {
            "model": trainer.model,
            "tokenizer": tokenizer,
        }
        clean_model_name = args.model_name.split("/")[-1]
        registry_name = f"{clean_model_name}-Scam-Classifier"
        
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path=clean_model_name,
            registered_model_name=registry_name
        )
        print(f"Model successfully registered to DagsHub Model Registry as '{registry_name}'!")

if __name__ == "__main__":
    main()
