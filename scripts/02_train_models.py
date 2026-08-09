"""
Fine-tune ModernBERT to classify text as scam (1) or legit (0).

Usage:
    python scripts/02_train_models.py
"""

import os
import json
import random
import numpy as np
import pandas as pd
import torch
import mlflow
import dagshub
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv

load_dotenv()

from datasets import Dataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    average_precision_score, 
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
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

def load_data(config):
    """Loads the train, val, and test datasets from DagsHub S3 based on config."""
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if not (repo_owner and repo_name):
        raise ValueError("Missing DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME in .env")
        
    s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
    
    paths = config["data_paths"]
    
    # Ensure local dirs exist
    os.makedirs(os.path.dirname(paths["train_local"]), exist_ok=True)
    
    print("Downloading train, val, and test datasets from DagsHub S3...")
    s3_client.download_file(repo_name, paths["train_remote"], paths["train_local"])
    s3_client.download_file(repo_name, paths["val_remote"], paths["val_local"])
    s3_client.download_file(repo_name, paths["test_remote"], paths["test_local"])
    
    train_df = pd.read_csv(paths["train_local"]).dropna(subset=["text", "label"])
    val_df = pd.read_csv(paths["val_local"]).dropna(subset=["text", "label"])
    test_df = pd.read_csv(paths["test_local"]).dropna(subset=["text", "label"])
    
    return train_df, val_df, test_df

def generate_and_log_plots(y_true, pos_probs, preds, output_dir):
    """Generates ROC, PR Curve, and Confusion Matrix and logs to MLflow"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, pos_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    roc_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(roc_path, "plots")
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, pos_probs)
    pr_auc = average_precision_score(y_true, pos_probs)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', lw=2, label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(alpha=0.3)
    pr_path = os.path.join(output_dir, 'pr_curve.png')
    plt.savefig(pr_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(pr_path, "plots")

    # 3. Confusion Matrix
    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legitimate', 'Scam'], 
                yticklabels=['Legitimate', 'Scam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    mlflow.log_artifact(cm_path, "plots")
    
    print("Plots generated and logged to MLflow successfully!")

def main():
    # Load configuration
    config_path = "configs/training_config.json"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
        
    with open(config_path, "r") as f:
        config = json.load(f)

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
        train_df, val_df, test_df = load_data(config)
    except Exception as e:
        print(f"Failed to load data: {e}")
        return
        
    print(f"Train dataset size: {len(train_df)}")
    print(f"Validation dataset size: {len(val_df)}")
    
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)
    test_ds = Dataset.from_pandas(test_df)

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    max_len = tokenizer.model_max_length
    if max_len > 100000:
        max_len = 8192
        
    print(f"Tokenization max_length set to: {max_len}")

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    train_tokenized = train_ds.map(tokenize_fn, batched=True)
    val_tokenized = val_ds.map(tokenize_fn, batched=True)
    test_tokenized = test_ds.map(tokenize_fn, batched=True)

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(config["model_name"], num_labels=2)
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # 4. Training config
    physical_batch_size = 1 if max_len > 512 else 16
    gradient_accumulation_steps = config["batch_size"] // physical_batch_size
    if gradient_accumulation_steps < 1:
        gradient_accumulation_steps = 1
        
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=physical_batch_size,
        per_device_eval_batch_size=physical_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        gradient_checkpointing=True,
        num_train_epochs=config["epochs"],
        weight_decay=config["weight_decay"],
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
    print(f"Setting MLflow experiment to '{config['mlflow_experiment']}'...")
    mlflow.set_experiment(config["mlflow_experiment"])
    
    with mlflow.start_run():
        # Log config
        mlflow.log_dict(config, "configs/training_config.json")
        
        # Train
        trainer.train()
        
        # Evaluate on test set
        print("Evaluating on held-out test set...")
        metrics = trainer.evaluate(test_tokenized)
        print("Final Test Metrics:", metrics)
        
        # Generate Predictions for plotting
        predictions = trainer.predict(test_tokenized)
        logits = predictions.predictions
        y_true = predictions.label_ids
        
        preds = np.argmax(logits, axis=-1)
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)
        pos_probs = probs[:, 1]
        
        # Log Plots
        plot_dir = os.path.join(config["output_dir"], "plots")
        generate_and_log_plots(y_true, pos_probs, preds, plot_dir)
    
        # 6. Save final model + tokenizer
        trainer.save_model(config["output_dir"])
        tokenizer.save_pretrained(config["output_dir"])
        print(f"Model saved locally to {config['output_dir']}")
        
        # 7. Log the model to DagsHub MLflow
        print("Uploading model weights to DagsHub MLflow registry...")
        components = {
            "model": trainer.model,
            "tokenizer": tokenizer,
        }
        clean_model_name = config["model_name"].split("/")[-1]
        registry_name = f"{clean_model_name}-Scam-Classifier"
        
        mlflow.transformers.log_model(
            transformers_model=components,
            artifact_path=clean_model_name,
            registered_model_name=registry_name
        )
        print(f"Model successfully registered to DagsHub Model Registry as '{registry_name}'!")

if __name__ == "__main__":
    main()
