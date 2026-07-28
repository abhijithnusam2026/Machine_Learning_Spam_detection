"""
Fine-tune DistilBERT to classify text as scam (1) or legit (0).

Usage:
    python train_scam_classifier.py --data scam_data.csv

Expects a CSV with two columns: "text" and "label" (0/1).
"""

import argparse
import numpy as np
import pandas as pd
import torch
import os
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
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
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to CSV with text,label columns")
    parser.add_argument("--model_name", type=str, default="distilbert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./scam-classifier-model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    # 1. Load and split data
    df = pd.read_csv(args.data)
    
    # If standard composite dataset doesn't have exactly text/label, adapt here.
    if "transcript" in df.columns and "is_scam" in df.columns:
        df = df.rename(columns={"transcript": "text", "is_scam": "label"})
    
    assert "text" in df.columns and "label" in df.columns, "CSV must have 'text' and 'label' columns"

    train_df, val_df = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["label"]
    )

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    hf_token = os.environ.get("HF_TOKEN")
    push_to_hub = bool(hf_token)
    hub_model_id = "tanu011235/distilbert-scam-classifier" if push_to_hub else None

    # 4. Training config
    training_args = TrainingArguments(
        output_dir=args.output_dir,
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
    trainer.train()

    # 6. Evaluate
    metrics = trainer.evaluate()
    print("Final validation metrics:", metrics)

    # 7. Save final model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved locally to {args.output_dir}")

    if push_to_hub:
        print(f"Pushing model to Hugging Face Hub (repo: {hub_model_id})...")
        trainer.push_to_hub()
        print("Model successfully pushed to Hugging Face Hub!")


if __name__ == "__main__":
    main()
