"""
Fine-tune ModernBERT to classify text as scam (1) or legit (0).

Usage:
    python 03_train_modernbert.py --train_data data/train.csv --test_data data/test.csv
"""

import argparse
import numpy as np
import pandas as pd
import torch
import os
import random
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
    parser.add_argument("--train_data", type=str, default="data/train.csv", help="Path to train CSV")
    parser.add_argument("--test_data", type=str, default="data/test.csv", help="Path to test CSV")
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--output_dir", type=str, default="./scam-classifier-model")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--push_to_hub", action="store_true", help="Push model to Hugging Face Hub privately")
    args = parser.parse_args()

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

    # 1. Load data
    train_df = pd.read_csv(args.train_data)
    test_df = pd.read_csv(args.test_data)
    
    assert "text" in train_df.columns and "label" in train_df.columns, "Train CSV must have 'text' and 'label' columns"
    assert "text" in test_df.columns and "label" in test_df.columns, "Test CSV must have 'text' and 'label' columns"

    train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
    val_ds = Dataset.from_pandas(test_df.reset_index(drop=True))

    # 2. Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=8192)

    train_ds = train_ds.map(tokenize_fn, batched=True)
    val_ds = val_ds.map(tokenize_fn, batched=True)

    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2
    )
    model.to(device)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    push_to_hub = args.push_to_hub
    hf_token = os.environ.get("HF_TOKEN")
    if push_to_hub and not hf_token:
        print("ERROR: --push_to_hub requested but HF_TOKEN environment variable not set.")
        import sys
        sys.exit(1)

    hub_model_id = "tanu011235/modernbert-scam-classifier" if push_to_hub else None

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
    long_mask = test_df["token_count"] > 8192
    short_mask = ~long_mask
    
    if short_mask.sum() > 0:
        acc_short = accuracy_score(y_true[short_mask], preds[short_mask])
        print(f"ModernBERT accuracy, short transcripts (≤ 8192 tokens): {acc_short:.2%} (N={short_mask.sum()})")
    
    if long_mask.sum() > 0:
        acc_long = accuracy_score(y_true[long_mask], preds[long_mask])
        print(f"ModernBERT accuracy, long transcripts (> 8192 tokens): {acc_long:.2%} (N={long_mask.sum()})")


    # 7. Save final model + tokenizer
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved locally to {args.output_dir}")

    if push_to_hub:
        print(f"Pushing model to Hugging Face Hub (repo: {hub_model_id})...")
        # Ensure it is pushed privately to respect the data privacy proposal
        trainer.push_to_hub()
        print("Model successfully pushed to Hugging Face Hub as a PRIVATE repository!")


if __name__ == "__main__":
    main()
