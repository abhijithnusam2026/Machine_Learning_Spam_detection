"""
Cross-Dataset Evaluation
Tests the ModernBERT model on the legacy Kaggle Composite dataset.
"""

import os
import argparse
import subprocess
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import mlflow
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import matplotlib.pyplot as plt
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_STAGE_MAP = {
    "feature/phase-1.5-ultimate-dataset": "model-modernbert-universal",
    "model-long-context": "model-modernbert-universal",
    "feature/phase-2-audio-asr": "feature/phase-2-audio-asr",
    "feature/phase-3-serving-quantization": "feature/phase-3-serving-quantization",
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

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_test_data(stage):
    """Downloads the mathematically balanced 20% held-out test set from DagsHub S3."""
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if not (repo_owner and repo_name):
        raise ValueError("Missing DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME in .env")
        
    s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
    local_dir = f"data/{stage}/phase1.5"
    os.makedirs(local_dir, exist_ok=True)
    
    test_csv = f"{local_dir}/test.csv"
    
    print(f"Downloading held-out test dataset from DagsHub S3: {test_csv}...")
    try:
        s3_client.download_file(repo_name, test_csv, test_csv)
    except Exception as e:
        print(f"Warning: Could not download {test_csv} from DagsHub. Exception: {e}")
    
    if not os.path.exists(test_csv):
        raise FileNotFoundError(f"Could not find test dataset at {test_csv}. Run 01_prep_data.py first.")
        
    df_test = pd.read_csv(test_csv)
    print(f"Loaded held-out test dataset: {len(df_test)} rows.")
    return df_test

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./scam-classifier-model", help="Path to model directory or 'mlflow' to pull from DagsHub")
    parser.add_argument("--registry_name", type=str, default=None, help="Name of the model in DagsHub MLflow Registry (if --model_dir=mlflow)")
    args = parser.parse_args()
    branch = detect_branch()
    stage = stage_for_branch(branch)
    registry_name = args.registry_name or f"ModernBERT-base-Scam-Classifier-{stage}"
    
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if username and password:
        os.environ["DAGSHUB_USER"] = username
        os.environ["DAGSHUB_TOKEN"] = password
        dagshub.auth.add_app_token(password)
        
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = "https://dagshub.com"
        os.environ["MLFLOW_S3_IGNORE_TLS"] = "true"
        os.environ["AWS_ACCESS_KEY_ID"] = username
        os.environ["AWS_SECRET_ACCESS_KEY"] = password
        
    device = get_device()
    print(f"Using device: {device}")
    
    # 1. Download & Load Data
    try:
        df = load_test_data(stage)
    except Exception as e:
        print(f"Failed to load legacy data: {e}")
        return
        
    if "text" not in df.columns or "label" not in df.columns:
        print("Dataset missing 'text' or 'label' columns.")
        return
        
    texts = df["text"].astype(str).tolist()
    true_labels = df["label"].astype(int).tolist()
    
    # 2. Load Model
    if args.model_dir == "mlflow":
        print("Fetching latest model from DagsHub MLflow registry...")
        mlflow.set_experiment(f"scam-detection/{stage}/eval")
        
        # Load directly from the Model Registry
        model_uri = f"models:/{registry_name}/latest"
        print(f"Loading model from registry: {model_uri}")
        
        try:
            pipeline = mlflow.transformers.load_model(model_uri, return_type="components")
            model = pipeline["model"]
            tokenizer = pipeline["tokenizer"]
        except Exception as e:
            print(f"ERROR: Failed to load model from registry ({e}). Please ensure you have trained and registered it.")
            return
    else:
        print(f"Loading model locally from {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        
    model.to(device)
    model.eval()
    
    # 3. Inference (Batched for speed and memory efficiency)
    batch_size = 4  # Reduced batch size to prevent OOM
    
    max_len = tokenizer.model_max_length
    if max_len > 100000:
        max_len = 8192
        
    print(f"Tokenization max_length set to: {max_len}")
    
    # Sort by length to minimize padding overhead in batches
    print("Sorting dataset by text length to optimize memory usage...")
    lengths = [len(str(t)) for t in texts]
    sorted_indices = np.argsort(lengths)
    
    sorted_texts = [texts[i] for i in sorted_indices]
    sorted_labels = [true_labels[i] for i in sorted_indices]
    
    all_preds_sorted = []
    
    print(f"Starting batched inference on {len(sorted_texts)} samples...")
    with torch.no_grad():
        for i in range(0, len(sorted_texts), batch_size):
            batch_texts = sorted_texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len).to(device)
            
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
                
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds_sorted.extend(preds)
            
            # Clear CUDA cache periodically to prevent memory fragmentation
            if i % 100 == 0 and device == "cuda":
                torch.cuda.empty_cache()
            
            if (i // batch_size) % 50 == 0:
                print(f"Processed {i}/{len(sorted_texts)}...")
                
    # Unsort predictions to match original true_labels order
    unsorted_preds = [0] * len(all_preds_sorted)
    for sorted_idx, original_idx in enumerate(sorted_indices):
        unsorted_preds[original_idx] = all_preds_sorted[sorted_idx]
        
    all_preds = unsorted_preds
                
    # 4. Metrics
    print("\n--- Cross-Dataset Evaluation Results ---")
    acc = accuracy_score(true_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, all_preds, average="binary")
    cm = confusion_matrix(true_labels, all_preds)
    
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\nConfusion Matrix:")
    print(cm)
    
    # Plotting and saving the Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Legit', 'Scam'], yticklabels=['Legit', 'Scam'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {args.registry_name}')
    
    report_dir = Path("reports") / stage
    report_dir.mkdir(parents=True, exist_ok=True)
    cm_path = report_dir / f"confusion_matrix_{registry_name}.png"
    plt.savefig(cm_path)
    print(f"\nSaved confusion matrix plot to {cm_path}")
    
    # Log the figure to MLflow if tracking is enabled
    if args.model_dir == "mlflow":
        with mlflow.start_run(run_name=f"{stage}-eval-{registry_name}"):
            mlflow.log_figure(plt.gcf(), "confusion_matrix.png")
            mlflow.log_metric("eval_accuracy_cross", acc)
            mlflow.log_metric("eval_f1_cross", f1)
            mlflow.set_tag("project_stage", stage)
            mlflow.set_tag("git_branch", branch)
            print("Logged evaluation metrics and confusion matrix to MLflow.")
            
    plt.close()
    
    print("\n----------------------------------------")
    
if __name__ == "__main__":
    main()
