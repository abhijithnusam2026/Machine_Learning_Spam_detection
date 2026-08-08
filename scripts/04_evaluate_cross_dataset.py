"""
Cross-Dataset Evaluation
Tests the ModernBERT model on the legacy Kaggle Composite dataset.
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import mlflow
import dagshub
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer, AutoModelForSequenceClassification

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def load_legacy_data():
    """Downloads and merges the entire legacy composite dataset from DagsHub S3."""
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    if not (repo_owner and repo_name):
        raise ValueError("Missing DAGSHUB_REPO_OWNER or DAGSHUB_REPO_NAME in .env")
        
    s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
    os.makedirs("data/legacy_composite", exist_ok=True)
    
    csv1 = "data/legacy_composite/composite_train.csv"
    csv2 = "data/legacy_composite/composite_test.csv"
    
    print("Downloading legacy datasets from DagsHub S3...")
    s3_client.download_file(repo_name, csv1, csv1)
    s3_client.download_file(repo_name, csv2, csv2)
    
    df1 = pd.read_csv(csv1)
    df2 = pd.read_csv(csv2)
    df_all = pd.concat([df1, df2], ignore_index=True)
    
    print(f"Loaded legacy dataset: {len(df_all)} rows.")
    return df_all

def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, default="./scam-classifier-model", help="Path to model directory or 'mlflow' to pull from DagsHub")
    args = parser.parse_args()
    
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
        df = load_legacy_data()
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
        mlflow.set_experiment("modernbert-scam-detection")
        runs = mlflow.search_runs(order_by=["start_time DESC"], max_results=1)
        
        if len(runs) > 0:
            run_id = runs.iloc[0].run_id
            print(f"Loading model from run: {run_id}")
            model_uri = f"runs:/{run_id}/modernbert-scam-classifier"
            pipeline = mlflow.transformers.load_model(model_uri, return_type="components")
            model = pipeline["model"]
            tokenizer = pipeline["tokenizer"]
        else:
            print("No runs found in MLflow. Please train first.")
            return
    else:
        print(f"Loading model locally from {args.model_dir}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
        
    model.to(device)
    model.eval()
    
    # 3. Inference (Batched for speed)
    batch_size = 16
    all_preds = []
    
    print(f"Starting batched inference on {len(texts)} samples...")
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
            
            if "token_type_ids" in inputs:
                del inputs["token_type_ids"]
                
            outputs = model(**inputs)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().tolist()
            all_preds.extend(preds)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i}/{len(texts)}...")
                
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
    print("\n----------------------------------------")
    
if __name__ == "__main__":
    main()
