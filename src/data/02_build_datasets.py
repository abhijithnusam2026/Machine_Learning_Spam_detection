import os
import json
import random
import re
import glob
import zipfile
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import mlflow
import dagshub
from dotenv import load_dotenv
from src.utils.mlflow_reporting import log_dataframe_artifact, log_split_profile

SEED = 42
PTQ_CALIBRATION_ROWS = 256
random.seed(SEED)

def clean_text(text):
    text = str(text)
    text = re.sub(r'(?i)(innocent|suspect):\s*', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

def stratified_sample(df, n, strata_cols, random_state):
    if df.empty:
        return df.copy()

    n = min(n, len(df))
    selected_parts = []
    selected_indices = set()

    grouped = list(df.groupby(strata_cols, group_keys=False, dropna=False))
    for _, group in grouped:
        group_n = max(1, round(n * len(group) / len(df)))
        group_n = min(group_n, len(group))
        sampled = group.sample(n=group_n, random_state=random_state)
        selected_parts.append(sampled)
        selected_indices.update(sampled.index.tolist())

    sampled_df = pd.concat(selected_parts).drop_duplicates(subset=["text"])
    if len(sampled_df) > n:
        sampled_df = sampled_df.sample(n=n, random_state=random_state)
    elif len(sampled_df) < n:
        remaining = df.drop(index=list(selected_indices), errors="ignore")
        if not remaining.empty:
            fill_n = min(n - len(sampled_df), len(remaining))
            fill_df = remaining.sample(n=fill_n, random_state=random_state)
            sampled_df = pd.concat([sampled_df, fill_df]).drop_duplicates(subset=["text"])

    return sampled_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

def main():
    print("--- Building Unified Modular Datasets ---")
    
    # 1. Load Synthesized JSON data (LLM Data)
    print("Loading synthesized JSON data...")
    json_dir = "data/raw/jsons"
    json_files = ["scam_call_hard_examples_250.json", "scam_call_transcripts_250_combined.json"]
    synth_data = []
    for fname in json_files:
        path = os.path.join(json_dir, fname)
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                synth_data.extend(json.load(f))
                
    df_synth = pd.DataFrame(synth_data) if synth_data else pd.DataFrame(columns=["text", "label"])
    df_synth["source_domain"] = "written_text"
    df_synth["source_dataset"] = "synthetic"
    
    # 2. Extract and Load Teeconnie dataset
    print("Processing teeconnie dataset...")
    teeconnie_zip = "data/raw/teeconnie/teeconnie_dataset.zip"
    if os.path.exists(teeconnie_zip):
        with zipfile.ZipFile(teeconnie_zip, 'r') as zipf:
            zipf.extractall("data/raw/teeconnie/")
    
    teeconnie_files = glob.glob("data/raw/teeconnie/**/*", recursive=True)
    # The composite and synthetic written corpora are already scam-heavy.
    # Teeconnie is intentionally used as legitimate-call augmentation only,
    # so ModernBERT gains benign call variety without increasing positive-class skew.
    nonscam_txt = [f for f in teeconnie_files if f.lower().endswith(".txt") and "non" in f.lower() and "scam" in f.lower()]
    
    entries = []
    if nonscam_txt:
        with open(nonscam_txt[0], "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
            entries = [e.strip() for e in content.split("\n\n") if e.strip()]
            if len(entries) < 5:
                entries = [e.strip() for e in content.split("\n") if e.strip()]
    
    df_teeconnie = pd.DataFrame({"text": entries, "label": [0] * len(entries)})
    df_teeconnie["source_domain"] = "written_text"
    df_teeconnie["source_dataset"] = "teeconnie"
    
    # 3. Load Legacy Kaggle Composite (Phishing, SMS, Enron)
    print("Processing Legacy Kaggle composite...")
    csv1 = "data/raw/legacy_composite/composite_train.csv"
    csv2 = "data/raw/legacy_composite/composite_test.csv"
    df_legacy1 = pd.read_csv(csv1) if os.path.exists(csv1) else pd.DataFrame()
    df_legacy2 = pd.read_csv(csv2) if os.path.exists(csv2) else pd.DataFrame()
    df_legacy = pd.concat([df_legacy1, df_legacy2], ignore_index=True)
    df_legacy["source_domain"] = "written_text"
    df_legacy["source_dataset"] = "kaggle_composite"
    
    # 4. Load Raw ASR Transcripts
    print("Processing Raw ASR Transcripts...")
    asr_paths = sorted(glob.glob("data/raw/asr/*.csv"))
    asr_frames = []
    for asr_path in asr_paths:
        frame = pd.read_csv(asr_path)
        if "source" in frame.columns and "source_dataset" not in frame.columns:
            frame = frame.rename(columns={"source": "source_dataset"})
        asr_frames.append(frame)
    df_asr = pd.concat(asr_frames, ignore_index=True) if asr_frames else pd.DataFrame()
    if df_asr.empty:
        raise RuntimeError(
            "Required ASR transcript sources are missing or empty under data/raw/asr/*.csv. "
            "Run src/data/00_download_raw_data.py after fixing the DagsHub artifact path before building datasets."
        )
    missing_asr_columns = {"text", "label"} - set(df_asr.columns)
    if missing_asr_columns:
        raise RuntimeError(f"ASR transcript sources are missing required columns: {sorted(missing_asr_columns)}")
    df_asr["source_domain"] = "spoken_asr"
    if "source_dataset" not in df_asr.columns:
        df_asr["source_dataset"] = "asr_transcripts"
    
    # Combine ALL sources
    print("\nAssembling massive canonical dataset...")
    all_data = pd.concat([df_synth, df_teeconnie, df_legacy, df_asr], ignore_index=True)
    all_data = all_data.dropna(subset=["text", "label"])
    all_data["text"] = all_data["text"].apply(clean_text)
    
    # Drop duplicates to prevent train/test leakage
    all_data = all_data.drop_duplicates(subset=["text"]).reset_index(drop=True)
    
    print(f"Total Unique Samples: {len(all_data)}")
    
    # 5. IMMEDIATELY split off the 20% Global Hold-Out Set
    print("\nCreating stratified Global Hold-Out Set...")
    global_test = all_data.groupby(["label", "source_domain"], group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=SEED))
    
    # The remainder goes to train/val
    train_val_data = all_data.drop(global_test.index)
    
    # Split train/val (80/20 of the remainder)
    global_val = train_val_data.groupby(["label", "source_domain"], group_keys=False).apply(lambda x: x.sample(frac=0.2, random_state=SEED))
    global_train = train_val_data.drop(global_val.index)
    
    # Save datasets
    os.makedirs("data/processed", exist_ok=True)
    
    global_test.to_csv("data/processed/global_test.csv", index=False)
    global_train.to_csv("data/processed/global_train.csv", index=False)
    global_val.to_csv("data/processed/global_val.csv", index=False)

    ptq_calibration = stratified_sample(
        global_train,
        n=PTQ_CALIBRATION_ROWS,
        strata_cols=["label", "source_domain"],
        random_state=SEED,
    )
    ptq_calibration.to_csv("data/processed/ptq_calibration.csv", index=False)

    val_texts = set(global_val["text"].astype(str))
    test_texts = set(global_test["text"].astype(str))
    calibration_texts = set(ptq_calibration["text"].astype(str))
    assert calibration_texts.isdisjoint(val_texts), "PTQ calibration data leaked into validation split"
    assert calibration_texts.isdisjoint(test_texts), "PTQ calibration data leaked into global holdout split"
    
    print(f"Global Train Set: {len(global_train)} rows")
    print(f"Global Val Set:   {len(global_val)} rows")
    print(f"Global Test Set:  {len(global_test)} rows (Frozen for final evaluation)")
    print(f"PTQ Calibration:  {len(ptq_calibration)} rows (sampled from train only)")
    
    print("\nDataset building complete. Data ready for modeling in data/processed/.")

    # 6. Log processed datasets to MLflow
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/02_build_datasets")
        with mlflow.start_run(run_name="build_processed_datasets"):
            mlflow.set_tag("project_stage", "refactored_pipeline")
            mlflow.set_tag("pipeline_stage", "02_build_datasets")
            mlflow.log_param("seed", SEED)
            mlflow.log_param("split_strategy", "20% frozen global holdout, then 20% validation from remaining data")
            mlflow.log_param("deduplication_key", "cleaned_text")
            mlflow.log_param("stratification_columns", "label,source_domain")
            mlflow.log_param("ptq_calibration_source", "global_train only")
            mlflow.log_param("ptq_calibration_target_rows", PTQ_CALIBRATION_ROWS)

            log_split_profile(
                {
                    "global_train": global_train,
                    "global_val": global_val,
                    "global_test": global_test,
                    "ptq_calibration": ptq_calibration,
                },
                artifact_path="dataset_profile",
            )

            source_summary = (
                all_data.groupby(["source_domain", "source_dataset", "label"])
                .size()
                .reset_index(name="rows")
                .sort_values(["source_domain", "source_dataset", "label"])
            )
            log_dataframe_artifact(source_summary, "canonical_source_summary.csv", "dataset_profile")

            mlflow.log_artifact("data/processed/global_train.csv", artifact_path="processed")
            mlflow.log_artifact("data/processed/global_val.csv", artifact_path="processed")
            mlflow.log_artifact("data/processed/global_test.csv", artifact_path="processed")
            mlflow.log_artifact("data/processed/ptq_calibration.csv", artifact_path="processed")
            print("\nSuccessfully logged processed datasets to DagsHub MLflow.")

if __name__ == "__main__":
    main()
