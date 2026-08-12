"""Clean the raw scam dataset, deduplicate it, and write branch-scoped splits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import re
from sklearn.model_selection import train_test_split


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


def dagshub_repo() -> str | None:
    return os.getenv("DAGSHUB_REPO")


def upload_to_dagshub(local_dir: Path, stage: str) -> None:
    repo = dagshub_repo()
    if not repo:
        print("DAGSHUB_REPO is not set; skipping processed-data upload.")
        return

    try:
        import dagshub
    except ImportError:
        print("dagshub is not installed; skipping processed-data upload.")
        return

    remote_path = f"data/{stage}/processed"
    print(f"Uploading processed data to DagsHub bucket: {repo} -> {remote_path}")
    dagshub.upload_files(
        repo,
        local_dir,
        remote_path=remote_path,
        bucket=True,
        commit_message=f"Upload processed data for {stage}",
    )


def clean_text(text: object) -> str:
    if pd.isna(text):
        return ""
    text_str = str(text).strip("\"'")
    return re.sub(r"\s+", " ", text_str).strip()


def fix_label_noise(row: pd.Series) -> int:
    """Force obviously benign domains to label 0 if they were mislabeled."""
    text_lower = row["text"].lower()
    benign_keywords = ["taxi", "food delivery", "uber", "doordash", "pizza", "restaurant"]
    if any(keyword in text_lower for keyword in benign_keywords):
        return 0
    return int(row["label"])


def preprocess_and_split(
    raw_train_path: Path,
    raw_test_path: Path,
    proc_train_path: Path,
    proc_test_path: Path,
) -> None:
    print("Loading raw datasets...")
    df_train = pd.read_csv(raw_train_path)
    df_test = pd.read_csv(raw_test_path)

    df_all = pd.concat([df_train, df_test], ignore_index=True)

    if "transcript" in df_all.columns and "is_scam" in df_all.columns:
        df_all = df_all.rename(columns={"transcript": "text", "is_scam": "label"})

    print(f"Total raw rows (train + test): {len(df_all)}")

    print("Cleaning text data and stripping quote artifacts...")
    df_all["text"] = df_all["text"].apply(clean_text)
    df_all = df_all[df_all["text"] != ""]

    print("Performing global deduplication to fix train/test leakage...")
    df_all = df_all.drop_duplicates(subset=["text"], keep="first")
    print(f"Rows after deduplication: {len(df_all)}")

    print("Auditing provenance and fixing label noise (benign misclassifications)...")
    original_labels = df_all["label"].copy()
    df_all["label"] = df_all.apply(fix_label_noise, axis=1)

    flipped = df_all[original_labels != df_all["label"]]
    print(f"Flipped {len(flipped)} rows heuristically based on benign keywords.")
    if len(flipped) > 0:
        print("\n--- SAMPLE OF 3 FLIPPED ROWS FOR SANITY CHECK ---")
        for _, row in flipped.sample(min(3, len(flipped)), random_state=42).iterrows():
            print(f"Text snippet: {row['text'][:100]}...")
        print("--------------------------------------------------\n")

    print("Re-splitting dataset securely...")
    df_train_clean, df_test_clean = train_test_split(
        df_all,
        test_size=0.15,
        random_state=42,
        stratify=df_all["label"],
    )

    proc_train_path.parent.mkdir(parents=True, exist_ok=True)
    df_train_clean.to_csv(proc_train_path, index=False)
    df_test_clean.to_csv(proc_test_path, index=False)

    print(f"Saved processed train dataset to {proc_train_path} (Rows: {len(df_train_clean)})")
    print(f"Saved processed test dataset to {proc_test_path} (Rows: {len(df_test_clean)})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch",
        default=detect_branch(),
        help="Logical branch bucket to use for DagsHub uploads.",
    )
    parser.add_argument("--push_to_hub", action="store_true")
    args = parser.parse_args()
    stage = stage_for_branch(args.branch)

    raw_train = REPO_ROOT / "data" / "raw" / "composite_train.csv"
    raw_test = REPO_ROOT / "data" / "raw" / "composite_test.csv"

    proc_root = REPO_ROOT / "data" / stage / "processed"
    proc_train = proc_root / "composite_train.csv"
    proc_test = proc_root / "composite_test.csv"

    if not (raw_train.exists() and raw_test.exists()):
        print("ERROR: Raw data files not found in data/raw/. Please run download.py first.")
        return 1

    preprocess_and_split(raw_train, raw_test, proc_train, proc_test)

    manifest_path = proc_root / "dataset_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "stage": stage,
                "branch": args.branch,
                "source": "composite-scam-transcript-dataset",
                "files": ["composite_train.csv", "composite_test.csv"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Saved dataset manifest to {manifest_path}")

    upload_to_dagshub(proc_root, stage)

    if args.push_to_hub:
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            print("ERROR: --push_to_hub used but HF_TOKEN not found in environment variables.")
            return 1
        try:
            from datasets import load_dataset

            print("\nPushing processed dataset to Hugging Face Hub (Private Repo)...")
            dataset = load_dataset("csv", data_files={"train": proc_train, "test": proc_test})
            repo_id = "tanu011235/scam-alert-dataset"
            dataset.push_to_hub(repo_id, token=hf_token, private=True)
            print(f"Successfully pushed PRIVATE dataset to https://huggingface.co/datasets/{repo_id}")
        except Exception as exc:  # pragma: no cover - upload path
            print(f"Failed to push to Hugging Face Hub: {exc}")
            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
