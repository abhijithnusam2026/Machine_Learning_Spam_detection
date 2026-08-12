"""Download the raw scam transcript dataset and mirror it to DagsHub.

Usage:
    python scripts/download.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import kagglehub


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "data" / "raw"
METADATA_PATH = RAW_DIR / "metadata.json"

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


def upload_to_dagshub(stage: str) -> None:
    repo = dagshub_repo()
    if not repo:
        print("DAGSHUB_REPO is not set; skipping raw-data upload.")
        return

    try:
        import dagshub
    except ImportError:
        print("dagshub is not installed; skipping raw-data upload.")
        return

    remote_path = f"data/{stage}/raw"
    print(f"Uploading raw data to DagsHub bucket: {repo} -> {remote_path}")
    dagshub.upload_files(
        repo,
        RAW_DIR,
        remote_path=remote_path,
        bucket=True,
        commit_message=f"Upload raw scam dataset for {stage}",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--branch",
        default=detect_branch(),
        help="Logical branch bucket to use for DagsHub uploads.",
    )
    args = parser.parse_args()
    stage = stage_for_branch(args.branch)

    if METADATA_PATH.exists():
        print(
            f"Dataset metadata found at {METADATA_PATH}. Skipping download to avoid breaking data provenance."
        )
        print("If you need to re-download, delete the data/raw/ directory first.")
        return 0

    print("Downloading dataset from Kaggle...")
    path = Path(kagglehub.dataset_download("ibrahimbagwan12/composite-scam-transcript-dataset"))

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    copied_files: list[str] = []
    for file_path in path.iterdir():
        dst = RAW_DIR / file_path.name
        if file_path.is_dir():
            shutil.copytree(file_path, dst, dirs_exist_ok=True)
            copied_files.append(f"{file_path.name}/")
            print(f"Copied directory {file_path.name} to {RAW_DIR}")
        else:
            shutil.copy(file_path, dst)
            copied_files.append(file_path.name)
            print(f"Copied {file_path.name} to {RAW_DIR}")

    metadata = {
        "dataset": "composite-scam-transcript-dataset",
        "source": "kaggle",
        "stage": stage,
        "branch": args.branch,
        "download_timestamp": dt.datetime.now().isoformat(),
        "files": copied_files,
    }

    with METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to {METADATA_PATH}")
    upload_to_dagshub(stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
