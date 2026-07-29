import os
import sys
import kagglehub
import shutil
import json
import datetime

raw_dir = "data/raw"
metadata_path = os.path.join(raw_dir, "metadata.json")

# Idempotency check
if os.path.exists(metadata_path):
    print(f"Dataset metadata found at {metadata_path}. Skipping download to avoid breaking data provenance.")
    print("If you need to re-download, delete the data/raw/ directory first.")
    sys.exit(0)

# Download the dataset
print("Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("ibrahimbagwan12/composite-scam-transcript-dataset")

os.makedirs(raw_dir, exist_ok=True)

# The path contains the downloaded files. We move them to the raw directory.
copied_files = []
for file in os.listdir(path):
    src = os.path.join(path, file)
    dst = os.path.join(raw_dir, file)
    shutil.copy(src, dst)
    copied_files.append(file)
    print(f"Copied {file} to {raw_dir}")

# Simulate S3 versioning/metadata
metadata = {
    "dataset": "composite-scam-transcript-dataset",
    "source": "kaggle",
    "download_timestamp": datetime.datetime.now().isoformat(),
    "files": copied_files
}

metadata_path = os.path.join(raw_dir, "metadata.json")
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=4)

print(f"Metadata saved to {metadata_path}")
