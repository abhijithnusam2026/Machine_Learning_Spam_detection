import os
import subprocess

RAW_DIR = "data/raw_external"
os.makedirs(RAW_DIR, exist_ok=True)

datasets = [
    "rivalcults/youtube-scam-phone-call-transcripts",
    "mealss/call-transcripts-scam-determinations",
    "jxxn03x/thai-call-center-call-log-dataset"
]

print("1. Downloading Kaggle Datasets...")
for ds in datasets:
    print(f"Downloading {ds}...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", ds, "-p", RAW_DIR, "--unzip"],
            check=True
        )
    except Exception as e:
        print(f"Failed to download {ds}. Ensure your kaggle.json is configured. Error: {e}")

print("\n2. Downloading Switchboard Dialog Act Corpus (SWDA)...")
swda_dir = os.path.join(RAW_DIR, "swda")
if not os.path.exists(swda_dir):
    try:
        subprocess.run(
            ["git", "clone", "https://github.com/cgpotts/swda.git", swda_dir],
            check=True
        )
    except Exception as e:
        print(f"Failed to clone SWDA repo: {e}")
else:
    print("SWDA already cloned.")

print("\nAll external datasets processed!")
