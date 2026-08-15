import os
import random
import boto3
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

def main():
    print("Fetching audio samples from PolyAI/minds14 (en-US)...")
    ds = load_dataset("PolyAI/minds14", name="en-US", split="train")
    
    os.makedirs("data/test_audio", exist_ok=True)
    
    # Select 5 random indices
    indices = random.sample(range(len(ds)), 5)
    
    saved_files = []
    for i, idx in enumerate(indices):
        sample = ds[idx]
        audio_array = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        intent_class = sample["intent_class"]
        
        import soundfile as sf
        filename = f"data/test_audio/sample_{i}_class_{intent_class}.wav"
        sf.write(filename, audio_array, sr)
        saved_files.append(filename)
        print(f"Saved {filename}")

    print("Uploading to DagsHub S3...")
    owner = os.getenv("DAGSHUB_REPO_OWNER", "kureeltanishq")
    name = os.getenv("DAGSHUB_REPO_NAME", "2026SU_MS_DSP_422-DL_SEC61_Machine_Learning_Spam_detection")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    import dagshub
    if token:
        dagshub.auth.add_app_token(token)
        
    s3 = dagshub.get_repo_bucket_client(f"{owner}/{name}")
    bucket = name

    for file in saved_files:
        s3_path = f"artifacts/test_audio/{os.path.basename(file)}"
        s3.upload_file(file, bucket, s3_path)
        print(f"Uploaded {s3_path}")
    
    print("SUCCESS: 5 random audio test files successfully prepared and synced to DagsHub!")

if __name__ == "__main__":
    main()
