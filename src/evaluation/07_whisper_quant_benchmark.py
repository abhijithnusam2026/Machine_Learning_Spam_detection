import os
import time
import pandas as pd
import dagshub
import mlflow
from dotenv import load_dotenv

def get_whisper_model_path(variant):
    # Depending on what we exported in 06, we might have these variants
    paths = {
        "F16": "models/ggml_whisper/whisper_f16.bin",
        "BF16": "models/ggml_whisper/whisper_bf16.bin",
        "Q8_0": "models/ggml_whisper/whisper_q8_0.bin",
        "Q4_K": "models/ggml_whisper/whisper_q4_k.bin"
    }
    return paths.get(variant)

def evaluate_whisper():
    print("--- Evaluating Whisper Quantization Variants ---")
    
    manifest_path = "data/large_audio_test/manifest.csv"
    
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    
    # Fetch test data from DagsHub S3 if missing
    if repo_owner and repo_name:
        s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
        local_audio_dir = "data/large_audio_test"
        os.makedirs(local_audio_dir, exist_ok=True)
        
        try:
            s3_client.download_file(repo_name, "data/large_audio_test/manifest.csv", f"{local_audio_dir}/manifest.csv")
            manifest_df = pd.read_csv(f"{local_audio_dir}/manifest.csv")
            failed_downloads = []
            
            for _, row in manifest_df.iterrows():
                audio_filename = os.path.basename(row['file'])
                remote_audio_path = f"data/large_audio_test/{audio_filename}"
                local_audio_path = f"{local_audio_dir}/{audio_filename}"
                if not os.path.exists(local_audio_path):
                    print(f"Downloading {audio_filename}...")
                    try:
                        s3_client.download_file(repo_name, remote_audio_path, local_audio_path)
                    except Exception as inner_e:
                        print(f"Failed to fetch {audio_filename}: {inner_e}")
                        failed_downloads.append(audio_filename)
                        
            if failed_downloads:
                raise RuntimeError(f"Audio files failed to download: {failed_downloads}")
        except Exception as e:
            print(f"Failed to fetch manifest or audio files: {e}")
            raise
    
    if not os.path.exists(manifest_path):
        print(f"Manifest not found at {manifest_path}. Skipping.")
        return
        
    try:
        from pywhispercpp.model import Model
    except ImportError:
        print("pywhispercpp not installed. Skipping Whisper eval.")
        return
        
    df = pd.read_csv(manifest_path)
    
    # We will benchmark F16, BF16, Q8_0, Q4_K
    variants = ["F16", "BF16", "Q8_0", "Q4_K"]
    
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/07_whisper_quant_benchmark")
        
    for variant in variants:
        path = get_whisper_model_path(variant)
        if not path or not os.path.exists(path):
            print(f"Variant {variant} not found at {path}. Skipping.")
            continue
            
        print(f"Evaluating {variant}...")
        model = Model(path, n_threads=4, print_realtime=False, print_progress=False)
        
        start = time.time()
        for _, row in df.iterrows():
            audio_path = f"data/large_audio_test/{os.path.basename(row['file'])}"
            if os.path.exists(audio_path):
                try:
                    segments = model.transcribe(audio_path, new_segment_callback=None)
                except Exception as e:
                    print(f"Failed to transcribe {audio_path}: {e}")
        end = time.time()
        
        latency = (end - start) / len(df)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"{variant}: {latency:.3f} s/req, {size_mb:.2f} MB")
        
        if repo_owner and repo_name:
            with mlflow.start_run(run_name=f"whisper_{variant}"):
                mlflow.log_metric("latency_sec", latency)
                mlflow.log_metric("size_mb", size_mb)
                mlflow.log_artifact(path, artifact_path="models")
                
if __name__ == "__main__":
    evaluate_whisper()
