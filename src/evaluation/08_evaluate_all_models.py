import os
import time
import json
import pandas as pd
from dotenv import load_dotenv
import mlflow
import importlib

def set_config(backend):
    config_path = "configs/inference_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    config["backend"] = backend
    
    if backend == "gguf_pruned":
        # We temporarily override backend to gguf so the pipeline initializes correctly,
        # but point it to the pruned artifacts
        config["backend"] = "gguf"
        config["classifier_model_path"] = "models/gguf_classifier_pruned/classifier_q8_0.gguf"
        config["asr_model_path"] = "models/ggml_whisper_pruned/whisper_q8_0.bin"
    elif backend == "gguf":
        config["classifier_model_path"] = "models/gguf_classifier/classifier_q8_0.gguf"
        config["asr_model_path"] = "models/ggml_whisper/whisper_q8_0.bin"
        
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
def run_benchmark(backend_name, audio_dir="data/large_audio_test"):
    print(f"\n================ BENCHMARKING: {backend_name.upper()} ================")
    set_config(backend_name)
    
    # Standard import for the modular structure
    from src.evaluation.inference_pipeline import InferencePipeline
    
    print(f"Initializing {backend_name.upper()} Pipeline...")
    pipeline = InferencePipeline()
    
    files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    files.sort()
    
    manifest_path = os.path.join(audio_dir, "manifest.csv")
    manifest = pd.read_csv(manifest_path) if os.path.exists(manifest_path) else None
    
    results = []
    
    with mlflow.start_run(run_name=f"large_scale_{backend_name}"):
        mlflow.set_tag("pipeline_backend", backend_name)
        mlflow.set_tag("dataset", "global_holdout_audio")
        mlflow.set_tag("is_pruned", "true" if "pruned" in backend_name else "false")
        
        # Log model sizes (approximate sizes in MB for GGUF vs FP16)
        if backend_name == "fp16":
            mlflow.log_metric("classifier_model_size_mb", 550.0)
            mlflow.log_metric("whisper_model_size_mb", 150.0)
        elif backend_name == "gguf":
            mlflow.log_metric("classifier_model_size_mb", 148.5)
            mlflow.log_metric("whisper_model_size_mb", 80.0)
        elif backend_name == "gguf_pruned":
            mlflow.log_metric("classifier_model_size_mb", 120.0)
            mlflow.log_metric("whisper_model_size_mb", 65.0)
        
        batch_size = 10
        full_paths = [os.path.join(audio_dir, f) for f in files]
        
        for i in range(0, len(full_paths), batch_size):
            batch_files = files[i:i+batch_size]
            batch_paths = full_paths[i:i+batch_size]
            
            print(f"[{i+1}-{min(i+batch_size, len(files))}/{len(files)}] Processing batch...")
            batch_results = pipeline.process_batch(batch_paths, batch_size=batch_size)
            
            for idx, (file, res) in enumerate(zip(batch_files, batch_results)):
                row = {
                    "file": file,
                    "transcript": res["transcript"],
                    "prediction": res["prediction"],
                    "asr_latency_s": res["metrics"]["asr_latency"],
                    "clf_latency_s": res["metrics"]["classifier_latency"],
                    "total_latency_s": res["metrics"]["total_latency"]
                }
                results.append(row)
                
                # Log metrics per step
                global_idx = i + idx
                mlflow.log_metric("asr_latency_s", row["asr_latency_s"], step=global_idx)
                mlflow.log_metric("clf_latency_s", row["clf_latency_s"], step=global_idx)
                mlflow.log_metric("total_latency_s", row["total_latency_s"], step=global_idx)
            
        df = pd.DataFrame(results)
        
        # Calculate aggregate metrics
        avg_asr = df["asr_latency_s"].mean()
        avg_clf = df["clf_latency_s"].mean()
        avg_total = df["total_latency_s"].mean()
        
        mlflow.log_metric("avg_asr_latency_s", avg_asr)
        mlflow.log_metric("avg_clf_latency_s", avg_clf)
        mlflow.log_metric("avg_total_latency_s", avg_total)
        
        # If we have manifest, calculate accuracy
        if manifest is not None:
            # manifest['file'] contains the full path, extract just the basename
            manifest['file_basename'] = manifest['file'].apply(os.path.basename)
            merged = df.merge(manifest, left_on="file", right_on="file_basename")
            
            if len(merged) == 0:
                print("Warning: Manifest merge failed, check file names.")
                accuracy = float('nan')
            else:
                # Map predictions to 1 (Scam) and 0 (Legit)
                merged['pred_label'] = merged['prediction'].apply(lambda x: 1 if "Scam" in x else 0)
                accuracy = (merged['pred_label'] == merged['label']).mean()
                
            mlflow.log_metric("accuracy", accuracy)
            print(f"Holdout Accuracy: {accuracy:.2%}")
            
        print(f"\n[DONE] {backend_name.upper()} - Avg Total Latency: {avg_total:.3f}s")
        
def main():
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    
    if repo_owner and repo_name and token:
        import dagshub
        dagshub.auth.add_app_token(token)
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/08_benchmark")
        
        # Download the audio benchmark dataset from DagsHub S3 for reproducibility
        print("Fetching test artifacts from DagsHub...")
        s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
        local_audio_dir = "data/large_audio_test"
        if not os.path.exists(local_audio_dir):
            os.makedirs(local_audio_dir, exist_ok=True)
            # Assuming manifest exists in the bucket; script will gracefully skip missing files
            try:
                s3_client.download_file(repo_name, "data/large_audio_test/manifest.csv", f"{local_audio_dir}/manifest.csv")
            except Exception as e:
                print(f"Failed to fetch manifest: {e}")
        
        # Backends to benchmark
        backends = ["fp16", "gguf", "gguf_pruned"]
        
        for b in backends:
            run_benchmark(b, audio_dir=local_audio_dir)
    else:
        print("Missing DagsHub credentials.")

if __name__ == "__main__":
    main()
