import os
import time
import json
import itertools
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import dagshub
import mlflow
from dotenv import load_dotenv
from src.evaluation.inference_pipeline import InferencePipeline

def get_dir_size(path):
    if not os.path.exists(path):
        return 0
    if os.path.isfile(path):
        return os.path.getsize(path)
    total_size = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if not os.path.islink(fp):
                total_size += os.path.getsize(fp)
    return total_size

def evaluate_combinations():
    print("--- Running E2E Combinatorial Benchmarks ---")
    
    # Check for audio files
    audio_dir = "data/large_audio_test"
    if not os.path.exists(audio_dir):
        print(f"Audio directory {audio_dir} not found. Ensure raw data was downloaded.")
        return
        
    files = [f for f in os.listdir(audio_dir) if f.endswith(".wav")]
    files.sort()
    assert len(files) > 0, f"CRITICAL ERROR: No .wav files found in {audio_dir}."
    
    manifest_path = os.path.join(audio_dir, "manifest.csv")
    manifest = pd.read_csv(manifest_path) if os.path.exists(manifest_path) else None
    
    assert manifest is not None, f"CRITICAL ERROR: {manifest_path} not found. Audio benchmark cannot proceed."
    assert len(files) == len(manifest), "CRITICAL ERROR: Downloaded .wav files do not match expected manifest count."
        
    # Define the variants available
    classifier_variants = ["fp16", "gguf", "gguf_pruned"]
    whisper_variants = ["fp16", "bf16", "q8_0", "q4_k"]
    
    # Generate all combinations
    combinations = list(itertools.product(classifier_variants, whisper_variants))
    
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/08_combo_benchmark")
        
    results = []
    
    # We create a dummy config that we'll inject values into
    with open("configs/inference_config.json", "r") as f:
        base_config = json.load(f)
        
    for clf, asr in combinations:
        print(f"\\n--- Benchmarking Combo: Classifier={clf} | ASR={asr} ---")
        
        run_name = f"combo_{clf}_{asr}"
        config = base_config.copy()
        
        config["classifier_backend"] = "gguf" if "gguf" in clf else "fp16"
        config["asr_backend"] = "fp16" if asr == "fp16" else "gguf"
        
        if clf == "gguf":
            config["backend"] = "gguf"
            config["classifier_model_path"] = "models/gguf_classifier/classifier_q8_0.gguf"
        elif clf == "gguf_pruned":
            config["backend"] = "gguf"
            config["classifier_model_path"] = "models/gguf_classifier_pruned/classifier_q8_0.gguf"
        else:
            config["backend"] = "fp16"
            config["fp16_classifier_model_name"] = "./scam-classifier-model-transcript"
            
        if asr == "fp16":
            config["asr_model_path"] = "models/ggml_whisper/whisper_f16.bin"
        elif asr == "bf16":
            config["asr_model_path"] = "models/ggml_whisper/whisper_bf16.bin"
        elif asr == "q8_0":
            config["asr_model_path"] = "models/ggml_whisper/whisper_q8_0.bin"
        else:
            config["asr_model_path"] = "models/ggml_whisper/whisper_q4_k.bin"
            
        if not os.path.exists(config.get("classifier_model_path", "")) and clf != "fp16":
            print(f"Classifier path {config.get('classifier_model_path')} not found. Skipping combo.")
            continue
            
        if not os.path.exists(config.get("asr_model_path", "")) and asr != "fp16":
            print(f"ASR path {config.get('asr_model_path')} not found. Skipping combo.")
            continue

        try:
            pipeline = InferencePipeline(config)
        except Exception as e:
            print(f"Failed to initialize pipeline for {clf}+{asr}: {e}")
            continue
            
        y_true = []
        y_pred = []
        latencies = []
        
        for idx, row in manifest.iterrows():
            audio_path = os.path.join(audio_dir, os.path.basename(row['file']))
            if not os.path.exists(audio_path): continue
            
            y_true.append(row['label'])
            
            start_time = time.time()
            try:
                res = pipeline.process_audio(audio_file=audio_path)
                y_pred.append(res['prediction'])
            except Exception as e:
                print(f"Failed prediction on {audio_path}: {e}")
                y_pred.append(0)
            end_time = time.time()
            latencies.append(end_time - start_time)
            
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        avg_lat = sum(latencies) / len(latencies)
        
        clf_path = config.get("classifier_model_path") if clf != "fp16" else config.get("fp16_classifier_model_name")
        asr_path = config.get("asr_model_path") if asr != "fp16" else config.get("fp16_asr_model_name")
        
        clf_size = get_dir_size(clf_path)
        asr_size = get_dir_size(asr_path)
        total_size_mb = (clf_size + asr_size) / (1024 * 1024)
        
        print(f"Results -> Acc: {acc:.4f}, F1: {f1:.4f}, Latency: {avg_lat:.3f}s, Size: {total_size_mb:.1f}MB")
        
        res_dict = {
            "classifier": clf,
            "asr": asr,
            "accuracy": acc,
            "f1": f1,
            "latency": avg_lat,
            "size_mb": total_size_mb
        }
        results.append(res_dict)
        
        if repo_owner and repo_name:
            with mlflow.start_run(run_name=run_name):
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_metric("latency_sec", avg_lat)
                mlflow.log_metric("size_mb", total_size_mb)
                
    # Save combo benchmark results
    df_res = pd.DataFrame(results)
    df_res.to_csv("combo_benchmark_results.csv", index=False)
    print("Combinatorial benchmarking complete. Results saved to combo_benchmark_results.csv")

if __name__ == "__main__":
    evaluate_combinations()
