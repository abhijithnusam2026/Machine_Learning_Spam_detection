import os
import glob
import json
import mlflow
import dagshub
import pandas as pd
from dotenv import load_dotenv

def run_benchmark():
    load_dotenv()
    owner = os.getenv("DAGSHUB_REPO_OWNER", "kureeltanishq")
    name = os.getenv("DAGSHUB_REPO_NAME", "2026SU_MS_DSP_422-DL_SEC61_Machine_Learning_Spam_detection")
    dagshub.init(repo_name=name, repo_owner=owner, mlflow=True)
    mlflow.set_experiment("scam-detection/e2e-benchmark")

    # Fetch test audio files if not present locally
    test_files = glob.glob("data/test_audio/*.wav")
    if not test_files:
        print("No test files found locally! Run 10a_prep_test_audio.py first.")
        return

    # To benchmark properly, we will modify the config dynamically and run the pipeline
    backends = ["fp16", "gguf"]
    
    results = []
    
    for backend in backends:
        print(f"\n================ BENCHMARKING: {backend.upper()} ================")
        
        # Modify config
        with open("configs/inference_config.json", "r") as f:
            config = json.load(f)
        config["backend"] = backend
        with open("configs/inference_config.json", "w") as f:
            json.dump(config, f, indent=4)
            
        # Re-import pipeline to initialize with new config
        import sys
        if "scripts.10_inference_pipeline" in sys.modules:
            del sys.modules["scripts.10_inference_pipeline"]
            
        import importlib
        spec = importlib.util.spec_from_file_location("inference_pipeline", "scripts/10_inference_pipeline.py")
        infer_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(infer_module)
        
        pipeline = infer_module.InferencePipeline()
        
        with mlflow.start_run(run_name=f"benchmark_{backend}"):
            mlflow.log_param("backend", backend)
            
            total_asr_latency = 0
            total_clf_latency = 0
            
            for file in test_files:
                print(f"Processing {file}...")
                res = pipeline.process_audio(file)
                
                total_asr_latency += res['metrics']['asr_latency']
                total_clf_latency += res['metrics']['classifier_latency']
                
                results.append({
                    "Backend": backend.upper(),
                    "File": os.path.basename(file),
                    "Transcript": res['transcript'][:50] + "...",
                    "Prediction": res['prediction'],
                    "ASR Latency (s)": round(res['metrics']['asr_latency'], 3),
                    "Clf Latency (s)": round(res['metrics']['classifier_latency'], 3),
                    "Total Latency (s)": round(res['metrics']['total_latency'], 3)
                })
                
            avg_asr = total_asr_latency / len(test_files)
            avg_clf = total_clf_latency / len(test_files)
            
            mlflow.log_metric("avg_asr_latency", avg_asr)
            mlflow.log_metric("avg_classifier_latency", avg_clf)
            mlflow.log_metric("avg_total_latency", avg_asr + avg_clf)
            
            print(f"[SUMMARY {backend.upper()}] Avg ASR: {avg_asr:.3f}s | Avg Clf: {avg_clf:.3f}s | Total: {avg_asr + avg_clf:.3f}s")
            
    # Print comparison table
    df = pd.DataFrame(results)
    print("\n\n===== FINAL END-TO-END BENCHMARK RESULTS =====")
    print(df.to_string(index=False))

    df.to_csv("benchmark_results.csv", index=False)
    mlflow.log_artifact("benchmark_results.csv")

if __name__ == "__main__":
    run_benchmark()
