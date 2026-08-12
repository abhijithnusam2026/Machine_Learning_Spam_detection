"""
Benchmarks GGUF/GGML models using llama-cpp-python.
Logs metrics (Latency, Throughput) to DagsHub MLflow.
"""

import os
import time
import pandas as pd
import mlflow
from dotenv import load_dotenv

def benchmark_gguf_latency(model_path, texts, num_runs=50):
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
        
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed. Run: pip install llama-cpp-python")
        return None
        
    print(f"Loading {model_path} via llama.cpp...")
    # Using embedding=True for BERT style classification models
    llm = Llama(model_path=model_path, embedding=True, verbose=False)
    
    latencies = []
    print(f"Benchmarking {model_path} ({num_runs} runs)...")
    for text in texts[:num_runs]:
        start = time.time()
        # Create embedding / forward pass
        _ = llm.create_embedding(str(text))
        latencies.append((time.time() - start) * 1000) # ms
        
    avg_latency = sum(latencies) / len(latencies)
    print(f"  -> Avg Latency: {avg_latency:.2f} ms")
    return avg_latency

def main():
    print("Initializing GGUF Benchmarking Pipeline...\n")
    load_dotenv()
    
    # MLflow Setup
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        import dagshub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    mlflow.set_experiment("scam-detection-quantized")
    
    # Load dataset
    csv_path = "data/phase2_asr/ptq_calibration.csv"
    if not os.path.exists(csv_path):
        print(f"Data not found locally at {csv_path}. Fetching from DagsHub S3...")
        if repo_owner and repo_name:
            s3 = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            s3.download_file(repo_name, csv_path, csv_path)
            print("  [SUCCESS] Data downloaded.")
        else:
            print("  [FAILED] Cannot download data. Exiting.")
            return
            
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    
    # Benchmark the 4 GGUF precision variants
    models_to_test = {
        "F16": "models/gguf_classifier/classifier_f16.gguf",
        "BF16": "models/gguf_classifier/classifier_bf16.gguf",
        "Q8_0": "models/gguf_classifier/classifier_q8_0.gguf",
        "Q4_K_M": "models/gguf_classifier/classifier_q4_k_m.gguf"
    }
    
    results = {}
    for precision, path in models_to_test.items():
        if os.path.exists(path):
            lat = benchmark_gguf_latency(path, texts)
            if lat:
                results[precision] = lat
    
    if results:
        with mlflow.start_run(run_name="GGUF_llama.cpp_Benchmark"):
            for precision, lat in results.items():
                mlflow.log_metric(f"gguf_{precision.lower()}_latency_ms", lat)
            print("\n[SUCCESS] Logged GGUF benchmarks to MLflow!")
    else:
        print("\n[WARNING] No GGUF models were successfully benchmarked.")

if __name__ == "__main__":
    main()
