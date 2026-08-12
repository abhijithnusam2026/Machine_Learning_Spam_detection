"""Benchmark FP32, FP16, and INT8 audio-branch classifiers."""

import os
import time
import subprocess
from pathlib import Path

import torch
import pandas as pd
import mlflow
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_STAGE_MAP = {
    "feature/phase-2-audio-asr": "feature/phase-2-audio-asr",
    "feature/phase-3-serving-quantization": "feature/phase-3-serving-quantization",
    "feature/phase-1.5-ultimate-dataset": "model-modernbert-universal",
    "model-long-context": "model-modernbert-universal",
    "model-distilbert": "model-distilbert",
    "main": "main",
}


def detect_branch(default="main"):
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


def stage_for_branch(branch):
    return BRANCH_STAGE_MAP.get(branch, branch.replace("/", "-") or "main")

def benchmark_latency(model, tokenizer, texts, device, num_runs=50):
    """Measures average inference latency in milliseconds."""
    model.eval()
    model.to(device)
    
    # Warm up
    sample = tokenizer("Warmup text", return_tensors="pt").to(device)
    for _ in range(5):
        with torch.no_grad():
            _ = model(**sample)
            
    latencies = []
    for text in texts[:num_runs]:
        inputs = tokenizer(str(text), return_tensors="pt", max_length=512, truncation=True).to(device)
        start = time.time()
        with torch.no_grad():
            _ = model(**inputs)
        latencies.append((time.time() - start) * 1000) # ms
        
    avg_latency = sum(latencies) / len(latencies)
    return avg_latency


def benchmark_accuracy(model, tokenizer, texts, labels, device):
    model.eval()
    model.to(device)
    preds = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(str(text), return_tensors="pt", max_length=512, truncation=True).to(device)
            logits = model(**inputs).logits
            preds.append(int(torch.argmax(logits, dim=-1).item()))
    correct = sum(int(pred == label) for pred, label in zip(preds, labels))
    return correct / max(len(labels), 1)

def main():
    print("Initializing PTQ Benchmarking...")
    load_dotenv()
    branch = detect_branch()
    stage = stage_for_branch(branch)
    stage_slug = stage.replace("/", "-")
    
    # Check MLflow config
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        import dagshub
        dagshub.init(repo_owner=repo_owner, repo_name=repo_name, mlflow=True)
    
    mlflow.set_experiment(f"scam-detection/{stage}/quantization")
    
    # Load dataset
    csv_path = "data/phase2_asr/ptq_calibration.csv"
    if not os.path.exists(csv_path):
        print(f"Data not found locally at {csv_path}. Fetching from DagsHub S3...")
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
        repo_name = os.getenv("DAGSHUB_REPO_NAME")
        token = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if repo_owner and repo_name and token:
            import dagshub
            dagshub.auth.add_app_token(token)
            s3 = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
            os.makedirs(os.path.dirname(csv_path), exist_ok=True)
            try:
                s3.download_file(repo_name, f"data/{stage}/phase2_asr/ptq_calibration.csv", csv_path)
                print("  [SUCCESS] Data downloaded.")
            except Exception as e:
                print(f"  [FAILED] S3 Download error: {e}")
                return
        else:
            print("  [FAILED] Missing DagsHub credentials. Skipping benchmark.")
            return
        
    df = pd.read_csv(csv_path)
    texts = df['text'].tolist()
    labels = df['label'].tolist()
    
    model_name = f"models:/ModernBERT-Scam-Classifier-{stage_slug}/latest"
    print("Loading tokenizer and Baseline FP32 Model...")
    if model_name.startswith("models:/"):
        local_model_path = mlflow.artifacts.download_artifacts(artifact_uri=model_name)
        components = mlflow.transformers.load_model(local_model_path, return_type="components")
        tokenizer = components["tokenizer"]
        model_fp32 = components["model"]
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_fp32 = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Measure Baseline (CPU)
    print("Benchmarking Baseline (FP32) on CPU...")
    fp32_latency = benchmark_latency(model_fp32, tokenizer, texts, device="cpu")
    print(f"  FP32 CPU Latency: {fp32_latency:.2f} ms")
    fp32_accuracy = benchmark_accuracy(model_fp32, tokenizer, texts, labels, device="cpu")
    print(f"  FP32 CPU Accuracy: {fp32_accuracy:.4f}")

    fp16_latency = None
    fp16_accuracy = None
    if torch.cuda.is_available():
        print("Benchmarking Baseline (FP16) on CUDA...")
        model_fp16 = model_fp32.half().to("cuda")
        fp16_latency = benchmark_latency(model_fp16, tokenizer, texts, device="cuda")
        fp16_accuracy = benchmark_accuracy(model_fp16, tokenizer, texts, labels, device="cuda")
        print(f"  FP16 CUDA Latency: {fp16_latency:.2f} ms")
        print(f"  FP16 CUDA Accuracy: {fp16_accuracy:.4f}")
    
    # Measure Baseline (GPU/MPS if available)
    if torch.cuda.is_available():
        print("Benchmarking Baseline (FP32) on CUDA...")
        fp32_gpu_latency = benchmark_latency(model_fp32, tokenizer, texts, device="cuda")
        print(f"  FP32 CUDA Latency: {fp32_gpu_latency:.2f} ms")
    
    # Measure INT8 (Must be CPU)
    int8_path = "models/quantized_classifier/classifier_int8.pt"
    if os.path.exists(int8_path):
        print("Loading Quantized INT8 Model...")
        model_int8 = torch.load(int8_path, map_location="cpu")
        print("Benchmarking Quantized (INT8) on CPU...")
        int8_latency = benchmark_latency(model_int8, tokenizer, texts, device="cpu")
        int8_accuracy = benchmark_accuracy(model_int8, tokenizer, texts, labels, device="cpu")
        print(f"  INT8 CPU Latency: {int8_latency:.2f} ms")
        print(f"  INT8 CPU Accuracy: {int8_accuracy:.4f}")
        speedup = fp32_latency / int8_latency
        print(f"  Speedup vs FP32 CPU: {speedup:.2f}x")
        
        with mlflow.start_run(run_name=f"{stage}-int8-benchmark"):
            mlflow.log_metric("fp32_cpu_latency_ms", fp32_latency)
            mlflow.log_metric("fp32_cpu_accuracy", fp32_accuracy)
            if fp16_latency is not None:
                mlflow.log_metric("fp16_cuda_latency_ms", fp16_latency)
                mlflow.log_metric("fp16_cuda_accuracy", fp16_accuracy)
            mlflow.log_metric("int8_cpu_latency_ms", int8_latency)
            mlflow.log_metric("int8_cpu_accuracy", int8_accuracy)
            mlflow.log_metric("cpu_speedup_factor", speedup)
            if torch.cuda.is_available():
                mlflow.log_metric("fp32_cuda_latency_ms", fp32_gpu_latency)
    else:
        print(f"INT8 model not found at {int8_path}. Run 05_quantize_models.py first.")

if __name__ == "__main__":
    main()
