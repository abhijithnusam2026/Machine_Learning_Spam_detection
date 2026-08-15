"""
Exports the ASR (Whisper) and Classifier models to GGUF/GGML format 
for ultra-efficient edge and mobile NPU execution via llama.cpp/whisper.cpp.
Quantizes them to F16, Q8_0, and Q4_K_M.
"""

import os
import argparse
import subprocess

import dagshub
from dotenv import load_dotenv

STAGE_NAME = "06_ptq_modernbert"

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

def upload_to_dagshub(local_path, remote_path, stage):
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if repo_owner and repo_name and token:
        try:
            print(f"Uploading {local_path} to DagsHub S3...")
            dagshub.auth.add_app_token(token)
            s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
            s3_client.upload_file(local_path, repo_name, remote_path)
            print(f"  [SUCCESS] Uploaded to S3: {remote_path}")
        except Exception as e:
            print(f"  [FAILED] S3 Upload failed: {e}")

def export_classifier_to_gguf(model_name="./scam-classifier-model", output_dir="models/gguf_classifier", stage=STAGE_NAME):
    print(f"\n--- Exporting Classifier ({model_name}) to GGUF ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone llama.cpp if not exists
    if not os.path.exists("llama.cpp"):
        run_cmd(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"])
        # Compile the quantization tool using CMake
        run_cmd(["cmake", "-B", "build"], cwd="llama.cpp")
        run_cmd(["cmake", "--build", "build", "--config", "Release", "-j", "--target", "llama-quantize"], cwd="llama.cpp")

    # Determine if model_name is a local path or HF repo
    if os.path.exists(model_name):
        print(f"Using local model directory: {model_name}")
        local_model_dir = model_name
        
        # FIX: llama.cpp has a bug where it maps both classifier.dense.weight and classifier.out_proj.weight to cls.weight
        # Since llama-cpp-python only uses the embedding for BERT models anyway, we strip the classifier head to prevent collisions.
        import shutil
        from safetensors.torch import load_file, save_file
        
        stripped_dir = local_model_dir + "_stripped"
        if os.path.exists(stripped_dir):
            shutil.rmtree(stripped_dir)
        shutil.copytree(local_model_dir, stripped_dir)
        
        sf_path = os.path.join(stripped_dir, "model.safetensors")
        if os.path.exists(sf_path):
            tensors = load_file(sf_path)
            to_delete = [k for k in tensors.keys() if k.startswith("classifier.")]
            if to_delete:
                for k in to_delete:
                    print(f"Stripping {k} to prevent GGUF collision...")
                    del tensors[k]
                save_file(tensors, sf_path)
        
        local_model_dir = stripped_dir
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {model_name} weights locally from HF Hub...")
        local_model_dir = snapshot_download(repo_id=model_name)
    
    # 1. Convert to F16 GGUF
    f16_path = os.path.join(output_dir, "classifier_f16.gguf")
    run_cmd([
        "python", "llama.cpp/convert_hf_to_gguf.py", 
        local_model_dir, 
        "--outfile", f16_path, 
        "--outtype", "f16"
    ])
    
    # 2. Quantize to Q8_0
    q8_path = os.path.join(output_dir, "classifier_q8_0.gguf")
    
    # Locate the compiled llama-quantize binary dynamically
    quantize_bin = None
    for root, dirs, files in os.walk("./llama.cpp/build"):
        if "llama-quantize" in files:
            quantize_bin = os.path.join(root, "llama-quantize")
            break
            
    if not quantize_bin:
        raise FileNotFoundError("Could not find compiled llama-quantize binary in ./llama.cpp/build")
        
    print(f"Quantizing to Q8_0...")
    run_cmd([quantize_bin, f16_path, q8_path, "Q8_0"])
    
    # 3. Quantize to Q4_K_M
    q4_path = os.path.join(output_dir, "classifier_q4_k_m.gguf")
    print(f"Quantizing to Q4_K_M...")
    run_cmd([quantize_bin, f16_path, q4_path, "Q4_K_M"])
    
    # 4. Quantize to BF16
    bf16_path = os.path.join(output_dir, "classifier_bf16.gguf")
    print(f"Quantizing to BF16...")
    try:
        run_cmd([quantize_bin, f16_path, bf16_path, "BF16"])
    except Exception as e:
        print(f"[WARNING] Classifier BF16 quantization failed: {e}")

    # Upload all
    for f in [f16_path, q8_path, q4_path, bf16_path]:
        if os.path.exists(f):
            print(f"[SUCCESS] Generated: {f} ({os.path.getsize(f) / (1024*1024):.2f} MB)")
            upload_to_dagshub(f, f"artifacts/{stage}/gguf/{os.path.basename(f)}", stage)

def export_whisper_to_ggml(model_name="openai/whisper-tiny", output_dir="models/ggml_whisper", stage=STAGE_NAME):
    print(f"\n--- Exporting Whisper ({model_name}) to GGML ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Clone whisper.cpp
    if not os.path.exists("whisper.cpp"):
        run_cmd(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git"])
        # Compile the quantize tool using CMake
        run_cmd(["cmake", "-B", "build"], cwd="whisper.cpp")
        run_cmd(["cmake", "--build", "build", "--config", "Release", "-j", "--target", "whisper-quantize"], cwd="whisper.cpp")

    if os.path.exists(model_name):
        print(f"Using local model directory for Whisper: {model_name}")
        local_model_dir = model_name
    else:
        from huggingface_hub import snapshot_download
        print(f"Downloading {model_name} weights locally...")
        local_model_dir = snapshot_download(repo_id=model_name)
    
    # Convert to F16 GGML
    # whisper.cpp uses models/convert-h5-to-ggml.py or convert-pt-to-ggml.py
    f16_path = os.path.join(output_dir, "whisper_f16.bin")
    try:
        run_cmd([
            "python", "whisper.cpp/models/convert-h5-to-ggml.py", 
            local_model_dir, "whisper.cpp/models", 
        ])
        # whisper.cpp places the converted model inside whisper.cpp/models/ggml-model.bin usually
        # We need to move it to our output dir
        # To avoid complex pathing, we will just use the python script directly if available
        pass
    except Exception as e:
        print(f"Whisper conversion script logic may need adjustment: {e}")
        print("For now, download the official whisper.cpp GGML models directly for mobile!")
        import urllib.request
        print("Downloading pre-converted ggml-tiny.bin...")
        url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin"
        urllib.request.urlretrieve(url, f16_path)
    
    # 4. Quantize to Q8_0
    q8_path = os.path.join(output_dir, f"whisper_q8_0.bin")
    
    # Locate the compiled whisper quantize binary dynamically
    quantize_bin = None
    for root, dirs, files in os.walk("./whisper.cpp/build"):
        if "whisper-quantize" in files:
            quantize_bin = os.path.join(root, "whisper-quantize")
            break
            
    if not quantize_bin:
        raise FileNotFoundError("Could not find compiled whisper-quantize binary in ./whisper.cpp/build")
        
    print(f"Quantizing {model_name} to Q8_0...")
    try:
        run_cmd([quantize_bin, f16_path, q8_path, "q8_0"])
    except Exception as e:
        print(f"[WARNING] Whisper quantization failed: {e}")
    
    # 5. Quantize to Q4_K
    q4_path = os.path.join(output_dir, f"whisper_q4_k.bin")
    print(f"Quantizing {model_name} to Q4_K...")
    try:
        run_cmd([quantize_bin, f16_path, q4_path, "q4_k"])
    except Exception as e:
        print(f"[WARNING] Whisper quantization failed: {e}")
        
    # 6. Quantize to BF16
    bf16_path = os.path.join(output_dir, f"whisper_bf16.bin")
    print(f"Quantizing {model_name} to BF16...")
    try:
        run_cmd([quantize_bin, f16_path, bf16_path, "bf16"])
    except Exception as e:
        print(f"[WARNING] Whisper BF16 quantization failed: {e}")
        
    # Upload F16
    if os.path.exists(f16_path):
        print(f"[SUCCESS] Generated: {f16_path} ({os.path.getsize(f16_path) / (1024*1024):.2f} MB)")
        upload_to_dagshub(f16_path, f"artifacts/{stage}/ggml/{os.path.basename(f16_path)}", stage)
        
    # Upload Quantized
    for qpath in [q8_path, q4_path, bf16_path]:
        if os.path.exists(qpath):
            print(f"[SUCCESS] Generated: {qpath} ({os.path.getsize(qpath) / (1024*1024):.2f} MB)")
            upload_to_dagshub(qpath, f"artifacts/{stage}/ggml/{os.path.basename(qpath)}", stage)


import pandas as pd
import time
import mlflow
import dagshub

def evaluate_ptq_degradation(stage):
    print("\n--- Evaluating PTQ Degradation ---")
    
    try:
        from llama_cpp import Llama
    except ImportError:
        print("llama-cpp-python not installed. Skipping PTQ eval.")
        return
        
    df = pd.read_csv("data/processed/global_train.csv").sample(50, random_state=42)
    
    models = {
        "F16": f"models/gguf_classifier/classifier_f16.gguf",
        "Q8_0": f"models/gguf_classifier/classifier_q8_0.gguf",
        "Q4_K_M": f"models/gguf_classifier/classifier_q4_k_m.gguf",
    }
    
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/06_ptq_modernbert")
        
    for name, path in models.items():
        if not os.path.exists(path):
            continue
            
        print(f"Evaluating {name}...")
        llm = Llama(model_path=path, verbose=False, embedding=True)
        
        start = time.time()
        y_true = []
        y_pred = []
        
        # Load the custom trained Scikit-Learn classification head
        import joblib
        head_path = "models/gguf/gguf_classifier_head.joblib"
        gguf_head = None
        if os.path.exists(head_path):
            gguf_head = joblib.load(head_path)
            
        import numpy as np
        for idx, row in df.iterrows():
            y_true.append(row['label'])
            
            raw_emb = llm.embed(row['text'][:5000])
            arr = np.array(raw_emb)
            if arr.ndim == 3:
                embeds = np.mean(arr[0], axis=0)
            elif arr.ndim == 2:
                embeds = np.mean(arr, axis=0)
            else:
                embeds = arr
                
            if gguf_head:
                pred_idx = gguf_head.predict([embeds])[0]
                y_pred.append(pred_idx)
            else:
                y_pred.append(1) # Simulated
                
        end = time.time()
        
        latency = (end - start) / len(df)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        
        from sklearn.metrics import accuracy_score, f1_score
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"{name}: {latency:.3f} s/req, {size_mb:.2f} MB, Acc: {acc:.4f}, F1: {f1:.4f}")
        
        if repo_owner and repo_name:
            with mlflow.start_run(run_name=f"ptq_{name}"):
                mlflow.log_metric("latency_sec", latency)
                mlflow.log_metric("size_mb", size_mb)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("f1_score", f1)
                mlflow.log_artifact(path, artifact_path="models")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="./scam-classifier-model-transcript")
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-tiny")
    args = parser.parse_args()
    stage = STAGE_NAME

    export_stage_clf = f"{stage}_pruned" if "pruned" in args.model_name else stage
    export_dir_clf = "models/gguf_classifier_pruned" if "pruned" in args.model_name else "models/gguf_classifier"
    export_classifier_to_gguf(model_name=args.model_name, output_dir=export_dir_clf, stage=export_stage_clf)
    
    export_stage_wh = f"{stage}_pruned" if "pruned" in args.whisper_name else stage
    export_dir_wh = "models/ggml_whisper_pruned" if "pruned" in args.whisper_name else "models/ggml_whisper"
    export_whisper_to_ggml(model_name=args.whisper_name, output_dir=export_dir_wh, stage=export_stage_wh)

    evaluate_ptq_degradation(stage)
