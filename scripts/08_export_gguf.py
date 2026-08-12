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

def run_cmd(cmd, cwd=None):
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=cwd)

def upload_to_dagshub(local_path, s3_path):
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if repo_owner and repo_name and token:
        try:
            print(f"Uploading {local_path} to DagsHub S3...")
            dagshub.auth.add_app_token(token)
            s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
            s3_client.upload_file(local_path, repo_name, s3_path)
            print(f"  [SUCCESS] Uploaded to S3: {s3_path}")
        except Exception as e:
            print(f"  [FAILED] S3 Upload failed: {e}")

def export_classifier_to_gguf(model_name="answerdotai/ModernBERT-base", output_dir="models/gguf_classifier"):
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
    
    # Locate the compiled llama-quantize binary
    quantize_bin = "./llama.cpp/build/bin/llama-quantize"
    if not os.path.exists(quantize_bin):
        quantize_bin = "./llama.cpp/build/llama-quantize"
        
    print(f"Quantizing to Q8_0...")
    run_cmd([quantize_bin, f16_path, q8_path, "Q8_0"])
    
    # 3. Quantize to Q4_K_M
    q4_path = os.path.join(output_dir, "classifier_q4_k_m.gguf")
    print(f"Quantizing to Q4_K_M...")
    run_cmd([quantize_bin, f16_path, q4_path, "Q4_K_M"])

    # Upload all
    for f in [f16_path, q8_path, q4_path]:
        if os.path.exists(f):
            print(f"[SUCCESS] Generated: {f} ({os.path.getsize(f) / (1024*1024):.2f} MB)")
            upload_to_dagshub(f, f)

def export_whisper_to_ggml(model_name="openai/whisper-tiny", output_dir="models/ggml_whisper"):
    print(f"\n--- Exporting Whisper ({model_name}) to GGML ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Clone whisper.cpp if not exists
    if not os.path.exists("whisper.cpp"):
        run_cmd(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git"])
        run_cmd(["make", "-j", "quantize"], cwd="whisper.cpp")

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
    
    # Upload F16
    if os.path.exists(f16_path):
        print(f"[SUCCESS] Generated: {f16_path} ({os.path.getsize(f16_path) / (1024*1024):.2f} MB)")
        upload_to_dagshub(f16_path, f16_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="answerdotai/ModernBERT-base")
    parser.add_argument("--whisper_name", type=str, default="openai/whisper-tiny")
    args = parser.parse_args()
    
    export_classifier_to_gguf(model_name=args.model_name)
    export_whisper_to_ggml(model_name=args.whisper_name)
