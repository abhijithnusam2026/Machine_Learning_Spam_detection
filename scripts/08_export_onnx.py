"""
Exports the ASR (Whisper) and Classifier models to ONNX format 
for edge and mobile NPU execution (e.g. CoreML, NNAPI).
"""

import os
import torch
import warnings
import dagshub
import subprocess
from dotenv import load_dotenv
from transformers import WhisperForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer, AutoProcessor

warnings.filterwarnings("ignore")

def upload_to_dagshub(local_path, s3_path):
    """Uploads a local file to the DagsHub S3 bucket."""
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
    else:
        print("  [WARNING] Skipping S3 upload (missing DagsHub credentials).")

def export_classifier_to_onnx(model_name="answerdotai/ModernBERT-base", output_dir="models/onnx_classifier"):
    print(f"\n--- Exporting Classifier ({model_name}) to ONNX ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model and tokenizer
    print("Loading FP32 Classifier...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    model.eval()
    
    # Create dummy input
    text = "Hello, this is a test."
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    
    onnx_path = os.path.join(output_dir, "classifier.onnx")
    
    print("Exporting to ONNX...")
    torch.onnx.export(
        model, 
        (inputs["input_ids"], inputs["attention_mask"]), 
        onnx_path, 
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['logits'],
        dynamic_axes={'input_ids': {0: 'batch_size'}, 
                      'attention_mask': {0: 'batch_size'}, 
                      'logits': {0: 'batch_size'}}
    )
    
    print(f"[SUCCESS] Exported Classifier to ONNX: {onnx_path}")
    print(f"ONNX File Size: {os.path.getsize(onnx_path) / (1024 * 1024):.2f} MB")
    
    upload_to_dagshub(onnx_path, onnx_path)

def export_whisper_to_onnx(model_name="openai/whisper-tiny", output_dir="models/onnx_whisper"):
    print(f"\n--- Exporting Whisper ({model_name}) to ONNX ---")
    print("Note: Exporting Whisper to ONNX requires standardizing the encoder/decoder graphs.")
    print("For a production NPU deployment, we highly recommend using `optimum-cli` instead of raw torch.onnx.")
    print("Example command:")
    print(f"  optimum-cli export onnx --model {model_name} {output_dir}/")
    
    # We will simulate the optimum command execution for robust ONNX Whisper export
    import subprocess
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        subprocess.run(["optimum-cli", "export", "onnx", "--model", model_name, output_dir], check=True)
        print(f"[SUCCESS] Exported Whisper to ONNX via Optimum: {output_dir}")
        # Upload all files in the output_dir to S3
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_file = os.path.join(root, file)
                upload_to_dagshub(local_file, local_file)
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] Could not run optimum-cli. Ensure 'optimum[exporters]' is installed. Error: {e}")
    except FileNotFoundError:
        print(f"[FAILED] optimum-cli not found. Please run: pip install optimum[exporters]")

if __name__ == "__main__":
    print("Initializing ONNX Export Pipeline for Edge/NPU...\n")
    export_classifier_to_onnx()
    export_whisper_to_onnx()
