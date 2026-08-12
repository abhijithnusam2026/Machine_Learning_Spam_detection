"""Apply post-training quantization to the ASR and classifier models."""

import os
import subprocess
from pathlib import Path

import torch
import warnings
import dagshub
from dotenv import load_dotenv
from transformers import WhisperForConditionalGeneration, AutoModelForSequenceClassification

REPO_ROOT = Path(__file__).resolve().parents[1]
BRANCH_STAGE_MAP = {
    "feature/phase-2-audio-asr": "feature/phase-2-audio-asr",
    "feature/phase-3-serving-quantization": "feature/phase-3-serving-quantization",
    "feature/phase-1.5-ultimate-dataset": "model-modernbert-universal",
    "model-long-context": "model-modernbert-universal",
    "model-distilbert": "model-distilbert",
    "main": "main",
}

# Suppress warnings and set ARM quantization engine for Apple Silicon
warnings.filterwarnings("ignore")
if torch.backends.quantized.supported_engines and 'qnnpack' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'qnnpack'


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

def upload_to_dagshub(local_path, remote_path, stage):
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
            s3_client.upload_file(local_path, repo_name, remote_path)
            print(f"  [SUCCESS] Uploaded to S3: {remote_path}")
        except Exception as e:
            print(f"  [FAILED] S3 Upload failed: {e}")
    else:
        print("  [WARNING] Skipping S3 upload (missing DagsHub credentials).")

def quantize_model(model):
    """
    Applies dynamic quantization to Linear layers of the given model.
    """
    print("Applying dynamic quantization to INT8...")
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized_model

def quantize_whisper(model_name="openai/whisper-tiny", output_dir="models/quantized_whisper", stage="feature/phase-2-audio-asr"):
    print(f"\n--- Quantizing Whisper ASR ({model_name}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original FP32 model
    print("Loading original model...")
    model = WhisperForConditionalGeneration.from_pretrained(model_name)
    
    # Quantize
    quantized_model = quantize_model(model)
    
    # Save the quantized model using torch.save
    save_path = os.path.join(output_dir, "whisper_int8.pt")
    torch.save(quantized_model, save_path)
    
    # Compare sizes
    fp32_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    int8_size = os.path.getsize(save_path) / (1024 * 1024)
    
    print(f"[SUCCESS] Whisper Quantization Complete.")
    print(f"FP32 Estimated Size: {fp32_size:.2f} MB")
    print(f"INT8 Saved File Size: {int8_size:.2f} MB")
    print(f"Saved to: {save_path}")
    
    upload_to_dagshub(save_path, f"artifacts/{stage}/quantization/whisper/{os.path.basename(save_path)}", stage)

def quantize_classifier(model_name="models:/ModernBERT-Scam-Classifier-feature-phase-2-audio-asr/latest", output_dir="models/quantized_classifier", stage="feature/phase-2-audio-asr"):
    print(f"\n--- Quantizing Classifier ({model_name}) ---")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load original FP32 model
    # Note: Since Phase 2 hasn't finished, we default to the base model for testing,
    # or the user can provide a path to their downloaded Phase 1.5 weights.
    print("Loading original model...")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        print(f"Failed to load {model_name}. Ensure it exists or is a valid HF repo. Error: {e}")
        return

    # Quantize
    quantized_model = quantize_model(model)
    
    # Save
    save_path = os.path.join(output_dir, "classifier_int8.pt")
    torch.save(quantized_model, save_path)
    
    # Compare sizes
    fp32_size = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    int8_size = os.path.getsize(save_path) / (1024 * 1024)
    
    print(f"[SUCCESS] Classifier Quantization Complete.")
    print(f"FP32 Estimated Size: {fp32_size:.2f} MB")
    print(f"INT8 Saved File Size: {int8_size:.2f} MB")
    print(f"Saved to: {save_path}")

    upload_to_dagshub(save_path, f"artifacts/{stage}/quantization/classifier/{os.path.basename(save_path)}", stage)

if __name__ == "__main__":
    print("Initializing Post-Training Quantization (PTQ) Pipeline...\n")
    branch = detect_branch()
    stage = stage_for_branch(branch)
    # Using tiny for fast local CPU execution
    quantize_whisper(model_name="openai/whisper-tiny", stage=stage)
    quantize_classifier(stage=stage)
