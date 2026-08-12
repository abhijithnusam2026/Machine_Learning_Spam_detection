import os
import sys
from dotenv import load_dotenv
import dagshub

def validate_exports():
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not all([repo_owner, repo_name, token]):
        print("❌ Missing DagsHub credentials in .env file.")
        sys.exit(1)

    try:
        dagshub.auth.add_app_token(token)
        s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
    except Exception as e:
        print(f"❌ Failed to connect to DagsHub S3: {e}")
        sys.exit(1)

    stage = "feature/phase-2-audio-asr"
    
    expected_files = {
        "GGUF Classifier (F16)": f"artifacts/{stage}/gguf/classifier_f16.gguf",
        "GGUF Classifier (Q8_0)": f"artifacts/{stage}/gguf/classifier_q8_0.gguf",
        "GGUF Classifier (Q4_K_M)": f"artifacts/{stage}/gguf/classifier_q4_k_m.gguf",
        "GGUF Classifier (BF16)": f"artifacts/{stage}/gguf/classifier_bf16.gguf",
        "GGML Whisper (F16)": f"artifacts/{stage}/ggml/ggml-tiny.bin",
        "GGML Whisper (Q8_0)": f"artifacts/{stage}/ggml/whisper_q8_0.bin",
        "GGML Whisper (Q4_K)": f"artifacts/{stage}/ggml/whisper_q4_k.bin",
        "GGML Whisper (BF16)": f"artifacts/{stage}/ggml/whisper_bf16.bin",
    }

    print("\n🔍 Validating exported models in DagsHub S3...\n")
    
    all_present = True
    for name, path in expected_files.items():
        try:
            # Check if file exists by reading metadata
            s3_client.head_object(Bucket=repo_name, Key=path)
            print(f"✅ {name} is present in S3: {path}")
        except Exception:
            print(f"❌ MISSING: {name} was not found at {path}")
            all_present = False

    print("\n========================================")
    if all_present:
        print("🎉 SUCCESS: All exported models are successfully validated in DagsHub S3!")
        print("You are officially ready to move to Phase 3 (Dynamic Inference).")
    else:
        print("⚠️ WARNING: Some models are missing. Check your previous export logs for errors.")
        sys.exit(1)

if __name__ == "__main__":
    validate_exports()
