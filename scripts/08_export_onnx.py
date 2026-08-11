"""
Exports the ASR (Whisper) and Classifier models to ONNX format 
for edge and mobile NPU execution (e.g. CoreML, NNAPI).
"""

import os
import dagshub
from dotenv import load_dotenv

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
    
    try:
        from optimum.onnxruntime import ORTModelForSequenceClassification
        from transformers import AutoTokenizer
        
        print("Exporting via Optimum ORTModel...")
        model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[SUCCESS] Exported Classifier to ONNX: {output_dir}")
        
        # Upload
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_file = os.path.join(root, file)
                upload_to_dagshub(local_file, local_file)
                
    except ImportError as e:
        print("[FAILED] Optimum is not installed correctly. Please run: pip install optimum[onnxruntime]")
        print(f"Exception details: {e}")
    except Exception as e:
        print(f"[FAILED] Export error: {e}")

def export_whisper_to_onnx(model_name="openai/whisper-tiny", output_dir="models/onnx_whisper"):
    print(f"\n--- Exporting Whisper ({model_name}) to ONNX ---")
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        from optimum.onnxruntime import ORTModelForSpeechSeq2Seq
        from transformers import AutoProcessor
        
        print("Exporting via Optimum ORTModel...")
        model = ORTModelForSpeechSeq2Seq.from_pretrained(model_name, export=True)
        processor = AutoProcessor.from_pretrained(model_name)
        
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)
        print(f"[SUCCESS] Exported Whisper to ONNX: {output_dir}")
        
        # Upload
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                local_file = os.path.join(root, file)
                upload_to_dagshub(local_file, local_file)
                
    except ImportError:
        print("[FAILED] Optimum is not installed correctly. Please run: pip install optimum[onnxruntime]")
    except Exception as e:
        print(f"[FAILED] Export error: {e}")

if __name__ == "__main__":
    print("Initializing ONNX Export Pipeline for Edge/NPU...\n")
    export_classifier_to_onnx()
    export_whisper_to_onnx()
