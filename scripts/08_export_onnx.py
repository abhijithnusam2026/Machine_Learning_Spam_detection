"""
Exports the ASR (Whisper) and Classifier models to ONNX format 
for edge and mobile NPU execution (e.g. CoreML, NNAPI).
"""

import os
import torch
import warnings
from transformers import WhisperForConditionalGeneration, AutoModelForSequenceClassification, AutoTokenizer, AutoProcessor

warnings.filterwarnings("ignore")

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
    except subprocess.CalledProcessError as e:
        print(f"[FAILED] Could not run optimum-cli. Ensure 'optimum[exporters]' is installed. Error: {e}")
    except FileNotFoundError:
        print(f"[FAILED] optimum-cli not found. Please run: pip install optimum[exporters]")

if __name__ == "__main__":
    print("Initializing ONNX Export Pipeline for Edge/NPU...\n")
    export_classifier_to_onnx()
    export_whisper_to_onnx()
