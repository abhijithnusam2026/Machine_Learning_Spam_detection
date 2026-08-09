"""
Applies PyTorch Dynamic Post-Training Quantization (INT8) to the ASR and Classifier models.
"""

import os
import torch
import warnings
from transformers import WhisperForConditionalGeneration, AutoModelForSequenceClassification

# Suppress warnings and set ARM quantization engine for Apple Silicon
warnings.filterwarnings("ignore")
if torch.backends.quantized.supported_engines and 'qnnpack' in torch.backends.quantized.supported_engines:
    torch.backends.quantized.engine = 'qnnpack'

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

def quantize_whisper(model_name="openai/whisper-tiny", output_dir="models/quantized_whisper"):
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

def quantize_classifier(model_name="answerdotai/ModernBERT-base", output_dir="models/quantized_classifier"):
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

if __name__ == "__main__":
    print("Initializing Post-Training Quantization (PTQ) Pipeline...\n")
    # Using tiny for fast local CPU execution
    quantize_whisper(model_name="openai/whisper-tiny")
    
    # Assuming user might point this to their final Phase 1.5/2.0 weights later. 
    # For now, we quantize the base architecture.
    quantize_classifier(model_name="answerdotai/ModernBERT-base")
