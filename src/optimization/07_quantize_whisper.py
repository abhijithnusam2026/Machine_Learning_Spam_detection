import os
import subprocess

def main():
    print("--- Preparing Whisper Models (Quantized Variants) ---")
    
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # We fetch the FP16 base model and pre-quantized Q5 and Q8 models for the benchmark
    variants = {
        "ggml-base.en.bin": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin",
        "ggml-base.en-q5_0.bin": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_0.bin",
        "ggml-base.en-q8_0.bin": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q8_0.bin"
    }
    
    for filename, url in variants.items():
        model_path = os.path.join(models_dir, filename)
        if not os.path.exists(model_path):
            print(f"Downloading Whisper variant: {filename}...")
            subprocess.run(["curl", "-L", "-o", model_path, url], check=True)
            print(f"Downloaded {filename}.")
        else:
            print(f"Whisper variant {filename} already exists.")
            
    print("All Whisper models prepared.")

if __name__ == "__main__":
    main()
