import os
import subprocess

def main():
    print("--- Preparing Whisper Models ---")
    
    # Check if whisper.cpp models exist
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, "ggml-base.en.bin")
    if not os.path.exists(model_path):
        print("Downloading Whisper ggml-base.en model...")
        
        # We download directly using curl to the models directory
        url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
        subprocess.run(["curl", "-L", "-o", model_path, url], check=True)
        print("Download complete.")
    else:
        print("Whisper model already exists.")

if __name__ == "__main__":
    main()
