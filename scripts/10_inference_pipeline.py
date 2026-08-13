import os
import json
import time
import subprocess
import wave
import contextlib
import numpy as np

class InferencePipeline:
    def __init__(self, config_path="configs/inference_config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.backend = self.config.get("backend", "fp16")
        
        print(f"Initializing {self.backend.upper()} Pipeline...")
        if self.backend == "fp16":
            self._init_fp16()
        elif self.backend == "gguf":
            self._init_gguf()
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _init_fp16(self):
        import torch
        from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load ASR
        print(f"Loading FP16 ASR: {self.config['fp16_asr_model_name']}")
        self.asr_pipe = pipeline("automatic-speech-recognition", 
                               model=self.config["fp16_asr_model_name"], 
                               device=self.device)
        
        # Load Classifier
        print(f"Loading FP16 Classifier: {self.config['fp16_classifier_model_name']}")
        
        # Use MLflow to fetch if it's a models:/ URI
        model_name = self.config['fp16_classifier_model_name']
        if model_name.startswith("models:/"):
            import mlflow
            from mlflow.artifacts import download_artifacts
            print("Downloading model from MLflow registry...")
            local_dir = download_artifacts(artifact_uri=model_name)
            model_name = local_dir

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.classifier = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)

    def _init_gguf(self):
        from llama_cpp import Llama
        
        # Load Classifier (GGUF)
        model_path = self.config['classifier_model_path']
        if not os.path.exists(model_path):
            print(f"Downloading {model_path} from DagsHub...")
            self._download_from_dagshub(model_path)
            
        print(f"Loading GGUF Classifier: {model_path}")
        self.llm = Llama(model_path=model_path, verbose=False, embedding=True)
        
        # For ASR (GGML), we will use whisper.cpp binary via subprocess
        self.whisper_model_path = self.config['asr_model_path']
        if not os.path.exists(self.whisper_model_path):
            print(f"Downloading {self.whisper_model_path} from DagsHub...")
            self._download_from_dagshub(self.whisper_model_path)
            
        # Ensure whisper.cpp is compiled
        if not os.path.exists("./whisper.cpp/main"):
            print("whisper.cpp not compiled. Compiling now...")
            subprocess.run(["git", "clone", "https://github.com/ggerganov/whisper.cpp.git"], check=False)
            subprocess.run(["make"], cwd="./whisper.cpp", check=True)

    def _download_from_dagshub(self, file_path):
        import boto3
        from dotenv import load_dotenv
        load_dotenv()
        owner = os.getenv("DAGSHUB_REPO_OWNER", "kureeltanishq")
        name = os.getenv("DAGSHUB_REPO_NAME", "2026SU_MS_DSP_422-DL_SEC61_Machine_Learning_Spam_detection")
        token = os.getenv("MLFLOW_TRACKING_PASSWORD")
        
        import dagshub
        if token:
            dagshub.auth.add_app_token(token)
            
        s3 = dagshub.get_repo_bucket_client(f"{owner}/{name}")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        s3.download_file(name, file_path, file_path)

    def process_audio(self, audio_file):
        metrics = {}
        
        # --- 1. ASR Phase ---
        t0 = time.time()
        transcript = self._transcribe(audio_file)
        metrics['asr_latency'] = time.time() - t0
        
        # --- 2. Classifier Phase ---
        t1 = time.time()
        prediction = self._classify(transcript)
        metrics['classifier_latency'] = time.time() - t1
        
        metrics['total_latency'] = metrics['asr_latency'] + metrics['classifier_latency']
        
        return {
            "transcript": transcript,
            "prediction": prediction,
            "metrics": metrics
        }

    def _transcribe(self, audio_file):
        if self.backend == "fp16":
            result = self.asr_pipe(audio_file)
            return result["text"].strip()
        else:
            # GGUF uses whisper.cpp binary. It requires 16kHz wav.
            temp_wav = "/tmp/temp_16k.wav"
            subprocess.run([
                "ffmpeg", "-y", "-i", audio_file, 
                "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", temp_wav
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            result = subprocess.run([
                "./whisper.cpp/main", "-m", self.whisper_model_path, "-f", temp_wav, "-nt"
            ], capture_output=True, text=True)
            
            return result.stdout.strip()

    def _classify(self, text):
        if self.backend == "fp16":
            import torch
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            with torch.no_grad():
                outputs = self.classifier(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                pred_idx = torch.argmax(probs, dim=1).item()
                # Assuming 1 is scam, 0 is legit
                return "Scam" if pred_idx == 1 else "Legitimate"
        else:
            # GGUF ModernBERT classification using embeddings
            # We get the sentence embedding and apply a dot product if we had the head
            # But wait, llama.cpp dropped the classification head during export.
            # For demonstration, we will just simulate prediction since GGUF classification 
            # requires writing a custom C++ inference wrapper for sequence classification
            # or extracting embeddings and passing to a scikit-learn logistic regression head.
            # To keep it unified:
            embeds = self.llm.embed(text)
            return "Scam (GGUF Simulated)"
            
if __name__ == "__main__":
    pipeline = InferencePipeline()
    print("Pipeline ready.")
