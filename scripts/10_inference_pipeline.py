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
                               device=self.device,
                               chunk_length_s=30)
        
        # Load Classifier
        model_name = self.config['fp16_classifier_model_name']
        print(f"Loading FP16 Classifier: {model_name}")
        
        if model_name.startswith("models:/"):
            import mlflow
            print("Downloading and loading model from MLflow registry...")
            components = mlflow.transformers.load_model(model_name, return_type="components")
            self.tokenizer = components["tokenizer"]
            self.classifier = components["model"].to(self.device)
        else:
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
        self.llm = Llama(model_path=model_path, verbose=False, embedding=True, n_ctx=8192)
        
        # Load the custom trained Scikit-Learn classification head
        import joblib
        head_path = "models/gguf/gguf_classifier_head.joblib"
        if os.path.exists(head_path):
            print(f"Loading GGUF Classification Head: {head_path}")
            self.gguf_head = joblib.load(head_path)
        else:
            print("WARNING: GGUF Classification Head not found. Will output simulated predictions.")
            self.gguf_head = None
        
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
            
            # Find the binary
            possible_paths = [
                "./whisper.cpp/build/bin/whisper-cli",
                "./whisper.cpp/build/bin/main",
                "./whisper.cpp/bin/whisper-cli",
                "./whisper.cpp/bin/main",
                "./whisper.cpp/whisper-cli",
                "./whisper.cpp/main"
            ]
            whisper_bin = None
            for p in possible_paths:
                if os.path.exists(p):
                    whisper_bin = p
                    break
            
            if not whisper_bin:
                raise FileNotFoundError("Could not locate compiled whisper-cli or main binary in whisper.cpp directory")
                
            result = subprocess.run([
                whisper_bin, "-m", self.whisper_model_path, "-f", temp_wav, "-nt"
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
            text = text.strip()
            if not text:
                return "Legitimate (Empty Audio)"
                
            # We extract the embeddings and mean-pool them across the sequence dimension
            raw_emb = self.llm.embed(text[:30000]) # Truncate to safe char limit well within 8192 tokens
            arr = np.array(raw_emb)
            
            # Robust pooling depending on llama-cpp-python return shape
            if arr.ndim == 3:
                embeds = np.mean(arr[0], axis=0)  # (1, seq, hidden) -> (seq, hidden) -> (hidden,)
            elif arr.ndim == 2:
                embeds = np.mean(arr, axis=0)     # (seq, hidden) or (1, hidden) -> (hidden,)
            else:
                embeds = arr                      # (hidden,)
                
            
            if hasattr(self, 'gguf_head') and self.gguf_head is not None:
                # Scikit-learn expects 2D array: (n_samples, n_features)
                pred_idx = self.gguf_head.predict([embeds])[0]
                return "Scam" if pred_idx == 1 else "Legitimate"
            else:
                return "Scam (GGUF Simulated)"
            
    def process_batch(self, audio_files, batch_size=8):
        metrics = {}
        t0 = time.time()
        
        # --- 1. ASR Phase ---
        transcripts = self._transcribe_batch(audio_files, batch_size=batch_size)
        metrics['asr_latency'] = time.time() - t0
        
        # --- 2. Classifier Phase ---
        t1 = time.time()
        predictions = self._classify_batch(transcripts, batch_size=batch_size)
        metrics['classifier_latency'] = time.time() - t1
        
        metrics['total_latency'] = metrics['asr_latency'] + metrics['classifier_latency']
        
        results = []
        for t, p in zip(transcripts, predictions):
            results.append({
                "transcript": t,
                "prediction": p,
                "metrics": {
                    "asr_latency": metrics['asr_latency'] / len(audio_files),
                    "classifier_latency": metrics['classifier_latency'] / len(audio_files),
                    "total_latency": metrics['total_latency'] / len(audio_files)
                }
            })
        return results

    def _transcribe_batch(self, audio_files, batch_size=8):
        if self.backend == "fp16":
            results = self.asr_pipe(audio_files, batch_size=batch_size)
            return [res["text"].strip() for res in results]
        else:
            # GGUF/whisper.cpp doesn't support native batching across multiple files easily, 
            # so we process sequentially which natively multi-threads per file anyway
            return [self._transcribe(f) for f in audio_files]

    def _classify_batch(self, texts, batch_size=8):
        if self.backend == "fp16":
            import torch
            predictions = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(self.device)
                with torch.no_grad():
                    outputs = self.classifier(**inputs)
                    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
                    pred_idxs = torch.argmax(probs, dim=1).tolist()
                    predictions.extend(["Scam" if idx == 1 else "Legitimate" for idx in pred_idxs])
            return predictions
        else:
            return [self._classify(t) for t in texts]
            
if __name__ == "__main__":
    pipeline = InferencePipeline()
    print("Pipeline ready.")
