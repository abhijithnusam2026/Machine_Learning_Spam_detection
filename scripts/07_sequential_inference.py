"""
End-to-End Sequential Inference Pipeline.
Accepts a raw .wav file, transcibes it via Whisper (INT8), and 
classifies the transcript via ModernBERT (INT8 or FP16).
"""

import os
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import librosa

def transcribe_audio(audio_path, whisper_model, processor):
    print(f"Transcribing {audio_path}...")
    audio, sr = librosa.load(audio_path, sr=16000)
    input_features = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
    
    # Generate token ids
    predicted_ids = whisper_model.generate(input_features)
    
    # Decode
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

def classify_text(text, classifier_model, tokenizer, device="cpu"):
    print(f"Classifying text: '{text[:50]}...'")
    inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = classifier_model(**inputs)
    
    logits = outputs.logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    
    # Class 0: Legit, Class 1: Scam
    scam_prob = probs[0][1].item()
    return scam_prob

def main():
    print("Initializing Sequential Inference Pipeline...")
    
    # Audio Path
    sample_audio = "data/raw_audio/jNQXAC9IVRw.wav" # Replace with actual path in prod
    
    # Device routing
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Hardware Accelerator available: {device.upper()}")
    
    # Load Whisper (Using FP32 for demo, but can load INT8 if needed)
    print("\nLoading Whisper Processor and Model...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    
    # Load Classifier
    print("Loading Text Classifier...")
    model_name = "answerdotai/ModernBERT-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load INT8 Model if it exists, otherwise fallback to FP32 on device
    int8_path = "models/quantized_classifier/classifier_int8.pt"
    if os.path.exists(int8_path):
        print(f"Detected Quantized Model. Forcing execution to CPU.")
        classifier_model = torch.load(int8_path, map_location="cpu")
        active_device = "cpu"
    else:
        print(f"Quantized model not found. Loading FP32 and routing to {device.upper()}.")
        classifier_model = AutoModelForSequenceClassification.from_pretrained(model_name)
        classifier_model.to(device)
        active_device = device
        
    classifier_model.eval()
    
    if not os.path.exists(sample_audio):
        print(f"\n[INFO] Provide a valid .wav path to run the E2E pipeline.")
        return
        
    # 1. Transcribe
    transcript = transcribe_audio(sample_audio, whisper_model, processor)
    print(f"  -> Extracted Text: {transcript}")
    
    # 2. Classify
    scam_prob = classify_text(transcript, classifier_model, tokenizer, device=active_device)
    
    # 3. Output
    print(f"\n=============================")
    print(f"FINAL SCAM PROBABILITY: {scam_prob:.2%}")
    if scam_prob > 0.5:
        print(f"VERDICT: 🚨 SCAM ROBOCALL 🚨")
    else:
        print(f"VERDICT: ✅ LEGITIMATE CALL ✅")
    print(f"=============================\n")

if __name__ == "__main__":
    main()
