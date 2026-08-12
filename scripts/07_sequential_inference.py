"""End-to-end sequential inference for one audio file or a directory of clips."""

import argparse
import os
import time
from pathlib import Path

import librosa
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import WhisperForConditionalGeneration, WhisperProcessor


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


def load_whisper_model(model_path):
    if os.path.isdir(model_path):
        print(f"Loading Whisper model from local directory: {model_path}")
        processor = WhisperProcessor.from_pretrained(model_path)
        whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
        return whisper_model, processor

    if model_path.endswith((".pt", ".pth")) and os.path.exists(model_path):
        print(f"Loading Whisper model from quantized checkpoint: {model_path}")
        whisper_model = torch.load(model_path, map_location="cpu")
        processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
        return whisper_model, processor

    print(f"Loading Whisper model from Hugging Face: {model_path}")
    processor = WhisperProcessor.from_pretrained(model_path)
    whisper_model = WhisperForConditionalGeneration.from_pretrained(model_path)
    return whisper_model, processor


def load_classifier_model(model_path, device):
    if os.path.isdir(model_path):
        print(f"Loading classifier from local directory: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        return model, tokenizer, device

    if model_path.endswith((".pt", ".pth")) and os.path.exists(model_path):
        print(f"Loading classifier from quantized checkpoint: {model_path}")
        tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        model = torch.load(model_path, map_location="cpu")
        return model, tokenizer, "cpu"

    print(f"Loading classifier from Hugging Face or registry-compatible path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    return model, tokenizer, device


def iter_audio_paths(audio_input):
    path = Path(audio_input)
    if path.is_dir():
        return sorted([str(p) for p in path.glob("*.wav")])
    return [str(path)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio_path", type=str, default="data/raw_audio/jNQXAC9IVRw.wav")
    parser.add_argument("--whisper_model_path", type=str, default="openai/whisper-tiny")
    parser.add_argument("--classifier_model_path", type=str, default="models/quantized_classifier/classifier_int8.pt")
    args = parser.parse_args()

    print("Initializing Sequential Inference Pipeline...")

    # Device routing
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Hardware Accelerator available: {device.upper()}")
    
    print("\nLoading Whisper Processor and Model...")
    whisper_model, processor = load_whisper_model(args.whisper_model_path)

    print("Loading Text Classifier...")
    classifier_model, tokenizer, active_device = load_classifier_model(args.classifier_model_path, device)
    classifier_model.eval()

    audio_paths = iter_audio_paths(args.audio_path)
    if not audio_paths:
        print(f"No audio files found under {args.audio_path}")
        return

    for audio_file in audio_paths:
        if not os.path.exists(audio_file):
            print(f"[WARNING] Audio file not found: {audio_file}")
            continue

        start = time.perf_counter()
        transcript = transcribe_audio(audio_file, whisper_model, processor)
        scam_prob = classify_text(transcript, classifier_model, tokenizer, device=active_device)
        elapsed_ms = (time.perf_counter() - start) * 1000

        print(f"  -> Extracted Text: {transcript}")
        print(f"\n=============================")
        print(f"FILE: {audio_file}")
        print(f"FINAL SCAM PROBABILITY: {scam_prob:.2%}")
        print(f"END-TO-END LATENCY: {elapsed_ms:.2f} ms")
        if scam_prob > 0.5:
            print(f"VERDICT: 🚨 SCAM ROBOCALL 🚨")
        else:
            print(f"VERDICT: ✅ LEGITIMATE CALL ✅")
        print(f"=============================\n")

if __name__ == "__main__":
    main()
