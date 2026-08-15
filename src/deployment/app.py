import os
import json
import gradio as gr
from dotenv import load_dotenv

# We rely on configs/inference_config.json for the backend selection.
config_path = "configs/inference_config.json"
if os.path.exists(config_path):
    with open(config_path, "r") as f:
        config = json.load(f)

# Load pipeline (will auto-download models from DagsHub if DagsHub secrets are set in HF Spaces)
print("Initializing configured inference pipeline...")
import importlib
infer_module = importlib.import_module("src.evaluation.inference_pipeline")
InferencePipeline = infer_module.InferencePipeline
pipeline = InferencePipeline()
backend_label = f"ASR={pipeline.asr_backend.upper()} | Classifier={pipeline.clf_backend.upper()}"

def process_audio(audio_file_path):
    if not audio_file_path:
        return "No audio provided.", "N/A", "N/A"
        
    try:
        # Run inference using the CPU GGUF pipeline + Logistic Regression head
        result = pipeline.process_audio(audio_file_path)
        
        transcript = result.get("transcript", "")
        prediction = result.get("prediction", "Unknown")
        if prediction == 1:
            prediction = "Scam"
        elif prediction == 0:
            prediction = "Legitimate"
        metrics = result.get("metrics", {})
        
        latency_str = (f"ASR: {metrics.get('asr_latency', 0):.2f}s | "
                       f"Classifier: {metrics.get('classifier_latency', 0):.2f}s | "
                       f"Total: {metrics.get('total_latency', 0):.2f}s")
                       
        return transcript, prediction, latency_str
        
    except Exception as e:
        return f"Error: {str(e)}", "Error", "Error"

# Build Gradio UI
with gr.Blocks(title="Scam Detection AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Scam Detection AI")
    gr.Markdown(f"Upload an audio recording of a phone call. Active pipeline: `{backend_label}`.")
    
    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(type="filepath", label="Upload Audio Call (.wav, .mp3)")
            analyze_btn = gr.Button("Analyze Intent", variant="primary")
            
        with gr.Column():
            prediction_output = gr.Textbox(label="AI Prediction (Scam vs Legitimate)", lines=1)
            transcript_output = gr.Textbox(label="Whisper Transcript", lines=5)
            latency_output = gr.Textbox(label="Inference Latency", lines=1)
            
    analyze_btn.click(
        fn=process_audio,
        inputs=audio_input,
        outputs=[transcript_output, prediction_output, latency_output]
    )

if __name__ == "__main__":
    demo.launch()
