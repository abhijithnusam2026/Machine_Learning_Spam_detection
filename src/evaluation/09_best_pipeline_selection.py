import os
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import dagshub
import mlflow
from dotenv import load_dotenv
from src.utils.mlflow_reporting import log_benchmark_plots, log_dataframe_artifact, log_json_artifact

def select_best_pipeline():
    print("--- Selecting Best Pipeline Configuration ---")
    
    results_path = "combo_benchmark_results.csv"
    if not os.path.exists(results_path):
        print(f"Results file {results_path} not found. Ensure 08_combo_benchmark.py ran successfully.")
        return
        
    df = pd.read_csv(results_path)
    
    if df.empty:
        print("Results file is empty.")
        return
        
    # User-approved scoring rule
    # score = 100 * accuracy - 1.0 * latency - 0.01 * size
    w_acc = 100.0
    w_lat = 1.0
    w_size = 0.01
    
    df['score'] = (w_acc * df['accuracy']) - (w_lat * df['latency']) - (w_size * df['size_mb'])
    
    best_row = df.loc[df['score'].idxmax()]
    print("\\n=== BEST COMBINATION FOUND ===")
    print(f"Classifier: {best_row['classifier']}")
    print(f"ASR: {best_row['asr']}")
    print(f"Accuracy: {best_row['accuracy']:.4f}")
    print(f"Latency: {best_row['latency']:.3f} s")
    print(f"Size: {best_row['size_mb']:.1f} MB")
    print(f"Score: {best_row['score']:.4f}")
    print("==============================\\n")
    
    # Write to config
    config_path = "configs/inference_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
        
    clf = best_row['classifier']
    asr = best_row['asr']
    
    classifier_paths = {
        "gguf": "models/gguf_classifier/classifier_q8_0.gguf",
        "gguf_pruned": "models/gguf_classifier_pruned/classifier_q8_0.gguf",
    }
    whisper_paths = {
        "fp16": "openai/whisper-tiny.en",
        "bf16": "models/ggml_whisper/whisper_bf16.bin",
        "q8_0": "models/ggml_whisper/whisper_q8_0.bin",
        "q4_k": "models/ggml_whisper/whisper_q4_k.bin",
    }

    if clf in classifier_paths:
        config["classifier_backend"] = "gguf"
        config["classifier_model_path"] = classifier_paths[clf]
        config["gguf_classifier_head_path"] = "models/gguf/gguf_classifier_head.joblib"
    else:
        config["classifier_backend"] = "fp16"
        config["fp16_classifier_model_name"] = "./scam-classifier-model-transcript-lora"
        
    if asr == "fp16":
        config["asr_backend"] = "fp16"
        config["fp16_asr_model_name"] = whisper_paths[asr]
    else:
        config["asr_backend"] = "gguf"
        config["asr_model_path"] = whisper_paths[asr]
        
    # Clean up old/unused keys
    if "backend" in config:
        del config["backend"]
    if "gguf_classifier_model_path" in config:
        del config["gguf_classifier_model_path"]
    if "ggml_whisper_model_path" in config:
        del config["ggml_whisper_model_path"]
        
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
        
    print(f"Successfully wrote winning configuration to {config_path}")
    
    # Write a dedicated backup for reference
    best_config_path = "configs/best_inference_config.json"
    with open(best_config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"Saved a backup to {best_config_path}")

    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if repo_owner and repo_name:
        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/09_best_pipeline_selection")
        with mlflow.start_run(run_name="best_pipeline_selection"):
            mlflow.set_tag("project_stage", "refactored_pipeline")
            mlflow.set_tag("pipeline_stage", "09_best_pipeline_selection")
            mlflow.log_param("scoring_rule", "100 * accuracy - latency_sec - 0.01 * size_mb")
            mlflow.log_param("selected_classifier", best_row["classifier"])
            mlflow.log_param("selected_asr", best_row["asr"])
            mlflow.log_metric("selected_accuracy", float(best_row["accuracy"]))
            mlflow.log_metric("selected_f1_score", float(best_row["f1"]))
            mlflow.log_metric("selected_latency_sec", float(best_row["latency"]))
            mlflow.log_metric("selected_size_mb", float(best_row["size_mb"]))
            mlflow.log_metric("selected_score", float(best_row["score"]))
            ranked_df = df.sort_values("score", ascending=False)
            log_dataframe_artifact(ranked_df, "ranked_pipeline_combinations.csv", "selection")
            plot_df = ranked_df.copy()
            plot_df["combination"] = plot_df["classifier"].astype(str) + "+" + plot_df["asr"].astype(str)
            log_benchmark_plots(plot_df, "combination", ["score", "accuracy", "latency", "size_mb"], "selection", "selection")
            log_json_artifact(config, "best_inference_config.json", "selection")
            mlflow.log_artifact(results_path, artifact_path="selection")

if __name__ == "__main__":
    select_best_pipeline()
