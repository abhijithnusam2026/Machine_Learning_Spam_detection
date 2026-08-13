import os
import argparse
import torch
import torch.nn.utils.prune as prune
from transformers import AutoModelForSequenceClassification, AutoTokenizer, WhisperForConditionalGeneration, AutoProcessor
import mlflow

def prune_model(model, amount=0.2):
    """
    Applies L1 Unstructured Magnitude Pruning to all Linear layers in the model.
    """
    print(f"Applying {amount*100}% L1 Unstructured Magnitude Pruning...")
    parameters_to_prune = []
    
    # Collect all linear layers
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))
            
    if not parameters_to_prune:
        print("No linear layers found to prune.")
        return model
        
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Make the pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    print(f"Successfully pruned {len(parameters_to_prune)} linear layers.")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classifier", type=str, default="models:/ModernBERT-Scam-Classifier-feature-phase-2-audio-asr/latest")
    parser.add_argument("--whisper", type=str, default="openai/whisper-tiny.en")
    parser.add_argument("--prune_amount", type=float, default=0.2)
    parser.add_argument("--out_classifier", type=str, default="./pruned_classifier")
    parser.add_argument("--out_whisper", type=str, default="./pruned_whisper")
    args = parser.parse_args()
    
    # --- PRUNE CLASSIFIER ---
    print(f"\n--- Pruning Classifier: {args.classifier} ---")
    if args.classifier.startswith("models:/"):
        from dotenv import load_dotenv
        load_dotenv()
        repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
        repo_name = os.getenv("DAGSHUB_REPO_NAME")
        if repo_owner and repo_name:
            import dagshub
            dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
            
        print("Loading from MLflow...")
        components = mlflow.transformers.load_model(args.classifier, return_type="components")
        clf_model = components["model"]
        clf_tokenizer = components["tokenizer"]
    else:
        clf_model = AutoModelForSequenceClassification.from_pretrained(args.classifier)
        clf_tokenizer = AutoTokenizer.from_pretrained(args.classifier)
        
    clf_model = prune_model(clf_model, amount=args.prune_amount)
    
    os.makedirs(args.out_classifier, exist_ok=True)
    clf_model.save_pretrained(args.out_classifier)
    clf_tokenizer.save_pretrained(args.out_classifier)
    print(f"Saved pruned classifier to {args.out_classifier}")
    
    # --- PRUNE WHISPER ---
    print(f"\n--- Pruning Whisper: {args.whisper} ---")
    whisper_model = WhisperForConditionalGeneration.from_pretrained(args.whisper)
    whisper_processor = AutoProcessor.from_pretrained(args.whisper)
    
    whisper_model = prune_model(whisper_model, amount=args.prune_amount)
    
    os.makedirs(args.out_whisper, exist_ok=True)
    whisper_model.save_pretrained(args.out_whisper)
    whisper_processor.save_pretrained(args.out_whisper)
    print(f"Saved pruned whisper to {args.out_whisper}")

if __name__ == "__main__":
    main()
