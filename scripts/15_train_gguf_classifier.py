import os
import pandas as pd
import numpy as np
from llama_cpp import Llama, LLAMA_POOLING_TYPE_MEAN
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

def main():
    print("--- Training GGUF Classification Head ---")
    
    # 1. Load GGUF Model
    model_path = "models/gguf/ModernBERT-Scam-Classifier-Q8_0.gguf"
    if not os.path.exists(model_path):
        print(f"Error: Could not find GGUF model at {model_path}")
        print("Downloading from DagsHub...")
        from dotenv import load_dotenv
        import dagshub
        import boto3
        load_dotenv()
        owner = os.getenv("DAGSHUB_REPO_OWNER")
        name = os.getenv("DAGSHUB_REPO_NAME")
        token = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if token:
            dagshub.auth.add_app_token(token)
        s3 = dagshub.get_repo_bucket_client(f"{owner}/{name}")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        s3.download_file(name, model_path, model_path)

    print(f"Loading GGUF model for embeddings: {model_path}")
    llm = Llama(model_path=model_path, verbose=False, embedding=True, pooling_type=LLAMA_POOLING_TYPE_MEAN)

    # 2. Load Phase 1.5 Training Data
    train_path = "data/phase1.5/train.csv"
    val_path = "data/phase1.5/val.csv"
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print("Error: Could not find Phase 1.5 datasets.")
        return

    print("Loading datasets...")
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)
    
    # We will use a random sample of 1000 for training to speed up extraction on CPU
    print(f"Original Train size: {len(df_train)}, Val size: {len(df_val)}")
    
    n_samples = min(1000, len(df_train))
    df_train = df_train.sample(n_samples, random_state=42)
    df_val = df_val.sample(min(200, len(df_val)), random_state=42)
    print(f"Sampled Train size: {len(df_train)}, Sampled Val size: {len(df_val)}")

    # 3. Extract Embeddings
    def extract_embeddings(df):
        embeddings = []
        labels = []
        for idx, row in df.iterrows():
            text = str(row['text'])
            try:
                emb = llm.embed(text[:4000]) # Truncate slightly for context limits
                embeddings.append(emb)
                labels.append(row['label'])
            except Exception as e:
                print(f"Failed to embed row: {e}")
                
            if len(embeddings) % 100 == 0:
                print(f"Extracted {len(embeddings)} / {len(df)} embeddings...")
                
        return np.array(embeddings), np.array(labels)

    print("\nExtracting Train Embeddings...")
    X_train, y_train = extract_embeddings(df_train)
    
    print("\nExtracting Val Embeddings...")
    X_val, y_val = extract_embeddings(df_val)

    # 4. Train Logistic Regression
    print("\nTraining Logistic Regression Head...")
    clf = LogisticRegression(max_iter=1000, class_weight='balanced')
    clf.fit(X_train, y_train)

    # 5. Evaluate
    print("\nEvaluating on Validation Set:")
    y_pred = clf.predict(X_val)
    print(f"Accuracy: {accuracy_score(y_val, y_pred)*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["Legitimate", "Scam"]))

    # 6. Save Model
    os.makedirs("models/gguf", exist_ok=True)
    save_path = "models/gguf/gguf_classifier_head.joblib"
    joblib.dump(clf, save_path)
    print(f"\nSuccessfully saved classification head to {save_path}")

if __name__ == "__main__":
    main()
