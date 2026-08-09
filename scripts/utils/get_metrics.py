import os
import mlflow
from dotenv import load_dotenv
import dagshub

load_dotenv()
dagshub.auth.add_app_token(os.getenv("MLFLOW_TRACKING_PASSWORD"))
os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{os.getenv('DAGSHUB_REPO_OWNER')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"

mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
experiment = mlflow.get_experiment_by_name("scam-detection-ablation")

if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    for _, run in runs.iterrows():
        print(f"\n--- Run: {run.get('tags.mlflow.runName', 'Unnamed')} ---")
        metrics = [k for k in run.keys() if k.startswith('metrics.')]
        for m in metrics:
            print(f"{m.replace('metrics.', '')}: {run[m]}")
else:
    print("Experiment not found!")
