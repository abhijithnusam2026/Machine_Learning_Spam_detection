import os
import mlflow
from dotenv import load_dotenv
import dagshub

load_dotenv()
dagshub.auth.add_app_token(os.getenv("MLFLOW_TRACKING_PASSWORD"))
os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{os.getenv('DAGSHUB_REPO_OWNER')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

client = mlflow.tracking.MlflowClient()

print("--- Registered Models ---")
for mv in client.search_registered_models():
    print(f"Model: {mv.name}")

print("\n--- Experiments ---")
experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
for exp in experiments:
    print(f"Exp ID: {exp.experiment_id}, Name: {exp.name}, Lifecycle: {exp.lifecycle_stage}")

print("\n--- Runs in scam-detection-ablation ---")
ablation_exp = client.get_experiment_by_name("scam-detection-ablation")
if ablation_exp:
    runs = client.search_runs(experiment_ids=[ablation_exp.experiment_id])
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "Unnamed")
        acc = run.data.metrics.get("eval_accuracy")
        print(f"Run ID: {run.info.run_id}, Name: {run_name}, Acc: {acc}")
        # Add metadata tag
        client.set_tag(run.info.run_id, "phase", "Phase_1_Ablation_Study")
        if acc and float(acc) > 0.988:
            client.set_tag(run.info.run_id, "model", "ModernBERT")
            client.set_tag(run.info.run_id, "status", "Champion")
        elif acc and float(acc) > 0.986:
            client.set_tag(run.info.run_id, "model", "DistilBERT")
            client.set_tag(run.info.run_id, "status", "Runner_Up")
