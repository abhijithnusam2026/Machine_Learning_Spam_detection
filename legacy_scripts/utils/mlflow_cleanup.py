import os
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv
import dagshub

REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCH_STAGE_MAP = {
    "feature/phase-2-audio-asr": "feature/phase-2-audio-asr",
    "feature/phase-3-serving-quantization": "feature/phase-3-serving-quantization",
    "feature/phase-1.5-ultimate-dataset": "model-modernbert-universal",
    "model-long-context": "model-modernbert-universal",
    "model-distilbert": "model-distilbert",
    "main": "main",
}


def detect_branch(default="main"):
    env_branch = os.getenv("DAGSHUB_BRANCH") or os.getenv("GIT_BRANCH")
    if env_branch:
        return env_branch.strip()
    result = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    branch = result.stdout.strip()
    return branch or default


def stage_for_branch(branch):
    return BRANCH_STAGE_MAP.get(branch, branch.replace("/", "-") or "main")

load_dotenv()
dagshub.auth.add_app_token(os.getenv("MLFLOW_TRACKING_PASSWORD"))
os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{os.getenv('DAGSHUB_REPO_OWNER')}/{os.getenv('DAGSHUB_REPO_NAME')}.mlflow"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
stage = stage_for_branch(detect_branch())

client = mlflow.tracking.MlflowClient()

print("--- Registered Models ---")
for mv in client.search_registered_models():
    print(f"Model: {mv.name}")

print("\n--- Experiments ---")
experiments = client.search_experiments(view_type=mlflow.entities.ViewType.ALL)
for exp in experiments:
    print(f"Exp ID: {exp.experiment_id}, Name: {exp.name}, Lifecycle: {exp.lifecycle_stage}")

experiment_name = f"scam-detection/{stage}/train"
print(f"\n--- Runs in {experiment_name} ---")
experiment = client.get_experiment_by_name(experiment_name)
if experiment:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    for run in runs:
        run_name = run.data.tags.get("mlflow.runName", "Unnamed")
        acc = run.data.metrics.get("eval_accuracy")
        print(f"Run ID: {run.info.run_id}, Name: {run_name}, Acc: {acc}")
        # Add metadata tag
        client.set_tag(run.info.run_id, "phase", "Phase_2_Audio_Asr")
        if acc and float(acc) > 0.988:
            client.set_tag(run.info.run_id, "model", "ModernBERT")
            client.set_tag(run.info.run_id, "status", "Champion")
        elif acc and float(acc) > 0.986:
            client.set_tag(run.info.run_id, "model", "DistilBERT")
            client.set_tag(run.info.run_id, "status", "Runner_Up")
else:
    print("Experiment not found!")
