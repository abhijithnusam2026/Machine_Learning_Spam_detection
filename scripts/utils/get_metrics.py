import os
import subprocess
from pathlib import Path

import mlflow
from dotenv import load_dotenv
import dagshub

REPO_ROOT = Path(__file__).resolve().parents[2]
BRANCH_STAGE_MAP = {
    "feature/phase-1.5-ultimate-dataset": "model-modernbert-universal",
    "model-long-context": "model-modernbert-universal",
    "feature/phase-2-audio-asr": "feature/phase-2-audio-asr",
    "feature/phase-3-serving-quantization": "feature/phase-3-serving-quantization",
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
experiment = mlflow.get_experiment_by_name(f"scam-detection/{stage_for_branch(detect_branch())}/train")

if experiment:
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    for _, run in runs.iterrows():
        print(f"\n--- Run: {run.get('tags.mlflow.runName', 'Unnamed')} ---")
        metrics = [k for k in run.keys() if k.startswith('metrics.')]
        for m in metrics:
            print(f"{m.replace('metrics.', '')}: {run[m]}")
else:
    print("Experiment not found!")
