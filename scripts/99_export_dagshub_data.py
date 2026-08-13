import os
import json
import mlflow
from dotenv import load_dotenv

def export_mlflow_data(output_dir="dagshub_export"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Authenticate with DagsHub
    load_dotenv()
    owner = os.getenv("DAGSHUB_REPO_OWNER")
    name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if not all([owner, name, token]):
        print("Error: Missing DagsHub credentials in .env file.")
        return

    import dagshub
    dagshub.auth.add_app_token(token)
    os.environ["MLFLOW_TRACKING_USERNAME"] = owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(f"https://dagshub.com/{owner}/{name}.mlflow")

    client = mlflow.tracking.MlflowClient()
    
    # 2. Fetch all experiments
    experiments = client.search_experiments()
    export_data = {}

    print(f"Found {len(experiments)} MLflow experiments.")
    
    for exp in experiments:
        print(f"Exporting Experiment: {exp.name} (ID: {exp.experiment_id})")
        exp_data = {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "lifecycle_stage": exp.lifecycle_stage,
            "runs": []
        }
        
        # 3. Fetch all runs for this experiment
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        
        for run in runs:
            run_data = {
                "run_id": run.info.run_id,
                "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")}
            }
            exp_data["runs"].append(run_data)
            
        export_data[exp.name] = exp_data

    # 4. Save to JSON
    output_file = os.path.join(output_dir, "mlflow_experiments_export.json")
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=4)
        
    print(f"\nSuccessfully exported all MLflow data to {output_file}!")

if __name__ == "__main__":
    export_mlflow_data()
