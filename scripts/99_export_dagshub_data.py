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
    export_data = {"experiments": {}, "registered_models": []}
    
    # 2. Fetch all Registered Models
    print("Fetching Registered Models...")
    try:
        models = client.search_registered_models()
        for model in models:
            model_info = {
                "name": model.name,
                "creation_timestamp": model.creation_timestamp,
                "description": model.description,
                "versions": []
            }
            # Fetch versions for this model
            versions = client.search_model_versions(f"name='{model.name}'")
            for v in versions:
                model_info["versions"].append({
                    "version": v.version,
                    "current_stage": v.current_stage,
                    "status": v.status,
                    "run_id": v.run_id,
                    "source": v.source
                })
            export_data["registered_models"].append(model_info)
    except Exception as e:
        print(f"Warning: Could not fetch registered models: {e}")

    # 3. Fetch all experiments
    experiments = client.search_experiments()
    print(f"Found {len(experiments)} MLflow experiments.")
    
    for exp in experiments:
        print(f"Exporting Experiment: {exp.name}")
        exp_data = {
            "experiment_id": exp.experiment_id,
            "name": exp.name,
            "artifact_location": exp.artifact_location,
            "runs": []
        }
        
        runs = client.search_runs(experiment_ids=[exp.experiment_id])
        
        for run in runs:
            run_id = run.info.run_id
            run_data = {
                "run_id": run_id,
                "run_name": run.data.tags.get("mlflow.runName", "unnamed"),
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "params": run.data.params,
                "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
                "metrics_history": {},
                "artifacts": []
            }
            
            # Extract FULL metric history (e.g. loss over every epoch) instead of just the final value
            for metric_key in run.data.metrics.keys():
                history = client.get_metric_history(run_id, metric_key)
                run_data["metrics_history"][metric_key] = [
                    {"step": m.step, "value": m.value, "timestamp": m.timestamp} for m in history
                ]
            
            # Extract list of all output artifacts (e.g. weights, confusion matrices, dataset splits)
            try:
                artifacts = client.list_artifacts(run_id)
                for artifact in artifacts:
                    run_data["artifacts"].append({
                        "path": artifact.path,
                        "is_dir": artifact.is_dir,
                        "file_size": artifact.file_size
                    })
            except Exception:
                pass
                
            exp_data["runs"].append(run_data)
            
        export_data["experiments"][exp.name] = exp_data

    # 4. Save to JSON
    output_file = os.path.join(output_dir, "mlflow_exhaustive_export.json")
    with open(output_file, "w") as f:
        json.dump(export_data, f, indent=4)
        
    print(f"\nSuccessfully exported ALL exhaustive MLflow data to {output_file}!")

if __name__ == "__main__":
    export_mlflow_data()
