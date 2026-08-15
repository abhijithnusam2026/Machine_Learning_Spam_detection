import os
import shutil
import subprocess
import sys
from pathlib import Path

import dagshub
import mlflow
import pandas as pd
from dotenv import load_dotenv


PROCESSED_DATA_EXPERIMENT = "scam-detection/refactored_pipeline/02_build_datasets"


def _configure_dagshub():
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")

    if username and password:
        os.environ["DAGSHUB_USER"] = username
        os.environ["DAGSHUB_TOKEN"] = password
        dagshub.auth.add_app_token(password)

    if not repo_owner or not repo_name:
        raise RuntimeError(
            "DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set to fetch missing data artifacts."
        )

    dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
    return repo_owner, repo_name


def _download_processed_from_mlflow(local_paths):
    _configure_dagshub()
    experiment = mlflow.get_experiment_by_name(PROCESSED_DATA_EXPERIMENT)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment not found: {PROCESSED_DATA_EXPERIMENT}")

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=10,
    )
    if runs.empty:
        raise RuntimeError(f"No finished runs found in MLflow experiment: {PROCESSED_DATA_EXPERIMENT}")

    last_error = None
    for _, run in runs.iterrows():
        run_id = run["run_id"]
        try:
            for local_path in local_paths:
                local_path = Path(local_path)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                artifact_path = f"processed/{local_path.name}"
                print(f"Fetching {artifact_path} from MLflow run {run_id}...")
                downloaded_path = Path(mlflow.artifacts.download_artifacts(
                    run_id=run_id,
                    artifact_path=artifact_path,
                    dst_path=str(local_path.parent),
                ))
                if downloaded_path.is_dir():
                    downloaded_path = downloaded_path / local_path.name
                if downloaded_path.resolve() != local_path.resolve():
                    shutil.copy2(downloaded_path, local_path)
            return
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Could not fetch processed artifacts from run {run_id}: {exc}")

    raise RuntimeError(f"Failed to fetch processed artifacts from MLflow. Last error: {last_error}")


def _build_processed_locally():
    print("Falling back to raw DagsHub download + local processed-data build.")
    subprocess.run(
        [sys.executable, "src/data/00_download_raw_data.py", "--skip_mlflow"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "src/data/02_build_datasets.py", "--skip_mlflow"],
        check=True,
    )
    subprocess.run(
        [sys.executable, "src/data/03_validate_partitions.py", "--skip_mlflow"],
        check=True,
    )


def ensure_processed_data(local_paths, allow_rebuild=True):
    paths = [Path(path) for path in local_paths]
    missing = [path for path in paths if not path.exists()]
    if not missing:
        return

    print("Missing processed data files:")
    for path in missing:
        print(f"  - {path}")

    try:
        _download_processed_from_mlflow(missing)
    except Exception as exc:
        if not allow_rebuild:
            raise
        print(f"[WARN] Processed MLflow artifact fetch failed: {exc}")
        _build_processed_locally()

    still_missing = [path for path in paths if not path.exists()]
    if still_missing:
        missing_list = "\n".join(f"  - {path}" for path in still_missing)
        raise FileNotFoundError(f"Required processed data files are still missing:\n{missing_list}")

    print("Processed data is available locally.")


def ensure_audio_holdout(local_audio_dir="data/large_audio_test"):
    local_audio_dir = Path(local_audio_dir)
    manifest_path = local_audio_dir / "manifest.csv"
    wav_files = list(local_audio_dir.glob("*.wav")) if local_audio_dir.exists() else []
    if manifest_path.exists() and wav_files:
        return

    repo_owner, repo_name = _configure_dagshub()
    s3_client = dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")
    local_audio_dir.mkdir(parents=True, exist_ok=True)

    print("Fetching audio benchmark holdout from DagsHub...")
    remote_prefixes = [
        "data/large_audio_test",
        "artifacts/feature/phase-3.5-benchmark/large_audio_test",
    ]
    manifest_key = None
    last_error = None
    for prefix in remote_prefixes:
        candidate_key = f"{prefix}/manifest.csv"
        try:
            s3_client.download_file(repo_name, candidate_key, str(manifest_path))
            manifest_key = candidate_key
            break
        except Exception as exc:
            last_error = exc

    if manifest_key is None:
        raise FileNotFoundError(
            "Audio benchmark manifest was not found in DagsHub at any known path. "
            f"Tried: {[f'{prefix}/manifest.csv' for prefix in remote_prefixes]}. "
            f"Last error: {last_error}"
        )

    remote_prefix = manifest_key.rsplit("/", 1)[0]
    manifest_df = pd.read_csv(manifest_path)

    failed_downloads = []
    for _, row in manifest_df.iterrows():
        audio_filename = os.path.basename(row["file"])
        remote_audio_path = f"{remote_prefix}/{audio_filename}"
        local_audio_path = local_audio_dir / audio_filename
        if local_audio_path.exists():
            continue
        try:
            print(f"Downloading {audio_filename}...")
            s3_client.download_file(repo_name, remote_audio_path, str(local_audio_path))
        except Exception as exc:
            print(f"Failed to fetch {audio_filename}: {exc}")
            failed_downloads.append(audio_filename)

    if failed_downloads:
        raise RuntimeError(f"Audio files failed to download: {failed_downloads}")

    print("Audio benchmark holdout is available locally.")
