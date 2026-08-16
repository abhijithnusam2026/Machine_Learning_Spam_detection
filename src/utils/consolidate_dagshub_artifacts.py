"""
Consolidate model artifacts that were created during earlier branch/stage runs
into the canonical refactored-pipeline artifact prefix.

This script does not train anything. It copies existing DagsHub S3 objects from
legacy locations into:

    artifacts/refactored_pipeline/06_ptq_modernbert/

and writes an artifact_manifest.json so downstream scripts have one stable
source of truth after a server disconnect.
"""

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import dagshub
from dotenv import load_dotenv


CANONICAL_PREFIX = "artifacts/refactored_pipeline/06_ptq_modernbert"

ARTIFACTS = {
    "classifier_f16_gguf": {
        "canonical": f"{CANONICAL_PREFIX}/gguf/classifier_f16.gguf",
        "candidates": [
            f"{CANONICAL_PREFIX}/gguf/classifier_f16.gguf",
            "artifacts/06_ptq_modernbert/gguf/classifier_f16.gguf",
            "models/gguf_classifier/classifier_f16.gguf",
            "artifacts/feature/phase-2-audio-asr/gguf/classifier_f16.gguf",
        ],
    },
    "classifier_q8_0_gguf": {
        "canonical": f"{CANONICAL_PREFIX}/gguf/classifier_q8_0.gguf",
        "candidates": [
            f"{CANONICAL_PREFIX}/gguf/classifier_q8_0.gguf",
            "artifacts/06_ptq_modernbert/gguf/classifier_q8_0.gguf",
            "models/gguf_classifier/classifier_q8_0.gguf",
            "artifacts/feature/phase-2-audio-asr/gguf/classifier_q8_0.gguf",
        ],
    },
    "classifier_q4_k_m_gguf": {
        "canonical": f"{CANONICAL_PREFIX}/gguf/classifier_q4_k_m.gguf",
        "candidates": [
            f"{CANONICAL_PREFIX}/gguf/classifier_q4_k_m.gguf",
            "artifacts/06_ptq_modernbert/gguf/classifier_q4_k_m.gguf",
            "models/gguf_classifier/classifier_q4_k_m.gguf",
            "artifacts/feature/phase-2-audio-asr/gguf/classifier_q4_k_m.gguf",
        ],
    },
    "gguf_classifier_head": {
        "canonical": f"{CANONICAL_PREFIX}/gguf/gguf_classifier_head.joblib",
        "candidates": [
            f"{CANONICAL_PREFIX}/gguf/gguf_classifier_head.joblib",
            "artifacts/06_ptq_modernbert/gguf/gguf_classifier_head.joblib",
            "models/gguf/gguf_classifier_head.joblib",
            "artifacts/feature/phase-2-audio-asr/gguf/gguf_classifier_head.joblib",
            "artifacts/feature/phase-3.5-benchmark/gguf_classifier_head.joblib",
            "artifacts/feature/phase-3.5-benchmark/gguf/gguf_classifier_head.joblib",
        ],
        "local_fallback": "models/gguf/gguf_classifier_head.joblib",
    },
    "whisper_f16_ggml": {
        "canonical": f"{CANONICAL_PREFIX}/ggml/whisper_f16.bin",
        "candidates": [
            f"{CANONICAL_PREFIX}/ggml/whisper_f16.bin",
            "artifacts/06_ptq_modernbert/ggml/whisper_f16.bin",
            "models/ggml_whisper/whisper_f16.bin",
            "artifacts/feature/phase-2-audio-asr/ggml/whisper_f16.bin",
        ],
    },
    "whisper_q8_0_ggml": {
        "canonical": f"{CANONICAL_PREFIX}/ggml/whisper_q8_0.bin",
        "candidates": [
            f"{CANONICAL_PREFIX}/ggml/whisper_q8_0.bin",
            "artifacts/06_ptq_modernbert/ggml/whisper_q8_0.bin",
            "models/ggml_whisper/whisper_q8_0.bin",
            "artifacts/feature/phase-2-audio-asr/ggml/whisper_q8_0.bin",
        ],
    },
    "whisper_q5_1_ggml": {
        "canonical": f"{CANONICAL_PREFIX}/ggml/whisper_q5_1.bin",
        "candidates": [
            f"{CANONICAL_PREFIX}/ggml/whisper_q5_1.bin",
            "artifacts/06_ptq_modernbert/ggml/whisper_q5_1.bin",
            "models/ggml_whisper/whisper_q5_1.bin",
            "artifacts/feature/phase-2-audio-asr/ggml/whisper_q5_1.bin",
        ],
    },
}


def configure_dagshub():
    load_dotenv(".env")
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    token = os.getenv("MLFLOW_TRACKING_PASSWORD") or os.getenv("DAGSHUB_TOKEN")
    if not repo_owner or not repo_name:
        raise RuntimeError("DAGSHUB_REPO_OWNER and DAGSHUB_REPO_NAME must be set.")
    if token:
        dagshub.auth.add_app_token(token)
    return repo_owner, repo_name, dagshub.get_repo_bucket_client(f"{repo_owner}/{repo_name}")


def head_object(s3_client, bucket, key):
    try:
        return s3_client.head_object(Bucket=bucket, Key=key)
    except Exception:
        return None


def copy_remote_artifact(s3_client, bucket, source_key, target_key, dry_run=False):
    if source_key == target_key:
        return
    if dry_run:
        print(f"[DRY RUN] copy {source_key} -> {target_key}")
        return
    try:
        s3_client.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": source_key},
            Key=target_key,
        )
    except Exception as exc:
        print(f"[WARN] server-side copy failed for {source_key}: {exc}")
        print("[INFO] Falling back to download-then-upload.")
        with tempfile.TemporaryDirectory() as tmpdir:
            local_tmp = Path(tmpdir) / Path(source_key).name
            s3_client.download_file(bucket, source_key, str(local_tmp))
            s3_client.upload_file(str(local_tmp), bucket, target_key)


def upload_local_artifact(s3_client, bucket, local_path, target_key, dry_run=False):
    if dry_run:
        print(f"[DRY RUN] upload {local_path} -> {target_key}")
        return
    s3_client.upload_file(str(local_path), bucket, target_key)


def consolidate(dry_run=False):
    _, repo_name, s3_client = configure_dagshub()
    manifest = {
        "canonical_prefix": CANONICAL_PREFIX,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "artifacts": {},
    }

    for artifact_name, spec in ARTIFACTS.items():
        canonical_key = spec["canonical"]
        source_key = None
        source_head = None

        for candidate_key in spec["candidates"]:
            candidate_head = head_object(s3_client, repo_name, candidate_key)
            if candidate_head is not None:
                source_key = candidate_key
                source_head = candidate_head
                break

        status = "missing"
        local_fallback = spec.get("local_fallback")
        if source_key is not None:
            copy_remote_artifact(s3_client, repo_name, source_key, canonical_key, dry_run=dry_run)
            status = "available"
            print(f"[OK] {artifact_name}: {source_key} -> {canonical_key}")
        elif local_fallback and Path(local_fallback).exists():
            upload_local_artifact(s3_client, repo_name, Path(local_fallback), canonical_key, dry_run=dry_run)
            source_key = local_fallback
            source_head = {"ContentLength": Path(local_fallback).stat().st_size}
            status = "uploaded_from_local"
            print(f"[OK] {artifact_name}: local {local_fallback} -> {canonical_key}")
        else:
            print(f"[MISSING] {artifact_name}: no remote/local source found")

        manifest["artifacts"][artifact_name] = {
            "status": status,
            "canonical_uri": f"s3://{repo_name}/{canonical_key}",
            "canonical_key": canonical_key,
            "source": source_key,
            "size_bytes": source_head.get("ContentLength") if source_head else None,
        }

    manifest_path = Path("artifacts/refactored_pipeline/06_ptq_modernbert/artifact_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    upload_local_artifact(
        s3_client,
        repo_name,
        manifest_path,
        f"{CANONICAL_PREFIX}/artifact_manifest.json",
        dry_run=dry_run,
    )
    print(f"[OK] wrote {manifest_path}")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    consolidate(dry_run=args.dry_run)
