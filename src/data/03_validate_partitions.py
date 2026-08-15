import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv


REQUIRED_COLUMNS = {"text", "label", "source_domain", "source_dataset"}
EXPECTED_SPLITS = {
    "global_train": "global_train.csv",
    "global_val": "global_val.csv",
    "global_test": "global_test.csv",
    "ptq_calibration": "ptq_calibration.csv",
}
EXPECTED_LABELS = {0, 1}
STAGE_FILTERS = {
    "distilbert_baseline_train": lambda df: df[df["source_dataset"] == "kaggle_composite"],
    "modernbert_universal_train": lambda df: df[df["source_domain"] == "written_text"],
    "transcript_retraining_train": lambda df: df[df["source_domain"] == "spoken_asr"],
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_text_set(df):
    return set(df["text"].astype(str).str.strip())


def distribution(df, column):
    return {str(k): int(v) for k, v in df[column].value_counts(dropna=False).sort_index().items()}


def add_failure(failures, message):
    failures.append(message)
    print(f"[FAIL] {message}")


def add_warning(warnings, message):
    warnings.append(message)
    print(f"[WARN] {message}")


def get_commit_sha():
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return "unknown"


def validate_split_frame(name, path, df, failures, warnings):
    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        add_failure(failures, f"{name} is missing required columns: {sorted(missing_columns)}")
        return

    if df.empty:
        add_failure(failures, f"{name} is empty")
        return

    blank_text = df["text"].isna() | (df["text"].astype(str).str.strip() == "")
    if blank_text.any():
        add_failure(failures, f"{name} contains {int(blank_text.sum())} blank text rows")

    null_labels = df["label"].isna()
    if null_labels.any():
        add_failure(failures, f"{name} contains {int(null_labels.sum())} null labels")

    try:
        labels = set(df["label"].astype(int).unique().tolist())
    except Exception:
        add_failure(failures, f"{name} labels are not safely castable to integers")
        return

    if not labels.issubset(EXPECTED_LABELS):
        add_failure(failures, f"{name} contains labels outside {sorted(EXPECTED_LABELS)}: {sorted(labels)}")

    if name != "ptq_calibration" and labels != EXPECTED_LABELS:
        add_failure(failures, f"{name} must contain both labels 0 and 1; found {sorted(labels)}")

    duplicate_text_rows = int(df["text"].astype(str).duplicated().sum())
    if duplicate_text_rows:
        add_failure(failures, f"{name} contains {duplicate_text_rows} duplicate text rows")

    if df["source_domain"].isna().any():
        add_failure(failures, f"{name} contains null source_domain values")

    if df["source_dataset"].isna().any():
        add_failure(failures, f"{name} contains null source_dataset values")

    if path.stat().st_size == 0:
        add_failure(failures, f"{name} file is zero bytes")

    text_lengths = df["text"].astype(str).str.split().str.len()
    if int((text_lengths == 0).sum()):
        add_failure(failures, f"{name} contains tokenless text rows after stripping")

    if float(text_lengths.quantile(0.95)) > 10000:
        add_warning(warnings, f"{name} p95 word count is unusually high; verify raw parsing")


def validate_stage_subsets(train_df, val_df, failures):
    for stage_name, filter_fn in STAGE_FILTERS.items():
        stage_train = filter_fn(train_df)
        stage_val = filter_fn(val_df)
        if stage_train.empty:
            add_failure(failures, f"{stage_name} has zero training rows after its source filter")
        if stage_val.empty:
            add_failure(failures, f"{stage_name} has zero validation rows after its source filter")

        for split_name, split_df in [("train", stage_train), ("val", stage_val)]:
            if split_df.empty:
                continue
            labels = set(split_df["label"].astype(int).unique().tolist())
            if labels != EXPECTED_LABELS:
                add_failure(
                    failures,
                    f"{stage_name} {split_name} subset must contain both labels 0 and 1; found {sorted(labels)}",
                )


def validate_split_ratios(train_df, val_df, test_df, expected_test_frac, expected_val_frac, tolerance, warnings):
    total = len(train_df) + len(val_df) + len(test_df)
    if total == 0:
        return

    actual_test_frac = len(test_df) / total
    if abs(actual_test_frac - expected_test_frac) > tolerance:
        add_warning(
            warnings,
            f"global_test fraction is {actual_test_frac:.3f}, expected about {expected_test_frac:.3f}",
        )

    train_val_total = len(train_df) + len(val_df)
    if train_val_total == 0:
        return

    actual_val_frac = len(val_df) / train_val_total
    if abs(actual_val_frac - expected_val_frac) > tolerance:
        add_warning(
            warnings,
            f"global_val fraction within train/val is {actual_val_frac:.3f}, expected about {expected_val_frac:.3f}",
        )


def maybe_log_to_mlflow(report, report_path):
    load_dotenv()
    repo_owner = os.getenv("DAGSHUB_REPO_OWNER")
    repo_name = os.getenv("DAGSHUB_REPO_NAME")
    if not repo_owner or not repo_name:
        print("[INFO] DagsHub environment variables not set; skipped MLflow validation logging.")
        return

    try:
        import dagshub
        import mlflow

        username = os.getenv("MLFLOW_TRACKING_USERNAME")
        password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        if username and password:
            os.environ["DAGSHUB_USER"] = username
            os.environ["DAGSHUB_TOKEN"] = password
            dagshub.auth.add_app_token(password)

        dagshub.init(repo_name=repo_name, repo_owner=repo_owner, mlflow=True)
        mlflow.set_experiment("scam-detection/refactored_pipeline/03_validate_partitions")
        with mlflow.start_run(run_name="validate_processed_partitions"):
            mlflow.set_tag("project_stage", "refactored_pipeline")
            mlflow.set_tag("pipeline_stage", "03_validate_partitions")
            mlflow.set_tag("commit_sha", report["commit_sha"])
            mlflow.log_metric("validation_passed", int(report["passed"]))
            mlflow.log_metric("validation_failure_count", len(report["failures"]))
            mlflow.log_metric("validation_warning_count", len(report["warnings"]))
            for split_name, split_report in report["splits"].items():
                if isinstance(split_report, dict) and "rows" in split_report:
                    mlflow.log_metric(f"{split_name}_rows", split_report["rows"])
            mlflow.log_artifact(str(report_path), artifact_path="validation")
        print("[INFO] Logged partition validation report to DagsHub MLflow.")
    except Exception as exc:
        print(f"[WARN] MLflow validation logging skipped: {exc}")


def main():
    parser = argparse.ArgumentParser(
        description="Validate processed scam-detection data partitions before training."
    )
    parser.add_argument("--processed_dir", default="data/processed")
    parser.add_argument("--expected_calibration_rows", type=int, default=256)
    parser.add_argument("--expected_test_frac", type=float, default=0.20)
    parser.add_argument("--expected_val_frac", type=float, default=0.20)
    parser.add_argument("--ratio_tolerance", type=float, default=0.03)
    parser.add_argument("--skip_mlflow", action="store_true")
    parser.add_argument(
        "--log_failed",
        action="store_true",
        help="Also log failed validation runs to MLflow. By default, failed checks stay local to avoid noisy tracking runs.",
    )
    args = parser.parse_args()

    processed_dir = Path(args.processed_dir)
    failures = []
    warnings = []
    frames = {}
    split_reports = {}

    print("--- Validating Processed Data Partitions ---")
    if not processed_dir.exists():
        add_failure(failures, f"Processed directory does not exist: {processed_dir}")
    else:
        for split_name, filename in EXPECTED_SPLITS.items():
            path = processed_dir / filename
            if not path.exists():
                add_failure(failures, f"Missing required split file: {path}")
                continue
            try:
                df = pd.read_csv(path)
            except Exception as exc:
                add_failure(failures, f"Could not read {path}: {exc}")
                continue

            validate_split_frame(split_name, path, df, failures, warnings)
            split_reports[split_name] = {
                "path": str(path),
                "rows": int(len(df)),
                "sha256": sha256_file(path),
                "label_distribution": distribution(df, "label") if "label" in df.columns else {},
                "source_domain_distribution": distribution(df, "source_domain")
                if "source_domain" in df.columns
                else {},
                "source_dataset_distribution": distribution(df, "source_dataset")
                if "source_dataset" in df.columns
                else {},
            }
            if REQUIRED_COLUMNS.issubset(df.columns):
                frames[split_name] = df

    required_loaded = set(EXPECTED_SPLITS) <= set(frames)
    if required_loaded:
        train_texts = canonical_text_set(frames["global_train"])
        val_texts = canonical_text_set(frames["global_val"])
        test_texts = canonical_text_set(frames["global_test"])
        calibration_texts = canonical_text_set(frames["ptq_calibration"])

        overlaps = {
            "train_val": len(train_texts & val_texts),
            "train_test": len(train_texts & test_texts),
            "val_test": len(val_texts & test_texts),
            "calibration_val": len(calibration_texts & val_texts),
            "calibration_test": len(calibration_texts & test_texts),
        }
        for overlap_name, overlap_count in overlaps.items():
            if overlap_count:
                add_failure(failures, f"{overlap_name} text overlap contains {overlap_count} rows")

        missing_from_train = calibration_texts - train_texts
        if missing_from_train:
            add_failure(
                failures,
                f"ptq_calibration contains {len(missing_from_train)} rows not present in global_train",
            )

        if len(frames["ptq_calibration"]) != args.expected_calibration_rows:
            add_failure(
                failures,
                f"ptq_calibration has {len(frames['ptq_calibration'])} rows; expected {args.expected_calibration_rows}",
            )

        validate_split_ratios(
            frames["global_train"],
            frames["global_val"],
            frames["global_test"],
            args.expected_test_frac,
            args.expected_val_frac,
            args.ratio_tolerance,
            warnings,
        )
        validate_stage_subsets(frames["global_train"], frames["global_val"], failures)
        split_reports["overlaps"] = overlaps

    report = {
        "passed": not failures,
        "commit_sha": get_commit_sha(),
        "processed_dir": str(processed_dir),
        "required_columns": sorted(REQUIRED_COLUMNS),
        "expected_labels": sorted(EXPECTED_LABELS),
        "failures": failures,
        "warnings": warnings,
        "splits": split_reports,
    }

    report_path = processed_dir / "partition_validation_report.json"
    if processed_dir.exists():
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"[INFO] Wrote validation report: {report_path}")

    should_log = not args.skip_mlflow and processed_dir.exists() and (report["passed"] or args.log_failed)
    if should_log:
        maybe_log_to_mlflow(report, report_path)
    elif not args.skip_mlflow and processed_dir.exists() and not report["passed"]:
        print("[INFO] Skipped MLflow logging for failed validation. Re-run with --log_failed if you want to track it.")

    if failures:
        print(f"\nPartition validation failed with {len(failures)} issue(s).")
        sys.exit(1)

    print("\nPartition validation passed.")
    if warnings:
        print(f"Completed with {len(warnings)} warning(s). Review the JSON report before long training runs.")


if __name__ == "__main__":
    main()
