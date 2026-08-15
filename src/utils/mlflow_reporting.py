import json
import os
import tempfile

import mlflow
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix


def _safe_name(value):
    return str(value).replace("/", "_").replace(" ", "_")


def log_dataframe_artifact(df, name, artifact_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, name)
        df.to_csv(path, index=False)
        mlflow.log_artifact(path, artifact_path=artifact_path)


def log_json_artifact(payload, name, artifact_path):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        mlflow.log_artifact(path, artifact_path=artifact_path)


def describe_split(df, split_name):
    summary = {
        "split": split_name,
        "rows": int(len(df)),
        "labels": {},
        "source_domains": {},
        "source_datasets": {},
    }
    if "label" in df.columns:
        summary["labels"] = {str(k): int(v) for k, v in df["label"].value_counts(dropna=False).sort_index().items()}
    if "source_domain" in df.columns:
        summary["source_domains"] = {str(k): int(v) for k, v in df["source_domain"].value_counts(dropna=False).sort_index().items()}
    if "source_dataset" in df.columns:
        summary["source_datasets"] = {str(k): int(v) for k, v in df["source_dataset"].value_counts(dropna=False).sort_index().items()}
    if "text" in df.columns:
        lengths = df["text"].astype(str).str.split().str.len()
        summary["word_count"] = {
            "mean": float(lengths.mean()) if len(lengths) else 0.0,
            "median": float(lengths.median()) if len(lengths) else 0.0,
            "p95": float(lengths.quantile(0.95)) if len(lengths) else 0.0,
            "max": int(lengths.max()) if len(lengths) else 0,
        }
    return summary


def log_split_profile(split_frames, artifact_path="dataset_profile"):
    rows = []
    payload = {}
    for split_name, df in split_frames.items():
        summary = describe_split(df, split_name)
        payload[split_name] = summary
        mlflow.log_metric(f"{split_name}_rows", summary["rows"])
        for label, count in summary["labels"].items():
            mlflow.log_metric(f"{split_name}_label_{_safe_name(label)}_rows", count)
        for domain, count in summary["source_domains"].items():
            mlflow.log_metric(f"{split_name}_domain_{_safe_name(domain)}_rows", count)
        for source, count in summary["source_datasets"].items():
            mlflow.log_metric(f"{split_name}_source_{_safe_name(source)}_rows", count)
        rows.append(
            {
                "split": split_name,
                "rows": summary["rows"],
                "label_distribution": json.dumps(summary["labels"], sort_keys=True),
                "source_domain_distribution": json.dumps(summary["source_domains"], sort_keys=True),
                "source_dataset_distribution": json.dumps(summary["source_datasets"], sort_keys=True),
                "word_count_mean": summary.get("word_count", {}).get("mean", 0.0),
                "word_count_median": summary.get("word_count", {}).get("median", 0.0),
                "word_count_p95": summary.get("word_count", {}).get("p95", 0.0),
                "word_count_max": summary.get("word_count", {}).get("max", 0),
            }
        )
    log_json_artifact(payload, "split_profile.json", artifact_path)
    log_dataframe_artifact(pd.DataFrame(rows), "split_profile.csv", artifact_path)


def log_classification_artifacts(y_true, y_pred, artifact_path="evaluation", prefix="classification"):
    labels = [0, 1]
    target_names = ["Legitimate", "Scam"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=target_names, columns=target_names)
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=target_names,
        zero_division=0,
    )

    log_dataframe_artifact(cm_df.reset_index().rename(columns={"index": "actual"}), f"{prefix}_confusion_matrix.csv", artifact_path)
    log_dataframe_artifact(pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "class"}), f"{prefix}_classification_report.csv", artifact_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        report_path = os.path.join(tmpdir, f"{prefix}_classification_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        mlflow.log_artifact(report_path, artifact_path=artifact_path)

        try:
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(5, 4))
            image = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(len(target_names)), target_names)
            ax.set_yticks(range(len(target_names)), target_names)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="black")
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            fig.tight_layout()
            plot_path = os.path.join(tmpdir, f"{prefix}_confusion_matrix.png")
            fig.savefig(plot_path, dpi=160)
            plt.close(fig)
            mlflow.log_artifact(plot_path, artifact_path=artifact_path)
        except Exception as exc:
            fallback_path = os.path.join(tmpdir, f"{prefix}_plot_skipped.txt")
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(f"Confusion matrix plot was skipped: {exc}\n")
            mlflow.log_artifact(fallback_path, artifact_path=artifact_path)


def log_training_history(log_history, artifact_path="training", prefix="training"):
    if not log_history:
        return
    history_df = pd.DataFrame(log_history)
    log_dataframe_artifact(history_df, f"{prefix}_trainer_log_history.csv", artifact_path)

    metric_cols = [
        col
        for col in ["loss", "eval_loss", "eval_accuracy", "eval_f1", "eval_precision", "eval_recall", "eval_pr_auc"]
        if col in history_df.columns
    ]
    if not metric_cols:
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            import matplotlib.pyplot as plt

            x_col = "epoch" if "epoch" in history_df.columns else "step"
            for metric in metric_cols:
                metric_df = history_df[[x_col, metric]].dropna()
                if metric_df.empty:
                    continue
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(metric_df[x_col], metric_df[metric], marker="o")
                ax.set_xlabel(x_col)
                ax.set_ylabel(metric)
                ax.set_title(metric.replace("_", " ").title())
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                plot_path = os.path.join(tmpdir, f"{prefix}_{metric}.png")
                fig.savefig(plot_path, dpi=160)
                plt.close(fig)
                mlflow.log_artifact(plot_path, artifact_path=artifact_path)
        except Exception as exc:
            fallback_path = os.path.join(tmpdir, f"{prefix}_training_plots_skipped.txt")
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(f"Training plots were skipped: {exc}\n")
            mlflow.log_artifact(fallback_path, artifact_path=artifact_path)


def log_benchmark_plots(df, x_col, y_cols, artifact_path="benchmarks", prefix="benchmark"):
    if df.empty:
        return
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            import matplotlib.pyplot as plt

            for metric in y_cols:
                if metric not in df.columns:
                    continue
                plot_df = df[[x_col, metric]].dropna()
                if plot_df.empty:
                    continue
                fig, ax = plt.subplots(figsize=(7, 4))
                ax.bar(plot_df[x_col].astype(str), plot_df[metric])
                ax.set_xlabel(x_col.replace("_", " ").title())
                ax.set_ylabel(metric.replace("_", " ").title())
                ax.set_title(metric.replace("_", " ").title())
                ax.tick_params(axis="x", rotation=30)
                fig.tight_layout()
                plot_path = os.path.join(tmpdir, f"{prefix}_{metric}.png")
                fig.savefig(plot_path, dpi=160)
                plt.close(fig)
                mlflow.log_artifact(plot_path, artifact_path=artifact_path)
        except Exception as exc:
            fallback_path = os.path.join(tmpdir, f"{prefix}_benchmark_plots_skipped.txt")
            with open(fallback_path, "w", encoding="utf-8") as f:
                f.write(f"Benchmark plots were skipped: {exc}\n")
            mlflow.log_artifact(fallback_path, artifact_path=artifact_path)
