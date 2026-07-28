"""Detect transcript embedding drift against a training-data baseline.

The detector intentionally stays simple:
- Build or load a baseline centroid from training transcript embeddings.
- Embed incoming transcripts.
- Compare centroid shift using cosine distance.
- Flag drift when the distance exceeds a threshold.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def transcript_from_record(record: dict[str, Any]) -> str:
    for field in ("raw_dialogue", "transcript", "input"):
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if "messages" in record:
        for message in record["messages"]:
            if message.get("role") == "user":
                return message.get("content", "").strip()
    raise ValueError(f"Could not find transcript text in record: {record.get('id', '<unknown>')}")


def load_transcripts(path: Path, limit: int | None = None) -> list[str]:
    records = load_jsonl(path)
    transcripts = [transcript_from_record(record) for record in records]
    return transcripts[:limit] if limit else transcripts


def load_embedder(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embed_texts(model, texts: list[str], batch_size: int) -> np.ndarray:
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    return np.asarray(embeddings, dtype=np.float32)


def centroid(embeddings: np.ndarray) -> np.ndarray:
    center = embeddings.mean(axis=0)
    norm = np.linalg.norm(center)
    if norm == 0:
        return center
    return center / norm


def cosine_distance(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    denominator = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    if denominator == 0:
        return 1.0
    similarity = float(np.dot(vector_a, vector_b) / denominator)
    return 1.0 - similarity


def save_baseline(path: Path, model_name: str, embeddings: np.ndarray, source_path: Path) -> dict[str, Any]:
    baseline = {
        "embedding_model": model_name,
        "source_path": str(source_path),
        "sample_count": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
        "centroid": centroid(embeddings).tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(baseline, file, indent=2)
    return baseline


def load_baseline(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def build_report(
    baseline: dict[str, Any],
    incoming_embeddings: np.ndarray,
    threshold: float,
    incoming_path: Path,
) -> dict[str, Any]:
    baseline_centroid = np.asarray(baseline["centroid"], dtype=np.float32)
    incoming_centroid = centroid(incoming_embeddings)
    distance = cosine_distance(baseline_centroid, incoming_centroid)
    per_sample_distances = [
        cosine_distance(baseline_centroid, embedding) for embedding in incoming_embeddings
    ]
    return {
        "status": "drift_detected" if distance >= threshold else "ok",
        "threshold": threshold,
        "centroid_cosine_distance": distance,
        "incoming_path": str(incoming_path),
        "incoming_sample_count": int(incoming_embeddings.shape[0]),
        "baseline_sample_count": baseline["sample_count"],
        "embedding_model": baseline["embedding_model"],
        "per_sample_distance_mean": float(np.mean(per_sample_distances)),
        "per_sample_distance_p95": float(np.percentile(per_sample_distances, 95)),
        "recommendation": (
            "Trigger review and consider retraining if sustained for multiple windows."
            if distance >= threshold
            else "No retraining action needed for this window."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-file", type=Path, default=Path("data/processed/phase1/calls_all.jsonl"))
    parser.add_argument("--incoming-file", type=Path, required=True)
    parser.add_argument("--baseline-file", type=Path, default=Path("artifacts/monitoring/embedding_baseline.json"))
    parser.add_argument("--output", type=Path, default=Path("outputs/monitoring/drift_report.json"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--threshold", type=float, default=0.12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--limit-training", type=int, default=None)
    parser.add_argument("--rebuild-baseline", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.rebuild_baseline or not args.baseline_file.exists():
        model = load_embedder(args.embedding_model)
        training_transcripts = load_transcripts(args.training_file, args.limit_training)
        training_embeddings = embed_texts(model, training_transcripts, args.batch_size)
        baseline = save_baseline(
            path=args.baseline_file,
            model_name=args.embedding_model,
            embeddings=training_embeddings,
            source_path=args.training_file,
        )
    else:
        baseline = load_baseline(args.baseline_file)
        model = load_embedder(baseline["embedding_model"])

    incoming_transcripts = load_transcripts(args.incoming_file)
    incoming_embeddings = embed_texts(model, incoming_transcripts, args.batch_size)
    report = build_report(
        baseline=baseline,
        incoming_embeddings=incoming_embeddings,
        threshold=args.threshold,
        incoming_path=args.incoming_file,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
