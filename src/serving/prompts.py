"""Prompt builders for serving-time call analysis tasks."""

from __future__ import annotations

from src.data.schema import INTENT_LABELS, SYSTEM_PROMPT


def summarize_messages(transcript: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Summarize the customer support call. Include the customer issue, "
                "important context, the agent action, and the agreed next step.\n\n"
                f"Transcript:\n{transcript}"
            ),
        },
    ]


def classify_intent_messages(transcript: str) -> list[dict[str, str]]:
    labels = ", ".join(INTENT_LABELS)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Classify the primary intent of the customer support call.\n"
                f"Return exactly one label from this list: {labels}.\n\n"
                f"Transcript:\n{transcript}"
            ),
        },
    ]
