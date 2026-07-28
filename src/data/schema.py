"""Shared schema constants for call-center datasets."""

INTENT_LABELS = [
    "billing_issue",
    "technical_support",
    "cancellation_request",
    "complaint",
    "general_inquiry",
]

TASK_SUMMARIZATION = "summarization"
TASK_INTENT = "intent_classification"

SYSTEM_PROMPT = (
    "You are a call center intelligence assistant. Follow the user's instruction "
    "using only the provided transcript."
)


def label_display_name(label: str) -> str:
    """Return a human-readable version of a normalized intent label."""
    return label.replace("_", " ")
