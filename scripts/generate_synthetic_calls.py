"""Generate synthetic call-center transcripts as JSONL.

The script supports two modes:
- template: deterministic local generation for pipeline smoke tests.
- openai_compatible: calls any OpenAI-compatible chat-completions endpoint.

Expected output schema per JSONL row:
{
  "id": "call_000001",
  "intent": "billing_issue",
  "raw_dialogue": "Agent: ...\nCustomer: ...",
  "summary": "...",
  "metadata": {"source": "template", ...}
}
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.schema import INTENT_LABELS


SCENARIOS = {
    "billing_issue": [
        "unexpected late fee",
        "charged twice for the same month",
        "promotional discount missing",
        "refund not posted",
        "invoice higher than expected",
    ],
    "technical_support": [
        "internet modem keeps restarting",
        "mobile app crashes on login",
        "payment page times out",
        "account verification code never arrives",
        "device cannot connect after an update",
    ],
    "cancellation_request": [
        "moving to a different provider",
        "subscription is too expensive",
        "service is no longer needed",
        "trial period is ending",
        "customer wants to pause but cancellation is the only option",
    ],
    "complaint": [
        "previous agent promised a callback",
        "technician missed the appointment window",
        "support case has been open too long",
        "customer was transferred multiple times",
        "service outage caused business disruption",
    ],
    "general_inquiry": [
        "asks about plan features",
        "checks store hours",
        "asks whether international roaming is included",
        "wants to know installation availability",
        "asks how to update contact details",
    ],
}

AGENT_NAMES = ["Maya", "Jordan", "Priya", "Alex", "Sam", "Taylor"]
CUSTOMER_NAMES = ["Chris", "Avery", "Morgan", "Riley", "Jamie", "Casey"]


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float


def build_template_record(call_id: int, intent: str, rng: random.Random) -> dict[str, Any]:
    scenario = rng.choice(SCENARIOS[intent])
    agent = rng.choice(AGENT_NAMES)
    customer = rng.choice(CUSTOMER_NAMES)
    account_id = rng.randint(100000, 999999)
    resolution = {
        "billing_issue": "The agent reviewed the bill, opened an adjustment request, and gave the customer a 3-5 business day refund timeline.",
        "technical_support": "The agent walked through troubleshooting steps, reset the service from their side, and scheduled a follow-up if the issue returns.",
        "cancellation_request": "The agent confirmed the cancellation request, explained final billing, and documented the effective cancellation date.",
        "complaint": "The agent apologized, escalated the case to a supervisor queue, and committed to a status update within 24 hours.",
        "general_inquiry": "The agent answered the policy question, confirmed the customer's account details, and shared the next step.",
    }[intent]
    summary = (
        f"Customer {customer} contacted support about {scenario}. {resolution}"
    )
    raw_dialogue = "\n".join(
        [
            f"Agent ({agent}): Thank you for calling Acme Support. How can I help today?",
            f"Customer ({customer}): I'm calling because I have an issue: {scenario}.",
            f"Agent ({agent}): I can help with that. May I verify the account ending in {str(account_id)[-4:]}?",
            f"Customer ({customer}): Yes, that is correct. I need this resolved because it has been frustrating.",
            f"Agent ({agent}): I reviewed the account notes and see the same concern. Let me explain what I can do.",
            f"Customer ({customer}): Okay, I appreciate that. What happens next?",
            f"Agent ({agent}): {resolution}",
            f"Customer ({customer}): Thanks. Please make sure this is documented on my account.",
            f"Agent ({agent}): I added the notes and your reference number is REF-{account_id}.",
        ]
    )
    return {
        "id": f"call_{call_id:06d}",
        "intent": intent,
        "raw_dialogue": raw_dialogue,
        "summary": summary,
        "metadata": {
            "source": "template",
            "scenario": scenario,
        },
    }


def build_generation_prompt(intent: str, call_id: int) -> str:
    return f"""
Generate one realistic synthetic customer support call transcript.

Intent label: {intent}
Call id: call_{call_id:06d}

Requirements:
- Return only valid JSON.
- Use this schema: {{"id": "...", "intent": "{intent}", "raw_dialogue": "...", "summary": "..."}}
- raw_dialogue must be 8-14 turns with speaker prefixes "Agent:" and "Customer:".
- summary must be a concise gold-standard summary of the issue, key context, action taken, and next step.
- Do not include personally identifying information, real phone numbers, emails, addresses, or payment details.
- The intent field must be exactly "{intent}".
""".strip()


async def call_openai_compatible(
    client: httpx.AsyncClient,
    config: OpenAICompatibleConfig,
    prompt: str,
) -> dict[str, Any]:
    response = await client.post(
        f"{config.base_url.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}"},
        json={
            "model": config.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You generate clean synthetic ML training data as strict JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.8,
            "response_format": {"type": "json_object"},
        },
        timeout=config.timeout_seconds,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    return json.loads(content)


async def generate_with_api(
    output_path: Path,
    total: int,
    config: OpenAICompatibleConfig,
    concurrency: int,
    seed: int,
) -> None:
    rng = random.Random(seed)
    intents = balanced_intents(total, rng)
    semaphore = asyncio.Semaphore(concurrency)

    async with httpx.AsyncClient() as client:
        async def generate_one(index: int, intent: str) -> dict[str, Any]:
            async with semaphore:
                record = await call_openai_compatible(
                    client=client,
                    config=config,
                    prompt=build_generation_prompt(intent, index),
                )
                record["id"] = record.get("id") or f"call_{index:06d}"
                record["intent"] = intent
                record.setdefault("metadata", {})
                record["metadata"].update({"source": "openai_compatible", "model": config.model})
                return validate_record(record)

        tasks = [generate_one(index + 1, intent) for index, intent in enumerate(intents)]
        with output_path.open("w", encoding="utf-8") as file:
            for task in asyncio.as_completed(tasks):
                record = await task
                file.write(json.dumps(record, ensure_ascii=False) + "\n")


def balanced_intents(total: int, rng: random.Random) -> list[str]:
    labels = [INTENT_LABELS[index % len(INTENT_LABELS)] for index in range(total)]
    rng.shuffle(labels)
    return labels


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    required = ["id", "intent", "raw_dialogue", "summary"]
    missing = [field for field in required if not record.get(field)]
    if missing:
        raise ValueError(f"Generated record is missing required fields: {missing}")
    if record["intent"] not in INTENT_LABELS:
        raise ValueError(f"Unsupported intent label: {record['intent']}")
    record["raw_dialogue"] = normalize_text(record["raw_dialogue"])
    record["summary"] = normalize_text(record["summary"])
    return record


def normalize_text(value: str) -> str:
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()


def generate_with_templates(output_path: Path, total: int, seed: int) -> None:
    rng = random.Random(seed)
    intents = balanced_intents(total, rng)
    with output_path.open("w", encoding="utf-8") as file:
        for index, intent in enumerate(intents, start=1):
            record = build_template_record(index, intent, rng)
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/calls.jsonl"))
    parser.add_argument("--total", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--provider",
        choices=["template", "openai_compatible"],
        default="template",
        help="Use template for local smoke tests, openai_compatible for the LLM API you specify.",
    )
    parser.add_argument("--base-url", default=os.getenv("OPENAI_COMPATIBLE_BASE_URL"))
    parser.add_argument("--api-key", default=os.getenv("OPENAI_COMPATIBLE_API_KEY"))
    parser.add_argument("--model", default=os.getenv("OPENAI_COMPATIBLE_MODEL"))
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    if args.provider == "template":
        generate_with_templates(output_path=args.output, total=args.total, seed=args.seed)
        return

    missing = [
        name
        for name, value in {
            "--base-url or OPENAI_COMPATIBLE_BASE_URL": args.base_url,
            "--api-key or OPENAI_COMPATIBLE_API_KEY": args.api_key,
            "--model or OPENAI_COMPATIBLE_MODEL": args.model,
        }.items()
        if not value
    ]
    if missing:
        raise SystemExit(f"Missing API configuration: {', '.join(missing)}")

    config = OpenAICompatibleConfig(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout_seconds=args.timeout_seconds,
    )
    asyncio.run(
        generate_with_api(
            output_path=args.output,
            total=args.total,
            config=config,
            concurrency=args.concurrency,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
