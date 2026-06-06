"""Prompt template for AgentDQ.

The agent receives:
- task description + label meaning
- column descriptions (schema)
- a compact training-set profile (per-column stats, top-k for categoricals)
- one row of data
and must return a strict JSON object {prediction, confidence, rationale}.
"""
from __future__ import annotations

import json
from typing import Iterable

from consumers.schema import SCHEMA, TASKS

SYSTEM_PROMPT = (
    "You are a careful tabular-data classifier. "
    "Read the column descriptions and one row of data, then return a JSON object "
    "with exactly these keys: prediction (integer 0 or 1), confidence (float in [0,1]), "
    "rationale (string, <= 40 words). Output ONLY the JSON object, no prose, no markdown fences."
)


def _schema_block(feature_cols: Iterable[str]) -> str:
    lines = ["COLUMNS:"]
    for c in feature_cols:
        desc = SCHEMA.get(c, "(no description)")
        lines.append(f"- {c}: {desc}")
    return "\n".join(lines)


def _row_block(row: dict, feature_cols: Iterable[str]) -> str:
    cleaned = {c: (None if row.get(c) in (None, "") else row.get(c)) for c in feature_cols}
    return "ROW:\n" + json.dumps(cleaned, default=str, ensure_ascii=False)


def _task_block(task: str) -> str:
    t = TASKS[task]
    return (
        f"TASK: {t['name']}\n"
        f"GOAL: {t['description']}\n"
        f"LABEL: {t['label_meaning']}\n"
    )


def _output_block() -> str:
    return (
        'OUTPUT FORMAT (strict JSON, single line):\n'
        '{"prediction": 0 or 1, "confidence": 0.0..1.0, "rationale": "..."}'
    )


def build_prompt(task: str, row: dict, feature_cols: list[str], profile_summary: str) -> list[dict]:
    user = "\n\n".join([
        _task_block(task),
        _schema_block(feature_cols),
        "TRAINING-SET PROFILE (compact summary):\n" + profile_summary,
        _row_block(row, feature_cols),
        _output_block(),
    ])
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
