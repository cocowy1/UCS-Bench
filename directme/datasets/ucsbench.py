"""UCS-Bench dataset loader.

The on-disk format is a JSON or JSONL list of records like::

    {
      "video_uid": "demo",
      "query_timestamp": 0.0,
      "dimension": "category_quantity",
      "question": "How many red cups are there around me?",
      "expected_count": 2,
      "expected_labels": ["cup"]
    }

All fields except ``video_uid`` and ``question`` are optional. Per-dimension
gold annotations live under top-level keys (``expected_count``,
``expected_relation``, ``expected_reachable``, etc.) and are passed through
into ``options`` so the evaluator can consume them generically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# Top-level keys that the evaluator interprets as gold annotations.
_GOLD_KEYS: tuple[str, ...] = (
    "dimension",
    "expected_relation",
    "expected_target_label",
    "expected_target_node_id",
    "expected_target_place",
    "expected_query_pose",
    "expected_reachable",
    "expected_count",
    "expected_labels",
    "expected_path_labels",
)


@dataclass
class UCSQuestion:
    video_uid: str
    query_timestamp: float
    question: str
    answer: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict] = field(default_factory=list)


def _row_to_question(row: dict[str, Any]) -> UCSQuestion:
    options: dict[str, Any] = dict(row.get("options") or {})
    # Fold the canonical gold keys into options so the scorer finds them.
    for key in _GOLD_KEYS:
        if key in row and key not in options:
            options[key] = row[key]
    return UCSQuestion(
        video_uid=str(row["video_uid"]),
        query_timestamp=float(row.get("query_timestamp", row.get("timestamp", 0.0))),
        question=str(row["question"]),
        answer=row.get("answer"),
        options=options,
        evidence=list(row.get("evidence") or []),
    )


def load_ucs_questions(path: str | Path) -> list[UCSQuestion]:
    """Load UCS-Bench-style QA metadata from JSON or JSONL."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        rows = json.loads(text)
        if isinstance(rows, dict) and "questions" in rows:
            rows = rows["questions"]
    return [_row_to_question(r) for r in rows]
