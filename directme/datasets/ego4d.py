from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Ego4DClip:
    video_uid: str
    video_path: str
    start_sec: float = 0.0
    end_sec: float | None = None
    metadata: dict | None = None


def load_ego4d_manifest(path: str | Path) -> list[Ego4DClip]:
    """Load a lightweight Ego4D-style JSON/JSONL manifest.

    Expected JSONL fields:
        video_uid, video_path, start_sec, end_sec, metadata
    """
    path = Path(path)
    clips: list[Ego4DClip] = []
    if path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    for row in rows:
        clips.append(
            Ego4DClip(
                video_uid=row["video_uid"],
                video_path=row["video_path"],
                start_sec=float(row.get("start_sec", 0.0)),
                end_sec=row.get("end_sec"),
                metadata=row.get("metadata", {}),
            )
        )
    return clips
