"""Tests for frame-path portability helpers and plural query parsing."""

from __future__ import annotations

import os

from directme.datasets.frame_paths import (
    infer_frames_root_from_paths,
    iter_video_frames,
    relativize_frame_path,
    resolve_frame_path,
)
from directme.retrieval.query_parser import parse_query


def test_resolve_existing_absolute_unchanged(tmp_path):
    f = tmp_path / "frame_000001.png"
    f.write_bytes(b"")
    assert resolve_frame_path(str(f)) == str(f)


def test_resolve_missing_absolute_via_frames_root_suffix(tmp_path):
    root = tmp_path / "frames"
    (root / "01042").mkdir(parents=True)
    f = root / "01042" / "frame_000123.png"
    f.write_bytes(b"")
    missing = "/somewhere/else/01042/frame_000123.png"
    assert resolve_frame_path(missing, frames_root=root) == str(f)


def test_resolve_relative_path_against_root(tmp_path):
    root = tmp_path / "frames"
    (root / "vid").mkdir(parents=True)
    (root / "vid" / "frame_000000.png").write_bytes(b"")
    assert resolve_frame_path("vid/frame_000000.png", frames_root=root) is not None
    assert resolve_frame_path("vid/frame_000999.png", frames_root=root) is None


def test_resolve_none_and_no_root(tmp_path):
    assert resolve_frame_path(None) is None
    assert resolve_frame_path(str(tmp_path / "nope.png")) is None
    assert resolve_frame_path("x.png", frames_root=tmp_path) is None


def test_relativize_and_roundtrip(tmp_path):
    root = tmp_path / "frames"
    (root / "01042").mkdir(parents=True)
    f = root / "01042" / "frame_000001.png"
    f.write_bytes(b"")
    rel = relativize_frame_path(str(f), root)
    assert rel == "01042/frame_000001.png"
    # Round-trip: relativized path resolves back to the real file.
    assert resolve_frame_path(rel, frames_root=root) == str(f)
    # Outside the root stays absolute.
    outside = tmp_path / "other.png"
    assert relativize_frame_path(str(outside), root) == str(outside)
    assert relativize_frame_path(str(f), None) == str(f)


def test_iter_video_frames_sorted(tmp_path):
    root = tmp_path / "frames" / "v"
    root.mkdir(parents=True)
    for i in (2, 0, 1):
        (root / f"frame_{i:06d}.png").write_bytes(b"")
    (root / "notes.txt").write_text("x")
    got = iter_video_frames(tmp_path / "frames", "v")
    assert [os.path.basename(p) for p in got] == [
        "frame_000000.png", "frame_000001.png", "frame_000002.png",
    ]
    assert iter_video_frames(tmp_path / "frames", "missing") == []


def test_infer_frames_root(tmp_path):
    root = tmp_path / "frames"
    (root / "v").mkdir(parents=True)
    f = root / "v" / "frame_000000.png"
    f.write_bytes(b"")
    assert infer_frames_root_from_paths([str(f)]) == str(root)


# ── query parser ────────────────────────────────────────────────────────────


def test_plural_intent_parsing():
    assert parse_query("How many sofas are in the room?", language="en").labels == ["sofa"]
    assert parse_query("Where are the couches?", language="en").labels == ["sofa"]
    assert parse_query("how many fruits did I see?", language="en").labels == ["fruit"]
    # Singular still works.
    assert parse_query("Where is the cup?", language="en").labels == ["cup"]


def test_plural_does_not_overmatch():
    it = parse_query("There is a busstop nearby", language="en")
    assert "bus" not in it.labels
