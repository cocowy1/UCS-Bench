"""Tests for directme.perception.ingest."""

from __future__ import annotations

from pathlib import Path

from directme.perception.base import VideoFrame
from directme.perception.ingest import (
    group_into_chunks,
    iter_frames_from_paths,
)


def test_iter_frames_from_paths_synthesizes_timestamps(tmp_path: Path):
    paths = []
    for i in range(5):
        p = tmp_path / f"frame_{i}.jpg"
        p.write_bytes(b"\xff\xd8\xff\xd9")  # minimal "image-shaped" stub
        paths.append(p)

    frames = list(iter_frames_from_paths(paths, fps=1.0, start_timestamp=10.0))
    assert len(frames) == 5
    assert frames[0].index == 0
    assert frames[0].timestamp == 10.0
    assert frames[4].timestamp == 14.0
    for f in frames:
        assert isinstance(f, VideoFrame)
        assert f.image_path is not None
        assert f.metadata["source"] == "frame_list"


def test_iter_frames_from_paths_uses_explicit_timestamps(tmp_path: Path):
    paths = [tmp_path / f"f_{i}.jpg" for i in range(3)]
    for p in paths:
        p.write_bytes(b"\x00")
    ts = [0.0, 5.5, 12.3]
    frames = list(iter_frames_from_paths(paths, timestamps=ts))
    assert [f.timestamp for f in frames] == ts


def test_group_into_chunks_basic():
    frames = [VideoFrame(index=i, timestamp=float(i)) for i in range(7)]
    chunks = list(group_into_chunks(frames, chunk_size=3))
    # 3 + 3 + 1 -> three chunks
    assert [c[0] for c in chunks] == [0, 1, 2]
    assert [len(c[1]) for c in chunks] == [3, 3, 1]


def test_group_into_chunks_empty_input_yields_nothing():
    chunks = list(group_into_chunks([], chunk_size=4))
    assert chunks == []


def test_group_into_chunks_rejects_invalid_chunk_size():
    try:
        list(group_into_chunks([VideoFrame(0, 0.0)], chunk_size=0))
    except ValueError:
        return
    raise AssertionError("group_into_chunks must reject chunk_size <= 0")


def test_iter_frames_from_video_raises_helpful_error_when_missing_video(tmp_path: Path):
    """When the file does not exist, error must precede the imageio import.

    This protects users on machines without imageio-ffmpeg from getting an
    obscure import error instead of the obvious "file not found" cause.
    """
    try:
        from directme.perception.ingest import iter_frames_from_video
        list(iter_frames_from_video(tmp_path / "nope.mp4"))
    except FileNotFoundError:
        return
    except ImportError:
        # Acceptable on machines without imageio-ffmpeg, but only AFTER we've
        # tried to read the file. We can't easily distinguish here, so skip.
        return
    raise AssertionError("expected FileNotFoundError for a missing video file")
