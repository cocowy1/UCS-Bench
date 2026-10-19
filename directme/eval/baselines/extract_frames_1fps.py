#!/usr/bin/env python3
"""Extract 1 fps frames from UCS-Bench videos.

Produces the layout expected by the baseline evaluation scripts::

    <out_dir>/<video_uid>/frame_000000.png
    <out_dir>/<video_uid>/frame_000001.png
    ...
    <out_dir>/<video_uid>/index.jsonl

with one index line per frame::

    {"file": "frame_000000.png", "sec": 0}

This helper reproduces the format of the frame dump used in the paper
(`data_frames_1fps`). Exact PNG encoder settings of the original dump may
differ slightly, which can cause tiny pixel-level differences.

Usage::

    python extract_frames_1fps.py --video_root /path/to/videos --out_dir /path/to/data_frames_1fps

``<video_root>`` should contain one ``<video_uid>.mp4`` per video.
"""

import argparse
import json
import subprocess
from pathlib import Path


def extract_video(video_path: Path, out_dir: Path) -> bool:
    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "index.jsonl"

    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(video_path),
        "-vf", "fps=1",
        "-start_number", "0",
        str(out_dir / "frame_%06d.png"),
    ]
    ret = subprocess.run(cmd).returncode
    if ret != 0:
        print(f"[WARN] ffmpeg failed for {video_path}")
        return False

    frames = sorted(out_dir.glob("frame_*.png"))
    with open(index_path, "w", encoding="utf-8") as f:
        for i, fp in enumerate(frames):
            f.write(json.dumps({"file": fp.name, "sec": i}) + "\n")

    print(f"[INFO] {video_path.name}: {len(frames)} frames -> {out_dir}")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_root", required=True,
                        help="Directory containing <video_uid>.mp4 files")
    parser.add_argument("--out_dir", required=True,
                        help="Output root (one subdirectory per video)")
    args = parser.parse_args()

    video_root = Path(args.video_root)
    out_root = Path(args.out_dir)

    videos = sorted(video_root.glob("*.mp4"))
    if not videos:
        print(f"[WARN] No .mp4 files found in {video_root}")
        return

    ok, failed = 0, 0
    for vp in videos:
        if extract_video(vp, out_root / vp.stem):
            ok += 1
        else:
            failed += 1

    print(f"Done: {ok} extracted, {failed} failed.")


if __name__ == "__main__":
    main()
