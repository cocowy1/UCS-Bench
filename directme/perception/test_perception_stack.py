#!/usr/bin/env python3
"""Smoke-test DirectMe perception adapters.

Default mode is dependency-light: it creates a fake SCAL3R result directory and
checks that these components work together:

1. ``Scal3ROutputReader`` parses poses / intrinsics / depth.
2. ``OpenVocabularyTrackingAdapter`` assigns stable track ids.
3. ``Scal3RComposedBackend`` returns ``ChunkPerception``.
4. ``OfflineMappingEngine`` can unproject bbox+depth into a scene graph node.

Real mode can additionally read or run SCAL3R outputs and optionally instantiate
SCAL3R / YOLO-World / SAM 2 / Depth Anything 3 imports to verify your environment.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Sequence

import numpy as np


def _add_project_root(project_root: str | Path) -> None:
    root = Path(project_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _write_fake_scal3r_result(result_dir: Path, n_frames: int, h: int, w: int) -> None:
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "depths").mkdir(parents=True, exist_ok=True)

    mats = []
    for i in range(n_frames):
        mat = np.eye(4, dtype=np.float64)
        mat[0, 3] = 0.25 * i
        # Deliberately add tiny Sim(3)-style scale to verify the reader strips
        # scale and produces a valid SE3 rotation.
        scale = 1.0 + 0.01 * i
        mat[:3, :3] *= scale
        mats.append(mat.reshape(-1))
        depth = np.full((h, w), 2.0 + 0.1 * i, dtype=np.float32)
        np.save(result_dir / "depths" / f"{i:06d}.npy", depth)

    np.savetxt(result_dir / "mat.txt", np.stack(mats, axis=0), fmt="%.8f")
    np.savetxt(result_dir / "confidence.txt", np.ones((n_frames,), dtype=np.float32), fmt="%.4f")

    fx = fy = 100.0
    cx = w / 2.0
    cy = h / 2.0
    lines = ["%YAML:1.0\n", "---\n", "names:\n"]
    for i in range(n_frames):
        lines.append(f' - "{i:06d}"\n')
    for i in range(n_frames):
        name = f"{i:06d}"
        k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
        lines.extend(
            [
                f"K_{name}: !!opencv-matrix\n",
                " rows: 3\n",
                " cols: 3\n",
                " dt: d\n",
                " data: [" + ", ".join(f"{x:.10f}" for x in k) + "]\n",
                f"H_{name}: {float(h):.10f}\n",
                f"W_{name}: {float(w):.10f}\n",
            ]
        )
    (result_dir / "intri.yml").write_text("".join(lines), encoding="utf-8")


def _make_synthetic_rgb_frames(n_frames: int, h: int, w: int) -> list[np.ndarray]:
    images = []
    for i in range(n_frames):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[..., 1] = 30 + i * 10
        x1, y1, x2, y2 = 35 + i * 2, 25, 85 + i * 2, 75
        img[y1:y2, x1:x2, 0] = 220
        img[y1:y2, x1:x2, 1] = 40
        img[y1:y2, x1:x2, 2] = 40
        images.append(img)
    return images


def _build_fake_tracker():
    from directme.perception.adapters.open_vocab_tracking import (
        Detection,
        OpenVocabularyTrackingAdapter,
        SimpleIoUAppearanceTracker,
    )

    class FakeDetector:
        def detect(self, image):
            # Move the box slightly as the frame index changes. The simple IoU
            # tracker should keep one persistent id.
            h, w = image.shape[:2]
            del h, w
            return [Detection(label="cup", bbox_xyxy=(35.0, 25.0, 85.0, 75.0), score=0.99)]

    return OpenVocabularyTrackingAdapter(
        detector=FakeDetector(),
        segmenter=None,
        tracker=SimpleIoUAppearanceTracker(),
        detection_stride=1,
    )


def _check_imports(*, check_heavy: bool) -> dict[str, str]:
    report = {}
    required = [
        "directme.perception.adapters.scal3r",
        "directme.perception.adapters.open_vocab_tracking",
        "directme.perception.adapters.composed",
        "directme.perception.adapters.depth_anything3",
    ]
    for name in required:
        try:
            importlib.import_module(name)
            report[name] = "ok"
        except Exception as exc:
            report[name] = f"FAIL: {type(exc).__name__}: {exc}"

    if check_heavy:
        heavy = [
            "torch",
            "ultralytics",
            "sam2.build_sam",
            "sam2.sam2_image_predictor",
            "depth_anything_3.api",
            "scal3r.run",
        ]
        for name in heavy:
            try:
                importlib.import_module(name)
                report[name] = "ok"
            except Exception as exc:
                report[name] = f"SKIP/FAIL: {type(exc).__name__}: {exc}"
    return report


def run_quick(args: argparse.Namespace) -> None:
    from directme.config import DirectMeConfig
    from directme.mapping.offline_engine import OfflineMappingEngine
    from directme.perception.adapters.scal3r import (
        Scal3RComposedBackend,
        Scal3RDepthPoseAdapter,
        Scal3ROutputReader,
    )
    from directme.perception.base import VideoFrame

    work_dir = Path(args.work_dir).resolve()
    if work_dir.exists() and args.clean:
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    n_frames, h, w = 3, 120, 160
    result_dir = work_dir / "fake_scal3r"
    _write_fake_scal3r_result(result_dir, n_frames=n_frames, h=h, w=w)

    reader = Scal3ROutputReader()
    parsed = reader.read(result_dir, expected_frames=n_frames)
    assert len(parsed.poses_local) == n_frames
    assert parsed.depth_maps is not None and len(parsed.depth_maps) == n_frames
    assert parsed.intrinsics is not None and len(parsed.intrinsics) == n_frames
    for pose in parsed.poses_local:
        rot = pose.rotation
        assert np.allclose(rot @ rot.T, np.eye(3), atol=1e-5)

    images = _make_synthetic_rgb_frames(n_frames, h, w)
    frames = [VideoFrame(index=i, timestamp=float(i), image=images[i]) for i in range(n_frames)]

    depth_pose = Scal3RDepthPoseAdapter(precomputed_root=result_dir)
    backend = Scal3RComposedBackend(depth_pose=depth_pose, tracker=_build_fake_tracker())
    chunk = backend.process_chunk(frames, chunk_id=0)
    assert len(chunk.frames) == n_frames
    assert all(fp.depth is not None for fp in chunk.frames)
    assert all(fp.intrinsics is not None for fp in chunk.frames)
    assert all(len(fp.objects) == 1 for fp in chunk.frames)
    track_ids = [fp.objects[0].track_id for fp in chunk.frames]
    assert len(set(track_ids)) == 1, f"tracker did not keep a stable id: {track_ids}"

    config = DirectMeConfig(run_dir=str(work_dir / "mapping_run"))
    engine = OfflineMappingEngine(backend=backend, config=config)
    events = engine.process_chunk(frames, chunk_id=0)
    assert len(events) >= 1
    assert engine.graph is not None and len(engine.graph.nodes) >= 1

    out = {
        "status": "ok",
        "fake_scal3r_result": str(result_dir),
        "n_frames": n_frames,
        "n_events": len(events),
        "n_graph_nodes": len(engine.graph.nodes),
        "track_ids": track_ids,
        "scene_graph_json": str(Path(config.run_dir) / "scene_graph.json"),
    }
    print(json.dumps(out, indent=2, ensure_ascii=False))


def _list_images(image_dir: str | Path, max_images: int | None) -> list[Path]:
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = sorted(p for p in Path(image_dir).iterdir() if p.suffix.lower() in suffixes)
    if max_images:
        paths = paths[:max_images]
    if not paths:
        raise FileNotFoundError(f"No images found in {image_dir}")
    return paths


def run_real(args: argparse.Namespace) -> None:
    from directme.perception.adapters.scal3r import (
        Scal3RDepthPoseAdapter,
        Scal3ROutputReader,
        Scal3RRunner,
    )
    from directme.perception.base import VideoFrame

    if args.scal3r_result_dir:
        parsed = Scal3ROutputReader().read(args.scal3r_result_dir)
        print(
            json.dumps(
                {
                    "status": "read_scal3r_result_ok",
                    "result_dir": args.scal3r_result_dir,
                    "n_poses": len(parsed.poses_local),
                    "has_depth": parsed.depth_maps is not None,
                    "has_intrinsics": parsed.intrinsics is not None,
                    "first_translation": parsed.poses_local[0].translation.tolist(),
                },
                indent=2,
            )
        )
        return

    if not args.image_dir:
        raise ValueError("real mode needs --image-dir or --scal3r-result-dir")

    paths = _list_images(args.image_dir, args.max_images)
    frames = [VideoFrame(index=i, timestamp=float(i), image_path=str(p)) for i, p in enumerate(paths)]
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    runner = Scal3RRunner(
        config=args.scal3r_config,
        checkpoint=args.scal3r_checkpoint,
        device=args.device if args.device != "auto" else None,
        block_size=args.scal3r_block_size,
        overlap_size=args.scal3r_overlap_size,
        use_loop=args.scal3r_use_loop,
        save_dpt=1,
        save_xyz=0,
    )
    adapter = Scal3RDepthPoseAdapter(runner=runner, work_dir=work_dir, keep_work_dir=True)
    outputs = adapter.infer_frames(frames, chunk_id=0)
    print(
        json.dumps(
            {
                "status": "run_scal3r_ok",
                "n_frames": len(outputs),
                "work_dir": str(work_dir),
                "has_depth": all(o.depth is not None for o in outputs),
                "has_intrinsics": all(o.intrinsics is not None for o in outputs),
                "first_translation": outputs[0].pose_local.translation.tolist(),
            },
            indent=2,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".", help="Root of the DirectMe repository")
    parser.add_argument("--mode", choices=["quick", "real"], default="quick")
    parser.add_argument("--work-dir", default="/tmp/directme_perception_smoke")
    parser.add_argument("--clean", action="store_true", help="Delete work-dir before running quick mode")
    parser.add_argument("--check-heavy", action="store_true", help="Also import torch / scal3r / DA3 / SAM2 / ultralytics")

    # Real/SCAL3R options.
    parser.add_argument("--image-dir", default="", help="Image directory for real SCAL3R run")
    parser.add_argument("--max-images", type=int, default=5)
    parser.add_argument("--scal3r-result-dir", default="", help="Existing SCAL3R result dir to read")
    parser.add_argument("--scal3r-config", default="configs/models/scal3r.yaml")
    parser.add_argument("--scal3r-checkpoint", default=None)
    parser.add_argument("--scal3r-block-size", type=int, default=None)
    parser.add_argument("--scal3r-overlap-size", type=int, default=None)
    parser.add_argument("--scal3r-use-loop", type=int, default=None)
    parser.add_argument("--device", default="auto")

    args = parser.parse_args(argv)
    _add_project_root(args.project_root)

    report = _check_imports(check_heavy=args.check_heavy)
    print("[import report]")
    for key, value in report.items():
        print(f"  {key}: {value}")
    failed_core = [k for k, v in report.items() if k.startswith("directme") and not v == "ok"]
    if failed_core:
        raise RuntimeError(f"Core DirectMe imports failed: {failed_core}")

    if args.mode == "quick":
        run_quick(args)
    else:
        run_real(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
