#!/usr/bin/env python3
"""Build scene graphs for v20 videos that are missing one.

Loads the perception stack (DA3 + YOLO-World + SAM2 + tracker) ONCE and
loops over the missing uids (the per-uid helper in eval_ucsbench rebuilds
the stack per call, which wastes hours on model loading over 100+ videos).

Usage:
    CUDA_VISIBLE_DEVICES=0 python tools/build_missing_graphs.py \
        --qa_dir /data/ucsbench/MC_QAs_v20 \
        --graph_root runs/ucsbench_perception_graphs \
        --frames_root /data/ucsbench/data_frames_1fps
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)  # repo root (for configs/)
sys.path.insert(0, os.getcwd())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa_dir", default="/data/ywang/dataset/SpatialMemory/MC_QAs_v20")
    parser.add_argument("--graph_root", default="/data/ywang/my_projects/VideoUnderstanding/runs/ucsbench_perception_graphs")
    parser.add_argument("--frames_root", default="/data/ywang/dataset/SpatialMemory/data_frames_1fps")
    parser.add_argument("--uids_file", default="", help="optional: file with one uid per line (default: auto-detect missing)")
    parser.add_argument("--limit", type=int, default=0, help="stop after N videos (0 = all)")
    parser.add_argument("--perception_device", default="cuda")
    parser.add_argument("--perception_chunk_size", type=int, default=300)
    parser.add_argument("--detection_stride", type=int, default=1)
    parser.add_argument("--score_threshold", type=float, default=0.15)
    parser.add_argument("--class_limit", type=int, default=300)
    parser.add_argument("--da3_process_res", type=int, default=504)
    parser.add_argument("--input_fps", type=float, default=1.0)
    parser.add_argument("--max_perception_frames", type=int, default=0)
    parser.add_argument("--use_sam2", action="store_true", default=True)
    parser.add_argument("--disable_scene_tag", action="store_true")
    args = parser.parse_args()

    # NOTE: deliberately do NOT import directme.eval.eval_ucsbench here — its
    # module body hard-codes os.environ["CUDA_VISIBLE_DEVICES"] = "2", which
    # would silently relocate this build onto whatever that script considers
    # "its" GPU. strip_mcq_uid is copied below instead.
    import re as _re

    def strip_mcq_uid(video_uid: str) -> str:
        video_uid = (video_uid or "").strip()
        if not video_uid:
            return ""
        matches = list(_re.finditer(r"mcq", video_uid, flags=_re.IGNORECASE))
        if not matches:
            return video_uid
        m = matches[-1]
        i = m.start() - 1
        while i >= 0 and video_uid[i] in "-_":
            i -= 1
        return video_uid[: i + 1]

    from directme.mapping.offline_engine import OfflineMappingEngine
    from directme.perception.adapters.open_vocab_tracking import (
        OpenVocabularyTrackingAdapter,
        Sam2MaskRefiner,
        SimpleIoUAppearanceTracker,
        YoloWorldDetector,
    )
    from directme.perception.adapters.composed import build_unified_perception_backend
    from directme.perception.test_perception_stack import (
        _list_images,
        _make_video_frames,
        _load_classes_from_yaml,
    )
    from directme.config import DirectMeConfig
    import torch

    repo = Path.cwd()
    yolo_weights = str(repo / "ckpts/yolo/yolov8m-worldv2.pt")
    classes_file = str(repo / "directme/perception/adapters/Object.yaml")
    sam2_ckpt = str(repo / "ckpts/sam2/sam2.1_hiera_base_plus.pt")
    sam2_cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"

    # ── resolve missing uids ─────────────────────────────────────────────
    if args.uids_file:
        missing = [l.strip() for l in open(args.uids_file) if l.strip()]
    else:
        uids = set()
        for f in os.listdir(args.qa_dir):
            if not f.endswith(".json"):
                continue
            for it in json.load(open(os.path.join(args.qa_dir, f))):
                uids.add(strip_mcq_uid(str(it.get("video_uid", ""))))
        missing = sorted(
            u for u in uids
            if not os.path.exists(os.path.join(args.graph_root, u, "directme_mapping_run", "scene_graph.json"))
        )
    if args.limit:
        missing = missing[: args.limit]
    total_frames = sum(
        len([x for x in os.listdir(os.path.join(args.frames_root, u)) if x.endswith(".png")])
        for u in missing if os.path.isdir(os.path.join(args.frames_root, u))
    )
    print(f"[build] {len(missing)} missing graphs, ~{total_frames} frames total", flush=True)

    # ── persistent perception stack ──────────────────────────────────────
    classes = _load_classes_from_yaml(classes_file, class_limit=args.class_limit)
    detector = YoloWorldDetector(
        weights=yolo_weights, classes=classes,
        score_threshold=args.score_threshold, device=args.perception_device,
    )
    segmenter = None
    if args.use_sam2:
        segmenter = Sam2MaskRefiner(checkpoint=sam2_ckpt, config=sam2_cfg, device=args.perception_device)
    tracker = OpenVocabularyTrackingAdapter(
        detector=detector, segmenter=segmenter,
        tracker=SimpleIoUAppearanceTracker(), detection_stride=args.detection_stride,
    )
    backend = build_unified_perception_backend(
        depth_backend="da3", tracker=tracker, device=args.perception_device,
        da3_process_res=args.da3_process_res,
        enable_scene_tag=not args.disable_scene_tag,
    )

    # ── per-video loop ───────────────────────────────────────────────────
    t0 = time.time()
    frames_done = 0
    for i, uid in enumerate(missing):
        work_dir = Path(args.graph_root) / uid
        graph_json = work_dir / "directme_mapping_run" / "scene_graph.json"
        if graph_json.exists():
            print(f"[build] ({i+1}/{len(missing)}) {uid}: exists, skip", flush=True)
            continue
        image_paths = _list_images(
            Path(args.frames_root) / uid,
            max_images=args.max_perception_frames or None,
        )
        if not image_paths:
            print(f"[build] ({i+1}/{len(missing)}) {uid}: NO FRAMES, skip", flush=True)
            continue
        frames = _make_video_frames(image_paths)
        print(f"[build] ({i+1}/{len(missing)}) {uid}: {len(frames)} frames", flush=True)
        tv = time.time()

        # One OOM'd video must not kill the whole run: retry once with half
        # the chunk size (halves DA3 joint-inference memory), else move on.
        chunk = max(1, int(args.perception_chunk_size))
        for attempt in range(2):
            try:
                # Fresh engine + tracker per video: track ids must not leak
                # across videos, and a partial graph from a crashed attempt
                # must not be resumed with stale chunk numbering.
                if graph_json.parent.exists():
                    import shutil
                    shutil.rmtree(graph_json.parent, ignore_errors=True)
                fresh_tracker = OpenVocabularyTrackingAdapter(
                    detector=detector, segmenter=segmenter,
                    tracker=SimpleIoUAppearanceTracker(), detection_stride=args.detection_stride,
                )
                backend.tracker = fresh_tracker
                engine = OfflineMappingEngine(
                    backend=backend, config=DirectMeConfig(run_dir=str(graph_json.parent)),
                )
                for cid, start in enumerate(range(0, len(frames), chunk)):
                    engine.process_chunk(frames[start:start + chunk], chunk_id=cid)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                break
            except torch.OutOfMemoryError as exc:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(f"[build] {uid}: OOM at chunk={chunk} ({exc}); retrying with chunk={chunk//2}", flush=True)
                chunk = max(1, chunk // 2)
                if attempt == 1:
                    print(f"[build] {uid}: FAILED even at chunk={chunk}; skipping", flush=True)
                    import shutil
                    shutil.rmtree(graph_json.parent, ignore_errors=True)
                    continue

        frames_done += len(frames)
        rate = frames_done / max(time.time() - t0, 1e-6)
        eta_h = (total_frames - frames_done) / max(rate, 1e-6) / 3600
        print(
            f"[build] {uid} done in {time.time()-tv:.0f}s  "
            f"(graph exists: {graph_json.exists()}, chunk={chunk})  "
            f"overall {rate:.2f} f/s  ETA {eta_h:.1f} h", flush=True,
        )

    print(f"[build] ALL DONE: {len(missing)} videos, {frames_done} frames, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
