"""End-to-end DirectMe runner on a real frame folder.

This script wires together the real perception backbones (Depth Anything 3 +
YOLO-World + SAM 2 + simple tracker) with the DirectMe offline mapping engine,
then asks a single user-relative question against the resulting scene graph.

Usage
-----

1. Extract a video to 1 FPS images, e.g.::

     mkdir -p ./frames
     ffmpeg -i your_video.mp4 -vf fps=1 ./frames/frame_%06d.jpg

2. Install perception extras::

     pip install -e ".[perception,video]"
     # SAM 2 and Depth Anything 3 install from source — see docs/adapter_guide.md

3. Run::

     python examples/run_real_pipeline.py \\
         --frames ./frames \\
         --classes "cup,phone,bottle,laptop,chair,table,sink,fridge,door,bag" \\
         --out runs/my_session \\
         --question "我身边有几个杯子？在哪？" \\
         --language zh

For a smoke test with no GPU, pass ``--toy``::

     python examples/run_real_pipeline.py --toy --out runs/toy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from directme.config import DirectMeConfig
from directme.data.frame_source import ImageFolderFrameSource
from directme.geometry.poses import SE3
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.qa.generator import RuleBasedAnswerGenerator
from directme.retrieval.retriever import GraphRetriever


def build_real_backend(args: argparse.Namespace):
    """Lazy-instantiate the default real DirectMe backend."""
    from directme.perception.runtime import build_composed_backend, resolve_runtime_device

    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    resolved = resolve_runtime_device(args.device)
    print(f"[init] device={resolved} DA3={args.depth_model}")
    print(f"[init] YOLO-World weights={args.yolo_weights} ({len(classes)} classes)")
    if args.sam2_checkpoint and args.sam2_config:
        print(f"[init] SAM 2 checkpoint={args.sam2_checkpoint}")
    else:
        print("[init] SAM 2 disabled; using bbox-center depth fallback")
    return build_composed_backend(
        classes=classes,
        device=args.device,
        depth_model=args.depth_model,
        use_ray_pose=args.use_ray_pose,
        yolo_weights=args.yolo_weights,
        score_threshold=args.score_threshold,
        detection_stride=args.detection_stride,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        min_pose_confidence=args.min_pose_confidence,
    )


def build_toy_backend(out_dir: Path):
    from directme.perception.toy import build_living_room_kitchen_demo
    return build_living_room_kitchen_demo(out_dir / "keyframes")


def run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if args.config:
        cfg = DirectMeConfig.from_yaml(args.config)
    else:
        cfg = DirectMeConfig()
    cfg.run_dir = str(out)
    cfg.storage.backend = args.storage_backend
    cfg.retrieval.reachable_radius_m = args.reachable_radius_m
    cfg.retrieval.language = args.language

    if args.toy:
        frames, backend = build_toy_backend(out)
        chunk_size = 2
        current_pose = SE3.from_translation([7.0, 0.0, 0.0])
    else:
        if not args.frames:
            raise SystemExit("Either --toy or --frames is required.")
        source = ImageFolderFrameSource(args.frames, fps=args.fps)
        frames = source.frames()
        if not frames:
            raise SystemExit(f"No frames found under {args.frames}")
        backend = build_real_backend(args)
        chunk_size = cfg.stream.chunk_size_frames
        # Default to identity (sit at the latest world end). Real deployments
        # would pull T_world_from_camera_current from the live tracker.
        current_pose = SE3.identity()

    print(f"[mapping] processing {len(frames)} frames in chunks of {chunk_size}")
    engine = OfflineMappingEngine(backend=backend, config=cfg)
    events = engine.process_frames(frames, chunk_size=chunk_size)
    graph = engine.graph
    assert graph is not None

    print(f"[mapping] events={len(events)} nodes={len(graph.nodes)} "
          f"places={len(graph.place_nodes)} edges={len(graph.edges)}")
    rejected = [r for r in engine.chunk_reports if not r.accepted]
    if rejected:
        print(f"[mapping] {len(rejected)} chunk(s) rejected: "
              + ", ".join(f"chunk_{r.chunk_id}={r.rejection_reason}" for r in rejected))

    # Use the most recent valid world pose if available, otherwise the user-supplied default.
    if not args.toy:
        current_pose = engine.pose_propagator.current_world_end

    retriever = GraphRetriever(
        graph,
        reachable_radius_m=cfg.retrieval.reachable_radius_m,
        lateral_tolerance_ratio=cfg.retrieval.lateral_tolerance_ratio,
    )
    context = retriever.retrieve(args.question, current_pose, language=cfg.retrieval.language)
    answer = RuleBasedAnswerGenerator().answer(context)

    print()
    print(f"Question: {args.question}")
    print(f"Answer:   {answer}")
    print()
    print(GraphRetriever.render_summary(context))

    # Persist the retrieved subgraph for debugging.
    debug_path = out / "last_query.json"
    debug_path.write_text(
        json.dumps(
            {
                "question": args.question,
                "answer": answer,
                "items": [
                    {
                        "node_id": it.node.node_id,
                        "label": it.node.semantic_label,
                        "egocentric": it.egocentric,
                    }
                    for it in context.items
                ],
                "ego_edges": context.ego_edges,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[debug] last query payload written to {debug_path}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="DirectMe end-to-end real-perception runner")
    p.add_argument("--frames", help="folder of pre-extracted images")
    p.add_argument("--out", default="runs/real", help="output directory")
    p.add_argument("--config", default=None, help="optional YAML config to override defaults")

    p.add_argument("--question", default="我身边有几个杯子？在哪？")
    p.add_argument("--language", default="zh", choices=["zh", "en"])
    p.add_argument("--reachable-radius-m", type=float, default=5.0)

    p.add_argument("--classes", default="cup,phone,bottle,chair,table,laptop,sink,door,bag,book")
    p.add_argument("--fps", type=float, default=1.0)
    p.add_argument("--device", default="auto", help="auto | cuda | cpu | mps")
    p.add_argument("--score-threshold", type=float, default=0.20)
    p.add_argument("--detection-stride", type=int, default=5)

    p.add_argument("--depth-model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    p.add_argument("--use-ray-pose", action="store_true")

    p.add_argument("--yolo-weights", default="yolov8s-worldv2.pt")
    p.add_argument("--sam2-checkpoint", default=None, help="optional SAM 2 checkpoint path")
    p.add_argument("--sam2-config", default=None, help="optional SAM 2 config path")

    p.add_argument("--min-pose-confidence", type=float, default=0.30)

    p.add_argument("--storage-backend", default="json", choices=["json", "sqlite"])
    p.add_argument("--toy", action="store_true",
                   help="bypass GPU models and run the deterministic living-room→kitchen demo")
    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
