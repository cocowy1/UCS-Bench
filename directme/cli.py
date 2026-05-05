from __future__ import annotations

import argparse
import json
from pathlib import Path

from directme.config import DirectMeConfig
from directme.eval import UCSBenchEvaluator
from directme.datasets.ucsbench import load_ucs_questions
from directme.geometry.poses import SE3
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.mapping.scene_graph import SceneGraph
from directme.perception.toy import build_living_room_kitchen_demo
from directme.qa.generator import RuleBasedAnswerGenerator
from directme.retrieval.pose_lookup import pose_from_graph_timeline
from directme.retrieval.retriever import GraphRetriever


def _pose_from_json(value: str | None) -> SE3:
    if not value:
        return SE3.identity()
    return SE3.from_list(json.loads(value))


def _pose_from_json_or_graph(value: str | None, graph: SceneGraph) -> SE3:
    if value:
        return _pose_from_json(value)
    return pose_from_graph_timeline(graph)


# ---------------------------------------------------------------------------
# `directme demo`
# ---------------------------------------------------------------------------


def cmd_demo(args: argparse.Namespace) -> None:
    out = Path(args.out)
    config = DirectMeConfig()
    config.run_dir = str(out)
    frames, backend = build_living_room_kitchen_demo(out / "keyframes")
    engine = OfflineMappingEngine(backend=backend, config=config)
    events = engine.process_frames(frames, chunk_size=2)
    graph = engine.graph
    assert graph is not None
    graph_path = out / "scene_graph.json"
    graph.save_json(graph_path)

    current_pose = SE3.from_translation([7.0, 0.0, 0.0])
    question = "我身边有几个红杯子？在哪？"
    retriever = GraphRetriever(
        graph,
        reachable_radius_m=config.retrieval.reachable_radius_m,
        lateral_tolerance_ratio=config.retrieval.lateral_tolerance_ratio,
    )
    context = retriever.retrieve(question, current_pose, language="zh")
    answer = RuleBasedAnswerGenerator().answer(context)

    print(f"Graph saved to {graph_path}")
    print(f"Events: {len(events)}")
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print()
    print(GraphRetriever.render_summary(context))


# ---------------------------------------------------------------------------
# `directme query`
# ---------------------------------------------------------------------------


def cmd_query(args: argparse.Namespace) -> None:
    graph = SceneGraph.load_json(args.graph)
    current_pose = _pose_from_json_or_graph(args.current_pose_json, graph)
    retriever = GraphRetriever(
        graph,
        reachable_radius_m=args.reachable_radius_m,
        lateral_tolerance_ratio=args.lateral_tolerance_ratio,
    )
    context = retriever.retrieve(
        question=args.question,
        current_pose=current_pose,
        top_k=args.top_k,
        language=args.language,
    )
    if args.show_summary:
        print(GraphRetriever.render_summary(context))
    print(RuleBasedAnswerGenerator().answer(context))


# ---------------------------------------------------------------------------
# `directme eval`  ―  UCS-Bench 4-dimension evaluation
# ---------------------------------------------------------------------------


def _resolve_dataset(path_arg: str | None) -> Path:
    if path_arg:
        return Path(path_arg)
    here = Path(__file__).parent / "eval" / "sample_questions.jsonl"
    return here


def _ensure_demo_graph(graph_path: str | None, out_dir: Path) -> Path:
    """If the user did not supply --graph, build the toy graph on the fly.

    This keeps `directme eval` runnable as a single command without any prior
    setup. The user can always pass --graph to evaluate against a real graph.
    """
    if graph_path:
        return Path(graph_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    graph_file = out_dir / "scene_graph.json"
    if graph_file.exists():
        return graph_file
    config = DirectMeConfig()
    config.run_dir = str(out_dir)
    frames, backend = build_living_room_kitchen_demo(out_dir / "keyframes")
    engine = OfflineMappingEngine(backend=backend, config=config)
    engine.process_frames(frames, chunk_size=2)
    assert engine.graph is not None
    engine.graph.save_json(graph_file)
    return graph_file


def cmd_eval(args: argparse.Namespace) -> None:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = _resolve_dataset(args.dataset)
    graph_path = _ensure_demo_graph(args.graph, out_dir / "demo_graph")
    graph = SceneGraph.load_json(graph_path)

    questions = load_ucs_questions(dataset_path)
    pose_lookup = {}
    if args.current_pose_json:
        pose = _pose_from_json(args.current_pose_json)
        pose_lookup = {(q.video_uid, float(q.query_timestamp)): pose for q in questions}

    evaluator = UCSBenchEvaluator(
        graph=graph,
        pose_lookup=pose_lookup or None,
        reachable_radius_m=args.reachable_radius_m,
        lateral_tolerance_ratio=args.lateral_tolerance_ratio,
        top_k=args.top_k,
        language=args.language,
    )
    report = evaluator.run(questions)

    summary = report.render_summary()
    print(summary)

    if args.output_json:
        out_json = Path(args.output_json)
    else:
        out_json = out_dir / "eval_report.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nFull report written to {out_json}")


# ---------------------------------------------------------------------------
# `directme ingest`  ―  video / frame-stream → scene graph (v0.4)
# ---------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a video or a directory of frames into a scene graph.

    Two input modes (mutually exclusive):

    * ``--video PATH`` — decode and sample at ``--target-fps`` (default 1.0).
    * ``--frames-dir DIR`` — consume image files (sorted by name) at the
      configured chunk size.

    Each chunk runs through ``OfflineMappingEngine.process_chunk`` with
    chunk-level fault isolation (one bad chunk does not kill the whole run)
    and SQLite progress tracking (interrupted runs resume on restart when
    the SQLite store is selected).
    """
    import asyncio
    from directme.mapping.async_engine import ingest_frames_async
    from directme.perception.ingest import (
        iter_frames_from_paths,
        iter_frames_from_video,
    )

    if not args.video and not args.frames_dir:
        raise SystemExit("ingest requires --video PATH or --frames-dir DIR")
    if args.video and args.frames_dir:
        raise SystemExit("ingest: --video and --frames-dir are mutually exclusive")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = DirectMeConfig()
    config.run_dir = str(out_dir)
    config.storage.backend = args.storage_backend
    config.stream.chunk_size_frames = args.chunk_size

    # Pick a perception backend. ``toy`` keeps the core install tiny, while
    # ``composed`` wires the reference DA3 + YOLO-World + optional SAM 2 stack.
    if args.backend == "toy":
        print(
            "[ingest] using ToyPerceptionBackend — this produces a "
            "deterministic synthetic graph regardless of the input frames.",
            flush=True,
        )
        from directme.perception.toy import ToyPerceptionBackend
        backend = ToyPerceptionBackend(script={})
    elif args.backend == "composed":
        from directme.perception.runtime import build_composed_backend, resolve_runtime_device

        classes = [c.strip() for c in args.classes.split(",") if c.strip()]
        if not classes:
            raise SystemExit("--backend composed requires at least one class in --classes")
        resolved = resolve_runtime_device(args.device)
        print(f"[ingest] backend=composed device={resolved} depth={args.depth_model}", flush=True)
        if args.sam2_checkpoint and args.sam2_config:
            print(f"[ingest] SAM 2 enabled: {args.sam2_checkpoint}", flush=True)
        else:
            print("[ingest] SAM 2 disabled; using bbox-center depth fallback", flush=True)
        backend = build_composed_backend(
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
    else:
        raise SystemExit(f"Unknown backend: {args.backend!r}")

    engine = OfflineMappingEngine(backend=backend, config=config)

    # Build the frame iterator.
    if args.video:
        frame_dump_dir = out_dir / "keyframes" if args.dump_frames else None
        frames_iter = iter_frames_from_video(
            args.video,
            target_fps=args.target_fps,
            frame_dump_dir=frame_dump_dir,
            max_frames=args.max_frames,
        )
    else:
        frames_dir = Path(args.frames_dir)
        image_paths = sorted(
            p for p in frames_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        )
        if args.max_frames is not None:
            image_paths = image_paths[: args.max_frames]
        frames_iter = iter_frames_from_paths(image_paths, fps=args.target_fps)

    mapper = asyncio.run(
        ingest_frames_async(
            engine=engine,
            frames=frames_iter,
            chunk_size=args.chunk_size,
            queue_maxsize=args.queue_maxsize,
            swallow_chunk_failures=not args.fail_fast,
        )
    )

    print()
    print(f"[ingest] {mapper.stats}")
    if mapper.failed_chunks:
        print(f"[ingest] {len(mapper.failed_chunks)} chunk(s) failed:")
        for fc in mapper.failed_chunks[:10]:
            print(
                f"  - chunk {fc.chunk_id} (n={fc.n_frames}, "
                f"first_ts={fc.first_frame_timestamp:.2f}s): "
                f"{fc.error_type}: {fc.error_message}"
            )

    assert engine.graph is not None
    graph_json = out_dir / "scene_graph.json"
    engine.graph.save_json(graph_json)
    print(f"[ingest] graph saved to {graph_json}")
    drift = engine.graph.metadata.get("drift_telemetry", {})
    if drift.get("warnings"):
        print("[ingest] pose-drift warnings:")
        for w in drift["warnings"]:
            print(f"  - {w}")


# ---------------------------------------------------------------------------
# Argparse plumbing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DirectMe 2.0 CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # demo
    demo = sub.add_parser("demo", help="run deterministic living-room-to-kitchen demo")
    demo.add_argument("--out", default="runs/toy", help="output directory")
    demo.set_defaults(func=cmd_demo)

    # query
    query = sub.add_parser("query", help="query an existing scene graph")
    query.add_argument("--graph", required=True, help="path to scene_graph.json")
    query.add_argument("--question", required=True, help="question text")
    query.add_argument("--current-pose-json", default=None,
                       help="4x4 T_world_from_current_camera JSON matrix")
    query.add_argument("--language", default="zh", choices=["zh", "en"])
    query.add_argument("--top-k", type=int, default=8)
    query.add_argument("--reachable-radius-m", type=float, default=5.0,
                       help="distance (m) within which an object is considered reachable")
    query.add_argument("--lateral-tolerance-ratio", type=float, default=0.20,
                       help="|x| <= this * max(|z|, 1m) collapses to centered front/behind")
    query.add_argument("--show-summary", action="store_true")
    query.set_defaults(func=cmd_query)

    # eval
    ev = sub.add_parser(
        "eval",
        help="run UCS-Bench 4-dimension evaluation against a scene graph",
    )
    ev.add_argument("--dataset", default=None,
                    help="path to UCS-Bench JSON or JSONL "
                         "(default: directme/eval/sample_questions.jsonl)")
    ev.add_argument("--graph", default=None,
                    help="path to scene_graph.json "
                         "(default: build the toy demo graph automatically)")
    ev.add_argument("--out", default="runs/eval", help="output directory")
    ev.add_argument("--output-json", default=None,
                    help="explicit path for the JSON report; "
                         "defaults to <out>/eval_report.json")
    ev.add_argument("--current-pose-json", default=None,
                    help="optional 4x4 pose matrix used for every question; "
                         "for the toy graph use [[1,0,0,7],[0,1,0,0],[0,0,1,0],[0,0,0,1]]")
    ev.add_argument("--language", default="zh", choices=["zh", "en"])
    ev.add_argument("--top-k", type=int, default=8)
    ev.add_argument("--reachable-radius-m", type=float, default=5.0)
    ev.add_argument("--lateral-tolerance-ratio", type=float, default=0.20)
    ev.set_defaults(func=cmd_eval)

    # ingest (v0.4)
    ig = sub.add_parser(
        "ingest",
        help="ingest a video file or a directory of frames into a scene graph "
             "(async incremental, chunk-fault-isolated, resumable)",
    )
    src = ig.add_mutually_exclusive_group(required=True)
    src.add_argument("--video", default=None,
                     help="path to a video file (mp4 / mov / mkv / …)")
    src.add_argument("--frames-dir", default=None,
                     help="directory of pre-extracted frames "
                          "(jpg / png; sorted lexicographically)")
    ig.add_argument("--out", default="runs/ingest", help="output directory")
    ig.add_argument("--target-fps", type=float, default=1.0,
                    help="frame sampling rate; default 1.0 (DirectMe is "
                         "designed for 1-FPS egocentric capture)")
    ig.add_argument("--chunk-size", type=int, default=10,
                    help="frames per perception chunk (default 10)")
    ig.add_argument("--queue-maxsize", type=int, default=None,
                    help="frame queue cap (default 4 * chunk_size)")
    ig.add_argument("--max-frames", type=int, default=None,
                    help="hard cap on frames consumed (smoke tests)")
    ig.add_argument("--storage-backend", default="sqlite",
                    choices=["json", "sqlite"],
                    help="default sqlite — required for resume-on-restart")
    ig.add_argument("--backend", default="toy",
                    help="perception backend; only 'toy' is bundled. Real "
                         "backends (DA3 / SCAL3R / YOLO-World) are wired in "
                         "via the Python API; see docs/adapter_guide.md")
    ig.add_argument("--dump-frames", action="store_true",
                    help="when ingesting a video, also write each sampled "
                         "frame to <out>/keyframes/ so keyframes survive "
                         "the original video being moved or deleted")
    ig.add_argument("--fail-fast", action="store_true",
                    help="re-raise on the first chunk failure instead of "
                         "swallowing and continuing (useful for debugging)")
    ig.set_defaults(func=cmd_ingest)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
