#!/usr/bin/env python3
"""DirectMe QA retrieval and spatial relation generation demo.

Aligns UCS-Bench format QA JSON with the scene graph produced by the
mapping engine, resolves the ego pose at each QA item's
``question_timestamps``, retrieves scene graph nodes via GraphRetriever,
renders egocentric spatial relations, and finally calls the answer
generator to produce answers.

Mathematical basis of egocentric direction computation
-------------------------------------------------------
Let p_world be the world coordinate of a node in the scene graph, and
T_world_from_camera the camera pose at query time (stored in
ego_pose_timeline). Then:

    T_camera_from_world = T_world_from_camera^{-1}
    p_cam = T_camera_from_world · p_world

Camera coordinate convention (DirectMe): x=right, y=down, z=forward.
Direction classification:

    |x| ≤ α · max(|z|, 1)  →  pure front/behind (α = lateral_tolerance_ratio = 0.20)
    |z| ≤ δ (δ=0.30m)      →  pure left/right
    otherwise              →  8-way combos such as front_left / behind_right

Distance = ||p_cam||₂ (rigid transforms preserve Euclidean distance)
Reachability = distance ≤ reachable_radius_m

Usage
-----
# Rule-based generator (no GPU required)
python demo.py --graph-json scene_graph.json --qa-json qa.json --mode rule

# Qwen3-VL local inference (UCS-Bench multiple-choice evaluation)
python demo.py --graph-json scene_graph.json --qa-json qa.json \\
    --mode qwen --model-path /path/to/Qwen3-VL-8B-Instruct

# InternVL3 local inference
python demo.py --graph-json scene_graph.json --qa-json qa.json \\
    --mode internvl --model-path /path/to/InternVL3-8B

# Interactive
python demo.py --graph-json scene_graph.json --mode rule --interactive
"""

from __future__ import annotations
import os
import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _ts_sec(ts: str) -> float:
    """Convert an 'HH:MM:SS' or 'MM:SS' timestamp to seconds."""
    parts = [float(p) for p in ts.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _options_list(options: dict[str, str]) -> list[str]:
    return [options[k] for k in sorted(options.keys())]


def _options_labels(options: dict[str, str]) -> str:
    return "".join(sorted(options.keys()))


# ---------------------------------------------------------------------------
# Scene graph loading
# ---------------------------------------------------------------------------

def load_graph(graph_json: str | None, pipeline_summary: str | None):
    """Load a SceneGraph from a JSON file. Prefer graph_json, otherwise locate it from the summary."""
    from directme.mapping.scene_graph import SceneGraph

    def _from_path(p: Path):
        if not p.exists():
            raise FileNotFoundError(p)
        graph = SceneGraph.load_json(p)
        print(f"[graph] loaded {p}  nodes={len(graph.nodes)}  edges={len(graph.edges)}")
        return graph

    if graph_json:
        return _from_path(Path(graph_json))

    if pipeline_summary:
        summary = _load_json(pipeline_summary)
        run_dir = Path(summary.get("mapping", {}).get("run_dir", ""))
        for name in ("scene_graph.json", "graph.json"):
            for base in (run_dir, Path(pipeline_summary).parent):
                p = base / name
                if p.exists():
                    return _from_path(p)

    for p in (
        Path("directme_mapping_run/scene_graph.json"),
        Path("runs/default/scene_graph.json"),
        Path("scene_graph.json"),
    ):
        if p.exists():
            print(f"[graph] using default path {p}")
            return _from_path(p)

    raise FileNotFoundError(
        "Scene graph JSON not found. Please specify it via --graph-json or --pipeline-summary."
    )


# ---------------------------------------------------------------------------
# QA JSON loading (UCS-Bench format)
# ---------------------------------------------------------------------------

def load_qa(qa_json: str | Path) -> list[dict[str, Any]]:
    """Load a UCS-Bench format QA file and return the raw entry list.

    Required fields: question, question_timestamps, options, answer_label
    Optional fields: question_chinese, answer, answer_chinese, evidence, ...
    """
    items = _load_json(qa_json)
    if not isinstance(items, list):
        raise ValueError(f"{qa_json} should be a list, got {type(items)}")
    print(f"[qa]    loaded {qa_json}  {len(items)} QA items")
    return items


# ---------------------------------------------------------------------------
# DirectMe instance construction
# ---------------------------------------------------------------------------

def build_directme(
    mode: str,
    model_path: str,
    device_map: str,
    graph=None,
    config=None,
    max_new_tokens: int = 32,
    max_image_size: int = 512,
    internvl_image_size: int = 448,
):
    """Build a DirectMe instance and inject the pre-built SceneGraph.

    mode=rule     → RuleBasedAnswerGenerator (no GPU; MC falls back to Jaccard)
    mode=qwen     → QwenVLGenerator (Qwen3-VL local inference, MC + keyframe images)
    mode=internvl → InternVLGenerator (InternVL3 local inference, MC + keyframe images)
    """
    from directme.pipeline import DirectMe as _DirectMe

    if mode == "rule":
        dm = _DirectMe.with_empty_graph(config=config)
    elif mode == "qwen":
        dm = _DirectMe.with_qwen(
            model_path=model_path,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            max_image_size=max_image_size,
            config=config,
        )
    elif mode == "internvl":
        dm = _DirectMe.with_internvl(
            model_path=model_path,
            device_map=device_map,
            max_new_tokens=max_new_tokens,
            image_size=internvl_image_size,
            config=config,
        )
    else:
        raise ValueError(f"Unknown mode: {mode}; supported: rule / qwen / internvl")

    if graph is not None:
        dm.graph = graph
    return dm


# ---------------------------------------------------------------------------
# Build the graph from perception (video/frames dir → scene graph)
# ---------------------------------------------------------------------------

_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _natural_sort_key(path: Path):
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_image_paths(frames_dir: str | Path) -> list[Path]:
    frames_dir = Path(frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)
    paths = [p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES]
    if not paths:
        raise ValueError(f"No image frames found under {frames_dir}")
    return sorted(paths, key=_natural_sort_key)


def _dedupe_keep_order(names: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        name = str(name).strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
    return out


def _load_classes_from_yaml(path: str | Path, class_limit: int | None = None) -> list[str]:
    """Load YOLO/Objects365 style class names from a YAML file.

    Supported shapes:
      names: ["person", "cup", ...]
      names: {0: "person", 1: "cup", ...}
    """
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("Reading --classes-file requires PyYAML: pip install pyyaml") from exc

    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Classes YAML does not exist: {path}")

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    names = data.get("names", data) if isinstance(data, dict) else data

    if isinstance(names, dict):
        def _sort_key(k):
            try:
                return int(k)
            except Exception:
                return str(k)
        classes = [str(v).strip() for _, v in sorted(names.items(), key=lambda kv: _sort_key(kv[0]))]
    elif isinstance(names, list):
        classes = [str(v).strip() for v in names]
    else:
        raise ValueError(f"Unsupported classes YAML format in {path}")

    classes = [c for c in classes if c]
    if class_limit and class_limit > 0:
        classes = classes[:class_limit]
    return classes


def _load_classes_from_json(path: str | Path, class_limit: int | None = None) -> list[str]:
    """Load open-vocabulary classes from Objects365/COCO style JSON."""
    data = _load_json(path)
    names: list[str] = []

    def _extract(obj):
        if isinstance(obj, str):
            names.append(obj)
        elif isinstance(obj, dict):
            for key in ("name", "label", "category", "class"):
                if isinstance(obj.get(key), str):
                    names.append(obj[key])
                    return
        elif isinstance(obj, list):
            for item in obj:
                _extract(item)

    if isinstance(data, list):
        _extract(data)
    elif isinstance(data, dict):
        for key in ("categories", "classes", "names", "objects"):
            if key not in data:
                continue
            value = data[key]
            if isinstance(value, dict):
                for item in value.values():
                    _extract(item)
            else:
                _extract(value)

        # Also supports simple id→name dicts such as {"1": "person", "2": "cup"}.
        if not names:
            for value in data.values():
                _extract(value)
    else:
        raise ValueError(f"Unsupported --classes-json format: {type(data)}")

    names = [c for c in names if str(c).strip()]
    if class_limit and class_limit > 0:
        names = names[:class_limit]
    return names


def load_open_vocab_classes(
    classes: str | None,
    classes_json: str | None,
    classes_file: str | None,
    class_limit: int | None,
) -> list[str]:
    """Load YOLO-World open-vocabulary classes.

    Priority/sources:
    1. --classes-file: YAML; passing Object.yaml / Objects365 names directly is recommended;
    2. --classes-json: Objects365/COCO categories JSON;
    3. --classes: comma-separated extra classes;
    4. If none is given, fall back to a small smoke-test class list.
    """
    names: list[str] = []

    if classes_file:
        names.extend(_load_classes_from_yaml(classes_file, class_limit=class_limit))

    if classes_json:
        names.extend(_load_classes_from_json(classes_json, class_limit=class_limit))

    if classes:
        names.extend([c.strip() for c in classes.split(",") if c.strip()])

    if not names:
        names = [
            "person", "cup", "chair", "table", "bottle", "cell phone", "book",
            "backpack", "sink", "refrigerator", "door", "sofa", "bed", "laptop",
            "keyboard", "mouse",
        ]
        print("[WARN] No --classes-file / --classes-json / --classes specified; using a small fallback class list, only suitable for smoke tests.")

    return _dedupe_keep_order(names)


def _resolve_scal3r_result_dir(path: str | Path | None) -> Path | None:
    """Accept either an exact SCAL3R output dir or a parent containing one mat.txt."""
    if not path:
        return None
    
    # Check if string is empty, since sometimes argparse passes empty strings
    if str(path).strip() == "" or str(path).strip().lower() == "none":
        return None

    root = Path(path).resolve()
    if not root.exists():
        raise FileNotFoundError(f"SCAL3R result root does not exist: {root}")

    if (root / "mat.txt").exists():
        return root

    mats = sorted(root.rglob("mat.txt"))
    if len(mats) == 1:
        return mats[0].parent

    if len(mats) > 1:
        # A precomputed multi-chunk directory may look like root/chunk_000000/mat.txt;
        # return root directly and let Scal3RDepthPoseAdapter look it up by chunk_id.
        if all(m.parent.name.startswith("chunk_") for m in mats):
            return root
        msg = "\n".join(str(p) for p in mats[:20])
        raise RuntimeError(
            f"Multiple mat.txt files found under {root}. "
            f"Please pass the exact result directory, or a root containing chunk_*/mat.txt.\n"
            f"Candidates:\n{msg}"
        )

    raise FileNotFoundError(f"No mat.txt found under SCAL3R result root: {root}")


def build_perception_backend(args):
    """Build the perception backend.

    Important: the default backend=scal3r is constructed the same way as in
    the full SCAL3R + YOLO-World + SAM2 pipeline test:
      Scal3RDepthPoseAdapter + Scal3RRunner
      + YoloWorldDetector
      + optional Sam2MaskRefiner
      + SimpleIoUAppearanceTracker
      + OpenVocabularyTrackingAdapter
      + Scal3RComposedBackend
    """
    if args.backend == "toy":
        from directme.perception.toy import ToyPerceptionBackend
        print("[perception] using ToyPerceptionBackend (smoke tests only; does not read real visual content)")
        return ToyPerceptionBackend(script={})

    classes = load_open_vocab_classes(
        args.classes,
        args.classes_json,
        args.classes_file,
        args.class_limit,
    )
    if not classes:
        raise ValueError("--backend requires at least one open-vocabulary class via --classes-file / --classes-json / --classes")

    if args.backend in ("da3", "composed"):
        # Backward compatible DA3 path, kept for compatibility with the previous demo;
        # use the default --backend scal3r to stay aligned with full_pipeline_test.
        from directme.perception.runtime import build_composed_backend, resolve_runtime_device

        resolved = resolve_runtime_device(args.device)
        print(f"[perception] backend=da3 device={resolved} chunk_size={args.chunk_size}")
        print(f"[perception] classes={classes[:12]}{' ...' if len(classes) > 12 else ''}")
        if args.sam2_checkpoint and args.sam2_config:
            print(f"[perception] SAM2 enabled: {args.sam2_checkpoint}")
        else:
            print("[perception] SAM2 disabled; tracking images still save bbox + track_id, geometry uses the bbox-center depth fallback")

        return build_composed_backend(
            classes=classes,
            device=args.device,
            depth_model=args.depth_model,
            use_ray_pose=args.use_ray_pose,
            process_res=args.process_res,
            yolo_weights=args.yolo_weights,
            score_threshold=args.score_threshold,
            detection_stride=args.detection_stride,
            sam2_checkpoint=args.sam2_checkpoint,
            sam2_config=args.sam2_config,
            min_pose_confidence=args.min_pose_confidence,
        )

    if args.backend != "scal3r":
        raise ValueError(f"Unknown perception backend: {args.backend}")

    from directme.perception.adapters.open_vocab_tracking import (
        OpenVocabularyTrackingAdapter,
        Sam2MaskRefiner,
        SimpleIoUAppearanceTracker,
        YoloWorldDetector,
    )
    from directme.perception.adapters.scal3r import (
        Scal3RComposedBackend,
        Scal3RDepthPoseAdapter,
        Scal3RRunner,
    )

    precomputed_scal3r_root = getattr(args, 'precomputed_scal3r_root', None)
    precomputed_root = _resolve_scal3r_result_dir(precomputed_scal3r_root)
    scal3r_work_dir = (
        Path(args.scal3r_work_dir)
        if args.scal3r_work_dir
        else Path(args.work_dir) / "scal3r_work"
    )

    print(f"[perception] backend=scal3r device={args.device} chunk_size={args.chunk_size}")
    print(f"[perception] classes_count={len(classes)} preview={classes[:20]}")
    print(f"[perception] precomputed_scal3r_root={precomputed_root}")
    print(f"[perception] scal3r_work_dir={scal3r_work_dir}")

    depth_pose = Scal3RDepthPoseAdapter(
        runner=Scal3RRunner(
            config=args.scal3r_config,
            checkpoint=args.scal3r_checkpoint,
            device=args.device,
            save_dpt=1,
            save_xyz=0,
        ),
        precomputed_root=precomputed_root,
        work_dir=scal3r_work_dir,
        keep_work_dir=args.keep_scal3r_work_dir,
    )

    detector = YoloWorldDetector(
        weights=args.yolo_weights,
        classes=classes,
        score_threshold=args.score_threshold,
        device=args.device,
    )

    segmenter = None
    if args.use_sam2 and args.sam2_checkpoint and args.sam2_config:
        print(f"[perception] SAM2 enabled: {args.sam2_checkpoint}")
        segmenter = Sam2MaskRefiner(
            checkpoint=args.sam2_checkpoint,
            config=args.sam2_config,
            device=args.device,
        )
    else:
        print("[perception] SAM2 disabled; tracking images still save bbox + track_id, geometry uses the bbox fallback")

    tracker = OpenVocabularyTrackingAdapter(
        detector=detector,
        segmenter=segmenter,
        tracker=SimpleIoUAppearanceTracker(),
        detection_stride=args.detection_stride,
    )

    return Scal3RComposedBackend(
        depth_pose=depth_pose,
        tracker=tracker,
        min_pose_confidence=args.min_pose_confidence,
    )

def build_graph_from_perception(dm, args):
    """Run perception + offline incremental mapping starting from a video or frames directory."""
    if args.video and args.frames_dir:
        raise ValueError("Only one of --video and --frames-dir may be specified")
    if not args.video and not args.frames_dir:
        raise ValueError("Building the graph from perception requires --video or --frames-dir")

    backend = build_perception_backend(args)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir = Path(args.perception_artifact_dir) if args.perception_artifact_dir else Path(args.work_dir) / "perception_artifacts"

    print(
        f"[build] offline incremental mapping: target_fps={args.target_fps}, "
        f"chunk_size={args.chunk_size} frames, artifacts={artifact_dir}"
    )

    if args.video:
        frame_dump_dir = Path(args.frame_dump_dir) if args.frame_dump_dir else Path(args.work_dir) / "frames"
        graph = dm.build_memory_from_video(
            args.video,
            backend,
            target_fps=args.target_fps,
            chunk_size=args.chunk_size,
            frame_dump_dir=frame_dump_dir,
            artifact_dir=artifact_dir,
            max_frames=args.max_frames,
        )
    else:
        image_paths = list_image_paths(args.frames_dir)
        graph = dm.build_memory_from_frames(
            image_paths,
            backend,
            fps=args.target_fps,
            chunk_size=args.chunk_size,
            artifact_dir=artifact_dir,
            max_frames=args.max_frames,
        )

    graph_path = Path(args.run_dir) / "scene_graph.json"
    graph.save_json(graph_path)
    print(f"[build] scene graph saved: {graph_path.resolve()}")
    print(f"[build] perception depth/tracking frames and videos saved under: {artifact_dir.resolve()}")
    return graph


# ---------------------------------------------------------------------------
# Core query function
# ---------------------------------------------------------------------------

def run_query(
    dm,
    item: dict[str, Any],
    *,
    language: str | None = None,
    top_k: int = 16,
    reachable_radius_m: float = 10.0,
) -> dict[str, Any]:
    """Run the full retrieve-render-generate pipeline for one UCS-Bench QA item.

    Pipeline
    --------
    1. Parse question_timestamps → seconds
    2. Interpolate the nearest ego pose from ego_pose_timeline
    3. GraphRetriever fetches the top-k nodes; render_egocentric computes the 8-way direction
    4. Generate the answer:
       - with options → dm.answer_mc()
         * rule mode: RuleBasedAnswerGenerator + Jaccard fallback
         * qwen/internvl: MC prompt + keyframe PIL images → local VLM → letter label
       - without options → dm.answer() (free-form text)
    5. Return a result dict with the predicted label and a correctness flag
    """
    from directme.retrieval.pose_lookup import pose_from_graph_timeline
    from directme.retrieval.query_parser import parse_query
    from directme.retrieval.retriever import GraphRetriever

    dm.config.retrieval.top_k = top_k
    dm.config.retrieval.reachable_radius_m = reachable_radius_m

    # ── Language & question text ──────────────────────────────────────────────
    lang = language or ("zh" if re.search(r"[\u4e00-\u9fff]", item.get("question", "")) else "en")
    if lang == "zh" and item.get("question_chinese"):
        question = item["question_chinese"]
        ref_answer = item.get("answer_chinese", item.get("answer", ""))
    else:
        question = item["question"]
        ref_answer = item.get("answer", "")

    # ── Timestamp → ego pose ──────────────────────────────────────────────────
    ts_s = _ts_sec(item["question_timestamps"])
    current_pose = pose_from_graph_timeline(dm.graph, timestamp=ts_s)

    # ── Scene graph retrieval (populates the spatial_items display fields) ─────────
    intent = parse_query(question, language=lang)
    retriever = GraphRetriever(dm.graph, reachable_radius_m=reachable_radius_m)
    ctx = retriever.retrieve(
        question, current_pose,
        top_k=top_k,
        language=lang,
        as_of_timestamp=ts_s,                # ← key change: only observations recorded before ts_s are used
    )
    # ── Egocentric spatial relation assembly ──────────────────────────────────
    # render_egocentric is already invoked inside GraphRetriever.
    # relation   = classify_egocentric_relation(T_cam_from_world @ p_world)
    # distance_m = ||p_cam||₂ (rigid transforms preserve Euclidean distance)
    # reachable  = distance_m ≤ reachable_radius_m
    spatial_items = [
        {
            "node_id":          it.node.node_id,
            "label":            it.node.semantic_label,
            "color":            it.node.attributes.get("color"),
            "p_world":          [round(v, 3) for v in it.node.p_world.tolist()],
            "relation":         it.egocentric["relation"],
            "distance_m":       it.egocentric["distance_m"],
            "reachable":        it.egocentric["reachable"],
            "natural_language": it.egocentric["natural_language"],
            "keyframes":        it.node.keyframes,
        }
        for it in ctx.items
    ]

    # ── Answer generation ─────────────────────────────────────────────────────
    options: dict[str, str] = item.get("options", {})
    predicted_label: str | None = None
    raw_answer: str = ""

    if options:
        # with options → answer_mc(): the intended path is MC prompt + keyframe images → VLM
        labels_str = _options_labels(options)
        raw_answer, predicted_label = dm.answer_mc(
            question=question,
            options=_options_list(options),
            current_pose=current_pose,
            language=lang,
            option_labels=labels_str,
            qtype=item.get("qtype"),             # ← COUNT questions use deterministic counting
            as_of_timestamp=ts_s,                # ← consistent with retrieve above
        )
    else:
        raw_answer = dm.answer(
            question,
            current_pose,
            language=lang,
            as_of_timestamp=ts_s,
        )

    answer_label: str | None = item.get("answer_label")
    is_correct: bool | None = (
        predicted_label == answer_label if (predicted_label and answer_label) else None
    )

    return {
        "qid":             item.get("qid") or item.get("q_id"),
        "video_uid":       item.get("video_uid"),
        "question":        question,
        "language":        lang,
        "timestamp_s":     ts_s,
        "category":        item.get("category"),
        "subcategory":     item.get("subcategory"),
        "qtype":           item.get("qtype"),
        "task_difficulty": item.get("task_difficulty"),
        "intent": {
            "labels":             intent.labels,
            "colors":             intent.colors,
            "rooms":              intent.rooms,
            "wants_location":     intent.wants_location,
            "wants_count":        intent.wants_count,
            "wants_reachability": intent.wants_reachability,
            "wants_trajectory":   intent.wants_trajectory,
        },
        "matched_count":   ctx.count,
        "reachable_count": ctx.reachable_count,
        "spatial_items":   spatial_items,
        "raw_answer":       raw_answer,
        "predicted_label":  predicted_label,
        "answer_label":     answer_label,
        "is_correct":       is_correct,
        "reference_answer": ref_answer,
    }


# ---------------------------------------------------------------------------
# Batch evaluation
# ---------------------------------------------------------------------------

def run_batch(
    dm,
    qa_items: list[dict],
    *,
    language: str | None,
    top_k: int,
    reachable_radius_m: float,
    verbose: bool,
) -> list[dict[str, Any]]:
    from collections import defaultdict

    results: list[dict] = []
    n = len(qa_items)

    for i, item in enumerate(qa_items):
        qid = item.get("qid") or item.get("q_id") or f"#{i+1}"
        try:
            result = run_query(
                dm, item,
                language=language,
                top_k=top_k,
                reachable_radius_m=reachable_radius_m,
            )
            results.append(result)
            _print_result(result, verbose=verbose)
            print(f"  [{i+1}/{n}] {qid}  "
                  f"correct={result['is_correct']}  "
                  f"predicted={result['predicted_label']}  "
                  f"ground_truth={result['answer_label']}")
        except Exception as exc:
            print(f"  [{i+1}/{n}] {qid}  ERROR: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()

    judged = [r for r in results if r["is_correct"] is not None]
    if judged:
        acc = sum(r["is_correct"] for r in judged) / len(judged)
        print(f"\n[stats] {len(results)} total  {len(judged)} judged  accuracy {acc:.1%}")

        cat_stats: dict[str, list[bool]] = defaultdict(list)
        for r in judged:
            if r.get("category"):
                cat_stats[r["category"]].append(r["is_correct"])
        if cat_stats:
            print("[stats] accuracy by category:")
            for cat, bools in sorted(cat_stats.items()):
                print(f"  {cat:45s}  {sum(bools)}/{len(bools)} = {sum(bools)/len(bools):.1%}")

    return results


def _print_result(result: dict, verbose: bool = False) -> None:
    print(f"\n{'─'*60}")
    print(f"Question [{result['language']}]: {result['question']}")
    print(f"Timestamp: {result['timestamp_s']:.1f}s  |  "
        f"{result['matched_count']} candidate graph nodes, "
        f"{result['reachable_count']} of them reachable")


    if result["spatial_items"]:
        print("Spatial relations:")
        for it in result["spatial_items"]:
            color = (f"[{it['color']}] "
                     if it.get("color") and it["color"] not in ("unknown", None, "")
                     else "")
            reach = "✔" if it["reachable"] else "✘"
            print(f"  {it['node_id']:12s}  {color}{it['label']:16s}  "
                  f"{it['natural_language']}  {reach}")

    print(f"Answer: {result['raw_answer'][:200]}")
    if verbose and result.get("reference_answer"):
        print(f"Reference: {result['reference_answer']}")


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def run_interactive(dm, *, top_k: int, reachable_radius_m: float) -> None:
    print("\nInteractive mode (type quit to exit). The pose is fixed to the last frame of the timeline.")
    while True:
        try:
            question = input("\n> Question: ").strip()
        except (KeyboardInterrupt, EOFError):
            break
        if not question or question.lower() in ("quit", "exit", "q"):
            break
        dummy_item = {"question": question, "question_timestamps": "9999:00", "options": {}}
        try:
            result = run_query(
                dm, dummy_item, language=None,
                top_k=top_k, reachable_radius_m=reachable_radius_m,
            )
            _print_result(result, verbose=True)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description="DirectMe QA Demo",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    src = p.add_argument_group("Input: existing scene graph, or rebuild it from perception")
    src.add_argument(
        "--input-mode",
        choices=["auto", "from_graph", "from_perception"],
        default="auto",
        help=(
            "Input mode: "
            "from_graph=reason over an existing scene_graph.json; "
            "from_perception=run perception+mapping from a video/frames directory; "
            "auto=decide automatically from whether --video/--frames-dir is given"
        ),
    )
    src.add_argument("--graph-json", default=None, help="Existing scene_graph.json; used when --video/--frames-dir is not given")
    src.add_argument("--pipeline-summary", default=None, help="Existing full_pipeline_summary.json; used to locate scene_graph.json")
    src.add_argument("--video", default="/data/ywang/dataset/SpatialMemory/scene0715_00-0.mp4", help="Start from a raw video: sample frames → perception → scene graph → QA")
    src.add_argument("--frames-dir", default=None,
                     help="Start from an extracted-frames directory: perception → scene graph → QA; default matches full_pipeline_test's --image-dir")
    src.add_argument("--work-dir", default="/data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline",
                     help="Top-level working directory when building the graph from perception; default matches full_pipeline_test's --work-dir")
    src.add_argument("--run-dir", default=None,
                     help="DirectMe mapping output directory; defaults to <work-dir>/directme_mapping_run, matching full_pipeline_test")
    src.add_argument("--frame-dump-dir", default=None, help="Directory where sampled video frames are saved; defaults to <work-dir>/frames")
    src.add_argument("--target-fps", type=float, default=5.0, help="Video sampling frame rate; the paper default is 1 FPS")
    src.add_argument("--chunk-size", type=int, default=60, help="Frames per offline incremental perception/mapping chunk; default 30, aligned with full_pipeline_test's --max-images=30")
    src.add_argument("--max-frames", type=int, default=300, help="Only process the first N sampled frames; default matches full_pipeline_test's --max-images")
    src.add_argument("--perception-artifact-dir", default=None, help="Output directory for depth maps, detection tracking images and videos; defaults to <run-dir>/perception_artifacts")

    p.add_argument("--qa-json", default="/data/ywang/dataset/SpatialMemory/MC_QAs_v15_norm/scene0715_00-0-mcq.json",
                   help="UCS-Bench QA JSON; default aligned with full_pipeline_test's scene0715_00-0 input scene; pass an empty string to only build the graph")

    gen = p.add_argument_group("Answer generation")
    gen.add_argument("--mode", choices=["rule", "qwen", "internvl"], default="qwen",
                     help="rule=rule-based  qwen=Qwen3-VL  internvl=InternVL3")
    gen.add_argument(
        "--model-path",
        default="/data/ywang/my_projects/VideoUnderstanding/Qwen3-VL-8B-Instruct",
        help="Local model path (used when mode=qwen or mode=internvl)",
    )
    gen.add_argument("--device-map",   default="auto",
                     help="PyTorch device_map, e.g. auto / cuda:0 / cuda:2")
    gen.add_argument("--max-new-tokens", type=int, default=32,
                     help="Maximum number of tokens the VLM may generate (32 is enough for multiple choice)")
    gen.add_argument("--max-image-size", type=int, default=512,
                     help="Upper bound on the longest side of Qwen3-VL keyframes (pixels)")
    gen.add_argument("--internvl-image-size", type=int, default=448,
                     help="InternVL3 preprocessing resolution (pixels)")

    per = p.add_argument_group("Perception backend")
    per.add_argument("--backend", choices=["scal3r", "da3", "composed", "toy"], default="scal3r",
                     help="scal3r=SCAL3R+YOLO-World(+SAM2), consistent with full_pipeline_test; da3/composed=legacy DA3 path; toy=quick testing")
    per.add_argument(
        "--classes-file",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/directme/perception/adapters/Object.yaml",
        help="YOLO-World open-vocabulary YAML; default matches full_pipeline_test's --classes-file",
    )
    per.add_argument("--classes-json", default=None,
                     help="YOLO-World open-vocabulary JSON; supports the Objects365/COCO categories format")
    per.add_argument("--class-limit", type=int, default=300,
                     help="Maximum number of classes read from --classes-file / --classes-json; default matches full_pipeline_test's --class-limit; <=0 means no limit")
    per.add_argument("--classes", default=None,
                     help="Extra open-vocabulary classes, comma-separated; merged with file classes and deduplicated")
    per.add_argument("--device", type=str, default="cuda",
                     help="Device to run perception on; default matches full_pipeline_test's --device")
    per.add_argument(
        "--yolo-weights",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/yolo/yolov8m-worldv2.pt",
        help="YOLO-World weights; default matches full_pipeline_test's --yolo-weights",
    )
    per.add_argument("--score-threshold", type=float, default=0.15,
                     help="Default matches full_pipeline_test's --score-threshold")
    per.add_argument("--detection-stride", type=int, default=1,
                     help="Default matches full_pipeline_test's --detection-stride")

    # Consistent with full_pipeline_test: store_true with default=True, so SAM2 is enabled by default.
    per.add_argument("--use-sam2", action="store_true", default=True)
    per.add_argument(
        "--sam2-checkpoint",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/sam2/sam2.1_hiera_base_plus.pt",
    )
    per.add_argument(
        "--sam2-config",
        type=str,
        default="configs/sam2.1/sam2.1_hiera_b+.yaml",
    )
    per.add_argument("--min-pose-confidence", type=float, default=0.30)

    per.add_argument(
        "--precomputed-scal3r-root",
        type=str,
        default="",
        help="Existing SCAL3R result dir containing mat.txt/intri.yml/depths. Can also be a parent dir if it contains exactly one mat.txt.",
    )
    per.add_argument(
        "--scal3r-config",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/configs/scal3r/scal3r.yaml",
    )
    per.add_argument(
        "--scal3r-checkpoint",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/scal3r/scal3r.pt",
    )
    per.add_argument("--scal3r-work-dir", default=None,
                     help="SCAL3R scratch/output working directory; defaults to <work-dir>/scal3r_work, matching full_pipeline_test")
    per.add_argument("--keep-scal3r-work-dir", action="store_true", default=True,
                     help="True by default, matching full_pipeline_test's keep_work_dir=True")

    # Only used by the legacy da3/composed path.
    per.add_argument("--depth-model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    per.add_argument("--use-ray-pose", action="store_true", help="Only used by the DA3 path")
    per.add_argument("--process-res", type=int, default=504, help="Only used by the DA3 path")

    print("per.video", per.video)

    ret = p.add_argument_group("Retrieval")
    ret.add_argument("--top-k",            type=int,   default=16)
    ret.add_argument("--reachable-radius", type=float, default=10.0, metavar="M")

    p.add_argument("--language",    choices=["zh", "en"], default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--output-json", default="demo_qa_results.json")
    p.add_argument("--verbose",     action="store_true")

    args = p.parse_args()

    # Consistent with full_pipeline_test:
    # work_dir is the top-level directory; DirectMe mapping output goes to <work_dir>/directme_mapping_run.
    if not args.run_dir:
        args.run_dir = str(Path(args.work_dir) / "directme_mapping_run")

    # Allow passing --qa-json "" on the command line to only build the graph without running QA.
    if args.qa_json == "":
        args.qa_json = None

    from directme.config import DirectMeConfig

    config = DirectMeConfig()
    config.run_dir = args.run_dir
    config.stream.fps = args.target_fps
    config.stream.chunk_size_frames = args.chunk_size

    if args.input_mode == "auto":
        input_mode = "from_perception" if (args.video or args.frames_dir) else "from_graph"
    else:
        input_mode = args.input_mode

    graph = None

    if input_mode == "from_graph":
        if args.video or args.frames_dir:
            print(
                "[ERROR] input-mode=from_graph cannot be combined with --video or --frames-dir.",
                file=sys.stderr,
            )
            return 1

        try:
            graph = load_graph(args.graph_json or None, args.pipeline_summary or None)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 1

    elif input_mode == "from_perception":
        if not args.video and not args.frames_dir:
            print(
                "[ERROR] input-mode=from_perception requires --video or --frames-dir.",
                file=sys.stderr,
            )
            return 1

        if args.video and args.frames_dir:
            print("[ERROR] Only one of --video and --frames-dir may be specified.", file=sys.stderr)
            return 1

        if args.graph_json or args.pipeline_summary:
            print(
                "[WARN] input-mode=from_perception rebuilds the graph; "
                "--graph-json / --pipeline-summary will be ignored."
            )

    else:
        print(f"[ERROR] Unknown input_mode: {input_mode}", file=sys.stderr)
        return 1

    try:
        dm = build_directme(
            mode=args.mode,
            model_path=args.model_path,
            device_map=args.device_map,
            graph=graph,
            config=config,
            max_new_tokens=args.max_new_tokens,
            max_image_size=args.max_image_size,
            internvl_image_size=args.internvl_image_size,
        )
        if input_mode == "from_perception":
            graph = build_graph_from_perception(dm, args)
    except (ImportError, ValueError, FileNotFoundError) as e:
        print(f"[ERROR] {e}", file=sys.stderr); return 1

    if args.interactive:
        run_interactive(dm, top_k=args.top_k, reachable_radius_m=args.reachable_radius)
        return 0

    if not args.qa_json:
        if input_mode == "from_perception":
            print("[done] No --qa-json specified: perception→scene_graph mapping finished, skipping QA.")
        else:
            print("[done] No --qa-json specified: existing scene graph loaded, skipping QA.")
        return 0

    try:
        qa_items = load_qa(args.qa_json)
    except (FileNotFoundError, ValueError) as e:
        print(f"[ERROR] {e}", file=sys.stderr); return 1

    results = run_batch(
        dm, qa_items,
        language=args.language,
        top_k=args.top_k,
        reachable_radius_m=args.reachable_radius,
        verbose=args.verbose,
    )

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n[save] {out.resolve()}  ({len(results)} items)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
