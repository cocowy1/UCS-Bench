#!/usr/bin/env python3
"""DirectMe QA 检索与空间关系生成 Demo。

将 UCS-Bench 格式的 QA JSON 与建图引擎输出的场景图对齐，
在每条 QA 的 ``question_timestamps`` 时刻解析 ego 位姿，
通过 GraphRetriever 检索场景图节点并渲染自我中心空间关系，
最终调用答案生成器输出答案。

自我中心方向计算的数学基础
---------------------------
设场景图中某节点的世界坐标为 p_world，
查询时刻的相机位姿为 T_world_from_camera（ego_pose_timeline 中存储），
则：

    T_camera_from_world = T_world_from_camera^{-1}
    p_cam = T_camera_from_world · p_world

相机坐标系约定（DirectMe）：x=右，y=下，z=前。
方位分类：

    |x| ≤ α · max(|z|, 1)  →  纯前/后（α = lateral_tolerance_ratio = 0.20）
    |z| ≤ δ（δ=0.30m）     →  纯左/右
    其他                   →  front_left / behind_right 等 8 向组合

距离 = ||p_cam||₂（刚性变换不改变欧氏距离）
可达性 = 距离 ≤ reachable_radius_m

用法
----
# 规则生成器（无需 GPU）
python demo.py --graph-json scene_graph.json --qa-json qa.json --mode rule

# Qwen3-VL 本地推理（UCS-Bench 多选评测）
python demo.py --graph-json scene_graph.json --qa-json qa.json \\
    --mode qwen --model-path /path/to/Qwen3-VL-8B-Instruct

# InternVL3 本地推理
python demo.py --graph-json scene_graph.json --qa-json qa.json \\
    --mode internvl --model-path /path/to/InternVL3-8B

# 交互式
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
# 工具函数
# ---------------------------------------------------------------------------

def _ts_sec(ts: str) -> float:
    """将 'HH:MM:SS' 或 'MM:SS' 格式时间戳转换为秒数。"""
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
# 场景图加载
# ---------------------------------------------------------------------------

def load_graph(graph_json: str | None, pipeline_summary: str | None):
    """从 JSON 文件加载 SceneGraph。优先使用 graph_json，其次从 summary 定位。"""
    from directme.mapping.scene_graph import SceneGraph

    def _from_path(p: Path):
        if not p.exists():
            raise FileNotFoundError(p)
        graph = SceneGraph.load_json(p)
        print(f"[graph] 加载 {p}  节点={len(graph.nodes)}  边={len(graph.edges)}")
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
            print(f"[graph] 使用默认路径 {p}")
            return _from_path(p)

    raise FileNotFoundError(
        "未找到场景图 JSON，请通过 --graph-json 或 --pipeline-summary 指定。"
    )


# ---------------------------------------------------------------------------
# QA JSON 加载（UCS-Bench 格式）
# ---------------------------------------------------------------------------

def load_qa(qa_json: str | Path) -> list[dict[str, Any]]:
    """加载 UCS-Bench 格式 QA 文件，返回原始条目列表。

    必选字段：question, question_timestamps, options, answer_label
    可选字段：question_chinese, answer, answer_chinese, evidence, ...
    """
    items = _load_json(qa_json)
    if not isinstance(items, list):
        raise ValueError(f"{qa_json} 应为 list，实际为 {type(items)}")
    print(f"[qa]    加载 {qa_json}  共 {len(items)} 条 QA")
    return items


# ---------------------------------------------------------------------------
# DirectMe 实例构建
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
    """构建 DirectMe 实例并注入已建好的 SceneGraph。

    mode=rule     → RuleBasedAnswerGenerator（无 GPU，MC 用 Jaccard 降级）
    mode=qwen     → QwenVLGenerator（Qwen3-VL 本地推理，MC + 关键帧图像）
    mode=internvl → InternVLGenerator（InternVL3 本地推理，MC + 关键帧图像）
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
        raise ValueError(f"未知 mode：{mode}，支持 rule / qwen / internvl")

    if graph is not None:
        dm.graph = graph
    return dm


# ---------------------------------------------------------------------------
# 从 perception 开始建图（视频/帧目录 → 场景图）
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
        raise ValueError(f"未在 {frames_dir} 下找到图像帧")
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
        raise ImportError("读取 --classes-file 需要 PyYAML：pip install pyyaml") from exc

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

        # 兼容 {"1": "person", "2": "cup"} 这类简单 id→name 字典。
        if not names:
            for value in data.values():
                _extract(value)
    else:
        raise ValueError(f"--classes-json 格式不支持：{type(data)}")

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
    """加载 YOLO-World 开放词表类别。

    优先级/来源：
    1. --classes-file：YAML，推荐直接传 Object.yaml / Objects365 names；
    2. --classes-json：Objects365/COCO categories JSON；
    3. --classes：逗号分隔额外类别；
    4. 都不传时，仅使用小规模 smoke-test fallback。
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
        print("[WARN] 未指定 --classes-file / --classes-json / --classes，使用小规模 fallback 类别，仅适合 smoke test。")

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
        # 多 chunk 预计算目录允许形如 root/chunk_000000/mat.txt；
        # 直接返回 root，由 Scal3RDepthPoseAdapter 按 chunk_id 查找。
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
    """构建 perception backend。

    重要：默认 backend=scal3r，构造方式与 full SCAL3R + YOLO-World + SAM2
    pipeline test 保持一致：
      Scal3RDepthPoseAdapter + Scal3RRunner
      + YoloWorldDetector
      + optional Sam2MaskRefiner
      + SimpleIoUAppearanceTracker
      + OpenVocabularyTrackingAdapter
      + Scal3RComposedBackend
    """
    if args.backend == "toy":
        from directme.perception.toy import ToyPerceptionBackend
        print("[perception] 使用 ToyPerceptionBackend（仅用于 smoke test，不读取真实视觉内容）")
        return ToyPerceptionBackend(script={})

    classes = load_open_vocab_classes(
        args.classes,
        args.classes_json,
        args.classes_file,
        args.class_limit,
    )
    if not classes:
        raise ValueError("--backend 需要通过 --classes-file / --classes-json / --classes 指定至少一个开放词表类别")

    if args.backend in ("da3", "composed"):
        # Backward compatible DA3 path. 保留是为了兼容上一版 demo；
        # 正式与 full_pipeline_test 对齐时请使用默认的 --backend scal3r。
        from directme.perception.runtime import build_composed_backend, resolve_runtime_device

        resolved = resolve_runtime_device(args.device)
        print(f"[perception] backend=da3 device={resolved} chunk_size={args.chunk_size}")
        print(f"[perception] classes={classes[:12]}{' ...' if len(classes) > 12 else ''}")
        if args.sam2_checkpoint and args.sam2_config:
            print(f"[perception] SAM2 enabled: {args.sam2_checkpoint}")
        else:
            print("[perception] SAM2 disabled；tracking 图仍保存 bbox + track_id，几何用 bbox-center depth fallback")

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
        raise ValueError(f"未知 perception backend: {args.backend}")

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
        print("[perception] SAM2 disabled；tracking 图仍保存 bbox + track_id，几何用 bbox fallback")

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
    """从视频或帧目录开始运行 perception + offline incremental mapping。"""
    if args.video and args.frames_dir:
        raise ValueError("--video 与 --frames-dir 只能指定一个")
    if not args.video and not args.frames_dir:
        raise ValueError("从 perception 建图需要 --video 或 --frames-dir")

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
# 核心查询函数
# ---------------------------------------------------------------------------

def run_query(
    dm,
    item: dict[str, Any],
    *,
    language: str | None = None,
    top_k: int = 16,
    reachable_radius_m: float = 10.0,
) -> dict[str, Any]:
    """对单条 UCS-Bench QA 条目执行完整的检索–渲染–生成流程。

    流程
    ----
    1. 解析 question_timestamps → 秒数
    2. 从 ego_pose_timeline 插值最近 ego 位姿
    3. GraphRetriever 检索 top-k 节点，render_egocentric 计算 8 向方位
    4. 生成答案：
       - 有选项 → dm.answer_mc()
         * rule 模式：RuleBasedAnswerGenerator + Jaccard 降级
         * qwen/internvl：MC 提示 + keyframe PIL 图像 → 本地 VLM → 字母标签
       - 无选项 → dm.answer()（自由文本）
    5. 返回含预测标签和正误标志的结果字典
    """
    from directme.retrieval.pose_lookup import pose_from_graph_timeline
    from directme.retrieval.query_parser import parse_query
    from directme.retrieval.retriever import GraphRetriever

    dm.config.retrieval.top_k = top_k
    dm.config.retrieval.reachable_radius_m = reachable_radius_m

    # ── 语言 & 问题文本 ───────────────────────────────────────────────────────
    lang = language or ("zh" if re.search(r"[\u4e00-\u9fff]", item.get("question", "")) else "en")
    if lang == "zh" and item.get("question_chinese"):
        question = item["question_chinese"]
        ref_answer = item.get("answer_chinese", item.get("answer", ""))
    else:
        question = item["question"]
        ref_answer = item.get("answer", "")

    # ── 时间戳 → ego 位姿 ─────────────────────────────────────────────────────
    ts_s = _ts_sec(item["question_timestamps"])
    current_pose = pose_from_graph_timeline(dm.graph, timestamp=ts_s)

    # ── 场景图检索（用于 spatial_items 展示字段）──────────────────────────────
    intent = parse_query(question, language=lang)
    retriever = GraphRetriever(dm.graph, reachable_radius_m=reachable_radius_m)
    ctx = retriever.retrieve(
        question, current_pose,
        top_k=top_k,
        language=lang,
        as_of_timestamp=ts_s,                # ← 关键修改：只用 ts_s 之前的观测
    )
    # ── 自我中心空间关系整理 ──────────────────────────────────────────────────
    # render_egocentric 已由 GraphRetriever 内部调用。
    # relation   = classify_egocentric_relation(T_cam_from_world @ p_world)
    # distance_m = ||p_cam||₂（刚性变换保持欧氏距离）
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

    # ── 答案生成 ──────────────────────────────────────────────────────────────
    options: dict[str, str] = item.get("options", {})
    predicted_label: str | None = None
    raw_answer: str = ""

    if options:
        # 有选项 → answer_mc()：正确路径为 MC 提示 + keyframe 图像 → VLM
        labels_str = _options_labels(options)
        raw_answer, predicted_label = dm.answer_mc(
            question=question,
            options=_options_list(options),
            current_pose=current_pose,
            language=lang,
            option_labels=labels_str,
            qtype=item.get("qtype"),             # ← COUNT 题走确定性计数
            as_of_timestamp=ts_s,                # ← 与上面 retrieve 保持一致
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
# 批量评测
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
                  f"正确={result['is_correct']}  "
                  f"预测={result['predicted_label']}  "
                  f"标准={result['answer_label']}")
        except Exception as exc:
            print(f"  [{i+1}/{n}] {qid}  ERROR: {exc}", file=sys.stderr)
            import traceback; traceback.print_exc()

    judged = [r for r in results if r["is_correct"] is not None]
    if judged:
        acc = sum(r["is_correct"] for r in judged) / len(judged)
        print(f"\n[统计] 共 {len(results)} 条  已评判 {len(judged)} 条  准确率 {acc:.1%}")

        cat_stats: dict[str, list[bool]] = defaultdict(list)
        for r in judged:
            if r.get("category"):
                cat_stats[r["category"]].append(r["is_correct"])
        if cat_stats:
            print("[统计] 分类准确率：")
            for cat, bools in sorted(cat_stats.items()):
                print(f"  {cat:45s}  {sum(bools)}/{len(bools)} = {sum(bools)/len(bools):.1%}")

    return results


def _print_result(result: dict, verbose: bool = False) -> None:
    print(f"\n{'─'*60}")
    print(f"问题 [{result['language']}]：{result['question']}")
    print(f"时间戳：{result['timestamp_s']:.1f}s  |  "
        f"候选图节点 {result['matched_count']} 个，"
        f"其中 {result['reachable_count']} 个可达")


    if result["spatial_items"]:
        print("空间关系：")
        for it in result["spatial_items"]:
            color = (f"[{it['color']}] "
                     if it.get("color") and it["color"] not in ("unknown", None, "")
                     else "")
            reach = "✔" if it["reachable"] else "✘"
            print(f"  {it['node_id']:12s}  {color}{it['label']:16s}  "
                  f"{it['natural_language']}  {reach}")

    print(f"答案：{result['raw_answer'][:200]}")
    if verbose and result.get("reference_answer"):
        print(f"参考：{result['reference_answer']}")


# ---------------------------------------------------------------------------
# 交互式 REPL
# ---------------------------------------------------------------------------

def run_interactive(dm, *, top_k: int, reachable_radius_m: float) -> None:
    print("\n交互式模式（输入 quit 退出）。位姿固定为 timeline 末尾帧。")
    while True:
        try:
            question = input("\n> 问题：").strip()
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

    src = p.add_argument_group("输入：已有场景图或从 perception 重新建图")
    src.add_argument(
        "--input-mode",
        choices=["auto", "from_graph", "from_perception"],
        default="auto",
        help=(
            "输入模式："
            "from_graph=基于已有 scene_graph.json 推理；"
            "from_perception=从视频/帧目录开始运行 perception+mapping；"
            "auto=根据是否传入 --video/--frames-dir 自动判断"
        ),
    )
    src.add_argument("--graph-json", default=None, help="已有 scene_graph.json；未指定 --video/--frames-dir 时使用")
    src.add_argument("--pipeline-summary", default=None, help="已有 full_pipeline_summary.json；用于定位 scene_graph.json")
    src.add_argument("--video", default="/data/ywang/dataset/SpatialMemory/scene0715_00-0.mp4", help="从原始视频开始：采样帧 → perception → scene graph → QA")
    src.add_argument("--frames-dir", default=None,
                     help="从已抽帧目录开始：perception → scene graph → QA；默认与 full_pipeline_test 的 --image-dir 保持一致")
    src.add_argument("--work-dir", default="/data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline",
                     help="从 perception 建图时的总工作目录；默认与 full_pipeline_test 的 --work-dir 保持一致")
    src.add_argument("--run-dir", default=None,
                     help="DirectMe mapping 输出目录；默认 <work-dir>/directme_mapping_run，与 full_pipeline_test 保持一致")
    src.add_argument("--frame-dump-dir", default=None, help="视频抽帧保存目录；默认 <work-dir>/frames")
    src.add_argument("--target-fps", type=float, default=5.0, help="视频采样帧率；论文默认 1 FPS")
    src.add_argument("--chunk-size", type=int, default=60, help="perception/mapping 每个离线增量 chunk 的帧数；默认 30，与 full_pipeline_test 的 --max-images=30 对齐")
    src.add_argument("--max-frames", type=int, default=300, help="仅处理前 N 个采样帧；默认与 full_pipeline_test 的 --max-images 保持一致")
    src.add_argument("--perception-artifact-dir", default=None, help="深度图、检测 tracking 图和视频输出目录；默认 <run-dir>/perception_artifacts")

    p.add_argument("--qa-json", default="/data/ywang/dataset/SpatialMemory/MC_QAs_v15_norm/scene0715_00-0-mcq.json",
                   help="UCS-Bench QA JSON；默认与 full_pipeline_test 的 scene0715_00-0 输入场景对齐；若只想建图可传空字符串")

    gen = p.add_argument_group("答案生成")
    gen.add_argument("--mode", choices=["rule", "qwen", "internvl"], default="qwen",
                     help="rule=规则  qwen=Qwen3-VL  internvl=InternVL3")
    gen.add_argument(
        "--model-path",
        default="/data/ywang/my_projects/VideoUnderstanding/Qwen3-VL-8B-Instruct",
        help="本地模型路径（mode=qwen 或 mode=internvl 时使用）",
    )
    gen.add_argument("--device-map",   default="auto",
                     help="PyTorch device_map，如 auto / cuda:0 / cuda:2")
    gen.add_argument("--max-new-tokens", type=int, default=32,
                     help="VLM 最多生成 token 数（多选题 32 足够）")
    gen.add_argument("--max-image-size", type=int, default=512,
                     help="Qwen3-VL 关键帧最长边缩放上限（像素）")
    gen.add_argument("--internvl-image-size", type=int, default=448,
                     help="InternVL3 预处理分辨率（像素）")

    per = p.add_argument_group("Perception backend")
    per.add_argument("--backend", choices=["scal3r", "da3", "composed", "toy"], default="scal3r",
                     help="scal3r=SCAL3R+YOLO-World(+SAM2)，与 full_pipeline_test 保持一致；da3/composed=旧 DA3 路径；toy=快速测试")
    per.add_argument(
        "--classes-file",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/directme/perception/adapters/Object.yaml",
        help="YOLO-World 开放词表 YAML；默认与 full_pipeline_test 的 --classes-file 保持一致",
    )
    per.add_argument("--classes-json", default=None,
                     help="YOLO-World 开放词表 JSON，支持 Objects365/COCO categories 格式")
    per.add_argument("--class-limit", type=int, default=300,
                     help="从 --classes-file / --classes-json 读取的类别上限；默认与 full_pipeline_test 的 --class-limit 保持一致；<=0 表示不限制")
    per.add_argument("--classes", default=None,
                     help="额外开放词表类别，逗号分隔；会与文件类别合并并去重")
    per.add_argument("--device", type=str, default="cuda",
                     help="perception 运行设备；默认与 full_pipeline_test 的 --device 保持一致")
    per.add_argument(
        "--yolo-weights",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/yolo/yolov8m-worldv2.pt",
        help="YOLO-World 权重；默认与 full_pipeline_test 的 --yolo-weights 保持一致",
    )
    per.add_argument("--score-threshold", type=float, default=0.15,
                     help="默认与 full_pipeline_test 的 --score-threshold 保持一致")
    per.add_argument("--detection-stride", type=int, default=1,
                     help="默认与 full_pipeline_test 的 --detection-stride 保持一致")

    # 与 full_pipeline_test 保持一致：store_true 且 default=True，默认启用 SAM2。
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
                     help="SCAL3R 临时/输出工作目录；默认 <work-dir>/scal3r_work，与 full_pipeline_test 保持一致")
    per.add_argument("--keep-scal3r-work-dir", action="store_true", default=True,
                     help="默认 True，与 full_pipeline_test 的 keep_work_dir=True 保持一致")

    # 仅 da3/composed 旧路径使用。
    per.add_argument("--depth-model", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    per.add_argument("--use-ray-pose", action="store_true", help="仅 DA3 路径使用")
    per.add_argument("--process-res", type=int, default=504, help="仅 DA3 路径使用")

    print("per.video", per.video)

    ret = p.add_argument_group("检索")
    ret.add_argument("--top-k",            type=int,   default=16)
    ret.add_argument("--reachable-radius", type=float, default=10.0, metavar="M")

    p.add_argument("--language",    choices=["zh", "en"], default=None)
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--output-json", default="demo_qa_results.json")
    p.add_argument("--verbose",     action="store_true")

    args = p.parse_args()

    # 与 full_pipeline_test 保持一致：
    # work_dir 是总目录，DirectMe mapping 输出在 <work_dir>/directme_mapping_run。
    if not args.run_dir:
        args.run_dir = str(Path(args.work_dir) / "directme_mapping_run")

    # 允许命令行传 --qa-json "" 来只建图、不跑 QA。
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
                "[ERROR] input-mode=from_graph 时不能同时指定 --video 或 --frames-dir。",
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
                "[ERROR] input-mode=from_perception 需要指定 --video 或 --frames-dir。",
                file=sys.stderr,
            )
            return 1

        if args.video and args.frames_dir:
            print("[ERROR] --video 与 --frames-dir 只能指定一个。", file=sys.stderr)
            return 1

        if args.graph_json or args.pipeline_summary:
            print(
                "[WARN] input-mode=from_perception 会重新建图，"
                "--graph-json / --pipeline-summary 将被忽略。"
            )

    else:
        print(f"[ERROR] 未知 input_mode: {input_mode}", file=sys.stderr)
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
            print("[done] 未指定 --qa-json：已完成 perception→scene_graph 建图，跳过 QA。")
        else:
            print("[done] 未指定 --qa-json：已加载已有 scene graph，跳过 QA。")
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
    print(f"\n[保存] {out.resolve()}  ({len(results)} 条)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
