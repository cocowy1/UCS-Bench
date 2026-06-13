#!/usr/bin/env python3
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # 强制使用 GPU 0，避免默认 auto 选错 GPU
import sys
import re
from tqdm.auto import tqdm
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from directme.perception.test_perception_stack import (
    _extract_video_frames,
    _list_images,
    _make_video_frames,
    _load_classes_from_yaml,
    _resolve_scal3r_result_dir,
    CachedBackend,
)


# ===================== 数据结构 =====================
@dataclass
class QAItem:
    video_uid: str
    qid: str
    question: str
    options: Dict[str, str]
    answer_label: str
    qtype: str
    category: str
    subcategory: str
    question_timestamp: float


@dataclass
class EvalResult:
    total: int = 0
    correct: int = 0

    def add(self, is_correct: bool):
        self.total += 1
        if is_correct:
            self.correct += 1

    @property
    def acc(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0



def build_graph_from_perception_for_uid(args, video_uid: str):
    from directme.config import DirectMeConfig
    from directme.mapping.offline_engine import OfflineMappingEngine
    from directme.perception.adapters.open_vocab_tracking import (
        OpenVocabularyTrackingAdapter,
        Sam2MaskRefiner,
        SimpleIoUAppearanceTracker,
        YoloWorldDetector,
    )
    from directme.perception.adapters.composed import build_unified_perception_backend

    work_dir = Path(args.perception_work_root) / video_uid
    mapping_run_dir = work_dir / "directme_mapping_run"
    scene_graph_json = mapping_run_dir / "scene_graph.json"

    if scene_graph_json.exists() and not args.rebuild_perception_graph:
        return load_graph(str(scene_graph_json), None), str(scene_graph_json)

    work_dir.mkdir(parents=True, exist_ok=True)

    if args.video_root:
        video_path = Path(args.video_root) / f"{video_uid}.mp4"
        image_paths = _extract_video_frames(
            video_path,
            work_dir / "extracted_frames",
            fps=args.input_fps,
            max_frames=args.max_perception_frames or None,
        )
    else:
        image_paths = _list_images(
            Path(args.frames_root) / video_uid,
            max_images=args.max_perception_frames or None,
        )

    frames = _make_video_frames(image_paths)

    classes = _load_classes_from_yaml(
        args.classes_file,
        class_limit=args.class_limit,
    )

    detector = YoloWorldDetector(
        weights=args.yolo_weights,
        classes=classes,
        score_threshold=args.score_threshold,
        device=args.perception_device,
    )

    segmenter = None
    if args.use_sam2:
        segmenter = Sam2MaskRefiner(
            checkpoint=args.sam2_checkpoint,
            config=args.sam2_config,
            device=args.perception_device,
        )

    tracker = OpenVocabularyTrackingAdapter(
        detector=detector,
        segmenter=segmenter,
        tracker=SimpleIoUAppearanceTracker(),
        detection_stride=args.detection_stride,
    )

    precomputed_root = _resolve_scal3r_result_dir(args.precomputed_scal3r_root)

    backend = build_unified_perception_backend(
        depth_backend=args.depth_backend,
        tracker=tracker,
        device=args.perception_device,

        da3_model_id=args.da3_model_id,
        da3_process_res=args.da3_process_res,

        scal3r_config=args.scal3r_config,
        scal3r_checkpoint=args.scal3r_checkpoint,
        scal3r_work_dir=str(work_dir / "scal3r_work"),
        precomputed_scal3r_root=str(precomputed_root) if precomputed_root else None,

        enable_scene_tag=not args.disable_scene_tag,
    )

    config = DirectMeConfig(run_dir=str(mapping_run_dir))
    engine = OfflineMappingEngine(
        backend=backend,
        config=config,
    )

    chunk_size = max(1, int(args.perception_chunk_size))

    for chunk_id, start in enumerate(range(0, len(frames), chunk_size)):
        sub_frames = frames[start:start + chunk_size]

        print(
            f"[PERCEPTION] video={video_uid} "
            f"chunk={chunk_id} frames={start}:{start + len(sub_frames)}"
        )

        engine.process_chunk(sub_frames, chunk_id=chunk_id)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return load_graph(str(scene_graph_json), None), str(scene_graph_json)


def load_clip(model_name, device="cuda"):
    clip_model = CLIPModel.from_pretrained(
        model_name,
        local_files_only=True,
        use_safetensors=True,
    ).to(device).eval()

    clip_processor = CLIPProcessor.from_pretrained(
        model_name,
        local_files_only=True,
    )
    return clip_model, clip_processor

def collect_keyframe_paths(ctx) -> list[str]:
    paths = []
    for it in ctx.items:
        for kf in getattr(it.node, "keyframes", []) or []:
            if isinstance(kf, str):
                p = kf
            elif isinstance(kf, dict):
                p = kf.get("path") or kf.get("image_path") or kf.get("frame_path")
            else:
                p = getattr(kf, "path", None) or getattr(kf, "image_path", None)

            if p and os.path.exists(p):
                paths.append(p)

    # 去重，保持顺序
    seen = set()
    out = []
    for p in paths:
        if p not in seen:
            out.append(p)
            seen.add(p)
    return out


@torch.no_grad()
def clip_rerank_keyframes(
    question: str,
    keyframe_paths: list[str],
    clip_model,
    clip_processor,
    top_k: int = 8,
    device: str = "cuda",
) -> list[Image.Image]:
    if not keyframe_paths:
        return []

    images = []
    valid_paths = []

    for p in keyframe_paths:
        try:
            img = Image.open(p).convert("RGB")
            images.append(img)
            valid_paths.append(p)
        except Exception:
            continue

    if not images:
        return []

    text_inputs = clip_processor(
        text=[question],
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(device)

    text_feat = clip_model.get_text_features(**text_inputs)

    if not torch.is_tensor(text_feat):
        text_feat = text_feat.pooler_output

    text_feat = text_feat / (text_feat.norm(dim=-1, keepdim=True) + 1e-6)

    scores = []

    for img in images:
        image_inputs = clip_processor(
            images=[img],
            return_tensors="pt",
        ).to(device)

        image_feat = clip_model.get_image_features(**image_inputs)

        if not torch.is_tensor(image_feat):
            image_feat = image_feat.pooler_output

        image_feat = image_feat / (image_feat.norm(dim=-1, keepdim=True) + 1e-6)

        sim = (image_feat @ text_feat.T).item()
        scores.append(sim)

    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    order = order[:top_k]

    return [images[i] for i in order]

def qa_to_item(qa: QAItem, clip_model, clip_processor, clip_device):
    return {
        "qid": qa.qid,
        "q_id": qa.qid,
        "video_uid": qa.video_uid,
        "question": qa.question,
        "question_timestamps": qa.question_timestamp,
        "options": qa.options,
        "answer_label": qa.answer_label,
        "category": qa.category,
        "subcategory": qa.subcategory,
        "qtype": qa.qtype,

        "_clip_model": clip_model,
        "_clip_processor": clip_processor,
        "_clip_device": clip_device,
        "_clip_top_k": 32,
    }

def build_directme_prior_text(spatial_items: list[dict], max_items: int = 12) -> str:
    if not spatial_items:
        return "No relevant DirectMe spatial memory was retrieved."

    lines = []
    for i, it in enumerate(spatial_items[:max_items], 1):
        color = it.get("color")
        color_txt = f", color={color}" if color and color != "unknown" else ""
        lines.append(
            f"{i}. object={it.get('label')}{color_txt}, "
            f"relation={it.get('relation')}, "
            f"distance={it.get('distance_m'):.2f}m, "
            f"reachable={it.get('reachable')}, "
            f"description={it.get('natural_language')}"
        )

    return "\n".join(lines)

# ---------------------------------------------------------------------------
# QA JSON 加载（UCS-Bench 格式）
# ---------------------------------------------------------------------------
def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))

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
# 核心查询函数
# ---------------------------------------------------------------------------

def run_query(
    dm,
    item: dict[str, Any],
    *,
    language: str | None = None,
    top_k: int = 32,
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
        prior_text = build_directme_prior_text(spatial_items, max_items=32)

        keyframe_paths = collect_keyframe_paths(ctx)

        clip_model = item.get("_clip_model")
        clip_processor = item.get("_clip_processor")
        clip_device = item.get("_clip_device", "cuda")
        clip_top_k = item.get("_clip_top_k", 32)
        clip_query = question + "\n" + "\n".join(
            f"{k}. {v}" for k, v in sorted(options.items())
        )

        keyframe_images = clip_rerank_keyframes(
            question=clip_query,
            keyframe_paths=keyframe_paths,
            clip_model=clip_model,
            clip_processor=clip_processor,
            top_k=clip_top_k,
            device=clip_device,
        )

        prompt = build_mc_prompt_with_prior(
            question=question,
            options=options,
            prior_text=prior_text,
            timestamp_s=ts_s,
        )


        raw_answer = dm.generator.answer_multimodal(
            system_prompt="",
            text=prompt,
            images=keyframe_images,
        )

        predicted_label = parse_option_from_answer(raw_answer)

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
        "num_keyframes": len(keyframe_images) if options else 0,

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
# 工具函数
# ---------------------------------------------------------------------------
def _ts_sec(ts) -> float:
    if isinstance(ts, (int, float)):
        return float(ts)

    parts = [float(p) for p in str(ts).strip().split(":")]
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

def strip_mcq_uid(video_uid: str) -> str:
    """
    把 xxxx-mcq / xxxx__mcq / xxxx___--mcq 统一映射到 xxxx
    """
    video_uid = (video_uid or "").strip()
    if not video_uid:
        return ""

    matches = list(re.finditer(r"mcq", video_uid, flags=re.IGNORECASE))
    if not matches:
        return video_uid

    m = matches[-1]
    i = m.start() - 1
    while i >= 0 and video_uid[i] in "-_":
        i -= 1
    return video_uid[: i + 1]

# ===================== 数据加载辅助 =====================
def timestamp_to_seconds(ts: str) -> float:
    try:
        parts = ts.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        else:
            return float(ts)
    except Exception:
        return 0.0

def load_qa_items_from_dir(qa_dir: str) -> Dict[str, List[QAItem]]:
    """
    修改点：
    - 从 item["video_uid"] 读到的 uid 可能是 xxxx-mcq
    - 我们在这里统一 strip 成真实视频名字 xxxx
    - video2qas 的 key 使用 strip 后的 uid，保证与 frames_dir 匹配
    """
    video2qas = {}
    uid2files = {} 
    
    if not os.path.exists(qa_dir):
        return {}

    json_files = sorted(
        [os.path.join(qa_dir, f) for f in os.listdir(qa_dir) if f.lower().endswith(".json")]
    )

    for qa_path in tqdm(json_files, desc="Loading QA files"):
        base = os.path.basename(qa_path)
        with open(qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            raw_uid = str(item.get("video_uid", "")).strip()
            uid = strip_mcq_uid(raw_uid)  # ✅ 核心修改

            if not uid:
                continue
            

            ans_label = item.get("answer_label", "")
            if not ans_label and "answer" in item:
                ans_label = item["answer"]

            qa = QAItem(
                video_uid=uid,  # ✅ 用 strip 后的 uid
                qid=str(item["qid"]),
                question=str(item["question"]),
                options=item["options"],
                answer_label=str(ans_label),
                qtype=str(item.get("qtype", "general")),
                category=str(item.get("category", "general")),
                subcategory=str(item.get("subcategory", "unknown")),
                question_timestamp=timestamp_to_seconds(item.get("question_timestamps", "00:00:00")),
            )

            # （可选）保留原始 uid 方便 debug：给 qa 动态挂一个字段
            # qa.raw_video_uid = raw_uid

            uid2files.setdefault(uid, set()).add(base)  
            video2qas.setdefault(uid, []).append(qa)

    # ✅ 打印哪些 uid 来自多个文件（就是被合并的根源）
    merged = {u: sorted(list(fs)) for u, fs in uid2files.items() if len(fs) > 1}
    print(f"[INFO] Unique videos (uids): {len(uid2files)}")
    print(f"[INFO] UIDs merged from multiple JSON files: {len(merged)}")
    
    for uid in video2qas:
        video2qas[uid].sort(key=lambda x: x.question_timestamp)

    return video2qas

def parse_option_from_answer(raw: str) -> str:
    raw = (raw or "").strip()

    m = re.search(r"\b(?:Answer|answer)\s*[:：]?\s*([A-E])\b", raw)
    if m:
        return m.group(1).upper()

    m = re.search(r"\b([A-E])\b", raw)
    if m:
        return m.group(1).upper()

    m = re.search(r"\(([A-E])\)", raw)
    if m:
        return m.group(1).upper()

    return ""

def build_mc_prompt_with_prior(question, options, prior_text, timestamp_s=None):
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(options.items()))

    ts_text = f"The question timestamp is {timestamp_s:.1f} seconds.\n" if timestamp_s is not None else ""

    return (
        "You are answering a five-choice egocentric video question.\n"
        f"{ts_text}"
        "You are given:\n"
        "1. Visual keyframes retrieved from the video.\n"
        "2. DirectMe spatial memory retrieved from the scene graph.\n\n"

        "DirectMe spatial memory contains object labels, colors, egocentric relations, "
        "distances, reachability, and natural-language spatial descriptions.\n"
        "Use it as contextual evidence, but it may be incomplete or noisy.\n"
        "If visual evidence and spatial memory conflict, prefer visual evidence.\n\n"

        "Reasoning rules:\n"
        "- Compare every option A, B, C, D, and E against the visual keyframes and spatial memory.\n"
        "- Pay attention to egocentric relations such as left, right, front, behind, near, far, and reachable.\n"
        "- Use the timestamp context: answer according to what is visible/known at the question time.\n"
        "- Select the single best option.\n"
        "- Output only one uppercase letter: A, B, C, D, or E.\n\n"

        "DirectMe spatial memory:\n"
        f"{prior_text}\n\n"

        "Question:\n"
        f"{question}\n\n"

        "Options:\n"
        f"{options_text}\n\n"

        "Answer:"
    )

def find_graph_for_uid(graph_root: str, uid: str) -> str | None:
    uid = strip_mcq_uid(uid)
    candidates = [
        Path(graph_root) / uid / "scene_graph.json",
        Path(graph_root) / uid / "graph.json",
        Path(graph_root) / uid / "directme_mapping_run" / "scene_graph.json",
        Path(graph_root) / f"{uid}.json",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def stat_dict(x: EvalResult):
    return {
        "correct": x.correct,
        "total": x.total,
        "acc": x.acc,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa_dir", default="/data/ywang/dataset/SpatialMemory/MC_QAs_v20/")
    parser.add_argument("--graph_root", default="")
    parser.add_argument("--output", default="/data/ywang/my_projects/VideoUnderstanding/Directme/ucsbench_output/results_qwen.json")

    parser.add_argument("--mode", choices=["qwen", "internvl"], default="qwen")
    parser.add_argument("--model_path", default="/data/ywang/my_projects/VideoUnderstanding/Qwen3-VL-8B-Instruct", help="本地模型路径（mode=qwen 或 mode=internvl 时使用）")
    parser.add_argument("--device_map", default="cuda")
    parser.add_argument("--perception_chunk_size", type=int, default=300)

    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--reachable_radius", type=float, default=10.0)
    parser.add_argument("--language", choices=["en", "zh"], default="en")

    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_image_size", type=int, default=512)
    parser.add_argument("--internvl_image_size", type=int, default=448)

    parser.add_argument("--from_perception", action="store_true", default=True)
    parser.add_argument("--frames_root", default="/data/ywang/dataset/SpatialMemory/data_frames_1fps")
    parser.add_argument("--video_root", default="")
    parser.add_argument("--perception_work_root", default="runs/ucsbench_perception_graphs")
    parser.add_argument("--rebuild_perception_graph", action="store_true")

    parser.add_argument("--perception_device", default="cuda")
    parser.add_argument("--max_perception_frames", type=int, default=0)

    parser.add_argument("--depth_backend", choices=["da3", "scal3r"], default="da3")
    parser.add_argument("--da3_model_id", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--da3_process_res", type=int, default=504)

    parser.add_argument("--scal3r_config", default="")
    parser.add_argument("--scal3r_checkpoint", default="")
    parser.add_argument("--precomputed_scal3r_root", default="")

    parser.add_argument("--yolo_weights", default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/yolo/yolov8m-worldv2.pt")
    parser.add_argument("--classes_file", default="/data/ywang/my_projects/VideoUnderstanding/Directme/directme/perception/adapters/Object.yaml")
    parser.add_argument("--class_limit", type=int, default=300)
    parser.add_argument("--score_threshold", type=float, default=0.15)
    parser.add_argument("--detection_stride", type=int, default=1)

    parser.add_argument("--use_sam2", action="store_true", default=True)
    parser.add_argument("--sam2_checkpoint", default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/sam2/sam2.1_hiera_base_plus.pt")

    parser.add_argument("--sam2_config", default="configs/sam2.1/sam2.1_hiera_b+.yaml")
    parser.add_argument("--disable_scene_tag", action="store_true")
    parser.add_argument("--input_fps", type=float, default=1.0)

    args = parser.parse_args()
    clip_device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, clip_processor = load_clip(
        "/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/clip-vit-base-patch32",
        device=clip_device,
    )

    video2qas = load_qa_items_from_dir(args.qa_dir)

    overall = EvalResult()
    by_qtype = defaultdict(EvalResult)
    by_category = defaultdict(EvalResult)
    by_subcategory = defaultdict(EvalResult)
    by_category_subcategory = defaultdict(EvalResult)

    details = []

    for video_uid, qa_items in video2qas.items():
        if args.from_perception:
            try:
                graph, graph_json = build_graph_from_perception_for_uid(args, video_uid)
            except Exception as e:
                print(f"[WARN] perception graph failed for video_uid={video_uid}: {e}")
                continue
        else:
            graph_json = find_graph_for_uid(args.graph_root, video_uid)

            if graph_json is None:
                print(f"[WARN] graph not found for video_uid={video_uid}")
                continue

            graph = load_graph(graph_json, None)

            print(f"\n[VIDEO] {video_uid}")
            print(f"[GRAPH] {graph_json}")

        if graph_json is None:
            print(f"[WARN] graph not found for video_uid={video_uid}")
            continue

        print(f"\n[VIDEO] {video_uid}")
        print(f"[GRAPH] {graph_json}")


        dm = build_directme(
            mode=args.mode,
            model_path=args.model_path,
            device_map=args.device_map,
            graph=graph,
            max_new_tokens=args.max_new_tokens,
            max_image_size=args.max_image_size,
            internvl_image_size=args.internvl_image_size,
        )

        items = [
            qa_to_item(qa, clip_model, clip_processor, clip_device)
            for qa in qa_items
        ]

        batch_results = run_batch(
            dm,
            items,
            language=args.language,
            top_k=args.top_k,
            reachable_radius_m=args.reachable_radius,
            verbose=False,
        )

        qa_by_qid = {qa.qid: qa for qa in qa_items}

        for result in batch_results:
            qid = result.get("qid")
            qa = qa_by_qid.get(str(qid))
            if qa is None:
                continue

            pred = result.get("predicted_label")
            gt = qa.answer_label
            ok = pred == gt

            overall.add(ok)
            by_qtype[qa.qtype].add(ok)
            by_category[qa.category].add(ok)
            by_subcategory[qa.subcategory].add(ok)
            by_category_subcategory[f"{qa.category} || {qa.subcategory}"].add(ok)

            details.append({
                "video_uid": qa.video_uid,
                "qid": qa.qid,
                "question": qa.question,
                "options": qa.options,
                "answer_label": gt,
                "predicted_label": pred,
                "is_correct": ok,
                "raw_answer": result.get("raw_answer"),
                "num_keyframes": result.get("num_keyframes", 0),
                "qtype": qa.qtype,
                "category": qa.category,
                "subcategory": qa.subcategory,
                "matched_count": result.get("matched_count"),
                "reachable_count": result.get("reachable_count"),
                "spatial_items": result.get("spatial_items", []),
                "graph_json": graph_json,
            })

            print(
                f"[{qa.qid}] pred={pred} gt={gt} "
                f"correct={ok} overall={overall.acc:.2%}"
            )

    output = {
        "overall": stat_dict(overall),
        "by_qtype": {k: stat_dict(v) for k, v in by_qtype.items()},
        "by_category": {k: stat_dict(v) for k, v in by_category.items()},
        "by_subcategory": {k: stat_dict(v) for k, v in by_subcategory.items()},
        "by_category_subcategory": {
            k: stat_dict(v) for k, v in by_category_subcategory.items()
        },
        "details": details,

        "config": {
            "qa_dir": args.qa_dir,
            "graph_root": args.graph_root,
            "mode": args.mode,
            "model_path": args.model_path,
            "top_k": args.top_k,
            "reachable_radius": args.reachable_radius,
        },
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n[FINAL] acc={overall.correct}/{overall.total} = {overall.acc:.2%}")
    print(f"[SAVE] {args.output}")


if __name__ == "__main__":
    main()
