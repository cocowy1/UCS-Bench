#!/usr/bin/env python3
"""UCS-Bench baseline evaluation for Qwen2.5-VL / Qwen3-VL models.

Ported verbatim from the internal script used for the paper's baseline rows
(`Qwen_stream_eval.py`). The evaluation protocol is:

* Frames are pre-extracted at 1 fps (``<frames_dir>/<video_uid>/index.jsonl``).
* ``--frame_select_mode uniform``:
  for each question, sample ``--num_samples`` frames uniformly over
  [0, t_q] (inclusive), deduplicated after index rounding, so a question may
  receive slightly fewer than ``--num_samples`` frames.
* ``--frame_select_mode clip_window`` ("stream"):
  take the *most recent* ``--num_samples`` valid frames within
  [t_q - clip_window_sec, t_q].

Despite the historical name, no CLIP model is involved in either mode; the
name ``clip_window`` is kept for command compatibility with the original run
log. The prompt text (kept byte-identical to the original experiments for
reproducibility) still mentions CLIP for the window mode.

Select the GPU with the ``CUDA_VISIBLE_DEVICES`` environment variable.

Greedy decoding (temperature 0, max_new_tokens=32), seed 42. Questions whose
inference raises an exception are counted as *incorrect* (they stay in the
denominator), matching the original runs.
"""

import os
import json
import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from transformers import AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info


# ===================== Random seed =====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ===================== Resize =====================
def resize_image_if_needed(image: Image.Image, max_size: int = 512) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)
    if max_dim > max_size:
        scale = max_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.BICUBIC)
    return image


# ===================== Data structures =====================
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


# ===================== QA loading =====================
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
    video2qas: Dict[str, List[QAItem]] = {}
    if not os.path.exists(qa_dir):
        return {}

    json_files = sorted(
        [os.path.join(qa_dir, f) for f in os.listdir(qa_dir) if f.lower().endswith(".json")]
    )
    for qa_path in tqdm(json_files, desc="Loading QA files"):
        with open(qa_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            uid = str(item["video_uid"]).strip()
            if not uid:
                continue

            ans_label = item.get("answer_label", "")
            if not ans_label and "answer" in item:
                ans_label = item["answer"]

            qa = QAItem(
                video_uid=uid,
                qid=str(item["qid"]),
                question=str(item["question"]),
                options=item["options"],
                answer_label=str(ans_label),
                qtype=str(item.get("qtype", "general")),
                category=str(item.get("category", "general")),
                subcategory=str(item.get("subcategory", "unknown")),
                question_timestamp=timestamp_to_seconds(item.get("question_timestamps", "00:00:00")),
            )
            video2qas.setdefault(uid, []).append(qa)

    for uid in video2qas:
        video2qas[uid].sort(key=lambda x: x.question_timestamp)
    return video2qas


def parse_option_from_answer(raw: str) -> str:
    raw = (raw or "").strip()
    match = re.search(r"Answer:\s*([A-Z])", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    match = re.search(r"\(([A-Z])\)", raw)
    if match:
        return match.group(1).upper()
    match = re.search(r"The answer is\s*([A-Z])", raw, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    for ch in raw:
        if ch.isalpha() and ch.upper() in ["A", "B", "C", "D", "E"]:
            return ch.upper()
    return raw


# ===================== 1 fps frame index loading =====================
def _pick_first(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def find_frame_dir(frames_dir: str, video_uid: str) -> str:
    """Exact match first, then prefix match (handles "<uid>-mcq" QA ids)."""
    cand = os.path.join(frames_dir, video_uid)
    if os.path.isdir(cand):
        return cand
    if os.path.isdir(frames_dir):
        for name in os.listdir(frames_dir):
            p = os.path.join(frames_dir, name)
            if name.startswith(video_uid) and os.path.isdir(p):
                return p
    raise FileNotFoundError(f"Cannot find frame dir for uid={video_uid} in {frames_dir}")


def load_sec2path_from_index_jsonl_in_dir(
    frame_dir: str,
    index_name: str = "index.jsonl",
    strict_exists: bool = True,
) -> Dict[int, str]:
    """Read ``frame_dir/index.jsonl`` into a {sec(int): abs_path} map."""
    index_path = os.path.join(frame_dir, index_name)
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"{index_name} not found in {frame_dir} (expected {index_path})")

    sec2path: Dict[int, str] = {}
    bad = 0
    total = 0

    with open(index_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                obj = json.loads(line)

                t = _pick_first(obj, ["timestamp_sec", "time_sec", "sec", "second", "timestamp", "time", "t"])
                if t is None:
                    bad += 1
                    continue
                try:
                    sec = int(round(float(t)))
                except Exception:
                    bad += 1
                    continue
                if sec < 0:
                    continue

                p = _pick_first(obj, ["path", "file", "filepath", "image", "img", "frame", "frame_path"])
                if p is None:
                    bad += 1
                    continue

                p = str(p)
                if not os.path.isabs(p):
                    p = os.path.join(frame_dir, p)
                p = os.path.abspath(p)

                if strict_exists and (not os.path.exists(p)):
                    bad += 1
                    continue

                sec2path[sec] = p
            except Exception:
                bad += 1
                continue

    if total > 0:
        print(f"[INFO] Loaded {index_name} in {os.path.basename(frame_dir)}: lines={total}, bad={bad}, frames={len(sec2path)}")
    else:
        print(f"[WARN] {index_name} empty: {index_path}")

    return sec2path


def build_frame_paths_from_sec_map(sec2path: Dict[int, str]) -> List[Optional[str]]:
    """Per-second aligned list: frame_paths[sec] = path or None."""
    if not sec2path:
        return []
    max_sec = max(sec2path.keys())
    arr: List[Optional[str]] = [None] * (max_sec + 1)
    for s, p in sec2path.items():
        if 0 <= s < len(arr):
            arr[s] = p
    return arr


# ===================== Uniform sampling over [0, t_q] =====================
def compute_sample_indices_for_video(
    qa_items: List[QAItem],
    fps: float,
    frame_count: int,
    num_samples: int = 32,
) -> Tuple[Dict[str, List[int]], List[int]]:
    """Per QA, uniformly sample num_samples indices over [0, t_q] (dedup after rounding)."""
    qid2indices = {}
    all_indices_set = set()
    for qa in qa_items:
        if qa.question_timestamp <= 0:
            times = [0.0] * num_samples
        else:
            times = np.linspace(0, qa.question_timestamp, num=num_samples, endpoint=True)
        indices = [int(t * fps) for t in times]
        indices = sorted({min(max(idx, 0), frame_count - 1) for idx in indices})
        qid2indices[qa.qid] = indices
        all_indices_set.update(indices)
    return qid2indices, sorted(list(all_indices_set))


def build_frame_cache_pil_from_indexed_paths(
    frame_paths: List[Optional[str]],
    all_indices: List[int],
    max_image_size: int = 512,
) -> Dict[int, Image.Image]:
    frame_cache: Dict[int, Image.Image] = {}
    if not all_indices:
        return frame_cache

    n = len(frame_paths)
    for idx in all_indices:
        if idx < 0 or idx >= n:
            continue
        p = frame_paths[idx]
        if p is None:
            continue
        try:
            img = Image.open(p).convert("RGB")
            img = resize_image_if_needed(img, max_size=max_image_size)
            frame_cache[idx] = img
        except Exception as e:
            print(f"[WARN] Failed to read frame(sec={idx}): {p} ({e})")
            continue

    return frame_cache


# ===================== Evaluation loop =====================
def evaluate_video_ucsbench_qwen(
    model,
    processor,
    frame_paths: List[Optional[str]],  # 1 fps: frame_paths[sec] = path
    qa_items: List[QAItem],
    num_samples: int = 32,
    max_image_size: int = 512,
    generation_config: Dict[str, Any] = None,
    frame_select_mode: str = "uniform",  # "uniform" or "clip_window"
    clip_window_sec: int = 300,
):
    if generation_config is None:
        generation_config = dict(
            max_new_tokens=32,
            do_sample=False,
            top_p=1.0,
            temperature=0.0,
        )

    fps = 1.0
    frame_count = len(frame_paths)
    if frame_count <= 0:
        print("[ERROR] Empty frames for this video.")
        return [], EvalResult(), {}, {}, {}, {}

    frame_cache: Dict[int, Image.Image] = {}

    def get_frame(sec_idx: int) -> Optional[Image.Image]:
        if sec_idx in frame_cache:
            return frame_cache[sec_idx]
        if sec_idx < 0 or sec_idx >= frame_count:
            return None
        p = frame_paths[sec_idx]
        if p is None or (not os.path.exists(p)):
            return None
        try:
            img = Image.open(p).convert("RGB")
            img = resize_image_if_needed(img, max_size=max_image_size)
            frame_cache[sec_idx] = img
            return img
        except Exception as e:
            print(f"[WARN] Failed to read frame(sec={sec_idx}): {p} ({e})")
            return None

    # Uniform mode: precompute & pre-cache the frames that will be used.
    qid2indices: Dict[str, List[int]] = {}
    if frame_select_mode == "uniform":
        qid2indices, all_indices = compute_sample_indices_for_video(
            qa_items, fps, frame_count, num_samples
        )
        frame_cache.update(
            build_frame_cache_pil_from_indexed_paths(
                frame_paths, all_indices, max_image_size=max_image_size
            )
        )

    per_question_results = []
    overall_stats = EvalResult()
    qtype_stats: Dict[str, EvalResult] = {}
    category_stats: Dict[str, EvalResult] = {}
    subcategory_stats: Dict[str, EvalResult] = {}
    qtype_category_stats: Dict[str, EvalResult] = {}

    for qa in qa_items:
        # --------- indices selection ---------
        indices: List[int] = []

        if frame_select_mode == "uniform":
            indices = qid2indices.get(qa.qid, [])
            pil_images = [frame_cache[idx] for idx in indices if idx in frame_cache]

        elif frame_select_mode == "clip_window":
            q_sec = int(np.floor(float(qa.question_timestamp)))
            q_sec = min(max(q_sec, 0), frame_count - 1)
            start = max(0, q_sec - int(clip_window_sec))
            end = q_sec

            cand_indices = [i for i in range(start, end + 1) if frame_paths[i] is not None]

            if len(cand_indices) == 0:
                indices = []
                pil_images = []
            elif len(cand_indices) <= num_samples:
                indices = cand_indices  # not enough frames, use all of them
                pil_images = [get_frame(i) for i in indices]
                pil_images = [im for im in pil_images if im is not None]
            else:
                indices = cand_indices[-num_samples:]  # most recent K frames
                pil_images = [get_frame(i) for i in indices]
                pil_images = [im for im in pil_images if im is not None]

        else:
            raise ValueError(f"Unknown frame_select_mode={frame_select_mode}")

        if not pil_images:
            print(f"[WARN] No frames found for QID {qa.qid}, skip.")
            continue

        options_text = "\n".join([f"{k}. {v}" for k, v in qa.options.items()])

        if frame_select_mode == "uniform":
            sampling_desc = "sampled uniformly from the start of the video"
        else:
            # Historical wording, kept byte-identical to the original runs.
            sampling_desc = f"selected by CLIP within the last {clip_window_sec} seconds"

        text_prompt = (
            "You are an egocentric video QA assistant.\n"
            "This is a multiple-choice question answering task based strictly on visual content.\n"
            f"You will see a sequence of frames {sampling_desc} "
            f"up to the question time {qa.question_timestamp:.2f} seconds.\n"
            "Use only the frames as evidence to answer the question.\n\n"
            "Rules:\n"
            "- Rely only on visible evidence.\n"
            "- Do not hallucinate unseen objects or actions.\n"
            "- Use temporal reasoning if helpful.\n"
            "- There is exactly one best answer.\n"
            "- Only output a single uppercase letter (A, B, C, ...).\n"
            "- Do not output any explanations or extra text.\n\n"
            "Your job is to choose the option that best matches the visual content.\n\n"
            f"Question:\n{qa.question}\n\n"
            f"Options:\n{options_text}\n\n"
            f"There are {len(pil_images)} frames in temporal order.\n"
            "Only output the letter of the best option (A, B, C, ...). "
            "Note that there is only one best option."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    *[{"type": "image", "image": img} for img in pil_images],
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        try:
            chat_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)

            inputs = processor(
                text=[chat_text],
                images=image_inputs,
                videos=video_inputs,
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=generation_config["max_new_tokens"],
                    do_sample=generation_config["do_sample"],
                    top_p=generation_config["top_p"],
                    temperature=generation_config["temperature"],
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]

            pred_label = parse_option_from_answer(response)
            is_correct = pred_label == qa.answer_label

            print(
                f"[Q] {qa.qid} @ {qa.question_timestamp:.1f}s | Frames: {len(pil_images)} | "
                f"Pred: {pred_label} | GT: {qa.answer_label} | {is_correct} | {response}"
            )

            overall_stats.add(is_correct)
            qtype_stats.setdefault(qa.qtype, EvalResult()).add(is_correct)
            category_stats.setdefault(qa.category, EvalResult()).add(is_correct)
            subcategory_stats.setdefault(qa.subcategory, EvalResult()).add(is_correct)

            key_qc = f"{qa.qtype}||{qa.category}"
            qtype_category_stats.setdefault(key_qc, EvalResult()).add(is_correct)

            per_question_results.append(
                {
                    "video_uid": qa.video_uid,
                    "qid": qa.qid,
                    "question": qa.question,
                    "gt_label": qa.answer_label,
                    "pred_label": pred_label,
                    "is_correct": is_correct,
                    "raw_response": response,
                    "qtype": qa.qtype,
                    "category": qa.category,
                    "subcategory": qa.subcategory,
                    "frame_select_mode": frame_select_mode,
                    "selected_frame_indices_sec": indices,
                }
            )
        except Exception as e:
            print(f"[ERROR] Inference failed for {qa.qid}: {e}")
            import traceback
            traceback.print_exc()
            overall_stats.add(False)

        torch.cuda.empty_cache()

    return (
        per_question_results,
        overall_stats,
        qtype_stats,
        category_stats,
        subcategory_stats,
        qtype_category_stats,
    )


# ===================== Entry point =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2.5-VL-32B-Instruct",
        help="Qwen2.5-VL / Qwen3-VL model path (HF repo id or local checkpoint)",
    )
    parser.add_argument(
        "--qa_dir",
        type=str,
        default="data/UCS-Bench/QAs",
        help="Directory of per-video QA JSON files (UCS-Bench MC format)",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="data/UCS-Bench/frames_1fps",
        help="Root directory of per-video 1 fps frame folders (each with index.jsonl + images)",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="index.jsonl",
        help="Index filename inside each video folder (default: index.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/qwen_vl/64_uniform.json",
    )
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=512,
        help="Resize each frame so max(H, W) <= this before feeding the model",
    )

    parser.add_argument(
        "--frame_select_mode",
        type=str,
        default="uniform",
        choices=["uniform", "clip_window"],
        help="uniform: sample over [0, tq]; clip_window: most recent K frames in [tq-window, tq].",
    )
    parser.add_argument(
        "--clip_window_sec",
        type=int,
        default=300,
        help="Window length in seconds for clip_window mode (default: 300)",
    )

    parser.add_argument(
        "--print_every",
        type=int,
        default=1,
        help="Print running accuracy every N evaluated videos (0 disables).",
    )

    args = parser.parse_args()

    set_seed()
    args.model_path = args.model_path.rstrip("/")

    print(f"[INFO] Loading Qwen VL model & processor: {args.model_path}")

    model = AutoModelForVision2Seq.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model_path)

    generation_config = dict(
        do_sample=False,
        max_new_tokens=32,
        top_p=1.0,
        temperature=0.0,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    video2qas = load_qa_items_from_dir(args.qa_dir)

    all_results = []
    global_overall = EvalResult()
    global_qtype_stats: Dict[str, EvalResult] = {}
    global_category_stats: Dict[str, EvalResult] = {}
    global_subcategory_stats: Dict[str, EvalResult] = {}
    global_qtype_category_stats: Dict[str, EvalResult] = {}

    print(f"[INFO] Start Evaluating {len(video2qas)} videos...")

    evaluated_videos = 0

    for uid, qas in tqdm(video2qas.items(), desc="Evaluating"):
        try:
            frame_dir = find_frame_dir(args.frames_dir, uid)
        except FileNotFoundError:
            print(f"[WARN] Frame folder not found: {uid}")
            continue

        try:
            sec2path = load_sec2path_from_index_jsonl_in_dir(
                frame_dir=frame_dir,
                index_name=args.index_name,
                strict_exists=True,
            )
        except FileNotFoundError:
            print(f"[WARN] index.jsonl missing for video: {uid} (dir={frame_dir})")
            continue

        frame_paths = build_frame_paths_from_sec_map(sec2path)
        if len(frame_paths) == 0:
            print(f"[WARN] Empty frames built from index for video: {uid}")
            continue

        print(
            f"\n===== Evaluate video: {uid} | mode={args.frame_select_mode} | "
            f"K={args.num_samples} | window={args.clip_window_sec}s ====="
        )

        (
            per_video_results,
            overall_stats,
            qtype_stats,
            category_stats,
            subcategory_stats,
            qtype_category_stats,
        ) = evaluate_video_ucsbench_qwen(
            model=model,
            processor=processor,
            frame_paths=frame_paths,
            qa_items=qas,
            num_samples=args.num_samples,
            max_image_size=args.max_image_size,
            generation_config=generation_config,
            frame_select_mode=args.frame_select_mode,
            clip_window_sec=args.clip_window_sec,
        )

        all_results.extend(per_video_results)

        global_overall.total += overall_stats.total
        global_overall.correct += overall_stats.correct

        for qt, st in qtype_stats.items():
            global_qtype_stats.setdefault(qt, EvalResult())
            global_qtype_stats[qt].total += st.total
            global_qtype_stats[qt].correct += st.correct

        for cat, st in category_stats.items():
            global_category_stats.setdefault(cat, EvalResult())
            global_category_stats[cat].total += st.total
            global_category_stats[cat].correct += st.correct

        for subcat, st in subcategory_stats.items():
            global_subcategory_stats.setdefault(subcat, EvalResult())
            global_subcategory_stats[subcat].total += st.total
            global_subcategory_stats[subcat].correct += st.correct

        for key_qc, st in qtype_category_stats.items():
            global_qtype_category_stats.setdefault(key_qc, EvalResult())
            global_qtype_category_stats[key_qc].total += st.total
            global_qtype_category_stats[key_qc].correct += st.correct

        evaluated_videos += 1
        if args.print_every > 0 and (evaluated_videos % args.print_every == 0):
            print(
                f"\n[PROGRESS] evaluated_videos={evaluated_videos} | "
                f"running_acc={global_overall.acc * 100:.2f}% "
                f"({global_overall.correct}/{global_overall.total})\n"
            )

    print("\n" + "=" * 50)
    print("===== Final Statistics (UCS-Bench + Qwen VL) =====")
    print("=" * 50)
    print(
        f"Overall Accuracy: {global_overall.correct}/{global_overall.total} "
        f"= {global_overall.acc * 100:.2f}%"
    )

    print("\n--- Accuracy by QType ---")
    for qt, st in global_qtype_stats.items():
        print(f"  {qt}: {st.correct}/{st.total} = {st.acc * 100:.2f}%")

    print("\n--- Accuracy by Category ---")
    for cat, st in global_category_stats.items():
        print(f"  {cat}: {st.correct}/{st.total} = {st.acc * 100:.2f}%")

    print("\n--- Accuracy by Subcategory ---")
    for subcat, st in global_subcategory_stats.items():
        print(f"  {subcat}: {st.correct}/{st.total} = {st.acc * 100:.2f}%")

    print("\n--- Accuracy by (QType, Category) ---")
    for key_qc, st in global_qtype_category_stats.items():
        parts = key_qc.split("||", 1)
        q_str = parts[0]
        c_str = parts[1] if len(parts) > 1 else ""
        print(f"  ({q_str}, {c_str}): {st.correct}/{st.total} = {st.acc * 100:.2f}%")

    out_obj = {
        "overall": {
            "correct": global_overall.correct,
            "total": global_overall.total,
            "acc": global_overall.acc,
        },
        "by_qtype": {
            qt: {"correct": st.correct, "total": st.total, "acc": st.acc}
            for qt, st in global_qtype_stats.items()
        },
        "by_category": {
            cat: {"correct": st.correct, "total": st.total, "acc": st.acc}
            for cat, st in global_category_stats.items()
        },
        "by_subcategory": {
            subcat: {"correct": st.correct, "total": st.total, "acc": st.acc}
            for subcat, st in global_subcategory_stats.items()
        },
        "by_qtype_category": {
            key_qc: {
                "qtype": key_qc.split("||", 1)[0],
                "category": key_qc.split("||", 1)[1] if "||" in key_qc else "",
                "correct": st.correct,
                "total": st.total,
                "acc": st.acc,
            }
            for key_qc, st in global_qtype_category_stats.items()
        },
        "details": all_results,
        "frames_source": {
            "fps": 1.0,
            "frames_dir": args.frames_dir,
            "index_name": args.index_name,
        },
        "frame_selection": {
            "mode": args.frame_select_mode,
            "K": args.num_samples,
            "clip_window_sec": args.clip_window_sec if args.frame_select_mode == "clip_window" else None,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
