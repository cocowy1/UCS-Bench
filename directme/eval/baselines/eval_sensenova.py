#!/usr/bin/env python3
"""UCS-Bench baseline evaluation for SenseNova-SI (InternVL3-8B based).

Ported verbatim from the internal script used for the paper's baseline rows
(`sensenova_stream_eval_1fps.py`). SenseNova-SI-1.1-InternVL3-8B is loaded
with `trust_remote_code` on a single GPU (no flash-attention). Differences
vs `eval_internvl.py`: dynamic tiling uses **max_num=6** tiles per frame
(instead of 1), and the model is loaded with `.eval().cuda()`. The
evaluation protocol is:

* Frames are pre-extracted at 1 fps (``<frames_dir>/<video_uid>/index.jsonl``).
* ``--frame_select_mode uniform``:
  for each question, sample ``--num_samples`` frames uniformly over
  [0, t_q] (inclusive), deduplicated after index rounding.
* ``--frame_select_mode clip_window`` ("stream"):
  take the *most recent* ``--num_samples`` valid frames within
  [t_q - clip_window_sec, t_q].

Despite the historical name, no CLIP model is involved in either mode; the
name ``clip_window`` is kept for command compatibility with the original run
log. The prompt text (kept byte-identical to the original experiments for
reproducibility) still mentions CLIP for the window mode.

SenseNova-specific notes:
* Each frame is fed through InternVL-style dynamic tiling
  (``input_size=448, max_num=6``) preceded by an ``Image-N: <image>`` token
  block; frames are cached at native resolution.
* ``--max_image_size`` is accepted but has no effect (kept for command
  compatibility), matching the original run.

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
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from PIL import Image
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModel


# ===================== InternVL image preprocessing =====================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height), Image.LANCZOS)
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) > 1:
        thumbnail_img = image.resize((image_size, image_size), Image.LANCZOS)
        processed_images.append(thumbnail_img)
    return processed_images


def process_images_for_internvl(images: List[Image.Image], input_size=448, max_num=12):
    """Turn a list of PIL images into pixel_values and num_patches_list."""
    transform = build_transform(input_size=input_size)
    pixel_values_list = []
    num_patches_list = []

    for img in images:
        # For multi-frame video input, max_num=1 keeps memory bounded
        # (K frames x 1 tile = K patches).
        tiles = dynamic_preprocess(img, min_num=1, max_num=max_num, image_size=input_size, use_thumbnail=False)
        tile_tensors = [transform(tile) for tile in tiles]
        pixel_values_list.extend(tile_tensors)
        num_patches_list.append(len(tiles))

    pixel_values = torch.stack(pixel_values_list)
    return pixel_values, num_patches_list


# ===================== Shared utilities =====================
def strip_mcq_uid(video_uid: str) -> str:
    """Map "<uid>-mcq" / "<uid>__mcq" / "<uid>___--mcq" back to "<uid>"."""
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


def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
    """Group QAs by video_uid with the "-mcq" suffix stripped, so uids match
    the per-video frame folders."""
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
            uid = strip_mcq_uid(raw_uid)

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

            uid2files.setdefault(uid, set()).add(base)
            video2qas.setdefault(uid, []).append(qa)

    merged = {u: sorted(list(fs)) for u, fs in uid2files.items() if len(fs) > 1}
    print(f"[INFO] Unique videos (uids): {len(uid2files)}")
    print(f"[INFO] UIDs merged from multiple JSON files: {len(merged)}")

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


def _pick_first(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def find_frame_dir(frames_dir: str, video_uid: str) -> str:
    video_prefix = strip_mcq_uid(video_uid)
    cand = os.path.join(frames_dir, video_prefix)
    if video_prefix and os.path.isdir(cand):
        return cand
    if os.path.isdir(frames_dir) and video_prefix:
        for name in os.listdir(frames_dir):
            p = os.path.join(frames_dir, name)
            if name.startswith(video_prefix) and os.path.isdir(p):
                return p
    raise FileNotFoundError(
        f"Cannot find frame dir for uid={video_uid} "
        f"(parsed_prefix={video_prefix}) in {frames_dir}"
    )


def load_sec2path_from_index_jsonl_in_dir(
    frame_dir: str,
    index_name: str = "index.jsonl",
    strict_exists: bool = True,
) -> Dict[int, str]:
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
    return sec2path


def build_frame_paths_from_sec_map(sec2path: Dict[int, str]) -> List[Optional[str]]:
    if not sec2path:
        return []
    max_sec = max(sec2path.keys())
    arr: List[Optional[str]] = [None] * (max_sec + 1)
    for s, p in sec2path.items():
        if 0 <= s < len(arr):
            arr[s] = p
    return arr


def compute_sample_indices_for_video(
    qa_items: List[QAItem],
    fps: float,
    frame_count: int,
    num_samples: int = 32,
) -> Tuple[Dict[str, List[int]], List[int]]:
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
    # NOTE: frames are cached at native resolution (no pre-resize); InternVL's
    # dynamic_preprocess handles resizing. max_image_size is unused here, kept
    # in the signature for call-site compatibility.
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
            frame_cache[idx] = img
        except Exception as e:
            print(f"[WARN] Failed to read frame(sec={idx}): {p} ({e})")
            continue

    return frame_cache


# ===================== Evaluation loop =====================
def evaluate_video_ucsbench_sensenova(
    model,
    tokenizer,
    frame_paths: List[Optional[str]],
    qa_items: List[QAItem],
    num_samples: int = 32,
    max_image_size: int = 448,
    generation_config: Dict[str, Any] = None,
    frame_select_mode: str = "uniform",
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
            frame_cache[sec_idx] = img
            return img
        except Exception as e:
            print(f"[WARN] Failed to read frame(sec={sec_idx}): {p} ({e})")
            return None

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
                indices = cand_indices
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

        # ----------------- InternVL input construction -----------------
        try:
            # 1. Image preprocessing -> pixel_values.
            # SenseNova-SI used max_num=6 tiles per frame.
            pixel_values, num_patches_list = process_images_for_internvl(
                pil_images, input_size=448, max_num=6
            )
            pixel_values = pixel_values.to(torch.bfloat16).to(model.device)

            # 2. Text prompt construction with explicit stacked <image> tokens.
            image_tokens = "".join([f"Image-{i+1}: <image>\n" for i in range(len(pil_images))])

            text_prompt = (
                f"{image_tokens}"
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

            # 3. Chat inference (InternVL 3.0/3.5 share the 2.0-style API).
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=text_prompt,
                generation_config=generation_config,
                num_patches_list=num_patches_list,
                verbose=False,
            )

            pred_label = parse_option_from_answer(response)
            is_correct = pred_label == qa.answer_label

            print(
                f"[Q] {qa.qid} @ {qa.question_timestamp:.1f}s | Frames: {len(pil_images)} | "
                f"Pred: {pred_label} | GT: {qa.answer_label} | {is_correct}"
            )

            overall_stats.add(is_correct)
            qtype_stats.setdefault(qa.qtype, EvalResult()).add(is_correct)
            category_stats.setdefault(qa.category, EvalResult()).add(is_correct)
            subcategory_stats.setdefault(qa.subcategory, EvalResult()).add(is_correct)
            qtype_category_stats.setdefault(f"{qa.qtype}||{qa.category}", EvalResult()).add(is_correct)

            per_question_results.append({
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
            })

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
        default="path/to/SenseNova-SI-1.1-InternVL3-8B",
        help="SenseNova-SI checkpoint path",
    )
    parser.add_argument(
        "--chat_template_path",
        type=str,
        default=None,
        help="Jinja2 chat template file path. If None, tries chat_template.jinja in model_path",
    )
    parser.add_argument(
        "--qa_dir",
        type=str,
        default="data/UCS-Bench/QAs",
    )
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="data/UCS-Bench/frames_1fps",
    )
    parser.add_argument(
        "--index_name",
        type=str,
        default="index.jsonl",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/sensenova-si/results_1fps.json",
    )
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument(
        "--max_image_size",
        type=int,
        default=448,
        help="No effect for InternVL (frames are fed at native resolution into "
             "dynamic tiling); kept for compatibility with the original commands.",
    )

    parser.add_argument(
        "--frame_select_mode",
        type=str,
        default="uniform",
        choices=["uniform", "clip_window"],
    )
    parser.add_argument(
        "--clip_window_sec",
        type=int,
        default=300,
        help="Window length in seconds for clip_window mode (default: 300)",
    )
    parser.add_argument("--print_every", type=int, default=1)

    args = parser.parse_args()

    set_seed()
    args.model_path = args.model_path.rstrip("/")

    if args.chat_template_path is None:
        args.chat_template_path = os.path.join(args.model_path, "chat_template.jinja")

    print(f"[INFO] Loading SenseNova-SI model & tokenizer: {args.model_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    # Optionally override the tokenizer's chat template with a local file.
    if os.path.exists(args.chat_template_path):
        print(f"[INFO] Loading chat template from: {args.chat_template_path}")
        with open(args.chat_template_path, 'r', encoding='utf-8') as f:
            tokenizer.chat_template = f.read()
    else:
        print(f"[INFO] Using default tokenizer chat template (if available).")

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).eval().cuda()

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

    print(f"[INFO] Start Evaluating {len(video2qas)} videos with SenseNova-SI...")
    evaluated_videos = 0

    for uid, qas in tqdm(video2qas.items(), desc="Evaluating"):
        try:
            frame_dir = find_frame_dir(args.frames_dir, uid)
        except FileNotFoundError:
            print(f"[WARN] Frame folder not found: {uid}")
            continue

        try:
            sec2path = load_sec2path_from_index_jsonl_in_dir(frame_dir, args.index_name)
        except FileNotFoundError:
            continue

        frame_paths = build_frame_paths_from_sec_map(sec2path)
        if len(frame_paths) == 0:
            continue

        print(
            f"\n===== Evaluate video: {uid} | mode={args.frame_select_mode} | "
            f"K={args.num_samples} ====="
        )

        (
            per_video_results,
            overall_stats,
            qtype_stats,
            category_stats,
            subcategory_stats,
            qtype_category_stats,
        ) = evaluate_video_ucsbench_sensenova(
            model=model,
            tokenizer=tokenizer,
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
    print("===== Final Statistics (UCS-Bench + SenseNova-SI) =====")
    print("=" * 50)
    print(f"Overall Accuracy: {global_overall.correct}/{global_overall.total} = {global_overall.acc * 100:.2f}%")

    out_obj = {
        "overall": {"correct": global_overall.correct, "total": global_overall.total, "acc": global_overall.acc},
        "by_qtype": {qt: {"correct": st.correct, "total": st.total, "acc": st.acc}
                     for qt, st in global_qtype_stats.items()},
        "by_category": {cat: {"correct": st.correct, "total": st.total, "acc": st.acc}
                        for cat, st in global_category_stats.items()},
        "by_subcategory": {subcat: {"correct": st.correct, "total": st.total, "acc": st.acc}
                           for subcat, st in global_subcategory_stats.items()},
        "by_qtype_category": {
            key: {
                "qtype": key.split("||", 1)[0],
                "category": key.split("||", 1)[1] if "||" in key else "",
                "correct": st.correct,
                "total": st.total,
                "acc": st.acc,
            }
            for key, st in global_qtype_category_stats.items()
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
        "config": {"model": args.model_path, "samples": args.num_samples},
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
