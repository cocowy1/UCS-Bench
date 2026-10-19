#!/usr/bin/env python3
"""UCS-Bench baseline evaluation for proprietary API VLMs.

Covers the proprietary-model rows of the paper's main table (GPT-5,
DeepSeek-V3, Doubao-seed-1.6V, Gemini-3-Flash, ...), all evaluated through an
**OpenAI-compatible chat.completions endpoint** with up to `--num_samples 50`
input images per question.

Ported verbatim from the internal script used for the paper's baseline rows
(`gpt5_eval_sm_1fps.py`), with the hardcoded API key / proxy URL removed:
pass `--base_url` and `--api_key` (or set `OPENAI_API_KEY`).

The evaluation protocol is:

* Frames are pre-extracted at 1 fps (``<frames_dir>/<video_uid>/index.jsonl``).
* ``--frame_select_mode uniform`` (default, K=50):
  for each question, sample K frames uniformly over [0, t_q], deduplicated
  after index rounding.
* ``--frame_select_mode clip_window`` ("stream"):
  the most recent K valid frames within [t_q - clip_window_sec, t_q],
  guaranteeing the **last frame is the question frame** (if the exact second
  is missing, the latest available frame <= t_q is used).
* Frames are resized to max(H, W) <= ``--max_image_size`` (default 512),
  JPEG-encoded (quality 90) and sent as base64 data URLs.

Despite the historical name, no CLIP model is involved in either mode; the
name ``clip_window`` is kept for command compatibility with the original run
log. The prompt text (kept byte-identical to the original experiments for
reproducibility) still mentions CLIP for the window mode.

Notes:
* Requests that fail with server errors / timeouts are retried up to
  ``--max_retries`` times; a question that still fails is counted as
  *incorrect* (it stays in the denominator).
* ``--disable_thinking`` (default on) sends ``extra_body={"thinking":
  {"type": "disabled"}}``, which the OpenAI-compatible proxies used in our
  runs required to suppress reasoning output.
"""

import os
import io
import json
import time
import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from openai import OpenAI, InternalServerError, APITimeoutError


# ===================== Utilities =====================
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


def resize_image_if_needed(image: Image.Image, max_size: int = 512) -> Image.Image:
    width, height = image.size
    max_dim = max(width, height)
    if max_dim > max_size:
        scale = max_size / max_dim
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.BICUBIC)
    return image


def pil_to_base64_jpeg(img: Image.Image, quality: int = 90) -> str:
    import base64

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


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


# ===================== 1 fps frame index loading =====================
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


# ===================== Prompt (identical wording to the local-model runs) =====================
def build_question_prompt(
    question: str,
    options: Dict[str, str],
    question_timestamp: float,
    num_frames: int,
    sampling_desc: str,
) -> str:
    options_text = "\n".join([f"{k}. {v}" for k, v in options.items()])
    return (
        "You are an egocentric video QA assistant.\n"
        "This is a multiple-choice question answering task based strictly on visual content.\n"
        f"You will see a sequence of frames {sampling_desc} "
        f"up to the question time {question_timestamp:.2f} seconds.\n"
        "Use only the frames as evidence to answer the question.\n\n"
        "Rules:\n"
        "- Rely only on visible evidence.\n"
        "- Do not hallucinate unseen objects or actions.\n"
        "- Use temporal reasoning if helpful.\n"
        "- There is exactly one best answer.\n"
        "- Only output a single uppercase letter (A, B, C, ...).\n"
        "- Do not output any explanations or extra text.\n\n"
        "Your job is to choose the option that best matches the visual content.\n\n"
        f"Question:\n{question}\n\n"
        f"Options:\n{options_text}\n\n"
        f"There are {num_frames} frames in temporal order.\n"
        "Only output the letter of the best option (A, B, C, ...). "
        "Note that there is only one best option."
    )


def call_api_multi_frames(
    client: OpenAI,
    model: str,
    pil_images: List[Image.Image],
    question: str,
    options: Dict[str, str],
    question_timestamp: float,
    sampling_desc: str,
    temperature: float = 0.0,
    max_tokens: int = 16,
    timeout: float = 120.0,
    disable_thinking: bool = True,
) -> str:
    prompt_text = build_question_prompt(
        question=question,
        options=options,
        question_timestamp=question_timestamp,
        num_frames=len(pil_images),
        sampling_desc=sampling_desc,
    )

    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt_text}]
    for img in pil_images:
        b64 = pil_to_base64_jpeg(img)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            }
        )

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful vision-language assistant for egocentric video QA. "
                "You receive multiple frames sampled from the video."
            ),
        },
        {"role": "user", "content": content},
    ]

    kwargs = {}
    if disable_thinking:
        # Required by the OpenAI-compatible proxies used in our runs to
        # suppress reasoning output.
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    raw_resp = client.chat.completions.with_raw_response.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=timeout,
        **kwargs,
    )
    resp = raw_resp.parse()

    request_id = getattr(raw_resp, "request_id", None)
    if request_id:
        print("[DEBUG] request_id =", request_id)

    # Tolerate proxy implementations that return plain strings or dict shapes.
    if isinstance(resp, str):
        raw = resp.strip()
    elif hasattr(resp, "choices"):
        choice = resp.choices[0]
        msg = getattr(choice, "message", None)
        if msg is None and isinstance(choice, dict):
            msg = choice.get("message", None)
        if msg is None:
            raw = str(resp).strip()
        else:
            content_field = getattr(msg, "content", None)
            if content_field is None and isinstance(msg, dict):
                content_field = msg.get("content", None)

            if isinstance(content_field, str):
                raw = content_field.strip()
            elif isinstance(content_field, list):
                texts = []
                for c in content_field:
                    if isinstance(c, dict) and c.get("type") == "text":
                        texts.append(c.get("text", ""))
                raw = "\n".join(texts).strip()
            else:
                raw = str(content_field).strip()
    else:
        raw = str(resp).strip()

    return parse_option_from_answer(raw)


# ===================== Evaluation loop =====================
def evaluate_video_ucsbench_api(
    client: OpenAI,
    model_name: str,
    frame_paths: List[Optional[str]],
    qa_items: List[QAItem],
    num_samples: int = 32,
    max_image_size: int = 512,
    frame_select_mode: str = "uniform",
    clip_window_sec: int = 300,
    gpt_max_tokens: int = 16,
    gpt_timeout: float = 120.0,
    gpt_temperature: float = 0.0,
    max_retries: int = 5,
    retry_sleep_sec: int = 30,
    disable_thinking: bool = True,
):
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
            cand_valid_indices = []
            for i in cand_indices:
                if get_frame(i) is not None:
                    cand_valid_indices.append(i)

            if len(cand_valid_indices) == 0:
                pil_images = []
                indices = []
            else:
                # Ensure the question frame is present: if the exact second is
                # missing, use the latest available frame <= q_sec.
                question_frame_sec = cand_valid_indices[-1]

                k = int(num_samples)
                if len(cand_valid_indices) > k:
                    indices = cand_valid_indices[-k:]
                else:
                    indices = cand_valid_indices

                pil_images = [get_frame(i) for i in indices]
                pil_images = [im for im in pil_images if im is not None]

                # Fallback: if the last frame is not the question frame for
                # whatever reason, add it back and re-trim.
                if len(indices) > 0 and indices[-1] != question_frame_sec:
                    if question_frame_sec not in indices:
                        indices.append(question_frame_sec)
                    indices = sorted(set(indices))
                    if len(indices) > k:
                        indices = indices[-k:]
                    pil_images = [get_frame(i) for i in indices]
                    pil_images = [im for im in pil_images if im is not None]

        else:
            raise ValueError(f"Unknown frame_select_mode={frame_select_mode}")

        if not pil_images:
            print(f"[WARN] No frames found for QID {qa.qid}, skip.")
            continue

        if frame_select_mode == "uniform":
            sampling_desc = "sampled uniformly from the start of the video"
        else:
            # Historical wording, kept byte-identical to the original runs.
            sampling_desc = f"selected by CLIP within the last {clip_window_sec} seconds"

        pred_label = None
        raw_error = None

        for attempt in range(max_retries):
            try:
                pred_label = call_api_multi_frames(
                    client=client,
                    model=model_name,
                    pil_images=pil_images,
                    question=qa.question,
                    options=qa.options,
                    question_timestamp=qa.question_timestamp,
                    sampling_desc=sampling_desc,
                    temperature=gpt_temperature,
                    max_tokens=gpt_max_tokens,
                    timeout=gpt_timeout,
                    disable_thinking=disable_thinking,
                )
                break
            except (InternalServerError, APITimeoutError) as e:
                raw_error = f"{type(e).__name__}: {e}"
                print(f"[WARN] {raw_error} on qid={qa.qid}, attempt={attempt+1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(retry_sleep_sec)
            except Exception as e:
                raw_error = f"UnexpectedError: {e}"
                print(f"[ERROR] {raw_error} on qid={qa.qid}, attempt={attempt+1}/{max_retries}")
                break

        if pred_label is None:
            is_correct = False
        else:
            is_correct = (pred_label == qa.answer_label)

        print(
            f"[Q] {qa.qid} @ {qa.question_timestamp:.1f}s | Frames: {len(pil_images)} | "
            f"Pred: {pred_label} | GT: {qa.answer_label} | {is_correct}"
        )

        overall_stats.add(is_correct)
        qtype_stats.setdefault(qa.qtype, EvalResult()).add(is_correct)
        category_stats.setdefault(qa.category, EvalResult()).add(is_correct)
        subcategory_stats.setdefault(qa.subcategory, EvalResult()).add(is_correct)

        key_qc = f"{qa.qtype}||{qa.category}"
        qtype_category_stats.setdefault(key_qc, EvalResult()).add(is_correct)

        rec = {
            "video_uid": qa.video_uid,
            "qid": qa.qid,
            "question": qa.question,
            "gt_label": qa.answer_label,
            "pred_label": pred_label,
            "is_correct": is_correct,
            "qtype": qa.qtype,
            "category": qa.category,
            "subcategory": qa.subcategory,
            "frame_select_mode": frame_select_mode,
            "selected_frame_indices_sec": indices,
        }
        if raw_error is not None and pred_label is None:
            rec["error"] = raw_error

        per_question_results.append(rec)

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

    # --- OpenAI-compatible endpoint ---
    parser.add_argument("--base_url", type=str, default=None,
                        help="OpenAI-compatible base_url (default: official OpenAI endpoint)")
    parser.add_argument("--api_key", type=str,
                        default=os.environ.get("OPENAI_API_KEY"),
                        help="API key (default: $OPENAI_API_KEY)")
    parser.add_argument("--model", type=str, default="gpt-5",
                        help="Model name, e.g. gpt-5 / deepseek-v3 / doubao-seed-1.6v / gemini-3-flash")
    parser.add_argument("--gpt_timeout", type=float, default=300.0)
    parser.add_argument("--gpt_max_tokens", type=int, default=24)
    parser.add_argument("--gpt_temperature", type=float, default=0.0)
    parser.add_argument("--max_retries", type=int, default=6)
    parser.add_argument("--retry_sleep_sec", type=int, default=60)
    parser.add_argument("--disable_thinking", action=argparse.BooleanOptionalAction, default=True,
                        help="Send extra_body={'thinking': {'type': 'disabled'}} (needed by the "
                             "proxies used in our runs; use --no-disable_thinking to omit).")

    # --- dataset paths ---
    parser.add_argument("--qa_dir", type=str, default="data/UCS-Bench/QAs")
    parser.add_argument("--frames_dir", type=str,
                        default="data/UCS-Bench/frames_1fps")
    parser.add_argument("--index_name", type=str, default="index.jsonl")
    parser.add_argument("--output", type=str, default="results/api_vlm/50_uniform.json")

    parser.add_argument("--num_samples", type=int, default=50,
                        help="K frames per question (proprietary APIs cap around 50 images)")
    parser.add_argument("--max_image_size", type=int, default=512)

    parser.add_argument("--frame_select_mode", type=str, default="uniform",
                        choices=["uniform", "clip_window"])
    parser.add_argument("--clip_window_sec", type=int, default=50,
                        help="Window length in seconds for clip_window mode (default: 50)")

    parser.add_argument("--print_every", type=int, default=1)

    args = parser.parse_args()

    if not args.api_key:
        parser.error("--api_key or $OPENAI_API_KEY is required")

    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key,
        timeout=args.gpt_timeout,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    video2qas = load_qa_items_from_dir(args.qa_dir)

    all_results = []
    global_overall = EvalResult()
    global_qtype_stats: Dict[str, EvalResult] = {}
    global_category_stats: Dict[str, EvalResult] = {}
    global_subcategory_stats: Dict[str, EvalResult] = {}
    global_qtype_category_stats: Dict[str, EvalResult] = {}

    print(f"[INFO] Start Evaluating {len(video2qas)} videos with {args.model}...")
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
        ) = evaluate_video_ucsbench_api(
            client=client,
            model_name=args.model,
            frame_paths=frame_paths,
            qa_items=qas,
            num_samples=args.num_samples,
            max_image_size=args.max_image_size,
            frame_select_mode=args.frame_select_mode,
            clip_window_sec=args.clip_window_sec,
            gpt_max_tokens=args.gpt_max_tokens,
            gpt_timeout=args.gpt_timeout,
            gpt_temperature=args.gpt_temperature,
            max_retries=args.max_retries,
            retry_sleep_sec=args.retry_sleep_sec,
            disable_thinking=args.disable_thinking,
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
    print(f"===== Final Statistics (UCS-Bench + {args.model}) =====")
    print("=" * 50)
    print(
        f"Overall Accuracy: {global_overall.correct}/{global_overall.total} "
        f"= {global_overall.acc * 100:.2f}%"
    )

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
        "api": {
            "base_url": args.base_url,
            "model": args.model,
            "max_tokens": args.gpt_max_tokens,
            "temperature": args.gpt_temperature,
            "disable_thinking": args.disable_thinking,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
