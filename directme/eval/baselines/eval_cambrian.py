#!/usr/bin/env python3
"""UCS-Bench baseline evaluation for Cambrian-S models.

Ported verbatim from the internal script used for the paper's baseline rows
(`cambrian-s/sm_eval_1fps.py`). The evaluation protocol is:

* Frames are pre-extracted at 1 fps (``<frames_dir>/<video_uid>/index.jsonl``).
* ``--frame_select_mode uniform``:
  for each question, sample exactly ``--num_samples`` indices uniformly over
  [0, t_q] (inclusive). Indices are rounded and the list is padded by
  repeating the last index, so the model always receives exactly K frames
  (duplicates possible).
* ``--frame_select_mode clip_window`` ("stream"):
  collect every valid frame within [t_q - clip_window_sec, t_q] (default
  window: 64 s), keep the first K of them and pad with repeats of the last
  frame if fewer than K are decodable.

Despite the historical name, no CLIP model is involved in either mode; the
name ``clip_window`` is kept for command compatibility with the original run
log. The prompt text (kept byte-identical to the original experiments for
reproducibility) still mentions CLIP for the window mode.

Cambrian-specific notes:
* Requires the ``cambrian`` package (from the Cambrian-S repository). Pass its
  checkout path via ``--cambrian_repo`` or have it importable already
  (``pip install -e`` / ``PYTHONPATH``).
* Images are passed through Cambrian's own ``process_images`` /
  ``tokenizer_image_token`` / ``conv_templates`` (default conv mode
  ``qwen_2``), with ``<image>`` tokens placed before the text prompt.
* ``--resume_file`` skips QIDs already present in a previous result file and
  carries their statistics over (used to resume interrupted runs).

Select the GPU with the ``CUDA_VISIBLE_DEVICES`` environment variable.

Greedy decoding (temperature 0, max_new_tokens=32), seed 42. Questions whose
inference raises an exception are counted as *incorrect* (they stay in the
denominator), matching the original runs.
"""

import os
import sys
import json
import argparse
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm


def _import_cambrian(cambrian_repo: Optional[str]):
    """Make the `cambrian` package importable, optionally from a checkout."""
    if cambrian_repo:
        repo = os.path.abspath(os.path.expanduser(cambrian_repo))
        if repo not in sys.path:
            sys.path.insert(0, repo)

    from cambrian.model.builder import load_pretrained_model
    from cambrian.mm_utils import process_images, tokenizer_image_token
    from cambrian.conversation import conv_templates

    # IMAGE_TOKEN_INDEX lives in different modules across Cambrian versions.
    try:
        from cambrian.constants import IMAGE_TOKEN_INDEX
    except Exception:
        try:
            from cambrian.mm_utils import IMAGE_TOKEN_INDEX
        except Exception:
            IMAGE_TOKEN_INDEX = -200  # common default

    return load_pretrained_model, process_images, tokenizer_image_token, conv_templates, IMAGE_TOKEN_INDEX


# ===================== Seed =====================
def set_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ===================== Resize =====================
def resize_image_if_needed(image: Image.Image, max_size: int = 512) -> Image.Image:
    w, h = image.size
    m = max(w, h)
    if m > max_size:
        s = max_size / m
        nw = max(1, int(w * s))
        nh = max(1, int(h * s))
        image = image.resize((nw, nh), Image.BICUBIC)
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

    def add(self, ok: bool):
        self.total += 1
        if ok:
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
    m = re.search(r"Answer:\s*([A-Z])", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(r"\(([A-Z])\)", raw)
    if m:
        return m.group(1).upper()
    m = re.search(r"The answer is\s*([A-Z])", raw, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    for ch in raw:
        if ch.isalpha() and ch.upper() in ["A", "B", "C", "D", "E"]:
            return ch.upper()
    return "A"


# ===================== 1 fps frame index loading =====================
def _pick_first(d: dict, keys: List[str], default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def find_frame_dir(frames_dir: str, video_uid: str) -> str:
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
    if not sec2path:
        return []
    max_sec = max(sec2path.keys())
    arr: List[Optional[str]] = [None] * (max_sec + 1)
    for s, p in sec2path.items():
        if 0 <= s < len(arr):
            arr[s] = p
    return arr


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


# ===================== Frame selection =====================
def compute_uniform_indices_1fps_fixedlen(
    question_timestamp: float,
    frame_count: int,
    k: int,
) -> List[int]:
    """1 fps (index == second): uniformly sample k indices over [0, t].
    The output length is strictly k (duplicates allowed)."""
    if frame_count <= 0:
        return [0] * k

    t = max(0.0, float(question_timestamp))
    if t <= 1e-6:
        times = [0.0] * k
    else:
        times = np.linspace(0.0, t, num=k, endpoint=True)

    idx = [int(round(float(tt))) for tt in times]  # fps=1
    idx = [min(max(i, 0), frame_count - 1) for i in idx]

    if len(idx) < k:
        idx += [idx[-1]] * (k - len(idx))
    elif len(idx) > k:
        idx = idx[:k]
    return idx


def collect_valid_secs_in_window(
    frame_paths: List[Optional[str]],
    get_frame_fn,
    start_sec: int,
    end_sec: int,
) -> List[int]:
    """All decodable frame seconds within [start_sec, end_sec], ascending."""
    n = len(frame_paths)
    start_sec = max(0, int(start_sec))
    end_sec = min(int(end_sec), n - 1)

    valid_secs = []
    for sec in range(start_sec, end_sec + 1):
        if sec < 0 or sec >= n:
            continue
        if frame_paths[sec] is None:
            continue
        im = get_frame_fn(sec)
        if im is not None:
            valid_secs.append(sec)
    return valid_secs


# ===================== Cambrian wrappers =====================
def _cambrian_process_images(
    pil_images: List[Image.Image],
    image_processor,
    model,
    process_images_fn,
) -> Tuple[torch.Tensor, Any]:
    image_tensor, image_sizes = process_images_fn(pil_images, image_processor, model.config)

    if isinstance(image_tensor, (list, tuple)):
        image_tensor = torch.stack(list(image_tensor), dim=0)

    image_tensor = image_tensor.to(model.device)
    return image_tensor, image_sizes


def _cambrian_tokenize_prompt(prompt: str, tokenizer, tokenizer_image_token_fn, image_token_index: int) -> torch.Tensor:
    try:
        res = tokenizer_image_token_fn(prompt, tokenizer, image_token_index, return_tensors="pt")
    except TypeError:
        res = tokenizer_image_token_fn(prompt, tokenizer, return_tensors="pt")

    if isinstance(res, dict):
        res = res.get("input_ids", None)

    if isinstance(res, (tuple, list)) and len(res) > 0:
        if torch.is_tensor(res[0]) or isinstance(res[0], (list, np.ndarray)):
            res = res[0]

    if isinstance(res, (list, np.ndarray)):
        res = torch.tensor(res, dtype=torch.long)

    if res is None or not torch.is_tensor(res):
        raise RuntimeError(
            f"tokenizer_image_token returned invalid input_ids: type={type(res)}, value={str(res)[:200]}"
        )

    if res.dim() == 1:
        res = res.unsqueeze(0)

    return res


def _build_cambrian_prompt_n(
    num_images: int,
    text_prompt: str,
    conv_templates,
    conv_mode: str = "qwen_2",
) -> str:
    if conv_mode not in conv_templates:
        raise ValueError(
            f"conv_mode='{conv_mode}' not found in conv_templates. "
            f"Available keys: {list(conv_templates.keys())}"
        )

    conv = conv_templates[conv_mode].copy()
    image_tokens = "\n".join(["<image>"] * int(num_images))
    user_msg = f"{image_tokens}\n{text_prompt}"
    conv.append_message(conv.roles[0], user_msg)
    conv.append_message(conv.roles[1], None)
    return conv.get_prompt()


# ===================== Evaluation loop =====================
def evaluate_video_ucsbench_cambrian(
    model,
    tokenizer,
    image_processor,
    frame_paths: List[Optional[str]],  # 1 fps: frame_paths[sec] = path or None
    qa_items: List[QAItem],
    num_samples: int = 32,  # K
    max_image_size: int = 512,
    conv_mode: str = "qwen_2",
    generation_config: Optional[Dict[str, Any]] = None,
    frame_select_mode: str = "uniform",  # "uniform" or "clip_window"
    clip_window_sec: int = 64,
    cambrian_fns: Optional[Dict[str, Any]] = None,
):
    if generation_config is None:
        generation_config = dict(
            max_new_tokens=32,
            do_sample=False,
            top_p=1.0,
            temperature=0.0,
        )

    fns = cambrian_fns or {}

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

    # Uniform mode: precompute indices for all QAs + cache in one pass.
    qid2indices: Dict[str, List[int]] = {}
    if frame_select_mode == "uniform":
        all_set = set()
        for qa in qa_items:
            idxs = compute_uniform_indices_1fps_fixedlen(qa.question_timestamp, frame_count, num_samples)
            qid2indices[qa.qid] = idxs
            all_set.update(idxs)

        frame_cache.update(
            build_frame_cache_pil_from_indexed_paths(
                frame_paths=frame_paths,
                all_indices=sorted(list(all_set)),
                max_image_size=max_image_size,
            )
        )

    per_question_results = []
    overall_stats = EvalResult()
    qtype_stats: Dict[str, EvalResult] = {}
    category_stats: Dict[str, EvalResult] = {}
    subcategory_stats: Dict[str, EvalResult] = {}
    qtype_category_stats: Dict[str, EvalResult] = {}

    for qa in qa_items:
        selected_secs: List[int] = []

        if frame_select_mode == "uniform":
            selected_secs = qid2indices.get(qa.qid, [])
        elif frame_select_mode == "clip_window":
            q_sec = int(np.floor(float(qa.question_timestamp)))
            q_sec = min(max(q_sec, 0), frame_count - 1)
            start = max(0, q_sec - int(clip_window_sec))
            end = q_sec

            selected_secs = collect_valid_secs_in_window(
                frame_paths=frame_paths,
                get_frame_fn=get_frame,
                start_sec=start,
                end_sec=end,
            )
        else:
            raise ValueError(f"Unknown frame_select_mode={frame_select_mode}")

        # Assemble exactly K frames (repeat-padding with the last frame).
        pil_images: List[Image.Image] = []
        missing = 0
        for sec in selected_secs:
            im = get_frame(sec)
            if im is None:
                missing += 1
                continue
            pil_images.append(im)

        if len(pil_images) == 0:
            print(f"[WARN] No frames decoded for QID {qa.qid}, skip.")
            continue

        if len(pil_images) < num_samples:
            last = pil_images[-1]
            pil_images.extend([last] * (num_samples - len(pil_images)))
        pil_images = pil_images[:num_samples]  # strictly K

        options_text = "\n".join([f"{k}. {v}" for k, v in qa.options.items()])
        if frame_select_mode == "uniform":
            frame_desc = "sampled uniformly from the start of the video"
        else:
            # Historical wording, kept byte-identical to the original runs.
            frame_desc = f"selected by CLIP within the last {clip_window_sec} seconds"

        text_prompt = (
            "You are an egocentric video QA assistant.\n"
            "This is a multiple-choice question answering task based strictly on visual content.\n"
            f"You will see a sequence of frames {frame_desc} "
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
            f"There are {num_samples} frames in temporal order.\n"
            "Keep answer short. Output format: answer is (). "
        )

        try:
            prompt = _build_cambrian_prompt_n(
                num_samples, text_prompt,
                conv_templates=fns["conv_templates"], conv_mode=conv_mode,
            )

            image_tensor, image_sizes = _cambrian_process_images(
                pil_images, image_processor, model, fns["process_images"],
            )
            input_ids = _cambrian_tokenize_prompt(
                prompt, tokenizer, fns["tokenizer_image_token"], fns["image_token_index"],
            ).to(model.device)

            with torch.no_grad():
                torch.cuda.empty_cache()
                output_ids = model.generate(
                    inputs=input_ids,
                    images=image_tensor,
                    image_sizes=image_sizes,
                    max_new_tokens=generation_config["max_new_tokens"],
                    do_sample=generation_config["do_sample"],
                    top_p=generation_config["top_p"],
                    temperature=generation_config["temperature"],
                    pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
                )
                # Move to CPU immediately and free GPU memory.
                output_ids_cpu = output_ids.cpu()
                del output_ids, image_tensor, input_ids
                torch.cuda.empty_cache()

            gen_ids = output_ids_cpu[0]
            response = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            pred_label = parse_option_from_answer(response)
            is_correct = pred_label == qa.answer_label

            print(
                f"[Q] {qa.qid} @ {qa.question_timestamp:.1f}s | mode={frame_select_mode} | "
                f"FramesIn={len(pil_images)} (missing={missing}) | Pred={pred_label} | GT={qa.answer_label} | {is_correct} | {response}"
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
                    "question_timestamp": qa.question_timestamp,
                    "num_samples": num_samples,
                    "missing_decoded": missing,
                    "frame_select_mode": frame_select_mode,
                    "selected_frame_indices_sec": selected_secs,  # at 1 fps, index == sec
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

    # Cambrian model args
    parser.add_argument(
        "--model_path",
        type=str,
        default="cambrian-s/cambrian-s-7b",
        help="Cambrian checkpoint path (local path or HF repo id)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="cambrian-s-7b",
        help="model_name passed to load_pretrained_model (e.g. cambrian-s-7b / cambrian-s-3b)",
    )
    parser.add_argument(
        "--cambrian_repo",
        type=str,
        default=None,
        help="Path to the Cambrian-S repository containing the `cambrian` package. "
             "Optional if the package is already importable (pip install -e / PYTHONPATH).",
    )
    parser.add_argument(
        "--device_map",
        type=str,
        default="auto",
        help='device_map for load_pretrained_model (e.g. "cuda" / "auto")',
    )
    parser.add_argument(
        "--conv_mode",
        type=str,
        default="qwen_2",
        help='key into conv_templates (e.g. "qwen_2")',
    )
    parser.add_argument("--load_8bit", action="store_true")
    parser.add_argument("--load_4bit", action="store_true")
    parser.add_argument("--use_flash_attn", action="store_true")

    # QA + frames
    parser.add_argument("--qa_dir", type=str, default="data/UCS-Bench/QAs")
    parser.add_argument(
        "--frames_dir",
        type=str,
        default="data/UCS-Bench/frames_1fps",
        help="Root directory containing per-video frame folders (each has index.jsonl + images)",
    )
    parser.add_argument("--index_name", type=str, default="index.jsonl")

    # output
    parser.add_argument(
        "--output",
        type=str,
        default="results/cambrian/64_uniform.json",
    )

    # common
    parser.add_argument("--num_samples", type=int, default=64, help="K (uniform sample count / stream cap)")
    parser.add_argument("--max_image_size", type=int, default=512)

    # frame selection strategy
    parser.add_argument(
        "--frame_select_mode",
        type=str,
        default="uniform",
        choices=["uniform", "clip_window"],
        help="uniform: sampling over [0, tq]; clip_window: frames in [tq-window, tq].",
    )
    parser.add_argument(
        "--clip_window_sec",
        type=int,
        default=64,
        help="Window length in seconds for clip_window mode (default: 64)",
    )

    parser.add_argument(
        "--resume_file",
        type=str,
        default=None,
        help="Path to an existing result file to resume from (dedup by qid)",
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

    (
        load_pretrained_model,
        process_images,
        tokenizer_image_token,
        conv_templates,
        IMAGE_TOKEN_INDEX,
    ) = _import_cambrian(args.cambrian_repo)

    print(f"[INFO] Loading Cambrian model: {args.model_path}")

    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path,
        None,
        args.model_name,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        use_flash_attn=args.use_flash_attn,
        device_map=args.device_map,
    )

    # Align IMAGE_TOKEN_INDEX with the checkpoint config when possible.
    for k in ["IMAGE_TOKEN_INDEX", "image_token_index", "im_token_index"]:
        if hasattr(model.config, k):
            IMAGE_TOKEN_INDEX = getattr(model.config, k)
            print(f"[INFO] Use IMAGE_TOKEN_INDEX from model.config.{k} = {IMAGE_TOKEN_INDEX}")
            break
    print(f"[INFO] Final IMAGE_TOKEN_INDEX = {IMAGE_TOKEN_INDEX}")

    model.eval()

    generation_config = dict(
        do_sample=False,
        max_new_tokens=32,
        top_p=1.0,
        temperature=0.0,
    )

    cambrian_fns = {
        "process_images": process_images,
        "tokenizer_image_token": tokenizer_image_token,
        "conv_templates": conv_templates,
        "image_token_index": IMAGE_TOKEN_INDEX,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    video2qas = load_qa_items_from_dir(args.qa_dir)

    all_results = []
    global_overall = EvalResult()
    global_qtype_stats: Dict[str, EvalResult] = {}
    global_category_stats: Dict[str, EvalResult] = {}
    global_subcategory_stats: Dict[str, EvalResult] = {}
    global_qtype_category_stats: Dict[str, EvalResult] = {}

    # ======= Resume / dedup by qid =======
    processed_qids = set()
    if args.resume_file and os.path.exists(args.resume_file):
        print(f"[INFO] Loading existing results from {args.resume_file}...")
        try:
            with open(args.resume_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)

            for item in existing_data["details"]:
                qid = item.get("qid")
                if not qid:
                    raise ValueError("none of qid")

                # The resume file itself must not contain duplicates.
                if qid in processed_qids:
                    raise ValueError("duplicates of qids")

                processed_qids.add(qid)
                all_results.append(item)

                # Reconstruct stats from the resumed entries.
                is_correct = item.get("is_correct", False)
                global_overall.add(is_correct)

                qt = item["qtype"]
                global_qtype_stats.setdefault(qt, EvalResult()).add(is_correct)

                cat = item["category"]
                global_category_stats.setdefault(cat, EvalResult()).add(is_correct)

                subcat = item["subcategory"]
                global_subcategory_stats.setdefault(subcat, EvalResult()).add(is_correct)

                key_qc = f"{qt}||{cat}"
                global_qtype_category_stats.setdefault(key_qc, EvalResult()).add(is_correct)

            print(f"[INFO] Resumed {len(processed_qids)} QAs from {args.resume_file}. Current Acc: {global_overall.acc*100:.2f}%")
        except Exception as e:
            print(f"[WARN] Failed to load resume file: {e}")

    print(f"[INFO] Start Evaluating {len(video2qas)} videos...")

    evaluated_videos = 0

    for uid, qas in tqdm(video2qas.items(), desc="Evaluating"):
        # Skip QAs that were already processed.
        qas_to_run = [qa for qa in qas if qa.qid not in processed_qids]
        if not qas_to_run:
            continue

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
            f"\n===== Evaluate video ({args.model_name}) : {uid} | "
            f"mode={args.frame_select_mode} | K={args.num_samples} | window={args.clip_window_sec}s ====="
        )

        (
            per_video_results,
            overall_stats,
            qtype_stats,
            category_stats,
            subcategory_stats,
            qtype_category_stats,
        ) = evaluate_video_ucsbench_cambrian(
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            frame_paths=frame_paths,
            qa_items=qas_to_run,
            num_samples=args.num_samples,
            max_image_size=args.max_image_size,
            conv_mode=args.conv_mode,
            generation_config=generation_config,
            frame_select_mode=args.frame_select_mode,
            clip_window_sec=args.clip_window_sec,
            cambrian_fns=cambrian_fns,
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
    print("===== Final Statistics (UCS-Bench + Cambrian) =====")
    print("=" * 50)
    print(f"Overall Accuracy: {global_overall.correct}/{global_overall.total} = {global_overall.acc * 100:.2f}%")

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
        q_str, c_str = key_qc.split("||", 1) if "||" in key_qc else (key_qc, "")
        print(f"  ({q_str}, {c_str}): {st.correct}/{st.total} = {st.acc * 100:.2f}%")

    out_obj = {
        "overall": {"correct": global_overall.correct, "total": global_overall.total, "acc": global_overall.acc},
        "by_qtype": {qt: {"correct": st.correct, "total": st.total, "acc": st.acc} for qt, st in global_qtype_stats.items()},
        "by_category": {cat: {"correct": st.correct, "total": st.total, "acc": st.acc} for cat, st in global_category_stats.items()},
        "by_subcategory": {subcat: {"correct": st.correct, "total": st.total, "acc": st.acc} for subcat, st in global_subcategory_stats.items()},
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
        "cambrian": {
            "model_path": args.model_path,
            "model_name": args.model_name,
            "conv_mode": args.conv_mode,
            "image_token_index": IMAGE_TOKEN_INDEX,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
