#!/usr/bin/env python3
"""UCS-Bench baseline evaluation for VG-LLM (spatial-centric baseline).

Ported verbatim from the internal script used for the paper's baseline rows
(`VG-LLM/sm_eval_1fps.py`). The evaluation protocol is:

* Frames are pre-extracted at 1 fps (``<frames_dir>/<video_uid>/index.jsonl``).
* ``--frame_select_mode uniform``:
  sample K frames uniformly over [0, t_q] (dedup after index rounding).
* ``--frame_select_mode streaming_window`` (default; the mode behind the
  paper's VG-LLM row):
  the most recent K 1 fps frames before/at t_q, i.e. seconds
  [q-K+1, ..., q]. Optionally left-pad with frame 0 near the start of the
  video to keep exactly K frames (``--pad_streaming_window``; off by default,
  matching the reported run).

VG-LLM-specific notes:
* Requires the VG-LLM repository (the ``Qwen2_5_VLForConditionalGenerationWithVGGT``
  model class under ``src/qwen_vl``). Pass its checkout via ``--vgllm_repo``
  or make it importable. The reported row uses the QA-finetuned checkpoint
  (``vgllm-qa-vggt-8b``).
* Frames are resized so max(H, W) <= ``--max_image_size`` (default 412) and
  then cropped so W/H are divisible by the processor's patch/merge grid.
* The vision helpers from ``vg_llm.utils.vision`` are used when importable;
  otherwise the built-in fallbacks are used. **Our reported run used the
  fallbacks** (the official package was not installed), so the fallback path
  is the faithful one.

Select the GPU with the ``CUDA_VISIBLE_DEVICES`` environment variable. The
model is loaded on a single GPU (``device_map={"": 0}``).

Greedy decoding (temperature 0, max_new_tokens=16), seed 42. Questions whose
inference raises an exception are counted as *incorrect* (they stay in the
denominator), matching the original runs.
"""

import os
import sys
import re
import json
import copy
import argparse
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch

# Compatibility shim for flash-attn on torch<2.6
if not hasattr(torch.library, "wrap_triton"):
    torch.library.wrap_triton = lambda kernel: kernel

from PIL import Image
from tqdm.auto import tqdm

from transformers import AutoConfig, AutoTokenizer, AutoProcessor


def _import_vgllm(vgllm_repo: Optional[str]):
    """Make the VG-LLM model class importable and fetch the vision helpers."""
    if vgllm_repo:
        repo = os.path.abspath(os.path.expanduser(vgllm_repo))
        if repo not in sys.path:
            sys.path.insert(0, repo)

    from src.qwen_vl.model.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGenerationWithVGGT

    try:
        from vg_llm.utils.vision import extract_vision_info, load_and_preprocess_images
    except Exception:
        # Fallbacks (this is the path our reported run took).
        def extract_vision_info(message: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
            out = []
            for turn in message:
                content = turn.get("content", [])
                if isinstance(content, list):
                    for ele in content:
                        if isinstance(ele, dict) and ("image" in ele):
                            out.append(ele)
            return out

        def load_and_preprocess_images(images: List[Image.Image]) -> List[torch.Tensor]:
            feats = []
            for img in images:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                arr = np.array(img).astype(np.float32) / 255.0  # [H,W,3]
                t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # [3,H,W]
                feats.append(t)
            return feats

    return Qwen2_5_VLForConditionalGenerationWithVGGT, extract_vision_info, load_and_preprocess_images


# ===================== UID normalization =====================
def strip_mcq_uid(video_uid: str) -> str:
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


# ===================== Seed =====================
def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


# ===================== Image resize & patch alignment =====================
def resize_image_if_needed(image: Image.Image, max_size: int = 512) -> Image.Image:
    w, h = image.size
    max_dim = max(w, h)
    if max_dim > max_size:
        scale = max_size / max_dim
        nw = int(w * scale)
        nh = int(h * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
    return image


def resize_and_align_pil(
    image: Image.Image,
    max_size: int,
    patch_size: int,
    merge_size: int,
    min_divisor: Optional[int] = None,
) -> Image.Image:
    """Resize within max_size, then crop to a multiple of patch*merge."""
    image = resize_image_if_needed(image, max_size=max_size)

    if patch_size <= 0:
        return image

    if merge_size is None or merge_size <= 0:
        merge_size = 1

    divisor = patch_size * merge_size
    if min_divisor is not None:
        divisor = max(divisor, int(min_divisor))

    w, h = image.size
    new_w = (w // divisor) * divisor
    new_h = (h // divisor) * divisor

    new_w = max(divisor, new_w)
    new_h = max(divisor, new_h)

    new_w = min(new_w, w)
    new_h = min(new_h, h)

    if (new_w, new_h) != (w, h):
        # Crop the bottom-right (keeps the top-left region).
        image = image.crop((0, 0, new_w, new_h))
    return image


def align_tensor_hw_crop(
    x: torch.Tensor,
    patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """x: [3, H, W]; crop to patch multiples, then to merge-aligned patch grid."""
    if x.ndim != 3:
        return x
    _, H, W = x.shape
    if patch_size is None or patch_size <= 0:
        return x

    if merge_size is None or merge_size <= 0:
        merge_size = 1

    H2 = (H // patch_size) * patch_size
    W2 = (W // patch_size) * patch_size

    H2 = max(patch_size, H2)
    W2 = max(patch_size, W2)

    if merge_size > 1:
        h_p = H2 // patch_size
        w_p = W2 // patch_size
        if h_p % merge_size != 0:
            h_p2 = h_p - (h_p % merge_size)
            H2 = max(patch_size * merge_size, h_p2 * patch_size)
        if w_p % merge_size != 0:
            w_p2 = w_p - (w_p % merge_size)
            W2 = max(patch_size * merge_size, w_p2 * patch_size)

    H2 = min(H2, H)
    W2 = min(W2, W)

    return x[:, :H2, :W2]


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


# ===================== Data loading =====================
def timestamp_to_seconds(ts: str) -> float:
    try:
        parts = ts.split(":")
        if len(parts) == 3:
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        if len(parts) == 2:
            m, s = parts
            return int(m) * 60 + float(s)
        return float(ts)
    except Exception:
        return 0.0


def load_qa_items_from_dir(qa_dir: str) -> Dict[str, List[QAItem]]:
    video2qas: Dict[str, List[QAItem]] = {}
    uid2files: Dict[str, set] = {}

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


# ===================== Frame index.jsonl loader =====================
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
        f"Cannot find frame dir for uid={video_uid} (parsed_prefix={video_prefix}) in {frames_dir}"
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
                sec = int(round(float(t)))
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


# ===================== Sampling =====================
def compute_sample_indices_for_video_uniform(
    qa_items: List[QAItem],
    fps: float,
    frame_count: int,
    num_samples: int = 32,
) -> Tuple[Dict[str, List[int]], List[int]]:
    qid2indices: Dict[str, List[int]] = {}
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


def compute_sample_indices_for_video_streaming_window(
    qa_items: List[QAItem],
    frame_count: int,
    num_samples: int = 32,
    pad_to_n: bool = True,
) -> Tuple[Dict[str, List[int]], List[int]]:
    """
    The most recent N 1 fps frames before/at tq:
      q = floor(tq); indices = [q-N+1, ..., q], clipped to [0, frame_count-1].
    If pad_to_n=True and the window starts before the video start, left-pad by
    repeating frame 0 to keep length N.
    """
    qid2indices: Dict[str, List[int]] = {}
    all_indices_set = set()

    for qa in qa_items:
        if frame_count <= 0:
            indices = []
        else:
            q = int(np.floor(float(qa.question_timestamp)))
            q = min(max(q, 0), frame_count - 1)
            start = max(0, q - int(num_samples) + 1)
            indices = list(range(start, q + 1))

            if pad_to_n and len(indices) < num_samples:
                pad = [0] * (num_samples - len(indices))
                indices = pad + indices

        indices = [min(max(i, 0), frame_count - 1) for i in indices]
        qid2indices[qa.qid] = indices
        all_indices_set.update(indices)

    return qid2indices, sorted(list(all_indices_set))


def build_frame_cache_pil_from_indexed_paths(
    frame_paths: List[Optional[str]],
    all_indices: List[int],
    max_image_size: int,
    patch_size: int,
    merge_size: int,
) -> Dict[int, Image.Image]:
    cache: Dict[int, Image.Image] = {}
    if not all_indices:
        return cache

    n = len(frame_paths)
    for idx in all_indices:
        if idx < 0 or idx >= n:
            continue
        p = frame_paths[idx]
        if p is None:
            continue
        try:
            img = Image.open(p).convert("RGB")
            img = resize_and_align_pil(
                img,
                max_size=max_image_size,
                patch_size=patch_size,
                merge_size=merge_size,
            )
            cache[idx] = img
        except Exception as e:
            print(f"[WARN] Failed to read frame(sec={idx}): {p} ({e})")
            continue

    return cache


# ===================== Answer parsing =====================
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

    return raw


# ===================== VG-LLM inference wrapper =====================
class VGLLM_Inference:
    """Wraps the VG-LLM (Qwen2.5-VL + VGGT geometry encoder) checkpoint."""

    def __init__(
        self,
        pretrained: str,
        model_cls,
        extract_vision_info_fn,
        load_and_preprocess_images_fn,
        device_map: str = "auto",
        torch_dtype=torch.bfloat16,
        max_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
        attn_implementation: str = "sdpa",
    ):
        config = AutoConfig.from_pretrained(pretrained)
        self.model = model_cls.from_pretrained(
            pretrained,
            config=config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained, padding_side="left")

        proc_kwargs = {"padding_side": "left"}
        if max_pixels is not None:
            proc_kwargs["max_pixels"] = max_pixels
        if min_pixels is not None:
            proc_kwargs["min_pixels"] = min_pixels
        self.processor = AutoProcessor.from_pretrained(pretrained, **proc_kwargs)

        self._extract_vision_info = extract_vision_info_fn
        self._load_and_preprocess_images = load_and_preprocess_images_fn

    @torch.inference_mode()
    def call_model(
        self,
        contexts: List[str],
        visuals: List[Optional[List[Image.Image]]],
        add_frame_index: bool = False,
        gen_kwargs: Optional[dict] = None,
    ) -> List[str]:
        if gen_kwargs is None:
            gen_kwargs = {}

        messages = []

        for i, context in enumerate(contexts):
            message = [{"role": "system", "content": "You are a helpful assistant."}]

            visual = visuals[i] if (visuals is not None and i < len(visuals)) else None
            if visual is not None and isinstance(visual, (list, tuple)) and all(isinstance(v, Image.Image) for v in visual):
                image_content = []
                for j, v in enumerate(visual):
                    if add_frame_index:
                        image_content.append({"type": "text", "text": f"Frame-{j}: "})
                    image_content.append({"type": "image", "image": v})
                message.append({"role": "user", "content": image_content + [{"type": "text", "text": context}]})
            else:
                message.append({"role": "user", "content": [{"type": "text", "text": context}]})

            messages.append(message)

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        geometry_encoder_inputs = []
        image_inputs = []
        patch_size = getattr(self.processor.image_processor, "patch_size", 14)
        merge_size = getattr(self.processor.image_processor, "merge_size", 2)

        for message in messages:
            vision_info = self._extract_vision_info(message)
            cur_geometry_encoder_inputs = []

            for ele in vision_info:
                if "image" not in ele:
                    continue
                image = ele["image"]
                if not isinstance(image, Image.Image):
                    raise NotImplementedError(f"Unsupported image type: {type(image)}")

                # [3,H,W]
                geom_img = self._load_and_preprocess_images([image])[0]

                # Safety re-alignment of the tensor size.
                geom_img = align_tensor_hw_crop(geom_img, patch_size=patch_size, merge_size=merge_size)

                # The geometry encoder and the image encoder share the same
                # aligned input.
                cur_geometry_encoder_inputs.append(copy.deepcopy(geom_img))
                image_inputs.append(geom_img)

            if len(cur_geometry_encoder_inputs) > 0:
                geometry_encoder_inputs.append(torch.stack(cur_geometry_encoder_inputs))
            else:
                geometry_encoder_inputs.append(torch.empty(0))

        inputs = self.processor(
            text=text,
            images=image_inputs if len(image_inputs) > 0 else None,
            videos=None,
            padding=True,
            return_tensors="pt",
            do_rescale=False,
        )

        device = self.model.device
        if getattr(self.model.config, "use_geometry_encoder", False):
            inputs["geometry_encoder_inputs"] = [feat.to(device) for feat in geometry_encoder_inputs]
        inputs = inputs.to(device)

        gen_kwargs = dict(gen_kwargs)
        gen_kwargs.setdefault("max_new_tokens", 16)
        gen_kwargs.setdefault("temperature", 0.0)
        gen_kwargs.setdefault("top_p", None)
        gen_kwargs.setdefault("num_beams", 1)

        cont = self.model.generate(
            **inputs,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=True if (gen_kwargs["temperature"] is not None and gen_kwargs["temperature"] > 0) else False,
            temperature=gen_kwargs["temperature"],
            top_p=gen_kwargs["top_p"],
            num_beams=gen_kwargs["num_beams"],
            max_new_tokens=gen_kwargs["max_new_tokens"],
        )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, cont)
        ]
        answers = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return [a for a in answers]


# ===================== Evaluation per video =====================
def evaluate_video_ucsbench_vgllm(
    infer: VGLLM_Inference,
    frame_paths: List[Optional[str]],
    qa_items: List[QAItem],
    num_samples: int = 32,
    max_image_size: int = 512,
    generation_config: Optional[Dict[str, Any]] = None,
    frame_select_mode: str = "uniform",  # "uniform" or "streaming_window"
    pad_streaming_window: bool = True,
):
    if generation_config is None:
        generation_config = dict(
            max_new_tokens=16,
            temperature=0.0,
            top_p=None,
            num_beams=1,
        )

    fps = 1.0
    frame_count = len(frame_paths)
    if frame_count <= 0:
        print("[ERROR] Empty frames for this video.")
        return [], EvalResult(), {}, {}, {}, {}

    if frame_select_mode == "uniform":
        qid2indices, all_indices = compute_sample_indices_for_video_uniform(
            qa_items, fps, frame_count, num_samples
        )
    elif frame_select_mode == "streaming_window":
        qid2indices, all_indices = compute_sample_indices_for_video_streaming_window(
            qa_items, frame_count, num_samples, pad_to_n=pad_streaming_window
        )
    else:
        raise ValueError(f"Unknown frame_select_mode={frame_select_mode}")

    patch_size = getattr(infer.processor.image_processor, "patch_size", 14)
    merge_size = getattr(infer.processor.image_processor, "merge_size", 2)

    frame_cache = build_frame_cache_pil_from_indexed_paths(
        frame_paths,
        all_indices,
        max_image_size=max_image_size,
        patch_size=patch_size,
        merge_size=merge_size,
    )

    per_question_results = []
    overall_stats = EvalResult()
    qtype_stats: Dict[str, EvalResult] = {}
    category_stats: Dict[str, EvalResult] = {}
    subcategory_stats: Dict[str, EvalResult] = {}
    qtype_category_stats: Dict[str, EvalResult] = {}

    for qa in qa_items:
        indices = qid2indices.get(qa.qid, [])
        pil_images = [frame_cache[idx] for idx in indices if idx in frame_cache]

        if not pil_images:
            print(f"[WARN] No frames found for QID {qa.qid}, skip.")
            continue

        options_text = "\n".join([f"{k}. {v}" for k, v in qa.options.items()])

        sampling_desc = (
            "sampled uniformly from the start of the video"
            if frame_select_mode == "uniform"
            else f"the most recent {num_samples} frames at 1fps before/at the question time"
        )

        text_prompt = (
            "You are an egocentric video QA assistant.\n"
            "This is a multiple-choice question answering task based strictly on visual content.\n"
            f"You will see a sequence of frames ({sampling_desc}) "
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

        try:
            ans = infer.call_model(
                contexts=[text_prompt],
                visuals=[pil_images],
                add_frame_index=False,
                gen_kwargs=generation_config,
            )[0]

            pred_label = parse_option_from_answer(ans)
            is_correct = (pred_label == qa.answer_label)

            print(
                f"[Q] {qa.qid} @ {qa.question_timestamp:.1f}s | Mode: {frame_select_mode} | "
                f"Frames: {len(pil_images)} | Pred: {pred_label} | GT: {qa.answer_label} | {is_correct}"
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
                    "raw_response": ans,
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

    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the VG-LLM checkpoint (e.g. the QA-finetuned vgllm-qa-vggt-8b)")
    parser.add_argument("--vgllm_repo", type=str, default=None,
                        help="Path to the VG-LLM repository containing src/qwen_vl. "
                             "Optional if it is already importable (pip install -e / PYTHONPATH).")
    parser.add_argument("--qa_dir", type=str, default="data/UCS-Bench/QAs")
    parser.add_argument("--frames_dir", type=str, default="data/UCS-Bench/frames_1fps")
    parser.add_argument("--index_name", type=str, default="index.jsonl")
    parser.add_argument("--output", type=str, default="results/vgllm/64_streaming_window.json")

    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--max_image_size", type=int, default=412)

    parser.add_argument(
        "--frame_select_mode",
        type=str,
        default="streaming_window",
        choices=["uniform", "streaming_window"],
        help="uniform: sample N frames uniformly from [0,tq]; streaming_window: last N 1fps frames before/at tq.",
    )
    parser.add_argument(
        "--pad_streaming_window",
        action="store_true",
        help="If set, streaming_window left-pads with frame 0 to keep exactly N frames near the start.",
    )

    # generation
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.0)  # 0.0 -> None
    parser.add_argument("--num_beams", type=int, default=1)

    # processor pixel constraints (recommended for memory control)
    parser.add_argument("--max_pixels", type=int, default=None)
    parser.add_argument("--min_pixels", type=int, default=None)

    parser.add_argument(
        "--attn_impl",
        type=str,
        default="flash_attention_2",
        choices=["sdpa", "eager", "flash_attention_2"],
        help="Attention backend. Use sdpa/eager to avoid flash_attn installation issues.",
    )

    parser.add_argument("--print_every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    set_seed(args.seed)

    generation_config = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=(None if args.top_p == 0.0 else args.top_p),
        num_beams=args.num_beams,
    )

    model_cls, extract_vision_info_fn, load_and_preprocess_images_fn = _import_vgllm(args.vgllm_repo)

    print(f"[INFO] Loading VG-LLM inference wrapper: {args.model_path}")
    infer = VGLLM_Inference(
        pretrained=args.model_path,
        model_cls=model_cls,
        extract_vision_info_fn=extract_vision_info_fn,
        load_and_preprocess_images_fn=load_and_preprocess_images_fn,
        device_map={"": 0},
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
        attn_implementation=args.attn_impl,
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    video2qas = load_qa_items_from_dir(args.qa_dir)

    all_results = []
    global_overall = EvalResult()
    global_qtype_stats: Dict[str, EvalResult] = {}
    global_category_stats: Dict[str, EvalResult] = {}
    global_subcategory_stats: Dict[str, EvalResult] = {}
    global_qtype_category_stats: Dict[str, EvalResult] = {}

    print(f"[INFO] Start Evaluating {len(video2qas)} videos | mode={args.frame_select_mode} | K={args.num_samples}")
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

        print(f"\n===== Evaluate video: {uid} | mode={args.frame_select_mode} | K={args.num_samples} =====")

        (
            per_video_results,
            overall_stats,
            qtype_stats,
            category_stats,
            subcategory_stats,
            qtype_category_stats,
        ) = evaluate_video_ucsbench_vgllm(
            infer=infer,
            frame_paths=frame_paths,
            qa_items=qas,
            num_samples=args.num_samples,
            max_image_size=args.max_image_size,
            generation_config=generation_config,
            frame_select_mode=args.frame_select_mode,
            pad_streaming_window=args.pad_streaming_window,
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
    print("===== Final Statistics (UCS-Bench + VG-LLM) =====")
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
        q_str, c_str = key_qc.split("||", 1) if "||" in key_qc else (key_qc, "")
        print(f"  ({q_str}, {c_str}): {st.correct}/{st.total} = {st.acc * 100:.2f}%")

    out_obj = {
        "overall": {
            "correct": global_overall.correct,
            "total": global_overall.total,
            "acc": global_overall.acc,
        },
        "by_qtype": {qt: {"correct": st.correct, "total": st.total, "acc": st.acc} for qt, st in global_qtype_stats.items()},
        "by_category": {cat: {"correct": st.correct, "total": st.total, "acc": st.acc} for cat, st in global_category_stats.items()},
        "by_subcategory": {subcat: {"correct": st.correct, "total": st.total, "acc": st.acc} for subcat, st in global_subcategory_stats.items()},
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
            "pad_streaming_window": bool(args.pad_streaming_window),
        },
        "generation": generation_config,
        "model": {
            "checkpoint": args.model_path,
            "attn_impl": args.attn_impl,
            "max_pixels": args.max_pixels,
            "min_pixels": args.min_pixels,
        },
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, indent=2, ensure_ascii=False)

    print(f"\n[INFO] Detailed results saved to {args.output}")


if __name__ == "__main__":
    main()
