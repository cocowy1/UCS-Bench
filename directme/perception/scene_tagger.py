#!/usr/bin/env python3
"""Qwen-VL scene tagger smoke test for DirectMe.

This script tags SpatialMemory extracted frames with a lightweight VLM, e.g.
Qwen2-VL-2B-Instruct or Qwen2.5-VL small variants.

It supports stride-based tagging:
    frame_000000, frame_000005, frame_000010, ...

Outputs:
    qwen_scene_tags_sampled.json
    qwen_scene_tags_sampled.csv
    qwen_scene_tags_all.json      # optional when --propagate-to-all
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_SCENE_LABELS = [
    "kitchen",
    "living_room",
    "bedroom",
    "bathroom",
    "office",
    "corridor",
    "staircase",
    "storage_room",
    "dining_room",
    "classroom",
    "conference_room",
    "meeting_room",
    "lab",
    "workshop",
    "library",
    "lobby",
    "entrance",
    "elevator",
    "garage",
    "outdoor",
    "store",
    "unknown",
]


def _numeric_sort_key(path: Path):
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[-1]), path.name
    return 10**12, path.name


def _list_images(image_dir: str | Path, max_images: int | None = None) -> list[Path]:
    image_dir = Path(image_dir).resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    paths.sort(key=_numeric_sort_key)

    if max_images and max_images > 0:
        paths = paths[:max_images]

    if not paths:
        raise FileNotFoundError(f"No image files found in: {image_dir}")

    return paths


def _parse_labels(labels: str | None) -> list[str]:
    if not labels:
        return list(DEFAULT_SCENE_LABELS)
    out = [x.strip().lower().replace(" ", "_") for x in labels.split(",") if x.strip()]
    if "unknown" not in out:
        out.append("unknown")
    return out


def _normalize_scene_tag(text: str, allowed_labels: list[str]) -> str:
    s = str(text).strip().lower()
    s = s.replace("-", "_").replace(" ", "_")

    aliases = {
        "livingroom": "living_room",
        "living_room": "living_room",
        "dinning_room": "dining_room",
        "diningroom": "dining_room",
        "bath_room": "bathroom",
        "bed_room": "bedroom",
        "office_room": "office",
        "hallway": "corridor",
        "hall": "corridor",
        "stairs": "staircase",
        "storage": "storage_room",
    }
    s = aliases.get(s, s)

    if s in allowed_labels:
        return s

    # Fallback: scan raw text for allowed labels.
    raw = str(text).lower()
    for label in allowed_labels:
        if label.replace("_", " ") in raw or label in raw:
            return label

    return "unknown"


def _extract_json_object(raw: str) -> dict[str, Any] | None:
    raw = raw.strip()

    # Remove Markdown fences if the model produced them.
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.IGNORECASE).strip()
    raw = re.sub(r"```$", "", raw).strip()

    # Try direct JSON first.
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # Try first {...} span.
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if match:
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None

    return None


def _parse_qwen_response(raw: str, allowed_labels: list[str]) -> dict[str, Any]:
    obj = _extract_json_object(raw)

    if obj is not None:
        tag = _normalize_scene_tag(obj.get("scene_tag", "unknown"), allowed_labels)
        confidence = obj.get("confidence", None)
        try:
            confidence = float(confidence)
            confidence = max(0.0, min(1.0, confidence))
        except Exception:
            confidence = None

        evidence = obj.get("evidence", [])
        if isinstance(evidence, str):
            evidence = [evidence]
        if not isinstance(evidence, list):
            evidence = []

        reason = str(obj.get("brief_reason", "")).strip()

        return {
            "scene_tag": tag,
            "freeform_scene": str(obj.get("freeform_scene", "")).strip(),
            "confidence": confidence,
            "evidence": [str(x) for x in evidence],
            "brief_reason": reason,
            "parse_ok": True,
        }


    tag = _normalize_scene_tag(raw, allowed_labels)
    return {
        "scene_tag": tag,
        "freeform_scene": str(raw).strip()[:300],
        "confidence": None,
        "evidence": [],
        "brief_reason": "",
        "parse_ok": False,
    }



@dataclass
class SceneTagResult:
    frame_index: int
    image_path: str
    scene_tag: str
    freeform_scene: str
    confidence: float | None
    evidence: list[str]
    brief_reason: str
    raw_response: str
    parse_ok: bool

class QwenVLSceneTagger:
    """Minimal Qwen-VL image scene tagger."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 128,
        max_pixels: int = 512 * 512,
    ):
        import torch
        from transformers import AutoProcessor

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError as exc:
            raise ImportError(
                "QwenVLSceneTagger requires qwen-vl-utils. Install with:\n"
                "  pip install qwen-vl-utils\n"
            ) from exc

        self.process_vision_info = process_vision_info
        self.max_new_tokens = int(max_new_tokens)

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if self.device.startswith("cuda"):
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float32

        # Try Qwen2.5-VL first, then Qwen2-VL, then generic fallback.
        model_cls = None
        import transformers

        for cls_name in [
            "Qwen3VLForConditionalGeneration",
            "Qwen2_5_VLForConditionalGeneration",
            "Qwen2VLForConditionalGeneration",
            "AutoModelForVision2Seq",
        ]:

            if hasattr(transformers, cls_name):
                model_cls = getattr(transformers, cls_name)
                break

        if model_cls is None:
            raise ImportError(
                "Your transformers version does not expose Qwen VL model classes. "
                "Try upgrading transformers."
            )

        print(f"[QwenSceneTagger] model_path = {model_path}")
        print(f"[QwenSceneTagger] model_cls = {model_cls.__name__}")
        print(f"[QwenSceneTagger] device = {self.device}")
        print(f"[QwenSceneTagger] max_pixels = {max_pixels}")

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                min_pixels=224 * 224,
                max_pixels=int(max_pixels),
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )

        try:
            self.model = model_cls.from_pretrained(
                model_path,
                dtype=torch_dtype,
                trust_remote_code=True,
            )
        except TypeError:
            self.model = model_cls.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )

        self.model.to(self.device)
        self.model.eval()

    def _build_prompt(self, allowed_labels: list[str]) -> str:
        label_text = ", ".join(allowed_labels)

        return (
            "You are a scene / landmark tagger for egocentric indoor video frames.\n"
            "Classify the image into exactly one scene tag from this candidate list:\n"
            f"{label_text}\n\n"
            "If the scene is ambiguous or does not match any candidate, choose unknown.\n"
            "Also provide a short free-form scene description.\n\n"
            "Return strict JSON only, with this schema:\n"
            "{\n"
            '  "scene_tag": "one_label_from_the_candidate_list",\n'
            '  "freeform_scene": "short natural language scene description",\n'
            '  "confidence": 0.0,\n'
            '  "evidence": ["short visible evidence"],\n'
            '  "brief_reason": "one short sentence"\n'
            "}\n"
            "Do not output Markdown. Do not output any extra text."
        )


    def infer_image(self, image_path: str | Path, allowed_labels: list[str]) -> dict[str, Any]:
        import torch

        image_path = Path(image_path).resolve()
        prompt = self._build_prompt(allowed_labels)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": str(image_path)},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        image_inputs, video_inputs = self.process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        # Qwen3-VL examples often remove token_type_ids before generate.
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Decode only generated continuation, not the prompt tokens.
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        raw = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip()

        parsed = _parse_qwen_response(raw, allowed_labels)
        parsed["raw_response"] = raw
        return parsed


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_index",
        "image_path",
        "scene_tag",
        "freeform_scene",
        "confidence",
        "evidence",
        "brief_reason",
        "parse_ok",
    ]


    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = {k: r.get(k, "") for k in fieldnames}
            row["evidence"] = json.dumps(row["evidence"], ensure_ascii=False)
            writer.writerow(row)


def _propagate_tags_to_all_frames(
    image_paths: list[Path],
    sampled_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Propagate sampled tags to unsampled frames using nearest previous sample."""
    sampled_by_index = {int(r["frame_index"]): r for r in sampled_records}

    all_records = []
    current = None

    for i, p in enumerate(image_paths):
        if i in sampled_by_index:
            current = sampled_by_index[i]
            all_records.append(
                {
                    **current,
                    "image_path": str(p),
                    "source_sample_frame_index": int(current["frame_index"]),
                    "is_sampled": True,
                }
            )
            continue

        if current is None:
            # This should not happen if frame 0 is sampled, but keep robust.
            all_records.append(
            {
                "frame_index": i,
                "image_path": str(p),
                "scene_tag": "unknown",
                "freeform_scene": "",
                "confidence": None,
                "evidence": [],
                "brief_reason": "No previous sampled scene tag available.",
                "raw_response": "",
                "parse_ok": False,
                "source_sample_frame_index": None,
                "is_sampled": False,
            }

            )
        else:
            all_records.append(
                {
                    "frame_index": i,
                    "image_path": str(p),
                    "scene_tag": current["scene_tag"],
                    "freeform_scene": current.get("freeform_scene", ""),
                    "confidence": current.get("confidence", None),
                    "evidence": current.get("evidence", []),
                    "brief_reason": current.get("brief_reason", ""),
                    "raw_response": current.get("raw_response", ""),
                    "parse_ok": current.get("parse_ok", False),
                    "source_sample_frame_index": int(current["frame_index"]),
                    "is_sampled": False,
                }

            )

    return all_records


def main() -> int:
    parser = argparse.ArgumentParser(description="Stride-based Qwen-VL scene tagger test.")

    parser.add_argument(
        "--image-dir",
        type=str,
        default="/data/ywang/dataset/SpatialMemory/data_frames_1fps/scene0804_00-0",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/qwen3-vl-2B",
        help="Local path or HF id of Qwen3-VL-2B / Qwen2.5-VL model.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-pixels", type=int, default=512 * 512)

    parser.add_argument(
        "--labels",
        type=str,
        default=",".join(DEFAULT_SCENE_LABELS),
        help="Comma-separated scene labels.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/tmp/qwen_scene_tagger",
    )
    parser.add_argument(
        "--propagate-to-all",
        action="store_false",
        default=True,
        help="Fill non-sampled frames using nearest previous sampled tag.",
    )

    args = parser.parse_args()

    stride = max(1, int(args.stride))
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    allowed_labels = _parse_labels(args.labels)
    image_paths = _list_images(args.image_dir, max_images=args.max_images)

    sampled = [
        (i, p)
        for i, p in enumerate(image_paths)
        if i % stride == 0
    ]

    print(f"[INFO] image_dir: {Path(args.image_dir).resolve()}")
    print(f"[INFO] n_images: {len(image_paths)}")
    print(f"[INFO] stride: {stride}")
    print(f"[INFO] n_sampled: {len(sampled)}")
    print(f"[INFO] allowed_labels: {allowed_labels}")
    print(f"[INFO] output_dir: {output_dir}")

    tagger = QwenVLSceneTagger(
        model_path=args.model_path,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
        max_pixels=args.max_pixels,
    )

    sampled_records: list[dict[str, Any]] = []

    for frame_index, image_path in sampled:
        print(f"\n[SceneTag] frame={frame_index} image={image_path}")

        parsed = tagger.infer_image(image_path, allowed_labels)

        result = SceneTagResult(
            frame_index=int(frame_index),
            image_path=str(image_path),
            scene_tag=str(parsed["scene_tag"]),
            freeform_scene=str(parsed.get("freeform_scene", "")),
            confidence=parsed.get("confidence", None),
            evidence=parsed.get("evidence", []),
            brief_reason=str(parsed.get("brief_reason", "")),
            raw_response=str(parsed.get("raw_response", "")),
            parse_ok=bool(parsed.get("parse_ok", False)),
        )


        record = asdict(result)
        sampled_records.append(record)

        print(
            f"  tag={result.scene_tag} "
            f"conf={result.confidence} "
            f"parse_ok={result.parse_ok}"
        )
        if result.freeform_scene:
            print(f"  freeform_scene={result.freeform_scene}")

        if result.evidence:
            print(f"  evidence={result.evidence}")
        if result.brief_reason:
            print(f"  reason={result.brief_reason}")

    sampled_json = output_dir / "qwen_scene_tags_sampled.json"
    sampled_csv = output_dir / "qwen_scene_tags_sampled.csv"

    _write_json(sampled_json, sampled_records)
    _write_csv(sampled_csv, sampled_records)

    summary = {
        "status": "ok",
        "image_dir": str(Path(args.image_dir).resolve()),
        "model_path": args.model_path,
        "n_images": len(image_paths),
        "stride": stride,
        "n_sampled": len(sampled_records),
        "allowed_labels": allowed_labels,
        "sampled_json": str(sampled_json),
        "sampled_csv": str(sampled_csv),
    }

    if args.propagate_to_all:
        all_records = _propagate_tags_to_all_frames(image_paths, sampled_records)
        all_json = output_dir / "qwen_scene_tags_all.json"
        all_csv = output_dir / "qwen_scene_tags_all.csv"

        _write_json(all_json, all_records)
        _write_csv(all_csv, all_records)

        summary["all_json"] = str(all_json)
        summary["all_csv"] = str(all_csv)

    summary_path = output_dir / "qwen_scene_tagger_summary.json"
    _write_json(summary_path, summary)

    print("\n[OK] Qwen-VL scene tagger test passed.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
