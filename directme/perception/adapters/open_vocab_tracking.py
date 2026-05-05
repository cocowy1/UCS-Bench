"""Open-vocabulary detection + segmentation + tracking adapter.

Composes three real models:
  * **YOLO-World** (Ultralytics)  → low-frequency open-vocab boxes.
  * **SAM 2**     (facebookresearch/sam2) → high-quality masks from boxes.
  * Lightweight IoU+appearance tracker → persistent track ids.

Heavy deps are lazy-imported so the core DirectMe package stays tiny.

References:
  https://github.com/AILab-CVC/YOLO-World  (and the Ultralytics wrapper)
  https://github.com/facebookresearch/sam2
  https://github.com/siyuanliii/masa  (more accurate alternative for tracking)

If you need stronger long-term association, swap the simple tracker here with a
``masa.MasaPredictor`` wrapper. The interface only requires `track(boxes, image,
embeddings) -> List[track_id]`, so substitution is local.
"""

from __future__ import annotations
import contextlib
import torch
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from directme.perception.runtime import resolve_runtime_device

def _load_classes_file(classes_file: str | Path) -> list[str]:
    """
    Load class names from txt/json/yaml.

    Supported formats:
      1. txt:
           Cup
           Chair
           Refrigerator

      2. json list:
           ["Cup", "Chair", "Refrigerator"]

      3. COCO/Object365 style json:
           {"categories": [{"id": 1, "name": "Person"}, ...]}

      4. Ultralytics yaml:
           names:
             0: Person
             1: Sneakers
             ...
    """
    import json

    path = Path(classes_file)
    if not path.exists():
        raise FileNotFoundError(f"Classes file does not exist: {path}")

    suffix = path.suffix.lower()

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as exc:
            raise ImportError(
                "Loading yaml class files requires PyYAML. "
                "Install with: pip install pyyaml"
            ) from exc

        data = yaml.safe_load(path.read_text(encoding="utf-8"))

        if not isinstance(data, dict) or "names" not in data:
            raise ValueError(f"YAML file does not contain a 'names' field: {path}")

        names = data["names"]

        # Ultralytics Object365.yaml format:
        # names:
        #   0: Person
        #   1: Sneakers
        if isinstance(names, dict):
            items = sorted(names.items(), key=lambda kv: int(kv[0]))
            return [str(v).strip() for _, v in items if str(v).strip()]

        # Alternative:
        # names: ["Person", "Sneakers", ...]
        if isinstance(names, list):
            return [str(v).strip() for v in names if str(v).strip()]

        raise ValueError(f"Unsupported YAML names format in: {path}")

    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))

        if isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                return [x.strip() for x in data if x.strip()]
            if all(isinstance(x, dict) and "name" in x for x in data):
                return [str(x["name"]).strip() for x in data if str(x["name"]).strip()]

        if isinstance(data, dict) and "categories" in data:
            cats = data["categories"]
            return [
                str(x["name"]).strip()
                for x in cats
                if "name" in x and str(x["name"]).strip()
            ]

        raise ValueError(f"Unsupported json classes format: {path}")

    classes = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        classes.append(line)

    return classes

def _default_objects365_yaml_path() -> Path:
    """Default Object365 yaml path next to this file."""
    return Path(__file__).resolve().with_name("Object.yaml")

def _resolve_test_classes(
    classes: str | None,
    classes_file: str | None,
    category_preset: str,
    class_limit: int,
) -> list[str] | None:
    if classes:
        out = [x.strip() for x in classes.split(",") if x.strip()]

    elif classes_file:
        out = _load_classes_file(classes_file)

    elif category_preset == "objects365":
        default_yaml = _default_objects365_yaml_path()
        if default_yaml.exists():
            out = _load_classes_file(default_yaml)
            print(f"[INFO] Loaded Objects365 classes from: {default_yaml}")
        else:
            raise FileNotFoundError(
                f"category_preset='objects365' expects Object.yaml at {default_yaml}. "
                "Please put Object.yaml next to open_vocab_tracking.py or pass --classes-file."
            )

    elif category_preset == "basic":
        out = [
            "Person", "Cup", "Chair", "Dining Table", "Bottle",
            "Cell Phone", "Book", "Backpack", "Sink", "Refrigerator",
        ]

    elif category_preset == "none":
        return None

    else:
        raise ValueError(f"Unknown category_preset: {category_preset}")

    if class_limit and class_limit > 0:
        out = out[:class_limit]

    # Deduplicate but keep order.
    seen = set()
    dedup = []
    for c in out:
        key = c.lower()
        if key not in seen:
            dedup.append(c)
            seen.add(key)

    return dedup


@dataclass
class Detection:
    label: str
    bbox_xyxy: tuple[float, float, float, float]
    score: float
    mask: np.ndarray | None = None      # (H, W) bool
    embedding: np.ndarray | None = None  # (D,) float32, optional
    track_id: str | None = None


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def _embedding_cosine(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None:
        return 0.0
    na = np.linalg.norm(a) + 1e-8
    nb = np.linalg.norm(b) + 1e-8
    return float(np.clip(np.dot(a, b) / (na * nb), -1.0, 1.0))


@dataclass
class _Track:
    track_id: str
    last_bbox: tuple[float, float, float, float]
    last_label: str
    embedding_ema: np.ndarray | None
    last_seen_frame: int


class SimpleIoUAppearanceTracker:
    """Greedy tracker combining IoU and appearance cosine similarity.

    Sufficient for slow-moving egocentric scenes. Replace with MASA for crowded
    or long-occlusion scenarios.
    """

    def __init__(
        self,
        iou_weight: float = 0.7,
        appearance_weight: float = 0.3,
        match_threshold: float = 0.30,
        max_age_frames: int = 30,
        embedding_alpha: float = 0.6,
    ):
        self.iou_weight = iou_weight
        self.appearance_weight = appearance_weight
        self.match_threshold = match_threshold
        self.max_age_frames = max_age_frames
        self.embedding_alpha = embedding_alpha
        self._tracks: dict[str, _Track] = {}
        self._next_id = 0

    def step(self, frame_index: int, detections: list[Detection]) -> list[Detection]:
        # Drop stale tracks.
        for tid in list(self._tracks):
            if frame_index - self._tracks[tid].last_seen_frame > self.max_age_frames:
                del self._tracks[tid]

        active = list(self._tracks.values())
        used_tracks: set[str] = set()
        for det in detections:
            best_score, best_track = 0.0, None
            for tr in active:
                if tr.track_id in used_tracks or tr.last_label != det.label:
                    continue
                iou = _box_iou(det.bbox_xyxy, tr.last_bbox)
                cos = _embedding_cosine(det.embedding, tr.embedding_ema)
                # Map cos from [-1, 1] to [0, 1] for combination.
                cos01 = 0.5 * (cos + 1.0)
                score = self.iou_weight * iou + self.appearance_weight * cos01
                if score > best_score:
                    best_score, best_track = score, tr
            if best_track is not None and best_score >= self.match_threshold:
                det.track_id = best_track.track_id
                used_tracks.add(best_track.track_id)
                if det.embedding is not None:
                    a = self.embedding_alpha
                    if best_track.embedding_ema is None:
                        best_track.embedding_ema = det.embedding.copy()
                    else:
                        best_track.embedding_ema = (
                            a * det.embedding + (1.0 - a) * best_track.embedding_ema
                        )
                best_track.last_bbox = det.bbox_xyxy
                best_track.last_seen_frame = frame_index
            else:
                tid = f"track_{self._next_id:06d}"
                self._next_id += 1
                self._tracks[tid] = _Track(
                    track_id=tid,
                    last_bbox=det.bbox_xyxy,
                    last_label=det.label,
                    embedding_ema=det.embedding.copy() if det.embedding is not None else None,
                    last_seen_frame=frame_index,
                )
                det.track_id = tid
        return detections


class YoloWorldDetector:
    """Open-vocabulary detector using Ultralytics YOLO-World."""

    def __init__(
        self,
        weights: str = "yolov8s-worldv2.pt",
        classes: list[str] | None = None,
        score_threshold: float = 0.20,
        device: str = "auto",
    ):
        try:
            from ultralytics import YOLOWorld
        except ImportError as exc:
            raise ImportError(
                "YoloWorldDetector requires `ultralytics>=8.3`. "
                "Install with `pip install ultralytics`."
            ) from exc

        self.model = YOLOWorld(weights)
        self.device = resolve_runtime_device(device)
        self.score_threshold = score_threshold
        if classes:
            self.set_classes(classes)

    def set_classes(self, classes: list[str]) -> None:
        self.model.set_classes(classes)

    def detect(self, image: Any) -> list[Detection]:
        result = self.model.predict(image, device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        cls = result.boxes.cls.cpu().numpy().astype(int)
        names = result.names
        out: list[Detection] = []
        for i in range(xyxy.shape[0]):
            if scores[i] < self.score_threshold:
                continue
            label = names[cls[i]] if isinstance(names, dict) else names[cls[i]]
            out.append(
                Detection(
                    label=str(label),
                    bbox_xyxy=tuple(float(v) for v in xyxy[i]),
                    score=float(scores[i]),
                )
            )
        return out


class Sam2MaskRefiner:
    """SAM 2 image predictor that converts boxes into high-quality masks."""

    def __init__(
        self,
        checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
        config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
        device: str = "auto",
    ):
        try:
            from sam2.build_sam import build_sam2
            from sam2.sam2_image_predictor import SAM2ImagePredictor
        except ImportError as exc:
            raise ImportError(
                "Sam2MaskRefiner requires `torch` and the upstream `sam2` package. "
                "See https://github.com/facebookresearch/sam2#installation"
            ) from exc

        self._build_sam2 = build_sam2
        self._SAM2ImagePredictor = SAM2ImagePredictor
        self.device = resolve_runtime_device(device)
        self.predictor = SAM2ImagePredictor(build_sam2(config, checkpoint, device=self.device))

    def refine(self, image: np.ndarray, detections: list[Detection]) -> list[Detection]:
        if not detections:
            return detections

        device_str = str(self.device)
        if device_str.startswith("cuda") and torch.cuda.is_available():
            autocast_ctx = torch.autocast("cuda", dtype=torch.bfloat16)
        else:
            autocast_ctx = contextlib.nullcontext()

        with torch.inference_mode(), autocast_ctx:
            self.predictor.set_image(image)
            for det in detections:
                box = np.array(det.bbox_xyxy, dtype=np.float32)[None, :]  # (1, 4)
                masks, _scores, _ = self.predictor.predict(
                    box=box,
                    multimask_output=False,
                )

                mask = np.asarray(masks)
                mask = np.squeeze(mask)

                if mask.ndim != 2:
                    raise RuntimeError(f"Unexpected SAM2 mask shape: raw={np.asarray(masks).shape}, squeezed={mask.shape}")

                det.mask = mask.astype(bool)


        return detections


class OpenVocabularyTrackingAdapter:
    """End-to-end open-vocab discovery → segmentation → tracking pipeline.

    Schedule: detect every ``detection_stride`` frames, segment every detected
    frame, track every frame using box propagation between detections (simple
    last-known-box hold + IoU re-association on next detection).
    """

    def __init__(
        self,
        detector: YoloWorldDetector,
        segmenter: Sam2MaskRefiner | None = None,
        tracker: SimpleIoUAppearanceTracker | None = None,
        detection_stride: int = 5,
    ):
        self.detector = detector
        self.segmenter = segmenter
        self.tracker = tracker or SimpleIoUAppearanceTracker()
        self.detection_stride = detection_stride
        self._last_detections: list[Detection] = []

    def step(self, frame_index: int, image: np.ndarray) -> list[Detection]:
        if frame_index % self.detection_stride == 0 or not self._last_detections:
            detections = self.detector.detect(image)
            if self.segmenter is not None and detections:
                detections = self.segmenter.refine(image, detections)
            self._last_detections = detections
        else:
            # Cheap hold strategy between detection frames: re-segment the last
            # boxes against the current image to refresh masks.
            detections = []
            for last in self._last_detections:
                detections.append(
                    Detection(label=last.label, bbox_xyxy=last.bbox_xyxy, score=last.score)
                )
            if self.segmenter is not None and detections:
                detections = self.segmenter.refine(image, detections)
        detections = self.tracker.step(frame_index, detections)
        return detections


def _test_natural_sort_key(path: Path):
    import re

    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _test_list_images_in_dir(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in suffixes
    ]
    return sorted(paths, key=_test_natural_sort_key)


def _test_find_first_frame_dir(root: str | Path) -> Path:
    root = Path(root).resolve()

    direct_images = _test_list_images_in_dir(root)
    if direct_images:
        return root

    for subdir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        images = _test_list_images_in_dir(subdir)
        if images:
            return subdir

    raise FileNotFoundError(f"No frame images found under: {root}")


def _test_resolve_spatialmemory_frame_dir(
    frame_root: str | Path,
    video_uid: str | None = None,
) -> Path:
    root = Path(frame_root).resolve()

    if video_uid:
        candidates = [
            root / video_uid,
            root / f"{video_uid}.mp4",
            root / f"{video_uid}.avi",
            root / f"{video_uid}.mkv",
        ]

        for cand in candidates:
            if cand.exists() and cand.is_dir() and _test_list_images_in_dir(cand):
                return cand

        for subdir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
            if subdir.name.startswith(video_uid) and _test_list_images_in_dir(subdir):
                return subdir

        raise FileNotFoundError(
            f"Cannot find frame directory for video_uid={video_uid} under {root}"
        )

    return _test_find_first_frame_dir(root)


def _test_sample_image_paths(
    image_paths: list[Path],
    max_images: int = 8,
    stride: int = 1,
) -> list[Path]:
    if not image_paths:
        raise ValueError("No image paths to sample.")

    stride = max(1, int(stride))
    image_paths = image_paths[::stride]

    if max_images is not None and max_images > 0 and len(image_paths) > max_images:
        indices = np.linspace(0, len(image_paths) - 1, num=max_images)
        indices = [int(round(x)) for x in indices]
        image_paths = [image_paths[i] for i in indices]

    return image_paths

def _color_from_track_id(track_id: str | None) -> tuple[int, int, int]:
    if track_id is None:
        return (255, 0, 0)
    h = abs(hash(track_id))
    return (
        50 + (h % 180),
        50 + ((h // 181) % 180),
        50 + ((h // 181 // 181) % 180),
    )


def _draw_and_save_detections(
    image_rgb: np.ndarray,
    detections: list[Detection],
    out_path: str | Path,
    draw_masks: bool = True,
) -> None:
    from PIL import Image, ImageDraw

    vis = image_rgb.copy()

    if draw_masks:
        for det in detections:
            if det.mask is None:
                continue
            color = np.array(_color_from_track_id(det.track_id), dtype=np.float32)
            mask = det.mask.astype(bool)
            if mask.shape[:2] != vis.shape[:2]:
                continue
            vis[mask] = (0.55 * vis[mask].astype(np.float32) + 0.45 * color).astype(np.uint8)

    pil = Image.fromarray(vis)
    draw = ImageDraw.Draw(pil)

    for det in detections:
        x1, y1, x2, y2 = det.bbox_xyxy
        color = _color_from_track_id(det.track_id)
        label = f"{det.label} {det.score:.2f}"
        if det.track_id:
            label += f" {det.track_id}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        draw.text((x1 + 2, max(0, y1 - 12)), label, fill=color)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pil.save(out_path)


def _detections_to_jsonable(detections: list[Detection]) -> list[dict[str, Any]]:
    out = []
    for det in detections:
        item = {
            "label": det.label,
            "bbox_xyxy": [float(x) for x in det.bbox_xyxy],
            "score": float(det.score),
            "track_id": det.track_id,
            "has_mask": det.mask is not None,
        }
        if det.mask is not None:
            ys, xs = np.where(det.mask.astype(bool))
            if len(xs) > 0:
                item["mask_bbox_xyxy"] = [
                    float(xs.min()),
                    float(ys.min()),
                    float(xs.max()),
                    float(ys.max()),
                ]
                item["mask_area"] = int(det.mask.astype(bool).sum())
            else:
                item["mask_bbox_xyxy"] = None
                item["mask_area"] = 0
        out.append(item)
    return out


def test_open_vocab_tracking_on_spatialmemory_frames(
    frame_root: str | Path = "/data/ywang/dataset/SpatialMemory/data_frames_1fps",
    video_uid: str | None = "scene0804_00-0",
    max_images: int = 8,
    stride: int = 1,
    weights: str = "yolov8s-worldv2.pt",
    classes: list[str] | None = None,
    score_threshold: float = 0.20,
    device: str = "auto",
    detection_stride: int = 1,
    use_sam2: bool = False,
    sam2_checkpoint: str = "checkpoints/sam2.1_hiera_large.pt",
    sam2_config: str = "configs/sam2.1/sam2.1_hiera_l.yaml",
    save_dir: str | Path = "/data/ywang/my_projects/VideoUnderstanding/Directme/directme/vis_output/open_vocab_tracking",
) -> list[dict[str, Any]]:
    from PIL import Image
    import json

    frame_dir = _test_resolve_spatialmemory_frame_dir(frame_root, video_uid)
    all_images = _test_list_images_in_dir(frame_dir)
    image_paths = _test_sample_image_paths(all_images, max_images=max_images, stride=stride)

    print(f"[INFO] SpatialMemory frame root: {Path(frame_root).resolve()}")
    print(f"[INFO] Selected frame dir: {frame_dir}")
    print(f"[INFO] Total frames in dir: {len(all_images)}")
    print(f"[INFO] Test frames: {len(image_paths)}")
    print(f"[INFO] YOLO-World weights: {weights}")
    print(f"[INFO] use_sam2: {use_sam2}")
    print(f"[INFO] detection_stride: {detection_stride}")
    print(f"[INFO] score_threshold: {score_threshold}")

    if classes is None:
        print("[INFO] classes: None, using model default classes.")
    else:
        print(f"[INFO] classes count: {len(classes)}")
        print(f"[INFO] classes preview: {classes[:20]}")

    detector = YoloWorldDetector(
        weights=weights,
        classes=classes,
        score_threshold=score_threshold,
        device=device,
    )

    segmenter = None
    if use_sam2:
        segmenter = Sam2MaskRefiner(
            checkpoint=sam2_checkpoint,
            config=sam2_config,
            device=device,
        )

    adapter = OpenVocabularyTrackingAdapter(
        detector=detector,
        segmenter=segmenter,
        tracker=SimpleIoUAppearanceTracker(),
        detection_stride=detection_stride,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[dict[str, Any]] = []

    for local_i, image_path in enumerate(image_paths):
        image_rgb = np.asarray(Image.open(image_path).convert("RGB"))

        detections = adapter.step(frame_index=local_i, image=image_rgb)
        json_dets = _detections_to_jsonable(detections)

        out_img = save_dir / f"tracking_{local_i:04d}.png"
        _draw_and_save_detections(
            image_rgb=image_rgb,
            detections=detections,
            out_path=out_img,
            draw_masks=use_sam2,
        )

        frame_record = {
            "frame_index": int(local_i),
            "image_path": str(image_path),
            "vis_path": str(out_img),
            "detections": json_dets,
        }
        all_results.append(frame_record)

        print(f"\n[Frame {local_i}] {image_path}")
        print(f"  detections: {len(detections)}")
        for det in detections[:10]:
            print(
                f"  - {det.label:20s} score={det.score:.3f} "
                f"track={det.track_id} bbox={tuple(round(x, 1) for x in det.bbox_xyxy)}"
            )
        if len(detections) > 10:
            print("  ...")

    json_path = save_dir / "tracking_results.json"
    json_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[OK] Open-vocabulary tracking smoke test passed.")
    print(f"[INFO] Visualization dir: {save_dir}")
    print(f"[INFO] JSON results: {json_path}")

    return all_results


def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test YOLO-World + optional SAM2 + tracking.")

    parser.add_argument(
        "--spatialmemory-frame-root",
        type=str,
        default="/data/ywang/dataset/SpatialMemory/data_frames_1fps",
    )
    parser.add_argument(
        "--video-uid",
        type=str,
        default="scene0804_00-0",
    )
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument(
        "--weights",
        type=str,
        default="./ckpts/yolo/yolov8m-worldv2.pt",
        help="YOLO-World weights, e.g. yolov8s-worldv2.pt / yolov8m-worldv2.pt.",
    )
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--detection-stride", type=int, default=1)

    parser.add_argument(
        "--category-preset",
        type=str,
        default="objects365",
        choices=["none", "basic", "coco", "objects365"],
        help="Class preset for YOLO-World set_classes.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help='Comma-separated classes, e.g. "cup,chair,table,phone". Overrides preset.',
    )
    parser.add_argument(
        "--classes-file",
        type=str,
        default="directme/perception/adapters/Object.yaml",
        help="Txt/json/yaml class names file. Use this for full Objects365 categories.",
    )
    parser.add_argument(
        "--class-limit",
        type=int,
        default=100,
        help="Limit number of classes for faster smoke test. Use 0 or negative for all.",
    )

    parser.add_argument(
        "--use-sam2",
        action="store_true",
        default=True,
        help="Enable SAM2 mask refinement after YOLO-World detection.",
    )

    parser.add_argument(
        "--sam2-checkpoint",
        type=str,
        default="./ckpts/sam2/sam2.1_hiera_base_plus.pt",
    )
    parser.add_argument(
        "--sam2-config",
        type=str,
        default="./configs/sam2.1/sam2.1_hiera_b+.yaml",
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/directme/vis_output/open_vocab_tracking",
    )

    args = parser.parse_args()

    classes = _resolve_test_classes(
        classes=args.classes,
        classes_file=args.classes_file,
        category_preset=args.category_preset,
        class_limit=args.class_limit,
    )

    test_open_vocab_tracking_on_spatialmemory_frames(
        frame_root=args.spatialmemory_frame_root,
        video_uid=args.video_uid,
        max_images=args.max_images,
        stride=args.stride,
        weights=args.weights,
        classes=classes,
        score_threshold=args.score_threshold,
        device=args.device,
        detection_stride=args.detection_stride,
        use_sam2=args.use_sam2,
        sam2_checkpoint=args.sam2_checkpoint,
        sam2_config=args.sam2_config,
        save_dir=args.save_dir,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
