from __future__ import annotations

"""Runtime helpers for optional heavy perception backbones.

This module centralizes two things:

* device resolution (``auto`` -> ``cuda`` / ``mps`` / ``cpu``), and
* construction of the default real DirectMe backend.

Keeping this logic in one place avoids drift between the CLI, examples, and
individual adapters.
"""

from collections.abc import Sequence


def resolve_runtime_device(device: str = "auto") -> str:
    """Resolve ``device`` into a concrete runtime string.

    ``auto`` prefers CUDA, then MPS, then CPU. Explicit unavailable GPU/MPS
    requests fall back to CPU so community users can still run smoke tests on
    laptops without editing code.
    """
    try:
        import torch
    except Exception:
        return "cpu" if device == "auto" else device

    requested = str(device).strip().lower() or "auto"
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        return "cpu"
    return device


def build_composed_backend(
    classes: Sequence[str],
    *,
    device: str = "auto",
    depth_model: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    use_ray_pose: bool = False,
    process_res: int = 504,
    yolo_weights: str = "yolov8s-worldv2.pt",
    score_threshold: float = 0.20,
    detection_stride: int = 5,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    min_pose_confidence: float = 0.30,
):
    """Build the default DA3 + YOLO-World + optional SAM2 backend.

    SAM 2 is optional. When it is absent, DirectMe falls back to bbox-center
    depth unprojection, which is less precise than mask-centric geometry but
    makes the project much easier to run in the community.
    """
    from directme.perception.adapters.composed import ComposedRealBackend
    from directme.perception.adapters.depth_anything3 import DepthAnything3Adapter
    from directme.perception.adapters.open_vocab_tracking import (
        OpenVocabularyTrackingAdapter,
        Sam2MaskRefiner,
        SimpleIoUAppearanceTracker,
        YoloWorldDetector,
    )

    resolved_device = resolve_runtime_device(device)
    classes_list = [c.strip() for c in classes if str(c).strip()]
    depth = DepthAnything3Adapter(
        model_id=depth_model,
        device=resolved_device,
        use_ray_pose=use_ray_pose,
        process_res=process_res,
    )
    detector = YoloWorldDetector(
        weights=yolo_weights,
        classes=classes_list,
        score_threshold=score_threshold,
        device=resolved_device,
    )
    segmenter = None
    if sam2_checkpoint and sam2_config:
        segmenter = Sam2MaskRefiner(
            checkpoint=sam2_checkpoint,
            config=sam2_config,
            device=resolved_device,
        )
    tracker = OpenVocabularyTrackingAdapter(
        detector=detector,
        segmenter=segmenter,
        tracker=SimpleIoUAppearanceTracker(),
        detection_stride=detection_stride,
    )
    return ComposedRealBackend(
        depth=depth,
        tracker=tracker,
        min_pose_confidence=min_pose_confidence,
    )
