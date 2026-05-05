"""SCAL3R adapter for DirectMe perception.

SCAL3R (zju3dv/Scal3R) is a long-sequence 3D reconstruction backend. Its
public inference entrypoint writes a result directory containing:

* ``mat.txt``: one camera-to-world transform per frame, row-major 4x4.
* ``intri.yml`` / ``extri.yml``: EasyVolcap/OpenCV-style camera files.
* ``depths/``: optional depth maps when ``--save_dpt 1`` is used.

DirectMe needs exactly the same geometry signal described in the paper's
offline stage: per-frame camera pose, depth, intrinsics, and tracked semantic
objects. This module therefore provides:

* :class:`Scal3ROutputReader` for precomputed SCAL3R results.
* :class:`Scal3RRunner` to call ``python -m scal3r.run`` on a chunk directory.
* :class:`Scal3RDepthPoseAdapter` to expose SCAL3R as a depth+pose backend.
* :class:`Scal3RComposedBackend` to combine SCAL3R with DirectMe's open-vocab
  tracker and scene classifier into a full :class:`PerceptionBackend`.

The code keeps heavy dependencies lazy. Importing this file does not require
SCAL3R, OpenCV, PyTorch, SAM 2, or Ultralytics; those are needed only when the
corresponding execution path is used.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from directme.geometry.poses import SE3
from directme.perception.base import (
    ChunkPerception,
    FramePerception,
    ObjectObservation,
    PerceptionBackend,
    VideoFrame,
)
from directme.perception.color import dominant_hsv_color, hsv_histogram_from_image_mask
from directme.perception.scene_classifier import (
    RuleBasedSceneClassifier,
    SceneClassifier,
)


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
_DEPTH_SUFFIXES = {".npy", ".npz", ".exr", ".png", ".tif", ".tiff"}


@dataclass
class Scal3RChunkOutput:
    """Per-chunk SCAL3R output, indexed by frame order in the chunk."""

    poses_local: list[SE3]
    pose_confidences: list[float]
    depth_maps: list[np.ndarray] | None = None
    intrinsics: list[np.ndarray] | None = None
    pose_scales: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Scal3RFrameOutput:
    """Single-frame depth+pose record returned by :class:`Scal3RDepthPoseAdapter`."""

    pose_local: SE3
    pose_confidence: float = 1.0
    depth: np.ndarray | None = None
    intrinsics: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    """Sort paths by the last integer in the stem, then by name."""
    nums = re.findall(r"\d+", path.stem)
    if nums:
        return int(nums[-1]), path.name
    return 10**12, path.name


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Project a nearly-rotation matrix to SO(3)."""
    u, _s, vh = np.linalg.svd(matrix)
    r = u @ vh
    if np.linalg.det(r) < 0:
        u[:, -1] *= -1.0
        r = u @ vh
    return r


def sim3_or_se3_to_se3(c2w: np.ndarray, *, normalize_scale: bool = True) -> tuple[SE3, float]:
    """Convert a SCAL3R camera-to-world matrix into DirectMe's SE3 convention.

    SCAL3R writes ``mat.txt`` from the optimized camera-to-world transforms.
    Depending on alignment, the upper-left 3x3 block can contain a small global
    Sim(3) scale. DirectMe's :class:`SE3` validator requires a pure rotation, so
    we remove that scale and orthogonalize the rotation.

    Returns:
        ``(pose, scale)`` where ``pose`` is ``T_local_from_camera`` and ``scale``
        is the removed positive scale factor. The translation is left untouched;
        SCAL3R's exported depth maps are already scale-adjusted by its result
        writer when ``--save_dpt`` is enabled.
    """
    arr = np.asarray(c2w, dtype=np.float64)
    if arr.shape == (16,):
        arr = arr.reshape(4, 4)
    if arr.shape == (12,):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :4] = arr.reshape(3, 4)
        arr = mat
    if arr.shape == (3, 4):
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :4] = arr
        arr = mat
    if arr.shape != (4, 4):
        raise ValueError(f"SCAL3R pose must be 4x4, 3x4, 16, or 12 values; got {arr.shape}")

    rot_raw = arr[:3, :3]
    det = float(np.linalg.det(rot_raw))
    scale = 1.0
    rot = rot_raw
    if normalize_scale:
        if np.isfinite(det) and det > 1e-12:
            scale = float(np.cbrt(det))
            rot = rot_raw / scale
        else:
            # Bad determinant: still try to make a valid rotation, but surface
            # scale=1.0 in metadata so users can inspect the issue.
            scale = 1.0
    rot = _project_to_rotation(rot)

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = rot
    mat[:3, 3] = arr[:3, 3]
    return SE3(mat), scale


def _read_mat_txt(mat_path: Path, *, normalize_scale: bool = True) -> tuple[list[SE3], list[float]]:
    if not mat_path.exists():
        raise FileNotFoundError(f"Expected SCAL3R poses at {mat_path}")
    rows = np.loadtxt(mat_path, dtype=np.float64)
    if rows.ndim == 1:
        rows = rows.reshape(1, -1)
    if rows.shape[1] not in (12, 16):
        raise ValueError(f"Each row of mat.txt must have 12 or 16 floats, got shape {rows.shape}")

    poses: list[SE3] = []
    scales: list[float] = []
    for row in rows:
        pose, scale = sim3_or_se3_to_se3(row, normalize_scale=normalize_scale)
        poses.append(pose)
        scales.append(scale)
    return poses, scales


def _read_confidence_file(path: Path, n: int) -> list[float]:
    if not path.exists():
        return [1.0] * n
    confs = np.loadtxt(path, dtype=np.float32).reshape(-1).tolist()
    if len(confs) != n:
        raise ValueError(f"{path.name} has {len(confs)} rows but mat.txt has {n}")
    return [float(np.clip(c, 0.0, 1.0)) for c in confs]


def _read_depth(path: Path) -> np.ndarray:
    """Read one SCAL3R depth map.

    SCAL3R currently saves OpenEXR files. For smoke tests and custom exporters
    we also accept ``.npy`` / ``.npz`` / integer image formats.
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path)
    elif suffix == ".npz":
        payload = np.load(path)
        if "depth" in payload:
            arr = payload["depth"]
        else:
            first_key = sorted(payload.files)[0]
            arr = payload[first_key]
    else:
        # OpenCV needs this environment variable before importing cv2 on many
        # wheels. Setting it here is harmless for non-EXR formats.
        os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
        try:
            import cv2  # type: ignore

            arr = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if arr is None:
                raise ValueError(f"cv2.imread returned None for {path}")
        except Exception:
            try:
                import imageio.v3 as iio  # type: ignore
            except Exception as exc:
                raise ImportError(
                    f"Could not read depth file {path}. Install opencv-python with EXR support "
                    "or imageio, or export depths as .npy."
                ) from exc
            arr = iio.imread(path)
    arr = np.asarray(arr)
    if arr.ndim == 3:
        # EXR is usually HxWx1; some readers return HxWxC. Use the first channel.
        arr = arr[..., 0]
    arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"Depth map must be HxW after loading, got {arr.shape} from {path}")
    return arr


def _read_depth_dir(depth_dir: Path) -> list[np.ndarray] | None:
    if not depth_dir.is_dir():
        return None
    depth_files = [p for p in depth_dir.iterdir() if p.suffix.lower() in _DEPTH_SUFFIXES]
    depth_files.sort(key=_numeric_sort_key)
    if not depth_files:
        return None
    return [_read_depth(p) for p in depth_files]


def _read_intrinsics_opencv(intri_path: Path) -> list[np.ndarray] | None:
    try:
        import cv2  # type: ignore
    except Exception:
        return None
    try:
        fs = cv2.FileStorage(str(intri_path), cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            return None
        names_node = fs.getNode("names")
        names: list[str] = []
        if not names_node.empty():
            for i in range(int(names_node.size())):
                names.append(str(names_node.at(i).string()))
        if not names:
            fs.release()
            return None
        out: list[np.ndarray] = []
        for name in names:
            mat = fs.getNode(f"K_{name}").mat()
            if mat is None:
                continue
            out.append(np.asarray(mat, dtype=np.float32))
        fs.release()
        return out or None
    except Exception:
        return None


def _read_intrinsics_text(intri_path: Path) -> list[np.ndarray] | None:
    text = intri_path.read_text(encoding="utf-8", errors="ignore")
    pattern = re.compile(
        r"^K_([^:\n]+):\s*!!opencv-matrix\s*\n"
        r"\s*rows:\s*(\d+)\s*\n"
        r"\s*cols:\s*(\d+)\s*\n"
        r"\s*dt:\s*[^\n]+\n"
        r"\s*data:\s*\[([^\]]*)\]",
        flags=re.MULTILINE,
    )
    items: list[tuple[tuple[int, str], np.ndarray]] = []
    for match in pattern.finditer(text.replace("\r\n", "\n")):
        name = match.group(1).strip()
        rows = int(match.group(2))
        cols = int(match.group(3))
        data = [float(x) for x in re.split(r"[,\s]+", match.group(4).strip()) if x]
        if len(data) != rows * cols:
            continue
        mat = np.asarray(data, dtype=np.float32).reshape(rows, cols)
        if mat.shape == (3, 3):
            nums = re.findall(r"\d+", name)
            order = (int(nums[-1]) if nums else 10**12, name)
            items.append((order, mat))
    if not items:
        return None
    items.sort(key=lambda x: x[0])
    return [mat for _order, mat in items]


def _read_intrinsics(intri_path: Path) -> list[np.ndarray] | None:
    if not intri_path.exists():
        return None
    return _read_intrinsics_opencv(intri_path) or _read_intrinsics_text(intri_path)


def _load_image_rgb(path: str | Path) -> np.ndarray:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Scal3RComposedBackend requires opencv-python to read RGB frames. "
            "Install with `pip install opencv-python`."
        ) from exc
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _ensure_rgb_frame_path(frame: VideoFrame, out_path: Path) -> None:
    """Stage a frame as an image file for SCAL3R.

    Prefer symlinks/copies when ``image_path`` is available. If the frame only
    carries an in-memory image, write it as PNG.
    """
    if frame.image_path:
        src = Path(frame.image_path)
        if not src.exists():
            raise FileNotFoundError(f"VideoFrame.image_path does not exist: {src}")
        try:
            os.symlink(src.resolve(), out_path)
        except Exception:
            shutil.copy2(src, out_path)
        return

    if frame.image is None:
        raise ValueError("Each VideoFrame must provide either image_path or image for SCAL3R")
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise ImportError("Writing in-memory frames requires opencv-python") from exc
    image = np.asarray(frame.image)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"VideoFrame.image must be HxWx3/4 RGB/RGBA, got {image.shape}")
    if image.shape[2] == 4:
        image = image[..., :3]
    bgr = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2BGR)
    ok = cv2.imwrite(str(out_path), bgr)
    if not ok:
        raise IOError(f"Could not write staged frame to {out_path}")


class Scal3ROutputReader:
    """Read SCAL3R outputs from a result directory.

    Layout accepted::

        result_dir/
            mat.txt              # N rows of 16 or 12 floats, camera-to-world
            intri.yml            # optional EasyVolcap/OpenCV intrinsics
            confidence.txt       # optional, N scalars in [0, 1]
            depths/              # optional, 000000.exr / .npy / .npz / image

    The returned poses are DirectMe-compatible ``T_local_from_camera`` matrices.
    """

    def __init__(
        self,
        *,
        depth_subdir: str = "depths",
        confidence_filename: str = "confidence.txt",
        intrinsics_filename: str = "intri.yml",
        normalize_pose_scale: bool = True,
    ):
        self.depth_subdir = depth_subdir
        self.confidence_filename = confidence_filename
        self.intrinsics_filename = intrinsics_filename
        self.normalize_pose_scale = normalize_pose_scale

    def read(self, result_dir: str | Path, *, expected_frames: int | None = None) -> Scal3RChunkOutput:
        result_path = Path(result_dir)
        poses, scales = _read_mat_txt(
            result_path / "mat.txt",
            normalize_scale=self.normalize_pose_scale,
        )
        n = len(poses)
        if expected_frames is not None and n != expected_frames:
            raise ValueError(
                f"SCAL3R result contains {n} poses but {expected_frames} frames were expected"
            )

        pose_confidences = _read_confidence_file(result_path / self.confidence_filename, n)

        intrinsics = _read_intrinsics(result_path / self.intrinsics_filename)
        if intrinsics is not None and len(intrinsics) != n:
            raise ValueError(f"intrinsics count ({len(intrinsics)}) != pose count ({n})")

        depth_maps = _read_depth_dir(result_path / self.depth_subdir)
        if depth_maps is not None and len(depth_maps) != n:
            raise ValueError(f"depth count ({len(depth_maps)}) != pose count ({n})")

        return Scal3RChunkOutput(
            poses_local=poses,
            pose_confidences=pose_confidences,
            depth_maps=depth_maps,
            intrinsics=intrinsics,
            pose_scales=scales,
            metadata={
                "source": "scal3r",
                "result_dir": str(result_path),
                "has_depth": depth_maps is not None,
                "has_intrinsics": intrinsics is not None,
            },
        )


@dataclass
class Scal3RRunner:
    """Thin wrapper around SCAL3R's public inference CLI."""

    config: str = "configs/models/scal3r.yaml"
    checkpoint: str | None = None
    device: str | None = None
    block_size: int | None = None
    overlap_size: int | None = None
    use_loop: int | None = None
    use_xyz_align: int | None = None
    save_dpt: int = 1
    save_xyz: int = 0
    offload_batches: int | None = None
    offload_outputs: int | None = None
    streaming_state: int | None = None
    extra_args: Sequence[str] = field(default_factory=tuple)

    def run(self, input_dir: str | Path, output_dir: str | Path, *, tag: str | None = None) -> None:
        cmd = [
            sys.executable,
            "-m",
            "scal3r.run",
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--config",
            self.config,
            "--save_dpt",
            str(int(self.save_dpt)),
            "--save_xyz",
            str(int(self.save_xyz)),
        ]
        if tag:
            cmd.extend(["--tag", tag])
        if self.checkpoint:
            cmd.extend(["--checkpoint", self.checkpoint])
        if self.device:
            cmd.extend(["--device", self.device])
        if self.block_size is not None:
            cmd.extend(["--block_size", str(int(self.block_size))])
        if self.overlap_size is not None:
            cmd.extend(["--overlap_size", str(int(self.overlap_size))])
        if self.use_loop is not None:
            cmd.extend(["--use_loop", str(int(self.use_loop))])
        if self.use_xyz_align is not None:
            cmd.extend(["--use_xyz_align", str(int(self.use_xyz_align))])
        if self.offload_batches is not None:
            cmd.extend(["--offload_batches", str(int(self.offload_batches))])
        if self.offload_outputs is not None:
            cmd.extend(["--offload_outputs", str(int(self.offload_outputs))])
        if self.streaming_state is not None:
            cmd.extend(["--streaming_state", str(int(self.streaming_state))])
        cmd.extend(str(x) for x in self.extra_args)

        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError as exc:
            raise ImportError(
                "Could not execute SCAL3R. Install it with `bash scripts/install.sh` "
                "inside https://github.com/zju3dv/Scal3R or `pip install -e .` "
                "from the cloned Scal3R repository."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"SCAL3R inference failed with exit code {exc.returncode}: {cmd}") from exc


class Scal3RDepthPoseAdapter:
    """Run/read SCAL3R and return per-frame depth + pose records.

    Use ``precomputed_root`` when SCAL3R has already been run externally. If it
    is omitted, the adapter stages the chunk frames into a temporary folder and
    invokes :class:`Scal3RRunner`.
    """

    def __init__(
        self,
        *,
        runner: Scal3RRunner | None = None,
        reader: Scal3ROutputReader | None = None,
        work_dir: str | Path | None = None,
        precomputed_root: str | Path | None = None,
        keep_work_dir: bool = False,
    ):
        self.runner = runner or Scal3RRunner()
        self.reader = reader or Scal3ROutputReader()
        self.work_dir = Path(work_dir) if work_dir is not None else None
        self.precomputed_root = Path(precomputed_root) if precomputed_root is not None else None
        self.keep_work_dir = keep_work_dir

    def _precomputed_dir_for_chunk(self, chunk_id: int) -> Path | None:
        if self.precomputed_root is None:
            return None
        candidates = [
            self.precomputed_root / f"chunk_{chunk_id:06d}",
            self.precomputed_root / f"chunk_{chunk_id}",
            self.precomputed_root,
        ]
        for path in candidates:
            if (path / "mat.txt").exists():
                return path
        raise FileNotFoundError(
            f"No SCAL3R mat.txt found for chunk {chunk_id} under {self.precomputed_root}"
        )

    def infer_frames(self, frames: list[VideoFrame], chunk_id: int) -> list[Scal3RFrameOutput]:
        if not frames:
            return []

        precomputed_dir = self._precomputed_dir_for_chunk(chunk_id)
        if precomputed_dir is not None:
            chunk_output = self.reader.read(precomputed_dir, expected_frames=len(frames))
            return self._to_frame_outputs(chunk_output)

        base_tmp: tempfile.TemporaryDirectory[str] | None = None
        if self.work_dir is None:
            base_tmp = tempfile.TemporaryDirectory(prefix="directme_scal3r_")
            base = Path(base_tmp.name)
        else:
            base = self.work_dir / f"chunk_{chunk_id:06d}"
            if base.exists():
                shutil.rmtree(base)
            base.mkdir(parents=True, exist_ok=True)

        input_dir = base / "input"
        output_dir = base / "output"
        input_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            for local_idx, frame in enumerate(frames):
                _ensure_rgb_frame_path(frame, input_dir / f"{local_idx:06d}.png")
            self.runner.run(input_dir, output_dir, tag=f"directme_chunk_{chunk_id:06d}")
            chunk_output = self.reader.read(output_dir, expected_frames=len(frames))
            return self._to_frame_outputs(chunk_output)
        finally:
            if base_tmp is not None and not self.keep_work_dir:
                base_tmp.cleanup()

    @staticmethod
    def _to_frame_outputs(chunk_output: Scal3RChunkOutput) -> list[Scal3RFrameOutput]:
        n = len(chunk_output.poses_local)
        out: list[Scal3RFrameOutput] = []
        for i in range(n):
            out.append(
                Scal3RFrameOutput(
                    pose_local=chunk_output.poses_local[i],
                    pose_confidence=chunk_output.pose_confidences[i],
                    depth=chunk_output.depth_maps[i] if chunk_output.depth_maps is not None else None,
                    intrinsics=chunk_output.intrinsics[i]
                    if chunk_output.intrinsics is not None
                    else None,
                    metadata={
                        **chunk_output.metadata,
                        "pose_scale": chunk_output.pose_scales[i]
                        if i < len(chunk_output.pose_scales)
                        else 1.0,
                    },
                )
            )
        return out


@dataclass
class Scal3RComposedBackend(PerceptionBackend):
    """Full DirectMe perception backend: SCAL3R + open-vocab tracking.

    The tracker is intentionally typed as ``Any`` so this module does not import
    heavy tracking classes at module import time. In practice pass an
    ``OpenVocabularyTrackingAdapter`` configured with YOLO-World and optional
    SAM 2, or a light fake tracker for tests.
    """

    depth_pose: Scal3RDepthPoseAdapter
    tracker: Any
    min_pose_confidence: float = 0.30
    color_hist_bins: int = 12
    scene_classifier: SceneClassifier = field(default_factory=RuleBasedSceneClassifier)

    def process_chunk(self, frames: list[VideoFrame], chunk_id: int) -> ChunkPerception:
        if not frames:
            return ChunkPerception(chunk_id=chunk_id, frames=[])
        scal3r_outputs = self.depth_pose.infer_frames(frames, chunk_id=chunk_id)
        if len(scal3r_outputs) != len(frames):
            raise ValueError(
                f"SCAL3R returned {len(scal3r_outputs)} outputs for {len(frames)} frames"
            )

        outputs: list[FramePerception] = []
        for frame, geom in zip(frames, scal3r_outputs):
            if frame.image_path:
                image = _load_image_rgb(frame.image_path)
            elif frame.image is not None:
                image = np.asarray(frame.image)
            else:
                raise ValueError("Scal3RComposedBackend needs image_path or image for tracking")

            detections = self.tracker.step(frame_index=frame.index, image=image)
            objects: list[ObjectObservation] = []
            for det in detections:
                attrs: dict[str, Any] = {}
                if getattr(det, "mask", None) is not None:
                    try:
                        attrs["color_hsv_histogram"] = hsv_histogram_from_image_mask(
                            image, det.mask, bins=self.color_hist_bins
                        )
                        attrs["color"] = dominant_hsv_color(image, det.mask)
                    except Exception:
                        # Appearance attributes are useful, but never make the
                        # geometry path fail solely because color extraction did.
                        pass
                objects.append(
                    ObjectObservation(
                        label=str(det.label),
                        track_id=getattr(det, "track_id", None),
                        score=float(getattr(det, "score", 1.0)),
                        bbox_xyxy=getattr(det, "bbox_xyxy", None),
                        mask=getattr(det, "mask", None),
                        attributes=attrs,
                        keyframe_path=frame.image_path,
                    )
                )

            try:
                scene_tag = self.scene_classifier(image, [obj.label for obj in objects])
            except Exception:
                scene_tag = RuleBasedSceneClassifier()(image, [obj.label for obj in objects])

            metadata = {
                **frame.metadata,
                **geom.metadata,
                "pose_confidence": geom.pose_confidence,
                "low_confidence": geom.pose_confidence < self.min_pose_confidence,
            }
            outputs.append(
                FramePerception(
                    frame=VideoFrame(
                        index=frame.index,
                        timestamp=frame.timestamp,
                        image_path=frame.image_path,
                        image=None,
                        metadata=metadata,
                    ),
                    local_pose=geom.pose_local,
                    intrinsics=geom.intrinsics,
                    depth=geom.depth,
                    objects=objects,
                    scene_tag=scene_tag,
                )
            )

        return ChunkPerception(chunk_id=chunk_id, frames=outputs)


def build_scal3r_backend(
    classes: Sequence[str],
    *,
    precomputed_root: str | Path | None = None,
    work_dir: str | Path | None = None,
    device: str | None = None,
    yolo_weights: str = "yolov8s-worldv2.pt",
    score_threshold: float = 0.20,
    detection_stride: int = 5,
    sam2_checkpoint: str | None = None,
    sam2_config: str | None = None,
    scal3r_config: str = "configs/models/scal3r.yaml",
    scal3r_checkpoint: str | None = None,
    scal3r_block_size: int | None = None,
    scal3r_overlap_size: int | None = None,
    scal3r_use_loop: int | None = None,
    scal3r_save_xyz: int = 0,
) -> Scal3RComposedBackend:
    """Convenience factory matching ``runtime.build_composed_backend``.

    This is the recommended entry point when replacing DA3 with SCAL3R while
    keeping YOLO-World + optional SAM 2 for object-level semantics.
    """
    from directme.perception.adapters.open_vocab_tracking import (
        OpenVocabularyTrackingAdapter,
        Sam2MaskRefiner,
        SimpleIoUAppearanceTracker,
        YoloWorldDetector,
    )
    from directme.perception.runtime import resolve_runtime_device

    resolved_device = resolve_runtime_device(device or "auto")
    depth_pose = Scal3RDepthPoseAdapter(
        runner=Scal3RRunner(
            config=scal3r_config,
            checkpoint=scal3r_checkpoint,
            device=resolved_device,
            block_size=scal3r_block_size,
            overlap_size=scal3r_overlap_size,
            use_loop=scal3r_use_loop,
            save_dpt=1,
            save_xyz=scal3r_save_xyz,
        ),
        precomputed_root=precomputed_root,
        work_dir=work_dir,
    )
    detector = YoloWorldDetector(
        weights=yolo_weights,
        classes=[str(c).strip() for c in classes if str(c).strip()],
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
    return Scal3RComposedBackend(depth_pose=depth_pose, tracker=tracker)


def _list_images_for_test(image_dir: str | Path) -> list[Path]:
    image_dir = Path(image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    paths.sort(key=_numeric_sort_key)
    return paths


def _make_test_frames(image_paths: list[Path]) -> list[VideoFrame]:
    frames = []
    for i, p in enumerate(image_paths):
        frames.append(
            VideoFrame(
                index=i,
                timestamp=float(i),
                image_path=str(p),
                image=None,
                metadata={},
            )
        )
    return frames


def _main() -> int:
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Real SCAL3R adapter test for DirectMe.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="/data/ywang/dataset/SpatialMemory/data_frames_1fps/scene0804_00-0",
        help="Directory containing input RGB frames.",
    )
    parser.add_argument("--max-images", type=int, default=30)
    parser.add_argument(
        "--work-dir",
        type=str,
        default="/tmp/directme_scal3r_real_test",
        help="Working directory for staged frames and SCAL3R outputs.",
    )
    parser.add_argument(
        "--precomputed-root",
        type=str,
        default=None,
        help="Existing SCAL3R result directory containing mat.txt/intri.yml/depths.",
    )
    parser.add_argument(
        "--scal3r-config",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/configs/scal3r/scal3r.yaml",
    )
    parser.add_argument(
        "--scal3r-checkpoint",
        type=str,
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/ckpts/scal3r/scal3r.pt",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--overlap-size", type=int, default=None)
    parser.add_argument("--use-loop", type=int, default=None)
    parser.add_argument("--keep-work-dir", action="store_true")

    args = parser.parse_args()

    image_paths = _list_images_for_test(args.image_dir)
    if args.max_images and args.max_images > 0:
        image_paths = image_paths[: args.max_images]

    if not image_paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    print(f"[INFO] image_dir: {args.image_dir}")
    print(f"[INFO] n_images: {len(image_paths)}")
    for p in image_paths[:5]:
        print(f"  - {p}")
    if len(image_paths) > 5:
        print("  ...")

    frames = _make_test_frames(image_paths)

    adapter = Scal3RDepthPoseAdapter(
        runner=Scal3RRunner(
            config=args.scal3r_config,
            checkpoint=args.scal3r_checkpoint,
            device=args.device,
            block_size=args.block_size,
            overlap_size=args.overlap_size,
            use_loop=args.use_loop,
            save_dpt=1,
            save_xyz=0,
        ),
        work_dir=args.work_dir,
        precomputed_root=args.precomputed_root,
        keep_work_dir=args.keep_work_dir,
    )

    outputs = adapter.infer_frames(frames, chunk_id=0)

    summary = {
        "status": "ok",
        "n_outputs": len(outputs),
        "has_depth": all(o.depth is not None for o in outputs),
        "has_intrinsics": all(o.intrinsics is not None for o in outputs),
        "first_pose_translation": outputs[0].pose_local.translation.tolist(),
        "first_depth_shape": None if outputs[0].depth is None else list(outputs[0].depth.shape),
        "first_intrinsics": None if outputs[0].intrinsics is None else outputs[0].intrinsics.tolist(),
        "work_dir": args.work_dir,
    }

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("[OK] Real SCAL3R adapter test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
