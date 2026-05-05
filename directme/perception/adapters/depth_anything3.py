"""Adapter for Depth Anything 3 (ByteDance-Seed/Depth-Anything-3).

Wraps the official ``depth_anything_3.api.DepthAnything3`` interface. Heavy
deps (``torch``, ``depth_anything_3``) are imported lazily so the core
DirectMe package stays lightweight.

Reference:
    https://github.com/ByteDance-Seed/Depth-Anything-3
    Output: extrinsics are [N, 3, 4] world-to-camera (OpenCV/COLMAP).

Default checkpoint
------------------
``depth-anything/DA3NESTED-GIANT-LARGE-1.1`` is the refreshed checkpoint
recommended by the upstream project; the original ``-LARGE`` checkpoint
contains a known training bug. Override via ``model_id=...`` if you need an
older or smaller variant.

License caveat
--------------
The DA3 *Nested* checkpoints are released under **CC BY-NC 4.0**, i.e. the
weights are restricted to non-commercial use. The DirectMe glue code is
Apache-2.0, but bundling a DA3 Nested checkpoint into a commercial product
requires a separate license from the upstream authors. Users who need
commercial deployment should swap in a permissively-licensed depth /
pose backend (e.g. SCAL3R, MAST3R, DUSt3R) via the same adapter interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from directme.geometry.poses import SE3

def w2c_3x4_to_T_world_from_camera(w2c: np.ndarray) -> SE3:
    """Convert a [3, 4] world-to-camera matrix into our SE3 (T_world_from_camera).

    DA3 returns extrinsics in OpenCV / COLMAP convention: ``X_cam = R @ X_world + t``.
    DirectMe stores ``T_world_from_camera`` such that ``X_world = T @ X_cam``.
    The two are inverses of each other.
    """
    if w2c.shape == (4, 4):
        w2c = w2c[:3, :4]
    if w2c.shape != (3, 4):
        raise ValueError(f"DA3 extrinsics expected (3, 4) or (4, 4), got {w2c.shape}")
    R = w2c[:, :3]
    t = w2c[:, 3]
    R_inv = R.T
    t_inv = -R_inv @ t
    return SE3.from_rotation_translation(R_inv, t_inv)


def _resolve_torch_device(device: str):
    """Resolve a ``device`` string into a concrete ``torch.device``.

    Accepts ``"auto"`` and falls back to CPU if CUDA / MPS are unavailable.
    """
    import torch

    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "You requested CUDA, but torch.cuda.is_available() is False. "
            "Please install CUDA-enabled PyTorch or check CUDA_VISIBLE_DEVICES."
        )

    return torch.device(device)


@dataclass
class DA3Output:
    """Per-frame DA3 output."""

    depth: np.ndarray                # (H, W) float32, metric meters
    confidence: np.ndarray | None    # (H, W) float32 in [0, 1]
    intrinsics: np.ndarray           # (3, 3)
    pose_local: SE3                  # T_local_from_camera (chunk-local frame)
    pose_confidence: float           # mean depth confidence as a proxy


class DepthAnything3Adapter:
    """Run DA3 on a list of image paths and return per-frame depth + pose.

    Example::

        adapter = DepthAnything3Adapter(
            "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
            device="auto",
        )
        outputs = adapter.infer(["frame_000.jpg", "frame_001.jpg"])

    DA3 treats the first input view as the local reference (``extrinsics[0] ≈ I``).
    DirectMe's chunk pose propagator stitches chunk-local poses into the global
    world frame, so this is the correct convention.
    """

    def __init__(
        self,
        model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
        device: str = "auto",
        use_ray_pose: bool = False,
        process_res: int = 504,
        ref_view_strategy: str = "first",
    ):
        try:
            import torch  # noqa: F401
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "DepthAnything3Adapter requires `torch` and the upstream "
                "`depth_anything_3` package. Install per "
                "https://github.com/ByteDance-Seed/Depth-Anything-3#installation"
            ) from exc

        self.device = _resolve_torch_device(device)
        self.use_ray_pose = use_ray_pose
        self.process_res = process_res
        self.ref_view_strategy = ref_view_strategy

        print(f"[DA3] requested device = {device}")
        print(f"[DA3] resolved device = {self.device}")
        print(f"[DA3] ref_view_strategy = {self.ref_view_strategy}")
        print(f"[DA3] process_res = {self.process_res}")

        self.model = DepthAnything3.from_pretrained(model_id).to(self.device).eval()

        
    def infer(self, image_paths: list[str | Path]) -> list[DA3Output]:
        if not image_paths:
            return []

        import torch

        with torch.inference_mode():
            prediction = self.model.inference(
                [str(p) for p in image_paths],
                use_ray_pose=self.use_ray_pose,
                process_res=self.process_res,
                ref_view_strategy=self.ref_view_strategy,
            )

        depth = np.asarray(prediction.depth, dtype=np.float32)
        conf = (
            np.asarray(prediction.conf, dtype=np.float32)
            if getattr(prediction, "conf", None) is not None
            else None
        )
        extrinsics = np.asarray(prediction.extrinsics, dtype=np.float32)
        intrinsics = np.asarray(prediction.intrinsics, dtype=np.float32)

        outputs: list[DA3Output] = []
        for i in range(depth.shape[0]):
            
            pose_local = w2c_3x4_to_T_world_from_camera(extrinsics[i])
            mean_conf = float(conf[i].mean()) if conf is not None else 1.0
            outputs.append(
                DA3Output(
                    depth=depth[i],
                    confidence=conf[i] if conf is not None else None,
                    intrinsics=intrinsics[i],
                    pose_local=pose_local,
                    pose_confidence=mean_conf,
                )
            )
        return outputs


import re


_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def _natural_sort_key(path: Path):
    """Natural sort: frame_2.jpg comes before frame_10.jpg."""
    parts = re.split(r"(\d+)", path.name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def _list_images_in_dir(image_dir: Path) -> list[Path]:
    """List image files directly under one directory."""
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")

    paths = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_SUFFIXES
    ]
    return sorted(paths, key=_natural_sort_key)


def _find_first_frame_dir(root: Path) -> Path:
    """
    Find the first subdirectory that contains image frames.

    Supports structures like:
        data_frames_1fps/video_uid/000001.jpg
        data_frames_1fps/video_uid/frame_000001.jpg
    """
    if not root.exists():
        raise FileNotFoundError(f"Frame root does not exist: {root}")

    # Case 1: root itself already contains images.
    direct = _list_images_in_dir(root)
    if direct:
        return root

    # Case 2: each video has one subdirectory.
    for subdir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
        images = _list_images_in_dir(subdir)
        if images:
            return subdir

    raise FileNotFoundError(f"No image frames found under: {root}")


def _resolve_spatialmemory_frame_dir(
    frame_root: str | Path,
    video_uid: str | None = None,
) -> Path:
    """
    Resolve image directory from SpatialMemory 1fps frame root.

    If video_uid is provided, this function tries:
        frame_root / video_uid
        frame_root / f"{video_uid}.mp4"
        any subdir whose name starts with video_uid

    If video_uid is omitted, it finds the first directory containing images.
    """
    root = Path(frame_root).resolve()

    if video_uid:
        candidates = [
            root / video_uid,
            root / f"{video_uid}.mp4",
            root / f"{video_uid}.avi",
            root / f"{video_uid}.mkv",
        ]

        for cand in candidates:
            if cand.exists() and cand.is_dir() and _list_images_in_dir(cand):
                return cand

        for subdir in sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name):
            if subdir.name.startswith(video_uid) and _list_images_in_dir(subdir):
                return subdir

        raise FileNotFoundError(
            f"Cannot find frame directory for video_uid={video_uid} under {root}"
        )

    return _find_first_frame_dir(root)


def _sample_image_paths(
    image_paths: list[Path],
    max_images: int = 8,
    stride: int = 1,
) -> list[Path]:
    """
    Sample image paths for DA3 smoke test.

    - First applies stride.
    - Then uniformly samples at most max_images.
    """
    if not image_paths:
        raise ValueError("No image paths to sample.")

    stride = max(1, int(stride))
    image_paths = image_paths[::stride]

    if max_images is not None and max_images > 0 and len(image_paths) > max_images:
        indices = np.linspace(0, len(image_paths) - 1, num=max_images)
        indices = [int(round(x)) for x in indices]
        image_paths = [image_paths[i] for i in indices]

    return image_paths


def test_da3_on_spatialmemory_frames(
    frame_root: str | Path = "/data/ywang/dataset/SpatialMemory/data_frames_1fps",
    video_uid: str | None = None,
    max_images: int = 8,
    stride: int = 1,
    model_id: str = "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    device: str = "auto",
    process_res: int = 504,
    use_ray_pose: bool = False,
    ref_view_strategy: str = "first",
) -> list[DA3Output]:
    """
    Smoke test DepthAnything3Adapter on SpatialMemory extracted frames.

    Example:
        test_da3_on_spatialmemory_frames(
            frame_root="/data/ywang/dataset/SpatialMemory/data_frames_1fps",
            video_uid="xxx",
            max_images=8,
            device="cuda",
        )
    """
    frame_dir = _resolve_spatialmemory_frame_dir(frame_root, video_uid)
    all_images = _list_images_in_dir(frame_dir)
    image_paths = _sample_image_paths(all_images, max_images=max_images, stride=stride)

    print(f"[INFO] SpatialMemory frame root: {Path(frame_root).resolve()}")
    print(f"[INFO] Selected frame dir: {frame_dir}")
    print(f"[INFO] Total frames in dir: {len(all_images)}")
    print(f"[INFO] Test frames: {len(image_paths)}")
    
    for p in image_paths[:5]:
        print(f"  - {p}")
    if len(image_paths) > 5:
        print("  ...")

    adapter = DepthAnything3Adapter(
        model_id=model_id,
        device=device,
        use_ray_pose=use_ray_pose,
        process_res=process_res,
        ref_view_strategy=ref_view_strategy,
    )

    outputs = adapter.infer(image_paths)

    assert len(outputs) == len(image_paths), (
        f"Expected {len(image_paths)} outputs, got {len(outputs)}"
    )

    for i, out in enumerate(outputs):
        # depth_png_path = f"/data/ywang/my_projects/VideoUnderstanding/Directme/directme/vis_output/depth_{i:03d}.png"
        # plt.imsave(depth_png_path, out.depth, cmap="plasma")
        print(f"\n[Frame {i}]")
        print(f"  image_path: {image_paths[i]}")
        print(f"  depth.shape: {out.depth.shape}")
        print(f"  depth.dtype: {out.depth.dtype}")
        print(
            f"  depth.min/max: "
            f"{float(np.nanmin(out.depth)):.4f} / {float(np.nanmax(out.depth)):.4f}"
        )
        print(f"  confidence: {'None' if out.confidence is None else out.confidence.shape}")
        print(f"  intrinsics.shape: {out.intrinsics.shape}")
        print(f"  pose.translation: {out.pose_local.translation.tolist()}")
        print(f"  pose_confidence: {out.pose_confidence:.4f}")

        assert out.depth.ndim == 2
        assert out.intrinsics.shape == (3, 3)
        assert out.pose_local.rotation.shape == (3, 3)
        assert out.pose_local.translation.shape == (3,)

    print("\n[OK] DA3 SpatialMemory frame test passed.")
    return outputs

def _test_pose_conversion() -> None:
    """Lightweight unit test for DA3 w2c -> DirectMe c2w conversion."""
    w2c = np.eye(4, dtype=np.float32)
    w2c[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float32)

    pose = w2c_3x4_to_T_world_from_camera(w2c)

    expected_translation = np.array([-1.0, -2.0, -3.0], dtype=np.float32)
    assert np.allclose(pose.rotation, np.eye(3), atol=1e-6)
    assert np.allclose(pose.translation, expected_translation, atol=1e-6)

    print("[OK] w2c_3x4_to_T_world_from_camera conversion test passed.")
    
def _main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test DepthAnything3Adapter.")
    parser.add_argument(
        "--model-id",
        type=str,
        default="depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--use-ray-pose", action="store_true")

    parser.add_argument(
        "--spatialmemory-frame-root",
        type=str,
        default="/data/ywang/dataset/SpatialMemory/data_frames_1fps",
        help="SpatialMemory 1fps extracted frame root.",
    )
    parser.add_argument(
        "--video-uid",
        type=str,
        default="scene0804_00-0",
        help="Optional video uid. If omitted, the first valid frame folder is used.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=30,
        help="Maximum number of frames used for this smoke test.",
    )
    parser.add_argument(
        "--ref-view-strategy",
        type=str,
        default="first",
        choices=["first", "middle", "saddle_balanced", "saddle_sim_range"],
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Frame stride before uniform sampling.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Only test pose conversion; do not load DA3 model.",
    )

    args = parser.parse_args()

    _test_pose_conversion()

    if args.skip_inference:
        print("[OK] Skip inference requested.")
        return 0

    test_da3_on_spatialmemory_frames(
        frame_root=args.spatialmemory_frame_root,
        video_uid=args.video_uid,
        max_images=args.max_images,
        stride=args.stride,
        model_id=args.model_id,
        device=args.device,
        process_res=args.process_res,
        use_ray_pose=args.use_ray_pose,
        ref_view_strategy=args.ref_view_strategy,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
