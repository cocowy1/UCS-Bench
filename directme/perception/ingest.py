"""Unified ingest for DirectMe — video files and pre-extracted frame streams.

This module gives the offline / async mapping engine a single front door for
the two real-world input modes:

* **Video file** (``mp4``, ``mov``, ``mkv``, …). Frames are sampled at a
  user-supplied target FPS (default ``1.0`` — DirectMe is built for 1 FPS
  egocentric capture; sampling above that rarely improves graph quality and
  burns inference cost).
* **Stream of frame paths or numpy arrays**. For users who already have a
  separate decoding / sampling pipeline (e.g. an upstream Aria / Project
  Aria recorder that writes JPEGs) and just want DirectMe to consume them.

Both paths emit :class:`directme.perception.base.VideoFrame` instances, which
is the only shape the rest of the pipeline depends on. The chunk grouping is
handled separately by :func:`group_into_chunks`.

Heavy deps (``imageio``, ``imageio-ffmpeg``) are lazy-imported. If they are
not installed, video ingest raises a precise error pointing at the install
command; frame-list ingest works with no deps at all.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence

from directme.perception.base import VideoFrame


# --------------------------------------------------------------------------- #
# Video file ingest
# --------------------------------------------------------------------------- #
def iter_frames_from_video(
    video_path: str | Path,
    target_fps: float = 1.0,
    frame_dump_dir: str | Path | None = None,
    image_format: str = "jpg",
    max_frames: int | None = None,
) -> Iterator[VideoFrame]:
    """Stream :class:`VideoFrame` from a video file at ``target_fps``.

    The video's native FPS is read from its container; we then keep one frame
    every ``round(native_fps / target_fps)`` to land close to the requested
    rate without re-encoding.

    Args:
        video_path: path to the video file.
        target_fps: desired sampling rate in Hz. Default ``1.0`` matches
            DirectMe's recommended egocentric capture rate.
        frame_dump_dir: if given, each sampled frame is written as
            ``frame_<index>.<image_format>`` and the path is set on
            :attr:`VideoFrame.image_path`. If ``None``, frames are kept only
            as numpy arrays on :attr:`VideoFrame.image` (cheaper but means
            keyframes can't be referenced after the run).
        image_format: ``"jpg"`` (default, ~10x smaller) or ``"png"`` (lossless).
        max_frames: hard cap on how many frames to emit. Useful for smoke
            tests / dry runs.

    Yields:
        :class:`VideoFrame` with ``index``, ``timestamp`` (seconds from
        video start), and either ``image_path`` or ``image`` populated.

    Raises:
        ImportError: if ``imageio[ffmpeg]`` is not installed.
        FileNotFoundError: if ``video_path`` does not exist.
        ValueError: if the video's native FPS cannot be inferred or
            ``target_fps`` exceeds the native FPS by more than 5 %.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    try:
        import imageio.v3 as iio
    except ImportError as exc:
        raise ImportError(
            "Video ingest requires `imageio` and `imageio-ffmpeg`. "
            "Install with: pip install 'directme[ingest]' "
            "or: pip install imageio imageio-ffmpeg"
        ) from exc

    metadata = iio.immeta(str(video_path), plugin="pyav") if False else iio.immeta(str(video_path))
    native_fps = float(metadata.get("fps") or metadata.get("frame_rate") or 0.0)
    if native_fps <= 0:
        raise ValueError(
            f"Could not infer FPS for {video_path}; metadata={metadata!r}"
        )
    if target_fps > native_fps * 1.05:
        raise ValueError(
            f"Requested target_fps={target_fps:.3f} exceeds native FPS "
            f"{native_fps:.3f} for {video_path.name}. Up-sampling is not "
            f"supported; pick a lower target_fps."
        )

    stride = max(1, int(round(native_fps / target_fps)))
    dump_dir: Path | None = None
    if frame_dump_dir is not None:
        dump_dir = Path(frame_dump_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)

    n_emitted = 0
    for native_idx, image in enumerate(iio.imiter(str(video_path))):
        if native_idx % stride != 0:
            continue
        timestamp = native_idx / native_fps
        index = native_idx // stride

        image_path: str | None = None
        if dump_dir is not None:
            out_path = dump_dir / f"frame_{index:06d}.{image_format}"
            iio.imwrite(str(out_path), image)
            image_path = str(out_path)

        yield VideoFrame(
            index=index,
            timestamp=timestamp,
            image_path=image_path,
            image=None if image_path is not None else image,
            metadata={
                "source_video": str(video_path),
                "native_frame_index": native_idx,
                "sampled_at_fps": target_fps,
            },
        )

        n_emitted += 1
        if max_frames is not None and n_emitted >= max_frames:
            break


# --------------------------------------------------------------------------- #
# Frame-list ingest
# --------------------------------------------------------------------------- #
def iter_frames_from_paths(
    image_paths: Sequence[str | Path],
    timestamps: Sequence[float] | None = None,
    fps: float = 1.0,
    start_timestamp: float = 0.0,
) -> Iterator[VideoFrame]:
    """Stream :class:`VideoFrame` from an ordered list of image paths.

    Use this when an upstream pipeline already extracted frames at the
    desired rate (e.g. a wearable's onboard 1-FPS sampler) and DirectMe
    just needs to consume them.

    Args:
        image_paths: ordered list of image file paths. **Order is preserved
            and treated as time order.**
        timestamps: per-frame timestamps in seconds. If ``None``, timestamps
            are synthesized as ``start_timestamp + i / fps``.
        fps: only used when ``timestamps is None``. Default ``1.0``.
        start_timestamp: only used when ``timestamps is None``.

    Yields:
        :class:`VideoFrame` with ``index`` running 0, 1, 2, …
    """
    paths = [Path(p) for p in image_paths]
    if timestamps is not None and len(timestamps) != len(paths):
        raise ValueError(
            f"timestamps has {len(timestamps)} entries but image_paths has "
            f"{len(paths)}; the two must align."
        )

    for i, p in enumerate(paths):
        ts = timestamps[i] if timestamps is not None else start_timestamp + i / fps
        yield VideoFrame(
            index=i,
            timestamp=float(ts),
            image_path=str(p),
            image=None,
            metadata={"source": "frame_list"},
        )


# --------------------------------------------------------------------------- #
# Chunk grouping
# --------------------------------------------------------------------------- #
def group_into_chunks(
    frames: Iterable[VideoFrame],
    chunk_size: int,
) -> Iterator[tuple[int, list[VideoFrame]]]:
    """Group a frame stream into fixed-size chunks.

    Yields ``(chunk_id, chunk_frames)`` tuples. The trailing chunk may be
    shorter than ``chunk_size``. Empty chunks are not emitted.

    Args:
        frames: any iterable of :class:`VideoFrame` (a generator is fine).
        chunk_size: target chunk size in frames. At 1 FPS a chunk_size of
            10–30 corresponds to 10–30 seconds of capture, which is the
            sweet spot for SCAL3R / DA3 backends — large enough for them to
            run useful local BA, small enough to keep memory bounded and
            keep online QA latency low.

    Yields:
        ``(chunk_id, list[VideoFrame])`` pairs.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    chunk: list[VideoFrame] = []
    chunk_id = 0
    for frame in frames:
        chunk.append(frame)
        if len(chunk) >= chunk_size:
            yield chunk_id, chunk
            chunk = []
            chunk_id += 1
    if chunk:
        yield chunk_id, chunk
