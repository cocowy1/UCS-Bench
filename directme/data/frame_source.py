from __future__ import annotations

from pathlib import Path

from directme.perception.base import VideoFrame


def chunk_list(items: list[VideoFrame], chunk_size: int) -> list[list[VideoFrame]]:
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]


class ImageFolderFrameSource:
    """Loads a pre-extracted 1 FPS image folder as a frame source.

    File names are sorted lexicographically. Timestamp is index / fps.
    """

    def __init__(self, folder: str | Path, fps: float = 1.0, suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".webp")):
        self.folder = Path(folder)
        self.fps = fps
        self.suffixes = suffixes

    def frames(self) -> list[VideoFrame]:
        paths = sorted(p for p in self.folder.iterdir() if p.suffix.lower() in self.suffixes)
        return [
            VideoFrame(index=i, timestamp=i / self.fps, image_path=str(path))
            for i, path in enumerate(paths)
        ]
