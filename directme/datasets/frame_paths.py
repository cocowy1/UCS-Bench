"""Portable frame-path resolution.

Scene graphs persist the keyframe paths exactly as the perception backend saw
them — historically absolute, machine-specific paths such as
``/data/.../data_frames_1fps/01042/frame_000123.png``.  When graphs are moved
between machines (published artifacts, issue #3 / issue #4 reproduction), every
``os.path.exists`` check downstream silently drops them and the MLLM ends up
with zero images.

This module resolves a stored path against a configurable ``frames_root``:

* absolute path that still exists  → returned unchanged;
* relative path (``<uid>/frame_000123.png``) → joined onto ``frames_root``;
* missing absolute path → re-located by its trailing ``<uid>/<file>`` suffix
  under ``frames_root``.

It also provides the inverse operation (``relativize_frame_path``) used when
building portable graphs.
"""

from __future__ import annotations

import os
from pathlib import Path


def resolve_frame_path(path: str | None, frames_root: str | Path | None = None) -> str | None:
    """Return an existing filesystem path for a stored keyframe path.

    Returns ``None`` when the path cannot be located at all (caller decides
    whether to warn or drop).
    """
    if not path:
        return None
    p = Path(path)

    if p.is_absolute() and p.exists():
        return str(p)

    if frames_root is None:
        return str(p) if p.exists() else None

    root = Path(frames_root)

    # Relative stored path: <uid>/frame_000123.png (or deeper).
    if not p.is_absolute():
        candidate = root / p
        if candidate.exists():
            return str(candidate)
        return None

    # Missing absolute path: re-locate by suffix under the root. The UCS-Bench
    # layout is <frames_root>/<video_uid>/frame_XXXXXX.png, so the trailing two
    # components are sufficient and avoid any filesystem walk.
    parts = p.parts
    if len(parts) >= 2:
        candidate = root.joinpath(*parts[-2:])
        if candidate.exists():
            return str(candidate)
    if len(parts) >= 1:
        candidate = root / parts[-1]
        if candidate.exists():
            return str(candidate)
    return None


def relativize_frame_path(path: str | None, frames_root: str | Path | None = None) -> str | None:
    """Store paths portably: relative to ``frames_root`` when possible.

    If ``frames_root`` is not given or the path does not live under it, the
    path is returned unchanged (absolute) — callers that need portability
    should always pass the root used at build time.
    """
    if not path:
        return None
    p = Path(path)
    if frames_root is None:
        return str(p)
    try:
        rel = p.relative_to(Path(frames_root))
        return rel.as_posix()
    except ValueError:
        return str(p)


def iter_video_frames(frames_root: str | Path, video_uid: str) -> list[str]:
    """Sorted frame paths of one video under ``frames_root`` (empty if absent)."""
    d = Path(frames_root) / video_uid
    if not d.is_dir():
        return []
    return sorted(str(p) for p in d.glob("*.png") if p.is_file())


def infer_frames_root_from_paths(paths: list[str]) -> str | None:
    """Best-effort: infer a frames root from existing absolute frame paths.

    UCS-Bench layout is ``<root>/<uid>/frame_XXXXXX.png``; the root is the
    grandparent of any frame file.  Useful when no explicit ``frames_root``
    was configured but the graph still holds valid local paths.
    """
    for path in paths:
        if not path:
            continue
        p = Path(path)
        if p.is_file() and p.parent.parent.exists():
            return str(p.parent.parent)
    return None
