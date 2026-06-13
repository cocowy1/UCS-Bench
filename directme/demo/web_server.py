from __future__ import annotations

import argparse
import json
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "directme" / "demo" / "web" / "alignment_config.json"
DEMO_DATA_ROOT = REPO_ROOT / "tmp" / "directme_scal3r_full_pipeline"
FRAMES_DIR = DEMO_DATA_ROOT / "frames"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
DEFAULT_CONFIG: dict[str, Any] = {
    "version": 1,
    "graph": {"position": [0, 0, 0], "rotationDeg": [0, 0, 0], "scale": 1},
    "sceneGraph": {"position": [0, 0, 0], "rotationDeg": [0, 0, 0], "scale": 1},
    "scene": {"position": [0, 0, 0], "rotationDeg": [0, 0, 0], "scale": 1},
}


class DirectMeDemoHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(REPO_ROOT), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/alignment-config":
            self._send_json(_load_config())
            return
        if path == "/api/original-frames":
            self._send_json({"frames": _list_original_frames()})
            return
        if path == "/api/depth-frames":
            self._send_json({"frames": _list_perception_frames("depth_frames")})
            return
        if path == "/api/tracking-frames":
            self._send_json({"frames": _list_perception_frames("tracking_frames")})
            return
        alias = _legacy_asset_alias(path)
        if alias is not None:
            self.path = alias
        super().do_GET()

    def do_HEAD(self) -> None:
        path = self.path.split("?", 1)[0]
        alias = _legacy_asset_alias(path)
        if alias is not None:
            self.path = alias
        super().do_HEAD()

    def do_POST(self) -> None:
        if self.path.split("?", 1)[0] != "/api/alignment-config":
            self.send_error(404, "Unknown API endpoint")
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = self.rfile.read(length).decode("utf-8")
            data = json.loads(payload)
            config = _normalize_config(data)
            CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:
            self.send_error(400, f"Invalid alignment config: {exc}")
            return

        self._send_json({"ok": True, "path": str(CONFIG_PATH), "config": config})

    def _send_json(self, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def _load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        return DEFAULT_CONFIG
    return _normalize_config(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _list_original_frames() -> list[str]:
    if not FRAMES_DIR.exists():
        return []
    frames = sorted(
        (path for path in FRAMES_DIR.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_sort_key,
    )
    return ["/" + path.relative_to(REPO_ROOT).as_posix() for path in frames]


def _list_perception_frames(dirname: str) -> list[str]:
    preferred_dir = DEMO_DATA_ROOT / "perception_videos" / dirname
    frames_dir = preferred_dir if preferred_dir.exists() else DEMO_DATA_ROOT / "perception_artifacts" / dirname
    if not frames_dir.exists():
        return []
    frames = sorted(
        (path for path in frames_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=_natural_sort_key,
    )
    return ["/" + path.relative_to(REPO_ROOT).as_posix() for path in frames]


def _legacy_asset_alias(path: str) -> str | None:
    """Serve the current pipeline1 group for stale browser URLs from older demos."""
    video_aliases = {
        "/tmp/directme_scal3r_full_pipeline/perception_artifacts/videos/original_all.mp4":
            "/tmp/directme_scal3r_full_pipeline/perception_videos/original.mp4",
        "/tmp/directme_scal3r_full_pipeline/perception_artifacts/videos/depth_all.mp4":
            "/tmp/directme_scal3r_full_pipeline/perception_videos/depth_h264.mp4",
        "/tmp/directme_scal3r_full_pipeline/perception_artifacts/videos/tracking_all.mp4":
            "/tmp/directme_scal3r_full_pipeline/perception_videos/tracking_h264.mp4",
        "/tmp/directme_scal3r_full_pipeline/perception_videos/original.mp4":
            "/tmp/directme_scal3r_full_pipeline/perception_videos/original.mp4",
        "/tmp/directme_scal3r_full_pipeline/perception_videos/depth_h264.mp4":
            "/tmp/directme_scal3r_full_pipeline/perception_videos/depth_h264.mp4",
        "/tmp/directme_scal3r_full_pipeline/perception_videos/tracking_h264.mp4":
            "/tmp/directme_scal3r_full_pipeline/perception_videos/tracking_h264.mp4",
        "/tmp/long_video/perception_videos/original_1fps_h264.mp4":
            "/tmp/directme_scal3r_full_pipeline1/perception_artifacts/videos/original_all_h264.mp4",
        "/tmp/long_video/perception_videos/depth_1fps_h264.mp4":
            "/tmp/directme_scal3r_full_pipeline1/perception_artifacts/videos/depth_all.mp4",
        "/tmp/long_video/perception_videos/tracking_1fps_h264.mp4":
            "/tmp/directme_scal3r_full_pipeline1/perception_artifacts/videos/tracking_all.mp4",
        "/long_video1/directme_scal3r_full_pipeline1/perception_artifacts/videos/original_all_h264.mp4":
            "/tmp/directme_scal3r_full_pipeline1/perception_artifacts/videos/original_all_h264.mp4",
        "/long_video1/directme_scal3r_full_pipeline1/perception_artifacts/videos/depth_all.mp4":
            "/tmp/directme_scal3r_full_pipeline1/perception_artifacts/videos/depth_all.mp4",
        "/long_video1/directme_scal3r_full_pipeline1/perception_artifacts/videos/tracking_all.mp4":
            "/tmp/directme_scal3r_full_pipeline1/perception_artifacts/videos/tracking_all.mp4",
        "/tmp/directme_scal3r_full_pipeline1/perception_artifacts/videos/original_all_h264.mp4":
            "/tmp/directme_scal3r_full_pipeline/perception_videos/original.mp4",
        "/tmp/directme_scal3r_full_pipeline1/perception_artifacts/videos/depth_all.mp4":
            "/tmp/directme_scal3r_full_pipeline/perception_videos/depth_h264.mp4",
        "/tmp/directme_scal3r_full_pipeline1/perception_artifacts/videos/tracking_all.mp4":
            "/tmp/directme_scal3r_full_pipeline/perception_videos/tracking_h264.mp4",
    }
    if path in video_aliases:
        return video_aliases[path]

    prefix = "/tmp/directme_scal3r_full_pipeline/frames/frame_"
    suffix = ".jpg"
    if path.startswith(prefix) and path.endswith(suffix):
        try:
            legacy_index = int(path[len(prefix):-len(suffix)])
        except ValueError:
            return None
        current_frames = _list_original_frames()
        if not current_frames:
            return None
        current_index = round(legacy_index * (len(current_frames) - 1) / 99)
        current_index = max(0, min(len(current_frames) - 1, current_index))
        return current_frames[current_index]

    return None


def _natural_sort_key(path: Path) -> list[Any]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", path.name)]


def _normalize_config(data: Any) -> dict[str, Any]:
    out = json.loads(json.dumps(DEFAULT_CONFIG))
    if not isinstance(data, dict):
        return out

    for key in ("graph", "sceneGraph", "scene"):
        source = data.get(key) if isinstance(data.get(key), dict) else {}
        out[key]["position"] = _vector(source.get("position"), out[key]["position"])
        out[key]["rotationDeg"] = _vector(source.get("rotationDeg"), out[key]["rotationDeg"])
        out[key]["scale"] = _number(source.get("scale"), out[key]["scale"])
    return out


def _vector(value: Any, fallback: list[float]) -> list[float]:
    if not isinstance(value, list) or len(value) < 3:
        return list(fallback)
    return [_number(value[i], fallback[i]) for i in range(3)]


def _number(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the DirectMe 3D demo and alignment debug API.")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.bind, args.port), DirectMeDemoHandler)
    print(f"DirectMe demo server: http://{args.bind}:{args.port}/directme/demo/web/")
    print(f"Alignment debug page: http://{args.bind}:{args.port}/directme/demo/web/debug.html")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
