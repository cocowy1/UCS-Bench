"""Export a demo dense map in the same world frame as a saved scene graph."""

from __future__ import annotations

import argparse
import colorsys
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_DATA_ROOT = Path("tmp") / "directme_scal3r_full_pipeline"
DEFAULT_GRAPH_JSON = REPO_DATA_ROOT / "directme_mapping_run" / "scene_graph.json"
DEFAULT_OUTPUT = REPO_DATA_ROOT / "dense_pointcloud_world.ply"
DEPTH_SUFFIXES = {".exr", ".npy", ".npz", ".png", ".tif", ".tiff"}
POSE_SOURCES = {"scal3r", "graph"}


def _numeric_sort_key(path: Path) -> tuple[int, str]:
    numbers = re.findall(r"\d+", path.stem)
    return (int(numbers[-1]) if numbers else 10**12, path.name)


def _frame_color(frame_index: int) -> tuple[int, int, int]:
    hue = (0.57 + frame_index * 0.013) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.32, 0.92)
    return round(255 * r), round(255 * g), round(255 * b)


def _sample_frame_points(
    depth: Any,
    intrinsics: Any,
    transform: Any,
    *,
    pixel_stride: int,
    depth_min: float,
    depth_max: float,
    max_points: int,
) -> Any:
    import numpy as np

    h, w = depth.shape
    ys, xs = np.mgrid[0:h:pixel_stride, 0:w:pixel_stride]
    z = depth[ys, xs]
    valid = np.isfinite(z) & (z >= depth_min) & (z <= depth_max)
    xs = xs[valid].astype(np.float64)
    ys = ys[valid].astype(np.float64)
    z = z[valid].astype(np.float64)
    if not z.size:
        return np.empty((0, 3), dtype=np.float64)

    if max_points and z.size > max_points:
        keep = np.linspace(0, z.size - 1, max_points, dtype=np.int64)
        xs, ys, z = xs[keep], ys[keep], z[keep]

    fx = float(intrinsics[0, 0])
    fy = float(intrinsics[1, 1])
    cx = float(intrinsics[0, 2])
    cy = float(intrinsics[1, 2])
    x_cam = (xs - cx) / fx * z
    y_cam = (ys - cy) / fy * z
    points_cam = np.column_stack([x_cam, y_cam, z, np.ones_like(z)])
    return (transform @ points_cam.T).T[:, :3]


def _timeline_by_chunk(graph: dict[str, Any]) -> dict[int, list[dict[str, Any]]]:
    records: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in graph.get("metadata", {}).get("ego_pose_timeline", []):
        chunk_id = item.get("chunk_id")
        frame_index = item.get("frame_index")
        transform = item.get("T_world_from_camera")
        if chunk_id is None or frame_index is None or transform is None:
            continue
        records[int(chunk_id)].append(item)

    for chunk_records in records.values():
        chunk_records.sort(key=lambda item: (float(item.get("timestamp", 0.0)), int(item["frame_index"])))
    return records


def _world_poses_from_scal3r(
    *,
    result_dir: Path,
    timeline: list[dict[str, Any]],
) -> tuple[list[Any], float | None]:
    import numpy as np

    from directme.geometry.poses import SE3
    from directme.perception.adapters.scal3r import _read_mat_txt

    local_poses, _scales = _read_mat_txt(result_dir / "mat.txt")
    usable = min(len(timeline), len(local_poses))
    if not usable:
        return [], None

    graph_anchor = SE3.from_list(timeline[0]["T_world_from_camera"])
    world_from_local = graph_anchor.compose(local_poses[0].inverse())
    world_poses = [world_from_local.compose(pose).matrix for pose in local_poses[:usable]]

    graph_poses = np.asarray(
        [record["T_world_from_camera"] for record in timeline[:usable]],
        dtype=np.float64,
    )
    scal3r_poses = np.asarray(world_poses, dtype=np.float64)
    max_abs_diff = float(np.max(np.abs(graph_poses - scal3r_poses)))
    return world_poses, max_abs_diff


def export_dense_map(
    *,
    data_root: Path,
    graph_json: Path,
    output: Path,
    pixel_stride: int,
    frame_stride: int,
    depth_min: float,
    depth_max: float,
    max_points_per_frame: int,
    pose_source: str,
    dry_run: bool = False,
) -> dict[str, Any]:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError(
            "Dense map export requires numpy plus an EXR reader such as OpenCV or imageio."
        ) from exc

    # Reuse SCAL3R readers so EXR and OpenCV-y intrinsics stay consistent with
    # the perception adapter without moving this demo exporter into the core.
    from directme.perception.adapters.scal3r import _read_depth, _read_intrinsics

    if pose_source not in POSE_SOURCES:
        raise ValueError(f"pose_source must be one of {sorted(POSE_SOURCES)}, got {pose_source!r}")

    graph = json.loads(graph_json.read_text(encoding="utf-8"))
    timelines = _timeline_by_chunk(graph)
    if not timelines:
        raise ValueError(f"{graph_json} has no metadata.ego_pose_timeline records")

    point_blocks: list[tuple[int, Any, tuple[int, int, int]]] = []
    skipped: list[str] = []
    pose_diffs: dict[int, float] = {}
    for chunk_id, timeline in sorted(timelines.items()):
        result_dir = data_root / "scal3r_work" / f"chunk_{chunk_id:06d}" / "output"
        depth_dir = result_dir / "depths"
        depth_files = sorted(
            (path for path in depth_dir.iterdir() if path.suffix.lower() in DEPTH_SUFFIXES),
            key=_numeric_sort_key,
        ) if depth_dir.is_dir() else []
        intrinsics = _read_intrinsics(result_dir / "intri.yml") or []
        if not depth_files or not intrinsics:
            skipped.append(f"chunk {chunk_id}: missing depths or intrinsics in {result_dir}")
            continue

        if pose_source == "scal3r":
            transforms, pose_diff = _world_poses_from_scal3r(result_dir=result_dir, timeline=timeline)
            if pose_diff is not None:
                pose_diffs[chunk_id] = pose_diff
        else:
            transforms = [np.asarray(record["T_world_from_camera"], dtype=np.float64) for record in timeline]

        usable = min(len(timeline), len(depth_files), len(intrinsics), len(transforms))
        if usable < len(timeline):
            skipped.append(
                f"chunk {chunk_id}: timeline={len(timeline)}, depths={len(depth_files)}, "
                f"intrinsics={len(intrinsics)}, poses={len(transforms)}; exporting first {usable}"
            )

        if dry_run:
            continue

        for local_index, record in enumerate(timeline[:usable]):
            if local_index % frame_stride:
                continue
            transform = np.asarray(transforms[local_index], dtype=np.float64)
            if transform.shape != (4, 4):
                skipped.append(f"frame {record['frame_index']}: invalid T_world_from_camera")
                continue
            points = _sample_frame_points(
                _read_depth(depth_files[local_index]),
                np.asarray(intrinsics[local_index], dtype=np.float64),
                transform,
                pixel_stride=pixel_stride,
                depth_min=depth_min,
                depth_max=depth_max,
                max_points=max_points_per_frame,
            )
            if points.size:
                frame_index = int(record["frame_index"])
                point_blocks.append((frame_index, points, _frame_color(frame_index)))

    n_points = sum(points.shape[0] for _frame_index, points, _color in point_blocks)
    if dry_run:
        return {
            "output": str(output),
            "n_points": 0,
            "n_frames": 0,
            "chunks": sorted(timelines),
            "pose_source": pose_source,
            "max_abs_pose_diff_vs_graph_by_chunk": pose_diffs,
            "skipped": skipped,
            "dry_run": True,
        }
    if not n_points:
        raise RuntimeError("Dense map export produced no valid depth points")

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        handle.write("ply\n")
        handle.write("format ascii 1.0\n")
        handle.write("comment DirectMe demo dense map in scene graph world frame\n")
        handle.write(f"comment source_graph={graph_json}\n")
        handle.write(f"comment pose_source={pose_source}\n")
        handle.write(f"element vertex {n_points}\n")
        handle.write("property float x\n")
        handle.write("property float y\n")
        handle.write("property float z\n")
        handle.write("property uchar red\n")
        handle.write("property uchar green\n")
        handle.write("property uchar blue\n")
        handle.write("property int frame_index\n")
        handle.write("end_header\n")
        for frame_index, points, color in point_blocks:
            r, g, b = color
            for x, y, z in points:
                handle.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b} {frame_index}\n")

    return {
        "output": str(output),
        "n_points": n_points,
        "n_frames": len(point_blocks),
        "chunks": sorted(timelines),
        "pose_source": pose_source,
        "max_abs_pose_diff_vs_graph_by_chunk": pose_diffs,
        "skipped": skipped,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export a demo dense PLY from SCAL3R depths using scene graph world poses."
    )
    parser.add_argument("--data-root", type=Path, default=REPO_DATA_ROOT)
    parser.add_argument("--graph-json", type=Path, default=DEFAULT_GRAPH_JSON)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--pixel-stride", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--depth-min", type=float, default=0.05)
    parser.add_argument("--depth-max", type=float, default=20.0)
    parser.add_argument("--max-points-per-frame", type=int, default=5000)
    parser.add_argument(
        "--pose-source",
        choices=sorted(POSE_SOURCES),
        default="scal3r",
        help=(
            "which poses to use for dense geometry. 'scal3r' keeps depths and "
            "poses from the same SCAL3R output, anchored to the first graph pose; "
            "'graph' preserves the previous behavior."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="check pose/depth/intrinsics availability and pose diffs without reading depth files or writing PLY",
    )
    args = parser.parse_args()

    if args.pixel_stride < 1 or args.frame_stride < 1:
        parser.error("--pixel-stride and --frame-stride must be >= 1")
    summary = export_dense_map(
        data_root=args.data_root,
        graph_json=args.graph_json,
        output=args.output,
        pixel_stride=args.pixel_stride,
        frame_stride=args.frame_stride,
        depth_min=args.depth_min,
        depth_max=args.depth_max,
        max_points_per_frame=args.max_points_per_frame,
        pose_source=args.pose_source,
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
