#!/usr/bin/env python3
"""Global geometric-consistency diagnostics for DA3 on long egocentric videos.

Motivation (senior-reviewer question 1): on long sequences the per-chunk
depth/pose estimates are stitched by *rigid chaining* (see
``directme/mapping/pose_propagation.py`` — no overlap frames, no Umeyama, no
loop closure), so pose / reconstruction error accumulates and directly
degrades object association, place induction, and spatial reasoning.

This tool re-runs DA3 per chunk over a video's frames (chunk size
configurable), reproduces the exact production pose chaining, and produces:

Artifacts (``<out>/``)
  * ``poses.json``                     — per-frame local + world poses, chunk ids
  * ``depth/chunk_XXXXXX.npz``         — downsampled per-frame metric depth
  * ``points/chunk_XXXXXX.npz``        — subsampled unprojected world points
  * ``pointcloud_by_chunk.ply``        — full point cloud coloured by chunk
  * ``trajectory_topdown.png``         — ego trajectory, chunk colours + boundaries
  * ``boundary_overlap_<k>.png``       — chunk k (red) vs k+1 (blue) world points
  * ``scene_graph_map.png``            — scene graph map (if ``--graph_json`` given)
  * ``metrics.json``                   — all quantitative results below

Metrics
  1. boundary_frame_jump     — per-frame ego translation across each chunk
                               boundary window (continuity stress check; the
                               chained trajectory itself is continuous by
                               construction).
  2. boundary_overlap_m      — symmetric nearest-neighbour distance (median /
                               P95) between the world point clouds of the last
                               W frames of chunk k and the first W frames of
                               chunk k+1: direct measurement of static-structure
                               misalignment introduced at the boundary.
  3. boundary_scale_ratio    — median-depth ratio of the two frames straddling
                               the boundary (per-chunk metric-scale drift).
  4. revisit_overlap_m       — same overlap metric for loop revisits (camera
                               returns within ``revisit_radius_m`` after
                               ``revisit_min_gap_s``): long-range drift.
  5. static_object_stability — (if ``--graph_json``) MAD of static-labelled
                               nodes' world positions over time, plus node /
                               track fragmentation stats.

Usage
-----
    python -m directme.eval.geometry_diagnostics \
        --uid D3-P3-am-museum_part001 \
        --frames_root /data/ucsbench/data_frames_1fps \
        --chunk_size 60 \
        --out runs/geometry_diagnostics/D3-P3-am-museum_part001_c60
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Core per-chunk processing
# ─────────────────────────────────────────────────────────────────────────────


def load_da3_adapter(model_id: str, device: str, process_res: int):
    from directme.perception.adapters.depth_anything3 import DepthAnything3Adapter

    return DepthAnything3Adapter(model_id=model_id, device=device, process_res=process_res)


def unproject_frame(
    depth: np.ndarray,
    intrinsics: np.ndarray,
    stride: int = 4,
    depth_min_m: float = 0.05,
    depth_max_m: float = 20.0,
    max_points: int = 6000,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Subsampled pinhole back-projection of one frame → camera-frame points."""
    h, w = depth.shape
    vs, us = np.mgrid[0:h:stride, 0:w:stride]
    z = depth[vs, us].astype(np.float32)
    valid = np.isfinite(z) & (z >= depth_min_m) & (z <= depth_max_m)
    vs, us, z = vs[valid], us[valid], z[valid]
    if len(z) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    x = (us.astype(np.float32) - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (vs.astype(np.float32) - intrinsics[1, 2]) * z / intrinsics[1, 1]
    pts = np.stack([x, y, z], axis=1)
    if len(pts) > max_points:
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(len(pts), size=max_points, replace=False)
        pts = pts[idx]
    return pts


def nn_distances_symmetric(a: np.ndarray, b: np.ndarray, max_points: int = 30000) -> np.ndarray:
    """Symmetric nearest-neighbour distances between two point sets (metres)."""
    try:
        from scipy.spatial import cKDTree

        tree_a, tree_b = cKDTree(a), cKDTree(b)
        d_ab = tree_a.query(b, k=1)[0]
        d_ba = tree_b.query(a, k=1)[0]
    except ImportError:
        # Brute-force fallback (chunked to bound memory).
        rng = np.random.default_rng(0)

        def _sub(x: np.ndarray) -> np.ndarray:
            return x[rng.choice(len(x), size=min(len(x), max_points), replace=False)] if len(x) > max_points else x

        a2, b2 = _sub(a.astype(np.float32)), _sub(b.astype(np.float32))
        d_ab = np.min(np.linalg.norm(b2[None, :, :] - a2[:, None, :], axis=2), axis=1) \
            if len(b2) < 8000 else _bf_chunked(a2, b2)
        d_ba = _bf_chunked(b2, a2) if len(b2) >= 8000 else np.min(
            np.linalg.norm(a2[None, :, :] - b2[:, None, :], axis=2), axis=1
        )
    return np.concatenate([d_ab, d_ba])


def _bf_chunked(q: np.ndarray, ref: np.ndarray, chunk: int = 512) -> np.ndarray:
    out = np.empty(len(q), dtype=np.float32)
    for i in range(0, len(q), chunk):
        d = np.linalg.norm(ref[None, :, :] - q[i:i + chunk, None, :], axis=2)
        out[i:i + chunk] = d.min(axis=1)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PLY export
# ─────────────────────────────────────────────────────────────────────────────


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray | None = None) -> None:
    """Minimal binary PLY writer (compatible with MeshLab / three.js viewers)."""
    n = len(points)
    has_color = colors is not None and len(colors) == n
    header = [
        "ply",
        "format binary_little_endian 1.0",
        f"element vertex {n}",
        "property float x", "property float y", "property float z",
    ]
    if has_color:
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header += ["end_header"]
    if has_color:
        pts = points.astype(np.float32)
        cols = np.clip(colors, 0, 255).astype(np.uint8)
        data = np.empty(n, dtype=[("xyz", np.float32, 3), ("rgb", np.uint8, 3)])
        data["xyz"], data["rgb"] = pts, cols
    else:
        data = np.ascontiguousarray(points.astype(np.float32))
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        f.write(data.tobytes())


def chunk_color(chunk_id: int, n_chunks: int) -> tuple[int, int, int]:
    import colorsys

    h = (chunk_id / max(n_chunks, 1)) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 255.0)
    return int(r), int(g), int(b)


# ─────────────────────────────────────────────────────────────────────────────
# Static-object stability from an existing scene graph
# ─────────────────────────────────────────────────────────────────────────────

# Objects365-style labels that are essentially immovable; used to measure
# world-coordinate stability of static structure across time.
STATIC_LABEL_TOKENS = {
    "table", "desk", "bed", "sofa", "couch", "chair", "stool", "fridge",
    "refrigerator", "oven", "microwave", "sink", "toilet", "door", "window",
    "tv", "monitor", "picture", "frame", "cabinet", "shelf", "counter",
    "rail", "fence", "wall", "ceiling", "floor", "stairs", "elevator",
    "escalator", "machine", "pillar", "column", "desk", "wardrobe",
}


def _is_static_label(label: str) -> bool:
    toks = set(str(label).lower().replace("/", " ").replace("_", " ").split())
    return bool(toks & STATIC_LABEL_TOKENS)


def static_object_stability(graph) -> dict:
    """World-position MAD / span of static-labelled nodes + fragmentation stats."""
    per_label_nodes: dict[str, int] = {}
    per_label_tracks: dict[str, set] = {}
    stability: list[dict] = []
    for node in graph.nodes.values():
        lbl = node.semantic_label
        per_label_nodes[lbl] = per_label_nodes.get(lbl, 0) + 1
        per_label_tracks.setdefault(lbl, set()).update(node.track_ids)
        if not _is_static_label(lbl) or len(node.observations) < 3:
            continue
        pts = np.asarray([o.p_world for o in node.observations if o.p_world], dtype=float)
        if pts.ndim != 2 or len(pts) < 3 or not np.all(np.isfinite(pts)):
            continue
        mad = float(np.median(np.abs(pts - np.median(pts, axis=0))))
        stability.append(
            {
                "node_id": node.node_id,
                "label": lbl,
                "n_obs": int(len(pts)),
                "time_span_s": float(pts.shape[0] and (
                    node.observations[-1].timestamp - node.observations[0].timestamp
                )),
                "position_mad_m": round(mad, 4),
                "position_span_m": round(
                    float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))), 4
                ),
            }
        )
    fragmentation = {
        lbl: {"nodes": n, "track_ids": len(per_label_tracks[lbl])}
        for lbl, n in sorted(per_label_nodes.items(), key=lambda kv: -kv[1])[:20]
    }
    if stability:
        mads = [s["position_mad_m"] for s in stability]
        summary = {
            "n_static_nodes_measured": len(stability),
            "position_mad_m_median": float(np.median(mads)),
            "position_mad_m_p90": float(np.percentile(mads, 90)),
        }
    else:
        summary = {"n_static_nodes_measured": 0}
    return {"summary": summary, "worst_nodes": sorted(stability, key=lambda s: -s["position_mad_m"])[:20],
            "fragmentation_top20": fragmentation}


# ─────────────────────────────────────────────────────────────────────────────
# Main diagnostic loop
# ─────────────────────────────────────────────────────────────────────────────


def run_diagnostics(args) -> dict:
    import torch
    from directme.datasets.frame_paths import iter_video_frames
    from directme.geometry.poses import SE3
    from directme.mapping.pose_propagation import ChunkPosePropagator

    out_dir = Path(args.out)
    (out_dir / "depth").mkdir(parents=True, exist_ok=True)
    (out_dir / "points").mkdir(parents=True, exist_ok=True)

    frame_paths = iter_video_frames(args.frames_root, args.uid)
    if args.max_frames:
        frame_paths = frame_paths[: args.max_frames]
    if not frame_paths:
        raise FileNotFoundError(f"no frames under {args.frames_root}/{args.uid}")
    print(f"[diag] {len(frame_paths)} frames, chunk_size={args.chunk_size}")

    adapter = load_da3_adapter(args.da3_model_id, args.device, args.da3_process_res)
    propagator = ChunkPosePropagator(max_per_frame_jump_m=args.max_per_frame_jump_m)

    chunk_ids = list(range(0, len(frame_paths), args.chunk_size))
    n_chunks = len(chunk_ids)

    poses_records: list[dict] = []
    all_points: dict[int, np.ndarray] = {}
    boundary_metrics: list[dict] = []
    prev_chunk_tail_points = None
    prev_chunk_last_output = None
    chunk_summaries: list[dict] = []
    rejected_chunks: list[tuple] = []

    for ci, start in enumerate(chunk_ids):
        paths = frame_paths[start:start + args.chunk_size]
        chunk_id = ci
        print(f"[diag] chunk {chunk_id + 1}/{n_chunks}: frames {start}..{start + len(paths) - 1}")

        with torch.inference_mode():
            outputs = adapter.infer(paths)

        local_poses = [o.pose_local for o in outputs]
        result = propagator.propagate(chunk_id, local_poses)
        if not result.accepted:
            rejected_chunks.append((chunk_id, result.rejection_reason))
            prev_chunk_tail_points = None
            prev_chunk_last_output = None
            continue

        # Per-frame bookkeeping + unprojection.
        chunk_pts_world: list[np.ndarray] = []
        depth_store: dict[str, np.ndarray] = {}
        for j, (p, o, wp) in enumerate(zip(paths, outputs, result.world_poses)):
            fidx = start + j
            poses_records.append(
                {
                    "frame_index": fidx,
                    "timestamp": fidx / max(args.fps, 1e-6),
                    "chunk_id": chunk_id,
                    "T_local": o.pose_local.to_list(),
                    "T_world": wp.to_list(),
                    "pose_confidence": float(getattr(o, "pose_confidence", 1.0) or 1.0),
                }
            )
            pts_cam = unproject_frame(
                o.depth, o.intrinsics, stride=args.point_stride,
                max_points=args.max_points_per_frame,
            )
            pts_world = wp.transform_points(pts_cam) if len(pts_cam) else pts_cam
            chunk_pts_world.append(pts_world.astype(np.float32))
            # Downsampled depth snapshot (float16, stride 2) for later re-analysis.
            ds = o.depth[::2, ::2].astype(np.float16)
            depth_store[f"frame_{fidx:06d}"] = ds
            depth_store[f"intrinsics_{fidx:06d}"] = o.intrinsics.astype(np.float32)

        np.savez_compressed(
            out_dir / "depth" / f"chunk_{chunk_id:06d}.npz", **depth_store
        )
        pts_cat = np.concatenate(chunk_pts_world, axis=0) if chunk_pts_world else np.zeros((0, 3), np.float32)
        all_points[chunk_id] = pts_cat
        np.savez_compressed(
            out_dir / "points" / f"chunk_{chunk_id:06d}.npz", points=pts_cat
        )
        chunk_summaries.append(
            {
                "chunk_id": chunk_id,
                "n_frames": len(paths),
                "n_points": int(len(pts_cat)),
                "median_depth_m": float(
                    np.median(np.concatenate([o.depth.ravel() for o in outputs]))
                ),
                "path_length_m": float(
                    np.sum(
                        np.linalg.norm(
                            np.diff(
                                np.stack([p.translation for p in result.world_poses]), axis=0
                            ),
                            axis=1,
                        )
                    )
                ) if len(result.world_poses) > 1 else 0.0,
            }
        )

        # ── Boundary metrics vs previous accepted chunk ──
        if prev_chunk_tail_points is not None and prev_chunk_last_output is not None and len(result.world_poses) > 0:
            w = min(args.boundary_window, len(chunk_pts_world))
            head_points = np.concatenate(chunk_pts_world[:w], axis=0)
            tail_points = prev_chunk_tail_points

            nn = nn_distances_symmetric(tail_points, head_points)
            last_depth = prev_chunk_last_output["median_depth_m"]
            first_depth = chunk_summaries[-1]["median_depth_m"]

            boundary_metrics.append(
                {
                    "boundary": f"{prev_chunk_last_output['chunk_id']}->{chunk_id}",
                    "n_points_tail": int(len(tail_points)),
                    "n_points_head": int(len(head_points)),
                    "overlap_median_m": round(float(np.median(nn)), 4),
                    "overlap_p75_m": round(float(np.percentile(nn, 75)), 4),
                    "overlap_p95_m": round(float(np.percentile(nn, 95)), 4),
                    "scale_ratio_tail_over_head": round(
                        float(last_depth / max(first_depth, 1e-6)), 4
                    ),
                }
            )

            # Boundary overlap figure: last W frames of prev chunk (red) vs
            # first W frames of this chunk (blue), top-down XZ projection.
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(7, 7))
                ax.scatter(tail_points[:, 0], tail_points[:, 2], s=1.5, c="red", alpha=0.35, label=f"chunk {prev_chunk_last_output['chunk_id']} tail")
                ax.scatter(head_points[:, 0], head_points[:, 2], s=1.5, c="blue", alpha=0.35, label=f"chunk {chunk_id} head")
                ax.set_aspect("equal")
                ax.set_xlabel("X (m)")
                ax.set_ylabel("Z (m)")
                ax.set_title(
                    f"Boundary {prev_chunk_last_output['chunk_id']}->{chunk_id}  "
                    f"med NN={boundary_metrics[-1]['overlap_median_m']:.3f} m  "
                    f"P95={boundary_metrics[-1]['overlap_p95_m']:.3f} m"
                )
                ax.legend(markerscale=6)
                fig.tight_layout()
                fig.savefig(out_dir / f"boundary_overlap_{prev_chunk_last_output['chunk_id']:06d}.png", dpi=130)
                plt.close(fig)
            except Exception as exc:  # visualization must never kill the metrics
                print(f"[diag][warn] boundary figure failed: {exc}")

        # Keep tail window + chunk summary for the next boundary comparison.
        w = min(args.boundary_window, len(chunk_pts_world))
        prev_chunk_tail_points = (
            np.concatenate(chunk_pts_world[-w:], axis=0) if w else None
        )
        prev_chunk_last_output = {
            "chunk_id": chunk_id,
            "median_depth_m": chunk_summaries[-1]["median_depth_m"],
        }

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Per-frame jump across boundaries (continuity stress) ──
    trans = np.asarray(
        [np.asarray(p["T_world"])[None, :3, 3] for p in poses_records], dtype=float
    ).reshape(-1, 3)
    chunk_of = [p["chunk_id"] for p in poses_records]
    jumps = np.linalg.norm(np.diff(trans, axis=0), axis=1)
    boundary_jumps = [
        {
            "boundary": f"{chunk_of[i]}->{chunk_of[i + 1]}",
            "frame_jump_m": round(float(jumps[i]), 4),
            "prev_5_avg_m": round(float(np.mean(jumps[max(0, i - 5):i])), 4),
        }
        for i in range(len(jumps))
        if chunk_of[i] != chunk_of[i + 1]
    ]

    # ── Loop-revisit consistency ──
    revisit_metrics: list[dict] = []
    ts = np.asarray([p["timestamp"] for p in poses_records], dtype=float)
    chunk_arr = np.asarray(chunk_of)
    i_idx, j_idx = np.where(
        (np.linalg.norm(trans[None, :, :] - trans[:, None, :], axis=2) < args.revisit_radius_m)
        & (np.abs(ts[None, :] - ts[:, None]) > args.revisit_min_gap_s)
    )
    seen_pairs: set[tuple[int, int]] = set()
    for i, j in zip(i_idx.tolist(), j_idx.tolist()):
        ci_, cj = chunk_arr[i], chunk_arr[j]
        if ci_ >= cj or (ci_, cj) in seen_pairs:
            continue
        seen_pairs.add((ci_, cj))
        if len(seen_pairs) > args.max_revisit_pairs:
            break
        pi = all_points.get(int(ci_))
        pj = all_points.get(int(cj))
        if pi is None or pj is None or len(pi) == 0 or len(pj) == 0:
            continue
        nn = nn_distances_symmetric(
            pi[:: max(1, len(pi) // 30000)], pj[:: max(1, len(pj) // 30000)]
        )
        revisit_metrics.append(
            {
                "chunks": f"{ci_}<->{cj}",
                "t_i_s": round(float(ts[i]), 1),
                "t_j_s": round(float(ts[j]), 1),
                "ego_gap_s": round(float(abs(ts[j] - ts[i])), 1),
                "overlap_median_m": round(float(np.median(nn)), 4),
                "overlap_p95_m": round(float(np.percentile(nn, 95)), 4),
            }
        )

    # ── Combined PLY coloured by chunk ──
    all_pts, all_cols = [], []
    for cid, pts in all_points.items():
        if not len(pts):
            continue
        all_pts.append(pts)
        all_cols.append(np.tile(np.asarray(chunk_color(cid, n_chunks), dtype=np.uint8), (len(pts), 1)))
    if all_pts:
        write_ply(out_dir / "pointcloud_by_chunk.ply", np.concatenate(all_pts),
                  np.concatenate(all_cols) if all_cols else None)

    # ── Trajectory top-down figure ──
    timeline = [
        {
            "chunk_id": p["chunk_id"],
            "timestamp": p["timestamp"],
            "translation": (np.asarray(p["T_world"])[:3, 3]).tolist(),
        }
        for p in poses_records
    ]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from directme.viz.topdown import draw_trajectory

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_aspect("equal")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"Ego trajectory — {args.uid} (chunk={args.chunk_size}, red × = boundaries)")
        ax.grid(True, alpha=0.3)
        draw_trajectory(ax, timeline)
        fig.tight_layout()
        fig.savefig(out_dir / "trajectory_topdown.png", dpi=140)
        plt.close(fig)
    except Exception as exc:
        print(f"[diag][warn] trajectory figure failed: {exc}")

    # ── Scene graph map + static-object stability ──
    graph_stats = None
    if args.graph_json:
        try:
            from directme.mapping.scene_graph import SceneGraph
            from directme.viz.topdown import save_topdown_map

            graph = SceneGraph.load_json(args.graph_json)
            graph_stats = static_object_stability(graph)
            save_topdown_map(graph, out_dir / "scene_graph_map.png")
        except Exception as exc:
            print(f"[diag][warn] graph stats failed: {exc}")

    # ── Poses dump + metrics summary ──
    (out_dir / "poses.json").write_text(json.dumps(poses_records), encoding="utf-8")

    summary = {
        "uid": args.uid,
        "chunk_size": args.chunk_size,
        "n_frames": len(frame_paths),
        "n_chunks": n_chunks,
        "rejected_chunks": rejected_chunks,
        "drift_telemetry": propagator.drift_telemetry(),
        "chunk_summaries": chunk_summaries,
        "boundary_frame_jumps": boundary_jumps,
        "boundary_overlap": boundary_metrics,
        "revisit_overlap": revisit_metrics,
        "static_object_stability": graph_stats,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # Console digest.
    print("\n===== geometry diagnostics digest =====")
    print(f"chunks accepted: {n_chunks - len(rejected_chunks)}/{n_chunks};  rejected: {rejected_chunks}")
    if boundary_metrics:
        meds = [b["overlap_median_m"] for b in boundary_metrics]
        p95s = [b["overlap_p95_m"] for b in boundary_metrics]
        print(f"boundary overlap: median-of-medians={np.median(meds):.3f} m; median P95={np.median(p95s):.3f} m")
        worst = sorted(boundary_metrics, key=lambda b: -b["overlap_median_m"])[:5]
        for b in worst:
            print(f"  worst boundary {b['boundary']}: med={b['overlap_median_m']} m, p95={b['overlap_p95_m']} m, scale_ratio={b['scale_ratio_tail_over_head']}")
    if revisit_metrics:
        for rv in revisit_metrics[:5]:
            print(f"revisit {rv['chunks']} (Δt={rv['ego_gap_s']} s): med={rv['overlap_median_m']} m, p95={rv['overlap_p95_m']} m")
    else:
        print("revisit overlap: no loop revisits detected (trajectory without closures)")
    if graph_stats and graph_stats["summary"].get("n_static_nodes_measured"):
        s = graph_stats["summary"]
        print(f"static-object stability: {s['n_static_nodes_measured']} nodes, MAD median={s['position_mad_m_median']:.3f} m, p90={s['position_mad_m_p90']:.3f} m")
    print(f"artifacts → {out_dir}")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uid", required=True, help="video uid (subdir under --frames_root)")
    parser.add_argument("--frames_root", default="/data/ywang/dataset/SpatialMemory/data_frames_1fps")
    parser.add_argument("--out", required=True, help="output directory for artifacts")
    parser.add_argument("--chunk_size", type=int, default=60)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--da3_model_id", default="depth-anything/DA3NESTED-GIANT-LARGE-1.1")
    parser.add_argument("--da3_process_res", type=int, default=504)
    parser.add_argument("--max_per_frame_jump_m", type=float, default=5.0)
    parser.add_argument("--point_stride", type=int, default=4, help="pixel stride for unprojection subsampling")
    parser.add_argument("--max_points_per_frame", type=int, default=6000)
    parser.add_argument("--boundary_window", type=int, default=10, help="frames per side for boundary overlap")
    parser.add_argument("--revisit_radius_m", type=float, default=2.0)
    parser.add_argument("--revisit_min_gap_s", type=float, default=60.0)
    parser.add_argument("--max_revisit_pairs", type=int, default=5)
    parser.add_argument("--graph_json", default="", help="optional existing scene_graph.json for static-object stats/map")
    args = parser.parse_args()
    run_diagnostics(args)


if __name__ == "__main__":
    main()
