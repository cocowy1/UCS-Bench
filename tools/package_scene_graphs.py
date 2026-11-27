#!/usr/bin/env python3
"""Package processed DirectMe scene graphs for release (issue #3).

Converts the internal perception-graph runs into a portable release layout:

    <out>/
      <uid>/scene_graph.json     # keyframe paths relativized to the frames root
      manifest.json              # per-uid stats + schema info
      README.md                  # schema description + usage instructions

Keyframe paths inside the graphs are stored as
``<uid>/frame_XXXXXX.png`` (relative to the released frames root), so
consumers can re-locate them on their own machine via
``directme.datasets.frame_paths.resolve_frame_path`` or the
``--frames-root`` flag of the evaluation scripts.

This script only *prepares* the release directory; it does not upload
anything.

Usage
-----
    python tools/package_scene_graphs.py \
        --graph_root runs/ucsbench_perception_graphs \
        --frames_root /data/ucsbench/data_frames_1fps \
        --out release/ucsbench_scene_graphs_v1
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def relativize_graph(data: dict, frames_root: Path) -> tuple[dict, int, int]:
    """Rewrite every keyframe path in a graph dict to be frames-root-relative.

    Returns ``(new_data, n_relativized, n_absolute_left)``.
    """
    from directme.datasets.frame_paths import relativize_frame_path

    n_rel = n_abs = 0

    def _rewrite(p):
        nonlocal n_rel, n_abs
        if not isinstance(p, str) or not p:
            return p
        rel = relativize_frame_path(p, frames_root)
        if rel is not None and not rel.startswith("/"):
            n_rel += 1
            return rel
        n_abs += 1
        return rel if rel is not None else p

    for node in data.get("nodes", []):
        if "keyframes" in node:
            node["keyframes"] = [_rewrite(k) for k in node["keyframes"]]
        for obs in node.get("observations", []):
            if obs.get("keyframe_path"):
                obs["keyframe_path"] = _rewrite(obs["keyframe_path"])
    return data, n_rel, n_abs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--graph_root", required=True,
                        help="root containing <uid>/directme_mapping_run/scene_graph.json")
    parser.add_argument("--frames_root", required=True,
                        help="frames root the relative paths should be relative to")
    parser.add_argument("--out", required=True, help="release output directory")
    parser.add_argument("--schema_name", default="directme.scene_graph.v2")
    args = parser.parse_args()

    from directme.mapping.scene_graph import SceneGraph

    graph_root = Path(args.graph_root)
    frames_root = Path(args.frames_root)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    manifest: list[dict] = []
    n_done = 0
    for uid_dir in sorted(p for p in graph_root.iterdir() if p.is_dir()):
        src = uid_dir / "directme_mapping_run" / "scene_graph.json"
        if not src.exists():
            src = uid_dir / "scene_graph.json"
        if not src.exists():
            continue

        data = json.loads(src.read_text(encoding="utf-8"))
        data, n_rel, n_abs = relativize_graph(data, frames_root)
        if n_abs:
            print(f"[warn] {uid_dir.name}: {n_abs} keyframe paths could not be relativized")

        dst = out / uid_dir.name / "scene_graph.json"
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        # Validate the packaged graph still loads.
        try:
            g = SceneGraph.load_json(dst)
            n_nodes, n_edges = len(g.nodes), len(g.edges)
        except Exception as exc:
            print(f"[warn] {uid_dir.name}: packaged graph failed to load: {exc}")
            n_nodes = n_edges = -1

        manifest.append(
            {
                "uid": uid_dir.name,
                "path": str(dst.relative_to(out)),
                "schema": data.get("schema_version", args.schema_name),
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "n_place_nodes": len(data.get("place_nodes", {})),
                "n_keyframes_relativized": n_rel,
            }
        )
        n_done += 1
        if n_done % 50 == 0:
            print(f"[pack] {n_done} graphs packaged")

    (out / "manifest.json").write_text(
        json.dumps(
            {
                "name": "ucsbench_directme_scene_graphs",
                "schema": args.schema_name,
                "frames_root_layout": "<frames_root>/<uid>/frame_XXXXXX.png",
                "keyframe_path_convention": "relative to frames_root",
                "n_videos": len(manifest),
                "entries": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = f"""# UCS-Bench DirectMe Scene Graphs (processed)

Processed scene graphs produced by the DirectMe perception/mapping pipeline
(DA3 depth + YOLO-World open-vocabulary detection + SAM2 masks + IoU tracking
+ depth-based unprojection), one per UCS-Bench video.

## Layout

- `<uid>/scene_graph.json` — the scene graph (`{args.schema_name}` schema).
- `manifest.json` — per-video stats and file list.

## Schema overview

Each `scene_graph.json` contains:

- `nodes`: object nodes. Fields: `node_id`, `semantic_label`, `attributes`
  (color, HSV histogram, scene tag, ...), `spatial_absolute` (world-frame
  EMA anchor `p_world`, observation count, last-seen time), `observations`
  (per-detection records: timestamp, frame index, track id, `p_world`,
  detector confidence, `pose_confidence`, `keyframe_path`, bbox; newer graphs
  also carry per-observation `label` and `mask_score`), `keyframes`,
  `track_ids`, `created_at`/`updated_at`/`first_seen_timestamp`.
- `edges`: `near` (k-NN within 2 m) and `in_place` relations over final node
  positions.
- `place_nodes`: induced places (centroid + member nodes).
- `metadata`: `ego_pose_timeline` (per-frame `T_world_from_camera`),
  `place_visit_timeline`, `drift_telemetry`.
- `settings`: graph build parameters.

## Keyframe paths

Keyframe paths are stored **relative to the frames root** as
`<uid>/frame_XXXXXX.png`. To resolve them on your machine:

```python
from directme.datasets.frame_paths import resolve_frame_path
p = resolve_frame_path("01042/frame_000123.png", frames_root="/your/frames/root")
```

Both `directme/eval/eval_ucsbench.py` (`--frames_root`) and
`examples/evaluate_ucsbench_vlm.py` (`--frames-root`) accept the root and
resolve paths automatically.

## Causality note

Nodes whose first observation is after a question timestamp should be
excluded when answering that question. Graphs built with v0.6+ persist
`first_seen_timestamp` per node for this purpose; the retriever
(`directme.retrieval.retriever.GraphRetriever`, `strict_causal=True`) applies
the question-time filter to node existence, positions, keyframes, and
timelines.
"""
    (out / "README.md").write_text(readme, encoding="utf-8")

    print(f"[done] packaged {len(manifest)} graphs → {out}")


if __name__ == "__main__":
    main()
