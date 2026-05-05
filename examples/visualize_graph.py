"""Render a top-down PNG of an existing scene graph + an optional query overlay.

Usage::

    python examples/visualize_graph.py --graph runs/toy/scene_graph.json \\
        --question "我身边有几个红杯子？" \\
        --current-pose-json "[[1,0,0,7],[0,1,0,0],[0,0,1,0],[0,0,0,1]]" \\
        --out runs/toy/topdown.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph
from directme.retrieval.retriever import GraphRetriever
from directme.viz import save_topdown_map


def _pose_from_json(value: str | None) -> SE3:
    if not value:
        return SE3.identity()
    return SE3.from_list(json.loads(value))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--graph", required=True, help="path to scene_graph.json")
    p.add_argument("--out", default=None, help="output image path (.png or .pdf)")
    p.add_argument("--question", default=None,
                   help="optional question; the matched subgraph is overlaid")
    p.add_argument("--current-pose-json", default=None,
                   help="4x4 T_world_from_current_camera as a JSON matrix")
    p.add_argument("--language", default="zh", choices=["zh", "en"])
    p.add_argument("--reachable-radius-m", type=float, default=5.0)
    args = p.parse_args()

    graph = SceneGraph.load_json(args.graph)
    pose = _pose_from_json(args.current_pose_json)
    ctx = None
    if args.question:
        ctx = GraphRetriever(graph, reachable_radius_m=args.reachable_radius_m).retrieve(
            args.question, pose, language=args.language
        )

    out = Path(args.out) if args.out else Path(args.graph).with_name("topdown.png")
    saved = save_topdown_map(graph, out, current_pose=pose, retrieved_context=ctx)
    print(f"wrote {saved}")


if __name__ == "__main__":
    main()
