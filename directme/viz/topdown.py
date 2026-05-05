"""Top-down 2D visualization of a DirectMe scene graph.

The plot shows the world frame from above (XZ plane), with:
  * coloured circles for object nodes, scaled by observation count,
  * "in_place" group hulls for place nodes,
  * the user's current pose as a black arrow,
  * optional ego-relative annotations from a :class:`RetrievedContext`.

Intentionally simple: matplotlib only, no Open3D / Rerun dependency. Designed
for sanity-checking long sessions and debugging fusion drift.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph


_COLOR_PALETTE = {
    "red": "#d62728", "orange": "#ff7f0e", "yellow": "#bcbd22",
    "green": "#2ca02c", "cyan": "#17becf", "blue": "#1f77b4",
    "purple": "#9467bd", "pink": "#e377c2", "brown": "#8c564b",
    "gray": "#7f7f7f", "black": "#222222", "white": "#dddddd",
}


def _node_color(node) -> str:
    name = node.attributes.get("color")
    return _COLOR_PALETTE.get(str(name).lower(), "#444444") if name else "#444444"


def render_topdown_map(
    graph: SceneGraph,
    current_pose: SE3 | None = None,
    retrieved_context: Any = None,
    figsize: tuple[float, float] = (8.0, 8.0),
    annotate: bool = True,
):
    """Build a matplotlib Figure with a top-down view of the graph.

    Returns ``(fig, ax)``. Caller is responsible for ``plt.show()`` or
    ``fig.savefig(...)``.
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrow
    except ImportError as exc:
        raise ImportError(
            "directme.viz.topdown requires matplotlib; install with "
            "`pip install directme[viz]`."
        ) from exc

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_xlabel("X (right, m)")
    ax.set_ylabel("Z (forward, m)")
    ax.set_title(
        f"DirectMe Scene Graph — {len(graph.nodes)} nodes, "
        f"{len(graph.place_nodes)} places, {len(graph.edges)} edges"
    )
    ax.grid(True, alpha=0.3)

    # Place hulls (concentric translucent disks centered on the place centroid).
    for place in graph.place_nodes.values():
        cx, _cy, cz = place["centroid"]
        members = [graph.nodes[m] for m in place["member_node_ids"] if m in graph.nodes]
        if not members:
            continue
        radii = [
            float(np.linalg.norm(np.asarray(m.p_world) - np.array([cx, _cy, cz])))
            for m in members
        ]
        radius = max(radii) + 0.4
        circle = plt.Circle((cx, cz), radius, alpha=0.10, color="tab:blue")
        ax.add_patch(circle)
        ax.text(cx, cz, place.get("label", ""), ha="center", va="center",
                fontsize=10, alpha=0.6, weight="bold")

    # Object nodes.
    for node in graph.nodes.values():
        x, _y, z = node.p_world.tolist()
        n_obs = len(node.observations)
        size = 60 + 10 * min(n_obs, 12)
        ax.scatter([x], [z], s=size, c=_node_color(node), edgecolors="black",
                   linewidths=0.7, zorder=3)
        if annotate:
            ax.text(x + 0.1, z + 0.1, node.semantic_label, fontsize=8, zorder=4)

    # User pose.
    if current_pose is not None:
        ux, _uy, uz = current_pose.translation.tolist()
        # Forward direction in world frame: T @ (0, 0, 1).
        fwd_world = current_pose.transform_points(np.array([0.0, 0.0, 0.6])) - \
                    current_pose.translation
        ax.add_patch(FancyArrow(
            ux, uz, float(fwd_world[0]), float(fwd_world[2]),
            width=0.05, head_width=0.25, head_length=0.25,
            color="black", zorder=5,
        ))
        ax.scatter([ux], [uz], s=80, c="black", marker="o", zorder=6)
        ax.text(ux + 0.15, uz - 0.25, "you", fontsize=9, zorder=6)

        # Reachability disk.
        if retrieved_context is not None:
            radius = float(getattr(retrieved_context, "reachable_radius_m", 5.0))
            ax.add_patch(plt.Circle((ux, uz), radius, fill=False, linestyle="--",
                                    color="black", alpha=0.4))

    # Ego edges from retrieved context.
    if retrieved_context is not None and current_pose is not None:
        ux, _uy, uz = current_pose.translation.tolist()
        for edge in getattr(retrieved_context, "ego_edges", []):
            target_id = edge.get("target")
            if target_id not in graph.nodes:
                continue
            tx, _ty, tz = graph.nodes[target_id].p_world.tolist()
            color = "tab:green" if edge.get("reachable") else "tab:red"
            ax.plot([ux, tx], [uz, tz], color=color, alpha=0.5, linewidth=1.4, zorder=2)
            mid_x, mid_z = (ux + tx) / 2.0, (uz + tz) / 2.0
            ax.text(mid_x, mid_z, edge.get("relation", ""),
                    fontsize=7, color=color, alpha=0.8, zorder=2,
                    ha="center", va="center")

    # Object–object near edges.
    for e in graph.edges:
        if e.get("relation") != "near":
            continue
        src, tgt = graph.nodes.get(e["source"]), graph.nodes.get(e["target"])
        if src is None or tgt is None:
            continue
        ax.plot([src.p_world[0], tgt.p_world[0]],
                [src.p_world[2], tgt.p_world[2]],
                color="gray", alpha=0.25, linewidth=0.8, zorder=1)

    fig.tight_layout()
    return fig, ax


def save_topdown_map(
    graph: SceneGraph,
    out_path: str | Path,
    current_pose: SE3 | None = None,
    retrieved_context: Any = None,
    dpi: int = 140,
    **kwargs,
) -> Path:
    """Render and save a top-down view to ``out_path`` (e.g. ``.png`` or ``.pdf``)."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, _ax = render_topdown_map(graph, current_pose, retrieved_context, **kwargs)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out
