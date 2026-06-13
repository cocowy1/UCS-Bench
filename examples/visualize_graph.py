#!/usr/bin/env python3
"""Render a top-down PNG/PDF of an existing scene graph + optional QA/query overlay.

支持两种用法：

1. 手动传 question：

    python examples/visualize_graph.py \
        --graph tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json \
        --question "我身边有几个红杯子？" \
        --current-pose-json "[[1,0,0,7],[0,1,0,0],[0,0,1,0],[0,0,0,1]]" \
        --out tmp/directme_scal3r_full_pipeline/directme_mapping_run/topdown2.png

2. 从 UCS-Bench/DirectMe QA JSON 中读取 question + question_timestamps：

    python examples/visualize_graph.py \
        --graph tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json \
        --qa-json scene0804_00-0-mcq.json \
        --qid scene0804_00-0_3 \
        --language zh \
        --out tmp/directme_scal3r_full_pipeline/directme_mapping_run/topdown3.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from directme.geometry.poses import SE3
from directme.mapping.scene_graph import SceneGraph
from directme.retrieval.pose_lookup import pose_from_graph_timeline
from directme.retrieval.retriever import GraphRetriever
from directme.viz import save_topdown_map


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _ts_sec(ts: str | int | float | None) -> float | None:
    """Convert 'HH:MM:SS' / 'MM:SS' / seconds to float seconds."""
    if ts is None:
        return None

    if isinstance(ts, (int, float)):
        return float(ts)

    value = str(ts).strip()
    if not value:
        return None

    parts = [float(p) for p in value.split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return float(parts[0])


def _pose_from_json(value: str | None) -> SE3 | None:
    """Parse 4x4 T_world_from_current_camera JSON matrix."""
    if not value:
        return None
    return SE3.from_list(json.loads(value))


def _load_qa_items(path: str | Path) -> list[dict[str, Any]]:
    data = _load_json(path)

    if isinstance(data, list):
        return data

    # 兼容 {"items": [...]} / {"qas": [...]} / {"questions": [...]} 等格式。
    if isinstance(data, dict):
        for key in ("items", "qas", "questions", "data"):
            if isinstance(data.get(key), list):
                return data[key]

    raise ValueError(f"Unsupported QA JSON format: {path}")


def _select_qa_item(
    qa_items: list[dict[str, Any]],
    *,
    qid: str | None,
    qa_index: int,
) -> dict[str, Any]:
    if qid:
        for item in qa_items:
            item_qid = item.get("qid") or item.get("q_id")
            if str(item_qid) == str(qid):
                return item
        raise ValueError(f"Cannot find qid={qid} in QA JSON")

    if qa_index < 0 or qa_index >= len(qa_items):
        raise IndexError(
            f"--qa-index {qa_index} is out of range. "
            f"QA JSON contains {len(qa_items)} items."
        )

    return qa_items[qa_index]


def _question_from_qa(item: dict[str, Any], language: str) -> str:
    if language == "zh" and item.get("question_chinese"):
        return str(item["question_chinese"])
    return str(item.get("question") or item.get("question_chinese") or "")


def _resolve_out_path(out_arg: str | None, graph_path: str | Path, qid: str | None) -> Path:
    if out_arg:
        out = Path(out_arg)

        # 如果传的是目录，自动补一个文件名。
        if out.exists() and out.is_dir():
            name = f"topdown_{qid}.png" if qid else "topdown.png"
            return out / name

        # 如果没有后缀，也当成目录处理。
        if out.suffix.lower() not in {".png", ".pdf"}:
            name = f"topdown_{qid}.png" if qid else "topdown.png"
            return out / name

        return out

    graph_path = Path(graph_path)
    name = f"topdown_{qid}.png" if qid else "topdown.png"
    return graph_path.with_name(name)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Visualize DirectMe scene_graph.json with optional QA/query overlay.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--graph",
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json",
        help="path to scene_graph.json",
    )
    p.add_argument(
        "--out",
        default="/data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline/directme_mapping_run/",
        help="output image path (.png or .pdf), or a directory",
    )

    # 手动 query 模式。
    p.add_argument(
        "--question",
        default=None,
        help="optional manual question; matched subgraph will be overlaid",
    )
    p.add_argument(
        "--current-pose-json",
        default=None,
        help="optional 4x4 T_world_from_current_camera JSON matrix; overrides QA timestamp pose",
    )

    # QA JSON 模式。
    p.add_argument(
        "--qa-json",
        default="/data/ywang/dataset/SpatialMemory/MC_QAs_v15_norm/scene0715_00-0-mcq.json",
        help="optional UCS-Bench/DirectMe QA JSON. If provided, question and timestamp are read from it.",
    )
    p.add_argument(
        "--qid",
        default="scene0715_00-0_6",
        help="qid to visualize, e.g. scene0715_00-0_6. If omitted, --qa-index is used.",
    )
    p.add_argument(
        "--qa-index",
        type=int,
        default=0,
        help="QA item index used when --qid is not provided.",
    )
    p.add_argument(
        "--timestamp-s",
        type=float,
        default=None,
        help="optional timestamp in seconds; overrides question_timestamps from QA JSON",
    )

    p.add_argument("--language", default="en", choices=["zh", "en"])
    p.add_argument("--reachable-radius-m", type=float, default=5.0)

    args = p.parse_args()

    graph = SceneGraph.load_json(args.graph)

    qa_item: dict[str, Any] | None = None
    qid: str | None = None

    # ------------------------------------------------------------------
    # 1. Resolve question + timestamp
    # ------------------------------------------------------------------
    question = args.question
    timestamp_s = args.timestamp_s

    if args.qa_json:
        qa_items = _load_qa_items(args.qa_json)
        qa_item = _select_qa_item(
            qa_items,
            qid=args.qid,
            qa_index=args.qa_index,
        )

        qid = str(qa_item.get("qid") or qa_item.get("q_id") or args.qa_index)

        if not question:
            question = _question_from_qa(qa_item, args.language)

        if timestamp_s is None:
            timestamp_s = _ts_sec(qa_item.get("question_timestamps"))

    if not question:
        question = None

    # ------------------------------------------------------------------
    # 2. Resolve current pose
    #    Priority:
    #    --current-pose-json > QA timestamp / --timestamp-s > identity
    # ------------------------------------------------------------------
    pose = _pose_from_json(args.current_pose_json)

    if pose is None and timestamp_s is not None:
        pose = pose_from_graph_timeline(graph, timestamp=timestamp_s)

    if pose is None:
        pose = SE3.identity()

    # ------------------------------------------------------------------
    # 3. Retrieve matched context
    # ------------------------------------------------------------------
    ctx = None

    if question:
        ctx = GraphRetriever(
            graph,
            reachable_radius_m=args.reachable_radius_m,
        ).retrieve(
            question,
            pose,
            language=args.language,
            as_of_timestamp=timestamp_s,
        )

    # ------------------------------------------------------------------
    # 4. Save top-down map
    # ------------------------------------------------------------------
    out = _resolve_out_path(args.out, args.graph, qid)

    out.parent.mkdir(parents=True, exist_ok=True)

    saved = save_topdown_map(
        graph,
        out,
        current_pose=pose,
        retrieved_context=ctx,
    )

    print(f"[graph] {args.graph}")
    if args.qa_json:
        print(f"[qa]    {args.qa_json}")
        print(f"[qid]   {qid}")
    if question:
        print(f"[question] {question}")
    if timestamp_s is not None:
        print(f"[timestamp_s] {timestamp_s:.3f}")
    print(f"[out] wrote {saved}")


if __name__ == "__main__":
    main()
