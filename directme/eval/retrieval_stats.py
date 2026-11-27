#!/usr/bin/env python3
"""Retrieval-layer statistics over a whole QA set — no VLM required.

Measures the retrieval health metrics that drove issue #4's reproduction gap
(22.1% of questions retrieved zero keyframes):

* zero-retrieval rate       — questions where the graph returns no items
* zero-matched rate         — questions where no node positively matches the
                              parsed intent (COUNT would be 0; the VLM must
                              fall back to frames)
* zero-keyframe-path rate   — questions whose retrieved keyframe paths all
                              fail to resolve on disk
* fallback-item rate        — questions answered with weak (non-matching)
                              candidates instead of positive matches

Runs two configurations for comparison:

* ``legacy``  — strict_causal=False, soft_scoring=False (pre-v0.6 behaviour:
                hard label/color gates, existence-only causal filter)
* ``improved``— strict_causal=True,  soft_scoring=True  (v0.6 defaults)

Usage
-----
    python -m directme.eval.retrieval_stats \
        --qa_dir /data/ucsbench/MC_QAs_v20 \
        --graph_root runs/ucsbench_perception_graphs \
        --frames_root /data/ucsbench/data_frames_1fps \
        --out ucsbench_output/retrieval_stats.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def evaluate_config(video2qas, load_graph_fn, frames_root, *, strict_causal, soft_scoring, top_k, reachable_radius):
    from directme.eval.eval_ucsbench import collect_keyframe_paths
    from directme.retrieval.pose_lookup import pose_from_graph_timeline
    from directme.retrieval.retriever import GraphRetriever

    agg = defaultdict(int)
    rows = []
    n_questions = 0

    for uid, qas in video2qas.items():
        graph = load_graph_fn(uid)
        if graph is None:
            agg["videos_missing_graph"] += 1
            continue
        retriever = GraphRetriever(
            graph,
            reachable_radius_m=reachable_radius,
            strict_causal=strict_causal,
            soft_scoring=soft_scoring,
        )
        for qa in qas:
            n_questions += 1
            ts = float(qa.question_timestamp)
            pose = pose_from_graph_timeline(graph, timestamp=ts)
            ctx = retriever.retrieve(
                qa.question, pose, top_k=top_k, language="en", as_of_timestamp=ts
            )
            kf_paths = collect_keyframe_paths(ctx, frames_root=frames_root)
            n_fallback_items = sum(1 for it in ctx.items if it.match_type == "fallback")
            row = {
                "uid": uid,
                "qid": qa.qid,
                "category": qa.category,
                "n_items": len(ctx.items),
                "n_matched": ctx.count,
                "n_fallback_items": n_fallback_items,
                "n_keyframe_paths": len(kf_paths),
                "intent_labels": None,  # filled by caller if needed
            }
            rows.append(row)
            agg["n_zero_items"] += int(len(ctx.items) == 0)
            agg["n_zero_matched"] += int(ctx.count == 0)
            agg["n_zero_keyframes"] += int(len(kf_paths) == 0)
            agg["n_fallback_only"] += int(len(ctx.items) > 0 and ctx.count == 0)
            agg["items_total"] += len(ctx.items)
            agg["keyframes_total"] += len(kf_paths)

    return {
        "n_questions": n_questions,
        "zero_item_rate": agg["n_zero_items"] / max(n_questions, 1),
        "zero_matched_rate": agg["n_zero_matched"] / max(n_questions, 1),
        "zero_keyframe_rate": agg["n_zero_keyframes"] / max(n_questions, 1),
        "fallback_only_rate": agg["n_fallback_only"] / max(n_questions, 1),
        "avg_items": agg["items_total"] / max(n_questions, 1),
        "avg_keyframe_paths": agg["keyframes_total"] / max(n_questions, 1),
        "videos_missing_graph": agg["videos_missing_graph"],
        "raw_counts": dict(agg),
        "by_category": _by_category(rows),
    }, rows


def _by_category(rows):
    stats = defaultdict(lambda: {"n": 0, "zero_matched": 0, "zero_kf": 0})
    for r in rows:
        c = stats[r["category"]]
        c["n"] += 1
        c["zero_matched"] += int(r["n_matched"] == 0)
        c["zero_kf"] += int(r["n_keyframe_paths"] == 0)
    return {
        cat: {
            "n": s["n"],
            "zero_matched_rate": s["zero_matched"] / s["n"],
            "zero_keyframe_rate": s["zero_kf"] / s["n"],
        }
        for cat, s in stats.items()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--qa_dir", default="/data/ywang/dataset/SpatialMemory/MC_QAs_v20/")
    parser.add_argument("--graph_root", default="runs/ucsbench_perception_graphs")
    parser.add_argument("--frames_root", default="/data/ywang/dataset/SpatialMemory/data_frames_1fps")
    parser.add_argument("--out", default="ucsbench_output/retrieval_stats.json")
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--reachable_radius", type=float, default=10.0)
    parser.add_argument("--save_rows", action="store_true", help="also dump per-question rows")
    args = parser.parse_args()

    from directme.eval.eval_ucsbench import load_qa_items_from_dir
    from directme.mapping.scene_graph import SceneGraph

    video2qas = load_qa_items_from_dir(args.qa_dir)
    print(f"[stats] {sum(len(v) for v in video2qas.values())} questions over {len(video2qas)} videos")

    graph_cache: dict[str, object] = {}

    def load_graph_fn(uid: str):
        if uid in graph_cache:
            return graph_cache[uid]
        for cand in (
            Path(args.graph_root) / uid / "directme_mapping_run" / "scene_graph.json",
            Path(args.graph_root) / uid / "scene_graph.json",
        ):
            if cand.exists():
                g = SceneGraph.load_json(cand)
                graph_cache[uid] = g
                # Bound memory: keep only the most recent few graphs.
                if len(graph_cache) > 4:
                    for k in list(graph_cache)[:-4]:
                        del graph_cache[k]
                return g
        return None

    results = {}
    rows_all = {}
    for name, cfg in (
        ("legacy", dict(strict_causal=False, soft_scoring=False)),
        ("improved", dict(strict_causal=True, soft_scoring=True)),
    ):
        print(f"[stats] evaluating config: {name} ({cfg})")
        summary, rows = evaluate_config(
            video2qas, load_graph_fn, args.frames_root,
            top_k=args.top_k, reachable_radius=args.reachable_radius, **cfg,
        )
        results[name] = summary
        rows_all[name] = rows
        print(f"[stats] {name}: zero_item={summary['zero_item_rate']:.1%} "
              f"zero_matched={summary['zero_matched_rate']:.1%} "
              f"zero_kf={summary['zero_keyframe_rate']:.1%} "
              f"avg_items={summary['avg_items']:.1f} "
              f"avg_kf={summary['avg_keyframe_paths']:.1f}")

    if args.save_rows:
        results["rows"] = {k: [dict(r, intent_labels=None) for r in v] for k, v in rows_all.items()}

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[stats] saved → {out}")


if __name__ == "__main__":
    main()
