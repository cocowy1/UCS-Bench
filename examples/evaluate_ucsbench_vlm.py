#!/usr/bin/env python3
"""UCS-Bench evaluation with DirectMe + VLM (public reproduction entrypoint).

This script runs the DirectMe evaluation pipeline on UCS-Bench and is kept
**behaviourally aligned** with the internal evaluator
(``directme.eval.eval_ucsbench``):

1. Load pre-built scene graphs (one per video).
2. For each question: resolve the ego pose at the question timestamp and
   retrieve the relevant subgraph **causally** (only observations at/before
   the question time).
3. Build the same "spatial memory prior" text as the internal eval and
   CLIP-rerank the graph keyframes against the question (top-32). When too
   few keyframes survive, fill up with CLIP-selected original video frames
   from before the question time so the VLM always has visual evidence.
4. Assemble the multiple-choice prompt (graph = auxiliary evidence, verify
   against frames) and send it to the VLM.
5. Parse the answer, compare with ground truth, report per-dimension and
   overall accuracy.

Prerequisites
-------------

* Pre-built scene graphs: run the offline pipeline on each UCS-Bench video::

    python examples/run_real_pipeline.py \
        --frames /data/ucsbench/video_001/frames \
        --out /data/ucsbench/graphs/video_001 \
        --classes "cup,phone,bottle,chair,table,sink,fridge,..." \
        --storage-backend json

* UCS-Bench annotation file (JSONL), one JSON object per line with fields::

    {
      "video_uid":       "video_001",        # must match a subdir name in --graphs-dir
      "query_timestamp": 81.0,               # seconds into the video
      "question":        "Where is the vending machine relative to me?",
      "options":         ["A. On your left", "B. Behind you", ...],  # 5-way MC
      "answer_idx":      2,                  # 0-based index of the correct option
      "dimension":       "position_orientation"
    }

* ``--frames-root`` should point at the extracted 1-fps frames
  (``<frames-root>/<video_uid>/frame_XXXXXX.png``). Graphs built on another
  machine store absolute keyframe paths; this root lets the script re-locate
  them (see ``directme.datasets.frame_paths``).

Usage
-----

Via OpenAI-compatible API (vLLM serving Qwen3-VL)::

    python examples/evaluate_ucsbench_vlm.py \
        --questions /data/ucsbench/questions.jsonl \
        --graphs-dir /data/ucsbench/graphs \
        --frames-root /data/ucsbench/data_frames_1fps \
        --backend openai \
        --model qwen3-vl-8b-instruct \
        --base-url http://localhost:8000/v1 \
        --out results/directme_qwen3vl.json

Via local Transformers::

    python examples/evaluate_ucsbench_vlm.py \
        --questions /data/ucsbench/questions.jsonl \
        --graphs-dir /data/ucsbench/graphs \
        --frames-root /data/ucsbench/data_frames_1fps \
        --backend transformers \
        --model Qwen/Qwen3-VL-8B-Instruct \
        --clip-model ckpts/clip-vit-base-patch32 \
        --out results/directme_qwen3vl.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_graphs(graphs_dir: str) -> dict[str, Any]:
    """Load all scene graphs from a directory (one .json per video_uid)."""
    from directme.mapping.scene_graph import SceneGraph

    graphs: dict[str, SceneGraph] = {}
    gdir = Path(graphs_dir)
    for p in sorted(gdir.rglob("scene_graph.json")):
        vid = p.parent.name
        graphs[vid] = SceneGraph.load_json(p)
    # Also handle flat layout: graphs_dir/video_001.json
    for p in sorted(gdir.glob("*.json")):
        vid = p.stem
        if vid not in graphs:
            graphs[vid] = SceneGraph.load_json(p)
    return graphs


def _load_questions(path: str) -> list[dict[str, Any]]:
    """Load UCS-Bench questions from JSONL."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _get_vlm_answer(
    system: str,
    parts: list[dict[str, Any]],
    backend: str,
    model: str,
    base_url: str | None,
    api_key: str | None,
    max_new_tokens: int = 32,
    _vlm_cache: dict[str, Any] = {},
) -> str:
    """Send a prompt to the VLM and return the raw response text."""

    if backend == "openai":
        import base64
        import mimetypes
        from openai import OpenAI

        client = _vlm_cache.get("client")
        if client is None:
            client = OpenAI(api_key=api_key or "EMPTY", base_url=base_url)
            _vlm_cache["client"] = client

        content: list[dict] = []
        for part in parts:
            if part["type"] == "text":
                content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                p = Path(part["path"])
                if p.exists():
                    mime, _ = mimetypes.guess_type(str(p))
                    if mime and mime.startswith("image/"):
                        data = base64.b64encode(p.read_bytes()).decode()
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime};base64,{data}"},
                        })

        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            temperature=0.0,
            max_tokens=max_new_tokens,
        )
        return resp.choices[0].message.content or ""

    elif backend == "transformers":
        # Lazy-load model once.
        if "model" not in _vlm_cache:
            import torch
            # Qwen VL models: try Qwen3 class first, fall back to Qwen2.5.
            try:
                from transformers import Qwen3VLForConditionalGeneration as _QwenVL
            except ImportError:
                from transformers import Qwen2_5_VLForConditionalGeneration as _QwenVL
            from transformers import AutoProcessor

            print(f"[vlm] Loading {model} ...")
            _vlm_cache["model"] = _QwenVL.from_pretrained(
                model, torch_dtype=torch.bfloat16, device_map="auto",
            )
            _vlm_cache["processor"] = AutoProcessor.from_pretrained(model)

        m = _vlm_cache["model"]
        processor = _vlm_cache["processor"]

        user_content = []
        for part in parts:
            if part["type"] == "text":
                user_content.append({"type": "text", "text": part["text"]})
            elif part["type"] == "image":
                p = Path(part["path"])
                if p.exists():
                    user_content.append({"type": "image", "image": str(p.resolve())})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user", "content": user_content},
        ]

        from qwen_vl_utils import process_vision_info
        import torch

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(m.device)

        with torch.no_grad():
            gen = m.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.0, do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    else:
        raise ValueError(f"Unknown backend: {backend}")


def _build_spatial_items(ctx) -> list[dict[str, Any]]:
    """Same spatial-prior assembly as the internal evaluator."""
    items = []
    for it in ctx.items:
        items.append({
            "node_id": it.node.node_id,
            "label": it.node.semantic_label,
            "color": it.node.attributes.get("color"),
            "relation": it.egocentric["relation"],
            "distance_m": it.egocentric["distance_m"],
            "reachable": it.egocentric["reachable"],
            "natural_language": it.egocentric["natural_language"],
            "match_type": it.match_type,
            "reliability": it.reliability.get("reliability") if it.reliability else None,
        })
    return items


def _select_keyframe_paths(
    ctx,
    question: str,
    options: list[str],
    timestamp: float,
    args,
    video_uid: str | None = None,
    clip_model=None,
    clip_processor=None,
    clip_device: str = "cuda",
) -> tuple[list[str], int, int]:
    """CLIP-reranked causal graph keyframes + original-frame fallback.

    Mirrors ``directme.eval.eval_ucsbench.run_query``: top-``--top-k`` CLIP
    rerank over the retrieved keyframes; if fewer than ``--min-keyframes``
    survive, fill up with CLIP-selected original frames from before the
    question time. Returns ``(paths, n_graph, n_fallback)``.
    """
    from directme.eval.eval_ucsbench import (
        _frame_index_from_path,
        _uniform_sample,
        clip_rerank_paths,
        collect_keyframe_paths,
    )

    keyframe_paths = collect_keyframe_paths(ctx, frames_root=args.frames_root)
    clip_query = question + "\n" + "\n".join(
        f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options)
    )

    if clip_model is not None:
        ranked = clip_rerank_paths(
            question=clip_query,
            keyframe_paths=keyframe_paths,
            clip_model=clip_model,
            clip_processor=clip_processor,
            top_k=args.top_k,
            device=clip_device,
        )
    else:
        ranked = keyframe_paths[: args.max_keyframes]
    n_graph = len(ranked)

    n_fallback = 0
    if n_graph < args.min_keyframes and args.frames_root and not args.no_frame_fallback:
        from directme.datasets.frame_paths import iter_video_frames

        all_frames = iter_video_frames(args.frames_root, str(video_uid)) if video_uid else []
        fps = max(float(args.input_fps), 1e-6)
        causal_frames = [
            p for p in all_frames
            if (idx := _frame_index_from_path(p)) is not None and idx / fps <= timestamp + 0.5
        ]
        already = set(keyframe_paths)
        causal_frames = [p for p in causal_frames if p not in already]
        causal_frames = _uniform_sample(causal_frames, 256)
        if causal_frames:
            if clip_model is not None:
                filler = clip_rerank_paths(
                    question=clip_query,
                    keyframe_paths=causal_frames,
                    clip_model=clip_model,
                    clip_processor=clip_processor,
                    top_k=max(0, min(args.min_keyframes, args.top_k) - n_graph),
                    device=clip_device,
                )
            else:
                filler = causal_frames[: max(0, args.min_keyframes - n_graph)]
            ranked = ranked + filler
            n_fallback = len(filler)

    return ranked, n_graph, n_fallback


def run(args: argparse.Namespace) -> None:
    from directme.eval.eval_ucsbench import (
        build_directme_prior_text,
        build_mc_prompt_with_prior,
        load_clip,
    )
    from directme.geometry.poses import SE3
    from directme.qa.prompts import MultipleChoicePromptBuilder
    from directme.retrieval.pose_lookup import pose_from_graph_timeline
    from directme.retrieval.retriever import GraphRetriever

    # 1. Load data.
    print(f"[data] Loading graphs from {args.graphs_dir} ...")
    graphs = _load_graphs(args.graphs_dir)
    print(f"[data] Loaded {len(graphs)} graphs")

    print(f"[data] Loading questions from {args.questions} ...")
    questions = _load_questions(args.questions)
    print(f"[data] {len(questions)} questions")

    # Optional CLIP reranker (aligned with the internal eval's 32-frame setup).
    clip_model = clip_processor = None
    clip_device = "cpu"
    if not args.no_clip_rerank:
        try:
            import torch
            clip_device = "cuda" if torch.cuda.is_available() else "cpu"
            clip_model, clip_processor = load_clip(args.clip_model, device=clip_device)
            print(f"[clip] loaded {args.clip_model} on {clip_device}")
        except Exception as exc:
            print(f"[clip] unavailable ({exc}); falling back to first-N keyframes")
            clip_model = clip_processor = None

    builder = MultipleChoicePromptBuilder(
        max_keyframes=args.max_keyframes,
        frames_root=args.frames_root,
    )

    # 2. Evaluate.
    results: list[dict[str, Any]] = []
    dim_stats: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})
    skipped = 0
    t0 = time.time()

    for i, q in enumerate(questions):
        vid = q["video_uid"]
        if vid not in graphs:
            skipped += 1
            continue

        graph = graphs[vid]
        timestamp = float(q.get("query_timestamp", q.get("timestamp", 0.0)))

        # Resolve pose (causal: newest pose at/before the question time).
        try:
            pose = pose_from_graph_timeline(graph, timestamp=timestamp)
        except Exception:
            pose = SE3.identity()

        # Retrieve subgraph causally, with the same soft scoring as the
        # internal evaluator (label/color mismatches down-weight, never
        # hard-zero; reliability re-ranks).
        retriever = GraphRetriever(
            graph,
            reachable_radius_m=args.reachable_radius_m,
            strict_causal=not args.no_strict_causal,
            soft_scoring=not args.no_soft_scoring,
        )
        context = retriever.retrieve(
            q["question"], pose, language=args.language,
            top_k=args.top_k,
            as_of_timestamp=None if args.no_strict_causal else timestamp,
        )

        # Strip option letter prefixes if present (e.g. "A. On your left" → "On your left").
        options_raw = q.get("options", [])
        if not options_raw:
            skipped += 1
            continue
        options = [re.sub(r"^[A-E][.)]\s*", "", str(opt)) for opt in options_raw]

        # Keyframes: CLIP-reranked causal graph keyframes + frame fallback.
        keyframe_paths, n_graph_kf, n_fallback_kf = _select_keyframe_paths(
            context, q["question"], options, timestamp, args,
            video_uid=vid,
            clip_model=clip_model, clip_processor=clip_processor, clip_device=clip_device,
        )

        # Same prior text + MC prompt as the internal evaluator.
        spatial_items = _build_spatial_items(context)
        prior_text = build_directme_prior_text(spatial_items, max_items=32)
        prompt = build_mc_prompt_with_prior(
            question=q["question"],
            options={chr(65 + j): opt for j, opt in enumerate(options)},
            prior_text=prior_text,
            timestamp_s=timestamp,
        )

        parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for p in keyframe_paths:
            parts.append({"type": "image", "path": p})

        # Call VLM.
        try:
            raw_answer = _get_vlm_answer(
                "", parts, args.backend, args.model,
                args.base_url, args.api_key,
                max_new_tokens=args.max_new_tokens,
            )
        except Exception as exc:
            print(f"  [WARN] VLM error on q{i}: {exc}")
            raw_answer = ""

        parsed = builder.parse_answer(raw_answer)
        gt_idx = q.get("answer_idx", q.get("correct_idx"))

        if gt_idx is not None and parsed is not None:
            gt_letter = "ABCDE"[int(gt_idx)]
            correct = parsed == gt_letter
        else:
            correct = None

        dim = q.get("dimension", "unknown")
        dim_stats[dim]["total"] += 1
        if correct:
            dim_stats[dim]["correct"] += 1

        results.append({
            "video_uid": vid,
            "query_timestamp": timestamp,
            "question": q["question"],
            "dimension": dim,
            "gt_idx": gt_idx,
            "predicted_letter": parsed,
            "raw_answer": raw_answer,
            "correct": correct,
            "num_keyframes": len(keyframe_paths),
            "num_graph_keyframes": n_graph_kf,
            "num_fallback_keyframes": n_fallback_kf,
            "zero_retrieval": context.count == 0,
        })

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(questions)}] elapsed={elapsed:.0f}s "
                  f"skipped={skipped}")

    # 3. Report.
    elapsed = time.time() - t0
    scored = [r for r in results if r["correct"] is not None]
    n_correct = sum(1 for r in scored if r["correct"])
    accuracy = n_correct / len(scored) if scored else 0.0
    zero_retrieval = sum(1 for r in results if r.get("zero_retrieval"))
    zero_frames = sum(1 for r in results if r.get("num_keyframes", 0) == 0)

    print(f"\n{'='*60}")
    print(f"DirectMe + {args.model} on UCS-Bench")
    print(f"{'='*60}")
    print(f"Total questions:  {len(questions)}")
    print(f"Evaluated:        {len(scored)}")
    print(f"Skipped:          {skipped}")
    print(f"Correct:          {n_correct}")
    print(f"Overall accuracy: {accuracy * 100:.1f}%")
    print(f"Zero-retrieval:   {zero_retrieval} ({zero_retrieval / max(len(results), 1) * 100:.1f}%)")
    print(f"Zero-keyframe:    {zero_frames} ({zero_frames / max(len(results), 1) * 100:.1f}%)")
    print(f"Time:             {elapsed:.1f}s")
    print()
    print("Per-dimension breakdown:")
    for dim in sorted(dim_stats):
        s = dim_stats[dim]
        acc = s["correct"] / s["total"] * 100 if s["total"] > 0 else 0
        print(f"  {dim:<30} {s['correct']:>4}/{s['total']:<4} = {acc:.1f}%")

    # 4. Save results.
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        report = {
            "model": args.model,
            "backend": args.backend,
            "overall_accuracy": accuracy,
            "n_total": len(questions),
            "n_scored": len(scored),
            "n_correct": n_correct,
            "n_skipped": skipped,
            "n_zero_retrieval": zero_retrieval,
            "n_zero_keyframe": zero_frames,
            "elapsed_seconds": elapsed,
            "config": {
                "top_k": args.top_k,
                "reachable_radius_m": args.reachable_radius_m,
                "strict_causal": not args.no_strict_causal,
                "soft_scoring": not args.no_soft_scoring,
                "clip_rerank": not args.no_clip_rerank,
                "frame_fallback": not args.no_frame_fallback,
                "frames_root": args.frames_root,
            },
            "per_dimension": {
                dim: {
                    "total": s["total"],
                    "correct": s["correct"],
                    "accuracy": s["correct"] / s["total"] if s["total"] > 0 else None,
                }
                for dim, s in dim_stats.items()
            },
            "predictions": results,
        }
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nResults saved to {out}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UCS-Bench evaluation with DirectMe + VLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--questions", required=True, help="UCS-Bench JSONL path")
    p.add_argument("--graphs-dir", required=True,
                   help="directory containing per-video scene_graph.json files")
    p.add_argument("--frames-root", default="",
                   help="root of extracted frames (<root>/<uid>/frame_XXXXXX.png); "
                        "used to re-locate keyframe paths and for the frame fallback")
    p.add_argument("--out", default="results/ucsbench_eval.json",
                   help="output JSON report path")
    p.add_argument("--language", default="en", choices=["zh", "en"])
    p.add_argument("--reachable-radius-m", type=float, default=10.0,
                   help="reachability radius in metres (aligned with the internal eval)")
    p.add_argument("--top-k", type=int, default=32,
                   help="number of retrieved nodes / CLIP-selected keyframes")
    p.add_argument("--max-keyframes", type=int, default=32,
                   help="keyframes sent to the VLM when CLIP rerank is disabled")
    p.add_argument("--min-keyframes", type=int, default=8,
                   help="fill up with original frames when fewer graph keyframes survive")
    p.add_argument("--input-fps", type=float, default=1.0,
                   help="fps of the extracted frames (for causal frame selection)")

    # Retrieval behaviour switches (defaults match the internal evaluator).
    p.add_argument("--no-strict-causal", action="store_true",
                   help="disable question-time causal filtering")
    p.add_argument("--no-soft-scoring", action="store_true",
                   help="legacy hard label/color gates (can yield zero retrieval)")
    p.add_argument("--no-frame-fallback", action="store_true",
                   help="disable the original-frame fallback")

    # CLIP rerank (optional; aligned with the internal 32-frame setup).
    p.add_argument("--clip-model", default="ckpts/clip-vit-base-patch32",
                   help="local CLIP checkpoint for keyframe reranking")
    p.add_argument("--no-clip-rerank", action="store_true",
                   help="disable CLIP reranking (use first-N keyframes instead)")

    # VLM.
    p.add_argument("--backend", required=True, choices=["openai", "transformers"],
                   help="VLM backend")
    p.add_argument("--model", default="qwen3-vl-8b-instruct")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--max-new-tokens", type=int, default=32)

    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
