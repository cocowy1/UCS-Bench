#!/usr/bin/env python3
"""UCS-Bench evaluation with DirectMe + VLM.

This script runs the full DirectMe evaluation pipeline on UCS-Bench:

1. Load pre-built scene graphs (one per video).
2. For each question, retrieve the relevant subgraph + keyframes.
3. Assemble a multiple-choice prompt and send it to the VLM.
4. Parse the VLM's answer and compare against the ground truth.
5. Report per-dimension and overall accuracy.

This reproduces the "DirectMe (w/ Qwen3-VL)" row in Table 3 of the paper.

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
      "dimension":       "position_orientation"  # one of: position_orientation,
                                                 #   trajectory_movement,
                                                 #   proximity_reachability,
                                                 #   category_quantity
    }

Usage
-----

Via OpenAI-compatible API (vLLM serving Qwen3-VL)::

    python examples/evaluate_ucsbench_vlm.py \
        --questions /data/ucsbench/questions.jsonl \
        --graphs-dir /data/ucsbench/graphs \
        --backend openai \
        --model qwen3-vl-8b-instruct \
        --base-url http://localhost:8000/v1 \
        --out results/directme_qwen3vl.json

Via local Transformers::

    python examples/evaluate_ucsbench_vlm.py \
        --questions /data/ucsbench/questions.jsonl \
        --graphs-dir /data/ucsbench/graphs \
        --backend transformers \
        --model Qwen/Qwen3-VL-8B-Instruct \
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
            max_tokens=16,
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
            gen = m.generate(**inputs, max_new_tokens=16, temperature=0.0, do_sample=False)
        trimmed = gen[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(trimmed, skip_special_tokens=True)[0]

    else:
        raise ValueError(f"Unknown backend: {backend}")


def run(args: argparse.Namespace) -> None:
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

    builder = MultipleChoicePromptBuilder(max_keyframes=args.max_keyframes)

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

        # Resolve pose.
        try:
            pose = pose_from_graph_timeline(graph, timestamp=timestamp)
        except Exception:
            pose = SE3.identity()

        # Retrieve subgraph.
        retriever = GraphRetriever(graph, reachable_radius_m=args.reachable_radius_m)
        context = retriever.retrieve(
            q["question"], pose, language=args.language,
        )

        # Build MC prompt.
        options_raw = q.get("options", [])
        if not options_raw:
            skipped += 1
            continue

        # Strip option letter prefixes if present (e.g. "A. On your left" → "On your left").
        options = []
        for opt in options_raw:
            cleaned = re.sub(r"^[A-E]\.\s*", "", str(opt))
            options.append(cleaned)

        system, parts = builder.build(context, options=options)

        # Call VLM.
        try:
            raw_answer = _get_vlm_answer(
                system, parts, args.backend, args.model,
                args.base_url, args.api_key,
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

    print(f"\n{'='*60}")
    print(f"DirectMe + {args.model} on UCS-Bench")
    print(f"{'='*60}")
    print(f"Total questions:  {len(questions)}")
    print(f"Evaluated:        {len(scored)}")
    print(f"Skipped:          {skipped}")
    print(f"Correct:          {n_correct}")
    print(f"Overall accuracy: {accuracy * 100:.1f}%")
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
            "elapsed_seconds": elapsed,
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
    p.add_argument("--out", default="results/ucsbench_eval.json",
                   help="output JSON report path")
    p.add_argument("--language", default="en", choices=["zh", "en"])
    p.add_argument("--reachable-radius-m", type=float, default=5.0)
    p.add_argument("--max-keyframes", type=int, default=4)

    # VLM.
    p.add_argument("--backend", required=True, choices=["openai", "transformers"],
                   help="VLM backend")
    p.add_argument("--model", default="qwen3-vl-8b-instruct")
    p.add_argument("--base-url", default=None)
    p.add_argument("--api-key", default=None)

    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
