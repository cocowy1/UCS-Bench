#!/usr/bin/env python3
"""End-to-end DirectMe + VLM inference pipeline.

This script demonstrates the core paper contribution: using DirectMe's
pose-anchored scene graph memory to augment a multimodal LLM (Qwen3-VL,
InternVL, etc.) for egocentric spatial QA.

Pipeline overview
-----------------

1. **Offline**: Build the scene graph from video frames (or load a
   pre-built graph).
2. **Online**: For each question, retrieve the relevant subgraph +
   keyframes, assemble a structured prompt, and send it to the VLM.

Three VLM backends are supported:

* **OpenAI-compatible API** (vLLM, Together, DeepInfra, Ollama, etc.)
* **HuggingFace Transformers** (local GPU inference with Qwen3-VL)
* **Rule-based** (no VLM; uses the deterministic answer generator for
  debugging the graph quality in isolation)

Usage
-----

A) Using a pre-built scene graph + OpenAI-compatible API::

    python examples/run_vlm_pipeline.py \
        --graph runs/my_session/scene_graph.json \
        --question "Where is the sink relative to me?" \
        --backend openai \
        --model qwen3-vl-8b-instruct \
        --base-url http://localhost:8000/v1

B) Using a pre-built scene graph + local Transformers::

    python examples/run_vlm_pipeline.py \
        --graph runs/my_session/scene_graph.json \
        --question "Where is the sink relative to me?" \
        --backend transformers \
        --model Qwen/Qwen3-VL-8B-Instruct

C) Multiple-choice mode (UCS-Bench style)::

    python examples/run_vlm_pipeline.py \
        --graph runs/my_session/scene_graph.json \
        --question "Where is the sink relative to me?" \
        --options "On your left;Behind you;In front of you;To your right;Above you" \
        --backend openai \
        --model qwen3-vl-8b-instruct \
        --base-url http://localhost:8000/v1

D) Toy demo (no GPU, no VLM)::

    python examples/run_vlm_pipeline.py --toy --backend rule
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Lazy-import helpers (keep startup fast for --help)
# ---------------------------------------------------------------------------


def _load_graph(path: str):
    from directme.mapping.scene_graph import SceneGraph
    return SceneGraph.load_json(path)


def _build_toy_graph():
    """Build the deterministic living-room→kitchen demo graph."""
    from directme.config import DirectMeConfig
    from directme.mapping.offline_engine import OfflineMappingEngine
    from directme.perception.toy import build_living_room_kitchen_demo

    out = Path("runs/vlm_toy")
    out.mkdir(parents=True, exist_ok=True)
    cfg = DirectMeConfig()
    cfg.run_dir = str(out)

    frames, backend = build_living_room_kitchen_demo(out / "keyframes")
    engine = OfflineMappingEngine(backend=backend, config=cfg)
    engine.process_frames(frames, chunk_size=2)
    return engine.graph


def _get_current_pose(args, graph):
    from directme.geometry.poses import SE3
    if args.current_pose_json:
        mat = json.loads(args.current_pose_json)
        return SE3.from_list(mat)
    # Use the latest recorded ego pose if available.
    timeline = graph.metadata.get("ego_pose_timeline", [])
    if timeline:
        from directme.retrieval.pose_lookup import pose_from_graph_timeline
        return pose_from_graph_timeline(graph, timestamp=float("inf"))
    return SE3.identity()


# ---------------------------------------------------------------------------
# VLM backends
# ---------------------------------------------------------------------------


def _answer_rule_based(context):
    from directme.qa.generator import RuleBasedAnswerGenerator
    return RuleBasedAnswerGenerator().answer(context)


def _answer_openai(context, args, system: str, parts: list[dict]):
    """Send the prompt to an OpenAI-compatible multimodal endpoint."""
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: Install with `pip install directme[vlm]`")
        sys.exit(1)

    client = OpenAI(api_key=args.api_key or "EMPTY", base_url=args.base_url)

    content: list[dict] = []
    for part in parts:
        if part["type"] == "text":
            content.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image":
            path = Path(part["path"])
            if path.exists():
                mime, _ = mimetypes.guess_type(str(path))
                if mime and mime.startswith("image/"):
                    data = base64.b64encode(path.read_bytes()).decode()
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{data}"},
                    })

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": content},
    ]
    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        temperature=0.0,
        max_tokens=64,
    )
    return response.choices[0].message.content or ""


def _answer_transformers(context, args, system: str, parts: list[dict]):
    """Local inference with HuggingFace Transformers (Qwen3-VL / Qwen2.5-VL)."""
    try:
        # Qwen VL models: try Qwen3 class first, fall back to Qwen2.5.
        try:
            from transformers import Qwen3VLForConditionalGeneration as _QwenVL
        except ImportError:
            from transformers import Qwen2_5_VLForConditionalGeneration as _QwenVL
        from transformers import AutoProcessor
        from qwen_vl_utils import process_vision_info
        import torch
    except ImportError:
        print(
            "ERROR: Install transformers + qwen-vl-utils:\n"
            "  pip install transformers qwen-vl-utils torch"
        )
        sys.exit(1)

    print(f"[vlm] Loading {args.model} ...")
    model = _QwenVL.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(args.model)

    # Build the chat messages in Qwen-VL format.
    user_content = []
    for part in parts:
        if part["type"] == "text":
            user_content.append({"type": "text", "text": part["text"]})
        elif part["type"] == "image":
            path = Path(part["path"])
            if path.exists():
                user_content.append({"type": "image", "image": str(path.resolve())})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user", "content": user_content},
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    generated = model.generate(**inputs, max_new_tokens=64, temperature=0.0, do_sample=False)
    trimmed = generated[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    from directme.geometry.poses import SE3
    from directme.retrieval.retriever import GraphRetriever

    # 1. Load or build graph.
    if args.toy:
        print("[graph] Building toy demo graph ...")
        graph = _build_toy_graph()
    elif args.graph:
        print(f"[graph] Loading {args.graph} ...")
        graph = _load_graph(args.graph)
    else:
        print("ERROR: Provide --graph <path> or --toy")
        sys.exit(1)

    assert graph is not None
    print(f"[graph] {len(graph.nodes)} nodes, {len(graph.edges)} edges, "
          f"{len(graph.place_nodes)} places")

    # 2. Resolve the user's current pose.
    current_pose = _get_current_pose(args, graph)
    print(f"[pose]  T_world_from_camera = {current_pose.matrix.tolist()}")

    # 3. Retrieve the relevant subgraph.
    retriever = GraphRetriever(graph, reachable_radius_m=args.reachable_radius_m)
    context = retriever.retrieve(args.question, current_pose, language=args.language)

    print(f"\n[retrieval] {len(context.items)} items, "
          f"{len(context.keyframes)} keyframes")
    print(GraphRetriever.render_summary(context))

    # 4. Build the prompt.
    if args.options:
        from directme.qa.prompts import MultipleChoicePromptBuilder
        options = [o.strip() for o in args.options.split(";")]
        builder = MultipleChoicePromptBuilder(max_keyframes=args.max_keyframes)
        system, parts = builder.build(context, options=options)
    else:
        from directme.qa.prompts import DirectMePromptBuilder
        builder = DirectMePromptBuilder(max_keyframes=args.max_keyframes)
        system, parts = builder.build(context)

    # 5. Send to VLM.
    print(f"\n[vlm] backend={args.backend}, model={args.model}")
    if args.backend == "rule":
        answer = _answer_rule_based(context)
    elif args.backend == "openai":
        answer = _answer_openai(context, args, system, parts)
    elif args.backend == "transformers":
        answer = _answer_transformers(context, args, system, parts)
    else:
        print(f"ERROR: Unknown backend {args.backend}")
        sys.exit(1)

    # 6. Output.
    print(f"\n{'='*60}")
    print(f"Question: {args.question}")
    if args.options:
        for i, opt in enumerate(args.options.split(";")):
            print(f"  {chr(65+i)}. {opt.strip()}")
    print(f"Answer:   {answer}")
    print(f"{'='*60}")

    # Parse MC answer if applicable.
    if args.options:
        from directme.qa.prompts import MultipleChoicePromptBuilder
        parsed = MultipleChoicePromptBuilder.parse_answer(answer)
        print(f"Parsed option: {parsed}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="DirectMe + VLM inference pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input.
    p.add_argument("--graph", help="path to a pre-built scene_graph.json")
    p.add_argument("--toy", action="store_true", help="use the built-in toy demo graph")

    # Question.
    p.add_argument("--question", default="Where is the sink relative to me?",
                   help="free-form question")
    p.add_argument("--options", default=None,
                   help="semicolon-separated MC options for UCS-Bench style evaluation")
    p.add_argument("--language", default="en", choices=["zh", "en"])

    # Pose.
    p.add_argument("--current-pose-json", default=None,
                   help="4x4 row-major SE(3) matrix as JSON (default: latest from graph)")

    # Retrieval.
    p.add_argument("--reachable-radius-m", type=float, default=5.0)
    p.add_argument("--max-keyframes", type=int, default=4)

    # VLM backend.
    p.add_argument("--backend", default="rule", choices=["rule", "openai", "transformers"],
                   help="VLM backend: rule (no VLM), openai (API), transformers (local)")
    p.add_argument("--model", default="qwen3-vl-8b-instruct",
                   help="model name for openai/transformers backends")
    p.add_argument("--base-url", default=None,
                   help="OpenAI-compatible base URL (e.g. http://localhost:8000/v1)")
    p.add_argument("--api-key", default=None, help="API key (default: OPENAI_API_KEY env)")

    return p


def main() -> None:
    args = build_parser().parse_args()
    run(args)


if __name__ == "__main__":
    main()
