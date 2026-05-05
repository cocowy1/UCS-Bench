# DirectMe

> User-centric continual spatial intelligence for egocentric video streams.

[![CI](https://github.com/directme-ai/directme/actions/workflows/ci.yml/badge.svg)](https://github.com/directme-ai/directme/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

DirectMe builds a **pose-anchored 3D scene graph memory** from a streaming first-person camera and answers questions about the spatial world *as it relates to the wearer right now* — "where is X relative to me?", "can I reach X?", "how many of X have I seen?".

The implementation follows a strict **offline / online split**:

```
                       (1) offline (always running)                          (2) online (per query)

frames ─► perception ─► pose propagation ─► world-coord projection ─► graph fusion ─► JSON / SQLite
                                                                                          │
                                              ┌───────────────────────────────────────────┘
                                              ▼
                            question  ─► intent parser ─► subgraph retrieval
                                            + world→current-camera projection
                                            + reachability + ego relations  ─► answer
```

The heavy visual stream processing happens **before** any question arrives. Online QA only retrieves a compact subgraph and a few keyframes — never reprocesses the full video.

---

## ✨ What's in this version (v0.7.0)

**v0.7 — VLM integration release (May 2026):**

- **Multiple-choice prompt builder** — `MultipleChoicePromptBuilder` generates UCS-Bench-style 5-way MC prompts with graph summary + keyframes, bilingual system prompts, and robust answer parsing (`parse_answer()` handles "A", "A.", "The answer is B", "答案是 C", etc.).
- **End-to-end VLM pipeline example** — `examples/run_vlm_pipeline.py` demonstrates the full graph → retrieval → prompt → MLLM → answer flow with three backends: `rule` (no VLM), `openai` (vLLM / Together), `transformers` (local HuggingFace Qwen3-VL).
- **UCS-Bench VLM evaluation script** — `examples/evaluate_ucsbench_vlm.py` reproduces the "DirectMe (w/ Qwen3-VL)" results in Table 3 of the paper.
- **Comprehensive paper→code alignment** — `docs/paper_alignment.md` maps every paper section to the implementing code.

**Inherited from v0.6.0:**

- **Room-aware retrieval** — `GraphRetriever._score_node` now boosts nodes whose `scene_tag` / `place_id` matches the room token parsed from the question (e.g. *"客厅那个红杯子"*). Soft additive scoring, never a hard filter, so recall is preserved when the scene classifier is noisy.
- **Disambiguating P&O scoring** — the UCS-Bench evaluator now resolves the queried target via `expected_target_node_id` → `expected_target_place` → `expected_target_label`, so dataset authors can write unambiguous gold even when two same-label objects exist in the scene.
- **Reproducible query-time pose** — datasets can pin the user's pose at query time via `expected_query_pose` (a 4×4 row-major SE(3) matrix). Removes the previous implicit dependency on `query_timestamp` mapping cleanly to `ego_pose_timeline`.
- **Singleton-place induction** — `induce_places(min_members=1, ...)` is the new default and partitions by `scene_tag` before geometric clustering, so a single object in a small room still surfaces as a queryable place.
- **Toy eval is now genuinely 6/6 = 100 %** across all four UCS-Bench dimensions, not just three.
- **Video & frame-stream input** — `directme ingest --video my.mp4 --target-fps 1.0` decodes & samples a video at the egocentric 1-FPS sweet spot, or `--frames-dir` consumes pre-extracted frames. See [`docs/ingest.md`](docs/ingest.md).
- **Async incremental ingest, hardened** — bounded queue (backpressure), chunk-level fault isolation (one bad chunk no longer kills the run), and SQLite-backed resumability (interrupt → restart picks up where it left off).
- **Diversity-aware keyframes** — when a CLIP / DINO embedding is supplied with each observation, the per-node K-frame budget is filled by greedy farthest-point sampling on cosine distance instead of v0.3's bbox-area greedy. Falls back to v0.3 behaviour when no embedding is available.
- **Pose-drift telemetry** — `graph.metadata["drift_telemetry"]` records cumulative world-frame translation, rejected chunks, and threshold-driven warnings, so a long ingest tells you *which segment* of its trajectory may be unreliable. **No automatic correction** — that would be loop closure (out of scope).
- **Pose-anchored scene graph** with HSV histogram + optional CLIP-style semantic embedding fusion (multi-view identity, ConceptGraphs-style).
- **Drift-resistant pose propagation** with NaN / non-orthogonal-rotation / translation-jump rejection — corrupt chunks don't poison the world frame.
- **8-class egocentric relation labels** (`front`, `behind`, `left`, `right`, `front_left`, `front_right`, `behind_left`, `behind_right`) emitted as graph edges at query time.
- **Depth-based reachability** — objects within `reachable_radius_m` (default 5.0 m) are flagged `reachable: true` in every retrieval result.
- **Pluggable storage** — JSON for debugging, SQLite (WAL mode) for long sessions and concurrent online reads.
- **Real-model adapters**: Depth Anything 3 (default `DA3NESTED-GIANT-LARGE-1.1`, CC BY-NC weights), YOLO-World, optional SAM 2 (CPU/GPU autocast), simple IoU+appearance tracker — all lazy-imported, all swappable. When SAM 2 is unavailable, DirectMe falls back to bbox-center depth unprojection so the community can still run the full pipeline.
- **Multimodal QA generator** — `OpenAICompatibleMultimodalGenerator` sends keyframes as `image_url` parts to vision endpoints (Qwen-VL, GPT-4o, InternVL via vLLM).
- **UCS-Bench 4-dimension scoring** — Position & Orientation, Trajectory & Movement, Proximity & Reachability, Category & Quantity.

---

## 🚀 Quickstart

### One-command demo (no GPU, < 1 s)

```bash
git clone https://github.com/directme-ai/directme.git
cd directme
make install
make demo
```

Expected output:

```
Graph saved to runs/toy/scene_graph.json
Question: 我身边有几个红杯子？在哪？
Answer:   找到 2 个匹配目标：entity_001 在您的左前方约 5.8 米处（不可及）；
          entity_002 在您的右前方约 0.5 米处（伸手可及）。
[Ego edges]
  (ego) --[front_left, 5.83m, reachable=False]--> (entity_001)
  (ego) --[front_right, 0.50m, reachable=True]--> (entity_002)
```

### Render a top-down map (matplotlib)

```bash
make viz
# wrote runs/toy/topdown.png
```

### Run UCS-Bench evaluation

```bash
make eval
```

Expected output:

```
=== DirectMe UCS-Bench Evaluation ===
Total questions:    6
Scored questions:   6
Correct answers:    6
Overall accuracy:   100.0%

Per-dimension breakdown:
  position_orientation         1 q | 1/1 correct | 100.0%
  trajectory_movement          1 q | 1/1 correct | 100.0%
  proximity_reachability       2 q | 2/2 correct | 100.0%
  category_quantity            2 q | 2/2 correct | 100.0%
```

The eval harness scores against a structured `RetrievedContext`, not free-form
generator text — so it isolates scene-graph quality from QA-generator quality.
Trajectory & Movement is scored against the offline engine's
`place_visit_timeline`; questions whose gold field is not
`expected_path_labels` (e.g. raw turn-by-turn route reconstruction) are
reported as `partial: trajectory_movement_partial_only` and excluded from the
headline accuracy. See [`docs/evaluation.md`](docs/evaluation.md) for the
dataset schema and per-dimension scoring rules.

### Run on real frames (GPU)

1. Extract a video to 1 FPS images:

   ```bash
   ffmpeg -i your_video.mp4 -vf fps=1 ./frames/frame_%06d.jpg
   ```
2. Install perception extras:

   ```bash
   pip install -e ".[perception,video,viz]"
   # SAM 2, Depth Anything 3 install from source — see docs/adapter_guide.md
   ```
3. Run the end-to-end pipeline:

   ```bash
   python examples/run_real_pipeline.py \
       --frames ./frames \
       --classes "cup,phone,bottle,laptop,chair,table,sink,fridge,door,bag" \
       --question "我身边有几个杯子？我够得着吗？" \
       --reachable-radius-m 5.0 \
       --device auto \
       --out runs/my_session
   ```

   Or use the built-in async CLI directly:

   ```bash
   directme ingest \
       --frames-dir ./frames \
       --backend composed \
       --classes "cup,phone,bottle,laptop,chair,table,sink,fridge,door,bag" \
       --device auto \
       --storage-backend sqlite \
       --out runs/my_session
   ```

   Add `--sam2-checkpoint ... --sam2-config ...` when you have SAM 2 installed;
   otherwise the pipeline uses bbox-center depth fallback.

---

## 🧱 Repository layout

```text
directme/
├── geometry/          # SE(3), pose propagation, mask→world unprojection
├── perception/
│   ├── base.py        # PerceptionBackend interface
│   ├── color.py       # HSV histogram extraction + similarity
│   ├── toy.py         # deterministic backend used by demos & tests
│   └── adapters/
│       ├── depth_anything3.py   # ByteDance-Seed/Depth-Anything-3
│       ├── open_vocab_tracking.py  # YOLO-World + SAM 2 + tracker
│       ├── scal3r.py            # zju3dv/Scal3R output reader
│       └── composed.py          # end-to-end real backend
├── mapping/
│   ├── pose_propagation.py      # chunk anchoring + failure rejection
│   ├── scene_graph.py           # nodes, edges, HSV + CLIP fusion
│   ├── place_induction.py       # cluster nodes into place regions
│   └── offline_engine.py        # the asynchronous incremental engine
├── retrieval/
│   ├── query_parser.py          # zh / en intent parsing
│   ├── egocentric.py            # 8-class relation classifier + reachability
│   └── retriever.py             # subgraph retrieval + ego_edges
├── qa/
│   ├── prompts.py     # free-form + multiple-choice prompt builders
│   ├── generator.py   # rule-based + OpenAI-compatible generators
│   └── openai_compatible.py  # multimodal VLM generator
├── storage/           # JSON + SQLite scene-graph stores
├── viz/               # top-down map renderer (matplotlib)
└── cli.py             # `directme demo` and `directme query`

examples/
├── run_vlm_pipeline.py        # ★ DirectMe + VLM end-to-end inference
├── evaluate_ucsbench_vlm.py   # ★ Full UCS-Bench evaluation with VLM
├── run_real_pipeline.py       # end-to-end with real perception models
├── run_toy_demo.py            # deterministic toy demo (no GPU)
└── visualize_graph.py         # matplotlib top-down visualization
```

---

## 🧠 Scene graph node schema

```json
{
  "node_id": "entity_002",
  "semantic_label": "cup",
  "place_id": "place_001",
  "attributes": {
    "color": "red",
    "color_hsv_histogram": [0.85, 0.10, 0.05, ...],
    "semantic_embedding": [0.12, -0.04, ...],
    "is_movable": true,
    "count_contribution": 1
  },
  "spatial_absolute": {
    "reference_frame": "Frame_0_World_Origin",
    "p_world": [6.02, 1.10, 5.05],
    "observation_count": 7,
    "last_seen_timestamp": 1245.6
  },
  "spatial_egocentric_dynamic": {
    "reference_frame": "Current_Camera",
    "p_cam": [0.30, -0.20, 0.40],
    "relation": "front_right",
    "distance_m": 0.50,
    "reachable": true,
    "natural_language": "在您的右前方约 0.5 米处（伸手可及）"
  },
  "observations": [...],
  "keyframes": ["frames/00:36:00.jpg", ...],
  "track_ids": ["track_kitchen_cup"]
}
```

`spatial_absolute` is stable across queries; `spatial_egocentric_dynamic` is recomputed per query from the user's current camera pose.

---

## 🛡 Failure handling

DirectMe is conservative about pose: a single bad chunk should never corrupt the world frame.

| Failure mode                                      | Mechanism                                                                  |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| NaN / Inf in `T_local`                          | `is_valid_se3()` rejects → chunk skipped                                |
| Non-orthogonal rotation                           | `is_valid_se3()` rejects → chunk skipped                                |
| Per-frame translation >`max_per_frame_jump_m`   | `max_translation_jump()` rejects → chunk skipped                        |
| Low DA3 depth confidence                          | Per-observation `pose_confidence` scales the EMA update                  |
| Movable object teleported                         | Motion-aware overwrite when displacement >`motion_overwrite_threshold_m` |
| Visually different objects same label, same color | Optional CLIP embedding gate prevents wrong merges                         |

Every rejection is logged in `engine.chunk_reports[]` for offline auditing.

---

## 🤖 DirectMe + VLM Integration

The core contribution of DirectMe is enhancing multimodal LLMs (Qwen3-VL, InternVL, etc.) with structured spatial memory. The pipeline works in two phases:

```
┌──────────────────────────────────────────────────────────────────────┐
│  OFFLINE: Build scene graph memory from egocentric video stream     │
│                                                                      │
│  Video ─► DA3 (depth+pose) ─► YOLO-World+SAM2 (detect+segment)     │
│       ─► Tracker (associate) ─► Scene Graph (upsert+fuse)           │
└────────────────────────────────────┬─────────────────────────────────┘
                                     │  scene_graph.json
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  ONLINE: Answer spatial questions with VLM                           │
│                                                                      │
│  Question ─► QueryParser ─► GraphRetriever                           │
│    (subgraph + ego-relations + keyframes)                            │
│       ─► PromptBuilder ─► Qwen3-VL / InternVL ─► Answer             │
└──────────────────────────────────────────────────────────────────────┘
```

### Quick example: DirectMe → Qwen3-VL

```python
from directme.mapping.scene_graph import SceneGraph
from directme.retrieval.retriever import GraphRetriever
from directme.qa.prompts import MultipleChoicePromptBuilder

# 1. Load pre-built graph
graph = SceneGraph.load_json("runs/my_session/scene_graph.json")

# 2. Set the user's current pose (from DA3 or live tracker)
from directme.retrieval.pose_lookup import pose_from_graph_timeline
pose = pose_from_graph_timeline(graph, timestamp=81.0)

# 3. Retrieve relevant subgraph + keyframes
retriever = GraphRetriever(graph, reachable_radius_m=5.0)
context = retriever.retrieve("Where is the sink?", pose, language="en")

# 4. Build prompt (with multiple-choice options for UCS-Bench)
builder = MultipleChoicePromptBuilder(max_keyframes=4)
system, parts = builder.build(
    context,
    options=["On your left", "Behind you", "In front of you", ...]
)

# 5. Send to VLM (OpenAI-compatible API, e.g. vLLM serving Qwen3-VL)
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
# ... (see examples/run_vlm_pipeline.py for full code)
```

### Run the VLM pipeline

```bash
# Free-form QA with OpenAI-compatible API
python examples/run_vlm_pipeline.py \
    --graph runs/my_session/scene_graph.json \
    --question "Where is the sink relative to me?" \
    --backend openai --model qwen3-vl-8b-instruct \
    --base-url http://localhost:8000/v1

# Local inference with HuggingFace Transformers
python examples/run_vlm_pipeline.py \
    --graph runs/my_session/scene_graph.json \
    --question "Where is the sink relative to me?" \
    --backend transformers --model Qwen/Qwen3-VL-8B-Instruct
```

### Reproduce UCS-Bench results (Table 3)

```bash
# Step 1: Build scene graphs for all 540 videos
for vid in /data/ucsbench/videos/*/; do
    python examples/run_real_pipeline.py \
        --frames "$vid/frames" --out "/data/ucsbench/graphs/$(basename $vid)" \
        --classes "cup,phone,bottle,chair,table,sink,fridge,oven,..." \
        --device auto
done

# Step 2: Evaluate with Qwen3-VL via vLLM
python examples/evaluate_ucsbench_vlm.py \
    --questions /data/ucsbench/questions.jsonl \
    --graphs-dir /data/ucsbench/graphs \
    --backend openai --model qwen3-vl-8b-instruct \
    --base-url http://localhost:8000/v1 \
    --out results/directme_qwen3vl.json
```

---

## 🔬 Comparison with ConceptGraphs

| Feature                  | ConceptGraphs       | DirectMe                                           |
| ------------------------ | ------------------- | -------------------------------------------------- |
| Scene graph construction | ✅ Static, one-pass | ✅**Incremental, streaming**                 |
| Object identity fusion   | ✅ CLIP multi-view  | ✅ CLIP + HSV histogram + tracker                  |
| Coordinate frame         | World-only          | World +**egocentric per-query**              |
| Ego-motion tracking      | ❌                  | ✅ Pose propagation + drift rejection              |
| Egocentric relations     | ❌                  | ✅ 8-class (front/behind/left/right/...)           |
| Reachability reasoning   | ❌                  | ✅ Depth-based, configurable radius                |
| Place/room induction     | ❌                  | ✅ Scene-tag + geometric clustering                |
| Trajectory memory        | ❌                  | ✅`ego_pose_timeline` + `place_visit_timeline` |
| Online QA pipeline       | ❌                  | ✅ Retrieve → prompt → MLLM → answer            |
| Evaluation benchmark     | ❌                  | ✅ UCS-Bench, 4 dimensions                         |
| Storage backends         | Memory only         | JSON + SQLite (WAL, resumable)                     |
| Async streaming ingest   | ❌                  | ✅ Bounded queue, fault isolation                  |

---

## 📚 Documentation

- [`docs/architecture.md`](docs/architecture.md) — the offline / online split in detail
- [`docs/algorithm_notes.md`](docs/algorithm_notes.md) — design choices and gotchas
- [`docs/adapter_guide.md`](docs/adapter_guide.md) — wiring DA3 / YOLO-World / SAM 2
- [`docs/schema.md`](docs/schema.md) — full scene graph JSON schema
- [`docs/paper_alignment.md`](docs/paper_alignment.md) — mapping from paper to code

---

## 🤝 Contributing

```bash
make dev          # install runtime + dev + viz extras
pre-commit install
make check        # lint + test
```

PRs welcome. Please:

1. Add tests for new functionality.
2. Run `make format` before committing.
3. Update `docs/` if the public API changes.

---

## 📝 License

Apache-2.0. See [`LICENSE`](LICENSE).

## 🙏 Acknowledgements

DirectMe's identity-fusion design draws on **ConceptGraphs** (Gu et al., ICRA 2024). The real-model adapters ship against **Depth Anything 3**, **YOLO-World**, **SAM 2**, and (optionally) **Scal3R**.
