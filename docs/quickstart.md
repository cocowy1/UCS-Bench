# Quickstart

## Toy demo (~1 second, no GPU)

```bash
make install
make demo
```

This runs a deterministic 4-frame "living room → kitchen, two red cups"
scenario. The scene graph is persisted to `runs/toy/scene_graph.json`,
and a sample question is answered:

```
Question: 我身边有几个红杯子？在哪？
Answer:   找到 2 个匹配目标：
          entity_001 在您的左前方约 5.8 米处（不可及）；
          entity_002 在您的右前方约 0.5 米处（伸手可及）。
```

## Top-down visualization

```bash
make viz   # writes runs/toy/topdown.png
```

The PNG shows every object node coloured by its dominant colour, the
user's current pose as a black arrow, the reachability disk, and the
ego-relative edges (green = reachable, red = out of reach).

## Querying a saved graph

```bash
directme query \
  --graph runs/toy/scene_graph.json \
  --question "杯子在哪？我够得着吗？" \
  --current-pose-json "[[1,0,0,7],[0,1,0,0],[0,0,1,0],[0,0,0,1]]" \
  --language zh \
  --reachable-radius-m 5.0 \
  --show-summary
```

The `--show-summary` flag prints the matched subgraph including the
`(ego) --[relation, distance]--> (target)` edges.

## Programmatic API

```python
from directme.config import DirectMeConfig
from directme.geometry.poses import SE3
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.perception.toy import build_living_room_kitchen_demo
from directme.qa.generator import RuleBasedAnswerGenerator
from directme.retrieval.retriever import GraphRetriever

frames, backend = build_living_room_kitchen_demo("runs/toy/keyframes")
cfg = DirectMeConfig()
cfg.run_dir = "runs/toy"

engine = OfflineMappingEngine(backend=backend, config=cfg)
engine.process_frames(frames, chunk_size=2)

retriever = GraphRetriever(
    engine.graph,
    reachable_radius_m=cfg.retrieval.reachable_radius_m,
)
ctx = retriever.retrieve(
    "我身边有几个红杯子？我够得着吗？",
    SE3.from_translation([7, 0, 0]),
    language="zh",
)
print(RuleBasedAnswerGenerator().answer(ctx))
print(ctx.ego_edges)
```

## Ingesting a video or a folder of frames *(v0.4)*

DirectMe's normal entry point for real data is `directme ingest`, which
runs the asynchronous incremental pipeline end-to-end:

```bash
# Decode a video at 1 FPS and build the graph
directme ingest --video walk.mp4 --target-fps 1.0 \
                --chunk-size 10 --storage-backend sqlite \
                --out runs/walk

# Or consume frames already extracted by an upstream tool
directme ingest --frames-dir frames/ --chunk-size 10 \
                --storage-backend sqlite --out runs/walk
```

The default `--backend toy` produces a deterministic synthetic graph
regardless of the input — useful for verifying plumbing. For real
backbones (DA3 / SCAL3R / YOLO-World), wire `ComposedPerceptionBackend`
through the Python API; see [`docs/ingest.md`](ingest.md) and
[`docs/adapter_guide.md`](adapter_guide.md).

Three properties to know about:

- **Backpressure.** Frame queue is bounded; producer waits when full.
  Memory cannot blow up on a 4-hour video.
- **Chunk fault isolation.** A chunk that crashes inside the backend is
  logged and skipped; subsequent chunks keep running. Pass
  `--fail-fast` to disable for debugging.
- **Resumability.** With `--storage-backend sqlite`, every committed
  chunk's id is persisted; a Ctrl-C'd run resumes from the next chunk
  on restart.

After ingest, query the resulting graph the same way as the toy demo:

```bash
directme query --graph runs/walk/scene_graph.json \
               --question "我刚才看到的红杯子在哪？" \
               --current-pose-json '[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]'
```

## Real perception backend

See [`docs/adapter_guide.md`](adapter_guide.md). The end-to-end runner is
`examples/run_real_pipeline.py`.
