# DirectMe

> 面向第一视角视频流的、以用户为中心的持续空间智能。

[![CI](https://github.com/directme-ai/directme/actions/workflows/ci.yml/badge.svg)](https://github.com/directme-ai/directme/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

DirectMe 会从流式第一视角相机中构建一个**以姿态锚定的 3D 场景图记忆**，并回答关于空间世界的问题，而且这些问题都与“佩戴者此时此刻所处的位置和朝向”相关——例如：“X 相对于我在哪里？”、“我够得到 X 吗？”、“我见过多少个 X？”。

该实现严格遵循**离线 / 在线分离**的设计：

```text
                       (1) 离线阶段（持续运行）                          (2) 在线阶段（每次查询）

帧 ─► 感知 ─► 姿态传播 ─► 世界坐标投影 ─► 图融合 ─► JSON / SQLite
                                                                                          │
                                              ┌───────────────────────────────────────────┘
                                              ▼
                            问题  ─► 意图解析器 ─► 子图检索
                                            + 世界坐标→当前相机坐标投影
                                            + 可达性 + 自我中心关系  ─► 答案
```

繁重的视频流视觉处理会在任何问题到来之前就完成。在线问答只会检索一个紧凑的子图和少量关键帧——绝不会重新处理整段视频。

---

## ✨ 本版本包含的内容（v0.7.0）

**v0.7 — VLM 集成版本（2026 年 5 月）：**

- **多项选择提示词构建器** — `MultipleChoicePromptBuilder` 会生成 UCS-Bench 风格的五选一选择题提示词，包含图摘要 + 关键帧、中英双语系统提示词，以及稳健的答案解析能力（`parse_answer()` 可处理 “A”、“A.”、“The answer is B”、“答案是 C”等格式）。
- **端到端 VLM 流水线示例** — `examples/run_vlm_pipeline.py` 展示完整的 图 → 检索 → 提示词 → MLLM → 答案 流程，支持三个后端：`rule`（不使用 VLM）、`openai`（vLLM / Together）、`transformers`（本地 HuggingFace Qwen3-VL）。
- **UCS-Bench VLM 评估脚本** — `examples/evaluate_ucsbench_vlm.py` 可复现论文表 3 中 “DirectMe (w/ Qwen3-VL)” 的结果。
- **完整的论文→代码对应说明** — `docs/paper_alignment.md` 将论文中的每个章节映射到对应的实现代码。

**继承自 v0.6.0 的内容：**

- **房间感知检索** — 当问题中解析出的房间词元与节点的 `scene_tag` / `place_id` 匹配时，`GraphRetriever._score_node` 现在会提升这些节点的分数（例如 *“客厅那个红杯子”*）。这是软加性评分，不是硬过滤，因此当场景分类器有噪声时仍能保持召回率。
- **带消歧的 P&O 评分** — UCS-Bench 评估器现在会按 `expected_target_node_id` → `expected_target_place` → `expected_target_label` 的顺序解析被查询的目标，因此即使场景中存在两个标签相同的物体，数据集作者也能编写无歧义的标准答案。
- **可复现的查询时刻姿态** — 数据集可以通过 `expected_query_pose`（一个 4×4 行主序 SE(3) 矩阵）固定用户在查询时的姿态。这样就移除了过去隐式依赖 `query_timestamp` 能够干净映射到 `ego_pose_timeline` 的问题。
- **单例地点归纳** — `induce_places(min_members=1, ...)` 成为新的默认设置，并且会先按 `scene_tag` 分区，再进行几何聚类。因此，即便一个小房间里只有一个物体，它仍然会作为可查询的地点浮现出来。
- **玩具评估现在真正达到 6/6 = 100%**，覆盖 UCS-Bench 的全部四个维度，而不只是三个维度。
- **视频与帧流输入** — `directme ingest --video my.mp4 --target-fps 1.0` 会以第一视角任务较合适的 1 FPS 对视频进行解码与采样；或者使用 `--frames-dir` 读取预先抽取好的帧。参见 [`docs/ingest.md`](docs/ingest.md)。
- **更稳健的异步增量导入** — 有界队列（背压）、chunk 级故障隔离（一个坏 chunk 不再导致整个运行失败），以及基于 SQLite 的可恢复能力（中断 → 重启后会从上次位置继续）。
- **多样性感知关键帧** — 当每个 observation 都提供 CLIP / DINO embedding 时，每个节点的 K 帧预算会通过基于余弦距离的贪心最远点采样填充，而不是 v0.3 中基于 bbox 面积的贪心策略。如果没有可用 embedding，则回退到 v0.3 的行为。
- **姿态漂移遥测** — `graph.metadata["drift_telemetry"]` 会记录累计的世界坐标系平移、被拒绝的 chunk，以及由阈值触发的警告。因此，长时间导入会告诉你轨迹中的**哪一段**可能不可靠。**不会自动校正**——那属于回环检测 / 闭环优化范畴，不在本项目范围内。
- **以姿态锚定的场景图**，支持 HSV 直方图 + 可选的 CLIP 风格语义 embedding 融合（多视角身份识别，类似 ConceptGraphs）。
- **抗漂移的姿态传播**，带有 NaN / 非正交旋转 / 平移跳变拒绝机制——损坏的 chunk 不会污染世界坐标系。
- **8 类第一视角关系标签**（`front`、`behind`、`left`、`right`、`front_left`、`front_right`、`behind_left`、`behind_right`），会在查询时作为图边输出。
- **基于深度的可达性判断** — 位于 `reachable_radius_m`（默认 5.0 m）范围内的物体，会在每个检索结果中被标记为 `reachable: true`。
- **可插拔存储** — JSON 用于调试，SQLite（WAL 模式）用于长时会话和并发在线读取。
- **真实模型适配器**：Depth Anything 3（默认 `DA3NESTED-GIANT-LARGE-1.1`，权重为 CC BY-NC 许可）、YOLO-World、可选 SAM 2（CPU/GPU autocast）、简单的 IoU+外观 tracker——全部延迟导入，且全部可替换。当 SAM 2 不可用时，DirectMe 会回退到 bbox 中心深度反投影，因此社区仍然可以运行完整流水线。
- **多模态问答生成器** — `OpenAICompatibleMultimodalGenerator` 会将关键帧作为 `image_url` 部分发送到视觉端点（例如通过 vLLM 提供的 Qwen-VL、GPT-4o、InternVL）。
- **UCS-Bench 四维评分** — 位置与朝向（Position & Orientation）、轨迹与运动（Trajectory & Movement）、邻近性与可达性（Proximity & Reachability）、类别与数量（Category & Quantity）。

---

## 🚀 快速开始

### 一条命令运行演示（无 GPU，< 1 秒）

```bash
git clone https://github.com/directme-ai/directme.git
cd directme
make install
make demo
```

期望输出：

```text
Graph saved to runs/toy/scene_graph.json
Question: 我身边有几个红杯子？在哪？
Answer:   找到 2 个匹配目标：entity_001 在您的左前方约 5.8 米处（不可及）；
          entity_002 在您的右前方约 0.5 米处（伸手可及）。
[Ego edges]
  (ego) --[front_left, 5.83m, reachable=False]--> (entity_001)
  (ego) --[front_right, 0.50m, reachable=True]--> (entity_002)
```

### 渲染俯视地图（matplotlib）

```bash
make viz
# wrote runs/toy/topdown.png
```

### 运行 UCS-Bench 评估

```bash
make eval
```

期望输出：

```text
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

评估框架会基于结构化的 `RetrievedContext` 进行评分，而不是基于自由文本形式的生成器输出——因此它能将场景图质量与问答生成器质量隔离开来。轨迹与运动维度会基于离线引擎的 `place_visit_timeline` 进行评分；如果某些问题的金标准字段不是 `expected_path_labels`（例如原始逐步路线重建），则会被报告为 `partial: trajectory_movement_partial_only`，并从主指标准确率中排除。有关数据集 schema 和各维度评分规则，请参见 [`docs/evaluation.md`](docs/evaluation.md)。

### 在真实帧上运行（GPU）

1. 将视频抽取为 1 FPS 图像：

   ```bash
   ffmpeg -i your_video.mp4 -vf fps=1 ./frames/frame_%06d.jpg
   ```

2. 安装感知相关额外依赖：

   ```bash
   pip install -e ".[perception,video,viz]"
   # SAM 2、Depth Anything 3 需要从源码安装——见 docs/adapter_guide.md
   ```

3. 运行端到端流水线：

   ```bash
   python examples/run_real_pipeline.py \
       --frames ./frames \
       --classes "cup,phone,bottle,laptop,chair,table,sink,fridge,door,bag" \
       --question "我身边有几个杯子？我够得着吗？" \
       --reachable-radius-m 5.0 \
       --device auto \
       --out runs/my_session
   ```

   或者直接使用内置的异步 CLI：

   ```bash
   directme ingest \
       --frames-dir ./frames \
       --backend composed \
       --classes "cup,phone,bottle,laptop,chair,table,sink,fridge,door,bag" \
       --device auto \
       --storage-backend sqlite \
       --out runs/my_session
   ```

   如果你已经安装了 SAM 2，请添加 `--sam2-checkpoint ... --sam2-config ...`；否则流水线会使用 bbox 中心深度回退方案。

---

## 🧱 仓库结构

```text
directme/
├── geometry/          # SE(3)、姿态传播、mask→world 反投影
├── perception/
│   ├── base.py        # PerceptionBackend 接口
│   ├── color.py       # HSV 直方图提取 + 相似度
│   ├── toy.py         # 演示和测试使用的确定性后端
│   └── adapters/
│       ├── depth_anything3.py   # ByteDance-Seed/Depth-Anything-3
│       ├── open_vocab_tracking.py  # YOLO-World + SAM 2 + tracker
│       ├── scal3r.py            # zju3dv/Scal3R 输出读取器
│       └── composed.py          # 端到端真实后端
├── mapping/
│   ├── pose_propagation.py      # chunk 锚定 + 失败拒绝
│   ├── scene_graph.py           # 节点、边、HSV + CLIP 融合
│   ├── place_induction.py       # 将节点聚类为地点区域
│   └── offline_engine.py        # 异步增量引擎
├── retrieval/
│   ├── query_parser.py          # 中文 / 英文意图解析
│   ├── egocentric.py            # 8 类关系分类器 + 可达性
│   └── retriever.py             # 子图检索 + ego_edges
├── qa/
│   ├── prompts.py     # 自由形式 + 多项选择提示词构建器
│   ├── generator.py   # 基于规则 + OpenAI 兼容生成器
│   └── openai_compatible.py  # 多模态 VLM 生成器
├── storage/           # JSON + SQLite 场景图存储
├── viz/               # 俯视图渲染器（matplotlib）
└── cli.py             # `directme demo` 和 `directme query`

examples/
├── run_vlm_pipeline.py        # ★ DirectMe + VLM 端到端推理
├── evaluate_ucsbench_vlm.py   # ★ 使用 VLM 的完整 UCS-Bench 评估
├── run_real_pipeline.py       # 使用真实感知模型的端到端流程
├── run_toy_demo.py            # 确定性玩具演示（无 GPU）
└── visualize_graph.py         # matplotlib 俯视可视化
```

---

## 🧠 场景图节点 schema

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

`spatial_absolute` 在不同查询之间保持稳定；`spatial_egocentric_dynamic` 会在每次查询时根据用户当前相机姿态重新计算。

---

## 🛡 故障处理

DirectMe 对姿态采取保守策略：单个坏 chunk 绝不应破坏整个世界坐标系。

| 失效模式 | 处理机制 |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| `T_local` 中存在 NaN / Inf | `is_valid_se3()` 拒绝 → 跳过该 chunk |
| 非正交旋转 | `is_valid_se3()` 拒绝 → 跳过该 chunk |
| 单帧平移 > `max_per_frame_jump_m` | `max_translation_jump()` 拒绝 → 跳过该 chunk |
| DA3 深度置信度低 | 每个 observation 的 `pose_confidence` 会缩放 EMA 更新 |
| 可移动物体发生“瞬移” | 当位移 > `motion_overwrite_threshold_m` 时，使用运动感知覆盖 |
| 视觉上不同但标签和颜色相同的物体 | 可选的 CLIP embedding gate 可防止错误合并 |

每一次拒绝都会记录在 `engine.chunk_reports[]` 中，方便离线审计。

---

## 🤖 DirectMe + VLM 集成

DirectMe 的核心贡献是用结构化空间记忆增强多模态 LLM（Qwen3-VL、InternVL 等）。该流水线分为两个阶段：

```text
┌──────────────────────────────────────────────────────────────────────┐
│  离线：从第一视角视频流构建场景图记忆                               │
│                                                                      │
│  视频 ─► DA3（深度+姿态） ─► YOLO-World+SAM2（检测+分割）             │
│       ─► Tracker（关联） ─► Scene Graph（插入或更新+融合）            │
└────────────────────────────────────┬─────────────────────────────────┘
                                     │  scene_graph.json
                                     ▼
┌──────────────────────────────────────────────────────────────────────┐
│  在线：使用 VLM 回答空间问题                                         │
│                                                                      │
│  问题 ─► QueryParser ─► GraphRetriever                               │
│    （子图 + 自我中心关系 + 关键帧）                                  │
│       ─► PromptBuilder ─► Qwen3-VL / InternVL ─► 答案                │
└──────────────────────────────────────────────────────────────────────┘
```

### 快速示例：DirectMe → Qwen3-VL

```python
from directme.mapping.scene_graph import SceneGraph
from directme.retrieval.retriever import GraphRetriever
from directme.qa.prompts import MultipleChoicePromptBuilder

# 1. 加载预先构建好的图
graph = SceneGraph.load_json("runs/my_session/scene_graph.json")

# 2. 设置用户当前姿态（来自 DA3 或实时 tracker）
from directme.retrieval.pose_lookup import pose_from_graph_timeline
pose = pose_from_graph_timeline(graph, timestamp=81.0)

# 3. 检索相关子图 + 关键帧
retriever = GraphRetriever(graph, reachable_radius_m=5.0)
context = retriever.retrieve("Where is the sink?", pose, language="en")

# 4. 构建提示词（UCS-Bench 中使用多项选择选项）
builder = MultipleChoicePromptBuilder(max_keyframes=4)
system, parts = builder.build(
    context,
    options=["On your left", "Behind you", "In front of you", ...]
)

# 5. 发送给 VLM（OpenAI 兼容 API，例如由 vLLM 提供服务的 Qwen3-VL）
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
# ...（完整代码见 examples/run_vlm_pipeline.py）
```

### 运行 VLM 流水线

```bash
# 使用 OpenAI 兼容 API 进行自由形式问答
python examples/run_vlm_pipeline.py \
    --graph runs/my_session/scene_graph.json \
    --question "Where is the sink relative to me?" \
    --backend openai --model qwen3-vl-8b-instruct \
    --base-url http://localhost:8000/v1

# 使用 HuggingFace Transformers 进行本地推理
python examples/run_vlm_pipeline.py \
    --graph runs/my_session/scene_graph.json \
    --question "Where is the sink relative to me?" \
    --backend transformers --model Qwen/Qwen3-VL-8B-Instruct
```

### 复现 UCS-Bench 结果（表 3）

```bash
# 第 1 步：为全部 540 个视频构建场景图
for vid in /data/ucsbench/videos/*/; do
    python examples/run_real_pipeline.py \
        --frames "$vid/frames" --out "/data/ucsbench/graphs/$(basename $vid)" \
        --classes "cup,phone,bottle,chair,table,sink,fridge,oven,..." \
        --device auto
done

# 第 2 步：通过 vLLM 使用 Qwen3-VL 进行评估
python examples/evaluate_ucsbench_vlm.py \
    --questions /data/ucsbench/questions.jsonl \
    --graphs-dir /data/ucsbench/graphs \
    --backend openai --model qwen3-vl-8b-instruct \
    --base-url http://localhost:8000/v1 \
    --out results/directme_qwen3vl.json
```

---

## 🔬 与 ConceptGraphs 的比较

| 特性 | ConceptGraphs | DirectMe |
| ------------------------ | ------------------- | -------------------------------------------------- |
| 场景图构建 | ✅ 静态、单次处理 | ✅ **增量式、流式处理** |
| 物体身份融合 | ✅ CLIP 多视角 | ✅ CLIP + HSV 直方图 + tracker |
| 坐标系 | 仅世界坐标 | 世界坐标 + **每次查询的自我中心坐标** |
| 自身运动跟踪 | ❌ | ✅ 姿态传播 + 漂移拒绝 |
| 第一视角关系 | ❌ | ✅ 8 类（front/behind/left/right/...） |
| 可达性推理 | ❌ | ✅ 基于深度，可配置半径 |
| 地点/房间归纳 | ❌ | ✅ 场景标签 + 几何聚类 |
| 轨迹记忆 | ❌ | ✅ `ego_pose_timeline` + `place_visit_timeline` |
| 在线问答流水线 | ❌ | ✅ 检索 → 提示词 → MLLM → 答案 |
| 评估基准 | ❌ | ✅ UCS-Bench，4 个维度 |
| 存储后端 | 仅内存 | JSON + SQLite（WAL，可恢复） |
| 异步流式导入 | ❌ | ✅ 有界队列，故障隔离 |

---

## 📚 文档

- [`docs/architecture.md`](docs/architecture.md) — 离线 / 在线分离的详细说明
- [`docs/algorithm_notes.md`](docs/algorithm_notes.md) — 设计选择与注意事项
- [`docs/adapter_guide.md`](docs/adapter_guide.md) — 如何接入 DA3 / YOLO-World / SAM 2
- [`docs/schema.md`](docs/schema.md) — 完整的场景图 JSON schema
- [`docs/paper_alignment.md`](docs/paper_alignment.md) — 从论文到代码的映射

---

## 🤝 贡献

```bash
make dev          # 安装运行时 + 开发 + 可视化额外依赖
pre-commit install
make check        # lint + test
```

欢迎提交 PR。请：

1. 为新功能添加测试。
2. 在提交前运行 `make format`。
3. 如果公共 API 发生变化，请更新 `docs/`。

---

## 📝 许可证

Apache-2.0。参见 [`LICENSE`](LICENSE)。

## 🙏 致谢

DirectMe 的身份融合设计借鉴了 **ConceptGraphs**（Gu 等，ICRA 2024）。真实模型适配器基于 **Depth Anything 3**、**YOLO-World**、**SAM 2**，以及（可选的）**Scal3R**。
