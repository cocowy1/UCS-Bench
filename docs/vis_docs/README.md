# Visual Demo Docs

这个目录是 DirectMe demo 的可视化编程约束和说明文档。

## 目标

当前 demo 主要覆盖三层可视化：

1. 感知层：深度图视频、tracking 图视频。
2. 建图层：语义点云、场景图、topdown 俯视图。
3. 问答层：基于场景图检索的流式视频理解回答。

## 当前仓库里已经具备的内容

- `directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json`
- `directme_scal3r_full_pipeline/directme_mapping_run/topdown2.png`
- `directme_scal3r_full_pipeline/directme_mapping_run/topdown_scene0715_00-0_6.png`
- `directme_scal3r_full_pipeline/perception_artifacts/videos/depth_all.mp4`
- `directme_scal3r_full_pipeline/perception_artifacts/videos/tracking_all.mp4`
- `directme_scal3r_full_pipeline/semantic_pointcloud.ply`
- `directme_scal3r_full_pipeline/semantic_pointcloud_fused.ply`

## 当前机器上已经验证可运行的内容

- `PYTHONPATH=. python3 -m directme.demo --qa-json '' --graph-json directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json`
  - 成功加载 `directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json`
- `http://127.0.0.1:8765/directme/demo/web/`
  - 成功加载 depth/tracking 视频、上下同步 3D scene graph / dense mapping、语义点云和时间轴增长动画
- `http://127.0.0.1:8765/directme/demo/web/debug.html`
  - 成功加载坐标调试页面，可保存 `alignment_config.json`
- `PYTHONPATH=. python3 examples/visualize_graph.py --graph directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json --qa-json '' --question '我现在周围有什么？' --timestamp-s 20 --language zh --out directme_scal3r_full_pipeline/directme_mapping_run/topdown_from_cli.png`
  - 成功生成新的 topdown PNG

## 当前仍缺失或未验证项

- `torch`
- `transformers`
- SCAL3R / YOLO-World / SAM2 本地权重
- 端到端重跑所需的完整 Python 推理环境

这意味着：

- 现在已经能展示 `depth/tracking video + scene_graph + semantic pointcloud + retrieval/topdown/answer`。
- 现在还不能确认本机可重新执行 `video -> perception -> scene graph` 的完整重建链路。

## 可视化开发约束

- demo 代码只在 `directme/demo/` 里改。
- 所有新默认参数都必须优先指向仓库内已有资源。
- 当仓库内没有真实视觉产物时，文档必须明确“已有图”“缺失图”“待补输入”。
- 流程展示优先用 SVG，便于后续继续编辑和嵌入汇报材料。

## 推荐本地命令

3D 汇报页面：

```bash
python3 -m directme.demo.web_server --port 8765
```

打开 `http://127.0.0.1:8765/directme/demo/web/`。

坐标调试页面：

```text
http://127.0.0.1:8765/directme/demo/web/debug.html
```

默认 sample QA：

```bash
PYTHONPATH=. python3 -m directme.demo
```

只加载图，不跑 QA：

```bash
PYTHONPATH=. python3 -m directme.demo --qa-json ''
```

交互式检索问答：

```bash
PYTHONPATH=. python3 -m directme.demo --interactive
```

手动生成 topdown：

```bash
PYTHONPATH=. python3 examples/visualize_graph.py \
  --graph directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json \
  --qa-json '' \
  --question '我现在周围有什么？' \
  --timestamp-s 20 \
  --language zh \
  --out directme_scal3r_full_pipeline/directme_mapping_run/topdown_from_cli.png
```

## Demo 代码流程图

见 [demo_flow.svg](./demo_flow.svg)。
