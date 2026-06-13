# DirectMe Demo 跑通指南

在 `/data/ywang/my_projects/VideoUnderstanding/Directme/` 仓库下，用 conda 环境跑通 `directme/demo/demo.py`。

> 以下命令已在 2026-06-05 测试通过。

## 0. 前置准备

```bash
cd /data/ywang/my_projects/VideoUnderstanding/Directme
```

确认以下文件存在：

- 场景图：`tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json`
- QA 文件：`directme/demo/sample_qa.json`
- 关键帧：`tmp/directme_scal3r_full_pipeline/frames/frame_*.jpg`（100 帧）
- Qwen3-VL 权重：`/data/ywang/my_projects/VideoUnderstanding/Qwen3-VL-8B-Instruct/`（bf16，~17 GB，需 ≥16 GB 显存）
- `directme` 已 editable 安装到 conda 环境

## 1. 选择 conda 环境

| conda env   | transformers | Qwen3-VL |
|-------------|--------------|----------|
| `qwen_vl`   | 4.37.2       | ❌       |
| `vgllm`     | 4.50.0       | ❌       |
| `video_chat`| 4.57.5       | ✅       |
| **`cambrian`** | 5.9.0    | ✅       |

**推荐使用 `cambrian`**（rule 和 qwen 模式均已跑通）：

```bash
conda activate cambrian
```

验证环境：

```bash
python -c "import torch, transformers; from transformers import Qwen3VLForConditionalGeneration; print('torch', torch.__version__, '| transformers', transformers.__version__, '| Qwen3-VL OK')"
```

## 2. rule 模式（无需 GPU，快速验证）

```bash
PYTHONPATH=. python -m directme.demo \
    --mode rule \
    --graph-json tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json \
    --qa-json   directme/demo/sample_qa.json \
    --output-json /tmp/demo_rule_results.json
```

## 3. qwen 模式（需 GPU ≥16 GB）

### GPU 推理（推荐）
```bash
PYTHONPATH=. python -m directme.demo \
    --mode qwen \
    --model-path /data/ywang/my_projects/VideoUnderstanding/Qwen3-VL-8B-Instruct \
    --no-load-in-4bit \
    --device-map cuda:1 \
    --graph-json tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json \
    --qa-json   directme/demo/sample_qa.json \
    --vlm-frame-budget 4 \
    --output-json /tmp/demo_qwen_results.json
```

### CPU 推理（仅验证，单条 3~5 分钟）
```bash
PYTHONPATH=. python -m directme.demo \
    --mode qwen \
    --model-path /data/ywang/my_projects/VideoUnderstanding/Qwen3-VL-8B-Instruct \
    --no-load-in-4bit \
    --device-map cpu \
    --graph-json tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json \
    --qa-json   directme/demo/sample_qa.json \
    --vlm-frame-budget 2\
    --max-image-size 224 \
    --output-json /tmp/demo_qwen_results.json
```

## 4. 其他命令

```bash
# 仅验证场景图
PYTHONPATH=. python -m directme.demo --mode rule \
    --graph-json tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json \
    --qa-json ''

# 交互式问答 
PYTHONPATH=. python -m directme.demo --mode rule --interactive

# 使用 pipeline summary 自动定位场景图
PYTHONPATH=. python -m directme.demo --mode rule \
    --pipeline-summary tmp/directme_scal3r_full_pipeline/full_pipeline_summary.json \
    --qa-json directme/demo/sample_qa.json

# 启动 3D 汇报 web 服务器
python -m directme.demo.web_server --port 8765
```

## 5. 故障排查

| 现象 | 处理 |
|------|------|
| `ImportError: Qwen3VLForConditionalGeneration` | 改用 `cambrian` 或 `video_chat` |
| `FileNotFoundError: 场景图 JSON` | 显式传 `--graph-json` |
| `未找到 Qwen3-VL 模型` | 检查 `config.json` + 4 个 `.safetensors` 文件 |
| GPU `OutOfMemoryError` | 减小 `--vlm-frame-budget` 或 `--max-image-size`，或改用 CPU |
| 推理特别慢 | 检查是否跑在 CPU 上，有 GPU 时加 `--device-map cuda:0` |

## 6. 一键运行

### rule 模式
```bash
cd /data/ywang/my_projects/VideoUnderstanding/Directme && \
PYTHONPATH=. conda run -n cambrian python -m directme.demo \
    --mode rule \
    --graph-json tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json \
    --qa-json   directme/demo/sample_qa.json \
    --output-json /tmp/demo_rule_results.json
```

### qwen 模式
```bash
cd /data/ywang/my_projects/VideoUnderstanding/Directme && \
PYTHONPATH=. conda run -n cambrian python -m directme.demo \
    --mode qwen \
    --model-path /data/ywang/my_projects/VideoUnderstanding/Qwen3-VL-8B-Instruct \
    --no-load-in-4bit \
    --device-map cuda:0 \
    --graph-json tmp/directme_scal3r_full_pipeline/directme_mapping_run/scene_graph.json \
    --qa-json   directme/demo/sample_qa.json \
    --vlm-frame-budget 4 \
    --output-json /tmp/demo_qwen_results.json
```

> CPU 运行时，将 `--device-map cuda:0` 改为 `--device-map cpu`，并添加 `--vlm-frame-budget 2 --max-image-size 224`。
