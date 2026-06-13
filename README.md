

# UCS-Bench / DirectMe

**Keep It in Mind: User-Centric Continual Spatial Intelligence Reasoning in Egocentric Video Streams**

[![Paper](https://img.shields.io/badge/Paper-ICML%202026-blue)](https://icml.cc/virtual/2026/poster/63682)
[![Dataset](https://img.shields.io/badge/Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/cocowy1/UCS-Bench)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/cocowy1/UCS-Bench)

UCS-Bench is a benchmark and codebase for evaluating **user-centric continual spatial intelligence** in long egocentric video streams. The goal is to test whether models can perceive, remember, and reason about spatial environments from a user's first-person viewpoint over time.

<img width="5197" height="2598" alt="ucsbench" src="https://github.com/user-attachments/assets/9530fd91-7f43-458c-b900-25cbe7374290" />


This repository contains **DirectMe**, a spatial-memory-based video understanding framework that builds metric 3D scene graphs from egocentric videos and performs spatial question answering through structured retrieval.

## News

* **Dataset released:** [cocowy1/UCS-Bench on Hugging Face](https://huggingface.co/datasets/cocowy1/UCS-Bench)
* **Paper page:** [ICML 2026 poster](https://icml.cc/virtual/2026/poster/63682)
* **Code released:** DirectMe pipeline, evaluation scripts, third-party perception adapters, and demo videos.

## Demo Videos

GitHub should render the following uploaded videos directly in the README.

### Indoor Demo

https://github.com/user-attachments/assets/2a0cf554-4a35-46db-98be-e0e8c6f411f6

### Outdoor Demo

https://github.com/user-attachments/assets/01d55770-abed-45de-ae98-2ef60cbc6c9b

## Overview

**UCS-Bench** focuses on long-horizon spatial reasoning in continuous egocentric video streams. Unlike standard video QA benchmarks that mainly test short clips or offline reasoning, UCS-Bench asks models to answer timestamped questions while respecting the user's current viewpoint and previously observed spatial memory.

**DirectMe** tackles this setting with a structured pipeline:
<img width="3458" height="1258" alt="method" src="https://github.com/user-attachments/assets/bc692f50-06d3-47ef-951e-a0ce662fd007" />


```text
Egocentric Video
      ↓
Perception Modules
Depth Anything 3 / Scal3R / YOLO-World / SAM2
      ↓
Metric 3D Scene Graph
objects, places, 3D positions, temporal observations
      ↓
Spatial Retrieval
causal retrieval at the query timestamp
      ↓
Question Answering
rule-based or VLM-based answer generation
```

## Key Features

* **Continual egocentric spatial memory:** builds a scene graph over time instead of answering from isolated frames.
* **Metric 3D reasoning:** stores object positions, camera poses, distances, and egocentric directions.
* **Timestamp-aware QA:** retrieves only the information available before the question timestamp.
* **Open-vocabulary perception:** supports YOLO-World detection and optional SAM2 mask refinement.
* **Third-party perception backbones:** integrates Depth Anything 3, Scal3R, and SAM2 through Git submodules.
* **Benchmark evaluation:** includes scripts for evaluating DirectMe + VLM models on UCS-Bench.

## Repository Structure

```text
UCS-Bench/
├── configs/                     # Runtime configs, including SAM2 and Scal3R configs
├── directme/                    # Core DirectMe package
│   ├── demo/                    # Demo utilities and web visualization helpers
│   ├── eval/                    # Evaluation utilities
│   ├── geometry/                # SE3 poses and geometric utilities
│   ├── mapping/                 # 3D scene graph and offline mapping engine
│   ├── perception/              # Perception adapters and runtime builder
│   ├── qa/                      # QA prompt and answer generation
│   ├── retrieval/               # Spatial retrieval and query parsing
│   ├── storage/                 # JSON / SQLite storage helpers
│   └── viz/                     # Visualization utilities
├── examples/
│   ├── run_real_pipeline.py     # End-to-end DirectMe pipeline
│   ├── evaluate_ucsbench_vlm.py # DirectMe + VLM evaluation on UCS-Bench
│   ├── run_vlm_pipeline.py      # VLM pipeline example
│   └── visualize_graph.py       # Scene graph visualization
├── third_party/
│   ├── Depth-Anything-3/        # Submodule
│   ├── Scal3R/                  # Submodule, adapted fork
│   └── sam2/                    # Submodule
├── video/                       # Demo videos
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone the repository with submodules

Recommended:

```bash
git clone --recursive https://github.com/cocowy1/UCS-Bench.git
cd UCS-Bench
```

If you already cloned without `--recursive`, initialize third-party libraries manually:

```bash
git submodule update --init --recursive
```

The third-party libraries are managed as Git submodules:

```text
third_party/Depth-Anything-3
third_party/Scal3R
third_party/sam2
```

### 2. Create a conda environment

```bash
conda create -n ucsbench python=3.10 -y
conda activate ucsbench
pip install --upgrade pip
```

### 3. Install PyTorch

Install the PyTorch version matching your CUDA environment. For example, for CUDA 12.1:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If your CUDA version is different, please follow the official PyTorch installation selector.

### 4. Install DirectMe

```bash
pip install -e ".[video,perception,viz,api,vlm]"
```

### 5. Install third-party libraries from local submodules

This step is important. The perception backbones are not ordinary Python dependencies only; they live under `third_party/` and should be installed locally:

```bash
pip install -e third_party/Depth-Anything-3
pip install -e third_party/Scal3R
pip install -e third_party/sam2
```

If you want to reproduce the full development environment, you can additionally install:

```bash
pip install -r requirements.txt
```

If `requirements.txt` reinstalls a remote version of a third-party library, simply reinstall the local submodule afterwards:

```bash
pip install -e third_party/Depth-Anything-3
pip install -e third_party/Scal3R
pip install -e third_party/sam2
pip install -e .
```

## Checkpoints and Large Files

Model checkpoints are **not included** in this repository. Please download them separately and place them under `ckpts/`.

A recommended layout is:

```text
ckpts/
├── sam2/
│   └── sam2.1_hiera_tiny.pt
├── yolo/
│   └── yolov8s-worldv2.pt
├── scal3r/
│   └── scal3r.pt
└── depth_anything_3/
```

Typical checkpoints include:

* Depth Anything 3 model weights
* Scal3R checkpoint for depth and camera pose estimation
* YOLO-World weights for open-vocabulary object detection
* SAM2 checkpoint for optional mask refinement

Large files such as datasets, checkpoints, output videos, and experiment logs should not be committed to GitHub.

## UCS-Bench Dataset

The UCS-Bench dataset is available on Hugging Face:

```text
https://huggingface.co/datasets/cocowy1/UCS-Bench
```

You can download it with:

```bash
pip install -U huggingface_hub

huggingface-cli download cocowy1/UCS-Bench \
  --repo-type dataset \
  --local-dir data/UCS-Bench
```

After downloading, inspect the dataset layout:

```bash
find data/UCS-Bench -maxdepth 3 -type f | head -50
```

The dataset contains egocentric videos and timestamped QA annotations. Depending on your local layout, you may organize the data as:

```text
data/UCS-Bench/
├── videos/
├── metadata.jsonl
├── questions.jsonl
└── ...
```

For evaluation, the script expects a question annotation file and pre-built scene graphs. If your downloaded annotation file has a different name or schema, please adapt the path or convert it to the expected JSONL format.

## Quick Start

### Option A: Smoke test without heavy GPU perception

Run the toy pipeline:

```bash
python examples/run_real_pipeline.py \
  --toy \
  --out runs/toy
```

### Option B: Run DirectMe on your own video

First extract frames from a video:

```bash
mkdir -p frames/demo

ffmpeg -i video/indoor_demo.mp4 \
  -vf fps=1 \
  frames/demo/frame_%06d.jpg
```

Then run the real perception pipeline:

```bash
python examples/run_real_pipeline.py \
  --frames frames/demo \
  --out runs/demo \
  --classes "cup,phone,bottle,laptop,chair,table,sink,fridge,door,bag,book" \
  --question "我身边有什么物体？它们在哪里？" \
  --language zh \
  --device cuda \
  --yolo-weights ckpts/yolo/yolov8s-worldv2.pt \
  --sam2-checkpoint ckpts/sam2/sam2.1_hiera_tiny.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml
```

If you do not want to use SAM2, omit the SAM2 arguments:

```bash
python examples/run_real_pipeline.py \
  --frames frames/demo \
  --out runs/demo_no_sam2 \
  --classes "cup,phone,bottle,laptop,chair,table,sink,fridge,door,bag,book" \
  --question "What objects are around me?" \
  --language en \
  --device cuda \
  --yolo-weights ckpts/yolo/yolov8s-worldv2.pt
```

The output directory will contain the generated spatial memory and debugging files, such as retrieved objects and scene graph information.

## Running the Full Demo Pipeline

You can also run the packaged pipeline script:

```bash
bash run_pipeline.sh
```

Before running it, please edit the script to match your local paths, GPU ID, video path, checkpoints, and output directory.

## Evaluating on UCS-Bench

The evaluation script is:

```text
examples/evaluate_ucsbench_vlm.py
```

It evaluates DirectMe + VLM on UCS-Bench by:

1. loading pre-built scene graphs,
2. retrieving relevant spatial context at each question timestamp,
3. assembling a multiple-choice VLM prompt,
4. calling a VLM backend,
5. computing overall and per-dimension accuracy.

### 1. Build scene graphs for the videos

For each video, extract frames and build a scene graph:

```bash
python examples/run_real_pipeline.py \
  --frames data/UCS-Bench/videos/video_001/frames \
  --out data/UCS-Bench/graphs/video_001 \
  --classes "cup,phone,bottle,chair,table,sink,fridge,door,bag,book" \
  --storage-backend json \
  --device cuda
```

The expected graph output is typically:

```text
data/UCS-Bench/graphs/video_001/scene_graph.json
```

### 2. Evaluate with an OpenAI-compatible VLM endpoint

For example, if you are serving Qwen3-VL through vLLM:

```bash
python examples/evaluate_ucsbench_vlm.py \
  --questions data/UCS-Bench/questions.jsonl \
  --graphs-dir data/UCS-Bench/graphs \
  --backend openai \
  --model qwen3-vl-8b-instruct \
  --base-url http://localhost:8000/v1 \
  --api-key EMPTY \
  --out results/directme_qwen3vl.json
```

### 3. Evaluate with local Transformers

```bash
python examples/evaluate_ucsbench_vlm.py \
  --questions data/UCS-Bench/questions.jsonl \
  --graphs-dir data/UCS-Bench/graphs \
  --backend transformers \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --out results/directme_qwen3vl.json
```

## Output Files

Typical output files include:

```text
runs/
├── demo/
│   ├── scene_graph.json
│   ├── last_query.json
│   ├── keyframes/
│   └── ...
results/
└── directme_qwen3vl.json
```

`scene_graph.json` stores the spatial memory built from the video. `last_query.json` stores the most recent retrieved objects and egocentric spatial relations for debugging.

## Third-Party Libraries

This project uses the following third-party components:

* [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) for depth estimation
* [Scal3R](https://github.com/cocowy1/Scal3R) for depth and camera pose estimation
* [SAM2](https://github.com/facebookresearch/sam2) for segmentation mask refinement
* YOLO-World / Ultralytics for open-vocabulary object detection

They are included as submodules where applicable. To update them:

```bash
git submodule update --init --recursive
```

To check submodule status:

```bash
git submodule status
```

## Paper

Please refer to the ICML 2026 paper page:

```text
https://icml.cc/virtual/2026/poster/63682
```

## Dataset

Please refer to the Hugging Face dataset page:

```text
https://huggingface.co/datasets/cocowy1/UCS-Bench
```

## Citation

If you find UCS-Bench or DirectMe useful, please cite our work:

```bibtex
@misc{ucsbench2026,
  title = {Keep It in Mind: User-Centric Continual Spatial Intelligence Reasoning in Egocentric Video Streams},
  year = {2026},
  note = {ICML 2026},
  url = {https://icml.cc/virtual/2026/poster/63682}
}
```

The full BibTeX will be updated after the official proceedings metadata is available.

## Acknowledgements

We thank the authors and maintainers of Depth Anything 3, Scal3R, SAM2, YOLO-World, and the open-source egocentric video datasets that support research in spatial intelligence.

## License

This repository is released under the license specified in `LICENSE`. Please also follow the licenses and terms of use of the third-party libraries, pretrained models, and source datasets.
