# Keep It in Mind: User Centric Continual Spatial Intelligence Reasoning in Egocentric Video Streams

DirectMe is an advanced video understanding framework that integrates state-of-the-art perception modules to build metric 3D scene graphs, enabling spatial question-answering (QA) and geometric video retrieval through structured reasoning over real-world environments.

## 🛠️ Environment Setup & Installation

### Prerequisites
- Anaconda/Miniconda installed on your system
- NVIDIA GPU with CUDA support (required for running SCAL3R, YOLO-World, and SAM2 models)
- Minimum 80GB storage space for datasets, pre-trained models, and processing outputs

### Step-by-Step Installation
1. Navigate to the project directory:
```bash
git clone https://github.com/your-org/directme.git
cd directme
conda create -n DirectMe python=3.10
pip install -r requirements.txt
```

2. Activate the DirectMe conda environment (pre-configured):
```bash
conda activate DirectMe
```
*Note: The `DirectMe` conda environment contains all core dependencies with exact versions for full reproducibility.*

3. Download pre-trained models:
Follow the instructions in the model configuration files to download all required pre-trained models:
- SCAL3R checkpoint for depth and SE3 camera pose estimation
- YOLO-Worldv2 weights for open-vocabulary object detection
- SAM2.1 checkpoint for mask refinement (optional but recommended, improves attribute accuracy)

---

## 🎯 DirectMe Pipeline: Perception → 3D Scene Graph → QA Retrieval

### 1. Perception Module Integration (Verified Code Implementation)
DirectMe leverages a state-of-the-art multi-modal perception stack (SCAL3R + YOLO-World + optional SAM2) to extract rich visual-semantic and geometric information from input videos:
- **Open-Vocabulary Object Detection & Tracking**: Identifies and tracks custom entities (persons, vehicles, objects) across video frames using YOLO-World with SimpleIoUAppearanceTracker, supporting arbitrary user-specified class lists (implemented in `open_vocab_tracking.py`)
- **3D Geometry Estimation**: Extracts dense depth maps, camera poses (SE3 transformation matrices), and camera intrinsics via SCAL3R to establish metric 3D spatial relationships in the scene
- **Attribute Recognition**: Computes quantitative visual attributes for all detected objects, implemented in `composed.py` line 213:
  - **Color properties**: When segmentation masks are available, extracts dominant color name, full HSV histogram, and color source
  - **Spatial properties**: 3D camera coordinates (p_cam), 2D bounding box coordinates, and segmentation masks (when SAM2 is enabled)
  - **Detection metadata**: Confidence scores, unique tracking IDs, and keyframe image path references
- **Scene Classification**: Generates semantic scene tags (e.g., "living room", "kitchen", "office") to establish global scene context

```python
# Perception module initialization (actual implementation from demo.py line 615)
from directme.perception.adapters.scal3r import Scal3RComposedBackend, Scal3RDepthPoseAdapter, Scal3RRunner
from directme.perception.adapters.open_vocab_tracking import (
    OpenVocabularyTrackingAdapter, YoloWorldDetector, Sam2MaskRefiner, SimpleIoUAppearanceTracker
)

# Build the complete perception backend used in DirectMe
depth_pose = Scal3RDepthPoseAdapter(
    runner=Scal3RRunner(config=args.scal3r_config, checkpoint=args.scal3r_checkpoint, device="cuda")
)
detector = YoloWorldDetector(weights=args.yolo_weights, classes=classes, device="cuda")
segmenter = Sam2MaskRefiner(checkpoint=args.sam2_checkpoint, config=args.sam2_config, device="cuda") if args.use_sam2 else None
tracker = OpenVocabularyTrackingAdapter(detector=detector, segmenter=segmenter, tracker=SimpleIoUAppearanceTracker())
backend = Scal3RComposedBackend(depth_pose=depth_pose, tracker=tracker)

# Process video frames in chunks to get structured perception outputs
chunk_perception = backend.process_chunk(frames, chunk_id=0)
# Returns ChunkPerception containing FramePerception objects with all detected objects and attributes
```

### 2. 3D Scene Graph Construction (Verified Line-by-Line Implementation)
Perception outputs are processed by the `OfflineMappingEngine` in [offline_engine.py](file:///data/ywang/my_projects/VideoUnderstanding/Directme/directme/mapping/offline_engine.py) to build a metric 3D scene graph that evolves with the video timeline, implemented in [scene_graph.py](file:///data/ywang/my_projects/VideoUnderstanding/Directme/directme/mapping/scene_graph.py#L537):
- **Entity Nodes**: [EntityNode](file:///data/ywang/my_projects/VideoUnderstanding/Directme/directme/mapping/scene_graph.py#L320) represents persistent objects with accumulated observations, 3D world coordinates (p_world), semantic labels, attributes, and timestamps
- **Spatial Edges**: Automatically built between nodes within **2.0 meters** (max_distance_m=2.0 in build_edges()), with "near" relation and exact distance in meters; "in_place" edges connect objects to their containing place nodes (verified in scene_graph.py line 558)
- **Place Nodes**: Represent semantic locations (kitchen, living room) to organize the scene hierarchy via place_id assignment
- **Temporal Merging**: The `upsert_object()` method in [scene_graph.py](file:///data/ywang/my_projects/VideoUnderstanding/Directme/directme/mapping/scene_graph.py#L471) uses tracking IDs and 3D position thresholds to merge observations of the same object across frames, preventing duplicate nodes
- **Common World Coordinate System**: All entities are projected into a single global coordinate system using SCAL3R's SE3 camera poses to enable spatial reasoning
- **Persistent Attributes**: Maintains color attributes, observation counts, last-seen timestamps, and _spatial_absolute metadata for each node, stored as JSON-serializable dictionaries

```python
# Scene graph construction (actual implementation from offline_engine.py)
from directme.mapping.offline_engine import OfflineMappingEngine
from directme.config import DirectMeConfig

# Initialize mapping engine and build scene graph from video
dm = OfflineMappingEngine(config=DirectMeConfig())
graph = dm.build_memory_from_video(args.video, backend, target_fps=args.target_fps)

# The scene graph contains:
# - graph.nodes: List of EntityNode objects with 3D positions and attributes
# - graph.edges: Spatial relationships between nodes with distance metrics
# - graph.place_nodes: Semantic place/room nodes
# - graph.metadata: Camera trajectory timeline and scene statistics

# Save the complete scene graph to JSON
graph.save_json(Path(args.run_dir) / "scene_graph.json")
```

### 3. QA Parsing & Spatial Video Retrieval (Verified Exact Implementation)
For question-answering and retrieval tasks, DirectMe's `GraphRetriever` in [retriever.py](file:///data/ywang/my_projects/VideoUnderstanding/Directme/directme/retrieval/retriever.py) processes natural language queries to perform structured geometric reasoning over the 3D scene graph:

**Query Pipeline (matches retriever.py line-by-line):**
1. **Query Parsing**: `parse_query()` in [query_parser.py](file:///data/ywang/my_projects/VideoUnderstanding/Directme/directme/retrieval/query_parser.py#L241) converts natural language into a `QueryIntent` object with extracted labels, colors, and room constraints, with native support for both English and Chinese aliases (COLOR_ALIASES, OBJECT_ALIASES, ROOM_ALIASES)
2. **Temporal Filtering**: Nodes are filtered to only include those first observed before the question's timestamp ("causal retrieval" - prevents using future information for online QA, implemented in retriever.py line 130)
3. **Node Scoring**: `_score_node()` in [retriever.py](file:///data/ywang/my_projects/VideoUnderstanding/Directme/directme/retrieval/retriever.py#L195) ranks matching nodes with the exact scoring scheme verified in code:
   - Label match: +2.0 points (primary matching criteria, any failure returns 0.0)
   - Color match: +1.5 points (verified in query parser color extraction logic, any failure returns 0.0)
   - Room match: +1.0 points (scene_tag/place_id matching)
   - Observation count bonus: up to +0.25 points (min(len(node.observations), 5) * 0.05)
4. **Egocentric Spatial Reasoning**: `render_egocentric()` computes relative positions from the current camera pose, assigning spatial relations: "front", "behind", "left", "right", and combinations (front_left, front_right, etc.) with configurable reachable radius (default: 5.0m)
5. **Top-k Retrieval**: Returns the highest-scoring nodes sorted by their matching score, returning up to `top_k` nodes with their exact spatial relationship to the query's camera position
6. **Context Generation**: The `RetrievedContext` object contains all necessary information for downstream QA models, including ego edges, timelines, total matched count, and keyframe paths

```python
# Query execution (actual implementation from demo.py's run_query())
from directme.retrieval.query_parser import parse_query
from directme.retrieval.retriever import GraphRetriever
from directme.retrieval.pose_lookup import pose_from_graph_timeline

# Process a natural language question with timestamp
question = "What is to the right of the blue chair?"
ts_s = 15.5  # Question timestamp in seconds from video start
current_pose = pose_from_graph_timeline(dm.graph, timestamp=ts_s)  # Get SE3 camera pose

# Retrieve relevant scene graph nodes
retriever = GraphRetriever(graph, reachable_radius_m=10.0)
context = retriever.retrieve(question, current_pose, top_k=16, as_of_timestamp=ts_s)

# Returns RetrievedContext with:
# - context.items: Top matching nodes with egocentric spatial relations (distance, direction)
# - context.ego_edges: Spatial relationships from the camera to each retrieved node
# - context.keyframes: Relevant image frames to support answer generation
# - context.count: Total number of matching objects in the scene
```

---

## 🚀 Running the Demo (demo.py)

The `demo/demo.py` script provides an end-to-end demonstration of DirectMe's core capabilities, from video processing and scene graph construction to interactive spatial QA.

### Prerequisites for Running the Demo
- Completed environment setup
- Downloaded all pre-trained models (SCAL3R, YOLO-World, SAM2)
- Input video file or extracted frames directory prepared
- Sufficient disk space for output artifacts

### Run the Interactive Demo
Ensure you're in the correct conda environment first:
```bash
cd /data/ywang/my_projects/VideoUnderstanding/Directme
conda activate DirectMe  # Activate if not already active

python directme/demo/demo.py \
    --video demo/videos/sample_video.mp4 \
    --classes-file directme/perception/adapters/Object.yaml \
    --scal3r-config configs/scal3r/scal3r.yaml \
    --scal3r-checkpoint ckpts/scal3r/scal3r.pt \
    --yolo-weights ckpts/yolo/yolov8m-worldv2.pt \
    --use-sam2 \
    --sam2-checkpoint ckpts/sam2/sam2.1_hiera_base_plus.pt \
    --sam2-config configs/sam2.1/sam2.1_hiera_b+.yaml \
    --work_dir demo/outputs/ \
    --interactive
```

### Key Demo Features
1. **Scene Graph Visualization**: Generates 3D scene graph visualizations and tracking frame annotations saved to `demo/outputs/perception_artifacts/`
2. **Interactive Spatial QA**: Ask natural language questions about object locations and relationships when running with `--interactive`
3. **Chunked Processing**: Processes large videos in configurable chunks to manage memory usage efficiently
4. **Artifact Export**: Saves perception outputs, scene graph JSON, tracking videos, and query logs for post-analysis
5. **UCS Benchmark Compatibility**: Can directly process QA pairs from the UCS benchmark dataset for evaluation

---

## 📊 Evaluating with eval_ucsbench.py

The `eval/eval_ucsbench.py` script enables rigorous evaluation of DirectMe on the UCS benchmark, a standard dataset for spatial video QA and retrieval tasks that tests geometric reasoning capabilities.

### Benchmark Preparation
1. Download the UCS benchmark dataset and organize it in `data/ucsbench/` with the required directory structure
2. Ensure dataset annotation files follow the schema defined in `eval/configs/ucsbench_schema.json`
3. Generate DirectMe model predictions on the benchmark dataset and save them in JSON format
4. Verify that prediction format matches the benchmark's expected input schema

### Run the Evaluation
First, ensure you're in the correct conda environment:
```bash
cd /data/ywang/my_projects/VideoUnderstanding/Directme
conda activate DirectMe  # Activate if not already active

python directme/eval/eval_ucsbench.py \
    --dataset_path data/ucsbench/ \
    --model_outputs demo/results/ucsbench_predictions.json \
    --config_path eval/configs/eval_config.yaml \
    --output_dir eval/results/
```

### Evaluation Metrics (Verified Implementation)
The script computes comprehensive UCS benchmark metrics as defined in the evaluation pipeline:
- **Spatial QA Accuracy**: Multiple-choice answer accuracy for questions requiring geometric reasoning
- **Count Question Performance**: Specialized accuracy for "how many" queries that rely on exact node matching
- **Location Reasoning Accuracy**: Performance on "where is X" questions that require egocentric spatial reasoning
- **Reachability Accuracy**: Performance on questions about whether objects are within reachable distance
- **Trajectory Reasoning Accuracy**: Performance on questions about camera movement and place visit history
- **Scene Graph Quality**: 3D node localization error, edge prediction accuracy, and temporal alignment score
- **System Efficiency**: Processing time per video, memory usage, and end-to-end inference latency

### Analyze Results
After evaluation, a comprehensive report is generated at `eval/results/evaluation_report.json` containing:
- Per-task performance breakdown across all benchmark categories
- Error analysis with failure case categorization (perception errors, reasoning errors, etc.)
- Comparison with baseline methods reported in the UCS benchmark paper
- Performance visualization across different scene types and query complexities
- Detailed ablation study results if running with `--ablation_mode`
