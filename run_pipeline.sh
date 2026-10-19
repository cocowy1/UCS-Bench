#!/bin/bash
set -e

echo "=== 1. Set up environment ==="
cd /data/ywang/my_projects/VideoUnderstanding/Directme
source /home/ywang/anaconda3/etc/profile.d/conda.sh
conda activate cambrian

# Change this to the index of an idle GPU found via nvidia-smi
export CUDA_VISIBLE_DEVICES=3

echo "=== 2. Run DirectMe pipeline (mapping & perception) ==="
PYTHONPATH=. python -m directme.demo \
  --mode rule \
  --backend scal3r \
  --input-mode from_perception \
  --video "tmp/qa_demo2.mp4" \
  --work-dir /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1 \
  --frame-dump-dir /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/frames \
  --run-dir /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/directme_mapping_run \
  --perception-artifact-dir /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/perception_artifacts \
  --target-fps 1 \
  --chunk-size 60 \
  --device cuda:0 \
  --qa-json '' \
  --use-sam2 \
  --sam2-checkpoint ckpts/sam2/sam2.1_hiera_tiny.pt \
  --sam2-config configs/sam2.1/sam2.1_hiera_t.yaml \
  --yolo-weights ckpts/yolo/yolov8m-worldv2.pt \
  --output-json /tmp/demo_test_results1.json

# Record the target FPS into the temp dir used by the Web UI
echo '{"fps": 1.0}' > /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/fps_record.json

echo "=== 3. Export dense pointcloud ==="
PYTHONPATH=. python directme/demo/export_dense_map.py \
  --data-root /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1 \
  --graph-json /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/directme_mapping_run/scene_graph.json \
  --output /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/dense_pointcloud_world.ply

echo "=== 4. Done ==="
echo "All data has been generated in /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/"
echo "You can now start the web server to view the results:"
echo "python -m directme.demo.web_server --port 8765"
