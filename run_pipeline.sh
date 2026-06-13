#!/bin/bash
set -e

echo "=== 1. 设置环境 ==="
cd /data/ywang/my_projects/VideoUnderstanding/Directme
source /home/ywang/anaconda3/etc/profile.d/conda.sh
conda activate cambrian

# 请在此处修改为您通过 nvidia-smi 查到的空闲显卡编号
export CUDA_VISIBLE_DEVICES=3

echo "=== 2. 运行 DirectMe Pipeline (建图 & 感知) ==="
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

# 记录目标 FPS 值到 Web UI 所在的临时目录中
echo '{"fps": 1.0}' > /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/fps_record.json

echo "=== 3. 导出密集点云 (Dense Pointcloud) ==="
PYTHONPATH=. python directme/demo/export_dense_map.py \
  --data-root /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1 \
  --graph-json /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/directme_mapping_run/scene_graph.json \
  --output /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/dense_pointcloud_world.ply

echo "=== 4. 任务完成 ==="
echo "所有数据已生成在 /data/ywang/my_projects/VideoUnderstanding/Directme/tmp/directme_scal3r_full_pipeline1/"
echo "您现在可以启动 Web 服务查看结果："
echo "python -m directme.demo.web_server --port 8765"
