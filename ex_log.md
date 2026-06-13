source ~/.bashrc


tmux new -s cam7
conda activate cambrian
python /data/ywang/my_projects/VideoUnderstanding/cambrian-s/sm_eval.py --qa_dir /data/ywang/dataset/SpatialMemory/MC_QAs_v20/data_v20_合并_选项_other_v17_llm --output /data/ywang/dataset/SpatialMemory/outputs/cambrian-s-7b/cambrian-s_timestamp64_rerun.json


modelscope download --model OpenGVLab/InternVL3-38B-Instruct


modelscope download --model OpenGVLab/InternVL3-38B-Instruct  --local_dir ./InternVL3-38B-Instruct
