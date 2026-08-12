# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

ROOT_DIR="$1"
SCENE_NAME="$2"

EXP_NAME="exp_ds_bm_nv"
sh run_bm_nv_trace.sh ${ROOT_DIR} ${SCENE_NAME} ${EXP_NAME} 60000


python3 summarize_results/benchmark_table_nv.py \
	--root /workspace/outputs/${EXP_NAME} \
	--out /workspace/outputs/${EXP_NAME}/results_bm_nv.csv
	
	
python3 summarize_results/benchmark_qualitative_nv.py \
    --root /workspace/outputs/${EXP_NAME} \
    --out-dir /workspace/outputs/${EXP_NAME}/output_plots 

sh run_bm_poses_trace.sh ${ROOT_DIR} ${SCENE_NAME} ${EXP_NAME} 30000


python3 summarize_results/benchmark_table_poses.py \
	--root /workspace/outputs/${EXP_NAME} \
	--out /workspace/outputs/${EXP_NAME}/results_bm_poses.csv
	
	
python3 summarize_results/benchmark_qualitative_poses.py \
    --root /workspace/outputs/${EXP_NAME} \
    --out-dir /workspace/outputs/${EXP_NAME}/output_plots 

EXP_NAME="exp_ds_bm_cam_cal"
sh run_bm_cam_calibr_trace.sh ${ROOT_DIR} ${SCENE_NAME} ${EXP_NAME} 30000


python3 summarize_results/src/benchmark_table_cam_calibr.py \
	--root /workspace/outputs/${EXP_NAME} \
	--out /workspace/outputs/${EXP_NAME}/results_bm_poses.csv
	
	
python3 summarize_results/benchmark_qualitative_cam_calibr.py \
    --root /workspace/outputs/${EXP_NAME} \
    --out-dir /workspace/outputs/${EXP_NAME}/output_plots 


