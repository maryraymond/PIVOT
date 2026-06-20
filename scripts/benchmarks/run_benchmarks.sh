ROOT_DIR="/workspace/datasets"

EXP_NAME="exp_ds_bm_poses"
sh /code/scripts/run_bm_poses_trace.sh ${ROOT_DIR} "backyard_sunny" ${EXP_NAME} 30000


python3 /code/scripts/src/benchmark_table_poses.py \
	--root /workspace/outputs/${EXP_NAME} \
	--out /workspace/outputs/${EXP_NAME}/results_bm_poses.csv
	
	
python3 /code/scripts/src/benchmark_qualitative_poses.py \
    --root /workspace/outputs/${EXP_NAME} \
    --out-dir /workspace/outputs/${EXP_NAME}/output_plots 

EXP_NAME="exp_ds_bm_cam_cal"
sh /code/scripts/run_bm_cam_calibr_trace.sh ${ROOT_DIR} "backyard_sunny" ${EXP_NAME} 30000


python3 /code/scripts/src/benchmark_table_cam_calibr.py \
	--root /workspace/outputs/${EXP_NAME} \
	--out /workspace/outputs/${EXP_NAME}/results_bm_poses.csv
	
	
python3 /code/scripts/src/benchmark_qualitative_cam_calibr.py \
    --root /workspace/outputs/${EXP_NAME} \
    --out-dir /workspace/outputs/${EXP_NAME}/output_plots 


EXP_NAME="exp_ds_bm_nv"
sh /code/scripts/run_bm_nv_trace.sh ${ROOT_DIR} "backyard_sunny" ${EXP_NAME} 60000


python3 /code/scripts/src/benchmark_table_nv.py \
	--root /workspace/outputs/${EXP_NAME} \
	--out /workspace/outputs/${EXP_NAME}/results_bm_nv.csv
	
	
python3 /code/scripts/src/benchmark_qualitative_nv.py \
    --root /workspace/outputs/${EXP_NAME} \
    --out-dir /workspace/outputs/${EXP_NAME}/output_plots 