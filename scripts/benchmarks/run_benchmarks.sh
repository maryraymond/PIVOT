# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

ROOT_DIR="$1"
SCENE_NAME="$2"

EXP_NAME="exp_ds_bm_nv"
run_bm_nv_trace ${ROOT_DIR} ${SCENE_NAME} ${EXP_NAME} 60000


EXP_NAME="exp_ds_bm_poses"

run_bm_poses_trace ${ROOT_DIR} ${SCENE_NAME} ${EXP_NAME} 30000


EXP_NAME="exp_ds_bm_cam_cal"
run_bm_cam_calibr_trace ${ROOT_DIR} ${SCENE_NAME} ${EXP_NAME} 30000



