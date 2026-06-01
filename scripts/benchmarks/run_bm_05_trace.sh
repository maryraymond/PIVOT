# temproary
cp /code/nerfstudio/scripts/eval.py  /home/user/nerfstudio/nerfstudio/scripts/eval.py
cp /code/nerfstudio/scripts/nerfstudio_drone_eval.py  /home/user/nerfstudio/nerfstudio/scripts/nerfstudio_drone_eval.py

## input for the script
# ROOT_DIR="/workspace/datasets/"
# SCENE_NAME="backyard_sunny"
# EXP_FOLDER="exp_ds_06"
# NUM_ITERATIONS=2000

ROOT_DIR="$1"
SCENE_NAME="$2"
EXP_FOLDER="$3"
NUM_ITERATIONS="$4"

DST_DS=${SCENE_NAME}_bm_05_nv_2

TIME_STAMP=${SCENE_NAME}_nv_2
echo ${TIME_STAMP}


python /code/scripts/exp/nerfstudio_integration.py \
  --scene-dir ${ROOT_DIR}/processed/${SCENE_NAME} \
  --dst-dir ${ROOT_DIR}/ns_processed/${DST_DS} \
  --scene-config '{
    "train": {
      "traversal_forward_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "traversal_left_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "traversal_right_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "traversal_backward_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "orbit_inward_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "orbit_inward_mid": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "orbit_inward_high": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "bev_orbit_area": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "rocket_upward": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      }
    },
    "eval": {
      "traversal_forward_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "traversal_left_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "traversal_right_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "traversal_backward_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      }
      "orbit_inward_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "orbit_inward_mid": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "orbit_inward_high": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "bev_orbit_area": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "rocket_upward": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "scattered_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "panorama_360_station_a": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "panorama_360_station_b": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "panorama_360_station_c": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      }
    }
  }'


echo splatfacto 
ns-train splatfacto --output-dir /workspace/outputs --experiment-name ${EXP_FOLDER} --save-only-latest-checkpoint True \
--vis viewer_legacy --data ${ROOT_DIR}/ns_processed/${DST_DS} --pipeline.model.random-scale 2 --pipeline.model.camera-optimizer.mode SO3xR3 \
--timestamp ${TIME_STAMP} --max-num-iterations ${NUM_ITERATIONS} --viewer.quit-on-train-completion True --pipeline.datamanager.camera-res-scale-factor 0.5 \
nerfstudio-data --eval-mode filename 

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/splatfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/splatfacto/${TIME_STAMP}/eval \
--run-drone-ds-eval

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/splatfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/splatfacto/${TIME_STAMP}/eval_standard.json \
--render-output-path /workspace/outputs/${EXP_FOLDER}/splatfacto/${TIME_STAMP}/eval_standard


echo nerfacto 
ns-train nerfacto --output-dir /workspace/outputs --experiment-name ${EXP_FOLDER} --save-only-latest-checkpoint True \
--vis viewer_legacy --data ${ROOT_DIR}/ns_processed/${DST_DS} --pipeline.model.camera-optimizer.mode SO3xR3 \
--timestamp ${TIME_STAMP} --max-num-iterations ${NUM_ITERATIONS} --viewer.quit-on-train-completion True  --pipeline.datamanager.camera-res-scale-factor 0.5 \
nerfstudio-data --eval-mode filename 

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval \
--run-drone-ds-eval

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval_standard.json \
--render-output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval_standard