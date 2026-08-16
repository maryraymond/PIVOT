# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.


ROOT_DIR="$1"
SCENE_NAME="$2"
EXP_FOLDER="$3"
NUM_ITERATIONS="$4"

DST_DS=${SCENE_NAME}_bm_03_opt_camera
TIME_STAMP=${SCENE_NAME}_opt_camera
echo ${TIME_STAMP}

python "$(command -v export_dataset.py)" \
  --scene-dir ${ROOT_DIR}/processed/${SCENE_NAME} \
  --dst-dir ${ROOT_DIR}/ns_processed/${DST_DS} \
  --scene-config '{
    "train": {

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
      }
    },
    "eval": {
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
      }
    }
  }'

echo splatfacto 
ns-train splatfacto --output-dir /workspace/outputs --experiment-name ${EXP_FOLDER} --save-only-latest-checkpoint True \
--vis viewer_legacy --data ${ROOT_DIR}/ns_processed/${DST_DS} --pipeline.model.random-scale 10 --pipeline.model.camera-optimizer.mode off \
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
--vis viewer_legacy --data ${ROOT_DIR}/ns_processed/${DST_DS} --pipeline.model.camera-optimizer.mode off \
--timestamp ${TIME_STAMP} --max-num-iterations ${NUM_ITERATIONS} --viewer.quit-on-train-completion True  --pipeline.datamanager.camera-res-scale-factor 0.5 \
nerfstudio-data --eval-mode filename 

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval \
--run-drone-ds-eval

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval_standard.json \
--render-output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval_standard




DST_DS=${SCENE_NAME}_bm_03_calib_camera
TIME_STAMP=${SCENE_NAME}_calib_camera
echo ${TIME_STAMP}

python "$(command -v export_dataset.py)" \
  --scene-dir ${ROOT_DIR}/processed/${SCENE_NAME} \
  --dst-dir ${ROOT_DIR}/ns_processed/${DST_DS} \
  --scene-config '{
    "train": {
      "orbit_inward_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": false,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "orbit_inward_mid": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": false,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      },
      "orbit_inward_high": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": false,
        "fill_missing_poses_with_non_optimized": false,
        "percentage": 0.9
      }
    },
    "eval": {
      "orbit_inward_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": false,
        "fill_missing_poses_with_non_optimized": false
      },
      "orbit_inward_mid": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": false,
        "fill_missing_poses_with_non_optimized": false
      },
      "orbit_inward_high": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": false,
        "fill_missing_poses_with_non_optimized": false
      }
    }
  }'

echo splatfacto 
ns-train splatfacto --output-dir /workspace/outputs --experiment-name ${EXP_FOLDER} --save-only-latest-checkpoint True \
--vis viewer_legacy --data ${ROOT_DIR}/ns_processed/${DST_DS} --pipeline.model.random-scale 10 --pipeline.model.camera-optimizer.mode off \
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
--vis viewer_legacy --data ${ROOT_DIR}/ns_processed/${DST_DS} --pipeline.model.camera-optimizer.mode off \
--timestamp ${TIME_STAMP} --max-num-iterations ${NUM_ITERATIONS} --viewer.quit-on-train-completion True --pipeline.datamanager.camera-res-scale-factor 0.5 \
nerfstudio-data --eval-mode filename 

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval \
--run-drone-ds-eval

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval_standard.json \
--render-output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval_standard

