# temporary
cp /code/nerfstudio/scripts/eval.py  /home/user/nerfstudio/nerfstudio/scripts/eval.py
cp /code/nerfstudio/scripts/nerfstudio_drone_eval.py  /home/user/nerfstudio/nerfstudio/scripts/nerfstudio_drone_eval.py
cp /code/nerfstudio/data/dataparsers/nerfstudio_dataparser.py  /home/user/nerfstudio/nerfstudio/data/dataparsers/nerfstudio_dataparser.py
cp /code/nerfstudio/cameras/cameras.py  /home/user/nerfstudio/nerfstudio/cameras/cameras.py

## Novel View Generalisation Benchmark (BM-NV)
#
# Training:  3 inward orbits (low/mid/high) + 3 traversals (forward/left/right)
#            90% train / 10% eval per trajectory, optimised poses, optimised intrinsics
#            NeRF camera-optimizer: OFF
#
# Eval-only: traversal_backward_low   -- unseen traversal direction
#            traverse_loop_low        -- perimeter path, different topology
#            orbit_outward_low        -- same spatial path as training orbits, camera flipped
#            rocket_upward            -- vertical ascent, scene-inward camera
#            bev_orbit_area           -- nadir, high altitude
#            panorama_360_station_a/b/c -- sweep_360, 4:3 intrinsics, inside scene
#
# Note: eval includes 4:3 panoramas alongside 16:9 training data (mixed intrinsics).
#       The dataparser and cameras.py copies above are required for this to work.

## input for the script
# ROOT_DIR="/workspace/datasets/"
# SCENE_NAME="backyard_sunny"
# EXP_FOLDER="exp_ds_bm_nv"
# NUM_ITERATIONS=30000

ROOT_DIR="$1"
SCENE_NAME="$2"
EXP_FOLDER="$3"
NUM_ITERATIONS="$4"

DST_DS=${SCENE_NAME}_bm_nv
TIME_STAMP=${SCENE_NAME}_bm_nv
echo ${TIME_STAMP}

python /code/scripts/exp/nerfstudio_integration.py \
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
      },
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
      },
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
      },
      "traverse_loop_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "orbit_outward_low": {
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
      "bev_orbit_area": {
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
      },
      "scattered_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      }
    }
  }'

# A workaround for training NeRF since evaluation with images of different resolutions is not supported
DST_DS_NERF=${SCENE_NAME}_bm_nv_nerf

python /code/scripts/exp/nerfstudio_integration.py \
  --scene-dir ${ROOT_DIR}/processed/${SCENE_NAME} \
  --dst-dir ${ROOT_DIR}/ns_processed/${DST_DS_NERF} \
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
      },
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
      },
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
      },
      "traverse_loop_low": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      },
      "orbit_outward": {
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
      "bev_orbit_area": {
        "c2w_rot_optimized": true,
        "c2w_trans_optimized": true,
        "camera_intrinsics_optimized": true,
        "fill_missing_poses_with_non_optimized": false
      }
    }
  }'


echo splatfacto
ns-train splatfacto --output-dir /workspace/outputs --experiment-name ${EXP_FOLDER} --save-only-latest-checkpoint True \
--vis viewer_legacy --data ${ROOT_DIR}/ns_processed/${DST_DS} --pipeline.model.random-scale 2 --pipeline.model.camera-optimizer.mode off \
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
--vis viewer_legacy --data ${ROOT_DIR}/ns_processed/${DST_DS_NERF} --pipeline.model.camera-optimizer.mode off \
--timestamp ${TIME_STAMP} --max-num-iterations ${NUM_ITERATIONS} --viewer.quit-on-train-completion True --pipeline.datamanager.camera-res-scale-factor 0.5 \
nerfstudio-data --eval-mode filename

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval \
--run-drone-ds-eval --eval-dataset ${ROOT_DIR}/ns_processed/${DST_DS}

ns-eval --load-config /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/config.yml \
--output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval_standard.json \
--render-output-path /workspace/outputs/${EXP_FOLDER}/nerfacto/${TIME_STAMP}/eval_standard
