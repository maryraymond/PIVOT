import json
from typing import Dict
import os
from pathlib import Path
import shutil

def process_split(scene_dir:str, dst_dir:str, 
                  scene_tranjectories:Dict, split_config:Dict, 
                  images_dir:str, split_prefix:str,
                  debug_prints:bool=False):
    frames = []
    for tranjectory, traj_config in split_config.items():
        if tranjectory in scene_tranjectories:
            pose_key = "colmap_pose_c2w" if traj_config["c2w_pose_optimized"] else "measured_pose_c2w"
            cam_intr_key = "camera_intrinsic_colmap" if traj_config["c2w_pose_optimized"] else "camera_intrinsic_calibration"
            alter_pose_key = "measured_pose_c2w" if traj_config["fill_missing_poses_with_non_optimized"] else None
            max_num_frames = int(traj_config["max_num_frames"]) if "max_num_frames" in traj_config else None
            ids = traj_config["indices"] if "indices" in traj_config else None

            if debug_prints:
                print(f"For {tranjectory} using c2w pose = {pose_key}, alternative pose = {alter_pose_key} and camera intr = {cam_intr_key}")

            traj_data = scene_tranjectories[tranjectory]

            cam_intr = traj_data[cam_intr_key]

            traj_num_frames = len(traj_data["frames"])

            if ids is not None :
                print(f"Indices is set for {tranjectory} is will use indices")
                traj_frame_range = list(ids)
                max_num_frames = None
            elif max_num_frames is None:
                max_num_frames = -1
            
            if max_num_frames is not None:
                
                if max_num_frames == -1:
                    step = 1
                    traj_frame_range = list(range(0, traj_num_frames, step))
                else:
                    n = min(max_num_frames, traj_num_frames)
                    traj_frame_range = [
                        round(i * (traj_num_frames - 1) / (n - 1))
                        for i in range(n)
                    ]
                            

            for i in traj_frame_range:

                if i > traj_num_frames:
                    print(f"Warning idex {i} is out of range for {tranjectory} skipping")
                    continue
                
                traj_frame = traj_data["frames"][i]

                if pose_key in traj_frame:
                    c2w = traj_frame[pose_key]
                elif alter_pose_key is not None:
                    c2w = traj_frame[alter_pose_key]
                else:
                    # This frame have no optimized pose and alternative pose is not enabled 
                    # we will need to skip this frame
                    if debug_prints:
                        print(f"Warning: skipping frames {traj_frame['file_name']} as it has no optimized pose and alternative measuered pose is not enabled")
                    continue
                
                frame = dict(cam_intr)
                src_file_name = Path(traj_frame["file_name"]).name
                src_file = scene_dir + f"/{traj_frame['file_name']}"
                dst_file_name = f"{images_dir}/{split_prefix}{tranjectory}_{src_file_name}"
                dst_file = dst_dir + "/" + dst_file_name

                frame["file_path"] = dst_file_name
                frame["transform_matrix"] = c2w

                frames.append(frame)

                shutil.copyfile(src_file, dst_file)
        else:
            raise ValueError(f"The specified trajectory {tranjectory} is not found in the scene data at {scene_dir}")
    return frames


def create_ns_dataset_from_scene(scene_dir:str, scene_config:Dict, 
                                 dst_dir:str, debug_prints:bool=False):
    transforms_data = {}
    transforms_file = dst_dir + "/transforms.json"
    frames = []
    if "train" in scene_config:
        # check if we have specific eval trajectories 
        # If eval is set to None or missing it means we will default to
        # eval mode of fraction and all the images will be in the images folder
        images_dir = "images"
        os.makedirs(dst_dir + "/" + images_dir, exist_ok=True)
        if "eval" in scene_config and scene_config["eval"] is not None:
            # we have a defined train eval split
            eval_prefix = "eval_"
            train_prefix = "train_"
        else:
            eval_prefix = None
            train_prefix = ""

        # start copying and populating the training dataset
        # now we read the scene data 
        scene_data_file = scene_dir + "/scene_data.json"
        with open(scene_data_file, 'r') as f:
            scene_data = json.load(f)
            scene_tranjectories = scene_data["trajectories"]
        # now for trajectory we will read and populate the frames
        frames += process_split(scene_dir=scene_dir, dst_dir=dst_dir,
                                scene_tranjectories=scene_tranjectories, split_config=scene_config["train"], 
                                split_prefix=train_prefix, images_dir=images_dir,
                                debug_prints=debug_prints)

        # see if we have a defined eval as well
        if eval_prefix is not None:
            frames += process_split(scene_dir=scene_dir, dst_dir=dst_dir,
                                    scene_tranjectories=scene_tranjectories, split_config=scene_config["eval"], 
                                    split_prefix=eval_prefix, images_dir=images_dir,
                                    debug_prints=debug_prints)
        
        transforms_data["frames"] = frames

        with open(transforms_file, 'w') as f:
            json.dump(transforms_data, f, indent=4)
            
    else:
        raise ValueError("No train configuration available")