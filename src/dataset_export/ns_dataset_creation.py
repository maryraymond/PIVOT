# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

from __future__ import annotations

import json
from typing import Dict
import os
from pathlib import Path
import shutil
import numpy as np
import random


def create_nerfstudio_dataset_from_data(images_data, cameras_data, dataset_dir, 
                              src_images_dir, crop_larger_images=False):

    """ The input images data should be in the drone formate i.e: with c2w poses"""
    os.makedirs(dataset_dir, exist_ok=True)
    dataset_images_dir = f"{dataset_dir}/images"
    os.makedirs(dataset_images_dir, exist_ok=True)

    if len(images_data) == 0:
        raise ValueError(f"No images found!")
    
    if "pose_c2w" not in images_data[0]:
        raise ValueError("The data provided does not have c2w pose information")
    
    # check if the dataset have sub-directories to be created
    if "/" in images_data[0]["file_name"]:
        subdirs = set()
        for image in images_data:
            if image["file_name"].split("/")[0] not in subdirs:
                subdirs.add(image["file_name"].split("/")[0])
        
        for subdir in subdirs:
            os.makedirs(f"{dataset_images_dir}/{subdir}", exist_ok=True)


    # if we have more than one camera we might need to crop to the smallest resolution
    if crop_larger_images:
        target_res = {}
        for camera_data in cameras_data.values():
            if len(target_res) == 0:
                target_res["h"] = int(camera_data["h"])
                target_res["w"] = int(camera_data["w"])
            else:
                if int(camera_data["h"]) <= target_res["h"]:
                    if int(camera_data["w"]) <= target_res["w"]:
                        target_res["h"] = int(camera_data["h"])
                        target_res["w"] = int(camera_data["w"])
                    else:
                        raise ValueError(f"Can't find a target crop resolution for target {target_res} and camera H={camera_data['h']}, W={camera_data['w']}")
                elif int(camera_data["w"]) <= target_res["w"]:
                    if int(camera_data["h"]) <= target_res["h"]:
                        target_res["h"] = int(camera_data["h"])
                        target_res["w"] = int(camera_data["w"])
                    else:
                        raise ValueError(f"Can't find a target crop resolution for target {target_res} and camera H={camera_data['h']}, W={camera_data['w']}")

        print(f"Crop to min enabled, target image size is {target_res}")     

    frames = []
    transforms = {}
    for image in images_data:

        camera_data = cameras_data[image["camera_id"]]

        image_name = image["file_name"]
        dst_image = f"{dataset_images_dir}/{image_name}"
        dst_image_path = f"{Path(dataset_images_dir).name}/{image_name}"

        if crop_larger_images and ((int(camera_data["h"]) > target_res["h"]) or (int(camera_data["w"]) > target_res["w"])):
            x0 = int ((int(camera_data["w"]) - target_res["w"]) / 2)
            y0 = int ((int(camera_data["h"]) - target_res["h"]) / 2)

            H = target_res["h"]
            W = target_res["w"]
            cx = int(camera_data["cx"]) - x0
            cy = int(camera_data["cy"]) - y0

            src_img = cv2.imread(f"{src_images_dir}/{image_name}")
            cv2.imwrite(dst_image, src_img[y0:y0+target_res["h"], x0:x0+target_res["w"]])
        else:
            H = camera_data["h"]
            W = camera_data["w"]
            cx = camera_data["cx"]
            cy = camera_data["cy"]
            shutil.copyfile(f"{src_images_dir}/{image_name}", dst_image)

        frame_data = {"camera_model": camera_data["camera_type"],
                      "w": W,         "h": H,
                      "fl_x": camera_data["fl_x"],   "fl_y": camera_data["fl_y"],
                      "cx": cx,       "cy": cy,
                      "k1": camera_data["k1"],       "k2": camera_data["k2"],    "k3":camera_data["k3"],
                      "p1": camera_data["p1"],       "p2": camera_data["p2"],
                      "file_path": dst_image_path,
                      "transform_matrix":  image["pose_c2w"]}

        frames.append(frame_data)

    transforms["frames"] = frames

    transforms_file = f"{dataset_dir}/transforms.json"

    with open (transforms_file, 'w') as f:
        json.dump(transforms, f, indent=4)

def random_train_eval_ids(n_frames: int, train_percent: float = 0.9, seed: int = 42):
    ids = list(range(n_frames))  # frame IDs: 0 to N-1

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_train = round(n_frames * train_percent)

    train_ids = sorted(ids[:n_train])
    eval_ids = sorted(ids[n_train:])

    return train_ids, eval_ids

def _expand_all_key(split_config, scene_trajectories):
    if "all" not in split_config:
        return split_config
    all_conf = split_config.pop("all")
    for traj in scene_trajectories:
        if traj not in split_config:
            split_config[traj] = dict(all_conf)
    return split_config

def update_config_with_indices(scene_config, scene_tranjectories):
    # get a set of traj in train and eval that have a percenatge set
    train_config = scene_config["train"]
    train_set_with_percentage = set()
    for traj in train_config.keys():
        if "percentage" in train_config[traj]:
            train_set_with_percentage.add(traj)

    eval_config = scene_config["eval"] if "eval" in scene_config else None

    # check if we have eval with percentage then we need to create a split
    for traj in train_set_with_percentage:
        # at least for train we need to set the IDs for the percentage
        train_percenatge = float(train_config[traj]["percentage"])
        train_ids, eval_ids = random_train_eval_ids(n_frames=scene_tranjectories[traj]["number_frames_in_traj"], 
                                                    train_percent=train_percenatge)
        train_config[traj]["indices"] = train_ids
        if eval_config is not None and traj in eval_config.keys():
            print(f"Setting the eval percentage for traj {traj} as {1-train_percenatge} %")
            eval_config[traj]["indices"] = eval_ids

    # update the overall config 
    scene_config["train"] = train_config
    scene_config["eval"] = eval_config

    return scene_config

def process_split(scene_dir:str, dst_dir:str, 
                  scene_tranjectories:Dict, split_config:Dict, 
                  images_dir:str, split_prefix:str,
                  debug_prints:bool=False):
    frames = []
    for tranjectory, traj_config in split_config.items():
        if tranjectory in scene_tranjectories:
            cam_intr_key = "camera_intrinsic_colmap" if traj_config["camera_intrinsics_optimized"] else "camera_intrinsic_calibration"
            alter_pose_key = "measured_pose_c2w" if traj_config["fill_missing_poses_with_non_optimized"] else None
            max_num_frames = int(traj_config["max_num_frames"]) if "max_num_frames" in traj_config else None
            ids = traj_config["indices"] if "indices" in traj_config else None

            if debug_prints:
                print(f"For {tranjectory} using c2w rot optimized = {traj_config['c2w_rot_optimized']}, \
                      c2w trans optimized = {traj_config['c2w_trans_optimized']}, alternative pose = {alter_pose_key} \
                      and camera intr = {cam_intr_key}")

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

                if i >= traj_num_frames:
                    print(f"Warning idex {i} is out of range for {tranjectory} skipping")
                    continue
                
                traj_frame = traj_data["frames"][i]

                if "colmap_pose_c2w" in traj_frame:
                    c2w_opt = np.array(traj_frame["colmap_pose_c2w"])
                elif alter_pose_key is not None:
                    c2w_opt = np.array(traj_frame[alter_pose_key])
                else:
                    # This frame have no optimized pose and alternative pose is not enabled 
                    # we will need to skip this frame
                    if debug_prints:
                        print(f"Warning: skipping frames {traj_frame['file_name']} as it has no optimized pose and alternative measuered pose is not enabled")
                    continue

                c2w_measured = np.array(traj_frame["measured_pose_c2w"])

                c2w = np.eye(4)

                if traj_config["c2w_rot_optimized"]:
                    c2w[:3, :3] = c2w_opt[:3, :3]
                else:
                    c2w[:3, :3] = c2w_measured[:3, :3]

                if traj_config["c2w_trans_optimized"]:
                    c2w[:3, 3] = c2w_opt[:3, 3]
                else:
                    c2w[:3, 3] = c2w_measured[:3, 3]

                frame = dict(cam_intr)
                # change the camera type key name
                frame["camera_model"] = frame.pop("camera_type")
                src_file_name = Path(traj_frame["file_name"]).name
                src_file = scene_dir + f"/{traj_frame['file_name']}"
                dst_file_name = f"{images_dir}/{split_prefix}{tranjectory}_{src_file_name}"
                dst_file = dst_dir + "/" + dst_file_name

                frame["file_path"] = dst_file_name
                frame["transform_matrix"] = c2w.tolist()

                frames.append(frame)

                shutil.copyfile(src_file, dst_file)
        else:
            raise ValueError(f"The specified trajectory {tranjectory} is not found in the scene data at {scene_dir}")
    return frames


def create_ns_dataset_from_scene(scene_dir:str, scene_config:Dict,
                                 dst_dir:str, use_sparse_pc:bool=False,
                                 debug_prints:bool=False):
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

        # expand "all" key and update the scene config to set ids for the percentages
        scene_config_expanded = dict(scene_config)
        scene_config_expanded["train"] = _expand_all_key(dict(scene_config_expanded["train"]), scene_tranjectories)
        if "eval" in scene_config_expanded and scene_config_expanded["eval"] is not None:
            scene_config_expanded["eval"] = _expand_all_key(dict(scene_config_expanded["eval"]), scene_tranjectories)
        updated_scene_config = update_config_with_indices(scene_config_expanded, scene_tranjectories)
        # now for trajectory we will read and populate the frames
        frames += process_split(scene_dir=scene_dir, dst_dir=dst_dir,
                                scene_tranjectories=scene_tranjectories, split_config=updated_scene_config["train"], 
                                split_prefix=train_prefix, images_dir=images_dir,
                                debug_prints=debug_prints)

        # see if we have a defined eval as well
        if eval_prefix is not None:
            frames += process_split(scene_dir=scene_dir, dst_dir=dst_dir,
                                    scene_tranjectories=scene_tranjectories, split_config=updated_scene_config["eval"], 
                                    split_prefix=eval_prefix, images_dir=images_dir,
                                    debug_prints=debug_prints)
        
        if use_sparse_pc:
            shutil.copyfile(scene_dir + "/sparse_model.ply", dst_dir + "/sparse_pc.ply")
            transforms_data["ply_file_path"] = "sparse_pc.ply"

        transforms_data["frames"] = frames

        with open(transforms_file, 'w') as f:
            json.dump(transforms_data, f, indent=4)
            
    else:
        raise ValueError("No train configuration available")

