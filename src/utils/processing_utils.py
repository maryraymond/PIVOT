import json
import cv2
import numpy as np
from typing import List
from numpy.typing import NDArray

from data_processing.core.metadata_utils import read_metadata_exiftool
from utils.metrics import get_traj_directed_chamfer_distance



def get_cameras_data(cal_file:str=None, sample_image_file:str=None, camera_id=1)->List:
    cameras_data = {}

    if cal_file is not None:
            with open(cal_file, 'r') as f:
                cal_data = json.load(f)
                camera_type =  cal_data["camera_type"]
                H = cal_data["h"]
                W = cal_data["w"]
                fx = cal_data["fx"]
                fy = cal_data["fy"]
                cx = cal_data["cx"]
                cy = cal_data["cy"]
                k1 = cal_data["k1"]
                k2 = cal_data["k2"]
                k3 = cal_data.get("k3", 0)
                p1 = cal_data.get("p1", 0)
                p2 = cal_data.get("p2", 0)
    elif sample_image_file is not None:
            sample_image = cv2.imread(sample_image_file)
            H, W = sample_image.shape[:2]
            sample_exif_data = read_metadata_exiftool(sample_image_file)

            # we will estimate the intrinsic from the available metadata
            fov_deg = sample_exif_data["Composite:FOV"]

            camera_type = "OPENCV"
            
            fx = (W / 2) / np.tan(np.deg2rad(fov_deg / 2))
            fy = fx
            cx = W / 2
            cy = H / 2
            # assume no distortion
            k1 = k2 = k3 = p1 = p2 = 0
    else:
        raise ValueError("Please provide either a calibration file of sample image to read the camera intrinsics")
    
    cameras_data[camera_id] = {"camera_type":camera_type, "h":H, "w":W, "fl_x":fx, "fl_y":fy, "cx": cx, "cy": cy, "k1":k1, "k2": k2, "k3":k3, "p1":p1, "p2":p2}

    return cameras_data

def get_poses_from_data(images_data:List)->List[NDArray]:

    """ The input images data should be in the drone formate i.e: with c2w poses"""
    # check if we have c2w poses or w2c quats
    if len(images_data) > 0:
        if "pose_c2w" not in images_data[0]:
            raise ValueError("The data provided does not have c2w pose information")
        poses = []

        for image in images_data:
            poses.append(np.array(image["pose_c2w"]))
    return poses

def get_images_from_data(images_data:List, image_src_dir)->List[NDArray]:
    images =  []
    for image_data in images_data:
        images.append(cv2.imread(f"{image_src_dir}/{image_data['file_name']}"))
    
    return images

def get_traj_frames_data(scene_traj_data:List, trajectory_name:str, 
                         cam_intrinsics_type:str="camera_intrinsic_colmap", 
                         c2w_pose_type:str="colmap_pose_c2w"):
    traj_data = scene_traj_data[trajectory_name]
    cam_intrinsics = traj_data[cam_intrinsics_type]

    frames = traj_data["frames"]

    loaded_frames = []

    for frame in frames:
        if c2w_pose_type not in frame:
            continue

        loaded_frame = {"file_name":frame["file_name"],
                        "pose_c2w":frame[c2w_pose_type],
                        "intrinsics":cam_intrinsics}
        loaded_frames.append(loaded_frame)

    return loaded_frames


def get_trajectories_diff(scene_data, traj_a_name, traj_b_name, 
                          pose_distance_fn, 
                          k_neighbor_size=1,
                          pose_type="colmap_pose_c2w"):
    
    traj_a = get_traj_frames_data(scene_traj_data=scene_data["trajectories"], trajectory_name=traj_a_name,
                                cam_intrinsics_type="camera_intrinsic_colmap", c2w_pose_type=pose_type)

    traj_b = get_traj_frames_data(scene_traj_data=scene_data["trajectories"], trajectory_name=traj_b_name,
                                    cam_intrinsics_type="camera_intrinsic_colmap", c2w_pose_type=pose_type)
    
    traj_a_poses = [np.array(frame["pose_c2w"]) for frame in traj_a]
    traj_b_poses = [np.array(frame["pose_c2w"]) for frame in traj_b]

    traj_mean_dis = get_traj_directed_chamfer_distance(traj_a_poses=traj_a_poses, 
                                                       traj_b_poses=traj_b_poses,
                                                       pose_distance_fn=pose_distance_fn,
                                                       k_neighbor_size=k_neighbor_size)
            
    return traj_mean_dis