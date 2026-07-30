# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

from utils.geometry_utils import get_rotation_diff, get_3d_point_distance
from numpy.typing import NDArray
import numpy as np

def compute_scene_diameter(camera_centers):
    pts = np.asarray(camera_centers)

    max_dist = 0.0
    for i in range(len(pts)):
        dists = np.linalg.norm(pts[i+1:] - pts[i], axis=1)
        if len(dists):
            max_dist = max(max_dist, dists.max())

    return float(max_dist)

def compute_aabb_diagonal(camera_centers):
    pts = np.asarray(camera_centers)
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)

    scene_scale = np.linalg.norm(bbox_max - bbox_min).item()

    return scene_scale

def get_pose_t_distance(pose_a:NDArray,
                        pose_b:NDArray,
                        translation_scale:float=1.0, 
                        translation_weight:float=1.0):
    d_t = translation_weight * (get_3d_point_distance(pose_a[:3, 3].tolist(), pose_b[:3, 3].tolist()) / translation_scale)

    return np.abs(d_t).item()

def get_pose_r_distance(pose_a:NDArray,
                        pose_b:NDArray,
                        rotation_scale:float=1.0,
                        rotation_weight:float=1.0):
    d_r = rotation_weight * (get_rotation_diff(pose_a[:3, :3], pose_b[:3,:3]) / rotation_scale)

    return np.abs(d_r).item()


def get_pose_tr_distance(pose_a:NDArray, pose_b:NDArray, 
                                          translation_scale:float=1.0, rotation_scale:float=1,
                                          translation_weight:float=1.0, rotation_weight:float=1.0):
    
    d_t = get_pose_t_distance(pose_a=pose_a, 
                              pose_b=pose_b,
                              translation_scale=translation_scale, 
                              translation_weight=translation_weight)
    
    d_r = get_pose_r_distance(pose_a=pose_a, 
                              pose_b=pose_b,
                              rotation_scale=rotation_scale, 
                              rotation_weight=rotation_weight)

    d_tr = (0.5 * d_t) + (0.5 * d_r)

    return d_tr


def get_pose_k_nearest_neighbor(pose, 
                                ref_poses,
                                pose_distance_fn, 
                                k=1):
    pose_distances = []
    for ref_pose in ref_poses:
        pose_distances.append(pose_distance_fn(pose, ref_pose))
    
    nearest_ids = np.argsort(pose_distances)[:k]
    distance = sum([pose_distances[nearest_id] for nearest_id in nearest_ids]) / k
    if len(nearest_ids) > 0:
        nearest_pose = ref_poses[nearest_ids[0]]
    else:
        nearest_pose = None

    return nearest_pose, distance, nearest_ids

def get_traj_directed_chamfer_distance(traj_a_poses, 
                                       traj_b_poses, 
                                       pose_distance_fn,
                                       k_neighbor_size=1):
    traj_a_distances = []
    traj_a_nearest_pose = []
    
    #TODO: This is brute force need to be changed to something smarted
    for pose_a in traj_a_poses:
        nearest_pose, distance, _ = get_pose_k_nearest_neighbor(pose_a, 
                                                                traj_b_poses, 
                                                                pose_distance_fn=pose_distance_fn,
                                                                k=k_neighbor_size)
        traj_a_distances.append(distance)
        traj_a_nearest_pose.append(nearest_pose)
    
    traj_a_to_b_distance = np.mean(traj_a_distances).item()

    return traj_a_to_b_distance


def get_traj_symmetric_chamfer_distance(traj_a_poses, 
                                        traj_b_poses, 
                                        pose_distance_fn,
                                        k_neighbor_size=1):
    
    traj_a_to_b_distance = get_traj_directed_chamfer_distance(traj_a_poses=traj_a_poses,
                                                                     traj_b_poses=traj_b_poses,
                                                                     pose_distance_fn=pose_distance_fn,
                                                                     k_neighbor_size=k_neighbor_size)
    
    traj_b_to_a_distance = get_traj_directed_chamfer_distance(traj_a_poses=traj_b_poses,
                                                                     traj_b_poses=traj_a_poses,
                                                                     pose_distance_fn=pose_distance_fn,
                                                                     k_neighbor_size=k_neighbor_size)

    return (0.5* traj_a_to_b_distance) + (0.5* traj_b_to_a_distance)
