from data_processing.utils.geometry_utils import get_rotation_diff, get_3d_point_distance
from numpy.typing import NDArray
import numpy as np

def get_pose_weight_rot_distance(pose_a:NDArray, pose_b:NDArray, rot_w:float=0.1):
    d_t = get_3d_point_distance(pose_a[:3, 3].tolist(), pose_b[:3, 3].tolist())
    d_r = get_rotation_diff(pose_a[:3, :3], pose_b[:3,:3])

    d = np.abs(d_t) + (rot_w * np.abs(d_r))

    return d

def get_pose_k_nearest_neighbor(pose, ref_poses, k=1,
                                pose_distance_fn=get_pose_weight_rot_distance):
    pose_distances = []
    for ref_pose in ref_poses:
        pose_distances.append(pose_distance_fn(pose, ref_pose))
    
    nearest_ids = np.argsort(pose_distances)[:k]
    distance = sum([pose_distances[nearest_id] for nearest_id in nearest_ids]) / k
    nearest_pose = ref_poses[nearest_ids[0]]

    return nearest_pose, distance, nearest_ids

def get_pose_chamfer_distance_directed_a_to_b(traj_a_poses,
                                             traj_b_poses,
                                             pose_distance_fn=get_pose_weight_rot_distance,
                                             k_neighbor_size=1):
    if len(traj_a_poses) == 0 or len(traj_b_poses) == 0:
        return float("nan")

    traj_a_distances = []

    #TODO: This is brute force need to be changed to something smarted
    for pose_a in traj_a_poses:
        _, distance, _ = get_pose_k_nearest_neighbor(pose_a,
                                                     traj_b_poses,
                                                     pose_distance_fn=pose_distance_fn,
                                                     k=k_neighbor_size)
        traj_a_distances.append(distance)

    traj_a_to_b_distance = np.mean(traj_a_distances).item()

    return traj_a_to_b_distance


def get_pose_chamfer_distance_symmetric(traj_a_poses, 
                                     traj_b_poses, 
                                     pose_distance_fn=get_pose_weight_rot_distance,
                                     k_neighbor_size=1):
    
    traj_a_to_b_distance = get_pose_chamfer_distance_directed_a_to_b(traj_a_poses=traj_a_poses,
                                                                     traj_b_poses=traj_b_poses,
                                                                     pose_distance_fn=pose_distance_fn,
                                                                     k_neighbor_size=k_neighbor_size)
    
    traj_b_to_a_distance = get_pose_chamfer_distance_directed_a_to_b(traj_a_poses=traj_b_poses,
                                                                     traj_b_poses=traj_a_poses,
                                                                     pose_distance_fn=pose_distance_fn,
                                                                     k_neighbor_size=k_neighbor_size)

    return (traj_a_to_b_distance + traj_b_to_a_distance) / 2
