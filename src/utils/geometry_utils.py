# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
import math
from scipy.spatial.transform import Rotation as Rot


def get_ned_rotation_from_yaw_pitch_roll(yaw_degree:float, pitch_degree:float, roll_degree:float)->NDArray:

    yaw = math.radians(yaw_degree)
    pitch = math.radians(pitch_degree)
    roll = math.radians(roll_degree)

    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0],
                   [math.sin(yaw), math.cos(yaw), 0], 
                   [0,              0,            1]])
    
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                   [0,              1,        0       ],
                   [-math.sin(pitch), 0, math.cos(pitch)]])
    

    Rx = np.array([[1,          0,       0            ],
                   [0, math.cos(roll), -math.sin(roll)],
                   [0, math.sin(roll), math.cos(roll)]])
    

    R = Rz @ Ry @ Rx

    return R

def R_to_quat(R: np.ndarray):
    r = Rot.from_matrix(R)
    qx, qy, qz, qw = r.as_quat()  # SciPy returns (x, y, z, w)
    return (float(qx), float(qy), float(qz), float(qw))

def quat_to_R(quat:Tuple[float])->NDArray:
    quat = list(quat)

    r = Rot.from_quat(quat)

    R = r.as_matrix()

    return R

def homo_pose_to_quat(pose_c2w: NDArray)-> Tuple[Tuple, Tuple]:

    R = pose_c2w[:3, :3]
    T = pose_c2w[:3, 3]
    quat = R_to_quat(R)
    
    return(quat, T.tolist())

def quat_to_homo_pose(quat:Tuple[float], T:Tuple[float]):
    R = quat_to_R(quat)
    w2c_pose = np.eye(4)
    w2c_pose[:3, :3] = R

    w2c_pose[:3, 3] = np.array(T)

    return w2c_pose

def get_yaw_pitch_roll_from_ned_rotation(R_mat):
    rot = Rot.from_matrix(R_mat)
    yaw, pitch, roll = rot.as_euler('ZYX', degrees=True)

    return yaw.item(), pitch.item(), roll.item()

def get_euler_diff(angle1, angle2):
    diff = ((angle2 - angle1 + 180) % 360) - 180
    return diff

def get_rotation_euler_diff(r1, r2):
    yaw1, pitch1, roll1 = get_yaw_pitch_roll_from_ned_rotation(r1)
    yaw2, pitch2, roll2 = get_yaw_pitch_roll_from_ned_rotation(r2)

    yaw_diff = get_euler_diff(yaw2, yaw1)
    pitch_diff = get_euler_diff(pitch2, pitch1)
    roll_diff = get_euler_diff(roll2, roll1)

    return yaw_diff, pitch_diff, roll_diff

def get_rotation_diff(r1, r2):
    r_rel = r1.T @ r2
    cos_theta = (np.trace(r_rel) - 1) / 2
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.degrees(np.arccos(cos_theta))

    return theta

def get_3d_point_distance(p1, p2):
    # first we get the Abslout XYZ camera center value to measure the distance
    X1, Y1, Z1 = p1
    X2, Y2, Z2 = p2
    
    d = np.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)

    return d

def get_homogenous_matrix(R: NDArray, x: float, y: float, z: float) -> NDArray:
    pose = np.eye(4)
    pose[:3, :3] = R
    pose[0, 3] = x
    pose[1, 3] = y
    pose[2, 3] = z
    return pose

def get_3d_point_xyz_diff(p1, p2):
    X1, Y1, Z1 = p1
    X2, Y2, Z2 = p2

    return np.abs(X2 - X1), np.abs(Y2 - Y1), np.abs(Z2 - Z1)
