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


if __name__ == "__main__":
    pass