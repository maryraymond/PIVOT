# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict, Any
import sqlite3
import re
import subprocess
from pathlib import Path
from typing import Any

from utils.geometry_utils import quat_to_homo_pose, homo_pose_to_quat

def convert_opengl_cam_to_colmap(pose_c2w:NDArray)->NDArray:
    # colmap (same as opencv) camera coordinate system = camera looks in Z +ve direction, Y +ve down and X +ve right
    # OpenGL camera coordinate system = camera looks in Z -vee direction, Y +ve is up and X +ve right
    # here we are changing the camera bases using a c2w pose so we multiply from the right and only rotation
    R_opengl_to_colmap = np.array([[1, 0, 0],
                                   [0, -1, 0],
                                   [0, 0, -1]])
    colmap_c2w = pose_c2w.copy()
    colmap_c2w[:3, :3] = colmap_c2w[:3, :3] @ R_opengl_to_colmap

    return colmap_c2w

def convert_colmap_cam_to_opengl(pose_c2w:NDArray)->NDArray:
    # colmap (same as opencv)camera coordinate system = camera looks in Z +ve direction, Y +ve down and X +ve right
    # OpenGL camera coordinate system = camera looks in Z -vee direction, Y +ve is up and X +ve right
    # here we are changing the camera bases using a c2w pose so we multiply from the right and only rotation
    R_colmap_to_opengl = np.array([[1, 0, 0],
                                   [0, -1, 0],
                                   [0, 0, -1]])
    opengl_c2w = pose_c2w.copy()
    opengl_c2w[:3, :3] = opengl_c2w[:3, :3] @ R_colmap_to_opengl

    return opengl_c2w

def convert_colmap_world_to_ned(pose_c2w:NDArray)->NDArray:
    # colmap world corrdinate coordinate system align with the first camera = camera looks in Z +ve direction, Y +ve down and X +ve right
    # NED corrdinate system =  X +ve aligns with Geographic north, Y +ve east, and Z +ve down toward Earth
    # here we are changing the world bases using a c2w so we multiply from the left with full homogenous matrix
    homogenous_colmap_to_ned = np.array([[0, 0, 1, 0],
                                         [1, 0, 0, 0],
                                         [0, 1, 0, 0],
                                         [0, 0, 0, 1]])

    ned_c2w = homogenous_colmap_to_ned @ pose_c2w

    return ned_c2w

def convert_ned_world_to_colmap(pose_c2w:NDArray)->NDArray:
    # colmap  world corrdinate coordinate system align with the first camera = camera looks in Z +ve direction, Y +ve down and X +ve right
    # NED corrdinate system =  X +ve aligns with Geographic north, Y +ve east, and Z +ve down toward Earth
    # here we are changing the world bases using a c2w so we multiply from the left with full homogenous matrix
    homogenous_ned_to_colmap = np.array([[0, 1, 0, 0],
                                         [0, 0, 1, 0],
                                         [1, 0, 0, 0],
                                         [0, 0, 0, 1]])

    ned_c2w = homogenous_ned_to_colmap @ pose_c2w

    return ned_c2w

def convert_colmap_pose_to_drone(quat:Tuple, translation:Tuple, transform_world_coord=True)->NDArray:
    # Drone data set uses NED as the world corrdinate and openGL as the camera coordinate
    # Colmap uses openCV as the camera coordinate and the first camera pose as the world cooridnate
    # Colmap defines the camera pose as Quaternian rotation and translation, while Drone dataset uses
    # Homogenouse 4x4 matrix
    # Colmap uses W2C while the drone dataset uses C2W pose
    
    # first we convert the quaternian roatation and translation into homogenous pose (w2c)
    pose_w2c = quat_to_homo_pose(quat, translation)

    #Then we get the c2w
    pose_c2w = np.linalg.inv(pose_w2c)

    #Then we change the camera bases from colmap (opencv) to opengl
    pose_c2w = convert_colmap_cam_to_opengl(pose_c2w)

    # If colmap was uisng pose prior then the world corrdinate is 
    # already in NED if that was used for the input
    if transform_world_coord:
        #We finally change the world bases from colmap to NED
        pose_c2w = convert_colmap_world_to_ned(pose_c2w)

    return pose_c2w

def convert_drone_pose_to_colmap(homo_c2w:NDArray)->Tuple[Tuple]:
    # Drone data set uses NED as the world corrdinate and openGL as the camera coordinate
    # Colmap uses openCV as the camera coordinate and the first camera pose as the world cooridnate
    # Colmap defines the camera pose as Quaternian rotation and translation, while Drone dataset uses
    # Homogenouse 4x4 matrix
    # Colmap uses W2C while the drone dataset uses C2W pose

    # change the camera bases from opengl to colmap (opencv)
    pose_c2w = convert_opengl_cam_to_colmap(homo_c2w)

    # Change world bases from NED to colmap
    pose_c2w = convert_ned_world_to_colmap(pose_c2w)

    # change to w2c
    pose_w2c = np.linalg.inv(pose_c2w)

    # now get the colmap format of quat and trans
    quat, trans = homo_pose_to_quat(pose_w2c)

    return quat, trans

def convert_data_to_colmap(images_data:List)->List:

    images_data_colmap = []
    for image_data in images_data:
        quat, T = convert_drone_pose_to_colmap(np.array(image_data["pose_c2w"]))

        images_data_colmap.append({"file_name": image_data["file_name"],
                                   "camera_id": image_data["camera_id"],
                                   "quat": quat,
                                   "translation": T})
    
    return images_data_colmap


def convert_data_from_colmap(images_data_colmap:List, transform_world_coord=True)->List:
    images_data = []

    for image_data_colmap in images_data_colmap:
        pose_c2w = convert_colmap_pose_to_drone(image_data_colmap["quat"], 
                                                image_data_colmap["translation"], 
                                                transform_world_coord=transform_world_coord)

        images_data.append({"file_name": image_data_colmap["file_name"],
                            "camera_id": image_data_colmap["camera_id"],
                            "pose_c2w": pose_c2w.tolist()})
    
    return images_data


def write_colmap_cameras_txt(cameras_data:Dict, camera_txt:str)->None:
    colmap_cameras_header =  "# Camera list with one line of data per camera: \n\
# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[fx fy cx cy k1 k2 p1 p2] \n\
# Number of cameras: \n\
"
    
    cameras_data_colmap = []
    for camera_id, camera_data in cameras_data.items():

        #Params: fx fy cx cy k1 k2 p1 p2
        camera_data_colmap = f"{camera_id} {camera_data['camera_type']} {camera_data['w']} {camera_data['h']} \
{camera_data['fl_x']} {camera_data['fl_y']} {camera_data['cx']} {camera_data['cy']} \
{camera_data['k1']} {camera_data['k2']} {camera_data['p1']} {camera_data['p2']} \n"

        cameras_data_colmap.append(camera_data_colmap)

    camera_txt_data = colmap_cameras_header + "".join(cameras_data_colmap)
    with open(camera_txt, "w") as wf:
        wf.write(camera_txt_data)


def read_db_id_map(colmap_database:str)->Dict:
    conn = sqlite3.connect(colmap_database)
    cursor = conn.cursor()

    cursor.execute("SELECT image_id, name FROM images;")
    rows = cursor.fetchall()

    id_map = {name: image_id for image_id, name in rows}

    return id_map


def write_colmap_images_txt(images_data:List, colmap_database:str, images_txt:str):
    colmap_images_header = "# Image list with two lines of data per image: \n\
#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME \n\
#   POINTS2D[] as (X, Y, POINT3D_ID) \n\
# Number of images: \n\
"

    images_data_colmap = []
    id_map = read_db_id_map(colmap_database)

    if "quat" not in images_data[0]:
        raise ValueError("The images data need to be provided int colmap formate with quad and T")
    
    for image_data in images_data:

        image_name = image_data["file_name"]
        quat = image_data["quat"]
        T = image_data["translation"]

        #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME 
        image_data_colmap = f"{id_map[image_name]} {quat[3]} {quat[0]} {quat[1]} {quat[2]} {T[0]} {T[1]} {T[2]} 1 {image_name} \n\n"

        images_data_colmap.append(image_data_colmap)

    with open(images_txt, "w") as wf:
            
            wf.write(colmap_images_header + "".join(images_data_colmap))


def get_colmap_number_points(points_txt_file:str)->int:
    with open(points_txt_file, 'r') as f:
        # skip the header
        for _ in range(2):
            f.readline()
        target_line = f.readline()
    stats = target_line.split(",")
    points_number = int(stats[0].split("# Number of points:")[-1])

    return points_number

def get_colmap_registered_images_stat(images_txt_file:str)-> int:
    with open(images_txt_file, 'r') as f:
        # skip the header
        for _ in range(3):
            f.readline()
        target_line = f.readline()
        stats = target_line.split(",")
        regestered_frames = int(stats[0].split("# Number of images:")[-1])
        mean_obser_per_image = float(stats[1].split('mean observations per image:')[-1])

        return regestered_frames, mean_obser_per_image


def read_colmap_image_txt(images_txt_file:str)-> List:
    images_data = []
    with open(images_txt_file, 'r') as f:
        # skip the header
        for _ in range(4):
            f.readline()
        while True:
            line = f.readline()
            if not line:
                break
            image_data = line.split(" ")

            qw_index = 1
            qx_index = 2
            qy_index = 3
            qz_index = 4
            tx_index = 5
            ty_index = 6
            tz_index = 7
            camera_id_index =  8
            file_name_index = 9

            qw = image_data[qw_index]
            qx = image_data[qx_index]
            qy = image_data[qy_index]
            qz = image_data[qz_index]
            tx = image_data[tx_index]
            ty = image_data[ty_index]
            tz = image_data[tz_index]
            camera_id = image_data[camera_id_index]
            file_name  = image_data[file_name_index]

            images_data.append({"file_name":file_name.strip(),
                                "camera_id":camera_id,
                                "quat":(float(qx), float(qy), float(qz), float(qw)),
                                "translation":(float(tx), float(ty), float(tz))})
            
            # skipe the next line
            skip = f.readline()
    return images_data


def read_colmap_cameras_txt(camera_txt_file:str)->List:
    with open(camera_txt_file, 'r') as f:
            cameras_data = {}
            # skip the header
            for _ in range(3):
                f.readline()
            while True:
                line = f.readline()
                if not line:
                    break
                camera_data = line.split(" ")

                camera_id_index = 0
                camera_type_index = 1
                width_index = 2
                height_index = 3
                fx_index = 4
                fy_index = 5
                cx_index = 6
                cy_index = 7
                k1_index = 8
                k2_index = 9
                p1_index = 10
                p2_index = 11

                cameras_data[camera_data[camera_id_index].strip()] ={"camera_type": camera_data[camera_type_index].strip(),
                                                                     "w": camera_data[width_index].strip(),
                                                                     "h": camera_data[height_index].strip(),
                                                                     "fl_x": camera_data[fx_index].strip(),
                                                                     "fl_y": camera_data[fy_index].strip(),
                                                                     "cx": camera_data[cx_index].strip(),
                                                                     "cy": camera_data[cy_index].strip(),
                                                                     "k1": camera_data[k1_index].strip(),
                                                                     "k2": camera_data[k2_index].strip(),
                                                                     "k3": 0
                                                                     }
                if cameras_data[camera_data[camera_id_index].strip()]["camera_type"] == "OPENCV":
                    cameras_data[camera_data[camera_id_index].strip()]["p1"] = camera_data[p1_index].strip()
                    cameras_data[camera_data[camera_id_index].strip()]["p2"] = camera_data[p2_index].strip()
                elif cameras_data[camera_data[camera_id_index].strip()]["camera_type"] == "OPENCV_FISHEYE":
                    cameras_data[camera_data[camera_id_index].strip()]["k3"] = camera_data[p1_index].strip()
                    cameras_data[camera_data[camera_id_index].strip()]["k4"] = camera_data[p2_index].strip()
                else:
                    raise ValueError(f"Unsupported camera type {cameras_data[camera_data[camera_id_index].strip()]['camera_type']}")

    return cameras_data



def run_colmap_model_analyzer(
    sparse_model_path: str | Path,
    colmap_exe: str = "colmap",
) -> dict[str, Any]:
    """
    Run `colmap model_analyzer` and parse its output into a dictionary.

    Args:
        sparse_model_path:
            Path to COLMAP sparse model directory, e.g. ".../sparse/0".
        colmap_exe:
            COLMAP executable name or full path.

    Returns:
        Dictionary containing parsed model statistics.
    """
    sparse_model_path = Path(sparse_model_path)

    if not sparse_model_path.exists():
        raise FileNotFoundError(f"Sparse model path does not exist: {sparse_model_path}")

    cmd = [
        colmap_exe,
        "model_analyzer",
        "--path",
        str(sparse_model_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    output = result.stdout

    if result.returncode != 0:
        raise RuntimeError(
            f"COLMAP model_analyzer failed with exit code {result.returncode}\n"
            f"Command: {' '.join(cmd)}\n"
            f"Output:\n{output}"
        )

    stats: dict[str, Any] = {}

    patterns = {
        "rigs": r"Rigs:\s+(\d+)",
        "cameras": r"Cameras:\s+(\d+)",
        "frames": r"Frames:\s+(\d+)",
        "registered_frames": r"Registered frames:\s+(\d+)",
        "images": r"Images:\s+(\d+)",
        "registered_images": r"Registered images:\s+(\d+)",
        "points3D": r"Points:\s+(\d+)",
        "observations": r"Observations:\s+(\d+)",
        "mean_track_length": r"Mean track length:\s+([0-9.eE+-]+)",
        "mean_observations_per_image": r"Mean observations per image:\s+([0-9.eE+-]+)",
        "mean_reprojection_error_px": r"Mean reprojection error:\s+([0-9.eE+-]+)px",
    }

    int_keys = {
        "rigs",
        "cameras",
        "frames",
        "registered_frames",
        "images",
        "registered_images",
        "points3D",
        "observations",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, output)

        if match is None:
            stats[key] = None
            continue

        value = match.group(1)

        if key in int_keys:
            stats[key] = int(value)
        else:
            stats[key] = float(value)


    return stats

if __name__ == "__main__":
    pass