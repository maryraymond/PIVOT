import numpy as np
import cv2
from numpy.typing import NDArray
from typing import Tuple, Optional, List, Dict
import json
import subprocess
from pathlib import Path
import math
import shutil
import os

from geometry import get_ned_rotation_from_yaw_pitch_roll

def read_metadata_exiftool(img_path:str) -> dict:
    cmd = ["exiftool", "-j", "-G1", "-a", "-n", img_path]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)[0]


def get_gps_value(exif_data, absolute_altitude=True):
    latituide = float(exif_data["XMP-drone-dji:GPSLatitude"])
    longitude = float(exif_data["XMP-drone-dji:GPSLongitude"])
    if absolute_altitude:
        altitude = float(exif_data["XMP-drone-dji:AbsoluteAltitude"])
    else:
        altitude = float(exif_data["XMP-drone-dji:RelativeAltitude"])
    
    return latituide, longitude, altitude

def get_flight_rotation(exif_data):
    # need to verify if we should multiply by -ve mathimaticaly +ve is look right)
    # flight_yaw = -1 * float(exif_data["XMP-drone-dji:FlightYawDegree"])
    flight_yaw = float(exif_data["XMP-drone-dji:FlightYawDegree"])
    flight_Pitch = float(exif_data["XMP-drone-dji:FlightPitchDegree"])
    flight_roll = float(exif_data["XMP-drone-dji:FlightRollDegree"])

    R = get_ned_rotation_from_yaw_pitch_roll(flight_yaw, flight_Pitch, flight_roll)

    return R, (flight_yaw, flight_Pitch, flight_roll)

def get_gimbal_rotation(exif_data):
    # need to verify if we should multiply by -ve mathimaticaly +ve is look right)
    # gimbal_yaw = -1 * float(exif_data["XMP-drone-dji:GimbalYawDegree"])
    gimbal_yaw = float(exif_data["XMP-drone-dji:GimbalYawDegree"])
    gimbal_Pitch = float(exif_data["XMP-drone-dji:GimbalPitchDegree"])
    gimbal_roll = float(exif_data["XMP-drone-dji:GimbalRollDegree"])

    R = get_ned_rotation_from_yaw_pitch_roll(gimbal_yaw, gimbal_Pitch, gimbal_roll)

    return R, (gimbal_yaw, gimbal_Pitch, gimbal_roll)


def ned_from_gps(lat_deg, lon_deg, alt_m, lat0_deg=0, lon0_deg=0, alt0_m=0.0, use_absolute_altitude=True):
    
    R_EARTH=6378137.0 # meters (WGS84)
    # NED -> North east down
    # Convert degrees to radians
    lat = math.radians(lat_deg)
    lon = math.radians(lon_deg)
    lat0 = math.radians(lat0_deg)
    lon0 = math.radians(lon0_deg)

    dlat = lat - lat0
    dlon = lon - lon0

    x_north = R_EARTH * dlat
    
    y_east  = R_EARTH * math.cos(lat0) * dlon

    if not use_absolute_altitude:
        # DJI "world" is NED: z is Down
        # RelativeAltitude is typically "Up from takeoff", so convert to Down:
        z_down = -(alt_m - alt0_m)
    else: 
        # Abslout is from sea level and increases up where Ned z+ve is down so
        # we need to multiply by -ve
        z_down = -(alt_m - alt0_m)

    return x_north, y_east, z_down

def ecef_from_gps(lat_deg, long_deg, alt_m):

    a = 6378137.0
    e2 = 6.69437999014e-3 

    lat = math.radians(lat_deg)
    long = math.radians(long_deg)

    N = a / math.sqrt(1 - (e2 * math.sin(lat)**2))

    X = (N + alt_m) * math.cos(lat) * math.cos(long)
    Y = (N + alt_m) * math.cos(lat) * math.sin(long)
    Z = (N * (1 - e2) + alt_m) * math.sin(lat)

    return X, Y, Z

def ned_from_ecef(point, X0, Y0, Z0, lat0_deg, long0_deg):

    lat0 = math.radians(lat0_deg)
    long0 = math.radians(long0_deg)
    
    R_ecef2ned = np.array([[-math.sin(lat0)*math.cos(long0),  -math.sin(lat0)*math.sin(long0),   math.cos(lat0)],
                           [-math.sin(long0),                  math.cos(long0),                  0                ],
                           [-math.cos(lat0)* math.cos(long0),  -math.cos(lat0)*math.sin(long0),   -math.sin(lat0) ]])
    
    points_ref = np.array(point)
    points_ref -= np.array([X0, Y0, Z0])

    ned_point = points_ref @ R_ecef2ned.T

    return ned_point[0], ned_point[1], ned_point[2]

def get_homogenous_matrix(R: NDArray, x:float, y:float, z:float) ->NDArray:
    eye = np.eye(4)
    eye[:3, :3] = R
    eye[:3, 3] = np.array([x, y, z]).T

    return eye

def convert_ned_cam_to_opengl(drone_pose:NDArray) -> NDArray:

    # assume camera coordinate system is openGL (so camera facing z -ve)
    c2gimble = np.array([[0, 0, -1],
                        [1, 0, 0],
                        [0, -1, 0]])
    
    pose = drone_pose.copy()
    pose[:3, :3] = pose[:3, :3] @  c2gimble

    return pose

def frame_num(image_name:str):
    if image_name.split("_")[-1] == "D.JPG":
       frame_num = int(image_name.split("_")[-2])
    else:
        frame_num = int(image_name.split("_")[-1].split(".")[0])
    return frame_num

def read_images_data_from_folder(data_path:str, camera_id:int=1, 
                                 sorting_func=frame_num, use_absloute_altitude=True)->List:

    images = sorted([file for file in Path(data_path).iterdir() if file.is_file() and ".JPG" in file.name], 
                    key=lambda x: sorting_func(x.name))

    subfolders_available  = False
    if len(images) == 0:
        print(f"No images found at {data_path} will check for sub-folders")

        subfolders = sorted(folder for folder in Path(data_path).iterdir() if folder.is_dir())

        images = []
        for subfolder in subfolders:
            images += sorted([file for file in Path(subfolder).iterdir() if file.is_file() and ".JPG" in file.name], 
                            key=lambda x: sorting_func(x.name))
            
        if  len(images) == 0:
            raise ValueError(f"No images found in folder {data_path} or subfolder {subfolders}")
        else:
            subfolders_available = True

    
    images_data = []
    ref_exif_data = read_metadata_exiftool(images[0])
    lat_0, long_0, alt_0 = get_gps_value(ref_exif_data)

    # image_0 = cv2.imread(images[0])

    # H, W = image_0.shape[:2]

    for image in images:
        exif_data = read_metadata_exiftool(image)
        lat, long, alt = get_gps_value(exif_data, absolute_altitude=use_absloute_altitude)
        x, y, z = ned_from_gps(lat, long, alt, lat_0, long_0, alt_0, use_absolute_altitude=use_absloute_altitude)

        Rg2f, _ = get_gimbal_rotation(exif_data)

        pose_g2w = get_homogenous_matrix(Rg2f, x, y, z)
        pose_c2w = convert_ned_cam_to_opengl(pose_g2w)

        file_name = f"{image.parts[-2]}/{image.parts[-1]}" if subfolders_available else Path(image).name
        images_data.append({"camera_id":camera_id,
                            "file_name": file_name,
                            "pose_c2w":pose_c2w.tolist()})
    
    return images_data


def get_cameras_data(cal_file:str=None, sample_image_file:str=None)->List:
    cameras_data = []

    if cal_file is not None:
            with open(cal_file, 'r') as f:
                cal_data = json.load(f)
                camera_type =  cal_data["camera_type"]
                H = cal_data["H"]
                W = cal_data["W"]
                fx = cal_data["fx"]
                fy = cal_data["fy"]
                cx = cal_data["cx"]
                cy = cal_data["cy"]
                k1 = cal_data["k1"]
                k2 = cal_data["k2"]
                k3 = cal_data["k3"]
                p1 = cal_data["p1"]
                p2 = cal_data["p2"]
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
    
    cameras_data.append({"camera_id":1, "camera_type":camera_type, "H":H, "W":W, "fl_x":fx, "fl_y":fy, "cx": cx, "cy": cy, "k1":k1, "k2": k2, "k3":k3, "p1":p1, "p2":p2})

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

def create_nerfstudio_dataset(images_data, camera_data, dataset_dir, src_images_dir):

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

    transforms = {"camera_model": camera_data["camera_type"],
                  "w": camera_data["W"],         "h": camera_data["H"],
                  "fl_x": camera_data["fl_x"],   "fl_y": camera_data["fl_y"],
                  "cx": camera_data["cx"],       "cy": camera_data["cy"],
                  "k1": camera_data["k1"],       "k2": camera_data["k2"],    "k3":camera_data["k3"],
                  "p1": camera_data["p1"],       "p2": camera_data["p2"]}

    frames = []
    for image in images_data:

        image_name = image["file_name"]
        dst_image = f"{dataset_images_dir}/{image_name}"
        dst_image_path = f"{Path(dataset_images_dir).name}/{image_name}"

        shutil.copyfile(f"{src_images_dir}/{image_name}", dst_image)


        frame_data = {"file_path": dst_image_path,
                    "transform_matrix":  image["pose_c2w"]}

        frames.append(frame_data)

    transforms["frames"] = frames

    transforms_file = f"{dataset_dir}/transforms.json"

    with open (transforms_file, 'w') as f:
        json.dump(transforms, f, indent=4)


def get_images_from_data(images_data:List, image_src_dir)->List[NDArray]:
    images =  []
    for image_data in images_data:
        images.append(cv2.imread(f"{image_src_dir}/{image_data['file_name']}"))
    
    return images


if __name__ == "__main__":
    pass