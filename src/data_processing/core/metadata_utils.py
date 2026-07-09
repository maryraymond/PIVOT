# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

import subprocess
import json
from typing import Dict

from data_processing.core.camera_metadata_abc import CameraTagsMap, GpsTagsMap
from utils.geometry_utils import get_ned_rotation_from_yaw_pitch_roll, get_euler_diff, get_3d_point_distance
from utils.gps_utils import ecef_from_gps

def read_metadata_exiftool(img_path:str) -> Dict:
    cmd = ["exiftool", "-j", "-G1", "-a", "-n", img_path]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)[0]

def read_video_metadata_exiftool(video_path:str) -> dict:
    cmd = ["exiftool", "-ee", "-u", "-j", "-G3", "-a", "-n", video_path]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)[0]

def get_gps_value(metadata:Dict, tags_map:CameraTagsMap, absolute_altitude=True):
    latituide = float(metadata[tags_map.get_image_latitude_tag()])
    longitude = float(metadata[tags_map.get_image_longitude_tag()])
    if absolute_altitude:
        altitude = float(metadata[tags_map.get_image_abs_altitude_tag()])
    else:
        altitude = float(metadata[tags_map.get_image_rel_altitude_tag()])
    
    return latituide, longitude, altitude

def get_robot_rotation(metadata:Dict, tags_map:CameraTagsMap):
    flight_yaw = float(metadata[tags_map.get_robot_image_yaw_tag()])
    flight_Pitch = float(metadata[tags_map.get_robot_image_pitch_tag()])
    flight_roll = float(metadata[tags_map.get_robot_image_roll_tag()])

    R = get_ned_rotation_from_yaw_pitch_roll(flight_yaw, flight_Pitch, flight_roll)

    return R, (flight_yaw, flight_Pitch, flight_roll)

def get_camera_rotation(metadata:Dict, tags_map:CameraTagsMap):
    gimbal_yaw = float(metadata[tags_map.get_camera_image_yaw_tag()])
    gimbal_Pitch = float(metadata[tags_map.get_camera_image_pitch_tag()])
    gimbal_roll = float(metadata[tags_map.get_camera_image_roll_tag()])

    R = get_ned_rotation_from_yaw_pitch_roll(gimbal_yaw, gimbal_Pitch, gimbal_roll)

    return R, (gimbal_yaw, gimbal_Pitch, gimbal_roll)

def write_camera_tags(metadata:Dict, image_file:str, tags_map:CameraTagsMap):
    cmd = ["exiftool", "-overwrite_original",
           f"-{tags_map.get_camera_image_yaw_tag()}={metadata[tags_map.get_camera_image_yaw_tag()]}",
           f"-{tags_map.get_camera_image_pitch_tag()}={metadata[tags_map.get_camera_image_pitch_tag()]}",
           f"-{tags_map.get_camera_image_roll_tag()}={metadata[tags_map.get_camera_image_roll_tag()]}",
           f"{image_file}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result	

def write_gps_tags(metadata, image_file, tags_map:CameraTagsMap):
    cmd = ["exiftool", "-overwrite_original",
           f"-{tags_map.get_image_latitude_tag()}={metadata[tags_map.get_image_latitude_tag()]}",
           f"-{tags_map.get_image_longitude_tag()}={metadata[tags_map.get_image_longitude_tag()]}",
           f"-{tags_map.get_image_rel_altitude_tag()}={metadata[tags_map.get_image_rel_altitude_tag()]}",
           f"-{tags_map.get_image_abs_altitude_tag()}={metadata[tags_map.get_image_abs_altitude_tag()]}",
           f"{image_file}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def write_robot_tags(metadata, image_file, tags_map:CameraTagsMap):
    cmd = ["exiftool", "-overwrite_original",
           f"-{tags_map.get_robot_image_yaw_tag()}={metadata[tags_map.get_robot_image_yaw_tag()]}",
           f"-{tags_map.get_robot_image_pitch_tag()}={metadata[tags_map.get_robot_image_pitch_tag()]}",
           f"-{tags_map.get_robot_image_roll_tag()}={metadata[tags_map.get_robot_image_roll_tag()]}",
           f"{image_file}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result

def write_standard_gps_tags(metadata, image_file, tags_map:CameraTagsMap, gps_tags_map:GpsTagsMap):
    latitudeRef = 'N' if metadata[tags_map.get_image_latitude_tag()] > 0 else 'S'
    longitudeRef = 'E' if metadata[tags_map.get_image_longitude_tag()] > 0 else 'W'
    altitudeRef = 0 if metadata[tags_map.get_image_abs_altitude_tag()] > 0 else 1

    cmd = ["exiftool", "-overwrite_original",
           f"-{gps_tags_map.get_ref_latitude_tag()}={latitudeRef}",
           f"-{gps_tags_map.get_abs_latitude_tag()}={abs(metadata[tags_map.get_image_latitude_tag()])}",
           f"-{gps_tags_map.get_ref_longitude_tag()}={longitudeRef}",
           f"-{gps_tags_map.get_abs_longitude_tag()}={abs(metadata[tags_map.get_image_longitude_tag()])}",
           f"-{gps_tags_map.get_ref_altitude_tag()}={altitudeRef}",
           f"-{gps_tags_map.get_abs_altitude_tag()}={abs(metadata[tags_map.get_image_abs_altitude_tag()])}",
           f"{image_file}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result
		
def get_distance_between_camera_centers(camera1_metadata:Dict, camera2_metadata:Dict, tags_map:CameraTagsMap):
    # first we get the Abslout XYZ camera center value to measure the distance
    X1, Y1, Z1 = ecef_from_gps(lat_deg=camera1_metadata[tags_map.get_image_latitude_tag()],
                               long_deg=camera1_metadata[tags_map.get_image_longitude_tag()],
                               alt_m=camera1_metadata[tags_map.get_image_abs_altitude_tag()])
    
    X2, Y2, Z2 = ecef_from_gps(lat_deg=camera2_metadata[tags_map.get_image_latitude_tag()],
                               long_deg=camera2_metadata[tags_map.get_image_longitude_tag()],
                               alt_m=camera2_metadata[tags_map.get_image_abs_altitude_tag()])
    
    d = get_3d_point_distance([X1, Y1, Z1], [X2, Y2, Z2])

    return d

def get_euler_diff_between_cameras(camera1_metadata:Dict, camera2_metadata:Dict, tags_map:CameraTagsMap):
    # we will get the shortest path using the formula (diff + 180) % 360 - 180
    
    yaw_diff = get_euler_diff(camera2_metadata[tags_map.get_camera_image_yaw_tag()], camera1_metadata[tags_map.get_camera_image_yaw_tag()])
    pitch_diff = get_euler_diff(camera2_metadata[tags_map.get_camera_image_pitch_tag()], camera1_metadata[tags_map.get_camera_image_pitch_tag()])
    roll_diff = get_euler_diff(camera2_metadata[tags_map.get_camera_image_roll_tag()], camera1_metadata[tags_map.get_camera_image_roll_tag()])

    return yaw_diff, pitch_diff, roll_diff

def get_frame_metadata(video_metadata:Dict, frame_number:int, camera_tags_map:CameraTagsMap, video_tags_map:CameraTagsMap):
    
    frame_data = {camera_tags_map.get_image_latitude_tag(): video_metadata.get(video_tags_map.get_frame_latitude_tag(idx=frame_number), None),
                  camera_tags_map.get_image_longitude_tag(): video_metadata.get(video_tags_map.get_frame_longitude_tag(idx=frame_number), None),
                  camera_tags_map.get_image_abs_altitude_tag(): video_metadata.get(video_tags_map.get_frame_abs_altitude_tag(idx=frame_number), None),
                  camera_tags_map.get_image_rel_altitude_tag(): video_metadata.get(video_tags_map.get_frame_rel_altitude_tag(idx=frame_number), None),
                  camera_tags_map.get_camera_image_yaw_tag(): video_metadata.get(video_tags_map.get_camera_frame_yaw_tag(idx=frame_number), None),
                  camera_tags_map.get_camera_image_pitch_tag(): video_metadata.get(video_tags_map.get_camera_frame_pitch_tag(idx=frame_number), None),
                  camera_tags_map.get_camera_image_roll_tag(): video_metadata.get(video_tags_map.get_camera_frame_roll_tag(idx=frame_number), None),
                  camera_tags_map.get_robot_image_yaw_tag(): video_metadata.get(video_tags_map.get_robot_frame_yaw_tag(idx=frame_number), None),
                  camera_tags_map.get_robot_image_pitch_tag(): video_metadata.get(video_tags_map.get_robot_frame_pitch_tag(idx=frame_number), None),
                  camera_tags_map.get_robot_image_roll_tag(): video_metadata.get(video_tags_map.get_robot_frame_roll_tag(idx=frame_number), None)}
    return frame_data