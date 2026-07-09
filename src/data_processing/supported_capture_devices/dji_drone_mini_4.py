# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

from data_processing.core.camera_metadata_abc import CameraTagsMap, CameraImageMetaData, GpsTagsMap, CameraVideoMetaData
from data_processing.core.metadata_utils import (read_metadata_exiftool, get_gps_value, get_camera_rotation,
                                                  write_gps_tags, write_standard_gps_tags, write_camera_tags,
                                                  write_robot_tags, get_distance_between_camera_centers,
                                                  get_euler_diff_between_cameras, get_frame_metadata)
from utils.gps_utils import ned_from_gps, convert_ned_cam_to_opengl
from utils.geometry_utils import get_homogenous_matrix
from data_processing.core.metadata_utils import read_video_metadata_exiftool


class DjiDroneMini4TagsMap(CameraTagsMap):
    def __init__(self):
        self.tags_map = {
            "image_longitude":"XMP-drone-dji:GPSLongitude",
            "image_latitude":"XMP-drone-dji:GPSLatitude",
            "image_altitude":"XMP-drone-dji:RelativeAltitude",
            "image_altitude_abs":"XMP-drone-dji:AbsoluteAltitude",

            "frame_longitude":"GPSLongitude",
            "frame_latitude":"GPSLatitude",
            "frame_altitude":"RelativeAltitude",
            "frame_altitude_abs":"AbsoluteAltitude",

            "camera_image_yaw":"XMP-drone-dji:GimbalYawDegree",
            "camera_image_pitch":"XMP-drone-dji:GimbalPitchDegree",
            "camera_image_roll":"XMP-drone-dji:GimbalRollDegree",

            "camera_frame_yaw":"GimbalYaw",
            "camera_frame_pitch":"GimbalPitch",
            "camera_frame_roll":"GimbalRoll",

            "robot_image_yaw":"XMP-drone-dji:FlightYawDegree",
            "robot_image_pitch":"XMP-drone-dji:FlightPitchDegree",
            "robot_image_roll":"XMP-drone-dji:FlightRollDegree",

            "robot_frame_yaw":"DroneYaw",
            "robot_frame_pitch":"DronePitch",
            "robot_frame_roll":"DroneRoll",
        }

    def get_image_longitude_tag(self):
        return self.tags_map['image_longitude']
    
    def get_image_latitude_tag(self):
        return self.tags_map['image_latitude']
    
    def get_image_rel_altitude_tag(self):
        return self.tags_map['image_altitude']
    
    def get_image_abs_altitude_tag(self):
        return self.tags_map['image_altitude_abs']
    
    def get_frame_longitude_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['frame_longitude']}"
    
    def get_frame_latitude_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['frame_latitude']}"
    
    def get_frame_abs_altitude_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['frame_altitude_abs']}"

    def get_frame_rel_altitude_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['frame_altitude']}"

    def get_camera_image_yaw_tag(self):
        return self.tags_map['camera_image_yaw']
    
    def get_camera_image_pitch_tag(self):
        return self.tags_map['camera_image_pitch']
    
    def get_camera_image_roll_tag(self):
        return self.tags_map['camera_image_roll']
    
    def get_camera_frame_yaw_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['camera_frame_yaw']}"
    
    def get_camera_frame_pitch_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['camera_frame_pitch']}"
    
    def get_camera_frame_roll_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['camera_frame_roll']}"

    def get_robot_image_yaw_tag(self):
        return self.tags_map['robot_image_yaw']
    
    def get_robot_image_pitch_tag(self):
        return self.tags_map['robot_image_pitch']
    
    def get_robot_image_roll_tag(self):
        return self.tags_map['robot_image_roll']
    
    def get_robot_frame_yaw_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['robot_frame_yaw']}"
    
    def get_robot_frame_pitch_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['robot_frame_pitch']}"
    
    def get_robot_frame_roll_tag(self, idx):
        return f"Doc{str(idx)}:{self.tags_map['robot_frame_roll']}"
    
class DjiDroneMini4ImageMetaData(CameraImageMetaData):
    def __init__(self, 
                 image_file:str, 
                 camera_tags_map:CameraTagsMap,
                 gps_tags_map:GpsTagsMap,
                 absolute_altitude:bool=True):
        self.image_file = image_file
        self.camera_tags_map = camera_tags_map
        self.gps_tags_map = gps_tags_map
        self.absolute_altitude = absolute_altitude
        self.metadata = read_metadata_exiftool(image_file)

        self.lat_0 = 0
        self.long_0 = 0
        self.alt_0 = 0
        
    def set_metadata(self, metadata):
        self.metadata = metadata
        
    def get_gps_value(self):
        return get_gps_value(self.metadata, self.camera_tags_map, self.absolute_altitude)
    
    def set_ref_gps(self, longitude, latitude, altitude):
        self.lat_0 = latitude
        self.long_0 = longitude
        self.alt_0 = altitude

    def get_camera_rotation(self):
        return get_camera_rotation(self.metadata, self.camera_tags_map)
    
    def get_c2w_opengl(self):
        lat, longitude, alt = self.get_gps_value()
        x, y, z = ned_from_gps(lat, longitude, alt, 
                               self.lat_0, self.long_0, self.alt_0, 
                               use_absolute_altitude=self.absolute_altitude)

        Rg2f, _ = self.get_camera_rotation()

        pose_g2w = get_homogenous_matrix(Rg2f, x, y, z)
        pose_c2w = convert_ned_cam_to_opengl(pose_g2w)

        return pose_c2w


    def write_camera_gps_tags(self):
        write_gps_tags(self.metadata, self.image_file, self.camera_tags_map)

    def write_standard_gps_tags(self):
        write_standard_gps_tags(self.metadata, self.image_file, 
                                tags_map=self.camera_tags_map, 
                                gps_tags_map=self.gps_tags_map)

    def write_camera_tags(self):
        write_camera_tags(self.metadata, self.image_file, self.camera_tags_map)

    def write_robot_tags(self):
        write_robot_tags(self.metadata, self.image_file, self.camera_tags_map)

    def write_all_camera_specific_tags(self):
        self.write_camera_gps_tags()
        self.write_camera_tags()
        self.write_robot_tags()


class DjiDroneMini4VideoMetaData(CameraVideoMetaData):
    def __init__(self, 
                 video_file, 
                 camera_tags_map,
                 absolute_altitude=True):
        self.metadata = read_video_metadata_exiftool(video_file)
        self.camera_tags_map = camera_tags_map
        self.absolute_altitude = absolute_altitude
        
    def get_frame_metadata(self, idx):
        return get_frame_metadata(self.metadata, frame_number=idx, camera_tags_map=self.camera_tags_map, video_tags_map=self.camera_tags_map)

    def get_distance_between_camera_centers(self, camera1_metadata, camera2_metadata):
        return get_distance_between_camera_centers(camera1_metadata, camera2_metadata, self.camera_tags_map)

    def get_euler_diff_between_cameras(self, camera1_metadata, camera2_metadata):
        return get_euler_diff_between_cameras(camera1_metadata, camera2_metadata, self.camera_tags_map)