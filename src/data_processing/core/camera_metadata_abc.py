from abc import ABC, abstractmethod


class CameraTagsMap(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def get_image_longitude_tag(self):
        pass

    @abstractmethod
    def get_image_latitude_tag(self):
        pass

    @abstractmethod
    def get_image_rel_altitude_tag(self):
        pass

    @abstractmethod
    def get_image_abs_altitude_tag(self):
        pass

    @abstractmethod
    def get_frame_longitude_tag(self, idx):
        pass

    @abstractmethod
    def get_frame_latitude_tag(self, idx):
        pass

    @abstractmethod
    def get_frame_abs_altitude_tag(self, idx):
        pass

    @abstractmethod
    def get_frame_rel_altitude_tag(self, idx):
        pass

    @abstractmethod
    def get_camera_image_yaw_tag(self):
        pass

    @abstractmethod
    def get_camera_image_pitch_tag(self):
        pass

    @abstractmethod
    def get_camera_image_roll_tag(self):
        pass

    @abstractmethod
    def get_camera_frame_yaw_tag(self, idx):
        pass

    @abstractmethod
    def get_camera_frame_pitch_tag(self, idx):
        pass

    @abstractmethod
    def get_camera_frame_roll_tag(self, idx):
        pass

    @abstractmethod
    def get_robot_image_yaw_tag(self):
        pass

    @abstractmethod
    def get_robot_image_pitch_tag(self):
        pass

    @abstractmethod
    def get_robot_image_roll_tag(self):
        pass

    @abstractmethod
    def get_robot_frame_yaw_tag(self, idx):
        pass

    @abstractmethod
    def get_robot_frame_pitch_tag(self, idx):
        pass

    @abstractmethod
    def get_robot_frame_roll_tag(self, idx):
        pass


class GpsTagsMap():
    def __init__(self):
        self.tags_map = {
            "longitude_abs":"GPS:GPSLongitude",
            "longitude_ref":"GPS:GPSLongitudeRef",
            "latitude_abs":"GPS:GPSLatitude",
            "latitude_ref":"GPS:GPSLatitudeRef",
            "altitude_abs":"GPS:GPSAltitude",
            "altitude_ref":"GPS:GPSAltitudeRef",
        }
    
    def get_abs_longitude_tag(self):
        return self.tags_map["longitude_abs"]
    
    def get_ref_longitude_tag(self):
        return self.tags_map["longitude_ref"]
    
    
    def get_abs_latitude_tag(self):
        return self.tags_map["latitude_abs"]
    
    def get_ref_latitude_tag(self):
        return self.tags_map["latitude_ref"]
    

    def get_abs_altitude_tag(self):
        return self.tags_map["altitude_abs"]
    
    def get_ref_altitude_tag(self):
        return self.tags_map["altitude_ref"]
    

class CameraImageMetaData(ABC):
    def __init__(self,
                 frame_file:str,
                 camera_tags_map:CameraTagsMap,
                 gps_tags_map:GpsTagsMap,
                 absolute_altitude:bool=True):
        pass

    @abstractmethod
    def set_metadata(self, metadata):
        pass

    @abstractmethod
    def get_gps_value(self):
        pass

    @abstractmethod
    def set_ref_gps(self, longitude, latitude, altitude):
        pass

    @abstractmethod
    def get_camera_rotation(self):
        pass

    @abstractmethod
    def get_c2w_opengl(self):
        pass

    @abstractmethod
    def write_camera_gps_tags(self):
        pass

    @abstractmethod
    def write_standard_gps_tags(self):
        pass

    @abstractmethod
    def write_all_camera_specific_tags(self):
        pass

    @abstractmethod
    def write_camera_tags(self):
        pass



class CameraVideoMetaData(ABC):
    def __init__(self,
                 video_file,
                 camera_tags_map,
                 absolute_altitude=True):
        pass

    @abstractmethod
    def get_frame_metadata(self, idx):
        pass

    @abstractmethod
    def get_distance_between_camera_centers(self, camera1_metadata, camera2_metadata):
        pass

    @abstractmethod
    def get_euler_diff_between_cameras(self, camera1_metadata, camera2_metadata):
        pass