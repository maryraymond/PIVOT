# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

from pathlib import Path
import json
import shutil
import os
from scipy.spatial.transform import Rotation as R
import numpy as np
import argparse
from typing import Dict, Callable

from data_processing.core.video_processing import process_video_to_images

from data_processing.core.image_processing import read_images_data_from_sub_folders, frame_num

from utils.processing_utils import get_cameras_data, get_trajectories_diff

from utils.metrics import (get_pose_tr_distance,
                           get_pose_r_distance,
                           get_pose_t_distance,
                           compute_scene_diameter,
                           compute_aabb_diagonal)

from data_processing.core.camera_metadata_abc import GpsTagsMap

from utils.geometry_utils import (get_rotation_euler_diff,
                                  get_rotation_diff,
                                  get_3d_point_distance,
                                  get_3d_point_xyz_diff)

from utils.camera_utils import (get_diag_fov_fe,
                                get_vertical_fov_fe,
                                get_horizontal_fov_fe,
                                get_diag_fov,
                                get_horizontal_fov,
                                get_vertical_fov)


from colmap_utils.colmap_conversion import (read_colmap_cameras_txt,
                                            read_colmap_image_txt,
                                            convert_data_from_colmap,
                                            get_colmap_registered_images_stat,
                                            get_colmap_number_points,
                                            run_colmap_model_analyzer)

from colmap_utils.colmap_processing import run_colmap_with_soft_priors, CameraMode


def _make_metadata_dispatcher(metadata_map):
    def dispatch(image_path, use_absolute_altitude):
        traj_name = Path(image_path).parent.name
        return metadata_map[traj_name](image_path, use_absolute_altitude)
    return dispatch


class RawDataProcessingPipeline():
    
    def __init__(self, 
                 supported_image_capture_devices:Dict[str, Callable], 
                 supported_video_capture_devices:Dict[str, Callable], 
                 colors=None):

        self.supported_image_capture_devices = supported_image_capture_devices
        self.supported_video_capture_devices = supported_video_capture_devices

        if colors is not None:
            self.COLORS=colors
        else:
            self.COLORS = {
                        # red is reserved for error visualisation
                        # base: Tableau-20 minus red pair
                        "blue":         ( 31, 119, 180),
                        "light_blue":   (174, 199, 232),
                        "orange":       (255, 127,  14),
                        "light_orange": (255, 187, 120),
                        "green":        ( 44, 160,  44),
                        "light_green":  (152, 223, 138),
                        "purple":       (148, 103, 189),
                        "light_purple": (197, 176, 213),
                        "brown":        (140,  86,  75),
                        "light_brown":  (196, 156, 148),
                        "pink":         (227, 119, 194),
                        "light_pink":   (247, 182, 210),
                        "gray":         (127, 127, 127),
                        "light_gray":   (199, 199, 199),
                        "olive":        (188, 189,  34),
                        "light_olive":  (219, 219, 141),
                        "teal":         ( 23, 190, 207),
                        "light_teal":   (158, 218, 229),
                        # extensions to reach 25
                        "lime":         (139, 195,  74),
                        "light_lime":   (197, 225, 165),
                        "sea_green":    (  0, 121, 107),
                        "violet":       ( 94,  53, 177),
                        "gold":         (255, 196,  15),
                        "sky_blue":     (  3, 155, 229),
                        "slate":        (100, 116, 139),
                    }

    def config_processing_pipeline(self, copy_images=True, run_colmap=True, copy_point_cloud=True,
                                   transform_world_coord=False, add_statistics=True, add_camera_center_distance_error=True,
                                   add_camera_center_components_error=True, add_camera_rotation_angle_error=True, add_camera_rotation_euler_error=True,
                                   min_distance_m=0.3, min_rot_degree=10, pos_covariance=[6, 6, 6], max_num_models=4, fov_cal_colmap=True,
                                   wfov_as_fisheye=False, absolute_altitude=True, chamfer_k_neighbor=1,
                                   chamfer_translation_scale: str = "aabb_diagonal", chamfer_rotation_scale: float = 180.0):
        self.min_distance_m = min_distance_m
        self.min_rot_degree = min_rot_degree
        self.pos_covariance = pos_covariance
        self.max_num_models = max_num_models
        self.copy_images = copy_images
        self.run_colmap = run_colmap
        self.copy_point_cloud = copy_point_cloud
        self.transform_world_coord = transform_world_coord
        self.add_camera_center_distance_error = add_camera_center_distance_error
        self.add_camera_center_components_error = add_camera_center_components_error
        self.add_camera_rotation_angle_error = add_camera_rotation_angle_error
        self.add_camera_rotation_euler_error = add_camera_rotation_euler_error
        self.add_statistics = add_statistics
        self.fov_cal_colmap = fov_cal_colmap
        self.wfov_as_fisheye = wfov_as_fisheye
        self.absolute_altitude = absolute_altitude
        self.chamfer_k_neighbor = chamfer_k_neighbor
        self.chamfer_translation_scale = chamfer_translation_scale
        self.chamfer_rotation_scale = chamfer_rotation_scale
    
    def configure_scene(self, scene_raw_dir, scene_processed_dir, scene_description_file_name="traj_description.json",
                        scene_processed_json_file_name="scene_data.json", calibration_files_path="/code/data/",
                        colmap_folder_name="PYCOLMAP_soft_prior", trajectories=None):
        self.scene_raw_dir = scene_raw_dir
        self.scene_description_file = f"{scene_raw_dir}/{scene_description_file_name}"
        self.calibration_files_path = calibration_files_path

        self.scene_processed_dir = scene_processed_dir
        self.scene_processed_json = f"{scene_processed_dir}/{scene_processed_json_file_name}"
        self.scene_colmap = f"{scene_processed_dir}/{colmap_folder_name}"
        self.scene_trajectories_dir = f"{scene_processed_dir}/trajectories"

        if trajectories is not None:
            self.trajectories = [Path(scene_raw_dir) / t for t in trajectories]
        else:
            self.trajectories = [item for item in Path(scene_raw_dir).iterdir()]
        self.scene_data = {}

        with open(self.scene_description_file, "r") as f:
                trajectories_description = json.load(f)
                
        # create a map for the capture device metadata for each trajectory
        self.image_metadata_map = {}
        self.video_metadata_map = {}
        for traj_name in trajectories_description.keys():
            if "capture_device" not in trajectories_description[traj_name]:
                raise ValueError(f"No capture device is defined for {traj_name}")
            else:
                self.image_metadata_map[traj_name] = self.supported_image_capture_devices[trajectories_description[traj_name]["capture_device"]]
                self.video_metadata_map[traj_name] = self.supported_video_capture_devices[trajectories_description[traj_name]["capture_device"]]

    @staticmethod
    def _get_trajectory_name(trajectory_path):
        if ".json" in trajectory_path.name:
            trajectory_name = None
        else:    
            trajectory_name = trajectory_path.name.split('.')[0]
        
        return trajectory_name
    
    @staticmethod
    def _get_frame_name(frame_index):
        return f"frame_{frame_index:06d}.JPG"
        
    def populate_trajectories_basic_data(self):
        # populate scene description
        with open(self.scene_description_file, "r") as f:
            trajectories_description = json.load(f)

            self.scene_data["trajectories"] ={}

        for trajectory in self.trajectories:

            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue
            
            if trajectory_name in trajectories_description:
                self.scene_data["trajectories"][trajectory_name] = {}
                for info_tag in trajectories_description[trajectory_name].keys():
                    self.scene_data["trajectories"][trajectory_name][info_tag] = trajectories_description[trajectory_name][info_tag]
            else:
                raise ValueError(f"Missing description for trajectory {trajectory_name}")
            
            if trajectory.is_file() and trajectory.suffix.upper() == ".MP4":
                self.scene_data["trajectories"][trajectory_name]["sampling_method"] = "distance_and_rot"
                self.scene_data["trajectories"][trajectory_name]["sample_details"] = {
                    "min_distance_m": self.min_distance_m,
                    "min_rotation_degree": self.min_rot_degree
                } 


    def copy_image_dir(self, trajectory, dst_dir):
        images = sorted([image.name for image in Path(trajectory).iterdir() if image.is_file()], key=lambda x:frame_num(x))
        for i, image in enumerate(images):
            dst_name = self._get_frame_name(i)
            shutil.copyfile(src=f"{trajectory}/{image}", dst= f"{dst_dir}/{dst_name}")


    def copy_trajectories_images(self):
        # We will loop over all the trajectories, if the type is folder it means those are images
        # and we will copy them and add the coresponding info to the output dict
        # if the type is a file with .mp4 extension then we will process them to the correspodning location

        # images copy and video processing
        os.makedirs(self.scene_trajectories_dir, exist_ok=True)
        for trajectory in self.trajectories:

            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue

            dst_dir = f"{self.scene_trajectories_dir}/{trajectory_name}"
            os.makedirs(dst_dir, exist_ok=True)
            
            if trajectory.is_dir():
                # those are images
                print("processing images")
                self.copy_image_dir(trajectory, dst_dir)
                
            elif trajectory.is_file() and trajectory.suffix.upper() == ".MP4":
                # this is a video
                print(f"processing video {trajectory.name}")
                _, _ = process_video_to_images(video_file=trajectory,
                                               image_dir=dst_dir,
                                               create_image_metadata=self.image_metadata_map[trajectory_name],
                                               create_video_metadata=self.video_metadata_map[trajectory_name],
                                               use_absloute_altitude=self.absolute_altitude,
                                               min_camera_distanct_m=self.min_distance_m,
                                               min_camera_rot_deg=self.min_rot_degree,
                                               debug_prints=self.debug_prints,
                                               frame_name_fn=self._get_frame_name)
                                            
    def populate_colors_for_trajectories(self):
        
        colors=iter(self.COLORS.items())
        for trajectory in self.trajectories:
            
            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue

            color = next(colors)

            self.scene_data["trajectories"][trajectory_name]["color_name"] = color[0]
            self.scene_data["trajectories"][trajectory_name]["color_value"] = color[1]
    
    # @staticmethod
    #todo needs fixing and change to a static method
    def _get_calibration_file_name(self, trajectory_description):
        calibration_name_parts = []
        if "video" in trajectory_description["capture_mode"]:
            calibration_name_parts.append("video_camera_")
        else:
            calibration_name_parts.append("images_camera_")
        
        if trajectory_description["aspect_ratio"] == "16:9":
            calibration_name_parts.append("16_9_")
        elif trajectory_description["aspect_ratio"] == "4:3":
            calibration_name_parts.append("4_3_")
        else:
            raise ValueError(f"Error unsupported aspect ratio of {trajectory_description['aspect_ratio']}")
        
        if "w" in trajectory_description["lens_type"]:
            calibration_name_parts.append("WFOV_")

        calibration_name_parts.append("calibration.json")

        return "".join(calibration_name_parts)
    
    def get_traj_to_camera_type_map(self):
        with open(self.scene_description_file, 'r') as f:
            traj_data = json.load(f)

        traj_to_camera = {}
        for traj in traj_data.keys():
            if "w" in traj_data[traj]["lens_type"]:
                traj_to_camera[traj] = "OPENCV_FISHEYE"
            else:
                traj_to_camera[traj] = "OPENCV"
        
        return traj_to_camera

    def populate_camera_intrinsic_calibration(self):
        # populate scene description
        with open(self.scene_description_file, "r") as f:
                trajectories_description = json.load(f)

        for trajectory in self.trajectories:

            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue
                
            if trajectory_name in trajectories_description:
                calibration_file_name = self._get_calibration_file_name(trajectories_description[trajectory_name])
                cal_file = f"{self.calibration_files_path}/{calibration_file_name}"
                camera_calibration = get_cameras_data(cal_file=cal_file, 
                                                    camera_id = 1)[1]
                self.scene_data["trajectories"][trajectory_name]["camera_intrinsic_calibration"] = camera_calibration
            else:
                raise ValueError(f"Missing description for trajectory {trajectory_name}")
                  
    def populate_camera_fov_calibration(self):
        # populate scene description
        with open(self.scene_description_file, "r") as f:
                trajectories_description = json.load(f)

        for trajectory in self.trajectories:

            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue
                
            if trajectory_name in trajectories_description:
                cal_file = f"{self.calibration_files_path}/{trajectories_description[trajectory_name]['calibration_file']}"
                camera_calibration = get_cameras_data(cal_file=cal_file, 
                                                    camera_id = 1)[1]
                fov_h, fov_v, fov_diag = self.get_camera_fov_from_intrincic(camera_calibration)
                
            else:
                raise ValueError(f"Missing description for trajectory {trajectory_name}")
                fov_h, fov_v, fov_diag = 0, 0, 0
            
            self.scene_data["trajectories"][trajectory_name]["fov_h"] = fov_h
            self.scene_data["trajectories"][trajectory_name]["fov_v"] = fov_v
            self.scene_data["trajectories"][trajectory_name]["fov_diag"] = fov_diag
        
    def find_reconstruction_with_max_images(self):
        # find the best reconstruction
        reconstructions_paths = [path for path in (Path(self.scene_colmap)/"sparse").iterdir() if path.is_dir()]

        max_images_number = 0
        seleceted_reonstrcution = ""

        # loop ones to find all the reconstruction numbers
        for reconstruction_path in reconstructions_paths:
            images_txt = reconstruction_path/"images.txt"
            try:
                with open(images_txt, "r") as f:
                    lines = f.readlines()
                    images_number_line = lines[3]
                    images_number = int(images_number_line.split(":")[1].split(",")[0])
                    print(f"Number of registered images in reconstruction {reconstruction_path.name} = {images_number}")
                    if images_number > max_images_number:
                        max_images_number = images_number
                        seleceted_reonstrcution = reconstruction_path
            except Exception as e:
                print(f"Could not open the file {images_txt} due to execption {e}")
        
        return seleceted_reonstrcution, max_images_number

    def get_camera_fov_from_intrincic(self, intrinsic):

        w = float(intrinsic["w"])
        h = float(intrinsic["h"])
        fx = float(intrinsic["fl_x"])
        fy = float(intrinsic["fl_y"])

        if intrinsic["camera_type"] == "OPENCV_FISHEYE":
            fov_h = get_horizontal_fov_fe(W=w, fx=fx, degrees=True)
            fov_v = get_vertical_fov_fe(H=h, fy=fy, degrees=True)
            fov_diag = get_diag_fov_fe(W=w, H=h, fx=fx, fy=fy, cx=None, cy=None, degrees=True)

        elif intrinsic["camera_type"] == "OPENCV":
            fov_h = get_horizontal_fov(W=w, fx=fx)
            fov_v = get_vertical_fov(H=h, fy=fy)
            fov_diag = get_diag_fov(W=w, H=h, fx=fx, fy=fy, degrees=True)
        else:
            raise ValueError(f"Error: Unsupported camera type {intrinsic['camera_type']}")
        
        return fov_h, fov_v,  fov_diag
    
    def populate_camera_fov_colmap(self, colmap_images_data_dict, colmap_cameras_data):

        for trajectory in self.trajectories:

            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue
                
            sample_frame = next((k for k in colmap_images_data_dict.keys() if k.startswith(trajectory_name)), None)
            if sample_frame is not None:
                colmap_intrinsics = colmap_cameras_data[colmap_images_data_dict[sample_frame]["camera_id"]]
                fov_h, fov_v, fov_diag = self.get_camera_fov_from_intrincic(colmap_intrinsics)
            else:
                colmap_intrinsics = None
                fov_h, fov_v, fov_diag = 0, 0, 0

            self.scene_data["trajectories"][trajectory_name]["fov_h"] = fov_h
            self.scene_data["trajectories"][trajectory_name]["fov_v"] = fov_v
            self.scene_data["trajectories"][trajectory_name]["fov_diag"] = fov_diag

    def populate_camera_intrinsic_colmap(self, colmap_images_data_dict, colmap_cameras_data):

        for trajectory in self.trajectories:

            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue
                
            sample_frame = next((k for k in colmap_images_data_dict.keys() if k.startswith(trajectory_name)), None)
            if sample_frame is not None:
                colmap_intrinsics = colmap_cameras_data[colmap_images_data_dict[sample_frame]["camera_id"]]
            else:
                colmap_intrinsics = None
            self.scene_data["trajectories"][trajectory_name]["camera_intrinsic_colmap"] = colmap_intrinsics
            
        
    def populate_pointcloud(self, colmap_point_cloud_ply):
        
        try:
            dst_file = f"{self.scene_processed_dir}/{Path(colmap_point_cloud_ply).name}"
            shutil.copyfile(colmap_point_cloud_ply, dst_file)
            self.scene_data["pointcloud"] = Path(dst_file).name
        except Exception as e:
            print(f"Could not copy point cloud {colmap_point_cloud_ply} due to execption {e}")

    def populate_trajectories_difference_metrics(self):

        trajectories_names = []			

        for trajectory in self.trajectories:
            
            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue
            trajectories_names.append(trajectory_name)
        
        trajectories_names = sorted(trajectories_names)

        # we need to measure the translation scale, we have two methods
        # aabb digonal or scene diameter, ideally this need to be configured
        camera_centers = [np.array(frame_data["colmap_pose_c2w"])[:3, 3]  \
                          for traj in trajectories_names \
                          for frame_data in self.scene_data["trajectories"][traj]["frames"] \
                          if "colmap_pose_c2w" in frame_data]
        
        scene_diameter = compute_scene_diameter(camera_centers)
        aabb_diagonal = compute_aabb_diagonal(camera_centers)

        if self.chamfer_translation_scale == "scene_diameter":
            translation_scale = scene_diameter
        else:  # "aabb_diagonal"
            translation_scale = aabb_diagonal

        self.scene_data["scene_diameter"] = round(scene_diameter, 3)
        self.scene_data["aabb_diagonal"] = round(aabb_diagonal, 3)

        for traj_a in trajectories_names:
            trajectories_matrix = {}
            for traj_b in trajectories_names:
                trajectories_matrix[traj_b] = {}

                traj_tr_distance = get_trajectories_diff(self.scene_data,
                                                    traj_a_name=traj_a,
                                                    traj_b_name=traj_b,
                                                    k_neighbor_size=self.chamfer_k_neighbor,
                                                    pose_distance_fn=lambda pose_a, pose_b, _ts=translation_scale: (
                                                        get_pose_tr_distance(pose_a=pose_a,
                                                                             pose_b=pose_b,
                                                                             translation_scale=_ts,
                                                                             rotation_scale=self.chamfer_rotation_scale)
                                                    ))
                trajectories_matrix[traj_b]["directed_norm_chamfer_tr_distance"] = round(traj_tr_distance, 3)

                traj_t_distance = get_trajectories_diff(self.scene_data,
                                                    traj_a_name=traj_a,
                                                    traj_b_name=traj_b,
                                                    k_neighbor_size=self.chamfer_k_neighbor,
                                                    pose_distance_fn=lambda pose_a, pose_b, _ts=translation_scale: (
                                                        get_pose_t_distance(pose_a=pose_a,
                                                                            pose_b=pose_b,
                                                                            translation_scale=_ts)
                                                    ))
                trajectories_matrix[traj_b]["directed_norm_chamfer_t_distance"] = round(traj_t_distance, 3)

                traj_r_distance = get_trajectories_diff(self.scene_data,
                                                    traj_a_name=traj_a,
                                                    traj_b_name=traj_b,
                                                    k_neighbor_size=self.chamfer_k_neighbor,
                                                    pose_distance_fn=lambda pose_a, pose_b: (
                                                        get_pose_r_distance(pose_a=pose_a,
                                                                            pose_b=pose_b,
                                                                            rotation_scale=self.chamfer_rotation_scale)
                                                    ))
                trajectories_matrix[traj_b]["directed_norm_chamfer_r_distance"] = round(traj_r_distance, 3)

            self.scene_data["trajectories"][traj_a]["trajectory_metrics"] = trajectories_matrix


    def populate_frames_data(self, colmap_images_data_dict):
        # populate per frame meta data and calculate scene statistics
        # read the images per folder 

        images_per_folder = read_images_data_from_sub_folders(self.scene_trajectories_dir,
                                                              create_image_metadata_map=self.image_metadata_map,
                                                              use_absloute_altitude=self.absolute_altitude)
        
        # Now we will loop over the trajectories to read the metadata
        total_frames_number = 0

        for trajectory in self.trajectories:
            
            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue

            traj_rot_error = 0
            traj_rot_error_yaw = 0
            traj_rot_error_pitch = 0
            traj_rot_error_roll = 0
            traj_camera_center_error_distance = 0
            traj_camera_center_error_x = 0
            traj_camera_center_error_y = 0
            traj_camera_center_error_z = 0
            traj_missing_colmap = 0
            traj_total_frames_num = 0
            
            frames = []
            for frame in images_per_folder[trajectory_name]:
                colmap_key = f"{trajectory_name}/{frame['file_name']}"
                frame_full = {}
                frame_full["file_name"] = f"trajectories/{trajectory_name}/{frame['file_name']}"
                frame_full["measured_pose_c2w"] = frame["pose_c2w"]

                if colmap_key in colmap_images_data_dict:
                    c2w_colmap = colmap_images_data_dict[colmap_key]["pose_c2w"]
                    frame_full["colmap_pose_c2w"] = c2w_colmap
                    # add difference
                    if self.add_statistics:
                        
                        c2w_measured = np.array(frame["pose_c2w"])
                        c2w_colmap = np.array(c2w_colmap)

                        if self.add_camera_rotation_angle_error:
                            rot_diff = get_rotation_diff(c2w_measured[:3,:3], c2w_colmap[:3,:3])
                            frame_full["rot_error"] = rot_diff
                            traj_rot_error += rot_diff

                        if self.add_camera_rotation_euler_error:
                            yaw_diff, pitch_diff, roll_diff = get_rotation_euler_diff(c2w_measured[:3,:3], c2w_colmap[:3,:3])
                            frame_full["rot_error_yaw"] = yaw_diff
                            frame_full["rot_error_pitch"] = pitch_diff
                            frame_full["rot_error_roll"] = roll_diff

                            traj_rot_error_yaw += yaw_diff
                            traj_rot_error_pitch += pitch_diff
                            traj_rot_error_roll += roll_diff

                        if self.add_camera_center_distance_error:
                            camera_center_distance = get_3d_point_distance(c2w_measured[:3, 3].tolist(), c2w_colmap[:3, 3].tolist())
                            frame_full["camera_center_error_distance"] = camera_center_distance
                            traj_camera_center_error_distance += camera_center_distance

                        if self.add_camera_center_components_error:
                            x_diff, y_diff, z_diff = get_3d_point_xyz_diff(c2w_measured[:3, 3].tolist(), c2w_colmap[:3, 3].tolist())
                            frame_full["camera_center_error_x"] = x_diff
                            frame_full["camera_center_error_y"] = y_diff
                            frame_full["camera_center_error_z"] = z_diff

                            traj_camera_center_error_x += x_diff
                            traj_camera_center_error_y += y_diff
                            traj_camera_center_error_z += z_diff            
                    
                else:
                    # frame_full["colmap_pose_c2w"] = None
                    traj_missing_colmap += 1
                
                traj_total_frames_num += 1
                total_frames_number += 1

                frames.append(frame_full)

            if self.add_camera_rotation_angle_error:    
                self.scene_data["trajectories"][trajectory_name]["average_rot_error"] = traj_rot_error / (traj_total_frames_num - traj_missing_colmap)

            if self.add_camera_rotation_euler_error:
                self.scene_data["trajectories"][trajectory_name]["average_rot_error_yaw"] = traj_rot_error_yaw / (traj_total_frames_num - traj_missing_colmap)
                self.scene_data["trajectories"][trajectory_name]["average_rot_error_pitch"] = traj_rot_error_pitch / (traj_total_frames_num - traj_missing_colmap)
                self.scene_data["trajectories"][trajectory_name]["average_rot_error_roll"] = traj_rot_error_roll / (traj_total_frames_num - traj_missing_colmap)

            if self.add_camera_center_distance_error:
                self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_distance"] = traj_camera_center_error_distance / (traj_total_frames_num - traj_missing_colmap)

            if self.add_camera_center_components_error:
                self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_x"] = traj_camera_center_error_x / (traj_total_frames_num - traj_missing_colmap)
                self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_y"] = traj_camera_center_error_y / (traj_total_frames_num - traj_missing_colmap)
                self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_z"] = traj_camera_center_error_z / (traj_total_frames_num - traj_missing_colmap)

            self.scene_data["trajectories"][trajectory_name]["missing_colmap_frames"] = traj_missing_colmap
            self.scene_data["trajectories"][trajectory_name]["number_frames_in_traj"] = traj_total_frames_num
            self.scene_data["trajectories"][trajectory_name]["frames"] = frames

        self.scene_data["total_frames_number"] =  total_frames_number



    def  print_trajectories_stats(self):
        for trajectory in self.trajectories:
            
            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue
            
            print(f"{trajectory_name}:")
            print("average_rot_error", self.scene_data["trajectories"][trajectory_name]["average_rot_error"])
            print("average_rot_error_yaw", self.scene_data["trajectories"][trajectory_name]["average_rot_error_yaw"])
            print("average_rot_error_pitch", self.scene_data["trajectories"][trajectory_name]["average_rot_error_pitch"])
            print("average_rot_error_roll", self.scene_data["trajectories"][trajectory_name]["average_rot_error_roll"])
            print("average_cam_center_error_distance", self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_distance"])
            print("average_cam_center_error_x", self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_x"])
            print("average_cam_center_error_y", self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_y"])
            print("average_cam_center_error_z", self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_z"])
            print("missing_colmap_frames", self.scene_data["trajectories"][trajectory_name]["missing_colmap_frames"])
            print("number_frames_in_traj", self.scene_data["trajectories"][trajectory_name]["number_frames_in_traj"])
            print(" ")
            
    def save_scene_data_file(self):
        # save the scene data
        with open(self.scene_processed_json, "w") as f:
            json.dump(self.scene_data, f, indent=4)


    def process_scene_from_raw(self, debug_prints=False):

        self.debug_prints = debug_prints

        # first we will populate the basic data for each trajectory 
        self.populate_trajectories_basic_data()

        # populate the colors for each trajectory
        self.populate_colors_for_trajectories()

        # if we need to copy the data (if it is not  already copied we will copy it here)
        if self.copy_images:
            self.copy_trajectories_images()

        if self.run_colmap:
            if self.wfov_as_fisheye:
                traj_camera_map = self.get_traj_to_camera_type_map()
            else:
                traj_camera_map = None
            run_colmap_with_soft_priors(dataset_images_dir=Path(self.scene_trajectories_dir),
                                        create_image_metadata=_make_metadata_dispatcher(self.image_metadata_map),
                                        colmap_dir=Path(self.scene_colmap),
                                        pos_var=self.pos_covariance, update_positions=True, cartesian_system=True,
                                        camera_mode=CameraMode.PER_FOLDER, max_num_models=self.max_num_models, camera_model_map=traj_camera_map)
            
        # Get the best colmap reconstrcution and read the data
        colmap_spares, registered_images = self.find_reconstruction_with_max_images()
        print(f"Found best reconstrcution at {colmap_spares} with {registered_images} registered images")

        colmap_cameras_txt = colmap_spares/"cameras.txt"
        colmap_images_txt = colmap_spares/"images.txt"
        colmap_points_txt = colmap_spares/"points3D.txt"
        colmap_point_cloud_ply = colmap_spares/"sparse_model.ply"

        colmap_cameras_data = read_colmap_cameras_txt(colmap_cameras_txt)
        colmap_images_data = read_colmap_image_txt(colmap_images_txt)
        colmap_images_data = convert_data_from_colmap(colmap_images_data, self.transform_world_coord)
        colmap_images_data_dict = {data["file_name"]:data for data in  colmap_images_data}

        # Populate the FOV info
        if self.fov_cal_colmap:
            self.populate_camera_fov_colmap(colmap_images_data_dict, colmap_cameras_data)
        else: 
            self.populate_camera_fov_calibration()

        # populate the camera intrinsics from calibration
        self.populate_camera_intrinsic_calibration()

        # populate the camera intrinscis from colmap
        self.populate_camera_intrinsic_colmap(colmap_images_data_dict, colmap_cameras_data)

        if self.copy_point_cloud:
            self.populate_pointcloud(colmap_point_cloud_ply)

        self.populate_frames_data(colmap_images_data_dict)

        # the trajectory metrics could only be populated after the frames data is populated
        self.populate_trajectories_difference_metrics()

        colmap_frames, per_image_obser = get_colmap_registered_images_stat(colmap_images_txt)

        self.scene_data["colmap_reg_frames_number"] = colmap_frames
        self.scene_data["colmap_per_image_observation"] = per_image_obser

        number_points = get_colmap_number_points(colmap_points_txt)
        self.scene_data["pointcloud_number"] = number_points

        colmap_stat = run_colmap_model_analyzer(colmap_spares)
        self.scene_data["observations"] = colmap_stat["observations"]
        self.scene_data["mean_track_length"] = colmap_stat["mean_track_length"]
        self.scene_data["mean_observations_per_image"] = colmap_stat["mean_observations_per_image"]
        self.scene_data["mean_reprojection_error_px"] = colmap_stat["mean_reprojection_error_px"]
        
        self.save_scene_data_file()
