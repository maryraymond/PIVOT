from pathlib import Path
import json
import shutil
import os
from scipy.spatial.transform import Rotation as R
import numpy as np


from drone_video import process_video_to_images
from drone_core import (get_cameras_data,
                        read_images_data_from_sub_folders, 
                        frame_num)

from colmap_conversion import (read_colmap_cameras_txt, 
                               read_colmap_image_txt, 
                               convert_data_from_colmap,
                               get_colmap_registered_images_stat,
                               get_colmap_number_points)

from colmap_processing import run_colmap_with_soft_priors, CameraMode
from geometry import (get_rotation_euler_diff,
                      get_rotation_diff,
                      get_3d_point_distance,
                      get_3d_point_xyz_diff)

from camera_utils import (get_diag_fov_fe, 
                          get_vertical_fov_fe, 
                          get_horizontal_fov_fe, 
                          get_diag_fov, 
                          get_horizontal_fov, 
                          get_vertical_fov)



class RawDataProcessingPipeline():
    
    def __init__(self):
        self.COLORS = {
                    # "red":     (255, 0, 0),
                    "green":   (0, 128, 0),
                    "blue":    (0, 0, 255),
                    "yellow":  (255, 255, 0),
                    "cyan":    (0, 255, 255),
                    "magenta": (255, 0, 255),
                    "orange":  (255, 165, 0),
                    "purple":  (128, 0, 128),
                    "pink":    (255, 192, 203),
                    "lime":    (50, 205, 50),
                    "teal":    (0, 128, 128),
                    "navy":    (0, 0, 128),
                    "brown":   (165, 42, 42),
                    "olive":   (128, 128, 0),
                    "gold":    (255, 215, 0),
                    "maroon":    (128, 0, 0),
                    "indigo":    (75, 0, 130),
                    "violet":    (238, 130, 238),
                    "coral":     (255, 127, 80),
                    "salmon":    (250, 128, 114),
                    "turquoise": (64, 224, 208),
                    "skyblue":   (135, 206, 235),
                    "khaki":     (240, 230, 140),
                    "crimson":   (220, 20, 60),
                    "orchid":    (218, 112, 214),
                }

    def config_processing_pipeline(self, copy_images=True, run_colmap=True, copy_point_cloud=True, 
                                   transform_world_coord=False, add_statistics=True, add_camera_center_distance_error=True,
                                   add_camera_center_components_error=True, add_camera_rotation_angle_error=True, add_camera_rotation_euler_error=True,
                                   min_distance_m=0.3, min_rot_degree=10, pos_covariance=[6, 6, 6], max_num_models=4, fov_cal_colmap=True,
                                   wfov_as_fisheye=False):
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
    
    def configure_scene(self, scene_raw_dir, scene_processed_dir,  scene_description_file_name="traj_description.json",
                        scene_processed_json_file_name="scene_data.json", calibration_files_path="/code/data/",
                        colmap_folder_name="PYCOLMAP_soft_prior"):
        self.scene_raw_dir = scene_raw_dir
        self.scene_description_file = f"{scene_raw_dir}/{scene_description_file_name}"
        self.calibration_files_path = calibration_files_path

        self.scene_processed_dir = scene_processed_dir
        self.scene_processed_json = f"{scene_processed_dir}/{scene_processed_json_file_name}"
        self.scene_colmap = f"{scene_processed_dir}/{colmap_folder_name}"

        self.trajectories = [item for item in Path(scene_raw_dir).iterdir()]
        self.scene_data = {}

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
            
            if trajectory.is_file() and ".MP4" in trajectory.name:
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
        for trajectory in self.trajectories:

            trajectory_name = self._get_trajectory_name(trajectory)

            if trajectory_name is None:
                continue
        
            dst_dir = f"{self.scene_processed_dir}/{trajectory_name}"
            os.makedirs(dst_dir, exist_ok=True)
            
            if trajectory.is_dir():
                # those are images
                print("processing images")
                self.copy_image_dir(trajectory, dst_dir)
                
            elif trajectory.is_file() and ".MP4" in trajectory.name:
                # this is a video
                print(f"processing video {trajectory.name}")
                _, _ = process_video_to_images(video_file=trajectory, image_dir=dst_dir,
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

        
    def populate_frames_data(self, colmap_images_data_dict):
        # populate per frame meta data and calculate scene statistics
        # read the images per folder 
        images_per_folder = read_images_data_from_sub_folders(self.scene_processed_dir)
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
                frame_full_name = f"{trajectory_name}/{frame['file_name']}"
                frame_full = {}
                frame_full["file_name"] = frame_full_name
                frame_full["measured_pose_c2w"] = frame["pose_c2w"]

                if frame_full_name in colmap_images_data_dict:
                    c2w_colmap = colmap_images_data_dict[frame_full_name]["pose_c2w"]
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
                self.scene_data["trajectories"][trajectory_name]["average_rot_error"] = traj_rot_error / len(images_per_folder[trajectory_name])

            if self.add_camera_rotation_euler_error:
                self.scene_data["trajectories"][trajectory_name]["average_rot_error_yaw"] = traj_rot_error_yaw / len(images_per_folder[trajectory_name])
                self.scene_data["trajectories"][trajectory_name]["average_rot_error_pitch"] = traj_rot_error_pitch / len(images_per_folder[trajectory_name])
                self.scene_data["trajectories"][trajectory_name]["average_rot_error_roll"] = traj_rot_error_roll / len(images_per_folder[trajectory_name])

            if self.add_camera_center_distance_error:
                self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_distance"] = traj_camera_center_error_distance / len(images_per_folder[trajectory_name])

            if self.add_camera_center_components_error:
                self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_x"] = traj_camera_center_error_x / len(images_per_folder[trajectory_name])
                self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_y"] = traj_camera_center_error_y / len(images_per_folder[trajectory_name])
                self.scene_data["trajectories"][trajectory_name]["average_cam_center_error_z"] = traj_camera_center_error_z / len(images_per_folder[trajectory_name])

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
            run_colmap_with_soft_priors(dataset_images_dir=Path(self.scene_processed_dir), colmap_dir=Path(self.scene_colmap),
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

        colmap_frames, per_image_obser = get_colmap_registered_images_stat(colmap_images_txt)

        self.scene_data["colmap_reg_frames_number"] = colmap_frames
        self.scene_data["colmap_per_image_observation"] = per_image_obser

        number_points = get_colmap_number_points(colmap_points_txt)
        self.scene_data["pointcloud_number"] = number_points
        
        self.save_scene_data_file()