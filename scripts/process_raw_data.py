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

from data_processing.core.raw_data_processing_pipeline import RawDataProcessingPipeline
from data_processing.core.camera_metadata_abc import GpsTagsMap

from data_processing.supported_capture_devices.dji_drone_mini_4 import (DjiDroneMini4ImageMetaData, 
                                                                        DjiDroneMini4TagsMap, 
                                                                        DjiDroneMini4VideoMetaData)



# this is the configuration that need to be extended if more capture devices need to be supported
supported_image_capture_devices = {"dji_drone_mini_4_pro": lambda name, abs_alt: ( DjiDroneMini4ImageMetaData(image_file=name, 
                                                                                                        camera_tags_map=DjiDroneMini4TagsMap(),
                                                                                                        gps_tags_map=GpsTagsMap(),
                                                                                                        absolute_altitude=abs_alt)
                                                                                  )
                                    }

supported_video_capture_devices = {"dji_drone_mini_4_pro": lambda name, abs_alt: ( DjiDroneMini4VideoMetaData(video_file=name, 
                                                                                                        camera_tags_map=DjiDroneMini4TagsMap(),
                                                                                                        absolute_altitude=abs_alt)
                                                                                  )
                                  }

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Process raw capture data from videos and images into the dataset format."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- required paths ---
    parser.add_argument("--scene-raw-dir", required=True,
                        help="Root directory of the raw captured scene data.")
    parser.add_argument("--scene-processed-dir", required=True,
                        help="Output directory for the processed dataset.")
    parser.add_argument("--calibration-files-path", required=True,
                        help="Directory containing per-camera calibration JSON files.")

    # --- optional paths (sensible defaults) ---
    parser.add_argument("--scene-description-file", default="trajectories_metadata.json",
                        help="Trajectory description JSON filename (default: trajectories_metadata.json).")
    parser.add_argument("--scene-output-json", default="scene_data.json",
                        help="Output scene data JSON filename (default: scene_data.json).")
    parser.add_argument("--colmap-folder", default="PYCOLMAP_soft_prior",
                        help="Name of the COLMAP output subdirectory (default: PYCOLMAP_soft_prior).")
    parser.add_argument("--trajectories", nargs="+", default=None,
                        help="Explicit list of trajectory names to process (space-separated). "
                             "Names are resolved relative to --scene-raw-dir. "
                             "If omitted, all entries under --scene-raw-dir are used.")

    # --- pipeline steps ---
    parser.add_argument("--copy-images", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy images to the processed directory (default: True).")
    parser.add_argument("--run-colmap", action=argparse.BooleanOptionalAction, default=True,
                        help="Run COLMAP reconstruction (default: True).")
    parser.add_argument("--copy-point-cloud", action=argparse.BooleanOptionalAction, default=True,
                        help="Copy the COLMAP point cloud PLY file (default: True).")
    parser.add_argument("--transform-world-coord", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply world coordinate transform to COLMAP poses (default: False).")

    # --- COLMAP parameters ---
    parser.add_argument("--pos-covariance", nargs=3, type=float, default=[2.0, 2.0, 2.0],
                        metavar=("X", "Y", "Z"),
                        help="Position prior covariance in X Y Z (default: 2 2 2).")
    parser.add_argument("--max-num-models", type=int, default=4,
                        help="Maximum number of COLMAP sub-models to accept (default: 4).")

    # --- frame selection parameters ---
    parser.add_argument("--min-distance", type=float, default=0.3,
                        help="Minimum camera centre distance (m) between saved frames (default: 0.3).")
    parser.add_argument("--min-rotation", type=float, default=10.0,
                        help="Minimum camera rotation (deg) between saved frames (default: 10).")

    # --- error / statistics flags ---
    parser.add_argument("--add-statistics", action=argparse.BooleanOptionalAction, default=True,
                        help="Compute and store per-trajectory statistics (default: True).")
    parser.add_argument("--add-distance-error", action=argparse.BooleanOptionalAction, default=True,
                        help="Store camera centre distance error per frame (default: True).")
    parser.add_argument("--add-components-error", action=argparse.BooleanOptionalAction, default=True,
                        help="Store per-axis camera centre error components per frame (default: True).")
    parser.add_argument("--add-rotation-angle-error", action=argparse.BooleanOptionalAction, default=True,
                        help="Store full rotation angle error per frame (default: True).")
    parser.add_argument("--add-rotation-euler-error", action=argparse.BooleanOptionalAction, default=True,
                        help="Store per-axis Euler rotation error per frame (default: True).")

    # --- trajectory difference metric ---
    parser.add_argument("--chamfer-k-neighbor", type=int, default=1,
                        help="Number of nearest neighbours used for the directed pose Chamfer distance (default: 1).")
    parser.add_argument("--chamfer-translation-scale", default="aabb_diagonal",
                        choices=["aabb_diagonal", "scene_diameter"],
                        help="Scene scale used to normalise the translation component of the Chamfer distance (default: aabb_diagonal).")
    parser.add_argument("--chamfer-rotation-scale", type=float, default=180.0,
                        help="Maximum rotation angle (degrees) used to normalise the rotation component of the Chamfer distance (default: 180).")

    # --- camera model flags ---
    parser.add_argument("--use-fisheye-for-wfov", action=argparse.BooleanOptionalAction, default=True,
                        help="Use fisheye camera model for wide-FOV lenses (default: True).")

    return parser


def main():
    args = build_parser().parse_args()

    raw_data_pipeline = RawDataProcessingPipeline(supported_image_capture_devices=supported_image_capture_devices,
                                                  supported_video_capture_devices=supported_video_capture_devices)

    raw_data_pipeline.config_processing_pipeline(
        copy_images=args.copy_images,
        run_colmap=args.run_colmap,
        copy_point_cloud=args.copy_point_cloud,
        transform_world_coord=args.transform_world_coord,
        add_statistics=args.add_statistics,
        add_camera_center_distance_error=args.add_distance_error,
        add_camera_center_components_error=args.add_components_error,
        add_camera_rotation_angle_error=args.add_rotation_angle_error,
        add_camera_rotation_euler_error=args.add_rotation_euler_error,
        min_distance_m=args.min_distance,
        min_rot_degree=args.min_rotation,
        pos_covariance=args.pos_covariance,
        max_num_models=args.max_num_models,
        wfov_as_fisheye=args.use_fisheye_for_wfov,
        chamfer_k_neighbor=args.chamfer_k_neighbor,
        chamfer_translation_scale=args.chamfer_translation_scale,
        chamfer_rotation_scale=args.chamfer_rotation_scale,
    )

    raw_data_pipeline.configure_scene(
        scene_raw_dir=args.scene_raw_dir,
        scene_processed_dir=args.scene_processed_dir,
        scene_description_file_name=args.scene_description_file,
        scene_processed_json_file_name=args.scene_output_json,
        calibration_files_path=args.calibration_files_path,
        colmap_folder_name=args.colmap_folder,
        trajectories=args.trajectories,
    )

    raw_data_pipeline.process_scene_from_raw(debug_prints=True)

    raw_data_pipeline.print_trajectories_stats()


if __name__ == "__main__":
    main()



