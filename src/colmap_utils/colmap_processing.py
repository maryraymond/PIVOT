
import pycolmap
import numpy as np
from typing import List, Callable
import subprocess
import os
from pathlib import Path
from enum import Enum

from data_processing.core.camera_metadata_abc import CameraImageMetaData
from data_processing.core.image_processing import read_images_data_from_folder
from data_processing.utils.processing_utils import get_cameras_data

from colmap_utils.colmap_conversion import write_colmap_cameras_txt, write_colmap_images_txt, convert_data_to_colmap

class CameraMode(Enum):
    SINGLE=pycolmap.CameraMode.SINGLE
    PER_FOLDER=pycolmap.CameraMode.PER_FOLDER
    PER_IMAGE=pycolmap.CameraMode.PER_IMAGE 

def update_db_pose_prior(colmap_data_base:str, images_data:List, pose_covar:List, 
                         update_position=True, cartesian_system=True):
    db = pycolmap.Database().open(colmap_data_base)
    db_images = db.read_all_images()
    # build the image name to ID map
    image_to_id = {image.name: image.image_id for image in db_images}

    for image_data in images_data:
        pose_prior = pycolmap.PosePrior()
        # check if we will update the position as well or keep the value as read
        if update_position:
            xyz = (np.array(image_data["pose_c2w"])[:3, 3]).tolist()
            pose_prior.position = xyz
        # check which coordinate system to set
        if cartesian_system:
            pose_prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        else:
            pose_prior.coordinate_system = pycolmap.PosePriorCoordinateSystem.WGS84
        pose_prior.position_covariance = np.diag(pose_covar)
        db.update_pose_prior(image_to_id[image_data["file_name"]], pose_prior)
    db.close()

def get_traj_name(image_name):
    return image_name.split("/")[0]

def get_camera_to_traj_map(db_images):
    cam_id_map = {}
    camera_set = set()
    for image in db_images:
        if image.camera_id in camera_set:
            continue
        else:
            camera_set.add(image.camera_id)
            cam_id_map[image.camera_id] = get_traj_name(image.name)
    
    return cam_id_map

def update_cameras_type(colmap_db, folder_to_cam):
    db = pycolmap.Database().open(colmap_db)
    db_images = db.read_all_images()
    db_cameras = db.read_all_cameras()

    cam_id_map = get_camera_to_traj_map(db_images)

    for camera in db_cameras:
        camera.model = folder_to_cam[cam_id_map[camera.camera_id]]
        db.update_camera(camera)

def convert_model_txt(input_path, output_path=None):
    if output_path is None:
        output_path = input_path
    
    cmd = ["colmap", "model_converter", "--input_path", input_path,
            "--output_path", output_path, "--output_type", "TXT"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result

def convert_model_ply(input_path, output_path=None):
    if output_path is None:
        output_path = input_path
    
    cmd = ["colmap", "model_converter", "--input_path", input_path,
            "--output_path", f"{output_path}/sparse_model.ply", "--output_type", "PLY"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result

def run_colmap(dataset_images_dir, colmap_dir, 
               camera_mode:CameraMode=CameraMode.SINGLE, max_num_models=3,
               camera_model="OPENCV", input_path='', camera_model_map=None):
    colmap_db = colmap_dir / "database.db"
    colmap_out_dir = colmap_dir / "sparse"

    os.makedirs(colmap_dir, exist_ok=True)

    image_reader_options = pycolmap.ImageReaderOptions(camera_model=camera_model)
    #Run feature extraction and force the camera to OPENCV
    pycolmap.extract_features(
        database_path=colmap_db,
        image_path=dataset_images_dir,
        camera_mode=camera_mode.value,
        camera_model = camera_model,
        reader_options=image_reader_options,
        input_path=input_path
    )

    if camera_model_map is not None:
        if camera_mode != CameraMode.PER_FOLDER:
            raise ValueError("The camera model map modification is only supoorted for camera mode Per Foler")
        update_cameras_type(colmap_db=colmap_db, folder_to_cam=camera_model_map)

    pycolmap.match_exhaustive(database_path=colmap_db)

    pipeline_options = pycolmap.IncrementalPipelineOptions(max_num_models=max_num_models)

    maps = pycolmap.incremental_mapping(
        database_path=colmap_db,
        image_path=dataset_images_dir,
        output_path=colmap_out_dir,
        options=pipeline_options)

    sparse_models = [folder for folder in Path(colmap_out_dir).iterdir() if folder.is_dir()] 
    for sparse_model in sparse_models:
        convert_model_txt(str(sparse_model))
        convert_model_ply(str(sparse_model))

def run_colmap_with_initialization(dataset_images_dir, 
                                   create_image_metadata:Callable[[str, bool], CameraImageMetaData],
                                   camera_calibration_file,
                                   colmap_dir,
                                   use_absloute_altitude=True,
                                   camera_mode:CameraMode=CameraMode.SINGLE,
                                   camera_model="OPENCV", camera_model_map=None):
    
    colmap_db = colmap_dir / "database.db"
    colmap_out_dir = colmap_dir / "sparse"
    colmap_init_dir = colmap_dir / "sparse_init"
    colmap_init_images = colmap_init_dir / "images.txt"
    colmap_init_cameras = colmap_init_dir / "cameras.txt"
    colmap_init_points = colmap_init_dir / "points3D.txt"

    cameras_data = get_cameras_data(cal_file=camera_calibration_file)
    images_data = read_images_data_from_folder(dataset_images_dir, 
                                               create_image_metadata=create_image_metadata, 
                                               use_absloute_altitude=use_absloute_altitude)

    os.makedirs(colmap_dir, exist_ok=True)

    image_reader_options = pycolmap.ImageReaderOptions(camera_model=camera_model)
    #Run feature extraction and force the camera to OPENCV
    pycolmap.extract_features(
        database_path=colmap_db,
        image_path=dataset_images_dir,
        camera_mode=camera_mode.value,
        camera_model = camera_model,
        reader_options=image_reader_options
    )

    if camera_model_map is not None:
        if camera_mode != CameraMode.PER_FOLDER:
            raise ValueError("The camera model map modification is only supoorted for camera mode Per Foler")
        update_cameras_type(colmap_db=colmap_db, folder_to_cam=camera_model_map)

    pycolmap.match_exhaustive(database_path=colmap_db)

    os.makedirs(colmap_init_dir, exist_ok=True)
    # Write the camera calibration in camera.txt
    write_colmap_cameras_txt(cameras_data, colmap_init_cameras)
    # Convert the images data to colmap formate (specificaly the camera extrinsic)
    images_data_colmap = convert_data_to_colmap(images_data)
    # Write the images.txt (Note: the colmap database hase to exist for this to work 
    # to be able to read the image name to image id mapping)
    write_colmap_images_txt(images_data_colmap, 
                            colmap_database=colmap_db, 
                            images_txt=colmap_init_images)
    
    # create an empty file for 3D points
    with open(colmap_init_points, "w"):
        pass
    
    os.makedirs(colmap_out_dir, exist_ok=True)
    # create an initial reconstrcution for the initial model values
    rec = pycolmap.Reconstruction(colmap_init_dir)
    # Run point Triangulation to find the 3D points
    new_rec = pycolmap.triangulate_points(
        reconstruction=rec,
        database_path=colmap_db,
        image_path=dataset_images_dir,
        output_path=colmap_out_dir/"0"
    )
    # Run bundle adjustment to optimize the initial model
    pycolmap.bundle_adjustment(new_rec)
        
    sparse_models = [folder for folder in Path(colmap_out_dir).iterdir() if folder.is_dir()] 
    for sparse_model in sparse_models:
        convert_model_txt(str(sparse_model))
        convert_model_ply(str(sparse_model))

def pose_prior_mapping(database_path, image_path, 
                       output_path, max_num_models=3,
                       input_path=''):
    os.makedirs(output_path, exist_ok=True)

    cmd = ["colmap", "pose_prior_mapper", 
              "--database_path", str(database_path), 
              "--image_path", str(image_path),
              "--output_path", str(output_path), 
              "--Mapper.max_num_models", str(max_num_models)]
    if input_path != '':
        cmd.append("--input_path")
        cmd.append(str(input_path))

    result = subprocess.run(cmd, capture_output=True, text=True)

    return result

def run_colmap_with_soft_priors(dataset_images_dir,
                                create_image_metadata:Callable[[str, bool], CameraImageMetaData], 
                                colmap_dir, 
                                use_absloute_altitude=True,
                                max_num_models=3,
                                camera_mode:CameraMode=CameraMode.SINGLE, 
                                pos_var = None,
                                update_positions=True, 
                                cartesian_system=True, 
                                camera_model="OPENCV",
                                input_path='', 
                                sparse_dir_name="sparse", 
                                camera_model_map=None):
    
    colmap_db = colmap_dir / "database.db"
    colmap_out_dir = colmap_dir / sparse_dir_name
    os.makedirs(colmap_dir, exist_ok=True)

    image_reader_options = pycolmap.ImageReaderOptions(camera_model=camera_model)
    #Run feature extraction and force the camera to OPENCV
    pycolmap.extract_features(
        database_path=colmap_db,
        image_path=dataset_images_dir,
        camera_mode=camera_mode.value,
        camera_model = camera_model,
        reader_options=image_reader_options
    )

    if camera_model_map is not None:
        if camera_mode != CameraMode.PER_FOLDER:
            raise ValueError("The camera model map modification is only supoorted for camera mode Per Foler")
        update_cameras_type(colmap_db=colmap_db, folder_to_cam=camera_model_map)

    #Run exhaustive matcher
    pycolmap.match_exhaustive(database_path=colmap_db)

    #Read the Images data to load the positions in NED coordinate system
    images_data = read_images_data_from_folder(dataset_images_dir, 
                                               create_image_metadata=create_image_metadata, 
                                               use_absloute_altitude=use_absloute_altitude)

    if pos_var is None:
        pos_var = [4, 4, 4]

    #Update the db as required
    update_db_pose_prior(colmap_db, images_data, pos_var,
                        update_position=update_positions, cartesian_system=cartesian_system)

    # run the pose prior mapping                    
    res = pose_prior_mapping(colmap_db, dataset_images_dir, 
                             colmap_out_dir, max_num_models=max_num_models,
                             input_path=input_path)
    print(res.stderr)
    print(res.stdout)

    sparse_models = [folder for folder in Path(colmap_out_dir).iterdir() if folder.is_dir()] 
    for sparse_model in sparse_models:
        convert_model_txt(str(sparse_model))
        convert_model_ply(str(sparse_model))
    
