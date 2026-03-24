
import pycolmap
import numpy as np
from typing import List
import subprocess
import os

from drone_core import read_images_data_from_folder, get_cameras_data
from colmap_conversion import write_colmap_cameras_txt, write_colmap_images_txt, convert_data_to_colmap

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


def convert_model_txt(input_path, output_path=None):
    if output_path is None:
        output_path = input_path
    
    cmd = ["colmap", "model_converter", "--input_path", input_path,
            "--output_path", output_path, "--output_type", "TXT"]
    result = subprocess.run(cmd, capture_output=True, text=True)

    return result

def run_colmap(dataset_images_dir, colmap_dir):
    colmap_db = colmap_dir / "database.db"
    colmap_out_dir = colmap_dir / "sparse"

    os.makedirs(colmap_dir, exist_ok=True)

    image_reader_options = pycolmap.ImageReaderOptions(camera_model="OPENCV")
    #Run feature extraction and force the camera to OPENCV
    pycolmap.extract_features(
        database_path=colmap_db,
        image_path=dataset_images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model = "OPENCV",
        reader_options=image_reader_options
    )

    pycolmap.match_exhaustive(database_path=colmap_db)

    maps = pycolmap.incremental_mapping(
        database_path=colmap_db,
        image_path=dataset_images_dir,
        output_path=colmap_out_dir)
        
    convert_model_txt(str(colmap_out_dir/"0"))


def run_colmap_with_initialization(dataset_images_dir, camera_calibration_file, colmap_dir):
    colmap_db = colmap_dir / "database.db"
    colmap_out_dir = colmap_dir / "sparse"
    colmap_init_dir = colmap_dir / "sparse_init"
    colmap_init_images = colmap_init_dir / "images.txt"
    colmap_init_cameras = colmap_init_dir / "cameras.txt"
    colmap_init_points = colmap_init_dir / "points3D.txt"

    cameras_data = get_cameras_data(cal_file=camera_calibration_file)
    images_data, _, _ = read_images_data_from_folder(dataset_images_dir)

    os.makedirs(colmap_dir, exist_ok=True)

    image_reader_options = pycolmap.ImageReaderOptions(camera_model="OPENCV")
    #Run feature extraction and force the camera to OPENCV
    pycolmap.extract_features(
        database_path=colmap_db,
        image_path=dataset_images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model = "OPENCV",
        reader_options=image_reader_options
    )

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
        
    convert_model_txt(str(colmap_out_dir/"0"))


def pose_prior_mapping(database_path, image_path, output_path):
    os.makedirs(output_path, exist_ok=True)

    cmd = ["colmap", "pose_prior_mapper", 
              "--database_path", str(database_path), 
              "--image_path", str(image_path),
              "--output_path", str(output_path)]

    result = subprocess.run(cmd, capture_output=True, text=True)

    return result


def run_colmap_with_soft_priors(dataset_images_dir, colmap_dir, pos_var = None,
                                update_positions=True, cartesian_system=True):
    colmap_db = colmap_dir / "database.db"
    colmap_out_dir = colmap_dir / "sparse"
    os.makedirs(colmap_dir, exist_ok=True)

    image_reader_options = pycolmap.ImageReaderOptions(camera_model="OPENCV")
    #Run feature extraction and force the camera to OPENCV
    pycolmap.extract_features(
        database_path=colmap_db,
        image_path=dataset_images_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model = "OPENCV",
        reader_options=image_reader_options
    )

    #Run exhaustive matcher
    pycolmap.match_exhaustive(database_path=colmap_db)

    #Read the Images data to load the positions in NED coordinate system
    images_data, _, _ = read_images_data_from_folder(dataset_images_dir)

    if pos_var is None:
        pos_var = [4, 4, 4]

    #Update the db as required
    update_db_pose_prior(colmap_db, images_data, pos_var,
                        update_position=update_positions, cartesian_system=cartesian_system)

    # run the pose prior mapping                    
    res = pose_prior_mapping(colmap_db, dataset_images_dir, colmap_out_dir)
    print(res.stderr)
    print(res.stdout)

    convert_model_txt(str(colmap_out_dir/"0"))
    
if __name__ == "__main__":
    pass