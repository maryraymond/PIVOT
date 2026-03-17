
import pycolmap
import numpy as np

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


if __name__ == "__main__":
    pass