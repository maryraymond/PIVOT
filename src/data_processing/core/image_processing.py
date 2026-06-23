from typing import List,  Callable, Dict
from pathlib import Path

from .camera_metadata_abc import CameraImageMetaData

def frame_num(image_name:str):
    if image_name.split("_")[-1] == "D.JPG":
       frame_num = int(image_name.split("_")[-2])
    else:
        frame_num = int(image_name.split("_")[-1].split(".")[0])
    return frame_num

def read_images_data_from_folder(data_path:str, 
                                 create_image_metadata:Callable[[str, bool], CameraImageMetaData], 
                                 camera_id:int=1, 
                                 sorting_func=frame_num, 
                                 use_absloute_altitude=True,
                                 refrence_point = None)->List:

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

    
    if refrence_point is not None:
        lat_0, long_0, alt_0 = refrence_point
    else:
        image_metadata = create_image_metadata(images[0], use_absloute_altitude)
        lat_0, long_0, alt_0 = image_metadata.get_gps_value()

    images_data = []

    for image in images:
        image_metadata = create_image_metadata(image, use_absloute_altitude)
        image_metadata.set_ref_gps(longitude=long_0, latitude=lat_0, altitude=alt_0)
        pose_c2w = image_metadata.get_c2w_opengl()

        file_name = f"{image.parts[-2]}/{image.parts[-1]}" if subfolders_available else Path(image).name
        images_data.append({"camera_id":camera_id,
                            "file_name": file_name,
                            "pose_c2w":pose_c2w.tolist()})
    
    return images_data
    
def read_images_data_from_sub_folders(data_path:str, 
                                      create_image_metadata_map:Dict[str, Callable[[str, bool], CameraImageMetaData]],
                                      sorting_func=frame_num, 
                                      use_absloute_altitude=True)->List:
    
    folders = [item.name for item in Path(data_path).iterdir() if item.is_dir()]

    # read a refrence point
    images = sorted([file for file in Path(f"{data_path}/{folders[0]}").iterdir() if file.is_file() and ".JPG" in file.name], 
                    key=lambda x: sorting_func(x.name))
    
    image_metadata = create_image_metadata_map[folders[0]](images[0], use_absloute_altitude)
    lat_0, long_0, alt_0 = image_metadata.get_gps_value()

    #temp camera ID
    camera_id = 1

    images_per_folder = {}

    for folder in folders:
        #make sure it is not a colmap folder
        if "COLMAP" in folder:
            continue

        full_path = f"{data_path}/{folder}"
        folder_images = read_images_data_from_folder(full_path, 
                                                     create_image_metadata=create_image_metadata_map[folder],
                                                     use_absloute_altitude=use_absloute_altitude, 
                                                     camera_id=camera_id, 
                                                     refrence_point=(lat_0, long_0, alt_0))
        
        images_per_folder[folder] = folder_images
    
    return images_per_folder
