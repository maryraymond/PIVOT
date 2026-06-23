
import subprocess
import json
from typing import Dict, Callable
import os
import cv2

from .camera_metadata_abc import CameraImageMetaData, CameraVideoMetaData

def process_video_to_images(video_file, 
                            image_dir, 
                            create_image_metadata:Callable[[str, bool], CameraImageMetaData],
                            create_video_metadata:Callable[[str, bool], CameraVideoMetaData],
                            use_absloute_altitude=True,
                            frame_step=1, 
                            max_frame_number=None, 
                            use_constant_step=False, 
                            use_camera_movement=True, 
                            min_camera_distanct_m=0.3, 
                            min_camera_rot_deg=10, 
                            use_prev_if_value_missing=False, 
                            assume_missing_zero=True,
                            write_standard_gps=True, 
                            debug_prints=True, 
                            frame_name_fn=None):
    
    os.makedirs(image_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    frame_index = 0
    video_metadata =  create_video_metadata(video_file, use_absloute_altitude)
    prev_frame_metadata = None
    prev_saved_frame_metadata = None
    missing_values = 0
    
    while True:

        ret, frame = cap.read()
        save_frame = False

        if not ret:
            break

        if max_frame_number is not None:
            if frame_index >= max_frame_number:
                break
        
        # The meta data frame number starts from 1
        frame_metadata = video_metadata.get_frame_metadata(idx=frame_index+1)
        
        for key, value in frame_metadata.items():
            if value is None:
                if assume_missing_zero:
                    frame_metadata[key] = 0
                    if debug_prints:
                        print(f"Warning: frame: {frame_index} missing value: {key} assumed zero")
                elif prev_frame_metadata is not None and use_prev_if_value_missing and prev_frame_metadata[key] is not None:
                    frame_metadata[key] = prev_frame_metadata[key]
                    if debug_prints:
                        print(f"Warning: frame: {frame_index} missing value: {key} previous value used")
                missing_values += 1
        
        if use_constant_step and frame_index % frame_step == 0:
            save_frame = True
        elif use_camera_movement:
            if prev_saved_frame_metadata is None:
                # this is the first frame so we will save it anyway
                save_frame = True
            else:
                distance = video_metadata.get_distance_between_camera_centers(prev_saved_frame_metadata, frame_metadata)
                yaw_diff, pitch_diff, roll_diff = video_metadata.get_euler_diff_between_cameras(prev_saved_frame_metadata, frame_metadata)

                if distance >= min_camera_distanct_m or abs(yaw_diff) >= min_camera_rot_deg or \
                   pitch_diff >= min_camera_rot_deg or abs(roll_diff) >= min_camera_rot_deg:
                    save_frame = True
                    if debug_prints:
                        print(f"Saving frame with camera dist={distance}, yaw_diff={yaw_diff}, roll_diff={roll_diff}, pitch_diff={pitch_diff}")

        # We read all the meta data to keep track of missing params but only write when needed
        if save_frame:
            if frame_name_fn is None:
                file_name = f"frame_{frame_index:06d}.JPG"
            else:
                file_name = frame_name_fn(frame_index)
                
            frame_filename = f"{image_dir}/{file_name}"
            cv2.imwrite(frame_filename, frame)
            image_metadata = create_image_metadata(frame_filename, use_absloute_altitude)
            image_metadata.set_metadata(frame_metadata)
            # Write the gimble meta data
            image_metadata.write_all_camera_specific_tags()
            prev_saved_frame_metadata = frame_metadata
            
            if write_standard_gps:
                image_metadata.write_standard_gps_tags()

        prev_frame_metadata = frame_metadata

        frame_index += 1

    cap.release()

    # return the number of frames in the video and the number of missing data
    return frame_index+1, missing_values

