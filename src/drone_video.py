
import subprocess
import json
from typing import Dict
import os
import cv2

from drone_core import get_distance_between_camera_centers, get_rotation_diff_between_cameras

def read_video_metadata_exiftool(video_path:str) -> dict:
    cmd = ["exiftool", "-ee", "-u", "-j", "-G3", "-a", "-n", video_path]
    out = subprocess.check_output(cmd, text=True)
    return json.loads(out)[0]


def write_gimble_tags(metadata, image_file):
    cmd = ["exiftool", "-overwrite_original",
           f"-XMP-drone-dji:GimbalYawDegree={metadata['XMP-drone-dji:GimbalYawDegree']}",
           f"-XMP-drone-dji:GimbalPitchDegree={metadata['XMP-drone-dji:GimbalPitchDegree']}",
           f"-XMP-drone-dji:GimbalRollDegree={metadata['XMP-drone-dji:GimbalRollDegree']}",
           f"{image_file}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def write_gps_tags(metadata, image_file):
    cmd = ["exiftool", "-overwrite_original",
           f"-XMP-drone-dji:GPSLatitude={metadata['XMP-drone-dji:GPSLatitude']}",
           f"-XMP-drone-dji:GPSLongitude={metadata['XMP-drone-dji:GPSLongitude']}",
           f"-XMP-drone-dji:AbsoluteAltitude={metadata['XMP-drone-dji:AbsoluteAltitude']}",
           f"-XMP-drone-dji:RelativeAltitude={metadata['XMP-drone-dji:RelativeAltitude']}",
           f"{image_file}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def write_standard_gps_tags(metadata, image_file):
    latitudeRef = 'N' if metadata['XMP-drone-dji:GPSLatitude'] > 0 else 'S'
    longitudeRef = 'E' if metadata['XMP-drone-dji:GPSLongitude'] > 0 else 'W'
    altitudeRef = 0 if metadata['XMP-drone-dji:AbsoluteAltitude'] > 0 else 1

    cmd = ["exiftool", "-overwrite_original",
           f"-GPS:GPSLatitudeRef={latitudeRef}",
           f"-GPS:GPSLatitude={abs(metadata['XMP-drone-dji:GPSLatitude'])}",
           f"-GPS:GPSLongitudeRef={longitudeRef}",
           f"-GPS:GPSLongitude={abs(metadata['XMP-drone-dji:GPSLongitude'])}",
           f"-GPS:GPSAltitudeRef={altitudeRef}",
           f"-GPS:GPSAltitude={abs(metadata['XMP-drone-dji:AbsoluteAltitude'])}",
           f"{image_file}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def write_drone_tags(metadata, image_file):
    cmd = ["exiftool", "-overwrite_original",
           f"-XMP-drone-dji:FlightYawDegree={metadata['XMP-drone-dji:FlightYawDegree']}",
           f"-XMP-drone-dji:FlightPitchDegree={metadata['XMP-drone-dji:FlightPitchDegree']}",
           f"-XMP-drone-dji:FlightRollDegree={metadata['XMP-drone-dji:FlightRollDegree']}",
           f"{image_file}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def write_all_drone_tags(metadata, image_file):
    cmd = ["exiftool", "-overwrite_original",
           f"-XMP-drone-dji:GimbalYawDegree={metadata['XMP-drone-dji:GimbalYawDegree']}",
           f"-XMP-drone-dji:GimbalPitchDegree={metadata['XMP-drone-dji:GimbalPitchDegree']}",
           f"-XMP-drone-dji:GimbalRollDegree={metadata['XMP-drone-dji:GimbalRollDegree']}",
           f"-XMP-drone-dji:GPSLatitude={metadata['XMP-drone-dji:GPSLatitude']}",
           f"-XMP-drone-dji:GPSLongitude={metadata['XMP-drone-dji:GPSLongitude']}",
           f"-XMP-drone-dji:AbsoluteAltitude={metadata['XMP-drone-dji:AbsoluteAltitude']}",
           f"-XMP-drone-dji:RelativeAltitude={metadata['XMP-drone-dji:RelativeAltitude']}",
           f"-XMP-drone-dji:FlightYawDegree={metadata['XMP-drone-dji:FlightYawDegree']}",
           f"-XMP-drone-dji:FlightPitchDegree={metadata['XMP-drone-dji:FlightPitchDegree']}",
           f"-XMP-drone-dji:FlightRollDegree={metadata['XMP-drone-dji:FlightRollDegree']}",
           f"{image_file}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


def get_frame_metadata(exif_video_data:Dict, frame_number:int):
    frame_data = {"XMP-drone-dji:GPSLatitude": exif_video_data.get(f"Doc{str(frame_number)}:GPSLatitude", None),
                  "XMP-drone-dji:GPSLongitude":exif_video_data.get(f"Doc{str(frame_number)}:GPSLongitude", None),
                  "XMP-drone-dji:AbsoluteAltitude":exif_video_data.get(f"Doc{str(frame_number)}:AbsoluteAltitude", None),
                  "XMP-drone-dji:RelativeAltitude":exif_video_data.get(f"Doc{str(frame_number)}:RelativeAltitude", None),
                  "XMP-drone-dji:FlightYawDegree":exif_video_data.get(f"Doc{str(frame_number)}:DroneYaw", None),
                  "XMP-drone-dji:FlightPitchDegree":exif_video_data.get(f"Doc{str(frame_number)}:DronePitch", None),
                  "XMP-drone-dji:FlightRollDegree":exif_video_data.get(f"Doc{str(frame_number)}:DroneRoll", None),
                  "XMP-drone-dji:GimbalYawDegree":exif_video_data.get(f"Doc{str(frame_number)}:GimbalYaw", None),
                  "XMP-drone-dji:GimbalPitchDegree":exif_video_data.get(f"Doc{str(frame_number)}:GimbalPitch", None),
                  "XMP-drone-dji:GimbalRollDegree":exif_video_data.get(f"Doc{str(frame_number)}:GimbalRoll", None)}
    return frame_data

def process_video_to_images(video_file, image_dir, 
                            frame_step=1, max_frame_number=None, use_constant_step=False, 
                            use_camera_movement=True, min_camera_distanct_m=0.3, min_camera_rot_deg=10, 
                            use_prev_if_value_missing=False, assume_missing_zero=True,
                            write_standard_gps=True, debug_prints=True):
    os.makedirs(image_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_file)
    frame_index = 0
    video_metadata =  read_video_metadata_exiftool(video_file)
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
        frame_metadata = get_frame_metadata(video_metadata, frame_number=frame_index+1)
        
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
                # this is the frame so we will save it anyway
                save_frame = True
            else:
                distance = get_distance_between_camera_centers(prev_saved_frame_metadata, frame_metadata)
                yaw_diff, pitch_diff, roll_diff = get_rotation_diff_between_cameras(prev_saved_frame_metadata, frame_metadata)

                if distance >= min_camera_distanct_m or yaw_diff >= min_camera_rot_deg or \
                   pitch_diff >= min_camera_rot_deg or roll_diff >= min_camera_rot_deg:
                    save_frame = True
                    if debug_prints:
                        print(f"Saving frame with camera dist={distance}, yaw_diff={yaw_diff}, roll_diff={roll_diff}, pitch_diff={pitch_diff}")

        # We read all the meta data to keep track of missing params but only write when needed
        if save_frame:
            frame_filename = f"{image_dir}/frame_{frame_index:06d}.JPG"
            cv2.imwrite(frame_filename, frame)
            # Write the gimble meta data
            write_all_drone_tags(frame_metadata, frame_filename)
            prev_saved_frame_metadata = frame_metadata
            
            if write_standard_gps:
                write_standard_gps_tags(frame_metadata, frame_filename)

        prev_frame_metadata = frame_metadata

        frame_index += 1

    cap.release()

    # return the number of frames in the video and the number of missing data
    return frame_index+1, missing_values


if __name__ == "__main__":
    pass