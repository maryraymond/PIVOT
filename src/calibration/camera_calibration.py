
import glob
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

def prepare_calibration_images(input_path: str,
                               sample_every_n_frames: int = 30,
                               image_extension: str = "JPG") -> str:
    """
    Accepts either a folder of images or a video file.

    - Folder: returns a glob pattern matching images in that folder.
    - Video file: samples one frame every `sample_every_n_frames` frames,
      writes them as JPGs into an `images/` subfolder next to the video,
      and returns a glob pattern for that subfolder.

    Returns the glob pattern string to pass to calibrate_camera_from_checkerboard.
    """
    path = Path(input_path)

    if path.is_dir():
        glob_pattern = str(path / f"*.{image_extension}")
        if not glob.glob(glob_pattern):
            raise FileNotFoundError(f"No .{image_extension} images found in {path}")
        print(f"Using image folder: {path}")
        return glob_pattern

    if path.is_file():
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {path}")

        images_dir = path.parent / "images"
        os.makedirs(images_dir, exist_ok=True)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        print(f"Video: {path.name} | {total_frames} frames @ {fps:.1f} fps")
        print(f"Sampling every {sample_every_n_frames} frames → "
              f"~{total_frames // sample_every_n_frames} images")

        saved = 0
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_index % sample_every_n_frames == 0:
                out_path = images_dir / f"frame_{frame_index:06d}.{image_extension}"
                cv2.imwrite(str(out_path), frame)
                saved += 1
            frame_index += 1
        cap.release()

        print(f"Saved {saved} frames to {images_dir}")
        return str(images_dir / f"*.{image_extension}")

    raise ValueError(f"Input path is neither a folder nor a file: {input_path}")


def calibrate_camera_from_checkerboard(
    images_glob: str,
    pattern_size=(9, 6),
    square_size_m=0.025,
    visualize=False,
    K=None,
    fisheye=False,
):
    nx, ny = pattern_size

    objp = np.zeros((nx * ny, 3), np.float32)
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
    objp *= float(square_size_m)

    objpoints = []
    imgpoints = []

    images = sorted(glob.glob(images_glob))
    if not images:
        raise FileNotFoundError(f"No images found for glob: {images_glob}")

    print(f"Total number of available images = {len(images)}")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        40,
        1e-6,
    )

    image_size = None

    for fname in images:
        img = cv2.imread(fname)
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        current_size = (gray.shape[1], gray.shape[0])
        if image_size is None:
            image_size = current_size
            print(f"image size = {image_size}")
        elif current_size != image_size:
            print(f"[WARN] Skipping {fname}: size {current_size} != expected {image_size}")
            continue

        found, corners = cv2.findChessboardCorners(
            gray,
            (nx, ny),
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_ACCURACY
            ),
        )

        if not found:
            print(f"[WARN] No corners found: {fname}")
            continue

        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria,
        )

        if fisheye:
            objpoints.append(objp.reshape(1, -1, 3).astype(np.float64))
            imgpoints.append(corners_refined.reshape(1, -1, 2).astype(np.float64))
        else:
            objpoints.append(objp.astype(np.float32))
            imgpoints.append(corners_refined.astype(np.float32))

        if visualize:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, (nx, ny), corners_refined, found)
            vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
            plt.imshow(vis_rgb)
            plt.show()

    if len(objpoints) < 10:
        raise RuntimeError(f"Too few valid images ({len(objpoints)}). Aim for 15–30+.")

    print(f"Images with corners = {len(objpoints)}")

    if fisheye:
        K_init = np.zeros((3, 3), dtype=np.float64) if K is None else K.astype(np.float64)
        D_init = np.zeros((4, 1), dtype=np.float64)

        cal_flags = (
            cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            + cv2.fisheye.CALIB_CHECK_COND
            + cv2.fisheye.CALIB_FIX_SKEW
        )

        if K is not None:
            cal_flags += cv2.fisheye.CALIB_USE_INTRINSIC_GUESS

        rms, K_out, dist, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            image_size,
            K_init,
            D_init,
            flags=cal_flags,
            criteria=criteria,
        )

    else:
        cal_flags = cv2.CALIB_FIX_K3

        if K is not None:
            cal_flags += cv2.CALIB_USE_INTRINSIC_GUESS

        rms, K_out, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints,
            imgpoints,
            image_size,
            K,
            None,
            flags=cal_flags,
        )

    # Normalise dist to a flat 1-D array regardless of calibration path:
    #   cv2.calibrateCamera  → (1, 5): [k1, k2, p1, p2, k3]
    #   cv2.fisheye.calibrate → (4, 1): [k1, k2, k3, k4]
    dist_flat = dist.flatten()

    return image_size[0], image_size[1], K_out, dist_flat, rvecs, tvecs, rms

def save_calibration(H, W,
                     K, dist,  RMS, 
                     calibration_file, 
                     fisheye=False):
    fx = float(K[0, 0])
    cx = float(K[0, 2])
    fy = float(K[1, 1])
    cy = float(K[1, 2])

    # dist is guaranteed to be 1-D (flattened in calibrate_camera_from_checkerboard)
    # Non-fisheye: [k1, k2, p1, p2, k3]   (indices 0-4)
    # Fisheye:     [k1, k2, k3, k4]        (indices 0-3)
    d = dist.flatten()
    k1 = float(d[0])
    k2 = float(d[1])

    if fisheye:
        camera_type = "OPENCV_FISHEYE"
    else:
        camera_type = "OPENCV"

    cam_calib = {"camera_type": camera_type,
                 "h": H,
                 "w": W,
                 "fx": fx,
                 "fy": fy,
                 "cx": cx,
                 "cy": cy,
                 "k1": k1,
                 "k2": k2}

    if fisheye:
        k3 = float(d[2])
        k4 = float(d[3])
        cam_calib["k3"] = k3
        cam_calib["k4"] = k4
    else:
        p1 = float(d[2])
        p2 = float(d[3])
        k3 = float(d[4])
        cam_calib["p1"] = p1
        cam_calib["p2"] = p2
        cam_calib["k3"] = k3

    cam_calib["RMS"] = RMS

    with open(calibration_file, 'w') as f:
        json.dump(cam_calib, f, indent=4)