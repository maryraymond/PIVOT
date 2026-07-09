# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

import argparse
from calibration.camera_calibration import (
    prepare_calibration_images,
    calibrate_camera_from_checkerboard,
    save_calibration,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run checkerboard camera calibration from images or a video file.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- input / output ---
    parser.add_argument("--input", required=True,
                        help="Folder of calibration images OR a video file.\n"
                             "If a video is given, frames are sampled and saved to an images/ subfolder.")
    parser.add_argument("--output", required=True,
                        help="Path for the output calibration JSON file.")

    # --- video sampling ---
    parser.add_argument("--sample-every-n-frames", type=int, default=30,
                        help="When input is a video, save one frame every N frames (default: 30).")
    parser.add_argument("--image-extension", default="JPG",
                        help="Image file extension to look for / save (default: JPG).")

    # --- checkerboard ---
    parser.add_argument("--pattern-cols", type=int, default=9,
                        help="Number of inner corners along the board columns (default: 9).")
    parser.add_argument("--pattern-rows", type=int, default=6,
                        help="Number of inner corners along the board rows (default: 6).")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="Physical size of one checkerboard square in metres (default: 0.025).")

    # --- calibration options ---
    parser.add_argument("--fisheye", action=argparse.BooleanOptionalAction, default=False,
                        help="Use fisheye (OPENCV_FISHEYE) calibration model (default: False).")
    parser.add_argument("--visualize", action=argparse.BooleanOptionalAction, default=False,
                        help="Show detected corners for each image (default: False).")

    return parser


def main():
    args = build_parser().parse_args()

    images_glob = prepare_calibration_images(
        input_path=args.input,
        sample_every_n_frames=args.sample_every_n_frames,
        image_extension=args.image_extension,
    )

    W, H, K, dist, rvecs, tvecs, rms = calibrate_camera_from_checkerboard(
        images_glob=images_glob,
        pattern_size=(args.pattern_cols, args.pattern_rows),
        square_size_m=args.square_size,
        visualize=args.visualize,
        K=None,
        fisheye=args.fisheye,
    )

    print("RMS reprojection error:", rms)
    print("K (intrinsics):\n", K)
    print("dist (distortion):\n", dist)

    save_calibration(H, W, K, dist, rms, args.output, args.fisheye)
    print(f"Calibration saved to {args.output}")


if __name__ == "__main__":
    main()
