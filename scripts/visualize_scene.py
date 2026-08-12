# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.

import time
import argparse

from visualization.viser_visualization import SceneVisualization

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Launch the viser scene visualizer for a processed dataset scene.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--dataset-root", required=True,
                        help="Root directory containing processed scene folders.")
    parser.add_argument("--scene-name", required=True,
                        help="Name of the scene folder to visualize.")
    parser.add_argument("--port", type=int, default=8080,
                        help="Port for the viser server (default: 8080).")
    parser.add_argument("--trajectories", nargs="+", default=None,
                        metavar="TRAJ",
                        help="Subset of trajectory names to display. Displays all if omitted.")
    parser.add_argument("--frustums-visible", action=argparse.BooleanOptionalAction, default=False,
                        help="Show camera frustums by default on launch (default: False).")
    parser.add_argument("--max-frame-number", type=int, default=3000,
                        help="Maximum total number of camera frustums to render across all "
                             "shown trajectories, counting both the optimized/colmap and "
                             "measured layers combined (default: 3000). If the configured "
                             "--step would exceed this, the step is increased automatically.")
    parser.add_argument("--step", type=int, default=1,
                        help="Frame stride used when sampling camera frustums (default: 1, "
                             "i.e. every frame). May be increased automatically to stay under "
                             "--max-frame-number.")
    return parser


def main():
    args = build_parser().parse_args()

    print(f"Will run the viewer for {args.scene_name}")
    scene_vis = SceneVisualization(dataset_root=args.dataset_root,
                                   scene_name=args.scene_name,
                                   port=args.port,
                                   trajectories=args.trajectories)
    scene_vis.visualize_scene(frustums_visible_default=args.frustums_visible,
                              max_frame_number=args.max_frame_number,
                              step=args.step)

    report = scene_vis.get_frustum_subsampling_report()
    scene_report = report["scene"]
    configured_step = scene_report["configured_step"]
    used_step = scene_report["used_step"]
    if used_step != configured_step:
        step_note = f"used step={used_step} (adjusted from configured step={configured_step})"
    else:
        step_note = f"used step={used_step} (matches configured step)"
    print(f"Frustum subsampling: total frames across shown trajectories="
          f"{scene_report['total_frames_available']} (~{scene_report['total_frustums_estimate']} frustums "
          f"combining optimized + measured layers), max frame number={scene_report['max_frame_number']}, "
          f"{step_note}")
    for traj_name, pose_counts in report["trajectories"].items():
        for pose_type, counts in pose_counts.items():
            print(f"  [{traj_name}/{pose_type}] frames available={counts['num_frames_available']}, "
                  f"frames shown={counts['num_frames_shown']}")

    print("Press ctrl+c to stop the viewer")
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()
    