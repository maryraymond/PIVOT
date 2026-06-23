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
    return parser


def main():
    args = build_parser().parse_args()

    print(f"Will run the viewer for {args.scene_name}")
    scene_vis = SceneVisualization(dataset_root=args.dataset_root,
                                   scene_name=args.scene_name,
                                   port=args.port,
                                   trajectories=args.trajectories)
    scene_vis.visualize_scene(frustums_visible_default=args.frustums_visible)

    print("Press ctrl+c to stop the viewer")
    while True:
        time.sleep(10)

if __name__ == "__main__":
    main()
    