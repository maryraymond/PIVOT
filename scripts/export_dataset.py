# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.


from __future__ import annotations

import json
from typing import Dict, Any
from pathlib import Path
import argparse
from dataset_export.ns_dataset_creation import create_ns_dataset_from_scene


def parse_scene_config(scene_config_arg: str) -> Dict[str, Any]:
    """
    Parse scene config either from:
    1. A JSON string
    2. A JSON file path
    """

    if "{" in scene_config_arg:
        # try parsing directly as JSON string
        try:
            return json.loads(scene_config_arg)
        except json.JSONDecodeError as e:
            raise ValueError(
                "scene_config must either be:\n"
                "- a valid JSON string\n"
                "- or a path to a JSON file"
            ) from e
    else:
        # Try loading as file first
        possible_path = Path(scene_config_arg)

        if possible_path.exists():
            with open(possible_path, "r", encoding="utf-8") as f:
                return json.load(f)

   


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a multi-trajectory drone scene configuration "
            "for Nerfstudio processing."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--scene-dir",
        type=str,
        required=True,
        help=(
            "Path to the processed scene directory.\n"
            "Example:\n"
            "  --scene-dir /data/processed/backyard_2"
        ),
    )

    parser.add_argument(
        "--dst-dir",
        type=str,
        required=True,
        help=(
            "Destination directory where the processed output will be written.\n"
            "Example:\n"
            "  --dst-dir /data/ns_processed/backyard_2_t2"
        ),
    )

    parser.add_argument(
        "--scene-config",
        type=str,
        required=True,
        help=(
            "Scene configuration as either:\n"
            "1. Path to a JSON file\n"
            "2. Inline JSON string\n\n"
            "File example:\n"
            "  --scene-config configs/backyard_2.json\n\n"
            "Inline example:\n"
            "  --scene-config '{\"train\": {\"traj_1\": {\"c2w_pose_optimized\": true}}}'"
        ),
    )

    parser.add_argument(
        "--use-sparse-pc",
        action="store_true",
        help="Copy sparse_model.ply from the scene dir and add it to transforms.json.",
    )

    parser.add_argument(
        "--debug-prints",
        action="store_true",
        help="Enable verbose debug printing.",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    scene_config = parse_scene_config(args.scene_config)

    print("\nParsed arguments:")
    print(f"scene_dir: {args.scene_dir}")
    print(f"dst_dir: {args.dst_dir}")
    print(f"use_sparse_pc: {args.use_sparse_pc}")
    print(f"debug_prints: {args.debug_prints}")

    print("\nParsed scene_config:")
    print(json.dumps(scene_config, indent=2))

    create_ns_dataset_from_scene(scene_dir=args.scene_dir, scene_config=scene_config,
                                 dst_dir=args.dst_dir, use_sparse_pc=args.use_sparse_pc,
                                 debug_prints=args.debug_prints)


if __name__ == "__main__":
    main()