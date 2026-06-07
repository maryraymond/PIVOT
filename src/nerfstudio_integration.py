from __future__ import annotations

import json
from typing import Dict, Any
import os
from pathlib import Path
import shutil
import numpy as np
import argparse
import random


def random_train_eval_ids(n_frames: int, train_percent: float = 0.9, seed: int = 42):
    ids = list(range(n_frames))  # frame IDs: 0 to N-1

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_train = round(n_frames * train_percent)

    train_ids = sorted(ids[:n_train])
    eval_ids = sorted(ids[n_train:])

    return train_ids, eval_ids

def update_config_with_indices(scene_config, scene_tranjectories):
    # get a set of traj in train and eval that have a percenatge set
    train_config = scene_config["train"]
    train_set_with_percentage = set()
    for traj in train_config.keys():
        if "percentage" in train_config[traj]:
            train_set_with_percentage.add(traj)

    eval_config = scene_config["eval"] if "eval" in scene_config else None

    # check if we have eval with percentage then we need to create a split
    for traj in train_set_with_percentage:
        # at least for train we need to set the IDs for the percentage
        train_percenatge = float(train_config[traj]["percentage"])
        train_ids, eval_ids = random_train_eval_ids(n_frames=scene_tranjectories[traj]["number_frames_in_traj"], 
                                                    train_percent=train_percenatge)
        train_config[traj]["indices"] = train_ids
        if eval_config is not None and traj in eval_config.keys():
            print(f"Setting the eval percentage for traj {traj} as {1-train_percenatge} %")
            eval_config[traj]["indices"] = eval_ids

    # update the overall config 
    scene_config["train"] = train_config
    scene_config["eval"] = eval_config

    return scene_config

def process_split(scene_dir:str, dst_dir:str, 
                  scene_tranjectories:Dict, split_config:Dict, 
                  images_dir:str, split_prefix:str,
                  debug_prints:bool=False):
    frames = []
    for tranjectory, traj_config in split_config.items():
        if tranjectory in scene_tranjectories:
            cam_intr_key = "camera_intrinsic_colmap" if traj_config["camera_intrinsics_optimized"] else "camera_intrinsic_calibration"
            alter_pose_key = "measured_pose_c2w" if traj_config["fill_missing_poses_with_non_optimized"] else None
            max_num_frames = int(traj_config["max_num_frames"]) if "max_num_frames" in traj_config else None
            ids = traj_config["indices"] if "indices" in traj_config else None

            if debug_prints:
                print(f"For {tranjectory} using c2w rot optimized = {traj_config['c2w_rot_optimized']}, \
                      c2w trans optimized = {traj_config['c2w_trans_optimized']}, alternative pose = {alter_pose_key} \
                      and camera intr = {cam_intr_key}")

            traj_data = scene_tranjectories[tranjectory]

            cam_intr = traj_data[cam_intr_key]

            traj_num_frames = len(traj_data["frames"])

            if ids is not None :
                print(f"Indices is set for {tranjectory} is will use indices")
                traj_frame_range = list(ids)
                max_num_frames = None
            elif max_num_frames is None:
                max_num_frames = -1
            
            if max_num_frames is not None:
                
                if max_num_frames == -1:
                    step = 1
                    traj_frame_range = list(range(0, traj_num_frames, step))
                else:
                    n = min(max_num_frames, traj_num_frames)
                    traj_frame_range = [
                        round(i * (traj_num_frames - 1) / (n - 1))
                        for i in range(n)
                    ]
                            
            for i in traj_frame_range:

                if i > traj_num_frames:
                    print(f"Warning idex {i} is out of range for {tranjectory} skipping")
                    continue
                
                traj_frame = traj_data["frames"][i]

                if "colmap_pose_c2w" in traj_frame:
                    c2w_opt = np.array(traj_frame["colmap_pose_c2w"])
                elif alter_pose_key is not None:
                    c2w_opt = np.array(traj_frame[alter_pose_key])
                else:
                    # This frame have no optimized pose and alternative pose is not enabled 
                    # we will need to skip this frame
                    if debug_prints:
                        print(f"Warning: skipping frames {traj_frame['file_name']} as it has no optimized pose and alternative measuered pose is not enabled")
                    continue

                c2w_measured = np.array(traj_frame["measured_pose_c2w"])

                c2w = np.eye(4)

                if traj_config["c2w_rot_optimized"]:
                    c2w[:3, :3] = c2w_opt[:3, :3]
                else:
                    c2w[:3, :3] = c2w_measured[:3, :3]

                if traj_config["c2w_trans_optimized"]:
                    c2w[:3, 3] = c2w_opt[:3, 3]
                else:
                    c2w[:3, 3] = c2w_measured[:3, 3]

                frame = dict(cam_intr)
                # change the camera type key name
                frame["camera_model"] = frame.pop("camera_type")
                src_file_name = Path(traj_frame["file_name"]).name
                src_file = scene_dir + f"/{traj_frame['file_name']}"
                dst_file_name = f"{images_dir}/{split_prefix}{tranjectory}_{src_file_name}"
                dst_file = dst_dir + "/" + dst_file_name

                frame["file_path"] = dst_file_name
                frame["transform_matrix"] = c2w.tolist()

                frames.append(frame)

                shutil.copyfile(src_file, dst_file)
        else:
            raise ValueError(f"The specified trajectory {tranjectory} is not found in the scene data at {scene_dir}")
    return frames


def create_ns_dataset_from_scene(scene_dir:str, scene_config:Dict, 
                                 dst_dir:str, debug_prints:bool=False):
    transforms_data = {}
    transforms_file = dst_dir + "/transforms.json"
    frames = []
    if "train" in scene_config:
        # check if we have specific eval trajectories 
        # If eval is set to None or missing it means we will default to
        # eval mode of fraction and all the images will be in the images folder
        images_dir = "images"
        os.makedirs(dst_dir + "/" + images_dir, exist_ok=True)
        if "eval" in scene_config and scene_config["eval"] is not None:
            # we have a defined train eval split
            eval_prefix = "eval_"
            train_prefix = "train_"
        else:
            eval_prefix = None
            train_prefix = ""

        # start copying and populating the training dataset
        # now we read the scene data 
        scene_data_file = scene_dir + "/scene_data.json"
        with open(scene_data_file, 'r') as f:
            scene_data = json.load(f)
            scene_tranjectories = scene_data["trajectories"]

        # update the scene config to set ids for the percentages
        updated_scene_config = update_config_with_indices(dict(scene_config),  scene_tranjectories)
        # now for trajectory we will read and populate the frames
        frames += process_split(scene_dir=scene_dir, dst_dir=dst_dir,
                                scene_tranjectories=scene_tranjectories, split_config=updated_scene_config["train"], 
                                split_prefix=train_prefix, images_dir=images_dir,
                                debug_prints=debug_prints)

        # see if we have a defined eval as well
        if eval_prefix is not None:
            frames += process_split(scene_dir=scene_dir, dst_dir=dst_dir,
                                    scene_tranjectories=scene_tranjectories, split_config=updated_scene_config["eval"], 
                                    split_prefix=eval_prefix, images_dir=images_dir,
                                    debug_prints=debug_prints)
        
        transforms_data["frames"] = frames

        with open(transforms_file, 'w') as f:
            json.dump(transforms_data, f, indent=4)
            
    else:
        raise ValueError("No train configuration available")


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
    print(f"debug_prints: {args.debug_prints}")

    print("\nParsed scene_config:")
    print(json.dumps(scene_config, indent=2))

    create_ns_dataset_from_scene(scene_dir=args.scene_dir, scene_config=scene_config,
                                 dst_dir=args.dst_dir, debug_prints=args.debug_prints)


if __name__ == "__main__":
    main()