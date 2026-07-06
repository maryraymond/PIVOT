
import json
from pathlib import Path
import torch
import cv2
import os
import numpy as np
from typing import Dict

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import SSIM


from utils.metrics import (
    get_traj_directed_chamfer_distance,
    get_pose_tr_distance,
    get_pose_t_distance,
    get_pose_r_distance,
    compute_scene_diameter,
    compute_aabb_diagonal,
)

from abc import ABC, abstractmethod


class ReconsModel(ABC):
    def __init__(self):
      pass
    
    @abstractmethod
    def get_generated_image(self, c2w, H, W, K, D, C=3, camera_model="OPENCV"):
       """ This function should return the image with pixel value 0 to 1
       The output shape should be H, W, C"""
       pass
    
    @abstractmethod
    def process_gt_image(self, gt_image):
       """This function should implement any gt manpulation like rescaling, undistorting and so on
       the input image pixel range is uint8 0 to 255 The output shape should be H, W, C"""
       pass
    
    @abstractmethod
    def get_ds_root(self):
       """Return the root path for the dataset that was used to train the model, if this information
       does not exist for this model return None"""
       pass

def calculate_eval_metrics(model:ReconsModel, eval_data:Dict, train_data:Dict,
                           ds_root:str, out_dir:str, normalization_scale: str = "aabb_diagonal",
                           debug=False):
    
    traj_metrics = {}
    ds_root = str(ds_root)
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    total_frame_number = 0
    traj_metrics["total_avrg"] = {"ssim":0.0,
                                    "lpips":0.0,
                                    "psnr": 0.0,
                                    "fid": 0.0
                                 }
    seen_traj_number_frames = 0
    traj_metrics["seen_traj_avrg"] = {"ssim":0.0,
                                      "lpips":0.0,
                                      "psnr": 0.0,
                                      "fid": 0.0
                                    }
    
    novel_traj_number_frames = 0
    traj_metrics["novel_traj_avrg"] = {"ssim":0.0,
                                      "lpips":0.0,
                                      "psnr": 0.0,
                                      "fid": 0.0
                                    }
    
    # Prepare all the training frames poses to be used for directed chamfer distance metric
    training_poses = []
    for train_data_traj in train_data.values():
        for frame in train_data_traj:
            training_poses.append(np.array(frame["transform_matrix"]).astype(np.float32))
    
    training_camera_centers = [train_pose[:3, 3] for train_pose in training_poses]
    camera_centers = list(training_camera_centers)

    for eval_data_traj in eval_data.values():
        for frame in eval_data_traj:
            camera_centers.append(np.array(frame["transform_matrix"])[:3, 3].astype(np.float32))

    training_diameter      = compute_scene_diameter(training_camera_centers)
    training_bbox_diameter = compute_aabb_diagonal(training_camera_centers)
    full_diameter          = compute_scene_diameter(camera_centers)
    full_bbox_diameter     = compute_aabb_diagonal(camera_centers)

    if normalization_scale == "scene_diameter":
        translation_scale = full_diameter
    else:  # "aabb_diagonal"
        translation_scale = full_bbox_diameter


    for traj in eval_data.keys():
        if debug:
            print(f"Evaluating for trajectory {traj}")
        gt_dir = f"{out_dir}/{traj}/gt"
        gen_dir = f"{out_dir}/{traj}/gen"
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        traj_metrics[traj] = {"ssim":0.0,
                            "lpips":0.0,
                            "psnr": 0.0,
                            "fid": 0.0
                            }

        eval_traj_poses = []
    
        frames = eval_data[traj]
    
        for frame in frames:
            
            c2w = np.array(frame["transform_matrix"]).astype(np.float32)
            eval_traj_poses.append(c2w)
        
            # Read the camera params
            camera_model = frame["camera_model"] if "camera_model" in frame else "OPENCV"

            fx=float(frame["fl_x"]) if "fl_x" in frame else 0.0
            fy=float(frame["fl_y"]) if "fl_y" in frame else 0.0
            cx=float(frame["cx"]) if "cx" in frame else 0.0
            cy=float(frame["cy"]) if "cy" in frame else 0.0

            K = np.eye(3)

            K[0, 0] = fx
            K[0, 2] = cx
            K[1, 1] = fy
            K[1, 2] = cy

            height=int(frame["h"])
            width=int(frame["w"])

            D = {}

            D["k1"] = float(frame["k1"]) if "k1" in frame else 0.0
            D["k2"] = float(frame["k2"]) if "k2" in frame else 0.0
            D["k3"] = float(frame["k3"]) if "k3" in frame else 0.0
            D["k4"] = float(frame["k4"]) if "k4" in frame else 0.0
            D["p1"] = float(frame["p1"]) if "p1" in frame else 0.0
            D["p2"] = float(frame["p2"]) if "p2" in frame else 0.0

            # get the generated image for this camera config
            # Note ReconsModel.get_generated_image need to be called before ReconsModel.process_gt_image to initialize any internal variables

            gen_image = model.get_generated_image(c2w=c2w, H=height, W=width,
                                                  K=K, D=D, camera_model=camera_model,
                                                  C=3)

            # Read th gt image and process it
            gt_image = cv2.imread(f"{ds_root}/{frame['file_path']}")
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)

            if gt_image.dtype ==  np.uint8:
                gt_image = gt_image.astype(np.float32)/255

            gt_image = model.process_gt_image(gt_image)

            image_name = frame['file_path'].split('/')[-1]
            # Save the images
            cv2.imwrite(f"{gen_dir}/{image_name}", (cv2.cvtColor(gen_image.astype(np.float32), cv2.COLOR_RGB2BGR) * 255))
            cv2.imwrite(f"{gt_dir}/{image_name}", (cv2.cvtColor(gt_image.astype(np.float32), cv2.COLOR_RGB2BGR)*255))
           
            
            # Reshape both images to the required shape for our metrics
            gen_image = torch.tensor(gen_image).permute(2, 0, 1).unsqueeze(0)
            gt_image = torch.tensor(gt_image).permute(2, 0, 1).unsqueeze(0)
            

            psnr_val = psnr(gt_image, gen_image).item()
            ssim_val = ssim(gt_image, gen_image).item()
            lpips_val = lpips(gt_image, gen_image).item()

            traj_metrics[traj]["ssim"] += ssim_val
            traj_metrics[traj]["lpips"] += lpips_val
            traj_metrics[traj]["psnr"]  += psnr_val


            traj_metrics["total_avrg"]["ssim"] += ssim_val
            traj_metrics["total_avrg"]["lpips"] += lpips_val
            traj_metrics["total_avrg"]["psnr"]  += psnr_val

            if traj in train_data.keys():
                traj_metrics["seen_traj_avrg"]["ssim"] += ssim_val
                traj_metrics["seen_traj_avrg"]["lpips"] += lpips_val
                traj_metrics["seen_traj_avrg"]["psnr"]  += psnr_val
            else:
                traj_metrics["novel_traj_avrg"]["ssim"] += ssim_val
                traj_metrics["novel_traj_avrg"]["lpips"] += lpips_val
                traj_metrics["novel_traj_avrg"]["psnr"]  += psnr_val

            if debug:
                print(f"psnr = {psnr_val}, ssim = {ssim_val}, lpips = {lpips_val}")
        
        traj_metrics[traj]["ssim"] /= len(frames)
        traj_metrics[traj]["lpips"] /= len(frames)
        traj_metrics[traj]["psnr"]  /= len(frames)

        traj_metrics[traj]["ssim"] =  round(traj_metrics[traj]["ssim"], 4)
        traj_metrics[traj]["lpips"] =  round(traj_metrics[traj]["lpips"], 4)
        traj_metrics[traj]["psnr"] =  round(traj_metrics[traj]["psnr"], 4)

        # Directed chamfer distance: eval trajectory → training poses, three variants
        chamfer_tr = get_traj_directed_chamfer_distance(
            traj_a_poses=eval_traj_poses,
            traj_b_poses=training_poses,
            pose_distance_fn=lambda pa, pb, _ts=translation_scale: get_pose_tr_distance(
                pose_a=pa, pose_b=pb, translation_scale=_ts, rotation_scale=180),
        )
        chamfer_t = get_traj_directed_chamfer_distance(
            traj_a_poses=eval_traj_poses,
            traj_b_poses=training_poses,
            pose_distance_fn=lambda pa, pb, _ts=translation_scale: get_pose_t_distance(
                pose_a=pa, pose_b=pb, translation_scale=_ts),
        )
        chamfer_r = get_traj_directed_chamfer_distance(
            traj_a_poses=eval_traj_poses,
            traj_b_poses=training_poses,
            pose_distance_fn=lambda pa, pb: get_pose_r_distance(
                pose_a=pa, pose_b=pb, rotation_scale=180),
        )

        traj_metrics[traj]["directed_chamfer_tr_norm"] = round(chamfer_tr, 4)
        traj_metrics[traj]["directed_chamfer_t_norm"]  = round(chamfer_t,  4)
        traj_metrics[traj]["directed_chamfer_r_norm"]  = round(chamfer_r,  4)

        total_frame_number += len(frames)
        
        if traj in train_data.keys():
            seen_traj_number_frames += len(frames)
            traj_metrics[traj]["trajectory_type"] = "seen_traj"
        else:
            novel_traj_number_frames += len(frames)
            traj_metrics[traj]["trajectory_type"] = "novel_traj"
            
    
    traj_metrics["chamfer_normalization"] = {
        "type": normalization_scale,
        "translation_scale": round(float(translation_scale), 4),
        "full_scene_diameter": round(float(full_diameter), 4),
        "full_bbox_diagonal": round(float(full_bbox_diameter), 4),
    }

    traj_metrics["total_avrg"]["ssim"] /= total_frame_number
    traj_metrics["total_avrg"]["lpips"] /= total_frame_number
    traj_metrics["total_avrg"]["psnr"]  /= total_frame_number

    traj_metrics["total_avrg"]["ssim"] = round(traj_metrics["total_avrg"]["ssim"], 4)
    traj_metrics["total_avrg"]["lpips"] = round(traj_metrics["total_avrg"]["lpips"], 4)
    traj_metrics["total_avrg"]["psnr"] = round(traj_metrics["total_avrg"]["psnr"], 4)


    if seen_traj_number_frames > 0:
        traj_metrics["seen_traj_avrg"]["ssim"] /= seen_traj_number_frames
        traj_metrics["seen_traj_avrg"]["lpips"] /= seen_traj_number_frames
        traj_metrics["seen_traj_avrg"]["psnr"]  /= seen_traj_number_frames

        traj_metrics["seen_traj_avrg"]["ssim"] = round(traj_metrics["seen_traj_avrg"]["ssim"], 4)
        traj_metrics["seen_traj_avrg"]["lpips"] = round(traj_metrics["seen_traj_avrg"]["lpips"], 4)
        traj_metrics["seen_traj_avrg"]["psnr"] = round(traj_metrics["seen_traj_avrg"]["psnr"], 4)
 

    if novel_traj_number_frames > 0:
        traj_metrics["novel_traj_avrg"]["ssim"] /= novel_traj_number_frames
        traj_metrics["novel_traj_avrg"]["lpips"] /= novel_traj_number_frames
        traj_metrics["novel_traj_avrg"]["psnr"]  /= novel_traj_number_frames

        traj_metrics["novel_traj_avrg"]["ssim"] = round(traj_metrics["novel_traj_avrg"]["ssim"], 4)
        traj_metrics["novel_traj_avrg"]["lpips"] = round(traj_metrics["novel_traj_avrg"]["lpips"], 4)
        traj_metrics["novel_traj_avrg"]["psnr"] = round(traj_metrics["novel_traj_avrg"]["psnr"], 4)

    return traj_metrics


def calculate_drone_ds_metrcis(model:ReconsModel, out_dir:str, eval_dataset=None,
                               normalization_scale: str = "aabb_diagonal", debug=False):
    """This function assumes that the dataset is following NS dataset transform.json file style
    If the dataset has been processed to some other format a custom function need to be implemented"""
    
    out_eval_file = f"{out_dir}/results.json"

    os.makedirs(out_dir, exist_ok=True)

    if eval_dataset is not None:
        ds_root = Path(eval_dataset)
    elif model.get_ds_root() is not None:
        ds_root = model.get_ds_root()
    else:
        raise ValueError("No eval dataset root is available")
    
    dataset_json = f"{ds_root}/transforms.json"

    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
        
        
    eval_data = {}
    train_data = {}

    # Populatte the training and eval data dicts
    for frame in dataset["frames"]:
        if "eval" in frame["file_path"]:
            traj = frame["file_path"].split('/')[-1].split("eval_")[-1].split('_frame')[0]
            eval_data.setdefault(traj, []).append(frame)
        elif "train" in frame["file_path"]:
            traj = frame["file_path"].split('/')[-1].split("train_")[-1].split('_frame')[0]
            train_data.setdefault(traj, []).append(frame)
            
    traj_metrics = calculate_eval_metrics(model=model,
                                          eval_data=eval_data,
                                          train_data=train_data, 
                                          ds_root=ds_root, 
                                          out_dir=out_dir, 
                                          normalization_scale=normalization_scale,
                                          debug=debug)


    with open (out_eval_file, 'w') as f:
        json.dump(traj_metrics, f, indent=4)