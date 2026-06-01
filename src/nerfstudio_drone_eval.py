
import json
from pathlib import Path
import torch
import cv2
import os
import numpy as np

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from pytorch_msssim import SSIM

from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras import camera_utils

def get_data_transforms(transforms_file):

    R = np.eye(4)
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)

        R[:3, :4] = np.array(transforms["transform"])
        scale = float(transforms["scale"])

    return R, scale


def calculate_eval_metrics(eval_data, model, model_dataparser_transforms, 
                           ds_root, out_dir, compose_background=False):
    traj_metrics = {}
    ds_root = str(ds_root)
    R, scale = get_data_transforms(model_dataparser_transforms)
    psnr = PeakSignalNoiseRatio(data_range=1.0)
    ssim = SSIM(data_range=1.0, size_average=True, channel=3)
    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

    for traj in eval_data.keys():

        frames = eval_data[traj]
        c2ws = []
        fx = []
        fy = []
        cx = []
        cy = []
        height = []
        width = []
        distort = []

        gt_dir = f"{out_dir}/{traj}/gt"
        gen_dir = f"{out_dir}/{traj}/gen"
        os.makedirs(gen_dir, exist_ok=True)
        os.makedirs(gt_dir, exist_ok=True)

        traj_metrics[traj] = {"ssim":0.0,
                            "lpips":0.0,
                            "psnr": 0.0,
                            "fid": 0.0
                            }

        for frame in frames:
            
            gt_image = cv2.imread(f"{ds_root}/{frame['file_path']}")
            # Save the image to the gt dir
            image_name = frame['file_path'].split('/')[-1]
            cv2.imwrite(f"{gt_dir}/{image_name}", gt_image)

            # now prepare the image for the metrics comparison
            gt_image = cv2.cvtColor(gt_image, cv2.COLOR_BGR2RGB)/255
            # gt_image = torch.tensor(gt_image, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
            gt_image = torch.tensor(gt_image, dtype=torch.float32)

            # BUILD UP THE CAMERA AND THE POSES
            if "camera_model" not in frame:
                camera_type = CameraType.PERSPECTIVE
            elif frame["camera_model"] == "OPENCV":
                camera_type = CameraType.PERSPECTIVE
            elif frame["camera_model"] == "OPENCV_FISHEYE":
                camera_type = CameraType.FISHEYE
            

            c2w = torch.tensor(frame["transform_matrix"]).view(4, 4).numpy()
            c2w = R @ c2w 
            c2w *=  scale
            c2w = torch.tensor(c2w)
            c2w = c2w[:3].unsqueeze(0).to(dtype=torch.float32)

            fx=float(frame["fl_x"])
            fy=float(frame["fl_y"])
            cx=float(frame["cx"])
            cy=float(frame["cy"])
            height=int(frame["h"])
            width=int(frame["w"])

            # read the distaortions
            distort.append(camera_utils.get_distortion_params(
                            k1=float(frame["k1"]) if "k1" in frame else 0.0,
                            k2=float(frame["k2"]) if "k2" in frame else 0.0,
                            k3=float(frame["k3"]) if "k3" in frame else 0.0,
                            k4=float(frame["k4"]) if "k4" in frame else 0.0,
                            p1=float(frame["p1"]) if "p1" in frame else 0.0,
                            p2=float(frame["p2"]) if "p2" in frame else 0.0,
                        ))
            
            camera = Cameras(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                distortion_params=distort,
                height=height,
                width=width,
                camera_to_worlds=c2w[:, :3, :4],
                camera_type=camera_type,
            )

            with torch.no_grad():
                gen_image = model.get_outputs_for_camera(camera=camera)

            if compose_background:
                gt_image = model.composite_with_background(gt_image, gen_image["background"])

            gt_image = gt_image.permute(2, 0, 1).unsqueeze(0)


            rgb = gen_image['rgb'].cpu()
            # save the image to the gen folder 
            cv2.imwrite(f"{gen_dir}/{image_name}", (cv2.cvtColor(rgb.numpy(), cv2.COLOR_RGB2BGR) * 255))
            rgb = rgb.permute(2, 0, 1).unsqueeze(0)

            psnr_val = psnr(gt_image, rgb).item()
            ssim_val = ssim(gt_image, rgb).item()
            lpips_val = lpips(gt_image, rgb).item()

            traj_metrics[traj]["ssim"] += ssim_val
            traj_metrics[traj]["lpips"] += lpips_val
            traj_metrics[traj]["psnr"]  += psnr_val


            print(f"psnr = {psnr_val}, ssim = {ssim_val}, lpips = {lpips_val}")
        
        traj_metrics[traj]["ssim"] /= len(frames)
        traj_metrics[traj]["lpips"] /= len(frames)
        traj_metrics[traj]["psnr"]  /= len(frames)

    return traj_metrics


def calculate_eval_metrics_avrg(traj_metrics):
    traj_metrics["avrg"] = {"ssim":0.0,
                            "lpips":0.0,
                            "psnr": 0.0,
                            "fid": 0.0
                            }
    for traj in traj_metrics.keys():

        traj_metrics["avrg"]["ssim"] += traj_metrics[traj]["ssim"]
        traj_metrics["avrg"]["lpips"] += traj_metrics[traj]["lpips"]
        traj_metrics["avrg"]["psnr"] += traj_metrics[traj]["psnr"]
        traj_metrics["avrg"]["fid"] += traj_metrics[traj]["fid"]

    traj_metrics["avrg"]["ssim"] /= len(traj_metrics.keys())
    traj_metrics["avrg"]["lpips"] /= len(traj_metrics.keys())
    traj_metrics["avrg"]["psnr"] /= len(traj_metrics.keys())
    traj_metrics["avrg"]["fid"] /= len(traj_metrics.keys())

    return traj_metrics


def calculate_drone_ds_metrcis(model_config_file, out_dir):    
    out_eval_file = f"{out_dir}/results.json"
    model_dataparser_transforms = f"{model_config_file.parent}/dataparser_transforms.json"
    compose_background = False

    config, pipeline, _, _ = eval_setup(
                Path(model_config_file),
                eval_num_rays_per_chunk=None,
                test_mode="inference",
            )
            
    ds_root = config.data
    dataset_json = f"{ds_root}/transforms.json"


    with open(dataset_json, 'r') as f:
        dataset = json.load(f)
        
        
    eval_data = {}

    for frame in dataset["frames"]:
        if "eval" in frame["file_path"]:
            traj = frame["file_path"].split('/')[-1].split("eval_")[-1].split('_frame')[0]
            eval_data.setdefault(traj, []).append(frame)
            
    model = pipeline.model

    os.makedirs(out_dir, exist_ok=True)


    traj_metrics = calculate_eval_metrics(eval_data, model, model_dataparser_transforms, 
                                          ds_root, out_dir, compose_background)

    traj_metrics = calculate_eval_metrics_avrg(traj_metrics)


    with open (out_eval_file, 'w') as f:
        json.dump(traj_metrics, f, indent=4)