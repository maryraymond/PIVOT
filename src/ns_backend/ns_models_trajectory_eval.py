# PIVOT: Pose, Intrinsics and Viewpoint Oriented Testbed
# Copyright (c) 2026 Mary Raymond
# Date: 2026-07-07
# MIT License — see LICENSE in the project root for details.


import json
from pathlib import Path
import torch
import cv2
import numpy as np
from copy import deepcopy


from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.cameras import camera_utils
from nerfstudio.models.splatfacto import SplatfactoModel
from nerfstudio.data.datamanagers.full_images_datamanager import _undistort_image

from reconstruction_eval.model_trajectory_eval import ReconsModel

def get_data_transforms(transforms_file):

    R = np.eye(4)
    with open(transforms_file, 'r') as f:
        transforms = json.load(f)

        R[:3, :4] = np.array(transforms["transform"]).astype(np.float32)
        scale = float(transforms["scale"])


    return R, scale


class NsModel(ReconsModel):
    def __init__(self, config_file, compose_background=False):
        
        super().__init__()
        model_dataparser_transforms = f"{config_file.parent}/dataparser_transforms.json"
        self.R, self.scale = get_data_transforms(model_dataparser_transforms)
        self.compose_background = compose_background

        config, pipeline, _, _ = eval_setup(
                    Path(config_file),
                    eval_num_rays_per_chunk=None,
                    test_mode="inference",
                )
        self.config = config
        self.model = pipeline.model
        self.scale_factor = config.pipeline.datamanager.camera_res_scale_factor
        

    def get_ds_root(self):
         return self.config.data
    
    def get_generated_image(self, c2w, H, W, K, D, C=3, camera_model="OPENCV"):
        """ This function should return the image with float pixel value 0 to 1
         The output should be a numpy array of shape should be H, W, C"""
        c2w = self.R @  np.array(c2w, copy=True)
        c2w[:3, 3] *=  self.scale
        c2w = torch.tensor(c2w, dtype=torch.float32)
        c2w = c2w[:3].unsqueeze(0)

        fx=float(K[0, 0])
        fy=float(K[1, 1])
        cx=float(K[0, 2])
        cy=float(K[1, 2])
       
       # BUILD UP THE CAMERA AND THE POSES
        if camera_model == "OPENCV":
            camera_type = CameraType.PERSPECTIVE
        elif camera_model == "OPENCV_FISHEYE":
            camera_type = CameraType.FISHEYE

        # read the distaortions
        distort = camera_utils.get_distortion_params(
                        k1=float(D["k1"]) if "k1" in D else 0.0,
                        k2=float(D["k2"]) if "k2" in D else 0.0,
                        k3=float(D["k3"]) if "k3" in D else 0.0,
                        k4=float(D["k4"]) if "k4" in D else 0.0,
                        p1=float(D["p1"]) if "p1" in D else 0.0,
                        p2=float(D["p2"]) if "p2" in D else 0.0,
                    )
        
        dummy_img = np.random.randint(
                low=0,
                high=256,
                size=(H, W, C),
                dtype=np.uint8,
        )

        if self.scale_factor < 1.0:

            dummy_img = cv2.resize(dummy_img, None, 
                                   fx=self.scale_factor, 
                                   fy=self.scale_factor, 
                                   interpolation=cv2.INTER_LINEAR)
            fx *= self.scale_factor
            fy *= self.scale_factor
            cx *= self.scale_factor
            cy *= self.scale_factor
            
            H = dummy_img.shape[0]
            W = dummy_img.shape[1]

            # convert the gt image to float
            dummy_img = dummy_img.astype(np.float32)/255
            
        self.source_camera = Cameras(
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                distortion_params=distort,
                height=torch.tensor(H, dtype=torch.int),
                width=torch.tensor(W, dtype=torch.int),
                camera_to_worlds=c2w[:, :3, :4],
                camera_type=camera_type,
            )     
        
        self.K = self.source_camera.get_intrinsics_matrices().squeeze(0).numpy().copy()
        self.distortion_params = self.source_camera.distortion_params.squeeze(0).numpy().copy()
        self.processed_camera = deepcopy(self.source_camera)

        if isinstance(self.model, SplatfactoModel):
            # we need to undistort the image
            data = {}
            K = self.source_camera.get_intrinsics_matrices().squeeze(0).numpy()
            distortion_params = self.source_camera.distortion_params.squeeze(0).numpy()
            
            Knew, dummy_img, mask = _undistort_image(self.source_camera, 
                                                     distortion_params, 
                                                     data, 
                                                     dummy_img, 
                                                     K)
            # we create each camera sepreatly so idx  = 0
            idx = 0
            self.processed_camera.fx[idx] = float(Knew[0, 0])
            self.processed_camera.fy[idx] = float(Knew[1, 1])
            self.processed_camera.cx[idx] = float(Knew[0, 2])
            self.processed_camera.cy[idx] = float(Knew[1, 2])
            self.processed_camera.width[idx] = dummy_img.shape[1]
            self.processed_camera.height[idx] = dummy_img.shape[0]

        with torch.no_grad():
                    gen_image = self.model.get_outputs_for_camera(camera=self.processed_camera)

                    if "background" in gen_image:
                        # safe the background info
                        self.gen_background = gen_image["background"]
                    
                    rgb = gen_image['rgb'].cpu()
        
        return rgb.numpy()
    
    def process_gt_image(self, gt_image):
        """This function should implement any gt manpulation like rescaling, undistorting and so on
        the input image pixel range is float 0 to 1 and the return should be in range 0 to 1 
        numpy array of shape H, W, 3"""
        if self.scale_factor < 1.0:
            gt_image = cv2.resize(gt_image, 
                                  None, 
                                  fx=self.scale_factor, 
                                  fy=self.scale_factor, 
                                  interpolation=cv2.INTER_LINEAR)


        if isinstance(self.model, SplatfactoModel):
            # we need to undistort the image
            data = {}
            _, gt_image, _ = _undistort_image(self.source_camera, self.distortion_params, data, gt_image, self.K)
            
        if self.compose_background:
            gt_image = self.model.composite_with_background(gt_image, self.gen_background)

        return gt_image

