#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import torch


import sys
sys.path.append("../flowmap")
sys.path.append("../../flowmap")
import geometry

from torchcubicspline import (natural_cubic_spline_coeffs, NaturalCubicSpline)
import splines.quaternion

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

        #resolution = [864,608] # our mipnerf res
    if orig_w>orig_h: resolution=[1024,672]

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,norm_K=cam_info.norm_K)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def render_time_interp(all_poses,wobble=False):
    pos_spline_idxs=torch.linspace(0,all_poses.size(0)-1,15 if 1 else 40)
    rot_spline_idxs=torch.linspace(0,all_poses.size(0)-1,15 if 1 else 40)

    all_pos_splines=[]
    all_quat_splines=[]

    b_i=0
    all_pos_spline=[]
    all_quat_spline=[]

    obj_i=0
    all_pos_spline.append(NaturalCubicSpline(natural_cubic_spline_coeffs(pos_spline_idxs, all_poses[pos_spline_idxs.long(),:3,-1].cpu())))
    quats=geometry.matrix_to_quaternion(all_poses[:,:3,:3])
    all_quat_spline.append(splines.quaternion.PiecewiseSlerp([splines.quaternion.UnitQuaternion.from_unit_xyzw(quat_) 
                                    for quat_ in quats[rot_spline_idxs.long()].detach().cpu().numpy()],grid=rot_spline_idxs.detach().tolist()))
    all_pos_splines.append(all_pos_spline)
    all_quat_splines.append(all_quat_spline)

    n=100
    thetas=np.linspace(0,np.pi*10*len(all_poses)/60,n)

    query_poses=[]
    for t_i,t in enumerate(torch.linspace(0,all_poses.size(0)-1,n)):
        print(t)

        custom_poses_b=[]
        pos_splines=all_pos_splines[0]
        quat_splines_=all_quat_splines[0]
        custom_poses=[]
        pos_spline=pos_splines[0]
        quat_spline_=quat_splines_[0]
        custom_pose=torch.eye(4).cuda()
        custom_pose[:3,-1]=pos_spline.evaluate(t)
        scale = .015 #.3 * model.far/30
        if wobble:
            custom_pose[0,-1]+=np.cos(thetas[t_i]) * scale
            custom_pose[1,-1]+=np.sin(thetas[t_i]) * scale
        quat_eval=quat_spline_.evaluate(t.item())
        curr_quats = torch.tensor(list(quat_eval.vector)+[quat_eval.scalar])
        custom_pose[:3,:3] = geometry.quaternion_to_matrix(curr_quats)
        query_poses.append(custom_pose)
    return torch.stack(query_poses)
