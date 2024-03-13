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

from torchcubicspline import (natural_cubic_spline_coeffs, NaturalCubicSpline)
import splines.quaternion
import torch.nn.functional as F

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    # from pytorch3d
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    # from pytorch3d
    """
    Convert rotations given as rotation matrices to quaternions.
    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))

def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    # from pytorch3d
    """
    Convert rotations given as quaternions to rotation matrices.
    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

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
    #pos_spline_idxs=torch.linspace(0,all_poses.size(0)-1,15 if 1 else 40)
    #rot_spline_idxs=torch.linspace(0,all_poses.size(0)-1,15 if 1 else 40)
    pos_spline_idxs=torch.arange(all_poses.size(0)).float()
    rot_spline_idxs=torch.arange(all_poses.size(0)).float()

    all_pos_splines=[]
    all_quat_splines=[]

    all_pos_spline=[]
    all_quat_spline=[]

    all_pos_spline.append(NaturalCubicSpline(natural_cubic_spline_coeffs(pos_spline_idxs, all_poses[pos_spline_idxs.long(),:3,-1].cpu())))
    quats = matrix_to_quaternion(all_poses[:,:3,:3])
    all_quat_spline.append(splines.quaternion.PiecewiseSlerp([splines.quaternion.UnitQuaternion.from_unit_xyzw(quat_) 
                                    for quat_ in quats[rot_spline_idxs.long()].detach().cpu().numpy()],grid=rot_spline_idxs.detach().tolist()))
    all_pos_splines.append(all_pos_spline)
    all_quat_splines.append(all_quat_spline)

    n=100
    thetas=np.linspace(0,np.pi*10*len(all_poses)/60,n)

    query_poses=[]
    for t_i,t in enumerate(torch.linspace(0,all_poses.size(0)-1,n)):
        print(t)

        pos_splines=all_pos_splines[0]
        quat_splines_=all_quat_splines[0]
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
        custom_pose[:3,:3] = quaternion_to_matrix(curr_quats)
        query_poses.append(custom_pose)
    return torch.stack(query_poses)
