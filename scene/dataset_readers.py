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

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

barf=False

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    norm_K: list = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    gt_cameras: list = None
    test_idxs: list = None
    bwd_flows: list = None
    gt_depths: list = None

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if "SIMPLE" in intr.model:#intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        else:#elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        #else: assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):

    eval=True
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = cam_infos_=sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    start_i,max_i=0,1000
    #start_i=0 if "cat" not in path else 35
    #max_i = (65 if "kitchen" in path else 60 if "horns" in path else 30 if "flower" in path else 60 if "teddybear" in path else 60 if "hydrant" in path else 150)+start_i
    print("max i",max_i)
    if 0:
        import torch
        tmp= {k:v.cpu() if type(v)==torch.Tensor else v for k,v in torch.load("../flowmap/aligned.pt").items()}
        cam_infos = readFlowCamCameras(tmp=tmp)[0]

    cam_infos=cam_infos[start_i:max_i]
    test_idxs=list(range(len(cam_infos)))[1:-1:10] if eval else []
    print(test_idxs)
    if eval:
        train_cam_infos = cam_infos#[c for idx, c in enumerate(cam_infos) if idx not in test_idxs]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx in test_idxs]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           gt_cameras=cam_infos_[start_i:max_i],
                           test_cameras=test_cam_infos,
                           test_idxs=test_idxs,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))
            
    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readFlowCamCameras(tmp):
    cam_infos = []
    cam_infos_= []
    cam_infos_test= []

    def get_poses(pose_name,intrinsics_name):
        cams=[]
        for idx in range(len(tmp[pose_name])):
            image_path_="%04d.png"%idx
            image_name_="%04d"%idx
            height_,width_=tmp["rgb"].shape[-2:]
            image_= Image.fromarray((tmp["rgb"][min(idx,len(tmp["rgb"])-1)].permute(1,2,0)).numpy().astype(np.uint8))
            c2w=np.array(tmp[pose_name][idx])
            w2c = np.linalg.inv(c2w)

            if barf and pose_name is not "gt_poses":w2c=np.eye(4)#+np.random.randn(*np.eye(4).shape)/10

            R_ = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T_ = w2c[:3, 3]
            norm_K=tmp[intrinsics_name][0].clone()
            K=np.array(tmp[intrinsics_name][0])
            #if idx==0: print(K)

            K[0]*=width_
            K[1]*=height_
            FovY_= focal2fov(K[1,1], height_)
            FovX_= focal2fov(K[0,0], width_)
            cam_info_= CameraInfo(uid=1, R=R_, T=T_, FovY=FovY_, FovX=FovX_, image=image_,
                                  image_path=image_path_, image_name=image_name_, width=width_, height=height_,norm_K=norm_K)
            cams.append(cam_info_)
        return cams
    out= [get_poses("poses","intrinsics"),get_poses("novel_poses" if "novel_poses" in tmp else "poses","intrinsics"),get_poses("gt_poses","gt_intrinsics") if "gt_poses" in tmp else get_poses("poses","intrinsics")]
    #print("USING GT INTRINSICS\n*10")
    #out= [get_poses("poses","gt_intrinsics"),get_poses("novel_poses" if "novel_poses" in tmp else "poses","intrinsics"),get_poses("gt_poses","gt_intrinsics")]
    return out
def readFlowCamSceneInfo(path, images, eval, llffhold=8):

    #same coordinate system as colmap -- just verify c2w vs w2c, unnormalize intrinsics here
    #just load in point cloud xyz and rgb below
    import torch
    tmp = {k:v.cpu() for k,v in torch.load(path).items()}
    if tmp["rgb"].max()<2: tmp["rgb"]=255*(tmp["rgb"]*.5+.5)

    eval="gt_poses" in tmp

    #sf=.5
    #with torch.no_grad(): tmp["world_crds"]=torch.nn.functional.interpolate(tmp["world_crds"].permute(0,2,1).unflatten(-1,tmp["rgb"].shape[-2:]),scale_factor=sf).flatten(-2,-1).permute(0,2,1)
    #with torch.no_grad(): tmp["rgb"]=torch.nn.functional.interpolate(tmp["rgb"],scale_factor=sf)

    reading_dir = "images" if images == None else images
    cam_infos = readFlowCamCameras(tmp=tmp)

    nerf_normalization = getNerfppNorm(cam_infos[0])

    stride=max(1,int(len(tmp["world_crds"].flatten(0,1))/150_000))
    print("stride: ",stride)
    #points=tmp["world_crds"].flatten(0,1)[::5].numpy()
    #rgb=tmp["rgb_crds"].flatten(0,1)[::5].numpy()
    points=tmp["world_crds"].flatten(0,1)[::5].numpy()
    rgb=tmp["rgb_crds"].flatten(0,1)[::5].numpy()

    #idxs=torch.randperm(len(tmp["rgb"][:,0].flatten()))[:200000]
    #points=torch.nn.functional.interpolate(tmp["world_crds"].permute(0,2,1).unflatten(-1,tmp["flow_inp_"][0].shape[1:]),tmp["rgb"].shape[-2:]).flatten(-2,-1).permute(0,2,1).flatten(0,1)[idxs].numpy()
    #rgb=tmp["rgb"].flatten(-2,-1).permute(0,2,1).flatten(0,1)[idxs].numpy()/255

    if barf:
        rgb=np.random.rand(*rgb.shape)
        pmax,pmin=torch.from_numpy(points).max(dim=0)[0].numpy(),torch.from_numpy(points).min(dim=0)[0].numpy()
        points=np.random.rand(*points.shape)*(pmax-pmin)+pmin

    pcd= BasicPointCloud(points=points, colors=rgb, normals=points*0)
    ply_path="tmp.ply"
    print("writing ply")
    storePly(ply_path, points, rgb)

    test_idxs=list(range(len(cam_infos[0])))[1:-1:10] if eval else []
    print(test_idxs)
    if eval:
        #train_cam_infos = [c for idx, c in enumerate(cam_infos[0]) if idx not in test_idxs]
        train_cam_infos =cam_infos[0] #[c for idx, c in enumerate(cam_infos[0]) ]
        test_cam_infos = [c for idx, c in enumerate(cam_infos[0]) if idx in test_idxs]
    else:
        train_cam_infos = cam_infos[0]
        test_cam_infos = []

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           gt_cameras=cam_infos[2],
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           test_idxs=test_idxs,
                           )
    return scene_info




sceneLoadTypeCallbacks = {
    "FlowCam": readFlowCamSceneInfo,
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
