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

import json
import os
import torch
import numpy as np
from matplotlib import cm
import imageio
import scipy.spatial
from random import randint
from utils.loss_utils import l1_loss, ssim
import random
import wandb
from pathlib import Path
import matplotlib.pyplot as plt
from gaussian_renderer import render,render_
import sys
from scene import Scene, GaussianModel
from utils.graphics_utils import matrix_to_euler_angles,lift_to_poses
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from lpipsPyTorch import lpips
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from utils.camera_utils import render_time_interp

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset,args.name)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_stack_gt = scene.gt_cameras


    # Setup poses and focals as parameters to optionally fine-tune
    poses = torch.stack([cam.world_view_transform.T for cam in viewpoint_stack])
    focal_params = torch.tensor([viewpoint_stack[0].FoVx,viewpoint_stack[0].FoVy]).detach().clone().cuda()
    transf_params = torch.nn.Parameter(torch.cat((matrix_to_euler_angles(poses[:,:3,:3]),poses[:,:3,-1]),-1).detach().clone(),requires_grad=True)
    focal_params = torch.nn.Parameter(focal_params,requires_grad=True)

    poses_gt = torch.stack([cam.world_view_transform.T for cam in viewpoint_stack_gt])
    focal_params_gt = torch.tensor([viewpoint_stack_gt[0].FoVx,viewpoint_stack_gt[0].FoVy]).detach().clone()
    transf_params_gt = torch.cat((matrix_to_euler_angles(poses_gt[:,:3,:3]),poses_gt[:,:3,-1]),-1).detach().clone()

    cam_lr=5e-5
    cam_optim = torch.optim.Adam(lr=cam_lr, params=[transf_params])
    focal_lr=5e-5
    focal_optim = torch.optim.Adam(lr=focal_lr, params=[focal_params])

    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    since_cam_step=0
    for iteration in range(first_iter, opt.iterations + 1):        
        since_cam_step+=1

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0 and not args.no_sh:
            print("increasing sh degree")
            gaussians.oneupSHdegree()

        s=1
        c_iteration = iteration - args.start_cam_opt 
        cam_lr = (5e-5 if c_iteration<s*15000 else 5e-6 if c_iteration<s*20000 else 0)
        tb_writer.add_scalar('train_loss_patches/cam_lr', cam_lr, iteration)

        for param_group in cam_optim.param_groups: param_group['lr'] = cam_lr
        for param_group in focal_optim.param_groups: param_group['lr'] = cam_lr

        # Pick a random Camera
        max_cam_i = len(viewpoint_stack)-1#min(len(viewpoint_stack)-1,int(iteration/1000))
        tb_writer.add_scalar('train_loss_patches/max_cam_i', max_cam_i, iteration)
        
        cam_i = randint(0, max_cam_i)
        if max_cam_i!=len(viewpoint_stack)-1: cam_i=random.choice([max_cam_i,cam_i])

        viewpoint_cam = viewpoint_stack[cam_i]

        transf=lift_to_poses(transf_params[cam_i])

        # Render
        if (iteration - 1) == debug_from: pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        if iteration > args.start_cam_opt: 
            render_pkg = render(viewpoint_cam, gaussians, pipe, background,pre_transf=transf,fov=focal_params) # with pose gradients
        else: 
            render_pkg = render_(viewpoint_cam, gaussians, pipe, bg) # without (faster)

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        if iteration>args.start_cam_opt:
            cam_optim.step(); cam_optim.zero_grad(); 
            focal_optim.step(); focal_optim.zero_grad(); 

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            if args.render_checkpoint:
                gaussians.load_ply(args.render_checkpoint)
                transf_params,focal_params=torch.load("/".join(args.render_checkpoint.split("/")[:-3])+"/poses.pt")
                all_cams_=sorted(scene.getTrainCameras(),key=lambda x:x.image_name)
                all_cams=torch.stack([lift_to_poses(transf_params[int(cam.image_name)]) for cam in all_cams_])
                interp_poses=render_time_interp(all_cams)
                interp_poses2=render_time_interp(all_cams,wobble=False)
                novel_images,vid_depths,novel_both=[],[],[]

                #fig=plt.figure();ax = fig.add_subplot(111, projection='3d');
                #ax.plot(*all_cams[:,:3,-1].cpu().unbind(1),c="red",label="Estimated Trajectory");
                #ax.plot(*interp_poses[:,:3,-1].cpu().unbind(1),c="black",label="GT Trajectory");
                #ax.plot(*interp_poses2[:,:3,-1].cpu().unbind(1),c="blue",label="GT Trajectory");
                #plt.legend()
                #plt.savefig("/home/camsmith/tmp.png")
                #plt.close()

                for i,cam in enumerate(interp_poses):
                    print(i)
                    print("rendering")
                    vid_image = render(all_cams_[0], gaussians, pipe, background,pre_transf=cam,fov=focal_params)["render"]
                    novel_images.append(vid_image)
                    vid_depths.append( render(all_cams_[0], gaussians, pipe, background+10,pre_transf=cam,fov=focal_params,render_depth=True)["render"][0] )
                    print("rendered")
                min_depth=min([x.cpu().min() for x in vid_depths])
                print("doing colormaps")
                vid_depths =[ torch.from_numpy(cm.get_cmap('magma')(min_depth/vid_depth.cpu().numpy())).squeeze(-2)[...,:3].permute(2,0,1) for vid_depth in vid_depths] 
                novel_both =[ torch.cat((x.cpu(),y.cpu()),-1) for x,y in zip(novel_images,vid_depths)] 
                frames = [(255*x.permute(1,2,0).cpu().numpy()).astype(np.uint8) for x in torch.stack(novel_both).clip(0,1)]
                print("writing frames")
                os.makedirs("output/renders",exist_ok=True)
                imageio.mimwrite("output/renders/"+args.render_checkpoint.split("/")[1]+"_both.mp4", frames, fps=8, quality=7)
                print("output/renders/"+args.render_checkpoint.split("/")[1]+"_both.mp4")
                frames = [(255*x.permute(1,2,0).cpu().numpy()).astype(np.uint8) for x in torch.stack(vid_depths).clip(0,1)]
                imageio.mimwrite("output/renders/"+args.render_checkpoint.split("/")[1]+"_depth.mp4", frames, fps=8, quality=7)
                frames = [(255*x.permute(1,2,0).cpu().numpy()).astype(np.uint8) for x in torch.stack(novel_images).clip(0,1)]
                imageio.mimwrite("output/renders/"+args.render_checkpoint.split("/")[1]+"_rgb.mp4", frames, fps=8, quality=7)
                done

            if iteration%100==1:torch.save([transf_params.detach().clone(),focal_params.detach().clone()],scene.model_path+"/poses.pt")
            if (iteration in [300,500,1000] or iteration%3000==1 and 1) and 0:
                raise NotImplementedError("This is still trying to auto-detect the colmap_cams flag. If you get rid of this error, you need to fix that.")

                colmap_cams=any([x in scene.getTrainCameras()[0].image_name for x in ["DSC","IMG","DJI","frame"]]) or 0
                if not colmap_cams:
                    #stride=10 if len(transf_params)>200 else 1
                    all_cams=sorted(scene.getTestCameras()+scene.getTrainCameras(),key=lambda x:x.image_name)
                    torch.cuda.empty_cache()
                    print("rendering all poses")
                    novel_images,novel_depth,novel_images_gt=[],[],[]
                    #for i,pose in enumerate(viewpoint_stack_test):
                    for i,cam in enumerate(all_cams[:200]):
                        print(i)
                        # TODO this should be using the optimized camera poses, otherwise these are novel renderings and not video reconstructions
                        vid_image = render(cam, gaussians, pipe, background,pre_transf=lift_to_poses(transf_params[int(cam.image_name)]),fov=focal_params)["render"]
                        vid_depth = render(cam, gaussians, pipe, background,pre_transf=lift_to_poses(transf_params[int(cam.image_name)]),fov=focal_params,render_depth=True)["render"]
                        novel_images.append(vid_image)
                        novel_depth.append(vid_depth[0])
                else:
                    print("rendering all poses")
                    novel_images,novel_depth,novel_images_gt=[],[],[]
                    for i,pose in enumerate(viewpoint_stack):
                        # TODO this should be using the optimized camera poses, otherwise these are novel renderings and not video reconstructions
                        vid_image = render(pose, gaussians, pipe, background)["render"]
                        novel_images.append(vid_image)
                        #vid_depth = render(pose, gaussians, pipe, background,render_depth=True)["render"]
                        #novel_depth.append(vid_depth[0])
                print("done rendering")
                if len(novel_images): 
                    frames=torch.stack(novel_images).clip(0,1)
                    frames = [(255*x.permute(1,2,0).cpu().numpy()).astype(np.uint8) for x in frames]
                    frames = frames + frames[::-1]
                    f = "novel.mp4"
                    if iteration<800: f="init_"+f
                    print("writing video at",f)
                    imageio.mimwrite(f, frames, fps=8, quality=7)
                    wandb.log({f'render_novel':wandb.Video(f, format='mp4', fps=8)})
                    torch.cuda.empty_cache()
                if len(novel_depth): 
                    frames=[((11-x).clip(min=1e-5)*255/11).cpu().numpy().astype(np.uint8) for x in novel_depth]
                    #frames=torch.stack(novel_depth)
                    #frames = [(255*x.permute(1,2,0).cpu().numpy()).astype(np.uint8) for x in frames]
                    frames = frames + frames[::-1]
                    f = "depth.mp4"
                    if iteration<800: f="init_"+f
                    print("writing video at",f)
                    imageio.mimwrite(f, frames, fps=8, quality=7)
                    wandb.log({f'render_depth':wandb.Video(f, format='mp4', fps=8)})
                    torch.cuda.empty_cache()

            # Pose logging
            #try:
            #our_poses = transf_params[...,3:].detach().cpu()
            #gt_poses = transf_params_gt[...,3:].detach().cpu()
            our_poses=lift_to_poses(transf_params).inverse()[...,:3,-1].detach().cpu()
            gt_poses=lift_to_poses(transf_params_gt).inverse()[...,:3,-1].detach().cpu()

            pos_gt_,pos_est_ = [torch.from_numpy(x) for x in scipy.spatial.procrustes(gt_poses.cpu().numpy(),our_poses.numpy())[:2]]
            ATE=(pos_gt_-pos_est_).square().mean().sqrt()
            tb_writer.add_scalar('train_loss_patches/ATE', ATE, iteration)

            if iteration%500==1 and 1:
                print("ATE",ATE)
                #print("making pose plot")
                tb_writer.add_images("render_vs_gt", torch.stack((image,gt_image)).clip(0,1), global_step=iteration)

                fig=plt.figure();ax = fig.add_subplot(111, projection='3d');
                ax.plot(*pos_est_.cpu().unbind(1),c="red",label="Estimated Trajectory");
                ax.plot(*pos_gt_.cpu().unbind(1),c="black",label="GT Trajectory");
                plt.legend()
                fig.canvas.draw()
                image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
                tb_writer.add_images("poses", torch.from_numpy(image_from_plot/255).permute(2,0,1)[None].clip(0,1), global_step=iteration)
                plt.close()

            #except:
            #    print("pose plotting error")

            tb_writer.add_scalar('train_loss_patches/Focal_Est_X', focal_params[0], iteration)
            tb_writer.add_scalar('train_loss_patches/Focal_GT_X', focal_params_gt[0], iteration)
            tb_writer.add_scalar('train_loss_patches/Focal_Est_Y', focal_params[1], iteration)
            tb_writer.add_scalar('train_loss_patches/Focal_GT_Y', focal_params_gt[1], iteration)

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),transf_params=transf_params,focal_params=focal_params, viewpoint_stack=viewpoint_stack)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if (iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter)) and max_cam_i==len(viewpoint_stack)-1:
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations and not (cam_i in scene.test_idxs and args.use_test): gaussians.optimizer.step()
            else:print("skipping cam",cam_i,scene.test_idxs)
            gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args,name):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    args.model_path = os.path.join("./output/", name)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,transf_params,focal_params, viewpoint_stack=None):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if 1 and iteration%500==1:# in testing_iterations:
        Path(f"metrics/{args.name}").mkdir(exist_ok=True, parents=True)

        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]}
                              )

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                psnrs = []
                lpipss = []
                ssims = []

                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    # Get the camera index within the viewpoint stack.
                    camera_index = [camera.image_name for camera in viewpoint_stack].index(viewpoint.image_name)

                    image = render(viewpoint, scene.gaussians, pre_transf=lift_to_poses(transf_params[camera_index]),fov=focal_params,*renderArgs)["render"].clip(0,1)
               
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                    if config["name"] == "test":
                        torchvision.utils.save_image(image, f"metrics/{args.name}/{config['cameras'][idx].image_name}.png")

                    psnrs.append(psnr(image, gt_image).mean().item())
                    lpipss.append(lpips(image, gt_image).item())
                    ssims.append(ssim(image, gt_image).item())

                psnrs = float(np.mean(psnrs))
                lpipss = float(np.mean(lpipss))
                ssims = float(np.mean(ssims))

                # Save the metrics.
                if config["name"] == "test":
                    metrics = {
                        "psnr": psnrs,
                        "lpips": lpipss,
                        "ssim": ssims,
                        "step": iteration,
                    }
                    with Path(f"metrics/{args.name}/metrics.json").open("w") as f:
                        json.dump(metrics, f)

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--start_cam_opt', type=int, default=3000)
    parser.add_argument('--cam_lr', type=float, default=0)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000,2_000,5_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[1_000,5_000, 30_000])
    #parser.add_argument("--res", nargs="+", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('-o','--online', action='store_true', default=False)
    parser.add_argument('--no_sh', action='store_true', default=False)
    parser.add_argument('--no_anneal_cam_lr', action='store_true', default=False)
    parser.add_argument('--use_test', action='store_true', default=True)
    parser.add_argument('-n','--name', type=str, default="gauss_splat")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--render_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    run = wandb.init(entity="scene-representation-group",project="gaussian-barf",mode="online" if args.online else "disabled",
                    name=args.name, sync_tensorboard=True)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    #safe_state(args.quiet)

    # Start GUI server, configure and run training
    #network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,args)

    # All done
    print("\nTraining complete.")
