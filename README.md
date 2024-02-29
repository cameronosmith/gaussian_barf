example run: `python train.py -s /home/camsmith/repos/flowmap/pose_exps/poses_bonsai_150.pt --start_cam_opt 100 -o`

relevant arguments:

-s : the scene path, either as a flowmap.pt file or colmap scene directory as in the original splatting code usage

-start_cam_opt : what iteration to start the camera optimization -- don't start it at 0 since the point cloud is very cloudy at the beginning and it's not the right time to refine the camera parameters since the scene needs to be quickly refined first


use the conda environment at `conda activate /home/camsmith/miniconda3/envs/envfile_gaussian_splatting`
