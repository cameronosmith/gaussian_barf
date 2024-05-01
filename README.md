# Gaussian Splatting Modified for Joint Camera Refinement

This project just modifies the official gaussian splatting code to jointly optimize camera poses alongside the splat

In practice, this is just a slight refinement to the poses for more pixel-perfect alignment rather than a robust pose optimization

The code is not very polished but we hope it's useful

## Usage

Just point your colmap-formatted scene as follows: `python train.py -s /nobackup/flowmap_results/colmap/tandt_caterpillar --name test_caterpillar -o` and it should launch a wandb run for the optimization 

## Requirements
The environment to run gaussian splatting is a bit tricky but you should start with getting the official splatting code running (https://github.com/graphdeco-inria/gaussian-splatting) and then come here after and it should just work
