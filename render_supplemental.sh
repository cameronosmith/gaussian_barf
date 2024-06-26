for METHOD in \
    colmap_rerun \
    flowmap_ablation_depth_var_rerun \
    flowmap_ablation_focal_var_rerun \
    flowmap_ablation_implicit_rerun \
    flowmap_ablation_pose_var_rerun \
    flowmap_ablation_two_stage_rerun
do
    for SCENE in \
        bench \
        bonsai \
        caterpillar \
        flower \
        horns \
        hydrant \
        kitchen \
        playground
    do
        python3 run_slurm.py python3 train.py -s /scratch/charatan/flowmap_rerun_ablations_converted/${METHOD}/${SCENE} --name paper_c2_${SCENE}_${METHOD} --render_checkpoint output/paper_c2_${SCENE}_${METHOD}/point_cloud/iteration_30000/point_cloud.ply
    done
done
