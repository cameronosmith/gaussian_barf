for METHOD in \
    flowmap_ablation_depth_var_rerun \
    flowmap_ablation_focal_var_rerun \
    flowmap_ablation_pose_var_rerun \
    flowmap_ablation_two_stage_rerun
do
    for SCENE in \
        fortress \
        fern \
        orchids \
        trex \
        counter \
        room \
        garden
    do
        python3 run_slurm.py python3 train.py -s /scratch/charatan/flowmap_rerun_ablations_converted/${METHOD}/${SCENE} --name paper_c2_${SCENE}_${METHOD} -o
    done
done
