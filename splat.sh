for METHOD in \
    colmap \
    mvscolmap \
    flowmap \
    droid \
    flowmap_ablation_depth_var \
    flowmap_ablation_focal_var \
    flowmap_ablation_pose_var \
    flowmap_ablation_scratch \
    flowmap_ablation_no_tracks
do
    for SCENE in \
        bench \
        hydrant \
        flower \
        horns \
        bonsai \
        kitchen \
        caterpillar \
        playground \
        office0 \
        room0
    do
        python3 run_slurm.py python3 train.py -s /scratch/charatan/flowmap_converted/${METHOD}/${SCENE} --name paper_c2_${SCENE}_${METHOD} -o
    done
done
