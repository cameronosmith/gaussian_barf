for METHOD in \
    flowmap_ablation_none \
    colmap \
    mvscolmap \
    flowmap_ablation_random_initialization \
    flowmap_ablation_random_initialization_long \
    flowmap_ablation_explicit_depth \
    flowmap_ablation_explicit_focal_length \
    flowmap_ablation_explicit_pose \
    flowmap_ablation_no_correspondence_weights \
    flowmap_ablation_no_tracks \
    flowmap_ablation_single_stage
do
    for SCENE in /scratch/datasets/flowmap/*/ ; do
        python3 run_slurm.py python3 train.py -s /scratch/charatan/projects/flowmap/results/${METHOD}/$(basename ${SCENE}) --name paper_v17_${METHOD}_$(basename ${SCENE}) -o
    done
done
