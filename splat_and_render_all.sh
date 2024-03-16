for METHOD in \
    ablation_none \
    ablation_random_initialization \
    ablation_midas_initialization \
    ablation_explicit_depth \
    ablation_explicit_focal_length \
    ablation_explicit_pose \
    ablation_no_correspondence_weights \
    ablation_no_tracks \
    ablation_single_stage
do
    for SCENE in /scratch/charatan/projects/flowmap/results/colmap/*/ ; do
        python3 run_slurm.py python3 train.py -s /scratch/charatan/projects/flowmap/results/${METHOD}/${SCENE} --name paper_v14_${SCENE}_${METHOD}
        python3 run_slurm.py python3 train.py -s /scratch/charatan/projects/flowmap/results/${METHOD}/${SCENE} --name paper_v14_${SCENE}_${METHOD} --render_checkpoint output/paper_v14_${SCENE}_${METHOD}/point_cloud/iteration_30000/point_cloud.ply
    done
done
