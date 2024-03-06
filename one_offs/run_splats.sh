for METHOD in \
    colmap \
    flowmap
do
    for SCENE in \
        co3d_bench \
        co3d_hydrant \
        llff_flower \
        llff_horns \
        mipnerf360_bonsai \
        mipnerf360_kitchen \
        tandt_caterpillar \
        tandt_playground
    do
        python3 run_slurm.py python3 train.py -s ~/datasets/flowmap/${METHOD}/${SCENE} --name paper_v1_${SCENE}_${METHOD} -o
    done
done
