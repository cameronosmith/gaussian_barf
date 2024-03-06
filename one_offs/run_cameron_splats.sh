for METHOD in \
    colmap \
    mvscolmap \
    flowmap \
    droid
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
