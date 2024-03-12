for METHOD in \
    colmap \
    mvscolmap \
    flowmap \
    droid
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
        python3 run_slurm.py python3 train.py -s /scratch/charatan/flowmap_supplemental_converted/${METHOD}/${SCENE} --name paper_c2_${SCENE}_${METHOD} -o
    done
done
