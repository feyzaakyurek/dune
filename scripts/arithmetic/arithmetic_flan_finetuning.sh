#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N artfinetuning           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 10          # Specify the parallel environment and the number of cores
#$ -t 1-5

source /projectnb/llamagrp/feyzanb/anaconda3/etc/profile.d/conda.sh
conda activate dunetrans


cnt=0
for MODELNAME in small base large xl xxl; do
    (( cnt++ ))
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic_locality/finetuning_flan-t5-${MODELNAME}"
        MODELPATH="/projectnb/llamagrp/feyzanb/dune/outputs/"
        CACHE="/projectnb/llamagrp/feyzanb/dune/cache/Arithmetic"
        mkdir -p $OUTDIR
        mkdir -p $CACHE
        python eval.py \
        --model_name $MODELPATH/all_fine_tune_flant5$MODELNAME \
        --dataset_name Arithmetic \
        --output_dir $OUTDIR \
        --generations_cache $CACHE/all_fine_tune_flant5$MODELNAME.json \
        --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_locality_qa.json" \
        --from_flax > ${OUTDIR}/log.txt 2>&1
    fi
done