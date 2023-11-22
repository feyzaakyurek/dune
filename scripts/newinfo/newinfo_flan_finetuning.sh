#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N nifinetuning           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 10          # Specify the parallel environment and the number of cores
#$ -t 1-5

module load conda
conda activate dune2

PROJECTP="/projectnb/llamagrp/feyzanb/dune"

cnt=0
for MODELNAME in small base large xl xxl; do
    (( cnt++ ))
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        OUTDIR="${PROJECTP}/outputs/NewInfo_post_emnlp/finetuning_flan-t5-${MODELNAME}"
        MODELPATH="${PROJECTP}/outputs/"
        CACHE="${PROJECTP}/cache/NewInfo"
        mkdir -p $OUTDIR
        mkdir -p $CACHE
        python eval.py \
        --model_name $MODELPATH/all_fine_tune_flant5$MODELNAME \
        --dataset_name NewInfo \
        --output_dir $OUTDIR \
        --generations_cache $CACHE/all_fine_tune_flant5$MODELNAME.json \
        --filename_queries "${PROJECTP}/dune/new_info.json" \
        --from_flax > ${OUTDIR}/log.txt 2>&1
    fi
done