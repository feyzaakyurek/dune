#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N niretapi           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -t 1-3

source /projectnb/llamagrp/feyzanb/anaconda3/etc/profile.d/conda.sh
conda activate dune


cnt=0
for MODELNAME in "gpt-3.5-turbo" "gpt-4" "bard"; do
    (( cnt++ ))
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/NewInfo_locality/retrieval_${MODELNAME}"
        CACHE="/projectnb/llamagrp/feyzanb/dune/cache/NewInfo"
        mkdir -p $OUTDIR
        mkdir -p $CACHE
        python eval.py \
        --model_name $MODELNAME \
        --dataset_name NewInfo \
        --output_dir $OUTDIR \
        --retriever_mechanism bm25 \
        --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/newinfo/new_info_locality_processed.json" \
        --with_edit > ${OUTDIR}/log.txt 2>&1
    fi
done