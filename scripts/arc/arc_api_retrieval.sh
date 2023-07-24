#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N arcretapi           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -t 1

source /projectnb/llamagrp/feyzanb/anaconda3/etc/profile.d/conda.sh
conda activate dune


cnt=0
for MODELNAME in "bard"; do
    (( cnt++ ))
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/ARC_locality/retrieval_${MODELNAME}"
        CACHE="/projectnb/llamagrp/feyzanb/dune/cache/ARC"
        mkdir -p $OUTDIR
        mkdir -p $CACHE
        python eval.py \
        --model_name $MODELNAME \
        --dataset_name ARC \
        --output_dir $OUTDIR \
        --retriever_mechanism bm25 \
        --chat_prompt_dict_path "/projectnb/llamagrp/feyzanb/dune/source/arc/chat_prompt_dict.json" \
        --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/arc/arc_locality_processed.json" \
        --with_edit > ${OUTDIR}/log.txt 2>&1
    fi
done