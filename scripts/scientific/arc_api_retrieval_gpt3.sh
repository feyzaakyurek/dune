#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N arcretapi           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -t 1-3

module load conda
conda activate dune2

PROJECTP="/projectnb/llamagrp/feyzanb/dune"

cnt=0
for MODELNAME in "gpt-3.5-turbo" "bard" "gpt-4"; do
    (( cnt++ ))
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        OUTDIR="${PROJECTP}/outputs/ARC_locality/retrieval_gpt3_${MODELNAME}"
        CACHE="${PROJECTP}/cache/ARC"
        mkdir -p $OUTDIR
        mkdir -p $CACHE
        python eval.py \
        --model_name $MODELNAME \
        --dataset_name ARC \
        --output_dir $OUTDIR \
        --retriever_mechanism gpt3 \
        --chat_prompt_dict_path "${PROJECTP}/source/arc/chat_prompt_dict.json" \
        --filename_queries "${PROJECTP}/arc/arc_locality.json" \
        --scope_cache ${PROJECTP}/outputs/scope_classifier/distilbert-base-cased/all_shuffled_edits_cache.json \
        --with_edit > ${OUTDIR}/log.txt 2>&1
    fi
done
