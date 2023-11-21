#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N ARCapinedit           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -t 1-3

module load conda
conda activate dune2

PROJECTP="/projectnb/llamagrp/feyzanb/dune"

cnt=0
for MODELNAME in "bard" "gpt-3.5-turbo" "gpt-4"; do
    for EDIT in "no_edit" "with_edit"; do
        (( cnt++ ))
        if [[ $cnt -eq $SGE_TASK_ID ]]; then
            OUTDIR="$PROJECTP/outputs/ARC_locality/${EDIT}_${MODELNAME}"
            mkdir -p $OUTDIR
            python eval.py \
            --model_name $MODELNAME \
            --dataset_name ARC \
            --output_dir $OUTDIR \
            --chat_prompt_dict_path "$PROJECTP/source/arc/chat_prompt_dict.json" \
            --filename_queries "$PROJECTP/dune/scientific_locality.json" \
            --$EDIT > ${OUTDIR}/log${edit}.txt 2>&1
        fi
    done
done