#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N arteditnedit           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=32G  # Request 48GB of GPU memory per GPU
#$ -l gpu_c=6.0		# Request GPU compute capability
#$ -t 1-2

source /projectnb/llamagrp/feyzanb/anaconda3/etc/profile.d/conda.sh
conda activate dune


cnt=0
for MODELNAME in xl xxl; do
    for EDIT in "no_edit"; do
        (( cnt++ ))
        if [[ $cnt -eq $SGE_TASK_ID ]]; then
            OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/NewInfo_locality/${EDIT}_flan-t5-${MODELNAME}"
            CACHE="/projectnb/llamagrp/feyzanb/dune/cache/NewInfo"
            mkdir -p $OUTDIR
            mkdir -p $CACHE
            python eval.py \
            --model_name google/flan-t5-$MODELNAME \
            --dataset_name NewInfo \
            --output_dir $OUTDIR \
            --generations_cache $CACHE/flan-t5-${MODELNAME}.json \
            --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/newinfo/new_info_locality_processed.json" \
            --$EDIT > ${OUTDIR}/log${edit}.txt 2>&1
        fi
    done
done

# EDIT="no_edit"
# MODELNAME="small"
# OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/NewInfo/${EDIT}_flan-t5-${MODELNAME}"
# CACHE="/projectnb/llamagrp/feyzanb/dune/cache/NewInfo"
# mkdir -p $OUTDIR
# mkdir -p $CACHE
# python eval.py \
# --model_name google/flan-t5-$MODELNAME \
# --dataset_name NewInfo \
# --output_dir $OUTDIR \
# --generations_cache $CACHE/flan-t5-${MODELNAME}.json \
# --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/newinfo/new_info.json" \
# --$EDIT