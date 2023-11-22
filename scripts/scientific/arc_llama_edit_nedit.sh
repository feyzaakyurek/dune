#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N arceditnedit           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=48G  # Request 48GB of GPU memory per GPU
#$ -l gpu_c=6.0		# Request GPU compute capability
#$ -t 1-2

module load miniconda
conda activate dune2

PROJECTP="/projectnb/llamagrp/feyzanb/dune"

cnt=0
MODELHOST="meta-llama"
MODELSUFFIX="Llama-2-7b-chat-hf"
for MODELNAME in "${MODELHOST}/${MODELSUFFIX}"; do
    for EDIT in "no_edit" "with_edit"; do
        (( cnt++ ))
        if [[ $cnt -eq $SGE_TASK_ID ]]; then
            OUTDIR="${PROJECTP}/outputs/ARC_locality/${EDIT}_${MODELSUFFIX}"
            CACHE="${PROJECTP}/cache/ARC"
            mkdir -p $OUTDIR
            mkdir -p $CACHE
            python eval.py \
            --model_name $MODELNAME \
            --dataset_name ARC \
            --output_dir $OUTDIR \
            --llama \
            --batch_size 1 \
            --generations_cache $CACHE/${MODELSUFFIX}.json \
            --filename_queries "${PROJECTP}/dune/scientific_locality.json" \
            --$EDIT > ${OUTDIR}/log${edit}.txt 2>&1
        fi
    done
done
