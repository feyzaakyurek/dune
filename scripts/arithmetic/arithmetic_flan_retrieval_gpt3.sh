#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N artretgpt3           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=16G  # Request 48GB of GPU memory per GPU
#$ -l gpu_c=6.0		# Request GPU compute capability
#$ -t 1-2

source /projectnb/llamagrp/feyzanb/anaconda3/etc/profile.d/conda.sh
conda activate dune


cnt=0
for MODELNAME in small large; do
    (( cnt++ ))
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic_locality/retrieval_gpt3_flan-t5-${MODELNAME}"
        CACHE="/projectnb/llamagrp/feyzanb/dune/cache/Arithmetic"
        mkdir -p $OUTDIR
        mkdir -p $CACHE
        python eval.py \
        --model_name google/flan-t5-$MODELNAME \
        --dataset_name Arithmetic \
        --output_dir $OUTDIR \
        --retriever_mechanism gpt3 \
        --generations_cache $CACHE/flan-t5-${MODELNAME}.json \
        --with_edit \
        --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_locality_qa.json" > ${OUTDIR}/log.txt 2>&1
    fi
done