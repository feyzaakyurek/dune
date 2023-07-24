#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N artscopebard           # Give job a name
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
for MODELNAME in "gpt-3.5-turbo" "gpt-4"; do
    (( cnt++ ))
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic_locality/retrieval_conditional_${MODELNAME}"
        CACHE="/projectnb/llamagrp/feyzanb/dune/cache/Arithmetic"
        mkdir -p $OUTDIR
        mkdir -p $CACHE
        python eval.py \
        --model_name $MODELNAME \
        --dataset_name Arithmetic \
        --output_dir $OUTDIR \
        --retriever_mechanism scope \
        --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_locality_qa.json" \
        --scope_cache /projectnb/llamagrp/feyzanb/dune/outputs/scope_classifier/distilbert-base-cased/all_shuffled_edits_cache.json \
        --with_edit > ${OUTDIR}/log.txt 2>&1
    fi
done