#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N niscope           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=48G  # Request 48GB of GPU memory per GPU
#$ -l gpu_c=6.0		# Request GPU compute capability
#$ -t 1-3

source /projectnb/llamagrp/feyzanb/anaconda3/etc/profile.d/conda.sh
conda activate dune


cnt=0
for MODELNAME in xxl; do
for NUM in 1 2 4; do
    (( cnt++ ))
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/NewInfo/retrieval_conditional_flan-t5-${MODELNAME}_num_retrievals_${NUM}"
        CACHE="/projectnb/llamagrp/feyzanb/dune/cache/NewInfo"
        mkdir -p $OUTDIR
        mkdir -p $CACHE
        python eval.py \
        --model_name google/flan-t5-$MODELNAME \
        --dataset_name NewInfo \
        --output_dir $OUTDIR \
        --retriever_mechanism scope \
        --batch_size 4 \
        --num_retrievals $NUM \
        --generations_cache $CACHE/flan-t5-${MODELNAME}.json \
        --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/newinfo/new_info.json" \
        --scope_cache /projectnb/llamagrp/feyzanb/dune/outputs/scope_classifier/distilbert-base-cased/all_shuffled_edits_cache_${NUM}.json \
        --with_edit > ${OUTDIR}/log.txt 2>&1
    fi
done
done