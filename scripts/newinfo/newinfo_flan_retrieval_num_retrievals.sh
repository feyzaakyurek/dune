#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N niretrieval           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=48G  # Request 48GB of GPU memory per GPU
#$ -l gpu_c=6.0		# Request GPU compute capability
#$ -t 1-2

source /projectnb/llamagrp/feyzanb/anaconda3/etc/profile.d/conda.sh
conda activate dune


cnt=0
for MODELNAME in xxl; do
for NUM in 8 16; do
    (( cnt++ ))
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/NewInfo/retrieval_flan-t5-${MODELNAME}_num_retrievals_${NUM}"
        CACHE="/projectnb/llamagrp/feyzanb/dune/cache/NewInfo"
        mkdir -p $OUTDIR
        mkdir -p $CACHE
        python eval.py \
        --model_name google/flan-t5-$MODELNAME \
        --dataset_name NewInfo \
        --output_dir $OUTDIR \
        --retriever_mechanism bm25 \
        --batch_size 2 \
        --num_retrievals $NUM \
        --generations_cache $CACHE/flan-t5-${MODELNAME}.json \
        --with_edit \
        --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/newinfo/new_info.json" > ${OUTDIR}/log.txt 2>&1
    fi
done
done