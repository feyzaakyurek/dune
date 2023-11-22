#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N niretapi           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -t 1

module load conda
conda activate dune2

PROJECTP="/projectnb/llamagrp/feyzanb/dune"
MODELNAME="gpt-4"
OUTDIR="${PROJECTP}/outputs/NewInfo_deneme/retrieval_${MODELNAME}"
CACHE="${PROJECTP}/cache/NewInfo"
mkdir -p $OUTDIR
mkdir -p $CACHE
python eval.py \
--model_name $MODELNAME \
--dataset_name NewInfo \
--output_dir $OUTDIR \
--retriever_mechanism bm25 \
--filename_queries "${PROJECTP}/dune/new_info.json" \
--with_edit

# cnt=0
# for MODELNAME in "bard"; do
#     (( cnt++ ))
#     if [[ $cnt -eq $SGE_TASK_ID ]]; then
#         OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/NewInfo_post_emnlp/retrieval_${MODELNAME}"
#         CACHE="/projectnb/llamagrp/feyzanb/dune/cache/NewInfo"
#         mkdir -p $OUTDIR
#         mkdir -p $CACHE
#         python eval.py \
#         --model_name $MODELNAME \
#         --dataset_name NewInfo \
#         --output_dir $OUTDIR \
#         --retriever_mechanism bm25 \
#         --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/newinfo/new_info.json" \
#         --with_edit > ${OUTDIR}/log.txt 2>&1
#     fi
# done
