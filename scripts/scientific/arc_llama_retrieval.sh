#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N arcretrieval           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=48G  # Request 48GB of GPU memory per GPU
#$ -l gpu_c=6.0		# Request GPU compute capability
#$ -t 1

module load miniconda
conda activate dune2


MODELHOST="meta-llama"
MODELSUFFIX="Llama-2-7b-chat-hf"
MODELNAME="${MODELHOST}/${MODELSUFFIX}"
OUTDIR="${PROJECTP}/outputs/ARC_locality/retrieval_${MODELNAME}"
CACHE="${PROJECTP}/cache/ARC"
mkdir -p $OUTDIR
mkdir -p $CACHE
python eval.py \
--model_name $MODELNAME \
--dataset_name ARC \
--output_dir $OUTDIR \
--retriever_mechanism bm25 \
--generations_cache $CACHE/${MODELSUFFIX}.json \
--with_edit \
--llama \
--batch_size 1 \
--filename_queries "${PROJECTP}/dune/scientific_locality.json" > ${OUTDIR}/log.txt 2>&1