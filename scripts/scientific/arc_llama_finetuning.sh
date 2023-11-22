#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N arcfinet           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=48G  # Request 48GB of GPU memory per GPU
#$ -l gpu_c=6.0		# Request GPU compute capability
#$ -t 1

module load miniconda
conda activate dune2

PROJECTP="/projectnb/llamagrp/feyzanb/dune"
MODELPATH="/projectnb/llamagrp/feyzanb/peft/trained_models"
MODELNAME="all_shuffled_dune_edits_meta-llama_Llama-2-7b-chat-hf_LORA_CAUSAL_LM_epoch7"
OUTDIR="${PROJECTP}/outputs/ARC_locality/finetuning_${MODELNAME}"
CACHE="${PROJECTP}/cache/ARC"
mkdir -p $OUTDIR
mkdir -p $CACHE
python eval.py \
--model_name $MODELPATH/$MODELNAME \
--dataset_name ARC \
--output_dir $OUTDIR \
--generations_cache $CACHE/$MODELNAME.json \
--filename_queries "${PROJECTP}/dune/scientific_locality.json" \
--llama \
--peft \
--batch_size 1