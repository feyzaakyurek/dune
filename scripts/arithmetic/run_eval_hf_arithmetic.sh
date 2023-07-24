#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N arithmetic           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m ea             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -l gpus=1           # Request GPU
#$ -l gpu_memory=48G  # Request 48GB of GPU memory per GPU
#$ -l gpu_c=6.0		# Request GPU compute capability
#$ -t 1-6

source /projectnb/llamagrp/feyzanb/anaconda3/etc/profile.d/conda.sh
conda activate dune

cnt=0
for model_name in "flan-t5-large" "flan-t5-xl" "flan-t5-xxl"; do
for edit in "--edit" "--no-edit"; do
    (( cnt++ ))
    output_dir=/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic/${model_name}
    filename="/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_qa.csv"
    if [[ $cnt -eq $SGE_TASK_ID ]]; then
        mkdir -p ${output_dir}
        python eval_arithmetic.py \
        --model_name google/$model_name \
        --output_dir $output_dir \
        --filename $filename \
        --batch_size 8 \
        $edit > ${output_dir}/log${edit}.txt 2>&1
    fi
done
done

# model_name="flan-t5-small"
# edit="--edit"
# output_dir=/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic/${model_name}
# filename="/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_qa_short.csv"
# mkdir -p ${output_dir}
# python eval_arithmetic.py \
# --model_name google/$model_name \
# --output_dir $output_dir \
# --filename $filename \
# $edit