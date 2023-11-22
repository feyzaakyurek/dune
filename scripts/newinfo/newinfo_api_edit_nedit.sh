#!/bin/bash -l

#$ -P llamagrp       # Specify the SCC project name you want to use
#$ -l h_rt=12:00:00   # Specify the hard time limit for the job
#$ -N newinfoapinedit           # Give job a name
#$ -j y               # Merge the error and output streams into a single file
#$ -m a             # Send email when job begins, ends and aborts
#$ -pe omp 4          # Specify the parallel environment and the number of cores
#$ -t 1-2

module load conda
conda activate dune2

PROJECTP="/projectnb/llamagrp/feyzanb/dune"

EDIT="no_edit"
MODELNAME="bard"
OUTDIR="${PROJECTP}/outputs/NewInfo/${EDIT}_${MODELNAME}"
mkdir -p $OUTDIR
python eval.py \
--model_name $MODELNAME \
--dataset_name NewInfo \
--output_dir $OUTDIR \
--filename_queries "${PROJECTP}/dune/new_info.json" \
--$EDIT


# cnt=0
# for MODELNAME in "bard"; do
#     for EDIT in "with_edit" "no_edit"; do
#         (( cnt++ ))
#         if [[ $cnt -eq $SGE_TASK_ID ]]; then
#             OUTDIR="${PROJECTP}/outputs/NewInfo_post_emnlp/${EDIT}_${MODELNAME}"
#             mkdir -p $OUTDIR
#             python eval.py \
#             --model_name $MODELNAME \
#             --dataset_name NewInfo \
#             --output_dir $OUTDIR \
#             --filename_queries "${PROJECTP}/source/newinfo/new_info.json" \
#             --$EDIT > ${OUTDIR}/log${edit}.txt 2>&1
#         fi
#     done
# done

