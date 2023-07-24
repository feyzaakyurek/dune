#!/bin/bash -l

# No edit
OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic"
MODELNAME="gpt-3.5-turbo"
python eval.py \
--model_name $MODELNAME \
--dataset_name Arithmetic \
--generations_cache /projectnb/llamagrp/feyzanb/dune/cache/Arithmetic/${MODELNAME}.json \
--filename_queries "/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_qa.json" \
--output_dir $OUTDIR/no_edit_$MODELNAME

# Gold edit
# OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic"
# MODELNAME="flan-t5-small"
# python eval.py \
# --model_name google/$MODELNAME \
# --dataset_name Arithmetic \
# --generations_cache /projectnb/llamagrp/feyzanb/dune/cache/Arithmetic/${MODELNAME}.json \
# --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_qa.json" \
# --output_dir $OUTDIR/with_edit_$MODELNAME \
# --with_edit

# Retrieval
# OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic"
# MODELNAME="flan-t5-small"
# python eval.py \
# --model_name google/$MODELNAME \
# --dataset_name Arithmetic \
# --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_qa.json" \
# --output_dir $OUTDIR/retrieval_$MODELNAME \
# --retriever_mechanism bm25 \
# --with_edit \
# --generations_cache /projectnb/llamagrp/feyzanb/dune/cache/$MODELNAME.json

# Retrieval Conditional
# OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic"
# MODELNAME="flan-t5-small"
# python eval.py \
# --model_name google/$MODELNAME \
# --dataset_name Arithmetic \
# --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_qa.json" \
# --output_dir $OUTDIR/retrieval_conditional_$MODELNAME \
# --retriever_mechanism scope \
# --with_edit \
# --generations_cache /projectnb/llamagrp/feyzanb/dune/outputs/cache/$MODELNAME.json \
# --scope_cache /projectnb/llamagrp/feyzanb/dune/outputs/scope_classifier/distilbert-base-cased/all_shuffled_edits_cache.json

# Fine-tuning
# OUTDIR="/projectnb/llamagrp/feyzanb/dune/outputs/Arithmetic"
# MODELPATH="/projectnb/llamagrp/feyzanb/dune/outputs/"
# MODELNAME="all_fine_tune_flant5small"
# python eval.py \
# --model_name $MODELPATH/$MODELNAME \
# --dataset_name Arithmetic \
# --generations_cache /projectnb/llamagrp/feyzanb/dune/cache/Arithmetic/${MODELNAME}.json \
# --filename_queries "/projectnb/llamagrp/feyzanb/dune/source/arithmetic/arithmetic_qa.json" \
# --output_dir $OUTDIR/no_edit_$MODELNAME \
# --from_flax