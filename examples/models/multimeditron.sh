#!/bin/bash

export CHECKPOINT=$1
export TOKENIZER_TYPE=$2
export TASKS=$3

python3 -m accelerate.commands.launch \
    --num_processes 4 \
    -m lmms_eval \
    --model multimeditron \
    --model_args pretrained="${CHECKPOINT}",default_llm="meta-llama/Llama-3.1-8B-Instruct",tokenizer_type="${TOKENIZER_TYPE}",device_map="auto" \
    --tasks ${TASKS} \
    --batch_size 1 \
    --verbosity DEBUG \
    --log_samples \
    --log_samples_suffix multiple_bench \
    --output_path ./debug/

