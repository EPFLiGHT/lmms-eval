#!/bin/bash
# Install package at https://github.com/OpenMeditron/MultiMeditron.git
# and run this script to evaluate the model on MultiMeditron dataset.
pip install git+https://github.com/OpenMeditron/MultiMeditron.git

python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model multimeditron \
    --model_args pretrained="/mloscratch/users/miczhang/multimodal/MultiMeditron/models/MultiMeditron-8B-Alignment-Image-CLIP/July8/checkpoint-2124/",default_llm="meta-llama/Llama-3.1-8B-Instruct" \
    --tasks mme \
    --batch_size 64 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme \
    --output_path ./logs/

