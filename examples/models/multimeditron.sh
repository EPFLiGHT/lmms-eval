#!/bin/bash
# Install package at https://github.com/OpenMeditron/MultiMeditron.git
# and run this script to evaluate the model on MultiMeditron dataset.
# pip install git+https://github.com/EPFLiGHT/MultiMeditron.git

export OPENAI_API_URL=""http://localhost:8000/v1""
export OPENAI_API_KEY="PROUT"
accelerate launch\
    --num_processes=8 \
    -m lmms_eval \
    --model multimeditron \
    --model_args pretrained="ClosedMeditron/MultiMeditron-LLaMA-8B-CLIP",default_llm="meta-llama/Llama-3.1-8B-Instruct" \
    --tasks path_vqa \
    --batch_size 64 \
    --log_samples \
    --log_samples_suffix llava_v1.5_mme \
    --output_path ./logs/
   
  

