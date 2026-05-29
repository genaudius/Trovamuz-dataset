#!/bin/bash
# train_trovamuz.sh — TrovaMUZ_V1 LoRA Training para RunPod H100
# Uso: bash train_trovamuz.sh
set -e

source .venv/bin/activate

export PYTHONUTF8=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python finetune_musicgen.py \
    --dataset training/datasets \
    --model_id melody \
    --epochs 2000 \
    --lr 5e-5 \
    --batch_size 4 \
    --grad_acc 4 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --warmup_steps 200 \
    --sample_every 250 \
    --output models/TrovaMUZ_V1_LoRA_v2 \
    --test_prompt "musicgau_adn style, TrovaMUZ_V1 Dominican bachata romantica, syncopated bicheo rhythm, requinto guitar with expressive bends, 116 BPM, A minor, romantic melancholic mood"
