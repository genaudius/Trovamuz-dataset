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

# ── Auto-upload a Cloudflare R2 al terminar ───────────────────────────────────
echo ""
echo "Subiendo modelos a R2..."
python3 - << 'PYEOF'
import boto3, os
from pathlib import Path

s3 = boto3.client('s3',
    endpoint_url=f'https://{os.environ["R2_ACCOUNT_ID"]}.r2.cloudflarestorage.com',
    aws_access_key_id=os.environ['R2_ACCESS_KEY'],
    aws_secret_access_key=os.environ['R2_SECRET_KEY'],
    region_name='auto')

bucket = os.environ['R2_BUCKET_NAME']
output_dir = Path('models/TrovaMUZ_V1_LoRA_v2')

files = list(output_dir.rglob('*.pt')) + list(output_dir.rglob('*.wav'))
print(f"Subiendo {len(files)} archivos a R2...")

for f in files:
    key = f'TrovaMUZ_V1/adapters/{f}'
    s3.upload_file(str(f), bucket, key)
    print(f"  ✓ {f}")

print(f"\nTodo guardado en R2: s3://{bucket}/TrovaMUZ_V1/adapters/")
print("Adapter principal: TrovaMUZ_V1/adapters/models/TrovaMUZ_V1_LoRA_v2/best/lora_adapter.pt")
PYEOF
