#!/bin/bash
# train_trovamuz.sh — TrovaMUZ_V1 LoRA Training para RunPod H100
# V3: Resume from epoch 176, AdamW8bit, LR=1e-4, caption_dropout=0.05
# Uso: bash train_trovamuz.sh
set -e

source .venv/bin/activate

export PYTHONUTF8=1
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Descargar adapter V2 desde R2 (para resume) ──────────────────────────────
ADAPTER_DIR="models/TrovaMUZ_V1_LoRA_v2/best"
ADAPTER_FILE="$ADAPTER_DIR/lora_adapter.pt"

if [ ! -f "$ADAPTER_FILE" ]; then
    echo "Descargando adapter V2 desde R2..."
    mkdir -p "$ADAPTER_DIR"
    python3 - << 'PYEOF'
import boto3, os
from pathlib import Path

s3 = boto3.client('s3',
    endpoint_url=f'https://{os.environ["R2_ACCOUNT_ID"]}.r2.cloudflarestorage.com',
    aws_access_key_id=os.environ['R2_ACCESS_KEY'],
    aws_secret_access_key=os.environ['R2_SECRET_KEY'],
    region_name='auto')

bucket = os.environ['R2_BUCKET_NAME']
files = [
    ('TrovaMUZ_V1/adapters/models/TrovaMUZ_V1_LoRA_v2/best/lora_adapter.pt',
     'models/TrovaMUZ_V1_LoRA_v2/best/lora_adapter.pt'),
    ('TrovaMUZ_V1/adapters/models/TrovaMUZ_V1_LoRA_v2/best/lora_config.json',
     'models/TrovaMUZ_V1_LoRA_v2/best/lora_config.json'),
]
for key, local in files:
    Path(local).parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(bucket, key, local)
        print(f"  ✓ {local}")
    except Exception as e:
        print(f"  ✗ {key}: {e}")
PYEOF
    echo "Adapter descargado."
else
    echo "Adapter V2 ya existe en $ADAPTER_FILE — OK"
fi

# ── Instalar bitsandbytes si no está ─────────────────────────────────────────
python3 -c "import bitsandbytes" 2>/dev/null || pip install bitsandbytes -q || echo "[warn] bitsandbytes no disponible — usando AdamW estándar"

# ── Training V3: resume desde epoch 176 ──────────────────────────────────────
python finetune_musicgen.py \
    --dataset training/datasets \
    --model_id melody \
    --epochs 2000 \
    --start_epoch 177 \
    --resume models/TrovaMUZ_V1_LoRA_v2/best \
    --lr 1e-4 \
    --weight_decay 1e-4 \
    --batch_size 4 \
    --grad_acc 4 \
    --lora_rank 32 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --caption_dropout 0.05 \
    --warmup_steps 10 \
    --sample_every 250 \
    --upload_every 50 \
    --output models/TrovaMUZ_V1_LoRA_v3 \
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
output_dir = Path('models/TrovaMUZ_V1_LoRA_v3')

files = list(output_dir.rglob('*.pt')) + list(output_dir.rglob('*.wav')) + list(output_dir.rglob('*.json'))
print(f"Subiendo {len(files)} archivos a R2...")

for f in files:
    key = f'TrovaMUZ_V1/adapters/{f}'
    s3.upload_file(str(f), bucket, key)
    print(f"  ✓ {f}")

print(f"\nTodo guardado en R2: s3://{bucket}/TrovaMUZ_V1/adapters/")
print("Adapter principal: TrovaMUZ_V1/adapters/models/TrovaMUZ_V1_LoRA_v3/best/lora_adapter.pt")
PYEOF
